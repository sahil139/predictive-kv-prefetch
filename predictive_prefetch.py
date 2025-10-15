#!/usr/bin/env python3
"""
Predictive Prefetch implementation for KV cache optimization.

This module implements intelligent prefetching based on attention patterns,
using asynchronous transfers to overlap prefetching with computation.
"""

import time
import json
import logging
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
import queue
import asyncio
from pathlib import Path
import os
import sys
import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from kv_paging import PagedKVCache, KVPage
from metrics import MetricsCollector


@dataclass
class AttentionPattern:
    """Represents attention pattern for a token."""
    token_idx: int
    attention_scores: torch.Tensor  # Shape: (num_heads, seq_len)
    timestamp: float
    layer_idx: int = 0


@dataclass
class PrefetchPrediction:
    """Represents a prefetch prediction."""
    page_id: int
    confidence: float
    predicted_access_time: float
    attention_weight: float
    priority: int


@dataclass
class PrefetchMetrics:
    """Metrics for predictive prefetching performance."""
    total_prefetches: int = 0
    correct_prefetches: int = 0
    incorrect_prefetches: int = 0
    wasted_transfers: int = 0
    total_prefetch_time_ms: float = 0.0
    total_wasted_time_ms: float = 0.0
    prediction_accuracy: float = 0.0
    prefetch_hit_rate: float = 0.0
    misprediction_cost_ms: float = 0.0
    overlap_efficiency: float = 0.0


class AttentionAnalyzer:
    """Analyzes attention patterns to predict future page access."""
    
    def __init__(self, window_size: int = 10, decay_factor: float = 0.9):
        """
        Initialize attention analyzer.
        
        Args:
            window_size: Number of recent attention patterns to consider
            decay_factor: Decay factor for older attention patterns
        """
        self.window_size = window_size
        self.decay_factor = decay_factor
        self.attention_history: deque = deque(maxlen=window_size)
        self.page_access_counts: Dict[int, float] = defaultdict(float)
        self.page_attention_weights: Dict[int, float] = defaultdict(float)
        
        logging.info(f"Initialized AttentionAnalyzer: window_size={window_size}, decay_factor={decay_factor}")
    
    def add_attention_pattern(self, pattern: AttentionPattern, page_size: int) -> None:
        """Add a new attention pattern for analysis."""
        self.attention_history.append(pattern)
        
        # Update page attention weights based on attention scores
        seq_len = pattern.attention_scores.shape[1]
        
        for token_idx in range(seq_len):
            page_id = token_idx // page_size
            
            # Calculate attention weight for this page
            page_start = page_id * page_size
            page_end = min(page_start + page_size, seq_len)
            
            # Average attention across heads for this page
            page_attention = pattern.attention_scores[:, page_start:page_end].mean()
            
            # Apply temporal decay
            self.page_attention_weights[page_id] *= self.decay_factor
            self.page_attention_weights[page_id] += page_attention.item()
            
            # Update access counts
            self.page_access_counts[page_id] *= self.decay_factor
            self.page_access_counts[page_id] += 1.0
    
    def predict_future_pages(self, current_token_idx: int, page_size: int, top_k: int = 5) -> List[PrefetchPrediction]:
        """
        Predict which pages are likely to be accessed next.
        
        Args:
            current_token_idx: Current token index
            page_size: Size of each page
            top_k: Number of top predictions to return
            
        Returns:
            List of prefetch predictions sorted by priority
        """
        predictions = []
        current_page_id = current_token_idx // page_size
        
        # Look ahead window (predict next few pages)
        look_ahead_pages = 3
        
        for page_id in range(current_page_id + 1, current_page_id + look_ahead_pages + 1):
            if page_id in self.page_attention_weights:
                attention_weight = self.page_attention_weights[page_id]
                access_count = self.page_access_counts[page_id]
                
                # Calculate confidence based on attention weight and access pattern
                confidence = min(1.0, attention_weight * (1.0 + access_count * 0.1))
                
                # Calculate priority (higher is better)
                priority = int(confidence * 1000)
                
                # Estimate access time (simplified)
                predicted_access_time = time.time() + (page_id - current_page_id) * 0.01
                
                prediction = PrefetchPrediction(
                    page_id=page_id,
                    confidence=confidence,
                    predicted_access_time=predicted_access_time,
                    attention_weight=attention_weight,
                    priority=priority
                )
                
                predictions.append(prediction)
        
        # Sort by priority (descending)
        predictions.sort(key=lambda x: x.priority, reverse=True)
        
        return predictions[:top_k]
    
    def get_page_relevance(self, page_id: int) -> float:
        """Get relevance score for a specific page."""
        return self.page_attention_weights.get(page_id, 0.0)


class AsyncPrefetcher:
    """Handles asynchronous prefetching operations."""
    
    def __init__(self, device: torch.device, max_concurrent_transfers: int = 4):
        """
        Initialize async prefetcher.
        
        Args:
            device: Target device for prefetching
            max_concurrent_transfers: Maximum concurrent transfers
        """
        self.device = device
        self.max_concurrent_transfers = max_concurrent_transfers
        self.active_transfers: Dict[int, torch.cuda.Stream] = {}
        self.transfer_queue = queue.Queue()
        self.completed_transfers: Dict[int, bool] = {}
        
        # Create CUDA streams for async transfers
        if torch.cuda.is_available():
            self.streams = [torch.cuda.Stream() for _ in range(max_concurrent_transfers)]
        else:
            self.streams = []
        
        logging.info(f"Initialized AsyncPrefetcher: device={device}, max_concurrent={max_concurrent_transfers}")
    
    def prefetch_page_async(self, page: KVPage, stream_idx: int = 0) -> Tuple[bool, float]:
        """
        Prefetch a page asynchronously.
        
        Args:
            page: Page to prefetch
            stream_idx: CUDA stream index to use
            
        Returns:
            Tuple of (success, transfer_time_ms)
        """
        if page.storage_location == "gpu":
            return True, 0.0  # Already in GPU
        
        if not torch.cuda.is_available() or not self.streams:
            # Fallback to synchronous transfer
            start_time = time.time()
            try:
                page.key_cache = page.key_cache.to(self.device, non_blocking=False)
                page.value_cache = page.value_cache.to(self.device, non_blocking=False)
                page.storage_location = "gpu"
                transfer_time = (time.time() - start_time) * 1000
                return True, transfer_time
            except Exception as e:
                logging.error(f"Failed to prefetch page {page.page_id}: {e}")
                return False, 0.0
        
        start_time = time.time()
        
        try:
            # Use specific CUDA stream for async transfer
            stream = self.streams[stream_idx % len(self.streams)]
            
            with torch.cuda.stream(stream):
                # Non-blocking transfer
                page.key_cache = page.key_cache.to(self.device, non_blocking=True)
                page.value_cache = page.value_cache.to(self.device, non_blocking=True)
            
            # Update storage location
            page.storage_location = "gpu"
            
            transfer_time = (time.time() - start_time) * 1000
            
            # Track active transfer
            self.active_transfers[page.page_id] = stream
            self.completed_transfers[page.page_id] = False
            
            logging.debug(f"Started async prefetch for page {page.page_id} in {transfer_time:.2f}ms")
            
            return True, transfer_time
            
        except Exception as e:
            logging.error(f"Failed to prefetch page {page.page_id}: {e}")
            return False, 0.0
    
    def check_transfer_completion(self, page_id: int) -> bool:
        """Check if a transfer has completed."""
        if page_id not in self.active_transfers:
            return True  # Not an active transfer
        
        stream = self.active_transfers[page_id]
        
        if stream.query():
            # Transfer completed
            del self.active_transfers[page_id]
            self.completed_transfers[page_id] = True
            return True
        
        return False
    
    def wait_for_transfer(self, page_id: int) -> None:
        """Wait for a specific transfer to complete."""
        if page_id in self.active_transfers:
            stream = self.active_transfers[page_id]
            stream.synchronize()
            del self.active_transfers[page_id]
            self.completed_transfers[page_id] = True
    
    def cleanup_completed_transfers(self) -> None:
        """Clean up completed transfers."""
        completed = [pid for pid, completed in self.completed_transfers.items() if completed]
        for pid in completed:
            if pid in self.active_transfers:
                del self.active_transfers[pid]
            del self.completed_transfers[pid]


class PredictivePrefetcher:
    """
    Main predictive prefetcher that coordinates attention analysis and async prefetching.
    """
    
    def __init__(
        self,
        cache: PagedKVCache,
        page_size: int = 256,
        prefetch_window: int = 3,
        max_prefetch_pages: int = 5,
        confidence_threshold: float = 0.3,
        enable_async: bool = True
    ):
        """
        Initialize predictive prefetcher.
        
        Args:
            cache: Paged KV cache to prefetch into
            page_size: Size of each page
            prefetch_window: Number of pages to look ahead
            max_prefetch_pages: Maximum pages to prefetch at once
            confidence_threshold: Minimum confidence for prefetching
            enable_async: Whether to use async prefetching
        """
        self.cache = cache
        self.page_size = page_size
        self.prefetch_window = prefetch_window
        self.max_prefetch_pages = max_prefetch_pages
        self.confidence_threshold = confidence_threshold
        self.enable_async = enable_async
        
        # Components
        self.attention_analyzer = AttentionAnalyzer(window_size=10)
        self.async_prefetcher = AsyncPrefetcher(cache.device) if enable_async else None
        
        # Metrics
        self.metrics = PrefetchMetrics()
        self.prefetch_history: List[Dict[str, Any]] = []
        
        # State tracking
        self.current_token_idx = 0
        self.active_prefetches: Dict[int, Dict[str, Any]] = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        logging.info(f"Initialized PredictivePrefetcher: page_size={page_size}, prefetch_window={prefetch_window}")
        logging.info(f"Max prefetch pages: {max_prefetch_pages}, confidence_threshold: {confidence_threshold}")
    
    def update_attention_pattern(self, token_idx: int, attention_scores: torch.Tensor, layer_idx: int = 0) -> None:
        """
        Update attention pattern after token generation.
        
        Args:
            token_idx: Token index that was generated
            attention_scores: Attention scores from the model
            layer_idx: Layer index (for multi-layer models)
        """
        with self.lock:
            pattern = AttentionPattern(
                token_idx=token_idx,
                attention_scores=attention_scores,
                timestamp=time.time(),
                layer_idx=layer_idx
            )
            
            self.attention_analyzer.add_attention_pattern(pattern, self.page_size)
            self.current_token_idx = token_idx
            
            logging.debug(f"Updated attention pattern for token {token_idx}")
    
    def predict_and_prefetch(self) -> Dict[str, Any]:
        """
        Predict future page access and initiate prefetching.
        
        Returns:
            Dictionary with prefetch results and metrics
        """
        with self.lock:
            start_time = time.time()
            
            # Get predictions
            predictions = self.attention_analyzer.predict_future_pages(
                self.current_token_idx, 
                self.page_size, 
                self.max_prefetch_pages
            )
            
            prefetch_results = {
                'predictions_made': len(predictions),
                'prefetches_initiated': 0,
                'prefetches_successful': 0,
                'prefetches_failed': 0,
                'wasted_transfers': 0,
                'total_time_ms': 0.0,
                'predictions': []
            }
            
            # Process predictions
            for prediction in predictions:
                if prediction.confidence < self.confidence_threshold:
                    continue
                
                page_id = prediction.page_id
                
                # Check if page exists and is not already in GPU
                if page_id in self.cache.pages:
                    page = self.cache.pages[page_id]
                    
                    if page.storage_location == "gpu":
                        # Page already in GPU, mark as wasted transfer
                        self.metrics.wasted_transfers += 1
                        prefetch_results['wasted_transfers'] += 1
                        continue
                    
                    # Check if already being prefetched
                    if page_id in self.active_prefetches:
                        continue
                    
                    # Initiate prefetch
                    prefetch_start = time.time()
                    
                    if self.enable_async and self.async_prefetcher:
                        success, transfer_time = self.async_prefetcher.prefetch_page_async(page)
                    else:
                        # Synchronous prefetch
                        success, transfer_time = self._prefetch_page_sync(page)
                    
                    prefetch_time = (time.time() - prefetch_start) * 1000
                    
                    # Update metrics
                    self.metrics.total_prefetches += 1
                    prefetch_results['prefetches_initiated'] += 1
                    
                    if success:
                        self.metrics.total_prefetch_time_ms += transfer_time
                        prefetch_results['prefetches_successful'] += 1
                        
                        # Track active prefetch
                        self.active_prefetches[page_id] = {
                            'prediction': prediction,
                            'start_time': prefetch_start,
                            'transfer_time': transfer_time
                        }
                    else:
                        self.metrics.total_wasted_time_ms += prefetch_time
                        prefetch_results['prefetches_failed'] += 1
                    
                    # Record prediction
                    prefetch_results['predictions'].append({
                        'page_id': page_id,
                        'confidence': prediction.confidence,
                        'priority': prediction.priority,
                        'success': success,
                        'transfer_time_ms': transfer_time
                    })
            
            total_time = (time.time() - start_time) * 1000
            prefetch_results['total_time_ms'] = total_time
            
            # Update overall metrics
            self._update_prediction_accuracy()
            
            logging.debug(f"Prefetch cycle completed: {prefetch_results['prefetches_initiated']} initiated, {prefetch_results['prefetches_successful']} successful")
            
            return prefetch_results
    
    def _prefetch_page_sync(self, page: KVPage) -> Tuple[bool, float]:
        """Synchronous prefetch fallback."""
        start_time = time.time()
        
        try:
            page.key_cache = page.key_cache.to(self.cache.device, non_blocking=False)
            page.value_cache = page.value_cache.to(self.cache.device, non_blocking=False)
            page.storage_location = "gpu"
            
            transfer_time = (time.time() - start_time) * 1000
            return True, transfer_time
            
        except Exception as e:
            logging.error(f"Failed to prefetch page {page.page_id}: {e}")
            return False, 0.0
    
    def check_prefetch_hits(self) -> Dict[str, int]:
        """
        Check which prefetched pages were actually used.
        
        Returns:
            Dictionary with hit/miss statistics
        """
        with self.lock:
            hits = 0
            misses = 0
            
            # Check completed prefetches
            completed_prefetches = []
            
            for page_id, prefetch_info in self.active_prefetches.items():
                if page_id in self.cache.pages:
                    page = self.cache.pages[page_id]
                    
                    # Check if page was accessed since prefetch
                    if page.last_accessed > prefetch_info['start_time']:
                        hits += 1
                        self.metrics.correct_prefetches += 1
                    else:
                        misses += 1
                        self.metrics.incorrect_prefetches += 1
                    
                    completed_prefetches.append(page_id)
            
            # Clean up completed prefetches
            for page_id in completed_prefetches:
                del self.active_prefetches[page_id]
            
            # Update metrics
            self._update_prediction_accuracy()
            
            return {
                'hits': hits,
                'misses': misses,
                'total_checked': hits + misses
            }
    
    def _update_prediction_accuracy(self) -> None:
        """Update prediction accuracy metrics."""
        total_predictions = self.metrics.correct_prefetches + self.metrics.incorrect_prefetches
        if total_predictions > 0:
            self.metrics.prediction_accuracy = self.metrics.correct_prefetches / total_predictions
        
        if self.metrics.total_prefetches > 0:
            self.metrics.prefetch_hit_rate = self.metrics.correct_prefetches / self.metrics.total_prefetches
    
    def get_prefetch_stats(self) -> Dict[str, Any]:
        """Get comprehensive prefetch statistics."""
        with self.lock:
            stats = {
                'total_prefetches': self.metrics.total_prefetches,
                'correct_prefetches': self.metrics.correct_prefetches,
                'incorrect_prefetches': self.metrics.incorrect_prefetches,
                'wasted_transfers': self.metrics.wasted_transfers,
                'prediction_accuracy': self.metrics.prediction_accuracy,
                'prefetch_hit_rate': self.metrics.prefetch_hit_rate,
                'total_prefetch_time_ms': self.metrics.total_prefetch_time_ms,
                'total_wasted_time_ms': self.metrics.total_wasted_time_ms,
                'average_prefetch_time_ms': self.metrics.total_prefetch_time_ms / max(1, self.metrics.total_prefetches),
                'active_prefetches': len(self.active_prefetches),
                'config': {
                    'page_size': self.page_size,
                    'prefetch_window': self.prefetch_window,
                    'max_prefetch_pages': self.max_prefetch_pages,
                    'confidence_threshold': self.confidence_threshold,
                    'enable_async': self.enable_async
                }
            }
            
            # Calculate efficiency metrics
            if self.metrics.total_prefetches > 0:
                stats['efficiency'] = {
                    'hit_rate': self.metrics.correct_prefetches / self.metrics.total_prefetches,
                    'waste_rate': self.metrics.wasted_transfers / self.metrics.total_prefetches,
                    'time_efficiency': 1.0 - (self.metrics.total_wasted_time_ms / max(1, self.metrics.total_prefetch_time_ms))
                }
            
            return stats
    
    def dump_prefetch_metrics(self, filepath: str) -> None:
        """Dump prefetch metrics to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        stats = self.get_prefetch_stats()
        
        # Add timestamp and experiment info
        stats['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        stats['experiment_type'] = 'predictive_prefetch'
        stats['prefetch_history'] = self.prefetch_history[-100:]  # Last 100 entries
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logging.info(f"Prefetch metrics dumped to {filepath}")
    
    def reset_metrics(self) -> None:
        """Reset all prefetch metrics."""
        with self.lock:
            self.metrics = PrefetchMetrics()
            self.prefetch_history.clear()
            self.active_prefetches.clear()
            self.attention_analyzer = AttentionAnalyzer()
            
            logging.info("Prefetch metrics reset")


class PredictiveInferenceEngine:
    """
    Inference engine that integrates predictive prefetching with paged KV cache.
    """
    
    def __init__(self, cache: PagedKVCache, prefetcher: PredictivePrefetcher, num_layers: int = 2):
        """
        Initialize predictive inference engine.
        
        Args:
            cache: Paged KV cache
            prefetcher: Predictive prefetcher
            num_layers: Number of transformer layers
        """
        self.cache = cache
        self.prefetcher = prefetcher
        self.num_layers = num_layers
        self.num_heads = 8
        self.head_dim = 64
        self.hidden_dim = 512
        
        # Mock layers for testing
        self.layers = [
            MockTransformerLayer(self.num_heads, self.head_dim, self.hidden_dim)
            for _ in range(num_layers)
        ]
        
        logging.info(f"Initialized PredictiveInferenceEngine with {num_layers} layers")
    
    def run_inference_with_prefetch(self, input_ids: torch.Tensor, max_new_tokens: int = 50) -> Dict[str, Any]:
        """Run inference with predictive prefetching."""
        batch_size, input_length = input_ids.shape
        
        # Initialize hidden states
        hidden_states = torch.randn(batch_size, input_length, self.hidden_dim)
        
        total_time = 0.0
        total_prefetch_time = 0.0
        tokens_generated = 0
        
        start_time = time.time()
        
        # Process input tokens
        for token_idx in range(input_length):
            token_start = time.time()
            
            # Get KV cache (may trigger prefetch)
            key_cache, value_cache, stall_time = self.cache.get_kv_page(
                token_idx, self.num_layers, self.num_heads, self.head_dim
            )
            
            # Simulate attention computation
            attention_scores = torch.randn(self.num_heads, token_idx + 1)
            
            # Update prefetcher with attention pattern
            self.prefetcher.update_attention_pattern(token_idx, attention_scores)
            
            # Process through layers
            layer_output = hidden_states[:, token_idx:token_idx+1, :]
            for layer in self.layers:
                layer_output, (new_key, new_value) = layer.forward(layer_output)
                self.cache.update_kv_cache(token_idx, new_key, new_value)
            
            token_time = time.time() - token_start
            total_time += token_time
        
        # Generate new tokens with prefetching
        for gen_idx in range(max_new_tokens):
            token_start = time.time()
            
            # Predict and prefetch before computation
            prefetch_results = self.prefetcher.predict_and_prefetch()
            total_prefetch_time += prefetch_results['total_time_ms']
            
            # Get KV cache
            cache_idx = input_length + gen_idx
            key_cache, value_cache, stall_time = self.cache.get_kv_page(
                cache_idx, self.num_layers, self.num_heads, self.head_dim
            )
            
            # Simulate attention computation
            attention_scores = torch.randn(self.num_heads, cache_idx + 1)
            
            # Update prefetcher
            self.prefetcher.update_attention_pattern(cache_idx, attention_scores)
            
            # Generate next token
            next_token = torch.randint(0, 1000, (batch_size, 1))
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Process through layers
            hidden_states = torch.randn(batch_size, input_length + gen_idx + 1, self.hidden_dim)
            layer_output = hidden_states[:, -1:, :]
            
            for layer in self.layers:
                layer_output, (new_key, new_value) = layer.forward(layer_output)
                self.cache.update_kv_cache(cache_idx, new_key, new_value)
            
            tokens_generated += 1
            token_time = time.time() - token_start
            total_time += token_time
            
            # Check prefetch hits periodically
            if gen_idx % 10 == 0:
                hit_stats = self.prefetcher.check_prefetch_hits()
                logging.debug(f"Prefetch hits: {hit_stats}")
        
        total_inference_time = time.time() - start_time
        
        # Get final statistics
        cache_stats = self.cache.get_cache_stats()
        prefetch_stats = self.prefetcher.get_prefetch_stats()
        
        metrics = {
            'total_time_ms': total_inference_time * 1000,
            'total_prefetch_time_ms': total_prefetch_time,
            'prefetch_overhead_percent': (total_prefetch_time / (total_inference_time * 1000)) * 100,
            'tokens_generated': tokens_generated,
            'input_length': input_length,
            'total_tokens': input_length + tokens_generated,
            'tokens_per_second': tokens_generated / total_inference_time,
            'cache_stats': cache_stats,
            'prefetch_stats': prefetch_stats
        }
        
        return metrics


class MockTransformerLayer:
    """Mock transformer layer for testing."""
    
    def __init__(self, num_heads: int = 8, head_dim: int = 64, hidden_dim: int = 512):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        
        self.attention_weights = torch.randn(hidden_dim, hidden_dim)
        self.output_weights = torch.randn(hidden_dim, hidden_dim)
    
    def forward(self, hidden_states: torch.Tensor, past_key_values: Tuple[torch.Tensor, torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Mock forward pass."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Generate mock key and value tensors
        key_cache = torch.randn(1, self.num_heads, seq_len, self.head_dim)
        value_cache = torch.randn(1, self.num_heads, seq_len, self.head_dim)
        
        # Simulate computation time
        time.sleep(0.001)
        
        # Generate output
        output = torch.matmul(hidden_states, self.attention_weights)
        
        return output, (key_cache, value_cache)


def create_predictive_prefetcher(
    cache: PagedKVCache,
    page_size: int = 256,
    prefetch_window: int = 3,
    max_prefetch_pages: int = 5,
    confidence_threshold: float = 0.3,
    enable_async: bool = True
) -> PredictivePrefetcher:
    """Factory function to create a predictive prefetcher."""
    return PredictivePrefetcher(
        cache=cache,
        page_size=page_size,
        prefetch_window=prefetch_window,
        max_prefetch_pages=max_prefetch_pages,
        confidence_threshold=confidence_threshold,
        enable_async=enable_async
    )


if __name__ == "__main__":
    # Example usage and testing
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    # Create cache and prefetcher
    from kv_paging import create_paged_cache
    
    cache = create_paged_cache(page_size=128, max_gpu_pages=8)
    prefetcher = create_predictive_prefetcher(
        cache=cache,
        page_size=128,
        prefetch_window=3,
        max_prefetch_pages=4,
        confidence_threshold=0.2
    )
    
    # Create inference engine
    engine = PredictiveInferenceEngine(cache, prefetcher, num_layers=2)
    
    print("Testing Predictive Prefetching...")
    
    # Test inference
    input_ids = torch.randint(0, 1000, (1, 100))
    metrics = engine.run_inference_with_prefetch(input_ids, max_new_tokens=50)
    
    print(f"\nResults:")
    print(f"Total time: {metrics['total_time_ms']:.2f}ms")
    print(f"Prefetch time: {metrics['total_prefetch_time_ms']:.2f}ms")
    print(f"Prefetch overhead: {metrics['prefetch_overhead_percent']:.1f}%")
    print(f"Tokens/sec: {metrics['tokens_per_second']:.2f}")
    
    # Print prefetch stats
    prefetch_stats = metrics['prefetch_stats']
    print(f"\nPrefetch Stats:")
    print(f"Total prefetches: {prefetch_stats['total_prefetches']}")
    print(f"Correct prefetches: {prefetch_stats['correct_prefetches']}")
    print(f"Prediction accuracy: {prefetch_stats['prediction_accuracy']:.2%}")
    print(f"Prefetch hit rate: {prefetch_stats['prefetch_hit_rate']:.2%}")
    
    # Dump metrics
    prefetcher.dump_prefetch_metrics("logs/predictive_prefetch_test.json")
    print("Test completed!")
