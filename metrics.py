#!/usr/bin/env python3
"""
Metrics collection and analysis utilities for KV cache experiments.

This module provides classes and functions for collecting, storing, and analyzing
performance metrics during inference experiments.
"""

import time
import json
import logging
import torch
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics
import os


@dataclass
class InferenceMetrics:
    """Data class for storing inference metrics."""
    # Timing metrics
    total_latency_ms: float
    per_token_latency_ms: float
    prefill_latency_ms: float
    decode_latency_ms: float
    
    # Token metrics
    num_input_tokens: int
    num_output_tokens: int
    tokens_per_second: float
    
    # Memory metrics
    memory_before_mb: Dict[str, float]
    memory_after_mb: Dict[str, float]
    memory_delta_mb: Dict[str, float]
    peak_memory_mb: Dict[str, float]
    
    # Cache metrics
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0
    
    # Additional metadata
    model_name: str = ""
    device: str = ""
    batch_size: int = 1
    context_length: int = 0
    timestamp: str = ""


class MetricsCollector:
    """Collects and manages inference metrics."""
    
    def __init__(self):
        self.metrics_history: List[InferenceMetrics] = []
        self.session_start_time = time.time()
        self.current_metrics: Optional[InferenceMetrics] = None
        
        # Performance counters
        self.total_tokens_generated = 0
        self.total_inference_time = 0.0
        self.total_cache_hits = 0
        self.total_cache_misses = 0
        
        logging.info("MetricsCollector initialized")
    
    def start_inference(self, model_name: str, device: str, batch_size: int = 1) -> None:
        """Start tracking inference metrics."""
        self.current_metrics = InferenceMetrics(
            total_latency_ms=0.0,
            per_token_latency_ms=0.0,
            prefill_latency_ms=0.0,
            decode_latency_ms=0.0,
            num_input_tokens=0,
            num_output_tokens=0,
            tokens_per_second=0.0,
            memory_before_mb={},
            memory_after_mb={},
            memory_delta_mb={},
            peak_memory_mb={},
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )
        
        # Record initial memory state
        self.current_metrics.memory_before_mb = self._get_memory_usage()
        
        logging.debug("Started inference metrics tracking")
    
    def record_prefill_metrics(self, input_length: int, prefill_time: float) -> None:
        """Record prefill phase metrics."""
        if self.current_metrics is None:
            logging.warning("No active inference session")
            return
        
        self.current_metrics.num_input_tokens = input_length
        self.current_metrics.prefill_latency_ms = prefill_time * 1000
        
        logging.debug(f"Recorded prefill metrics: {input_length} tokens, {prefill_time*1000:.2f}ms")
    
    def record_decode_metrics(self, output_length: int, decode_time: float) -> None:
        """Record decode phase metrics."""
        if self.current_metrics is None:
            logging.warning("No active inference session")
            return
        
        self.current_metrics.num_output_tokens = output_length
        self.current_metrics.decode_latency_ms = decode_time * 1000
        self.current_metrics.total_latency_ms = (
            self.current_metrics.prefill_latency_ms + self.current_metrics.decode_latency_ms
        )
        
        # Calculate per-token metrics
        total_tokens = self.current_metrics.num_input_tokens + self.current_metrics.num_output_tokens
        if total_tokens > 0:
            self.current_metrics.per_token_latency_ms = self.current_metrics.total_latency_ms / total_tokens
        
        if self.current_metrics.total_latency_ms > 0:
            self.current_metrics.tokens_per_second = (
                self.current_metrics.num_output_tokens * 1000 / self.current_metrics.total_latency_ms
            )
        
        logging.debug(f"Recorded decode metrics: {output_length} tokens, {decode_time*1000:.2f}ms")
    
    def record_cache_metrics(self, hits: int, misses: int) -> None:
        """Record cache hit/miss metrics."""
        if self.current_metrics is None:
            logging.warning("No active inference session")
            return
        
        self.current_metrics.cache_hits = hits
        self.current_metrics.cache_misses = misses
        
        total_requests = hits + misses
        if total_requests > 0:
            self.current_metrics.cache_hit_rate = hits / total_requests
        
        logging.debug(f"Recorded cache metrics: {hits} hits, {misses} misses, {self.current_metrics.cache_hit_rate:.2%} hit rate")
    
    def finish_inference(self) -> InferenceMetrics:
        """Finish tracking inference metrics and return the collected metrics."""
        if self.current_metrics is None:
            logging.warning("No active inference session")
            return InferenceMetrics(
                total_latency_ms=0.0, per_token_latency_ms=0.0, prefill_latency_ms=0.0,
                decode_latency_ms=0.0, num_input_tokens=0, num_output_tokens=0,
                tokens_per_second=0.0, memory_before_mb={}, memory_after_mb={},
                memory_delta_mb={}, peak_memory_mb={}
            )
        
        # Record final memory state
        self.current_metrics.memory_after_mb = self._get_memory_usage()
        
        # Calculate memory deltas
        for key in self.current_metrics.memory_after_mb:
            before = self.current_metrics.memory_before_mb.get(key, 0)
            after = self.current_metrics.memory_after_mb[key]
            self.current_metrics.memory_delta_mb[key] = after - before
        
        # Record peak memory usage
        self.current_metrics.peak_memory_mb = self._get_peak_memory_usage()
        
        # Update session totals
        self.total_tokens_generated += self.current_metrics.num_output_tokens
        self.total_inference_time += self.current_metrics.total_latency_ms / 1000
        self.total_cache_hits += self.current_metrics.cache_hits
        self.total_cache_misses += self.current_metrics.cache_misses
        
        # Store metrics
        metrics = self.current_metrics
        self.metrics_history.append(metrics)
        
        # Reset current metrics
        self.current_metrics = None
        
        logging.info(f"Finished inference metrics: {metrics.total_latency_ms:.2f}ms total, {metrics.tokens_per_second:.2f} tok/s")
        
        return metrics
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in MB."""
        memory_usage = {}
        
        if torch.cuda.is_available():
            memory_usage.update({
                'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                'reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
            })
        
        return memory_usage
    
    def _get_peak_memory_usage(self) -> Dict[str, float]:
        """Get peak memory usage in MB."""
        peak_memory = {}
        
        if torch.cuda.is_available():
            peak_memory.update({
                'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
                'max_reserved_mb': torch.cuda.max_memory_reserved() / 1024 / 1024,
            })
        
        return peak_memory
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics collected in this session."""
        if not self.metrics_history:
            return {}
        
        # Calculate aggregate statistics
        latencies = [m.total_latency_ms for m in self.metrics_history]
        per_token_latencies = [m.per_token_latency_ms for m in self.metrics_history]
        tokens_per_second = [m.tokens_per_second for m in self.metrics_history]
        cache_hit_rates = [m.cache_hit_rate for m in self.metrics_history]
        
        # Memory statistics
        max_memory_allocated = max(
            m.peak_memory_mb.get('max_allocated_mb', 0) for m in self.metrics_history
        )
        max_memory_reserved = max(
            m.peak_memory_mb.get('max_reserved_mb', 0) for m in self.metrics_history
        )
        
        session_duration = time.time() - self.session_start_time
        
        return {
            'session_duration_seconds': session_duration,
            'total_inferences': len(self.metrics_history),
            'total_tokens_generated': self.total_tokens_generated,
            'total_inference_time_seconds': self.total_inference_time,
            'average_latency_ms': statistics.mean(latencies) if latencies else 0,
            'median_latency_ms': statistics.median(latencies) if latencies else 0,
            'std_latency_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0,
            'min_latency_ms': min(latencies) if latencies else 0,
            'max_latency_ms': max(latencies) if latencies else 0,
            'average_per_token_latency_ms': statistics.mean(per_token_latencies) if per_token_latencies else 0,
            'average_tokens_per_second': statistics.mean(tokens_per_second) if tokens_per_second else 0,
            'average_cache_hit_rate': statistics.mean(cache_hit_rates) if cache_hit_rates else 0,
            'total_cache_hits': self.total_cache_hits,
            'total_cache_misses': self.total_cache_misses,
            'max_memory_allocated_mb': max_memory_allocated,
            'max_memory_reserved_mb': max_memory_reserved,
            'overall_tokens_per_second': self.total_tokens_generated / self.total_inference_time if self.total_inference_time > 0 else 0,
        }
    
    def save_metrics(self, filepath: str) -> None:
        """Save all collected metrics to a JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert metrics to dictionaries
        metrics_data = {
            'session_summary': self.get_session_summary(),
            'individual_metrics': [asdict(metrics) for metrics in self.metrics_history],
            'metadata': {
                'session_start_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.session_start_time)),
                'session_end_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_metrics_collected': len(self.metrics_history),
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        logging.info(f"Saved {len(self.metrics_history)} metrics to {filepath}")
    
    def load_metrics(self, filepath: str) -> None:
        """Load metrics from a JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Load individual metrics
            self.metrics_history = [
                InferenceMetrics(**metrics_dict) 
                for metrics_dict in data.get('individual_metrics', [])
            ]
            
            logging.info(f"Loaded {len(self.metrics_history)} metrics from {filepath}")
            
        except Exception as e:
            logging.error(f"Failed to load metrics from {filepath}: {e}")
    
    def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        self.metrics_history.clear()
        self.total_tokens_generated = 0
        self.total_inference_time = 0.0
        self.total_cache_hits = 0
        self.total_cache_misses = 0
        self.session_start_time = time.time()
        
        logging.info("Cleared all metrics")


class PerformanceProfiler:
    """Context manager for profiling inference performance."""
    
    def __init__(self, metrics_collector: MetricsCollector, model_name: str, device: str):
        self.metrics_collector = metrics_collector
        self.model_name = model_name
        self.device = device
        self.start_time = None
        self.prefill_start = None
        self.decode_start = None
    
    def __enter__(self):
        self.metrics_collector.start_inference(self.model_name, self.device)
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.metrics_collector.finish_inference()
    
    def start_prefill(self):
        """Mark the start of prefill phase."""
        self.prefill_start = time.time()
    
    def end_prefill(self, input_length: int):
        """Mark the end of prefill phase."""
        if self.prefill_start is not None:
            prefill_time = time.time() - self.prefill_start
            self.metrics_collector.record_prefill_metrics(input_length, prefill_time)
    
    def start_decode(self):
        """Mark the start of decode phase."""
        self.decode_start = time.time()
    
    def end_decode(self, output_length: int):
        """Mark the end of decode phase."""
        if self.decode_start is not None:
            decode_time = time.time() - self.decode_start
            self.metrics_collector.record_decode_metrics(output_length, decode_time)


def compare_metrics(metrics_list: List[Dict[str, Any]], labels: List[str]) -> Dict[str, Any]:
    """Compare metrics from different experiments."""
    if len(metrics_list) != len(labels):
        raise ValueError("Number of metrics and labels must match")
    
    comparison = {}
    
    for i, (metrics, label) in enumerate(zip(metrics_list, labels)):
        comparison[label] = {
            'average_latency_ms': metrics.get('average_latency_ms', 0),
            'average_tokens_per_second': metrics.get('average_tokens_per_second', 0),
            'average_per_token_latency_ms': metrics.get('average_per_token_latency_ms', 0),
            'max_memory_allocated_mb': metrics.get('max_memory_allocated_mb', 0),
            'average_cache_hit_rate': metrics.get('average_cache_hit_rate', 0),
            'total_inferences': metrics.get('total_inferences', 0),
        }
    
    # Calculate relative improvements
    if len(metrics_list) >= 2:
        baseline_label = labels[0]
        baseline_metrics = comparison[baseline_label]
        
        for label in labels[1:]:
            current_metrics = comparison[label]
            
            # Calculate improvement percentages
            latency_improvement = (
                (baseline_metrics['average_latency_ms'] - current_metrics['average_latency_ms']) 
                / baseline_metrics['average_latency_ms'] * 100
            ) if baseline_metrics['average_latency_ms'] > 0 else 0
            
            throughput_improvement = (
                (current_metrics['average_tokens_per_second'] - baseline_metrics['average_tokens_per_second'])
                / baseline_metrics['average_tokens_per_second'] * 100
            ) if baseline_metrics['average_tokens_per_second'] > 0 else 0
            
            memory_reduction = (
                (baseline_metrics['max_memory_allocated_mb'] - current_metrics['max_memory_allocated_mb'])
                / baseline_metrics['max_memory_allocated_mb'] * 100
            ) if baseline_metrics['max_memory_allocated_mb'] > 0 else 0
            
            comparison[f"{label}_improvements"] = {
                'latency_improvement_percent': latency_improvement,
                'throughput_improvement_percent': throughput_improvement,
                'memory_reduction_percent': memory_reduction,
            }
    
    return comparison


def format_metrics_summary(metrics: Dict[str, Any]) -> str:
    """Format metrics summary as a readable string."""
    summary_lines = [
        "=" * 60,
        "PERFORMANCE METRICS SUMMARY",
        "=" * 60,
        f"Total inferences: {metrics.get('total_inferences', 0)}",
        f"Total tokens generated: {metrics.get('total_tokens_generated', 0)}",
        f"Session duration: {metrics.get('session_duration_seconds', 0):.2f} seconds",
        "",
        "LATENCY METRICS:",
        f"  Average latency: {metrics.get('average_latency_ms', 0):.2f} ms",
        f"  Median latency: {metrics.get('median_latency_ms', 0):.2f} ms",
        f"  Min latency: {metrics.get('min_latency_ms', 0):.2f} ms",
        f"  Max latency: {metrics.get('max_latency_ms', 0):.2f} ms",
        f"  Std deviation: {metrics.get('std_latency_ms', 0):.2f} ms",
        "",
        "THROUGHPUT METRICS:",
        f"  Average tokens/second: {metrics.get('average_tokens_per_second', 0):.2f}",
        f"  Overall tokens/second: {metrics.get('overall_tokens_per_second', 0):.2f}",
        f"  Average per-token latency: {metrics.get('average_per_token_latency_ms', 0):.2f} ms",
        "",
        "MEMORY METRICS:",
        f"  Max memory allocated: {metrics.get('max_memory_allocated_mb', 0):.2f} MB",
        f"  Max memory reserved: {metrics.get('max_memory_reserved_mb', 0):.2f} MB",
        "",
        "CACHE METRICS:",
        f"  Average hit rate: {metrics.get('average_cache_hit_rate', 0):.2%}",
        f"  Total cache hits: {metrics.get('total_cache_hits', 0)}",
        f"  Total cache misses: {metrics.get('total_cache_misses', 0)}",
        "=" * 60,
    ]
    
    return "\n".join(summary_lines)
