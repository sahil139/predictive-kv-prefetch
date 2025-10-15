#!/usr/bin/env python3
"""
Paged KV Cache implementation with reactive offloading.

This module implements a paged KV cache system that splits the KV cache into
manageable pages, stores them on CPU/GPU as needed, and provides reactive
offloading with LRU eviction policy.
"""

import time
import json
import logging
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from collections import OrderedDict
import threading
from pathlib import Path
import os

# Add current directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from metrics import MetricsCollector


@dataclass
class KVPage:
    """Represents a single page of KV cache."""
    page_id: int
    start_token_idx: int
    end_token_idx: int
    key_cache: Optional[torch.Tensor] = None
    value_cache: Optional[torch.Tensor] = None
    storage_location: str = "cpu"  # "cpu" or "gpu"
    last_accessed: float = 0.0
    access_count: int = 0
    size_bytes: int = 0


@dataclass
class CacheMetrics:
    """Metrics for cache performance."""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_stall_time_ms: float = 0.0
    total_offload_time_ms: float = 0.0
    total_eviction_time_ms: float = 0.0
    pages_evicted: int = 0
    pages_offloaded: int = 0
    pages_loaded: int = 0
    gpu_memory_used_mb: float = 0.0
    cpu_memory_used_mb: float = 0.0


class PagedKVCache:
    """
    Paged KV Cache with reactive offloading and LRU eviction.
    
    This class manages KV cache in pages, automatically moving pages between
    CPU and GPU memory based on access patterns and available memory.
    """
    
    def __init__(
        self,
        page_size: int = 256,
        max_gpu_pages: int = 16,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
        enable_metrics: bool = True
    ):
        """
        Initialize the paged KV cache.
        
        Args:
            page_size: Number of tokens per page
            max_gpu_pages: Maximum number of pages to keep in GPU memory
            device: Target device for GPU operations
            dtype: Data type for cache tensors
            enable_metrics: Whether to collect performance metrics
        """
        self.page_size = page_size
        self.max_gpu_pages = max_gpu_pages
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        self.enable_metrics = enable_metrics
        
        # Storage for pages
        self.pages: Dict[int, KVPage] = {}
        self.gpu_pages: OrderedDict[int, KVPage] = OrderedDict()  # LRU cache
        self.cpu_pages: Dict[int, KVPage] = {}
        
        # Mapping from token index to page ID
        self.token_to_page: Dict[int, int] = {}
        
        # Metrics
        self.metrics = CacheMetrics()
        self.metrics_collector = MetricsCollector() if enable_metrics else None
        
        # Thread safety
        self.lock = threading.RLock()
        
        logging.info(f"Initialized PagedKVCache: page_size={page_size}, max_gpu_pages={max_gpu_pages}")
        logging.info(f"Device: {self.device}, dtype: {dtype}")
    
    def _get_page_id(self, token_idx: int) -> int:
        """Get page ID for a given token index."""
        return token_idx // self.page_size
    
    def _get_page_range(self, page_id: int) -> Tuple[int, int]:
        """Get token range for a page."""
        start_idx = page_id * self.page_size
        end_idx = start_idx + self.page_size - 1
        return start_idx, end_idx
    
    def _create_empty_page(self, page_id: int, num_layers: int, num_heads: int, head_dim: int) -> KVPage:
        """Create an empty KV page."""
        start_idx, end_idx = self._get_page_range(page_id)
        
        # Create empty tensors
        key_shape = (num_layers, num_heads, self.page_size, head_dim)
        value_shape = (num_layers, num_heads, self.page_size, head_dim)
        
        key_cache = torch.zeros(key_shape, dtype=self.dtype, device='cpu')
        value_cache = torch.zeros(value_shape, dtype=self.dtype, device='cpu')
        
        # Calculate size
        size_bytes = key_cache.numel() * key_cache.element_size() + value_cache.numel() * value_cache.element_size()
        
        page = KVPage(
            page_id=page_id,
            start_token_idx=start_idx,
            end_token_idx=end_idx,
            key_cache=key_cache,
            value_cache=value_cache,
            storage_location="cpu",
            last_accessed=time.time(),
            size_bytes=size_bytes
        )
        
        return page
    
    def _move_page_to_gpu(self, page: KVPage) -> float:
        """Move a page from CPU to GPU. Returns time taken in milliseconds."""
        if page.storage_location == "gpu":
            return 0.0
        
        start_time = time.time()
        
        try:
            # Move tensors to GPU
            page.key_cache = page.key_cache.to(self.device, non_blocking=True)
            page.value_cache = page.value_cache.to(self.device, non_blocking=True)
            
            # Synchronize to ensure transfer is complete
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            
            page.storage_location = "gpu"
            page.last_accessed = time.time()
            
            # Update metrics
            if self.enable_metrics:
                self.metrics.pages_loaded += 1
            
            transfer_time = (time.time() - start_time) * 1000
            logging.debug(f"Moved page {page.page_id} to GPU in {transfer_time:.2f}ms")
            
            return transfer_time
            
        except Exception as e:
            logging.error(f"Failed to move page {page.page_id} to GPU: {e}")
            return 0.0
    
    def _move_page_to_cpu(self, page: KVPage) -> float:
        """Move a page from GPU to CPU. Returns time taken in milliseconds."""
        if page.storage_location == "cpu":
            return 0.0
        
        start_time = time.time()
        
        try:
            # Move tensors to CPU
            page.key_cache = page.key_cache.cpu()
            page.value_cache = page.value_cache.cpu()
            
            page.storage_location = "cpu"
            
            # Update metrics
            if self.enable_metrics:
                self.metrics.pages_offloaded += 1
            
            transfer_time = (time.time() - start_time) * 1000
            logging.debug(f"Moved page {page.page_id} to CPU in {transfer_time:.2f}ms")
            
            return transfer_time
            
        except Exception as e:
            logging.error(f"Failed to move page {page.page_id} to CPU: {e}")
            return 0.0
    
    def _evict_lru_page(self) -> float:
        """Evict the least recently used page from GPU. Returns time taken in milliseconds."""
        if not self.gpu_pages:
            return 0.0
        
        start_time = time.time()
        
        # Get the least recently used page (first in OrderedDict)
        page_id, page = self.gpu_pages.popitem(last=False)
        
        # Move to CPU
        offload_time = self._move_page_to_cpu(page)
        
        # Update storage
        self.cpu_pages[page_id] = page
        
        # Update metrics
        if self.enable_metrics:
            self.metrics.pages_evicted += 1
        
        eviction_time = (time.time() - start_time) * 1000
        logging.debug(f"Evicted page {page_id} in {eviction_time:.2f}ms")
        
        return eviction_time
    
    def _make_space_for_page(self, required_pages: int = 1) -> float:
        """Make space in GPU for new pages. Returns total time taken in milliseconds."""
        total_time = 0.0
        
        # Calculate how many pages we need to evict
        current_gpu_pages = len(self.gpu_pages)
        pages_to_evict = max(0, current_gpu_pages + required_pages - self.max_gpu_pages)
        
        for _ in range(pages_to_evict):
            eviction_time = self._evict_lru_page()
            total_time += eviction_time
            
            if self.enable_metrics:
                self.metrics.total_eviction_time_ms += eviction_time
        
        return total_time
    
    def get_kv_page(self, token_idx: int, num_layers: int, num_heads: int, head_dim: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Get KV cache page for a given token index.
        
        Args:
            token_idx: Token index to get cache for
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
            
        Returns:
            Tuple of (key_cache, value_cache, stall_time_ms)
        """
        with self.lock:
            start_time = time.time()
            stall_time = 0.0
            
            # Get page ID
            page_id = self._get_page_id(token_idx)
            
            # Update metrics
            if self.enable_metrics:
                self.metrics.total_requests += 1
            
            # Check if page exists
            if page_id not in self.pages:
                # Create new page
                page = self._create_empty_page(page_id, num_layers, num_heads, head_dim)
                self.pages[page_id] = page
                self.cpu_pages[page_id] = page
                
                # Update token mapping
                start_idx, end_idx = self._get_page_range(page_id)
                for idx in range(start_idx, end_idx + 1):
                    self.token_to_page[idx] = page_id
                
                logging.debug(f"Created new page {page_id} for token {token_idx}")
            
            page = self.pages[page_id]
            
            # Update access info
            page.last_accessed = time.time()
            page.access_count += 1
            
            # Check if page is in GPU
            if page.storage_location == "gpu":
                # Page is in GPU, update LRU order
                if page_id in self.gpu_pages:
                    self.gpu_pages.move_to_end(page_id)
                
                if self.enable_metrics:
                    self.metrics.cache_hits += 1
                
                logging.debug(f"Cache hit for page {page_id}")
                
            else:
                # Page is in CPU, need to load to GPU
                if self.enable_metrics:
                    self.metrics.cache_misses += 1
                
                # Make space if needed
                eviction_time = self._make_space_for_page()
                stall_time += eviction_time
                
                # Move page to GPU
                load_time = self._move_page_to_gpu(page)
                stall_time += load_time
                
                # Update storage tracking
                if page_id in self.cpu_pages:
                    del self.cpu_pages[page_id]
                
                self.gpu_pages[page_id] = page
                
                if self.enable_metrics:
                    self.metrics.total_stall_time_ms += stall_time
                    self.metrics.total_offload_time_ms += load_time
                
                logging.debug(f"Cache miss for page {page_id}, loaded in {load_time:.2f}ms")
            
            # Get the specific slice for this token
            token_offset = token_idx - page.start_token_idx
            
            # Ensure we have valid tensors
            if page.key_cache is None or page.value_cache is None:
                raise RuntimeError(f"Page {page_id} has None cache tensors")
            
            # Return the specific token's cache
            key_slice = page.key_cache[:, :, token_offset:token_offset+1, :]
            value_slice = page.value_cache[:, :, token_offset:token_offset+1, :]
            
            return key_slice, value_slice, stall_time
    
    def update_kv_cache(self, token_idx: int, key_cache: torch.Tensor, value_cache: torch.Tensor) -> None:
        """
        Update KV cache for a specific token.
        
        Args:
            token_idx: Token index to update
            key_cache: New key cache tensor
            value_cache: New value cache tensor
        """
        with self.lock:
            page_id = self._get_page_id(token_idx)
            
            if page_id not in self.pages:
                raise ValueError(f"Page {page_id} does not exist for token {token_idx}")
            
            page = self.pages[page_id]
            token_offset = token_idx - page.start_token_idx
            
            # Update the cache
            if page.key_cache is not None and page.value_cache is not None:
                page.key_cache[:, :, token_offset:token_offset+1, :] = key_cache
                page.value_cache[:, :, token_offset:token_offset+1, :] = value_cache
                
                page.last_accessed = time.time()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        gpu_memory = 0.0
        cpu_memory = 0.0
        
        for page in self.pages.values():
            if page.storage_location == "gpu":
                gpu_memory += page.size_bytes
            else:
                cpu_memory += page.size_bytes
        
        return {
            'gpu_memory_mb': gpu_memory / (1024 * 1024),
            'cpu_memory_mb': cpu_memory / (1024 * 1024),
            'total_pages': len(self.pages),
            'gpu_pages': len(self.gpu_pages),
            'cpu_pages': len(self.cpu_pages)
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        if self.enable_metrics:
            hit_rate = self.metrics.cache_hits / max(1, self.metrics.total_requests)
            miss_rate = self.metrics.cache_misses / max(1, self.metrics.total_requests)
        else:
            hit_rate = miss_rate = 0.0
        
        memory_usage = self.get_memory_usage()
        
        return {
            'cache_hit_rate': hit_rate,
            'cache_miss_rate': miss_rate,
            'total_requests': self.metrics.total_requests,
            'cache_hits': self.metrics.cache_hits,
            'cache_misses': self.metrics.cache_misses,
            'total_stall_time_ms': self.metrics.total_stall_time_ms,
            'average_stall_time_ms': self.metrics.total_stall_time_ms / max(1, self.metrics.cache_misses),
            'total_offload_time_ms': self.metrics.total_offload_time_ms,
            'total_eviction_time_ms': self.metrics.total_eviction_time_ms,
            'pages_evicted': self.metrics.pages_evicted,
            'pages_offloaded': self.metrics.pages_offloaded,
            'pages_loaded': self.metrics.pages_loaded,
            'memory_usage': memory_usage,
            'config': {
                'page_size': self.page_size,
                'max_gpu_pages': self.max_gpu_pages,
                'device': str(self.device),
                'dtype': str(self.dtype)
            }
        }
    
    def dump_metrics(self, filepath: str) -> None:
        """Dump cache metrics to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        stats = self.get_cache_stats()
        
        # Add timestamp
        stats['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        stats['experiment_type'] = 'paged_kv_cache'
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logging.info(f"Cache metrics dumped to {filepath}")
    
    def clear_cache(self) -> None:
        """Clear all cached pages."""
        with self.lock:
            self.pages.clear()
            self.gpu_pages.clear()
            self.cpu_pages.clear()
            self.token_to_page.clear()
            
            # Reset metrics
            self.metrics = CacheMetrics()
            
            logging.info("Cache cleared")
    
    def warmup(self, num_tokens: int, num_layers: int, num_heads: int, head_dim: int) -> None:
        """Warm up the cache by pre-allocating pages."""
        logging.info(f"Warming up cache for {num_tokens} tokens")
        
        num_pages = (num_tokens + self.page_size - 1) // self.page_size
        
        for page_id in range(num_pages):
            if page_id not in self.pages:
                page = self._create_empty_page(page_id, num_layers, num_heads, head_dim)
                self.pages[page_id] = page
                self.cpu_pages[page_id] = page
                
                # Update token mapping
                start_idx, end_idx = self._get_page_range(page_id)
                for idx in range(start_idx, min(end_idx + 1, num_tokens)):
                    self.token_to_page[idx] = page_id
        
        logging.info(f"Cache warmup complete: {len(self.pages)} pages created")


class PagedKVCacheManager:
    """
    Manager class for integrating paged KV cache with transformer models.
    
    This class provides a higher-level interface for using paged KV cache
    with Hugging Face transformers.
    """
    
    def __init__(self, cache: PagedKVCache):
        self.cache = cache
        self.current_context_length = 0
        self.model_config = None
    
    def set_model_config(self, config: Dict[str, Any]) -> None:
        """Set model configuration for cache management."""
        self.model_config = config
    
    def get_past_key_values(self, input_ids: torch.Tensor, layer_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get past key values for attention computation.
        
        Args:
            input_ids: Input token IDs
            layer_idx: Layer index (for multi-layer models)
            
        Returns:
            Tuple of (past_key_values, stall_time_ms)
        """
        if self.model_config is None:
            raise ValueError("Model config not set")
        
        batch_size, seq_len = input_ids.shape
        num_layers = self.model_config.get('num_layers', 1)
        num_heads = self.model_config.get('num_heads', 8)
        head_dim = self.model_config.get('head_dim', 64)
        
        total_stall_time = 0.0
        
        # Get cache for each token in the sequence
        past_keys = []
        past_values = []
        
        for token_idx in range(seq_len):
            key_slice, value_slice, stall_time = self.cache.get_kv_page(
                token_idx, num_layers, num_heads, head_dim
            )
            total_stall_time += stall_time
            
            past_keys.append(key_slice)
            past_values.append(value_slice)
        
        # Concatenate along sequence dimension
        if past_keys:
            past_key_values = torch.cat(past_keys, dim=2)
            past_value_values = torch.cat(past_values, dim=2)
        else:
            past_key_values = torch.empty(0)
            past_value_values = torch.empty(0)
        
        return past_key_values, past_value_values, total_stall_time
    
    def update_past_key_values(self, input_ids: torch.Tensor, new_keys: torch.Tensor, new_values: torch.Tensor) -> None:
        """Update past key values in the cache."""
        batch_size, seq_len = input_ids.shape
        
        for token_idx in range(seq_len):
            key_slice = new_keys[:, :, token_idx:token_idx+1, :]
            value_slice = new_values[:, :, token_idx:token_idx+1, :]
            
            self.cache.update_kv_cache(token_idx, key_slice, value_slice)
        
        self.current_context_length = max(self.current_context_length, seq_len)


def create_paged_cache(
    page_size: int = 256,
    max_gpu_pages: int = 16,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float16
) -> PagedKVCache:
    """Factory function to create a paged KV cache."""
    return PagedKVCache(
        page_size=page_size,
        max_gpu_pages=max_gpu_pages,
        device=device,
        dtype=dtype,
        enable_metrics=True
    )


if __name__ == "__main__":
    # Example usage and testing
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    # Create cache
    cache = create_paged_cache(page_size=128, max_gpu_pages=8)
    
    # Test parameters
    num_layers = 2
    num_heads = 8
    head_dim = 64
    
    print("Testing PagedKVCache...")
    
    # Test cache operations
    for i in range(10):
        key, value, stall_time = cache.get_kv_page(i, num_layers, num_heads, head_dim)
        print(f"Token {i}: stall_time={stall_time:.2f}ms, shape={key.shape}")
    
    # Print stats
    stats = cache.get_cache_stats()
    print(f"\nCache Stats:")
    print(f"Hit rate: {stats['cache_hit_rate']:.2%}")
    print(f"Total requests: {stats['total_requests']}")
    print(f"Memory usage: {stats['memory_usage']}")
    
    # Dump metrics
    cache.dump_metrics("logs/paged_cache_test.json")
    print("Test completed!")
