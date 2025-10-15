#!/usr/bin/env python3
"""
Utility functions for KV cache experiments.

This module provides helper functions for model loading, configuration management,
logging setup, and device information.
"""

import json
import logging
import os
import sys
import torch
from pathlib import Path
from typing import Dict, Any, Optional
import platform
import psutil


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """Setup logging configuration."""
    # Create logs directory if it doesn't exist
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logging.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logging.warning(f"Config file {config_path} not found, using defaults")
        return get_default_config()
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in config file {config_path}: {e}")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return {
        'model_name': 'meta-llama/Llama-2-7b-hf',
        'batch_size': 1,
        'max_context_length': 4096,
        'quantization_config': {
            'load_in_8bit': False,
            'load_in_4bit': False
        },
        'device': 'auto',
        'torch_dtype': 'float16',
        'trust_remote_code': True,
        'use_cache': True,
        'do_sample': False,
        'temperature': 1.0,
        'top_p': 1.0,
        'top_k': 50,
        'repetition_penalty': 1.0,
        'max_new_tokens': 100,
        'pad_token_id': None,
        'eos_token_id': None,
    }


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to JSON file."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logging.info(f"Configuration saved to {config_path}")


def get_device_info() -> Dict[str, Any]:
    """Get information about available devices."""
    device_info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cpu_count': psutil.cpu_count(),
        'memory_total_gb': psutil.virtual_memory().total / (1024**3),
        'memory_available_gb': psutil.virtual_memory().available / (1024**3),
    }
    
    if torch.cuda.is_available():
        device_info.update({
            'cuda_version': torch.version.cuda,
            'cuda_device_count': torch.cuda.device_count(),
            'cuda_devices': []
        })
        
        for i in range(torch.cuda.device_count()):
            device_info['cuda_devices'].append({
                'device_id': i,
                'name': torch.cuda.get_device_name(i),
                'memory_total_gb': torch.cuda.get_device_properties(i).total_memory / (1024**3),
                'memory_allocated_gb': torch.cuda.memory_allocated(i) / (1024**3),
                'memory_reserved_gb': torch.cuda.memory_reserved(i) / (1024**3),
            })
    
    return device_info


def get_optimal_device() -> torch.device:
    """Get the optimal device for inference."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def format_bytes(bytes_value: int) -> str:
    """Format bytes into human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def format_time(seconds: float) -> str:
    """Format time in seconds into human readable string."""
    if seconds < 1:
        return f"{seconds*1000:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    elif seconds < 3600:
        return f"{seconds/60:.2f} min"
    else:
        return f"{seconds/3600:.2f} h"


def ensure_directory(path: str) -> None:
    """Ensure directory exists, create if it doesn't."""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_model_size_info(model) -> Dict[str, Any]:
    """Get information about model size and parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate model size in bytes
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_bytes = param_size + buffer_size
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_bytes': model_size_bytes,
        'model_size_mb': model_size_bytes / (1024 * 1024),
        'model_size_gb': model_size_bytes / (1024 * 1024 * 1024),
    }


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration parameters."""
    required_keys = ['model_name', 'batch_size', 'max_context_length']
    
    for key in required_keys:
        if key not in config:
            logging.error(f"Missing required config key: {key}")
            return False
    
    # Validate batch size
    if not isinstance(config['batch_size'], int) or config['batch_size'] < 1:
        logging.error("batch_size must be a positive integer")
        return False
    
    # Validate max context length
    if not isinstance(config['max_context_length'], int) or config['max_context_length'] < 1:
        logging.error("max_context_length must be a positive integer")
        return False
    
    # Validate quantization config
    if 'quantization_config' in config:
        quant_config = config['quantization_config']
        if isinstance(quant_config, dict):
            if 'load_in_8bit' in quant_config and 'load_in_4bit' in quant_config:
                if quant_config['load_in_8bit'] and quant_config['load_in_4bit']:
                    logging.error("Cannot use both 8-bit and 4-bit quantization")
                    return False
    
    return True


def create_experiment_directory(experiment_name: str) -> str:
    """Create directory for experiment results."""
    base_dir = "logs"
    experiment_dir = os.path.join(base_dir, experiment_name)
    ensure_directory(experiment_dir)
    return experiment_dir


def get_timestamp() -> str:
    """Get current timestamp as string."""
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_prompts_from_file(file_path: str) -> list:
    """Load prompts from a text file (one prompt per line)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        logging.info(f"Loaded {len(prompts)} prompts from {file_path}")
        return prompts
    except FileNotFoundError:
        logging.error(f"Prompts file {file_path} not found")
        return []
    except Exception as e:
        logging.error(f"Error loading prompts from {file_path}: {e}")
        return []


def save_prompts_to_file(prompts: list, file_path: str) -> None:
    """Save prompts to a text file (one prompt per line)."""
    ensure_directory(os.path.dirname(file_path))
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for prompt in prompts:
            f.write(prompt + '\n')
    
    logging.info(f"Saved {len(prompts)} prompts to {file_path}")


def get_sample_prompts() -> list:
    """Get a list of sample prompts for testing."""
    return [
        "The future of artificial intelligence is",
        "In a world where technology advances rapidly,",
        "The key to solving climate change lies in",
        "When I think about the meaning of life,",
        "The most important skill for the 21st century is",
        "If I could travel back in time, I would",
        "The secret to happiness is",
        "What makes a great leader is",
        "The impact of social media on society is",
        "The most beautiful thing about nature is",
    ]


def cleanup_gpu_memory():
    """Clean up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics."""
    memory_info = psutil.virtual_memory()
    
    result = {
        'system_memory_total_gb': memory_info.total / (1024**3),
        'system_memory_available_gb': memory_info.available / (1024**3),
        'system_memory_used_gb': memory_info.used / (1024**3),
        'system_memory_percent': memory_info.percent,
    }
    
    if torch.cuda.is_available():
        result.update({
            'gpu_memory_allocated_gb': torch.cuda.memory_allocated() / (1024**3),
            'gpu_memory_reserved_gb': torch.cuda.memory_reserved() / (1024**3),
            'gpu_memory_max_allocated_gb': torch.cuda.max_memory_allocated() / (1024**3),
            'gpu_memory_max_reserved_gb': torch.cuda.max_memory_reserved() / (1024**3),
        })
    
    return result
