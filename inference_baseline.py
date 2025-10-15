#!/usr/bin/env python3
"""
Baseline inference script for KV cache experiments.

This script loads a specified model from Hugging Face and runs autoregressive
inference with full KV cache in GPU memory, measuring performance metrics.
"""

import argparse
import json
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional, Tuple
import logging
import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import setup_logging, load_config, get_device_info
from metrics import MetricsCollector


class BaselineInference:
    """Baseline inference with full KV cache in GPU memory."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_name = config.get('model_name', 'meta-llama/Llama-2-7b-hf')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = config.get('batch_size', 1)
        self.max_context_length = config.get('max_context_length', 4096)
        self.quantization_config = config.get('quantization_config', None)
        
        # Initialize metrics collector
        self.metrics = MetricsCollector()
        
        # Model and tokenizer will be loaded in load_model()
        self.model = None
        self.tokenizer = None
        
        logging.info(f"Initialized BaselineInference with model: {self.model_name}")
        logging.info(f"Device: {self.device}")
        logging.info(f"Batch size: {self.batch_size}")
        logging.info(f"Max context length: {self.max_context_length}")
    
    def load_model(self) -> None:
        """Load the model and tokenizer from Hugging Face."""
        logging.info(f"Loading model: {self.model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side='left'
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optional quantization
            model_kwargs = {
                'torch_dtype': torch.float16,
                'device_map': 'auto',
                'trust_remote_code': True,
            }
            
            if self.quantization_config:
                if self.quantization_config.get('load_in_8bit', False):
                    model_kwargs['load_in_8bit'] = True
                elif self.quantization_config.get('load_in_4bit', False):
                    model_kwargs['load_in_4bit'] = True
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            self.model.eval()
            
            logging.info(f"Model loaded successfully")
            logging.info(f"Model device: {next(self.model.parameters()).device}")
            
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise
    
    def measure_memory_usage(self) -> Dict[str, float]:
        """Measure current GPU memory usage."""
        if torch.cuda.is_available():
            return {
                'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                'reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
                'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
                'max_reserved_mb': torch.cuda.max_memory_reserved() / 1024 / 1024,
            }
        return {}
    
    def find_max_context_length(self) -> int:
        """Find the maximum context length before OOM."""
        logging.info("Finding maximum context length...")
        
        # Start with a reasonable context length
        test_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768]
        max_length = 0
        
        for length in test_lengths:
            try:
                # Create dummy input
                dummy_input = torch.randint(
                    0, self.tokenizer.vocab_size, 
                    (self.batch_size, length), 
                    device=self.device
                )
                
                # Try forward pass
                with torch.no_grad():
                    _ = self.model(dummy_input)
                
                max_length = length
                logging.info(f"Context length {length} works")
                
                # Clear cache
                torch.cuda.empty_cache()
                
            except torch.cuda.OutOfMemoryError:
                logging.info(f"OOM at context length {length}")
                break
            except Exception as e:
                logging.warning(f"Error at context length {length}: {e}")
                break
        
        return max_length
    
    def run_inference(self, prompt: str, max_new_tokens: int = 100) -> Tuple[str, Dict]:
        """Run autoregressive inference on a prompt."""
        logging.info(f"Running inference on prompt: {prompt[:50]}...")
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt, 
            return_tensors='pt', 
            padding=True,
            truncation=True,
            max_length=self.max_context_length
        ).to(self.device)
        
        input_length = inputs.input_ids.shape[1]
        logging.info(f"Input length: {input_length}")
        
        # Measure memory before inference
        memory_before = self.measure_memory_usage()
        
        # Run inference
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,  # Enable KV cache
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        total_time = time.time() - start_time
        
        # Measure memory after inference
        memory_after = self.measure_memory_usage()
        
        # Decode output
        generated_text = self.tokenizer.decode(
            outputs.sequences[0][input_length:], 
            skip_special_tokens=True
        )
        
        # Calculate per-token latency
        num_new_tokens = outputs.sequences.shape[1] - input_length
        per_token_latency = total_time / num_new_tokens if num_new_tokens > 0 else 0
        
        # Collect metrics
        metrics = {
            'total_latency_ms': total_time * 1000,
            'per_token_latency_ms': per_token_latency * 1000,
            'num_input_tokens': input_length,
            'num_output_tokens': num_new_tokens,
            'tokens_per_second': num_new_tokens / total_time if total_time > 0 else 0,
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_delta_mb': {
                k: memory_after.get(k, 0) - memory_before.get(k, 0) 
                for k in memory_after.keys()
            }
        }
        
        logging.info(f"Inference completed in {total_time:.2f}s")
        logging.info(f"Generated {num_new_tokens} tokens")
        logging.info(f"Per-token latency: {per_token_latency*1000:.2f}ms")
        
        return generated_text, metrics
    
    def run_benchmark(self, prompts: List[str], max_new_tokens: int = 100) -> Dict:
        """Run inference benchmark on multiple prompts."""
        logging.info(f"Starting benchmark with {len(prompts)} prompts")
        
        all_metrics = []
        total_tokens = 0
        total_time = 0
        
        for i, prompt in enumerate(prompts):
            logging.info(f"Processing prompt {i+1}/{len(prompts)}")
            
            try:
                generated_text, metrics = self.run_inference(prompt, max_new_tokens)
                all_metrics.append(metrics)
                
                total_tokens += metrics['num_output_tokens']
                total_time += metrics['total_latency_ms'] / 1000
                
                # Clear cache between prompts
                torch.cuda.empty_cache()
                
            except Exception as e:
                logging.error(f"Error processing prompt {i+1}: {e}")
                continue
        
        # Calculate aggregate metrics
        if all_metrics:
            avg_per_token_latency = sum(m['per_token_latency_ms'] for m in all_metrics) / len(all_metrics)
            avg_tokens_per_second = sum(m['tokens_per_second'] for m in all_metrics) / len(all_metrics)
            
            # Find max memory usage
            max_memory = {}
            for key in ['allocated_mb', 'reserved_mb', 'max_allocated_mb', 'max_reserved_mb']:
                max_memory[key] = max(m['memory_after_mb'].get(key, 0) for m in all_metrics)
        else:
            avg_per_token_latency = 0
            avg_tokens_per_second = 0
            max_memory = {}
        
        # Find maximum context length
        max_context_length = self.find_max_context_length()
        
        benchmark_results = {
            'model_name': self.model_name,
            'device': str(self.device),
            'batch_size': self.batch_size,
            'num_prompts': len(prompts),
            'total_tokens_generated': total_tokens,
            'total_time_seconds': total_time,
            'average_per_token_latency_ms': avg_per_token_latency,
            'average_tokens_per_second': avg_tokens_per_second,
            'max_context_length': max_context_length,
            'max_memory_usage_mb': max_memory,
            'per_prompt_metrics': all_metrics,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        return benchmark_results
    
    def save_metrics(self, metrics: Dict, output_path: str) -> None:
        """Save metrics to JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logging.info(f"Metrics saved to {output_path}")


def main():
    """Main function to run baseline inference."""
    parser = argparse.ArgumentParser(description='Run baseline inference with KV cache')
    parser.add_argument('--config', type=str, default='configs/baseline_config.json',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, help='Model name (overrides config)')
    parser.add_argument('--batch-size', type=int, help='Batch size (overrides config)')
    parser.add_argument('--max-context', type=int, help='Max context length (overrides config)')
    parser.add_argument('--output', type=str, default='logs/baseline_metrics.json',
                       help='Output file for metrics')
    parser.add_argument('--prompts', type=str, nargs='+', 
                       default=["The future of artificial intelligence is"],
                       help='Prompts to test')
    parser.add_argument('--max-new-tokens', type=int, default=100,
                       help='Maximum new tokens to generate')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.model:
        config['model_name'] = args.model
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.max_context:
        config['max_context_length'] = args.max_context
    
    logging.info("Starting baseline inference benchmark")
    logging.info(f"Configuration: {config}")
    
    try:
        # Initialize inference engine
        inference = BaselineInference(config)
        
        # Load model
        inference.load_model()
        
        # Run benchmark
        metrics = inference.run_benchmark(args.prompts, args.max_new_tokens)
        
        # Save metrics
        inference.save_metrics(metrics, args.output)
        
        # Print summary
        print("\n" + "="*50)
        print("BASELINE INFERENCE RESULTS")
        print("="*50)
        print(f"Model: {metrics['model_name']}")
        print(f"Device: {metrics['device']}")
        print(f"Average per-token latency: {metrics['average_per_token_latency_ms']:.2f} ms")
        print(f"Average tokens per second: {metrics['average_tokens_per_second']:.2f}")
        print(f"Max context length: {metrics['max_context_length']}")
        print(f"Max memory usage: {metrics['max_memory_usage_mb'].get('allocated_mb', 0):.2f} MB")
        print(f"Total tokens generated: {metrics['total_tokens_generated']}")
        print(f"Total time: {metrics['total_time_seconds']:.2f} seconds")
        print(f"Metrics saved to: {args.output}")
        print("="*50)
        
    except Exception as e:
        logging.error(f"Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
