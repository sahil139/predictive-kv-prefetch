#!/usr/bin/env python3
"""
Main orchestrator for KV cache prefetch experiments.

This script coordinates experiments across different KV cache strategies:
- Baseline: Standard KV caching
- Reactive: Paged KV cache with reactive offloading
- Predictive: Paged KV cache with predictive prefetching

It parses experiment configurations, runs experiments, aggregates metrics,
and generates comparative analysis.
"""

import argparse
import json
import logging
import time
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import subprocess

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference_baseline import BaselineInference
from kv_paging import create_paged_cache, PagedKVCache
from predictive_prefetch import create_predictive_prefetcher, PredictiveInferenceEngine
from utils import setup_logging, load_config, get_device_info, get_sample_prompts
from metrics import MetricsCollector, format_metrics_summary


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    mode: str  # 'baseline', 'reactive', 'predictive'
    model_name: str
    page_size: int = 256
    max_gpu_pages: int = 16
    prefetch_k: int = 5
    context_lengths: List[int] = None
    num_runs: int = 3
    max_new_tokens: int = 100
    batch_size: int = 1
    quantization_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context_lengths is None:
            self.context_lengths = [100, 500, 1000]
        if self.quantization_config is None:
            self.quantization_config = {"load_in_8bit": False, "load_in_4bit": False}


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    config: ExperimentConfig
    run_id: int
    metrics: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None
    duration_seconds: float = 0.0


class ExperimentRunner:
    """Runs individual experiments for different KV cache strategies."""
    
    def __init__(self, output_dir: str = "logs"):
        self.output_dir = output_dir
        self.device_info = get_device_info()
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        logging.info(f"Initialized ExperimentRunner with output directory: {output_dir}")
    
    def run_baseline_experiment(self, config: ExperimentConfig, run_id: int) -> ExperimentResult:
        """Run baseline inference experiment."""
        logging.info(f"Running baseline experiment {run_id} with model {config.model_name}")
        
        start_time = time.time()
        
        try:
            # Create baseline configuration
            baseline_config = {
                'model_name': config.model_name,
                'batch_size': config.batch_size,
                'max_context_length': max(config.context_lengths),
                'quantization_config': config.quantization_config,
                'max_new_tokens': config.max_new_tokens
            }
            
            # Initialize baseline inference
            inference = BaselineInference(baseline_config)
            inference.load_model()
            
            # Get sample prompts
            prompts = get_sample_prompts()[:5]  # Use first 5 prompts
            
            # Run benchmark
            metrics = inference.run_benchmark(prompts, config.max_new_tokens)
            
            # Add experiment metadata
            metrics.update({
                'experiment_mode': 'baseline',
                'experiment_config': asdict(config),
                'run_id': run_id,
                'device_info': self.device_info,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            })
            
            # Save metrics
            filename = f"baseline_{config.model_name.replace('/', '_')}_run{run_id}.json"
            filepath = os.path.join(self.output_dir, filename)
            inference.save_metrics(metrics, filepath)
            
            duration = time.time() - start_time
            
            logging.info(f"Baseline experiment {run_id} completed in {duration:.2f}s")
            
            return ExperimentResult(
                config=config,
                run_id=run_id,
                metrics=metrics,
                success=True,
                duration_seconds=duration
            )
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Baseline experiment {run_id} failed: {str(e)}"
            logging.error(error_msg)
            
            return ExperimentResult(
                config=config,
                run_id=run_id,
                metrics={},
                success=False,
                error_message=error_msg,
                duration_seconds=duration
            )
    
    def run_reactive_experiment(self, config: ExperimentConfig, run_id: int) -> ExperimentResult:
        """Run reactive paged KV cache experiment."""
        logging.info(f"Running reactive experiment {run_id} with page_size {config.page_size}")
        
        start_time = time.time()
        
        try:
            # Create paged cache
            cache = create_paged_cache(
                page_size=config.page_size,
                max_gpu_pages=config.max_gpu_pages
            )
            
            # Create mock inference engine (simplified for testing)
            from test_kv_paging import PagedInferenceEngine
            engine = PagedInferenceEngine(cache, num_layers=2)
            
            # Test with different context lengths
            all_metrics = []
            
            for context_length in config.context_lengths:
                logging.info(f"Testing reactive with context length {context_length}")
                
                # Create mock input
                input_ids = torch.randint(0, 1000, (config.batch_size, context_length))
                
                # Run inference
                metrics = engine.run_inference(input_ids, config.max_new_tokens)
                
                # Add context length info
                metrics['context_length'] = context_length
                all_metrics.append(metrics)
            
            # Aggregate metrics
            aggregated_metrics = self._aggregate_reactive_metrics(all_metrics)
            aggregated_metrics.update({
                'experiment_mode': 'reactive',
                'experiment_config': asdict(config),
                'run_id': run_id,
                'device_info': self.device_info,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'per_context_metrics': all_metrics
            })
            
            # Save metrics
            filename = f"reactive_page{config.page_size}_run{run_id}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(aggregated_metrics, f, indent=2)
            
            # Also save cache metrics
            cache.dump_metrics(os.path.join(self.output_dir, f"reactive_cache_run{run_id}.json"))
            
            duration = time.time() - start_time
            
            logging.info(f"Reactive experiment {run_id} completed in {duration:.2f}s")
            
            return ExperimentResult(
                config=config,
                run_id=run_id,
                metrics=aggregated_metrics,
                success=True,
                duration_seconds=duration
            )
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Reactive experiment {run_id} failed: {str(e)}"
            logging.error(error_msg)
            
            return ExperimentResult(
                config=config,
                run_id=run_id,
                metrics={},
                success=False,
                error_message=error_msg,
                duration_seconds=duration
            )
    
    def run_predictive_experiment(self, config: ExperimentConfig, run_id: int) -> ExperimentResult:
        """Run predictive prefetching experiment."""
        logging.info(f"Running predictive experiment {run_id} with prefetch_k {config.prefetch_k}")
        
        start_time = time.time()
        
        try:
            # Create paged cache
            cache = create_paged_cache(
                page_size=config.page_size,
                max_gpu_pages=config.max_gpu_pages
            )
            
            # Create predictive prefetcher
            prefetcher = create_predictive_prefetcher(
                cache=cache,
                page_size=config.page_size,
                prefetch_window=3,
                max_prefetch_pages=config.prefetch_k,
                confidence_threshold=0.3
            )
            
            # Create inference engine
            engine = PredictiveInferenceEngine(cache, prefetcher, num_layers=2)
            
            # Test with different context lengths
            all_metrics = []
            
            for context_length in config.context_lengths:
                logging.info(f"Testing predictive with context length {context_length}")
                
                # Create mock input
                input_ids = torch.randint(0, 1000, (config.batch_size, context_length))
                
                # Run inference with prefetching
                metrics = engine.run_inference_with_prefetch(input_ids, config.max_new_tokens)
                
                # Add context length info
                metrics['context_length'] = context_length
                all_metrics.append(metrics)
            
            # Aggregate metrics
            aggregated_metrics = self._aggregate_predictive_metrics(all_metrics)
            aggregated_metrics.update({
                'experiment_mode': 'predictive',
                'experiment_config': asdict(config),
                'run_id': run_id,
                'device_info': self.device_info,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'per_context_metrics': all_metrics
            })
            
            # Save metrics
            filename = f"predictive_k{config.prefetch_k}_run{run_id}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(aggregated_metrics, f, indent=2)
            
            # Also save prefetch metrics
            prefetcher.dump_prefetch_metrics(os.path.join(self.output_dir, f"predictive_prefetch_run{run_id}.json"))
            
            duration = time.time() - start_time
            
            logging.info(f"Predictive experiment {run_id} completed in {duration:.2f}s")
            
            return ExperimentResult(
                config=config,
                run_id=run_id,
                metrics=aggregated_metrics,
                success=True,
                duration_seconds=duration
            )
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Predictive experiment {run_id} failed: {str(e)}"
            logging.error(error_msg)
            
            return ExperimentResult(
                config=config,
                run_id=run_id,
                metrics={},
                success=False,
                error_message=error_msg,
                duration_seconds=duration
            )
    
    def _aggregate_reactive_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics from multiple reactive experiment runs."""
        if not metrics_list:
            return {}
        
        # Calculate averages across context lengths
        avg_latency = np.mean([m.get('total_time_ms', 0) for m in metrics_list])
        avg_tokens_per_sec = np.mean([m.get('tokens_per_second', 0) for m in metrics_list])
        avg_stall_time = np.mean([m.get('total_stall_time_ms', 0) for m in metrics_list])
        
        # Cache statistics
        cache_hit_rates = [m.get('cache_stats', {}).get('cache_hit_rate', 0) for m in metrics_list]
        avg_cache_hit_rate = np.mean(cache_hit_rates) if cache_hit_rates else 0
        
        return {
            'average_latency_ms': avg_latency,
            'average_tokens_per_second': avg_tokens_per_sec,
            'average_stall_time_ms': avg_stall_time,
            'average_cache_hit_rate': avg_cache_hit_rate,
            'stall_fraction': avg_stall_time / max(avg_latency, 1) if avg_latency > 0 else 0,
            'num_context_lengths_tested': len(metrics_list)
        }
    
    def _aggregate_predictive_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics from multiple predictive experiment runs."""
        if not metrics_list:
            return {}
        
        # Calculate averages across context lengths
        avg_latency = np.mean([m.get('total_time_ms', 0) for m in metrics_list])
        avg_tokens_per_sec = np.mean([m.get('tokens_per_second', 0) for m in metrics_list])
        avg_prefetch_time = np.mean([m.get('total_prefetch_time_ms', 0) for m in metrics_list])
        avg_prefetch_overhead = np.mean([m.get('prefetch_overhead_percent', 0) for m in metrics_list])
        
        # Cache statistics
        cache_hit_rates = [m.get('cache_stats', {}).get('cache_hit_rate', 0) for m in metrics_list]
        avg_cache_hit_rate = np.mean(cache_hit_rates) if cache_hit_rates else 0
        
        # Prefetch statistics
        prefetch_hit_rates = [m.get('prefetch_stats', {}).get('prefetch_hit_rate', 0) for m in metrics_list]
        avg_prefetch_hit_rate = np.mean(prefetch_hit_rates) if prefetch_hit_rates else 0
        
        prediction_accuracies = [m.get('prefetch_stats', {}).get('prediction_accuracy', 0) for m in metrics_list]
        avg_prediction_accuracy = np.mean(prediction_accuracies) if prediction_accuracies else 0
        
        return {
            'average_latency_ms': avg_latency,
            'average_tokens_per_second': avg_tokens_per_sec,
            'average_prefetch_time_ms': avg_prefetch_time,
            'average_prefetch_overhead_percent': avg_prefetch_overhead,
            'average_cache_hit_rate': avg_cache_hit_rate,
            'average_prefetch_hit_rate': avg_prefetch_hit_rate,
            'average_prediction_accuracy': avg_prediction_accuracy,
            'stall_fraction': avg_prefetch_overhead / 100.0,
            'num_context_lengths_tested': len(metrics_list)
        }


class ExperimentOrchestrator:
    """Main orchestrator for running all experiments."""
    
    def __init__(self, config_file: str, output_dir: str = "logs"):
        self.config_file = config_file
        self.output_dir = output_dir
        self.runner = ExperimentRunner(output_dir)
        self.results: List[ExperimentResult] = []
        
        # Load experiment configuration
        self.experiment_configs = self._load_experiment_config()
        
        logging.info(f"Initialized ExperimentOrchestrator with {len(self.experiment_configs)} experiment configurations")
    
    def _load_experiment_config(self) -> List[ExperimentConfig]:
        """Load experiment configuration from YAML file."""
        try:
            with open(self.config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            experiment_configs = []
            
            # Parse each experiment mode
            for mode_config in config_data.get('experiments', []):
                mode = mode_config['mode']
                
                # Create base configuration
                base_config = ExperimentConfig(
                    mode=mode,
                    model_name=mode_config.get('model_name', 'microsoft/DialoGPT-small'),
                    page_size=mode_config.get('page_size', 256),
                    max_gpu_pages=mode_config.get('max_gpu_pages', 16),
                    prefetch_k=mode_config.get('prefetch_k', 5),
                    context_lengths=mode_config.get('context_lengths', [100, 500, 1000]),
                    num_runs=mode_config.get('num_runs', 3),
                    max_new_tokens=mode_config.get('max_new_tokens', 100),
                    batch_size=mode_config.get('batch_size', 1),
                    quantization_config=mode_config.get('quantization_config', {})
                )
                
                experiment_configs.append(base_config)
            
            logging.info(f"Loaded {len(experiment_configs)} experiment configurations")
            return experiment_configs
            
        except Exception as e:
            logging.error(f"Failed to load experiment config: {e}")
            raise
    
    def run_all_experiments(self) -> List[ExperimentResult]:
        """Run all configured experiments."""
        logging.info("Starting experiment orchestration")
        
        total_experiments = sum(config.num_runs for config in self.experiment_configs)
        completed_experiments = 0
        
        for config in self.experiment_configs:
            logging.info(f"Running {config.num_runs} experiments for mode: {config.mode}")
            
            for run_id in range(config.num_runs):
                logging.info(f"Starting {config.mode} experiment run {run_id + 1}/{config.num_runs}")
                
                # Run appropriate experiment
                if config.mode == 'baseline':
                    result = self.runner.run_baseline_experiment(config, run_id)
                elif config.mode == 'reactive':
                    result = self.runner.run_reactive_experiment(config, run_id)
                elif config.mode == 'predictive':
                    result = self.runner.run_predictive_experiment(config, run_id)
                else:
                    logging.error(f"Unknown experiment mode: {config.mode}")
                    continue
                
                self.results.append(result)
                completed_experiments += 1
                
                # Print progress
                progress = (completed_experiments / total_experiments) * 100
                logging.info(f"Progress: {completed_experiments}/{total_experiments} ({progress:.1f}%)")
        
        logging.info("All experiments completed")
        return self.results
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive summary of all experiments."""
        if not self.results:
            return {}
        
        # Group results by mode
        results_by_mode = {}
        for result in self.results:
            if result.success:
                mode = result.config.mode
                if mode not in results_by_mode:
                    results_by_mode[mode] = []
                results_by_mode[mode].append(result)
        
        # Calculate summary statistics
        summary = {
            'experiment_summary': {
                'total_experiments': len(self.results),
                'successful_experiments': len([r for r in self.results if r.success]),
                'failed_experiments': len([r for r in self.results if not r.success]),
                'total_duration_seconds': sum(r.duration_seconds for r in self.results)
            },
            'mode_summaries': {},
            'comparative_analysis': {}
        }
        
        # Generate mode summaries
        for mode, mode_results in results_by_mode.items():
            if not mode_results:
                continue
            
            # Calculate averages
            avg_latency = np.mean([r.metrics.get('average_latency_ms', 0) for r in mode_results])
            avg_tokens_per_sec = np.mean([r.metrics.get('average_tokens_per_second', 0) for r in mode_results])
            avg_stall_fraction = np.mean([r.metrics.get('stall_fraction', 0) for r in mode_results])
            
            mode_summary = {
                'num_runs': len(mode_results),
                'average_latency_ms': avg_latency,
                'average_tokens_per_second': avg_tokens_per_sec,
                'average_stall_fraction': avg_stall_fraction,
                'configurations': [asdict(r.config) for r in mode_results]
            }
            
            # Add mode-specific metrics
            if mode == 'reactive':
                avg_cache_hit_rate = np.mean([r.metrics.get('average_cache_hit_rate', 0) for r in mode_results])
                mode_summary['average_cache_hit_rate'] = avg_cache_hit_rate
            elif mode == 'predictive':
                avg_prefetch_hit_rate = np.mean([r.metrics.get('average_prefetch_hit_rate', 0) for r in mode_results])
                avg_prediction_accuracy = np.mean([r.metrics.get('average_prediction_accuracy', 0) for r in mode_results])
                mode_summary['average_prefetch_hit_rate'] = avg_prefetch_hit_rate
                mode_summary['average_prediction_accuracy'] = avg_prediction_accuracy
            
            summary['mode_summaries'][mode] = mode_summary
        
        # Generate comparative analysis
        if 'baseline' in results_by_mode and len(results_by_mode) > 1:
            baseline_results = results_by_mode['baseline']
            baseline_latency = np.mean([r.metrics.get('average_latency_ms', 0) for r in baseline_results])
            baseline_tokens_per_sec = np.mean([r.metrics.get('average_tokens_per_second', 0) for r in baseline_results])
            
            comparative_analysis = {
                'baseline_reference': {
                    'average_latency_ms': baseline_latency,
                    'average_tokens_per_second': baseline_tokens_per_sec
                }
            }
            
            # Compare other modes to baseline
            for mode, mode_results in results_by_mode.items():
                if mode == 'baseline':
                    continue
                
                mode_latency = np.mean([r.metrics.get('average_latency_ms', 0) for r in mode_results])
                mode_tokens_per_sec = np.mean([r.metrics.get('average_tokens_per_second', 0) for r in mode_results])
                
                latency_speedup = ((baseline_latency - mode_latency) / baseline_latency) * 100 if baseline_latency > 0 else 0
                throughput_speedup = ((mode_tokens_per_sec - baseline_tokens_per_sec) / baseline_tokens_per_sec) * 100 if baseline_tokens_per_sec > 0 else 0
                
                comparative_analysis[mode] = {
                    'average_latency_ms': mode_latency,
                    'average_tokens_per_second': mode_tokens_per_sec,
                    'latency_speedup_percent': latency_speedup,
                    'throughput_speedup_percent': throughput_speedup,
                    'relative_performance': {
                        'latency_improvement': latency_speedup > 0,
                        'throughput_improvement': throughput_speedup > 0
                    }
                }
            
            summary['comparative_analysis'] = comparative_analysis
        
        # Add timestamp
        summary['generated_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        return summary
    
    def print_summary_table(self, summary: Dict[str, Any]) -> None:
        """Print a formatted summary table to console."""
        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)
        
        # Overall statistics
        exp_summary = summary.get('experiment_summary', {})
        print(f"Total Experiments: {exp_summary.get('total_experiments', 0)}")
        print(f"Successful: {exp_summary.get('successful_experiments', 0)}")
        print(f"Failed: {exp_summary.get('failed_experiments', 0)}")
        print(f"Total Duration: {exp_summary.get('total_duration_seconds', 0):.1f}s")
        
        # Mode comparison table
        print(f"\n{'Mode':<12} {'Latency (ms)':<12} {'Speedup %':<10} {'Stall %':<8} {'Hit Rate %':<10} {'Mispred Cost':<12}")
        print("-" * 80)
        
        mode_summaries = summary.get('mode_summaries', {})
        comparative_analysis = summary.get('comparative_analysis', {})
        
        for mode, mode_summary in mode_summaries.items():
            latency = mode_summary.get('average_latency_ms', 0)
            
            # Get speedup from comparative analysis
            speedup = 0
            if mode in comparative_analysis:
                speedup = comparative_analysis[mode].get('latency_speedup_percent', 0)
            
            stall_fraction = mode_summary.get('average_stall_fraction', 0) * 100
            
            # Get hit rate (cache or prefetch)
            hit_rate = 0
            if mode == 'reactive':
                hit_rate = mode_summary.get('average_cache_hit_rate', 0) * 100
            elif mode == 'predictive':
                hit_rate = mode_summary.get('average_prefetch_hit_rate', 0) * 100
            
            # Misprediction cost (only for predictive)
            mispred_cost = 0
            if mode == 'predictive':
                prediction_accuracy = mode_summary.get('average_prediction_accuracy', 0)
                mispred_cost = (1 - prediction_accuracy) * 100
            
            print(f"{mode:<12} {latency:<12.1f} {speedup:<10.1f} {stall_fraction:<8.1f} {hit_rate:<10.1f} {mispred_cost:<12.1f}")
        
        print("=" * 80)
    
    def save_summary(self, summary: Dict[str, Any], filename: str = "experiment_summary.json") -> None:
        """Save summary to JSON file."""
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logging.info(f"Summary saved to {filepath}")
    
    def generate_plots(self, summary: Dict[str, Any]) -> None:
        """Generate comparative plots."""
        try:
            # Check if matplotlib is available
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # Create plots directory
            plots_dir = os.path.join(self.output_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            # Extract data for plotting
            modes = []
            latencies = []
            throughputs = []
            hit_rates = []
            
            mode_summaries = summary.get('mode_summaries', {})
            
            for mode, mode_summary in mode_summaries.items():
                modes.append(mode)
                latencies.append(mode_summary.get('average_latency_ms', 0))
                throughputs.append(mode_summary.get('average_tokens_per_second', 0))
                
                if mode == 'reactive':
                    hit_rates.append(mode_summary.get('average_cache_hit_rate', 0) * 100)
                elif mode == 'predictive':
                    hit_rates.append(mode_summary.get('average_prefetch_hit_rate', 0) * 100)
                else:
                    hit_rates.append(0)
            
            # Create comparison plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('KV Cache Strategy Comparison', fontsize=16)
            
            # Latency comparison
            axes[0, 0].bar(modes, latencies)
            axes[0, 0].set_title('Average Latency')
            axes[0, 0].set_ylabel('Latency (ms)')
            
            # Throughput comparison
            axes[0, 1].bar(modes, throughputs)
            axes[0, 1].set_title('Average Throughput')
            axes[0, 1].set_ylabel('Tokens/Second')
            
            # Hit rate comparison
            axes[1, 0].bar(modes, hit_rates)
            axes[1, 0].set_title('Hit Rate')
            axes[1, 0].set_ylabel('Hit Rate (%)')
            
            # Speedup comparison
            comparative_analysis = summary.get('comparative_analysis', {})
            speedups = []
            for mode in modes:
                if mode in comparative_analysis:
                    speedups.append(comparative_analysis[mode].get('latency_speedup_percent', 0))
                else:
                    speedups.append(0)
            
            axes[1, 1].bar(modes, speedups)
            axes[1, 1].set_title('Latency Speedup vs Baseline')
            axes[1, 1].set_ylabel('Speedup (%)')
            axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(plots_dir, "kv_cache_comparison.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Plots saved to {plot_path}")
            
        except ImportError:
            logging.warning("Matplotlib not available, skipping plot generation")
        except Exception as e:
            logging.error(f"Failed to generate plots: {e}")


def create_default_experiment_config() -> str:
    """Create a default experiment configuration file."""
    config_content = """
# KV Cache Prefetch Experiment Configuration

experiments:
  - mode: baseline
    model_name: "microsoft/DialoGPT-small"
    context_lengths: [100, 500, 1000]
    num_runs: 3
    max_new_tokens: 50
    batch_size: 1
    quantization_config:
      load_in_8bit: false
      load_in_4bit: false

  - mode: reactive
    model_name: "microsoft/DialoGPT-small"
    page_size: 128
    max_gpu_pages: 8
    context_lengths: [100, 500, 1000]
    num_runs: 3
    max_new_tokens: 50
    batch_size: 1

  - mode: predictive
    model_name: "microsoft/DialoGPT-small"
    page_size: 128
    max_gpu_pages: 8
    prefetch_k: 4
    context_lengths: [100, 500, 1000]
    num_runs: 3
    max_new_tokens: 50
    batch_size: 1

# Global settings
global_settings:
  output_dir: "logs"
  generate_plots: true
  verbose_logging: true
  save_detailed_metrics: true
"""
    
    config_path = "experiment_config.yaml"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    return config_path


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='KV Cache Prefetch Experiment Orchestrator')
    parser.add_argument('--config', type=str, default='experiment_config.yaml',
                       help='Path to experiment configuration file')
    parser.add_argument('--output', type=str, default='logs',
                       help='Output directory for results')
    parser.add_argument('--create-config', action='store_true',
                       help='Create default configuration file and exit')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    # Create default config if requested
    if args.create_config:
        config_path = create_default_experiment_config()
        print(f"Default configuration created: {config_path}")
        return
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Configuration file {args.config} not found.")
        print("Use --create-config to create a default configuration file.")
        return
    
    try:
        # Initialize orchestrator
        orchestrator = ExperimentOrchestrator(args.config, args.output)
        
        # Run all experiments
        print("Starting KV Cache Prefetch Experiments")
        print("=" * 50)
        
        results = orchestrator.run_all_experiments()
        
        # Generate summary
        print("\nGenerating summary...")
        summary = orchestrator.generate_summary()
        
        # Print summary table
        orchestrator.print_summary_table(summary)
        
        # Save summary
        orchestrator.save_summary(summary)
        
        # Generate plots
        if not args.no_plots:
            print("\nGenerating plots...")
            orchestrator.generate_plots(summary)
        
        print(f"\nExperiments completed successfully!")
        print(f"Results saved to: {args.output}")
        print(f"Summary saved to: {os.path.join(args.output, 'experiment_summary.json')}")
        
    except Exception as e:
        logging.error(f"Experiment orchestration failed: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
