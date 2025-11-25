#!/usr/bin/env python3
"""
Task 1: Comprehensive Benchmarking Suite for Compact Models
Measures inference speed, throughput, and memory usage across all three models
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Add src to path
import sys
sys.path.insert(0, '/home/aaron/ATX/software/MLFF_Distiller/src')

from mlff_distiller.models.student_model import StudentForceField


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load a model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model = StudentForceField(hidden_dim=128, max_z=100)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = checkpoint if isinstance(checkpoint, StudentForceField) else torch.jit.load(checkpoint)

    model.to(device)
    model.eval()
    return model


def benchmark_model(model: torch.nn.Module, batch_sizes: List[int], device: torch.device,
                   num_runs: int = 10, max_atoms: int = 32) -> Dict:
    """Benchmark a model across different batch sizes."""
    results = {
        'batch_results': {},
        'summary': {}
    }

    model.eval()

    with torch.no_grad():
        for batch_size in batch_sizes:
            batch_times = []
            memory_usage = []
            throughputs = []

            for _ in range(num_runs):
                # Create random input: (batch_size, num_atoms, 3) positions and (batch_size, num_atoms) atom types
                positions = torch.randn(batch_size, max_atoms, 3, device=device)
                atom_numbers = torch.randint(1, 8, (batch_size, max_atoms), device=device)

                # Warm up
                _ = model(atom_numbers, positions)

                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()

                # Forward pass - note: atomic_numbers comes before positions
                energy = model(atom_numbers, positions)

                torch.cuda.synchronize() if torch.cuda.is_available() else None
                elapsed = time.time() - start_time

                batch_times.append(elapsed)
                throughputs.append(batch_size / elapsed)  # samples/sec

                # Memory tracking
                if torch.cuda.is_available():
                    memory_usage.append(torch.cuda.max_memory_allocated() / 1e6)  # MB

            results['batch_results'][batch_size] = {
                'avg_latency_ms': np.mean(batch_times) * 1000,
                'std_latency_ms': np.std(batch_times) * 1000,
                'min_latency_ms': np.min(batch_times) * 1000,
                'max_latency_ms': np.max(batch_times) * 1000,
                'avg_throughput_samples_per_sec': np.mean(throughputs),
                'avg_memory_mb': np.mean(memory_usage) if memory_usage else 0,
                'latency_per_sample_ms': np.mean(batch_times) * 1000 / batch_size
            }

    return results


def benchmark_cpu_inference(model: torch.nn.Module, batch_sizes: List[int],
                           num_runs: int = 5, max_atoms: int = 32) -> Dict:
    """Benchmark model on CPU."""
    device = torch.device('cpu')
    model = model.cpu()
    results = {'batch_results': {}}

    model.eval()
    with torch.no_grad():
        for batch_size in batch_sizes:
            batch_times = []

            for _ in range(num_runs):
                positions = torch.randn(batch_size, max_atoms, 3, device=device)
                atom_numbers = torch.randint(1, 8, (batch_size, max_atoms), device=device)

                _ = model(positions, atom_numbers)

                start_time = time.time()
                energy, forces = model(positions, atom_numbers)
                elapsed = time.time() - start_time
                batch_times.append(elapsed)

            results['batch_results'][batch_size] = {
                'avg_latency_ms': np.mean(batch_times) * 1000,
                'latency_per_sample_ms': np.mean(batch_times) * 1000 / batch_size,
                'throughput_samples_per_sec': batch_size / np.mean(batch_times)
            }

    return results


def get_model_size(checkpoint_path: str) -> float:
    """Get model size in MB."""
    return os.path.getsize(checkpoint_path) / 1e6


def create_comparison_plots(benchmark_results: Dict, output_dir: str):
    """Create comparison plots for all three models."""
    os.makedirs(output_dir, exist_ok=True)

    models = list(benchmark_results.keys())
    batch_sizes = sorted(list(benchmark_results[models[0]]['GPU']['batch_results'].keys()))

    # Plot 1: Latency Comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Compact Models: Inference Performance Comparison', fontsize=16, fontweight='bold')

    # Latency vs batch size
    ax = axes[0, 0]
    for model in models:
        latencies = [benchmark_results[model]['GPU']['batch_results'][bs]['avg_latency_ms']
                    for bs in batch_sizes]
        ax.plot(batch_sizes, latencies, marker='o', label=model)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('GPU Latency vs Batch Size')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Throughput vs batch size
    ax = axes[0, 1]
    for model in models:
        throughputs = [benchmark_results[model]['GPU']['batch_results'][bs]['avg_throughput_samples_per_sec']
                      for bs in batch_sizes]
        ax.plot(batch_sizes, throughputs, marker='s', label=model)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Throughput (samples/sec)')
    ax.set_title('GPU Throughput vs Batch Size')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Memory usage
    ax = axes[1, 0]
    for model in models:
        memory = [benchmark_results[model]['GPU']['batch_results'][bs]['avg_memory_mb']
                 for bs in batch_sizes]
        ax.plot(batch_sizes, memory, marker='^', label=model)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Memory (MB)')
    ax.set_title('GPU Memory Usage vs Batch Size')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Per-sample latency
    ax = axes[1, 1]
    for model in models:
        per_sample = [benchmark_results[model]['GPU']['batch_results'][bs]['latency_per_sample_ms']
                     for bs in batch_sizes]
        ax.plot(batch_sizes, per_sample, marker='d', label=model)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Latency per Sample (ms)')
    ax.set_title('Per-Sample Latency vs Batch Size')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/performance_comparison.png")

    # Plot 2: Model characteristics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Model Characteristics Comparison', fontsize=14, fontweight='bold')

    model_sizes = [benchmark_results[m]['model_size_mb'] for m in models]
    params = [benchmark_results[m]['num_parameters'] for m in models]

    ax = axes[0]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    ax.bar(models, model_sizes, color=colors, alpha=0.7)
    ax.set_ylabel('Model Size (MB)')
    ax.set_title('Checkpoint File Size')
    for i, (m, s) in enumerate(zip(models, model_sizes)):
        ax.text(i, s + 0.1, f'{s:.2f}MB', ha='center', va='bottom')

    ax = axes[1]
    ax.bar(models, params, color=colors, alpha=0.7)
    ax.set_ylabel('Number of Parameters')
    ax.set_title('Model Complexity')
    for i, (m, p) in enumerate(zip(models, params)):
        ax.text(i, p + 10000, f'{p/1e3:.0f}K', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_characteristics.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/model_characteristics.png")


def main():
    print("="*80)
    print("TASK 1: COMPREHENSIVE BENCHMARKING SUITE")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Model configurations
    models_config = {
        'Original Student (427K)': {
            'checkpoint': '/home/aaron/ATX/software/MLFF_Distiller/checkpoints/best_model.pt',
            'arch_params': {'hidden_dims': [128, 128, 128], 'output_dim': 1}
        },
        'Tiny Model (77K)': {
            'checkpoint': '/home/aaron/ATX/software/MLFF_Distiller/checkpoints/tiny_model/best_model.pt',
            'arch_params': {'hidden_dims': [32, 32], 'output_dim': 1}
        },
        'Ultra-tiny Model (21K)': {
            'checkpoint': '/home/aaron/ATX/software/MLFF_Distiller/checkpoints/ultra_tiny_model/best_model.pt',
            'arch_params': {'hidden_dims': [16], 'output_dim': 1}
        }
    }

    batch_sizes = [1, 2, 4, 8, 16, 32]
    benchmark_results = {}

    # Benchmark each model
    for model_name, config in models_config.items():
        checkpoint_path = config['checkpoint']

        if not os.path.exists(checkpoint_path):
            print(f"Warning: {checkpoint_path} not found. Skipping {model_name}")
            continue

        print(f"\n{model_name}")
        print("-" * 60)

        model_size = get_model_size(checkpoint_path)
        print(f"Checkpoint size: {model_size:.2f} MB")

        # Load model
        model = load_model(checkpoint_path, device)
        print(f"Model loaded successfully")

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {num_params:,}")

        # GPU Benchmarking
        print("Benchmarking GPU inference...")
        gpu_results = benchmark_model(model, batch_sizes, device, num_runs=10)

        # CPU Benchmarking
        print("Benchmarking CPU inference...")
        cpu_results = benchmark_cpu_inference(model, [1, 2, 4, 8], num_runs=5)

        benchmark_results[model_name] = {
            'GPU': gpu_results,
            'CPU': cpu_results,
            'model_size_mb': model_size,
            'num_parameters': num_params,
            'checkpoint_path': checkpoint_path
        }

        # Print summary
        print(f"\nGPU Results (batch_size=1):")
        print(f"  Latency: {gpu_results['batch_results'][1]['avg_latency_ms']:.3f} ms")
        print(f"  Throughput: {gpu_results['batch_results'][1]['avg_throughput_samples_per_sec']:.1f} samples/sec")

        print(f"\nGPU Results (batch_size=32):")
        print(f"  Latency: {gpu_results['batch_results'][32]['avg_latency_ms']:.3f} ms")
        print(f"  Throughput: {gpu_results['batch_results'][32]['avg_throughput_samples_per_sec']:.1f} samples/sec")
        print(f"  Memory: {gpu_results['batch_results'][32]['avg_memory_mb']:.1f} MB")

    # Save results
    output_dir = '/home/aaron/ATX/software/MLFF_Distiller/benchmarks'
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'{output_dir}/compact_models_benchmark_{timestamp}.json'

    # Prepare JSON-serializable results
    serializable_results = {}
    for model_name, results in benchmark_results.items():
        serializable_results[model_name] = {
            'GPU': {
                'batch_results': {
                    str(bs): {
                        'avg_latency_ms': float(v['avg_latency_ms']),
                        'std_latency_ms': float(v['std_latency_ms']),
                        'min_latency_ms': float(v['min_latency_ms']),
                        'max_latency_ms': float(v['max_latency_ms']),
                        'avg_throughput_samples_per_sec': float(v['avg_throughput_samples_per_sec']),
                        'avg_memory_mb': float(v['avg_memory_mb']),
                        'latency_per_sample_ms': float(v['latency_per_sample_ms'])
                    }
                    for bs, v in results['GPU']['batch_results'].items()
                }
            },
            'CPU': {
                'batch_results': {
                    str(bs): {
                        'avg_latency_ms': float(v['avg_latency_ms']),
                        'latency_per_sample_ms': float(v['latency_per_sample_ms']),
                        'throughput_samples_per_sec': float(v['throughput_samples_per_sec'])
                    }
                    for bs, v in results['CPU']['batch_results'].items()
                }
            },
            'model_size_mb': float(results['model_size_mb']),
            'num_parameters': int(results['num_parameters']),
            'checkpoint_path': results['checkpoint_path']
        }

    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\n✓ Saved benchmark results: {output_file}")

    # Create comparison plots
    try:
        create_comparison_plots(benchmark_results, output_dir)
    except Exception as e:
        print(f"Warning: Could not create plots: {e}")

    print("\n" + "="*80)
    print("TASK 1 COMPLETE: Benchmarking Results Saved")
    print("="*80)


if __name__ == '__main__':
    main()
