#!/usr/bin/env python3
"""
Simplified benchmarking script for compact models
"""

import os
import json
import torch
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

def benchmark_model_simple(model_path, device, batch_sizes=[1, 2, 4, 8, 16, 32], num_runs=5):
    """Simple benchmark that works with saved checkpoints."""

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_size = os.path.getsize(model_path) / 1e6

    # Extract model from checkpoint
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Import here to avoid issues
        import sys
        sys.path.insert(0, '/home/aaron/ATX/software/MLFF_Distiller/src')
        from mlff_distiller.models.student_model import StudentForceField

        model = StudentForceField(hidden_dim=128, max_z=100)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = checkpoint

    model = model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    results = {'batch_results': {}}

    with torch.no_grad():
        for batch_size in batch_sizes:
            times = []

            for _ in range(num_runs):
                # Create inputs - must match expected shapes [N]
                # For batching, we concatenate atoms across structures
                num_atoms = 16
                total_atoms = batch_size * num_atoms

                atomic_numbers = torch.randint(1, 8, (total_atoms,), device=device, dtype=torch.long)
                positions = torch.randn(total_atoms, 3, device=device, dtype=torch.float32)

                # Warm up
                _ = model(atomic_numbers, positions)

                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start = time.time()
                energy = model(atomic_numbers, positions)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                elapsed = time.time() - start

                times.append(elapsed)

            avg_time = np.mean(times) * 1000  # ms
            results['batch_results'][batch_size] = {
                'avg_latency_ms': float(avg_time),
                'throughput_samples_per_sec': float(batch_size * num_atoms / (avg_time / 1000)),
                'latency_per_sample_ms': float(avg_time / (batch_size * num_atoms))
            }

    return {
        'model_path': str(model_path),
        'model_size_mb': float(model_size),
        'num_parameters': int(num_params),
        'gpu_results': results
    }


def main():
    print("="*80)
    print("SIMPLIFIED BENCHMARKING")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    models = {
        'Original Student (427K)': 'checkpoints/best_model.pt',
        'Tiny Model (77K)': 'checkpoints/tiny_model/best_model.pt',
        'Ultra-tiny Model (21K)': 'checkpoints/ultra_tiny_model/best_model.pt'
    }

    results = {}

    for name, path in models.items():
        if os.path.exists(path):
            print(f"Benchmarking {name}...")
            try:
                results[name] = benchmark_model_simple(path, device)
                print(f"  Size: {results[name]['model_size_mb']:.2f} MB")
                print(f"  Params: {results[name]['num_parameters']:,}")
                print(f"  Latency (bs=1): {results[name]['gpu_results']['batch_results'][1]['avg_latency_ms']:.3f} ms")
                print(f"  Latency (bs=32): {results[name]['gpu_results']['batch_results'][32]['avg_latency_ms']:.3f} ms\n")
            except Exception as e:
                print(f"  Error: {e}\n")
        else:
            print(f"Not found: {path}\n")

    # Save results
    output_dir = 'benchmarks'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'{output_dir}/compact_models_benchmark_{timestamp}.json'

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {output_file}")

    # Create comparison plot
    if len(results) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))

        batch_sizes = [1, 2, 4, 8, 16, 32]
        for model_name, data in results.items():
            latencies = [data['gpu_results']['batch_results'][bs]['avg_latency_ms'] for bs in batch_sizes]
            ax.plot(batch_sizes, latencies, marker='o', label=model_name)

        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Inference Latency Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plot_path = f'{output_dir}/benchmark_comparison_{timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot_path}")

    print("\nBenchmarking complete!")


if __name__ == '__main__':
    main()
