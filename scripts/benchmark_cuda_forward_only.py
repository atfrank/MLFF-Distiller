#!/usr/bin/env python3
"""
Benchmark CUDA Optimizations (Forward Pass Only)

Tests forward pass speedup of Triton kernels.
Note: Current implementation doesn't support autograd (forces), so we benchmark
energy computation only.

For force computation speedup, we need to implement analytical gradients
in the Triton kernels (future work).

Usage:
    python scripts/benchmark_cuda_forward_only.py --device cuda
"""

import sys
import time
import argparse
from pathlib import Path
import logging
import json

import numpy as np
import torch
from ase.build import molecule

# Add src to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT))

from mlff_distiller.models.student_model import StudentForceField
from mlff_distiller.models.student_model_optimized import StudentForceFieldOptimized

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def benchmark_forward_pass(model, atomic_numbers, positions, n_iterations=100):
    """Benchmark forward pass (energy only)."""
    device = next(model.parameters()).device

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(atomic_numbers, positions)

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(n_iterations):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.perf_counter()

            energy = model(atomic_numbers, positions)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'p50_ms': float(np.median(times)),
        'p95_ms': float(np.percentile(times, 95)),
        'max_ms': float(np.max(times)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=str, default='benchmarks/cuda_optimizations')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        return 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("CUDA OPTIMIZATION BENCHMARKS (Forward Pass Only)")
    logger.info("=" * 80)

    # Load models
    logger.info("Loading models...")
    baseline_model = StudentForceField.load(args.checkpoint, device=args.device)
    baseline_model.eval()

    optimized_model = StudentForceFieldOptimized.load(
        args.checkpoint,
        device=args.device,
        use_triton_kernels=True
    )
    optimized_model.eval()

    # Test molecule
    mol = molecule('C6H6')
    atomic_numbers = torch.tensor(mol.get_atomic_numbers(), dtype=torch.long, device=args.device)
    positions = torch.tensor(mol.get_positions(), dtype=torch.float32, device=args.device)

    # Benchmark
    logger.info("\nBenchmarking baseline model...")
    baseline_results = benchmark_forward_pass(baseline_model, atomic_numbers, positions)
    logger.info(f"  Mean: {baseline_results['mean_ms']:.3f} ± {baseline_results['std_ms']:.3f} ms")

    logger.info("\nBenchmarking CUDA-optimized model...")
    optimized_results = benchmark_forward_pass(optimized_model, atomic_numbers, positions)
    logger.info(f"  Mean: {optimized_results['mean_ms']:.3f} ± {optimized_results['std_ms']:.3f} ms")

    # Analysis
    speedup = baseline_results['mean_ms'] / optimized_results['mean_ms']

    print("\n" + "=" * 80)
    print("SPEEDUP ANALYSIS")
    print("=" * 80)
    print(f"\nForward Pass (Energy Computation Only):")
    print(f"  Baseline: {baseline_results['mean_ms']:.3f} ms")
    print(f"  Optimized: {optimized_results['mean_ms']:.3f} ms")
    print(f"  Speedup: {speedup:.2f}x")

    # Save results
    results = {
        'baseline': baseline_results,
        'optimized': optimized_results,
        'speedup': speedup
    }

    with open(output_dir / 'forward_only_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_dir / 'forward_only_results.json'}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
