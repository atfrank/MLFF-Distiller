#!/usr/bin/env python3
"""
Benchmark CUDA Optimizations

Compares performance of:
1. Baseline StudentForceField (PyTorch)
2. StudentForceFieldOptimized (Triton kernels)
3. Individual kernel benchmarks

This script validates the 5-7x speedup target for Phase 3.

Usage:
    python scripts/benchmark_cuda_optimizations.py --device cuda
    python scripts/benchmark_cuda_optimizations.py --quick
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List
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


def benchmark_model(
    model,
    atomic_numbers,
    positions_np,
    n_iterations=100,
    warmup=10,
    model_name="Model"
):
    """Benchmark model performance."""
    device = next(model.parameters()).device

    # Warmup
    for _ in range(warmup):
        pos = torch.tensor(
            positions_np,
            dtype=torch.float32,
            device=device,
            requires_grad=True
        )
        energy, forces = model.predict_energy_and_forces(atomic_numbers, pos)

    # Benchmark
    times = []
    for _ in range(n_iterations):
        pos = torch.tensor(
            positions_np,
            dtype=torch.float32,
            device=device,
            requires_grad=True
        )

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()

        energy, forces = model.predict_energy_and_forces(atomic_numbers, pos)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    results = {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'p50_ms': float(np.median(times)),
        'p95_ms': float(np.percentile(times, 95)),
        'max_ms': float(np.max(times)),
    }

    logger.info(f"{model_name}: {results['mean_ms']:.3f} ± {results['std_ms']:.3f} ms")

    return results


def benchmark_kernels(device='cuda'):
    """Benchmark individual CUDA kernels."""
    logger.info("\n" + "=" * 80)
    logger.info("INDIVIDUAL KERNEL BENCHMARKS")
    logger.info("=" * 80)

    from kernels.fused_rbf_cutoff import fused_rbf_cutoff_pytorch, fused_rbf_cutoff_triton
    from kernels.fused_edge_features import fused_edge_features_pytorch, fused_edge_features_triton

    # Test parameters
    n_edges = 132  # Benzene
    n_rbf = 20
    cutoff = 5.0
    n_atoms = 12

    # Generate test data
    torch.manual_seed(42)
    distances = torch.rand(n_edges, device=device) * cutoff
    centers = torch.linspace(0, cutoff, n_rbf, device=device)
    widths = torch.ones(n_rbf, device=device) * (cutoff / n_rbf)
    gamma = (1.0 / (widths[0] ** 2)).item()

    positions = torch.randn(n_atoms, 3, device=device)
    src = torch.randint(0, n_atoms, (n_edges,), device=device)
    dst = torch.randint(0, n_atoms, (n_edges,), device=device)
    edge_index = torch.stack([src, dst], dim=0)

    n_iterations = 1000

    results = {}

    # 1. RBF + Cutoff
    logger.info("\n1. Fused RBF + Cutoff")
    logger.info("-" * 80)

    # Warmup
    for _ in range(10):
        _ = fused_rbf_cutoff_pytorch(distances, centers, gamma, cutoff)
        _ = fused_rbf_cutoff_triton(distances, centers, gamma, cutoff)

    # PyTorch
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = fused_rbf_cutoff_pytorch(distances, centers, gamma, cutoff)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / n_iterations * 1000

    # Triton
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = fused_rbf_cutoff_triton(distances, centers, gamma, cutoff)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / n_iterations * 1000

    speedup = pytorch_time / triton_time

    logger.info(f"  PyTorch: {pytorch_time:.3f} ms")
    logger.info(f"  Triton:  {triton_time:.3f} ms")
    logger.info(f"  Speedup: {speedup:.2f}x")

    results['rbf_cutoff'] = {
        'pytorch_ms': pytorch_time,
        'triton_ms': triton_time,
        'speedup': speedup
    }

    # 2. Edge Features
    logger.info("\n2. Fused Edge Features")
    logger.info("-" * 80)

    # Warmup
    for _ in range(10):
        _ = fused_edge_features_pytorch(positions, edge_index)
        _ = fused_edge_features_triton(positions, edge_index)

    # PyTorch
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = fused_edge_features_pytorch(positions, edge_index)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / n_iterations * 1000

    # Triton
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = fused_edge_features_triton(positions, edge_index)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / n_iterations * 1000

    speedup = pytorch_time / triton_time

    logger.info(f"  PyTorch: {pytorch_time:.3f} ms")
    logger.info(f"  Triton:  {triton_time:.3f} ms")
    logger.info(f"  Speedup: {speedup:.2f}x")

    results['edge_features'] = {
        'pytorch_ms': pytorch_time,
        'triton_ms': triton_time,
        'speedup': speedup
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark CUDA optimizations"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='benchmarks/cuda_optimizations',
        help='Output directory'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick benchmark with fewer iterations'
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        logger.error("CUDA not available! CUDA optimizations require GPU.")
        return 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("CUDA OPTIMIZATION BENCHMARKS")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Output: {output_dir}\n")

    # Test molecule
    mol = molecule('C6H6')  # Benzene, 12 atoms
    n_atoms = len(mol)

    logger.info(f"Test molecule: C6H6 ({n_atoms} atoms)\n")

    # Prepare inputs
    device = torch.device(args.device)
    atomic_numbers = torch.tensor(
        mol.get_atomic_numbers(),
        dtype=torch.long,
        device=device
    )
    positions_np = mol.get_positions()

    n_iter = 50 if args.quick else 100

    # ========================================================================
    # Part 1: Individual Kernel Benchmarks
    # ========================================================================
    kernel_results = benchmark_kernels(device=args.device)

    # ========================================================================
    # Part 2: End-to-End Model Benchmarks
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("END-TO-END MODEL BENCHMARKS")
    logger.info("=" * 80)

    # Load baseline model
    logger.info("\nLoading baseline model...")
    baseline_model = StudentForceField.load(args.checkpoint, device=args.device)
    baseline_model.eval()

    # Load optimized model
    logger.info("Loading CUDA-optimized model...")
    optimized_model = StudentForceFieldOptimized.load(
        args.checkpoint,
        device=args.device,
        use_triton_kernels=True
    )
    optimized_model.eval()

    # Benchmark baseline
    logger.info("\nBenchmarking baseline model...")
    baseline_results = benchmark_model(
        baseline_model,
        atomic_numbers,
        positions_np,
        n_iterations=n_iter,
        model_name="Baseline"
    )

    # Benchmark optimized
    logger.info("\nBenchmarking CUDA-optimized model...")
    optimized_results = benchmark_model(
        optimized_model,
        atomic_numbers,
        positions_np,
        n_iterations=n_iter,
        model_name="Optimized"
    )

    # ========================================================================
    # Part 3: Analysis and Speedup Calculation
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("SPEEDUP ANALYSIS")
    logger.info("=" * 80)

    baseline_time = baseline_results['mean_ms']
    optimized_time = optimized_results['mean_ms']
    speedup = baseline_time / optimized_time

    print(f"\n{'Metric':<30} {'Baseline':<15} {'Optimized':<15} {'Speedup':<10}")
    print("-" * 70)
    print(f"{'Mean time (ms)':<30} {baseline_time:<15.3f} {optimized_time:<15.3f} {speedup:<10.2f}x")
    print(f"{'Std dev (ms)':<30} {baseline_results['std_ms']:<15.3f} {optimized_results['std_ms']:<15.3f}")
    print(f"{'Min time (ms)':<30} {baseline_results['min_ms']:<15.3f} {optimized_results['min_ms']:<15.3f}")
    print(f"{'P95 time (ms)':<30} {baseline_results['p95_ms']:<15.3f} {optimized_results['p95_ms']:<15.3f}")

    # Calculate throughput
    baseline_throughput = 1000 / baseline_time
    optimized_throughput = 1000 / optimized_time

    print(f"\n{'Throughput (struct/s)':<30} {baseline_throughput:<15.1f} {optimized_throughput:<15.1f} "
          f"{optimized_throughput/baseline_throughput:<10.2f}x")

    # Combined with Phase 3A batching
    phase3a_speedup = 3.42  # From previous benchmarks
    total_speedup = speedup * phase3a_speedup

    print(f"\n{'Combined with Phase 3A batching:':<30}")
    print(f"  Phase 3A speedup: {phase3a_speedup:.2f}x")
    print(f"  CUDA kernels speedup: {speedup:.2f}x")
    print(f"  Total speedup: {total_speedup:.2f}x")

    # Check if target met
    target_min = 5.0
    target_max = 7.0

    if total_speedup >= target_min:
        print(f"\n TARGET MET: {total_speedup:.2f}x >= {target_min:.1f}x")
        if total_speedup >= target_max:
            print(f" EXCEEDED STRETCH GOAL: {total_speedup:.2f}x >= {target_max:.1f}x")
    else:
        print(f"\n TARGET NOT MET: {total_speedup:.2f}x < {target_min:.1f}x")
        print(f" Additional optimization needed: {target_min/total_speedup:.2f}x more")

    # ========================================================================
    # Part 4: Save Results
    # ========================================================================
    results = {
        'kernels': kernel_results,
        'baseline': baseline_results,
        'optimized': optimized_results,
        'analysis': {
            'speedup': speedup,
            'phase3a_speedup': phase3a_speedup,
            'total_speedup': total_speedup,
            'target_met': total_speedup >= target_min,
            'stretch_met': total_speedup >= target_max
        },
        'config': {
            'checkpoint': args.checkpoint,
            'device': args.device,
            'molecule': 'C6H6',
            'n_atoms': n_atoms,
            'n_iterations': n_iter
        }
    }

    results_file = output_dir / 'cuda_optimization_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n\nResults saved to {results_file}")

    # Generate summary report
    summary_file = output_dir / 'CUDA_OPTIMIZATION_SUMMARY.md'
    with open(summary_file, 'w') as f:
        f.write("# CUDA Optimization Results\n\n")
        f.write(f"**Date**: 2025-11-24\n")
        f.write(f"**Model**: StudentForceField (427K parameters)\n")
        f.write(f"**Test System**: Benzene (12 atoms, 132 edges)\n\n")

        f.write("## Performance Summary\n\n")
        f.write(f"| Metric | Baseline | Optimized | Speedup |\n")
        f.write(f"|--------|----------|-----------|----------|\n")
        f.write(f"| Mean Time | {baseline_time:.3f} ms | {optimized_time:.3f} ms | **{speedup:.2f}x** |\n")
        f.write(f"| Throughput | {baseline_throughput:.1f} struct/s | {optimized_throughput:.1f} struct/s | **{optimized_throughput/baseline_throughput:.2f}x** |\n\n")

        f.write("## Kernel-Level Speedups\n\n")
        for kernel_name, kernel_data in kernel_results.items():
            f.write(f"- **{kernel_name}**: {kernel_data['speedup']:.2f}x speedup\n")

        f.write("\n## Cumulative Speedup\n\n")
        f.write(f"- Phase 3A (batching): {phase3a_speedup:.2f}x\n")
        f.write(f"- CUDA kernels: {speedup:.2f}x\n")
        f.write(f"- **Total**: **{total_speedup:.2f}x**\n\n")

        if total_speedup >= target_min:
            f.write(f" **TARGET MET**: {total_speedup:.2f}x >= {target_min:.1f}x\n\n")
        else:
            f.write(f" **TARGET NOT MET**: {total_speedup:.2f}x < {target_min:.1f}x\n\n")

        f.write("## Next Steps\n\n")
        if total_speedup >= target_max:
            f.write("- Exceeded stretch goal!\n")
            f.write("- Deploy to production\n")
            f.write("- Document optimization guide\n")
        elif total_speedup >= target_min:
            f.write("- Met minimum target\n")
            f.write("- Consider additional optimizations for stretch goal\n")
        else:
            f.write("- Implement batched force computation\n")
            f.write("- Optimize message passing kernels\n")
            f.write("- Profile remaining bottlenecks\n")

    logger.info(f"Summary saved to {summary_file}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
