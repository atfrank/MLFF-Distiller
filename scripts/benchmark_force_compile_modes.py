#!/usr/bin/env python3
"""
Benchmark torch.compile() modes specifically for FORCE COMPUTATION.

Focus: Energy+Forces workload (real MD use case), not just energy-only.

Tests:
1. Baseline (no compile, autograd forces)
2. Baseline + batched (current Phase 3A)
3. torch.compile() default mode
4. torch.compile() reduce-overhead mode
5. torch.compile() max-autotune mode
6. Best mode + batching

Usage:
    conda run -n mlff-py312 python scripts/benchmark_force_compile_modes.py
"""

import sys
from pathlib import Path
import time
import torch
import numpy as np
from ase.build import molecule

# Add src to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from mlff_distiller.inference import StudentForceFieldCalculator

def create_test_molecules():
    """Create test molecules of various sizes."""
    return {
        'H2O (3 atoms)': molecule('H2O'),
        'CH4 (5 atoms)': molecule('CH4'),
        'C6H6 (12 atoms)': molecule('C6H6'),
        'C2H6 (8 atoms)': molecule('C2H6'),
    }

def benchmark_configuration(calc, molecules, n_trials=30, warmup=5):
    """Benchmark a calculator configuration."""
    results = {}

    for name, mol in molecules.items():
        # Warmup
        for _ in range(warmup):
            mol.calc = calc
            _ = mol.get_potential_energy()
            _ = mol.get_forces()

        # Benchmark
        times = []
        for _ in range(n_trials):
            mol.calc = calc
            start = time.perf_counter()
            energy = mol.get_potential_energy()
            forces = mol.get_forces()
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)

        results[name] = {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'median_ms': np.median(times),
            'min_ms': np.min(times),
        }

    return results

def benchmark_batched(calc, molecules, batch_sizes=[1, 2, 4], n_trials=30):
    """Benchmark batched force computation."""
    results = {}

    mol_list = list(molecules.values())

    for batch_size in batch_sizes:
        if batch_size > len(mol_list):
            continue

        batch = mol_list[:batch_size]

        # Warmup
        for _ in range(5):
            _ = calc.calculate_batch(batch)

        # Benchmark
        times = []
        for _ in range(n_trials):
            start = time.perf_counter()
            results_batch = calc.calculate_batch(batch)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            time_per_mol = elapsed / batch_size
            times.append(time_per_mol)

        results[f'batch_{batch_size}'] = {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'median_ms': np.median(times),
            'total_ms': np.mean(times) * batch_size,
        }

    return results

def main():
    print("="*80)
    print("FORCE COMPUTATION OPTIMIZATION - torch.compile() Benchmarks")
    print("="*80)
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print(f"torch.compile() available: {hasattr(torch, 'compile')}")
    print()

    checkpoint = Path('checkpoints/best_model.pt')
    if not checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint}")
        return 1

    molecules = create_test_molecules()
    print(f"Test molecules: {list(molecules.keys())}")
    print()

    # Configuration 1: Baseline (no compile)
    print("="*80)
    print("Configuration 1: Baseline (No compilation)")
    print("="*80)
    calc_baseline = StudentForceFieldCalculator(
        checkpoint_path=str(checkpoint),
        device='cuda',
        use_compile=False,
    )

    results_baseline = benchmark_configuration(calc_baseline, molecules)

    print("\nResults (Energy + Forces):")
    for name, res in results_baseline.items():
        print(f"  {name}: {res['mean_ms']:.3f} ± {res['std_ms']:.3f} ms")

    baseline_mean = np.mean([r['mean_ms'] for r in results_baseline.values()])
    print(f"\n  Average: {baseline_mean:.3f} ms")
    print()

    # Configuration 2: Baseline + Batching
    print("="*80)
    print("Configuration 2: Baseline + Batching (Phase 3A)")
    print("="*80)

    results_batched = benchmark_batched(calc_baseline, molecules, batch_sizes=[1, 2, 4])

    print("\nResults (Batched):")
    for name, res in results_batched.items():
        speedup = baseline_mean / res['mean_ms']
        print(f"  {name}: {res['mean_ms']:.3f} ms/mol ({speedup:.2f}x speedup)")
    print()

    # Configuration 3: torch.compile() default
    print("="*80)
    print("Configuration 3: torch.compile() - default mode")
    print("="*80)
    try:
        calc_compile_default = StudentForceFieldCalculator(
            checkpoint_path=str(checkpoint),
            device='cuda',
            use_compile=True,
            compile_mode='default',
        )

        results_compile_default = benchmark_configuration(calc_compile_default, molecules)

        print("\nResults (torch.compile default):")
        for name, res in results_compile_default.items():
            speedup = results_baseline[name]['mean_ms'] / res['mean_ms']
            print(f"  {name}: {res['mean_ms']:.3f} ms ({speedup:.2f}x vs baseline)")

        compile_default_mean = np.mean([r['mean_ms'] for r in results_compile_default.values()])
        speedup = baseline_mean / compile_default_mean
        print(f"\n  Average: {compile_default_mean:.3f} ms ({speedup:.2f}x speedup)")
    except Exception as e:
        print(f"ERROR: {e}")
        results_compile_default = None
    print()

    # Configuration 4: torch.compile() reduce-overhead
    print("="*80)
    print("Configuration 4: torch.compile() - reduce-overhead mode")
    print("="*80)
    try:
        calc_compile_overhead = StudentForceFieldCalculator(
            checkpoint_path=str(checkpoint),
            device='cuda',
            use_compile=True,
            compile_mode='reduce-overhead',
        )

        results_compile_overhead = benchmark_configuration(calc_compile_overhead, molecules)

        print("\nResults (torch.compile reduce-overhead):")
        for name, res in results_compile_overhead.items():
            speedup = results_baseline[name]['mean_ms'] / res['mean_ms']
            print(f"  {name}: {res['mean_ms']:.3f} ms ({speedup:.2f}x vs baseline)")

        compile_overhead_mean = np.mean([r['mean_ms'] for r in results_compile_overhead.values()])
        speedup = baseline_mean / compile_overhead_mean
        print(f"\n  Average: {compile_overhead_mean:.3f} ms ({speedup:.2f}x speedup)")
    except Exception as e:
        print(f"ERROR: {e}")
        results_compile_overhead = None
    print()

    # Configuration 5: torch.compile() max-autotune
    print("="*80)
    print("Configuration 5: torch.compile() - max-autotune mode")
    print("="*80)
    try:
        calc_compile_autotune = StudentForceFieldCalculator(
            checkpoint_path=str(checkpoint),
            device='cuda',
            use_compile=True,
            compile_mode='max-autotune',
        )

        results_compile_autotune = benchmark_configuration(calc_compile_autotune, molecules)

        print("\nResults (torch.compile max-autotune):")
        for name, res in results_compile_autotune.items():
            speedup = results_baseline[name]['mean_ms'] / res['mean_ms']
            print(f"  {name}: {res['mean_ms']:.3f} ms ({speedup:.2f}x vs baseline)")

        compile_autotune_mean = np.mean([r['mean_ms'] for r in results_compile_autotune.values()])
        speedup = baseline_mean / compile_autotune_mean
        print(f"\n  Average: {compile_autotune_mean:.3f} ms ({speedup:.2f}x speedup)")
    except Exception as e:
        print(f"ERROR: {e}")
        results_compile_autotune = None
    print()

    # Configuration 6: Best compile mode + batching
    print("="*80)
    print("Configuration 6: Best torch.compile() mode + Batching")
    print("="*80)

    # Use reduce-overhead as it's typically best for latency
    calc_best = StudentForceFieldCalculator(
        checkpoint_path=str(checkpoint),
        device='cuda',
        use_compile=True,
        compile_mode='reduce-overhead',
    )

    results_best_batched = benchmark_batched(calc_best, molecules, batch_sizes=[1, 2, 4])

    print("\nResults (reduce-overhead + Batching):")
    for name, res in results_best_batched.items():
        speedup = baseline_mean / res['mean_ms']
        print(f"  {name}: {res['mean_ms']:.3f} ms/mol ({speedup:.2f}x vs baseline)")
    print()

    # Final Summary
    print("="*80)
    print("FINAL SUMMARY - Force Computation Speedups")
    print("="*80)
    print(f"Baseline (no opts):              {baseline_mean:.3f} ms  (1.00x)")

    if results_compile_default:
        compile_default_speedup = baseline_mean / compile_default_mean
        print(f"torch.compile() default:         {compile_default_mean:.3f} ms  ({compile_default_speedup:.2f}x)")

    if results_compile_overhead:
        compile_overhead_speedup = baseline_mean / compile_overhead_mean
        print(f"torch.compile() reduce-overhead: {compile_overhead_mean:.3f} ms  ({compile_overhead_speedup:.2f}x)")

    if results_compile_autotune:
        compile_autotune_speedup = baseline_mean / compile_autotune_mean
        print(f"torch.compile() max-autotune:    {compile_autotune_mean:.3f} ms  ({compile_autotune_speedup:.2f}x)")

    batch4_time = results_batched['batch_4']['mean_ms']
    batch4_speedup = baseline_mean / batch4_time
    print(f"Batching (batch=4):              {batch4_time:.3f} ms  ({batch4_speedup:.2f}x)")

    best_batch4_time = results_best_batched['batch_4']['mean_ms']
    best_batch4_speedup = baseline_mean / best_batch4_time
    print(f"Best mode + Batch=4:             {best_batch4_time:.3f} ms  ({best_batch4_speedup:.2f}x) ← BEST")

    print()
    print("="*80)

    # Calculate total speedup from original baseline (16.65 ms from earlier)
    original_baseline = 16.65  # ms from Phase 3A analysis
    total_speedup = original_baseline / best_batch4_time
    print(f"TOTAL SPEEDUP vs Original Baseline: {total_speedup:.1f}x")
    print(f"  (Original: {original_baseline:.2f} ms → Optimized: {best_batch4_time:.3f} ms)")
    print("="*80)

    return 0

if __name__ == '__main__':
    sys.exit(main())
