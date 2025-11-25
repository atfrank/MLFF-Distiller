#!/usr/bin/env python3
"""
Simple benchmark: FORCE COMPUTATION ONLY (what MD needs).

Focus: Just measure how fast we can compute forces for MD.
"""

import sys
from pathlib import Path
import time
import torch
import numpy as np
from ase.build import molecule

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from mlff_distiller.inference import StudentForceFieldCalculator

def benchmark_forces(calc, mol, n_trials=50, warmup=5):
    """Benchmark force computation speed."""
    mol.calc = calc

    # Warmup
    for _ in range(warmup):
        _ = mol.get_forces()

    # Benchmark
    times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        forces = mol.get_forces()
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'median': np.median(times),
    }

print("="*60)
print("FORCE COMPUTATION BENCHMARK (MD Use Case)")
print("="*60)
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print()

# Test molecule
benzene = molecule('C6H6')
print(f"Test molecule: Benzene (12 atoms)")
print()

# Config 1: Baseline
print("-"*60)
print("1. Baseline (no optimizations)")
print("-"*60)
calc1 = StudentForceFieldCalculator(
    checkpoint_path='checkpoints/best_model.pt',
    device='cuda',
    use_compile=False,
)
result1 = benchmark_forces(calc1, benzene)
print(f"Force computation: {result1['mean']:.3f} ± {result1['std']:.3f} ms")
print(f"Speedup: 1.00x (baseline)")
print()

# Config 2: torch.compile() default
print("-"*60)
print("2. torch.compile() - default mode")
print("-"*60)
try:
    calc2 = StudentForceFieldCalculator(
        checkpoint_path='checkpoints/best_model.pt',
        device='cuda',
        use_compile=True,
        compile_mode='default',
    )
    result2 = benchmark_forces(calc2, benzene)
    speedup2 = result1['mean'] / result2['mean']
    print(f"Force computation: {result2['mean']:.3f} ± {result2['std']:.3f} ms")
    print(f"Speedup: {speedup2:.2f}x")
except Exception as e:
    print(f"FAILED: {e}")
    result2 = None
print()

# Config 3: torch.compile() reduce-overhead
print("-"*60)
print("3. torch.compile() - reduce-overhead mode")
print("-"*60)
try:
    calc3 = StudentForceFieldCalculator(
        checkpoint_path='checkpoints/best_model.pt',
        device='cuda',
        use_compile=True,
        compile_mode='reduce-overhead',
    )
    result3 = benchmark_forces(calc3, benzene)
    speedup3 = result1['mean'] / result3['mean']
    print(f"Force computation: {result3['mean']:.3f} ± {result3['std']:.3f} ms")
    print(f"Speedup: {speedup3:.2f}x")
except Exception as e:
    print(f"FAILED: {e}")
    result3 = None
print()

# Config 4: torch.compile() max-autotune
print("-"*60)
print("4. torch.compile() - max-autotune mode")
print("-"*60)
try:
    calc4 = StudentForceFieldCalculator(
        checkpoint_path='checkpoints/best_model.pt',
        device='cuda',
        use_compile=True,
        compile_mode='max-autotune',
    )
    result4 = benchmark_forces(calc4, benzene)
    speedup4 = result1['mean'] / result4['mean']
    print(f"Force computation: {result4['mean']:.3f} ± {result4['std']:.3f} ms")
    print(f"Speedup: {speedup4:.2f}x")
except Exception as e:
    print(f"FAILED: {e}")
    result4 = None
print()

# Summary
print("="*60)
print("SUMMARY - Force Computation for MD")
print("="*60)
print(f"Baseline:            {result1['mean']:.3f} ms  (1.00x)")
if result2:
    print(f"compile (default):   {result2['mean']:.3f} ms  ({result1['mean']/result2['mean']:.2f}x)")
if result3:
    print(f"compile (overhead):  {result3['mean']:.3f} ms  ({result1['mean']/result3['mean']:.2f}x)")
if result4:
    print(f"compile (autotune):  {result4['mean']:.3f} ms  ({result1['mean']/result4['mean']:.2f}x)")

# Find best
best_time = result1['mean']
best_name = "Baseline"

if result2 and result2['mean'] < best_time:
    best_time = result2['mean']
    best_name = "compile (default)"
if result3 and result3['mean'] < best_time:
    best_time = result3['mean']
    best_name = "compile (reduce-overhead)"
if result4 and result4['mean'] < best_time:
    best_time = result4['mean']
    best_name = "compile (max-autotune)"

speedup = result1['mean'] / best_time
print()
print(f"BEST: {best_name} - {best_time:.3f} ms ({speedup:.2f}x speedup)")
print("="*60)
