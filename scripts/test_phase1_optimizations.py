#!/usr/bin/env python3
"""
Test Phase 1 Optimizations: torch.compile() and FP16

This script tests the Phase 1 optimizations (torch.compile + FP16) to verify:
1. Correctness: Results match baseline
2. Performance: Speedup achieved
3. Stability: No errors or crashes

Usage:
    # Test in Python 3.12 environment
    conda activate mlff-py312
    python scripts/test_phase1_optimizations.py
"""

import sys
from pathlib import Path
import time
import numpy as np
import torch
from ase.build import molecule

# Add src to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from mlff_distiller.inference import StudentForceFieldCalculator

def test_optimization(use_compile=False, use_fp16=False, n_iter=10):
    """Test an optimization configuration."""

    opt_name = []
    if use_compile:
        opt_name.append("compile")
    if use_fp16:
        opt_name.append("fp16")
    opt_str = "+".join(opt_name) if opt_name else "baseline"

    print(f"\n{'='*60}")
    print(f"Testing: {opt_str}")
    print(f"{'='*60}")

    # Create calculator
    calc = StudentForceFieldCalculator(
        checkpoint_path='checkpoints/best_model.pt',
        device='cuda',
        use_compile=use_compile,
        use_fp16=use_fp16
    )

    # Create test molecule
    atoms = molecule('H2O')
    atoms.calc = calc

    # Warm-up (important for torch.compile!)
    print("Warming up...")
    for _ in range(3):
        _ = atoms.get_potential_energy()
        _ = atoms.get_forces()

    # Benchmark
    print(f"Benchmarking {n_iter} iterations...")
    times = []
    energies = []
    forces_list = []

    for i in range(n_iter):
        start = time.perf_counter()
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        elapsed = time.perf_counter() - start

        times.append(elapsed)
        energies.append(energy)
        forces_list.append(forces)

        if i == 0:
            print(f"  First call: {elapsed*1000:.2f} ms")

    # Results
    times = np.array(times)
    mean_time = times.mean()
    std_time = times.std()
    min_time = times.min()

    print(f"\nResults:")
    print(f"  Mean time: {mean_time*1000:.2f} ± {std_time*1000:.2f} ms")
    print(f"  Min time: {min_time*1000:.2f} ms")
    print(f"  Energy: {energies[0]:.6f} eV")
    print(f"  Max force: {np.abs(forces_list[0]).max():.6f} eV/Å")

    return {
        'opt': opt_str,
        'mean_time': mean_time,
        'std_time': std_time,
        'min_time': min_time,
        'energy': energies[0],
        'forces': forces_list[0]
    }

def main():
    print("="*60)
    print("Phase 1 Optimization Testing")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"torch.compile available: {hasattr(torch, 'compile')}")

    # Test all configurations
    results = {}

    # 1. Baseline
    results['baseline'] = test_optimization(use_compile=False, use_fp16=False)

    # 2. torch.compile() only
    results['compile'] = test_optimization(use_compile=True, use_fp16=False)

    # 3. FP16 only
    results['fp16'] = test_optimization(use_compile=False, use_fp16=True)

    # 4. Both optimizations
    results['compile+fp16'] = test_optimization(use_compile=True, use_fp16=True)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    baseline_time = results['baseline']['mean_time']

    print(f"\n{'Configuration':<20} {'Time (ms)':<15} {'Speedup':>10} {'Energy Match':>15}")
    print("-" * 60)

    for name in ['baseline', 'compile', 'fp16', 'compile+fp16']:
        r = results[name]
        speedup = baseline_time / r['mean_time']
        energy_diff = abs(r['energy'] - results['baseline']['energy'])
        energy_match = "✓" if energy_diff < 0.001 else f"✗ ({energy_diff:.6f})"

        print(f"{name:<20} {r['mean_time']*1000:>7.2f} ± {r['std_time']*1000:>4.2f}   "
              f"{speedup:>6.2f}x    {energy_match:>15}")

    # Check correctness
    print("\n" + "="*60)
    print("CORRECTNESS CHECK")
    print("="*60)

    baseline_energy = results['baseline']['energy']
    baseline_forces = results['baseline']['forces']

    all_correct = True
    for name in ['compile', 'fp16', 'compile+fp16']:
        r = results[name]
        energy_err = abs(r['energy'] - baseline_energy)
        forces_err = np.abs(r['forces'] - baseline_forces).max()

        energy_ok = energy_err < 0.010  # 10 meV tolerance (realistic for FP16)
        forces_ok = forces_err < 0.010  # 10 meV/Å tolerance (realistic for FP16)

        status = "✓ PASS" if (energy_ok and forces_ok) else "✗ FAIL"
        all_correct = all_correct and energy_ok and forces_ok

        print(f"\n{name}:")
        print(f"  Energy error: {energy_err:.6f} eV ({status if energy_ok else '✗ FAIL'})")
        print(f"  Max force error: {forces_err:.6f} eV/Å ({status if forces_ok else '✗ FAIL'})")

    # Final verdict
    print("\n" + "="*60)
    if all_correct:
        print("✓ ALL TESTS PASSED")
        print("  - torch.compile() and FP16 produce correct results")
        compile_speedup = baseline_time / results['compile']['mean_time']
        fp16_speedup = baseline_time / results['fp16']['mean_time']
        combined_speedup = baseline_time / results['compile+fp16']['mean_time']
        print(f"  - torch.compile() speedup: {compile_speedup:.2f}x")
        print(f"  - FP16 speedup: {fp16_speedup:.2f}x")
        print(f"  - Combined speedup: {combined_speedup:.2f}x")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("  - Results do not match baseline within tolerance")
        return 1

if __name__ == '__main__':
    sys.exit(main())
