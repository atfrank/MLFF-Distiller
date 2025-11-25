#!/usr/bin/env python3
"""
Benchmark and Validate Analytical Force Computation

This script comprehensively tests the analytical force implementation against
autograd baseline, measuring:
1. Numerical accuracy (force MAE, max error)
2. Performance speedup
3. Memory efficiency
4. Correctness across different system sizes and edge cases

Author: CUDA Optimization Engineer
Date: 2025-11-24
Phase: 3B Week 1
Target: 1.8-2x speedup, <1e-4 eV/Å error
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import numpy as np
import torch
from ase import Atoms
from ase.build import molecule, bulk
from ase.calculators.singlepoint import SinglePointCalculator

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mlff_distiller.models.student_model import StudentForceField
from src.mlff_distiller.inference import StudentForceFieldCalculator


def create_test_structures() -> List[Tuple[str, Atoms]]:
    """
    Create diverse test structures for validation.

    Returns:
        List of (name, atoms) tuples
    """
    structures = []

    # Small molecules
    structures.append(("H2O", molecule("H2O")))
    structures.append(("CH4", molecule("CH4")))
    structures.append(("NH3", molecule("NH3")))
    structures.append(("Benzene", molecule("C6H6")))

    # Medium molecules
    try:
        structures.append(("Ethanol", molecule("C2H5OH")))
        structures.append(("Acetone", molecule("CH3COCH3")))
    except:
        pass  # Some molecules may not be available

    # Small clusters (manually created)
    # C20 fullerene-like
    if True:  # Placeholder - create simple cluster
        c20 = Atoms(
            'C20',
            positions=np.random.randn(20, 3) * 2.0 + 5.0,
            cell=[10, 10, 10],
            pbc=False
        )
        structures.append(("C20_cluster", c20))

    # Larger cluster
    c50 = Atoms(
        'C50',
        positions=np.random.randn(50, 3) * 3.0 + 7.0,
        cell=[15, 15, 15],
        pbc=False
    )
    structures.append(("C50_cluster", c50))

    # Edge cases
    # Single atom
    structures.append(("H_atom", Atoms('H', positions=[[0, 0, 0]])))

    # Two atoms (very close)
    structures.append(("H2_close", Atoms('H2', positions=[[0, 0, 0], [0.5, 0, 0]])))

    # Two atoms (far apart, no interaction)
    structures.append(("H2_far", Atoms('H2', positions=[[0, 0, 0], [10, 0, 0]])))

    return structures


def test_force_accuracy(
    model: StudentForceField,
    atoms: Atoms,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Test force accuracy: analytical vs autograd.

    Args:
        model: StudentForceField model
        atoms: Test structure
        device: Device to run on

    Returns:
        Dictionary with accuracy metrics
    """
    model.eval()

    # Convert to tensors
    atomic_numbers = torch.tensor(
        atoms.get_atomic_numbers(),
        dtype=torch.long,
        device=device
    )
    positions = torch.tensor(
        atoms.get_positions(),
        dtype=torch.float32,
        device=device
    )

    # NOTE: Don't use torch.no_grad() here - we need gradients for force computation!
    # Autograd forces (baseline)
    energy_auto, forces_auto = model.predict_energy_and_forces(
        atomic_numbers, positions
    )

    # Analytical forces (optimized)
    energy_anal, forces_anal = model.forward_with_analytical_forces(
        atomic_numbers, positions
    )

    # Convert to numpy
    energy_auto = energy_auto.cpu().item()
    energy_anal = energy_anal.cpu().item()
    forces_auto = forces_auto.cpu().numpy()
    forces_anal = forces_anal.cpu().numpy()

    # Compute errors
    energy_error = abs(energy_auto - energy_anal)
    force_diff = forces_auto - forces_anal
    force_mae = np.abs(force_diff).mean()
    force_max = np.abs(force_diff).max()
    force_rmse = np.sqrt((force_diff ** 2).mean())

    # Relative error
    force_norm = np.linalg.norm(forces_auto)
    force_rel_error = np.linalg.norm(force_diff) / (force_norm + 1e-8)

    return {
        'energy_error': energy_error,
        'force_mae': force_mae,
        'force_max': force_max,
        'force_rmse': force_rmse,
        'force_rel_error': force_rel_error,
        'n_atoms': len(atoms),
    }


def benchmark_force_computation(
    model: StudentForceField,
    atoms: Atoms,
    device: str = 'cuda',
    n_warmup: int = 10,
    n_runs: int = 50
) -> Dict[str, float]:
    """
    Benchmark force computation speed: analytical vs autograd.

    Args:
        model: StudentForceField model
        atoms: Test structure
        device: Device to run on
        n_warmup: Warmup iterations
        n_runs: Benchmark iterations

    Returns:
        Dictionary with timing metrics
    """
    model.eval()

    # Convert to tensors
    atomic_numbers = torch.tensor(
        atoms.get_atomic_numbers(),
        dtype=torch.long,
        device=device
    )
    positions = torch.tensor(
        atoms.get_positions(),
        dtype=torch.float32,
        device=device
    )

    # Warmup
    # NOTE: Don't use torch.no_grad() - forces require gradients!
    for _ in range(n_warmup):
        _ = model.predict_energy_and_forces(atomic_numbers, positions)

    if device == 'cuda':
        torch.cuda.synchronize()

    # Benchmark autograd forces
    times_auto = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model.predict_energy_and_forces(atomic_numbers, positions)
        if device == 'cuda':
            torch.cuda.synchronize()
        times_auto.append((time.perf_counter() - start) * 1000)  # ms

    # Warmup analytical
    for _ in range(n_warmup):
        _ = model.forward_with_analytical_forces(atomic_numbers, positions)

    if device == 'cuda':
        torch.cuda.synchronize()

    # Benchmark analytical forces
    times_anal = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model.forward_with_analytical_forces(atomic_numbers, positions)
        if device == 'cuda':
            torch.cuda.synchronize()
        times_anal.append((time.perf_counter() - start) * 1000)  # ms

    # Compute statistics
    auto_mean = np.mean(times_auto)
    auto_std = np.std(times_auto)
    anal_mean = np.mean(times_anal)
    anal_std = np.std(times_anal)
    speedup = auto_mean / anal_mean

    return {
        'autograd_mean_ms': auto_mean,
        'autograd_std_ms': auto_std,
        'analytical_mean_ms': anal_mean,
        'analytical_std_ms': anal_std,
        'speedup': speedup,
        'n_atoms': len(atoms),
    }


def benchmark_energy_only(
    model: StudentForceField,
    atoms: Atoms,
    device: str = 'cuda',
    n_warmup: int = 10,
    n_runs: int = 50
) -> float:
    """
    Benchmark energy-only computation (no forces).

    This provides a baseline for understanding overhead.

    Args:
        model: StudentForceField model
        atoms: Test structure
        device: Device to run on
        n_warmup: Warmup iterations
        n_runs: Benchmark iterations

    Returns:
        Mean time in ms
    """
    model.eval()

    # Convert to tensors
    atomic_numbers = torch.tensor(
        atoms.get_atomic_numbers(),
        dtype=torch.long,
        device=device
    )
    positions = torch.tensor(
        atoms.get_positions(),
        dtype=torch.float32,
        device=device
    )

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model.forward(atomic_numbers, positions)

    if device == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model.forward(atomic_numbers, positions)
        if device == 'cuda':
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    return np.mean(times)


def run_comprehensive_validation(
    checkpoint_path: str,
    device: str = 'cuda',
    output_path: str = 'benchmarks/analytical_forces_validation.json'
) -> Dict:
    """
    Run comprehensive validation and benchmarking.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to run on
        output_path: Where to save results

    Returns:
        Results dictionary
    """
    print("=" * 80)
    print("ANALYTICAL FORCE VALIDATION AND BENCHMARKING")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print()

    # Load model
    print("Loading model...")
    model = StudentForceField.load(checkpoint_path, device=device)
    model.eval()
    print(f"Model loaded: {model.num_parameters():,} parameters")
    print()

    # Create test structures
    print("Creating test structures...")
    structures = create_test_structures()
    print(f"Created {len(structures)} test structures")
    print()

    # Run validation
    results = {
        'config': {
            'checkpoint': checkpoint_path,
            'device': device,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
        },
        'accuracy': {},
        'performance': {},
        'summary': {},
    }

    print("-" * 80)
    print("ACCURACY VALIDATION")
    print("-" * 80)

    all_force_mae = []
    all_force_max = []
    max_error_structure = None
    max_error_value = 0

    for name, atoms in structures:
        print(f"\n{name} ({len(atoms)} atoms):")

        try:
            accuracy = test_force_accuracy(model, atoms, device)
            results['accuracy'][name] = accuracy

            print(f"  Energy error: {accuracy['energy_error']:.2e} eV")
            print(f"  Force MAE:    {accuracy['force_mae']:.2e} eV/Å")
            print(f"  Force max:    {accuracy['force_max']:.2e} eV/Å")
            print(f"  Force RMSE:   {accuracy['force_rmse']:.2e} eV/Å")
            print(f"  Force rel:    {accuracy['force_rel_error']:.2e}")

            # Track statistics
            all_force_mae.append(accuracy['force_mae'])
            all_force_max.append(accuracy['force_max'])

            if accuracy['force_max'] > max_error_value:
                max_error_value = accuracy['force_max']
                max_error_structure = name

            # Check if error is acceptable
            if accuracy['force_mae'] > 1e-4:
                print(f"  ⚠️  WARNING: MAE exceeds target (1e-4 eV/Å)")
            else:
                print(f"  ✓ PASS: MAE within target")

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results['accuracy'][name] = {'error': str(e)}

    # Accuracy summary
    print("\n" + "=" * 80)
    print("ACCURACY SUMMARY")
    print("=" * 80)
    print(f"Mean Force MAE:    {np.mean(all_force_mae):.2e} eV/Å")
    print(f"Median Force MAE:  {np.median(all_force_mae):.2e} eV/Å")
    print(f"Max Force error:   {np.max(all_force_max):.2e} eV/Å ({max_error_structure})")
    print(f"Target:            1.00e-04 eV/Å")

    if np.max(all_force_max) < 1e-4:
        print("✓ ALL TESTS PASSED")
    else:
        print("⚠️  SOME TESTS EXCEED TARGET")

    results['summary']['accuracy'] = {
        'mean_force_mae': float(np.mean(all_force_mae)),
        'median_force_mae': float(np.median(all_force_mae)),
        'max_force_error': float(np.max(all_force_max)),
        'max_error_structure': max_error_structure,
        'passed': bool(np.max(all_force_max) < 1e-4),
    }

    # Run performance benchmarks
    print("\n" + "-" * 80)
    print("PERFORMANCE BENCHMARKING")
    print("-" * 80)

    # Select representative structures for detailed benchmarking
    benchmark_structures = [
        ("H2O", structures[0][1]),
        ("CH4", structures[1][1]),
        ("Benzene", structures[3][1]),
    ]

    # Add medium and large if available
    for name, atoms in structures:
        if "C20" in name or "C50" in name:
            benchmark_structures.append((name, atoms))

    all_speedups = []
    all_energy_times = []

    for name, atoms in benchmark_structures:
        print(f"\n{name} ({len(atoms)} atoms):")

        try:
            # Energy only (baseline)
            energy_time = benchmark_energy_only(model, atoms, device)
            all_energy_times.append(energy_time)
            print(f"  Energy only:    {energy_time:.3f} ms")

            # Force computation
            perf = benchmark_force_computation(model, atoms, device)
            results['performance'][name] = perf
            results['performance'][name]['energy_only_ms'] = energy_time

            print(f"  Autograd:       {perf['autograd_mean_ms']:.3f} ± {perf['autograd_std_ms']:.3f} ms")
            print(f"  Analytical:     {perf['analytical_mean_ms']:.3f} ± {perf['analytical_std_ms']:.3f} ms")
            print(f"  Speedup:        {perf['speedup']:.2f}x")

            # Overhead analysis
            auto_overhead = perf['autograd_mean_ms'] - energy_time
            anal_overhead = perf['analytical_mean_ms'] - energy_time
            overhead_reduction = (auto_overhead - anal_overhead) / auto_overhead * 100

            print(f"  Overhead (auto): {auto_overhead:.3f} ms")
            print(f"  Overhead (anal): {anal_overhead:.3f} ms")
            print(f"  Overhead saved:  {overhead_reduction:.1f}%")

            all_speedups.append(perf['speedup'])

            if perf['speedup'] >= 1.8:
                print(f"  ✓ TARGET MET (1.8x speedup)")
            elif perf['speedup'] >= 1.5:
                print(f"  ⚠️  CLOSE TO TARGET ({perf['speedup']:.2f}x < 1.8x)")
            else:
                print(f"  ❌ BELOW TARGET ({perf['speedup']:.2f}x < 1.8x)")

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results['performance'][name] = {'error': str(e)}

    # Performance summary
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"Mean speedup:      {np.mean(all_speedups):.2f}x")
    print(f"Median speedup:    {np.median(all_speedups):.2f}x")
    print(f"Min speedup:       {np.min(all_speedups):.2f}x")
    print(f"Max speedup:       {np.max(all_speedups):.2f}x")
    print(f"Target:            1.80x")

    if np.min(all_speedups) >= 1.8:
        print("✓ ALL BENCHMARKS MEET TARGET")
    elif np.mean(all_speedups) >= 1.8:
        print("⚠️  AVERAGE MEETS TARGET, BUT SOME BELOW")
    else:
        print("❌ BELOW TARGET")

    results['summary']['performance'] = {
        'mean_speedup': float(np.mean(all_speedups)),
        'median_speedup': float(np.median(all_speedups)),
        'min_speedup': float(np.min(all_speedups)),
        'max_speedup': float(np.max(all_speedups)),
        'target_met': bool(np.min(all_speedups) >= 1.8),
    }

    # Calculate total speedup vs original baseline
    # From Phase 3A profiling: original autograd = 7.0 ms (3 atoms)
    # This is our baseline for total speedup calculation
    h2o_perf = results['performance'].get('H2O', {})
    if 'analytical_mean_ms' in h2o_perf:
        original_baseline_ms = 7.0  # From phase3_week1_results.json
        current_time_ms = h2o_perf['analytical_mean_ms']
        total_speedup = original_baseline_ms / current_time_ms

        print("\n" + "=" * 80)
        print("TOTAL SPEEDUP vs ORIGINAL BASELINE")
        print("=" * 80)
        print(f"Original baseline: {original_baseline_ms:.2f} ms (autograd, no optimizations)")
        print(f"Current (H2O):     {current_time_ms:.2f} ms (analytical forces)")
        print(f"Total speedup:     {total_speedup:.2f}x")
        print(f"Target:            9-10x")

        if total_speedup >= 9.0:
            print("✓✓✓ PHASE 3B WEEK 1 TARGET ACHIEVED!")
        elif total_speedup >= 7.0:
            print("⚠️  CLOSE TO TARGET (need 9-10x)")
        else:
            print("❌ BELOW TARGET")

        results['summary']['total_speedup'] = {
            'original_baseline_ms': original_baseline_ms,
            'current_time_ms': current_time_ms,
            'total_speedup': float(total_speedup),
            'target_met': bool(total_speedup >= 9.0),
        }

    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark and validate analytical force computation"
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
        default='cuda',
        help='Device to run on (cuda or cpu)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='benchmarks/analytical_forces_validation.json',
        help='Output path for results'
    )

    args = parser.parse_args()

    # Run validation
    results = run_comprehensive_validation(
        checkpoint_path=args.checkpoint,
        device=args.device,
        output_path=args.output
    )

    # Print final status
    print("\n" + "=" * 80)
    print("FINAL STATUS")
    print("=" * 80)

    accuracy_passed = results['summary']['accuracy']['passed']
    performance_passed = results['summary']['performance'].get('target_met', False)
    total_speedup_passed = results['summary'].get('total_speedup', {}).get('target_met', False)

    print(f"Accuracy:         {'✓ PASS' if accuracy_passed else '❌ FAIL'}")
    print(f"Performance:      {'✓ PASS' if performance_passed else '❌ FAIL'}")
    print(f"Total Speedup:    {'✓ PASS' if total_speedup_passed else '❌ FAIL'}")

    if accuracy_passed and performance_passed and total_speedup_passed:
        print("\n🎉 ALL TARGETS MET! Phase 3B Week 1 Complete!")
        return 0
    elif accuracy_passed and performance_passed:
        print("\n⚠️  Analytical forces working, but total speedup needs improvement")
        return 1
    else:
        print("\n❌ Some targets not met. Further optimization needed.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
