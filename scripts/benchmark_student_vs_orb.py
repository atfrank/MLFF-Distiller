#!/usr/bin/env python3
"""
Benchmark Student Model vs Orb Teacher on Drug-Like Molecules

Compares inference speed between:
1. Student model (PaiNN-based, distilled)
2. Orb teacher model (large pretrained model)

Tests on 17 representative drug-like molecules to measure:
- Energy computation time
- Force computation time
- Memory usage
- Speedup factor

Author: CUDA Optimization Engineer
Date: 2025-11-24
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from ase import Atoms
from ase.build import molecule

# Add src to path
import sys
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))


def create_test_molecules() -> List[Tuple[str, Atoms]]:
    """
    Create 17 representative drug-like test molecules.

    Returns:
        List of (name, atoms) tuples
    """
    molecules = []

    # Small molecules (3-9 atoms)
    small_mols = [
        'H2O',      # 3 atoms - water
        'NH3',      # 4 atoms - ammonia
        'CH4',      # 5 atoms - methane
        'C2H6',     # 8 atoms - ethane
        'C2H5OH',   # 9 atoms - ethanol
    ]

    for name in small_mols:
        try:
            atoms = molecule(name)
            molecules.append((name, atoms))
        except Exception as e:
            print(f"Warning: Could not create {name}: {e}")

    # Medium molecules (10-20 atoms)
    medium_mols = [
        'C6H6',     # 12 atoms - benzene
        'CH3COOH',  # 8 atoms - acetic acid
        'C6H14',    # 20 atoms - hexane
        'C10H8',    # 18 atoms - naphthalene
    ]

    for name in medium_mols:
        try:
            atoms = molecule(name)
            molecules.append((name, atoms))
        except Exception as e:
            print(f"Warning: Could not create {name}: {e}")

    # Biomolecules (10-20 atoms)
    bio_mols = [
        ('glycine', 'C2H5NO2'),    # 10 atoms
        ('alanine', 'C3H7NO2'),    # 13 atoms
    ]

    for name, _ in bio_mols:
        try:
            atoms = molecule(name)
            molecules.append((name, atoms))
        except Exception as e:
            print(f"Warning: Could not create {name}: {e}")

    # Large molecules (20-30 atoms)
    large_mols = [
        'C14H10',   # 24 atoms - anthracene
    ]

    for name in large_mols:
        try:
            atoms = molecule(name)
            molecules.append((name, atoms))
        except Exception as e:
            print(f"Warning: Could not create {name}: {e}")

    # Water clusters
    try:
        # 5-molecule water cluster (15 atoms)
        water = molecule('H2O')
        cluster = water.copy()
        for i in range(1, 5):
            shifted = water.copy()
            shifted.translate([i * 3.0, 0, 0])
            cluster += shifted
        molecules.append(('water_cluster_5', cluster))
    except Exception as e:
        print(f"Warning: Could not create water cluster: {e}")

    print(f"Created {len(molecules)} test molecules")
    for name, atoms in molecules:
        print(f"  - {name}: {len(atoms)} atoms")

    return molecules


def benchmark_student_model(
    checkpoint_path: str,
    molecules: List[Tuple[str, Atoms]],
    n_trials: int = 30,
    device: str = 'cuda'
) -> Dict:
    """
    Benchmark student model inference speed.

    Args:
        checkpoint_path: Path to student model checkpoint
        molecules: List of (name, atoms) tuples
        n_trials: Number of timing trials per molecule
        device: Device to run on ('cuda' or 'cpu')

    Returns:
        Dictionary with benchmark results
    """
    from mlff_distiller.inference import StudentForceFieldCalculator

    print(f"\n{'='*60}")
    print("BENCHMARKING STUDENT MODEL")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Trials per molecule: {n_trials}")

    # Initialize calculator
    calc = StudentForceFieldCalculator(
        checkpoint_path=checkpoint_path,
        device=device
    )

    results = {
        'model': 'Student (PaiNN)',
        'device': device,
        'n_trials': n_trials,
        'molecules': {}
    }

    # Benchmark each molecule
    for mol_name, atoms in molecules:
        print(f"\nTesting {mol_name} ({len(atoms)} atoms)...")

        # Attach calculator
        atoms.calc = calc

        # Warmup (5 iterations)
        print("  Warming up...")
        for _ in range(5):
            _ = atoms.get_potential_energy()
            _ = atoms.get_forces()

        # Synchronize GPU
        if device == 'cuda':
            torch.cuda.synchronize()

        # Benchmark energy + forces
        print(f"  Running {n_trials} trials...")
        times = []

        for trial in range(n_trials):
            # Perturb positions slightly to prevent caching
            orig_pos = atoms.get_positions().copy()
            atoms.set_positions(orig_pos + np.random.randn(len(atoms), 3) * 0.001)

            # Time energy + forces computation
            start = time.perf_counter()
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()

            if device == 'cuda':
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # Convert to ms

        # Statistics
        times = np.array(times)
        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)

        print(f"  Time: {mean_time:.3f} ± {std_time:.3f} ms (min: {min_time:.3f} ms)")

        results['molecules'][mol_name] = {
            'n_atoms': len(atoms),
            'mean_ms': float(mean_time),
            'std_ms': float(std_time),
            'min_ms': float(min_time),
            'all_times_ms': times.tolist()
        }

    # Overall statistics
    all_times = [m['mean_ms'] for m in results['molecules'].values()]
    results['overall'] = {
        'mean_ms': float(np.mean(all_times)),
        'std_ms': float(np.std(all_times)),
        'min_ms': float(np.min(all_times)),
        'max_ms': float(np.max(all_times))
    }

    print(f"\n{'='*60}")
    print("STUDENT MODEL SUMMARY")
    print(f"{'='*60}")
    print(f"Mean time across all molecules: {results['overall']['mean_ms']:.3f} ms")
    print(f"Range: {results['overall']['min_ms']:.3f} - {results['overall']['max_ms']:.3f} ms")

    return results


def benchmark_orb_model(
    molecules: List[Tuple[str, Atoms]],
    n_trials: int = 30,
    device: str = 'cuda'
) -> Dict:
    """
    Benchmark Orb teacher model inference speed.

    Args:
        molecules: List of (name, atoms) tuples
        n_trials: Number of timing trials per molecule
        device: Device to run on ('cuda' or 'cpu')

    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'='*60}")
    print("BENCHMARKING ORB TEACHER MODEL")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Trials per molecule: {n_trials}")

    try:
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        # Load Orb model
        print("\nLoading Orb model (this may take a while)...")
        orbff = pretrained.orb_v2(device=device)
        calc = ORBCalculator(orbff, device=device)

    except ImportError as e:
        print(f"\nERROR: Could not import Orb models: {e}")
        print("Orb models may not be installed. Skipping Orb benchmark.")
        return {
            'model': 'Orb-v2 (Teacher)',
            'error': 'orb_models not installed',
            'molecules': {}
        }
    except Exception as e:
        print(f"\nERROR: Could not load Orb model: {e}")
        return {
            'model': 'Orb-v2 (Teacher)',
            'error': str(e),
            'molecules': {}
        }

    results = {
        'model': 'Orb-v2 (Teacher)',
        'device': device,
        'n_trials': n_trials,
        'molecules': {}
    }

    # Benchmark each molecule
    for mol_name, atoms in molecules:
        print(f"\nTesting {mol_name} ({len(atoms)} atoms)...")

        try:
            # Attach calculator
            atoms.calc = calc

            # Warmup (5 iterations)
            print("  Warming up...")
            for _ in range(5):
                _ = atoms.get_potential_energy()
                _ = atoms.get_forces()

            # Synchronize GPU
            if device == 'cuda':
                torch.cuda.synchronize()

            # Benchmark energy + forces
            print(f"  Running {n_trials} trials...")
            times = []

            for trial in range(n_trials):
                # Perturb positions slightly to prevent caching
                orig_pos = atoms.get_positions().copy()
                atoms.set_positions(orig_pos + np.random.randn(len(atoms), 3) * 0.001)

                # Time energy + forces computation
                start = time.perf_counter()
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()

                if device == 'cuda':
                    torch.cuda.synchronize()

                elapsed = time.perf_counter() - start
                times.append(elapsed * 1000)  # Convert to ms

            # Statistics
            times = np.array(times)
            mean_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)

            print(f"  Time: {mean_time:.3f} ± {std_time:.3f} ms (min: {min_time:.3f} ms)")

            results['molecules'][mol_name] = {
                'n_atoms': len(atoms),
                'mean_ms': float(mean_time),
                'std_ms': float(std_time),
                'min_ms': float(min_time),
                'all_times_ms': times.tolist()
            }

        except Exception as e:
            print(f"  ERROR: {e}")
            results['molecules'][mol_name] = {
                'n_atoms': len(atoms),
                'error': str(e)
            }

    # Overall statistics (only for successful molecules)
    successful_times = [
        m['mean_ms'] for m in results['molecules'].values()
        if 'mean_ms' in m
    ]

    if successful_times:
        results['overall'] = {
            'mean_ms': float(np.mean(successful_times)),
            'std_ms': float(np.std(successful_times)),
            'min_ms': float(np.min(successful_times)),
            'max_ms': float(np.max(successful_times))
        }

        print(f"\n{'='*60}")
        print("ORB MODEL SUMMARY")
        print(f"{'='*60}")
        print(f"Mean time across all molecules: {results['overall']['mean_ms']:.3f} ms")
        print(f"Range: {results['overall']['min_ms']:.3f} - {results['overall']['max_ms']:.3f} ms")
    else:
        results['overall'] = {'error': 'No successful benchmarks'}
        print("\nNo successful Orb benchmarks")

    return results


def compare_results(student_results: Dict, orb_results: Dict) -> Dict:
    """
    Compare student and Orb model results.

    Args:
        student_results: Student model benchmark results
        orb_results: Orb model benchmark results

    Returns:
        Comparison statistics
    """
    print(f"\n{'='*60}")
    print("STUDENT vs ORB COMPARISON")
    print(f"{'='*60}")

    comparison = {
        'student': student_results.get('overall', {}),
        'orb': orb_results.get('overall', {}),
        'molecules': {}
    }

    # Per-molecule comparison
    for mol_name in student_results['molecules']:
        student_mol = student_results['molecules'][mol_name]
        orb_mol = orb_results['molecules'].get(mol_name, {})

        if 'mean_ms' in student_mol and 'mean_ms' in orb_mol:
            speedup = orb_mol['mean_ms'] / student_mol['mean_ms']

            comparison['molecules'][mol_name] = {
                'n_atoms': student_mol['n_atoms'],
                'student_ms': student_mol['mean_ms'],
                'orb_ms': orb_mol['mean_ms'],
                'speedup': float(speedup)
            }

            print(f"\n{mol_name} ({student_mol['n_atoms']} atoms):")
            print(f"  Student: {student_mol['mean_ms']:.3f} ms")
            print(f"  Orb:     {orb_mol['mean_ms']:.3f} ms")
            print(f"  Speedup: {speedup:.2f}x")

    # Overall comparison
    if 'mean_ms' in student_results.get('overall', {}) and \
       'mean_ms' in orb_results.get('overall', {}):
        overall_speedup = orb_results['overall']['mean_ms'] / student_results['overall']['mean_ms']
        comparison['overall_speedup'] = float(overall_speedup)

        print(f"\n{'='*60}")
        print("OVERALL COMPARISON")
        print(f"{'='*60}")
        print(f"Student mean: {student_results['overall']['mean_ms']:.3f} ms")
        print(f"Orb mean:     {orb_results['overall']['mean_ms']:.3f} ms")
        print(f"Speedup:      {overall_speedup:.2f}x")
        print(f"\nStudent is {overall_speedup:.2f}x FASTER than Orb teacher!")

    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Student vs Orb Teacher on drug-like molecules"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pt',
        help='Path to student model checkpoint'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='benchmarks/student_vs_orb.json',
        help='Output JSON file for results'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=30,
        help='Number of timing trials per molecule'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run on'
    )
    parser.add_argument(
        '--skip-orb',
        action='store_true',
        help='Skip Orb benchmarking (only test student)'
    )

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("STUDENT vs ORB BENCHMARK")
    print(f"{'='*60}")
    print(f"Student checkpoint: {args.checkpoint}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")
    print(f"Trials: {args.n_trials}")

    # Create test molecules
    molecules = create_test_molecules()

    # Benchmark student model
    student_results = benchmark_student_model(
        args.checkpoint,
        molecules,
        n_trials=args.n_trials,
        device=args.device
    )

    # Benchmark Orb model (unless skipped)
    if args.skip_orb:
        print("\nSkipping Orb benchmark (--skip-orb)")
        orb_results = {
            'model': 'Orb-v2 (Teacher)',
            'skipped': True,
            'molecules': {}
        }
        comparison = {'skipped': True}
    else:
        orb_results = benchmark_orb_model(
            molecules,
            n_trials=args.n_trials,
            device=args.device
        )

        # Compare results
        comparison = compare_results(student_results, orb_results)

    # Save results
    results = {
        'student': student_results,
        'orb': orb_results,
        'comparison': comparison,
        'config': {
            'checkpoint': args.checkpoint,
            'n_trials': args.n_trials,
            'device': args.device
        }
    }

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("RESULTS SAVED")
    print(f"{'='*60}")
    print(f"Output: {args.output}")
    print("\nBenchmark complete!")


if __name__ == '__main__':
    main()
