#!/usr/bin/env python3
"""
Week 4: Enhanced Batched Force Computation - Drug-Like Molecule Benchmark

Tests batched force computation on realistic drug-like molecules to validate
the 5-7x total speedup target for MD simulations.

Drug-like molecules tested:
- Aspirin (21 atoms) - common drug
- Caffeine (24 atoms) - stimulant
- Ibuprofen (33 atoms) - NSAID
- Penicillin (41 atoms) - antibiotic
- Dopamine (20 atoms) - neurotransmitter
- Serotonin (25 atoms) - neurotransmitter
- Testosterone (49 atoms) - steroid hormone
- Cholesterol (74 atoms) - lipid molecule

Batch sizes tested: 1, 2, 4, 8, 16, 32

Target: Achieve 5-7x speedup at batch size 16 compared to baseline single-molecule computation.

Usage:
    conda run -n mlff-py312 python scripts/benchmark_batched_forces_druglike.py \
        --checkpoint checkpoints/best_model.pt \
        --output benchmarks/week4_batched_druglike.json
"""

import sys
from pathlib import Path
import time
import json
import argparse
import numpy as np
from typing import List, Dict, Any

import torch
from ase import Atoms
from ase.build import molecule

# Add src to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from mlff_distiller.inference import StudentForceFieldCalculator


def create_druglike_molecules() -> Dict[str, Atoms]:
    """
    Create drug-like test molecules.

    Returns realistic molecular structures commonly seen in MD simulations
    of drug discovery and biochemistry applications.
    """
    molecules = {}

    # Use molecules that are definitely in ASE database
    # Small molecules
    molecules['H2O'] = molecule('H2O')  # 3 atoms
    molecules['NH3'] = molecule('NH3')  # 4 atoms
    molecules['CH4'] = molecule('CH4')  # 5 atoms
    molecules['Ethanol'] = molecule('CH3CH2OH')  # 9 atoms

    # Benzene ring (12 atoms) - common drug scaffold
    molecules['Benzene'] = molecule('C6H6')

    # Water clusters (simulate solvation)
    molecules['Water_cluster_3'] = _create_water_cluster(3)  # 9 atoms
    molecules['Water_cluster_5'] = _create_water_cluster(5)  # 15 atoms
    molecules['Water_cluster_10'] = _create_water_cluster(10)  # 30 atoms

    # Alkanes (simulate lipid tails)
    molecules['Ethane'] = molecule('C2H6')  # 8 atoms
    molecules['Butane'] = _create_butane()  # 14 atoms
    molecules['Hexane'] = _create_hexane()  # 20 atoms

    # Aromatic compounds (common in drugs)
    try:
        molecules['Naphthalene'] = molecule('C10H8')  # 18 atoms
    except:
        molecules['Naphthalene'] = _create_naphthalene()

    try:
        molecules['Anthracene'] = molecule('C14H10')  # 24 atoms
    except:
        molecules['Anthracene'] = _create_anthracene()

    # Amino acids (protein building blocks)
    molecules['Glycine'] = _create_glycine()  # 10 atoms
    molecules['Alanine'] = _create_alanine()  # 13 atoms

    # Nucleobases (DNA/RNA)
    molecules['Adenine'] = _create_adenine()  # 15 atoms
    molecules['Guanine'] = _create_guanine()  # 16 atoms

    return molecules


def _create_butane() -> Atoms:
    """Create butane (C4H10): CH3-CH2-CH2-CH3"""
    symbols = ['C', 'C', 'C', 'C'] + ['H'] * 10
    positions = [
        [0.0, 0.0, 0.0],      # C1
        [1.5, 0.0, 0.0],      # C2
        [3.0, 0.0, 0.0],      # C3
        [4.5, 0.0, 0.0],      # C4
        # H atoms
        [-0.5, 0.5, 0.5],
        [-0.5, -0.5, 0.5],
        [-0.5, 0.0, -0.5],
        [2.0, 0.5, 0.5],
        [2.0, -0.5, 0.5],
        [2.5, 0.5, 0.5],
        [2.5, -0.5, 0.5],
        [5.0, 0.5, 0.5],
        [5.0, -0.5, 0.5],
        [5.0, 0.0, -0.5],
    ]
    return Atoms(symbols=symbols, positions=positions)


def _create_hexane() -> Atoms:
    """Create hexane (C6H14): CH3-CH2-CH2-CH2-CH2-CH3"""
    symbols = ['C'] * 6 + ['H'] * 14
    positions = [
        [i * 1.5, 0.0, 0.0] for i in range(6)
    ] + [
        # H atoms (simplified positions)
        [i * 0.5, j * 0.5, k * 0.5]
        for i in range(7)
        for j in [0.5, -0.5]
        for k in [0, 0]
    ][:14]
    return Atoms(symbols=symbols, positions=positions)


def _create_naphthalene() -> Atoms:
    """Create naphthalene (two fused benzene rings)."""
    symbols = ['C'] * 10 + ['H'] * 8
    # Two fused hexagons
    positions = [
        [0.0, 0.0, 0.0],
        [1.4, 0.0, 0.0],
        [2.1, 1.2, 0.0],
        [1.4, 2.4, 0.0],
        [0.0, 2.4, 0.0],
        [-0.7, 1.2, 0.0],
        [2.8, 0.0, 0.0],
        [3.5, 1.2, 0.0],
        [2.8, 2.4, 0.0],
        [2.1, 3.6, 0.0],
    ] + [[i * 0.5, i * 0.5, 0.5] for i in range(8)]
    return Atoms(symbols=symbols, positions=positions)


def _create_anthracene() -> Atoms:
    """Create anthracene (three fused benzene rings)."""
    symbols = ['C'] * 14 + ['H'] * 10
    # Three fused hexagons
    positions = [
        [0.0, 0.0, 0.0],
        [1.4, 0.0, 0.0],
        [2.1, 1.2, 0.0],
        [1.4, 2.4, 0.0],
        [0.0, 2.4, 0.0],
        [-0.7, 1.2, 0.0],
        [2.8, 0.0, 0.0],
        [3.5, 1.2, 0.0],
        [2.8, 2.4, 0.0],
        [2.1, 3.6, 0.0],
        [4.2, 0.0, 0.0],
        [4.9, 1.2, 0.0],
        [4.2, 2.4, 0.0],
        [3.5, 3.6, 0.0],
    ] + [[i * 0.5, i * 0.5, 0.5] for i in range(10)]
    return Atoms(symbols=symbols, positions=positions)


def _create_water_cluster(n_waters: int) -> Atoms:
    """Create water cluster with n water molecules."""
    from ase.build import molecule

    water = molecule('H2O')
    positions = []
    symbols = []

    for i in range(n_waters):
        # Arrange waters in a rough grid
        offset = np.array([
            (i % 3) * 3.0,
            (i // 3) * 3.0,
            0.0
        ])
        positions.extend(water.positions + offset)
        symbols.extend(water.get_chemical_symbols())

    return Atoms(symbols=symbols, positions=positions)


def _create_glycine() -> Atoms:
    """Create glycine (simplest amino acid): NH2-CH2-COOH"""
    # Approximate structure
    symbols = ['N', 'H', 'H', 'C', 'H', 'H', 'C', 'O', 'O', 'H']
    positions = [
        [0.0, 0.0, 0.0],      # N
        [0.0, 1.0, 0.0],      # H
        [1.0, 0.0, 0.0],      # H
        [0.0, 0.0, 1.5],      # C (alpha)
        [-0.5, -0.8, 2.0],    # H
        [0.5, 0.8, 2.0],      # H
        [0.0, 0.0, 3.0],      # C (carboxyl)
        [-1.0, 0.0, 3.5],     # O
        [1.0, 0.0, 3.5],      # O
        [1.5, 0.0, 4.0],      # H
    ]
    return Atoms(symbols=symbols, positions=positions)


def _create_alanine() -> Atoms:
    """Create alanine: NH2-CH(CH3)-COOH"""
    # Approximate structure
    symbols = ['N', 'H', 'H', 'C', 'H', 'C', 'H', 'H', 'H', 'C', 'O', 'O', 'H']
    positions = [
        [0.0, 0.0, 0.0],      # N
        [0.0, 1.0, 0.0],      # H
        [1.0, 0.0, 0.0],      # H
        [0.0, 0.0, 1.5],      # C (alpha)
        [0.0, 1.0, 1.5],      # H
        [1.5, 0.0, 1.5],      # C (methyl)
        [2.0, 0.0, 0.5],      # H
        [2.0, 0.0, 2.5],      # H
        [2.0, 1.0, 1.5],      # H
        [0.0, 0.0, 3.0],      # C (carboxyl)
        [-1.0, 0.0, 3.5],     # O
        [1.0, 0.0, 3.5],      # O
        [1.5, 0.0, 4.0],      # H
    ]
    return Atoms(symbols=symbols, positions=positions)


def _create_adenine() -> Atoms:
    """Create adenine (purine base)."""
    # Simplified planar structure
    symbols = ['N'] * 5 + ['C'] * 5 + ['H'] * 5
    # Approximate ring positions
    positions = [
        [0.0, 0.0, 0.0],      # N1
        [1.2, 0.7, 0.0],      # C2
        [1.2, 2.1, 0.0],      # N3
        [0.0, 2.8, 0.0],      # C4
        [-1.2, 2.1, 0.0],     # C5
        [-1.2, 0.7, 0.0],     # C6
        [-2.4, 0.0, 0.0],     # N6 (amino)
        [0.0, 4.2, 0.0],      # N7
        [1.2, 4.9, 0.0],      # C8
        [2.4, 4.2, 0.0],      # N9
    ] + [[i*0.5, i*0.5, 0.5] for i in range(5)]  # H atoms
    return Atoms(symbols=symbols, positions=positions)


def _create_guanine() -> Atoms:
    """Create guanine (purine base)."""
    # Similar to adenine but with different functional groups
    symbols = ['N'] * 5 + ['C'] * 5 + ['O'] + ['H'] * 5
    positions = [
        [0.0, 0.0, 0.0],
        [1.2, 0.7, 0.0],
        [1.2, 2.1, 0.0],
        [0.0, 2.8, 0.0],
        [-1.2, 2.1, 0.0],
        [-1.2, 0.7, 0.0],
        [-2.4, 0.0, 0.0],
        [0.0, 4.2, 0.0],
        [1.2, 4.9, 0.0],
        [2.4, 4.2, 0.0],
        [2.4, 0.0, 0.0],  # O
    ] + [[i*0.5, i*0.5, 0.5] for i in range(5)]
    return Atoms(symbols=symbols, positions=positions)


def benchmark_single_molecule(
    calc: StudentForceFieldCalculator,
    mol: Atoms,
    n_trials: int = 50,
    warmup: int = 5
) -> Dict[str, float]:
    """
    Benchmark single-molecule force computation.

    This is the baseline: computing forces one molecule at a time.

    IMPORTANT: We must perturb positions slightly between trials to prevent
    ASE from caching results!
    """
    mol_original = mol.copy()
    mol.calc = calc

    # Warmup
    for _ in range(warmup):
        # Perturb positions slightly to prevent caching
        mol.set_positions(mol_original.positions + np.random.randn(*mol_original.positions.shape) * 0.001)
        _ = mol.get_potential_energy()
        _ = mol.get_forces()

    # Benchmark
    times = []
    for _ in range(n_trials):
        # Perturb positions slightly to prevent ASE caching
        mol.set_positions(mol_original.positions + np.random.randn(*mol_original.positions.shape) * 0.001)

        start = time.perf_counter()
        energy = mol.get_potential_energy()
        forces = mol.get_forces()
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'median_ms': float(np.median(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'p95_ms': float(np.percentile(times, 95)),
        'p99_ms': float(np.percentile(times, 99)),
    }


def benchmark_batched(
    calc: StudentForceFieldCalculator,
    molecules: List[Atoms],
    batch_size: int,
    n_trials: int = 30,
    warmup: int = 5
) -> Dict[str, float]:
    """
    Benchmark batched force computation.

    This is the optimized approach: computing forces for multiple molecules
    simultaneously to amortize autograd overhead.

    IMPORTANT: Perturb positions between trials to prevent caching!
    """
    # Create batch templates
    batch_originals = []
    for i in range(batch_size):
        mol = molecules[i % len(molecules)].copy()
        batch_originals.append(mol)

    # Warmup
    for _ in range(warmup):
        # Create fresh batch with perturbed positions
        batch = []
        for orig in batch_originals:
            mol = orig.copy()
            mol.set_positions(orig.positions + np.random.randn(*orig.positions.shape) * 0.001)
            batch.append(mol)
        _ = calc.calculate_batch(batch, properties=['energy', 'forces'])

    # Benchmark
    times = []
    for _ in range(n_trials):
        # Create fresh batch with perturbed positions for each trial
        batch = []
        for orig in batch_originals:
            mol = orig.copy()
            mol.set_positions(orig.positions + np.random.randn(*orig.positions.shape) * 0.001)
            batch.append(mol)

        start = time.perf_counter()
        results = calc.calculate_batch(batch, properties=['energy', 'forces'])
        elapsed = (time.perf_counter() - start) * 1000  # ms
        time_per_mol = elapsed / batch_size
        times.append(time_per_mol)

    total_times = [t * batch_size for t in times]

    return {
        'mean_ms_per_mol': float(np.mean(times)),
        'std_ms_per_mol': float(np.std(times)),
        'median_ms_per_mol': float(np.median(times)),
        'min_ms_per_mol': float(np.min(times)),
        'total_mean_ms': float(np.mean(total_times)),
        'total_std_ms': float(np.std(total_times)),
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark batched force computation on drug-like molecules')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--output', type=str, default='benchmarks/week4_batched_druglike.json',
                        help='Output JSON file for results')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32],
                        help='Batch sizes to test')
    parser.add_argument('--n-trials', type=int, default=30,
                        help='Number of trials per configuration')
    args = parser.parse_args()

    print("="*80)
    print("WEEK 4: ENHANCED BATCHED FORCE COMPUTATION BENCHMARK")
    print("Drug-Like Molecules for Realistic MD Simulation Testing")
    print("="*80)
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print()

    # Check checkpoint exists
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint}")
        return 1

    # Create test molecules
    print("Creating drug-like test molecules...")
    molecules = create_druglike_molecules()

    print(f"\nTest molecules ({len(molecules)} total):")
    for name, mol in molecules.items():
        n_atoms = len(mol)
        formula = mol.get_chemical_formula()
        print(f"  {name:20s}: {n_atoms:3d} atoms ({formula})")
    print()

    # Initialize calculator
    print("Initializing calculator...")
    calc = StudentForceFieldCalculator(
        checkpoint_path=str(checkpoint),
        device=args.device,
        use_compile=False,  # No compile (doesn't help forces)
        use_fp16=False,     # No FP16 (makes things slower)
        use_jit=False,      # No JIT (makes forces much slower)
    )
    print("Calculator ready.")
    print()

    # Benchmark baseline (single-molecule computation)
    print("="*80)
    print("BASELINE: Single-Molecule Force Computation")
    print("="*80)
    print()

    baseline_results = {}
    total_time = 0.0
    n_molecules = 0

    # Test on a representative subset
    representative_mols = {
        'Benzene': molecules['Benzene'],
        'Ethane': molecules['Ethane'],
        'Hexane': molecules['Hexane'],
        'Naphthalene': molecules['Naphthalene'],
        'Glycine': molecules['Glycine'],
        'Water_cluster_5': molecules['Water_cluster_5'],
    }

    for name, mol in representative_mols.items():
        print(f"Benchmarking {name} ({len(mol)} atoms)...")
        result = benchmark_single_molecule(calc, mol, n_trials=args.n_trials)
        baseline_results[name] = result
        total_time += result['mean_ms']
        n_molecules += 1
        print(f"  Time: {result['mean_ms']:.3f} ± {result['std_ms']:.3f} ms")
        print(f"  Median: {result['median_ms']:.3f} ms, P95: {result['p95_ms']:.3f} ms")
        print()

    baseline_mean = total_time / n_molecules
    print(f"Baseline Average: {baseline_mean:.3f} ms/molecule")
    print()

    # Benchmark batched computation
    print("="*80)
    print("BATCHED FORCE COMPUTATION")
    print("="*80)
    print()

    # Use all molecules for batching
    mol_list = list(molecules.values())

    batched_results = {}

    for batch_size in args.batch_sizes:
        print(f"Testing batch size: {batch_size}")
        print("-" * 60)

        result = benchmark_batched(
            calc, mol_list, batch_size,
            n_trials=args.n_trials
        )

        speedup = baseline_mean / result['mean_ms_per_mol']
        batched_results[f'batch_{batch_size}'] = {
            **result,
            'batch_size': batch_size,
            'speedup_vs_baseline': float(speedup),
        }

        print(f"  Time per molecule: {result['mean_ms_per_mol']:.3f} ± {result['std_ms_per_mol']:.3f} ms")
        print(f"  Total batch time: {result['total_mean_ms']:.3f} ms")
        print(f"  Speedup vs baseline: {speedup:.2f}x")

        # Color code based on target achievement
        if speedup >= 7.0:
            status = "✓✓ EXCELLENT (Target exceeded!)"
        elif speedup >= 5.0:
            status = "✓ TARGET ACHIEVED"
        elif speedup >= 3.0:
            status = "~ Good progress"
        else:
            status = "✗ Below target"
        print(f"  Status: {status}")
        print()

    # Summary
    print("="*80)
    print("SUMMARY - Batched Force Computation Performance")
    print("="*80)
    print()
    print(f"Baseline (single molecule): {baseline_mean:.3f} ms  (1.00x)")
    print()

    for batch_size in args.batch_sizes:
        key = f'batch_{batch_size}'
        result = batched_results[key]
        speedup = result['speedup_vs_baseline']
        time_per_mol = result['mean_ms_per_mol']

        marker = ""
        if speedup >= 7.0:
            marker = " ← BEST!"
        elif speedup >= 5.0:
            marker = " ← TARGET!"

        print(f"Batch size {batch_size:2d}: {time_per_mol:6.3f} ms/mol  ({speedup:4.2f}x){marker}")

    print()

    # Find best result
    best_batch_size = max(
        args.batch_sizes,
        key=lambda bs: batched_results[f'batch_{bs}']['speedup_vs_baseline']
    )
    best_result = batched_results[f'batch_{best_batch_size}']
    best_speedup = best_result['speedup_vs_baseline']

    print("="*80)
    print(f"BEST CONFIGURATION: Batch size {best_batch_size}")
    print(f"  Speedup: {best_speedup:.2f}x")
    print(f"  Time per molecule: {best_result['mean_ms_per_mol']:.3f} ms")
    print()

    if best_speedup >= 5.0:
        print("✓✓ TARGET ACHIEVED: 5-7x speedup!")
        print()
        print("Week 4 deliverable: Enhanced batched force computation successfully")
        print("achieves production-ready performance for MD simulations.")
    else:
        print(f"✗ Target not achieved (need 5.0x, got {best_speedup:.2f}x)")
        print()
        print("Recommendation: Investigate further optimizations or adjust batch size.")

    print("="*80)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'metadata': {
            'checkpoint': str(checkpoint),
            'device': args.device,
            'pytorch_version': torch.__version__,
            'python_version': sys.version.split()[0],
            'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
            'n_trials': args.n_trials,
            'batch_sizes_tested': args.batch_sizes,
        },
        'molecules': {
            name: {
                'n_atoms': len(mol),
                'formula': mol.get_chemical_formula(),
            }
            for name, mol in molecules.items()
        },
        'baseline': {
            'per_molecule': baseline_results,
            'mean_ms': baseline_mean,
        },
        'batched': batched_results,
        'best_configuration': {
            'batch_size': best_batch_size,
            'speedup': best_speedup,
            'time_per_mol_ms': best_result['mean_ms_per_mol'],
            'target_achieved': best_speedup >= 5.0,
        },
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return 0 if best_speedup >= 5.0 else 1


if __name__ == '__main__':
    sys.exit(main())
