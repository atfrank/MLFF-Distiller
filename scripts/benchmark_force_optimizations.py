#!/usr/bin/env python3
"""
Comprehensive Force Computation Benchmarks

Compare different optimization strategies for force computation:
1. Baseline (autograd)
2. Batched computation
3. Optimized neighbor search
4. Combined optimizations

Usage:
    python scripts/benchmark_force_optimizations.py --device cuda
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List
import logging

import numpy as np
import torch
from ase.build import molecule
from ase import Atoms

# Add src to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from mlff_distiller.models.student_model import StudentForceField

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_molecules() -> List[Atoms]:
    """Create test molecules."""
    return [
        molecule('H2O'),
        molecule('CH4'),
        molecule('NH3'),
        molecule('C2H6'),
        molecule('C6H6'),
    ]


def benchmark_baseline_forces(
    model: StudentForceField,
    molecules: List[Atoms],
    n_iterations: int = 50
) -> Dict:
    """Benchmark baseline autograd forces."""
    logger.info(f"Benchmarking baseline (autograd) forces ({n_iterations} iter)...")

    device = model.embedding.weight.device
    times = []

    for mol in molecules:
        atomic_numbers = torch.tensor(
            mol.get_atomic_numbers(),
            dtype=torch.long,
            device=device
        )
        positions_np = mol.get_positions()

        # Warmup
        for _ in range(5):
            pos = torch.tensor(positions_np, dtype=torch.float32, device=device)
            energy, forces = model.predict_energy_and_forces(atomic_numbers, pos)

        # Benchmark
        mol_times = []
        for _ in range(n_iterations):
            pos = torch.tensor(positions_np, dtype=torch.float32, device=device)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.perf_counter()

            energy, forces = model.predict_energy_and_forces(atomic_numbers, pos)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.perf_counter()
            mol_times.append((end - start) * 1000)

        times.append(np.mean(mol_times))
        logger.info(f"  {len(mol)} atoms: {np.mean(mol_times):.3f} ms")

    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'times_by_size': times
    }


def benchmark_batched_forces(
    model: StudentForceField,
    molecules: List[Atoms],
    batch_sizes: List[int] = [1, 2, 4, 8],
    n_iterations: int = 20
) -> Dict:
    """Benchmark batched force computation."""
    logger.info("Benchmarking batched force computation...")

    device = model.embedding.weight.device
    results = {}

    for batch_size in batch_sizes:
        if len(molecules) < batch_size:
            continue

        batch = molecules[:batch_size]

        # Prepare batched tensors
        atomic_numbers_list = []
        positions_list = []
        batch_idx_list = []

        for i, mol in enumerate(batch):
            n_atoms = len(mol)
            atomic_numbers_list.append(torch.tensor(
                mol.get_atomic_numbers(),
                dtype=torch.long,
                device=device
            ))
            positions_list.append(torch.tensor(
                mol.get_positions(),
                dtype=torch.float32,
                device=device
            ))
            batch_idx_list.append(torch.full(
                (n_atoms,), i,
                dtype=torch.long,
                device=device
            ))

        atomic_numbers = torch.cat(atomic_numbers_list)
        positions_batch = torch.cat(positions_list)
        batch_idx = torch.cat(batch_idx_list)

        # Warmup
        for _ in range(5):
            pos = positions_batch.clone().requires_grad_(True)
            energies = model(atomic_numbers, pos, batch=batch_idx)
            forces = -torch.autograd.grad(
                energies.sum(),
                pos,
                create_graph=False
            )[0]

        # Benchmark
        times = []
        for _ in range(n_iterations):
            pos = positions_batch.clone().requires_grad_(True)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.perf_counter()

            energies = model(atomic_numbers, pos, batch=batch_idx)
            forces = -torch.autograd.grad(
                energies.sum(),
                pos,
                create_graph=False
            )[0]

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.perf_counter()
            times.append((end - start) * 1000)

        mean_time = np.mean(times)
        time_per_mol = mean_time / batch_size
        throughput = batch_size / (mean_time / 1000)

        results[batch_size] = {
            'total_time_ms': float(mean_time),
            'time_per_mol_ms': float(time_per_mol),
            'throughput_mol_per_sec': float(throughput),
            'std_ms': float(np.std(times))
        }

        logger.info(f"  Batch {batch_size}: {mean_time:.2f} ms total, "
                    f"{time_per_mol:.2f} ms/mol, {throughput:.1f} mol/s")

    return results


def compute_speedups(baseline: Dict, optimized: Dict) -> Dict:
    """Compute speedup factors."""
    baseline_time = baseline['mean_ms']

    speedups = {}
    for batch_size, data in optimized.items():
        time_per_mol = data['time_per_mol_ms']
        speedup = baseline_time / time_per_mol
        speedups[batch_size] = {
            'speedup': float(speedup),
            'time_reduction_pct': float((1 - time_per_mol/baseline_time) * 100)
        }

    return speedups


def main():
    parser = argparse.ArgumentParser(description="Benchmark force optimizations")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pt',
        help='Model checkpoint'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='benchmarks/force_optimizations'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick benchmark'
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("FORCE COMPUTATION OPTIMIZATION BENCHMARKS")
    logger.info("=" * 80)
    logger.info(f"Device: {args.device}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info("")

    # Load model
    logger.info("Loading model...")
    model = StudentForceField.load(args.checkpoint, device=args.device)
    model.eval()
    logger.info(f"Loaded: {model.num_parameters():,} parameters\n")

    # Create test molecules
    molecules = create_test_molecules()
    logger.info(f"Test molecules: {len(molecules)}\n")

    n_iter = 20 if args.quick else 50
    batch_sizes = [1, 2, 4] if args.quick else [1, 2, 4, 8, 16]

    # 1. Baseline
    print("=" * 80)
    print("1. BASELINE (AUTOGRAD)")
    print("=" * 80)
    baseline_results = benchmark_baseline_forces(model, molecules, n_iter)

    # 2. Batched
    print("\n" + "=" * 80)
    print("2. BATCHED COMPUTATION")
    print("=" * 80)
    batched_results = benchmark_batched_forces(
        model, molecules, batch_sizes, n_iter
    )

    # 3. Compute speedups
    print("\n" + "=" * 80)
    print("SPEEDUP ANALYSIS")
    print("=" * 80)
    speedups = compute_speedups(baseline_results, batched_results)

    print(f"Baseline (single molecule): {baseline_results['mean_ms']:.2f} ms")
    print(f"\nBatching speedups:")
    for batch_size in sorted(speedups.keys()):
        sp = speedups[batch_size]
        print(f"  Batch {batch_size:2d}: {sp['speedup']:.2f}x speedup "
              f"({sp['time_reduction_pct']:.1f}% faster)")

    # Save results
    results = {
        'config': {
            'checkpoint': args.checkpoint,
            'device': args.device,
            'n_iterations': n_iter
        },
        'baseline': baseline_results,
        'batched': batched_results,
        'speedups': speedups
    }

    results_file = output_dir / 'force_optimization_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {results_file}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Baseline force computation: {baseline_results['mean_ms']:.2f} ms")
    if batched_results:
        best_batch = max(batched_results.keys())
        best_time = batched_results[best_batch]['time_per_mol_ms']
        best_speedup = baseline_results['mean_ms'] / best_time
        print(f"✓ Best batched (size {best_batch}): {best_time:.2f} ms/mol ({best_speedup:.2f}x)")
    print(f"\nNext steps:")
    print(f"1. Integrate batching into ASE calculator")
    print(f"2. Optimize neighbor search with custom CUDA kernel")
    print(f"3. Implement custom force kernels for critical operations")

    return 0


if __name__ == '__main__':
    sys.exit(main())
