#!/usr/bin/env python3
"""
Detailed Force Computation Profiling

Profile force computation bottlenecks to guide optimization strategy.
Compares energy-only vs energy+forces performance and identifies where
autograd overhead is coming from.

Usage:
    python scripts/profile_force_computation.py --device cuda
    python scripts/profile_force_computation.py --quick
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import logging

import numpy as np
import torch
from ase.build import molecule, bulk
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
    """Create test molecules of various sizes."""
    molecules = [
        molecule('H2O'),      # 3 atoms
        molecule('NH3'),      # 4 atoms
        molecule('CH4'),      # 5 atoms
        molecule('C2H6'),     # 8 atoms
        molecule('C6H6'),     # 12 atoms (benzene)
    ]

    logger.info(f"Created {len(molecules)} test molecules (3-12 atoms)")
    return molecules


def profile_energy_only(
    model: StudentForceField,
    molecules: List[Atoms],
    n_iterations: int = 50
) -> Dict:
    """
    Profile energy-only computation (no forces).

    This establishes the baseline forward pass performance
    without any autograd overhead.
    """
    logger.info(f"Profiling energy-only computation ({n_iterations} iterations)...")

    results_by_size = {}

    for mol in molecules:
        n_atoms = len(mol)

        # Prepare inputs
        atomic_numbers = torch.tensor(
            mol.get_atomic_numbers(),
            dtype=torch.long,
            device=model.embedding.weight.device
        )
        positions = torch.tensor(
            mol.get_positions(),
            dtype=torch.float32,
            device=model.embedding.weight.device
        )

        # Warmup
        with torch.no_grad():
            for _ in range(5):
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

        results_by_size[n_atoms] = {
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'median_ms': float(np.median(times))
        }

        logger.info(f"  {n_atoms} atoms: {np.mean(times):.3f} ± {np.std(times):.3f} ms")

    return results_by_size


def profile_energy_and_forces_autograd(
    model: StudentForceField,
    molecules: List[Atoms],
    n_iterations: int = 50
) -> Dict:
    """
    Profile energy+forces computation using current autograd approach.

    This shows the REAL performance bottleneck - autograd overhead
    dominates for small molecules.
    """
    logger.info(f"Profiling energy+forces with autograd ({n_iterations} iterations)...")

    results_by_size = {}

    for mol in molecules:
        n_atoms = len(mol)

        # Prepare inputs
        atomic_numbers = torch.tensor(
            mol.get_atomic_numbers(),
            dtype=torch.long,
            device=model.embedding.weight.device
        )
        positions_np = mol.get_positions()

        # Warmup
        for _ in range(5):
            positions = torch.tensor(
                positions_np,
                dtype=torch.float32,
                device=model.embedding.weight.device,
                requires_grad=True
            )
            energy, forces = model.predict_energy_and_forces(
                atomic_numbers, positions
            )

        # Benchmark
        times = []
        for _ in range(n_iterations):
            # Fresh tensor each iteration (required for autograd)
            positions = torch.tensor(
                positions_np,
                dtype=torch.float32,
                device=model.embedding.weight.device,
                requires_grad=True
            )

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.perf_counter()

            energy, forces = model.predict_energy_and_forces(
                atomic_numbers, positions
            )

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

        results_by_size[n_atoms] = {
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'median_ms': float(np.median(times))
        }

        logger.info(f"  {n_atoms} atoms: {np.mean(times):.3f} ± {np.std(times):.3f} ms")

    return results_by_size


def profile_forward_pass_components(
    model: StudentForceField,
    mol: Atoms
) -> Dict:
    """
    Profile individual components of forward pass to identify bottlenecks.

    Breaks down timing into:
    - Neighbor search
    - RBF computation
    - Message passing
    - Energy readout
    - Force computation (autograd)
    """
    logger.info(f"Profiling forward pass components ({len(mol)} atoms)...")

    # Prepare inputs
    device = model.embedding.weight.device
    atomic_numbers = torch.tensor(
        mol.get_atomic_numbers(),
        dtype=torch.long,
        device=device
    )
    positions = torch.tensor(
        mol.get_positions(),
        dtype=torch.float32,
        device=device,
        requires_grad=True
    )

    n_iterations = 100
    results = {}

    # Import radius_graph
    from mlff_distiller.models.student_model import radius_graph

    # 1. Neighbor search
    times = []
    for _ in range(n_iterations):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()

        edge_index = radius_graph(
            positions,
            r=model.cutoff,
            batch=None,
            loop=False,
            use_torch_cluster=model.use_torch_cluster
        )

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.perf_counter()
        times.append((end - start) * 1000)

    results['neighbor_search'] = {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times))
    }
    logger.info(f"  Neighbor search: {np.mean(times):.3f} ± {np.std(times):.3f} ms")

    # Get edge_index for subsequent steps
    edge_index = radius_graph(
        positions,
        r=model.cutoff,
        batch=None,
        loop=False,
        use_torch_cluster=model.use_torch_cluster
    )
    src, dst = edge_index

    # 2. Edge feature computation (RBF + cutoff)
    times = []
    for _ in range(n_iterations):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()

        edge_vector = positions[src] - positions[dst]
        edge_distance = torch.norm(edge_vector, dim=1)
        edge_vector_normalized = edge_vector / (edge_distance.unsqueeze(1) + 1e-8)
        edge_rbf = model.rbf(edge_distance)
        cutoff_values = model.cutoff_fn(edge_distance)
        edge_rbf = edge_rbf * cutoff_values.unsqueeze(-1)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.perf_counter()
        times.append((end - start) * 1000)

    results['edge_features'] = {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times))
    }
    logger.info(f"  Edge features: {np.mean(times):.3f} ± {np.std(times):.3f} ms")

    # 3. Full forward pass (energy only)
    times = []
    with torch.no_grad():
        for _ in range(n_iterations):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.perf_counter()

            energy = model(atomic_numbers, positions.detach())

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.perf_counter()
            times.append((end - start) * 1000)

    results['forward_pass_energy'] = {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times))
    }
    logger.info(f"  Forward pass (energy): {np.mean(times):.3f} ± {np.std(times):.3f} ms")

    # 4. Force computation via autograd
    times = []
    for _ in range(n_iterations):
        # Fresh tensor
        pos = torch.tensor(
            mol.get_positions(),
            dtype=torch.float32,
            device=device,
            requires_grad=True
        )

        # Forward pass
        energy = model(atomic_numbers, pos)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()

        # Backward pass for forces
        forces = -torch.autograd.grad(
            energy,
            pos,
            create_graph=False,
            retain_graph=False
        )[0]

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.perf_counter()
        times.append((end - start) * 1000)

    results['autograd_forces'] = {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times))
    }
    logger.info(f"  Autograd forces: {np.mean(times):.3f} ± {np.std(times):.3f} ms")

    # 5. Combined energy + forces
    times = []
    for _ in range(n_iterations):
        pos = torch.tensor(
            mol.get_positions(),
            dtype=torch.float32,
            device=device,
            requires_grad=True
        )

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()

        energy, forces = model.predict_energy_and_forces(
            atomic_numbers, pos
        )

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.perf_counter()
        times.append((end - start) * 1000)

    results['energy_and_forces_total'] = {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times))
    }
    logger.info(f"  Energy+forces total: {np.mean(times):.3f} ± {np.std(times):.3f} ms")

    return results


def profile_with_pytorch_profiler(
    model: StudentForceField,
    mol: Atoms,
    output_dir: Path
):
    """
    Use PyTorch profiler to get detailed operation-level timing.
    """
    logger.info("Running PyTorch profiler for force computation...")

    output_dir.mkdir(parents=True, exist_ok=True)

    device = model.embedding.weight.device
    atomic_numbers = torch.tensor(
        mol.get_atomic_numbers(),
        dtype=torch.long,
        device=device
    )
    positions_np = mol.get_positions()

    # Warmup
    for _ in range(10):
        pos = torch.tensor(
            positions_np,
            dtype=torch.float32,
            device=device,
            requires_grad=True
        )
        energy, forces = model.predict_energy_and_forces(atomic_numbers, pos)

    # Profile
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_modules=True
    ) as prof:
        for _ in range(20):
            pos = torch.tensor(
                positions_np,
                dtype=torch.float32,
                device=device,
                requires_grad=True
            )
            energy, forces = model.predict_energy_and_forces(atomic_numbers, pos)

    # Save results
    profile_file = output_dir / 'force_computation_profile.txt'
    with open(profile_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write(f"Force Computation Profile - {len(mol)} atoms\n")
        f.write("=" * 100 + "\n\n")

        f.write("Top 30 operations by CUDA time:\n")
        f.write("-" * 100 + "\n")
        f.write(prof.key_averages().table(
            sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
            row_limit=30
        ))
        f.write("\n\n")

        f.write("Top 30 operations by CPU time:\n")
        f.write("-" * 100 + "\n")
        f.write(prof.key_averages().table(
            sort_by="cpu_time_total",
            row_limit=30
        ))
        f.write("\n\n")

        f.write("Operations with 'autograd' in name:\n")
        f.write("-" * 100 + "\n")
        autograd_ops = [
            event for event in prof.key_averages()
            if 'autograd' in event.key.lower() or 'backward' in event.key.lower()
        ]
        if autograd_ops:
            for event in sorted(
                autograd_ops,
                key=lambda x: x.self_cuda_time_total if torch.cuda.is_available() else x.self_cpu_time_total,
                reverse=True
            )[:20]:
                cpu_ms = event.self_cpu_time_total / 1000
                cuda_ms = event.self_cuda_time_total / 1000 if torch.cuda.is_available() else 0
                f.write(f"  {event.key}\n")
                f.write(f"    CPU: {cpu_ms:.3f} ms, CUDA: {cuda_ms:.3f} ms, Calls: {event.count}\n")
        else:
            f.write("  (no autograd operations found)\n")

    logger.info(f"Profiling results saved to {profile_file}")

    # Also save Chrome trace for visualization
    trace_file = output_dir / 'force_computation_trace.json'
    prof.export_chrome_trace(str(trace_file))
    logger.info(f"Chrome trace saved to {trace_file}")
    logger.info(f"  View at: chrome://tracing")


def analyze_autograd_overhead(
    energy_only_results: Dict,
    energy_forces_results: Dict
) -> Dict:
    """
    Analyze how much overhead autograd adds vs energy-only computation.
    """
    logger.info("\nAnalyzing autograd overhead...")

    overhead_analysis = {}

    print("\n" + "=" * 80)
    print("AUTOGRAD OVERHEAD ANALYSIS")
    print("=" * 80)
    print(f"{'Atoms':<8} {'Energy-Only':<15} {'Energy+Forces':<15} {'Overhead':<15} {'Slowdown':<10}")
    print("-" * 80)

    for n_atoms in sorted(energy_only_results.keys()):
        energy_time = energy_only_results[n_atoms]['mean_ms']
        forces_time = energy_forces_results[n_atoms]['mean_ms']
        overhead = forces_time - energy_time
        slowdown = forces_time / energy_time

        overhead_analysis[n_atoms] = {
            'energy_only_ms': energy_time,
            'energy_forces_ms': forces_time,
            'overhead_ms': overhead,
            'slowdown_factor': slowdown,
            'overhead_percentage': (overhead / energy_time) * 100
        }

        print(f"{n_atoms:<8} {energy_time:<15.3f} {forces_time:<15.3f} "
              f"{overhead:<15.3f} {slowdown:<10.2f}x")

    print("=" * 80)

    # Calculate average overhead
    avg_slowdown = np.mean([v['slowdown_factor'] for v in overhead_analysis.values()])
    avg_overhead_pct = np.mean([v['overhead_percentage'] for v in overhead_analysis.values()])

    print(f"\nAverage slowdown from autograd: {avg_slowdown:.2f}x")
    print(f"Average overhead: {avg_overhead_pct:.1f}%")
    print()

    return overhead_analysis


def generate_optimization_recommendations(
    component_profile: Dict,
    overhead_analysis: Dict
) -> List[str]:
    """
    Generate specific optimization recommendations based on profiling.
    """
    recommendations = []

    # Analyze component timings
    forward_time = component_profile['forward_pass_energy']['mean_ms']
    autograd_time = component_profile['autograd_forces']['mean_ms']
    total_time = component_profile['energy_and_forces_total']['mean_ms']

    autograd_fraction = autograd_time / total_time

    recommendations.append(
        f"Force computation (autograd) takes {autograd_fraction*100:.1f}% of total time"
    )

    # High-priority recommendations
    if autograd_fraction > 0.5:
        recommendations.append(
            "HIGH PRIORITY: Implement analytical gradients to eliminate autograd overhead"
        )
        recommendations.append(
            f"  - Expected speedup: {1/autograd_fraction:.1f}x (eliminate {autograd_fraction*100:.0f}% overhead)"
        )
        recommendations.append(
            "  - Implementation: Compute force gradients analytically during forward pass"
        )

    # Analyze neighbor search
    neighbor_time = component_profile['neighbor_search']['mean_ms']
    neighbor_fraction = neighbor_time / total_time

    if neighbor_fraction > 0.2:
        recommendations.append(
            f"MEDIUM PRIORITY: Optimize neighbor search ({neighbor_fraction*100:.1f}% of time)"
        )
        recommendations.append(
            "  - Consider: Custom CUDA kernel for radius search"
        )
        recommendations.append(
            "  - Consider: Cache neighbor lists if geometry doesn't change much"
        )

    # Edge features
    edge_time = component_profile['edge_features']['mean_ms']
    edge_fraction = edge_time / total_time

    if edge_fraction > 0.15:
        recommendations.append(
            f"MEDIUM PRIORITY: Optimize edge features ({edge_fraction*100:.1f}% of time)"
        )
        recommendations.append(
            "  - Consider: Fused RBF+cutoff kernel"
        )
        recommendations.append(
            "  - Consider: Custom Triton kernel for RBF computation"
        )

    # Batching recommendation
    recommendations.append(
        "OPTIMIZATION: Implement batched force computation"
    )
    recommendations.append(
        "  - Compute forces for multiple structures simultaneously"
    )
    recommendations.append(
        "  - Expected speedup: 5-10x throughput improvement"
    )

    return recommendations


def main():
    parser = argparse.ArgumentParser(
        description="Profile force computation bottlenecks"
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
        default='benchmarks/force_profiling',
        help='Output directory'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick profiling with fewer iterations'
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Force Computation Profiling")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Output: {output_dir}")
    logger.info("")

    # Load model
    logger.info("Loading model...")
    model = StudentForceField.load(args.checkpoint, device=args.device)
    model.eval()
    logger.info(f"Loaded model: {model.num_parameters():,} parameters\n")

    # Create test molecules
    molecules = create_test_molecules()

    # Profiling iterations
    n_iter = 20 if args.quick else 50

    # 1. Energy-only profiling
    print("\n" + "=" * 80)
    print("1. ENERGY-ONLY COMPUTATION")
    print("=" * 80)
    energy_only_results = profile_energy_only(model, molecules, n_iter)

    # 2. Energy+forces profiling
    print("\n" + "=" * 80)
    print("2. ENERGY + FORCES COMPUTATION (AUTOGRAD)")
    print("=" * 80)
    energy_forces_results = profile_energy_and_forces_autograd(model, molecules, n_iter)

    # 3. Component profiling
    print("\n" + "=" * 80)
    print("3. COMPONENT-LEVEL PROFILING")
    print("=" * 80)
    test_mol = molecules[2]  # CH4 - medium size
    component_profile = profile_forward_pass_components(model, test_mol)

    # 4. Overhead analysis
    overhead_analysis = analyze_autograd_overhead(
        energy_only_results,
        energy_forces_results
    )

    # 5. PyTorch profiler
    if not args.quick:
        print("\n" + "=" * 80)
        print("4. PYTORCH PROFILER (DETAILED)")
        print("=" * 80)
        profile_with_pytorch_profiler(model, test_mol, output_dir)

    # Generate recommendations
    print("\n" + "=" * 80)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 80)
    recommendations = generate_optimization_recommendations(
        component_profile,
        overhead_analysis
    )
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")

    # Save results
    import json
    results_file = output_dir / 'force_profiling_results.json'
    results = {
        'energy_only': energy_only_results,
        'energy_and_forces': energy_forces_results,
        'component_profile': component_profile,
        'overhead_analysis': overhead_analysis,
        'recommendations': recommendations,
        'config': {
            'checkpoint': args.checkpoint,
            'device': args.device,
            'n_iterations': n_iter
        }
    }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {results_file}")
    logger.info("\nNext steps:")
    logger.info("1. Review profiling results and recommendations")
    logger.info("2. Implement analytical force gradients")
    logger.info("3. Benchmark analytical forces vs autograd")
    logger.info("4. Optimize critical kernels with CUDA/Triton")

    return 0


if __name__ == '__main__':
    sys.exit(main())
