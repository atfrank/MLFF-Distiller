#!/usr/bin/env python3
"""
Detailed CUDA Kernel Profiling for Optimization

Profiles individual operations in the force computation pipeline to identify
exact bottlenecks for CUDA/Triton kernel optimization.

This script provides:
1. Per-operation timing (RBF, message passing, aggregation, etc.)
2. Memory bandwidth utilization estimates
3. Kernel launch overhead analysis
4. Fusion opportunity identification

Usage:
    python scripts/profile_kernels_detailed.py --device cuda
    python scripts/profile_kernels_detailed.py --nsight  # with Nsight profiling
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import json

import numpy as np
import torch
from ase.build import molecule

# Add src to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from mlff_distiller.models.student_model import StudentForceField, radius_graph

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def profile_operation(func, n_iterations=100, warmup=10):
    """Profile a single operation with warmup."""
    # Warmup
    for _ in range(warmup):
        func()

    # Benchmark
    times = []
    for _ in range(n_iterations):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()

        result = func()

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


def profile_neighbor_search(model, positions, n_iterations=100):
    """Profile neighbor search operation."""
    logger.info("Profiling neighbor search...")

    def neighbor_search_op():
        return radius_graph(
            positions,
            r=model.cutoff,
            batch=None,
            loop=False,
            use_torch_cluster=model.use_torch_cluster
        )

    result = profile_operation(neighbor_search_op, n_iterations)

    # Get edge count for analysis
    edge_index = neighbor_search_op()
    n_edges = edge_index.shape[1]
    n_atoms = positions.shape[0]

    result['n_atoms'] = n_atoms
    result['n_edges'] = n_edges
    result['edges_per_atom'] = n_edges / n_atoms

    logger.info(f"  Neighbor search: {result['mean_ms']:.3f} ± {result['std_ms']:.3f} ms")
    logger.info(f"  Edges: {n_edges} ({result['edges_per_atom']:.1f} per atom)")

    return result


def profile_rbf_computation(model, positions, edge_index, n_iterations=100):
    """Profile RBF and cutoff computation separately and fused."""
    logger.info("Profiling RBF computation...")

    src, dst = edge_index
    edge_vector = positions[src] - positions[dst]
    edge_distance = torch.norm(edge_vector, dim=1)

    results = {}

    # 1. RBF only
    def rbf_op():
        return model.rbf(edge_distance)

    results['rbf_only'] = profile_operation(rbf_op, n_iterations)
    logger.info(f"  RBF only: {results['rbf_only']['mean_ms']:.3f} ms")

    # 2. Cutoff only
    def cutoff_op():
        return model.cutoff_fn(edge_distance)

    results['cutoff_only'] = profile_operation(cutoff_op, n_iterations)
    logger.info(f"  Cutoff only: {results['cutoff_only']['mean_ms']:.3f} ms")

    # 3. RBF + cutoff (current implementation)
    def rbf_cutoff_op():
        edge_rbf = model.rbf(edge_distance)
        cutoff_values = model.cutoff_fn(edge_distance)
        return edge_rbf * cutoff_values.unsqueeze(-1)

    results['rbf_cutoff_fused'] = profile_operation(rbf_cutoff_op, n_iterations)
    logger.info(f"  RBF+cutoff (current): {results['rbf_cutoff_fused']['mean_ms']:.3f} ms")

    # Fusion opportunity
    sequential_time = results['rbf_only']['mean_ms'] + results['cutoff_only']['mean_ms']
    fused_time = results['rbf_cutoff_fused']['mean_ms']
    results['fusion_opportunity'] = {
        'sequential_time_ms': sequential_time,
        'fused_time_ms': fused_time,
        'overhead_ms': fused_time - sequential_time,
        'fusion_benefit_potential': sequential_time / fused_time
    }

    logger.info(f"  Fusion opportunity: {results['fusion_opportunity']['fusion_benefit_potential']:.2f}x")

    return results


def profile_edge_features(model, positions, edge_index, n_iterations=100):
    """Profile edge feature computation (vectors, distances, normalization)."""
    logger.info("Profiling edge feature computation...")

    src, dst = edge_index

    results = {}

    # 1. Edge vector computation
    def edge_vector_op():
        return positions[src] - positions[dst]

    results['edge_vectors'] = profile_operation(edge_vector_op, n_iterations)
    logger.info(f"  Edge vectors: {results['edge_vectors']['mean_ms']:.3f} ms")

    # 2. Distance computation
    def distance_op():
        edge_vector = positions[src] - positions[dst]
        return torch.norm(edge_vector, dim=1)

    results['distances'] = profile_operation(distance_op, n_iterations)
    logger.info(f"  Distances: {results['distances']['mean_ms']:.3f} ms")

    # 3. Normalization
    def normalize_op():
        edge_vector = positions[src] - positions[dst]
        edge_distance = torch.norm(edge_vector, dim=1)
        return edge_vector / (edge_distance.unsqueeze(1) + 1e-8)

    results['normalization'] = profile_operation(normalize_op, n_iterations)
    logger.info(f"  Normalization: {results['normalization']['mean_ms']:.3f} ms")

    # 4. Full edge feature pipeline
    def full_pipeline():
        edge_vector = positions[src] - positions[dst]
        edge_distance = torch.norm(edge_vector, dim=1)
        edge_vector_normalized = edge_vector / (edge_distance.unsqueeze(1) + 1e-8)
        edge_rbf = model.rbf(edge_distance)
        cutoff_values = model.cutoff_fn(edge_distance)
        edge_rbf = edge_rbf * cutoff_values.unsqueeze(-1)
        return edge_rbf, edge_vector_normalized

    results['full_pipeline'] = profile_operation(full_pipeline, n_iterations)
    logger.info(f"  Full edge pipeline: {results['full_pipeline']['mean_ms']:.3f} ms")

    return results


def profile_message_passing(model, positions, edge_index, n_iterations=50):
    """Profile message passing layers."""
    logger.info("Profiling message passing...")

    device = positions.device
    n_atoms = positions.shape[0]

    # Prepare edge features
    src, dst = edge_index
    edge_vector = positions[src] - positions[dst]
    edge_distance = torch.norm(edge_vector, dim=1)
    edge_vector_normalized = edge_vector / (edge_distance.unsqueeze(1) + 1e-8)
    edge_rbf = model.rbf(edge_distance)
    cutoff_values = model.cutoff_fn(edge_distance)
    edge_rbf = edge_rbf * cutoff_values.unsqueeze(-1)

    # Initial features
    atomic_numbers = torch.ones(n_atoms, dtype=torch.long, device=device)
    scalar_features = model.embedding(atomic_numbers)
    vector_features = torch.zeros(n_atoms, 3, model.hidden_dim, device=device)

    results = {}

    # Profile each interaction layer
    for i, interaction in enumerate(model.interactions):
        logger.info(f"  Layer {i+1}...")

        # Message passing
        def message_op():
            return interaction.message(
                scalar_features,
                vector_features,
                edge_index,
                edge_rbf,
                edge_vector_normalized
            )

        layer_results = {}
        layer_results['message'] = profile_operation(message_op, n_iterations)
        logger.info(f"    Message: {layer_results['message']['mean_ms']:.3f} ms")

        # Update
        def update_op():
            return interaction.update(scalar_features, vector_features)

        layer_results['update'] = profile_operation(update_op, n_iterations)
        logger.info(f"    Update: {layer_results['update']['mean_ms']:.3f} ms")

        # Combined
        def combined_op():
            s, v = interaction.message(
                scalar_features,
                vector_features,
                edge_index,
                edge_rbf,
                edge_vector_normalized
            )
            return interaction.update(s, v)

        layer_results['combined'] = profile_operation(combined_op, n_iterations)
        logger.info(f"    Combined: {layer_results['combined']['mean_ms']:.3f} ms")

        results[f'layer_{i+1}'] = layer_results

        # Update for next layer
        scalar_features, vector_features = interaction(
            scalar_features,
            vector_features,
            edge_index,
            edge_rbf,
            edge_vector_normalized
        )

    return results


def profile_energy_readout(model, n_atoms, n_iterations=100):
    """Profile energy readout head."""
    logger.info("Profiling energy readout...")

    device = model.embedding.weight.device

    # Create dummy features
    scalar_features = torch.randn(n_atoms, model.hidden_dim, device=device)

    def readout_op():
        atomic_energies = model.energy_head(scalar_features)
        return torch.sum(atomic_energies)

    results = profile_operation(readout_op, n_iterations)
    logger.info(f"  Energy readout: {results['mean_ms']:.3f} ms")

    return results


def profile_autograd_backward(model, positions, atomic_numbers, n_iterations=50):
    """Profile autograd backward pass for force computation."""
    logger.info("Profiling autograd backward pass...")

    positions_np = positions.cpu().numpy()

    def autograd_op():
        pos = torch.tensor(
            positions_np,
            dtype=torch.float32,
            device=positions.device,
            requires_grad=True
        )
        energy = model(atomic_numbers, pos)
        forces = -torch.autograd.grad(
            energy,
            pos,
            create_graph=False,
            retain_graph=False
        )[0]
        return forces

    results = profile_operation(autograd_op, n_iterations, warmup=5)
    logger.info(f"  Autograd backward: {results['mean_ms']:.3f} ms")

    return results


def estimate_memory_bandwidth(operation_name, data_size_bytes, time_ms):
    """Estimate memory bandwidth utilization."""
    bandwidth_gb_s = (data_size_bytes / 1e9) / (time_ms / 1000)

    # RTX 3080 Ti theoretical peak: ~760 GB/s
    peak_bandwidth = 760.0
    utilization = (bandwidth_gb_s / peak_bandwidth) * 100

    return {
        'bandwidth_gb_s': bandwidth_gb_s,
        'peak_bandwidth_gb_s': peak_bandwidth,
        'utilization_percent': utilization
    }


def identify_fusion_opportunities(profile_results):
    """Identify kernel fusion opportunities from profiling results."""
    logger.info("\nIdentifying fusion opportunities...")

    opportunities = []

    # 1. RBF + Cutoff fusion
    if 'rbf_computation' in profile_results:
        rbf_data = profile_results['rbf_computation']
        if 'fusion_opportunity' in rbf_data:
            fus_opp = rbf_data['fusion_opportunity']
            opportunities.append({
                'name': 'Fused RBF + Cutoff',
                'description': 'Fuse Gaussian RBF and cosine cutoff into single kernel',
                'current_time_ms': fus_opp['fused_time_ms'],
                'potential_speedup': fus_opp['fusion_benefit_potential'],
                'priority': 'HIGH' if fus_opp['fusion_benefit_potential'] > 1.3 else 'MEDIUM'
            })

    # 2. Edge features fusion
    if 'edge_features' in profile_results:
        edge_data = profile_results['edge_features']
        total_time = edge_data['full_pipeline']['mean_ms']
        component_time = (
            edge_data['edge_vectors']['mean_ms'] +
            edge_data['distances']['mean_ms'] +
            edge_data['normalization']['mean_ms']
        )
        if component_time > 0:
            opportunities.append({
                'name': 'Fused Edge Features',
                'description': 'Fuse edge vector, distance, normalization into single kernel',
                'current_time_ms': total_time,
                'potential_speedup': component_time / total_time if total_time > 0 else 1.0,
                'priority': 'MEDIUM'
            })

    # 3. Message passing fusion
    if 'message_passing' in profile_results:
        for layer_name, layer_data in profile_results['message_passing'].items():
            if isinstance(layer_data, dict) and 'message' in layer_data and 'update' in layer_data:
                message_time = layer_data['message']['mean_ms']
                update_time = layer_data['update']['mean_ms']
                combined_time = layer_data['combined']['mean_ms']
                sequential_time = message_time + update_time

                if sequential_time > combined_time:
                    opportunities.append({
                        'name': f'Fused Message+Update ({layer_name})',
                        'description': f'Fuse message and update for {layer_name}',
                        'current_time_ms': combined_time,
                        'potential_speedup': sequential_time / combined_time,
                        'priority': 'MEDIUM'
                    })

    # Sort by potential speedup
    opportunities.sort(key=lambda x: x['potential_speedup'], reverse=True)

    print("\n" + "=" * 100)
    print("KERNEL FUSION OPPORTUNITIES")
    print("=" * 100)
    print(f"{'Priority':<10} {'Opportunity':<30} {'Current (ms)':<15} {'Speedup':<10} {'Description':<30}")
    print("-" * 100)

    for opp in opportunities:
        print(f"{opp['priority']:<10} {opp['name']:<30} {opp['current_time_ms']:<15.3f} "
              f"{opp['potential_speedup']:<10.2f}x {opp['description']:<30}")

    print("=" * 100)

    return opportunities


def generate_optimization_strategy(profile_results):
    """Generate optimization strategy based on profiling."""
    logger.info("\nGenerating optimization strategy...")

    strategy = {
        'week_3': [],
        'week_4': [],
        'expected_speedup': {}
    }

    # Week 3: Quick wins
    strategy['week_3'].extend([
        {
            'task': 'Implement fused RBF + cutoff Triton kernel',
            'expected_speedup': '1.3-1.5x',
            'effort': '1-2 days',
            'priority': 'P0'
        },
        {
            'task': 'Implement fused edge features kernel',
            'expected_speedup': '1.2-1.3x',
            'effort': '1-2 days',
            'priority': 'P1'
        },
        {
            'task': 'Test and validate kernel correctness',
            'expected_speedup': '0x (validation)',
            'effort': '1 day',
            'priority': 'P0'
        }
    ])

    # Week 4: Advanced optimizations
    strategy['week_4'].extend([
        {
            'task': 'Implement fused message passing kernel',
            'expected_speedup': '1.2-1.4x',
            'effort': '2-3 days',
            'priority': 'P1'
        },
        {
            'task': 'Optimize kernel parameters (block size, etc.)',
            'expected_speedup': '1.1-1.2x',
            'effort': '1-2 days',
            'priority': 'P2'
        },
        {
            'task': 'Integrate with batched force computation',
            'expected_speedup': '1.5-2x (for batches)',
            'effort': '2-3 days',
            'priority': 'P1'
        }
    ])

    # Calculate cumulative speedup
    strategy['expected_speedup'] = {
        'week_3_target': '1.5-2x',
        'week_4_target': '2-3x cumulative',
        'total_with_phase3a': '5-7x cumulative (including 3.42x from batching)'
    }

    print("\n" + "=" * 100)
    print("OPTIMIZATION STRATEGY")
    print("=" * 100)

    print("\nWEEK 3 (Days 1-7): Quick Wins")
    print("-" * 100)
    for i, task in enumerate(strategy['week_3'], 1):
        print(f"{i}. [{task['priority']}] {task['task']}")
        print(f"   Expected: {task['expected_speedup']}, Effort: {task['effort']}")

    print("\nWEEK 4 (Days 8-14): Advanced Optimizations")
    print("-" * 100)
    for i, task in enumerate(strategy['week_4'], 1):
        print(f"{i}. [{task['priority']}] {task['task']}")
        print(f"   Expected: {task['expected_speedup']}, Effort: {task['effort']}")

    print("\nEXPECTED CUMULATIVE SPEEDUP")
    print("-" * 100)
    for key, value in strategy['expected_speedup'].items():
        print(f"  {key}: {value}")

    print("=" * 100)

    return strategy


def main():
    parser = argparse.ArgumentParser(
        description="Detailed CUDA kernel profiling for optimization"
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
        default='benchmarks/cuda_kernel_profiling',
        help='Output directory'
    )
    parser.add_argument(
        '--molecule',
        type=str,
        default='C6H6',
        help='Test molecule (default: benzene)'
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        logger.error("CUDA not available! This profiler requires GPU.")
        return 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 100)
    logger.info("DETAILED CUDA KERNEL PROFILING")
    logger.info("=" * 100)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Test molecule: {args.molecule}")
    logger.info(f"Output: {output_dir}")
    logger.info("")

    # Load model
    logger.info("Loading model...")
    model = StudentForceField.load(args.checkpoint, device=args.device)
    model.eval()
    logger.info(f"Loaded model: {model.num_parameters():,} parameters\n")

    # Create test molecule
    mol = molecule(args.molecule)
    n_atoms = len(mol)
    logger.info(f"Test molecule: {args.molecule} ({n_atoms} atoms)\n")

    # Prepare inputs
    device = torch.device(args.device)
    atomic_numbers = torch.tensor(
        mol.get_atomic_numbers(),
        dtype=torch.long,
        device=device
    )
    positions = torch.tensor(
        mol.get_positions(),
        dtype=torch.float32,
        device=device
    )

    # Compute edge index once
    edge_index = radius_graph(
        positions,
        r=model.cutoff,
        batch=None,
        loop=False,
        use_torch_cluster=model.use_torch_cluster
    )

    profile_results = {}

    # 1. Neighbor search
    print("\n" + "=" * 100)
    print("1. NEIGHBOR SEARCH")
    print("=" * 100)
    profile_results['neighbor_search'] = profile_neighbor_search(model, positions)

    # 2. RBF computation
    print("\n" + "=" * 100)
    print("2. RBF COMPUTATION")
    print("=" * 100)
    profile_results['rbf_computation'] = profile_rbf_computation(model, positions, edge_index)

    # 3. Edge features
    print("\n" + "=" * 100)
    print("3. EDGE FEATURES")
    print("=" * 100)
    profile_results['edge_features'] = profile_edge_features(model, positions, edge_index)

    # 4. Message passing
    print("\n" + "=" * 100)
    print("4. MESSAGE PASSING")
    print("=" * 100)
    profile_results['message_passing'] = profile_message_passing(model, positions, edge_index)

    # 5. Energy readout
    print("\n" + "=" * 100)
    print("5. ENERGY READOUT")
    print("=" * 100)
    profile_results['energy_readout'] = profile_energy_readout(model, n_atoms)

    # 6. Autograd backward
    print("\n" + "=" * 100)
    print("6. AUTOGRAD BACKWARD PASS")
    print("=" * 100)
    profile_results['autograd_backward'] = profile_autograd_backward(model, positions, atomic_numbers)

    # Identify fusion opportunities
    fusion_opportunities = identify_fusion_opportunities(profile_results)
    profile_results['fusion_opportunities'] = fusion_opportunities

    # Generate optimization strategy
    optimization_strategy = generate_optimization_strategy(profile_results)
    profile_results['optimization_strategy'] = optimization_strategy

    # Save results
    results_file = output_dir / 'kernel_profiling_detailed.json'
    with open(results_file, 'w') as f:
        json.dump(profile_results, f, indent=2)

    logger.info(f"\n\nResults saved to {results_file}")
    logger.info("\nNext steps:")
    logger.info("1. Review fusion opportunities (highest priority first)")
    logger.info("2. Implement fused RBF + cutoff Triton kernel")
    logger.info("3. Benchmark kernel performance")
    logger.info("4. Integrate with student model")
    logger.info("5. Measure end-to-end speedup")

    return 0


if __name__ == '__main__':
    sys.exit(main())
