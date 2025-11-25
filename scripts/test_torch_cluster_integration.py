#!/usr/bin/env python3
"""
Test torch-cluster Integration

Quick test script to verify torch-cluster is properly integrated
and produces correct results.

Author: CUDA Optimization Engineer
Date: 2025-11-24
Issue: #25
"""

import sys
from pathlib import Path
import logging

import torch
import numpy as np
from ase.build import molecule

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mlff_distiller.models.student_model import StudentForceField, TORCH_CLUSTER_AVAILABLE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_torch_cluster_availability():
    """Test if torch-cluster is available."""
    logger.info(f"torch-cluster available: {TORCH_CLUSTER_AVAILABLE}")
    if TORCH_CLUSTER_AVAILABLE:
        import torch_cluster
        logger.info(f"torch-cluster version: {torch_cluster.__version__}")
    return TORCH_CLUSTER_AVAILABLE


def test_neighbor_search_equivalence():
    """Test that torch-cluster produces same results as native PyTorch."""
    logger.info("\nTesting neighbor search equivalence...")

    # Create test molecule
    atoms = molecule('H2O')
    positions = torch.tensor(atoms.get_positions(), dtype=torch.float32, device='cuda')
    cutoff = 5.0

    # Import neighbor search functions
    from src.mlff_distiller.models.student_model import (
        radius_graph_native,
        radius_graph_torch_cluster
    )

    # Native PyTorch
    edge_index_native = radius_graph_native(positions, cutoff, loop=False)
    logger.info(f"Native neighbor search: {edge_index_native.shape[1]} edges")

    if not TORCH_CLUSTER_AVAILABLE:
        logger.warning("torch-cluster not available, skipping comparison")
        return True

    # torch-cluster
    edge_index_cluster = radius_graph_torch_cluster(positions, cutoff, loop=False)
    logger.info(f"torch-cluster neighbor search: {edge_index_cluster.shape[1]} edges")

    # Compare results
    # Sort edge indices for comparison
    edge_index_native_sorted = edge_index_native[:, torch.argsort(edge_index_native[0] * 1000 + edge_index_native[1])]
    edge_index_cluster_sorted = edge_index_cluster[:, torch.argsort(edge_index_cluster[0] * 1000 + edge_index_cluster[1])]

    if edge_index_native_sorted.shape != edge_index_cluster_sorted.shape:
        logger.error(f"Shape mismatch: native {edge_index_native_sorted.shape} vs cluster {edge_index_cluster_sorted.shape}")
        return False

    if not torch.allclose(edge_index_native_sorted.float(), edge_index_cluster_sorted.float()):
        logger.error("Edge indices don't match!")
        return False

    logger.info("Neighbor search equivalence: PASS")
    return True


def test_model_forward_equivalence():
    """Test that model produces same results with/without torch-cluster."""
    logger.info("\nTesting model forward pass equivalence...")

    checkpoint_path = Path('checkpoints/best_model.pt')
    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint not found: {checkpoint_path}, skipping test")
        return True

    # Create test molecule
    atoms = molecule('H2O')
    atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device='cuda')
    positions = torch.tensor(atoms.get_positions(), dtype=torch.float32, device='cuda')

    # Load model with native PyTorch
    model_native = StudentForceField.load(checkpoint_path, device='cuda')
    model_native.use_torch_cluster = False
    model_native.eval()

    with torch.no_grad():
        energy_native = model_native(atomic_numbers, positions)

    logger.info(f"Native PyTorch energy: {energy_native.item():.6f} eV")

    if not TORCH_CLUSTER_AVAILABLE:
        logger.warning("torch-cluster not available, skipping comparison")
        return True

    # Load model with torch-cluster
    model_cluster = StudentForceField.load(checkpoint_path, device='cuda')
    model_cluster.use_torch_cluster = True
    model_cluster.eval()

    with torch.no_grad():
        energy_cluster = model_cluster(atomic_numbers, positions)

    logger.info(f"torch-cluster energy: {energy_cluster.item():.6f} eV")

    # Compare results
    energy_diff = abs(energy_native.item() - energy_cluster.item())
    logger.info(f"Energy difference: {energy_diff:.6e} eV")

    if energy_diff > 1e-5:
        logger.error(f"Energy difference too large: {energy_diff:.6e} eV")
        return False

    logger.info("Model forward pass equivalence: PASS")
    return True


def benchmark_speedup():
    """Quick benchmark to measure torch-cluster speedup."""
    logger.info("\nBenchmarking torch-cluster speedup...")

    checkpoint_path = Path('checkpoints/best_model.pt')
    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint not found: {checkpoint_path}, skipping benchmark")
        return

    # Create test molecule (medium size)
    atoms = molecule('C6H6')  # Benzene
    atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device='cuda')
    positions = torch.tensor(atoms.get_positions(), dtype=torch.float32, device='cuda')

    import time

    # Benchmark native PyTorch
    model_native = StudentForceField.load(checkpoint_path, device='cuda')
    model_native.use_torch_cluster = False
    model_native.eval()

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model_native(atomic_numbers, positions)
    torch.cuda.synchronize()

    # Benchmark
    times_native = []
    for _ in range(100):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model_native(atomic_numbers, positions)
        torch.cuda.synchronize()
        times_native.append((time.perf_counter() - start) * 1000)

    mean_native = np.mean(times_native)
    logger.info(f"Native PyTorch: {mean_native:.4f} ms")

    if not TORCH_CLUSTER_AVAILABLE:
        logger.warning("torch-cluster not available, skipping benchmark")
        return

    # Benchmark torch-cluster
    model_cluster = StudentForceField.load(checkpoint_path, device='cuda')
    model_cluster.use_torch_cluster = True
    model_cluster.eval()

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model_cluster(atomic_numbers, positions)
    torch.cuda.synchronize()

    # Benchmark
    times_cluster = []
    for _ in range(100):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model_cluster(atomic_numbers, positions)
        torch.cuda.synchronize()
        times_cluster.append((time.perf_counter() - start) * 1000)

    mean_cluster = np.mean(times_cluster)
    logger.info(f"torch-cluster: {mean_cluster:.4f} ms")

    speedup = mean_native / mean_cluster
    logger.info(f"Speedup: {speedup:.2f}x")

    if speedup < 1.0:
        logger.warning("torch-cluster is SLOWER than native PyTorch!")
    elif speedup < 1.2:
        logger.warning("torch-cluster speedup is minimal (<1.2x)")
    else:
        logger.info(f"torch-cluster provides {speedup:.2f}x speedup")


def main():
    logger.info("="*60)
    logger.info("torch-cluster Integration Test")
    logger.info("="*60)

    # Test 1: Availability
    if not test_torch_cluster_availability():
        logger.warning("\ntorch-cluster not available - model will use native PyTorch")
        logger.warning("Install with: pip install torch-cluster --no-build-isolation")
        return

    # Test 2: Neighbor search equivalence
    if not test_neighbor_search_equivalence():
        logger.error("\nNeighbor search equivalence test FAILED")
        sys.exit(1)

    # Test 3: Model forward equivalence
    if not test_model_forward_equivalence():
        logger.error("\nModel forward equivalence test FAILED")
        sys.exit(1)

    # Test 4: Speedup benchmark
    benchmark_speedup()

    logger.info("\n" + "="*60)
    logger.info("All tests PASSED")
    logger.info("="*60)


if __name__ == '__main__':
    main()
