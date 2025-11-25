#!/usr/bin/env python3
"""
Detailed Profiling for CUDA-X Library Analysis

This script performs comprehensive profiling to identify computational bottlenecks
and recommend appropriate CUDA-X libraries for optimization.

Usage:
    python benchmarks/profile_detailed.py --checkpoint checkpoints/best_model.pt
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import logging

import numpy as np
import torch
import torch.nn.functional as F
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


def profile_with_nvtx(model: StudentForceField, atoms: Atoms, device: str) -> Dict:
    """
    Profile model with NVTX markers for detailed GPU analysis.

    This enables detailed profiling with nsys/nvprof:
        nsys profile -o profile.qdrep python benchmarks/profile_detailed.py
    """
    logger.info("Profiling with NVTX markers...")

    # Prepare inputs
    atomic_numbers = torch.tensor(atoms.numbers, dtype=torch.long, device=device)
    positions = torch.tensor(atoms.positions, dtype=torch.float32, device=device, requires_grad=True)

    # Warmup
    for _ in range(10):
        energy = model(atomic_numbers, positions)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Profile with NVTX markers
    times = {}

    # Full forward pass
    start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push("forward_pass")

    energy = model(atomic_numbers, positions)

    if torch.cuda.is_available():
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()

    times['forward_pass'] = (time.perf_counter() - start) * 1000

    # Force computation
    start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push("force_computation")

    forces = -torch.autograd.grad(energy, positions, create_graph=False)[0]

    if torch.cuda.is_available():
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()

    times['force_computation'] = (time.perf_counter() - start) * 1000

    return times


def profile_operations(model: StudentForceField, atoms: Atoms, device: str) -> Dict:
    """
    Profile individual operations to identify bottlenecks.

    Returns timing breakdown for:
    - Neighbor search (radius_graph)
    - RBF computation
    - Message passing (per layer)
    - Update layers (per layer)
    - Energy readout
    - Force computation
    """
    logger.info("Profiling individual operations...")

    # Prepare inputs
    atomic_numbers = torch.tensor(atoms.numbers, dtype=torch.long, device=device)
    positions = torch.tensor(atoms.positions, dtype=torch.float32, device=device, requires_grad=True)

    # Warmup
    for _ in range(10):
        _ = model(atomic_numbers, positions)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = {}
    iterations = 100

    # Profile neighbor search
    logger.info("  Profiling neighbor search...")
    neighbor_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        from mlff_distiller.models.student_model import radius_graph_native
        edge_index = radius_graph_native(positions, r=model.cutoff, batch=None, loop=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        neighbor_times.append((time.perf_counter() - start) * 1000)

    times['neighbor_search'] = {
        'mean_ms': float(np.mean(neighbor_times)),
        'std_ms': float(np.std(neighbor_times)),
        'min_ms': float(np.min(neighbor_times)),
        'max_ms': float(np.max(neighbor_times))
    }

    # Get edge info for subsequent operations
    from mlff_distiller.models.student_model import radius_graph_native
    edge_index = radius_graph_native(positions, r=model.cutoff, batch=None, loop=False)
    src, dst = edge_index
    edge_vector = positions[src] - positions[dst]
    edge_distance = torch.norm(edge_vector, dim=1)

    # Profile RBF computation
    logger.info("  Profiling RBF computation...")
    rbf_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        edge_rbf = model.rbf(edge_distance)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        rbf_times.append((time.perf_counter() - start) * 1000)

    times['rbf_computation'] = {
        'mean_ms': float(np.mean(rbf_times)),
        'std_ms': float(np.std(rbf_times)),
    }

    # Profile cutoff function
    logger.info("  Profiling cutoff function...")
    cutoff_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        cutoff_values = model.cutoff_fn(edge_distance)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        cutoff_times.append((time.perf_counter() - start) * 1000)

    times['cutoff_function'] = {
        'mean_ms': float(np.mean(cutoff_times)),
        'std_ms': float(np.std(cutoff_times)),
    }

    # Profile embedding
    logger.info("  Profiling atomic embedding...")
    embed_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        scalar_features = model.embedding(atomic_numbers)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        embed_times.append((time.perf_counter() - start) * 1000)

    times['embedding'] = {
        'mean_ms': float(np.mean(embed_times)),
        'std_ms': float(np.std(embed_times)),
    }

    # Profile message passing layers
    logger.info("  Profiling message passing layers...")
    num_atoms = len(atomic_numbers)
    scalar_features = model.embedding(atomic_numbers)
    vector_features = torch.zeros(num_atoms, 3, model.hidden_dim, dtype=positions.dtype, device=device)
    edge_rbf = model.rbf(edge_distance)
    cutoff_values = model.cutoff_fn(edge_distance)
    edge_rbf = edge_rbf * cutoff_values.unsqueeze(-1)
    edge_vector_normalized = edge_vector / (edge_distance.unsqueeze(1) + 1e-8)

    for layer_idx, interaction in enumerate(model.interactions):
        message_times = []
        update_times = []

        for _ in range(iterations):
            # Profile message passing
            start = time.perf_counter()
            s_msg, v_msg = interaction.message(
                scalar_features, vector_features, edge_index, edge_rbf, edge_vector_normalized
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            message_times.append((time.perf_counter() - start) * 1000)

            # Profile update
            start = time.perf_counter()
            s_upd, v_upd = interaction.update(s_msg, v_msg)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            update_times.append((time.perf_counter() - start) * 1000)

            scalar_features = s_upd
            vector_features = v_upd

        times[f'message_layer_{layer_idx}'] = {
            'mean_ms': float(np.mean(message_times)),
            'std_ms': float(np.std(message_times)),
        }

        times[f'update_layer_{layer_idx}'] = {
            'mean_ms': float(np.mean(update_times)),
            'std_ms': float(np.std(update_times)),
        }

    # Profile energy readout
    logger.info("  Profiling energy readout...")
    readout_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        atomic_energies = model.energy_head(scalar_features)
        total_energy = torch.sum(atomic_energies)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        readout_times.append((time.perf_counter() - start) * 1000)

    times['energy_readout'] = {
        'mean_ms': float(np.mean(readout_times)),
        'std_ms': float(np.std(readout_times)),
    }

    # Profile full forward pass for comparison
    logger.info("  Profiling full forward pass...")
    forward_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        energy = model(atomic_numbers, positions)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        forward_times.append((time.perf_counter() - start) * 1000)

    times['full_forward'] = {
        'mean_ms': float(np.mean(forward_times)),
        'std_ms': float(np.std(forward_times)),
    }

    # Profile force computation
    logger.info("  Profiling force computation...")
    force_times = []
    for _ in range(iterations):
        energy = model(atomic_numbers, positions)
        start = time.perf_counter()
        forces = -torch.autograd.grad(energy, positions, create_graph=False)[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        force_times.append((time.perf_counter() - start) * 1000)

    times['force_computation'] = {
        'mean_ms': float(np.mean(force_times)),
        'std_ms': float(np.std(force_times)),
    }

    return times


def analyze_memory_patterns(model: StudentForceField, atoms: Atoms, device: str) -> Dict:
    """
    Analyze memory access patterns and allocation overhead.
    """
    logger.info("Analyzing memory patterns...")

    if not torch.cuda.is_available():
        return {'error': 'CUDA not available'}

    # Prepare inputs
    atomic_numbers = torch.tensor(atoms.numbers, dtype=torch.long, device=device)
    positions = torch.tensor(atoms.positions, dtype=torch.float32, device=device, requires_grad=True)

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    baseline = torch.cuda.memory_allocated()

    # Forward pass
    energy = model(atomic_numbers, positions)
    after_forward = torch.cuda.memory_allocated()

    # Force computation
    forces = -torch.autograd.grad(energy, positions, create_graph=False)[0]
    after_forces = torch.cuda.memory_allocated()

    peak = torch.cuda.max_memory_allocated()

    return {
        'baseline_mb': baseline / 1024**2,
        'after_forward_mb': after_forward / 1024**2,
        'after_forces_mb': after_forces / 1024**2,
        'peak_mb': peak / 1024**2,
        'forward_overhead_mb': (after_forward - baseline) / 1024**2,
        'force_overhead_mb': (after_forces - after_forward) / 1024**2,
    }


def profile_with_pytorch_profiler(model: StudentForceField, atoms: Atoms, device: str, output_dir: Path) -> Dict:
    """
    Use PyTorch profiler for detailed operation-level profiling.
    """
    logger.info("Running PyTorch profiler...")

    # Prepare inputs
    atomic_numbers = torch.tensor(atoms.numbers, dtype=torch.long, device=device)
    positions = torch.tensor(atoms.positions, dtype=torch.float32, device=device, requires_grad=True)

    # Warmup
    for _ in range(10):
        energy = model(atomic_numbers, positions)
        forces = -torch.autograd.grad(energy, positions, create_graph=False)[0]

    # Profile
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
    ) as prof:
        for _ in range(10):
            energy = model(atomic_numbers, positions)
            forces = -torch.autograd.grad(energy, positions, create_graph=False)[0]

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed table
    profile_file = output_dir / 'pytorch_profiler_detailed.txt'
    with open(profile_file, 'w') as f:
        f.write("=" * 120 + "\n")
        f.write("PyTorch Profiler Results - Sorted by CUDA Time\n")
        f.write("=" * 120 + "\n\n")
        f.write(prof.key_averages().table(
            sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
            row_limit=100,
            max_name_column_width=60
        ))
        f.write("\n\n")
        f.write("=" * 120 + "\n")
        f.write("PyTorch Profiler Results - Sorted by CPU Time\n")
        f.write("=" * 120 + "\n\n")
        f.write(prof.key_averages().table(
            sort_by="cpu_time_total",
            row_limit=100,
            max_name_column_width=60
        ))
        f.write("\n\n")
        f.write("=" * 120 + "\n")
        f.write("PyTorch Profiler Results - Memory Usage\n")
        f.write("=" * 120 + "\n\n")
        f.write(prof.key_averages().table(
            sort_by="self_cuda_memory_usage" if torch.cuda.is_available() else "self_cpu_memory_usage",
            row_limit=50,
            max_name_column_width=60
        ))

    logger.info(f"  Saved detailed profiling to {profile_file}")

    # Export Chrome trace for visualization
    trace_file = output_dir / 'trace.json'
    try:
        prof.export_chrome_trace(str(trace_file))
        logger.info(f"  Saved Chrome trace to {trace_file}")
        logger.info(f"  View trace at: chrome://tracing/")
    except Exception as e:
        logger.warning(f"  Could not export Chrome trace: {e}")
        trace_file = None

    # Extract key statistics
    key_averages = prof.key_averages()

    # Get top operations by CUDA time
    top_cuda_ops = []
    for event in sorted(
        key_averages,
        key=lambda x: x.cuda_time_total if hasattr(x, 'cuda_time_total') else 0,
        reverse=True
    )[:20]:
        top_cuda_ops.append({
            'name': event.key,
            'cuda_time_us': event.cuda_time_total if hasattr(event, 'cuda_time_total') else 0,
            'cpu_time_us': event.cpu_time_total,
            'calls': event.count,
            'cuda_memory_mb': event.cuda_memory_usage / 1024**2 if hasattr(event, 'cuda_memory_usage') else 0
        })

    return {
        'profile_file': str(profile_file),
        'trace_file': str(trace_file) if trace_file else None,
        'top_cuda_operations': top_cuda_ops
    }


def analyze_scalability(model: StudentForceField, device: str) -> Dict:
    """
    Analyze how different operations scale with system size.
    """
    logger.info("Analyzing scalability with system size...")

    sizes = [5, 10, 20, 30, 50, 75, 100]
    results = []

    for n_atoms in sizes:
        logger.info(f"  Testing {n_atoms} atoms...")

        # Create test system
        positions = torch.randn(n_atoms, 3, device=device) * 5.0  # Random positions in 5A box
        atomic_numbers = torch.ones(n_atoms, dtype=torch.long, device=device) * 6  # All carbon
        positions.requires_grad_(True)

        # Warmup
        for _ in range(5):
            _ = model(atomic_numbers, positions)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Time operations
        iterations = 50

        # Neighbor search
        from mlff_distiller.models.student_model import radius_graph_native
        neighbor_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            edge_index = radius_graph_native(positions, r=model.cutoff, batch=None, loop=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            neighbor_times.append((time.perf_counter() - start) * 1000)

        # Full forward
        forward_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            energy = model(atomic_numbers, positions)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_times.append((time.perf_counter() - start) * 1000)

        # Forces
        force_times = []
        for _ in range(iterations):
            energy = model(atomic_numbers, positions)
            start = time.perf_counter()
            forces = -torch.autograd.grad(energy, positions, create_graph=False)[0]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            force_times.append((time.perf_counter() - start) * 1000)

        # Get edge count
        edge_index = radius_graph_native(positions, r=model.cutoff, batch=None, loop=False)
        n_edges = edge_index.shape[1]

        results.append({
            'n_atoms': n_atoms,
            'n_edges': n_edges,
            'neighbor_search_ms': float(np.mean(neighbor_times)),
            'forward_pass_ms': float(np.mean(forward_times)),
            'force_computation_ms': float(np.mean(force_times)),
            'total_ms': float(np.mean(forward_times) + np.mean(force_times))
        })

    return {'scalability': results}


def generate_cuda_x_recommendations(profiling_data: Dict, output_file: Path):
    """
    Generate CUDA-X library recommendations based on profiling data.
    """
    logger.info("Generating CUDA-X library recommendations...")

    with open(output_file, 'w') as f:
        f.write("# CUDA-X Library Analysis and Recommendations\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Model**: PaiNN Student (427K parameters)\n\n")

        f.write("---\n\n")

        f.write("## Executive Summary\n\n")

        # Analyze bottlenecks
        op_times = profiling_data.get('operation_breakdown', {})

        f.write("### Primary Bottlenecks\n\n")

        # Calculate percentages
        if 'full_forward' in op_times:
            total_time = op_times['full_forward']['mean_ms']
            neighbor_pct = (op_times.get('neighbor_search', {}).get('mean_ms', 0) / total_time) * 100
            message_pct = sum(
                op_times.get(f'message_layer_{i}', {}).get('mean_ms', 0)
                for i in range(3)
            ) / total_time * 100
            update_pct = sum(
                op_times.get(f'update_layer_{i}', {}).get('mean_ms', 0)
                for i in range(3)
            ) / total_time * 100
            force_pct = (op_times.get('force_computation', {}).get('mean_ms', 0) / total_time) * 100

            f.write(f"1. **Neighbor Search**: {neighbor_pct:.1f}% of forward pass time\n")
            f.write(f"2. **Message Passing**: {message_pct:.1f}% of forward pass time\n")
            f.write(f"3. **Update Layers**: {update_pct:.1f}% of forward pass time\n")
            f.write(f"4. **Force Computation**: {force_pct:.1f}% of total time\n\n")

        f.write("---\n\n")

        f.write("## Detailed Analysis by Operation\n\n")

        # 1. Neighbor Search
        f.write("### 1. Neighbor Search (radius_graph)\n\n")
        if 'neighbor_search' in op_times:
            ns_time = op_times['neighbor_search']['mean_ms']
            f.write(f"**Current Performance**: {ns_time:.3f} ms\n\n")

        f.write("**Current Implementation**: Pure PyTorch with pairwise distance matrix\n")
        f.write("- Complexity: O(N²) for N atoms\n")
        f.write("- Memory: O(N²) for distance matrix\n")
        f.write("- Not optimized for sparse neighbor lists\n\n")

        f.write("**CUDA-X Recommendation**: ❌ **No direct library support**\n\n")
        f.write("**Rationale**:\n")
        f.write("- cuSPARSE: Not applicable (not a sparse matrix operation)\n")
        f.write("- cuGraph: Designed for static graph analytics, not dynamic k-NN\n")
        f.write("- Thrust/CUB: Could help with sorting/filtering, but won't solve core O(N²) issue\n\n")

        f.write("**Best Optimization Strategy**: 🎯 **Custom CUDA Kernel**\n\n")
        f.write("**Approach**:\n")
        f.write("1. Cell list (spatial hashing) algorithm: O(N) instead of O(N²)\n")
        f.write("2. Use CUB for efficient atomic operations and prefix sums\n")
        f.write("3. Or integrate existing library: torch-cluster, PyG radius\n\n")

        f.write("**Expected Speedup**: 5-10x for systems >50 atoms\n\n")
        f.write("**Implementation Difficulty**: Medium (or Easy if using torch-cluster)\n\n")

        f.write("---\n\n")

        # 2. Message Passing
        f.write("### 2. Message Passing Layers\n\n")

        if all(f'message_layer_{i}' in op_times for i in range(3)):
            msg_times = [op_times[f'message_layer_{i}']['mean_ms'] for i in range(3)]
            total_msg = sum(msg_times)
            f.write(f"**Current Performance**: {total_msg:.3f} ms total (3 layers)\n")
            for i, t in enumerate(msg_times):
                f.write(f"  - Layer {i}: {t:.3f} ms\n")
            f.write("\n")

        f.write("**Current Implementation**: PyTorch ops (matmul, index_add)\n")
        f.write("- Linear layers: Already using cuBLAS via PyTorch\n")
        f.write("- Scatter operations: Using PyTorch index_add\n")
        f.write("- Element-wise ops: Using PyTorch kernels\n\n")

        f.write("**CUDA-X Recommendation**: ⚠️ **Limited direct benefit**\n\n")
        f.write("**Analysis**:\n")
        f.write("- ✅ **cuBLAS**: Already used by PyTorch for matmul/linear layers\n")
        f.write("- ✅ **cuDNN**: Already used for activations (SiLU)\n")
        f.write("- ❌ **cuGraph**: Not applicable (GNN message passing ≠ graph analytics)\n")
        f.write("- ❓ **CUB**: Possible for scatter operations, but PyTorch is already optimized\n\n")

        f.write("**cuGraph Applicability**: ⚠️ **Not Suitable for GNN Message Passing**\n\n")
        f.write("cuGraph is designed for:\n")
        f.write("- Static graph algorithms (PageRank, BFS, community detection)\n")
        f.write("- Graph analytics on large, sparse graphs\n")
        f.write("- CPU preprocessing of graph structure\n\n")

        f.write("Our GNN requires:\n")
        f.write("- Dynamic graph construction per inference\n")
        f.write("- Feature propagation with learned transformations\n")
        f.write("- Differentiable operations for backprop\n")
        f.write("- Integration with PyTorch autograd\n\n")

        f.write("**Best Optimization Strategy**: 🎯 **Kernel Fusion**\n\n")
        f.write("**Approach**:\n")
        f.write("1. Fuse message computation + aggregation into single kernel\n")
        f.write("2. Reduce memory bandwidth by avoiding intermediate tensors\n")
        f.write("3. Use Triton for easier implementation\n\n")

        f.write("**Expected Speedup**: 1.5-2x\n\n")
        f.write("**Implementation Difficulty**: Medium-Hard\n\n")

        f.write("---\n\n")

        # 3. Update Layers
        f.write("### 3. Update Layers\n\n")

        if all(f'update_layer_{i}' in op_times for i in range(3)):
            upd_times = [op_times[f'update_layer_{i}']['mean_ms'] for i in range(3)]
            total_upd = sum(upd_times)
            f.write(f"**Current Performance**: {total_upd:.3f} ms total (3 layers)\n\n")

        f.write("**Current Implementation**: PyTorch MLPs + einsum\n")
        f.write("- Linear layers: cuBLAS via PyTorch\n")
        f.write("- Activations: cuDNN via PyTorch\n")
        f.write("- einsum: PyTorch optimized\n\n")

        f.write("**CUDA-X Recommendation**: ✅ **Already Optimized**\n\n")
        f.write("PyTorch already uses cuBLAS and cuDNN for these operations.\n\n")

        f.write("**Best Optimization Strategy**: 🎯 **torch.compile() + Kernel Fusion**\n\n")
        f.write("Expected speedup: 1.3-1.5x\n\n")

        f.write("---\n\n")

        # 4. Force Computation
        f.write("### 4. Force Computation (Autograd)\n\n")

        if 'force_computation' in op_times:
            force_time = op_times['force_computation']['mean_ms']
            f.write(f"**Current Performance**: {force_time:.3f} ms\n\n")

        f.write("**Current Implementation**: PyTorch autograd\n")
        f.write("- Backward pass through entire network\n")
        f.write("- Computes ∇E/∇positions\n\n")

        f.write("**CUDA-X Recommendation**: ❌ **No direct library support**\n\n")
        f.write("**Rationale**:\n")
        f.write("- Autograd is fundamental PyTorch operation\n")
        f.write("- No CUDA-X library provides automatic differentiation\n")
        f.write("- cuBLAS/cuDNN already used for individual ops in backward pass\n\n")

        f.write("**Best Optimization Strategy**: 🎯 **Optimize Forward Pass**\n\n")
        f.write("Force computation is already efficient. Speedup comes from:\n")
        f.write("1. Faster forward pass (less to differentiate)\n")
        f.write("2. torch.compile() to fuse backward ops\n")
        f.write("3. CUDA graphs to reduce launch overhead\n\n")

        f.write("Expected speedup: 1.2-1.5x\n\n")

        f.write("---\n\n")

        # Summary table
        f.write("## CUDA-X Library Applicability Summary\n\n")

        f.write("| CUDA-X Library | Applicable? | Use Case | Expected Speedup |\n")
        f.write("|----------------|-------------|----------|------------------|\n")
        f.write("| cuBLAS | ✅ Already used | Linear layers via PyTorch | N/A (baseline) |\n")
        f.write("| cuDNN | ✅ Already used | Activations via PyTorch | N/A (baseline) |\n")
        f.write("| cuSPARSE | ❌ Not applicable | No sparse matrix ops | N/A |\n")
        f.write("| cuGraph | ❌ Not suitable | GNN ≠ static graph analytics | N/A |\n")
        f.write("| CUB | ⚠️ Marginal | Atomic ops in neighbor search | <1.2x |\n")
        f.write("| Thrust | ⚠️ Marginal | Sorting/filtering in neighbor search | <1.2x |\n")
        f.write("| NCCL | ❌ Not applicable | Single GPU inference | N/A |\n\n")

        f.write("---\n\n")

        # Recommendations
        f.write("## Recommended Optimization Strategy\n\n")

        f.write("### Priority 1: Quick Wins (1-2 days)\n\n")
        f.write("1. **torch.compile()** (Python 3.12 required)\n")
        f.write("   - Expected: 1.3-1.5x speedup\n")
        f.write("   - Difficulty: Easy\n")
        f.write("   - Action: Test with `mode='reduce-overhead'`\n\n")

        f.write("2. **FP16 Mixed Precision** (with proper autocast)\n")
        f.write("   - Expected: 1.5-2x speedup\n")
        f.write("   - Difficulty: Easy\n")
        f.write("   - Action: Fix current implementation (autocast only)\n\n")

        f.write("3. **Use torch-cluster for neighbor search**\n")
        f.write("   - Expected: 2-3x speedup on neighbor search\n")
        f.write("   - Difficulty: Easy (drop-in replacement)\n")
        f.write("   - Action: `pip install torch-cluster` and use radius()\n\n")

        f.write("**Combined Expected**: 3-5x speedup\n\n")

        f.write("### Priority 2: Custom CUDA Kernels (1 week)\n\n")
        f.write("1. **Custom Neighbor Search**\n")
        f.write("   - Cell list algorithm with CUB primitives\n")
        f.write("   - Expected: 5-10x on neighbor search\n")
        f.write("   - Difficulty: Medium\n\n")

        f.write("2. **Fused Message Passing Kernel** (Triton)\n")
        f.write("   - Fuse RBF + message + aggregation\n")
        f.write("   - Expected: 1.5-2x on message passing\n")
        f.write("   - Difficulty: Medium-Hard\n\n")

        f.write("**Combined Expected with Priority 1**: 5-10x total speedup\n\n")

        f.write("### Priority 3: Advanced Optimizations (1-2 weeks)\n\n")
        f.write("1. **CUDA Graphs**\n")
        f.write("   - Reduce kernel launch overhead\n")
        f.write("   - Expected: 1.2-1.3x\n\n")

        f.write("2. **Kernel Tuning**\n")
        f.write("   - Optimize block sizes, shared memory\n")
        f.write("   - Expected: 1.1-1.2x\n\n")

        f.write("**Combined Expected with Priorities 1+2**: 6-13x total speedup\n\n")

        f.write("---\n\n")

        f.write("## Detailed Operation Timing Breakdown\n\n")

        if op_times:
            f.write("| Operation | Mean (ms) | Std (ms) | % of Forward |\n")
            f.write("|-----------|-----------|----------|-------------|\n")

            forward_time = op_times.get('full_forward', {}).get('mean_ms', 1.0)

            for key in sorted(op_times.keys()):
                if key == 'full_forward':
                    continue
                mean = op_times[key].get('mean_ms', 0)
                std = op_times[key].get('std_ms', 0)
                pct = (mean / forward_time) * 100
                f.write(f"| {key.replace('_', ' ').title()} | {mean:.4f} | {std:.4f} | {pct:.1f}% |\n")

            f.write(f"| **Full Forward** | **{forward_time:.4f}** | "
                   f"**{op_times['full_forward'].get('std_ms', 0):.4f}** | **100%** |\n\n")

        # Scalability
        if 'scalability' in profiling_data:
            f.write("---\n\n")
            f.write("## Scalability Analysis\n\n")
            f.write("| Atoms | Edges | Neighbor (ms) | Forward (ms) | Forces (ms) | Total (ms) |\n")
            f.write("|-------|-------|---------------|--------------|-------------|------------|\n")

            for data in profiling_data['scalability']['scalability']:
                f.write(f"| {data['n_atoms']} | {data['n_edges']} | "
                       f"{data['neighbor_search_ms']:.3f} | "
                       f"{data['forward_pass_ms']:.3f} | "
                       f"{data['force_computation_ms']:.3f} | "
                       f"{data['total_ms']:.3f} |\n")
            f.write("\n")

    logger.info(f"Recommendations saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Detailed profiling for CUDA-X analysis")
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
        default='benchmarks/cuda_x_analysis',
        help='Output directory'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick profiling (fewer iterations)'
    )

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("CUDA-X Library Analysis - Detailed Profiling")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Output: {output_dir}")
    logger.info("")

    # Load model
    logger.info("Loading model...")
    from mlff_distiller.models.student_model import StudentForceField
    model = StudentForceField.load(args.checkpoint, device=args.device)
    model.eval()

    logger.info(f"Model loaded: {model.num_parameters():,} parameters")
    logger.info("")

    # Create test molecules
    logger.info("Creating test molecules...")
    test_molecule = molecule('C6H6')  # Benzene - 12 atoms

    # Collect profiling data
    profiling_data = {}

    # 1. Operation breakdown
    logger.info("\n" + "=" * 80)
    logger.info("1. Profiling Individual Operations")
    logger.info("=" * 80)
    profiling_data['operation_breakdown'] = profile_operations(model, test_molecule, args.device)

    # 2. Memory patterns
    logger.info("\n" + "=" * 80)
    logger.info("2. Analyzing Memory Patterns")
    logger.info("=" * 80)
    profiling_data['memory_patterns'] = analyze_memory_patterns(model, test_molecule, args.device)

    # 3. PyTorch profiler
    logger.info("\n" + "=" * 80)
    logger.info("3. Running PyTorch Profiler")
    logger.info("=" * 80)
    profiling_data['pytorch_profiler'] = profile_with_pytorch_profiler(
        model, test_molecule, args.device, output_dir
    )

    # 4. Scalability
    if not args.quick:
        logger.info("\n" + "=" * 80)
        logger.info("4. Analyzing Scalability")
        logger.info("=" * 80)
        profiling_data['scalability'] = analyze_scalability(model, args.device)

    # Save raw profiling data
    json_file = output_dir / 'profiling_data.json'
    with open(json_file, 'w') as f:
        json.dump(profiling_data, f, indent=2)
    logger.info(f"\nRaw profiling data saved to {json_file}")

    # Generate recommendations
    logger.info("\n" + "=" * 80)
    logger.info("Generating CUDA-X Recommendations")
    logger.info("=" * 80)

    recommendations_file = output_dir / 'CUDA_X_RECOMMENDATIONS.md'
    generate_cuda_x_recommendations(profiling_data, recommendations_file)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("Summary")
    logger.info("=" * 80)
    logger.info(f"\nProfiling complete!")
    logger.info(f"Results saved to: {output_dir}/")
    logger.info(f"\nKey files:")
    logger.info(f"  - CUDA-X recommendations: {recommendations_file}")
    logger.info(f"  - Detailed profiling: {output_dir}/pytorch_profiler_detailed.txt")
    logger.info(f"  - Chrome trace: {output_dir}/trace.json")
    logger.info(f"  - Raw data: {json_file}")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Review {recommendations_file}")
    logger.info(f"  2. Implement Priority 1 optimizations (quick wins)")
    logger.info(f"  3. Consider custom CUDA kernels for neighbor search")

    return 0


if __name__ == '__main__':
    sys.exit(main())
