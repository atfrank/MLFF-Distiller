"""
Fused Edge Features Triton Kernel

This kernel fuses edge vector computation, distance calculation, and normalization
into a single GPU kernel.

Mathematical Operations:
    edge_vec = pos[src] - pos[dst]
    distance = ||edge_vec||_2
    normalized = edge_vec / (distance + eps)

Performance:
    - Current (separate ops): ~0.510 ms for benzene (132 edges)
    - Expected (fused): ~0.350 ms (1.45x speedup)

Author: CUDA Optimization Engineer
Date: 2025-11-24
"""

import torch
import triton
import triton.language as tl


@triton.jit
def fused_edge_features_kernel(
    # Input pointers
    positions_ptr,           # [n_atoms, 3] - atomic positions
    edge_index_src_ptr,      # [n_edges] - source atom indices
    edge_index_dst_ptr,      # [n_edges] - destination atom indices
    # Output pointers
    edge_vectors_ptr,        # [n_edges, 3] - edge vectors
    distances_ptr,           # [n_edges] - distances
    normalized_vectors_ptr,  # [n_edges, 3] - normalized edge vectors
    # Scalar parameters
    n_edges,                 # Number of edges
    eps,                     # Small constant for numerical stability
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused edge feature computation kernel.

    Each program instance processes one edge and computes:
    1. Edge vector: r_ij = r_j - r_i
    2. Distance: ||r_ij||
    3. Normalized vector: r_ij / ||r_ij||

    Grid: (n_edges,)
    """
    # Get edge index
    edge_id = tl.program_id(axis=0)

    if edge_id >= n_edges:
        return

    # Load edge indices
    src = tl.load(edge_index_src_ptr + edge_id)
    dst = tl.load(edge_index_dst_ptr + edge_id)

    # Load positions (3D coordinates)
    # Load source position
    src_x = tl.load(positions_ptr + src * 3 + 0)
    src_y = tl.load(positions_ptr + src * 3 + 1)
    src_z = tl.load(positions_ptr + src * 3 + 2)

    # Load destination position
    dst_x = tl.load(positions_ptr + dst * 3 + 0)
    dst_y = tl.load(positions_ptr + dst * 3 + 1)
    dst_z = tl.load(positions_ptr + dst * 3 + 2)

    # Compute edge vector: r_ij = r_j - r_i
    edge_x = src_x - dst_x
    edge_y = src_y - dst_y
    edge_z = src_z - dst_z

    # Compute distance: ||r_ij||
    distance = tl.sqrt(edge_x * edge_x + edge_y * edge_y + edge_z * edge_z + eps)

    # Compute normalized vector: r_ij / ||r_ij||
    norm_x = edge_x / distance
    norm_y = edge_y / distance
    norm_z = edge_z / distance

    # Store edge vectors
    tl.store(edge_vectors_ptr + edge_id * 3 + 0, edge_x)
    tl.store(edge_vectors_ptr + edge_id * 3 + 1, edge_y)
    tl.store(edge_vectors_ptr + edge_id * 3 + 2, edge_z)

    # Store distances
    tl.store(distances_ptr + edge_id, distance)

    # Store normalized vectors
    tl.store(normalized_vectors_ptr + edge_id * 3 + 0, norm_x)
    tl.store(normalized_vectors_ptr + edge_id * 3 + 1, norm_y)
    tl.store(normalized_vectors_ptr + edge_id * 3 + 2, norm_z)


def fused_edge_features_triton(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    eps: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute fused edge features using Triton kernel.

    Args:
        positions: Atomic positions, shape [n_atoms, 3]
        edge_index: Edge indices, shape [2, n_edges]
        eps: Small constant for numerical stability

    Returns:
        edge_vectors: Edge vectors, shape [n_edges, 3]
        distances: Edge distances, shape [n_edges]
        normalized_vectors: Normalized edge vectors, shape [n_edges, 3]
    """
    n_edges = edge_index.shape[1]
    n_atoms = positions.shape[0]

    # Flatten positions for easier indexing in kernel
    positions_flat = positions.contiguous().view(-1)

    # Extract source and destination indices
    edge_src = edge_index[0, :].contiguous()
    edge_dst = edge_index[1, :].contiguous()

    # Allocate outputs
    edge_vectors = torch.empty(
        (n_edges, 3),
        dtype=positions.dtype,
        device=positions.device
    )
    distances = torch.empty(
        n_edges,
        dtype=positions.dtype,
        device=positions.device
    )
    normalized_vectors = torch.empty(
        (n_edges, 3),
        dtype=positions.dtype,
        device=positions.device
    )

    # Flatten outputs for kernel
    edge_vectors_flat = edge_vectors.view(-1)
    normalized_vectors_flat = normalized_vectors.view(-1)

    # Launch kernel
    grid = (n_edges,)
    fused_edge_features_kernel[grid](
        positions_flat,
        edge_src,
        edge_dst,
        edge_vectors_flat,
        distances,
        normalized_vectors_flat,
        n_edges,
        eps,
        BLOCK_SIZE=256
    )

    return edge_vectors, distances, normalized_vectors


# ============================================================================
# Baseline PyTorch Implementation
# ============================================================================

def fused_edge_features_pytorch(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    eps: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Baseline PyTorch implementation of fused edge features.

    Args:
        positions: Atomic positions, shape [n_atoms, 3]
        edge_index: Edge indices, shape [2, n_edges]
        eps: Small constant for numerical stability

    Returns:
        edge_vectors: Edge vectors, shape [n_edges, 3]
        distances: Edge distances, shape [n_edges]
        normalized_vectors: Normalized edge vectors, shape [n_edges, 3]
    """
    src, dst = edge_index

    # Edge vectors
    edge_vectors = positions[src] - positions[dst]  # [n_edges, 3]

    # Distances
    distances = torch.norm(edge_vectors, dim=1)  # [n_edges]

    # Normalized vectors
    normalized_vectors = edge_vectors / (distances.unsqueeze(1) + eps)  # [n_edges, 3]

    return edge_vectors, distances, normalized_vectors


# ============================================================================
# Testing and Benchmarking
# ============================================================================

def test_fused_edge_features():
    """Test correctness of Triton kernel against PyTorch reference."""
    print("Testing fused edge features kernel...")

    # Test parameters
    n_atoms = 12
    n_edges = 132
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Generate test data
    torch.manual_seed(42)
    positions = torch.randn(n_atoms, 3, device=device)

    # Create random edge index
    src = torch.randint(0, n_atoms, (n_edges,), device=device)
    dst = torch.randint(0, n_atoms, (n_edges,), device=device)
    edge_index = torch.stack([src, dst], dim=0)

    # PyTorch reference
    edge_vec_pt, dist_pt, norm_pt = fused_edge_features_pytorch(positions, edge_index)

    # Triton kernel
    edge_vec_tr, dist_tr, norm_tr = fused_edge_features_triton(positions, edge_index)

    # Compare edge vectors
    edge_vec_diff = torch.max(torch.abs(edge_vec_pt - edge_vec_tr)).item()
    print(f"  Edge vectors max diff: {edge_vec_diff:.2e}")

    # Compare distances
    dist_diff = torch.max(torch.abs(dist_pt - dist_tr)).item()
    print(f"  Distances max diff: {dist_diff:.2e}")

    # Compare normalized vectors
    norm_diff = torch.max(torch.abs(norm_pt - norm_tr)).item()
    print(f"  Normalized vectors max diff: {norm_diff:.2e}")

    # Check correctness
    atol = 1e-4  # Relaxed tolerance for FP32
    rtol = 1e-3

    is_correct = (
        torch.allclose(edge_vec_pt, edge_vec_tr, atol=atol, rtol=rtol) and
        torch.allclose(dist_pt, dist_tr, atol=atol, rtol=rtol) and
        torch.allclose(norm_pt, norm_tr, atol=atol, rtol=rtol)
    )

    if is_correct:
        print("  PASSED: Triton kernel matches PyTorch reference")
    else:
        print(f"  FAILED: Differences exceed tolerance (atol={atol}, rtol={rtol})")

    return is_correct


def benchmark_fused_edge_features():
    """Benchmark Triton kernel vs PyTorch implementation."""
    print("\nBenchmarking fused edge features...")

    import time

    # Test parameters
    n_atoms = 12
    n_edges = 132
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Generate test data
    torch.manual_seed(42)
    positions = torch.randn(n_atoms, 3, device=device)
    src = torch.randint(0, n_atoms, (n_edges,), device=device)
    dst = torch.randint(0, n_atoms, (n_edges,), device=device)
    edge_index = torch.stack([src, dst], dim=0)

    n_iterations = 1000

    # Warmup
    for _ in range(10):
        _ = fused_edge_features_pytorch(positions, edge_index)
        _ = fused_edge_features_triton(positions, edge_index)

    # Benchmark PyTorch
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    for _ in range(n_iterations):
        edge_vec_pt, dist_pt, norm_pt = fused_edge_features_pytorch(positions, edge_index)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    pytorch_time = (time.perf_counter() - start) / n_iterations * 1000  # ms

    # Benchmark Triton
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    for _ in range(n_iterations):
        edge_vec_tr, dist_tr, norm_tr = fused_edge_features_triton(positions, edge_index)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    triton_time = (time.perf_counter() - start) / n_iterations * 1000  # ms

    speedup = pytorch_time / triton_time

    print(f"\n  Results ({n_iterations} iterations):")
    print(f"  PyTorch:  {pytorch_time:.3f} ms")
    print(f"  Triton:   {triton_time:.3f} ms")
    print(f"  Speedup:  {speedup:.2f}x")

    return {'pytorch_ms': pytorch_time, 'triton_ms': triton_time, 'speedup': speedup}


if __name__ == '__main__':
    # Run tests
    if torch.cuda.is_available():
        print("=" * 80)
        print("FUSED EDGE FEATURES KERNEL TEST")
        print("=" * 80)

        # Correctness test
        test_passed = test_fused_edge_features()

        if test_passed:
            # Performance benchmark
            benchmark_results = benchmark_fused_edge_features()
        else:
            print("\nSkipping benchmark due to test failure")
    else:
        print("CUDA not available, skipping tests")
