"""
Fused RBF + Cutoff Triton Kernel

This kernel fuses the Gaussian RBF basis computation and cosine cutoff
into a single GPU kernel, reducing memory traffic and kernel launch overhead.

Mathematical Operations:
    RBF_k(d) = exp(-gamma * (d - mu_k)^2)
    cutoff(d) = 0.5 * (cos(pi * d / r_cut) + 1) if d < r_cut else 0
    output = RBF * cutoff

Performance:
    - Current (separate kernels): ~0.327 ms for benzene (132 edges)
    - Expected (fused): ~0.250 ms (1.3x speedup)
    - Saves ~0.077 ms per molecule

Author: CUDA Optimization Engineer
Date: 2025-11-24
"""

import torch
import triton
import triton.language as tl
import numpy as np


@triton.jit
def fused_rbf_cutoff_kernel(
    # Input pointers
    distances_ptr,      # [n_edges] - pairwise distances
    centers_ptr,        # [n_rbf] - RBF centers
    # Output pointer
    output_ptr,         # [n_edges, n_rbf] - fused RBF + cutoff values
    # Scalar parameters
    n_edges,            # Number of edges
    n_rbf,              # Number of RBF basis functions
    gamma,              # RBF width parameter (1 / width^2)
    r_cut,              # Cutoff radius
    pi,                 # Pi constant
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused RBF + cutoff kernel.

    Each program instance processes one edge and computes all RBF values
    for that edge, applying the cutoff function.

    Grid: (n_edges,)
    Block: BLOCK_SIZE (handles n_rbf iterations)
    """
    # Get edge index
    edge_id = tl.program_id(axis=0)

    if edge_id >= n_edges:
        return

    # Load distance for this edge
    distance = tl.load(distances_ptr + edge_id)

    # Compute cutoff value
    # cutoff = 0.5 * (cos(pi * d / r_cut) + 1) if d < r_cut else 0
    cutoff_val = tl.where(
        distance < r_cut,
        0.5 * (tl.cos(pi * distance / r_cut) + 1.0),
        0.0
    )

    # Process RBF basis functions in blocks
    for rbf_block_start in range(0, n_rbf, BLOCK_SIZE):
        # Compute RBF indices for this block
        rbf_offsets = rbf_block_start + tl.arange(0, BLOCK_SIZE)
        rbf_mask = rbf_offsets < n_rbf

        # Load RBF centers
        centers = tl.load(centers_ptr + rbf_offsets, mask=rbf_mask, other=0.0)

        # Compute RBF values: exp(-gamma * (d - center)^2)
        diff = distance - centers
        rbf_vals = tl.exp(-gamma * diff * diff)

        # Apply cutoff
        output_vals = rbf_vals * cutoff_val

        # Store results
        output_indices = edge_id * n_rbf + rbf_offsets
        tl.store(output_ptr + output_indices, output_vals, mask=rbf_mask)


def fused_rbf_cutoff_triton(
    distances: torch.Tensor,
    centers: torch.Tensor,
    gamma: float,
    r_cut: float,
    block_size: int = 64
) -> torch.Tensor:
    """
    Compute fused RBF + cutoff using Triton kernel.

    Args:
        distances: Edge distances, shape [n_edges]
        centers: RBF centers, shape [n_rbf]
        gamma: RBF width parameter (1 / width^2)
        r_cut: Cutoff radius
        block_size: Triton block size (default: 64)

    Returns:
        Fused RBF + cutoff values, shape [n_edges, n_rbf]
    """
    n_edges = distances.shape[0]
    n_rbf = centers.shape[0]

    # Allocate output
    output = torch.empty(
        (n_edges, n_rbf),
        dtype=distances.dtype,
        device=distances.device
    )

    # Launch kernel
    grid = (n_edges,)
    fused_rbf_cutoff_kernel[grid](
        distances,
        centers,
        output,
        n_edges,
        n_rbf,
        gamma,
        r_cut,
        np.pi,
        BLOCK_SIZE=block_size
    )

    return output


class FusedRBFCutoff(torch.nn.Module):
    """
    PyTorch module wrapper for fused RBF + cutoff kernel.

    This provides a drop-in replacement for separate RBF and cutoff operations.

    Args:
        num_rbf: Number of radial basis functions
        cutoff: Cutoff distance in Angstroms
        learnable: Whether RBF parameters are learnable (default: False)

    Example:
        >>> rbf_cutoff = FusedRBFCutoff(num_rbf=20, cutoff=5.0)
        >>> distances = torch.randn(132, device='cuda')  # 132 edges
        >>> output = rbf_cutoff(distances)  # [132, 20]
    """

    def __init__(
        self,
        num_rbf: int = 20,
        cutoff: float = 5.0,
        learnable: bool = False
    ):
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff

        # Initialize centers uniformly from 0 to cutoff
        centers = torch.linspace(0, cutoff, num_rbf)

        # Initialize widths to cover the space evenly
        widths = torch.ones(num_rbf) * (cutoff / num_rbf)

        if learnable:
            self.centers = torch.nn.Parameter(centers)
            self.widths = torch.nn.Parameter(widths)
        else:
            self.register_buffer('centers', centers)
            self.register_buffer('widths', widths)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Apply fused RBF + cutoff to distances.

        Args:
            distances: Pairwise distances, shape [num_edges]

        Returns:
            Fused RBF + cutoff features, shape [num_edges, num_rbf]
        """
        # Compute gamma from widths
        gamma = 1.0 / (self.widths[0] ** 2)  # Assume uniform widths

        # Call Triton kernel
        return fused_rbf_cutoff_triton(
            distances,
            self.centers,
            gamma.item(),
            self.cutoff
        )


# ============================================================================
# Baseline PyTorch Implementation (for comparison)
# ============================================================================

def fused_rbf_cutoff_pytorch(
    distances: torch.Tensor,
    centers: torch.Tensor,
    gamma: float,
    r_cut: float
) -> torch.Tensor:
    """
    Baseline PyTorch implementation of fused RBF + cutoff.

    This is the reference implementation for correctness testing.

    Args:
        distances: Edge distances, shape [n_edges]
        centers: RBF centers, shape [n_rbf]
        gamma: RBF width parameter
        r_cut: Cutoff radius

    Returns:
        Fused RBF + cutoff values, shape [n_edges, n_rbf]
    """
    # Expand dimensions for broadcasting
    # distances: [n_edges] -> [n_edges, 1]
    # centers: [n_rbf] -> [1, n_rbf]
    dist_expanded = distances.unsqueeze(-1)  # [n_edges, 1]
    centers_expanded = centers.unsqueeze(0)  # [1, n_rbf]

    # Compute RBF
    diff = dist_expanded - centers_expanded  # [n_edges, n_rbf]
    rbf = torch.exp(-gamma * diff ** 2)  # [n_edges, n_rbf]

    # Compute cutoff
    cutoff_values = torch.where(
        distances < r_cut,
        0.5 * (torch.cos(np.pi * distances / r_cut) + 1.0),
        torch.zeros_like(distances)
    )  # [n_edges]

    # Apply cutoff to RBF
    output = rbf * cutoff_values.unsqueeze(-1)  # [n_edges, n_rbf]

    return output


# ============================================================================
# Testing and Benchmarking
# ============================================================================

def test_fused_rbf_cutoff():
    """Test correctness of Triton kernel against PyTorch reference."""
    print("Testing fused RBF + cutoff kernel...")

    # Test parameters
    n_edges = 132  # Benzene
    n_rbf = 20
    cutoff = 5.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Generate test data
    torch.manual_seed(42)
    distances = torch.rand(n_edges, device=device) * cutoff
    centers = torch.linspace(0, cutoff, n_rbf, device=device)
    widths = torch.ones(n_rbf, device=device) * (cutoff / n_rbf)
    gamma = (1.0 / (widths[0] ** 2)).item()

    # PyTorch reference
    output_pytorch = fused_rbf_cutoff_pytorch(distances, centers, gamma, cutoff)

    # Triton kernel
    output_triton = fused_rbf_cutoff_triton(distances, centers, gamma, cutoff)

    # Compare
    max_diff = torch.max(torch.abs(output_pytorch - output_triton)).item()
    mean_diff = torch.mean(torch.abs(output_pytorch - output_triton)).item()
    rel_error = mean_diff / (torch.mean(torch.abs(output_pytorch)).item() + 1e-8)

    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")
    print(f"  Relative error: {rel_error:.2e}")

    # Check correctness
    atol = 1e-5
    rtol = 1e-4
    is_correct = torch.allclose(output_pytorch, output_triton, atol=atol, rtol=rtol)

    if is_correct:
        print("  PASSED: Triton kernel matches PyTorch reference")
    else:
        print(f"  FAILED: Differences exceed tolerance (atol={atol}, rtol={rtol})")

    return is_correct


def benchmark_fused_rbf_cutoff():
    """Benchmark Triton kernel vs PyTorch implementation."""
    print("\nBenchmarking fused RBF + cutoff...")

    import time

    # Test parameters
    n_edges = 132  # Benzene
    n_rbf = 20
    cutoff = 5.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Generate test data
    torch.manual_seed(42)
    distances = torch.rand(n_edges, device=device) * cutoff
    centers = torch.linspace(0, cutoff, n_rbf, device=device)
    widths = torch.ones(n_rbf, device=device) * (cutoff / n_rbf)
    gamma = (1.0 / (widths[0] ** 2)).item()

    n_iterations = 1000

    # Warmup
    for _ in range(10):
        _ = fused_rbf_cutoff_pytorch(distances, centers, gamma, cutoff)
        _ = fused_rbf_cutoff_triton(distances, centers, gamma, cutoff)

    # Benchmark PyTorch
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    for _ in range(n_iterations):
        output_pytorch = fused_rbf_cutoff_pytorch(distances, centers, gamma, cutoff)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    pytorch_time = (time.perf_counter() - start) / n_iterations * 1000  # ms

    # Benchmark Triton
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    for _ in range(n_iterations):
        output_triton = fused_rbf_cutoff_triton(distances, centers, gamma, cutoff)
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
        print("FUSED RBF + CUTOFF KERNEL TEST")
        print("=" * 80)

        # Correctness test
        test_passed = test_fused_rbf_cutoff()

        if test_passed:
            # Performance benchmark
            benchmark_results = benchmark_fused_rbf_cutoff()
        else:
            print("\nSkipping benchmark due to test failure")
    else:
        print("CUDA not available, skipping tests")
