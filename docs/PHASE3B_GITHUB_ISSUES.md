# Phase 3B: GitHub Issues - Advanced Optimizations

**Created**: 2025-11-24
**Coordinator**: ml-distillation-coordinator
**Milestone**: M4 - CUDA Optimization & Advanced Performance
**Target**: 10-50x total speedup

---

## Issue Creation Plan

This document outlines the GitHub Issues to be created for Phase 3B: Advanced Optimizations. Issues are organized by week and assigned to specialized agents.

---

## Week 1: Analytical Gradient Issues

### Issue #25: [CUDA] [M4] Design and derive analytical gradient formulas for PaiNN

**Assigned to**: cuda-optimization-engineer (primary), ml-architecture-designer (reviewer)
**Priority**: P0 (Critical Path)
**Labels**: `M4-optimization`, `cuda`, `analytical-gradients`, `design`
**Estimated Effort**: 2 days (16 hours)

**Description**:

Design and mathematically derive analytical gradient formulas for computing forces directly in PaiNN without autograd backward pass.

**Problem Statement**:

Current force computation uses autograd.grad() which incurs 1.6-2x overhead from:
- Graph construction (20-30% overhead)
- Backward pass execution (40-50% of forward time)
- Intermediate activation storage

We need analytical gradients to compute forces directly during forward pass.

**Requirements**:

1. Derive ∂E/∂r formulas for each PaiNN layer:
   - Embedding layer
   - Message passing layers (3 interactions)
   - Scalar update layers
   - Vector update layers
   - Readout layer

2. Identify intermediate values needed:
   - Edge features (RBF expansions)
   - Messages (scalar and vector)
   - Node features (at each layer)

3. Design storage strategy:
   - Memory vs recomputation tradeoff
   - Caching structure for forward pass
   - Gradient backpropagation plan

4. Prove mathematical correctness:
   - Chain rule application
   - Verify against finite differences
   - Document assumptions

**Deliverables**:

- [ ] `docs/ANALYTICAL_GRADIENT_DERIVATION.md` with full mathematical derivation
- [ ] Memory footprint analysis (baseline vs analytical)
- [ ] Pseudocode for each gradient component
- [ ] Mathematical correctness proof

**Acceptance Criteria**:

- [ ] Complete mathematical derivation for all PaiNN layers
- [ ] Memory footprint estimated and acceptable (<2x forward pass)
- [ ] Design reviewed and approved by ml-architecture-designer
- [ ] Pseudocode clear and implementable

**Related Issues**: #26 (implementation depends on this)

**References**:
- PaiNN paper: https://arxiv.org/abs/2102.03150
- Current implementation: `src/mlff_distiller/models/student_model.py`

**Timeline**:
- Day 1: Derive formulas for embedding + first interaction layer
- Day 2: Complete remaining layers + documentation

---

### Issue #26: [CUDA] [M4] Implement analytical force computation in StudentForceField

**Assigned to**: cuda-optimization-engineer
**Priority**: P0 (Critical Path)
**Labels**: `M4-optimization`, `cuda`, `analytical-gradients`, `implementation`
**Estimated Effort**: 2 days (16 hours)
**Depends on**: #25

**Description**:

Implement analytical force computation in `StudentForceField` class to compute forces directly without autograd.

**Problem Statement**:

Based on the mathematical derivation from Issue #25, implement the analytical gradient computation in the model's forward pass.

**Requirements**:

1. Modify `StudentForceField.forward()`:
   - Add `compute_forces=False` parameter
   - Cache intermediate activations during forward pass
   - Implement `_compute_forces_analytical()` method

2. Implement gradient computation:
   - Geometric derivatives: ∂r_ij/∂r_i
   - Energy gradients: ∂E/∂features
   - Force assembly: scatter operations

3. Memory-efficient caching:
   - Store only essential intermediate values
   - Optional recomputation for memory-constrained cases
   - Clear cache after force computation

4. Maintain backward compatibility:
   - Keep autograd option for validation
   - Switch via parameter or environment variable
   - Ensure existing tests still pass

**Implementation Plan**:

```python
class StudentForceField(nn.Module):
    def forward(self, batch, compute_forces=False, use_analytical=True):
        # Forward pass with caching
        energy = self._compute_energy_with_cache(batch)

        if not compute_forces:
            self._clear_cache()
            return energy

        if use_analytical:
            forces = self._compute_forces_analytical(batch)
        else:
            # Fallback to autograd
            forces = -torch.autograd.grad(energy.sum(), batch.positions)[0]

        self._clear_cache()
        return energy, forces

    def _compute_energy_with_cache(self, batch):
        # Store intermediate values in self._cache
        ...

    def _compute_forces_analytical(self, batch):
        # Use cached values to compute forces
        geometric_grads = self._compute_geometric_derivatives(batch)
        energy_grads = self._backprop_readout()
        forces = self._assemble_forces(geometric_grads, energy_grads, batch)
        return forces
```

**Deliverables**:

- [ ] Modified `src/mlff_distiller/models/student_model.py` (+300 lines)
- [ ] Implementation of `_compute_forces_analytical()`
- [ ] Caching mechanism for intermediate values
- [ ] Backward compatibility maintained

**Acceptance Criteria**:

- [ ] Code compiles and runs without errors
- [ ] Energy computation unchanged (bit-exact)
- [ ] Forces computed (accuracy validated in #27)
- [ ] Memory overhead <2x forward pass
- [ ] All existing tests pass

**Testing Strategy**:

Unit tests (in #27):
- Test each gradient component separately
- Verify cache management works correctly
- Test backward compatibility

**Timeline**:
- Day 3: Implement caching and geometric derivatives
- Day 4: Implement energy gradient backprop and force assembly

---

### Issue #27: [Testing] [M4] Validate analytical gradients vs autograd

**Assigned to**: testing-benchmark-engineer (primary), cuda-optimization-engineer (support)
**Priority**: P0 (Critical Path)
**Labels**: `M4-optimization`, `testing`, `analytical-gradients`, `validation`
**Estimated Effort**: 1 day (8 hours)
**Depends on**: #26

**Description**:

Comprehensive validation of analytical force computation against autograd reference.

**Problem Statement**:

Analytical gradients must produce numerically identical results to autograd to ensure correctness for MD simulations.

**Requirements**:

1. Numerical accuracy tests:
   - Compare analytical vs autograd forces
   - Tolerance: <1e-4 eV/Å (production requirement)
   - Stretch: <1e-6 eV/Å (ideal)

2. Test coverage:
   - Various molecule sizes (3-100 atoms)
   - Different element compositions
   - Edge cases (single atom, no neighbors)
   - Random structure generation

3. Gradient checking:
   - Finite difference validation
   - Per-atom force comparison
   - Per-component (x, y, z) analysis

4. MD stability validation:
   - Short NVE simulation (100 steps)
   - Energy conservation check (<0.1%/ns target)
   - No NaN/Inf values

**Implementation Plan**:

Create `tests/unit/test_analytical_forces.py`:

```python
import pytest
import torch
from mlff_distiller.models.student_model import StudentForceField

@pytest.fixture
def model():
    return StudentForceField.load_checkpoint('checkpoints/best_model.pt')

def test_analytical_vs_autograd_small_molecule(model):
    # Test on water molecule
    batch = create_test_batch(n_atoms=3, species=['H', 'H', 'O'])

    # Analytical forces
    energy_ana, forces_ana = model(batch, compute_forces=True, use_analytical=True)

    # Autograd forces
    energy_auto, forces_auto = model(batch, compute_forces=True, use_analytical=False)

    # Compare
    assert torch.allclose(energy_ana, energy_auto, atol=1e-6)
    assert torch.allclose(forces_ana, forces_auto, atol=1e-4)

def test_gradient_check_finite_difference(model):
    # Validate against finite differences
    batch = create_test_batch(n_atoms=10)
    epsilon = 1e-5

    forces_ana = model(batch, compute_forces=True, use_analytical=True)[1]

    # Compute finite difference forces
    forces_fd = compute_finite_difference_forces(model, batch, epsilon)

    assert torch.allclose(forces_ana, forces_fd, rtol=1e-3, atol=1e-3)

def test_md_stability(model):
    # Run short MD simulation
    trajectory = run_nve_simulation(model, n_steps=100, use_analytical=True)

    # Check energy conservation
    energy_drift = (trajectory.energies[-1] - trajectory.energies[0]) / trajectory.energies[0]
    assert abs(energy_drift) < 0.001  # <0.1% drift

    # Check no NaN/Inf
    assert not torch.isnan(trajectory.forces).any()
    assert not torch.isinf(trajectory.forces).any()
```

**Deliverables**:

- [ ] `tests/unit/test_analytical_forces.py` (+150 lines)
- [ ] Numerical accuracy validation report
- [ ] MD stability validation report
- [ ] Per-atom force error analysis

**Acceptance Criteria**:

- [ ] All tests pass with tolerance <1e-4 eV/Å
- [ ] Gradient check passes (finite difference comparison)
- [ ] MD simulation stable (energy drift <0.1%)
- [ ] No numerical issues (NaN/Inf)

**Timeline**:
- Day 5 morning: Write and run tests
- Day 5 afternoon: Generate validation reports

---

### Issue #28: [CUDA] [M4] Benchmark analytical gradients and measure speedup

**Assigned to**: cuda-optimization-engineer (primary), testing-benchmark-engineer (support)
**Priority**: P0 (Critical Path)
**Labels**: `M4-optimization`, `cuda`, `analytical-gradients`, `benchmarking`
**Estimated Effort**: 0.5 days (4 hours)
**Depends on**: #27

**Description**:

Benchmark analytical gradient implementation and measure speedup vs autograd.

**Problem Statement**:

Validate that analytical gradients provide expected 1.8-2.0x speedup over current batched autograd approach.

**Requirements**:

1. Comprehensive benchmarking:
   - System sizes: 10, 20, 50, 100 atoms
   - Batch sizes: 1, 2, 4
   - Compare: baseline, batched autograd, analytical

2. Metrics to measure:
   - Latency (ms/molecule)
   - Throughput (molecules/second)
   - GPU memory usage
   - Speedup vs baseline

3. Statistical rigor:
   - Warmup runs: 10
   - Benchmark runs: 100
   - Report: mean, std, median, p95, p99

4. Visualizations:
   - Speedup vs system size
   - Latency breakdown (energy vs forces)
   - Cumulative speedup progress

**Implementation Plan**:

Create `scripts/benchmark_analytical_forces.py`:

```python
import torch
import time
from mlff_distiller.models.student_model import StudentForceField

def benchmark_configuration(model, batch, config_name, n_iterations=100):
    timings = []

    # Warmup
    for _ in range(10):
        _ = model(batch, **config)

    # Benchmark
    torch.cuda.synchronize()
    for _ in range(n_iterations):
        start = time.perf_counter()
        energy, forces = model(batch, **config)
        torch.cuda.synchronize()
        end = time.perf_counter()
        timings.append((end - start) * 1000)  # Convert to ms

    return {
        'config': config_name,
        'mean_ms': np.mean(timings),
        'std_ms': np.std(timings),
        'median_ms': np.median(timings),
        'p95_ms': np.percentile(timings, 95),
    }

def main():
    model = StudentForceField.load_checkpoint('checkpoints/best_model.pt')

    results = {}

    for n_atoms in [10, 20, 50, 100]:
        batch = create_test_batch(n_atoms=n_atoms)

        # Baseline (sequential)
        results[f'baseline_{n_atoms}'] = benchmark_configuration(
            model, batch, 'baseline', n_iterations=100
        )

        # Batched autograd
        results[f'batched_autograd_{n_atoms}'] = benchmark_configuration(
            model, batch, 'batched_autograd', n_iterations=100
        )

        # Analytical gradients
        results[f'analytical_{n_atoms}'] = benchmark_configuration(
            model, batch, 'analytical', n_iterations=100
        )

    # Compute speedups
    for n_atoms in [10, 20, 50, 100]:
        baseline = results[f'baseline_{n_atoms}']['mean_ms']
        analytical = results[f'analytical_{n_atoms}']['mean_ms']
        results[f'speedup_{n_atoms}'] = baseline / analytical

    # Save results
    save_json(results, 'benchmarks/analytical_forces_results.json')

    # Generate report
    generate_report(results)

if __name__ == '__main__':
    main()
```

**Deliverables**:

- [ ] `scripts/benchmark_analytical_forces.py` (+200 lines)
- [ ] `benchmarks/analytical_forces_results.json` (performance data)
- [ ] `docs/ANALYTICAL_FORCES_IMPLEMENTATION.md` (summary report)
- [ ] Speedup visualization plots

**Acceptance Criteria**:

- [ ] Analytical gradients achieve 1.8-2.0x speedup (minimum 1.5x)
- [ ] Total speedup vs original baseline: 9-10x (minimum 7.5x)
- [ ] Performance consistent across system sizes
- [ ] Report generated with clear results

**Success Metrics**:

- Minimum Success: 1.5x speedup, 7.5x total
- Target Success: 1.8x speedup, 9x total
- Stretch Success: 2.0x speedup, 10x total

**Timeline**:
- Day 5 afternoon: Run benchmarks and generate report

---

## Week 2-3: Custom CUDA Kernel Issues

### Issue #29: [CUDA] [M4] Profile student model and identify kernel bottlenecks

**Assigned to**: cuda-optimization-engineer
**Priority**: P1 (High)
**Labels**: `M4-optimization`, `cuda`, `profiling`, `kernels`
**Estimated Effort**: 2 days (16 hours)
**Depends on**: #28

**Description**:

Profile the analytical gradient implementation using NVIDIA Nsight Systems and Nsight Compute to identify bottleneck kernels for optimization.

**Problem Statement**:

Need detailed profiling data to prioritize which operations to optimize with custom CUDA kernels.

**Requirements**:

1. Timeline profiling (Nsight Systems):
   - Identify which kernels take most time
   - Measure kernel launch overhead
   - Analyze GPU utilization
   - Detect idle time and synchronization

2. Kernel-level profiling (Nsight Compute):
   - Memory bandwidth utilization
   - Compute throughput
   - Occupancy metrics
   - Warp efficiency

3. Target operations:
   - Neighbor search (radius_graph)
   - RBF computation
   - Message passing scatter/gather
   - Geometric derivative computation
   - Force assembly

4. Prioritization matrix:
   - Time spent in each operation
   - Optimization potential (bandwidth vs compute bound)
   - Implementation complexity
   - Expected speedup

**Implementation Plan**:

```bash
# Timeline profiling
nsys profile -o profile_analytical \
    --trace=cuda,nvtx,osrt \
    --gpu-metrics-device=all \
    python scripts/benchmark_analytical_forces.py --quick

# Kernel profiling (top 10 kernels)
ncu --set full -o profile_kernels \
    --target-processes all \
    python scripts/benchmark_analytical_forces.py --quick

# Analyze results
nsys stats profile_analytical.nsys-rep --report cuda_gpu_kernel_sum
ncu -i profile_kernels.ncu-rep --page details
```

Create `scripts/analyze_profiling_results.py`:

```python
def analyze_nsight_results(nsys_report, ncu_report):
    # Parse Nsight Systems report
    kernel_times = parse_kernel_times(nsys_report)

    # Sort by time
    bottlenecks = sorted(kernel_times.items(), key=lambda x: x[1], reverse=True)

    # Analyze top 10 kernels
    for kernel_name, time_ms in bottlenecks[:10]:
        # Get detailed metrics from Nsight Compute
        metrics = parse_ncu_metrics(ncu_report, kernel_name)

        # Classify bottleneck type
        bottleneck_type = classify_bottleneck(metrics)

        # Estimate optimization potential
        potential = estimate_speedup_potential(metrics, bottleneck_type)

        print(f"{kernel_name}: {time_ms:.2f}ms ({potential:.1f}x potential)")

    # Generate priority matrix
    generate_priority_matrix(bottlenecks)
```

**Deliverables**:

- [ ] `docs/CUDA_PROFILING_REPORT.md` with full analysis
- [ ] Kernel priority matrix (which to optimize first)
- [ ] Profiling data files (`.nsys-rep`, `.ncu-rep`)
- [ ] `scripts/analyze_profiling_results.py` (+150 lines)

**Acceptance Criteria**:

- [ ] Top 10 bottleneck kernels identified
- [ ] Each kernel classified (memory-bound vs compute-bound)
- [ ] Optimization priority matrix created
- [ ] Expected speedup estimates documented

**Expected Findings**:

Based on previous profiling, expect:
1. Neighbor search (radius_graph): 15-20% of time, memory-bound
2. Message passing scatter: 15-20% of time, memory-bound
3. RBF computation: 8-12% of time, compute-bound
4. Geometric derivatives: 10-15% of time, mixed
5. Force scatter: 5-10% of time, memory-bound

**Timeline**:
- Week 2, Day 1: Run profiling with Nsight Systems
- Week 2, Day 2: Run profiling with Nsight Compute, analyze results

---

### Issue #30: [CUDA] [M4] Implement optimized neighbor search with cell lists (Triton)

**Assigned to**: cuda-optimization-engineer
**Priority**: P1 (High)
**Labels**: `M4-optimization`, `cuda`, `triton`, `kernels`, `neighbor-search`
**Estimated Effort**: 3 days (24 hours)
**Depends on**: #29

**Description**:

Implement optimized neighbor search using cell list algorithm in Triton to replace the current O(N²) implementation.

**Problem Statement**:

Current `radius_graph_native()` computes full N×N distance matrix, which is inefficient for large systems. Cell list algorithm reduces complexity to O(N).

**Requirements**:

1. Cell list data structure:
   - Divide space into cells of size `cutoff`
   - Assign atoms to cells
   - Only search 27 neighboring cells (3x3x3)

2. Triton kernel implementation:
   - Input: positions, cell, cutoff
   - Output: edge_index, distances
   - Handle periodic boundary conditions (PBC)
   - Thread-safe (avoid race conditions)

3. Performance targets:
   - 2-3x faster than current implementation
   - Scales better with system size
   - Memory efficient

4. Correctness validation:
   - Exact match with PyTorch reference
   - Test with and without PBC
   - Edge cases (empty cells, boundary atoms)

**Implementation Plan**:

Create `src/mlff_distiller/cuda/neighbor_search.py`:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def neighbor_search_kernel(
    positions_ptr,      # [N, 3]
    cell_ptr,           # [3, 3] (cell matrix for PBC)
    cell_assignments_ptr,  # [N] (which cell each atom is in)
    cell_start_ptr,     # [n_cells] (start index for each cell)
    cell_end_ptr,       # [n_cells] (end index for each cell)
    neighbors_out_ptr,  # [N, max_neighbors] (output neighbor indices)
    distances_out_ptr,  # [N, max_neighbors] (output distances)
    num_neighbors_ptr,  # [N] (output number of neighbors per atom)
    cutoff: tl.constexpr,
    max_neighbors: tl.constexpr,
    use_pbc: tl.constexpr,
    N: tl.constexpr,
    n_cells_x: tl.constexpr,
    n_cells_y: tl.constexpr,
    n_cells_z: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get atom index
    atom_i = tl.program_id(0)

    if atom_i >= N:
        return

    # Load position of atom i
    pos_i = tl.load(positions_ptr + atom_i * 3 + tl.arange(0, 3))

    # Get cell index
    cell_i = tl.load(cell_assignments_ptr + atom_i)
    cx = cell_i % n_cells_x
    cy = (cell_i // n_cells_x) % n_cells_y
    cz = cell_i // (n_cells_x * n_cells_y)

    # Initialize neighbor count
    n_neighbors = 0

    # Search 27 neighboring cells (3x3x3)
    for dcx in range(-1, 2):
        for dcy in range(-1, 2):
            for dcz in range(-1, 2):
                # Neighbor cell index (with PBC wrapping)
                ncx = (cx + dcx) % n_cells_x if use_pbc else cx + dcx
                ncy = (cy + dcy) % n_cells_y if use_pbc else cy + dcy
                ncz = (cz + dcz) % n_cells_z if use_pbc else cz + dcz

                # Skip if out of bounds (no PBC)
                if not use_pbc:
                    if ncx < 0 or ncx >= n_cells_x:
                        continue
                    if ncy < 0 or ncy >= n_cells_y:
                        continue
                    if ncz < 0 or ncz >= n_cells_z:
                        continue

                # Cell index
                neighbor_cell = ncx + ncy * n_cells_x + ncz * n_cells_x * n_cells_y

                # Get atoms in this cell
                cell_start = tl.load(cell_start_ptr + neighbor_cell)
                cell_end = tl.load(cell_end_ptr + neighbor_cell)

                # Iterate atoms in cell
                for j in range(cell_start, cell_end):
                    if j == atom_i:
                        continue  # Skip self

                    # Load position of atom j
                    pos_j = tl.load(positions_ptr + j * 3 + tl.arange(0, 3))

                    # Compute displacement (with PBC)
                    disp = pos_j - pos_i
                    if use_pbc:
                        # Apply minimum image convention
                        # (simplified, actual needs cell matrix)
                        disp = apply_pbc(disp, cell_ptr)

                    # Compute distance
                    dist = tl.sqrt(tl.sum(disp * disp))

                    # Add to neighbor list if within cutoff
                    if dist < cutoff and n_neighbors < max_neighbors:
                        tl.store(neighbors_out_ptr + atom_i * max_neighbors + n_neighbors, j)
                        tl.store(distances_out_ptr + atom_i * max_neighbors + n_neighbors, dist)
                        n_neighbors += 1

    # Store final neighbor count
    tl.store(num_neighbors_ptr + atom_i, n_neighbors)


class OptimizedNeighborSearch(torch.nn.Module):
    def __init__(self, cutoff=5.0, max_neighbors=32):
        super().__init__()
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors

    def forward(self, positions, cell=None):
        N = positions.shape[0]
        device = positions.device

        # Build cell list
        cell_assignments, cell_start, cell_end, n_cells = self._build_cell_list(
            positions, cell, self.cutoff
        )

        # Allocate output
        neighbors = torch.zeros((N, self.max_neighbors), dtype=torch.long, device=device)
        distances = torch.zeros((N, self.max_neighbors), dtype=torch.float32, device=device)
        num_neighbors = torch.zeros(N, dtype=torch.long, device=device)

        # Launch kernel
        grid = (N,)
        neighbor_search_kernel[grid](
            positions, cell if cell is not None else torch.eye(3, device=device),
            cell_assignments, cell_start, cell_end,
            neighbors, distances, num_neighbors,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
            use_pbc=(cell is not None),
            N=N,
            n_cells_x=n_cells[0],
            n_cells_y=n_cells[1],
            n_cells_z=n_cells[2],
            BLOCK_SIZE=128
        )

        # Convert to edge_index format
        edge_index, edge_distances = self._to_edge_index(neighbors, distances, num_neighbors)

        return edge_index, edge_distances
```

**Deliverables**:

- [ ] `src/mlff_distiller/cuda/neighbor_search.py` (+250 lines)
- [ ] Cell list construction utilities
- [ ] PBC handling (minimum image convention)
- [ ] `tests/unit/test_neighbor_search_cuda.py` (+100 lines)

**Acceptance Criteria**:

- [ ] Kernel produces exact same neighbors as PyTorch reference
- [ ] 2-3x faster than current implementation
- [ ] Works with and without PBC
- [ ] No race conditions or numerical errors
- [ ] All tests passing

**Timeline**:
- Week 2, Day 3: Implement cell list data structure
- Week 2, Day 4: Implement Triton kernel
- Week 2, Day 5: Test, validate, and benchmark

---

### Issue #31: [CUDA] [M4] Implement fused message passing kernel (Triton)

**Assigned to**: cuda-optimization-engineer
**Priority**: P1 (High)
**Labels**: `M4-optimization`, `cuda`, `triton`, `kernels`, `message-passing`
**Estimated Effort**: 3 days (24 hours)
**Depends on**: #30

**Description**:

Implement fused message passing kernel that combines RBF computation, message generation, and aggregation into a single Triton kernel.

**Problem Statement**:

Current implementation has separate kernels for:
1. RBF computation
2. Message generation (MLP)
3. Scatter aggregation

Fusing these reduces memory traffic and kernel launch overhead.

**Requirements**:

1. Fused operations:
   - Inline RBF computation (no intermediate storage)
   - Message generation (simplified MLP)
   - Atomic aggregation to destination nodes

2. Memory optimization:
   - Minimize global memory access
   - Maximize register usage
   - Coalesced memory access patterns

3. Performance targets:
   - 1.5-2x faster than separate kernels
   - Scales with number of edges

4. Correctness:
   - Bit-exact match with PyTorch (or <1e-6 error)
   - Handle variable message sizes
   - Thread-safe aggregation

**Implementation Plan**:

Create `src/mlff_distiller/cuda/message_passing.py`:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def fused_message_passing_kernel(
    positions_ptr,      # [N, 3]
    node_features_ptr,  # [N, hidden_dim]
    edge_index_ptr,     # [2, n_edges]
    edge_attr_out_ptr,  # [n_edges, rbf_dim] (output, optional)
    messages_out_ptr,   # [N, hidden_dim] (output, aggregated)
    # RBF parameters
    rbf_centers_ptr,    # [rbf_dim]
    rbf_widths_ptr,     # [rbf_dim]
    cutoff: tl.constexpr,
    # MLP parameters (simplified: one layer)
    mlp_weight_ptr,     # [hidden_dim, rbf_dim]
    mlp_bias_ptr,       # [hidden_dim]
    # Dimensions
    hidden_dim: tl.constexpr,
    rbf_dim: tl.constexpr,
    n_edges: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Each thread processes one edge
    edge_id = tl.program_id(0)

    if edge_id >= n_edges:
        return

    # Load edge indices
    src = tl.load(edge_index_ptr + edge_id * 2)
    dst = tl.load(edge_index_ptr + edge_id * 2 + 1)

    # Load positions
    pos_src = tl.load(positions_ptr + src * 3 + tl.arange(0, 3))
    pos_dst = tl.load(positions_ptr + dst * 3 + tl.arange(0, 3))

    # Compute distance
    disp = pos_dst - pos_src
    dist = tl.sqrt(tl.sum(disp * disp))

    # Compute RBF (inline, no storage)
    # Use Gaussian RBF: exp(-(dist - center)^2 / width^2)
    rbf = tl.zeros([rbf_dim], dtype=tl.float32)
    for i in range(rbf_dim):
        center = tl.load(rbf_centers_ptr + i)
        width = tl.load(rbf_widths_ptr + i)
        rbf[i] = tl.exp(-((dist - center) ** 2) / (width ** 2))

    # Apply cutoff envelope
    envelope = cutoff_envelope(dist, cutoff)
    rbf = rbf * envelope

    # Load source node features
    feat_src = tl.load(node_features_ptr + src * hidden_dim + tl.arange(0, hidden_dim))

    # Compute message: MLP(rbf) * feat_src
    # Simplified: message = (W @ rbf + b) * feat_src
    message = tl.zeros([hidden_dim], dtype=tl.float32)
    for i in range(hidden_dim):
        mlp_out = tl.load(mlp_bias_ptr + i)
        for j in range(rbf_dim):
            weight = tl.load(mlp_weight_ptr + i * rbf_dim + j)
            mlp_out += weight * rbf[j]
        message[i] = mlp_out * feat_src[i]

    # Aggregate to destination (atomic add)
    for i in range(hidden_dim):
        tl.atomic_add(messages_out_ptr + dst * hidden_dim + i, message[i])


class FusedMessagePassing(torch.nn.Module):
    def __init__(self, hidden_dim=128, rbf_dim=20, cutoff=5.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rbf_dim = rbf_dim
        self.cutoff = cutoff

        # RBF parameters
        self.rbf_centers = torch.linspace(0, cutoff, rbf_dim)
        self.rbf_widths = torch.ones(rbf_dim) * (cutoff / rbf_dim)

        # Simplified MLP (1 layer)
        self.mlp_weight = torch.nn.Parameter(torch.randn(hidden_dim, rbf_dim))
        self.mlp_bias = torch.nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, positions, node_features, edge_index):
        N = positions.shape[0]
        n_edges = edge_index.shape[1]
        device = positions.device

        # Allocate output
        messages = torch.zeros((N, self.hidden_dim), dtype=torch.float32, device=device)

        # Launch kernel
        grid = (n_edges,)
        fused_message_passing_kernel[grid](
            positions, node_features, edge_index,
            None,  # edge_attr_out (not needed)
            messages,
            self.rbf_centers.to(device), self.rbf_widths.to(device),
            cutoff=self.cutoff,
            self.mlp_weight, self.mlp_bias,
            hidden_dim=self.hidden_dim,
            rbf_dim=self.rbf_dim,
            n_edges=n_edges,
            BLOCK_SIZE=128
        )

        return messages
```

**Deliverables**:

- [ ] `src/mlff_distiller/cuda/message_passing.py` (+300 lines)
- [ ] Fused RBF + message generation kernel
- [ ] Atomic aggregation implementation
- [ ] `tests/unit/test_message_passing_cuda.py` (+120 lines)

**Acceptance Criteria**:

- [ ] Kernel matches PyTorch reference (<1e-5 error)
- [ ] 1.5-2x faster than separate kernels
- [ ] No race conditions in atomic operations
- [ ] Memory access patterns optimized (coalesced)
- [ ] All tests passing

**Timeline**:
- Week 3, Day 1: Implement kernel structure and RBF computation
- Week 3, Day 2: Implement message generation and aggregation
- Week 3, Day 3: Test, optimize, and validate

---

### Issue #32: [CUDA] [M4] Implement fused analytical force kernel (Triton)

**Assigned to**: cuda-optimization-engineer
**Priority**: P1 (High)
**Labels**: `M4-optimization`, `cuda`, `triton`, `kernels`, `analytical-forces`
**Estimated Effort**: 2 days (16 hours)
**Depends on**: #31

**Description**:

Implement fused kernel for analytical force computation that combines geometric derivative computation and force assembly.

**Problem Statement**:

Current analytical force implementation (from Issue #26) uses separate operations for:
1. Computing geometric derivatives (∂r_ij/∂r_i)
2. Backpropagating energy gradients
3. Force assembly via scatter

Fusing these operations reduces memory traffic.

**Requirements**:

1. Fused operations:
   - Geometric derivative computation
   - Energy gradient multiplication (chain rule)
   - Force contribution scatter

2. Correctness:
   - Exact match with PyTorch scatter implementation
   - Proper handling of Newton's third law
   - Thread-safe atomic operations

3. Performance targets:
   - 1.5-2x faster than PyTorch scatter
   - Scales with number of edges

**Implementation Plan**:

Create `src/mlff_distiller/cuda/force_kernels.py`:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def fused_analytical_forces_kernel(
    positions_ptr,         # [N, 3]
    edge_index_ptr,        # [2, n_edges]
    energy_grad_edges_ptr, # [n_edges] (∂E/∂edge_features)
    forces_out_ptr,        # [N, 3] (output)
    n_edges: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Each thread processes one edge
    edge_id = tl.program_id(0)

    if edge_id >= n_edges:
        return

    # Load edge indices
    src = tl.load(edge_index_ptr + edge_id * 2)
    dst = tl.load(edge_index_ptr + edge_id * 2 + 1)

    # Load positions
    pos_src = tl.load(positions_ptr + src * 3 + tl.arange(0, 3))
    pos_dst = tl.load(positions_ptr + dst * 3 + tl.arange(0, 3))

    # Compute displacement vector
    r_vec = pos_dst - pos_src
    r_norm = tl.sqrt(tl.sum(r_vec * r_vec))

    # Geometric derivative: ∂r/∂r_src = -r_hat, ∂r/∂r_dst = r_hat
    r_hat = r_vec / r_norm

    # Load energy gradient w.r.t. this edge
    energy_grad = tl.load(energy_grad_edges_ptr + edge_id)

    # Compute force contribution (chain rule)
    # F_src = -∂E/∂r_src = -energy_grad * (-r_hat) = energy_grad * r_hat
    # F_dst = -∂E/∂r_dst = -energy_grad * r_hat
    force_contrib = energy_grad * r_hat

    # Atomic add to source (positive)
    for i in range(3):
        tl.atomic_add(forces_out_ptr + src * 3 + i, force_contrib[i])

    # Atomic add to destination (negative, Newton's third law)
    for i in range(3):
        tl.atomic_add(forces_out_ptr + dst * 3 + i, -force_contrib[i])


class FusedAnalyticalForces(torch.nn.Module):
    def forward(self, positions, edge_index, energy_grad_edges):
        N = positions.shape[0]
        n_edges = edge_index.shape[1]
        device = positions.device

        # Allocate output
        forces = torch.zeros((N, 3), dtype=torch.float32, device=device)

        # Launch kernel
        grid = (n_edges,)
        fused_analytical_forces_kernel[grid](
            positions, edge_index, energy_grad_edges,
            forces,
            n_edges=n_edges,
            BLOCK_SIZE=128
        )

        return forces
```

**Deliverables**:

- [ ] `src/mlff_distiller/cuda/force_kernels.py` (+200 lines)
- [ ] Fused geometric derivative + force assembly kernel
- [ ] Integration with analytical gradient implementation
- [ ] `tests/unit/test_force_kernels_cuda.py` (+100 lines)

**Acceptance Criteria**:

- [ ] Kernel matches PyTorch scatter (<1e-6 error)
- [ ] 1.5-2x faster than PyTorch scatter operations
- [ ] Forces satisfy Newton's third law (sum to zero)
- [ ] No race conditions or numerical errors
- [ ] All tests passing

**Timeline**:
- Week 3, Day 4: Implement kernel
- Week 3, Day 5: Test and validate

---

### Issue #33: [Testing] [M4] Integration testing and final benchmarking for Phase 3B

**Assigned to**: testing-benchmark-engineer (primary), cuda-optimization-engineer (support)
**Priority**: P1 (High)
**Labels**: `M4-optimization`, `testing`, `benchmarking`, `integration`
**Estimated Effort**: 1 day (8 hours)
**Depends on**: #32

**Description**:

Comprehensive integration testing and benchmarking of all Phase 3B optimizations.

**Problem Statement**:

Validate that all optimizations work together correctly and achieve target speedup of 15-25x over baseline.

**Requirements**:

1. Integration tests:
   - Analytical gradients + custom neighbor search
   - Analytical gradients + fused message passing
   - Analytical gradients + fused force kernels
   - All optimizations combined

2. Benchmarking:
   - Compare: baseline, analytical-only, analytical+kernels
   - System sizes: 10, 20, 50, 100, 200 atoms
   - Batch sizes: 1, 2, 4
   - MD workload simulation

3. Validation:
   - Numerical accuracy (<1e-5 error)
   - MD stability (NVE, 1000 steps)
   - Energy conservation (<0.1%/ns)
   - No regression in any metric

4. Performance analysis:
   - Per-component speedup breakdown
   - Cumulative speedup tracking
   - Memory usage comparison
   - GPU utilization metrics

**Implementation Plan**:

Create `scripts/benchmark_phase3b_final.py`:

```python
def benchmark_all_configurations():
    configs = {
        'baseline': {
            'use_analytical': False,
            'use_custom_neighbor_search': False,
            'use_fused_message_passing': False,
            'use_fused_forces': False,
        },
        'analytical_only': {
            'use_analytical': True,
            'use_custom_neighbor_search': False,
            'use_fused_message_passing': False,
            'use_fused_forces': False,
        },
        'analytical_custom_neighbor': {
            'use_analytical': True,
            'use_custom_neighbor_search': True,
            'use_fused_message_passing': False,
            'use_fused_forces': False,
        },
        'analytical_fused_mp': {
            'use_analytical': True,
            'use_custom_neighbor_search': True,
            'use_fused_message_passing': True,
            'use_fused_forces': False,
        },
        'full_optimized': {
            'use_analytical': True,
            'use_custom_neighbor_search': True,
            'use_fused_message_passing': True,
            'use_fused_forces': True,
        },
    }

    results = {}

    for config_name, config in configs.items():
        print(f"Benchmarking: {config_name}")

        for n_atoms in [10, 20, 50, 100, 200]:
            batch = create_test_batch(n_atoms=n_atoms)

            timing = benchmark_configuration(model, batch, config)

            results[f'{config_name}_{n_atoms}'] = timing

    # Compute cumulative speedups
    for n_atoms in [10, 20, 50, 100, 200]:
        baseline_time = results[f'baseline_{n_atoms}']['mean_ms']

        for config_name in configs.keys():
            optimized_time = results[f'{config_name}_{n_atoms}']['mean_ms']
            speedup = baseline_time / optimized_time
            results[f'{config_name}_{n_atoms}']['speedup'] = speedup

    return results
```

**Deliverables**:

- [ ] `scripts/benchmark_phase3b_final.py` (+300 lines)
- [ ] `benchmarks/phase3b_final_results.json` (comprehensive data)
- [ ] `docs/PHASE3B_FINAL_REPORT.md` (summary report)
- [ ] Speedup visualization plots
- [ ] Integration test suite

**Acceptance Criteria**:

- [ ] All optimizations work together without conflicts
- [ ] Total speedup: 15-25x over baseline (minimum 10x)
- [ ] Numerical accuracy maintained (<1e-5 error)
- [ ] MD stable (energy conservation <0.1%/ns)
- [ ] Production-ready deployment

**Success Metrics**:

- Minimum Success: 10x total speedup
- Target Success: 15-25x total speedup
- Stretch Success: 25-50x total speedup

**Timeline**:
- Week 3, Day 5: Run all benchmarks and generate final report

---

## Week 4 (Optional): Advanced Tuning Issues

### Issue #34: [CUDA] [M4] CUDA graphs integration for static workloads

**Assigned to**: cuda-optimization-engineer
**Priority**: P2 (Optional)
**Labels**: `M4-optimization`, `cuda`, `cuda-graphs`, `advanced`
**Estimated Effort**: 2 days (16 hours)
**Depends on**: #33

**Description**:

Integrate CUDA graphs to capture and replay inference pipeline for static molecule sizes.

**Requirements**:

1. Graph capture for common sizes (10, 20, 50 atoms)
2. Automatic fallback for unsupported cases
3. Benchmark kernel launch overhead reduction
4. Expected: 1.2-1.3x additional speedup

**Acceptance Criteria**:

- [ ] CUDA graphs work for static sizes
- [ ] 1.2x+ speedup for captured workloads
- [ ] Graceful fallback for dynamic cases

---

### Issue #35: [CUDA] [M4] Multi-stream execution for batched inference

**Assigned to**: cuda-optimization-engineer
**Priority**: P3 (Optional)
**Labels**: `M4-optimization`, `cuda`, `multi-stream`, `advanced`
**Estimated Effort**: 1.5 days (12 hours)
**Depends on**: #34

**Description**:

Implement multi-stream execution to overlap computation and memory transfers.

**Requirements**:

1. Create multiple CUDA streams
2. Pipeline batches across streams
3. Double-buffering for input/output
4. Expected: 1.1-1.2x additional speedup

**Acceptance Criteria**:

- [ ] Multi-stream pipeline working
- [ ] 1.1x+ throughput improvement
- [ ] No synchronization bugs

---

### Issue #36: [CUDA] [M4] Kernel auto-tuning for optimal performance

**Assigned to**: cuda-optimization-engineer
**Priority**: P3 (Optional)
**Labels**: `M4-optimization`, `cuda`, `auto-tuning`, `advanced`
**Estimated Effort**: 1 day (8 hours)
**Depends on**: #35

**Description**:

Add Triton auto-tuning to find optimal grid/block sizes for target hardware.

**Requirements**:

1. Auto-tune all custom kernels
2. Search optimal BLOCK_SIZE, num_warps
3. Save tuned configs for deployment
4. Expected: 1.05-1.1x additional speedup

**Acceptance Criteria**:

- [ ] Auto-tuning framework working
- [ ] Optimal configs found and saved
- [ ] 5-10% additional speedup

---

## Summary: Issue Roadmap

### Week 1 (Analytical Gradients)
- Issue #25: Derive analytical gradient formulas (2 days)
- Issue #26: Implement analytical forces (2 days)
- Issue #27: Validate vs autograd (1 day)
- Issue #28: Benchmark analytical gradients (0.5 days)
- **Target**: 9-10x total speedup

### Week 2-3 (Custom CUDA Kernels)
- Issue #29: Profile and identify bottlenecks (2 days)
- Issue #30: Neighbor search kernel (3 days)
- Issue #31: Fused message passing kernel (3 days)
- Issue #32: Fused force kernel (2 days)
- Issue #33: Integration testing and final benchmarking (1 day)
- **Target**: 15-25x total speedup

### Week 4 (Optional Advanced Tuning)
- Issue #34: CUDA graphs (2 days)
- Issue #35: Multi-stream execution (1.5 days)
- Issue #36: Kernel auto-tuning (1 day)
- **Target**: 25-50x total speedup

---

## Issue Labels

```
M4-optimization          # Milestone 4 (CUDA Optimization)
cuda                     # CUDA-related work
triton                   # Triton kernel implementation
analytical-gradients     # Analytical gradient computation
kernels                  # Custom kernel development
profiling                # Performance profiling
testing                  # Testing and validation
benchmarking             # Performance benchmarking
integration              # Integration testing
neighbor-search          # Neighbor search optimization
message-passing          # Message passing optimization
forces                   # Force computation optimization
cuda-graphs              # CUDA graphs (advanced)
multi-stream             # Multi-stream execution (advanced)
auto-tuning              # Kernel auto-tuning (advanced)
advanced                 # Advanced optimizations (Week 4)

priority:critical        # P0 - Must complete
priority:high            # P1 - Should complete
priority:medium          # P2 - Optional, high value
priority:low             # P3 - Optional, lower value
```

---

## Next Action

**Coordinator**: Review this issue plan and create Issues #25-#36 in GitHub.

**Immediate Next Steps**:
1. Get user approval for Phase 3B plan
2. Create GitHub Issues #25-#28 (Week 1)
3. Assign to cuda-optimization-engineer
4. Begin work on Issue #25 immediately

**Status**: Awaiting approval to proceed
