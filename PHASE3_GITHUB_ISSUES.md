# Phase 3 CUDA Optimization - GitHub Issues

**Created**: 2025-11-24
**Milestone**: M5 (CUDA Optimization)
**Target**: 5-10x speedup

---

## Issue #25: [M5] [CUDA] Install and integrate torch-cluster for optimized neighbor search

### Description

Integrate the torch-cluster library to replace our current O(N²) neighbor search implementation with an optimized O(N log N) algorithm. This is the first step in Phase 3 CUDA optimization and provides quick wins with minimal code changes.

### Background

Current implementation uses PyTorch pairwise distance matrix (O(N²) complexity), which becomes a bottleneck for systems with >50 atoms. torch-cluster provides battle-tested, CUDA-optimized implementations used by PyTorch Geometric.

**Profiling data** (from `benchmarks/cuda_x_analysis/`):
- Current neighbor search: 0.485 ms (12.3% of forward pass)
- Expected speedup: 2-3x on neighbor search
- Overall impact: 10-30% improvement for small molecules, more for large

### Acceptance Criteria

- [ ] torch-cluster installed successfully (`pip install torch-cluster`)
- [ ] Current `radius_graph_native` replaced with `torch_cluster.radius`
- [ ] All correctness tests passing (numerical equivalence)
- [ ] Benchmark shows 2-3x improvement on neighbor search operation
- [ ] Tested on 10, 50, 100 atom systems
- [ ] No accuracy regression (<1e-5 tolerance)
- [ ] Documentation updated with installation instructions

### Implementation Notes

**Installation**:
```bash
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.x.x+cu121.html
```

**Code changes** (in `src/mlff_distiller/models/student_model.py`):
```python
# Before
from mlff_distiller.models.student_model import radius_graph_native
edge_index = radius_graph_native(positions, batch, self.cutoff)

# After
from torch_cluster import radius
edge_index = radius(
    positions, positions,
    r=self.cutoff,
    batch_x=batch, batch_y=batch,
    max_num_neighbors=128
)
```

**Testing**:
```python
# Validate numerical equivalence
old_edges = radius_graph_native(positions, batch, cutoff)
new_edges = radius(positions, positions, r=cutoff, batch_x=batch, batch_y=batch)
assert torch.allclose(old_edges, new_edges)
```

**Benchmarking**:
```bash
python scripts/benchmark_phase3.py --optimization torch-cluster --system-sizes "10,50,100"
```

### Dependencies

None (this is the first optimization)

### Estimated Time

2-3 days

### Assignee

@cuda-optimization-engineer

### Priority

HIGH

### Labels

- `M5-cuda-optimization`
- `priority-high`
- `type-performance`
- `agent-cuda`

---

## Issue #26: [M5] [CUDA] Implement Triton fused message passing kernels

### Description

Implement a custom Triton kernel to fuse message passing operations (RBF computation, filter network, message aggregation) into a single GPU kernel. This reduces memory bandwidth and kernel launch overhead.

### Background

Current message passing implementation has multiple separate operations:
1. RBF computation (scalar operations)
2. Filter network (linear layers)
3. Element-wise multiplication
4. Scatter/aggregation (index_add)

Each operation launches a separate kernel and writes intermediate results to global memory. Fusing these operations reduces memory traffic significantly.

**Profiling data**:
- Current message passing: 1.630 ms (41% of forward pass)
- Expected speedup: 1.5-2x on message passing
- Overall impact: 20-30% improvement

### Acceptance Criteria

- [ ] Triton kernel implemented for message passing layer
- [ ] Fuses: RBF + filter network + aggregation
- [ ] Numerical equivalence validated (<1e-5 error vs PyTorch implementation)
- [ ] Benchmark shows 1.5-2x improvement on message passing
- [ ] Backward pass compatible with PyTorch autograd
- [ ] Works for variable-size molecules
- [ ] Memory usage not increased
- [ ] Unit tests covering edge cases

### Implementation Approach

**File**: `src/mlff_distiller/cuda/triton_message_passing.py`

**Kernel structure**:
```python
import triton
import triton.language as tl

@triton.jit
def fused_message_passing_kernel(
    # Inputs
    scalar_features,     # [N, hidden_dim]
    edge_index,          # [2, num_edges]
    edge_distances,      # [num_edges]
    filter_weights,      # [num_edges, hidden_dim*3]
    # Outputs
    scalar_out,          # [N, hidden_dim]
    # Constants
    N, num_edges, hidden_dim, cutoff
):
    # Each block processes one destination node
    dst_idx = tl.program_id(0)

    # Load scalar features for this node
    scalar_offset = dst_idx * hidden_dim
    scalar = tl.load(scalar_features + scalar_offset + tl.arange(0, hidden_dim))

    # Accumulate messages from all incoming edges
    message_sum = tl.zeros([hidden_dim], dtype=tl.float32)

    for edge_idx in range(num_edges):
        edge_dst = tl.load(edge_index + 1 * num_edges + edge_idx)

        if edge_dst == dst_idx:
            # This edge points to our node
            edge_src = tl.load(edge_index + 0 * num_edges + edge_idx)

            # Compute RBF
            dist = tl.load(edge_distances + edge_idx)
            rbf = compute_rbf(dist, cutoff)

            # Load and apply filter
            filter_offset = edge_idx * hidden_dim * 3
            filter_vec = tl.load(filter_weights + filter_offset + tl.arange(0, hidden_dim))

            # Load source features and compute message
            src_offset = edge_src * hidden_dim
            src_scalar = tl.load(scalar_features + src_offset + tl.arange(0, hidden_dim))

            message = src_scalar * filter_vec * rbf
            message_sum += message

    # Write output
    output = scalar + message_sum
    tl.store(scalar_out + scalar_offset + tl.arange(0, hidden_dim), output)
```

**Integration** (in `StudentForceField`):
```python
from mlff_distiller.cuda.triton_message_passing import fused_message_passing

# In forward pass
if self.use_fused_kernels:
    scalar_out = fused_message_passing(
        scalar, edge_index, edge_attr, filter_weights,
        cutoff=self.cutoff
    )
else:
    # Original PyTorch implementation
    scalar_out = self.message_passing_layer(scalar, edge_index, edge_attr)
```

### Testing Requirements

**Numerical equivalence**:
```python
def test_fused_kernel_equivalence():
    pytorch_out = pytorch_message_passing(scalar, edge_index, edge_attr)
    triton_out = fused_message_passing(scalar, edge_index, edge_attr)
    assert torch.allclose(pytorch_out, triton_out, atol=1e-5)
```

**Gradient correctness**:
```python
def test_gradient_correctness():
    scalar.requires_grad = True
    energy = model(atomic_numbers, positions)
    energy.backward()
    grad_pytorch = scalar.grad.clone()

    scalar.grad = None
    model.use_fused_kernels = True
    energy = model(atomic_numbers, positions)
    energy.backward()
    grad_triton = scalar.grad

    assert torch.allclose(grad_pytorch, grad_triton, atol=1e-4)
```

**Benchmark**:
```bash
python scripts/benchmark_phase3.py --optimization triton-fusion --system-sizes "10,50,100"
```

### Dependencies

- Issue #25 (torch-cluster) should be completed first
- Requires: `pip install triton`

### Estimated Time

5-7 days

### Assignee

@cuda-optimization-engineer

### Priority

HIGH

### Labels

- `M5-cuda-optimization`
- `priority-high`
- `type-performance`
- `agent-cuda`
- `complexity-high`

---

## Issue #27: [M5] [CUDA] Implement CUDA graphs for reduced overhead

### Description

Implement CUDA graph capture to reduce kernel launch overhead. CUDA graphs allow capturing an entire sequence of GPU operations and replaying them with a single launch, eliminating per-kernel CPU overhead.

### Background

Current implementation launches kernels one-by-one through PyTorch, incurring CPU overhead for each launch. For small, fast operations, this overhead can be significant (10-20% of total time).

**Profiling data**:
- Expected speedup: 1.2-1.3x additional improvement
- Most beneficial for small molecules (high kernel launch overhead)

### Acceptance Criteria

- [ ] CUDA graph capture implemented for forward pass
- [ ] Works for common molecule sizes (10, 20, 50, 100 atoms)
- [ ] Automatic size-based graph selection
- [ ] Fallback to non-graph mode for uncommon sizes
- [ ] Benchmark shows 1.2-1.3x additional improvement
- [ ] No accuracy loss vs non-graph implementation
- [ ] Memory usage documented
- [ ] Thread-safe implementation

### Implementation Approach

**File**: `src/mlff_distiller/cuda/cuda_graphs.py`

**Graph capture**:
```python
class CUDAGraphInference:
    def __init__(self, model, common_sizes=[10, 20, 50, 100]):
        self.model = model
        self.graphs = {}
        self.static_inputs = {}

        # Pre-capture graphs for common sizes
        for size in common_sizes:
            self._capture_graph(size)

    def _capture_graph(self, num_atoms):
        # Create dummy inputs
        atomic_numbers = torch.ones(num_atoms, dtype=torch.long, device='cuda')
        positions = torch.randn(num_atoms, 3, device='cuda', requires_grad=True)

        # Warmup
        for _ in range(10):
            energy = self.model(atomic_numbers, positions)

        # Capture graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            energy = self.model(atomic_numbers, positions)

        self.graphs[num_atoms] = graph
        self.static_inputs[num_atoms] = {
            'atomic_numbers': atomic_numbers,
            'positions': positions,
            'energy': energy
        }

    def predict(self, atomic_numbers, positions):
        num_atoms = atomic_numbers.shape[0]

        if num_atoms in self.graphs:
            # Use pre-captured graph
            inputs = self.static_inputs[num_atoms]

            # Copy input data to static buffers
            inputs['atomic_numbers'].copy_(atomic_numbers)
            inputs['positions'].copy_(positions)

            # Replay graph
            self.graphs[num_atoms].replay()

            # Return result from static buffer
            return inputs['energy'].clone()
        else:
            # Fall back to normal execution
            return self.model(atomic_numbers, positions)
```

**Integration**:
```python
# In inference code
if use_cuda_graphs:
    inference_engine = CUDAGraphInference(model, common_sizes=[10, 20, 50, 100])
    energy = inference_engine.predict(atomic_numbers, positions)
else:
    energy = model(atomic_numbers, positions)
```

### Testing Requirements

**Correctness**:
```python
def test_cuda_graph_correctness():
    # Non-graph execution
    energy_baseline = model(atomic_numbers, positions)

    # Graph execution
    graph_engine = CUDAGraphInference(model)
    energy_graph = graph_engine.predict(atomic_numbers, positions)

    assert torch.allclose(energy_baseline, energy_graph, atol=1e-6)
```

**Performance**:
```bash
python scripts/benchmark_phase3.py --optimization cuda-graphs --system-sizes "10,20,50,100"
```

**Memory**:
```python
def test_memory_usage():
    # Measure memory before and after graph capture
    torch.cuda.reset_peak_memory_stats()
    graph_engine = CUDAGraphInference(model, common_sizes=[10, 20, 50, 100])
    memory_used = torch.cuda.max_memory_allocated() / 1024**2
    print(f"Memory for graphs: {memory_used:.2f} MB")
```

### Dependencies

- Issue #26 (Triton kernels) should be completed first
- Works best with fused kernels (fewer nodes in graph)

### Estimated Time

2-3 days

### Assignee

@cuda-optimization-engineer

### Priority

MEDIUM

### Labels

- `M5-cuda-optimization`
- `priority-medium`
- `type-performance`
- `agent-cuda`

---

## Issue #28: [M5] [Testing] Comprehensive benchmark suite for 5-10x validation

### Description

Create a comprehensive benchmarking suite to validate that we achieve the 5-10x speedup target. This suite will track performance across all optimization stages and provide detailed analysis.

### Background

We need rigorous benchmarking to:
1. Validate each optimization independently
2. Measure cumulative speedup
3. Identify performance regressions
4. Track accuracy across optimizations
5. Provide data for final report

### Acceptance Criteria

- [ ] Benchmark suite covers all system sizes (10, 20, 50, 100, 200 atoms)
- [ ] Measures latency (mean, median, std, min, max)
- [ ] Measures throughput (structures/second)
- [ ] Measures GPU memory usage
- [ ] Tracks accuracy (energy MAE, force RMSE)
- [ ] Compares against baseline and intermediate optimizations
- [ ] Validates 5-10x speedup target achieved
- [ ] Results exported to JSON and markdown reports
- [ ] Visualization plots generated
- [ ] Automated regression detection

### Implementation Approach

**File**: `benchmarks/benchmark_phase3.py`

**Structure**:
```python
class Phase3Benchmark:
    def __init__(self, baseline_model, optimized_models):
        self.baseline = baseline_model
        self.optimized = optimized_models  # Dict of optimization configs
        self.results = {}

    def benchmark_config(self, model, config_name, test_molecules):
        """Benchmark one configuration"""
        results = {
            'config': config_name,
            'system_results': {},
            'memory': torch.cuda.max_memory_allocated() / 1024**2
        }

        for mol in test_molecules:
            num_atoms = mol['atomic_numbers'].shape[0]

            # Warmup
            for _ in range(10):
                energy, forces = self.predict(model, mol)

            # Benchmark
            times = []
            for _ in range(100):
                start = time.perf_counter()
                energy, forces = self.predict(model, mol)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

            results['system_results'][num_atoms] = {
                'mean_time_ms': np.mean(times) * 1000,
                'std_time_ms': np.std(times) * 1000,
                'median_time_ms': np.median(times) * 1000,
                'min_time_ms': np.min(times) * 1000,
                'throughput': 1.0 / np.mean(times),
                'energy': energy.item(),
                'forces_norm': torch.norm(forces).item()
            }

        return results

    def compare_accuracy(self, baseline_results, optimized_results):
        """Compare accuracy between configurations"""
        energy_errors = []
        force_errors = []

        for system_size in baseline_results['system_results']:
            baseline = baseline_results['system_results'][system_size]
            optimized = optimized_results['system_results'][system_size]

            energy_error = abs(baseline['energy'] - optimized['energy'])
            force_error = abs(baseline['forces_norm'] - optimized['forces_norm'])

            energy_errors.append(energy_error)
            force_errors.append(force_error)

        return {
            'mean_energy_error_eV': np.mean(energy_errors),
            'max_energy_error_eV': np.max(energy_errors),
            'mean_force_error': np.mean(force_errors),
            'max_force_error': np.max(force_errors)
        }

    def generate_report(self, output_path):
        """Generate markdown report"""
        # Create detailed report with tables, plots, analysis
        pass
```

**Usage**:
```bash
python benchmarks/benchmark_phase3.py \
    --baseline models/student_model_jit.pt \
    --torch-cluster \
    --triton-fusion \
    --cuda-graphs \
    --system-sizes "10,20,50,100,200" \
    --iterations 100 \
    --output results/phase3_benchmark.json \
    --report results/phase3_report.md
```

### Testing Requirements

**Validation**:
- [ ] Benchmark runs without errors
- [ ] Results reproducible (±5% variance)
- [ ] Accuracy metrics computed correctly
- [ ] Speedup calculations correct
- [ ] Memory measurements accurate

**Regression detection**:
```python
def detect_regression(current_results, previous_results, threshold=0.95):
    """Alert if performance regresses by >5%"""
    current_time = current_results['mean_time_ms']
    previous_time = previous_results['mean_time_ms']

    if current_time > previous_time * 1.05:
        print(f"WARNING: Regression detected! {previous_time:.2f}ms -> {current_time:.2f}ms")
        return True
    return False
```

### Dependencies

- All optimization issues (#25, #26, #27) for comprehensive testing
- Can start in parallel with basic baseline benchmarking

### Estimated Time

Ongoing throughout Phase 3 (3-5 days total effort)

### Assignee

@testing-benchmark-engineer

### Priority

HIGH

### Labels

- `M5-cuda-optimization`
- `priority-high`
- `type-testing`
- `agent-testing`

---

## Issue #29: [M5] [Testing] MD stability validation with optimized kernels

### Description

Validate that all CUDA optimizations maintain numerical stability for molecular dynamics (MD) simulations. This ensures production readiness for downstream applications.

### Background

CUDA optimizations (especially custom kernels) can introduce subtle numerical errors that accumulate over long MD simulations. We need to validate:
1. Energy conservation (NVE ensemble)
2. Force accuracy over time
3. No catastrophic failures
4. Comparable stability to baseline

**Target**: Stable MD for 1000+ steps with <1% energy drift

### Acceptance Criteria

- [ ] MD simulations run with optimized model (1000 steps)
- [ ] Energy conservation validated (NVE ensemble, <1% drift)
- [ ] Force accuracy maintained (<0.01 eV/Å RMSE)
- [ ] Compare stability against baseline TorchScript model
- [ ] Test on multiple systems (water, ethanol, benzene, peptide)
- [ ] Document any drift or instabilities
- [ ] Provide recommendations for production use

### Implementation Approach

**File**: `scripts/validate_md_stability.py`

**MD setup**:
```python
from ase import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
from mlff_distiller.inference import StudentForceFieldCalculator

def run_md_validation(calculator, atoms, steps=1000, temperature=300):
    """Run MD and track energy drift"""

    # Set up calculator
    atoms.calc = calculator

    # Initialize velocities
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)

    # Create MD integrator (NVE)
    dyn = VelocityVerlet(atoms, timestep=0.5 * units.fs)

    # Track energy
    energies = []
    forces_norms = []

    def track_energy():
        energies.append(atoms.get_potential_energy())
        forces_norms.append(np.linalg.norm(atoms.get_forces()))

    dyn.attach(track_energy, interval=1)

    # Run MD
    dyn.run(steps)

    # Analyze
    energies = np.array(energies)
    energy_drift = (energies[-1] - energies[0]) / abs(energies[0])
    energy_std = np.std(energies) / abs(np.mean(energies))

    return {
        'initial_energy': energies[0],
        'final_energy': energies[-1],
        'energy_drift_percent': energy_drift * 100,
        'energy_std_percent': energy_std * 100,
        'mean_force_norm': np.mean(forces_norms),
        'std_force_norm': np.std(forces_norms)
    }

# Test multiple systems
test_systems = [
    create_water_molecule(),
    create_ethanol_molecule(),
    create_benzene_molecule(),
    create_small_peptide()
]

for i, system in enumerate(test_systems):
    print(f"\n=== System {i+1}: {system.get_chemical_formula()} ===")

    # Baseline
    baseline_calc = StudentForceFieldCalculator(
        checkpoint_path='checkpoints/best_model.pt',
        jit_path='models/student_model_jit.pt',
        use_jit=True
    )
    baseline_results = run_md_validation(baseline_calc, system.copy(), steps=1000)

    # Optimized
    optimized_calc = StudentForceFieldCalculator(
        checkpoint_path='checkpoints/best_model.pt',
        use_torch_cluster=True,
        use_triton_fusion=True,
        use_cuda_graphs=True
    )
    optimized_results = run_md_validation(optimized_calc, system.copy(), steps=1000)

    # Compare
    print(f"Baseline energy drift: {baseline_results['energy_drift_percent']:.2f}%")
    print(f"Optimized energy drift: {optimized_results['energy_drift_percent']:.2f}%")
```

**Success criteria**:
- Energy drift <1% (acceptable for NVE)
- Similar stability between baseline and optimized
- No NaN or Inf values during simulation
- Forces remain bounded

### Testing Requirements

**Test cases**:
1. **Short test** (100 steps): Quick sanity check
2. **Medium test** (1000 steps): Standard validation
3. **Long test** (10000 steps): Production-level validation

**Systems to test**:
- Water (H2O): 3 atoms, simple
- Ethanol (C2H5OH): 9 atoms, hydrogen bonding
- Benzene (C6H6): 12 atoms, aromatic
- Small peptide: 20-50 atoms, complex

**Ensembles**:
- NVE (microcanonical): Energy conservation
- NVT (canonical): Temperature control

### Dependencies

- Issue #26 (Triton kernels) - main optimization to validate
- Issue #27 (CUDA graphs) - ensure graphs don't break MD
- Can start with baseline validation in parallel

### Estimated Time

3-4 days

### Assignee

@testing-benchmark-engineer

### Priority

HIGH

### Labels

- `M5-cuda-optimization`
- `priority-high`
- `type-testing`
- `agent-testing`
- `validation-md`

---

## Issue Dependency Graph

```
#25 (torch-cluster)
  ↓
#26 (Triton kernels) ----→ #29 (MD stability)
  ↓                             ↓
#27 (CUDA graphs) -----------→ #29 (MD stability)
  ↓                             ↓
#28 (Comprehensive benchmarks) ←┘
```

**Critical path**: #25 → #26 → #27 → #28
**Parallel work**: #28 and #29 can start baseline validation early

---

## Project Board Setup

### Columns

1. **Backlog**: Issues not yet started
2. **In Progress**: Currently being worked on
3. **Review**: Code complete, awaiting review
4. **Testing**: Under validation
5. **Done**: Completed and merged

### Initial State

**Backlog**:
- Issue #25
- Issue #26
- Issue #27
- Issue #28 (partial - can start baseline)
- Issue #29 (partial - can start baseline)

**Target weekly progress**:
- Week 1: #25 → Done, #28 baseline → Done
- Week 2-3: #26 → Done, #29 baseline → Done
- Week 4: #27 → Done, #28 final → Done, #29 final → Done

---

## Summary

**Total Issues**: 5
**High Priority**: 4
**Medium Priority**: 1

**Estimated Total Time**: 4 weeks
**Critical Path**: 15-18 days
**Parallel Work**: 5-7 days

**Success Metric**: All 5 issues completed, 5-10x speedup validated, MD stable

---

**Created by**: Lead Project Coordinator
**Date**: 2025-11-24
**Status**: Ready for agent assignment
