# Python 3.13 Optimization Alternatives (Without torch.compile)

**Context:** torch.compile() is not available on Python 3.13. This document explores alternative optimization paths that DO work on Python 3.13.

**Goal:** Achieve 2-3x speedup without torch.compile()

**Date:** 2025-11-24

---

## Viable Optimizations for Python 3.13

### 1. Custom CUDA Kernels (HIGHEST IMPACT)

**Speedup:** 1.5-2.5x for single inference
**Timeline:** 1-2 weeks
**Complexity:** High

**What:**
Write custom CUDA kernels using PyTorch C++ extensions or CuPy.

**Target Operations:**
1. **Neighbor Search (radius_graph)**
   - Current: O(N²) distance computation
   - Optimized: Cell list algorithm O(N)
   - Expected: 1.5-2x faster

2. **RBF Computation + Cutoff**
   - Current: Separate ops (RBF → cutoff → multiply)
   - Optimized: Fused kernel
   - Expected: 1.2-1.5x faster

3. **Message Passing Aggregation**
   - Current: PyTorch scatter_add
   - Optimized: Custom CUDA kernel with shared memory
   - Expected: 1.3-1.7x faster

4. **Force Scatter**
   - Current: Autograd backward
   - Optimized: Custom gradient kernel
   - Expected: 1.2-1.5x faster

**Implementation:**

```python
# Example: Fused RBF + Cutoff kernel
import torch
from torch.utils.cpp_extension import load

# Compile custom CUDA extension
rbf_cuda = load(
    name="rbf_cutoff_fused",
    sources=["cuda_ops/rbf_cutoff.cu"],
    extra_cuda_cflags=["-O3"]
)

# Use in forward pass
def forward(self, distances):
    # Fused RBF + cutoff computation
    return rbf_cuda.forward(distances, self.centers, self.widths, self.cutoff)
```

**Files to Create:**
- `cuda_ops/rbf_cutoff.cu` - Fused RBF kernel
- `cuda_ops/neighbor_search.cu` - Cell list algorithm
- `cuda_ops/message_passing.cu` - Aggregation kernel
- `setup.py` - Build script

---

### 2. Batch Processing Fix (CRITICAL FOR BATCHES)

**Speedup:** 10-20x for batched inference
**Timeline:** 3-5 days
**Complexity:** Medium

**Problem:**
Current batch implementation is 28x **slower** than single inference.

**Expected:**
Batch-16 should be ~10x **faster** per structure.

**Root Cause Analysis:**

```python
# Current (SLOW) - processes sequentially
for atoms in batch:
    energy = model(atoms)  # Separate forward pass!

# Expected (FAST) - true batching
batch_energies = model(batched_atoms)  # Single forward pass
```

**Fix:**
1. Concatenate all atoms into single tensor
2. Create batch index tensor
3. Single model forward pass
4. Scatter results back per structure

**Implementation:**

```python
def calculate_batch_optimized(atoms_list):
    # Concatenate all atoms
    all_positions = torch.cat([a.positions for a in atoms_list])
    all_numbers = torch.cat([a.numbers for a in atoms_list])

    # Create batch indices
    batch_idx = torch.cat([
        torch.full((len(a),), i) for i, a in enumerate(atoms_list)
    ])

    # Single forward pass (FAST!)
    energies = model(all_numbers, all_positions, batch=batch_idx)

    # Compute forces (batched autograd)
    forces = -torch.autograd.grad(
        energies,
        all_positions,
        grad_outputs=torch.ones_like(energies)
    )[0]

    return energies, forces
```

**Impact:**
- Single inference: No change
- Batch-16 inference: **10-20x faster per structure**

---

### 3. Triton Kernels (Alternative to torch.compile)

**Speedup:** 1.5-2x
**Timeline:** 1 week
**Complexity:** Medium

**What:**
Triton is a Python-based CUDA kernel language that may work on Python 3.13 (need to verify).

**Advantages over raw CUDA:**
- Python syntax
- Automatic memory management
- Easier to write than CUDA C++

**Example:**

```python
import triton
import triton.language as tl

@triton.jit
def rbf_kernel(
    distances_ptr,
    centers_ptr,
    output_ptr,
    n_edges: tl.constexpr,
    n_rbf: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Triton kernel for RBF computation
    edge_id = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Load distance
    distance = tl.load(distances_ptr + edge_id, mask=edge_id < n_edges)

    # Compute RBF for all centers
    for k in range(n_rbf):
        center = tl.load(centers_ptr + k)
        diff = distance - center
        rbf_val = tl.exp(-diff * diff / width)
        tl.store(output_ptr + edge_id * n_rbf + k, rbf_val)
```

**Need to test:**
```bash
# Check if Triton works on Python 3.13
pip install triton
python -c "import triton; print(triton.__version__)"
```

If Triton works → Implement kernels in 1 week → 1.5-2x speedup

---

### 4. Memory Optimizations

**Speedup:** 1.1-1.3x
**Timeline:** 2-3 days
**Complexity:** Low

**Techniques:**

1. **Pre-allocate Buffers**
```python
class StudentForceField(nn.Module):
    def __init__(self, ...):
        # Pre-allocate reusable buffers
        self.register_buffer('edge_buffer', torch.empty(...))
        self.register_buffer('rbf_buffer', torch.empty(...))

    def forward(self, ...):
        # Reuse buffers (no allocation overhead)
        self.edge_buffer[:n_edges] = compute_edges(...)
```

2. **Pinned Memory for CPU-GPU Transfer**
```python
# Use pinned memory for faster transfers
positions_gpu = torch.from_numpy(positions).pin_memory().cuda(non_blocking=True)
```

3. **Avoid Intermediate Tensors**
```python
# Before (creates intermediate)
rbf = compute_rbf(distances)
cutoff = compute_cutoff(distances)
modulated = rbf * cutoff

# After (in-place)
rbf = compute_rbf(distances)
rbf *= compute_cutoff(distances)  # In-place multiplication
```

4. **Gradient Checkpointing for Large Molecules**
```python
# For very large systems (>100 atoms)
from torch.utils.checkpoint import checkpoint

def forward(self, ...):
    # Save memory by recomputing forward during backward
    return checkpoint(self._forward_impl, ...)
```

---

### 5. Torch Scripting (Limited)

**Speedup:** 1.1-1.2x (for specific ops only)
**Timeline:** 2-3 days
**Complexity:** Low

**What:**
Use `@torch.jit.script` for individual functions (not whole model).

**Why Full JIT Failed:**
Whole model JIT adds too much overhead.

**What Might Work:**
JIT-compile small, compute-intensive functions only.

**Example:**

```python
@torch.jit.script
def compute_rbf_jit(
    distances: torch.Tensor,
    centers: torch.Tensor,
    widths: torch.Tensor
) -> torch.Tensor:
    """JIT-compiled RBF computation."""
    diff = distances.unsqueeze(-1) - centers
    gamma = 1.0 / (widths ** 2)
    return torch.exp(-gamma * diff ** 2)

class GaussianRBF(nn.Module):
    def forward(self, distances):
        # Use JIT version
        return compute_rbf_jit(distances, self.centers, self.widths)
```

**Target Functions:**
- RBF computation
- Cutoff function
- Distance calculations

**Expected:** 1.1-1.2x speedup (modest)

---

### 6. Operator Fusion (Manual)

**Speedup:** 1.2-1.5x
**Timeline:** 3-5 days
**Complexity:** Medium

**What:**
Manually fuse operations that PyTorch keeps separate.

**Example 1: RBF + Cutoff Fusion**

```python
# Before (2 kernel launches)
rbf = gaussian_rbf(distances)  # Kernel 1
cutoff = cosine_cutoff(distances)  # Kernel 2
modulated = rbf * cutoff  # Kernel 3

# After (1 kernel launch)
def fused_rbf_cutoff(distances, centers, widths, cutoff_dist):
    diff = distances.unsqueeze(-1) - centers
    gamma = 1.0 / (widths ** 2)
    rbf = torch.exp(-gamma * diff ** 2)

    # Fuse cutoff computation
    cutoff_val = 0.5 * (torch.cos(np.pi * distances / cutoff_dist) + 1.0)
    cutoff_val = cutoff_val * (distances < cutoff_dist).float()

    # Fuse multiplication
    return rbf * cutoff_val.unsqueeze(-1)
```

**Example 2: Message + Update Fusion**

```python
# Before (separate passes)
scalar_out, vector_out = message_layer(...)
scalar_final, vector_final = update_layer(scalar_out, vector_out)

# After (fused)
def fused_message_update(...):
    # Compute message
    scalar_msg = ...
    vector_msg = ...

    # Update immediately (reuse activations)
    scalar_final = scalar_msg + update_scalar(scalar_msg)
    vector_final = vector_msg + update_vector(vector_msg)

    return scalar_final, vector_final
```

---

## Recommended Optimization Sequence (Python 3.13)

### Week 1: Batch Processing Fix (CRITICAL)

**Days 1-2:** Debug current batch bug
- Profile batch vs single inference
- Identify sequential processing
- Implement true batching

**Days 3-4:** Test and validate
- Benchmark batch sizes 1, 4, 8, 16, 32
- Verify numerical correctness
- Validate against single inference

**Day 5:** Documentation
- Document batch API
- Create usage examples
- Update benchmarks

**Expected:** 10-20x speedup for batched workloads

---

### Week 2: Memory + Operator Fusion

**Days 1-2:** Memory optimizations
- Pre-allocate buffers
- Pinned memory
- Reduce intermediate tensors

**Days 3-4:** Operator fusion
- Fuse RBF + cutoff
- Fuse message + update
- Test TorchScript for hot functions

**Day 5:** Benchmark and validate
- Measure speedup
- Validate accuracy
- Document changes

**Expected:** 1.2-1.5x additional speedup

---

### Weeks 3-4: Custom CUDA Kernels

**Week 3:** Neighbor search + RBF
- Implement cell list algorithm
- Fused RBF kernel
- Test Triton compatibility

**Week 4:** Message passing + Forces
- Custom aggregation kernel
- Optimized force scatter
- Integration and testing

**Expected:** 1.5-2x additional speedup

---

## Cumulative Speedup (Python 3.13 Path)

| Optimization | Speedup | Cumulative | Timeline |
|--------------|---------|------------|----------|
| **Baseline** | 1.0x | 1.0x | - |
| Batch processing fix | 1.0x (single) | 1.0x | Week 1 |
| Memory + Fusion | 1.3x | **1.3x** | Week 2 |
| CUDA kernels | 2.0x | **2.6x** | Weeks 3-4 |
| **Total** | - | **2-3x single, 10-20x batch** | 4 weeks |

**Note:** Batch speedup is separate (applies when using batched inference)

---

## Comparison: Python 3.12 vs 3.13 Paths

| Metric | Python 3.12 (torch.compile) | Python 3.13 (CUDA) |
|--------|----------------------------|-------------------|
| **Week 2 Speedup** | 2-3x | 1.3x |
| **Week 4 Speedup** | 3-5x | 2-3x |
| **Timeline** | 3 days → 2 weeks | 4 weeks |
| **Complexity** | Low (just config) | High (CUDA code) |
| **Maintenance** | Easy (PyTorch updates) | Hard (custom kernels) |
| **Batch Benefit** | + torch.compile | Same (10-20x) |

**Conclusion:** Python 3.12 path is **faster and easier** for similar results.

---

## Testing Triton on Python 3.13

Let's verify if Triton works:

```bash
# Install Triton
pip install triton

# Test import
python -c "import triton; print(f'Triton {triton.__version__} works!')"

# If successful:
# → Use Triton for kernels (easier than raw CUDA)
# → Expected: 1.5-2x speedup in 1 week

# If failed:
# → Fall back to PyTorch C++ extensions
# → Timeline: 2 weeks for same speedup
```

---

## Recommended Action

**Still recommend Python 3.12 migration:**

**Why:**
1. Faster timeline (3 days vs 4 weeks)
2. Easier implementation (config vs CUDA)
3. Better maintainability
4. Similar final performance

**But if Python 3.13 is required:**
1. Week 1: Fix batch processing
2. Week 2: Memory + fusion optimizations
3. Weeks 3-4: CUDA kernels (or Triton if supported)
4. Target: 2-3x single, 10-20x batch

**Best path:** Python 3.12 for Week 2, then CUDA in Weeks 3-4 for maximum performance (3-6x total).

---

## Files Created

1. Benchmark script: `scripts/benchmark_compile_modes.py`
2. CUDA ops directory: `cuda_ops/` (to be created)
3. Batch processing fix: `src/mlff_distiller/inference/batch_processing.py` (to be created)
4. Memory optimizations: Applied to `student_model.py`

---

Last Updated: 2025-11-24
Status: Python 3.13 alternatives documented, awaiting user decision
