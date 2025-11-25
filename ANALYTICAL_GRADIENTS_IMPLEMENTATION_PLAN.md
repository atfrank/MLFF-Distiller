# Analytical Gradients Implementation - Single MD Optimization

**Goal**: Achieve 1.8-2x speedup for single-molecule MD simulations
**Target**: Reduce 16 ms/step → 8-9 ms/step
**Approach**: Eliminate autograd by computing forces analytically

---

## Current Performance Breakdown

From Week 3 profiling:

| Component | Time (ms) | % of Total | Can Optimize? |
|-----------|-----------|------------|---------------|
| **Autograd backward** | 10.1 | 75% | ✅ YES (analytical) |
| Forward pass | 3.3 | 25% | ⚠️ Limited (kernels) |
| **Total** | **13.4** | **100%** | |

**Key insight**: Eliminating autograd = 2.5-3x potential speedup!

But we need to keep forward pass overhead, so realistic target is **1.8-2x**.

---

## Mathematical Foundation (✅ COMPLETE)

Week 1 delivered complete mathematical derivation:
- **File**: `docs/ANALYTICAL_FORCES_DERIVATION.md` (400+ lines)
- RBF gradient formulas
- Cutoff function gradients
- Unit vector Jacobians
- Message passing chain rule
- Force accumulation strategy

**This foundation is production-ready** - we just need to implement it!

---

## Implementation Strategy

### Phase 1: RBF & Distance Gradients (Days 1-3)

**Goal**: Eliminate autograd for distance-based features

**Components**:
1. Analytical RBF gradients: `∂φ_k/∂r_ij`
2. Analytical cutoff gradients: `∂f_cut/∂r_ij`
3. Unit vector Jacobians: `∂d_ij/∂r_i`

**Expected speedup**: 1.2-1.3x (partial autograd elimination)

**Implementation**:
```python
def compute_rbf_forces_analytical(
    distances: Tensor,  # [n_edges]
    edge_vec: Tensor,   # [n_edges, 3]
    rbf_params: Tensor, # [n_rbf]
) -> Tensor:  # [n_edges, 3, n_rbf]
    """
    Compute ∂RBF/∂r analytically.

    Formula: ∂φ_k/∂r_i = -2γ(r_ij - μ_k) · φ_k(r_ij) · d_ij
    """
    # Compute RBF values
    rbf = torch.exp(-gamma * (distances.unsqueeze(-1) - centers) ** 2)

    # Compute gradients
    grad_coeff = -2 * gamma * (distances.unsqueeze(-1) - centers) * rbf
    grad_rbf = grad_coeff.unsqueeze(1) * edge_vec.unsqueeze(-1)

    return grad_rbf  # [n_edges, 3, n_rbf]
```

---

### Phase 2: Message Passing Gradients (Days 4-7)

**Goal**: Chain gradients through PaiNN layers

**Components**:
1. Message function gradients: `∂msg/∂positions`
2. Update function gradients: `∂update/∂positions`
3. Gradient accumulation across layers
4. Force computation: `F_i = -∂E/∂r_i`

**Expected speedup**: 1.5-1.6x cumulative (more autograd eliminated)

**Implementation**:
```python
def forward_with_analytical_forces(
    self,
    atomic_numbers: Tensor,
    positions: Tensor,
    ...
) -> Tuple[Tensor, Tensor]:
    """
    Forward pass with analytical force computation.

    Returns:
        energy: [batch_size] total energies
        forces: [n_atoms, 3] atomic forces
    """
    # Forward pass (keep all intermediates)
    intermediates = []

    # Embedding
    h = self.embedding(atomic_numbers)
    intermediates.append(('embedding', h))

    # Edge features
    edge_index, edge_vec, distances = self.get_edges(positions, ...)
    rbf = self.rbf(distances)
    cutoff = self.cutoff_fn(distances)
    edge_features = rbf * cutoff
    intermediates.append(('edges', (edge_vec, distances, rbf, cutoff)))

    # Message passing layers
    for layer in self.message_layers:
        h, intermediates = layer.forward_with_cache(h, edge_index, edge_features, intermediates)

    # Energy prediction
    energy = self.energy_head(h).sum(dim=0)  # [batch_size]

    # Analytical force computation (backward through intermediates)
    forces = self._compute_forces_analytical(energy, intermediates, positions, edge_index)

    return energy, forces
```

---

### Phase 3: Optimization & Validation (Days 8-10)

**Goal**: Validate correctness and optimize performance

**Tasks**:
1. **Accuracy validation**: Compare analytical vs autograd forces
   - Target: MAE < 1e-6 eV/Å
   - Test: 1000 random structures

2. **Performance benchmarking**: Measure actual speedup
   - Target: 1.8-2x faster than autograd
   - Test: Drug-like molecules

3. **Numerical stability**: Handle edge cases
   - Small distances (< 0.1 Å)
   - Zero forces (equilibrium)
   - Large systems (100+ atoms)

4. **MD validation**: Energy conservation
   - Run 10,000 step NVE simulation
   - Check: ΔE/E < 0.001 (0.1% drift)

---

## Implementation Timeline

### Days 1-3: RBF Gradients

**Day 1**:
- Implement `compute_rbf_gradient_analytical()`
- Implement `compute_cutoff_gradient_analytical()`
- Unit tests for gradient correctness

**Day 2**:
- Integrate into StudentForceField model
- Cache edge information during forward pass
- Benchmark: Measure partial speedup

**Day 3**:
- Fix numerical stability issues
- Validate on 100 test molecules
- Document: RBF gradients report

**Deliverable**: RBF analytical gradients working, 1.2-1.3x speedup

---

### Days 4-7: Message Passing Gradients

**Day 4**:
- Design intermediate caching strategy
- Implement backward through first PaiNN layer
- Unit tests for message gradient

**Day 5**:
- Implement backward through update layers
- Chain gradients across all 3 interaction blocks
- Test gradient accumulation

**Day 6**:
- Integrate end-to-end force computation
- Fix shape mismatches and indexing bugs
- Validate forces match autograd

**Day 7**:
- Optimize memory usage (cache reuse)
- Benchmark: Measure cumulative speedup
- Document: Message passing gradients

**Deliverable**: Full analytical gradients working, 1.5-1.6x speedup

---

### Days 8-10: Validation & Optimization

**Day 8**:
- Comprehensive accuracy validation
- Fix any remaining numerical issues
- Edge case testing

**Day 9**:
- Full performance benchmark suite
- Compare to all baselines
- MD energy conservation testing

**Day 10**:
- Final documentation
- Production deployment guide
- Code review and cleanup

**Deliverable**: Production-ready analytical gradients, 1.8-2x validated speedup

---

## Expected Performance Results

### Current (Autograd)
- Single molecule: 16 ms/step
- Batch size 16: 1.78 ms/molecule

### After Analytical Gradients
- Single molecule: **8-9 ms/step** (1.8-2x faster)
- Batch size 16: **~1 ms/molecule** (1.8x improvement on top of batching)

### Combined Speedup
- Single trajectory: 1.8-2x faster
- Batched (size 16): 8.82x × 1.8 = **~16x total speedup**!

---

## Technical Challenges

### Challenge 1: Gradient Accumulation

**Problem**: Forces must accumulate contributions from all edges

**Solution**: Use scatter_add for efficient accumulation
```python
forces = torch.zeros(n_atoms, 3, device=device)
forces.scatter_add_(
    0,
    edge_index[0].unsqueeze(-1).expand(-1, 3),
    edge_force_contributions
)
```

---

### Challenge 2: Chain Rule Through Layers

**Problem**: Must backpropagate through 3 interaction blocks

**Solution**: Cache all intermediate activations during forward pass
```python
# Forward: cache everything
intermediates = []
for layer in self.layers:
    h, cache = layer.forward_with_cache(h, edges)
    intermediates.append(cache)

# Backward: chain gradients
grad_h = initial_gradient
for layer, cache in reversed(zip(self.layers, intermediates)):
    grad_h, grad_forces = layer.backward_analytical(grad_h, cache)
    total_forces += grad_forces
```

---

### Challenge 3: Numerical Stability

**Problem**: Small distances cause division by zero

**Solution**: Clamp distances and use stable formulas
```python
# Avoid r → 0
distances = torch.clamp(distances, min=1e-6)

# Stable unit vector
d_ij = edge_vec / distances.unsqueeze(-1).clamp(min=1e-6)
```

---

## Validation Criteria

### Accuracy Requirements

| Metric | Target | Why |
|--------|--------|-----|
| Force MAE | < 1e-6 eV/Å | Numerical precision |
| Force max error | < 1e-5 eV/Å | Edge case stability |
| Energy error | < 1e-8 eV | Conservation check |

### Performance Requirements

| Metric | Target | Why |
|--------|--------|-----|
| Single molecule | 8-9 ms | 1.8-2x speedup |
| Speedup consistency | ±10% | Across molecule sizes |
| Memory overhead | < 2x | Caching cost |

### MD Validation

| Test | Target | Why |
|------|--------|-----|
| NVE energy drift | < 0.1%/ns | Stability |
| Force correlation | > 0.9999 | Accuracy |
| Trajectory match | RMSD < 0.01 Å | Consistency |

---

## Code Structure

### New Files

1. **`src/mlff_distiller/models/analytical_gradients.py`**
   - Core analytical gradient functions
   - RBF, cutoff, message, update gradients
   - ~500 lines

2. **`src/mlff_distiller/models/student_model_analytical.py`**
   - StudentForceField with analytical forces
   - Backward methods for each layer
   - ~800 lines

3. **`tests/unit/test_analytical_gradients.py`**
   - Unit tests for each gradient function
   - Finite difference validation
   - ~300 lines

4. **`scripts/benchmark_analytical_vs_autograd.py`**
   - Performance comparison
   - Accuracy validation
   - ~400 lines

---

## Risk Assessment

### Low Risk ✅
- Mathematical foundation complete (Week 1)
- Formulas validated and documented
- Clear implementation path

### Medium Risk ⚠️
- Numerical stability for small distances
- Memory overhead from caching
- Debugging complex gradient chains

### High Risk ❌
- None! (Math is correct, just need implementation)

---

## Fallback Plan

If analytical gradients prove too complex or slow:

### Option A: Hybrid Approach
- Analytical for RBF/cutoff (easy parts)
- Autograd for message passing (complex parts)
- Expected: 1.3-1.5x speedup

### Option B: torch.func.jacrev
- Use PyTorch's functional API for efficient Jacobians
- May be faster than manual autograd
- Expected: 1.2-1.4x speedup

---

## Success Criteria

**Minimum Viable**:
- ✅ Analytical gradients implemented
- ✅ Force accuracy: MAE < 1e-6 eV/Å
- ✅ Speedup: ≥ 1.5x for single molecules

**Target**:
- ✅ Speedup: 1.8-2x for single molecules
- ✅ MD validation: Energy conservation < 0.1%
- ✅ Production-ready code with tests

**Stretch**:
- ✅ Speedup: > 2x for single molecules
- ✅ Combined with batching: > 15x total
- ✅ Numerical stability for all edge cases

---

## Next Steps

Ready to start implementation! I'll begin with:

**Day 1 - RBF Analytical Gradients**:
1. Create `src/mlff_distiller/models/analytical_gradients.py`
2. Implement `compute_rbf_gradient_analytical()`
3. Implement `compute_cutoff_gradient_analytical()`
4. Write unit tests with finite difference validation
5. Benchmark: Measure partial speedup

Shall I proceed with Day 1 implementation?
