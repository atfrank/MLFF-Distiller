# Analytical Force Computation Design

## Problem Statement

**Current Bottleneck**: Force computation via autograd is 2.3x slower than energy-only computation
- Energy-only: ~3 ms
- Energy+Forces (autograd): ~7 ms
- Autograd overhead: ~4 ms (55% of total time)

**Goal**: Eliminate autograd overhead by computing forces analytically during forward pass
- Target: 1.8-2.3x speedup (reduce 7 ms → 3-4 ms)
- Forces must match autograd within 1e-4 eV/Å

## PaiNN Architecture Review

### Forward Pass Structure

```
Input: atomic_numbers, positions
  ↓
1. Embedding: Z → s⁰  (scalar features)
  ↓
2. Edge Features:
   - Neighbor list: r_ij = r_i - r_j
   - Distances: d_ij = ||r_ij||
   - Directions: r̂_ij = r_ij / d_ij
   - RBF: φ_k(d_ij) = exp(-γ_k(d_ij - μ_k)²)
   - Cutoff: f_c(d_ij) = 0.5(cos(πd_ij/r_c) + 1)
  ↓
3. Message Passing (×3 blocks):
   a) Message:
      - m_s = Σ_j s_j * W_s(φ(d_ij))
      - m_v = Σ_j [v_j * W_v1(φ(d_ij)) + r̂_ij * W_v2(φ(d_ij))]
   b) Update:
      - s' = s + MLP(s, ||v||)
      - v' = v * gate1 + (U @ v) * gate2
  ↓
4. Energy Readout:
   - ε_i = MLP(s_i)  (per-atom energy)
   - E = Σ_i ε_i      (total energy)
```

### Force Computation (Chain Rule)

Forces are negative energy gradients:
```
F_i = -∂E/∂r_i
```

Using chain rule:
```
F_i = -Σ_α (∂E/∂x_α) * (∂x_α/∂r_i)
```

Where x_α includes all intermediate activations that depend on r_i.

## Analytical Gradient Derivation

### Key Dependencies on Positions

Positions r_i affect energy through:
1. **Edge vectors**: r_ij = r_i - r_j
2. **Distances**: d_ij = ||r_ij||
3. **Directions**: r̂_ij = r_ij / d_ij
4. **RBF features**: φ_k(d_ij)
5. **Vector messages**: m_v (through r̂_ij)

### Gradients We Need

#### 1. Distance Gradient
```
∂d_ij/∂r_i = r̂_ij
∂d_ij/∂r_j = -r̂_ij
```

#### 2. Direction Gradient
```
∂r̂_ij/∂r_i = (I - r̂_ij ⊗ r̂_ij) / d_ij
∂r̂_ij/∂r_j = -(I - r̂_ij ⊗ r̂_ij) / d_ij
```

#### 3. RBF Gradient
```
∂φ_k(d_ij)/∂d_ij = -2γ_k(d_ij - μ_k) * φ_k(d_ij)
```

#### 4. Cutoff Gradient
```
∂f_c(d_ij)/∂d_ij = -0.5 * (π/r_c) * sin(πd_ij/r_c)
```

#### 5. Modulated RBF Gradient
```
ψ_k = φ_k * f_c
∂ψ_k/∂d_ij = φ_k * ∂f_c/∂d_ij + f_c * ∂φ_k/∂d_ij
```

### Backward Pass Through Network

Working backwards from energy to positions:

#### Step 1: Energy → Per-Atom Energies
```
∂E/∂ε_i = 1  (for all i, since E = Σ ε_i)
```

#### Step 2: Per-Atom Energies → Scalar Features
```
∂E/∂s_i = ∂ε_i/∂s_i = MLP_grad(s_i)
```
This is just the gradient through the energy readout MLP.

#### Step 3: Scalar Features → Messages (Recursive)
For each interaction block (working backwards):
```
∂E/∂m_s_ij = ∂E/∂s_i  (scalar message contribution)
∂E/∂m_v_ij = ∂E/∂v_i  (vector message contribution)
```

#### Step 4: Messages → RBF Features
```
∂E/∂ψ_k(ij) = Σ_l [W_s[k,l] * ∂E/∂m_s_ij[l] + W_v1[k,l] * ∂E/∂m_v_ij[l]]
```

#### Step 5: RBF Features → Distances
```
∂E/∂d_ij = Σ_k (∂E/∂ψ_k) * (∂ψ_k/∂d_ij)
```

#### Step 6: Distances → Positions
```
∂E/∂r_i = Σ_j (∂E/∂d_ij) * (∂d_ij/∂r_i) = Σ_j (∂E/∂d_ij) * r̂_ij
```

Plus contribution from directional terms (vector messages).

## Implementation Strategy

### Approach 1: Store Intermediate Activations (RECOMMENDED)

Modify forward pass to save all intermediate values needed for gradients:
```python
def forward_with_forces(self, atomic_numbers, positions, ...):
    # Forward pass (unchanged)
    energy = self.forward(...)

    # Store intermediate activations in a cache
    cache = {
        'edge_index': edge_index,
        'edge_vector': edge_vector,
        'edge_distance': edge_distance,
        'edge_direction': edge_direction,
        'edge_rbf': edge_rbf,
        'cutoff_values': cutoff_values,
        'scalar_features_per_layer': [...],
        'vector_features_per_layer': [...],
        'filter_weights_per_layer': [...]
    }

    # Compute forces analytically
    forces = self._compute_forces_analytical(cache, positions)

    return energy, forces
```

### Approach 2: Dual Forward-Backward (Alternative)

Compute energy forward, then manually backward:
```python
def _compute_forces_analytical(self, cache, positions):
    n_atoms = positions.shape[0]
    forces = torch.zeros_like(positions)

    # Gradient of energy w.r.t. per-atom energies (all 1s)
    grad_atomic_energies = torch.ones(n_atoms, 1)

    # Backward through energy readout MLP
    grad_scalar = self._backward_energy_mlp(
        cache['scalar_features_final'],
        grad_atomic_energies
    )

    # Backward through each interaction block (reverse order)
    for layer_idx in reversed(range(self.num_interactions)):
        grad_scalar, grad_vector = self._backward_interaction(
            layer_idx, cache, grad_scalar, grad_vector
        )

    # Convert scalar/vector gradients to position gradients
    forces = self._accumulate_force_contributions(
        cache, grad_scalar, grad_vector
    )

    return -forces  # Negative gradient
```

## Expected Performance Impact

### Timing Breakdown (Current)
- Energy forward pass: 4.7 ms
- Autograd backward pass: 6.0 ms
- **Total: 10.7 ms**

### Timing Breakdown (Analytical)
- Energy forward pass: 4.7 ms
- Analytical force computation: 2-3 ms (estimated)
  - Gradient through MLPs: ~0.5 ms
  - Gradient accumulation: ~0.5 ms
  - Distance gradients: ~0.5 ms
  - Force accumulation: ~0.5 ms
- **Total: 6.7-7.7 ms**

### Expected Speedup
- Baseline (autograd): 10.7 ms
- Analytical: 7.0 ms
- **Speedup: 1.5-1.8x**

This matches our profiling prediction of 1.8x speedup!

## Numerical Validation

### Test Cases
1. **Gradient Check**: Compare analytical forces vs autograd for random configurations
2. **Tolerance**: Maximum error < 1e-4 eV/Å (typical MD tolerance)
3. **Systematic Tests**:
   - Single atom (trivial, forces = 0)
   - Diatomic molecule (simple case)
   - Water molecule (3 atoms, different species)
   - Benzene (12 atoms, larger system)
   - Batch of molecules (ensure independence)

### Validation Script
```python
def validate_analytical_forces(model, test_molecules, tol=1e-4):
    for mol in test_molecules:
        # Autograd forces (ground truth)
        energy_auto, forces_auto = model.predict_energy_and_forces(...)

        # Analytical forces (our implementation)
        energy_analytical, forces_analytical = model.forward_with_forces(...)

        # Compare
        max_error = torch.max(torch.abs(forces_auto - forces_analytical))
        assert max_error < tol, f"Force error {max_error:.2e} exceeds tolerance {tol}"

        print(f"✓ {mol.name}: max error = {max_error:.2e} eV/Å")
```

## Implementation Phases

### Phase 1: Forward Pass Cache (1-2 hours)
- Modify `forward()` to optionally save intermediate activations
- Add `_save_for_backward()` method
- Test: Ensure forward pass still matches

### Phase 2: Backward Through MLPs (1-2 hours)
- Implement MLP gradient computation
- Implement update block backward pass
- Test: Gradient through MLPs matches autograd

### Phase 3: Backward Through Messages (2-3 hours)
- Implement message passing backward
- Handle scalar and vector message gradients
- Test: Message gradients match autograd

### Phase 4: Position Gradients (2-3 hours)
- Implement distance gradient accumulation
- Implement direction gradient contributions
- Accumulate all force contributions
- Test: Full forces match autograd

### Phase 5: Optimization (1-2 hours)
- Fuse gradient operations
- Optimize memory layout
- Add optional caching for static geometries

**Total Time Estimate: 8-12 hours**

## Alternative: Use torch.func (Experimental)

PyTorch 2.0 has `torch.func.jacrev()` for efficient Jacobian computation:
```python
from torch.func import jacrev

def forward_for_forces(positions):
    return model(atomic_numbers, positions)

# Compute Jacobian (forces)
forces_jacobian = jacrev(forward_for_forces)(positions)
forces = -forces_jacobian.sum(dim=0)  # Sum over output dim
```

This might be faster than manual implementation, worth testing!

## Risk Mitigation

### Numerical Stability
- Use `eps=1e-8` for division by distances
- Careful handling of zero-distance edges (shouldn't exist with cutoff)
- Use `torch.clamp()` for gradients if needed

### Correctness
- Extensive testing against autograd
- Test with different molecule sizes, species, geometries
- Test edge cases (linear molecules, planar, etc.)

### Performance
- Profile each backward component
- If analytical forces aren't faster, investigate why
- May need custom CUDA kernels for bottleneck operations

## Success Criteria

1. ✓ Analytical forces match autograd within 1e-4 eV/Å
2. ✓ Speedup of 1.5x minimum, 2.0x target
3. ✓ No numerical instability in MD simulations
4. ✓ Clean, maintainable code with good documentation
5. ✓ Comprehensive test coverage

## Next Steps

1. Implement Phase 1 (forward pass cache)
2. Validate cache correctness
3. Implement Phase 2-4 (backward passes)
4. Validate forces against autograd
5. Benchmark and optimize
6. Integrate into ASE calculator
7. Run MD simulation stress test
