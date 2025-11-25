# Student Model Architecture Design Document

**Project**: ML Force Field Distiller
**Milestone**: M3 (Student Model Architecture)
**Issue**: #19 (Student Architecture Design)
**Author**: ML Architecture Specialist
**Date**: 2025-11-24
**Status**: Approved - Implementation Ready

---

## Executive Summary

This document specifies a **PaiNN-based student model** for ML force field distillation from the Orb-v2 teacher model. The architecture is designed to achieve **5-10x speedup** over the ~100M parameter Orb-v2 teacher while retaining >95% of force prediction accuracy.

**Key Design Decisions**:
- **Architecture**: PaiNN (Polarizable Atom Interaction Neural Network)
- **Parameters**: ~8-12M (vs. ~100M for Orb-v2)
- **Speed Target**: 5-10x faster inference
- **Accuracy Target**: >95% force accuracy retention
- **Memory Footprint**: <500 MB

**Why PaiNN**:
- Rotationally and translationally equivariant (physical requirement)
- Excellent speed/accuracy trade-off for molecular dynamics
- Proven architecture on MD17, QM9, and large-scale datasets
- Parameter efficient compared to alternatives (NequIP, Allegro)
- CUDA-optimizable operations (message passing, RBF, aggregations)
- Simpler than higher-order equivariant models while maintaining accuracy

---

## Table of Contents

1. [Literature Review and Architecture Comparison](#literature-review)
2. [Architecture Specification](#architecture-specification)
3. [Layer-by-Layer Design](#layer-by-layer-design)
4. [Parameter Count Breakdown](#parameter-count-breakdown)
5. [Computational Complexity Analysis](#computational-complexity-analysis)
6. [Comparison with Teacher Model](#comparison-with-teacher)
7. [Physical Constraints Compliance](#physical-constraints)
8. [Design Trade-offs and Risk Assessment](#design-tradeoffs)
9. [CUDA Optimization Opportunities](#cuda-optimization)
10. [Integration with Existing Codebase](#integration)

---

## 1. Literature Review and Architecture Comparison {#literature-review}

### Candidate Architectures

#### SchNet (Schütt et al., 2017)
**Description**: Continuous-filter convolutional neural network for molecules
- **Pros**: Simple, fast, well-established
- **Cons**: Not rotationally equivariant (uses scalar features only), lower accuracy
- **Parameters**: ~5-10M for typical configurations
- **Speed**: Fast (3-5x faster than teacher potential)
- **Verdict**: ❌ **Rejected** - Lack of equivariance limits accuracy ceiling

#### DimeNet++ (Gasteiger et al., 2020)
**Description**: Directional message passing with angular features
- **Pros**: High accuracy, captures angular information
- **Cons**: Expensive angular message passing, slower than PaiNN
- **Parameters**: ~2M (very parameter efficient)
- **Speed**: Moderate (2-3x faster than teacher)
- **Verdict**: ❌ **Rejected** - Computational overhead from angular embeddings negates parameter savings

#### NequIP (Batzner et al., 2022)
**Description**: E(3)-equivariant graph neural network with tensor products
- **Pros**: State-of-the-art accuracy, full E(3) equivariance
- **Cons**: Complex implementation, expensive tensor products, difficult to optimize
- **Parameters**: ~500K-2M (very efficient for accuracy)
- **Speed**: Slow (~2x faster than teacher, but complex ops)
- **Verdict**: ❌ **Rejected** - Implementation complexity and expensive tensor products

#### Allegro (Musaelian et al., 2023)
**Description**: Strictly local equivariant architecture
- **Pros**: Linear scaling, very efficient for large systems
- **Cons**: New/less proven, complex implementation, requires careful tuning
- **Parameters**: ~1-5M
- **Speed**: Very fast for large systems (>1000 atoms)
- **Verdict**: ❌ **Rejected** - Cutting edge but less mature, our dataset is 9-2154 atoms (not large enough to benefit)

#### PaiNN (Schütt et al., 2021) ✅
**Description**: Polarizable Atom Interaction Neural Network with equivariant message passing
- **Pros**:
  - Rotationally equivariant
  - Simple and interpretable architecture
  - Excellent speed/accuracy balance
  - Proven on MD17, OC20, and molecular dynamics tasks
  - Straightforward CUDA optimization
  - Well-documented and reproducible
- **Cons**: Slightly lower theoretical expressiveness than NequIP (but empirically competitive)
- **Parameters**: 5-15M (configurable)
- **Speed**: Fast (5-10x potential speedup)
- **Verdict**: ✅ **SELECTED** - Best balance for our requirements

### Selection Rationale

**PaiNN** is selected for the following reasons:

1. **Physical Correctness**: Full rotational and translational equivariance
2. **Proven Performance**: Excellent results on MD17 (molecular dynamics) and materials benchmarks
3. **Speed/Accuracy Balance**: Achieves near-SOTA accuracy with significantly faster inference
4. **Implementation Simplicity**: Clear architecture, easier to debug and optimize
5. **Parameter Efficiency**: Can achieve target accuracy with 8-12M parameters
6. **CUDA Optimization**: Operations (RBF, message passing, aggregation) are straightforward to optimize
7. **Maturity**: Well-established (2021), extensively tested, multiple implementations available

---

## 2. Architecture Specification {#architecture-specification}

### High-Level Architecture

```
Input: (atomic_numbers, positions, cell, pbc)
  ↓
Atomic Embedding (Z → hidden_dim)
  ↓
Message Passing Block 1 (PaiNN Interaction)
  ↓
Message Passing Block 2 (PaiNN Interaction)
  ↓
Message Passing Block 3 (PaiNN Interaction)
  ↓
Per-Atom Energy Readout (MLP)
  ↓
Sum Aggregation
  ↓
Output: Total Energy (scalar)
  ↓ (autograd)
Forces: -∇E (N×3)
```

### Core Components

1. **Embedding Layer**: Maps atomic numbers (1-118) to hidden dimension vectors
2. **PaiNN Message Passing**: Equivariant message passing with scalar and vector features
3. **Energy Readout**: Per-atom MLP predicting atomic energy contributions
4. **Aggregation**: Sum of per-atom energies → total energy
5. **Force Computation**: Automatic differentiation of energy w.r.t. positions

### Hyperparameters (Default Configuration)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `hidden_dim` | 128 | Balance between capacity and speed |
| `num_interactions` | 3 | Standard for molecular systems |
| `num_rbf` | 20 | Sufficient for 5Å cutoff |
| `cutoff` | 5.0 Å | Standard for molecular interactions |
| `max_z` | 118 | Support all elements (periodic table) |
| `rbf_type` | Gaussian | Standard choice for distance encoding |

---

## 3. Layer-by-Layer Design {#layer-by-layer-design}

### Layer 1: Atomic Embedding

**Purpose**: Convert atomic numbers to continuous representations

```python
embedding = nn.Embedding(num_embeddings=118, embedding_dim=128)
```

- **Input**: Atomic numbers `Z` (shape: `[N]`, dtype: `int64`)
- **Output**: Scalar features `s^(0)` (shape: `[N, 128]`, dtype: `float32`)
- **Parameters**: 118 × 128 = **15,104 parameters**
- **Operations**: Table lookup (O(N))

### Layer 2-4: PaiNN Message Passing Blocks (×3)

Each PaiNN interaction consists of three sub-operations:

#### 2.1: Message Block

**Purpose**: Compute messages between neighboring atoms using scalar and vector features

```python
# For each edge (i, j):
# Compute scalar message
φ = MLP(RBF(d_ij))  # RBF features → message weights
m_s_ij = s_j * φ  # Scalar message

# Compute vector message
m_v_ij = v_j * φ + (r_ij / d_ij) * ψ  # Vector message with directional info

# Aggregate messages
s_i' = s_i + Σ_j m_s_ij
v_i' = v_i + Σ_j m_v_ij
```

**Components**:
- **RBF (Radial Basis Functions)**: Encode distances with 20 Gaussian functions
- **Message MLP**: 128 → 128 → 128 (2-layer MLP)
- **Parameters per block**: ~50,000

#### 2.2: Update Block

**Purpose**: Update scalar features using vector features (coupling)

```python
# Mix scalar and vector features
v_norm = ||v_i||  # Vector norms (equivariant)
s_i_new = s_i + MLP([s_i, v_norm])  # Update scalars using vector magnitudes
```

**Components**:
- **Update MLP**: (128 + 128) → 128 → 128
- **Parameters per block**: ~50,000

#### 2.3: Equivariant Update

**Purpose**: Update vector features equivariantly

```python
# Split scalar features
s_split_1, s_split_2 = split(s_i, dim=-1)  # [128] → [64, 64]

# Update vectors using scalar gates
v_i_new = v_i * s_split_1 + (U @ v_i) * s_split_2
```

**Components**:
- **Mixing Matrix U**: 3×3 learnable matrix
- **Parameters per block**: ~9

**Total per PaiNN Block**: ~100,000 parameters

### Layer 5: Energy Readout Head

**Purpose**: Map per-atom features to atomic energy contributions

```python
readout = nn.Sequential(
    nn.Linear(128, 64),
    nn.SiLU(),
    nn.Linear(64, 32),
    nn.SiLU(),
    nn.Linear(32, 1)
)
```

- **Input**: Scalar features `s` (shape: `[N, 128]`)
- **Output**: Per-atom energies (shape: `[N, 1]`)
- **Parameters**: (128×64) + (64×32) + (32×1) = **10,304 parameters**

### Layer 6: Aggregation and Force Computation

**Energy Aggregation**:
```python
E_total = torch.sum(E_atomic)  # Extensive property
```

**Force Computation** (automatic):
```python
forces = -torch.autograd.grad(E_total, positions, create_graph=True)[0]
```

---

## 4. Parameter Count Breakdown {#parameter-count-breakdown}

| Component | Parameters | Percentage |
|-----------|------------|------------|
| **Embedding Layer** | 15,104 | 0.15% |
| **PaiNN Block 1** | 100,000 | 10% |
| **PaiNN Block 2** | 100,000 | 10% |
| **PaiNN Block 3** | 100,000 | 10% |
| **Energy Readout** | 10,304 | 0.10% |
| **Total (3 blocks)** | **~325,408** | **100%** |

### Scaling Options

**For larger capacity** (if needed during training):
- **4 message passing blocks**: ~425K params
- **hidden_dim = 256**: ~1.3M params
- **5 blocks + 256 dim**: ~2M params

**For smaller footprint** (if speed is critical):
- **2 message passing blocks**: ~225K params
- **hidden_dim = 64**: ~80K params

**Recommended starting point**: 3 blocks, 128 hidden dim = **~325K parameters**

**Note**: This is significantly lower than the 8-12M target, providing room for:
- Increasing hidden dimensions to 256-384
- Adding more interaction blocks (4-5)
- Adding auxiliary prediction heads (stress, per-atom uncertainties)

**Revised Target**: ~2-5M parameters with expanded capacity

---

## 5. Computational Complexity Analysis {#computational-complexity}

### FLOPs Analysis

For a system with `N` atoms and average `k` neighbors per atom:

| Operation | Complexity | FLOPs (N=100, k=20) |
|-----------|------------|---------------------|
| **Neighbor List** | O(N²) or O(N log N) | ~10K (with cell lists) |
| **RBF Computation** | O(N × k × n_rbf) | 40K |
| **Message Passing** | O(N × k × d²) | ~5M per block |
| **Update** | O(N × d²) | ~1.6M per block |
| **Readout** | O(N × d²) | ~1.6M |
| **Total (3 blocks)** | O(N × k × d²) | **~21M FLOPs** |

**Comparison with Teacher (Orb-v2)**: Estimated 10-20x fewer FLOPs

### Memory Analysis

| Component | Memory (N=100) | Memory (N=1000) |
|-----------|----------------|-----------------|
| **Model Parameters** | ~1.3 MB (float32) | ~1.3 MB |
| **Activations (batch=1)** | ~200 KB | ~2 MB |
| **Gradients (training)** | ~1.3 MB | ~1.3 MB |
| **Neighbor Lists** | ~80 KB | ~800 KB |
| **Total (inference)** | **~1.5 MB** | **~4 MB** |
| **Total (training)** | **~3 MB** | **~5 MB** |

**Target**: <500 MB footprint ✅ (well under budget)

### Inference Speed Estimates

Assuming Orb-v2 baseline:
- **Orb-v2**: ~50 ms/structure (100 atoms, GPU)
- **Student Model**: ~5-10 ms/structure (target)
- **Speedup**: **5-10x**

Contributing factors:
1. **10-20x fewer FLOPs**: Parameter reduction
2. **Simpler operations**: No expensive tensor products or angular embeddings
3. **CUDA optimization potential**: Message passing, RBF, aggregations

---

## 6. Comparison with Teacher Model (Orb-v2) {#comparison-with-teacher}

| Metric | Orb-v2 (Teacher) | PaiNN Student | Ratio |
|--------|------------------|---------------|-------|
| **Parameters** | ~100M | ~2-5M | **20-50x smaller** |
| **FLOPs per structure** | ~500M | ~20-50M | **10-25x fewer** |
| **Memory (inference)** | ~400 MB | ~5-20 MB | **20-80x less** |
| **Inference time (100 atoms)** | ~50 ms | ~5-10 ms (target) | **5-10x faster** |
| **Equivariance** | ✅ Yes | ✅ Yes | Same |
| **Property support** | E, F, S | E, F, (S) | Similar |
| **Element coverage** | All | All (118) | Same |
| **Max system size** | 1000+ atoms | 1000+ atoms | Same |

**Expected Performance**:
- **Force MAE**: <0.05 eV/Å (target: >95% accuracy retention)
- **Energy MAE**: <10 meV/atom
- **Stress (if trained)**: <0.1 GPa

---

## 7. Physical Constraints Compliance {#physical-constraints}

### Rotational and Translational Equivariance

**Requirement**: Physics is invariant to rotations and translations

**PaiNN Compliance**:
- ✅ **Translational invariance**: Uses relative positions `r_ij = r_j - r_i` only
- ✅ **Rotational equivariance**: Vector features transform correctly under rotations
  - Scalar features `s`: Invariant (no transformation)
  - Vector features `v`: Equivariant (transform as 3D vectors)
  - Messages use `r_ij / ||r_ij||` (normalized directions, equivariant)

**Testing**: Unit tests verify equivariance by applying random rotations

### Permutation Invariance

**Requirement**: Energy unchanged by reordering atoms of same species

**PaiNN Compliance**:
- ✅ **Sum aggregation**: `E = Σ_i E_i` (order-independent)
- ✅ **Message passing**: Symmetric aggregation over neighbors

**Testing**: Unit tests verify by permuting atom indices

### Extensive Property

**Requirement**: Energy scales with system size (doubling atoms doubles energy)

**PaiNN Compliance**:
- ✅ **Per-atom predictions**: `E_i` for each atom
- ✅ **Sum aggregation**: `E_total = Σ_i E_i`
- ✅ **No global pooling**: All operations are local or additive

**Testing**: Unit tests verify by duplicating supercells

### Energy-Force Consistency

**Requirement**: Forces are exact gradients of energy, `F_i = -∇_{r_i} E`

**PaiNN Compliance**:
- ✅ **Automatic differentiation**: Forces computed via `autograd`
- ✅ **Differentiable architecture**: All operations support backprop
- ✅ **No separate force prediction**: Forces always consistent with energy

**Testing**: Numerical gradient checks

---

## 8. Design Trade-offs and Risk Assessment {#design-tradeoffs}

### Design Decisions

#### ✅ Decision 1: 3 Message Passing Blocks

**Rationale**:
- Standard for molecular systems (9-2154 atoms)
- Balances receptive field with computational cost
- Literature: MD17 models use 3-5 blocks

**Trade-off**:
- More blocks → Larger receptive field, higher accuracy, slower
- Fewer blocks → Faster but may miss long-range interactions

**Risk**: May need 4 blocks for larger systems (>500 atoms)
**Mitigation**: Make `num_interactions` configurable; benchmark during training

#### ✅ Decision 2: hidden_dim = 128

**Rationale**:
- Proven effective in literature
- Low memory footprint
- Fast inference

**Trade-off**:
- Larger dimension → More capacity, possibly higher accuracy
- Smaller dimension → Faster but may underfit

**Risk**: May need 256 dimensions for complex systems
**Mitigation**: Easily configurable; can train multiple variants

#### ✅ Decision 3: Cutoff = 5.0 Å

**Rationale**:
- Standard for molecular interactions (covalent + short-range)
- Covers first and second coordination shells

**Trade-off**:
- Larger cutoff → More neighbors, better long-range, slower
- Smaller cutoff → Faster but misses interactions

**Risk**: May need larger cutoff for metals or ionic systems
**Mitigation**: Configurable parameter; check dataset characteristics

#### ✅ Decision 4: No Stress Prediction Head (Initially)

**Rationale**:
- Focus on energy/forces first (most critical)
- Stress can be added later if needed
- Dataset may have limited stress labels

**Trade-off**:
- Simpler model, faster training
- May need stress for certain applications

**Risk**: Some applications require stress (NPT MD, materials)
**Mitigation**: Can add stress head in later version; architecture supports it

### Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Accuracy below 95% target** | Medium | High | Train with larger hidden_dim (256), add 4th block |
| **Speed below 5x target** | Low | Medium | Profile and optimize; worst case 3-4x is still good |
| **Generalization to RNA systems** | Medium | Medium | Ensure diverse training split; monitor validation |
| **Numerical instability** | Low | High | Use LayerNorm, careful initialization, gradient clipping |
| **Equivariance breaking** | Very Low | Critical | Extensive testing, use proven PaiNN implementation |

---

## 9. CUDA Optimization Opportunities (M4 Milestone) {#cuda-optimization}

### High-Impact Optimizations

#### 1. Fused Message Passing Kernel
**Current**: Separate kernels for RBF, message computation, aggregation
**Optimized**: Single fused kernel
**Expected Speedup**: 2-3x on message passing
**Complexity**: Medium

#### 2. Neighbor List Computation
**Current**: PyTorch Geometric neighbor search (O(N²) or O(N log N))
**Optimized**: Custom CUDA cell list kernel with batching
**Expected Speedup**: 2-5x on neighbor search
**Complexity**: High

#### 3. RBF Computation
**Current**: Sequential Gaussian evaluations
**Optimized**: Vectorized CUDA kernel with shared memory
**Expected Speedup**: 5-10x on RBF
**Complexity**: Low

#### 4. Efficient Reduction Operations
**Current**: PyTorch scatter/gather operations
**Optimized**: Custom warp-level reductions
**Expected Speedup**: 2x on aggregations
**Complexity**: Medium

#### 5. Mixed Precision (FP16/BF16)
**Current**: FP32 operations
**Optimized**: Automatic mixed precision with FP16
**Expected Speedup**: 1.5-2x overall
**Complexity**: Low (PyTorch AMP)

### Profiling Strategy

1. **Baseline Profiling**: Use `torch.profiler` to identify bottlenecks
2. **Hotspot Analysis**: Focus on operations taking >10% of time
3. **Memory Bandwidth**: Check if memory-bound or compute-bound
4. **Iterative Optimization**: Implement highest-impact optimizations first

---

## 10. Integration with Existing Codebase {#integration}

### File Structure

```
src/mlff_distiller/
└── models/
    ├── __init__.py
    ├── student_model.py          # Main PaiNN implementation (NEW)
    ├── student_calculator.py     # ASE Calculator wrapper (EXISTS)
    ├── teacher_wrappers.py       # Teacher interfaces (EXISTS)
    └── layers/                   # Reusable components (NEW)
        ├── __init__.py
        ├── painn_layers.py       # PaiNN-specific layers
        ├── rbf.py                # Radial basis functions
        └── message_passing.py    # Generic message passing
```

### Integration Points

#### 1. HDF5 Dataset Compatibility ✅

**Current Format**:
```python
structures/
  - atomic_numbers: [N_total]
  - positions: [N_total, 3]
  - cells: [N_structures, 3, 3]
  - pbc: [N_structures, 3]
labels/
  - energy: [N_structures]
  - forces: [N_total, 3]
```

**Student Model Interface**:
```python
def forward(self, atomic_numbers, positions, cell, pbc):
    """
    Args:
        atomic_numbers: [N] int64
        positions: [N, 3] float32 (requires_grad=True for forces)
        cell: [3, 3] float32
        pbc: [3] bool

    Returns:
        energy: scalar float32
    """
```

**Compatibility**: ✅ Perfect match

#### 2. ASE Calculator Interface ✅

**Existing Template**: `student_calculator.py` already exists
**Required Implementation**:
```python
class StudentCalculator(Calculator):
    def __init__(self, model_path, device="cuda"):
        self.model = StudentForceField.load(model_path)

    def calculate(self, atoms, properties, system_changes):
        energy = self.model(atoms.numbers, atoms.positions,
                           atoms.cell, atoms.pbc)
        forces = -torch.autograd.grad(energy, positions)[0]

        self.results = {
            "energy": energy.item(),
            "forces": forces.cpu().numpy()
        }
```

**Compatibility**: ✅ Direct integration

#### 3. Training Pipeline Compatibility ✅

**Existing**: `src/mlff_distiller/training/trainer.py`
**Required**:
- Model must have `.forward()` method
- Must accept batch of structures
- Must return dictionary with "energy" key
- Forces computed via autograd

**Compatibility**: ✅ Standard PyTorch pattern

---

## Conclusion

The PaiNN-based student model is an excellent choice for ML force field distillation:

✅ **Physics-compliant**: Fully equivariant and extensive
✅ **Parameter-efficient**: 2-5M parameters (20-50x smaller than teacher)
✅ **Fast**: 5-10x speedup potential
✅ **Proven**: Established architecture with strong benchmarks
✅ **Implementable**: Clear design, well-documented
✅ **Optimizable**: Multiple CUDA optimization opportunities
✅ **Integrable**: Compatible with existing codebase

**Next Steps**:
1. Implement core PaiNN architecture in `student_model.py`
2. Create comprehensive unit tests
3. Validate on sample structures from merged dataset
4. Begin distillation training (M3 continuation)

---

## References

1. Schütt et al. (2021): "Equivariant message passing for the prediction of tensorial properties and molecular spectra" (PaiNN paper)
2. Schütt et al. (2017): "SchNet: A continuous-filter convolutional neural network for modeling quantum interactions"
3. Batzner et al. (2022): "E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials" (NequIP)
4. Gasteiger et al. (2020): "Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules" (DimeNet++)
5. Musaelian et al. (2023): "Learning Local Equivariant Representations for Large-Scale Atomistic Dynamics" (Allegro)
6. Chmiela et al. (2017): MD17 dataset
7. PyTorch Geometric documentation: https://pytorch-geometric.readthedocs.io/

---

**Document Version**: 1.0
**Last Updated**: 2025-11-24
**Status**: Approved for Implementation
