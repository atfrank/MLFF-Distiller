# Phase 1 Optimization Specification

**Goal**: Achieve 2-3x inference speedup through torch.compile() + FP16 mixed precision
**Timeline**: 3 days (November 24-26, 2025)
**Risk**: Low (software-only optimizations, fully reversible)

---

## Executive Summary

Phase 1 focuses on **low-hanging fruit** optimizations that require minimal code changes and provide substantial speedup with negligible risk:

1. **torch.compile()**: Graph-level optimizations (1.3-1.5x speedup)
2. **FP16 Mixed Precision**: Reduced precision inference (1.5-2x speedup)

**Combined Expected Speedup**: 2-3x
**Development Time**: 3 days
**Risk Level**: Low
**Reversibility**: Full (simple parameter flags)

---

## Optimization 1: torch.compile()

### Overview

**torch.compile()** is a PyTorch 2.x feature that applies graph-level optimizations to neural networks without requiring code changes. It works by:
1. Tracing the computational graph
2. Fusing operations
3. Optimizing memory access patterns
4. Reducing Python overhead

### Technical Details

**Requirements**:
- Python 3.12+
- PyTorch 2.x
- CUDA 12.x

**Implementation Approach**:
```python
# Simple wrapper in ASE Calculator
self.model = torch.compile(
    self.model,
    mode='reduce-overhead',  # Options: 'default', 'reduce-overhead', 'max-autotune'
    fullgraph=True,          # Compile entire graph
    disable=False            # Enable compilation
)
```

**Compile Modes**:
- `default`: Balanced optimization (30s compile time)
- `reduce-overhead`: Optimize for low latency (60s compile time)
- `max-autotune`: Maximum performance (5-10 min compile time)

**Recommendation**: Use `reduce-overhead` for best balance

---

### Performance Expectations

**Speedup**:
- Single-structure inference: 1.3-1.5x
- Batch inference: 1.4-1.6x (better with larger batches)
- Compilation overhead: 30-60s (one-time cost)

**Memory**:
- No change in memory usage
- Slight increase during compilation

**Accuracy**:
- Identical to baseline (numerical precision preserved)

---

### Implementation Plan

**File**: `src/mlff_distiller/inference/ase_calculator.py`

**Step 1**: Add Parameters
```python
def __init__(
    self,
    checkpoint_path: Union[str, Path],
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    use_compile: bool = True,           # NEW
    compile_mode: str = 'reduce-overhead',  # NEW
    use_fp16: bool = False,
    logger: Optional[logging.Logger] = None,
):
```

**Step 2**: Apply Compilation
```python
# After model loading
if self.use_compile:
    try:
        if not hasattr(torch, 'compile'):
            self.logger.warning("torch.compile() not available (requires PyTorch 2.x)")
            self.use_compile = False
        else:
            self.logger.info(f"Compiling model with mode={compile_mode}...")
            self.model = torch.compile(
                self.model,
                mode=compile_mode,
                fullgraph=True,
                disable=False
            )
            self.logger.info(f"✓ Model compiled successfully")
    except Exception as e:
        self.logger.warning(f"torch.compile() failed: {e}")
        self.logger.warning("Falling back to eager mode")
        self.use_compile = False
```

**Step 3**: Add Warm-up
```python
def _warmup_compiled_model(self):
    """Warm up compiled model to trigger compilation."""
    if not self.use_compile:
        return

    self.logger.info("Warming up compiled model (this may take 30-60s)...")
    dummy_atoms = ase.Atoms('H2', positions=[[0,0,0], [0,0,1]])
    dummy_atoms.calc = self

    try:
        _ = dummy_atoms.get_potential_energy()
        self.logger.info("✓ Model compilation complete")
    except Exception as e:
        self.logger.error(f"Warm-up failed: {e}")
        raise
```

---

### Testing Strategy

**Correctness Tests**:
```bash
# Test that results match baseline
pytest tests/integration/test_ase_calculator.py -v -k "test_energy_forces"

# Test with different compile modes
python -c "
from mlff_distiller.inference import StudentForceFieldCalculator
from ase.build import molecule

for mode in ['default', 'reduce-overhead', 'max-autotune']:
    calc = StudentForceFieldCalculator(
        checkpoint_path='checkpoints/best_model.pt',
        device='cuda',
        use_compile=True,
        compile_mode=mode
    )
    atoms = molecule('H2O')
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    print(f'{mode}: {energy:.6f} eV')
"
```

**Performance Tests**:
```bash
# Benchmark speedup
python scripts/benchmark_inference.py \
    --use-compile \
    --compile-mode reduce-overhead \
    --n-structures 1000 \
    --output benchmarks/with_compile/
```

---

### Known Limitations

1. **First-run overhead**: 30-60s compilation time on first inference
2. **Dynamic shapes**: Recompilation if input shapes change significantly
3. **Memory spike**: Temporary memory increase during compilation
4. **PyTorch version**: Requires PyTorch 2.x

**Mitigation**:
- Warm up model after initialization
- Cache compiled models
- Use consistent batch sizes
- Document PyTorch version requirement

---

## Optimization 2: FP16 Mixed Precision

### Overview

**FP16 mixed precision** reduces compute and memory requirements by using 16-bit floating point for most operations while keeping critical operations in 32-bit. This provides:
- 1.5-2x speedup on modern GPUs (Ampere, Ada)
- 50% memory reduction
- <1% accuracy loss (typically)

### Technical Details

**Approach**: **Autocast-only** (recommended)
- Use `torch.cuda.amp.autocast()` context manager
- Keep model weights in FP32
- Automatically cast operations to FP16
- Gradient computation stays in FP32

**Why Autocast-only?**
- Simpler implementation (no explicit model.half())
- Better numerical stability
- Easier to debug
- More flexible (can disable easily)

---

### Implementation Plan

**File**: `src/mlff_distiller/inference/ase_calculator.py`

**Step 1**: Update Forward Pass
```python
def _batch_forward(self, batch_data):
    """Single forward pass for entire batch."""

    # Apply autocast for FP16
    if self.use_fp16 and torch.cuda.is_available():
        with torch.cuda.amp.autocast():
            energies = self.model(
                batch_data['atomic_numbers'],
                batch_data['positions'],
                cell=None,
                pbc=None,
                batch=batch_data['batch']
            )
    else:
        energies = self.model(
            batch_data['atomic_numbers'],
            batch_data['positions'],
            cell=None,
            pbc=None,
            batch=batch_data['batch']
        )

    # Forces computation (autocast automatically converts back to FP32)
    forces = -torch.autograd.grad(
        energies.sum(),
        batch_data['positions'],
        create_graph=False,
        retain_graph=False
    )[0]

    return {
        'energies': energies,
        'forces': forces,
        'batch': batch_data['batch'],
    }
```

**Step 2**: No Model Conversion Needed!
```python
# DON'T DO THIS (old approach):
# if self.use_fp16:
#     self.model = self.model.half()  # BAD: causes type errors

# DO THIS INSTEAD (autocast-only):
# Just use autocast context manager in forward pass
# Model stays in FP32, operations auto-cast to FP16
```

---

### Performance Expectations

**Speedup**:
- Single-structure inference: 1.5-1.8x
- Batch inference: 1.8-2.0x (better with larger batches)
- No overhead (instant)

**Memory**:
- 30-50% reduction in activation memory
- Model weights stay in FP32 (no change)

**Accuracy**:
- Energy MAE: <0.1% increase
- Force RMSE: <1% increase
- Typically negligible for MD simulations

---

### Testing Strategy

**Accuracy Tests**:
```bash
# Compare to baseline
python scripts/validate_student_on_test_molecule.py \
    --use-fp16 \
    --compare-to-baseline \
    --output validation_results/fp16_accuracy/
```

**Performance Tests**:
```bash
# Benchmark speedup
python scripts/benchmark_inference.py \
    --use-fp16 \
    --n-structures 1000 \
    --output benchmarks/with_fp16/
```

**Combined Test**:
```bash
# Test torch.compile() + FP16
python scripts/benchmark_inference.py \
    --use-compile \
    --use-fp16 \
    --n-structures 1000 \
    --output benchmarks/combined/
```

---

### Accuracy Validation

**Acceptance Criteria**:
- Energy MAE vs baseline: <0.5 meV/atom
- Force RMSE vs baseline: <0.01 eV/Å
- Correlation coefficient: >0.999

**Test Script**: `scripts/validate_fp16_accuracy.py`
```python
import torch
from mlff_distiller.inference import StudentForceFieldCalculator
from ase.build import bulk
import numpy as np

# Test structures
structures = [
    bulk('Si', 'diamond', a=5.43),
    bulk('NaCl', 'rocksalt', a=5.64),
    # ... more structures
]

# Baseline (FP32)
calc_fp32 = StudentForceFieldCalculator(
    checkpoint_path='checkpoints/best_model.pt',
    device='cuda',
    use_fp16=False
)

# FP16
calc_fp16 = StudentForceFieldCalculator(
    checkpoint_path='checkpoints/best_model.pt',
    device='cuda',
    use_fp16=True
)

# Compare
energy_errors = []
force_errors = []

for atoms in structures:
    # FP32
    atoms.calc = calc_fp32
    e_fp32 = atoms.get_potential_energy()
    f_fp32 = atoms.get_forces()

    # FP16
    atoms.calc = calc_fp16
    e_fp16 = atoms.get_potential_energy()
    f_fp16 = atoms.get_forces()

    # Errors
    energy_errors.append(abs(e_fp16 - e_fp32) / len(atoms))
    force_errors.append(np.sqrt(((f_fp16 - f_fp32)**2).mean()))

print(f"Energy MAE: {np.mean(energy_errors)*1000:.3f} meV/atom")
print(f"Force RMSE: {np.mean(force_errors):.4f} eV/Å")
```

---

### Known Limitations

1. **Slightly reduced accuracy**: <1% typical, acceptable for MD
2. **GPU-only**: FP16 only beneficial on CUDA devices
3. **Numerical instability**: Rare, but possible for extreme values

**Mitigation**:
- Test accuracy on validation set
- Keep option to disable FP16
- Monitor for NaN/Inf during inference
- Document accuracy trade-offs

---

## Combined Optimization

### torch.compile() + FP16

**Expected Speedup**: 2-3x (multiplicative)

**Implementation**:
```python
calc = StudentForceFieldCalculator(
    checkpoint_path='checkpoints/best_model.pt',
    device='cuda',
    use_compile=True,
    compile_mode='reduce-overhead',
    use_fp16=True
)
```

**Key Considerations**:
1. Apply torch.compile() **first**, then use FP16
2. Warm up compiled model before benchmarking
3. Test accuracy with both optimizations
4. Monitor memory usage

---

### Performance Expectations

**Single-structure Inference**:
```
Baseline:              100 ms/structure
+ torch.compile():     70 ms/structure  (1.4x)
+ FP16:                56 ms/structure  (1.8x)
+ both:                40 ms/structure  (2.5x)
```

**Batch-16 Inference**:
```
Baseline:              6.25 ms/structure
+ torch.compile():     4.5 ms/structure  (1.4x)
+ FP16:                3.5 ms/structure  (1.8x)
+ both:                2.5 ms/structure  (2.5x)
+ batching advantage:  16x vs single-structure
= Total speedup:       40x vs baseline single-structure
```

---

## Implementation Timeline

### Day 1: Environment + Validation
- [x] Python 3.12 environment setup (2 hours)
- [ ] Quick MD validation (100ps NVE) (1.5 hours)
- [ ] Documentation (1 hour)

### Day 2: torch.compile() + FP16
- [ ] torch.compile() implementation (3 hours)
- [ ] FP16 implementation (2.5 hours)
- [ ] Initial testing (2 hours)

### Day 3: Testing + Reporting
- [ ] Combined optimization testing (2 hours)
- [ ] Comprehensive benchmarking (3 hours)
- [ ] Phase 1 completion report (2 hours)

---

## Testing & Validation

### Correctness Tests
```bash
# Run full integration test suite
pytest tests/integration/ -v

# Expected: All 21 tests passing
```

### Performance Benchmarks
```bash
# Baseline
python scripts/benchmark_inference.py \
    --output benchmarks/baseline/

# torch.compile()
python scripts/benchmark_inference.py \
    --use-compile \
    --output benchmarks/with_compile/

# FP16
python scripts/benchmark_inference.py \
    --use-fp16 \
    --output benchmarks/with_fp16/

# Combined
python scripts/benchmark_inference.py \
    --use-compile \
    --use-fp16 \
    --output benchmarks/combined/
```

### Accuracy Validation
```bash
# Test accuracy on validation set
python scripts/validate_fp16_accuracy.py \
    --checkpoint checkpoints/best_model.pt \
    --output validation_results/fp16_accuracy/
```

---

## Success Criteria

Phase 1 is successful when:
- [ ] torch.compile() implemented and tested
- [ ] FP16 implemented and tested
- [ ] Combined speedup 2-3x measured
- [ ] Accuracy degradation <1%
- [ ] All integration tests passing
- [ ] Documentation complete
- [ ] Phase 1 report generated

---

## Risk Assessment

### Risk 1: torch.compile() Overhead
**Probability**: Low
**Impact**: Low
**Mitigation**: Warm-up model after initialization, one-time cost

### Risk 2: FP16 Accuracy Loss
**Probability**: Low
**Impact**: Medium
**Mitigation**: Measure accuracy, revert if >1% degradation

### Risk 3: Compatibility Issues
**Probability**: Low
**Impact**: Medium
**Mitigation**: Comprehensive testing, fallback to baseline

---

## Next Steps (Phase 2)

After Phase 1 completion, proceed to custom CUDA optimizations:
1. Custom CUDA kernels for force computation
2. Memory-efficient batching strategies
3. JIT-compiled distance calculations
4. Advanced graph optimizations
5. Multi-GPU support

**Target**: Additional 2-3x speedup → 10x total vs original baseline

---

## References

- [PyTorch torch.compile() Documentation](https://pytorch.org/docs/stable/generated/torch.compile.html)
- [PyTorch AMP Documentation](https://pytorch.org/docs/stable/amp.html)
- [NVIDIA Mixed Precision Training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)
- [torch.compile() Tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)

---

## Contact

**Coordinator**: ml-distillation-coordinator
**Working Directory**: `/home/aaron/ATX/software/MLFF_Distiller`
**Environment**: `mlff-py312`
**Checkpoint**: `checkpoints/best_model.pt`
