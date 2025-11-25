# GitHub Issues for Week 1 - Phase 1 Optimizations

**Create Date**: November 24, 2025
**Milestone**: Phase 1 Optimizations
**Total Issues**: 3 (Issues #27, #28, #29)

---

## Issue #27: Enable torch.compile() for Inference Optimization

**Title**: `[Optimization] Enable torch.compile() for 1.3-1.5x inference speedup`

**Labels**: `enhancement`, `optimization`, `priority-high`
**Milestone**: Phase 1 Optimizations
**Assignee**: cuda-optimization-engineer
**Estimated Time**: 1 day

### Description

Enable PyTorch 2.x's `torch.compile()` feature to achieve 1.3-1.5x inference speedup through graph-level optimizations without requiring code changes to the model architecture.

**Context**:
- Python 3.12 environment created with PyTorch 2.9.1+cu128
- torch.compile() verified available
- Zero code changes to model architecture required
- Fully reversible (simple parameter flag)

**Expected Benefit**:
- Speedup: 1.3-1.5x for single-structure inference
- Compilation time: 30-60s one-time cost
- Accuracy: Identical to baseline (numerical precision preserved)
- Memory: No change

---

### Tasks

#### Implementation
- [ ] Add `use_compile` parameter to StudentForceFieldCalculator.__init__()
- [ ] Add `compile_mode` parameter (default: 'reduce-overhead')
- [ ] Implement torch.compile() wrapper in calculator initialization
- [ ] Add fallback for PyTorch versions without compile support
- [ ] Add warm-up method to trigger compilation
- [ ] Add logging for compilation status

#### Testing
- [ ] Run integration test suite (pytest tests/integration/)
- [ ] Verify numerical accuracy matches baseline
- [ ] Test different compile modes (default, reduce-overhead, max-autotune)
- [ ] Test with single-structure inference
- [ ] Test with batch inference
- [ ] Measure compilation overhead

#### Benchmarking
- [ ] Benchmark single-structure inference (1000 steps)
- [ ] Benchmark batch inference (sizes: 4, 8, 16)
- [ ] Compare to baseline performance
- [ ] Document speedup results
- [ ] Generate performance plots

#### Documentation
- [ ] Update ASE Calculator docstring
- [ ] Add usage examples
- [ ] Document compile modes
- [ ] Update README with torch.compile() usage
- [ ] Add troubleshooting guide

---

### Acceptance Criteria

- [ ] torch.compile() working in ASE Calculator
- [ ] All integration tests passing (21/21)
- [ ] Speedup measured: 1.3-1.5x for single-structure
- [ ] Speedup measured: 1.4-1.6x for batch-16
- [ ] Results match baseline (within numerical precision)
- [ ] Compilation overhead documented
- [ ] Usage documentation complete
- [ ] No regressions in existing functionality

---

### Implementation Details

**File**: `src/mlff_distiller/inference/ase_calculator.py`

**Changes**:
```python
def __init__(
    self,
    checkpoint_path: Union[str, Path],
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    use_compile: bool = True,           # NEW PARAMETER
    compile_mode: str = 'reduce-overhead',  # NEW PARAMETER
    use_fp16: bool = False,
    logger: Optional[logging.Logger] = None,
):
    # ... existing initialization ...

    # Apply torch.compile() if requested
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
                self.logger.info("✓ Model compiled successfully")
        except Exception as e:
            self.logger.warning(f"torch.compile() failed: {e}")
            self.use_compile = False
```

**Testing Command**:
```bash
# Correctness
pytest tests/integration/test_ase_calculator.py -v

# Benchmark
python scripts/benchmark_inference.py \
    --use-compile \
    --compile-mode reduce-overhead \
    --output benchmarks/with_compile/
```

---

### Related Documentation

- [Week 1 Coordination Plan](WEEK1_COORDINATION_PLAN.md)
- [Python 3.12 Setup Guide](docs/PYTHON312_SETUP_GUIDE.md)
- [Phase 1 Optimization Spec](docs/PHASE1_OPTIMIZATION_SPEC.md)

---

### Dependencies

- **Requires**: Python 3.12 environment (Issue: completed)
- **Blocks**: Combined optimization benchmarking
- **Related**: Issue #28 (FP16), Issue #29 (MD Validation)

---

### Timeline

**Day 2 Morning (November 25)**:
- 8:00-9:00 AM: Implementation
- 9:00-10:00 AM: Correctness testing
- 10:00-11:00 AM: Benchmarking
- 11:00-12:00 PM: Documentation

**Estimated Completion**: Day 2, 12:00 PM

---

## Issue #28: Implement FP16 Mixed Precision for Inference

**Title**: `[Optimization] Implement FP16 mixed precision for 1.5-2x speedup`

**Labels**: `enhancement`, `optimization`, `priority-high`
**Milestone**: Phase 1 Optimizations
**Assignee**: cuda-optimization-engineer
**Estimated Time**: 0.5 days

### Description

Implement FP16 mixed precision inference using autocast-only approach to achieve 1.5-2x speedup with <1% accuracy degradation.

**Context**:
- GPU: NVIDIA RTX 3080 Ti (Ampere architecture, excellent FP16 support)
- Approach: Autocast-only (no explicit model.half())
- Benefit: 1.5-2x speedup + 50% memory reduction
- Risk: Low (<1% typical accuracy loss)

**Key Decision**: Use autocast-only approach instead of explicit model.half() for better numerical stability and easier debugging.

---

### Tasks

#### Implementation
- [ ] Remove any explicit .half() conversions (if present)
- [ ] Implement autocast in _batch_forward() method
- [ ] Wrap forward pass with torch.cuda.amp.autocast()
- [ ] Verify gradient computation stays in FP32
- [ ] Add use_fp16 parameter handling
- [ ] Add logging for FP16 status

#### Testing
- [ ] Run integration test suite with FP16
- [ ] Verify no type errors occur
- [ ] Test with single-structure inference
- [ ] Test with batch inference
- [ ] Measure accuracy degradation vs baseline
- [ ] Check for NaN/Inf values

#### Accuracy Validation
- [ ] Compare energies to baseline (MAE <0.1%)
- [ ] Compare forces to baseline (RMSE <1%)
- [ ] Test on validation set (multiple structures)
- [ ] Generate accuracy comparison plots
- [ ] Document accuracy statistics

#### Benchmarking
- [ ] Benchmark single-structure inference
- [ ] Benchmark batch inference
- [ ] Measure memory usage reduction
- [ ] Compare to baseline and torch.compile()
- [ ] Test combined torch.compile() + FP16

#### Documentation
- [ ] Update ASE Calculator docstring
- [ ] Add usage examples
- [ ] Document accuracy trade-offs
- [ ] Add troubleshooting guide
- [ ] Update README

---

### Acceptance Criteria

- [ ] FP16 working without type errors
- [ ] All integration tests passing
- [ ] Accuracy degradation <1% (energy MAE, force RMSE)
- [ ] Speedup measured: 1.5-1.8x for single-structure
- [ ] Speedup measured: 1.8-2.0x for batch-16
- [ ] Memory reduction: 30-50%
- [ ] Combined with torch.compile(): 2-3x total speedup
- [ ] Documentation complete

---

### Implementation Details

**File**: `src/mlff_distiller/inference/ase_calculator.py`

**Changes to _batch_forward()**:
```python
def _batch_forward(self, batch_data):
    """Single forward pass for entire batch."""

    # Use autocast for FP16 (don't convert model to half!)
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

    # Forces computation (autocast automatically handles conversion)
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

**Testing Commands**:
```bash
# Accuracy validation
python scripts/validate_student_on_test_molecule.py \
    --use-fp16 \
    --compare-to-baseline \
    --output validation_results/fp16_accuracy/

# Benchmark
python scripts/benchmark_inference.py \
    --use-fp16 \
    --output benchmarks/with_fp16/

# Combined optimization
python scripts/benchmark_inference.py \
    --use-compile \
    --use-fp16 \
    --output benchmarks/combined/
```

---

### Related Documentation

- [Week 1 Coordination Plan](WEEK1_COORDINATION_PLAN.md)
- [Phase 1 Optimization Spec](docs/PHASE1_OPTIMIZATION_SPEC.md)

---

### Dependencies

- **Requires**: torch.compile() implementation (Issue #27)
- **Blocks**: Phase 1 completion report
- **Related**: Issue #27 (torch.compile()), Issue #29 (MD Validation)

---

### Timeline

**Day 2 Afternoon (November 25)**:
- 1:00-3:00 PM: Implementation
- 3:00-4:00 PM: Accuracy testing
- 4:00-5:00 PM: Benchmarking

**Estimated Completion**: Day 2, 5:00 PM

---

## Issue #29: Quick MD Stability Validation (100ps NVE)

**Title**: `[Validation] Run quick 100ps NVE validation to verify MD stability`

**Labels**: `validation`, `testing`, `priority-high`
**Milestone**: Phase 1 Optimizations
**Assignee**: testing-benchmark-engineer
**Estimated Time**: 0.5 days

### Description

Run a quick 100 picosecond NVE (constant energy) trajectory to verify that the student model is stable for molecular dynamics simulations before investing in heavy optimization work.

**Context**:
- Quick validation: ~30-45 min compute time
- Test system: Benzene (C6H6, 12 atoms)
- Pass criteria: Energy drift <1%
- Decision: Go/No-Go for Phase 1 optimizations

**Purpose**: Catch critical issues (energy drift, instabilities, atomic explosions) before spending 3 days on optimization.

---

### Tasks

#### Setup
- [x] Create validation script: `scripts/quick_md_validation.py`
- [x] Document validation procedure in quickstart guide
- [ ] Verify script is executable
- [ ] Check checkpoint path is correct

#### Execution
- [ ] Run 100ps NVE trajectory (benzene)
- [ ] Monitor progress (200,000 MD steps)
- [ ] Wait for completion (~30-45 minutes)
- [ ] Handle any runtime errors

#### Analysis
- [ ] Calculate energy drift (%)
- [ ] Calculate energy fluctuations (%)
- [ ] Analyze temperature stability
- [ ] Check for atomic explosions
- [ ] Generate analysis plots

#### Reporting
- [ ] Generate validation report
- [ ] Save trajectory data
- [ ] Document pass/fail result
- [ ] Make go/no-go decision
- [ ] Update project status

#### Decision Point
- [ ] If PASS: Proceed with torch.compile() implementation
- [ ] If FAIL: Investigate and fix model before optimization

---

### Acceptance Criteria

- [ ] Trajectory completes successfully (100ps)
- [ ] Energy drift <1% over 100ps
- [ ] Energy fluctuations <0.5% (std/mean)
- [ ] Temperature stable (±20K from target 300K)
- [ ] No atomic explosions or NaN values
- [ ] Validation report generated
- [ ] Plots saved (energy, temperature)
- [ ] Go/No-Go decision documented

---

### Validation Parameters

**System**:
- Molecule: Benzene (C6H6)
- Atoms: 12 (6 carbon + 6 hydrogen)
- Initial structure: ASE built-in molecule

**MD Settings**:
- Ensemble: NVE (microcanonical)
- Duration: 100 ps (100,000 fs)
- Timestep: 0.5 fs
- Total steps: 200,000
- Recording interval: Every 10 steps
- Initial temperature: 300 K

**Pass Criteria**:
```python
passed = (
    abs(energy_drift) < 1.0 and      # <1% drift
    energy_std < 0.5 and             # <0.5% fluctuations
    abs(temp_mean - 300) < 20        # within 20K
)
```

---

### Run Command

```bash
# Activate environment
conda activate mlff-py312

# Run validation
python scripts/quick_md_validation.py \
    --checkpoint checkpoints/best_model.pt \
    --timestep 0.5 \
    --duration 100 \
    --output validation_results/quick_nve

# Check results
cat validation_results/quick_nve/validation_report.txt
```

---

### Expected Output Files

```
validation_results/quick_nve/
├── validation_report.txt           # Summary: PASS/FAIL + statistics
├── quick_md_validation.png         # Plots: energy + temperature
└── trajectory_data.npz             # Raw data: times, energies, temps
```

---

### Related Documentation

- [Week 1 Coordination Plan](WEEK1_COORDINATION_PLAN.md)
- [MD Validation Quickstart](docs/MD_VALIDATION_QUICKSTART.md)

---

### Dependencies

- **Requires**: Python 3.12 environment (completed)
- **Blocks**: torch.compile() implementation (conditional on PASS)
- **Related**: All Phase 1 optimizations depend on this validation

---

### Timeline

**Day 1 Afternoon (November 24)**:
- 1:00-2:00 PM: Setup and launch
- 2:00-3:00 PM: Wait for completion
- 3:00-4:00 PM: Analysis and decision

**Estimated Completion**: Day 1, 4:00 PM

---

### Decision Tree

```
Run MD Validation
    ↓
Energy drift <1%?
    ↓
  YES ↓             NO ↓
    ↓               ↓
  PASS          FAIL
    ↓               ↓
Proceed with     Investigate
optimizations    and fix model
    ↓               ↓
Issue #27        Pause Phase 1
torch.compile()  until fixed
```

---

## Summary

### Issue Overview

| Issue | Title | Priority | Estimated Time | Dependencies |
|-------|-------|----------|----------------|--------------|
| #27 | torch.compile() | High | 1 day | Python 3.12 env ✓ |
| #28 | FP16 Mixed Precision | High | 0.5 days | Issue #27 |
| #29 | MD Validation | High | 0.5 days | Python 3.12 env ✓ |

### Critical Path

```
Day 1:
  Python 3.12 environment ✓ (completed)
  Documentation ✓ (completed)
  ↓
  Issue #29: MD Validation (afternoon)
  ↓
Day 2:
  Issue #27: torch.compile() (morning)
  ↓
  Issue #28: FP16 (afternoon)
  ↓
Day 3:
  Combined benchmarking
  Phase 1 completion report
```

### Total Expected Outcomes

**Performance**:
- torch.compile(): 1.3-1.5x speedup
- FP16: 1.5-2x speedup
- Combined: 2-3x speedup
- With batching: 32-48x total (vs single-structure baseline)

**Quality**:
- Accuracy: >99% (FP16 <1% degradation)
- Stability: Verified via MD validation
- Tests: All integration tests passing

**Timeline**:
- Day 1: Environment + validation
- Day 2: Implementation
- Day 3: Benchmarking + reporting

---

## GitHub Project Board

### Column: Backlog
*Empty - all Week 1 work planned*

### Column: In Progress
- Issue #29: MD Validation (starts Day 1 afternoon)

### Column: Review
*Empty - no PRs yet*

### Column: Done
- Python 3.12 environment setup ✓
- Coordination documentation ✓

---

## Labels to Create

```
enhancement     - New feature or optimization
optimization    - Performance improvement
validation      - Testing and validation work
priority-high   - Must complete for Week 1
priority-medium - Important but not blocking
priority-low    - Nice to have
cuda            - CUDA-related work
documentation   - Documentation improvements
testing         - Test infrastructure
```

---

## Milestone: Phase 1 Optimizations

**Description**: Software-only optimizations to achieve 2-3x inference speedup

**Due Date**: November 26, 2025 (end of Day 3)

**Goals**:
- [ ] torch.compile() implemented (1.3-1.5x)
- [ ] FP16 implemented (1.5-2x)
- [ ] Combined 2-3x speedup achieved
- [ ] MD stability validated
- [ ] All tests passing
- [ ] Documentation complete

---

**Document Created**: November 24, 2025
**Status**: Ready to create issues
**Next Action**: Create Issues #27, #28, #29 in GitHub
