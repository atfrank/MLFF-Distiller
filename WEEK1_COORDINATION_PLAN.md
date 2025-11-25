# Week 1 Coordination Plan - Phase 1 Optimizations

**Project**: ML Force Field Distillation
**Duration**: 3 Days (November 24-26, 2025)
**Goal**: Achieve 2-3x speedup through torch.compile() + FP16 mixed precision

---

## Executive Summary

**Objective**: Complete Phase 1 optimizations to achieve 2-3x inference speedup before investing in custom CUDA kernels.

**Key Decisions**:
- Python 3.12 environment (enables torch.compile())
- Quick MD validation (100ps NVE) before heavy optimization
- Conservative CUDA phasing (Phase 1 first, Phase 2 after validation)

**Success Criteria**:
- [ ] Python 3.12 environment operational
- [ ] Quick MD validation PASSED (<1% energy drift)
- [ ] torch.compile() working (1.3-1.5x speedup)
- [ ] FP16 working (1.5-2x speedup)
- [ ] Combined 2-3x speedup achieved
- [ ] All integration tests passing

---

## Day-by-Day Schedule

### Day 1 (November 24, 2025) - Environment & Validation

#### Morning Session (4 hours)
**8:00 - 10:00 AM**: Environment Setup
- [x] Create Python 3.12 conda environment
- [x] Install PyTorch 2.x with CUDA 12.1+
- [x] Install all dependencies (PyG, ASE, h5py, etc.)
- [x] Verify torch.compile() availability
- [x] Run test suite to verify environment

**Checkpoint 1**: `python -c "import torch; print(hasattr(torch, 'compile'))"` returns True

**10:00 - 12:00 PM**: Documentation & Planning
- [ ] Create Week 1 coordination docs (4 files)
- [ ] Create GitHub Issues #27-29
- [ ] Update project board
- [ ] Review baseline benchmarks

**Checkpoint 2**: All documentation files created, Issues assigned

#### Afternoon Session (4 hours)
**1:00 - 2:00 PM**: Quick MD Validation Setup
- [ ] Create `scripts/quick_md_validation.py`
- [ ] Set up small test system (benzene, 12 atoms)
- [ ] Configure 100ps NVE trajectory
- [ ] Add energy conservation analysis

**2:00 - 3:00 PM**: Run Validation (Parallel with Implementation)
- [ ] Launch 100ps NVE validation
- [ ] Monitor progress (should take ~30-45 minutes)

**3:00 - 4:00 PM**: Begin torch.compile() Implementation
- [ ] Add `use_compile` parameter to ASE Calculator
- [ ] Implement torch.compile() wrapper
- [ ] Add mode selection (reduce-overhead, max-autotune)
- [ ] Write basic tests

**4:00 - 5:00 PM**: Validation Analysis & Decision
- [ ] Analyze NVE validation results
- [ ] Check energy conservation (<1% drift target)
- [ ] Make go/no-go decision
- [ ] Document results

**Checkpoint 3**: MD validation PASSED, torch.compile() implementation started

#### Evening Check (5:00 PM)
- [x] Python 3.12 environment: OPERATIONAL
- [ ] MD validation: PASS/FAIL (pending)
- [ ] torch.compile() implementation: IN PROGRESS
- [ ] Documentation: COMPLETE

---

### Day 2 (November 25, 2025) - Phase 1 Implementation

#### Morning Session (4 hours)
**8:00 - 9:00 AM**: Complete torch.compile()
- [ ] Finish implementation from Day 1
- [ ] Add error handling and fallback
- [ ] Test with different compile modes
- [ ] Document usage

**9:00 - 10:00 AM**: Correctness Testing
- [ ] Run integration tests
- [ ] Compare results to baseline (numerical precision)
- [ ] Verify forces/energies match
- [ ] Test batch processing

**10:00 - 11:00 AM**: Performance Benchmarking
- [ ] Run single-structure inference (1000 steps)
- [ ] Run batch inference (16 structures)
- [ ] Measure speedup vs baseline
- [ ] Document results

**Checkpoint 4**: torch.compile() working correctly, 1.3-1.5x speedup measured

**11:00 - 12:00 PM**: Documentation & Code Review
- [ ] Update ASE Calculator docs
- [ ] Add usage examples
- [ ] Update benchmark reports
- [ ] Commit changes

#### Afternoon Session (4 hours)
**1:00 - 3:00 PM**: FP16 Implementation
- [ ] Remove explicit `.half()` conversions
- [ ] Implement autocast-only approach
- [ ] Wrap forward pass with `torch.cuda.amp.autocast()`
- [ ] Ensure gradient computation stays FP32
- [ ] Test accuracy degradation (<1% target)

**3:00 - 4:00 PM**: FP16 Testing
- [ ] Run integration tests with FP16
- [ ] Measure accuracy degradation
- [ ] Compare forces/energies to baseline
- [ ] Verify no type errors

**4:00 - 5:00 PM**: FP16 Benchmarking
- [ ] Benchmark speedup (target 1.5-2x)
- [ ] Test different batch sizes
- [ ] Document results

**Checkpoint 5**: FP16 working correctly, 1.5-2x speedup measured

#### Evening Check (5:00 PM)
- [ ] torch.compile(): COMPLETE, 1.3-1.5x speedup
- [ ] FP16: COMPLETE, 1.5-2x speedup
- [ ] Correctness verified: ALL TESTS PASSING
- [ ] Documentation updated

---

### Day 3 (November 26, 2025) - Combined Testing & Reporting

#### Morning Session (4 hours)
**8:00 - 9:00 AM**: Combined Optimization Testing
- [ ] Test torch.compile() + FP16 together
- [ ] Verify correctness with both optimizations
- [ ] Measure combined speedup (target 2-3x)
- [ ] Test batch processing

**9:00 - 11:00 AM**: Comprehensive Benchmark Suite
- [ ] Single-structure inference (1000 steps)
- [ ] Batch inference (4, 8, 16 structures)
- [ ] Memory usage profiling
- [ ] Accuracy analysis across test set
- [ ] Compare to baseline benchmarks

**11:00 - 12:00 PM**: Batch Processing Verification
- [ ] Test batch sizes: 1, 4, 8, 16
- [ ] Verify 16x speedup from batching
- [ ] Test with/without optimizations
- [ ] Document scaling behavior

**Checkpoint 6**: Combined 2-3x speedup + 16x batch speedup = 32-48x total

#### Afternoon Session (3 hours)
**1:00 - 3:00 PM**: Phase 1 Completion Report
- [ ] Executive summary
- [ ] Performance results (tables, graphs)
- [ ] Accuracy analysis
- [ ] Memory usage comparison
- [ ] Recommendations for Phase 2
- [ ] Known issues and limitations

**3:00 - 4:00 PM**: Documentation Updates
- [ ] Update main README
- [ ] Update OPTIMIZATION_ROADMAP
- [ ] Add Phase 1 usage examples
- [ ] Update benchmarking docs

**4:00 - 5:00 PM**: Phase 2 Planning
- [ ] Create Issues for CUDA optimization
- [ ] Define Phase 2 milestones
- [ ] Update project board
- [ ] Prepare handoff

**Final Checkpoint**: All Week 1 deliverables complete

---

## Detailed Task Breakdown

### Task 1: Python 3.12 Environment
**Status**: COMPLETE
**Time**: 2 hours (actual)

**Steps**:
1. [x] Create conda environment with Python 3.12
2. [x] Install PyTorch 2.9.1 with CUDA 12.8
3. [x] Install PyTorch Geometric + extensions
4. [x] Install project dependencies
5. [x] Install project in development mode
6. [x] Verify torch.compile() availability
7. [x] Run test suite

**Acceptance Criteria**:
- [x] Python 3.12.12 installed
- [x] PyTorch 2.9.1+cu128 installed
- [x] torch.compile() available: True
- [x] CUDA available: True
- [x] 458/470 tests passing (12 failures in trainer - expected)

**Environment Details**:
- Conda env: `mlff-py312`
- Python: 3.12.12
- PyTorch: 2.9.1+cu128
- CUDA: 12.8
- torch.compile(): Available

---

### Task 2: Quick MD Validation
**Status**: PENDING
**Time**: 1.5 hours estimated

**Implementation**: `scripts/quick_md_validation.py`

**Test System**:
- Molecule: Benzene (C6H6, 12 atoms)
- Duration: 100ps
- Timestep: 0.5fs
- Ensemble: NVE (constant energy)
- Temperature: 300K (initial)

**Analysis Metrics**:
- Energy conservation (drift <1%)
- Energy fluctuations (std <0.5%)
- Temperature stability (±20K)
- No atomic explosions

**Pass/Fail Criteria**:
```python
passed = (
    abs(energy_drift) < 1.0 and      # <1% drift
    energy_std < 0.5 and             # <0.5% fluctuations
    abs(temp_mean - 300) < 20        # within 20K
)
```

**Outputs**:
- `validation_results/quick_nve/validation_report.txt`
- `validation_results/quick_nve/quick_md_validation.png`

**Run Command**:
```bash
conda activate mlff-py312
python scripts/quick_md_validation.py \
    --checkpoint checkpoints/best_model.pt \
    --timestep 0.5 \
    --duration 100 \
    --output validation_results/quick_nve
```

---

### Task 3: torch.compile() Implementation
**Status**: PENDING
**Time**: 3 hours estimated

**File**: `src/mlff_distiller/inference/ase_calculator.py`

**Changes**:
1. Add `use_compile` parameter (default: True)
2. Add `compile_mode` parameter (default: 'reduce-overhead')
3. Apply torch.compile() in __init__
4. Add fallback for older PyTorch versions
5. Log compilation status

**Code Changes**:
```python
def __init__(
    self,
    checkpoint_path: Union[str, Path],
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    use_compile: bool = True,  # NEW
    compile_mode: str = 'reduce-overhead',  # NEW
    use_fp16: bool = False,
    logger: Optional[logging.Logger] = None,
):
    # ... existing initialization ...

    # Apply torch.compile() if requested
    if self.use_compile:
        try:
            if hasattr(torch, 'compile'):
                self.model = torch.compile(
                    self.model,
                    mode=compile_mode,
                    fullgraph=True,
                    disable=False
                )
                self.logger.info(f"✓ Model compiled with torch.compile(mode={compile_mode})")
            else:
                self.logger.warning("torch.compile() not available")
                self.use_compile = False
        except Exception as e:
            self.logger.warning(f"torch.compile() failed: {e}")
            self.use_compile = False
```

**Testing**:
```bash
# Correctness
pytest tests/integration/test_ase_calculator.py -v

# Benchmark
python scripts/benchmark_inference.py \
    --use-compile \
    --output benchmarks/with_compile/
```

**Expected Results**:
- Speedup: 1.3-1.5x
- Accuracy: Identical to baseline (within numerical precision)
- Compilation time: ~30s on first run (one-time cost)

---

### Task 4: FP16 Implementation
**Status**: PENDING
**Time**: 2.5 hours estimated

**Approach**: Autocast-only (no explicit model.half())

**Changes to `_batch_forward()`**:
```python
def _batch_forward(self, batch_data):
    """Single forward pass for entire batch."""

    # Use autocast for FP16 (don't convert model!)
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
        energies = self.model(...)

    # Forces stay FP32 (autocast handles conversion)
    forces = -torch.autograd.grad(
        energies.sum(),
        batch_data['positions'],
        create_graph=False,
        retain_graph=False
    )[0]

    return {'energies': energies, 'forces': forces, 'batch': batch_data['batch']}
```

**Testing**:
```bash
# Accuracy
python scripts/validate_student_on_test_molecule.py \
    --use-fp16 \
    --compare-to-baseline

# Benchmark
python scripts/benchmark_inference.py \
    --use-fp16 \
    --output benchmarks/with_fp16/
```

**Expected Results**:
- Speedup: 1.5-2x
- Accuracy degradation: <1%
- No type errors
- Memory usage: 50% reduction

---

### Task 5: Combined Benchmarking
**Status**: PENDING
**Time**: 3 hours estimated

**Benchmark Suite**:
1. Single-structure inference (1000 steps)
2. Batch inference (4, 8, 16 structures)
3. Memory profiling
4. Accuracy analysis

**Test Configurations**:
- Baseline (no optimizations)
- torch.compile() only
- FP16 only
- torch.compile() + FP16

**Metrics to Collect**:
- Inference time per structure (ms)
- Throughput (structures/second)
- Memory usage (MB)
- Energy MAE vs baseline (eV)
- Force RMSE vs baseline (eV/Å)

**Expected Results**:
```
Configuration          | Speedup | Memory | Accuracy
-----------------------|---------|--------|----------
Baseline               | 1.0x    | 100%   | 100%
torch.compile()        | 1.4x    | 100%   | 100%
FP16                   | 1.8x    | 50%    | 99.5%
compile + FP16         | 2.5x    | 50%    | 99.5%
Batch-16               | 16x     | 150%   | 100%
Batch-16 + compile+FP16| 40x     | 75%    | 99.5%
```

---

### Task 6: Phase 1 Completion Report
**Status**: PENDING
**Time**: 2 hours estimated

**File**: `benchmarks/PHASE1_COMPLETION_REPORT.md`

**Contents**:
1. Executive Summary
2. Performance Results
   - Inference time comparison
   - Speedup analysis
   - Throughput metrics
3. Accuracy Analysis
   - Energy MAE
   - Force RMSE
   - Distribution plots
4. Memory Usage
   - Peak memory
   - Memory efficiency
5. Recommendations for Phase 2
6. Known Issues

**Deliverables**:
- Markdown report with tables and graphs
- Performance plots (PNG)
- Raw benchmark data (JSON)

---

## GitHub Issues

### Issue #27: torch.compile() Implementation
**Labels**: enhancement, optimization, priority-high
**Milestone**: Phase 1 Optimizations
**Assignee**: cuda-optimization-engineer
**Estimated Time**: 1 day

**Description**: Enable torch.compile() for 1.3-1.5x speedup

**Tasks**:
- [ ] Add use_compile parameter
- [ ] Implement torch.compile() wrapper
- [ ] Test correctness
- [ ] Benchmark speedup
- [ ] Update documentation

**Acceptance Criteria**:
- torch.compile() working
- All tests passing
- 1.3-1.5x speedup achieved

---

### Issue #28: FP16 Mixed Precision
**Labels**: enhancement, optimization, priority-high
**Milestone**: Phase 1 Optimizations
**Assignee**: cuda-optimization-engineer
**Estimated Time**: 0.5 days

**Description**: Implement FP16 mixed precision using autocast-only approach

**Tasks**:
- [ ] Remove explicit .half() conversions
- [ ] Implement autocast in forward pass
- [ ] Test accuracy degradation
- [ ] Benchmark speedup
- [ ] Update documentation

**Acceptance Criteria**:
- FP16 working without type errors
- Accuracy degradation <1%
- 1.5-2x speedup achieved

---

### Issue #29: Quick MD Validation
**Labels**: validation, testing, priority-high
**Milestone**: Phase 1 Optimizations
**Assignee**: testing-benchmark-engineer
**Estimated Time**: 0.5 days

**Description**: Run 100ps NVE trajectory to verify MD stability

**Tasks**:
- [ ] Create validation script
- [ ] Run 100ps NVE trajectory
- [ ] Analyze energy conservation
- [ ] Document results

**Acceptance Criteria**:
- Energy drift <1%
- Temperature stable (±10K)
- No atomic explosions

---

## Risk Management

### Risk 1: torch.compile() Incompatibility
**Probability**: Low
**Impact**: High
**Mitigation**: Fallback to baseline if compilation fails

### Risk 2: FP16 Accuracy Loss
**Probability**: Medium
**Impact**: High
**Mitigation**: Measure accuracy degradation, revert if >1%

### Risk 3: MD Instability
**Probability**: Low
**Impact**: Critical
**Mitigation**: Run validation before optimization investment

### Risk 4: Environment Issues
**Probability**: Medium
**Impact**: Medium
**Mitigation**: Document exact versions, test thoroughly

---

## Success Metrics

**Phase 1 Complete When**:
- [ ] Python 3.12 environment operational
- [ ] MD validation passed (<1% energy drift)
- [ ] torch.compile() implemented (1.3-1.5x)
- [ ] FP16 implemented (1.5-2x)
- [ ] Combined 2-3x speedup measured
- [ ] Batch processing 16x speedup verified
- [ ] All integration tests passing
- [ ] Documentation complete
- [ ] Phase 1 report generated

**Total Expected Speedup**:
- Single-structure: 2-3x (compile + FP16)
- Batch-16: 32-48x (batch × compile × FP16)

---

## Next Steps (Phase 2)

After Week 1 completion:
1. Custom CUDA kernels for force computation
2. Memory-efficient batching
3. JIT-compiled distance calculations
4. Advanced graph optimizations
5. Multi-GPU support

**Target**: Additional 2-3x speedup → 10x total

---

## Contact & Escalation

**Coordinator**: ml-distillation-coordinator
**Working Directory**: `/home/aaron/ATX/software/MLFF_Distiller`
**Environment**: `mlff-py312`
**Checkpoint**: `checkpoints/best_model.pt`

**Escalation Path**:
1. Check this coordination plan
2. Review GitHub Issues #27-29
3. Consult Phase 1 optimization spec
4. Contact coordinator for architectural decisions
