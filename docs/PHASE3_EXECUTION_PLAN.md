# Phase 3 CUDA Optimization - Execution Plan

**Date**: 2025-11-24
**Coordinator**: Lead Project Coordinator
**Status**: APPROVED - User confirmed "yes proceed with optimization"
**Goal**: Achieve 5-10x speedup over baseline (target: <2ms inference time)

---

## Executive Summary

User has approved proceeding with Phase 3 CUDA optimizations despite coordinator's recommendation to validate MD stability first. This document outlines the complete execution plan to achieve 5-10x speedup target.

### Current State

**Performance Baseline**:
- **Baseline (PyTorch eager)**: ~22.3 ms per inference (single molecule)
- **With TorchScript JIT**: ~0.43 ms per inference (2.0x speedup from profiling)
- **Target**: 0.086-0.172 ms per inference (5-10x total from baseline)
- **Gap**: Need 2.5-5x additional speedup over TorchScript

**Completed Work**:
- Phase 1: torch.compile() + FP16 testing (1.5-2x speedup potential)
- Phase 2: TorchScript JIT compilation (2.0x speedup achieved)
- CUDA-X Analysis: Comprehensive profiling and library evaluation
- Bottleneck identification: 64% time in force computation, 41% in message passing

**Key Finding**: CUDA-X libraries (cuGraph, cuSPARSE, Thrust) provide minimal direct benefit. Optimization path is PyTorch-level optimizations + custom kernels.

---

## Risk Assessment and Mitigation

### Critical Risks

**Risk 1: Optimization may not reach 5-10x target**
- Likelihood: Medium (optimistic estimates assume perfect speedup stacking)
- Impact: High (project goal not met)
- Mitigation: Phase 3.1 quick wins (3-5x) are low-risk; only proceed to custom kernels if needed
- Fallback: Accept 3-5x speedup as "good enough" for deployment

**Risk 2: Custom kernels may break model accuracy**
- Likelihood: Medium (FP16 and kernel fusion can introduce numerical errors)
- Impact: High (model unusable if accuracy degrades)
- Mitigation: Comprehensive validation suite with <10 meV energy error tolerance
- Rollback plan: Revert to TorchScript JIT (2x speedup, perfect accuracy)

**Risk 3: MD validation may still reveal instabilities**
- Likelihood: Medium (user chose to skip MD validation before optimization)
- Impact: Critical (requires model retraining, optimization wasted)
- Mitigation: Run MD validation in parallel during Week 1 optimization work
- Decision point: If MD fails, STOP optimization and fix model first

**Risk 4: Custom CUDA code adds maintenance burden**
- Likelihood: High (CUDA kernels are complex)
- Impact: Medium (harder to debug, update, port to new hardware)
- Mitigation: Use Triton for high-level kernel programming; document thoroughly
- Alternative: Stick with PyTorch-level optimizations only (torch.compile, FP16, torch-cluster)

**Risk 5: torch.compile() blocked by Python 3.13 incompatibility**
- Likelihood: High (current environment is Python 3.13)
- Impact: Medium (loses 1.3-1.5x speedup)
- Mitigation: Create separate Python 3.12 conda environment
- Workaround: Skip torch.compile() and focus on FP16 + torch-cluster + custom kernels

### Decision Gates

**Gate 1 (End of Week 1)**: Evaluate Phase 3.1 results
- If speedup ≥5x: STOP, declare victory, proceed to deployment
- If speedup 3-5x: EVALUATE if custom kernels worth effort
- If speedup <3x: INVESTIGATE what went wrong, may need to rethink approach

**Gate 2 (End of Week 2)**: MD validation results
- If MD stable: CONTINUE optimization work
- If MD unstable: STOP optimization, prioritize model fixes

**Gate 3 (End of Week 3)**: Custom kernel results
- If speedup ≥7x: CONTINUE to production tuning
- If speedup 5-7x: SKIP Phase 3.3, proceed to deployment
- If speedup <5x: ROLLBACK to Phase 3.1, investigate issues

---

## Phase 3.1: Quick Wins (Week 1)

**Objective**: Achieve 3-5x speedup with minimal code changes
**Timeline**: 5 working days
**Agent**: cuda-optimization-engineer (lead), testing-benchmark-engineer (validation)

### Tasks

**Day 1-2: Python Environment + torch.compile()**
- Create Python 3.12 conda environment
- Install PyTorch 2.x with CUDA 12.1 support
- Test torch.compile() with modes: 'default', 'reduce-overhead', 'max-autotune'
- Benchmark speedup on test molecules
- Validate numerical accuracy (<1e-5 energy error)

**Deliverable**: torch.compile() working, 1.3-1.5x speedup documented

---

**Day 3: FP16 Mixed Precision**
- Fix current FP16 implementation (use autocast only, no .half())
- Test on validation set (100 molecules)
- Measure accuracy degradation (target: <10 meV MAE increase)
- Benchmark speedup (expected: 1.5-2x on RTX 3080 Ti tensor cores)

**Deliverable**: FP16 autocast working, 1.5-2x additional speedup, accuracy validated

---

**Day 4: torch-cluster Integration**
- Install torch-cluster from PyG repository
- Replace radius_graph_native() with torch_cluster.radius()
- Test correctness on small/medium/large molecules
- Benchmark neighbor search speedup (expected: 2-3x faster neighbor search)

**Deliverable**: torch-cluster integrated, neighbor search optimized

---

**Day 5: Integration + End-to-End Benchmarking**
- Combine all optimizations (compile + FP16 + torch-cluster)
- Run comprehensive benchmark suite (10, 20, 50, 100, 200 atoms)
- Profile to identify remaining bottlenecks
- Generate Phase 3.1 performance report

**Deliverable**: Complete Phase 3.1 report with speedup analysis

---

### Expected Phase 3.1 Results

| Optimization | Speedup | Cumulative Speedup | Time (ms) |
|--------------|---------|-------------------|-----------|
| Baseline | 1.0x | 1.0x | 22.3 |
| + TorchScript (done) | 2.0x | 2.0x | 11.15 |
| + torch.compile() | 1.3x | 2.6x | 8.58 |
| + FP16 autocast | 1.7x | 4.4x | 5.07 |
| + torch-cluster | 1.2x | 5.3x | 4.21 |

**Conservative estimate**: 3-4x total speedup (7.4-5.6 ms)
**Optimistic estimate**: 5-6x total speedup (4.5-3.7 ms)

**Decision**: If ≥5x achieved, STOP and declare success. If 3-5x, evaluate cost/benefit of Phase 3.2.

---

## Phase 3.2: Custom CUDA Kernels (Week 2-3)

**Objective**: Achieve 7-10x total speedup with custom kernels
**Timeline**: 2 weeks
**Agent**: cuda-optimization-engineer (lead), ml-architecture-designer (integration)

**ONLY PROCEED IF**:
- Phase 3.1 achieved <5x speedup AND user wants to pursue further optimization
- MD validation passed (model is stable)
- Team has bandwidth for 2 weeks of kernel development

### Week 2: Custom Neighbor Search (Optional)

**Goal**: Replace torch-cluster with optimized cell-list kernel (if torch-cluster insufficient)

**Tasks**:
- Design cell-list (spatial hashing) algorithm
- Implement CUDA kernel with CUB for atomic operations
- Write PyTorch C++ extension for Python binding
- Unit test: compare output to torch-cluster (must be identical)
- Benchmark on 10-200 atom molecules

**Expected Speedup**: 1.2x additional (only significant for >50 atoms)

**Complexity**: Medium (3-5 days of CUDA development)

**Decision**: If torch-cluster already fast enough, SKIP this task

---

### Week 3: Triton Fused Message Passing Kernel

**Goal**: Fuse RBF computation + linear layer + message aggregation into single kernel

**Current bottleneck**: Message passing is 41% of forward pass, split across multiple kernels

**Approach**: Use Triton (high-level GPU language) to fuse operations

**Tasks**:
- Design kernel fusion strategy (what ops to combine)
- Implement Triton fused kernel for message passing layer
- Test numerical equivalence (must match PyTorch within 1e-5)
- Benchmark speedup on small/medium/large molecules
- Profile memory bandwidth utilization

**Expected Speedup**: 1.5-2x on message passing (20-30% overall improvement)

**Complexity**: Medium-Hard (5-7 days)

**Code Structure**:
```python
# src/mlff_distiller/cuda/message_passing_triton.py
@triton.jit
def fused_message_kernel(...):
    # Fuse: RBF + filter_net + multiply + scatter
    ...
```

---

### Expected Phase 3.2 Results

| Optimization | Speedup | Cumulative Speedup | Time (ms) |
|--------------|---------|-------------------|-----------|
| After Phase 3.1 | 5.3x | 5.3x | 4.21 |
| + Custom neighbor search | 1.2x | 6.4x | 3.48 |
| + Triton fused kernel | 1.5x | 9.6x | 2.32 |

**Conservative estimate**: 7x total speedup (3.2 ms)
**Optimistic estimate**: 10x total speedup (2.2 ms)

**Decision**: If ≥7x achieved, proceed to Phase 3.3 for final tuning. If 5-7x, evaluate if Phase 3.3 worth it.

---

## Phase 3.3: Production Tuning (Week 4)

**Objective**: Squeeze final 1.2-1.5x speedup for production deployment
**Timeline**: 1 week
**Agent**: cuda-optimization-engineer

**ONLY PROCEED IF**:
- Phase 3.2 completed successfully
- Speedup is close to 10x target and want to push further
- Model will be deployed in production (worth optimization effort)

### Day 1-2: CUDA Graphs

**Goal**: Reduce kernel launch overhead by capturing inference as single graph

**Challenge**: CUDA graphs require static input shapes (fixed number of atoms)

**Implementation**:
```python
# Capture graphs for common sizes
graphs = {}
for n_atoms in [10, 20, 50, 100]:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        energy, forces = model.predict(n_atoms)
    graphs[n_atoms] = graph

# Use at inference time
graph = graphs.get(n_atoms, None)
if graph:
    graph.replay()  # Fast path
else:
    energy, forces = model.predict(n_atoms)  # Fallback
```

**Expected Speedup**: 1.2-1.3x (reduces ~10% kernel launch overhead)

---

### Day 3-5: Kernel Parameter Tuning

**Goal**: Optimize custom CUDA kernels for maximum occupancy and memory bandwidth

**Approach**:
- Profile kernels with `ncu` (NVIDIA Nsight Compute)
- Identify bottlenecks: occupancy, memory bandwidth, register pressure
- Tune block sizes (32, 64, 128, 256 threads)
- Optimize shared memory usage
- Experiment with kernel launch parameters

**Expected Speedup**: 1.1-1.2x (incremental gains)

**Complexity**: Hard (requires deep CUDA profiling knowledge)

---

### Expected Phase 3.3 Results

| Optimization | Speedup | Cumulative Speedup | Time (ms) |
|--------------|---------|-------------------|-----------|
| After Phase 3.2 | 9.6x | 9.6x | 2.32 |
| + CUDA graphs | 1.2x | 11.5x | 1.94 |
| + Kernel tuning | 1.1x | 12.7x | 1.76 |

**Conservative estimate**: 10x total speedup (2.2 ms)
**Optimistic estimate**: 15x total speedup (1.5 ms)

**Target achieved**: 5-10x goal met or exceeded ✓

---

## Parallel Workstream: MD Validation (Week 1-2)

**CRITICAL**: Run MD validation in parallel during optimization work

**Objective**: Confirm student model is stable in molecular dynamics before investing in optimization

**Agent**: testing-benchmark-engineer

**Tasks**:
- Set up MD validation framework (ASE + NVE ensemble)
- Run 1ps MD simulations on 10 test molecules
- Measure energy conservation (<1% drift acceptable)
- Measure temperature stability
- Generate MD validation report

**Timeline**: 5 days (parallel to Phase 3.1)

**Decision Point**: If MD unstable, STOP optimization and prioritize model fixes

---

## Agent Assignments and Communication

### Agents

**Agent 1: cuda-optimization-engineer** (PRIMARY)
- Owns Phase 3.1, 3.2, 3.3 implementation
- Creates benchmark scripts
- Documents optimization strategies
- Reports speedup results

**Agent 2: testing-benchmark-engineer**
- Runs MD validation (parallel to Phase 3.1)
- Validates accuracy after each optimization
- Creates comprehensive test suite
- Generates validation reports

**Agent 3: ml-architecture-designer**
- Reviews custom kernel integration with model architecture
- Ensures optimizations don't break ASE calculator interface
- Helps with architectural decisions (when to use graphs vs. dynamic)

**Agent 4: data-pipeline-engineer** (BACKUP)
- On standby to help with data loading optimizations if needed
- Can assist with batching strategies

**Agent 5: training-engineer** (BACKUP)
- On standby if model needs retraining (if MD fails)
- Can help with accuracy validation

### Communication Protocol

**Daily Standups** (async via GitHub):
- Each agent comments on their assigned Issue with progress
- Coordinator monitors and responds to blockers within 4 hours
- Use @mentions to escalate urgent items

**Blocker Escalation**:
- Tag Issue with "status:blocked" label
- Comment with blocker description + @coordinator mention
- Coordinator responds with decision or action plan within 4 hours

**Weekly Reports**:
- End of Week 1: Phase 3.1 performance report
- End of Week 2: MD validation report + Phase 3.2 interim report
- End of Week 3: Phase 3.2 final report
- End of Week 4: Phase 3.3 final report + deployment guide

---

## Success Criteria

### Minimum Viable Product (MVP)

**Target**: 5x total speedup (4.5 ms inference time)

**Must Have**:
- [ ] torch.compile() working (or documented why skipped)
- [ ] FP16 autocast implemented and validated
- [ ] torch-cluster integrated
- [ ] All correctness tests passing (numerical equivalence to baseline)
- [ ] Accuracy degradation <1% on validation set
- [ ] MD validation passed (if not, model fixes prioritized)

**Timeline**: Week 1 (Phase 3.1)

---

### Full Success

**Target**: 7-10x total speedup (2.2-3.2 ms inference time)

**Must Have**:
- [ ] All MVP requirements
- [ ] Custom CUDA kernels implemented (neighbor search and/or message passing)
- [ ] Comprehensive benchmark suite across molecule sizes
- [ ] Production deployment guide
- [ ] All tests passing (unit, integration, accuracy, MD stability)

**Timeline**: Week 3 (Phase 3.2)

---

### Stretch Goal

**Target**: >10x total speedup (<2.2 ms inference time)

**Must Have**:
- [ ] All Full Success requirements
- [ ] CUDA graphs implemented for common sizes
- [ ] Custom kernels tuned for maximum performance
- [ ] Deployment guide includes production optimization tips

**Timeline**: Week 4 (Phase 3.3)

---

## Rollback Plan

If optimizations fail or introduce unacceptable accuracy loss:

**Rollback Level 1**: Revert Phase 3.3 tuning
- Keep Phase 3.1 + 3.2 optimizations
- Accept 7x speedup instead of 10x

**Rollback Level 2**: Revert Phase 3.2 custom kernels
- Keep Phase 3.1 PyTorch optimizations only
- Accept 3-5x speedup instead of 7-10x

**Rollback Level 3**: Revert all Phase 3 work
- Use TorchScript JIT model (Phase 2)
- Accept 2x speedup
- Perfect accuracy guaranteed

**Rollback Level 4**: Revert to baseline PyTorch model
- If TorchScript introduces issues
- Accept 1x performance (no speedup)
- Perfect accuracy guaranteed

---

## File Organization

```
MLFF_Distiller/
├── src/mlff_distiller/
│   ├── cuda/                           # NEW: Custom CUDA code
│   │   ├── __init__.py
│   │   ├── neighbor_search.cu          # Custom neighbor search kernel (Phase 3.2)
│   │   ├── neighbor_search.cpp         # PyTorch binding
│   │   ├── message_passing_triton.py   # Triton fused kernel (Phase 3.2)
│   │   └── setup.py                    # CUDA extension compilation
│   ├── inference/
│   │   ├── optimized_inference.py      # NEW: Optimized inference pipeline
│   │   ├── ase_calculator_optimized.py # NEW: Optimized ASE interface
│   │   └── cuda_graphs.py              # NEW: CUDA graph management (Phase 3.3)
│   └── models/
│       └── student_model.py            # Use optimized ops
├── benchmarks/
│   ├── benchmark_phase3.py             # NEW: Phase 3 benchmark script
│   ├── compare_optimizations.py        # NEW: Compare all optimization levels
│   └── profile_custom_kernels.py       # NEW: Detailed kernel profiling
├── tests/
│   ├── unit/
│   │   └── test_custom_kernels.py      # NEW: CUDA kernel tests
│   ├── integration/
│   │   └── test_optimized_inference.py # NEW: End-to-end tests
│   └── validation/
│       ├── test_accuracy.py            # NEW: Accuracy validation
│       └── test_md_stability.py        # NEW: MD validation
├── models/
│   ├── student_model_jit.pt            # EXISTS: TorchScript (Phase 2)
│   ├── student_model_compiled.pt       # NEW: torch.compile version (Phase 3.1)
│   └── student_model_optimized.pt      # NEW: Full optimization (Phase 3.2/3.3)
├── docs/
│   ├── PHASE3_EXECUTION_PLAN.md        # THIS FILE
│   ├── OPTIMIZATION_GUIDE.md           # NEW: User guide for optimizations
│   └── DEPLOYMENT_GUIDE.md             # NEW: Production deployment
└── configs/
    └── optimization_configs.yaml       # NEW: Optimization presets
```

---

## Testing and Validation Strategy

### Correctness Tests

**1. Numerical Equivalence**
```python
def test_numerical_equivalence():
    baseline_energy, baseline_forces = baseline_model(positions)
    optimized_energy, optimized_forces = optimized_model(positions)

    assert torch.allclose(baseline_energy, optimized_energy, atol=1e-5)
    assert torch.allclose(baseline_forces, optimized_forces, atol=1e-4)
```

**2. Validation Set Accuracy**
- Energy MAE must remain within 1% of baseline
- Force MAE must remain within 2% of baseline
- No catastrophic failures (NaN, inf)

**3. MD Stability**
- Energy conservation: <1% drift over 1ps
- Temperature stability: <5% fluctuation
- No explosions or unphysical behavior

---

### Performance Tests

**Benchmarking Protocol**:
```python
def benchmark(model, molecules, warmup=10, iterations=100):
    # Warmup GPU
    for _ in range(warmup):
        energy, forces = model.predict(molecules[0])

    # Benchmark
    times = []
    for mol in molecules:
        for _ in range(iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            energy, forces = model.predict(mol)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'p50_ms': np.median(times) * 1000,
        'p95_ms': np.percentile(times, 95) * 1000,
    }
```

**Test Molecules**: 3, 5, 10, 20, 50, 100, 200 atoms

**Metrics to Track**:
- Mean inference time
- Standard deviation (measure consistency)
- P95 latency (worst-case performance)
- Speedup vs baseline
- Memory usage

---

### Profiling

**Tools**:
1. **PyTorch Profiler**: High-level operation breakdown
2. **nsys**: CUDA kernel timeline (NVIDIA Nsight Systems)
3. **ncu**: Kernel metrics (NVIDIA Nsight Compute)

**Commands**:
```bash
# Timeline profiling
nsys profile -o phase3_profile.qdrep python benchmark_phase3.py

# Kernel metrics
ncu --set full -o kernel_metrics python benchmark_phase3.py

# PyTorch profiler
python -m torch.utils.bottleneck benchmark_phase3.py
```

---

## Timeline and Milestones

### Week 1: Phase 3.1 Quick Wins
- **Day 1-2**: Python 3.12 env + torch.compile()
- **Day 3**: FP16 mixed precision
- **Day 4**: torch-cluster integration
- **Day 5**: Integration + benchmarking

**Milestone**: 3-5x speedup achieved
**Deliverable**: Phase 3.1 Performance Report
**Decision Gate**: Continue to Phase 3.2 or stop?

---

### Week 2: Phase 3.2 Custom Neighbor Search
- **Day 1-2**: Design cell-list algorithm
- **Day 3-4**: Implement CUDA kernel
- **Day 5**: Test + benchmark

**Milestone**: Neighbor search optimized (if needed)
**Deliverable**: Custom kernel implementation + benchmarks

---

### Week 3: Phase 3.2 Triton Message Passing
- **Day 1-2**: Design kernel fusion
- **Day 3-4**: Implement Triton kernel
- **Day 5**: Test numerical equivalence + benchmark

**Milestone**: 7-10x speedup achieved
**Deliverable**: Phase 3.2 Final Report
**Decision Gate**: Continue to Phase 3.3 or stop?

---

### Week 4: Phase 3.3 Production Tuning
- **Day 1-2**: CUDA graphs implementation
- **Day 3-5**: Kernel parameter tuning

**Milestone**: 10-15x speedup achieved
**Deliverable**: Production Deployment Guide

---

## Dependencies and Prerequisites

### Environment
- CUDA 12.1+ (for latest cuBLAS/cuDNN)
- Python 3.12 (for torch.compile()) OR Python 3.13 (if skipping torch.compile())
- PyTorch 2.x with CUDA support
- NVIDIA GPU with tensor cores (RTX 30xx or better)

### Libraries
```bash
# Phase 3.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.x.x+cu121.html

# Phase 3.2
pip install triton

# Phase 3.3
# (uses CUDA runtime, no additional installs)
```

### Conda Environment
```yaml
name: mlff-optimized
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.12
  - pytorch=2.5
  - pytorch-cuda=12.1
  - pip
  - pip:
    - torch-cluster
    - triton
    - ase
    - numpy
    - pytest
```

---

## Monitoring and Reporting

### Weekly Reports

**Week 1 Report** (Phase 3.1):
- Speedup achieved for each optimization
- Cumulative speedup
- Accuracy validation results
- MD validation status
- Decision: Continue to Phase 3.2?

**Week 2 Report** (MD Validation):
- MD stability results
- Energy conservation
- Temperature fluctuations
- Decision: Model stable? Continue optimization?

**Week 3 Report** (Phase 3.2):
- Custom kernel speedup
- Total speedup vs baseline
- Complexity added (lines of CUDA code)
- Decision: Continue to Phase 3.3 or deploy?

**Week 4 Report** (Phase 3.3):
- Final speedup
- Production deployment guide
- Maintenance recommendations
- Lessons learned

### GitHub Project Board

Columns:
- **Backlog**: Future optimization ideas
- **In Progress**: Current week's tasks
- **Review**: Waiting for validation/benchmarking
- **Done**: Completed tasks

### Issue Labels
- `phase:3.1`, `phase:3.2`, `phase:3.3`
- `status:blocked`, `status:in-progress`, `status:review`
- `priority:critical`, `priority:high`, `priority:medium`
- `agent:cuda`, `agent:testing`, `agent:architecture`

---

## Conclusion

This execution plan provides a structured approach to achieving 5-10x speedup through three progressive phases:

1. **Phase 3.1 (Week 1)**: Low-risk PyTorch optimizations (3-5x speedup)
2. **Phase 3.2 (Week 2-3)**: Custom CUDA kernels (7-10x speedup)
3. **Phase 3.3 (Week 4)**: Production tuning (10-15x speedup)

**Decision gates** at each phase ensure we can stop early if targets are met or issues arise.

**Parallel MD validation** ensures we don't waste optimization effort on unstable models.

**Comprehensive rollback plan** allows reverting to previous optimization levels if issues occur.

**Clear agent assignments** and communication protocols keep the team coordinated.

**Next Step**: Create GitHub Issues for Week 1 tasks and begin execution.
