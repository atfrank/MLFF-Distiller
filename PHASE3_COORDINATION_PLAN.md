# Phase 3 CUDA Optimization - Coordination Plan

**Date**: 2025-11-24
**Status**: APPROVED - User authorized to proceed
**Target**: 5-10x speedup
**Timeline**: 4 weeks

---

## User Authorization

User approved proceeding with Phase 3 CUDA optimization to achieve 5-10x speedup target despite coordinator's earlier recommendation to validate MD stability first.

**User Decision**: "yes proceed with optimization"

---

## Current Baseline

### Performance Status

**Achieved to date**:
- Phase 1: torch.compile() + FP16 attempted (1.5-2x expected)
- Phase 2: TorchScript JIT implemented (2.0x achieved)
- Current best: 0.430 ms per inference (2.0x vs baseline)

**Profiling Results** (from detailed analysis):
- Current: 10.95 ms per inference (12-atom benzene)
- Bottlenecks identified:
  1. Update layers (45% of forward pass)
  2. Message passing (41% of forward pass)
  3. Force computation (64% of total time)
  4. Neighbor search (12% of forward pass)

### Accuracy Status

TorchScript JIT: Perfect accuracy (<1e-6 eV error)
FP16: Acceptable accuracy (0.009 eV energy error, 0.0015 eV/A force RMSE)

---

## Phase 3 Strategy

Based on comprehensive CUDA-X analysis in `benchmarks/cuda_x_analysis/`, the optimization path is:

**Key Finding**: CUDA-X libraries won't significantly help (PyTorch already uses cuBLAS/cuDNN optimally)

**Real path to 5-10x**:
1. torch-cluster for neighbor search
2. Custom CUDA kernels (Triton)
3. CUDA graphs
4. Kernel tuning

---

## 4-Week Implementation Plan

### Week 1: Quick Wins (Target: 3-5x total speedup)

**Priority 1: torch-cluster Integration**
- Install torch-cluster library
- Replace current neighbor search with torch-cluster.knn_graph
- Benchmark improvement
- Expected: 2-3x on neighbor search (10-30% overall for small molecules)
- Timeline: 2-3 days

**Priority 2: torch.compile Optimization**
- Already blocked by Python 3.13 incompatibility
- Alternative: Test TorchScript + additional optimizations
- Timeline: 1-2 days

**Priority 3: FP16 Refinement**
- Already implemented (1.37x speedup)
- Validate accuracy on larger test set
- Ensure <10 meV accuracy maintained
- Timeline: 1 day

**Week 1 Deliverables**:
- [ ] torch-cluster integrated and benchmarked
- [ ] Combined optimization tested (TorchScript + FP16 + torch-cluster)
- [ ] 3-5x total speedup achieved
- [ ] All accuracy tests passing

---

### Week 2-3: Custom CUDA Kernels (Target: 5-8x total speedup)

**Priority 4: Triton Fused Message Passing**
- Write Triton kernel for message passing
- Fuse: RBF computation + linear layer + element-wise multiply + scatter
- Reduce memory bandwidth
- Expected: 1.5-2x on message passing (20-30% overall improvement)
- Timeline: 5-7 days

**Priority 5: Custom Neighbor Search (if torch-cluster insufficient)**
- Implement cell-list neighbor search algorithm
- Use CUB for atomic operations
- Only if torch-cluster doesn't meet targets
- Expected: 5-10x on neighbor search
- Timeline: 3-5 days

**Week 2-3 Deliverables**:
- [ ] Triton fused message passing kernel implemented
- [ ] Numerical equivalence tests passing
- [ ] 5-8x total speedup achieved
- [ ] MD stability validated

---

### Week 4: Production Optimization (Target: 7-10x total speedup)

**Priority 6: CUDA Graphs**
- Capture forward pass as CUDA graph
- Reduce kernel launch overhead
- Handle fixed-size inputs
- Expected: 1.2-1.3x additional speedup
- Timeline: 2-3 days

**Priority 7: Kernel Tuning**
- Profile custom kernels with nsys/ncu
- Optimize block sizes and grid dimensions
- Tune shared memory usage
- Minimize register pressure
- Expected: 1.1-1.2x additional speedup
- Timeline: 2-3 days

**Week 4 Deliverables**:
- [ ] CUDA graphs implemented
- [ ] Kernel parameters tuned
- [ ] 7-10x total speedup achieved (GOAL MET!)
- [ ] Production deployment ready
- [ ] Complete documentation

---

## Agent Assignments

### Primary: CUDA Optimization Engineer

**Responsibilities**:
- Lead all CUDA optimization implementation
- Write custom kernels (Triton)
- Integrate torch-cluster
- Implement CUDA graphs
- Profile and tune performance

**Week-by-week tasks**:
- Week 1: torch-cluster integration
- Week 2-3: Triton fused kernels
- Week 4: CUDA graphs + tuning

### Secondary: Testing & Benchmarking Engineer

**Responsibilities**:
- Validate correctness of all optimizations
- Run comprehensive benchmarks
- MD stability testing
- Performance tracking
- Accuracy validation

**Week-by-week tasks**:
- Week 1: Benchmark torch-cluster integration
- Week 2-3: Validate custom kernels
- Week 4: Final validation and benchmarking

### Standby: ML Architecture Designer

**Responsibilities**:
- Consult on kernel design
- Advise on architecture changes if needed
- Model profiling support

**Availability**: On-call for technical consultations

---

## GitHub Issues Created

### Issue #25: [M5] [CUDA] Install and integrate torch-cluster for optimized neighbor search
- **Milestone**: M5 (CUDA Optimization)
- **Assignee**: cuda-optimization-engineer
- **Priority**: HIGH
- **Estimated Time**: 2-3 days
- **Dependencies**: None
- **Acceptance Criteria**:
  - [ ] torch-cluster installed successfully
  - [ ] radius_graph replaced with torch_cluster.radius
  - [ ] Correctness tests passing
  - [ ] Benchmark shows 2-3x improvement on neighbor search
  - [ ] Works for 10, 50, 100 atom systems

### Issue #26: [M5] [CUDA] Implement Triton fused message passing kernels
- **Milestone**: M5
- **Assignee**: cuda-optimization-engineer
- **Priority**: HIGH
- **Estimated Time**: 5-7 days
- **Dependencies**: Issue #25
- **Acceptance Criteria**:
  - [ ] Triton kernel implemented for message passing
  - [ ] Fuses RBF + filter + aggregation
  - [ ] Numerical equivalence validated (<1e-5 error)
  - [ ] Benchmark shows 1.5-2x improvement on message passing
  - [ ] Backward pass compatible with autograd

### Issue #27: [M5] [CUDA] Implement CUDA graphs for reduced overhead
- **Milestone**: M5
- **Assignee**: cuda-optimization-engineer
- **Priority**: MEDIUM
- **Estimated Time**: 2-3 days
- **Dependencies**: Issue #26
- **Acceptance Criteria**:
  - [ ] CUDA graph capture implemented
  - [ ] Works for common molecule sizes (10, 20, 50, 100 atoms)
  - [ ] Fallback to non-graph mode for uncommon sizes
  - [ ] Benchmark shows 1.2-1.3x additional improvement
  - [ ] No accuracy loss

### Issue #28: [M5] [Testing] Comprehensive benchmark suite for 5-10x validation
- **Milestone**: M5
- **Assignee**: testing-benchmark-engineer
- **Priority**: HIGH
- **Estimated Time**: Ongoing (throughout Phase 3)
- **Dependencies**: All optimization issues
- **Acceptance Criteria**:
  - [ ] Benchmark suite covers all system sizes (10-100 atoms)
  - [ ] Measures latency, throughput, memory usage
  - [ ] Compares against baseline and intermediate optimizations
  - [ ] Validates 5-10x speedup target achieved
  - [ ] Results exported to JSON for analysis

### Issue #29: [M5] [Testing] MD stability validation with optimized kernels
- **Milestone**: M5
- **Assignee**: testing-benchmark-engineer
- **Priority**: HIGH
- **Estimated Time**: 3-4 days
- **Dependencies**: Issue #26, #27
- **Acceptance Criteria**:
  - [ ] MD simulations run with optimized model
  - [ ] Energy conservation validated (NVE ensemble)
  - [ ] Force accuracy maintained over 1000+ steps
  - [ ] Compare stability against baseline TorchScript
  - [ ] Document any drift or instabilities

---

## Success Criteria

### Week 1 Success Criteria
- [ ] torch-cluster integrated and working
- [ ] 3-5x total speedup achieved
- [ ] Accuracy maintained (<10 meV energy error)
- [ ] Benchmarks on 10, 50, 100 atom systems complete
- [ ] No regression in numerical precision

### Week 2-3 Success Criteria
- [ ] Triton message passing kernel implemented
- [ ] 5-8x total speedup achieved
- [ ] All correctness tests passing
- [ ] MD simulations stable for 1000 steps
- [ ] Numerical equivalence validated

### Week 4 Success Criteria (FINAL)
- [ ] CUDA graphs implemented
- [ ] 7-10x total speedup achieved (GOAL MET!)
- [ ] Production-ready deployment
- [ ] Complete documentation
- [ ] Benchmarks published

### Overall Project Success
- [ ] Achieve 5-10x speedup target
- [ ] Maintain >95% accuracy vs teacher models
- [ ] MD stable for production use
- [ ] Easy deployment (documented)
- [ ] Comprehensive test coverage

---

## Risk Management

### Risk 1: Custom kernels introduce numerical errors
**Impact**: High - May break accuracy requirement
**Probability**: Medium
**Mitigation**:
- Rigorous testing with <1e-5 tolerance
- Compare against baseline for every optimization
- Maintain PyTorch implementation as reference
- Run extensive validation set tests
**Fallback**: Revert to TorchScript if accuracy loss >10 meV

### Risk 2: Optimization takes longer than 4 weeks
**Impact**: Medium - Delays project timeline
**Probability**: Medium
**Mitigation**:
- Phased approach with incremental improvements
- Deliver working optimizations each week
- Track velocity and adjust scope
**Fallback**: Stop at 5x speedup if 10x requires >6 weeks

### Risk 3: CUDA expertise bottleneck
**Impact**: High - Could block entire Phase 3
**Probability**: Low-Medium
**Mitigation**:
- Use Triton (Python-based) instead of raw CUDA C++
- Rely on torch-cluster library for neighbor search
- Leverage PyTorch profiler for guidance
- Consult documentation and examples
**Fallback**: Focus on library-based optimizations (torch-cluster + CUDA graphs)

### Risk 4: CUDA graphs incompatible with dynamic inputs
**Impact**: Medium - Limits applicability
**Probability**: Medium
**Mitigation**:
- Create multiple graphs for common sizes (10, 20, 50, 100 atoms)
- Implement size-based graph selection logic
- Fall back to non-graph mode for uncommon sizes
**Fallback**: Accept dynamic overhead for variable-size molecules

### Risk 5: Performance gains don't combine multiplicatively
**Impact**: Medium - May not reach 10x target
**Probability**: High
**Mitigation**:
- Use conservative speedup estimates
- Profile after each optimization
- Identify diminishing returns early
- Adjust strategy if needed
**Fallback**: Deliver 5-7x speedup (still meets lower target)

---

## Coordination Framework

### Daily Coordination
- **Check-in time**: End of day (async)
- **What to report**:
  - Progress on assigned issues
  - Any blockers encountered
  - Questions for coordinator or other agents
- **Coordinator actions**:
  - Review progress on all issues
  - Unblock agents
  - Update project board

### Weekly Milestones
- **Week 1 checkpoint**: Friday EOD
  - Go/no-go decision for Week 2
  - Review torch-cluster results
  - Adjust custom kernel plan if needed

- **Week 2 checkpoint**: Friday EOD
  - Review Triton kernel progress
  - Validate numerical accuracy
  - Decide on custom neighbor search need

- **Week 3 checkpoint**: Friday EOD
  - Review total speedup achieved
  - Go/no-go for Week 4 optimizations
  - Assess 5-10x target feasibility

- **Week 4 checkpoint**: Friday EOD
  - Final validation
  - Deploy decision
  - Documentation review

### Communication Channels
- **GitHub Issues**: All technical discussions, progress updates
- **PR Reviews**: Code review, feedback, approval
- **@mentions**: For urgent blockers or questions
- **Project Board**: High-level status tracking

---

## Performance Tracking

### Baseline (Pre-Phase 3)
- TorchScript JIT: 0.430 ms per inference (2.0x vs FP32 baseline)
- Accuracy: Perfect (<1e-6 eV error)

### Target Performance

| Milestone | Target Latency | Target Speedup | Status |
|-----------|---------------|----------------|--------|
| Week 1    | ~0.25 ms      | 3-5x           | Pending |
| Week 2-3  | ~0.15 ms      | 5-8x           | Pending |
| Week 4    | ~0.10 ms      | 7-10x          | Pending |

### Tracking Metrics
- **Latency**: Mean inference time (ms) for 10, 50, 100 atom systems
- **Throughput**: Structures per second
- **Memory**: GPU memory usage (MB)
- **Accuracy**: Energy MAE, Force RMSE vs baseline
- **Stability**: MD energy drift over 1000 steps

---

## Code Organization

```
MLFF_Distiller/
├── src/mlff_distiller/
│   ├── cuda/                          # NEW: Custom CUDA kernels
│   │   ├── __init__.py
│   │   ├── triton_message_passing.py  # Triton fused kernel
│   │   ├── cuda_graphs.py             # CUDA graph wrapper
│   │   └── neighbor_search.py         # torch-cluster integration
│   ├── models/
│   │   ├── student_model.py           # Base model
│   │   └── student_model_optimized.py # Optimized version (NEW)
│   └── inference/
│       ├── inference_optimized.py     # Production inference (NEW)
│       └── ase_calculator.py          # ASE interface (updated)
├── benchmarks/
│   ├── benchmark_phase3.py            # Phase 3 benchmarks (NEW)
│   ├── profile_kernels.py             # Kernel profiling (NEW)
│   └── cuda_x_analysis/               # Analysis from planning
├── tests/
│   ├── test_custom_kernels.py         # Kernel unit tests (NEW)
│   └── test_numerical_equivalence.py  # Accuracy tests (NEW)
├── scripts/
│   ├── install_torch_cluster.sh       # torch-cluster setup (NEW)
│   └── validate_phase3.py             # End-to-end validation (NEW)
└── docs/
    ├── PHASE3_CUDA_GUIDE.md           # Implementation guide (NEW)
    └── DEPLOYMENT_GUIDE.md            # Production deployment (NEW)
```

---

## Testing Strategy

### Level 1: Unit Tests (Per Optimization)
- Numerical equivalence (<1e-5 tolerance)
- Gradient correctness (autograd compatible)
- Edge cases (empty graphs, large molecules)
- Memory leaks

### Level 2: Integration Tests
- End-to-end inference pipeline
- ASE calculator compatibility
- Batch processing
- Variable-size molecules

### Level 3: Performance Tests
- Benchmark suite (10, 50, 100 atoms)
- Latency, throughput, memory
- Comparison vs baseline
- Scaling analysis

### Level 4: Stability Tests
- MD simulations (1000+ steps)
- Energy conservation
- Force accuracy over time
- Numerical stability

---

## Documentation Requirements

### Week 1
- [ ] torch-cluster integration guide
- [ ] Benchmark results (Week 1)
- [ ] Updated README with optimization flags

### Week 2-3
- [ ] Triton kernel implementation guide
- [ ] Custom CUDA kernel documentation
- [ ] Profiling results and analysis

### Week 4
- [ ] Complete optimization guide
- [ ] Production deployment guide
- [ ] Performance tuning guide
- [ ] Final benchmark report

---

## Deliverables Summary

### Code Deliverables
1. torch-cluster integration
2. Triton fused message passing kernel
3. CUDA graphs implementation
4. Optimized inference pipeline
5. Comprehensive test suite

### Documentation Deliverables
1. Implementation guides
2. Benchmark reports
3. Deployment guide
4. Troubleshooting guide

### Validation Deliverables
1. Numerical accuracy validation
2. MD stability validation
3. Performance benchmarks
4. Scaling analysis

---

## Next Steps

**Immediate Actions** (Coordinator):
1. ✅ Create this coordination plan
2. ⏳ Create GitHub Issues #25, #26, #27, #28, #29
3. ⏳ Update project board with M5 milestone
4. ⏳ Brief CUDA optimization engineer on Week 1 tasks
5. ⏳ Brief testing engineer on validation requirements

**Week 1 Kickoff** (CUDA Engineer):
1. Install torch-cluster library
2. Replace neighbor search implementation
3. Run correctness tests
4. Benchmark performance
5. Report results and blockers

**Week 1 Parallel** (Testing Engineer):
1. Set up benchmark suite for Phase 3
2. Establish baseline measurements
3. Create accuracy validation tests
4. Prepare MD stability tests

---

## Status: READY TO BEGIN

User has approved. Coordination plan complete. Ready to execute Phase 3.

**Target**: 5-10x speedup
**Timeline**: 4 weeks
**Confidence**: Medium-High (conservative estimates, phased approach)

Let's achieve this goal!

---

**Coordinator**: Lead Project Coordinator
**Date Created**: 2025-11-24
**Last Updated**: 2025-11-24
**Status**: APPROVED - Execution Phase
