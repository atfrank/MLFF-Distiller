# Phase 3 CUDA Optimization - Agent Briefing

**Date**: 2025-11-24
**Meeting Type**: Project Kickoff
**Attendees**: CUDA Optimization Engineer, Testing & Benchmarking Engineer, Lead Coordinator

---

## Mission Statement

Achieve 5-10x inference speedup for the StudentForceField model through CUDA optimization, maintaining >95% accuracy and production-level stability.

**User Authorization**: Approved to proceed (2025-11-24)
**Timeline**: 4 weeks
**Current Baseline**: 2.0x speedup (TorchScript JIT)
**Target**: 5-10x total speedup

---

## Project Context

### What We've Achieved

**Phase 1** (torch.compile + FP16 attempts):
- torch.compile: Blocked by Python 3.13 incompatibility
- FP16: Working (1.37x speedup, acceptable accuracy)
- Status: Partial success

**Phase 2** (TorchScript JIT):
- TorchScript: Success (2.0x speedup, perfect accuracy)
- Production ready
- Status: Complete

**Phase 3** (CUDA optimization - NOW):
- Goal: 5-10x total speedup
- Strategy: Custom kernels + torch-cluster + CUDA graphs
- Status: Starting

### Why CUDA Optimization?

Comprehensive profiling analysis (`benchmarks/cuda_x_analysis/`) revealed:
1. CUDA-X libraries won't help (PyTorch already optimal)
2. Real bottlenecks: Neighbor search, message passing, kernel overhead
3. Solution: Custom kernels, optimized libraries, graph-level optimization

**Key insight**: cuBLAS and cuDNN already used by PyTorch. Gains come from algorithm-level optimization, not library swapping.

---

## Agent Roles & Responsibilities

### CUDA Optimization Engineer (Primary)

**Your mission**: Implement all CUDA optimizations to achieve 5-10x speedup

**Week 1 Tasks**:
- [ ] Install and integrate torch-cluster library
- [ ] Replace `radius_graph_native` with `torch_cluster.radius`
- [ ] Validate numerical correctness
- [ ] Benchmark performance improvement
- [ ] Report results (target: 3-5x total speedup by end of Week 1)

**Week 2-3 Tasks**:
- [ ] Design Triton fused message passing kernel
- [ ] Implement kernel with RBF + filter + aggregation fusion
- [ ] Test gradient correctness (autograd compatible)
- [ ] Benchmark performance improvement
- [ ] Report results (target: 5-8x total speedup by end of Week 3)

**Week 4 Tasks**:
- [ ] Implement CUDA graph capture for common molecule sizes
- [ ] Optimize kernel parameters (block size, shared memory)
- [ ] Final integration and testing
- [ ] Report results (target: 7-10x total speedup by end of Week 4)

**Key Deliverables**:
1. torch-cluster integration (Week 1)
2. Triton fused kernels (Week 2-3)
3. CUDA graphs implementation (Week 4)
4. Performance benchmarks at each stage

**GitHub Issues**:
- Issue #25: torch-cluster integration (Week 1)
- Issue #26: Triton fused kernels (Week 2-3)
- Issue #27: CUDA graphs (Week 4)

**Communication**:
- Daily: Update GitHub issues with progress
- Weekly: Checkpoint meeting with coordinator (Fridays)
- Blockers: @mention coordinator immediately in GitHub issue

**Resources**:
- Implementation plan: `benchmarks/cuda_x_analysis/IMPLEMENTATION_PLAN.md`
- Profiling data: `benchmarks/cuda_x_analysis/profiling_data.json`
- Current code: `src/mlff_distiller/models/student_model.py`
- Triton docs: https://triton-lang.org/

---

### Testing & Benchmarking Engineer (Secondary)

**Your mission**: Validate correctness and measure performance of all optimizations

**Week 1 Tasks**:
- [ ] Set up Phase 3 benchmark suite
- [ ] Establish baseline measurements (TorchScript + current optimizations)
- [ ] Create numerical accuracy tests (<1e-5 tolerance)
- [ ] Benchmark torch-cluster integration
- [ ] Report baseline data

**Week 2-3 Tasks**:
- [ ] Validate Triton kernel numerical equivalence
- [ ] Test gradient correctness (autograd)
- [ ] Run comprehensive benchmarks (10, 50, 100 atoms)
- [ ] Start MD stability validation (baseline)
- [ ] Report accuracy and performance data

**Week 4 Tasks**:
- [ ] Validate CUDA graphs correctness
- [ ] Complete MD stability testing (1000 steps)
- [ ] Final comprehensive benchmarks
- [ ] Generate Phase 3 final report
- [ ] Verify 5-10x speedup target achieved

**Key Deliverables**:
1. Baseline benchmark data (Week 1)
2. Per-optimization validation reports (Weeks 2-3)
3. MD stability validation (Week 3-4)
4. Final Phase 3 benchmark report (Week 4)

**GitHub Issues**:
- Issue #28: Comprehensive benchmark suite (ongoing)
- Issue #29: MD stability validation (Week 3-4)

**Communication**:
- Daily: Update GitHub issues with test results
- Weekly: Present validation data at checkpoint meeting
- Issues: Report any numerical errors or regressions immediately

**Resources**:
- Benchmark reference: `scripts/benchmark_optimizations.py`
- MD validation: `scripts/validate_md_optimized.py`
- Profiling: `benchmarks/profile_detailed.py`

---

## Week 1 Detailed Plan

### Goal

Achieve 3-5x total speedup through torch-cluster integration and optimization tuning.

### Monday-Tuesday (CUDA Engineer)

**Task**: Install and integrate torch-cluster

**Steps**:
1. Install torch-cluster:
   ```bash
   pip install torch-cluster -f https://data.pyg.org/whl/torch-2.x.x+cu121.html
   ```

2. Locate neighbor search code in `src/mlff_distiller/models/student_model.py`:
   ```python
   # Find radius_graph_native function
   def radius_graph_native(positions, batch, cutoff):
       # Current O(N²) implementation
   ```

3. Replace with torch-cluster:
   ```python
   from torch_cluster import radius

   def radius_graph_optimized(positions, batch, cutoff):
       edge_index = radius(
           positions, positions,
           r=cutoff,
           batch_x=batch, batch_y=batch,
           max_num_neighbors=128
       )
       return edge_index
   ```

4. Add configuration flag:
   ```python
   class StudentForceField:
       def __init__(self, ..., use_torch_cluster=False):
           self.use_torch_cluster = use_torch_cluster
   ```

5. Test on small molecule:
   ```bash
   python -c "
   from mlff_distiller.inference import StudentForceFieldCalculator
   calc = StudentForceFieldCalculator(
       checkpoint_path='checkpoints/best_model.pt',
       use_torch_cluster=True
   )
   # Test inference
   "
   ```

**Deliverable**: torch-cluster integrated, basic tests passing

### Wednesday (CUDA Engineer + Testing Engineer)

**Task**: Validate numerical correctness

**CUDA Engineer**:
- Write correctness test comparing old vs new neighbor search
- Verify edge lists are identical (or reordered but equivalent)
- Test on multiple system sizes (10, 50, 100 atoms)

**Testing Engineer**:
- Run comprehensive accuracy validation
- Compare energies and forces against baseline
- Target: <1e-5 eV error
- Document any discrepancies

**Deliverable**: Correctness validation complete

### Thursday (Testing Engineer)

**Task**: Benchmark performance

**Steps**:
1. Create benchmark script (extend `scripts/benchmark_optimizations.py`)
2. Test configurations:
   - Baseline (TorchScript JIT)
   - TorchScript + torch-cluster
   - TorchScript + torch-cluster + FP16
3. System sizes: 10, 20, 50, 100 atoms
4. Iterations: 100 per configuration
5. Export results to JSON

**Deliverable**: Performance data collected

### Friday (All)

**Task**: Week 1 checkpoint meeting

**Agenda**:
1. Review torch-cluster integration (CUDA engineer presents)
2. Review benchmark results (testing engineer presents)
3. Discuss Week 2 plan (Triton kernels)
4. Address any blockers
5. Go/no-go decision for Week 2

**Success criteria**:
- [ ] torch-cluster integrated and working
- [ ] 3-5x speedup achieved (or on track)
- [ ] No numerical errors
- [ ] Ready to start Triton kernels

---

## Week 2-3 Plan Overview

**Goal**: Achieve 5-8x total speedup through Triton fused message passing kernels

**CUDA Engineer**:
- Design kernel fusion strategy
- Implement Triton kernel
- Test gradient correctness
- Optimize kernel parameters

**Testing Engineer**:
- Validate numerical equivalence
- Benchmark performance
- Start MD stability tests
- Track cumulative speedup

**Checkpoints**: End of Week 2, End of Week 3

---

## Week 4 Plan Overview

**Goal**: Achieve 7-10x total speedup through CUDA graphs and final tuning

**CUDA Engineer**:
- Implement CUDA graph capture
- Handle multiple graph sizes
- Final kernel tuning
- Integration and polish

**Testing Engineer**:
- Complete MD stability validation
- Final comprehensive benchmarks
- Generate Phase 3 report
- Verify target achieved

**Final Checkpoint**: End of Week 4 (Project completion)

---

## Communication Protocol

### Daily Updates

**Format** (GitHub issue comment):
```
## Daily Update - [Date]

### Progress Today
- [x] Task 1 completed
- [ ] Task 2 in progress (60% done)
- [x] Task 3 completed

### Blockers
- None / [Describe blocker]

### Plan for Tomorrow
- Complete Task 2
- Start Task 4
- Run benchmarks

### Questions
- [Any questions for coordinator or other agents]
```

**When to update**: End of day (or earlier if blocked)

### Weekly Checkpoints

**When**: Every Friday, 4pm
**Duration**: 30-60 minutes
**Format**: Async (GitHub discussion thread) or sync (video call)

**Agenda**:
1. CUDA engineer: Demo and progress report
2. Testing engineer: Validation results
3. Coordinator: Review metrics and adjust plan
4. Discussion: Blockers, questions, next week plan
5. Decision: Go/no-go for next phase

### Blocker Escalation

**If you're blocked**, immediately:
1. Document blocker in GitHub issue
2. Add label: `blocked`
3. @mention coordinator: `@lead-coordinator`
4. Propose 2-3 possible solutions
5. Coordinator will respond within 4 hours

**Types of blockers**:
- Technical: Can't figure out how to implement something
- Resource: Missing library, hardware, documentation
- Dependency: Waiting on another agent
- Clarification: Need architectural decision

---

## Success Metrics

### Week 1 Success
- [ ] torch-cluster integrated
- [ ] 3-5x speedup achieved (2.0x → 6-10x cumulative)
- [ ] Accuracy maintained
- [ ] No blockers for Week 2

### Week 2-3 Success
- [ ] Triton kernels implemented
- [ ] 5-8x speedup achieved
- [ ] Gradient correctness validated
- [ ] MD stable (baseline tests)

### Week 4 Success
- [ ] CUDA graphs working
- [ ] 7-10x speedup achieved (TARGET MET!)
- [ ] MD stable (1000 steps)
- [ ] Documentation complete

### Overall Project Success
- [ ] 5-10x speedup target achieved
- [ ] >95% accuracy maintained
- [ ] Production ready
- [ ] Complete test coverage
- [ ] Comprehensive documentation

---

## Risk Awareness

### Risk 1: Custom kernels break accuracy
**Mitigation**: Rigorous testing at every step, <1e-5 tolerance
**Your action**: Report any numerical errors immediately

### Risk 2: Timeline slips
**Mitigation**: Deliver incremental improvements each week
**Your action**: Update estimates if tasks take longer than expected

### Risk 3: Technical complexity
**Mitigation**: Use Triton (easier than CUDA C++), leverage libraries
**Your action**: Ask for help early, don't struggle in silence

---

## Key Files Reference

### Code
- Student model: `src/mlff_distiller/models/student_model.py`
- Inference: `src/mlff_distiller/inference/ase_calculator.py`
- CUDA kernels (NEW): `src/mlff_distiller/cuda/`

### Documentation
- Implementation plan: `benchmarks/cuda_x_analysis/IMPLEMENTATION_PLAN.md`
- Coordination plan: `PHASE3_COORDINATION_PLAN.md`
- GitHub issues: `PHASE3_GITHUB_ISSUES.md`

### Benchmarking
- Benchmark script: `scripts/benchmark_optimizations.py`
- Profiling: `benchmarks/profile_detailed.py`
- Results: `benchmarks/optimization_results.json`

### Analysis
- CUDA-X analysis: `benchmarks/cuda_x_analysis/CUDA_X_RECOMMENDATIONS.md`
- Profiling data: `benchmarks/cuda_x_analysis/profiling_data.json`
- Executive summary: `benchmarks/cuda_x_analysis/EXECUTIVE_SUMMARY.md`

---

## Questions & Answers

### Q: What if torch-cluster doesn't give 3-5x speedup?

**A**: That's okay! The estimate includes all Week 1 optimizations combined. If torch-cluster gives 1.5x, that's still progress. Report the actual numbers, and we'll adjust the plan. The goal is 5-10x by end of Week 4, not necessarily by Week 1.

### Q: What if Triton kernels are too hard to implement?

**A**: Start with a simple version that works, even if not fully fused. We can iterate. Alternatively, focus on torch-cluster + CUDA graphs if Triton is too complex. Coordinator will help prioritize.

### Q: What if MD simulations are unstable?

**A**: Document the instability (energy drift, force spikes, etc.) and report immediately. We may need to reduce kernel fusion or adjust optimization strategy. Production stability is a hard requirement.

### Q: What if we only achieve 5x instead of 10x?

**A**: That's success! The target range is 5-10x. Achieving 5x meets the goal. Anything above is bonus. Document what we achieved and what could be done for further speedup.

### Q: How much time should I spend on documentation?

**A**: Document as you go:
- Code comments: Inline (5-10% of coding time)
- README updates: Weekly (30 min)
- Benchmark reports: Automated (built into scripts)
- Final documentation: Week 4 (2-4 hours)

Don't let documentation block progress, but don't skip it entirely.

---

## Final Notes

**This is an ambitious project**, but we have:
- Clear roadmap (4-week plan)
- Strong baseline (2.0x already achieved)
- Good tools (Triton, torch-cluster)
- Profiling data (know where to optimize)

**You are empowered to**:
- Make implementation decisions within your domain
- Ask for help when needed
- Adjust approach if initial plan doesn't work
- Propose alternative solutions

**Coordinator will**:
- Unblock you quickly
- Make architectural decisions
- Track overall progress
- Adjust timeline if needed

**Let's achieve 5-10x speedup!**

---

## Immediate Next Steps

### CUDA Optimization Engineer
1. Read implementation plan: `benchmarks/cuda_x_analysis/IMPLEMENTATION_PLAN.md`
2. Review current code: `src/mlff_distiller/models/student_model.py`
3. Install torch-cluster: `pip install torch-cluster`
4. Start Issue #25 implementation
5. Update GitHub issue with progress by end of Monday

### Testing & Benchmarking Engineer
1. Review baseline results: `benchmarks/optimization_results.json`
2. Set up benchmark environment
3. Create test molecule dataset (10, 20, 50, 100 atoms)
4. Establish baseline measurements (TorchScript)
5. Update GitHub issue #28 with baseline data by end of Tuesday

### Coordinator (Me)
1. Monitor daily updates
2. Be available for questions
3. Review PRs quickly
4. Prepare for Friday checkpoint

---

**Ready to begin? Questions?**

Post questions in GitHub issue comments or @mention coordinator.

**Let's build the fastest ML force field inference engine!**

---

**Briefing Date**: 2025-11-24
**Briefing By**: Lead Project Coordinator
**Status**: Phase 3 Kickoff - APPROVED TO PROCEED
