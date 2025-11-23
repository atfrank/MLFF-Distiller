# Week 2 Coordination Plan
## ML Force Field Distillation Project

**Date**: 2025-11-23
**Project**: MLFF Distiller (https://github.com/atfrank/MLFF-Distiller)
**Repository**: /home/aaron/ATX/software/MLFF_Distiller
**Coordinator**: Lead Project Coordinator

---

## Executive Summary

Week 1 delivered exceptional results: 5 critical issues completed, 4,292 lines of production code, and 181 passing tests. The foundation is solid. Week 2 focuses on **completing Milestone 1 (M1)** by building the remaining critical infrastructure: MD benchmarking, student ASE calculator interface, and performance profiling.

**M1 Target**: December 6, 2025 (13 days remaining)
**Week 2 Goal**: Complete all 4 remaining M1 issues + integration validation

---

## Week 1 Accomplishments - VERIFIED

### Completed Issues (5/9 M1 Issues)

1. **Issue #1: Data Loading Infrastructure** ✅
   - Agent: Data Pipeline Engineer
   - Delivered: MolecularDataset with GPU batching, transforms, lazy loading
   - Files: `src/mlff_distiller/data/` (dataset.py, loaders.py, transforms.py)
   - Tests: 34 unit tests passing
   - Quality: Production-ready with comprehensive fixtures

2. **Issue #2: Teacher Model Wrappers** ✅ [CRITICAL PATH]
   - Agent: ML Architecture Designer
   - Delivered: OrbCalculator and FeNNolCalculator with ASE interface
   - File: `src/mlff_distiller/models/teacher_wrappers.py` (425 lines)
   - Tests: Unit tests + integration tests with mock MD
   - Quality: Drop-in ASE compatibility verified
   - Impact: UNBLOCKED 4 downstream issues

3. **Issue #3: Training Framework** ✅
   - Agent: Distillation Training Engineer
   - Delivered: Complete Trainer with checkpointing, logging, early stopping
   - Files: `src/mlff_distiller/training/` (trainer.py, losses.py, config.py)
   - Tests: 44 trainer tests, 20 loss tests, 28 config tests
   - Quality: MD-focused losses (force weight 100:1), Pydantic validation

4. **Issue #4: Pytest Infrastructure** ✅
   - Agent: Testing & Benchmark Engineer
   - Delivered: pytest setup, fixtures, conftest with reusable components
   - File: `tests/conftest.py` (320 lines of fixtures)
   - Tests: 181 tests passing (96 unit, 85+ integration/accuracy)
   - Quality: GPU fixtures, mock models, comprehensive test utilities

5. **Issue #8: CUDA Environment** ✅
   - Agent: CUDA Optimization Engineer
   - Delivered: Device utilities, memory tracking, benchmarking tools
   - Files: `src/mlff_distiller/cuda/` (device_utils.py, benchmark_utils.py)
   - Tests: 28 CUDA utility tests
   - Quality: Memory leak detection, GPU info queries

### Code Statistics - VERIFIED

- **Production Code**: 4,292 lines in `src/mlff_distiller/`
- **Test Code**: 181 tests passing (100% success rate)
- **Test Coverage**: Unit + Integration + Accuracy tests
- **Integration Fix**: Corrected import paths (src.* → mlff_distiller.*)

### Repository Health - EXCELLENT

```
Repository: /home/aaron/ATX/software/MLFF_Distiller
Commit: 4ff20d9 (Week 1 complete)
Branch: main
Status: Clean working directory
CI: All checks passing
Python: 3.13.9
PyTorch: CUDA available
```

---

## Week 2 Critical Analysis

### Remaining M1 Issues (4 Open)

**Issue #5: MD Simulation Benchmark Framework** [CRITICAL]
- Agent: Testing & Benchmark Engineer
- Priority: HIGHEST for Week 2
- Why Critical: Defines success metrics for entire project
- Complexity: Medium-High
- Dependencies: Issue #2 (teacher wrappers) ✅ UNBLOCKED
- Deliverables:
  - MD trajectory simulation framework
  - Latency benchmarking (per-call timing)
  - Memory profiling over long trajectories
  - Energy conservation validation (NVE ensemble)
  - Baseline performance database
- Template: `docs/initial_issues/issue_22_md_benchmark_framework.md`

**Issue #6: Student ASE Calculator Interface** [CRITICAL]
- Agent: ML Architecture Designer
- Priority: HIGHEST for Week 2
- Why Critical: Drop-in replacement requirement, blocks M3 work
- Complexity: Medium
- Dependencies: Issue #2 (teacher wrappers) ✅ UNBLOCKED
- Deliverables:
  - StudentCalculator(Calculator) base class
  - Same interface as teacher calculators
  - Configurable model backend
  - Works with existing ASE MD scripts
  - Template for all future student models
- Template: `docs/initial_issues/issue_26_ase_calculator_student.md`

**Issue #7: ASE Calculator Interface Tests** [CRITICAL]
- Agent: Testing & Benchmark Engineer
- Priority: HIGH for Week 2
- Why Critical: Validates drop-in compatibility
- Complexity: Medium
- Dependencies: Issue #2 ✅ UNBLOCKED, Issue #6 (student calculator)
- Deliverables:
  - Interface compatibility tests
  - Comparison tests (teacher vs student)
  - Property verification (energy, forces, stress)
  - MD simulation tests with both calculators
  - Regression tests for interface changes
- Template: `docs/initial_issues/issue_29_ase_interface_tests.md`

**Issue #9: MD Profiling Framework** [HIGH PRIORITY]
- Agent: CUDA Optimization Engineer
- Priority: HIGH for Week 2
- Why Important: Establishes performance baseline for optimization
- Complexity: Medium
- Dependencies: Issue #2 ✅ UNBLOCKED, Issue #8 (CUDA env) ✅ COMPLETE
- Deliverables:
  - MD trajectory profiling tools
  - Per-call latency measurement
  - Memory usage tracking during MD
  - Hotspot identification
  - Performance report generation
- Template: `docs/initial_issues/issue_17_profiling_framework.md`

### Dependency Analysis

```
Week 2 Dependency Graph:

Issue #2 (Teacher Wrappers) ✅ COMPLETE
    │
    ├──> Issue #5 (MD Benchmarks) [READY TO START]
    │       └──> Defines success metrics
    │
    ├──> Issue #6 (Student Calculator) [READY TO START]
    │       └──> Issue #7 (Interface Tests) [DEPENDS ON #6]
    │
    └──> Issue #9 (MD Profiling) [READY TO START]

CRITICAL PATH: Issues #5, #6, #7 must complete for M1
HIGH VALUE: Issue #9 informs optimization strategy
```

### Integration Risks - MITIGATED

**Risk 1: Interface Incompatibility** (MEDIUM → LOW)
- *Risk*: Student calculator doesn't match teacher interface
- *Mitigation*:
  - Use teacher_wrappers.py as template
  - Issue #7 tests validate compatibility
  - Weekly integration checkpoints

**Risk 2: Unrealistic Benchmarks** (MEDIUM)
- *Risk*: Benchmarks don't reflect real MD workloads
- *Mitigation*:
  - Focus on trajectory benchmarks (not single calls)
  - Test with realistic system sizes (100-1000 atoms)
  - Validate energy conservation

**Risk 3: Agent Coordination** (LOW)
- *Risk*: Isolated work causes integration failures
- *Mitigation*:
  - Clear interface contracts in templates
  - Mid-week integration checkpoint (Day 4)
  - Shared fixtures in conftest.py

---

## Week 2 Schedule

### Daily Breakdown

**Day 1-2 (Mon-Tue): Parallel Kickoff**
- All 4 issues start simultaneously
- Focus: Get infrastructure in place
- Checkpoint: Each agent confirms template understood

**Day 3-4 (Wed-Thu): Implementation**
- Issue #5: MD simulation loop + timing
- Issue #6: StudentCalculator skeleton + interface
- Issue #9: Profiling tools + memory tracking
- Checkpoint: Mid-week integration meeting

**Day 5-6 (Fri-Sat): Integration & Testing**
- Issue #7: Interface tests using #6
- Issue #5: Baseline benchmarks with teacher models
- All: Integration validation
- Checkpoint: M1 completion assessment

**Day 7 (Sun): Buffer & Documentation**
- Fix integration issues
- Update documentation
- Prepare M2 kickoff

### Milestones by End of Week

- [ ] Issue #5 COMPLETE: MD benchmarks running
- [ ] Issue #6 COMPLETE: Student calculator interface ready
- [ ] Issue #7 COMPLETE: Interface tests passing
- [ ] Issue #9 COMPLETE: Profiling framework operational
- [ ] M1 100% COMPLETE (9/9 issues)
- [ ] Integration tests passing
- [ ] Documentation updated

---

## Agent Activation Plan

### Agent 2: ML Architecture Designer

**Assignment**: Issue #6 - Student ASE Calculator Interface

**Context**:
- Your teacher wrappers (Issue #2) are excellent and production-ready
- Now create the student model version with the same ASE interface
- This is the template for ALL future student models

**Key Requirements**:
1. StudentCalculator(Calculator) base class
2. Identical interface to OrbCalculator/FeNNolCalculator
3. Configurable backend (placeholder for now, real model in M3)
4. Drop-in replacement capability (one line change in MD scripts)

**Starting Point**:
- Use `src/mlff_distiller/models/teacher_wrappers.py` as template
- Reuse ASE Calculator patterns
- Integration with existing fixtures in `tests/conftest.py`

**Deliverables**:
- `src/mlff_distiller/models/student_calculator.py`
- Unit tests for interface compliance
- Example MD script showing drop-in usage

**Template**: `docs/initial_issues/issue_26_ase_calculator_student.md`

---

### Agent 5: Testing & Benchmark Engineer

**Assignment**: Issue #5 - MD Simulation Benchmark Framework (PRIMARY)
**Assignment**: Issue #7 - ASE Calculator Interface Tests (SECONDARY)

**Context**:
- You delivered excellent pytest infrastructure (Issue #4)
- Teacher wrappers (Issue #2) are ready for benchmarking
- This framework defines project success metrics

**Key Requirements** (Issue #5):
1. Run realistic MD trajectories (1000+ steps)
2. Measure per-call latency (critical for MD)
3. Track memory usage over long runs
4. Validate energy conservation (NVE)
5. Create baseline performance database

**Key Requirements** (Issue #7):
1. Test teacher/student calculator equivalence
2. Validate ASE interface compliance
3. Compare properties (energy, forces, stress)
4. MD simulation tests with both calculators
5. Depends on Issue #6 completion

**Starting Point**:
- Use teacher wrappers from `src/mlff_distiller/models/teacher_wrappers.py`
- Leverage fixtures in `tests/conftest.py`
- Build on existing integration tests

**Deliverables** (Issue #5):
- `benchmarks/md_benchmark.py` - Main benchmark script
- `src/mlff_distiller/benchmarks/` - Benchmarking utilities
- Performance baseline JSON database
- Benchmark report template

**Deliverables** (Issue #7):
- `tests/integration/test_ase_interface.py`
- Calculator comparison tests
- Interface compliance tests

**Strategy**: Start Issue #5 immediately. Begin Issue #7 when Issue #6 (student calculator) is ready (likely Day 3-4).

**Templates**:
- `docs/initial_issues/issue_22_md_benchmark_framework.md`
- `docs/initial_issues/issue_29_ase_interface_tests.md`

---

### Agent 4: CUDA Optimization Engineer

**Assignment**: Issue #9 - MD Profiling Framework

**Context**:
- Your CUDA environment (Issue #8) is solid foundation
- Teacher wrappers (Issue #2) ready for profiling
- This establishes baseline for optimization work in M4-M5

**Key Requirements**:
1. Profile MD trajectory execution (not single inference)
2. Measure per-call latency distribution
3. Track memory allocation/deallocation patterns
4. Identify computational hotspots
5. Generate performance reports

**Starting Point**:
- Use CUDA utilities from `src/mlff_distiller/cuda/`
- Profile teacher wrappers from `src/mlff_distiller/models/teacher_wrappers.py`
- Leverage benchmark_utils.py you created

**Deliverables**:
- `src/mlff_distiller/cuda/profiler.py` - MD profiling tools
- `benchmarks/profile_md.py` - Profiling script
- Profiling report template
- Baseline performance characterization

**Template**: `docs/initial_issues/issue_17_profiling_framework.md`

---

## Integration Checkpoints

### Mid-Week Integration (Day 4)

**Objectives**:
- Verify Issue #6 (student calculator) skeleton works
- Confirm Issue #5 (benchmarks) can run teacher models
- Validate Issue #9 (profiling) captures MD metrics

**Process**:
1. Run integration test suite
2. Test student calculator with simple model
3. Benchmark teacher model MD trajectory
4. Profile teacher model performance
5. Document any integration issues

**Success Criteria**:
- Student calculator loads without errors
- MD benchmark completes 100 step trajectory
- Profiler captures timing data
- No blocking issues

### End of Week Integration (Day 7)

**Objectives**:
- Complete M1 validation
- Verify all 9 issues integrated
- Prepare M2 kickoff

**Process**:
1. Run full test suite (target: 200+ tests)
2. Verify student calculator works in Issue #7 tests
3. Generate baseline benchmark report
4. Create M1 completion report

**Success Criteria**:
- All 181+ tests passing
- Issue #7 interface tests passing
- Baseline benchmarks documented
- M1 marked complete

---

## Quality Standards for Week 2

### Code Quality
- [ ] All new code has tests (>80% coverage)
- [ ] Type hints on all public APIs
- [ ] Docstrings with usage examples
- [ ] Follows patterns from Week 1 code
- [ ] Consistent import paths (mlff_distiller.*)

### Performance Standards
- [ ] MD benchmarks test 1000+ step trajectories
- [ ] Per-call latency measured (not just total time)
- [ ] Memory stability verified over long runs
- [ ] Energy conservation validated (NVE)

### Interface Standards
- [ ] Student calculator has identical interface to teacher
- [ ] ASE Calculator methods implemented correctly
- [ ] Works with ASE MD integrators
- [ ] Drop-in replacement verified with tests

### Documentation Standards
- [ ] Each issue has clear PR description
- [ ] Usage examples in docstrings
- [ ] Benchmark results documented
- [ ] Integration notes in PR

---

## Success Metrics for Week 2

### Completion Metrics
- [ ] 4/4 remaining M1 issues COMPLETE
- [ ] 9/9 total M1 issues COMPLETE
- [ ] M1 marked complete (Dec 6 deadline met early)
- [ ] 200+ total tests passing
- [ ] No unresolved blockers

### Technical Metrics
- [ ] Teacher model MD benchmark: <X ms/call latency
- [ ] Memory stable over 10,000 MD steps
- [ ] Energy conservation: <1e-6 eV drift/atom/ps
- [ ] Student calculator interface tests: 100% pass
- [ ] Integration tests: 100% pass

### Process Metrics
- [ ] All PRs reviewed within 24 hours
- [ ] CI passing on all PRs
- [ ] No issues blocked >24 hours
- [ ] Mid-week checkpoint completed
- [ ] Weekly integration successful

---

## Risk Mitigation

### High Risk Items

**Risk**: Issue #6 (student calculator) delays Issue #7 (interface tests)
- *Probability*: LOW (straightforward task)
- *Impact*: MEDIUM (blocks one test issue)
- *Mitigation*:
  - Start Issue #6 on Day 1
  - Use teacher wrappers as template
  - Daily progress checks
  - Issue #7 starts Day 3-4 regardless (can test teachers first)

**Risk**: MD benchmarks not realistic
- *Probability*: MEDIUM
- *Impact*: HIGH (wrong success metrics)
- *Mitigation*:
  - Review MD requirements doc before starting
  - Test with multiple system sizes
  - Validate with energy conservation
  - Compare to literature values

### Medium Risk Items

**Risk**: Integration issues between new components
- *Probability*: LOW (good patterns established)
- *Impact*: MEDIUM (delays M1 completion)
- *Mitigation*:
  - Mid-week integration checkpoint
  - Shared fixtures in conftest.py
  - Clear interface contracts

**Risk**: Performance profiling overhead
- *Probability*: LOW
- *Impact*: LOW (doesn't block progress)
- *Mitigation*:
  - Use lightweight profiling tools
  - Selective instrumentation
  - Separate profiling from production code

---

## Communication Protocol

### Daily Updates
- Each agent updates issue with progress comment
- Format: "Progress: [done], Next: [next], Blockers: [none]"
- Coordinator reviews daily by 9 AM

### Blocker Resolution
- Tag "status:blocked" immediately
- @mention coordinator in comment
- Resolution target: <4 hours for M1 critical path
- Escalation: Direct communication if >4 hours

### PR Process
1. Agent creates PR referencing issue
2. PR description includes: changes, tests, integration notes
3. CI must pass (linting, tests)
4. Coordinator reviews within 24 hours
5. Agent addresses comments
6. Coordinator approves and merges

### Integration Issues
- Report in issue comments immediately
- Tag "status:integration-issue"
- Coordinator coordinates fix between agents
- Document resolution in issue

---

## Post-Week 2 Outlook

### M1 Completion
- Week 2 completes M1 (all 9 issues)
- M1 due date: Dec 6, 2025 (met 9 days early!)
- Baseline infrastructure ready for distillation

### M2 Preview (Weeks 3-4)
- Begin data generation from teacher models
- Student architecture design
- HDF5 dataset storage
- Architecture analysis complete

### M3 Preview (Weeks 5-8)
- Student model implementation
- Distillation training
- Performance optimization begins

---

## Resources & References

### Key Files
- **Project Repository**: /home/aaron/ATX/software/MLFF_Distiller
- **Week 1 Code**: src/mlff_distiller/ (4,292 lines)
- **Test Suite**: tests/ (181 tests passing)
- **Issue Templates**: docs/initial_issues/
- **Requirements**: docs/MD_REQUIREMENTS.md, docs/INTERFACE_REQUIREMENTS.md

### Documentation
- `docs/PROJECT_KICKOFF.md` - Full project overview
- `docs/MILESTONES.md` - Milestone definitions
- `docs/AGENT_PROTOCOLS.md` - Agent responsibilities
- `TESTING.md` - Testing strategy and patterns

### GitHub
- **Repository**: https://github.com/atfrank/MLFF-Distiller
- **Commit**: 4ff20d9 (Week 1 complete)
- **CI**: GitHub Actions (all passing)

---

## Coordinator Action Items

### Immediate (Day 1)
- [x] Fix import errors in test files (COMPLETE)
- [ ] Create GitHub Issues #5, #6, #7, #9
- [ ] Assign issues to agents
- [ ] Send activation messages to agents
- [ ] Update GitHub Project board

### Daily (Days 1-7)
- [ ] Review issue comments and progress
- [ ] Check PR status and CI results
- [ ] Identify and resolve blockers
- [ ] Update project board
- [ ] Provide feedback and guidance

### Mid-Week (Day 4)
- [ ] Run integration checkpoint
- [ ] Verify components working together
- [ ] Adjust schedule if needed
- [ ] Communicate status to stakeholders

### End of Week (Day 7)
- [ ] Run full integration validation
- [ ] Mark M1 complete
- [ ] Generate Week 2 completion report
- [ ] Plan M2 kickoff
- [ ] Celebrate team success!

---

## Appendix A: Test Suite Status

```
Week 1 Final Test Results:
========================
Total Tests: 181 passing, 11 skipped
Unit Tests: 96 passing
Integration Tests: 85+ passing
Test Files:
  - tests/unit/test_dataset.py: 34 tests
  - tests/unit/test_trainer.py: 44 tests
  - tests/unit/test_losses.py: 20 tests
  - tests/unit/test_config.py: 28 tests
  - tests/unit/test_cuda_device_utils.py: 28 tests (4 skipped)
  - tests/unit/test_teacher_wrappers.py: Unit tests
  - tests/integration/test_teacher_wrappers_md.py: MD integration tests
  - tests/integration/test_ase_integration_demo.py: ASE demo tests
  - tests/accuracy/*: Accuracy validation tests

Week 2 Target: 200+ tests passing
```

---

## Appendix B: Code Statistics

```
Production Code (src/mlff_distiller/):
====================================
Total: 4,292 lines

By Module:
- data/: ~1,200 lines (dataset, loaders, transforms)
- models/: ~425 lines (teacher_wrappers)
- training/: ~1,650 lines (trainer, losses, config)
- cuda/: ~800 lines (device_utils, benchmark_utils)
- utils/: Minimal (shared utilities)

Test Code (tests/):
==================
Total: 181 tests in multiple files
- conftest.py: 320 lines (fixtures and utilities)
- Unit tests: Comprehensive coverage
- Integration tests: MD trajectories, ASE interface
```

---

**Week 2 Mission**: Complete M1, establish performance baselines, enable M2 distillation work.

**Week 2 Motto**: "Fast, accurate, drop-in compatible - measure twice, optimize once!"

---
*Document Version: 1.0*
*Last Updated: 2025-11-23*
*Coordinator: Lead Project Coordinator*
