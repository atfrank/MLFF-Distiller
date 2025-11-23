# Week 2 Agent Activation Instructions
## ML Force Field Distillation Project

**Date**: 2025-11-23
**Week**: 2 of 14
**Milestone**: M1 Completion (9/9 issues)
**Repository**: /home/aaron/ATX/software/MLFF_Distiller

---

## Mission Statement

Week 1 was exceptional - you delivered production-quality code that passed all integration tests. Week 2 is equally critical: we complete M1 by building the performance measurement infrastructure and student model interface. These components define our success metrics and enable the distillation work in M2-M3.

**Week 2 Goal**: Complete M1 (4 remaining issues), establish baselines, prepare for distillation.

---

## Week 1 Achievements - Recap

Your Week 1 work is live in the repository at `/home/aaron/ATX/software/MLFF_Distiller`:

- **4,292 lines** of production code
- **181 tests** passing (100% success rate)
- **5 issues** completed (Issues #1, #2, #3, #4, #8)
- **Commit**: 4ff20d9 pushed to GitHub

All agents delivered on time with high quality. Well done!

---

## Week 2 Agent Assignments

### Agent 2: ML Architecture Designer

**Your Week 1 Success**:
- Issue #2 (Teacher Model Wrappers) - COMPLETE
- Delivered excellent OrbCalculator and FeNNolCalculator
- 425 lines of clean, well-documented code
- ASE Calculator interface perfectly implemented

**Your Week 2 Assignment**:

**Issue #6: Student ASE Calculator Interface** [CRITICAL]

**Why This Matters**:
- Core drop-in replacement requirement
- Template for ALL future student models
- Enables M3 distillation work
- Blocks Issue #7 (interface tests)

**What You're Building**:
A StudentCalculator class with identical ASE interface to your teacher wrappers, but configured to load distilled models instead. Users should be able to replace teacher calculators with student calculators by changing ONE line.

**Template Location**:
`/home/aaron/ATX/software/MLFF_Distiller/docs/initial_issues/issue_26_ase_calculator_student.md`

**Starting Point**:
- Your teacher wrappers: `src/mlff_distiller/models/teacher_wrappers.py`
- Reuse the ASE Calculator patterns
- Use fixtures from `tests/conftest.py`

**Key Deliverables**:
1. `src/mlff_distiller/models/student_calculator.py`
   - StudentCalculator(Calculator) base class
   - Same interface as OrbCalculator/FeNNolCalculator
   - Configurable model backend (placeholder for now)

2. Tests:
   - Unit tests for interface compliance
   - Mock model backend for testing
   - Example showing drop-in usage

3. Documentation:
   - Clear docstrings with examples
   - Drop-in replacement guide

**Success Criteria**:
- [ ] StudentCalculator implements ASE Calculator interface
- [ ] Same initialization parameters as teacher calculators
- [ ] get_potential_energy(), get_forces(), get_stress() methods work
- [ ] Can run simple MD trajectory (even with placeholder model)
- [ ] Tests pass and demonstrate drop-in compatibility
- [ ] Code follows Week 1 quality standards

**Timeline**:
- Day 1-2: Design and implement StudentCalculator skeleton
- Day 3-4: Add tests and integration with fixtures
- Day 5: Documentation and examples
- Day 6: Buffer for Issue #7 support

**Dependencies**:
- Issue #2 (teacher wrappers) ✅ COMPLETE (your own work!)
- Blocks: Issue #7 (interface tests)

**Integration Notes**:
- Agent 5 will use your StudentCalculator in Issue #7 tests
- Coordinate with Agent 5 starting Day 3-4
- Ensure interface matches teacher calculators exactly

---

### Agent 5: Testing & Benchmark Engineer

**Your Week 1 Success**:
- Issue #4 (Pytest Infrastructure) - COMPLETE
- Delivered comprehensive test suite (181 tests passing)
- Excellent fixtures in conftest.py (320 lines)
- Set foundation for all testing

**Your Week 2 Assignments**:

**PRIMARY: Issue #5 - MD Simulation Benchmark Framework** [CRITICAL]

**Why This Matters**:
- Defines success metrics for entire project (5-10x speedup target)
- Required for baseline performance characterization
- Critical for measuring progress in M4-M5 optimization
- Validates energy conservation (correctness check)

**What You're Building**:
A comprehensive MD benchmarking framework that measures performance on realistic trajectories (1000+ steps), not just single inference calls. This framework will track latency, memory, and energy conservation.

**Template Location**:
`/home/aaron/ATX/software/MLFF_Distiller/docs/initial_issues/issue_22_md_benchmark_framework.md`

**Starting Point**:
- Teacher wrappers: `src/mlff_distiller/models/teacher_wrappers.py`
- Your fixtures: `tests/conftest.py`
- Existing integration tests: `tests/integration/test_teacher_wrappers_md.py`

**Key Deliverables**:
1. `benchmarks/md_benchmark.py`
   - Run MD trajectories with timing
   - Support NVE, NVT, NPT protocols
   - Multiple system sizes (32-1024 atoms)
   - Per-step latency measurement

2. `src/mlff_distiller/benchmarks/` (utilities module)
   - Benchmark result classes
   - Memory profiling tools
   - JSON export/import
   - Plotting utilities

3. Baseline Performance Database:
   - JSON file with teacher model baselines
   - Multiple system sizes
   - Multiple protocols

4. Tests:
   - Unit tests for benchmark utilities
   - Integration test running full benchmark

**Success Criteria**:
- [ ] Benchmark runs 1000+ step MD trajectories
- [ ] Measures per-call latency (not just total time)
- [ ] Tracks memory usage over trajectory
- [ ] Validates energy conservation (NVE)
- [ ] Tests multiple system sizes
- [ ] Generates JSON output
- [ ] Documentation with usage examples
- [ ] Baseline results for teacher models

**Timeline**:
- Day 1-2: Core benchmark framework (MD simulation + timing)
- Day 3-4: Memory profiling, JSON output, utilities
- Day 5: Baseline measurements with teacher models
- Day 6-7: Documentation, plots, integration

**SECONDARY: Issue #7 - ASE Calculator Interface Tests** [CRITICAL]

**Why This Matters**:
- Validates drop-in compatibility requirement
- Tests teacher/student equivalence
- Regression tests for interface changes

**What You're Building**:
Comprehensive tests that verify student calculators work identically to teacher calculators in ASE MD simulations.

**Template Location**:
`/home/aaron/ATX/software/MLFF_Distiller/docs/initial_issues/issue_29_ase_interface_tests.md`

**Key Deliverables**:
1. `tests/integration/test_ase_interface.py`
   - Interface compliance tests
   - Teacher/student comparison tests
   - MD trajectory tests with both calculators
   - Property verification (energy, forces, stress)

**Success Criteria**:
- [ ] Tests verify ASE Calculator interface compliance
- [ ] Tests compare teacher and student calculators
- [ ] Tests run MD trajectories with both
- [ ] Tests pass with current teacher calculators
- [ ] Tests ready for student calculator (from Issue #6)

**Timeline**:
- Day 3-4: Start when Issue #6 (student calculator) ready
- Day 5-6: Complete tests and integration
- Day 7: Documentation

**Dependencies**:
- Issue #2 (teacher wrappers) ✅ COMPLETE
- Issue #6 (student calculator) - IN PROGRESS (Agent 2)
- **Coordinate with Agent 2 starting Day 3**

**Strategy**:
- **Days 1-3**: Focus 100% on Issue #5 (benchmarks)
- **Days 4-6**: Add Issue #7 (interface tests) in parallel
- Issue #7 can start with teacher-only tests, add student when ready

**Integration Notes**:
- Use Agent 2's StudentCalculator when available
- Start with teacher calculator tests (don't wait)
- Coordinate with Agent 2 on Day 3-4 for student integration

---

### Agent 4: CUDA Optimization Engineer

**Your Week 1 Success**:
- Issue #8 (CUDA Environment) - COMPLETE
- Delivered solid CUDA utilities and benchmarking tools
- Memory tracking, device utilities working great
- Foundation for optimization work

**Your Week 2 Assignment**:

**Issue #9: MD Profiling Framework** [HIGH PRIORITY]

**Why This Matters**:
- Establishes performance baseline for optimization
- Identifies computational hotspots
- Informs M4-M5 CUDA kernel work
- Critical for latency optimization strategy

**What You're Building**:
MD-specific profiling tools that measure performance during trajectory execution, not just single inference. This framework will guide optimization decisions in M4-M5.

**Template Location**:
`/home/aaron/ATX/software/MLFF_Distiller/docs/initial_issues/issue_17_profiling_framework.md`

**Starting Point**:
- Your CUDA utilities: `src/mlff_distiller/cuda/device_utils.py`
- Your benchmark utilities: `src/mlff_distiller/cuda/benchmark_utils.py`
- Teacher wrappers: `src/mlff_distiller/models/teacher_wrappers.py`

**Key Deliverables**:
1. `src/mlff_distiller/cuda/profiler.py`
   - MD trajectory profiling tools
   - Per-call latency distribution
   - Memory allocation tracking
   - Hotspot identification
   - PyTorch profiler integration

2. `benchmarks/profile_md.py`
   - Profiling script for MD trajectories
   - Support for teacher and student models
   - Multiple system sizes

3. Profiling Reports:
   - Performance characterization of teacher models
   - Hotspot analysis
   - Memory usage patterns
   - Optimization recommendations

4. Tests:
   - Unit tests for profiler utilities
   - Integration test profiling MD trajectory

**Success Criteria**:
- [ ] Profile full MD trajectories (not single calls)
- [ ] Measure per-call latency distribution
- [ ] Track memory allocation patterns
- [ ] Identify computational hotspots
- [ ] PyTorch profiler integration
- [ ] Generate profiling reports
- [ ] Baseline teacher model characterization
- [ ] Documentation with examples

**Timeline**:
- Day 1-2: Core profiler implementation (latency, memory)
- Day 3-4: PyTorch profiler integration, hotspot analysis
- Day 5-6: Profile teacher models, generate reports
- Day 7: Documentation and integration

**Dependencies**:
- Issue #2 (teacher wrappers) ✅ COMPLETE
- Issue #8 (CUDA environment) ✅ COMPLETE (your own work!)

**Integration Notes**:
- Your profiling will complement Agent 5's benchmarks
- Different goals: profiling (understand) vs benchmarking (measure)
- Can share timing utilities
- Coordinate on Day 4-5 for integrated reports

---

## Week 2 Schedule

### Day 1-2 (Mon-Tue): Parallel Kickoff
**All Agents**:
- Read your issue template thoroughly
- Confirm understanding in issue comment
- Set up skeleton code structure
- Initial implementation

**Agent 2** (Architecture):
- StudentCalculator skeleton with ASE interface

**Agent 5** (Testing):
- Issue #5: MD benchmark core loop

**Agent 4** (CUDA):
- Issue #9: Profiler skeleton

**Checkpoint**: Each agent comments progress in issue by EOD Day 2

### Day 3-4 (Wed-Thu): Implementation & Coordination
**All Agents**:
- Core functionality complete
- Tests written
- Initial integration

**Agent 2** (Architecture):
- StudentCalculator interface complete
- Notify Agent 5 it's ready for Issue #7

**Agent 5** (Testing):
- Issue #5: Memory profiling, JSON output
- Issue #7: START when Agent 2 ready

**Agent 4** (CUDA):
- PyTorch profiler integration
- Hotspot analysis

**Checkpoint**: Mid-week integration meeting (virtual via issue comments)

### Day 5-6 (Fri-Sat): Integration & Validation
**All Agents**:
- Complete deliverables
- Run integration tests
- Document results
- Create PRs

**Agent 2** (Architecture):
- Documentation and examples
- Support Agent 5 with Issue #7

**Agent 5** (Testing):
- Issue #5: Baseline measurements
- Issue #7: Complete interface tests

**Agent 4** (CUDA):
- Profile teacher models
- Generate reports

**Checkpoint**: Full integration test run

### Day 7 (Sun): Buffer & M1 Completion
**All Agents**:
- Address integration issues
- Finalize documentation
- PR reviews and merging

**Coordinator**:
- Mark M1 complete
- Generate Week 2 report
- Plan M2 kickoff

---

## Quality Standards

### Code Quality (Same as Week 1)
- [ ] Tests for all new code (>80% coverage)
- [ ] Type hints on public APIs
- [ ] Docstrings with usage examples
- [ ] Follows Week 1 code patterns
- [ ] Consistent imports (mlff_distiller.*)

### Performance Focus
- [ ] Measure on MD trajectories (1000+ steps)
- [ ] Per-call latency, not just total time
- [ ] Memory stability over long runs
- [ ] Realistic system sizes (100-1000 atoms)

### Documentation
- [ ] Clear README for each deliverable
- [ ] Usage examples in docstrings
- [ ] PR descriptions with integration notes

---

## Communication Protocol

### Daily Updates
- Update your issue with progress comment
- Format: "Progress: [done], Next: [next], Blockers: [none]"
- By 6 PM each day

### Blockers
- Tag "status:blocked" immediately
- @mention coordinator
- Expected resolution: <4 hours

### Questions
- Ask in issue comments
- @mention coordinator or other agents
- No question is too small

### PR Process
1. Create PR referencing issue
2. Ensure CI passes
3. Request review from coordinator
4. Address feedback
5. Merge when approved

---

## Integration Points

### Agent 2 ↔ Agent 5
- **What**: Student calculator interface
- **When**: Day 3-4
- **How**: Agent 2 notifies Agent 5 when StudentCalculator ready
- **Action**: Agent 5 uses StudentCalculator in Issue #7 tests

### Agent 4 ↔ Agent 5
- **What**: Profiling vs benchmarking coordination
- **When**: Day 4-5
- **How**: Share timing utilities, discuss metrics
- **Action**: Potentially integrated reports

### All Agents ↔ Coordinator
- **What**: Progress tracking, blocker resolution
- **When**: Daily
- **How**: Issue comments, PR reviews
- **Action**: Keep project on track

---

## Resources

### Repository
- **Location**: /home/aaron/ATX/software/MLFF_Distiller
- **Commit**: 4ff20d9 (Week 1 complete)
- **Tests**: Run with `pytest tests/` (181 passing)

### Key Files
- **Issue Templates**: docs/initial_issues/issue_*.md
- **Week 1 Code**: src/mlff_distiller/
- **Test Fixtures**: tests/conftest.py
- **Requirements**: docs/MD_REQUIREMENTS.md

### Documentation
- **Week 2 Plan**: docs/WEEK2_COORDINATION_PLAN.md
- **Project Overview**: docs/PROJECT_KICKOFF.md
- **Milestones**: docs/MILESTONES.md
- **Testing Guide**: TESTING.md

### GitHub
- **Repository**: https://github.com/atfrank/MLFF-Distiller
- **Issues**: Create and track progress
- **PRs**: Submit when ready for review

---

## Success Criteria for Week 2

### Individual Success
**Agent 2**:
- [ ] Issue #6 complete: StudentCalculator ready
- [ ] Tests passing
- [ ] Agent 5 using it successfully in Issue #7

**Agent 5**:
- [ ] Issue #5 complete: Benchmarks running
- [ ] Issue #7 complete: Interface tests passing
- [ ] Baseline performance documented

**Agent 4**:
- [ ] Issue #9 complete: Profiling framework ready
- [ ] Teacher model profiled
- [ ] Reports generated

### Team Success
- [ ] 4/4 remaining M1 issues complete
- [ ] M1 100% complete (9/9 issues)
- [ ] 200+ tests passing
- [ ] All PRs merged
- [ ] No unresolved blockers
- [ ] Documentation updated

---

## Motivation

You crushed Week 1. Week 2 is where we establish the metrics that define success for the entire project. Your benchmarking, interface design, and profiling work this week will:

1. **Prove the concept works** - Student models can be drop-in replacements
2. **Define success** - What does "5-10x faster" actually mean?
3. **Guide optimization** - Where should we focus CUDA efforts?
4. **Enable distillation** - M2-M3 work depends on this infrastructure

The ML/MD community needs fast, accurate force fields. You're building the tools to deliver them.

**Let's complete M1 and set up for distillation success!**

---

## Quick Start

### For Agent 2 (Architecture):
```bash
cd /home/aaron/ATX/software/MLFF_Distiller

# Read your template
cat docs/initial_issues/issue_26_ase_calculator_student.md

# Review your Week 1 work
cat src/mlff_distiller/models/teacher_wrappers.py

# Create new file
# src/mlff_distiller/models/student_calculator.py
```

### For Agent 5 (Testing):
```bash
cd /home/aaron/ATX/software/MLFF_Distiller

# Read your templates
cat docs/initial_issues/issue_22_md_benchmark_framework.md
cat docs/initial_issues/issue_29_ase_interface_tests.md

# Review teacher wrappers
cat src/mlff_distiller/models/teacher_wrappers.py

# Create benchmark directory
mkdir -p benchmarks
# Create file: benchmarks/md_benchmark.py
```

### For Agent 4 (CUDA):
```bash
cd /home/aaron/ATX/software/MLFF_Distiller

# Read your template
cat docs/initial_issues/issue_17_profiling_framework.md

# Review your Week 1 work
cat src/mlff_distiller/cuda/device_utils.py
cat src/mlff_distiller/cuda/benchmark_utils.py

# Create profiler file
# src/mlff_distiller/cuda/profiler.py
```

---

## Questions?

- **Issue Templates**: Read thoroughly, all details are there
- **Week 1 Code**: Use as reference and foundation
- **Blockers**: Report immediately, we'll solve together
- **Clarifications**: Ask in issue comments

**You've got this. Let's finish M1 strong!**

---

*Document Version: 1.0*
*Last Updated: 2025-11-23*
*Coordinator: Lead Project Coordinator*
