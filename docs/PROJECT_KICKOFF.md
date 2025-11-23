# MLFF Distiller Project Kickoff

**Date**: 2025-11-23
**Status**: Ready to Begin
**Repository**: /home/aaron/ATX/software/MLFF_Distiller

## Executive Summary

The MLFF Distiller project is now fully initialized and ready for specialized agent deployment. This document provides kickoff information for all five specialized agents, outlines the critical path for Week 1, and establishes coordination protocols.

### Project Status
- Repository structure: COMPLETE
- Documentation: COMPLETE (2,500+ lines)
- CI/CD pipeline: COMPLETE (tests passing)
- Issue planning: COMPLETE (33 issues planned across 6 milestones)
- GitHub setup: PENDING (manual setup required due to authentication)

### Critical Requirements
1. **MD Performance**: Models will be called millions of times in MD simulations
   - Latency optimization is paramount (not throughput)
   - Memory stability over long trajectories
   - Per-call overhead must be minimized

2. **Drop-In Replacement**: Student models must be plug-compatible
   - ASE Calculator interface (primary requirement)
   - Same inputs: atomic positions, species, cells
   - Same outputs: energies, forces, stresses
   - Works in existing MD scripts without modification

---

## GitHub Setup Instructions

Due to API authentication issues, the GitHub initialization script needs to be run with proper credentials. Here's how to proceed:

### Option 1: Manual GitHub Setup via Web Interface

1. **Create Repository** (if not already done):
   - Visit https://github.com/new
   - Name: `MLFF-Distiller` (note: hyphenated version)
   - Description: "Fast CUDA-optimized distilled force field models for molecular dynamics simulations"
   - Visibility: Public
   - Initialize: No (we already have code)

2. **Push Code**:
   ```bash
   cd /home/aaron/ATX/software/MLFF_Distiller
   git remote set-url origin https://github.com/atfrank/MLFF-Distiller.git
   git push -u origin main
   ```

3. **Create Labels** (via Settings → Labels):
   Run the script after fixing authentication or create manually using the label definitions in `scripts/initialize_project.sh`

4. **Create Milestones** (via Issues → Milestones):
   - M1: Setup & Baseline (Due: 2025-12-07)
   - M2: Data Pipeline (Due: 2025-12-21)
   - M3: Model Architecture (Due: 2026-01-04)
   - M4: Distillation Training (Due: 2026-01-25)
   - M5: CUDA Optimization (Due: 2026-02-15)
   - M6: Testing & Deployment (Due: 2026-03-01)

5. **Create Initial Issues**:
   Use the templates in `/home/aaron/ATX/software/MLFF_Distiller/docs/initial_issues/`
   Priority 1 issues to create first (see below)

### Option 2: Run Initialization Script

Once GitHub authentication is fixed:
```bash
bash /home/aaron/ATX/software/MLFF_Distiller/scripts/initialize_project.sh
```

---

## Priority 1 Issues (Create Immediately)

These 11 issues are critical for Week 1 progress and should be created first:

### Issue #1: [Data Pipeline] [M1] Set up data loading infrastructure
- **Agent**: Data Pipeline Engineer
- **Priority**: High
- **Complexity**: Medium
- **Template**: `/home/aaron/ATX/software/MLFF_Distiller/docs/initial_issues/issue_01_data_infrastructure.md`
- **Why Critical**: Foundation for all data operations; needed for teacher model loading

### Issue #2: [Data Pipeline] [M1] Create atomic structure data classes
- **Agent**: Data Pipeline Engineer
- **Priority**: High
- **Complexity**: Low
- **Template**: `/home/aaron/ATX/software/MLFF_Distiller/docs/initial_issues/issue_02_data_classes.md`
- **Why Critical**: Core data structures used across all agents

### Issue #6: [Architecture] [M1] Create teacher model wrapper with ASE Calculator interface
- **Agent**: ML Architecture Designer
- **Priority**: CRITICAL
- **Complexity**: High
- **Template**: `/home/aaron/ATX/software/MLFF_Distiller/docs/initial_issues/issue_06_teacher_wrappers.md`
- **Why Critical**: BLOCKS multiple other issues; enables drop-in compatibility; required for benchmarking

### Issue #7: [Architecture] [M1] Analyze Orb-models architecture
- **Agent**: ML Architecture Designer
- **Priority**: High
- **Complexity**: Medium
- **Template**: `/home/aaron/ATX/software/MLFF_Distiller/docs/initial_issues/issue_07_orb_analysis.md`
- **Why Critical**: Informs student model design

### Issue #11: [Training] [M1] Set up baseline training framework
- **Agent**: Distillation Training Engineer
- **Priority**: High
- **Complexity**: Medium
- **Template**: `/home/aaron/ATX/software/MLFF_Distiller/docs/initial_issues/issue_11_training_framework.md`
- **Why Critical**: Foundation for all training experiments

### Issue #16: [CUDA] [M1] Set up CUDA development environment
- **Agent**: CUDA Optimization Engineer
- **Priority**: High
- **Complexity**: Low
- **Template**: `/home/aaron/ATX/software/MLFF_Distiller/docs/initial_issues/issue_16_cuda_environment.md`
- **Why Critical**: Enables all CUDA work

### Issue #17: [CUDA] [M1] Create performance profiling framework for MD workloads
- **Agent**: CUDA Optimization Engineer
- **Priority**: High
- **Complexity**: Medium
- **Template**: `/home/aaron/ATX/software/MLFF_Distiller/docs/initial_issues/issue_17_profiling_framework.md`
- **Why Critical**: Establishes how we measure MD performance (not just single inference)

### Issue #21: [Testing] [M1] Configure pytest and test infrastructure
- **Agent**: Testing & Benchmark Engineer
- **Priority**: High
- **Complexity**: Low
- **Template**: `/home/aaron/ATX/software/MLFF_Distiller/docs/initial_issues/issue_21_pytest_setup.md`
- **Why Critical**: ENABLES ALL TESTING; blocks comprehensive validation

### Issue #22: [Testing] [M1] Create MD simulation benchmark framework
- **Agent**: Testing & Benchmark Engineer
- **Priority**: CRITICAL
- **Complexity**: Medium
- **Template**: `/home/aaron/ATX/software/MLFF_Distiller/docs/initial_issues/issue_22_benchmark_framework.md`
- **Why Critical**: Defines how we measure success for MD workloads; needed for all performance validation

### Issue #26: [Architecture] [M1] Implement ASE Calculator interface for student models
- **Agent**: ML Architecture Designer
- **Priority**: CRITICAL
- **Complexity**: Medium
- **Dependencies**: Issue #6
- **Template**: `/home/aaron/ATX/software/MLFF_Distiller/docs/initial_issues/issue_26_ase_calculator_student.md`
- **Why Critical**: Core requirement for drop-in replacement capability

### Issue #29: [Testing] [M1] Implement ASE Calculator interface tests
- **Agent**: Testing & Benchmark Engineer
- **Priority**: CRITICAL
- **Complexity**: Medium
- **Dependencies**: Issue #6
- **Template**: `/home/aaron/ATX/software/MLFF_Distiller/docs/initial_issues/issue_29_ase_interface_tests.md`
- **Why Critical**: Validates drop-in compatibility; prevents interface regressions

---

## Agent Kickoff Messages

### Agent 1: Data Pipeline Engineer

**Welcome to the team!** You are the Data Pipeline Engineer for the MLFF Distiller project.

#### Your Mission
Create robust, efficient data infrastructure that supports the entire distillation pipeline while maintaining compatibility with ASE (Atomic Simulation Environment) formats for seamless MD integration.

#### Week 1 Priorities

**PRIMARY TASK**: Issue #1 - Set up data loading infrastructure
- Location: `/home/aaron/ATX/software/MLFF_Distiller/src/data/`
- Key requirement: Must work with ASE Atoms objects (drop-in requirement)
- Deliverables:
  - AtomicDataLoader class supporting ASE Atoms format
  - Batch loading for GPU efficiency
  - Example loading script
  - Unit tests (depends on Issue #21)

**SECONDARY TASK**: Issue #2 - Create atomic structure data classes
- Build data classes compatible with ASE
- Support positions, species, cells, energies, forces, stresses
- Implement conversion to/from ASE Atoms
- Add validation methods

#### Critical Requirements to Remember
1. **ASE Compatibility**: All data structures must convert to/from ASE Atoms seamlessly
2. **MD Scale**: Design for datasets with millions of configurations
3. **GPU Efficiency**: DataLoader must saturate GPU during training
4. **Validation**: Every data point must be validated before storage

#### Key Documentation
- Contributing Guide: `/home/aaron/ATX/software/MLFF_Distiller/CONTRIBUTING.md`
- Agent Protocols: `/home/aaron/ATX/software/MLFF_Distiller/docs/AGENT_PROTOCOLS.md`
- Milestone M1: `/home/aaron/ATX/software/MLFF_Distiller/docs/MILESTONES.md`
- Data Requirements: `/home/aaron/ATX/software/MLFF_Distiller/docs/DATA_REQUIREMENTS.md`

#### Dependencies
- Your work blocks: Issue #5 (data generation), Issue #24 (data tests)
- You depend on: Issue #6 (teacher wrappers, for data generation in M2)

#### Expected Timeline
- Issue #1: Complete by end of Week 1
- Issue #2: Complete by end of Week 1
- Issue #3: Start Week 1, complete Week 2

#### Communication
- Tag @coordinator in issues when you need architectural decisions
- Use "status:blocked" label if dependencies aren't ready
- Update issue progress regularly
- Request reviews when PRs are ready

**Your success enables the entire training pipeline. Focus on correctness and ASE compatibility first, optimization second.**

---

### Agent 2: ML Architecture Designer

**Welcome to the team!** You are the ML Architecture Designer for the MLFF Distiller project.

#### Your Mission
Design and implement efficient student model architectures optimized for MD simulation workloads, with strict adherence to drop-in replacement requirements via ASE Calculator interface.

#### Week 1 Priorities

**CRITICAL TASK**: Issue #6 - Create teacher model wrapper with ASE Calculator interface
- Location: `/home/aaron/ATX/software/MLFF_Distiller/src/models/teacher_wrappers.py`
- **THIS IS THE HIGHEST PRIORITY TASK IN THE ENTIRE PROJECT**
- BLOCKS: Issues #5, #18, #23, #29, and all benchmarking
- Key requirements:
  - Implement ASE Calculator interface for Orb-models and FeNNol-PMC
  - Must accept ASE Atoms objects
  - Must return energies, forces, stresses in standard units
  - Must be usable in ASE MD engines (Verlet, Langevin, etc.)
  - Must handle periodic boundary conditions correctly
- Deliverables:
  - `OrbTeacherCalculator(Calculator)` class
  - `FeNNolTeacherCalculator(Calculator)` class
  - Example scripts showing MD simulation usage
  - Unit tests (once Issue #21 is complete)

**HIGH PRIORITY**: Issue #7 - Analyze Orb-models architecture
- Deep dive into Orb architecture
- Identify bottlenecks for repeated inference
- Document architecture choices
- Recommend simplifications for student model
- Focus on latency-critical components

**CRITICAL TASK**: Issue #26 - Implement ASE Calculator interface for student models
- Depends on Issue #6 completion
- Create StudentCalculator base class
- Ensure exact same interface as teacher wrappers
- Enable truly drop-in replacement

#### Critical Requirements to Remember
1. **ASE Calculator Interface**: Non-negotiable requirement for all models
   - Must inherit from `ase.calculators.calculator.Calculator`
   - Must implement `get_potential_energy()`, `get_forces()`, `get_stress()`
   - Must work with `ase.md` modules without modification
2. **MD Performance**: Optimize for LATENCY not throughput
   - Models called millions of times (1ns = 1000 calls at 1fs timestep)
   - Per-call overhead is critical
   - Memory must be stable over long trajectories
3. **Drop-In Replacement**: Users should be able to replace one line:
   ```python
   # Old: calc = OrbTeacherCalculator()
   # New: calc = OrbStudentCalculator()
   # Everything else stays the same!
   ```

#### Key Documentation
- Interface Requirements: `/home/aaron/ATX/software/MLFF_Distiller/docs/INTERFACE_REQUIREMENTS.md`
- MD Requirements: `/home/aaron/ATX/software/MLFF_Distiller/docs/MD_REQUIREMENTS.md`
- Contributing Guide: `/home/aaron/ATX/software/MLFF_Distiller/CONTRIBUTING.md`
- Agent Protocols: `/home/aaron/ATX/software/MLFF_Distiller/docs/AGENT_PROTOCOLS.md`

#### Dependencies
- Issue #6 BLOCKS: Issues #5, #18, #23, #26, #29
- Issue #26 depends on: Issue #6
- Issue #7 informs: Issue #9 (student architecture design)

#### Expected Timeline
- Issue #6: URGENT - Complete by end of Week 1 (critical path!)
- Issue #7: Complete by end of Week 1
- Issue #26: Start Week 2 (after Issue #6 complete)

#### Communication
- Issue #6 is highest priority - provide daily updates
- Immediately flag any blockers with "status:blocked" label
- Tag @coordinator for architectural decisions
- Coordinate with Testing Engineer for interface validation

**Your Issue #6 is the most critical task in Week 1. Everything depends on it. Focus all effort here first.**

---

### Agent 3: Distillation Training Engineer

**Welcome to the team!** You are the Distillation Training Engineer for the MLFF Distiller project.

#### Your Mission
Design and implement the knowledge distillation pipeline that transfers teacher model capabilities to fast, efficient student models while maintaining accuracy for MD simulations.

#### Week 1 Priorities

**PRIMARY TASK**: Issue #11 - Set up baseline training framework
- Location: `/home/aaron/ATX/software/MLFF_Distiller/src/training/`
- Key components:
  - Basic training loop with PyTorch
  - Checkpoint saving/loading
  - Metric tracking (energy MAE, force MAE, stress MAE)
  - Integration with DataLoader (from Issue #1)
- Focus on correctness and structure first, optimization later
- Deliverables:
  - `Trainer` base class
  - Example training script
  - Checkpoint management utilities
  - Training documentation

**SECONDARY TASK**: Issue #12 - Implement training configuration system
- YAML-based configuration
- Support for hyperparameter specification
- Model architecture selection
- Training regime parameters
- Reproducibility (random seeds, deterministic operations)

#### Critical Requirements to Remember
1. **MD-Relevant Metrics**: Track what matters for MD
   - Energy errors (kJ/mol or eV)
   - Force errors (kJ/mol/Å) - most critical for MD
   - Stress errors (for NPT simulations)
   - Energy conservation in NVE (later milestone)
2. **Accuracy Target**: >95% accuracy compared to teacher
3. **Multi-Teacher**: Support both Orb and FeNNol teachers
4. **Reproducibility**: All training must be exactly reproducible

#### Key Documentation
- Training Strategy: `/home/aaron/ATX/software/MLFF_Distiller/docs/TRAINING_STRATEGY.md`
- Contributing Guide: `/home/aaron/ATX/software/MLFF_Distiller/CONTRIBUTING.md`
- Agent Protocols: `/home/aaron/ATX/software/MLFF_Distiller/docs/AGENT_PROTOCOLS.md`
- Milestone M4: `/home/aaron/ATX/software/MLFF_Distiller/docs/MILESTONES.md`

#### Dependencies
- You depend on: Issue #1 (data loading), Issue #6 (teacher models for training data)
- You block: Issue #13 (distillation loss functions), Issue #15 (hyperparameter tuning)
- Later: Issue #9 (student architecture) will use your framework

#### Expected Timeline
- Issue #11: Complete by end of Week 1
- Issue #12: Complete by Week 2
- Issue #13: Start Week 3 (M4 milestone)

#### Communication
- Coordinate with Data Pipeline Engineer for DataLoader integration
- Tag @coordinator for training strategy decisions
- Update issue with training experiments and results
- Request code review when framework is ready

**Your framework will be used for all distillation experiments. Build it solid and modular.**

---

### Agent 4: CUDA Optimization Engineer

**Welcome to the team!** You are the CUDA Optimization Engineer for the MLFF Distiller project.

#### Your Mission
Optimize student model inference for maximum performance in MD simulation workloads, achieving 5-10x speedup through CUDA optimization while maintaining numerical stability.

#### Week 1 Priorities

**PRIMARY TASK**: Issue #16 - Set up CUDA development environment
- Location: `/home/aaron/ATX/software/MLFF_Distiller/src/cuda/`
- Components to set up:
  - CUDA toolkit installation verification
  - PyTorch CUDA environment
  - Profiling tools (nvprof, Nsight, PyTorch profiler)
  - Benchmarking infrastructure
  - CUDA kernel development template
- Deliverables:
  - Environment verification script
  - CUDA capability detection
  - Profiling wrapper utilities
  - Example CUDA kernel (simple operation)
  - Documentation

**HIGH PRIORITY**: Issue #17 - Create performance profiling framework for MD workloads
- This is NOT just single inference profiling!
- Must profile repeated inference over MD trajectories
- Key metrics:
  - Per-call latency (mean, median, p95, p99)
  - Throughput (calls/second)
  - Memory usage (peak and stable state)
  - GPU utilization over time
  - Kernel launch overhead
  - Memory transfer overhead
- Deliverables:
  - MD trajectory profiling script
  - Performance report generator
  - Comparison utilities (teacher vs student)
  - Profiling documentation

#### Critical Requirements to Remember
1. **MD Workload Characteristics**:
   - Millions of repeated inference calls (1ns MD = 1000 calls at 1fs timestep)
   - Small batch sizes (often single structures) - different from training!
   - Latency matters more than throughput
   - Memory must be stable (no leaks over long runs)
   - GPU should stay warm (avoid cold start penalties)

2. **Optimization Targets**:
   - 5-10x faster than teacher models
   - <1ms inference time per structure (for real-time MD)
   - Minimal per-call overhead
   - Stable memory usage over 1M+ calls

3. **Profiling Strategy**:
   - Always profile on MD-like workloads (repeated calls)
   - Measure warm-up vs steady-state performance
   - Track memory over time, not just peak
   - Test on various system sizes (10-1000 atoms)

#### Key Documentation
- MD Requirements: `/home/aaron/ATX/software/MLFF_Distiller/docs/MD_REQUIREMENTS.md`
- Performance Requirements: `/home/aaron/ATX/software/MLFF_Distiller/docs/PERFORMANCE_REQUIREMENTS.md`
- Contributing Guide: `/home/aaron/ATX/software/MLFF_Distiller/CONTRIBUTING.md`
- Agent Protocols: `/home/aaron/ATX/software/MLFF_Distiller/docs/AGENT_PROTOCOLS.md`

#### Dependencies
- Issue #17 depends on: Issue #6 (teacher wrappers to profile)
- Issue #18 depends on: Issues #6, #17
- You enable: Issue #19 (CUDA kernels), Issue #20 (memory optimization), Issue #33 (repeated inference)

#### Expected Timeline
- Issue #16: Complete by Week 1
- Issue #17: Start Week 1, complete Week 2
- Issue #18: Week 2 (after Issue #6 complete)

#### Communication
- Coordinate with Architecture Designer for teacher model profiling
- Share profiling results with all agents (informs design)
- Tag @coordinator for optimization strategy decisions
- Document all performance measurements

**Your profiling work will guide all optimization decisions. Measure what actually matters for MD!**

---

### Agent 5: Testing & Benchmark Engineer

**Welcome to the team!** You are the Testing & Benchmark Engineer for the MLFF Distiller project.

#### Your Mission
Establish comprehensive testing and benchmarking infrastructure focused on MD simulation workloads, ensuring correctness, performance, and drop-in compatibility throughout the project.

#### Week 1 Priorities

**CRITICAL TASK**: Issue #21 - Configure pytest and test infrastructure
- Location: `/home/aaron/ATX/software/MLFF_Distiller/tests/`
- **THIS ENABLES ALL OTHER TESTING**
- Components:
  - pytest configuration (`pytest.ini`, `conftest.py`)
  - Test directory structure (unit, integration, benchmarks)
  - Coverage reporting (pytest-cov)
  - CI integration (already set up, needs test runner)
  - Fixtures for common test data (ASE Atoms objects, etc.)
- Deliverables:
  - Working pytest setup
  - Example test files in each category
  - Coverage reporting configured
  - Testing documentation for other agents

**CRITICAL TASK**: Issue #22 - Create MD simulation benchmark framework
- **THIS IS NOT JUST SINGLE INFERENCE TIMING!**
- Must benchmark full MD trajectories
- Key measurements:
  - Trajectory performance (1000+ timesteps)
  - Per-step latency statistics
  - Memory usage over time
  - Energy conservation (NVE ensemble)
  - Temperature stability (NVT ensemble)
  - Comparison to teacher models
- Deliverables:
  - `MDBenchmark` class
  - Benchmark runner script
  - Result visualization tools
  - Benchmark report generator
  - Example benchmark configurations

**CRITICAL TASK**: Issue #29 - Implement ASE Calculator interface tests
- Depends on Issue #6
- Validates drop-in compatibility
- Key tests:
  - ASE Calculator interface compliance
  - Energy/force/stress calculation correctness
  - Periodic boundary condition handling
  - Integration with ASE MD engines (Verlet, Langevin)
  - Comparison to reference implementations
- Deliverables:
  - Calculator interface test suite
  - ASE MD integration tests
  - Compatibility validation script

#### Critical Requirements to Remember
1. **MD-Focused Testing**:
   - Don't just test single inference - test trajectories!
   - Energy conservation is critical (NVE ensemble)
   - Force errors accumulate - test long trajectories
   - Memory leaks appear over time - test extended runs

2. **Drop-In Compatibility**:
   - Student models must pass exact same tests as teachers
   - Interface must be identical
   - No modification to existing MD scripts
   - ASE integration must be seamless

3. **Performance Benchmarking**:
   - Always compare to baseline (teacher models)
   - Track metrics over time (regression detection)
   - Test on various system sizes
   - Document benchmark conditions (hardware, versions)

4. **Test Coverage**:
   - Aim for >80% coverage
   - Critical paths need 100% coverage
   - Edge cases matter (small/large systems, extreme forces)

#### Key Documentation
- Testing Strategy: `/home/aaron/ATX/software/MLFF_Distiller/docs/TESTING_STRATEGY.md`
- MD Requirements: `/home/aaron/ATX/software/MLFF_Distiller/docs/MD_REQUIREMENTS.md`
- Interface Requirements: `/home/aaron/ATX/software/MLFF_Distiller/docs/INTERFACE_REQUIREMENTS.md`
- Contributing Guide: `/home/aaron/ATX/software/MLFF_Distiller/CONTRIBUTING.md`

#### Dependencies
- Issue #21 ENABLES: All other testing work
- Issue #22 depends on: Issue #6 (teacher models to benchmark)
- Issue #29 depends on: Issue #6 (teacher calculators)
- Issue #23 depends on: Issues #6, #22 (baseline benchmarks)

#### Expected Timeline
- Issue #21: URGENT - Complete by Week 1 (enables other agents' testing)
- Issue #22: Complete by end of Week 1
- Issue #29: Start Week 1, complete Week 2 (after Issue #6)

#### Communication
- Issue #21 is urgent - other agents need it for their PRs
- Coordinate with all agents on test requirements
- Share benchmark results widely (influences design)
- Tag @coordinator for testing strategy decisions
- Create test templates for other agents to follow

**Your testing infrastructure enables quality across the project. Prioritize Issue #21 to unblock others!**

---

## Week 1 Critical Path Analysis

### Critical Path Items (Must Complete)

The following items form the critical path and must be completed in order for the project to progress:

```
Week 1 Critical Path:
─────────────────────

Day 1-2:
├─ Issue #21 (Testing): Configure pytest [ENABLES ALL TESTING]
├─ Issue #16 (CUDA): CUDA environment setup [ENABLES PROFILING]
├─ Issue #1 (Data): Data loading infrastructure [FOUNDATION]
└─ Issue #2 (Data): Data classes [FOUNDATION]

Day 2-5:
├─ Issue #6 (Architecture): Teacher ASE Calculator wrappers [BLOCKS EVERYTHING]
│   └─ HIGHEST PRIORITY - Multiple blockers waiting
│
└─ Issue #22 (Testing): MD benchmark framework [DEFINES SUCCESS METRICS]

Day 3-7:
├─ Issue #11 (Training): Baseline training framework
├─ Issue #7 (Architecture): Orb analysis
├─ Issue #17 (CUDA): MD profiling framework
└─ Issue #29 (Testing): ASE interface tests [depends on #6]

End of Week 1:
└─ Issue #26 (Architecture): Student ASE Calculator [depends on #6]
```

### Dependency Graph

```
Issue #21 (pytest) ────┬──> All testing work
                       │
Issue #6 (Teacher) ────┼──> Issue #5 (data generation)
    CRITICAL!          ├──> Issue #18 (profiling)
                       ├──> Issue #23 (baseline benchmarks)
                       ├──> Issue #26 (student calculator)
                       └──> Issue #29 (interface tests)

Issue #1 (data infra) ─┬──> Issue #4 (HDF5 storage)
                       └──> Issue #24 (data tests)

Issue #22 (MD bench) ──┬──> Issue #23 (baseline benchmarks)
                       └──> Issue #25 (regression tests)

Issue #17 (profiling) ──> Issue #18 (profile teacher)

Issue #26 (student calc) ──> Issue #27 (drop-in validation)
```

### Bottleneck Analysis

**HIGHEST RISK**: Issue #6 (Teacher ASE Calculator wrappers)
- Blocks 5+ other issues
- High complexity
- New interface implementation
- Mitigation: Assign to Architecture Agent immediately, daily check-ins

**MEDIUM RISK**: Issue #21 (pytest setup)
- Blocks all testing but low complexity
- Should complete quickly
- Mitigation: Prioritize early

**MEDIUM RISK**: Issue #22 (MD benchmark framework)
- Defines how we measure success
- Needs to be done correctly, not just quickly
- Mitigation: Clear requirements in issue template

---

## Week 1 Coordination Plan

### Day-by-Day Plan

#### Monday (Day 1)
**Focus**: Foundation setup

- **Testing Engineer**: Start Issue #21 (pytest setup) - TARGET: Complete today
- **CUDA Engineer**: Start Issue #16 (CUDA environment) - TARGET: Complete Day 2
- **Data Engineer**: Start Issue #1 (data infrastructure)
- **Architecture Designer**: Start Issue #6 (Teacher wrappers) - CRITICAL
- **Training Engineer**: Start Issue #11 (training framework)

**Coordinator Actions**:
- Create Priority 1 issues on GitHub (if not automated)
- Monitor Issue #6 progress closely
- Check in with all agents end-of-day

#### Tuesday (Day 2)
**Focus**: Critical path progress

- **Testing Engineer**:
  - COMPLETE Issue #21
  - Start Issue #22 (MD benchmark framework)
- **CUDA Engineer**:
  - COMPLETE Issue #16
  - Start Issue #17 (profiling framework)
- **Data Engineer**:
  - Continue Issue #1
  - Start Issue #2 (data classes)
- **Architecture Designer**:
  - FOCUS on Issue #6 (target: 50% complete)
  - Daily update required
- **Training Engineer**: Continue Issue #11

**Coordinator Actions**:
- Review Issue #21 completion (unblocks testing)
- Check Issue #6 progress (critical path)
- Verify test infrastructure working

#### Wednesday (Day 3)
**Focus**: Issue #6 completion target

- **Architecture Designer**:
  - TARGET: Complete Issue #6 by end of day
  - If blocked, flag immediately
- **Testing Engineer**:
  - Continue Issue #22
  - Prepare Issue #29 (depends on Issue #6)
- **CUDA Engineer**: Continue Issue #17
- **Data Engineer**: Complete Issue #1, Issue #2
- **Training Engineer**: Continue Issue #11

**Coordinator Actions**:
- Critical check: Is Issue #6 on track?
- If Issue #6 delayed, reprioritize dependent work
- Review any blocked issues

#### Thursday (Day 4)
**Focus**: Dependent work begins

- **Architecture Designer**:
  - Issue #6 should be DONE
  - Start Issue #7 (Orb analysis)
  - Start Issue #26 (Student calculator)
- **Testing Engineer**:
  - Complete Issue #22
  - Start Issue #29 (now unblocked)
- **CUDA Engineer**: Complete Issue #17
- **Data Engineer**: Start Issue #3 (validation)
- **Training Engineer**: Target Issue #11 completion

**Coordinator Actions**:
- Review Issue #6 PR (highest priority)
- Check baseline benchmark plan
- Verify all agents have work

#### Friday (Day 5)
**Focus**: Week completion and integration

- **All Agents**: Complete Week 1 assigned issues
- **Architecture Designer**: Continue Issue #7, #26
- **Testing Engineer**: Continue Issue #29
- **Integration**: Run end-to-end tests with teacher models

**Coordinator Actions**:
- Week 1 review
- Assess milestone M1 progress (target: 40% complete)
- Plan Week 2 issues
- Identify any blockers for next week

### Communication Protocol

**Daily Standup (Virtual)**:
- Each agent updates their assigned issue with progress comment
- Format: "Progress: [what was done], Next: [what's next], Blockers: [any blockers]"
- Time: End of day
- Coordinator reviews all updates each morning

**Blocker Resolution**:
- Tag issue with "status:blocked" immediately
- @mention coordinator in comment
- Expected resolution time: <4 hours for critical path items

**PR Review Process**:
1. Agent creates PR linking to issue
2. PR runs CI (tests, linting)
3. Agent requests review from coordinator
4. Coordinator reviews within 24 hours
5. Address feedback, re-request review
6. Merge when approved and tests pass

**Integration Points**:
- Mid-week (Wednesday): Check cross-agent compatibility
- End-of-week (Friday): Integration testing
- Document any interface changes immediately

---

## Success Metrics for Week 1

### Completion Targets
- [ ] 4/11 Priority 1 issues COMPLETE (Issues #21, #16, #1, #2)
- [ ] 7/11 Priority 1 issues IN PROGRESS (all others)
- [ ] Issue #6 COMPLETE or 80%+ complete (critical path)
- [ ] pytest infrastructure functional
- [ ] CUDA environment verified
- [ ] All agents have submitted at least 1 PR

### Quality Metrics
- [ ] All merged PRs have tests
- [ ] CI passing on all PRs
- [ ] No unresolved "status:blocked" issues >48 hours
- [ ] Code coverage >60% on new code

### Integration Metrics
- [ ] Teacher models load successfully
- [ ] Data classes work with ASE Atoms
- [ ] Baseline training loop runs (even if not trained)
- [ ] CUDA profiling captures meaningful metrics

### Team Metrics
- [ ] All agents responsive to comments within 24 hours
- [ ] No agent has >3 issues in progress simultaneously
- [ ] All blockers resolved within 48 hours
- [ ] Documentation updated with each PR

---

## Risk Assessment

### High Risks

**Risk**: Issue #6 (Teacher wrappers) takes longer than expected
- **Impact**: Critical - blocks multiple agents
- **Probability**: Medium (high complexity, new interface)
- **Mitigation**:
  - Daily progress checks
  - Pair with Testing Engineer for interface validation
  - Simplified first version (energy only, add forces later if needed)
  - Have Architecture Agent focus ONLY on this until done

**Risk**: ASE Calculator interface unfamiliarity
- **Impact**: High - core requirement misunderstood
- **Probability**: Medium
- **Mitigation**:
  - Detailed documentation in issue templates
  - Example implementations provided
  - Early validation with simple test cases
  - Testing Engineer validates interface immediately

### Medium Risks

**Risk**: CUDA environment setup issues
- **Impact**: Medium - blocks CUDA work but not other agents
- **Probability**: Low-Medium (environmental issues)
- **Mitigation**:
  - Detailed setup documentation
  - Multiple CUDA installation methods documented
  - Can work on profiling strategy while debugging

**Risk**: Agents work in isolation, integration fails
- **Impact**: Medium - rework needed
- **Probability**: Medium (distributed team)
- **Mitigation**:
  - Mid-week integration check
  - Clear interface contracts in issues
  - Shared test fixtures
  - Coordinator monitors cross-dependencies

### Low Risks

**Risk**: pytest setup complications
- **Impact**: Low - well-understood task
- **Probability**: Low
- **Mitigation**:
  - Standard pytest patterns
  - Many examples available
  - Simple initial setup, elaborate later

---

## Next Steps (Coordinator)

### Immediate (Today)
1. **Manually create GitHub repository if not exists**:
   - Name: MLFF-Distiller
   - Push code

2. **Create Priority 1 issues** (11 issues):
   - Use templates in `/home/aaron/ATX/software/MLFF_Distiller/docs/initial_issues/`
   - Ensure proper labels and milestones
   - Verify issue templates are complete

3. **Set up GitHub Project board**:
   - Columns: Backlog, Ready, In Progress, Review, Done
   - Add all Priority 1 issues

4. **Notify agents**:
   - Send kickoff messages
   - Assign initial issues
   - Clarify expectations

### Day 2-3
1. **Monitor Issue #6 progress** (critical path)
2. **Review Issue #21 completion** (enables testing)
3. **Check for blockers**
4. **First PR reviews**

### End of Week 1
1. **Week 1 retrospective**
2. **M1 milestone progress review** (target: 40%)
3. **Create Priority 2 issues** (if capacity available)
4. **Plan Week 2**
5. **Adjust timeline if needed**

---

## Resources

### Documentation
- Project README: `/home/aaron/ATX/software/MLFF_Distiller/README.md`
- Contributing Guide: `/home/aaron/ATX/software/MLFF_Distiller/CONTRIBUTING.md`
- Agent Protocols: `/home/aaron/ATX/software/MLFF_Distiller/docs/AGENT_PROTOCOLS.md`
- Milestones: `/home/aaron/ATX/software/MLFF_Distiller/docs/MILESTONES.md`
- All Issue Templates: `/home/aaron/ATX/software/MLFF_Distiller/docs/initial_issues/`

### Key Requirements
- Interface Requirements: `/home/aaron/ATX/software/MLFF_Distiller/docs/INTERFACE_REQUIREMENTS.md`
- MD Requirements: `/home/aaron/ATX/software/MLFF_Distiller/docs/MD_REQUIREMENTS.md`
- Performance Requirements: `/home/aaron/ATX/software/MLFF_Distiller/docs/PERFORMANCE_REQUIREMENTS.md`
- Data Requirements: `/home/aaron/ATX/software/MLFF_Distiller/docs/DATA_REQUIREMENTS.md`

### Technical Guides
- Training Strategy: `/home/aaron/ATX/software/MLFF_Distiller/docs/TRAINING_STRATEGY.md`
- Testing Strategy: `/home/aaron/ATX/software/MLFF_Distiller/docs/TESTING_STRATEGY.md`

### Setup Scripts
- Initialize GitHub: `/home/aaron/ATX/software/MLFF_Distiller/scripts/initialize_project.sh`
- Environment setup: Follow README installation instructions

---

## Questions & Support

### For Agents
- **Blockers**: Tag with "status:blocked" and @mention coordinator
- **Clarifications**: Comment on issue, @mention coordinator
- **Architectural decisions**: Tag with "status:needs-decision"
- **Urgent issues**: Direct message (if available) or urgent GitHub comment

### For Coordinator
- **Monitor**: All issues tagged "status:blocked" or "status:needs-decision"
- **Review cadence**: Daily check-ins, PR reviews within 24 hours
- **Decision-making**: Document all architectural decisions in issues
- **Escalation**: Tag issues with "needs-stakeholder" for external input

---

**Project Start Date**: 2025-11-23
**M1 Target Date**: 2025-12-07 (2 weeks)
**Current Status**: Ready to begin
**Next Milestone Review**: 2025-11-30 (1 week check-in)

**Let's build something amazing! The MD simulation community is counting on us.**
