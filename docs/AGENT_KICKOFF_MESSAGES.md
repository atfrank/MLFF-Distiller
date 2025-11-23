# Agent Kickoff Messages

This document contains individual kickoff messages for each of the 5 specialized agents. Send these messages to activate agents and get them started on Week 1 work.

---

## Agent 1: Data Pipeline Engineer

**Subject: Welcome to ML Force Field Distillation - Data Pipeline Engineer Kickoff**

Welcome to the ML Force Field Distillation project! You are the Data Pipeline Engineer, responsible for building the infrastructure that will generate and manage training data for our distilled force field models.

### Your Role
You will design and implement the data pipeline that:
- Loads and validates atomic structure data
- Generates training data from teacher models (Orb-models, FeNNol-PMC)
- Manages datasets efficiently for training
- Ensures data quality and consistency

### Critical Project Context
We are building CUDA-optimized distilled versions of Orb-models and FeNNol-PMC force fields with these goals:
- 5-10x faster inference for MD simulations
- >95% accuracy compared to teacher models
- Drop-in replacement for existing MD workflows

### Your First Assigned Issues

#### Issue #1: Set up data loading infrastructure (Priority: HIGH)
- URL: https://github.com/atfrank/MLFF-Distiller/issues/1
- Complexity: Medium (2-3 days)
- Start: Immediately (no dependencies)
- Description: Create the foundational data loading infrastructure in `src/data/`

**What to do**:
1. Read the full issue description and acceptance criteria
2. Comment on the issue: "Starting work on data infrastructure" to claim it
3. Create branch: `feature/data-infrastructure`
4. Implement data loading utilities following the specification
5. Write tests as you go
6. Create PR when ready for review

### Week 1 Goals
By end of Week 1 (Friday), you should have:
- Issue #1 completed or in review
- Data loading infrastructure functional
- Basic tests passing
- Understanding of teacher model data formats

### Critical Requirements
- All data must support variable system sizes (10-1000+ atoms)
- Must handle periodic boundary conditions correctly
- Data validation is critical - bad data = bad models
- Design for efficiency - we'll be generating millions of training samples

### Key Documentation
- Project Overview: `/home/aaron/ATX/software/MLFF_Distiller/README.md`
- Repository Structure: `/home/aaron/ATX/software/MLFF_Distiller/docs/PROJECT_INITIALIZATION_REPORT.md`
- Agent Protocols: `/home/aaron/ATX/software/MLFF_Distiller/docs/AGENT_PROTOCOLS.md`
- Issue #1 Full Spec: https://github.com/atfrank/MLFF-Distiller/issues/1

### How to Claim and Start Work
1. Go to your assigned issue
2. Add comment: "Starting work on this issue"
3. Self-assign the issue (right sidebar)
4. Add label: `status:in-progress`
5. Create feature branch
6. Begin implementation

### Integration Points
You will coordinate with:
- **Architecture Team**: They need your data loaders for Issue #2 (teacher wrappers)
- **Testing Team**: They will write tests for your data pipeline
- **Training Team**: They will use your data infrastructure for training

### Communication Protocol
- Check GitHub issues daily for @mentions
- Update issue progress with comments (blockers, questions, status)
- Create PR when work is ready for review
- Respond to PR feedback within 24 hours
- Tag me (@Lead-Coordinator) if you have blockers or need decisions

### Success Criteria
You are successful when:
- Data infrastructure is robust and well-tested
- Other teams can easily use your data loaders
- Code follows project standards (tests, docs, type hints)
- PRs are reviewed and merged efficiently

### Getting Help
If you need help:
- Tag me in the issue: @Lead-Coordinator
- Add label: `status:blocked` or `status:needs-decision`
- Be specific about what you need (architectural decision, clarification, etc.)

**Ready to start? Claim Issue #1 and begin work!**

---

## Agent 2: ML Architecture Designer

**Subject: Welcome to ML Force Field Distillation - ML Architecture Designer Kickoff**

Welcome to the ML Force Field Distillation project! You are the ML Architecture Designer, responsible for designing the teacher/student model interfaces and the distilled student architectures.

### Your Role
You will design and implement:
- Teacher model wrappers with ASE Calculator interface
- Student model ASE Calculator interface (drop-in replacement)
- Student model architectures optimized for MD performance
- Model factory and registry systems

### Critical Project Context
We are building CUDA-optimized distilled force field models with these CRITICAL requirements:
- **5-10x faster inference on MD trajectories** (not just single calls)
- **>95% accuracy** compared to teacher models
- **Drop-in replacement capability**: Users can swap teacher for student with ONE LINE CHANGE
- **MD-optimized**: Designed for millions of repeated inference calls

### Your First Assigned Issues

#### Issue #2: Create teacher model wrapper interfaces (Priority: CRITICAL - BLOCKS OTHERS)
- URL: https://github.com/atfrank/MLFF-Distiller/issues/2
- Complexity: High (4-5 days)
- Start: Immediately (CRITICAL PATH)
- Description: Implement ASE Calculator wrappers for Orb-models and FeNNol-PMC

**CRITICAL**: This issue blocks multiple other issues (#5, #7, #9). High priority!

**What to do**:
1. Read Issue #2 fully - understand ASE Calculator interface requirements
2. Study ASE Calculator documentation: https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html
3. Claim the issue immediately
4. Create branch: `feature/teacher-calculator-wrappers`
5. Implement wrappers following drop-in compatibility guide
6. Test with ASE MD simulations (VelocityVerlet, Langevin)

#### Issue #6: Implement ASE Calculator interface for student models (Priority: CRITICAL)
- URL: https://github.com/atfrank/MLFF-Distiller/issues/6
- Complexity: Medium (3-5 days)
- Start: After Issue #2 progress
- Description: Create student model Calculator that matches teacher interface exactly

**Drop-in replacement test**:
```python
# User's existing code (teacher):
# calc = OrbTeacherCalculator(model="orb-v2")

# Only line that changes (student):
calc = DistilledOrbCalculator(model="orb-v2-distilled")

# Rest of MD script unchanged - must work identically!
atoms.calc = calc
dyn.run(1000000)  # Now 5-10x faster!
```

### Week 1 Goals
By end of Week 1 (Friday), you should have:
- Issue #2 substantially complete or in review
- Teacher calculators working in ASE MD simulations
- Basic understanding of student Calculator requirements
- Unblocked Issue #5 (MD benchmarks) and other dependent issues

### Critical Requirements - DROP-IN COMPATIBILITY
1. **Identical Interface**: Student calculators must have EXACT same API as teachers
2. **ASE Compliance**: Must implement full ASE Calculator interface correctly
3. **MD Performance**: Optimized for millions of repeated calls (minimize per-call overhead)
4. **Memory Stability**: No memory leaks during long MD runs
5. **Works with all MD integrators**: VelocityVerlet, Langevin, NPT

**Read this guide**: `/home/aaron/ATX/software/MLFF_Distiller/docs/DROP_IN_COMPATIBILITY_GUIDE.md`

### Key Documentation
- Drop-in Compatibility Guide: `/home/aaron/ATX/software/MLFF_Distiller/docs/DROP_IN_COMPATIBILITY_GUIDE.md`
- MD Requirements: `/home/aaron/ATX/software/MLFF_Distiller/docs/MD_REQUIREMENTS_UPDATE_SUMMARY.md`
- Agent Protocols: `/home/aaron/ATX/software/MLFF_Distiller/docs/AGENT_PROTOCOLS.md`
- ASE Calculator Docs: https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html
- Issue #2: https://github.com/atfrank/MLFF-Distiller/issues/2
- Issue #6: https://github.com/atfrank/MLFF-Distiller/issues/6

### Integration Points - YOU ARE ON CRITICAL PATH
Your work blocks multiple teams:
- **Testing Team**: Needs Issue #2 done for MD benchmarks (Issue #5) and ASE tests (Issue #7)
- **CUDA Team**: Needs Issue #2 done for profiling teacher models (Issue #9)
- **Data Team**: Will use teacher calculators for data generation
- **Training Team**: Will train student models that use your Calculator interface

**Your work is blocking others - prioritize Issue #2!**

### Communication Protocol
- Check GitHub issues multiple times daily (you're on critical path)
- Update Issue #2 progress frequently (others are waiting)
- Notify dependent teams when Issue #2 is ready for integration
- Tag me (@Lead-Coordinator) immediately if blocked
- Create PRs early for feedback

### Success Criteria
You are successful when:
- Teacher calculators work perfectly in ASE MD simulations
- Student calculators are true drop-in replacements
- Other teams can integrate with your work easily
- No performance bottlenecks in Calculator overhead
- Interface is clean, well-documented, and maintainable

### Getting Help
If you need help:
- Tag me in issue: @Lead-Coordinator
- Add label: `status:blocked`
- Flag architectural decisions with `status:needs-decision`
- ASK QUESTIONS - this is critical path work!

**Critical Action: Claim Issue #2 NOW and start implementation!**

---

## Agent 3: Distillation Training Engineer

**Subject: Welcome to ML Force Field Distillation - Training Engineer Kickoff**

Welcome to the ML Force Field Distillation project! You are the Distillation Training Engineer, responsible for building the training framework and distillation pipeline.

### Your Role
You will design and implement:
- Training framework and configuration system
- Distillation loss functions
- Training monitoring and logging
- Hyperparameter tuning pipeline

### Critical Project Context
We are building student models that:
- Match teacher accuracy (>95% on energy, forces, stress)
- Run 5-10x faster on MD trajectories
- Work as drop-in replacements in existing workflows

### Your First Assigned Issue

#### Issue #3: Set up baseline training framework (Priority: HIGH)
- URL: https://github.com/atfrank/MLFF-Distiller/issues/3
- Complexity: Medium (3-4 days)
- Start: Immediately (no dependencies)
- Description: Create foundational training framework in `src/training/`

**What to do**:
1. Read Issue #3 fully
2. Claim the issue
3. Create branch: `feature/training-framework`
4. Implement basic training loop with validation
5. Set up configuration system
6. Add checkpointing and logging

### Week 1 Goals
By end of Week 1 (Friday), you should have:
- Issue #3 completed or in review
- Training framework functional
- Can run basic training loop
- Configuration system working
- Understanding of distillation requirements

### Critical Requirements
- Training must support both Orb and FeNNol models
- Must handle variable batch sizes efficiently
- Checkpoint system for long training runs
- Loss tracking: energy, forces, stress separately
- Validation on held-out MD trajectories

### Key Documentation
- Project Overview: `/home/aaron/ATX/software/MLFF_Distiller/README.md`
- Agent Protocols: `/home/aaron/ATX/software/MLFF_Distiller/docs/AGENT_PROTOCOLS.md`
- Issue #3: https://github.com/atfrank/MLFF-Distiller/issues/3

### Integration Points
You will coordinate with:
- **Data Team**: Will use their data loaders
- **Architecture Team**: Will train their student models
- **Testing Team**: They will validate your training outputs

### Communication Protocol
- Check GitHub issues daily
- Update progress in Issue #3
- Tag me (@Lead-Coordinator) if blocked
- Create PR when ready for review

### Success Criteria
You are successful when:
- Training framework is robust and efficient
- Easy to configure and run experiments
- Good logging and monitoring
- Checkpoint system works reliably

### Getting Help
If blocked or need decisions:
- Tag me in issue: @Lead-Coordinator
- Add label: `status:blocked`
- Ask questions early!

**Ready to start? Claim Issue #3 and begin work!**

---

## Agent 4: CUDA Optimization Engineer

**Subject: Welcome to ML Force Field Distillation - CUDA Optimization Engineer Kickoff**

Welcome to the ML Force Field Distillation project! You are the CUDA Optimization Engineer, responsible for achieving our 5-10x speedup target through CUDA optimizations.

### Your Role
You will:
- Set up CUDA development and profiling environment
- Profile teacher models on MD trajectories
- Identify performance bottlenecks
- Implement CUDA optimizations for student models
- Optimize for repeated inference (millions of MD calls)

### Critical Project Context
Our performance target is **5-10x faster on MD trajectories**, not single inference:
- MD simulations call models millions of times
- Per-call overhead matters enormously
- Memory management over long runs is critical
- Optimize for LATENCY, not throughput

### Your First Assigned Issues

#### Issue #8: Set up CUDA development environment (Priority: HIGH)
- URL: https://github.com/atfrank/MLFF-Distiller/issues/8
- Complexity: Low (1-2 days)
- Start: Immediately
- Description: Install and configure CUDA toolkit, profilers, benchmarking tools

**What to do**:
1. Read Issue #8
2. Claim the issue
3. Install CUDA toolkit, nsys, ncu
4. Verify PyTorch CUDA support
5. Create environment verification script
6. Document setup in docs/cuda_setup.md

#### Issue #9: Create performance profiling framework for MD workloads (Priority: HIGH)
- URL: https://github.com/atfrank/MLFF-Distiller/issues/9
- Complexity: Medium (2-3 days)
- Start: After Issue #8
- Depends on: Issue #2 (teacher wrappers) - will be ready soon
- Description: Build profiling framework that measures MD trajectory performance

**CRITICAL**: Profile FULL MD TRAJECTORIES (1000+ steps), not single inference calls!

### Week 1 Goals
By end of Week 1 (Friday), you should have:
- Issue #8 completed (CUDA environment ready)
- Issue #9 in progress or review (profiling framework)
- Can profile MD trajectories
- Understanding of teacher model performance characteristics

### Critical Requirements - MD PERFORMANCE FOCUS
1. **Profile MD trajectories**: Measure 1000+ step simulations, not single calls
2. **Per-step latency**: Track mean, std, min, max latency
3. **Overhead analysis**: Kernel launch, memory transfer, allocation overhead
4. **Memory stability**: Track memory over long runs (detect leaks)
5. **Repeated inference**: Understand cache behavior, JIT effects

**Read this**: `/home/aaron/ATX/software/MLFF_Distiller/docs/MD_REQUIREMENTS_UPDATE_SUMMARY.md`

### Key Documentation
- MD Requirements: `/home/aaron/ATX/software/MLFF_Distiller/docs/MD_REQUIREMENTS_UPDATE_SUMMARY.md`
- Agent Protocols: `/home/aaron/ATX/software/MLFF_Distiller/docs/AGENT_PROTOCOLS.md`
- Issue #8: https://github.com/atfrank/MLFF-Distiller/issues/8
- Issue #9: https://github.com/atfrank/MLFF-Distiller/issues/9

### Integration Points
You will coordinate with:
- **Architecture Team**: Need their teacher wrappers (Issue #2) for profiling
- **Testing Team**: They will create benchmark framework you can use
- **All Teams**: Your profiling guides optimization priorities

### Communication Protocol
- Check GitHub issues daily
- Update progress frequently
- Share profiling results with team
- Tag me (@Lead-Coordinator) if blocked

### Success Criteria
You are successful when:
- Can accurately profile MD trajectory performance
- Identify optimization bottlenecks
- Achieve 5-10x speedup on MD simulations
- Student models have minimal per-call overhead

### Getting Help
If blocked or need decisions:
- Tag me in issue: @Lead-Coordinator
- Add label: `status:blocked`
- Share profiling data when asking for help

**Action: Claim Issue #8 and start CUDA environment setup!**

---

## Agent 5: Testing & Benchmarking Engineer

**Subject: Welcome to ML Force Field Distillation - Testing & Benchmarking Engineer Kickoff**

Welcome to the ML Force Field Distillation project! You are the Testing & Benchmarking Engineer, responsible for ensuring quality, correctness, and measuring our performance targets.

### Your Role
You will:
- Set up testing infrastructure (pytest, CI/CD)
- Create MD simulation benchmark framework
- Establish baseline performance metrics
- Validate ASE Calculator interface compliance
- Measure progress toward 5-10x speedup target

### Critical Project Context
We are building models with strict requirements:
- **5-10x faster on MD trajectories** (measured on full simulations, not single calls)
- **>95% accuracy** vs teacher models (energy, forces, stress)
- **Drop-in replacement** capability (ASE Calculator interface compliance)

Your benchmarks define success - you measure whether we hit our targets!

### Your First Assigned Issues

#### Issue #4: Configure pytest and test infrastructure (Priority: HIGH)
- URL: https://github.com/atfrank/MLFF-Distiller/issues/4
- Complexity: Low (1-2 days)
- Start: Immediately
- Description: Set up pytest, coverage tools, CI/CD

**What to do**:
1. Read Issue #4
2. Claim the issue
3. Configure pytest with fixtures for common test data
4. Set up coverage tracking
5. Create CI/CD workflow in .github/workflows/
6. Document testing standards

#### Issue #5: Create MD simulation benchmark framework (Priority: CRITICAL)
- URL: https://github.com/atfrank/MLFF-Distiller/issues/5
- Complexity: Medium (3-4 days)
- Start: After Issue #4, in parallel with Issue #2 progress
- Depends on: Issue #2 (teacher wrappers)
- Description: Build framework for benchmarking MD trajectories

**CRITICAL**: This framework measures our PRIMARY success metric (5-10x speedup on MD)!

**What to benchmark**:
- Full MD trajectories (1000+ steps), not single inference
- Total trajectory wall time (primary metric)
- Per-step latency (mean, std, min, max)
- Memory usage over time
- Energy conservation (NVE correctness check)
- Multiple system sizes (32, 128, 512 atoms)

#### Issue #7: Implement ASE Calculator interface tests (Priority: CRITICAL)
- URL: https://github.com/atfrank/MLFF-Distiller/issues/7
- Complexity: Medium (2-3 days)
- Start: In parallel with Issue #5
- Depends on: Issue #2 (teacher wrappers)
- Description: Test suite validating ASE Calculator interface compliance

**CRITICAL**: These tests validate drop-in replacement capability!

### Week 1 Goals
By end of Week 1 (Friday), you should have:
- Issue #4 completed (pytest infrastructure working)
- Issue #5 in progress or review (MD benchmark framework)
- Issue #7 in progress (ASE interface tests)
- Understanding of teacher model baseline performance

### Critical Requirements - MD TRAJECTORY FOCUS
1. **Benchmark full MD simulations**: Not just single inference calls
2. **Measure what matters**: Total trajectory time, per-step latency distribution
3. **Track over time**: JSON output for regression detection
4. **Multiple protocols**: NVE (energy conservation), NVT, NPT
5. **ASE compliance**: Validate Calculator interface for drop-in replacement

**Read these**:
- `/home/aaron/ATX/software/MLFF_Distiller/docs/MD_REQUIREMENTS_UPDATE_SUMMARY.md`
- `/home/aaron/ATX/software/MLFF_Distiller/docs/DROP_IN_COMPATIBILITY_GUIDE.md`

### Key Documentation
- MD Requirements: `/home/aaron/ATX/software/MLFF_Distiller/docs/MD_REQUIREMENTS_UPDATE_SUMMARY.md`
- Drop-in Guide: `/home/aaron/ATX/software/MLFF_Distiller/docs/DROP_IN_COMPATIBILITY_GUIDE.md`
- Agent Protocols: `/home/aaron/ATX/software/MLFF_Distiller/docs/AGENT_PROTOCOLS.md`
- Issue #4: https://github.com/atfrank/MLFF-Distiller/issues/4
- Issue #5: https://github.com/atfrank/MLFF-Distiller/issues/5
- Issue #7: https://github.com/atfrank/MLFF-Distiller/issues/7

### Integration Points
You will coordinate with:
- **Architecture Team**: Need Issue #2 (teacher wrappers) for benchmarks and ASE tests
- **CUDA Team**: Share benchmark framework for their profiling work
- **All Teams**: Test their code, measure their performance improvements

### Communication Protocol
- Check GitHub issues daily (especially Issue #2 progress)
- Update progress on your issues
- Share benchmark results with team
- Tag me (@Lead-Coordinator) if blocked

### Success Criteria
You are successful when:
- MD benchmark framework accurately measures our 5-10x target
- ASE interface tests validate drop-in replacement capability
- Tests catch bugs and regressions early
- Clear metrics guide optimization priorities

### Getting Help
If blocked or need decisions:
- Tag me in issue: @Lead-Coordinator
- Add label: `status:blocked`
- Ask questions about metrics or testing approach

**Action: Claim Issue #4 immediately and start pytest setup! Begin planning Issue #5 (MD benchmarks).**

---

## Coordination Notes for All Agents

### Week 1 Priorities
1. **Architecture Team (Issue #2)**: CRITICAL PATH - blocks multiple teams
2. **Testing Team (Issues #4, #5, #7)**: CRITICAL - defines success metrics
3. **Data Team (Issue #1)**: HIGH - needed for data generation
4. **Training Team (Issue #3)**: HIGH - foundation for distillation
5. **CUDA Team (Issues #8, #9)**: HIGH - enables optimization

### Daily Check-ins
- Check your assigned issues for @mentions
- Update progress with comments
- Flag blockers immediately
- Respond to PR reviews within 24 hours

### Integration Points
- Architecture Issue #2 unblocks: Testing #5 & #7, CUDA #9
- All teams: Use common coding standards (tests, type hints, docs)
- Friday: Integration sync - what worked with what?

### Communication Channels
- GitHub Issues: Technical discussions, blockers, questions
- GitHub PRs: Code reviews, implementation feedback
- Tag @Lead-Coordinator: Architectural decisions, blockers, conflicts

### Success Definition
Week 1 is successful when:
- All agents have active work in progress
- Issue #2 (teacher wrappers) is complete or nearly complete
- Testing infrastructure is working
- No agents are blocked
- Team understands project goals and requirements

**Let's build something amazing! Start by claiming your issues and beginning work.**
