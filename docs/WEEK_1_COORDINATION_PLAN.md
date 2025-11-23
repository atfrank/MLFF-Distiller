# Week 1 Coordination Plan

## Overview
Week 1 (Nov 23-29, 2025) focuses on establishing foundational infrastructure and critical path work. Success means all teams are productive, Issue #2 (teacher wrappers) is complete, and no agents are blocked.

---

## Critical Path Analysis

### BLOCKING ISSUE: #2 (Teacher Model Wrappers)
**Owner**: ML Architecture Designer
**Priority**: CRITICAL
**Blocks**: Issues #5, #7, #9 (and indirectly many M2 issues)

**Why Critical**:
- Testing team needs it for MD benchmarks (#5) and ASE tests (#7)
- CUDA team needs it for profiling (#9)
- Data team needs it for data generation (M2)

**Action Plan**:
- Architecture agent: Start Issue #2 IMMEDIATELY (Monday morning)
- Lead Coordinator: Daily check-ins on Issue #2 progress
- Architecture agent: Create early PR for feedback (by Wednesday)
- Target: Issue #2 complete or in final review by Friday

**Mitigation if Delayed**:
- Testing team can work on pytest setup (#4) and ASE test design (#7) without implementation
- CUDA team can work on environment setup (#8) and profiling framework design (#9)
- Data team continues with data infrastructure (#1)

---

## Team-by-Team Week 1 Plan

### Agent 1: Data Pipeline Engineer

**Primary Issue**: #1 (Data loading infrastructure)

**Week 1 Schedule**:
- **Monday-Tuesday**:
  - Read Issue #1 fully
  - Claim issue, create branch
  - Design data loading interfaces
  - Start implementation of core loaders
- **Wednesday-Thursday**:
  - Complete data loaders
  - Write unit tests
  - Add validation utilities
  - Integration testing
- **Friday**:
  - Create PR for review
  - Document usage
  - Support other teams using data loaders

**Deliverables**:
- Data loading infrastructure functional
- Tests passing
- PR created
- Ready for M2 data generation work

**Dependencies**: None (can start immediately)

**Integration Points**:
- Friday: Coordinate with architecture team on data format for teacher models

---

### Agent 2: ML Architecture Designer

**Primary Issues**: #2 (Teacher wrappers - CRITICAL), #6 (Student calculator - start design)

**Week 1 Schedule**:
- **Monday Morning** (URGENT):
  - Claim Issue #2 immediately
  - Read ASE Calculator documentation thoroughly
  - Study existing ASE Calculator implementations
  - Create branch: feature/teacher-calculator-wrappers
- **Monday Afternoon - Tuesday**:
  - Implement OrbTeacherCalculator skeleton
  - Implement core calculate() method
  - Handle ASE Atoms input correctly
  - Early PR for feedback (Tuesday EOD)
- **Wednesday-Thursday**:
  - Complete OrbTeacherCalculator
  - Start FeNNolTeacherCalculator
  - Write unit tests for both
  - Test with ASE MD integrators (VelocityVerlet, Langevin)
- **Friday**:
  - Final testing and documentation
  - PR ready for final review and merge
  - Notify testing team and CUDA team that calculators are ready
  - Begin design work for Issue #6 (student calculator)

**Deliverables**:
- Teacher calculators working in ASE MD simulations
- Tests passing (unit + integration)
- PR merged or in final review
- Unblock Issues #5, #7, #9

**Dependencies**: None for start; Orb-models and FeNNol-PMC libraries must be available

**Integration Points**:
- Tuesday: Early PR for lead coordinator review
- Wednesday: Notify testing team of progress
- Friday: Hand off to testing/CUDA teams for benchmarks/profiling

**CRITICAL**: This is the blocking issue for Week 1. Daily updates required.

---

### Agent 3: Distillation Training Engineer

**Primary Issue**: #3 (Training framework)

**Week 1 Schedule**:
- **Monday-Tuesday**:
  - Claim Issue #3
  - Design training framework architecture
  - Set up configuration system (YAML/Hydra)
  - Create branch: feature/training-framework
- **Wednesday-Thursday**:
  - Implement basic training loop
  - Add validation loop
  - Implement checkpointing
  - Add logging (tensorboard/wandb)
  - Write tests
- **Friday**:
  - Integration testing
  - Create PR
  - Document configuration options

**Deliverables**:
- Training framework functional
- Can run basic training loop with dummy data
- Configuration system working
- PR created

**Dependencies**: None for basic framework (can use dummy data)

**Integration Points**:
- Friday: Coordinate with data team on data loader integration

---

### Agent 4: CUDA Optimization Engineer

**Primary Issues**: #8 (CUDA environment), #9 (Profiling framework)

**Week 1 Schedule**:
- **Monday**:
  - Claim Issue #8
  - Install CUDA toolkit
  - Verify PyTorch CUDA support
  - Install nsys, ncu profilers
- **Tuesday**:
  - Create environment verification script
  - Document setup in docs/cuda_setup.md
  - Test profiling tools
  - Complete Issue #8, create PR
- **Wednesday** (after Issue #2 progress):
  - Claim Issue #9
  - Design profiling framework for MD trajectories
  - Create branch: feature/md-profiling
- **Thursday-Friday**:
  - Implement MD profiling framework
  - Test with teacher calculators (from Issue #2)
  - Profile sample MD trajectories
  - Create PR

**Deliverables**:
- CUDA environment fully set up (Issue #8 complete)
- Profiling framework implemented (Issue #9 in progress/review)
- Can profile MD trajectories
- Initial profiling data on teacher models

**Dependencies**:
- Issue #8: None
- Issue #9: Depends on Issue #2 (teacher wrappers) - will be available Wed/Thu

**Integration Points**:
- Wednesday: Check Issue #2 progress, coordinate with architecture team
- Thursday: Use teacher calculators for profiling
- Friday: Share profiling results with team

---

### Agent 5: Testing & Benchmarking Engineer

**Primary Issues**: #4 (Pytest setup), #5 (MD benchmarks), #7 (ASE interface tests)

**Week 1 Schedule**:
- **Monday**:
  - Claim Issue #4
  - Configure pytest
  - Set up test fixtures
  - Configure coverage tools
  - Create branch: feature/test-infrastructure
- **Tuesday**:
  - Create CI/CD workflow (.github/workflows/tests.yml)
  - Document testing standards
  - Complete Issue #4, create PR
  - Claim Issue #5 and #7
- **Wednesday** (in parallel):
  - **Issue #5**: Design MD benchmark framework
    - Plan benchmark metrics
    - Design API for running MD benchmarks
  - **Issue #7**: Design ASE interface tests
    - Plan test cases for Calculator interface
- **Thursday** (after Issue #2 progress):
  - **Issue #5**: Implement MD benchmark framework
    - Use teacher calculators from Issue #2
  - **Issue #7**: Implement ASE interface tests
    - Test teacher calculators
- **Friday**:
  - Complete Issue #5 and #7 implementations
  - Run benchmarks on teacher models
  - Document baseline performance
  - Create PRs

**Deliverables**:
- Pytest infrastructure working (Issue #4 complete)
- MD benchmark framework functional (Issue #5 in progress/review)
- ASE interface tests implemented (Issue #7 in progress/review)
- Baseline teacher model performance documented

**Dependencies**:
- Issue #4: None
- Issues #5, #7: Depend on Issue #2 (teacher wrappers)

**Integration Points**:
- Tuesday: Notify teams that CI/CD is ready
- Wednesday: Check Issue #2 progress
- Thursday: Use teacher calculators for benchmarks and tests
- Friday: Share baseline performance results with team

---

## Daily Coordination Schedule

### Monday (Day 1)
**Morning**:
- All agents: Read kickoff messages
- All agents: Claim assigned issues
- Architecture agent: START ISSUE #2 IMMEDIATELY
- Other agents: Create branches, begin work

**Afternoon**:
- Lead coordinator: Check all agents have claimed issues
- Lead coordinator: Verify Issue #2 (critical path) has started

**EOD Check**:
- All agents have active work in progress
- Issue #2 has initial progress
- No immediate blockers

---

### Tuesday (Day 2)
**Morning**:
- Architecture agent: Create early PR for Issue #2 (for feedback)
- Testing agent: Complete Issue #4 (pytest setup)

**Afternoon**:
- Lead coordinator: Review Issue #2 PR, provide feedback
- Testing agent: Begin planning Issues #5 and #7

**EOD Check**:
- Issue #2 PR exists and is under review
- Issue #4 complete or nearly complete
- Other issues progressing

---

### Wednesday (Day 3 - MID-WEEK SYNC)
**Morning**:
- Architecture agent: Address PR feedback on Issue #2
- Testing agent: Check Issue #2 progress, coordinate on calculator usage
- CUDA agent: Check Issue #2 progress, coordinate on profiling

**Afternoon**:
- All agents: Update issue comments with progress
- Lead coordinator: Assess if Issue #2 on track for Friday completion
- Lead coordinator: Identify any blockers

**EOD Check**:
- Issue #2 substantially complete (90%+)
- Testing and CUDA teams ready to integrate
- No critical blockers

---

### Thursday (Day 4)
**Morning**:
- Architecture agent: Final work on Issue #2
- Testing agent: Begin using teacher calculators for Issues #5, #7
- CUDA agent: Begin using teacher calculators for Issue #9

**Afternoon**:
- Architecture agent: Final testing, documentation
- Other teams: Integration work with teacher calculators

**EOD Check**:
- Issue #2 ready for merge or in final review
- Integration work proceeding smoothly
- Other issues progressing well

---

### Friday (Day 5 - WEEK 1 COMPLETION)
**Morning**:
- All agents: Finalize Week 1 work
- Create PRs for review
- Write documentation

**Afternoon - WEEK 1 RETROSPECTIVE**:
- Lead coordinator: Review all Week 1 issues
- Assess: What's complete? What's in review? What's blocked?
- Identify: What worked well? What needs adjustment?
- Plan: Week 2 priorities

**EOD Success Criteria**:
- Issue #2 merged (CRITICAL)
- Issue #4 merged (pytest infrastructure)
- Issues #1, #3, #8 complete or in review
- Issues #5, #7, #9 in progress (50%+ complete)
- No agents blocked for Week 2 work

---

## Blocker Management Protocol

### If Issue #2 (Teacher Wrappers) Gets Blocked
**Immediate Actions**:
1. Architecture agent: Tag @Lead-Coordinator in issue with "blocked" label
2. Describe blocker clearly: Technical? Missing dependency? Needs decision?
3. Lead coordinator responds within 2 hours

**Mitigation Options**:
- **Technical blocker**: Lead coordinator makes architectural decision
- **Missing dependency**: Find workaround or stub implementation
- **Needs clarification**: Lead coordinator provides specification
- **Too complex**: Split into smaller issues, complete core functionality first

**Fallback Plan**:
- Testing team: Work on test design, pytest setup
- CUDA team: Work on environment setup, framework design
- Continue with mock/stub calculators if needed

### If Any Other Issue Gets Blocked
1. Agent: Add "status:blocked" label
2. Agent: Comment with clear description of blocker
3. Agent: Tag @Lead-Coordinator
4. Lead coordinator: Respond within 4 hours
5. Find parallel work or workaround

---

## Integration Points

### Architecture ↔ Testing
- **Handoff**: Issue #2 (teacher calculators) → Issues #5, #7 (benchmarks, tests)
- **Timing**: Wednesday/Thursday
- **What**: Testing team needs working calculator implementations
- **Validation**: Can run simple MD trajectory using calculators

### Architecture ↔ CUDA
- **Handoff**: Issue #2 (teacher calculators) → Issue #9 (profiling)
- **Timing**: Thursday
- **What**: CUDA team needs calculators to profile
- **Validation**: Can profile MD trajectory performance

### Data ↔ Training
- **Handoff**: Issue #1 (data loaders) → Issue #3 (training framework)
- **Timing**: Friday
- **What**: Training team needs to understand data loader API
- **Validation**: Can load dummy data into training loop

### Testing ↔ All Teams
- **Handoff**: Issue #4 (pytest setup) → All teams
- **Timing**: Tuesday
- **What**: All teams use pytest infrastructure for their tests
- **Validation**: CI runs tests for all PRs

---

## Communication Expectations

### Daily Updates
Each agent should:
- Check assigned issues for @mentions (at least twice daily)
- Update issue comments with progress (daily)
- Flag blockers immediately (don't wait)
- Respond to PR comments (within 24 hours)

### Lead Coordinator Daily Check
- Morning: Review all active issues for progress and blockers
- Afternoon: Check Issue #2 (critical path) status
- EOD: Verify no agents are blocked for next day

### GitHub Etiquette
- **Claiming issues**: Comment "Starting work on this" + self-assign + add "status:in-progress"
- **Asking questions**: Tag relevant people, be specific
- **Reporting blockers**: Use "status:blocked" label, describe clearly, suggest solutions if possible
- **PR ready**: Tag lead coordinator for review

---

## Success Metrics for Week 1

### Must Have (Critical Success)
- [ ] Issue #2 (teacher wrappers) merged - UNBLOCKS MULTIPLE TEAMS
- [ ] Issue #4 (pytest setup) merged - ENABLES TESTING
- [ ] All agents have active work in progress
- [ ] No agents blocked for Week 2

### Should Have (High Priority)
- [ ] Issue #1 (data loading) complete or in final review
- [ ] Issue #3 (training framework) complete or in final review
- [ ] Issue #8 (CUDA environment) complete
- [ ] Issues #5, #7, #9 at 50%+ completion

### Nice to Have (Bonus)
- [ ] All Week 1 issues complete or in review
- [ ] Baseline performance metrics documented
- [ ] Integration testing between components successful
- [ ] Team velocity and capacity understood

---

## Week 2 Preview

Based on Week 1 completion, Week 2 will focus on:
- **M2 Milestone Issues**: Data generation from teacher models
- **Continued M1 Work**: Complete Issues #5, #7, #9 if not done
- **Architecture Research**: Begin analyzing Orb/FeNNol architectures (M1)
- **Integration**: Components working together smoothly
- **Performance Baselines**: Full MD benchmark results documented

**Week 2 Critical Path**: Issue #5 (MD benchmarks) must complete to establish baseline for measuring progress.

---

## Escalation Process

### Level 1: Agent Self-Resolution
Agent attempts to resolve blocker independently (1 hour)

### Level 2: Issue Discussion
Agent posts in issue, tags relevant collaborators (2 hours)

### Level 3: Lead Coordinator
Agent tags @Lead-Coordinator with "status:blocked" (immediate response)

### Level 4: Architectural Decision
Lead coordinator makes decision or creates RFC for major changes (4 hours)

### Level 5: Stakeholder Input
Tag issue with "status:needs-decision" for project sponsor input (24 hours)

---

## Week 1 Checklist for Lead Coordinator

**Monday**:
- [ ] Verify all agents received kickoff messages
- [ ] Confirm Issue #2 work started
- [ ] Check all issues claimed

**Tuesday**:
- [ ] Review Issue #2 early PR
- [ ] Verify pytest infrastructure progressing
- [ ] Address any blockers

**Wednesday** (Mid-week sync):
- [ ] Assess Issue #2 completion percentage
- [ ] Check integration readiness
- [ ] Identify risks to Week 1 goals

**Thursday**:
- [ ] Monitor integration work (testing, CUDA using calculators)
- [ ] Push for Issue #2 completion
- [ ] Prepare for Friday retrospective

**Friday**:
- [ ] Conduct Week 1 retrospective
- [ ] Assess what's complete vs in-progress
- [ ] Plan Week 2 priorities
- [ ] Unblock any issues for Week 2
- [ ] Document lessons learned

---

**Week 1 Goal: Establish solid foundation, complete critical path work, enable all teams to be productive in Week 2 and beyond.**
