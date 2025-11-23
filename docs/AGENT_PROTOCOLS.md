# Agent Coordination Protocols

This document defines the workflows, responsibilities, and communication protocols for the specialized agent team working on MLFF Distiller.

## Agent Overview

### Team Structure

```
Lead Coordinator
    ├── Data Pipeline Engineer
    ├── ML Architecture Designer
    ├── Distillation Training Engineer
    ├── CUDA Optimization Engineer
    └── Testing & Benchmark Engineer
```

## Individual Agent Roles

### 1. Data Pipeline Engineer

**Primary Responsibilities**:
- Data generation from teacher models
- Preprocessing and augmentation pipelines
- Dataset management and storage
- Data validation and quality assurance
- DataLoader optimization

**Key Skills**: Python, HDF5, ASE, data engineering, parallel processing

**Typical Issues**:
- `[Data Pipeline] [M1] Set up data loading infrastructure`
- `[Data Pipeline] [M2] Implement data generation from Orb-models`
- `[Data Pipeline] [M2] Create preprocessing pipeline`
- `[Data Pipeline] [M2] Optimize DataLoader for GPU utilization`

**Interfaces**:
- **To ML Architecture**: Provide data format specifications
- **To Distillation Training**: Supply training/validation datasets
- **To Testing**: Provide test datasets and validation criteria

---

### 2. ML Architecture Designer

**Primary Responsibilities**:
- Student model architecture design
- Teacher model integration and analysis
- Model component interfaces
- Architecture optimization for performance
- Research and prototyping new architectures

**Key Skills**: PyTorch, neural architecture design, graph neural networks, equivariant networks

**Typical Issues**:
- `[Architecture] [M1] Analyze teacher model architectures`
- `[Architecture] [M3] Design student model architecture v1`
- `[Architecture] [M3] Implement teacher-student interface`
- `[Architecture] [M3] Benchmark architecture variants`

**Interfaces**:
- **To Data Pipeline**: Define input data requirements
- **To Distillation Training**: Provide model implementations
- **To CUDA Optimization**: Identify optimization opportunities
- **To Testing**: Provide model validation criteria

---

### 3. Distillation Training Engineer

**Primary Responsibilities**:
- Training loop implementation
- Loss function design and tuning
- Hyperparameter optimization
- Training monitoring and logging
- Model validation

**Key Skills**: PyTorch training, optimization algorithms, loss functions, experiment tracking

**Typical Issues**:
- `[Training] [M1] Create baseline training framework`
- `[Training] [M4] Implement distillation loss functions`
- `[Training] [M4] Set up training monitoring with Wandb`
- `[Training] [M4] Hyperparameter tuning pipeline`

**Interfaces**:
- **To Data Pipeline**: Request specific data formats or augmentations
- **To ML Architecture**: Request architecture modifications
- **To Testing**: Provide trained models for validation
- **To CUDA Optimization**: Identify training bottlenecks

---

### 4. CUDA Optimization Engineer

**Primary Responsibilities**:
- CUDA kernel development
- Memory optimization
- Performance profiling and analysis
- Inference optimization
- Batched computation implementation

**Key Skills**: CUDA, C++, PyTorch C++ extensions, performance profiling, GPU architecture

**Typical Issues**:
- `[CUDA] [M1] Set up CUDA development environment`
- `[CUDA] [M5] Implement custom CUDA kernels for inference`
- `[CUDA] [M5] Optimize memory usage`
- `[CUDA] [M5] Profile and optimize bottlenecks`

**Interfaces**:
- **To ML Architecture**: Request architecture changes for optimization
- **To Distillation Training**: Optimize training performance
- **To Testing**: Provide optimized models for benchmarking
- **To Data Pipeline**: Optimize data loading bottlenecks

---

### 5. Testing & Benchmark Engineer

**Primary Responsibilities**:
- Test framework setup
- Unit and integration tests
- Benchmark suite development
- Performance regression detection
- Validation against teacher models

**Key Skills**: pytest, benchmarking, CI/CD, validation metrics

**Typical Issues**:
- `[Testing] [M1] Configure pytest and coverage tools`
- `[Testing] [M1] Create baseline benchmark framework`
- `[Testing] [M2] Implement data validation tests`
- `[Testing] [M5] Performance regression test suite`

**Interfaces**:
- **To All Agents**: Provide testing requirements and feedback
- **To Data Pipeline**: Validate data quality
- **To ML Architecture**: Benchmark model performance
- **To Distillation Training**: Validate model accuracy
- **To CUDA Optimization**: Measure optimization impact

---

## Workflow Protocols

### Daily Workflow

1. **Morning Check** (Virtual - check at your convenience):
   - Review GitHub notifications and @mentions
   - Check assigned issues for updates or blockers
   - Review Project board for your agent column
   - Read comments on your active PRs

2. **Active Work**:
   - Work on current in-progress issue
   - Update issue with progress comments
   - Commit code frequently with clear messages
   - Run tests locally before pushing

3. **End of Day**:
   - Push work to feature branch
   - Update issue with current status
   - Tag coordinator if blocked
   - Move completed items to Review column

### Issue Workflow

```
1. Issue Created → Backlog
2. Agent Claims → In Progress (add comment)
3. Work Started → Update status in issue
4. PR Created → Link in issue comment
5. PR Reviewed → Address feedback
6. PR Merged → Move to Done, close issue
```

### Pull Request Workflow

#### Creating a PR

1. **Before Creating**:
   - Run all tests locally: `pytest tests/`
   - Format code: `black src tests && isort src tests`
   - Check types: `mypy src`
   - Review your own changes

2. **PR Title Format**:
   ```
   [Agent] [Milestone] Brief description

   Examples:
   [Data Pipeline] [M2] Add HDF5 dataset storage
   [Architecture] [M3] Implement student model v1
   [CUDA] [M5] Optimize inference kernel
   ```

3. **PR Description**:
   - Summary of changes
   - Link to related issue(s): `Closes #42`
   - Testing performed
   - Benchmarks (if applicable)
   - Screenshots/examples (if relevant)

4. **Labels**:
   - Add milestone label (e.g., `milestone:M2`)
   - Add agent label (e.g., `agent:data-pipeline`)
   - Add type label (e.g., `type:feature`)

#### Reviewing a PR

1. **Automated Checks**:
   - Wait for CI to pass
   - Check test coverage hasn't decreased
   - Verify benchmarks (if applicable)

2. **Code Review**:
   - Read all changes thoroughly
   - Check for code quality and style
   - Verify tests are comprehensive
   - Ensure documentation is updated
   - Test functionality locally if needed

3. **Providing Feedback**:
   - Be specific and constructive
   - Suggest improvements, don't demand
   - Ask questions for clarification
   - Approve when criteria are met

4. **Approval**:
   - At least 1 approval required
   - All comments addressed
   - CI passes
   - Coordinator review (for architectural changes)

### Handling Blockers

If you encounter a blocker:

1. **Identify the Type**:
   - **Technical**: Unclear how to implement
   - **Dependency**: Waiting on another issue/PR
   - **Resource**: Missing data, access, or tools
   - **Clarification**: Unclear requirements

2. **Immediate Actions**:
   - Add `status:blocked` label to issue
   - Comment with clear description of blocker
   - Tag relevant people:
     - `@coordinator` for decisions
     - `@agent-name` for dependencies
   - Suggest potential solutions if you have ideas

3. **Find Parallel Work**:
   - Look for other issues in your domain
   - Help review PRs
   - Write documentation
   - Improve tests

4. **Follow Up**:
   - Check for responses daily
   - Update issue when blocker is resolved
   - Remove `status:blocked` label when unblocked

## Communication Channels

### GitHub Issues
- **Primary** communication for work items
- Tag people with @username
- Use clear, professional language
- Provide context and links

### GitHub PR Comments
- Code-specific discussions
- Review feedback
- Implementation questions
- Mark resolved when addressed

### Project Board
- Visual status tracking
- Move cards as work progresses
- Keep columns current

### Documentation
- Technical specifications
- API documentation
- Architecture decisions
- Best practices

## Collaboration Patterns

### Cross-Agent Collaboration

#### Pattern 1: Sequential Dependency
```
Agent A completes Issue #1
    → Agent B starts Issue #2 (depends on #1)

Example:
Data Pipeline creates dataset format
    → Architecture designs models using that format
```

**Protocol**:
1. Agent A marks issue as done, notifies Agent B
2. Agent B reviews deliverable
3. Agent B starts dependent work

#### Pattern 2: Parallel Work
```
Agent A works on Issue #1
Agent B works on Issue #2 (independent)
Both complete in parallel
```

**Protocol**:
1. Ensure no conflicts in files/modules
2. Coordinate on shared utilities
3. Merge in sequence to avoid conflicts

#### Pattern 3: Collaborative Task
```
Agent A + Agent B work together on Issue #1

Example:
Architecture + CUDA work together on optimizing model
```

**Protocol**:
1. Create shared issue or linked issues
2. Use PR co-author feature
3. Regular sync via issue comments
4. Clear division of subtasks

### Integration Points

#### Data Pipeline ↔ Architecture
- Data format specification
- Input/output tensor shapes
- Normalization conventions
- Batching requirements

#### Architecture ↔ Distillation Training
- Model API and interfaces
- Loss function requirements
- Training hyperparameters
- Model checkpointing

#### Distillation Training ↔ CUDA Optimization
- Performance bottlenecks
- Memory usage patterns
- Batch size optimization
- Inference requirements

#### All Agents ↔ Testing
- Test requirements
- Validation criteria
- Benchmark specifications
- Bug reports and fixes

## Decision Making

### Agent-Level Decisions
**Agents decide**: Implementation details, algorithm choices, code organization within their domain

**Example**: Data Pipeline Engineer chooses HDF5 vs Zarr for storage

### Coordinator Decisions
**Coordinator decides**: Architecture, interfaces, priorities, blockers, conflicts

**Example**: Choice between model architecture variants

### Escalation Process
1. Agent identifies decision needed
2. Create RFC issue with `needs-decision` label
3. Present options with pros/cons
4. Tag coordinator
5. Discuss in issue comments
6. Coordinator makes final decision
7. Document rationale

## Quality Standards

### Code Quality
- **All code** must pass CI checks
- **All new features** must have tests
- **All public APIs** must have docstrings
- **All PRs** must be reviewed

### Documentation
- Update docs when changing behavior
- Include examples for new features
- Keep README and guides current
- Document design decisions

### Performance
- Benchmark before and after optimizations
- Document performance characteristics
- No regressions without justification
- Profile before optimizing

## Milestone Coordination

### Milestone Start
1. Coordinator creates all issues for milestone
2. Agents review and claim issues
3. Identify dependencies and order
4. Agree on timeline

### During Milestone
1. Daily progress updates in issues
2. Weekly milestone check-in (async)
3. Early flag for blockers or delays
4. Coordinate integrations

### Milestone End
1. Complete all acceptance criteria
2. Run full test suite
3. Update documentation
4. Create milestone report
5. Demo key features
6. Plan next milestone

## Best Practices

### For All Agents

1. **Communicate Early and Often**
   - Don't wait until you're completely stuck
   - Ask questions in issue comments
   - Share work-in-progress for feedback

2. **Write Clean, Maintainable Code**
   - Think about who will read it next
   - Add comments for complex logic
   - Use descriptive names

3. **Test Thoroughly**
   - Test happy path and edge cases
   - Run full test suite before PR
   - Add regression tests for bugs

4. **Document as You Go**
   - Update docstrings with code changes
   - Add examples for new features
   - Keep README current

5. **Be a Good Collaborator**
   - Review others' PRs promptly
   - Provide constructive feedback
   - Be open to feedback on your work
   - Celebrate team successes

### Red Flags to Avoid

- Long-running PRs (>1 week)
- Issues stuck in progress (>3 days)
- Failing tests in main branch
- Undocumented architectural changes
- Decreasing code coverage
- Silent blockers

## Emergency Protocols

### Critical Bug in Main
1. Create hotfix issue with `priority:critical`
2. Tag coordinator immediately
3. Create fix branch from main
4. Fast-track PR review
5. Merge and deploy ASAP

### Major Blocker
1. Label issue `status:blocked` + `priority:high`
2. Tag coordinator with full context
3. Propose workarounds if possible
4. Wait max 24h for response
5. Escalate if no response

### Milestone at Risk
1. Identify issues at risk early
2. Comment on milestone tracking issue
3. Propose: cut scope, extend time, or add resources
4. Coordinator makes final call
5. Update documentation and stakeholders

---

**Questions?** Tag `@coordinator` in any issue or PR.

**Last Updated**: 2025-11-23
