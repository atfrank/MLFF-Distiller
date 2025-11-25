# M7 Phase Plan: Deployment & Integration

**Phase Duration**: 2 weeks
**Start Date**: November 25, 2025
**Target Completion**: December 9, 2025
**Phase Status**: INITIATED

---

## Executive Summary

M7 is the final phase of the ML Force Field Distillation project, focused on packaging the production-validated student model for public release. The model has been validated through M6 with PRODUCTION APPROVED status (427K parameters, 0.14% energy drift).

### Phase Objectives

1. Create pip-installable package (`pip install mlff-distiller`)
2. Establish robust CI/CD pipeline with automated testing and releases
3. Write comprehensive user documentation
4. Build example Jupyter notebooks
5. Tag first official release (v0.1.0)

---

## Prerequisites (Completed)

| Prerequisite | Status | Reference |
|-------------|--------|-----------|
| Original Student Model PRODUCTION APPROVED | DONE | Issue #33 |
| MD Test Framework | DONE | Issue #37 |
| Performance Benchmarks | DONE | Issue #36 |
| Model Checkpoint | DONE | `checkpoints/best_model.pt` |

---

## GitHub Issues

| Issue | Title | Agent | Priority | Status |
|-------|-------|-------|----------|--------|
| #39 | M7 Phase Coordination | Coordinator | CRITICAL | OPEN |
| #40 | Package Structure for pip install | Architecture | HIGH | OPEN |
| #41 | CI/CD Pipeline | Testing | HIGH | OPEN |
| #42 | User Documentation | Documentation | MEDIUM | OPEN |
| #43 | Example Notebooks | Architecture | MEDIUM | OPEN |
| #44 | Fix Trainer Tests | Training | MEDIUM | OPEN |

---

## Detailed Timeline

### Week 1: Days 1-3 (Nov 25-27)

**Focus**: Package Structure & CI/CD Foundation

| Day | Task | Agent | Issue |
|-----|------|-------|-------|
| 1 | Verify pyproject.toml production-ready | Architecture | #40 |
| 1 | Add entry points for CLI (if applicable) | Architecture | #40 |
| 1-2 | Test `pip install -e .` works cleanly | Architecture | #40 |
| 2-3 | Enhance CI workflow with release automation | Testing | #41 |
| 2-3 | Add PyPI publishing workflow | Testing | #41 |
| 3 | Create MANIFEST.in if needed | Architecture | #40 |

**Milestone Check (Day 3)**:
- [ ] `pip install -e .` works
- [ ] `python -m build` creates dist/
- [ ] CI runs on all PRs
- [ ] Release workflow ready (not triggered)

### Week 1: Days 4-5 (Nov 28-29)

**Focus**: Documentation & Test Fixes

| Day | Task | Agent | Issue |
|-----|------|-------|-------|
| 4 | Update README with installation instructions | Documentation | #42 |
| 4 | Create Quick Start guide | Documentation | #42 |
| 4-5 | Fix trainer test failures | Training | #44 |
| 5 | Write API reference for StudentCalculator | Documentation | #42 |

**Milestone Check (Day 5)**:
- [ ] README updated with pip install instructions
- [ ] Quick start guide written
- [ ] At least 6/12 trainer tests fixed

### Week 2: Days 6-8 (Dec 1-3)

**Focus**: Example Notebooks & Remaining Documentation

| Day | Task | Agent | Issue |
|-----|------|-------|-------|
| 6 | Create 01_quickstart.ipynb | Architecture | #43 |
| 6-7 | Create 02_md_simulation.ipynb | Architecture | #43 |
| 7 | Create 03_geometry_optimization.ipynb | Architecture | #43 |
| 7-8 | Complete API documentation | Documentation | #42 |
| 8 | Fix remaining trainer tests | Training | #44 |

**Milestone Check (Day 8)**:
- [ ] 3 example notebooks complete
- [ ] API documentation complete
- [ ] All trainer tests pass

### Week 2: Days 9-10 (Dec 4-5)

**Focus**: Final Review & Release

| Day | Task | Agent | Issue |
|-----|------|-------|-------|
| 9 | Create 04_batch_processing.ipynb | Architecture | #43 |
| 9 | Final documentation review | Coordinator | #42 |
| 9 | Integration testing of full package | Testing | #41 |
| 10 | Tag v0.1.0 release | Coordinator | #39 |
| 10 | Publish to PyPI (if approved) | Coordinator | #39 |

**Milestone Check (Day 10)**:
- [ ] All notebooks run without errors
- [ ] Documentation complete and reviewed
- [ ] v0.1.0 tagged
- [ ] Release notes written

---

## Current State Assessment

### Package Structure (Issue #40)

**Status**: Partially Complete

The repository already has a well-configured `pyproject.toml`:
- Package name: `mlff-distiller`
- Version: `0.1.0`
- Dependencies properly specified
- Optional dependencies: `dev`, `cuda`, `docs`
- Build system: setuptools

**Remaining Work**:
- Verify clean installation
- Test package distribution build
- Add any missing entry points

### CI/CD Pipeline (Issue #41)

**Status**: Partially Complete

Existing `.github/workflows/ci.yml` includes:
- Multi-Python testing (3.9, 3.10, 3.11)
- Black formatting check
- isort import check
- mypy type checking
- pytest with coverage
- Codecov integration
- Ruff linting

**Remaining Work**:
- Add release workflow for tagged versions
- Configure PyPI publishing
- Add benchmark regression testing

### Test Status (Issue #44)

**Status**: 17 tests collected, some failing

Tests in `tests/unit/test_trainer.py`:
- TestTrainerInitialization: 4 tests
- TestTrainingLoop: 3 tests
- TestCheckpointing: 3 tests
- TestEarlyStopping: 1 test
- TestGradientHandling: 2 tests
- TestLossFunctionIntegration: 1 test
- TestMetricTracking: 2 tests
- TestReproducibility: 1 test

---

## Sub-Agent Assignments

### Agent 2: Architecture Specialist

**Assigned Issues**: #40, #43

**Responsibilities**:
1. Verify package structure for pip installation
2. Create entry points if CLI tools needed
3. Build all 4 example notebooks
4. Ensure notebooks use packaged model

### Agent 3: Training Engineer

**Assigned Issues**: #44

**Responsibilities**:
1. Fix all 17 trainer tests
2. Align trainer module with current model interface
3. Ensure no test regressions

### Agent 5: Testing & Benchmarking Engineer

**Assigned Issues**: #41

**Responsibilities**:
1. Enhance CI workflow
2. Create release automation workflow
3. Configure PyPI publishing
4. Add benchmark regression detection

### Coordinator

**Assigned Issues**: #39, #42

**Responsibilities**:
1. Monitor progress across all agents
2. Write/review documentation
3. Resolve blockers within 2-4 hours
4. Final approval of release artifacts
5. Tag v0.1.0 release

---

## Dependencies & Risk Mitigation

### Dependency Graph

```
Issue #40 (Package) ──┬──> Issue #42 (Docs) ──> Issue #43 (Notebooks)
                      │
Issue #41 (CI/CD) ────┤
                      │
Issue #44 (Tests) ────┴──> v0.1.0 Release
```

### Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Trainer test failures complex | HIGH | MEDIUM | Parallel work on other issues |
| PyPI publishing issues | MEDIUM | LOW | Test with TestPyPI first |
| Notebook dependencies | MEDIUM | LOW | Use minimal example structures |
| CI workflow failures | MEDIUM | LOW | Iterative testing |

---

## Success Criteria

**Phase Complete When**:

1. [ ] `pip install mlff-distiller` works from PyPI or test index
2. [ ] All 17 trainer tests pass
3. [ ] CI/CD pipeline green on main branch
4. [ ] 4 example notebooks execute without errors
5. [ ] README includes clear installation and usage instructions
6. [ ] API documentation covers StudentCalculator and StudentModel
7. [ ] v0.1.0 release tagged with release notes
8. [ ] No critical or high-priority issues open in M7

---

## Communication Protocol

**Daily Sync** (Virtual):
- Coordinator reviews all M7 issue comments
- Updates this plan with progress
- Identifies and resolves blockers

**Escalation Path**:
1. Agent identifies blocker
2. Labels issue with "blocked"
3. Coordinator responds within 2-4 hours
4. Architectural decisions documented in issue

**Progress Updates**:
- Update issue checkboxes as tasks complete
- Comment on issues with meaningful progress
- Cross-reference related PRs

---

## Immediate Next Actions

1. **Architecture Agent**: Start Issue #40 - verify `pip install -e .` works
2. **Testing Agent**: Start Issue #41 - review CI workflow, plan release automation
3. **Training Agent**: Start Issue #44 - run trainer tests, identify failure patterns
4. **Coordinator**: Monitor progress, prepare documentation outline for Issue #42

---

## Appendix: Key Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Package configuration |
| `.github/workflows/ci.yml` | CI pipeline |
| `checkpoints/best_model.pt` | Production model (427K params) |
| `src/mlff_distiller/` | Main package source |
| `tests/unit/test_trainer.py` | Failing tests to fix |

---

*Document created: November 25, 2025*
*Phase Status: INITIATED*
*Next Update: End of Day 1*
