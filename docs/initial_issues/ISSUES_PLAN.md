# Initial GitHub Issues Plan

This document lists all initial issues to be created for the project. Issues are organized by agent and milestone.

## How to Create These Issues

Use the GitHub CLI or web interface to create these issues:

```bash
# Example with gh CLI
gh issue create --title "[Data Pipeline] [M1] Set up data loading infrastructure" \
  --body-file docs/initial_issues/issue_01_data_infrastructure.md \
  --label "agent:data-pipeline,milestone:M1,type:task,priority:high"
```

## Milestone 1 Issues (Setup & Baseline)

### Data Pipeline Engineer (5 issues)

1. **Issue #1**: [Data Pipeline] [M1] Set up data loading infrastructure
   - Labels: `agent:data-pipeline`, `milestone:M1`, `type:task`, `priority:high`
   - Complexity: Medium
   - File: `issue_01_data_infrastructure.md`

2. **Issue #2**: [Data Pipeline] [M1] Create atomic structure data classes
   - Labels: `agent:data-pipeline`, `milestone:M1`, `type:task`, `priority:high`
   - Complexity: Low
   - File: `issue_02_data_classes.md`

3. **Issue #3**: [Data Pipeline] [M1] Implement data validation utilities
   - Labels: `agent:data-pipeline`, `milestone:M1`, `type:task`, `priority:medium`
   - Complexity: Medium
   - File: `issue_03_data_validation.md`

4. **Issue #4**: [Data Pipeline] [M2] Design HDF5 dataset storage format
   - Labels: `agent:data-pipeline`, `milestone:M2`, `type:task`, `priority:high`
   - Complexity: Medium
   - Depends on: #1, #2
   - File: `issue_04_hdf5_storage.md`

5. **Issue #5**: [Data Pipeline] [M2] Implement data generation from teacher models
   - Labels: `agent:data-pipeline`, `milestone:M2`, `type:task`, `priority:high`
   - Complexity: High
   - Depends on: #6 (teacher wrappers)
   - File: `issue_05_data_generation.md`

### ML Architecture Designer (8 issues)

6. **Issue #6**: [Architecture] [M1] Create teacher model wrapper with ASE Calculator interface
   - Labels: `agent:architecture`, `milestone:M1`, `type:task`, `priority:critical`
   - Complexity: High
   - File: `issue_06_teacher_wrappers.md`
   - **UPDATED**: Now emphasizes ASE Calculator interface implementation

7. **Issue #7**: [Architecture] [M1] Analyze Orb-models architecture
   - Labels: `agent:architecture`, `milestone:M1`, `type:research`, `priority:high`
   - Complexity: Medium
   - File: `issue_07_orb_analysis.md`

8. **Issue #8**: [Architecture] [M1] Analyze FeNNol-PMC architecture
   - Labels: `agent:architecture`, `milestone:M1`, `type:research`, `priority:medium`
   - Complexity: Medium
   - File: `issue_08_fennol_analysis.md`

9. **Issue #9**: [Architecture] [M3] Design student model architecture v1 optimized for MD latency
   - Labels: `agent:architecture`, `milestone:M3`, `type:task`, `priority:high`
   - Complexity: Extra High
   - Depends on: #7, #8
   - File: `issue_09_student_architecture.md`

10. **Issue #10**: [Architecture] [M3] Implement model factory and registry
    - Labels: `agent:architecture`, `milestone:M3`, `type:task`, `priority:medium`
    - Complexity: Medium
    - File: `issue_10_model_factory.md`

26. **Issue #26**: [Architecture] [M1] Implement ASE Calculator interface for student models
    - Labels: `agent:architecture`, `milestone:M1`, `type:task`, `priority:critical`
    - Complexity: Medium
    - Depends on: #6
    - File: `issue_26_ase_calculator_student.md`
    - **NEW**: Ensures student models are drop-in replacements

27. **Issue #27**: [Architecture] [M3] Validate drop-in replacement compatibility
    - Labels: `agent:architecture`, `milestone:M3`, `type:task`, `priority:high`
    - Complexity: Medium
    - Depends on: #26, #9
    - File: `issue_27_dropin_validation.md`
    - **NEW**: Tests that student models work in existing MD scripts

28. **Issue #28**: [Architecture] [M6] Implement LAMMPS pair_style interface
    - Labels: `agent:architecture`, `milestone:M6`, `type:task`, `priority:medium`
    - Complexity: High
    - Depends on: #9, #27
    - File: `issue_28_lammps_interface.md`
    - **NEW**: Production MD engine integration

### Distillation Training Engineer (5 issues)

11. **Issue #11**: [Training] [M1] Set up baseline training framework
    - Labels: `agent:training`, `milestone:M1`, `type:task`, `priority:high`
    - Complexity: Medium
    - File: `issue_11_training_framework.md`

12. **Issue #12**: [Training] [M1] Implement training configuration system
    - Labels: `agent:training`, `milestone:M1`, `type:task`, `priority:medium`
    - Complexity: Medium
    - File: `issue_12_training_config.md`

13. **Issue #13**: [Training] [M4] Design distillation loss functions
    - Labels: `agent:training`, `milestone:M4`, `type:task`, `priority:high`
    - Complexity: High
    - Depends on: #11
    - File: `issue_13_loss_functions.md`

14. **Issue #14**: [Training] [M4] Set up training monitoring and logging
    - Labels: `agent:training`, `milestone:M4`, `type:task`, `priority:medium`
    - Complexity: Medium
    - File: `issue_14_training_monitoring.md`

15. **Issue #15**: [Training] [M4] Implement hyperparameter tuning pipeline
    - Labels: `agent:training`, `milestone:M4`, `type:task`, `priority:medium`
    - Complexity: High
    - Depends on: #13
    - File: `issue_15_hyperparameter_tuning.md`

### CUDA Optimization Engineer (6 issues)

16. **Issue #16**: [CUDA] [M1] Set up CUDA development environment
    - Labels: `agent:cuda`, `milestone:M1`, `type:task`, `priority:high`
    - Complexity: Low
    - File: `issue_16_cuda_environment.md`

17. **Issue #17**: [CUDA] [M1] Create performance profiling framework for MD workloads
    - Labels: `agent:cuda`, `milestone:M1`, `type:task`, `priority:high`
    - Complexity: Medium
    - File: `issue_17_profiling_framework.md`
    - **UPDATED**: Focuses on MD trajectory profiling

18. **Issue #18**: [CUDA] [M1] Profile teacher model on MD trajectories
    - Labels: `agent:cuda`, `milestone:M1`, `type:task`, `priority:high`
    - Complexity: Medium
    - Depends on: #6, #17
    - File: `issue_18_profile_teacher.md`
    - **UPDATED**: Profiles repeated inference over MD trajectories

19. **Issue #19**: [CUDA] [M5] Implement custom CUDA kernels optimized for latency
    - Labels: `agent:cuda`, `milestone:M5`, `type:task`, `priority:high`
    - Complexity: Extra High
    - File: `issue_19_cuda_kernels.md`
    - **UPDATED**: Emphasizes latency over throughput

20. **Issue #20**: [CUDA] [M5] Optimize memory usage for long MD trajectories
    - Labels: `agent:cuda`, `milestone:M5`, `type:task`, `priority:high`
    - Complexity: High
    - File: `issue_20_memory_optimization.md`
    - **UPDATED**: Focuses on stable memory during long runs

33. **Issue #33**: [CUDA] [M5] Minimize per-call overhead for repeated inference
    - Labels: `agent:cuda`, `milestone:M5`, `type:task`, `priority:high`
    - Complexity: High
    - Depends on: #19
    - File: `issue_33_repeated_inference_optimization.md`
    - **NEW**: Optimizes for millions of repeated calls in MD

### Testing & Benchmark Engineer (9 issues)

21. **Issue #21**: [Testing] [M1] Configure pytest and test infrastructure
    - Labels: `agent:testing`, `milestone:M1`, `type:task`, `priority:high`
    - Complexity: Low
    - File: `issue_21_pytest_setup.md`

22. **Issue #22**: [Testing] [M1] Create MD simulation benchmark framework
    - Labels: `agent:testing`, `milestone:M1`, `type:task`, `priority:critical`
    - Complexity: Medium
    - File: `issue_22_benchmark_framework.md`
    - **UPDATED**: Now focuses on MD trajectory benchmarks, not just single inference

23. **Issue #23**: [Testing] [M1] Establish baseline MD trajectory benchmarks
    - Labels: `agent:testing`, `milestone:M1`, `type:task`, `priority:critical`
    - Complexity: Medium
    - Depends on: #6, #22
    - File: `issue_23_baseline_benchmarks.md`
    - **UPDATED**: Benchmarks full MD trajectories with teacher models

24. **Issue #24**: [Testing] [M2] Implement data pipeline tests
    - Labels: `agent:testing`, `milestone:M2`, `type:task`, `priority:medium`
    - Complexity: Medium
    - Depends on: #1, #2, #3
    - File: `issue_24_data_tests.md`

25. **Issue #25**: [Testing] [M5] Create MD performance regression test suite
    - Labels: `agent:testing`, `milestone:M5`, `type:task`, `priority:medium`
    - Complexity: Medium
    - File: `issue_25_regression_tests.md`
    - **UPDATED**: Tests for MD trajectory performance regressions

29. **Issue #29**: [Testing] [M1] Implement ASE Calculator interface tests
    - Labels: `agent:testing`, `milestone:M1`, `type:task`, `priority:critical`
    - Complexity: Medium
    - Depends on: #6
    - File: `issue_29_ase_interface_tests.md`
    - **NEW**: Validates ASE Calculator interface compliance

30. **Issue #30**: [Testing] [M3] Create drop-in replacement validation tests
    - Labels: `agent:testing`, `milestone:M3`, `type:task`, `priority:high`
    - Complexity: Medium
    - Depends on: #26, #27
    - File: `issue_30_dropin_tests.md`
    - **NEW**: Tests that student models work without modifying MD scripts

31. **Issue #31**: [Testing] [M3] Implement energy conservation tests for MD
    - Labels: `agent:testing`, `milestone:M3`, `type:task`, `priority:high`
    - Complexity: Medium
    - Depends on: #26
    - File: `issue_31_energy_conservation_tests.md`
    - **NEW**: Validates NVE energy conservation

32. **Issue #32**: [Testing] [M6] Create LAMMPS integration tests
    - Labels: `agent:testing`, `milestone:M6`, `type:task`, `priority:medium`
    - Complexity: High
    - Depends on: #28
    - File: `issue_32_lammps_tests.md`
    - **NEW**: Tests LAMMPS pair_style integration

## New Issues Summary

**Total New Issues Added**: 8
- Architecture: 3 new issues (#26, #27, #28)
- Testing: 4 new issues (#29, #30, #31, #32)
- CUDA: 1 new issue (#33)

**Updated Issues**: 6
- #6: Now emphasizes ASE Calculator interface
- #9: Now optimized for MD latency
- #22: Now focuses on MD trajectories
- #23: Now benchmarks full MD trajectories
- #18: Now profiles MD trajectories
- #19, #20: Now focus on latency and long MD runs

## Issue Creation Order

**Priority 1 (Create immediately - MD Critical)**: Issues 1, 2, 6, 7, 11, 16, 17, 21, 22, 26, 29
**Priority 2 (Create after Priority 1 started)**: Issues 3, 8, 12, 18, 23, 30
**Priority 3 (Create during M1)**: Issues 4, 5, 10, 14, 24, 27, 31
**Priority 4 (Create approaching M3-M5)**: Issues 9, 13, 15, 19, 20, 25, 33
**Priority 5 (Create approaching M6)**: Issues 28, 32

## Labels to Create

Create these labels in the GitHub repository:

**Agent Labels**:
- `agent:data-pipeline` (Color: #0E8A16)
- `agent:architecture` (Color: #1D76DB)
- `agent:training` (Color: #5319E7)
- `agent:cuda` (Color: #D93F0B)
- `agent:testing` (Color: #0052CC)

**Milestone Labels**:
- `milestone:M1` (Color: #C2E0C6)
- `milestone:M2` (Color: #BFDADC)
- `milestone:M3` (Color: #C5DEF5)
- `milestone:M4` (Color: #D4C5F9)
- `milestone:M5` (Color: #F9D0C4)
- `milestone:M6` (Color: #FEF2C0)

**Type Labels**:
- `type:task` (Color: #0075CA)
- `type:bug` (Color: #D73A4A)
- `type:feature` (Color: #A2EEEF)
- `type:research` (Color: #7057FF)
- `type:refactor` (Color: #FBCA04)
- `type:docs` (Color: #0E8A16)

**Priority Labels**:
- `priority:critical` (Color: #B60205)
- `priority:high` (Color: #D93F0B)
- `priority:medium` (Color: #FBCA04)
- `priority:low` (Color: #0E8A16)

**Status Labels**:
- `status:blocked` (Color: #D73A4A)
- `status:in-progress` (Color: #0075CA)
- `status:needs-review` (Color: #FBCA04)
- `status:needs-decision` (Color: #7057FF)

---

**Total Issues**: 33 (25 original + 8 new)
**M1 Issues**: 20 (15 original + 5 new)
**M2 Issues**: 4
**M3 Issues**: 5 (2 original + 3 new)
**M4 Issues**: 3
**M5 Issues**: 2 (1 original + 1 new)
**M6 Issues**: 2 (new)

**Critical MD/Interface Issues**: 11 issues (#6, #22, #23, #26, #27, #28, #29, #30, #31, #32, #33)
