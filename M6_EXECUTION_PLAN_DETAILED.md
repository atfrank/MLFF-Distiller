# M6 Phase Execution Plan - Detailed Implementation
## MD Integration Testing & Validation Phase

**Date**: November 25, 2025
**Coordinator**: Lead Coordinator
**Lead Engineer**: Agent 5 (Testing & Benchmarking)
**Phase Duration**: 12-14 calendar days (Target: December 8-9, 2025)
**Status**: EXECUTION INITIATED

---

## EXECUTIVE SUMMARY

M6 Phase is a critical validation phase that will:
1. Confirm the Original Student Model (427K) is production-ready for MD simulations
2. Build a reusable MD testing framework for future model validation
3. Characterize performance/accuracy tradeoffs for compressed variants
4. Provide clear recommendations for deployment and use cases

**Critical Success Metrics**:
- Original model: 10ps NVE MD without crashes, <1% energy drift, <0.2 eV/Å force RMSE
- Framework: Functional, unit tested, documented, reusable
- Recommendations: Production-ready decision for Original, clear use cases for Tiny/Ultra-tiny

---

## PART 1: IMMEDIATE ACTIONS (NEXT 2 HOURS)

### Action 1: Inventory & Verification (30 minutes)
**Responsible**: Lead Coordinator
**Tasks**:
1. Verify all 6 GitHub issues (#33-#38) are created and properly labeled
2. Confirm checkpoint files exist and are accessible:
   - Original: `/home/aaron/ATX/software/MLFF_Distiller/checkpoints/best_model.pt` (1.72 MB)
   - Tiny: `/home/aaron/ATX/software/MLFF_Distiller/checkpoints/tiny_model/best_model.pt`
   - Ultra-tiny: `/home/aaron/ATX/software/MLFF_Distiller/checkpoints/ultra_tiny_model/best_model.pt`
3. Verify ASE calculator at `/home/aaron/ATX/software/MLFF_Distiller/src/mlff_distiller/inference/ase_calculator.py`
4. Check existing test infrastructure in `/home/aaron/ATX/software/MLFF_Distiller/tests/integration/`

**Checklist**:
- [ ] All 6 issues created and visible in `gh issue list`
- [ ] All 3 checkpoint files load correctly (test with simple Python load)
- [ ] ASE calculator compiles without errors
- [ ] Existing test suite runs successfully (`pytest tests/integration/ -v`)

### Action 2: Background Process Cleanup (20 minutes)
**Responsible**: Lead Coordinator
**Tasks**:
1. Identify GPU-consuming processes from prior work (Phase 3 training/benchmarking)
2. Determine which are still needed (check running jobs, timestamps)
3. Terminate processes that are NOT critical
4. Free up GPU memory for MD testing (target: >80% GPU memory available)

**Commands**:
```bash
# Check for background Python processes
ps aux | grep python | grep -v "grep"

# Check GPU memory usage
nvidia-smi

# If needed, terminate old processes:
# kill -9 [PID]
```

**Note**: Keep any processes with modification time within last 24 hours if unclear.

### Action 3: Agent 5 Onboarding (30 minutes)
**Responsible**: Both Coordinator and Agent 5
**Tasks**:
1. Agent 5 reads documentation in order:
   - `M6_PHASE_INITIATION_REPORT.md` (this document's context)
   - `docs/M6_TESTING_ENGINEER_QUICKSTART.md` (detailed implementation guide)
   - `docs/M6_MD_INTEGRATION_COORDINATION.md` (full coordination plan)
2. Coordinator reviews Agent 5 questions and provides clarity
3. Both verify environment readiness
4. Establish daily standup protocol (Issue #38 comments at 9 AM daily)

**Success Criteria**:
- Agent 5 can explain Issue #37 acceptance criteria in own words
- Agent 5 understands critical path: #37 → #33 → (#34, #35, #36 parallel)
- Both agree on escalation protocol for blockers

### Action 4: Create Initial Standup (10 minutes)
**Responsible**: Lead Coordinator
**Task**: Post initial standup comment in Issue #38

**Template**:
```
## M6 Phase Execution - Day 1 Standup (November 25, 2025)

### Status Summary
- Phase: INITIATED
- Critical Path: Issue #37 (Framework) → Issue #33 (Original Model)
- Parallel Work: Issues #34, #35, #36

### Day 1 Plan
1. Agent 5: Environment setup and documentation review
2. Agent 5: Begin Issue #37 framework architecture design
3. Coordinator: Daily checkpoint availability verification
4. Both: Establish working cadence

### Blockers
None at startup.

### Next Checkpoint
End of Day 1: Framework architecture documented, prototyping begins
```

---

## PART 2: WEEK 1 EXECUTION PLAN (DAYS 1-5)

### DAYS 1-3: Issue #37 - Test Framework Enhancement (CRITICAL BLOCKER)

**Objective**: Build comprehensive MD simulation test infrastructure that unblocks all other work.

**Owner**: Agent 5
**Duration**: 3 calendar days (actual coding: ~12 hours)
**Estimated Effort**: 12-15 hours
**Dependency**: None (can start immediately)
**Blocks**: Issue #33 (Original Model MD Testing)

#### Day 1 Activities

**Morning (2-3 hours)**:
1. Read framework design guide in `M6_TESTING_ENGINEER_QUICKSTART.md`
2. Review existing test infrastructure:
   - `/home/aaron/ATX/software/MLFF_Distiller/tests/integration/test_ase_calculator.py` (387 lines - analyze structure)
   - Existing fixtures and test patterns
3. Plan framework architecture:
   - File structure: `tests/integration/test_md_integration.py` (primary)
   - Additional modules: `src/mlff_distiller/testing/md_harness.py`, `metrics.py`, `benchmark_utils.py`
   - Class hierarchy and method signatures
4. Document architecture decisions in Issue #37 comment

**Expected Output**: Architecture diagram and component list posted in Issue #37

**Afternoon (2-3 hours)**:
1. Begin implementation: `tests/integration/test_md_integration.py`
   - Start with fixtures: water, methane test molecules
   - Implement basic test class structure
   - Create placeholder for MDSimulationHarness

2. Create `/src/mlff_distiller/testing/` directory and stubs:
   - `md_harness.py` with MDSimulationHarness class skeleton
   - `metrics.py` with function stubs
   - `benchmark_utils.py` with decorator stub

3. Get code to compilation state (no errors, basic structure)

**End of Day 1 Checkpoint**:
- [ ] Framework architecture documented in Issue #37
- [ ] File structure created (stubs compile)
- [ ] Progress comment posted

---

**Days 2-3 (Full Implementation)**:

**Day 2 Morning - Core Implementation (4 hours)**:
1. Implement `MDSimulationHarness.run_nve_simulation()`:
   - Initialize ASE atoms with velocities (Maxwell-Boltzmann distribution)
   - Integrate VelocityVerlet from ASE
   - Track all frames (energies, forces, positions)
   - Return trajectory object with metadata

2. Implement energy conservation metrics:
   - `compute_energy_conservation()` function
   - Calculate total energy drift percentage
   - Track kinetic and potential energy separately

3. Unit tests for both components

**Day 2 Afternoon - Force & Trajectory Metrics (4 hours)**:
1. Implement `compute_force_metrics()`:
   - RMSE, MAE calculations
   - Component-wise analysis
   - Per-atom statistics
   - Angular error metrics

2. Implement trajectory analysis tools:
   - `compare_trajectories()` function
   - `analyze_trajectory_stability()` function
   - Anomaly detection

3. Unit tests for metrics

**Day 3 - Benchmarking & Integration (3-4 hours)**:
1. Implement benchmarking utilities:
   - `@benchmark_decorator` for timing
   - `measure_throughput()` function
   - `measure_memory()` function

2. Integration test: Run 100-step (100 fs) simulation end-to-end
   - Verify all components work together
   - Check output format and metrics calculation
   - Performance: Should complete in <2 minutes for water

3. Final code review and documentation:
   - Add docstrings to all public functions
   - Create example usage in comments
   - Verify >80% code coverage on test module

**Expected Deliverables**:
```
/home/aaron/ATX/software/MLFF_Distiller/
├── tests/integration/
│   ├── test_md_integration.py (500+ lines)
│   │   ├── TestMDFramework (unit tests)
│   │   ├── test_nve_harness()
│   │   ├── test_energy_metrics()
│   │   ├── test_force_metrics()
│   │   └── test_end_to_end_integration()
│   └── fixtures.py (updated with MD molecules)
├── src/mlff_distiller/testing/
│   ├── __init__.py
│   ├── md_harness.py (250+ lines)
│   │   ├── MDSimulationHarness class
│   │   ├── run_nve_simulation()
│   │   └── initialize_velocities()
│   ├── metrics.py (300+ lines)
│   │   ├── compute_energy_conservation()
│   │   ├── compute_force_metrics()
│   │   └── analyze_trajectory_stability()
│   └── benchmark_utils.py (150+ lines)
│       ├── @benchmark_decorator
│       └── measure_throughput()
└── docs/
    └── MD_FRAMEWORK_GUIDE.md (usage examples)
```

**Acceptance Criteria for Issue #37** (ALL MUST PASS):
- [ ] Framework supports 10+ ps simulations without memory overflow
- [ ] Energy conservation metrics accurate to machine precision
- [ ] Force metrics match expected ranges from force analysis
- [ ] Easy to add new molecules (demonstrated with 3 test molecules)
- [ ] All functions have unit tests with >80% coverage
- [ ] Documentation with usage examples provided
- [ ] Integration test passes: 100-step water simulation completes
- [ ] Metrics output matches expected format
- [ ] No warnings or deprecation errors

**Day 3 Completion Checklist**:
```
Framework Development:
- [ ] NVE simulation harness fully functional
- [ ] Energy metrics tested and validated
- [ ] Force metrics tested and validated
- [ ] Trajectory analysis utilities working
- [ ] Benchmarking decorators implemented
- [ ] All unit tests passing (pytest shows 100% framework tests pass)
- [ ] Code formatted (black, isort)
- [ ] Type hints added for public APIs

Documentation:
- [ ] Function docstrings complete
- [ ] Usage examples provided
- [ ] Expected output format documented
- [ ] Integration test example included

Issue #37 Ready to Close:
- [ ] Comment posted with test results
- [ ] Benchmark output: "100-step simulation: 8.2s wall time"
- [ ] All acceptance criteria confirmed
- [ ] Approval from Coordinator requested
```

---

### DAYS 2-6: Issue #33 - Original Model MD Testing (CRITICAL - PRODUCTION BLOCKER)

**Objective**: Validate that Original Student Model (427K, R²=0.9958) is stable and accurate for production MD simulations.

**Owner**: Agent 5
**Duration**: 4-5 days (actual testing: ~15 hours)
**Dependency**: Issue #37 must have working framework (starts Day 2 afternoon)
**Blocks**: Production deployment decision

**Note**: This issue can START on Day 2 afternoon (before #37 is finished) using prototype framework, but FULL VALIDATION waits for #37 completion.

#### Day 2 Afternoon - Setup & Quick Test (2 hours)
**Prerequisite**: Framework #37 has basic NVE harness working

1. Verify Original model loads and runs:
```python
import torch
from src.mlff_distiller.inference.ase_calculator import MLFFCalculator
from ase.build import molecule

atoms = molecule('H2O')
calc = MLFFCalculator('checkpoints/best_model.pt')
atoms.calc = calc
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
print(f"Energy: {energy:.4f} eV, Force shape: {forces.shape}")
```

2. Run 100-step quick test to verify framework + model work together
3. Post preliminary result in Issue #33

#### Days 3-4 - Phase 1: Basic Validation (6 hours)
**Test Plan**: Quick 5ps tests on 3 molecules

**Test 1: Water (H2O, 3 atoms)**
```python
# 5 ps simulation: 5000 timesteps × 1fs = 5ps
atoms = molecule('H2O')
atoms.calc = MLFFCalculator('checkpoints/best_model.pt')

trajectory = run_nve_simulation(
    atoms,
    timesteps=5000,
    timestep=1.0,
    temperature=300,
    name='H2O_5ps'
)

# Check metrics
energy_drift = compute_energy_conservation(trajectory)
force_rmse = compute_force_metrics(trajectory, reference_calc='teacher')
```

**Expected Results for Water**:
- Total energy drift: <0.5%
- Force RMSE: <0.15 eV/Å
- Simulation completes without crashes
- Kinetic energy stable (no spikes)

**Test 2: Methane (CH4, 5 atoms)**
- Same procedure as water
- Expected: Slightly more stable than water (heavier atoms)
- Expected drift: <0.5%
- Expected RMSE: <0.16 eV/Å

**Test 3: Alanine (C5H11NO2, 13 atoms)**
- Same procedure as water
- More complex system (larger forces)
- Expected drift: <1%
- Expected RMSE: <0.20 eV/Å

**Daily Checkpoint (End Day 4)**:
- [ ] All 3 molecules: 5ps tests complete
- [ ] Metrics calculated and within expected ranges
- [ ] Results table posted in Issue #33:

```markdown
## Basic Validation Results (Days 3-4)

| Molecule | Natoms | Duration | E-drift | F-RMSE | Status |
|----------|--------|----------|---------|--------|--------|
| H2O      | 3      | 5ps      | 0.42%   | 0.158  | PASS   |
| CH4      | 5      | 5ps      | 0.38%   | 0.162  | PASS   |
| C5H11NO2 | 13     | 5ps      | 0.68%   | 0.197  | PASS   |
```

#### Days 5-6 - Phase 2: Extended Validation (6 hours)

**Extended Test: 10+ ps Trajectories**

**Test 4: Water 10ps Extended**
```python
# 10,000 timesteps = 10ps
trajectory_long = run_nve_simulation(
    atoms_water,
    timesteps=10000,
    timestep=1.0,
    temperature=300,
    name='H2O_10ps_extended'
)
```

**Acceptance Criteria**:
- [ ] Total energy drift remains <1% for full 10ps
- [ ] No sudden spikes or crashes at any point
- [ ] Force RMSE stable (doesn't degrade over time)
- [ ] Kinetic energy shows realistic fluctuations (±5% of mean)
- [ ] Potential energy stable

**Test 5: Temperature Scaling (Day 6)**
**Test water at different temperatures** (if #37 framework permits):
```python
for temp in [100, 300, 500]:
    trajectory = run_nve_simulation(atoms_water, temperature=temp, timesteps=5000)
    # Verify stability at each T
```

**Expected**: Trajectory stability improves at lower T (less kinetic energy), maintains at high T

**Daily Checkpoint (End Day 6)**:
```
## Extended Validation Results (Days 5-6)

Water 10ps:
- E-drift: 0.51%
- F-RMSE: 0.159 eV/Å
- Kinetic stable: YES
- Status: PASS ✅

Temperature Scaling (5ps each):
- 100K: E-drift 0.22%, Status: PASS
- 300K: E-drift 0.42%, Status: PASS
- 500K: E-drift 0.61%, Status: PASS
```

#### Issue #33 Completion (End Day 6)

**Final Deliverables**:
1. Results table with all tests
2. Energy conservation plot showing drift over time
3. Force RMSE distribution plot
4. Production readiness recommendation

**Acceptance Criteria for Issue #33** (ALL MUST PASS):
- [ ] 10ps simulation completes without crashes (any molecule)
- [ ] Total energy drift <1% (measured, not assumed)
- [ ] Kinetic energy shows no sudden spikes
- [ ] Force RMSE during MD <0.2 eV/Å (per-frame average)
- [ ] Per-frame inference time <10ms (GPU)
- [ ] 3+ test molecules validated
- [ ] All metrics stable and documented
- [ ] Production-ready recommendation confirmed

**Example Results Section**:
```
ORIGINAL MODEL (427K) - PRODUCTION VALIDATION
==============================================

Summary:
- All tests PASSED ✅
- Model is PRODUCTION READY for deployment

Results:
- 3 molecules tested, 5-10ps trajectories
- Energy conservation: 0.38-0.68% (all <1%) ✅
- Force accuracy: 0.158-0.197 eV/Å (all <0.2) ✅
- Temperature scaling: Stable 100K-500K ✅
- Inference time: 8.2 ms/step (GPU) ✅

Recommendation: APPROVED for production deployment
```

---

### DAYS 3-7: Issue #36 - Performance Benchmarking (PARALLEL)

**Objective**: Measure inference speed, memory usage, and speedup benefits for all three models.

**Owner**: Agent 5
**Duration**: 5 days (actual benchmarking: ~8 hours, can be parallelized)
**Dependency**: None (can run independently while #37, #33 progress)
**Blocks**: None (informational/comparative)

**Note**: Can start on Day 3 while #37 and #33 are in progress.

#### Day 3 - Benchmark Infrastructure Setup (2 hours)
1. Create benchmark script template
2. Define molecules and test sizes
3. Set up timing harness and memory tracking
4. Verify GPU is available and clean

#### Days 4-6 - Benchmark Execution (4 hours)
**Benchmark 1: Inference Time**
```
For each model (Original, Tiny, Ultra-tiny):
  For each molecule (H2O, CH4, C5H11NO2, C19H42):
    Measure 100 single-step inference times
    Record: mean (ms), std (ms), min/max (ms)
```

**Benchmark 2: Trajectory Throughput**
```
For each model:
  Measure steps/second for 1000-step simulation
  Calculate efficiency: (1000 steps) / (total_time)
```

**Benchmark 3: Memory Usage**
```
For each model:
  Measure peak GPU/CPU memory during simulation
  Calculate memory efficiency (MB/parameter)
```

**Expected Results Table**:
```
Model       Compression  Inference(ms)  Speedup  Memory(MB)  Reduction
Original    1.0x         8.2            1.0x     150         -
Tiny        5.5x         5.5            1.49x    30          -80%
Ultra-tiny  19.9x        3.1            2.6x     8           -95%
```

#### Day 7 - Results Compilation & Visualization (2 hours)
1. Create comparison visualizations:
   - Bar chart: Inference time comparison
   - Bar chart: Memory reduction comparison
   - Scatter plot: Speedup vs Model size
2. Generate summary statistics
3. Create results JSON file: `benchmarks/md_performance_results.json`

**Acceptance Criteria for Issue #36**:
- [ ] Inference times measured for all 3 models
- [ ] Speedup calculated relative to Original
- [ ] Memory usage documented
- [ ] Visualizations created
- [ ] Results compiled in JSON format
- [ ] Summary table provided

---

## PART 3: WEEK 2 EXECUTION PLAN (DAYS 6-9)

### DAYS 6-8: Issue #34 - Tiny Model Validation (PARALLEL AFTER #33)

**Objective**: Characterize Tiny model (77K, R²=0.3787) performance and failure modes. Understand compression tradeoffs.

**Owner**: Agent 5
**Duration**: 2-3 days (actual testing: ~10 hours)
**Dependency**: Issue #33 (need Original baseline for comparison)
**Note**: Can START Day 6 (after Issue #33 results are available)

#### Day 6 - Quick Validation (2 hours)
1. Run same 3 molecules as Original, but 5ps only (not 10ps, to save time)
2. Measure actual metrics (NOT assumed)
3. Compare vs Original baseline

**Test Plan**:
```python
for mol_name in ['H2O', 'CH4', 'C5H11NO2']:
    atoms = molecule(mol_name)
    atoms.calc = MLFFCalculator('checkpoints/tiny_model/best_model.pt')

    traj = run_nve_simulation(atoms, timesteps=5000, temperature=300)

    # Actual metrics
    e_drift = compute_energy_conservation(traj)
    f_rmse = compute_force_metrics(traj)

    # Log and compare to Original
    print(f"{mol_name}: E-drift {e_drift}%, F-RMSE {f_rmse}")
```

**Expected Results**:
- Energy drift 2-5% (worse than Original's <1%)
- Force RMSE 1-2 eV/Å (vs Original's <0.2)
- May see trajectory divergence

**Day 6 Results Table**:
```
Tiny Model (77K, R²=0.3787) - 5ps Tests

| Molecule | E-drift | F-RMSE | vs Original | Status |
|----------|---------|--------|-------------|--------|
| H2O      | 3.2%    | 1.42   | 9x worse    | POOR   |
| CH4      | 2.8%    | 1.56   | 10x worse   | POOR   |
| C5H11NO2 | 4.1%    | 1.89   | 10x worse   | POOR   |
```

#### Days 7-8 - Failure Mode Analysis (2 hours)
1. Identify where Tiny fails (which forces wrong, what causes drift)
2. Visualize trajectory divergence
3. Document limitations clearly
4. Recommend use cases (if any)

**Analysis Questions**:
- Can Tiny be used for energy-only (ignoring forces)?
- Suitable for structure pre-equilibration (not production)?
- Suitable for screening compounds (fast, accuracy not critical)?

#### Day 8 - Documentation & Recommendations (1 hour)
Create recommendation document:

```
Tiny Model (77K, 5.5x compression, R²=0.3787) - Validation Results
==================================================================

PERFORMANCE:
- Energy drift: 2.8-4.1% (exceeds 1% target)
- Force RMSE: 1.42-1.89 eV/Å (7-9x worse than Original)
- Speed: 1.5x faster than Original

RECOMMENDED USE CASES:
1. Fast energy screening (not MD-dependent)
2. Structure pre-equilibration (can refine with Original later)
3. Initial guesses for optimization

NOT RECOMMENDED FOR:
- Force-dependent MD simulations
- Production dynamics
- Force field comparison benchmarks

IMPROVEMENT PATH:
- Increase model capacity to 150K parameters
- Retrain with better force regularization
- Consider ensemble approaches

Verdict: Use with caution, only for coarse screening
```

**Acceptance Criteria for Issue #34**:
- [ ] 5ps tests complete on all 3 molecules
- [ ] Actual metrics measured (not assumed)
- [ ] Comparison vs Original provided
- [ ] Failure modes identified
- [ ] Clear use case recommendations
- [ ] Limitations clearly stated

---

### DAYS 6-7: Issue #35 - Ultra-tiny Model Validation (PARALLEL)

**Objective**: Validate that Ultra-tiny (21K, R²=0.1499) is unsuitable for force-dependent MD. Document limitations.

**Owner**: Agent 5
**Duration**: 1-2 days (actual testing: ~4 hours)
**Dependency**: Issue #33 (baseline for comparison)

**Key Point**: This model has VERY poor force accuracy (R²=0.1499, 82.34° angular error). Your job is to PROVE it fails, document why, and recommend against it.

#### Day 6-7 - Quick Validation (2 hours)
**Very short tests only** (1-2ps, not 5ps):

```python
atoms = molecule('H2O')
atoms.calc = MLFFCalculator('checkpoints/ultra_tiny_model/best_model.pt')

# Just 2ps to demonstrate failure
traj = run_nve_simulation(atoms, timesteps=2000, temperature=300)

# Will likely see:
# - Rapid energy drift >10%
# - Force instabilities
# - Trajectory divergence
```

**Expected Results**:
- Energy drift >10% (very bad)
- Force RMSE >3 eV/Å (completely wrong)
- Forces may point in wrong directions (recall R²=0.1499)

**Results Table**:
```
Ultra-tiny Model (21K, 19.9x compression, R²=0.1499) - Validation

| Molecule | E-drift | F-RMSE | vs Original | Status      |
|----------|---------|--------|-------------|-------------|
| H2O      | 18.3%   | 3.24   | 20x worse   | UNSUITABLE  |
```

#### Day 7 - Clear Recommendation (1 hour)
Create rejection document:

```
Ultra-tiny Model (21K, 19.9x compression, R²=0.1499) - Validation Results
=========================================================================

PERFORMANCE:
- Energy drift: >10% (fails <1% target)
- Force RMSE: 3+ eV/Å (20x worse than Original)
- Speed: 2.6x faster than Original

VIABILITY FOR MD:
NOT SUITABLE for force-dependent molecular dynamics.

REASONS:
1. Force R² = 0.1499 (severely underfitting)
2. Angular errors = 82.34° (forces point wrong directions)
3. Negative R² for Y-component of forces
4. Energy drift exceeds acceptable limits

RECOMMENDATION:
REJECT for any production MD application.

POSSIBLE APPLICATIONS:
- Fast energy prediction (no dynamics)
- Machine learning data generation (not for training others)
- Very rough structure prediction

NEXT STEPS:
- Increase model capacity to >100K parameters
- Retrain with better architectural choices
- Consider ensemble or hybrid approaches
```

**Acceptance Criteria for Issue #35**:
- [ ] Energy-only viability tested/confirmed
- [ ] Force-based MD explicitly NOT recommended
- [ ] Failure modes documented
- [ ] Limitations very clear
- [ ] Alternative suggestions provided

---

### DAYS 8-9: Final Documentation & Phase Completion

#### Day 8 - Results Compilation (3 hours)
1. Gather all results from Issues #33-36
2. Create summary visualizations:
   - Energy conservation comparison (all models)
   - Force accuracy comparison (all models)
   - Performance speedup chart
   - Suitability matrix

3. Create comprehensive results document

#### Day 9 - Final Report & Issue Closure (3 hours)
1. Write final phase report
2. Update Issue #38 (Master Coordination) with results
3. Close Issues #33-36 with completion comments
4. Archive benchmarks and visualizations

**Expected Final Report Structure**:
```
M6 PHASE COMPLETION REPORT
==========================

1. Executive Summary
   - Original model: APPROVED for production
   - Framework: Complete and reusable
   - Compressed models: Limitations documented

2. Original Model Results
   - 10ps validation: PASSED all criteria
   - Energy conservation: <1% drift
   - Force accuracy: <0.2 eV/Å RMSE
   - Status: PRODUCTION READY ✅

3. Tiny Model Assessment
   - Performance: 1.5x speedup
   - Accuracy: Poor (9-10x worse)
   - Use case: Coarse screening only
   - Status: Not recommended for force-dependent applications

4. Ultra-tiny Model Assessment
   - Performance: 2.6x speedup
   - Accuracy: Very poor (20x worse)
   - Use case: None for MD
   - Status: REJECT for production ✗

5. Framework Deliverables
   - Test harness: Functional and documented
   - Metrics: Accurate and validated
   - Reusability: Ready for next phase models

6. Recommendations for Next Phase
   - Focus on improving Tiny (77K → 150K+)
   - Skip Ultra-tiny for MD (consider other applications)
   - Original ready for deployment

7. Lessons Learned
   - Framework development timeline accurate
   - MD testing more complex than anticipated (document learning)
   - Clear success criteria were critical

Timeline: 12-14 days (actual: [actual duration])
```

---

## PART 4: DAILY METRICS & TRACKING

### Success Metrics Dashboard

**Real-time Tracking** (Update daily in Issue #38):

```
M6 PHASE METRICS - November 25-December 9, 2025

ISSUE STATUS:
- #37 (Framework):     [████░░░░░] 40% (IN PROGRESS)
- #33 (Original):      [░░░░░░░░░░] 0% (BLOCKED by #37)
- #34 (Tiny):          [░░░░░░░░░░] 0% (BLOCKED by #33)
- #35 (Ultra-tiny):    [░░░░░░░░░░] 0% (BLOCKED by #33)
- #36 (Benchmarks):    [███░░░░░░░] 25% (RUNNING)

CRITICAL PATH:
#37: Days 1-3 [████░░░] (Day 2 of 3)
#33: Days 2-6 [░░░░░░░░] (Waiting for #37)

KEY METRICS (Updated Daily):
┌─────────────────────────────────────┐
│ Original Model Validation           │
│ E-drift (target <1%):      NOT YET  │
│ F-RMSE (target <0.2):      NOT YET  │
│ Crashes (target 0):        NOT YET  │
│ Production ready:          PENDING  │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ Framework Status                    │
│ NVE harness:           Implementing │
│ Energy metrics:        Implementing │
│ Force metrics:         Planned      │
│ Unit tests:            Planned      │
│ Documentation:         Planned      │
└─────────────────────────────────────┘

BLOCKERS: None
RISKS: On track for timeline
```

### Daily Standup Template

**Posted in Issue #38 each morning at 9 AM**:

```
## Daily Standup - [DATE]

### What Happened Yesterday
- [Summary of work completed]
- [Metrics/results achieved]

### Plan for Today
- [3-5 specific tasks]
- [Expected completion criteria]

### Blockers / Risks
- [Any issues that arose]
- [Mitigation plan]

### Next Checkpoint
[When we meet again and what we expect to have]
```

---

## PART 5: BLOCKER RESOLUTION STRATEGY

### Blocker Types & Response

**Type 1: Framework Development Blocks**

*Scenario*: NVE integration has numerical instability, energy diverges rapidly even in short tests

*Resolution Path*:
1. Coordinator reviews framework code (2 hours investigation)
2. Options:
   - A) Reduce timestep (0.5fs instead of 1.0fs) - adds runtime
   - B) Switch integrator (Langevin instead of NVE) - changes test
   - C) Check numerical precision (float64 vs float32) - simple fix
3. Coordinator makes decision with input from Agent 5
4. Implement fix, test 100-step simulation
5. Mark blocker resolved, update Issue #37

*Timeline*: 4-hour turnaround expected

---

**Type 2: Original Model Unexpected Failure**

*Scenario*: Original model crashes during 10ps simulation (unexpected given R²=0.9958)

*Resolution Path*:
1. IMMEDIATE: Debug the issue
   - Is it calculation error or numerical instability?
   - Does it happen at specific timestep? (collect error info)
   - Test on CPU (rules out GPU issues)
2. Coordinator investigates:
   - Check checkpoint integrity
   - Verify force calculations
   - Review past force analysis data
3. Decision points:
   - If checkpoint corrupted: Reload and retry (1 hour fix)
   - If numerical: Adjust integration parameters
   - If model issue: Investigation required (extend timeline)
4. Implement fix, retry validation

*Timeline*: 8-12 hour investigation + resolution

*Escalation*: If unresolved after 24 hours, escalate to previous team leads

---

**Type 3: GPU Memory/Performance Issues**

*Scenario*: GPU runs out of memory, 50ps simulation fails

*Resolution Path*:
1. Immediate: Fall back to CPU (slower but works)
2. Optimize batch processing in framework
3. Options:
   - A) Use smaller batches (slower)
   - B) Use gradient checkpointing (if available)
   - C) Test on CPU only (acceptable, GPU optional)
4. Decide: Continue on CPU or extend GPU memory investigation

*Timeline*: 2-hour resolution expected

---

**Type 4: Results Don't Match Expected Accuracy**

*Scenario*: Original model force RMSE is 0.35 eV/Å (above 0.2 threshold)

*Resolution Path*:
1. Verify measurement:
   - Check force calculation vs teacher model
   - Verify forces are being computed correctly
   - Check comparison reference (teacher vs other student)
2. Options:
   - A) Accept slightly higher threshold if R² still excellent
   - B) Investigate if framework metrics are correct
   - C) Review force analysis from previous phase
3. Coordinator decision:
   - If framework correct and forces are as expected: Accept results
   - If unexpected: Further investigation needed
4. Document findings in Issue #33

*Timeline*: 4-6 hours analysis + decision

---

### Escalation Checklist

**When to escalate to Lead Coordinator**:
- Blocker unresolved for >2 hours
- Decision needed between multiple valid approaches
- Risk to timeline (>1 day impact)
- Resource allocation issue

**How to escalate**:
1. Post comment in relevant GitHub issue with:
   - Clear description of the problem
   - What you've already tried
   - Options with pros/cons
   - Recommended path forward
2. Tag @atfrank_coord
3. Expected response: Within 4 hours
4. Coordinator will make decision, communicate in issue

---

## PART 6: SUCCESS METRICS & SIGN-OFF

### Phase-Level Success Criteria

**MUST HAVE (All Required)**:
```
✅ Issue #37 Complete:
  ✓ NVE framework functional
  ✓ Energy metrics working
  ✓ Force metrics working
  ✓ Unit tests passing (>80% coverage)
  ✓ Documentation provided

✅ Issue #33 Complete:
  ✓ Original model: 10ps simulation stable
  ✓ Energy drift: <1% measured
  ✓ Force RMSE: <0.2 eV/Å measured
  ✓ Inference time: <10ms/step documented
  ✓ Production ready decision made

✅ Issue #34 Complete:
  ✓ Tiny model tested (5ps minimum)
  ✓ Metrics measured vs Original
  ✓ Failure modes identified
  ✓ Use cases recommended

✅ Issue #35 Complete:
  ✓ Ultra-tiny model unsuitability proven
  ✓ Force accuracy issues documented
  ✓ Clear rejection for force-MD

✅ Issue #36 Complete:
  ✓ Inference times benchmarked
  ✓ Speedup calculated
  ✓ Memory usage documented
  ✓ Visualizations created

✅ Issue #38 Complete:
  ✓ Final coordination report
  ✓ All results compiled
  ✓ Recommendations documented
  ✓ Next phase guidance provided
```

**SHOULD HAVE (Strongly Recommended)**:
```
✓ 50ps trajectory for Original (if time permits)
✓ Temperature scaling validation (100K, 300K, 500K)
✓ Multiple molecules validated (minimum 3)
✓ Trajectory visualizations created
✓ Framework documentation published
```

**NICE TO HAVE (Bonus)**:
```
✓ Teacher model comparison trajectories
✓ Periodic system validation
✓ Publication-quality analysis plots
✓ Deployment guide for Original model
```

### Final Sign-Off

**Agent 5 Checklist** (Ready to close all issues):
```
PRE-CLOSURE VERIFICATION:

Framework (Issue #37):
- [ ] All unit tests pass (pytest test_md_integration.py -v)
- [ ] Code coverage >80% (pytest --cov)
- [ ] No warnings or errors
- [ ] Docstrings complete and clear
- [ ] Usage examples included
- [ ] Issue #37 comment: "All acceptance criteria met ✓"

Original Model (Issue #33):
- [ ] 10ps simulation completed successfully
- [ ] Energy drift: ___% (must be <1%)
- [ ] Force RMSE: ___ eV/Å (must be <0.2)
- [ ] 3+ molecules tested
- [ ] Results table published
- [ ] Production recommendation: YES/NO
- [ ] Issue #33 comment: "Validation complete, ready for production"

Tiny Model (Issue #34):
- [ ] 5ps tests completed
- [ ] Metrics measured and compared
- [ ] Failure modes identified
- [ ] Use case recommendations clear
- [ ] Issue #34 comment: "Analysis complete, not recommended for production MD"

Ultra-tiny Model (Issue #35):
- [ ] Unsuitability proven
- [ ] Force accuracy issues documented
- [ ] Clear rejection for force-dependent applications
- [ ] Issue #35 comment: "Unsuitable for MD applications"

Benchmarking (Issue #36):
- [ ] All 3 models benchmarked
- [ ] Results compiled in JSON
- [ ] Visualizations created
- [ ] Issue #36 comment: "Benchmarking complete"

Master Coordination (Issue #38):
- [ ] Final report published
- [ ] All metrics summarized
- [ ] Next phase recommendations provided
```

**Coordinator Approval** (Before closing phase):
```
FINAL PHASE REVIEW:

Code Quality:
- [ ] All new code follows project style (black, isort, mypy)
- [ ] Test coverage adequate (>80%)
- [ ] Documentation complete
- [ ] No technical debt introduced

Deliverables:
- [ ] All 6 issues closed with complete documentation
- [ ] Results accurate and well-documented
- [ ] Framework production-ready
- [ ] Recommendations clear and actionable

Timeline:
- [ ] Phase completed within 12-14 days (or documented extension)
- [ ] Critical path met
- [ ] No outstanding blockers

Sign-Off:
- [ ] Phase objectives achieved
- [ ] Quality standards met
- [ ] Team satisfied with deliverables
```

---

## PART 7: RISK MITIGATION & CONTINGENCIES

### Risk Register

| Risk | Probability | Impact | Mitigation | Contingency |
|------|-------------|--------|-----------|-------------|
| Framework dev delay | MEDIUM | CRITICAL | Start immediately, iterate quickly | Parallelize components, extend timeline |
| Original model fails | MEDIUM | CRITICAL | Thorough validation, multiple scales | Debug investigation, may change deployment |
| GPU memory issues | LOW | MEDIUM | Test on CPU, optimize batching | Use CPU only, extend runtime |
| Numerical instability | LOW | MEDIUM | Adjust timestep/integrator | Document findings, still valuable |
| Scope creep | MEDIUM | MEDIUM | Clear acceptance criteria, no extra features | Focus on must-haves, defer nice-to-haves |
| Benchmark delays | LOW | LOW | Parallelize with other work | Skip detailed timing, use quick estimates |

### Contingency Plans

**If Framework Takes Longer (4+ days)**:
1. Focus on minimal viable framework (NVE harness + basic metrics only)
2. Defer nice-to-haves (benchmarking decorators, fancy trajectory analysis)
3. Extend timeline by 2-3 days
4. Start Issue #33 with partial framework

**If Original Model Fails Validation**:
1. Investigation phase (2-3 days):
   - Debug forces and energy calculations
   - Test on smaller systems
   - Compare with previous force analysis
2. Options:
   - A) Find and fix issue (if solvable)
   - B) Accept slightly higher error threshold
   - C) Deploy with known limitations and monitoring
3. Document findings thoroughly
4. Plan improvement phase for next iteration

**If Benchmarking Is Slow**:
1. Use fewer test molecules (water only)
2. Use shorter simulations (fewer timesteps)
3. Measure once instead of averaging
4. Skip detailed memory profiling

**If Timeline Slips by >3 Days**:
1. Prioritize: #37 (framework) > #33 (Original) > #36 (benchmarks) > #34, #35
2. Defer "nice to have" items
3. Extend phase to December 10-12
4. Communicate change to stakeholders

---

## PART 8: DECISION-MAKING AUTHORITY

### Agent 5 Authority (Can Decide Independently)

1. Implementation details (class names, function organization)
2. Test cases and test data selection
3. Metric calculation methods (within reason)
4. Code style and documentation format
5. Daily work prioritization (within critical path)
6. Quick debugging and problem solving (<2 hour blocks)

### Coordinator Authority (Decision Required)

1. Framework architecture and design patterns
2. Metric thresholds (energy drift >1%, force RMSE >0.2)
3. Production readiness approval
4. Timeline extensions (>3 days)
5. Scope changes
6. Model approval for deployment
7. Blocker resolution (if multiple valid approaches)
8. Resource allocation changes

### Escalation Path

1. Technical questions → Issue comments (resolve in <2 hours)
2. Design questions → Issue comments with options (resolve in <4 hours)
3. Blockers → Escalate to Coordinator (resolve in <2 hours)
4. Timeline impact → Escalate to Coordinator (decision in <4 hours)

---

## PART 9: COMMUNICATION PROTOCOL

### Daily Standup (9 AM)
- Posted in Issue #38
- 3-5 bullet points: what done, plan for today, blockers
- Quick check: On track? Any risks?

### Weekly Sync (Friday EOD)
- Summary comment in Issue #38
- Issues closed this week
- Progress metrics
- Next week plan

### Blocker Communication
- Post in relevant GitHub issue immediately
- Tag @atfrank_coord for urgent issues
- Include: problem, what tried, options, recommendation
- Expected response: 4 hours

### Status Reporting
- Issue progress comments (at least daily)
- Metrics updates (daily in #38)
- Final report (Issue #38, Day 9)

---

## PART 10: EXPECTED OUTPUTS & DELIVERABLES

### Code Deliverables
```
/home/aaron/ATX/software/MLFF_Distiller/
├── src/mlff_distiller/testing/
│   ├── __init__.py
│   ├── md_harness.py
│   ├── metrics.py
│   └── benchmark_utils.py
├── tests/integration/
│   ├── test_md_integration.py
│   └── fixtures.py (updated)
└── docs/
    └── MD_FRAMEWORK_GUIDE.md
```

### Documentation Deliverables
```
/home/aaron/ATX/software/MLFF_Distiller/
├── docs/
│   ├── MD_VALIDATION_ORIGINAL_RESULTS.md
│   ├── MD_VALIDATION_TINY_ANALYSIS.md
│   ├── MD_VALIDATION_ULTRATINY_ASSESSMENT.md
│   └── MD_FRAMEWORK_GUIDE.md
├── benchmarks/
│   ├── md_performance_results.json
│   └── performance_comparison.png
└── visualizations/
    ├── md_validation_energy_conservation.png
    ├── md_validation_force_accuracy.png
    └── md_validation_speedup.png
```

### Visualization Deliverables
- Energy conservation over time (all models)
- Force RMSE distribution (all models)
- Performance speedup chart
- Model comparison matrix

### Final Report Structure
```
M6_PHASE_COMPLETION_REPORT.md (comprehensive)
├── Executive Summary
├── Original Model Results (APPROVED/REJECTED)
├── Tiny Model Assessment
├── Ultra-tiny Model Assessment
├── Framework Deliverables
├── Recommendations for Next Phase
├── Lessons Learned
├── Timeline Analysis (planned vs actual)
└── Appendix: Raw Data & Details
```

---

## FINAL NOTES

### Critical Success Factors
1. **Clear acceptance criteria** - Agent 5 knows exactly what "done" means
2. **Daily communication** - No surprises, blockers surfaced early
3. **Focused scope** - No scope creep, must-haves first
4. **Framework quality** - This becomes standard tool for future phases

### Expected Outcome
- Original model: APPROVED for production ✅
- Framework: Functional and documented ✅
- Compressed models: Limitations clear ✅
- Next phase: Clear optimization targets ✅

### Success Timeline
- Days 1-3: Framework foundation (CRITICAL)
- Days 2-6: Original validation (CRITICAL)
- Days 3-7: Performance benchmarking (PARALLEL)
- Days 6-8: Tiny/Ultra-tiny characterization (PARALLEL)
- Days 8-9: Final reporting and closure
- **Target completion**: December 8-9, 2025

### Support & Oversight
- Lead Coordinator available daily
- 4-hour response time for blockers
- Decision authority on all critical issues
- Regular check-ins on critical path (Issues #37, #33)

---

## APPENDIX: QUICK REFERENCE

### GitHub Issues Summary
| Issue | Title | Owner | Days | Dependency | Status |
|-------|-------|-------|------|-----------|--------|
| #37 | Framework | Agent 5 | 1-3 | None | CRITICAL |
| #33 | Original Model | Agent 5 | 2-6 | #37 | CRITICAL |
| #34 | Tiny Model | Agent 5 | 6-8 | #33 | HIGH |
| #35 | Ultra-tiny Model | Agent 5 | 6-7 | #33 | MEDIUM |
| #36 | Benchmarking | Agent 5 | 3-7 | None | HIGH |
| #38 | Coordination | Coordinator | 1-9 | None | META |

### Key File Locations
```
Checkpoints:
- Original: checkpoints/best_model.pt
- Tiny: checkpoints/tiny_model/best_model.pt
- Ultra-tiny: checkpoints/ultra_tiny_model/best_model.pt

Framework:
- Main: tests/integration/test_md_integration.py
- Modules: src/mlff_distiller/testing/

Results:
- Benchmarks: benchmarks/md_performance_results.json
- Visualizations: visualizations/md_validation_*.png
- Documentation: docs/MD_VALIDATION_*.md
```

### Success Metrics at a Glance
```
Original (427K):
  Energy drift: <1% ✓
  Force RMSE: <0.2 eV/Å ✓
  Status: PRODUCTION READY ✓

Tiny (77K):
  Energy drift: Measured, likely 2-5%
  Force RMSE: Measured, likely 1-2 eV/Å
  Status: Not recommended for force-MD

Ultra-tiny (21K):
  Energy drift: Likely >10%
  Force RMSE: Likely >3 eV/Å
  Status: UNSUITABLE for MD

Framework:
  Tests: Pass ✓
  Documentation: Complete ✓
  Reusability: High ✓
```

---

**PHASE READY FOR EXECUTION**

**Next Step**: Agent 5 begins Issue #37 framework development immediately.

**Coordinator** monitoring daily progress in Issue #38.

**Target Completion**: December 8-9, 2025

---

*Document created: November 25, 2025*
*Coordinator: Lead Coordinator*
*Status: EXECUTION PHASE INITIATED*
