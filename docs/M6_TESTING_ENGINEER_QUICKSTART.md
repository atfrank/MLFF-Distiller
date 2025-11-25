# M6 Testing Engineer Quick Start Guide
## MD Integration Testing & Validation Phase

**Target Audience**: Agent 5 (Testing & Benchmarking Engineer)
**Date**: November 25, 2025
**Phase**: M6 - MD Integration Testing & Validation
**Critical**: Read before starting work

---

## YOUR MISSION

You are responsible for:
1. Building the MD simulation test framework (Issue #37)
2. Validating the Original model in MD (Issue #33) - CRITICAL PRIORITY
3. Characterizing Tiny/Ultra-tiny models (Issues #34, #35)
4. Performance benchmarking across variants (Issue #36)

**Timeline**: ~12-14 days
**Priority Order**: #37 → #33 → (#34, #35, #36 in parallel)

---

## WHAT'S READY FOR YOU

### Existing Infrastructure ✅
```
/home/aaron/ATX/software/MLFF_Distiller/
├── src/mlff_distiller/inference/ase_calculator.py (production ready)
├── tests/integration/test_ase_calculator.py (387 lines, functional)
├── tests/integration/
│   ├── test_ase_integration_demo.py
│   ├── test_ase_interface_compliance.py
│   ├── test_drop_in_replacement.py
│   ├── test_student_calculator_integration.py
│   ├── test_teacher_wrappers_md.py
│   └── test_validation_integration.py
└── checkpoints/
    ├── best_model.pt (Original 427K)
    ├── tiny_model/best_model.pt (Tiny 77K)
    └── ultra_tiny_model/best_model.pt (Ultra-tiny 21K)
```

### Model Status ✅
- **Original (427K)**: R² = 0.9958, PRODUCTION READY, no blockers
- **Tiny (77K)**: R² = 0.3787, reference implementation
- **Ultra-tiny (21K)**: R² = 0.1499, limited use only

### ASE Calculator Features ✅
- Full ASE Calculator compliance
- Batch inference support
- GPU/CPU automatic fallback
- Memory-efficient tensor management
- Comprehensive error handling
- Performance tracking

---

## ISSUE #37: TEST FRAMEWORK (DAYS 1-3)

### What to Build
Create a comprehensive MD simulation test framework in `tests/integration/test_md_integration.py`:

#### 1. NVE Ensemble Harness
```python
class MDSimulationHarness:
    """Runs NVE ensemble simulations with metrics tracking."""

    def run_nve_simulation(atoms, calculator, timestep=1.0, steps=100,
                          temperature=300):
        """Run NVE dynamics simulation."""
        # Initialize velocities (Maxwell-Boltzmann)
        # Run VelocityVerlet integration
        # Track all frames
        # Return trajectory data
```

#### 2. Energy Conservation Metrics
```python
def compute_energy_conservation(trajectory):
    """
    Returns:
    - Total energy array [E_0, E_1, ..., E_N]
    - Drift percentage (E_final - E_0) / E_0 * 100
    - Kinetic energy array
    - Potential energy array
    - Energy smoothness metrics
    """
```

#### 3. Force Accuracy Metrics
```python
def compute_force_metrics(forces_predicted, forces_reference):
    """
    Returns:
    - RMSE (eV/Å)
    - MAE (eV/Å)
    - Component-wise R²
    - Angular errors
    - Per-atom statistics
    """
```

#### 4. Trajectory Analysis Tools
```python
def compare_trajectories(traj1, traj2, metric='rmsd'):
    """Compare two trajectories."""

def analyze_trajectory_stability(trajectory):
    """Detect instabilities, divergences, anomalies."""
```

#### 5. Benchmarking Utilities
```python
@benchmark_decorator
def inference_timing(atoms, calculator, n_runs=10):
    """Measure inference time (ms per step)."""
```

### Acceptance Criteria
- [ ] Supports 10+ ps simulations without memory overflow
- [ ] Energy conservation accurate to machine precision
- [ ] Force metrics match expected ranges
- [ ] Easy to add new molecules
- [ ] Unit tests for each metric function
- [ ] Documentation with examples

### Files to Create/Modify
- `tests/integration/test_md_integration.py` (NEW, ~500 lines)
- `src/mlff_distiller/testing/` (NEW directory)
  - `md_harness.py` (NVE simulator)
  - `metrics.py` (Energy, force, trajectory metrics)
  - `benchmark_utils.py` (Timing decorators)
- `tests/integration/fixtures.py` (TEST MOLECULES)

### Key Decisions
1. **Integrator**: Use ASE's VelocityVerlet (already works with calculator)
2. **Timestep**: 1.0 fs (standard), 0.5 fs for safety
3. **Temperature**: Use Maxwell-Boltzmann at 300K (standard)
4. **Trajectory format**: Keep in-memory as ASE objects (simple) or HDF5 (scalable)?
   - **Recommendation**: In-memory for <50ps, HDF5 for longer

### Expected Output
```
NVE Ensemble Simulation Results:
├── Total Energy: [E_0, E_1, ..., E_N]
├── Energy Drift: 0.42%
├── Kinetic Energy: [KE_0, ..., KE_N]
├── Potential Energy: [PE_0, ..., PE_N]
├── Force RMSE per frame: [RMSE_0, ..., RMSE_N]
├── Trajectory RMSD evolution
└── Inference Time: 8.3 ms/step
```

### Success Definition
- Framework passes all unit tests
- Can run 100-step (100 fs) simulation without errors
- Energy drift < 0.1% over short test (numerical precision check)
- Timing measurements have < 5% variance

---

## ISSUE #33: ORIGINAL MODEL MD TESTING (DAYS 2-6)

### What to Validate
Production readiness of Original Student Model (427K, R²=0.9958):

### Test Plan

#### Phase 1: Basic Validation (Days 2-3)
```python
# Test molecules
test_molecules = [
    ('H2O', 'water', 3),
    ('CH4', 'methane', 5),
]

for name, label, natoms in test_molecules:
    # 5 ps simulation
    atoms = molecule(name)
    atoms.calc = StudentForceFieldCalculator('checkpoints/best_model.pt')

    trajectory = run_nve_simulation(
        atoms,
        timesteps=5000,  # 5000 steps × 1fs = 5ps
        timestep=1.0,
        temperature=300
    )

    # Check metrics
    energy_drift = check_energy_conservation(trajectory)
    force_rmse = compute_force_metrics(trajectory)

    assert energy_drift < 1.0, f"Drift {energy_drift}% exceeds threshold"
    assert force_rmse < 0.2, f"RMSE {force_rmse} exceeds threshold"
```

#### Phase 2: Extended Validation (Days 4-5)
```python
# Longer trajectory
trajectory_long = run_nve_simulation(
    atoms,
    timesteps=50000,  # 50ps
    timestep=1.0,
    temperature=300
)

# More rigorous checks
assert total_energy_drift < 1.0
assert kinetic_energy_stable
assert force_predictions_consistent
assert no_crashes_or_warnings
```

#### Phase 3: Temperature Scaling (Day 6)
```python
# Test at different temperatures
for temp in [100, 300, 500]:
    trajectory = run_nve_simulation(atoms, temperature=temp)
    assert trajectory_stable(trajectory)
```

### Acceptance Criteria (All Must Pass)
- [ ] 10ps simulation completes without crashes
- [ ] Total energy drift < 1% over trajectory
- [ ] Kinetic energy shows no sudden spikes
- [ ] Force RMSE during MD < 0.2 eV/Å
- [ ] Per-frame inference time < 10ms (GPU)
- [ ] 3 test molecules validated
- [ ] All metrics stable and documented
- [ ] Production-ready recommendation confirmed

### Expected Results (From Force Analysis)
- R² = 0.9958 in offline → expect similar in MD
- RMSE = 0.1606 eV/Å in offline → expect <0.2 eV/Å in MD
- Angular error = 9.61° → should not cause instability

### Test Output Template
```
ORIGINAL MODEL (427K) - MD VALIDATION RESULTS
==============================================

Test Molecule: Water (H2O)
Duration: 10 ps
Temperature: 300 K
Timestep: 1.0 fs
Steps: 10000

RESULTS:
--------
Total Energy:
  Initial: -47.23 eV
  Final: -47.24 eV
  Drift: 0.02% ✅

Energy Components:
  Kinetic: stable [3.5 to 3.8 eV] ✅
  Potential: stable [-50.8 to -50.7 eV] ✅

Force Metrics:
  RMSE: 0.158 eV/Å ✅
  MAE: 0.103 eV/Å ✅
  Inference Time: 8.2 ms/step ✅

Status: ✅ PRODUCTION READY
```

### Create/Modify Files
- `tests/integration/test_md_integration.py` (add Original tests)
- `visualizations/md_validation_Original.png` (energy plots)
- `docs/MD_VALIDATION_ORIGINAL_RESULTS.md` (results summary)

---

## ISSUE #34: TINY MODEL VALIDATION (DAYS 6-8)

### What to Test
Tiny Student Model (77K, 5.5x compression, R²=0.3787):

### Scope
- Shorter trajectories (5ps max) - expect issues
- Compare metrics vs Original baseline
- Document limitations clearly
- Identify failure modes

### Test Plan
```python
# Same 3 test molecules as Original
# BUT shorter trajectories (5ps vs 10ps)

for mol_name in ['H2O', 'CH4', 'C5H11NO2']:
    atoms = molecule(mol_name)
    atoms.calc = StudentForceFieldCalculator('checkpoints/tiny_model/best_model.pt')

    traj = run_nve_simulation(atoms, timesteps=5000, temperature=300)

    # Document what happens
    # DON'T EXPECT PERFECT RESULTS
    energy_drift = check_energy_conservation(traj)
    force_rmse = compute_force_metrics(traj)

    # Log actual results for analysis
    print(f"Energy drift: {energy_drift}%")
    print(f"Force RMSE: {force_rmse} eV/Å")
    # (Will likely be worse than Original)
```

### Expected Results
- Energy drift likely 2-5% (worse than Original's <1%)
- Force RMSE likely 1-2 eV/Å (vs Original's <0.2)
- May see trajectory divergence
- Inference speed 1.5-3x faster

### Acceptance Criteria
- [ ] 5ps simulations complete
- [ ] Actual metrics documented (not assumed)
- [ ] Comparison vs Original provided
- [ ] Failure modes identified
- [ ] Use case recommendations
- [ ] Limitations clearly stated

### Deliverable
Analysis document showing:
- Per-molecule results comparison table
- Why it's worse (R² = 0.3787 → forces not learned)
- Suitable applications (e.g., pre-equilibration only?)
- Architecture improvement recommendations

---

## ISSUE #35: ULTRA-TINY MODEL VALIDATION (DAYS 6-7)

### What to Test
Ultra-tiny Model (21K, 19.9x compression, R²=0.1499):

### Scope
- Very short trajectories (1-2ps)
- Expect significant failures
- Characterize why it fails
- Recommend against force-dependent use

### Key Point
**This model is likely unsuitable for MD due to poor force accuracy.**
- R² = 0.1499 (severely underfitting)
- Angular errors = 82.34° (forces point wrong ways!)
- Negative component R² for Y direction
- Your job: PROVE it doesn't work for MD, document why

### Test Plan (Quick)
```python
# Very short test
atoms = molecule('H2O')
atoms.calc = StudentForceFieldCalculator('checkpoints/ultra_tiny_model/best_model.pt')

# Just 2ps to show the problem
traj = run_nve_simulation(atoms, timesteps=2000, temperature=300)

# Will likely see:
# - Rapid energy drift >5%
# - Force instabilities
# - Trajectory divergence
# → EXPECTED AND DOCUMENTED
```

### Acceptance Criteria
- [ ] Energy-only usage verified (if applicable)
- [ ] Force-based MD explicitly NOT recommended
- [ ] Failure modes documented
- [ ] Limitations very clear
- [ ] Alternative approaches suggested (hybrid, etc.)

### Expected Conclusion
"Ultra-tiny model is not suitable for force-dependent MD simulations. Recommended for energy screening only or hybrid approaches."

---

## ISSUE #36: PERFORMANCE BENCHMARKING (DAYS 3-7, PARALLEL)

### What to Measure

#### 1. Inference Time
```python
# Single forward pass timing
for model_path, label in [
    ('checkpoints/best_model.pt', 'Original'),
    ('checkpoints/tiny_model/best_model.pt', 'Tiny'),
    ('checkpoints/ultra_tiny_model/best_model.pt', 'Ultra-tiny'),
]:
    calc = StudentForceFieldCalculator(model_path)

    for mol_name in ['H2O', 'CH4', 'C5H11NO2']:
        atoms = molecule(mol_name)
        atoms.calc = calc

        # Time a single energy/force calculation
        time_ms = benchmark_inference(atoms, n_runs=100)
        print(f"{label:12} {mol_name:5} {time_ms:.3f} ms")
```

#### 2. Trajectory Throughput
```python
# Simulations per second
for model in [original, tiny, ultra_tiny]:
    time_per_step = measure_throughput(model, mol, n_steps=1000)
    steps_per_second = 1000 / time_per_step
    print(f"Steps/sec: {steps_per_second:.1f}")
```

#### 3. Memory Usage
```python
# Peak memory during simulation
for model in [original, tiny, ultra_tiny]:
    peak_memory = measure_memory_usage(model, mol, n_steps=1000)
    print(f"Peak memory: {peak_memory:.1f} MB")
```

### Benchmark Results Table
```
Model       Compression  Inference(ms)  Speedup  Memory(MB)  Memory Reduction
--------    -----------  -----------    -------  ---------   ---------------
Original    1.0x         8.2            1.0x     150         Baseline
Tiny        5.5x         5.5            1.5x     30          -80%
Ultra-tiny  19.9x        3.1            2.6x     8           -95%
```

### Output Format
- Create `benchmarks/md_performance_results.json`
- Create visualization: `benchmarks/performance_comparison.png`
- Summary table in documentation

---

## KEY SUCCESS METRICS

### Original Model (Must Pass All)
| Metric | Target | Test |
|--------|--------|------|
| MD Stability | 10+ ps, no crash | Issue #33 |
| Energy Drift | <1% | Issue #33 |
| Force RMSE | <0.2 eV/Å | Issue #33 |
| Inference Time | <10 ms/step | Issue #36 |
| Production Ready | YES | All tests pass |

### Tiny Model
| Metric | Document | Test |
|--------|----------|------|
| MD Stability | Actual results | Issue #34 |
| Energy Drift | Measure | Issue #34 |
| Force RMSE | Measure | Issue #34 |
| Speedup | 1.5-3x | Issue #36 |
| Recommendation | Clear | Issue #34 |

### Ultra-tiny Model
| Metric | Document | Test |
|--------|----------|------|
| Unsuitability | Very clear | Issue #35 |
| Why failure | Force analysis | Issue #35 |
| Speedup | 3-5x | Issue #36 |
| Best use | Energy only? | Issue #35 |

---

## CODE STRUCTURE RECOMMENDATION

```
tests/integration/
├── test_md_integration.py (NEW, ~500 lines)
│   ├── TestFramework
│   │   ├── test_nve_harness
│   │   ├── test_energy_metrics
│   │   └── test_force_metrics
│   ├── TestOriginalModel
│   │   ├── test_water_5ps
│   │   ├── test_methane_5ps
│   │   ├── test_alanine_5ps
│   │   └── test_50ps_long
│   ├── TestTinyModel
│   │   ├── test_water_5ps
│   │   ├── test_vs_original
│   │   └── test_limitations
│   └── TestUltraTinyModel
│       ├── test_unsuitability
│       └── test_energy_only

src/mlff_distiller/testing/ (NEW)
├── md_harness.py (500 lines)
│   ├── MDSimulationHarness
│   └── VelocityVerletRunner
├── metrics.py (400 lines)
│   ├── compute_energy_conservation()
│   ├── compute_force_metrics()
│   └── analyze_trajectory_stability()
└── benchmark_utils.py (200 lines)
    ├── @benchmark_decorator
    └── measure_throughput()
```

---

## EXECUTION CHECKLIST

### Phase 1: Setup (Day 1)
- [ ] Clone repository and set up environment
- [ ] Verify checkpoints exist and load correctly
- [ ] Verify ASE calculator works with water molecule
- [ ] Understand existing test structure

### Phase 2: Framework (Days 1-3) - Issue #37
- [ ] Implement NVE simulation harness
- [ ] Implement energy conservation metrics
- [ ] Implement force metrics
- [ ] Unit test each component
- [ ] Test on 5ps simulation
- [ ] All tests pass

### Phase 3: Original Model Testing (Days 2-6) - Issue #33
- [ ] Water molecule 5ps test
- [ ] Methane molecule 5ps test
- [ ] Alanine molecule 5ps test
- [ ] 10ps extended test (at least one molecule)
- [ ] All metrics below thresholds
- [ ] Document results
- [ ] Production-ready conclusion

### Phase 4: Tiny Model Testing (Days 6-8) - Issue #34
- [ ] Basic 5ps tests
- [ ] Compare vs Original
- [ ] Document limitations
- [ ] Identify failure modes
- [ ] Recommendations

### Phase 5: Ultra-tiny Model Testing (Days 6-7) - Issue #35
- [ ] Short 1-2ps tests
- [ ] Characterize failures
- [ ] Document unsuitability
- [ ] Alternative recommendations

### Phase 6: Performance Benchmarking (Days 3-7) - Issue #36
- [ ] Inference time measurements
- [ ] Memory measurements
- [ ] Speedup calculations
- [ ] Results visualization
- [ ] Documentation

### Phase 7: Final Documentation (Days 8-9)
- [ ] All results compiled
- [ ] Final report written
- [ ] Issues #33-36 closed
- [ ] Master coordination issue #38 updated

---

## CRITICAL REMINDERS

### ✅ DO:
- Run tests on GPU (CUDA) for realistic timing
- Document actual results (don't assume)
- Compare Tiny/Ultra-tiny vs Original baseline
- Test multiple molecules for robustness
- Track energy conservation carefully
- Measure inference time accurately
- Be clear about what fails and why

### ❌ DON'T:
- Expect Tiny/Ultra-tiny to match Original
- Skip testing just because you predict failure
- Modify models (they are frozen for this phase)
- Ignore blockers - escalate to Lead Coordinator
- Assume results without measurement
- Use unfair comparison metrics

---

## WHERE TO GET HELP

### For Technical Questions
- Look at existing tests in `tests/integration/`
- Review ASE documentation: https://wiki.fysik.dtu.dk/ase/
- Check mdanalysis for trajectory tools: https://www.mdanalysis.org/

### For Blockers
- Tag @atfrank_coord in issue comments
- Daily standup updates
- Escalate early, not at deadline

### For Design Questions
- Discuss framework architecture in Issue #37
- Get feedback before full implementation
- Iterate quickly on prototypes

---

## TIMELINE SUMMARY

```
Days 1-3: Issue #37 (Framework)     - CRITICAL PATH
Days 2-6: Issue #33 (Original)      - CRITICAL PATH (blocked by #37 initially)
Days 6-8: Issue #34 (Tiny)          - Secondary (parallel after #33 starts)
Days 6-7: Issue #35 (Ultra-tiny)    - Secondary (parallel)
Days 3-7: Issue #36 (Benchmarking)  - Parallel track
Days 8-9: Final docs & reporting    - Wrap up

Total: 12-14 calendar days
```

---

## SUCCESS DEFINITION FOR YOU

### Minimal Success
- Framework works for basic tests
- Original model passes 10ps without crashes
- Energy drift < 5%
- Clear recommendations for each model

### Target Success
- Framework fully functional and tested
- Original model excellent results (<1% drift, <0.2 RMSE)
- Tiny/Ultra-tiny limitations clearly documented
- Performance benchmarks show speedup path
- All issues closed with thorough documentation

### Exceptional Success
- All of above PLUS:
- 50ps+ trajectories validated
- Multiple molecule types tested
- Temperature scaling validated
- Publication-quality visualizations
- Framework documented for future engineers

---

**Good luck! This is critical work. The Original model's production readiness depends on your validation.**

*For questions or blockers: Create issue comment or tag @atfrank_coord*
