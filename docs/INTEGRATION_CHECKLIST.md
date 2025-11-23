# Integration Validation Checklist
## ML Force Field Distillation Project

**Purpose**: Ensure Week 2 components integrate correctly with Week 1 foundation and with each other.

**When to Use**:
- Mid-Week checkpoint (Day 4)
- End of Week validation (Day 7)
- Before marking M1 complete

---

## Pre-Integration Checks

### Environment Health
- [ ] Repository at `/home/aaron/ATX/software/MLFF_Distiller`
- [ ] On latest commit (Week 1: 4ff20d9)
- [ ] Working directory clean (no uncommitted changes during check)
- [ ] Python 3.13+ available
- [ ] PyTorch with CUDA available
- [ ] All Week 1 dependencies installed

**Validation Command**:
```bash
cd /home/aaron/ATX/software/MLFF_Distiller
git status  # Should be clean
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## Week 1 Foundation Validation

### Test Suite Health
- [ ] Run full test suite: `pytest tests/ -v`
- [ ] All 181+ tests passing
- [ ] No collection errors
- [ ] No import errors
- [ ] Skipped tests are expected (e.g., orb-models not installed)

**Validation Command**:
```bash
pytest tests/ -v --tb=short 2>&1 | tee test_results.log
grep -E "passed|failed|error" test_results.log
```

**Expected**: `181+ passed, ~11 skipped, 0 failed, 0 errors`

### Module Imports
- [ ] `from mlff_distiller.data import MolecularDataset` works
- [ ] `from mlff_distiller.models.teacher_wrappers import OrbCalculator` works
- [ ] `from mlff_distiller.training import Trainer, TrainingConfig` works
- [ ] `from mlff_distiller.cuda.device_utils import get_device` works

**Validation Command**:
```bash
python -c "
from mlff_distiller.data import MolecularDataset, DataLoader
from mlff_distiller.models.teacher_wrappers import OrbCalculator, FeNNolCalculator
from mlff_distiller.training import Trainer, TrainingConfig, ForceFieldLoss
from mlff_distiller.cuda.device_utils import get_device, GPUInfo
print('All Week 1 imports successful!')
"
```

### Fixtures Available
- [ ] `tests/conftest.py` loads without error
- [ ] Sample atoms fixtures available
- [ ] Mock model fixtures available
- [ ] GPU fixtures available

**Validation Command**:
```bash
pytest tests/unit/test_fixtures_demo.py -v
```

---

## Week 2 Component Validation

### Issue #6: Student Calculator (Agent 2)

#### File Existence
- [ ] `src/mlff_distiller/models/student_calculator.py` exists
- [ ] File is not empty (>100 lines expected)
- [ ] Imports without error

**Validation Command**:
```bash
ls -lh src/mlff_distiller/models/student_calculator.py
python -c "from mlff_distiller.models.student_calculator import StudentCalculator; print('StudentCalculator import OK')"
```

#### Interface Compliance
- [ ] StudentCalculator inherits from ase.calculators.calculator.Calculator
- [ ] Has `implemented_properties` attribute
- [ ] Has `calculate()` method
- [ ] Has `get_potential_energy()` method
- [ ] Has `get_forces()` method
- [ ] Has `get_stress()` method (if applicable)

**Validation Command**:
```bash
python -c "
from mlff_distiller.models.student_calculator import StudentCalculator
from ase.calculators.calculator import Calculator
import inspect

# Check inheritance
assert issubclass(StudentCalculator, Calculator), 'Not a Calculator subclass'

# Check methods exist
assert hasattr(StudentCalculator, 'calculate'), 'Missing calculate method'
assert hasattr(StudentCalculator, 'get_potential_energy'), 'Missing get_potential_energy'
assert hasattr(StudentCalculator, 'get_forces'), 'Missing get_forces'

print('StudentCalculator interface compliance: PASS')
"
```

#### Basic Functionality
- [ ] Can instantiate StudentCalculator
- [ ] Can attach to ASE Atoms object
- [ ] Can call get_potential_energy() (even with placeholder model)
- [ ] Can call get_forces() (even with placeholder model)

**Validation Command**:
```bash
python -c "
from mlff_distiller.models.student_calculator import StudentCalculator
from ase import Atoms

# Create calculator
calc = StudentCalculator(device='cpu')

# Create simple atoms
atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
atoms.calc = calc

# Try to get properties (may return placeholders, that's OK)
try:
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    print('StudentCalculator basic functionality: PASS')
except NotImplementedError:
    print('StudentCalculator basic functionality: PLACEHOLDER (OK for now)')
"
```

#### Tests
- [ ] Unit tests exist for StudentCalculator
- [ ] Tests run without error
- [ ] Tests pass

**Validation Command**:
```bash
pytest tests/unit/test_student_calculator.py -v
# OR
pytest tests/ -k "student_calculator" -v
```

---

### Issue #5: MD Benchmark Framework (Agent 5)

#### File Existence
- [ ] `benchmarks/md_benchmark.py` exists
- [ ] `src/mlff_distiller/benchmarks/__init__.py` exists (if module created)
- [ ] Benchmark utilities exist

**Validation Command**:
```bash
ls -lh benchmarks/md_benchmark.py
find src/mlff_distiller/benchmarks -name "*.py" 2>/dev/null || echo "No benchmarks module (OK if in benchmarks/ dir only)"
```

#### Imports
- [ ] `benchmarks/md_benchmark.py` imports without error
- [ ] Required dependencies available (ASE, numpy, torch)

**Validation Command**:
```bash
python -c "import sys; sys.path.insert(0, 'benchmarks'); import md_benchmark; print('Benchmark imports OK')"
```

#### Basic Functionality
- [ ] Can run a short benchmark (10 steps)
- [ ] Produces timing output
- [ ] Handles ASE calculators
- [ ] Doesn't crash on small system

**Validation Command**:
```bash
python -c "
# Test benchmark framework with mock calculator
from ase import Atoms
from ase.calculators.emt import EMT
import sys
sys.path.insert(0, 'benchmarks')

# If md_benchmark has a simple API:
# from md_benchmark import MDBenchmark
# benchmark = MDBenchmark(calculator=EMT(), n_steps=10)
# results = benchmark.run()
# print(f'Benchmark test: {results}')

print('Manual benchmark test - check script can run')
"
```

#### Integration with Teacher Models
- [ ] Can benchmark OrbCalculator (if orb-models installed)
- [ ] Can benchmark FeNNolCalculator (if fennol installed)
- [ ] Can benchmark with mock calculator

**Validation Command**:
```bash
# Run benchmark with EMT (always available)
cd benchmarks
python md_benchmark.py --calculator emt --n-steps 10 --system-size 32
# OR whatever CLI the benchmark provides
```

#### Tests
- [ ] Benchmark utility tests exist
- [ ] Tests pass

**Validation Command**:
```bash
pytest tests/ -k "benchmark" -v
```

---

### Issue #7: ASE Interface Tests (Agent 5)

#### File Existence
- [ ] `tests/integration/test_ase_interface.py` exists
- [ ] File is not empty (>100 lines expected)

**Validation Command**:
```bash
ls -lh tests/integration/test_ase_interface.py
wc -l tests/integration/test_ase_interface.py
```

#### Tests Run
- [ ] Tests collect without error
- [ ] Tests for teacher calculators pass
- [ ] Tests for student calculator exist (may skip if not ready)

**Validation Command**:
```bash
pytest tests/integration/test_ase_interface.py -v
```

#### Coverage
- [ ] Tests verify get_potential_energy() compatibility
- [ ] Tests verify get_forces() compatibility
- [ ] Tests verify get_stress() compatibility (if applicable)
- [ ] Tests compare teacher vs student (if student ready)
- [ ] Tests run MD trajectory with both calculators

**Validation Command**:
```bash
pytest tests/integration/test_ase_interface.py -v --tb=short
# Check test names include coverage areas
```

---

### Issue #9: MD Profiling Framework (Agent 4)

#### File Existence
- [ ] `src/mlff_distiller/cuda/profiler.py` exists
- [ ] `benchmarks/profile_md.py` exists (or similar)

**Validation Command**:
```bash
ls -lh src/mlff_distiller/cuda/profiler.py
ls -lh benchmarks/profile_md.py 2>/dev/null || echo "Profiler may be in different location"
```

#### Imports
- [ ] Profiler imports without error
- [ ] Integration with existing CUDA utilities

**Validation Command**:
```bash
python -c "from mlff_distiller.cuda.profiler import MDProfiler; print('Profiler import OK')"
# OR whatever the class is named
```

#### Basic Functionality
- [ ] Can profile a simple MD run
- [ ] Produces latency statistics
- [ ] Tracks memory usage
- [ ] Generates report

**Validation Command**:
```bash
python -c "
from mlff_distiller.cuda.profiler import MDProfiler
from ase.calculators.emt import EMT

# Test profiler with EMT
# profiler = MDProfiler()
# results = profiler.profile(calculator=EMT(), n_steps=10)
# print(f'Profiler test: {results}')

print('Manual profiler test - check if API works')
"
```

#### Tests
- [ ] Profiler tests exist
- [ ] Tests pass

**Validation Command**:
```bash
pytest tests/ -k "profiler" -v
```

---

## Cross-Component Integration

### Teacher + Benchmark
- [ ] Benchmark framework can run with OrbCalculator
- [ ] Benchmark framework can run with FeNNolCalculator
- [ ] Produces valid performance metrics

**Validation Command**:
```bash
# With mock or EMT calculator
python benchmarks/md_benchmark.py --calculator mock --n-steps 100
```

### Teacher + Profiler
- [ ] Profiler can analyze teacher calculator
- [ ] Produces latency distribution
- [ ] Identifies hotspots

**Validation Command**:
```bash
python benchmarks/profile_md.py --calculator mock --n-steps 100
```

### Student + Interface Tests
- [ ] StudentCalculator works in interface tests
- [ ] Interface tests pass for student calculator
- [ ] Comparison tests work (if implemented)

**Validation Command**:
```bash
pytest tests/integration/test_ase_interface.py -k "student" -v
```

### Student + Benchmark
- [ ] Benchmark framework can run with StudentCalculator
- [ ] Produces valid performance metrics
- [ ] Can compare to teacher benchmarks

**Validation Command**:
```bash
python benchmarks/md_benchmark.py --calculator student --n-steps 100
```

---

## Full Integration Test

### All Components Together
- [ ] Run full test suite: `pytest tests/`
- [ ] Expected: 200+ tests passing
- [ ] No failures
- [ ] Only expected skips (missing optional dependencies)

**Validation Command**:
```bash
pytest tests/ -v --tb=short --maxfail=5 2>&1 | tee integration_test.log
tail -20 integration_test.log
```

### Run Complete Workflow
- [ ] Create atoms
- [ ] Attach StudentCalculator
- [ ] Run MD with benchmark
- [ ] Profile execution
- [ ] Generate reports

**Validation Script**:
```python
# integration_test.py
from ase import Atoms
from ase.md.verlet import VelocityVerlet
from ase import units
from mlff_distiller.models.student_calculator import StudentCalculator
import sys
sys.path.insert(0, 'benchmarks')
from md_benchmark import MDBenchmark

# Create system
atoms = Atoms('H2O', positions=[[0,0,0], [1,0,0], [0,1,0]])

# Attach calculator
calc = StudentCalculator(device='cpu')
atoms.calc = calc

# Run MD
dyn = VelocityVerlet(atoms, timestep=1.0*units.fs)
dyn.run(10)

# Benchmark
benchmark = MDBenchmark(calc, system_size=3)
results = benchmark.run(n_steps=10)

print("Full integration test: PASS")
print(f"Results: {results}")
```

**Run**:
```bash
python integration_test.py
```

---

## Performance Validation

### Benchmark Results
- [ ] Teacher model baseline established
- [ ] Timing results reasonable (not 0, not infinite)
- [ ] Memory usage tracked
- [ ] JSON output valid

### Profiling Results
- [ ] Latency distribution captured
- [ ] Hotspots identified
- [ ] Report generated
- [ ] Data exportable

---

## Documentation Validation

### Code Documentation
- [ ] All new classes have docstrings
- [ ] All public methods have docstrings
- [ ] Usage examples in docstrings
- [ ] Type hints present

**Validation Command**:
```bash
# Check for missing docstrings
python -c "
import ast
import inspect
from mlff_distiller.models.student_calculator import StudentCalculator

# Check class docstring
assert StudentCalculator.__doc__ is not None, 'Missing class docstring'

# Check method docstrings
for name, method in inspect.getmembers(StudentCalculator, predicate=inspect.isfunction):
    if not name.startswith('_'):
        assert method.__doc__ is not None, f'Missing docstring for {name}'

print('Documentation validation: PASS')
"
```

### External Documentation
- [ ] README updated if needed
- [ ] Examples directory has usage examples
- [ ] TESTING.md updated with new tests
- [ ] CHANGELOG or similar updated

---

## M1 Completion Validation

### All Issues Complete
- [ ] Issue #1: Data loading ✅ (Week 1)
- [ ] Issue #2: Teacher wrappers ✅ (Week 1)
- [ ] Issue #3: Training framework ✅ (Week 1)
- [ ] Issue #4: Pytest infrastructure ✅ (Week 1)
- [ ] Issue #5: MD benchmark framework ✅ (Week 2)
- [ ] Issue #6: Student calculator ✅ (Week 2)
- [ ] Issue #7: ASE interface tests ✅ (Week 2)
- [ ] Issue #8: CUDA environment ✅ (Week 1)
- [ ] Issue #9: MD profiling framework ✅ (Week 2)

### M1 Success Criteria
- [ ] All 9 M1 issues complete
- [ ] 200+ tests passing
- [ ] Teacher models benchmarked
- [ ] Student interface defined
- [ ] Performance baseline established
- [ ] No blocking issues for M2

### Ready for M2
- [ ] Can generate data from teacher models
- [ ] Student architecture can be implemented
- [ ] Performance targets defined (5-10x speedup)
- [ ] Testing infrastructure ready for distillation

---

## Checklist Sign-Off

### Mid-Week Checkpoint (Day 4)

Date: __________
Performed by: __________

**Status**:
- [ ] Week 1 foundation healthy
- [ ] Week 2 components progressing
- [ ] No blocking integration issues
- [ ] On track for M1 completion

**Issues Found**:
_________________________________________
_________________________________________

**Resolution Plan**:
_________________________________________
_________________________________________

### End of Week Validation (Day 7)

Date: __________
Performed by: __________

**Status**:
- [ ] All Week 2 components complete
- [ ] Full integration successful
- [ ] Test suite passing (200+ tests)
- [ ] M1 complete

**Final Test Results**:
- Total tests: __________
- Passed: __________
- Failed: __________
- Skipped: __________

**M1 Completion**:
- [ ] All 9 issues complete
- [ ] Baseline established
- [ ] Documentation updated
- [ ] Ready for M2

**Signature**: __________

---

## Troubleshooting

### Common Issues

**Import Errors**:
- Check `PYTHONPATH` includes `src/`
- Verify `__init__.py` files exist
- Run `pip install -e .` for editable install

**Test Failures**:
- Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
- Update fixtures if interfaces changed
- Check for hardcoded paths

**Integration Failures**:
- Run components individually first
- Check interface compatibility
- Verify data types match
- Review Week 1 patterns

**Performance Issues**:
- Check device (CPU vs CUDA)
- Verify reasonable system sizes
- Check for debug mode overhead

---

## Contact

**Issues/Questions**:
- Create GitHub issue
- Comment in related issue
- Tag coordinator

**Urgent Integration Problems**:
- Tag coordinator immediately
- Use "status:integration-issue" label
- Provide reproducible test case

---

*Document Version: 1.0*
*Last Updated: 2025-11-23*
*Coordinator: Lead Project Coordinator*
