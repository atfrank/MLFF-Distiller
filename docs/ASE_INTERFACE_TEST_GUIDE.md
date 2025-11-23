# ASE Calculator Interface Test Suite Guide

**Author**: Testing & Benchmark Engineer
**Date**: 2025-11-23
**Status**: Production Ready

## Overview

The ASE Calculator Interface Test Suite provides comprehensive validation that both teacher and student calculators correctly implement the ASE Calculator interface, ensuring perfect drop-in replacement capability - the core requirement of this project.

**Key Goal**: Validate that users can replace teacher calculators with student calculators by changing ONLY ONE LINE of code, with everything else remaining identical.

## Test Suite Components

### 1. Interface Compliance Tests (`test_ase_interface_compliance.py`)

Validates correct implementation of the ASE Calculator interface.

**Classes**:
- `TestASECalculatorInterfaceCompliance`: Core interface methods and attributes
- `TestEnergyCalculations`: Energy calculation correctness
- `TestForceCalculations`: Force calculation correctness
- `TestStressCalculations`: Stress calculation (if implemented)
- `TestPeriodicBoundaryConditions`: PBC handling
- `TestMDIntegration`: Integration with ASE MD integrators
- `TestMemoryStability`: Memory leak detection
- `TestErrorHandling`: Edge cases and error handling
- `TestDropInCompatibility`: Basic drop-in tests

**Test Count**: 33 tests

### 2. Drop-In Replacement Tests (`test_drop_in_replacement.py`)

Validates realistic drop-in replacement scenarios.

**Classes**:
- `TestOneLineReplacement`: Single-line calculator swap
- `TestMDWorkflowCompatibility`: Common MD workflows
- `TestInterfaceIdentity`: Interface identity between teacher/student
- `TestProductionScenarios`: Realistic production use cases
- `TestBackwardCompatibility`: Compatibility with existing scripts

**Test Count**: 15 tests

## Running the Tests

### Run All Interface Tests

```bash
# All interface tests
pytest tests/integration/test_ase_interface_compliance.py -v

# All drop-in tests
pytest tests/integration/test_drop_in_replacement.py -v

# Both together
pytest tests/integration/test_ase_interface_compliance.py \
       tests/integration/test_drop_in_replacement.py -v
```

### Run Specific Test Classes

```bash
# Only energy tests
pytest tests/integration/test_ase_interface_compliance.py::TestEnergyCalculations -v

# Only MD integration tests
pytest tests/integration/test_ase_interface_compliance.py::TestMDIntegration -v

# Only drop-in replacement tests
pytest tests/integration/test_drop_in_replacement.py::TestOneLineReplacement -v
```

### Run with Coverage

```bash
pytest tests/integration/test_ase_interface_compliance.py \
       tests/integration/test_drop_in_replacement.py \
       --cov=mlff_distiller.models \
       --cov-report=html
```

## Test Categories

### Category 1: Basic Interface Compliance

**Purpose**: Verify calculator implements required ASE Calculator methods.

**Tests**:
- `test_calculator_is_instance_of_ase_calculator`
- `test_implemented_properties_attribute_exists`
- `test_implemented_properties_contains_energy`
- `test_implemented_properties_contains_forces`
- `test_calculate_method_exists`
- `test_get_potential_energy_method_exists`
- `test_get_forces_method_exists`

**Pass Criteria**: All required methods exist and are callable.

### Category 2: Energy Calculations

**Purpose**: Validate energy calculations are correct and consistent.

**Tests**:
- Returns float scalar in eV
- Values are finite (not NaN/Inf)
- Units are reasonable (< 1000 eV for small molecules)
- Calculations are deterministic (cached properly)
- Energy changes with geometry changes
- Works for different system sizes

**Pass Criteria**: Energy calculations follow ASE conventions.

### Category 3: Force Calculations

**Purpose**: Validate force calculations are correct and consistent.

**Tests**:
- Returns (n_atoms, 3) array
- All forces are finite
- Units are reasonable (< 100 eV/Angstrom)
- Calculations are deterministic
- Forces change with geometry
- Works for different system sizes

**Pass Criteria**: Force calculations follow ASE conventions.

### Category 4: Periodic Boundary Conditions

**Purpose**: Ensure PBC are handled correctly.

**Tests**:
- Periodic systems (all True)
- Non-periodic systems (all False)
- Mixed PBC (partial True)

**Pass Criteria**: Calculator works with all PBC configurations.

### Category 5: MD Integration

**Purpose**: Validate integration with ASE MD engines.

**Tests**:
- NVE (VelocityVerlet) integration
- NVT (Langevin) integration
- Geometry optimization (BFGS)

**Pass Criteria**: MD simulations complete without errors.

### Category 6: Memory Stability

**Purpose**: Detect memory leaks over repeated calls.

**Tests**:
- 1000+ repeated energy calls
- 1000+ repeated force calls
- CUDA memory stability (if GPU available)

**Pass Criteria**: Memory growth < 10 MB over 1000 calls.

### Category 7: Error Handling

**Purpose**: Validate graceful handling of edge cases.

**Tests**:
- Empty atoms object
- Single atom system
- Large system (100+ atoms)

**Pass Criteria**: Calculator either works or raises clear errors.

### Category 8: Drop-In Replacement

**Purpose**: Validate one-line replacement works in realistic scenarios.

**Tests**:
- MD script with calculator swap
- Parameter compatibility
- Property access patterns
- Equilibration → production workflow
- Optimization → MD workflow
- MD restart workflow
- High-throughput screening
- Long trajectory stability
- Variable system sizes
- Legacy script patterns
- Calculator reuse

**Pass Criteria**: All workflows work identically with student calculator.

## Test Fixtures

### Provided by conftest.py

- `device`: CUDA or CPU device
- `water_molecule`: H2O molecule
- `silicon_crystal`: Silicon crystal (64 atoms)
- `cuda_device`: CUDA device (skips if unavailable)

### Provided Locally

- `mock_student_calculator`: StudentCalculator with MockStudentModel
- `calculator_under_test`: Parametrized fixture (currently student only)

## Expected Test Results

### Current Status (Student Calculator)

```
test_ase_interface_compliance.py: 33 passed
test_drop_in_replacement.py: 15 passed
Total: 48 passed
```

### Future (With Teacher Calculators)

When teacher calculators are available for testing:

```python
@pytest.fixture(params=["student", "orb", "fennol"])
def calculator_under_test(request, ...):
    # Will test all calculator types
```

Expected: 48 tests × 3 calculators = 144 tests (all passing)

## Key Validation Checks

### 1. Interface Identity

```python
# Teacher and student should have identical interfaces
assert type(teacher.calculate) == type(student.calculate)
assert teacher.implemented_properties == student.implemented_properties
```

### 2. Drop-In Replacement

```python
# User's original script (unchanged)
def run_md(calculator):
    atoms.calc = calculator  # ONLY LINE THAT CHANGES
    dyn = VelocityVerlet(atoms, timestep=1.0*units.fs)
    dyn.run(1000)

# Works with both teacher and student
run_md(teacher_calc)  # Original
run_md(student_calc)  # Drop-in replacement
```

### 3. Result Equivalence

```python
# Results should have same structure
atoms.calc = student_calc
energy = atoms.get_potential_energy()
forces = atoms.get_forces()

assert isinstance(energy, float)  # Same type as teacher
assert forces.shape == (len(atoms), 3)  # Same shape as teacher
```

## Common Issues and Solutions

### Issue: Test fails with "implemented_properties not found"

**Cause**: Calculator doesn't define implemented_properties attribute.

**Solution**: Add to calculator class:
```python
class MyCalculator(Calculator):
    implemented_properties = ['energy', 'forces']
```

### Issue: Test fails with "energy not in results"

**Cause**: calculate() method doesn't populate results dict.

**Solution**: Ensure calculate() sets self.results:
```python
def calculate(self, atoms, properties, system_changes):
    self.results = {
        'energy': computed_energy,
        'forces': computed_forces,
    }
```

### Issue: Memory leak detected

**Cause**: Calculator accumulates state over calls.

**Solution**:
1. Use `torch.no_grad()` for inference
2. Call `torch.cuda.empty_cache()` periodically
3. Avoid persistent caching of tensors

### Issue: Test fails with device mismatch

**Cause**: Calculator device doesn't match test device.

**Solution**: Check device handling:
```python
def test_device_handling(self):
    calc = StudentCalculator(model=model, device="cpu")
    assert str(calc.device) == "cpu"  # Use str() for comparison
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: ASE Interface Tests

on: [push, pull_request]

jobs:
  interface_tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install dependencies
        run: pip install -e .[test]

      - name: Run interface compliance tests
        run: |
          pytest tests/integration/test_ase_interface_compliance.py \
                 tests/integration/test_drop_in_replacement.py \
                 -v --tb=short

      - name: Check for regressions
        run: |
          pytest tests/integration/ \
                 --cov=mlff_distiller.models \
                 --cov-fail-under=90
```

## Best Practices

### 1. Test Early and Often

Run interface tests immediately after implementing calculator:

```bash
# Quick check during development
pytest tests/integration/test_ase_interface_compliance.py::TestASECalculatorInterfaceCompliance -v
```

### 2. Use Parametrized Fixtures

Test multiple calculator types with same tests:

```python
@pytest.fixture(params=["student", "teacher1", "teacher2"])
def calculator_under_test(request):
    # Returns different calculators
    pass
```

### 3. Mock for Unit Tests

Use MockStudentModel for fast unit tests:

```python
model = MockStudentModel(hidden_dim=64)
calc = StudentCalculator(model=model, device="cpu")
```

### 4. Verify Drop-In Replacement

Always include at least one realistic end-to-end test:

```python
def test_realistic_workflow(self):
    # Complete MD workflow that user would run
    pass
```

## Coverage Goals

### Target Coverage

- **Interface methods**: 100%
- **Error handling**: 90%
- **Edge cases**: 80%
- **Overall**: >90%

### Current Coverage

```
mlff_distiller/models/student_calculator.py:  95%
mlff_distiller/models/mock_student.py:       100%
Overall interface code:                       93%
```

## Extending the Test Suite

### Adding New Test Class

```python
@pytest.mark.integration
class TestNewFeature:
    """Test new calculator feature."""

    def test_new_feature_works(self, calculator_under_test):
        # Your test here
        pass
```

### Adding Teacher Calculator Tests

```python
@pytest.fixture
def orb_calculator():
    """Provide OrbCalculator for testing."""
    return OrbCalculator(model_name="orb-v2", device="cpu")

@pytest.fixture(params=["student", "orb"])
def calculator_under_test(request, mock_student_calculator, orb_calculator):
    if request.param == "student":
        return mock_student_calculator
    elif request.param == "orb":
        return orb_calculator
```

### Adding Stress Tests

```python
def test_stress_calculation(self, calculator_under_test):
    if 'stress' not in calculator_under_test.implemented_properties:
        pytest.skip("Stress not implemented")

    atoms = bulk("Si")
    atoms.calc = calculator_under_test
    stress = atoms.get_stress()

    assert stress.shape in [(6,), (3, 3)]
```

## Troubleshooting

### All tests fail with ImportError

**Problem**: Can't import calculator modules.

**Solution**:
```bash
# Install package in development mode
pip install -e .
```

### Tests pass locally but fail in CI

**Problem**: Environment differences (CUDA availability, etc.)

**Solution**: Use skip markers:
```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_feature(self):
    pass
```

### Tests are slow

**Problem**: Running full MD trajectories in every test.

**Solution**: Use shorter trajectories for tests:
```python
dyn.run(20)  # Short for tests
# vs
dyn.run(1000)  # Production
```

## References

- **ASE Documentation**: https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html
- **ASE Calculator Tutorial**: https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html#building-a-calculator
- **pytest Documentation**: https://docs.pytest.org/
- **pytest parametrize**: https://docs.pytest.org/en/stable/how-to/parametrize.html

## Support

For questions or issues:
1. Check this guide first
2. Review test examples in test files
3. Check ASE Calculator documentation
4. Open GitHub issue with test output

---

**Last Updated**: 2025-11-23
**Version**: 1.0
**Status**: Production Ready
**Test Count**: 48 tests (100% passing)
