# Testing Guide for MLFF Distiller

This document provides comprehensive guidance on writing and running tests for the MLFF Distiller project.

## Table of Contents

- [Quick Start](#quick-start)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Test Markers](#test-markers)
- [Available Fixtures](#available-fixtures)
- [Writing Tests](#writing-tests)
- [Coverage Requirements](#coverage-requirements)
- [CI/CD Integration](#cicd-integration)
- [Best Practices](#best-practices)

## Quick Start

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests only
pytest -m accuracy          # Accuracy validation tests only

# Run tests excluding slow ones
pytest -m "not slow"

# Run CUDA tests only (requires GPU)
pytest -m cuda

# Run with verbose output
pytest -v
```

## Test Structure

Tests are organized into three main categories:

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── unit/                    # Unit tests (fast, isolated)
│   ├── __init__.py
│   ├── test_fixtures_demo.py
│   └── test_*.py
├── integration/             # Integration tests (component interactions)
│   ├── __init__.py
│   ├── test_ase_integration_demo.py
│   └── test_*.py
└── accuracy/                # Accuracy validation tests
    ├── __init__.py
    ├── test_accuracy_demo.py
    └── test_*.py
```

### Test Categories

1. **Unit Tests** (`tests/unit/`)
   - Test individual components in isolation
   - Fast execution (< 1 second per test)
   - No external dependencies
   - Mock complex dependencies
   - Marker: `@pytest.mark.unit`

2. **Integration Tests** (`tests/integration/`)
   - Test component interactions
   - ASE interface testing
   - Data pipeline integration
   - May take longer to run
   - Marker: `@pytest.mark.integration`

3. **Accuracy Tests** (`tests/accuracy/`)
   - Validate model predictions
   - Compare against teacher models
   - Check numerical consistency
   - Energy/force accuracy validation
   - Marker: `@pytest.mark.accuracy`

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_fixtures_demo.py

# Run specific test function
pytest tests/unit/test_fixtures_demo.py::test_water_molecule_fixture

# Run tests matching pattern
pytest -k "water"

# Run with verbose output
pytest -v

# Run with even more detail
pytest -vv
```

### Running with Coverage

```bash
# Generate coverage report
pytest --cov=src --cov-report=term-missing

# Generate HTML coverage report
pytest --cov=src --cov-report=html
# Open htmlcov/index.html in browser

# Generate XML coverage (for CI)
pytest --cov=src --cov-report=xml

# Fail if coverage below 80%
pytest --cov=src --cov-fail-under=80
```

### Parallel Test Execution

```bash
# Run tests in parallel (requires pytest-xdist)
pytest -n auto              # Auto-detect CPU count
pytest -n 4                 # Use 4 workers
```

## Test Markers

Tests can be marked with decorators to categorize them:

```python
@pytest.mark.unit           # Unit test
@pytest.mark.integration    # Integration test
@pytest.mark.accuracy       # Accuracy validation test
@pytest.mark.slow           # Test takes >5 seconds
@pytest.mark.cuda           # Requires CUDA/GPU
@pytest.mark.cpu            # CPU-only test
@pytest.mark.benchmark      # Performance benchmark
@pytest.mark.requires_teacher   # Requires trained teacher model
@pytest.mark.requires_student   # Requires trained student model
```

### Running Tests by Marker

```bash
# Run only unit tests
pytest -m unit

# Run only CUDA tests
pytest -m cuda

# Run tests excluding slow ones
pytest -m "not slow"

# Run accuracy tests that don't require models
pytest -m "accuracy and not requires_teacher"

# Combine markers
pytest -m "integration and not slow"
```

## Available Fixtures

Fixtures are defined in `tests/conftest.py` and are automatically available to all tests.

### Device Fixtures

```python
def test_example(device):
    """Device is CUDA if available, else CPU."""
    tensor = torch.randn(10, device=device)

def test_cpu_only(cpu_device):
    """Always uses CPU."""
    tensor = torch.randn(10, device=cpu_device)

@pytest.mark.cuda
def test_gpu_only(cuda_device):
    """Requires CUDA, skipped if not available."""
    tensor = torch.randn(10, device=cuda_device)

def test_check_cuda(has_cuda):
    """Boolean indicating CUDA availability."""
    if has_cuda:
        # GPU-specific code
        pass
```

### Molecular Structure Fixtures

```python
def test_with_water(water_molecule):
    """H2O molecule as ASE Atoms."""
    assert len(water_molecule) == 3

def test_with_methane(methane_molecule):
    """CH4 molecule as ASE Atoms."""
    assert len(methane_molecule) == 5

def test_with_ethanol(ethanol_molecule):
    """C2H5OH molecule."""
    pass

def test_with_ammonia(ammonia_molecule):
    """NH3 molecule."""
    pass

def test_batch(small_molecule_set):
    """List of 5 small molecules."""
    for mol in small_molecule_set:
        # Process each molecule
        pass
```

### Periodic Structure Fixtures

```python
def test_silicon(silicon_crystal):
    """2x2x2 supercell of diamond Si (64 atoms)."""
    assert len(silicon_crystal) == 64

def test_aluminum(fcc_aluminum):
    """2x2x2 supercell of FCC Al."""
    pass

def test_iron(bcc_iron):
    """2x2x2 supercell of BCC Fe."""
    pass

def test_nacl(nacl_crystal):
    """2x2x2 supercell of NaCl."""
    pass

def test_small_system(small_periodic_system):
    """Small Si cell (8 atoms) for quick tests."""
    pass
```

### Temporary Directory Fixtures

```python
def test_file_io(temp_dir):
    """Temporary directory, auto-cleaned."""
    output_file = temp_dir / "output.txt"
    output_file.write_text("test")
    assert output_file.exists()

def test_checkpoints(temp_checkpoint_dir):
    """Temporary checkpoint directory."""
    checkpoint = temp_checkpoint_dir / "model.pt"
    torch.save({}, checkpoint)

def test_data(temp_data_dir):
    """Temporary data directory."""
    pass
```

### Data Fixtures

```python
def test_training_data(sample_training_data):
    """Dictionary with positions, atomic_numbers, energies, forces."""
    positions = sample_training_data["positions"]  # (10, 8, 3)
    energies = sample_training_data["energies"]    # (10,)

def test_batch_data(sample_batch_data):
    """PyTorch batch with positions, atomic_numbers, batch indices."""
    positions = sample_batch_data["positions"]
    batch = sample_batch_data["batch"]
```

### Tolerance Fixtures

```python
def test_accuracy(energy_tolerance, force_tolerance):
    """Strict tolerances for exact comparisons."""
    assert energy_mae < energy_tolerance  # 1e-4 eV
    assert force_mae < force_tolerance    # 1e-3 eV/Ang

def test_distilled_model(loose_energy_tolerance, loose_force_tolerance):
    """Loose tolerances for distilled models."""
    assert energy_mae < loose_energy_tolerance  # 1e-3 eV
    assert force_mae < loose_force_tolerance    # 0.01 eV/Ang
```

## Writing Tests

### Unit Test Example

```python
import pytest
import torch

@pytest.mark.unit
def test_model_forward_pass(device):
    """Test model forward pass with dummy data."""
    from src.models.student_model import StudentModel

    model = StudentModel().to(device)
    positions = torch.randn(10, 3, device=device)
    atomic_numbers = torch.ones(10, dtype=torch.long, device=device)

    output = model(positions, atomic_numbers)

    assert "energy" in output
    assert output["energy"].shape == ()
    assert not torch.isnan(output["energy"])
```

### Integration Test Example

```python
import pytest
from ase import Atoms

@pytest.mark.integration
def test_ase_calculator(water_molecule, device):
    """Test ASE calculator interface."""
    from src.inference.ase_calculator import MLFFCalculator

    calc = MLFFCalculator(device=device)
    water_molecule.calc = calc

    energy = water_molecule.get_potential_energy()
    forces = water_molecule.get_forces()

    assert isinstance(energy, float)
    assert forces.shape == (3, 3)
```

### Accuracy Test Example

```python
import pytest
import numpy as np

@pytest.mark.accuracy
@pytest.mark.requires_teacher
def test_energy_accuracy(water_molecule, loose_energy_tolerance):
    """Compare student vs teacher energy predictions."""
    from src.models import load_teacher, load_student

    teacher = load_teacher()
    student = load_student()

    teacher_energy = teacher.predict(water_molecule)
    student_energy = student.predict(water_molecule)

    mae = abs(teacher_energy - student_energy)
    assert mae < loose_energy_tolerance
```

### Parametrized Tests

```python
@pytest.mark.unit
@pytest.mark.parametrize("n_atoms", [1, 5, 10, 50])
def test_varying_sizes(n_atoms, device):
    """Test with different system sizes."""
    positions = torch.randn(n_atoms, 3, device=device)
    # Test code here

@pytest.mark.unit
@pytest.mark.parametrize("element,z", [("H", 1), ("C", 6), ("O", 8)])
def test_elements(element, z):
    """Test different elements."""
    assert atomic_number(element) == z
```

### Using Multiple Fixtures

```python
@pytest.mark.integration
def test_complex_scenario(
    water_molecule,
    device,
    temp_dir,
    energy_tolerance,
):
    """Combine multiple fixtures."""
    # Use water_molecule
    # Use device for computations
    # Save outputs to temp_dir
    # Check accuracy with energy_tolerance
    pass
```

## Coverage Requirements

- **Target coverage: 80%** (enforced in `pyproject.toml`)
- Unit tests should cover >90% of core logic
- Integration tests cover component interactions
- Accuracy tests validate end-to-end behavior

### Checking Coverage

```bash
# Generate coverage report
pytest --cov=src --cov-report=term-missing

# View detailed HTML report
pytest --cov=src --cov-report=html
open htmlcov/index.html

# Check specific module
pytest --cov=src.models --cov-report=term-missing
```

### Coverage Exclusions

Lines excluded from coverage (configured in `pyproject.toml`):

- `pragma: no cover` comments
- `__repr__` methods
- Abstract methods (`@abstractmethod`)
- Type checking blocks (`if TYPE_CHECKING:`)
- Main guards (`if __name__ == "__main__":`)

## CI/CD Integration

Tests run automatically on GitHub Actions for:

- Pull requests to `main` or `develop`
- Pushes to `main` or `develop`

### CI Test Matrix

Tests run on:
- Python 3.9, 3.10, 3.11
- Ubuntu latest
- CPU only (GPU tests skipped in CI)

### Skipping Tests in CI

```python
import os

@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Requires GPU not available in CI"
)
def test_cuda_only():
    pass
```

### Environment Variables for CI

- `MLFF_TEST_CPU_ONLY=1` - Force CPU-only testing
- `MLFF_TEACHER_MODEL_PATH` - Path to teacher model (enables teacher tests)
- `MLFF_STUDENT_MODEL_PATH` - Path to student model (enables student tests)

## Best Practices

### Test Structure

1. **Arrange-Act-Assert pattern:**
   ```python
   def test_something():
       # Arrange: Set up test data
       data = create_test_data()

       # Act: Execute the code under test
       result = function_under_test(data)

       # Assert: Verify the results
       assert result == expected_value
   ```

2. **One concept per test:**
   - Test one thing at a time
   - Use descriptive test names
   - Keep tests focused

3. **Use fixtures for setup:**
   - Don't repeat setup code
   - Leverage conftest.py fixtures
   - Create custom fixtures for complex setups

### Test Naming

```python
# Good test names
def test_water_molecule_has_three_atoms():
    pass

def test_model_raises_error_on_invalid_input():
    pass

def test_optimizer_converges_for_simple_system():
    pass

# Less clear names
def test_water():
    pass

def test_error():
    pass

def test_optimization():
    pass
```

### Assertions

```python
# Use specific assertions with messages
assert result == expected, f"Expected {expected}, got {result}"

# Use numpy testing for arrays
np.testing.assert_allclose(arr1, arr2, rtol=1e-5, atol=1e-8)

# Use pytest.raises for exceptions
with pytest.raises(ValueError, match="Invalid input"):
    function_that_should_raise()

# Check multiple properties
assert len(output) > 0, "Output should not be empty"
assert all(x > 0 for x in output), "All values should be positive"
```

### Randomness and Reproducibility

```python
# All tests automatically use seed 42 (conftest.py)
# For test-specific seeds:
def test_with_custom_seed():
    rng = np.random.RandomState(123)
    data = rng.randn(10)
    # Test code
```

### Skipping Tests

```python
# Skip test unconditionally
@pytest.mark.skip(reason="Not implemented yet")
def test_future_feature():
    pass

# Skip based on condition
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_cuda_feature():
    pass

# Expected failure (test is known to fail)
@pytest.mark.xfail(reason="Known bug #123")
def test_known_issue():
    pass
```

### Test Data

1. **Use fixtures for common data:**
   - Defined in `conftest.py`
   - Automatically available
   - Cleaned up after use

2. **Keep test data small:**
   - Use minimal systems for unit tests
   - Use larger systems only when necessary

3. **Use fixed seeds:**
   - Ensure reproducibility
   - Makes debugging easier

### Performance

1. **Keep unit tests fast (<1s each):**
   - Mark slow tests with `@pytest.mark.slow`
   - Use small test systems
   - Mock expensive operations

2. **Run slow tests separately:**
   ```bash
   pytest -m "not slow"  # Fast tests only
   pytest -m slow        # Slow tests only
   ```

3. **Use pytest-xdist for parallelization:**
   ```bash
   pytest -n auto
   ```

## Troubleshooting

### Tests Pass Locally but Fail in CI

- Check Python version differences
- Verify all dependencies are in `pyproject.toml`
- Check for platform-specific code
- Look for timing-dependent tests

### CUDA Tests Skipped

- Check `torch.cuda.is_available()`
- Verify CUDA drivers installed
- Use `@pytest.mark.cuda` marker
- Set `MLFF_TEST_CPU_ONLY=0`

### Coverage Not Meeting Target

- Run `pytest --cov=src --cov-report=term-missing`
- Check missing lines in report
- Add tests for uncovered code
- Use `# pragma: no cover` for untestable code

### Fixtures Not Found

- Ensure `conftest.py` exists
- Check fixture name spelling
- Verify fixture scope
- Check if fixture is in correct `conftest.py`

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [ASE documentation](https://wiki.fysik.dtu.dk/ase/)
- [PyTorch testing](https://pytorch.org/docs/stable/testing.html)

## Getting Help

- Check existing tests for examples
- Review this documentation
- Ask in GitHub issues
- Tag `@testing-benchmark-engineer` in PRs
