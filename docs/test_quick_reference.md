# Test Infrastructure Quick Reference

## Quick Commands

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test types
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests only
pytest -m accuracy          # Accuracy tests only
pytest -m "not slow"        # Exclude slow tests
pytest -m cuda              # GPU tests only

# Run specific file
pytest tests/unit/test_fixtures_demo.py -v

# Run specific test
pytest tests/unit/test_fixtures_demo.py::test_water_molecule_fixture -v
```

## Available Fixtures

### Molecules
- `water_molecule` - H2O (3 atoms)
- `methane_molecule` - CH4 (5 atoms)
- `ethanol_molecule` - C2H5OH
- `ammonia_molecule` - NH3
- `small_molecule_set` - List of 5 molecules

### Crystals
- `silicon_crystal` - Si 2x2x2 supercell (64 atoms)
- `fcc_aluminum` - Al 2x2x2 supercell
- `bcc_iron` - Fe 2x2x2 supercell
- `nacl_crystal` - NaCl 2x2x2 supercell
- `small_periodic_system` - Si cell (8 atoms)

### Devices
- `device` - CUDA if available, else CPU
- `cpu_device` - Always CPU
- `cuda_device` - Always CUDA (skip if unavailable)
- `has_cuda` - Boolean CUDA availability

### Data
- `sample_training_data` - Dict with positions, energies, forces
- `sample_batch_data` - PyTorch batch tensors
- `random_positions` - Random positions (10, 3)
- `random_atomic_numbers` - Random Z values

### Utilities
- `temp_dir` - Temporary directory
- `temp_checkpoint_dir` - Checkpoint directory
- `temp_data_dir` - Data directory
- `energy_tolerance` - 1e-4 eV
- `force_tolerance` - 1e-3 eV/Ang
- `loose_energy_tolerance` - 1e-3 eV
- `loose_force_tolerance` - 0.01 eV/Ang

## Test Markers

```python
@pytest.mark.unit           # Unit test
@pytest.mark.integration    # Integration test
@pytest.mark.accuracy       # Accuracy validation
@pytest.mark.slow           # Takes >5 seconds
@pytest.mark.cuda           # Requires GPU
@pytest.mark.cpu            # CPU only
@pytest.mark.benchmark      # Performance test
@pytest.mark.requires_teacher   # Needs teacher model
@pytest.mark.requires_student   # Needs student model
```

## Example Test

```python
import pytest
import torch

@pytest.mark.unit
def test_example(water_molecule, device, energy_tolerance):
    """Example test using fixtures."""
    # Use water molecule
    positions = water_molecule.get_positions()

    # Convert to tensor on device
    pos_tensor = torch.tensor(positions, device=device)

    # Your test logic here
    result = my_function(pos_tensor)

    # Check with tolerance
    assert abs(result - expected) < energy_tolerance
```

## Common Patterns

### Test with parametrization
```python
@pytest.mark.unit
@pytest.mark.parametrize("n_atoms", [8, 16, 32])
def test_scaling(n_atoms, device):
    positions = torch.randn(n_atoms, 3, device=device)
    # Test code
```

### Test with temporary files
```python
@pytest.mark.unit
def test_save_load(temp_dir):
    output_file = temp_dir / "output.txt"
    output_file.write_text("data")
    assert output_file.exists()
```

### Test requiring CUDA
```python
@pytest.mark.cuda
def test_gpu_only(cuda_device):
    # This test skips if CUDA unavailable
    tensor = torch.randn(100, device=cuda_device)
```

### Test with multiple fixtures
```python
@pytest.mark.integration
def test_complex(silicon_crystal, device, temp_dir, energy_tolerance):
    # Use all fixtures together
    pass
```

## Coverage

```bash
# Generate HTML report
pytest --cov=src --cov-report=html
open htmlcov/index.html

# Check coverage threshold (80%)
pytest --cov=src --cov-fail-under=80
```

## Documentation

- Full guide: `TESTING.md`
- Fixture definitions: `tests/conftest.py`
- Example unit tests: `tests/unit/test_fixtures_demo.py`
- Example integration tests: `tests/integration/test_ase_integration_demo.py`
- Example accuracy tests: `tests/accuracy/test_accuracy_demo.py`
