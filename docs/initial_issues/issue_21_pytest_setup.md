# [Testing] [M1] Configure pytest and test infrastructure

## Assigned Agent
testing-benchmark-engineer

## Milestone
M1: Setup & Baseline

## Task Description
Set up comprehensive testing infrastructure using pytest. This includes configuring test directories, fixtures, coverage reporting, and establishing testing best practices for the project.

## Context & Background
A robust testing infrastructure is critical for maintaining code quality throughout the project. We need:
- Unit tests for individual components
- Integration tests for component interactions
- Test fixtures for common test data
- Coverage reporting to track test completeness
- CI integration for automated testing

## Acceptance Criteria
- [ ] Configure pytest in `pyproject.toml`
- [ ] Create test directory structure (`tests/unit/`, `tests/integration/`)
- [ ] Set up pytest fixtures in `tests/conftest.py`
- [ ] Configure coverage reporting (pytest-cov)
- [ ] Create testing utilities and helpers
- [ ] Add example tests demonstrating best practices
- [ ] Set up test data fixtures (molecular structures)
- [ ] Document testing guidelines in `docs/TESTING.md`
- [ ] Verify CI runs tests automatically
- [ ] Achieve initial test coverage baseline

## Technical Notes

### Test Directory Structure
```
tests/
├── conftest.py           # Shared fixtures
├── unit/                 # Unit tests
│   ├── test_data_loader.py
│   ├── test_models.py
│   ├── test_training.py
│   └── test_utils.py
├── integration/          # Integration tests
│   ├── test_data_pipeline.py
│   ├── test_training_pipeline.py
│   └── test_inference_pipeline.py
├── fixtures/             # Test data
│   ├── structures/       # Example molecular structures
│   └── configs/          # Test configurations
└── utils/                # Testing utilities
    └── helpers.py
```

### Key Fixtures

```python
# tests/conftest.py
import pytest
import torch
import numpy as np
from ase import Atoms

@pytest.fixture
def device():
    """Device fixture for tests."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def simple_molecule():
    """Create a simple water molecule for testing."""
    return Atoms(
        symbols="H2O",
        positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        cell=[10, 10, 10],
        pbc=True
    )

@pytest.fixture
def batch_data():
    """Create a batch of test data."""
    return {
        "positions": torch.randn(8, 20, 3),
        "species": torch.randint(1, 10, (8, 20)),
        "cells": torch.eye(3).unsqueeze(0).repeat(8, 1, 1),
        "mask": torch.ones(8, 20, dtype=torch.bool)
    }

@pytest.fixture
def tmp_checkpoint_dir(tmp_path):
    """Temporary directory for checkpoints."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir
```

### Testing Best Practices

1. **Test Organization**:
   - One test file per source file
   - Group related tests in classes
   - Use descriptive test names

2. **Test Patterns**:
```python
def test_data_loader_handles_variable_sizes():
    """Test that DataLoader correctly handles variable-sized molecules."""
    # Arrange
    molecules = [create_molecule(n_atoms=n) for n in [10, 20, 30]]
    loader = MolecularDataLoader(molecules, batch_size=3)

    # Act
    batch = next(iter(loader))

    # Assert
    assert batch["positions"].shape[0] == 3  # batch size
    assert batch["positions"].shape[1] == 30  # padded to max
    assert torch.all(batch["mask"].sum(dim=1) == torch.tensor([10, 20, 30]))
```

3. **Parametrized Tests**:
```python
@pytest.mark.parametrize("n_atoms", [10, 50, 100, 500])
def test_model_scales_with_system_size(n_atoms):
    """Test model handles different system sizes."""
    # Test implementation
    pass
```

4. **Mock External Dependencies**:
```python
from unittest.mock import Mock, patch

def test_teacher_wrapper_with_mock():
    """Test teacher wrapper with mocked model."""
    with patch('src.models.teacher_wrapper.load_pretrained') as mock_load:
        mock_model = Mock()
        mock_model.predict.return_value = {"energy": 1.0}
        mock_load.return_value = mock_model
        # Test logic
```

### Coverage Configuration

```toml
[tool.coverage.run]
source = ["src"]
omit = [
    "*/tests/*",
    "*/__init__.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
]
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_data_loader.py

# Run with verbose output
pytest -v

# Run tests matching pattern
pytest -k "data_loader"

# Run in parallel
pytest -n auto
```

## Related Issues
- Enables: All other issues (provides testing infrastructure)
- Related to: #22 (benchmark framework)

## Dependencies
- pytest
- pytest-cov (coverage)
- pytest-xdist (parallel execution)
- pytest-mock (mocking utilities)

## Estimated Complexity
Low (2-3 days)

## Definition of Done
- [ ] pytest configuration complete
- [ ] Test directory structure created
- [ ] Common fixtures implemented
- [ ] Example tests written and passing
- [ ] Coverage reporting working
- [ ] Testing documentation written
- [ ] CI successfully runs tests
- [ ] PR created and reviewed
- [ ] PR merged to main

## Resources
- pytest documentation: https://docs.pytest.org/
- pytest best practices: https://docs.pytest.org/en/stable/goodpractices.html
- pytest-cov: https://pytest-cov.readthedocs.io/
- Testing guidelines: https://realpython.com/pytest-python-testing/

## Notes
- Start with basic fixtures and expand as needed
- Ensure tests run quickly (mock slow operations)
- Document testing patterns for other agents
- Consider adding pytest markers for slow tests
