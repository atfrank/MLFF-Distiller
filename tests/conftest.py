"""Shared pytest fixtures for MLFF Distiller tests.

This module provides common test fixtures for molecular structures, devices,
and test data that are used across unit, integration, and accuracy tests.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Generator, List

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk, molecule


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Fixture providing the appropriate compute device (CUDA if available, else CPU).

    Returns:
        torch.device: CUDA device if available and not disabled, otherwise CPU.
    """
    # Allow tests to force CPU via environment variable
    force_cpu = os.environ.get("MLFF_TEST_CPU_ONLY", "0") == "1"

    if force_cpu:
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture(scope="session")
def has_cuda() -> bool:
    """Fixture indicating whether CUDA is available.

    Returns:
        bool: True if CUDA is available, False otherwise.
    """
    return torch.cuda.is_available()


@pytest.fixture
def cpu_device() -> torch.device:
    """Fixture providing CPU device explicitly.

    Returns:
        torch.device: CPU device.
    """
    return torch.device("cpu")


@pytest.fixture
def cuda_device() -> torch.device:
    """Fixture providing CUDA device.

    Returns:
        torch.device: CUDA device.

    Raises:
        pytest.skip: If CUDA is not available.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


# Molecular structure fixtures
@pytest.fixture
def water_molecule() -> Atoms:
    """Fixture providing a water molecule (H2O).

    Returns:
        Atoms: ASE Atoms object for water molecule.
    """
    return molecule("H2O")


@pytest.fixture
def methane_molecule() -> Atoms:
    """Fixture providing a methane molecule (CH4).

    Returns:
        Atoms: ASE Atoms object for methane molecule.
    """
    return molecule("CH4")


@pytest.fixture
def ethanol_molecule() -> Atoms:
    """Fixture providing an ethanol molecule (C2H5OH).

    Returns:
        Atoms: ASE Atoms object for ethanol molecule.
    """
    return molecule("CH3CH2OH")


@pytest.fixture
def ammonia_molecule() -> Atoms:
    """Fixture providing an ammonia molecule (NH3).

    Returns:
        Atoms: ASE Atoms object for ammonia molecule.
    """
    return molecule("NH3")


@pytest.fixture
def small_molecule_set() -> List[Atoms]:
    """Fixture providing a set of small molecules for batch testing.

    Returns:
        List[Atoms]: List of ASE Atoms objects for various small molecules.
    """
    molecules = ["H2O", "CH4", "NH3", "CO2", "H2"]
    return [molecule(mol) for mol in molecules]


# Periodic structure fixtures
@pytest.fixture
def silicon_crystal() -> Atoms:
    """Fixture providing a silicon crystal structure.

    Returns:
        Atoms: ASE Atoms object for diamond-structure silicon (2x2x2 supercell, 64 atoms).
    """
    si = bulk("Si", "diamond", a=5.43, cubic=True)
    return si.repeat((2, 2, 2))


@pytest.fixture
def fcc_aluminum() -> Atoms:
    """Fixture providing an FCC aluminum crystal.

    Returns:
        Atoms: ASE Atoms object for FCC aluminum (2x2x2 supercell).
    """
    al = bulk("Al", "fcc", a=4.05)
    return al.repeat((2, 2, 2))


@pytest.fixture
def bcc_iron() -> Atoms:
    """Fixture providing a BCC iron crystal.

    Returns:
        Atoms: ASE Atoms object for BCC iron (2x2x2 supercell).
    """
    fe = bulk("Fe", "bcc", a=2.87)
    return fe.repeat((2, 2, 2))


@pytest.fixture
def nacl_crystal() -> Atoms:
    """Fixture providing a NaCl (rock salt) crystal.

    Returns:
        Atoms: ASE Atoms object for NaCl crystal (2x2x2 supercell).
    """
    nacl = bulk("NaCl", "rocksalt", a=5.64)
    return nacl.repeat((2, 2, 2))


@pytest.fixture
def small_periodic_system() -> Atoms:
    """Fixture providing a small periodic system for quick testing.

    Returns:
        Atoms: ASE Atoms object for a simple cubic silicon cell (8 atoms).
    """
    return bulk("Si", "diamond", a=5.43)


# Random structure fixtures
@pytest.fixture
def random_positions() -> np.ndarray:
    """Fixture providing random atomic positions.

    Returns:
        np.ndarray: Random positions array of shape (10, 3).
    """
    rng = np.random.RandomState(42)  # Fixed seed for reproducibility
    return rng.rand(10, 3) * 10.0  # 10 atoms in a 10x10x10 box


@pytest.fixture
def random_atomic_numbers() -> np.ndarray:
    """Fixture providing random atomic numbers (H, C, N, O).

    Returns:
        np.ndarray: Random atomic numbers array of shape (10,).
    """
    rng = np.random.RandomState(42)
    # Common elements: H(1), C(6), N(7), O(8)
    return rng.choice([1, 6, 7, 8], size=10)


# Temporary directory fixtures
@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Fixture providing a temporary directory for test outputs.

    Yields:
        Path: Path to temporary directory that is cleaned up after test.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_checkpoint_dir(temp_dir: Path) -> Path:
    """Fixture providing a temporary directory for model checkpoints.

    Args:
        temp_dir: Temporary directory from temp_dir fixture.

    Returns:
        Path: Path to checkpoint directory within temp_dir.
    """
    checkpoint_dir = temp_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    return checkpoint_dir


@pytest.fixture
def temp_data_dir(temp_dir: Path) -> Path:
    """Fixture providing a temporary directory for test data.

    Args:
        temp_dir: Temporary directory from temp_dir fixture.

    Returns:
        Path: Path to data directory within temp_dir.
    """
    data_dir = temp_dir / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


# Data fixtures
@pytest.fixture
def sample_training_data() -> Dict[str, np.ndarray]:
    """Fixture providing sample training data.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing positions, atomic_numbers,
            energies, and forces for a small training set.
    """
    rng = np.random.RandomState(42)
    n_samples = 10
    n_atoms = 8

    return {
        "positions": rng.rand(n_samples, n_atoms, 3) * 10.0,
        "atomic_numbers": np.tile([14], (n_samples, n_atoms)),  # All silicon
        "energies": rng.randn(n_samples) * 5.0,
        "forces": rng.randn(n_samples, n_atoms, 3) * 0.1,
    }


@pytest.fixture
def sample_batch_data(device: torch.device) -> Dict[str, torch.Tensor]:
    """Fixture providing sample batch data as PyTorch tensors.

    Args:
        device: Device to place tensors on.

    Returns:
        Dict[str, torch.Tensor]: Dictionary containing positions, atomic_numbers,
            and batch indices as PyTorch tensors.
    """
    batch_size = 4
    n_atoms_per_structure = 8
    total_atoms = batch_size * n_atoms_per_structure

    rng = np.random.RandomState(42)

    # Create batch indices (use cpu tensors to avoid device comparison issues)
    positions_data = rng.rand(total_atoms, 3) * 10.0
    atomic_numbers_data = np.tile([14], total_atoms)
    batch_data = np.repeat(np.arange(batch_size), n_atoms_per_structure)

    return {
        "positions": torch.tensor(positions_data, dtype=torch.float32).to(device),
        "atomic_numbers": torch.tensor(atomic_numbers_data, dtype=torch.long).to(device),
        "batch": torch.tensor(batch_data, dtype=torch.long).to(device),
    }


# Tolerance fixtures
@pytest.fixture
def energy_tolerance() -> float:
    """Fixture providing default energy tolerance for comparisons (eV).

    Returns:
        float: Energy tolerance in eV (1e-4 eV = 0.1 meV).
    """
    return 1e-4


@pytest.fixture
def force_tolerance() -> float:
    """Fixture providing default force tolerance for comparisons (eV/Angstrom).

    Returns:
        float: Force tolerance in eV/Angstrom.
    """
    return 1e-3


@pytest.fixture
def loose_energy_tolerance() -> float:
    """Fixture providing loose energy tolerance for distilled models (eV).

    Returns:
        float: Loose energy tolerance in eV (1 meV).
    """
    return 1e-3


@pytest.fixture
def loose_force_tolerance() -> float:
    """Fixture providing loose force tolerance for distilled models (eV/Angstrom).

    Returns:
        float: Loose force tolerance in eV/Angstrom.
    """
    return 0.01


# Random seed fixture
@pytest.fixture(autouse=True)
def random_seed():
    """Fixture to set random seeds for reproducibility in all tests.

    This fixture runs automatically for all tests.
    """
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Skip markers
def pytest_configure(config):
    """Configure custom pytest markers and skip conditions."""
    config.addinivalue_line(
        "markers", "requires_model: mark test as requiring a trained model"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add skip markers based on environment.

    Args:
        config: Pytest config object.
        items: List of collected test items.
    """
    skip_cuda = pytest.mark.skip(reason="CUDA not available")
    skip_teacher = pytest.mark.skip(reason="Teacher model not available")
    skip_student = pytest.mark.skip(reason="Student model not available")

    has_cuda_available = torch.cuda.is_available()

    # Check for model availability (these would be set in CI or test environment)
    has_teacher = os.environ.get("MLFF_TEACHER_MODEL_PATH") is not None
    has_student = os.environ.get("MLFF_STUDENT_MODEL_PATH") is not None

    for item in items:
        # Skip CUDA tests if CUDA not available
        if "cuda" in item.keywords and not has_cuda_available:
            item.add_marker(skip_cuda)

        # Skip tests requiring teacher model if not available
        if "requires_teacher" in item.keywords and not has_teacher:
            item.add_marker(skip_teacher)

        # Skip tests requiring student model if not available
        if "requires_student" in item.keywords and not has_student:
            item.add_marker(skip_student)
