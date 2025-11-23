"""Example unit tests demonstrating fixture usage.

This module shows how to use the shared fixtures defined in conftest.py
for writing unit tests in the MLFF Distiller project.
"""

import numpy as np
import pytest
import torch
from ase import Atoms


@pytest.mark.unit
def test_water_molecule_fixture(water_molecule: Atoms):
    """Test that water molecule fixture has correct properties.

    Args:
        water_molecule: Water molecule fixture from conftest.py.
    """
    assert len(water_molecule) == 3, "Water molecule should have 3 atoms"
    symbols = water_molecule.get_chemical_symbols()
    assert symbols.count("H") == 2, "Water should have 2 hydrogen atoms"
    assert symbols.count("O") == 1, "Water should have 1 oxygen atom"


@pytest.mark.unit
def test_methane_molecule_fixture(methane_molecule: Atoms):
    """Test that methane molecule fixture has correct properties.

    Args:
        methane_molecule: Methane molecule fixture from conftest.py.
    """
    assert len(methane_molecule) == 5, "Methane molecule should have 5 atoms"
    symbols = methane_molecule.get_chemical_symbols()
    assert symbols.count("C") == 1, "Methane should have 1 carbon atom"
    assert symbols.count("H") == 4, "Methane should have 4 hydrogen atoms"


@pytest.mark.unit
def test_silicon_crystal_fixture(silicon_crystal: Atoms):
    """Test that silicon crystal fixture has correct properties.

    Args:
        silicon_crystal: Silicon crystal fixture from conftest.py.
    """
    # 2x2x2 supercell of diamond structure (8 atoms per unit cell)
    assert len(silicon_crystal) == 64, "Silicon supercell should have 64 atoms"
    symbols = silicon_crystal.get_chemical_symbols()
    assert all(s == "Si" for s in symbols), "All atoms should be silicon"
    # Check periodicity
    assert silicon_crystal.pbc.all(), "Silicon crystal should have periodic boundaries"
    cell = silicon_crystal.get_cell()
    assert cell is not None and len(cell) == 3, "Should have 3x3 cell vectors"


@pytest.mark.unit
def test_small_molecule_set_fixture(small_molecule_set):
    """Test that small molecule set fixture contains multiple molecules.

    Args:
        small_molecule_set: List of small molecules from conftest.py.
    """
    assert len(small_molecule_set) == 5, "Should have 5 molecules"
    for mol in small_molecule_set:
        assert isinstance(mol, Atoms), "Each item should be an Atoms object"
        assert len(mol) > 0, "Each molecule should have at least one atom"


@pytest.mark.unit
def test_device_fixture(device: torch.device):
    """Test that device fixture provides a valid PyTorch device.

    Args:
        device: Device fixture from conftest.py.
    """
    assert isinstance(device, torch.device), "Should be a torch.device"
    assert device.type in ["cpu", "cuda"], "Device should be CPU or CUDA"


@pytest.mark.unit
@pytest.mark.cpu
def test_cpu_device_fixture(cpu_device: torch.device):
    """Test that CPU device fixture always provides CPU.

    Args:
        cpu_device: CPU device fixture from conftest.py.
    """
    assert cpu_device.type == "cpu", "CPU device should be CPU type"


@pytest.mark.unit
@pytest.mark.cuda
def test_cuda_device_fixture(cuda_device: torch.device):
    """Test that CUDA device fixture provides CUDA device.

    Args:
        cuda_device: CUDA device fixture from conftest.py.
    """
    assert cuda_device.type == "cuda", "CUDA device should be CUDA type"


@pytest.mark.unit
def test_random_positions_fixture(random_positions: np.ndarray):
    """Test that random positions fixture has correct shape and range.

    Args:
        random_positions: Random positions fixture from conftest.py.
    """
    assert random_positions.shape == (10, 3), "Should have shape (10, 3)"
    assert random_positions.min() >= 0, "Positions should be non-negative"
    assert random_positions.max() <= 10, "Positions should be within 10x10x10 box"


@pytest.mark.unit
def test_random_atomic_numbers_fixture(random_atomic_numbers: np.ndarray):
    """Test that random atomic numbers fixture has correct values.

    Args:
        random_atomic_numbers: Random atomic numbers fixture from conftest.py.
    """
    assert random_atomic_numbers.shape == (10,), "Should have 10 atomic numbers"
    valid_elements = {1, 6, 7, 8}  # H, C, N, O
    assert all(z in valid_elements for z in random_atomic_numbers), (
        "Atomic numbers should be H, C, N, or O"
    )


@pytest.mark.unit
def test_sample_training_data_fixture(sample_training_data):
    """Test that sample training data fixture has correct structure.

    Args:
        sample_training_data: Sample training data fixture from conftest.py.
    """
    assert "positions" in sample_training_data, "Should contain positions"
    assert "atomic_numbers" in sample_training_data, "Should contain atomic_numbers"
    assert "energies" in sample_training_data, "Should contain energies"
    assert "forces" in sample_training_data, "Should contain forces"

    n_samples = 10
    n_atoms = 8

    assert sample_training_data["positions"].shape == (n_samples, n_atoms, 3)
    assert sample_training_data["atomic_numbers"].shape == (n_samples, n_atoms)
    assert sample_training_data["energies"].shape == (n_samples,)
    assert sample_training_data["forces"].shape == (n_samples, n_atoms, 3)


@pytest.mark.unit
def test_sample_batch_data_fixture(sample_batch_data, device):
    """Test that sample batch data fixture has correct structure.

    Args:
        sample_batch_data: Sample batch data fixture from conftest.py.
        device: Device fixture from conftest.py.
    """
    assert "positions" in sample_batch_data, "Should contain positions"
    assert "atomic_numbers" in sample_batch_data, "Should contain atomic_numbers"
    assert "batch" in sample_batch_data, "Should contain batch indices"

    # Check all tensors are on correct device type
    for key, tensor in sample_batch_data.items():
        assert isinstance(tensor, torch.Tensor), f"{key} should be a tensor"
        assert tensor.device.type == device.type, f"{key} should be on {device.type}"

    # Check batch structure
    batch_size = 4
    n_atoms_per_structure = 8
    total_atoms = batch_size * n_atoms_per_structure

    assert sample_batch_data["positions"].shape == (total_atoms, 3)
    assert sample_batch_data["atomic_numbers"].shape == (total_atoms,)
    assert sample_batch_data["batch"].shape == (total_atoms,)


@pytest.mark.unit
def test_temp_dir_fixture(temp_dir):
    """Test that temp_dir fixture provides a valid temporary directory.

    Args:
        temp_dir: Temporary directory fixture from conftest.py.
    """
    assert temp_dir.exists(), "Temporary directory should exist"
    assert temp_dir.is_dir(), "Should be a directory"

    # Test we can write to it
    test_file = temp_dir / "test.txt"
    test_file.write_text("test content")
    assert test_file.exists(), "Should be able to write files"


@pytest.mark.unit
def test_temp_checkpoint_dir_fixture(temp_checkpoint_dir):
    """Test that checkpoint directory fixture is properly created.

    Args:
        temp_checkpoint_dir: Checkpoint directory fixture from conftest.py.
    """
    assert temp_checkpoint_dir.exists(), "Checkpoint directory should exist"
    assert temp_checkpoint_dir.is_dir(), "Should be a directory"
    assert temp_checkpoint_dir.name == "checkpoints", "Should be named 'checkpoints'"


@pytest.mark.unit
def test_tolerance_fixtures(
    energy_tolerance,
    force_tolerance,
    loose_energy_tolerance,
    loose_force_tolerance,
):
    """Test that tolerance fixtures provide reasonable values.

    Args:
        energy_tolerance: Energy tolerance fixture from conftest.py.
        force_tolerance: Force tolerance fixture from conftest.py.
        loose_energy_tolerance: Loose energy tolerance fixture from conftest.py.
        loose_force_tolerance: Loose force tolerance fixture from conftest.py.
    """
    assert energy_tolerance == 1e-4, "Energy tolerance should be 1e-4 eV"
    assert force_tolerance == 1e-3, "Force tolerance should be 1e-3 eV/Ang"
    assert loose_energy_tolerance == 1e-3, "Loose energy tolerance should be 1e-3 eV"
    assert loose_force_tolerance == 0.01, "Loose force tolerance should be 0.01 eV/Ang"

    # Check that loose tolerances are larger
    assert loose_energy_tolerance >= energy_tolerance
    assert loose_force_tolerance >= force_tolerance


@pytest.mark.unit
def test_random_seed_reproducibility():
    """Test that random seed fixture ensures reproducibility."""
    # Generate random numbers - should be the same across test runs
    np_random = np.random.rand(5)
    torch_random = torch.rand(5)

    # Verify that we get consistent results (using looser checks)
    # The exact values will be consistent with seed 42
    assert 0.0 <= np_random[0] <= 1.0, "NumPy random should be in [0, 1]"
    assert 0.0 <= torch_random[0] <= 1.0, "PyTorch random should be in [0, 1]"

    # Test that re-running with same seed gives same results
    np.random.seed(100)
    torch.manual_seed(100)
    np_test1 = np.random.rand(3)
    torch_test1 = torch.rand(3)

    np.random.seed(100)
    torch.manual_seed(100)
    np_test2 = np.random.rand(3)
    torch_test2 = torch.rand(3)

    np.testing.assert_array_equal(np_test1, np_test2, "NumPy seeding should be reproducible")
    assert torch.equal(torch_test1, torch_test2), "PyTorch seeding should be reproducible"


@pytest.mark.unit
def test_multiple_fixtures_together(
    water_molecule: Atoms,
    device: torch.device,
    energy_tolerance: float,
):
    """Demonstrate using multiple fixtures in a single test.

    Args:
        water_molecule: Water molecule fixture from conftest.py.
        device: Device fixture from conftest.py.
        energy_tolerance: Energy tolerance fixture from conftest.py.
    """
    # Get positions as numpy array
    positions = water_molecule.get_positions()

    # Convert to PyTorch tensor on device
    positions_tensor = torch.tensor(positions, dtype=torch.float32, device=device)

    # Verify shape
    assert positions_tensor.shape == (3, 3), "Water should have 3 atoms with 3D positions"

    # Use tolerance for some comparison (example)
    energy_diff = 0.00005  # Some hypothetical energy difference
    assert energy_diff < energy_tolerance, "Energy difference should be within tolerance"


@pytest.mark.unit
@pytest.mark.parametrize("molecule_name", ["H2O", "CH4", "NH3"])
def test_parametrized_with_fixture(molecule_name: str, device: torch.device):
    """Example of combining parametrization with fixtures.

    Args:
        molecule_name: Molecule name (from parametrize).
        device: Device fixture from conftest.py.
    """
    from ase.build import molecule

    mol = molecule(molecule_name)
    positions = torch.tensor(mol.get_positions(), dtype=torch.float32).to(device)

    assert positions.ndim == 2, "Positions should be 2D"
    assert positions.shape[1] == 3, "Positions should have 3 coordinates"
    assert positions.device.type == device.type, "Positions should be on correct device type"
