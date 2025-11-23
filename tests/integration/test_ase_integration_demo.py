"""Example integration tests demonstrating ASE integration testing.

This module shows how to write integration tests that verify components
work together correctly, particularly with ASE (Atomic Simulation Environment).
"""

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize import BFGS


@pytest.mark.integration
def test_ase_atoms_to_torch_conversion(silicon_crystal: Atoms, device: torch.device):
    """Test converting ASE Atoms to PyTorch tensors.

    Args:
        silicon_crystal: Silicon crystal fixture from conftest.py.
        device: Device fixture from conftest.py.
    """
    # Extract data from ASE Atoms
    positions = silicon_crystal.get_positions()
    atomic_numbers = silicon_crystal.get_atomic_numbers()
    cell = silicon_crystal.get_cell()

    # Convert to PyTorch tensors
    pos_tensor = torch.tensor(positions, dtype=torch.float32, device=device)
    z_tensor = torch.tensor(atomic_numbers, dtype=torch.long, device=device)
    cell_tensor = torch.tensor(cell.array, dtype=torch.float32, device=device)

    # Verify conversions
    assert pos_tensor.shape == (64, 3), "Positions should have correct shape"
    assert z_tensor.shape == (64,), "Atomic numbers should have correct shape"
    assert cell_tensor.shape == (3, 3), "Cell should be 3x3"

    # Verify device placement
    assert pos_tensor.device.type == device.type
    assert z_tensor.device.type == device.type
    assert cell_tensor.device.type == device.type

    # Verify values are preserved
    np.testing.assert_allclose(pos_tensor.cpu().numpy(), positions, rtol=1e-6)
    np.testing.assert_array_equal(z_tensor.cpu().numpy(), atomic_numbers)


@pytest.mark.integration
def test_batch_processing_multiple_molecules(small_molecule_set, device: torch.device):
    """Test batching multiple molecules together.

    Args:
        small_molecule_set: List of small molecules from conftest.py.
        device: Device fixture from conftest.py.
    """
    # Extract positions and atomic numbers from all molecules
    all_positions = []
    all_atomic_numbers = []
    batch_indices = []

    for batch_idx, mol in enumerate(small_molecule_set):
        n_atoms = len(mol)
        all_positions.append(mol.get_positions())
        all_atomic_numbers.append(mol.get_atomic_numbers())
        batch_indices.extend([batch_idx] * n_atoms)

    # Concatenate into batch format
    positions = torch.tensor(
        np.vstack(all_positions), dtype=torch.float32, device=device
    )
    atomic_numbers = torch.tensor(
        np.concatenate(all_atomic_numbers), dtype=torch.long, device=device
    )
    batch = torch.tensor(batch_indices, dtype=torch.long, device=device)

    # Verify batch structure
    assert positions.ndim == 2 and positions.shape[1] == 3
    assert atomic_numbers.ndim == 1
    assert batch.ndim == 1
    assert len(positions) == len(atomic_numbers) == len(batch)

    # Verify batch indices are correct
    assert batch.min() == 0
    assert batch.max() == len(small_molecule_set) - 1
    assert len(batch.unique()) == len(small_molecule_set)


@pytest.mark.integration
def test_periodic_boundary_conditions(nacl_crystal: Atoms):
    """Test handling of periodic boundary conditions.

    Args:
        nacl_crystal: NaCl crystal fixture from conftest.py.
    """
    # Verify PBC is set
    assert nacl_crystal.pbc.all(), "All directions should be periodic"

    # Get cell and positions
    cell = nacl_crystal.get_cell()
    positions = nacl_crystal.get_positions()

    # Test wrapping positions into cell
    wrapped = nacl_crystal.get_positions(wrap=True)
    assert wrapped.shape == positions.shape

    # Verify all wrapped positions are within the cell
    # Convert to fractional coordinates
    fractional = np.linalg.solve(cell.T, wrapped.T).T
    assert np.all(fractional >= 0.0), "Fractional coords should be >= 0"
    assert np.all(fractional < 1.0), "Fractional coords should be < 1"


@pytest.mark.integration
def test_neighbor_list_computation(silicon_crystal: Atoms):
    """Test computing neighbor lists for periodic systems.

    Args:
        silicon_crystal: Silicon crystal fixture from conftest.py.
    """
    from ase.neighborlist import neighbor_list

    # Compute neighbors within 3.0 Angstrom cutoff
    cutoff = 3.0
    i, j, d = neighbor_list("ijd", silicon_crystal, cutoff)

    # Verify we found neighbors
    assert len(i) > 0, "Should find neighbors within cutoff"
    assert len(i) == len(j) == len(d), "Neighbor arrays should have same length"

    # Verify distances are within cutoff
    assert np.all(d <= cutoff), "All distances should be within cutoff"
    assert np.all(d > 0), "Distances should be positive"

    # Each atom should have neighbors (diamond structure)
    unique_atoms = np.unique(i)
    assert len(unique_atoms) > 0, "Should have atoms with neighbors"


@pytest.mark.integration
@pytest.mark.slow
def test_energy_conservation_simple_dynamics(water_molecule: Atoms, temp_dir):
    """Test energy conservation in simple dynamics (with dummy calculator).

    This is a placeholder for testing MD integration. Real implementation
    would use actual force field calculator.

    Args:
        water_molecule: Water molecule fixture from conftest.py.
        temp_dir: Temporary directory fixture from conftest.py.
    """

    # Create a simple harmonic calculator for testing
    class HarmonicCalculator(Calculator):
        """Simple harmonic potential for testing."""

        implemented_properties = ["energy", "forces"]

        def __init__(self, k=1.0):
            Calculator.__init__(self)
            self.k = k

        def calculate(self, atoms=None, properties=["energy"], system_changes=None):
            Calculator.calculate(self, atoms, properties, system_changes)

            # Simple harmonic potential centered at origin
            positions = self.atoms.get_positions()
            r = np.linalg.norm(positions, axis=1)

            energy = 0.5 * self.k * np.sum(r**2)
            forces = -self.k * positions

            self.results = {"energy": energy, "forces": forces}

    # Attach calculator
    water_molecule.calc = HarmonicCalculator(k=0.1)

    # Get initial energy
    initial_energy = water_molecule.get_potential_energy()
    assert initial_energy is not None, "Should compute energy"

    # Get forces
    forces = water_molecule.get_forces()
    assert forces.shape == (3, 3), "Should have forces for 3 atoms"


@pytest.mark.integration
def test_force_consistency_with_energy(methane_molecule: Atoms):
    """Test that forces are consistent with energy gradients.

    This demonstrates numerical gradient checking.

    Args:
        methane_molecule: Methane molecule fixture from conftest.py.
    """

    # Create a simple calculator
    class SimpleCalculator(Calculator):
        """Simple calculator for testing force consistency."""

        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=["energy"], system_changes=None):
            Calculator.calculate(self, atoms, properties, system_changes)

            positions = self.atoms.get_positions()
            # Simple potential: sum of squared distances
            energy = np.sum(positions**2)

            # Analytical forces: F = -dE/dr = -2*r
            forces = -2 * positions

            self.results = {"energy": energy, "forces": forces}

    methane_molecule.calc = SimpleCalculator()

    # Get analytical forces
    forces_analytical = methane_molecule.get_forces()

    # Compute numerical forces
    delta = 1e-5
    forces_numerical = np.zeros_like(forces_analytical)

    for i in range(len(methane_molecule)):
        for j in range(3):
            # Forward step
            positions = methane_molecule.get_positions()
            positions[i, j] += delta
            methane_molecule.set_positions(positions)
            e_plus = methane_molecule.get_potential_energy()

            # Backward step
            positions[i, j] -= 2 * delta
            methane_molecule.set_positions(positions)
            e_minus = methane_molecule.get_potential_energy()

            # Central difference
            forces_numerical[i, j] = -(e_plus - e_minus) / (2 * delta)

            # Reset positions
            positions[i, j] += delta
            methane_molecule.set_positions(positions)

    # Compare analytical and numerical forces
    np.testing.assert_allclose(
        forces_analytical,
        forces_numerical,
        rtol=1e-4,
        atol=1e-6,
        err_msg="Analytical forces should match numerical gradients",
    )


@pytest.mark.integration
def test_stress_tensor_for_periodic_system(silicon_crystal: Atoms):
    """Test stress tensor computation for periodic systems.

    Args:
        silicon_crystal: Silicon crystal fixture from conftest.py.
    """

    # Create a simple calculator that computes stress
    class StressCalculator(Calculator):
        """Calculator that computes stress tensor."""

        implemented_properties = ["energy", "forces", "stress"]

        def calculate(self, atoms=None, properties=["energy"], system_changes=None):
            Calculator.calculate(self, atoms, properties, system_changes)

            positions = self.atoms.get_positions()
            volume = self.atoms.get_volume()

            # Simple energy and forces
            energy = np.sum(positions**2) / volume
            forces = -2 * positions / volume

            # Simplified stress tensor (should be more sophisticated in reality)
            # Stress has shape (6,) in Voigt notation: xx, yy, zz, yz, xz, xy
            stress = np.array([energy, energy, energy, 0.0, 0.0, 0.0])

            self.results = {"energy": energy, "forces": forces, "stress": stress}

    silicon_crystal.calc = StressCalculator()

    # Get stress tensor
    stress = silicon_crystal.get_stress()
    assert stress.shape == (6,), "Stress should have shape (6,) in Voigt notation"

    # Verify stress components
    assert not np.any(np.isnan(stress)), "Stress should not contain NaN"
    assert not np.any(np.isinf(stress)), "Stress should not contain Inf"


@pytest.mark.integration
def test_cell_optimization_with_stress(bcc_iron: Atoms, temp_dir):
    """Test cell optimization using stress tensor.

    This demonstrates how cell optimization would work with a calculator.

    Args:
        bcc_iron: BCC iron crystal fixture from conftest.py.
        temp_dir: Temporary directory fixture from conftest.py.
    """

    # Create calculator with stress support
    class CellCalculator(Calculator):
        """Calculator for cell optimization testing."""

        implemented_properties = ["energy", "forces", "stress"]

        def __init__(self, a0=2.87):
            Calculator.__init__(self)
            self.a0 = a0  # Equilibrium lattice constant

        def calculate(self, atoms=None, properties=["energy"], system_changes=None):
            Calculator.calculate(self, atoms, properties, system_changes)

            volume = self.atoms.get_volume()
            # Simple energy that has minimum at a0
            energy = 0.1 * (volume - self.a0**3) ** 2

            # Zero forces for simplicity
            forces = np.zeros((len(self.atoms), 3))

            # Stress proportional to volume deviation
            stress_value = 0.2 * (volume - self.a0**3)
            stress = np.array([stress_value] * 3 + [0.0] * 3)

            self.results = {"energy": energy, "forces": forces, "stress": stress}

    bcc_iron.calc = CellCalculator()

    # Get initial energy and stress
    initial_energy = bcc_iron.get_potential_energy()
    initial_stress = bcc_iron.get_stress()

    assert initial_energy is not None
    assert initial_stress.shape == (6,)

    # In a real test, we would run cell optimization here
    # from ase.optimize import BFGS
    # from ase.constraints import StrainFilter
    # opt = BFGS(StrainFilter(bcc_iron), trajectory=str(temp_dir / "opt.traj"))
    # opt.run(fmax=0.01)


@pytest.mark.integration
def test_multi_species_system(nacl_crystal: Atoms):
    """Test handling of systems with multiple species.

    Args:
        nacl_crystal: NaCl crystal fixture from conftest.py.
    """
    symbols = nacl_crystal.get_chemical_symbols()
    atomic_numbers = nacl_crystal.get_atomic_numbers()

    # NaCl should have both Na and Cl
    unique_symbols = set(symbols)
    assert "Na" in unique_symbols, "Should contain sodium"
    assert "Cl" in unique_symbols, "Should contain chlorine"

    # Check atomic numbers
    na_z = 11  # Sodium
    cl_z = 17  # Chlorine
    unique_z = set(atomic_numbers)
    assert na_z in unique_z, "Should contain Na atomic number"
    assert cl_z in unique_z, "Should contain Cl atomic number"

    # Should have equal numbers of Na and Cl
    assert symbols.count("Na") == symbols.count("Cl"), (
        "Should have equal numbers of Na and Cl"
    )


@pytest.mark.integration
def test_data_pipeline_integration(
    small_molecule_set, temp_data_dir, device: torch.device
):
    """Test integration of data loading and batching pipeline.

    Args:
        small_molecule_set: List of molecules from conftest.py.
        temp_data_dir: Temporary data directory from conftest.py.
        device: Device fixture from conftest.py.
    """
    # Simulate saving molecule data
    import json

    data_file = temp_data_dir / "molecules.json"

    molecule_data = []
    for mol in small_molecule_set:
        mol_dict = {
            "positions": mol.get_positions().tolist(),
            "atomic_numbers": mol.get_atomic_numbers().tolist(),
            "symbols": mol.get_chemical_symbols(),
        }
        molecule_data.append(mol_dict)

    # Save to file
    with open(data_file, "w") as f:
        json.dump(molecule_data, f)

    # Load from file
    with open(data_file, "r") as f:
        loaded_data = json.load(f)

    assert len(loaded_data) == len(small_molecule_set)

    # Convert back to tensors
    for i, mol_dict in enumerate(loaded_data):
        positions = torch.tensor(mol_dict["positions"], dtype=torch.float32, device=device)
        atomic_numbers = torch.tensor(
            mol_dict["atomic_numbers"], dtype=torch.long, device=device
        )

        # Verify shapes
        assert positions.ndim == 2 and positions.shape[1] == 3
        assert atomic_numbers.ndim == 1
        assert len(positions) == len(atomic_numbers)
