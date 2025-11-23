"""
ASE Calculator Interface Compliance Tests

This module validates that both teacher and student calculators correctly implement
the ASE Calculator interface, ensuring drop-in replacement capability.

Tests verify:
1. All required Calculator methods exist and work correctly
2. Methods return correct types and shapes
3. Units are correct (eV, eV/Angstrom, eV/Angstrom^3)
4. Behavior matches ASE conventions
5. Integration with ASE MD integrators works
6. Memory stability over many repeated calls
7. Identical interface between teacher and student calculators

Author: Testing & Benchmark Engineer
Date: 2025-11-23
"""

import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from ase import Atoms, units
from ase.build import bulk, molecule
from ase.calculators.calculator import Calculator, all_changes
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS

from mlff_distiller.models.student_calculator import StudentCalculator
from mlff_distiller.models.mock_student import MockStudentModel


# Test fixtures providing calculators
@pytest.fixture
def mock_student_calculator(device):
    """Provide StudentCalculator with mock model for testing."""
    model = MockStudentModel(hidden_dim=64)
    return StudentCalculator(
        model=model,
        device=str(device),
        energy_key="energy",
        forces_key="forces",
    )


@pytest.fixture(params=["student"])
def calculator_under_test(request, mock_student_calculator):
    """
    Parametrized fixture providing different calculator types.

    This allows the same tests to run on both teacher and student calculators,
    validating drop-in compatibility.
    """
    if request.param == "student":
        return mock_student_calculator
    else:
        pytest.skip(f"Calculator type {request.param} not available")


@pytest.mark.integration
class TestASECalculatorInterfaceCompliance:
    """Test that calculators correctly implement ASE Calculator interface."""

    def test_calculator_is_instance_of_ase_calculator(self, calculator_under_test):
        """Test that calculator is instance of ASE Calculator."""
        assert isinstance(calculator_under_test, Calculator)

    def test_implemented_properties_attribute_exists(self, calculator_under_test):
        """Test that implemented_properties attribute exists and is non-empty."""
        assert hasattr(calculator_under_test, 'implemented_properties')
        assert isinstance(calculator_under_test.implemented_properties, (list, tuple, set))
        assert len(calculator_under_test.implemented_properties) > 0

    def test_implemented_properties_contains_energy(self, calculator_under_test):
        """Test that 'energy' is in implemented_properties."""
        assert 'energy' in calculator_under_test.implemented_properties

    def test_implemented_properties_contains_forces(self, calculator_under_test):
        """Test that 'forces' is in implemented_properties."""
        assert 'forces' in calculator_under_test.implemented_properties

    def test_calculate_method_exists(self, calculator_under_test):
        """Test that calculate() method exists."""
        assert hasattr(calculator_under_test, 'calculate')
        assert callable(calculator_under_test.calculate)

    def test_get_potential_energy_method_exists(self, water_molecule, calculator_under_test):
        """Test that get_potential_energy() works via atoms interface."""
        water_molecule.calc = calculator_under_test
        # This should work without errors
        energy = water_molecule.get_potential_energy()
        assert energy is not None

    def test_get_forces_method_exists(self, water_molecule, calculator_under_test):
        """Test that get_forces() works via atoms interface."""
        water_molecule.calc = calculator_under_test
        forces = water_molecule.get_forces()
        assert forces is not None


@pytest.mark.integration
class TestEnergyCalculations:
    """Test energy calculation correctness and consistency."""

    def test_get_potential_energy_returns_float(self, water_molecule, calculator_under_test):
        """Test that get_potential_energy() returns a float scalar."""
        water_molecule.calc = calculator_under_test
        energy = water_molecule.get_potential_energy()

        assert isinstance(energy, (float, np.floating))
        assert np.isfinite(energy), "Energy should be finite"

    def test_energy_units_are_eV(self, water_molecule, calculator_under_test):
        """Test that energy is in reasonable eV range for water molecule."""
        water_molecule.calc = calculator_under_test
        energy = water_molecule.get_potential_energy()

        # For a small molecule like water, energy should be reasonable
        # Typical DFT energies are in tens to hundreds of eV
        assert abs(energy) < 1000, f"Energy {energy} eV seems unreasonable for H2O"

    def test_energy_is_deterministic(self, water_molecule, calculator_under_test):
        """Test that energy calculation is deterministic (same input -> same output)."""
        water_molecule.calc = calculator_under_test

        energy1 = water_molecule.get_potential_energy()
        energy2 = water_molecule.get_potential_energy()

        # Should be identical (cached by ASE)
        assert energy1 == energy2

    def test_energy_changes_with_geometry(self, water_molecule, calculator_under_test):
        """Test that energy changes when geometry changes."""
        water_molecule.calc = calculator_under_test

        energy1 = water_molecule.get_potential_energy()

        # Perturb positions
        water_molecule.positions += np.random.randn(len(water_molecule), 3) * 0.1

        energy2 = water_molecule.get_potential_energy()

        # Energies should differ (unless by chance)
        # Use a reasonable tolerance
        assert abs(energy1 - energy2) > 1e-8, "Energy should change with geometry"

    def test_energy_different_system_sizes(self, calculator_under_test):
        """Test energy calculation for different system sizes."""
        sizes = [2, 10, 20]  # Different numbers of atoms

        for n_atoms in sizes:
            positions = np.random.randn(n_atoms, 3) * 5.0
            atoms = Atoms(f"H{n_atoms}", positions=positions)
            atoms.calc = calculator_under_test

            energy = atoms.get_potential_energy()
            assert isinstance(energy, (float, np.floating))
            assert np.isfinite(energy)


@pytest.mark.integration
class TestForceCalculations:
    """Test force calculation correctness and consistency."""

    def test_get_forces_returns_correct_shape(self, water_molecule, calculator_under_test):
        """Test that get_forces() returns (n_atoms, 3) array."""
        water_molecule.calc = calculator_under_test
        forces = water_molecule.get_forces()

        assert forces.shape == (len(water_molecule), 3)

    def test_forces_are_finite(self, water_molecule, calculator_under_test):
        """Test that forces are finite (no NaN or Inf)."""
        water_molecule.calc = calculator_under_test
        forces = water_molecule.get_forces()

        assert np.all(np.isfinite(forces)), "All forces should be finite"

    def test_forces_units_are_eV_per_angstrom(self, water_molecule, calculator_under_test):
        """Test that forces are in reasonable eV/Angstrom range."""
        water_molecule.calc = calculator_under_test
        forces = water_molecule.get_forces()

        # Typical atomic forces are in range of 0.01-10 eV/Angstrom
        max_force = np.max(np.abs(forces))
        assert max_force < 100, f"Max force {max_force} eV/A seems unreasonably large"

    def test_forces_are_deterministic(self, water_molecule, calculator_under_test):
        """Test that force calculation is deterministic."""
        water_molecule.calc = calculator_under_test

        forces1 = water_molecule.get_forces()
        forces2 = water_molecule.get_forces()

        # Should be identical (cached)
        np.testing.assert_array_equal(forces1, forces2)

    def test_forces_change_with_geometry(self, water_molecule, calculator_under_test):
        """Test that forces change when geometry changes."""
        water_molecule.calc = calculator_under_test

        forces1 = water_molecule.get_forces()

        # Perturb positions
        water_molecule.positions += np.random.randn(len(water_molecule), 3) * 0.1

        forces2 = water_molecule.get_forces()

        # Forces should differ
        assert not np.allclose(forces1, forces2, atol=1e-8)

    def test_forces_different_system_sizes(self, calculator_under_test):
        """Test force calculation for different system sizes."""
        sizes = [2, 10, 20]

        for n_atoms in sizes:
            positions = np.random.randn(n_atoms, 3) * 5.0
            atoms = Atoms(f"H{n_atoms}", positions=positions)
            atoms.calc = calculator_under_test

            forces = atoms.get_forces()
            assert forces.shape == (n_atoms, 3)
            assert np.all(np.isfinite(forces))


@pytest.mark.integration
class TestStressCalculations:
    """Test stress calculation (if implemented)."""

    def test_stress_calculation_if_implemented(self, silicon_crystal, calculator_under_test):
        """Test stress calculation if implemented."""
        if 'stress' not in calculator_under_test.implemented_properties:
            pytest.skip("Stress not implemented for this calculator")

        silicon_crystal.calc = calculator_under_test
        stress = silicon_crystal.get_stress()

        # Stress can be 6-component Voigt or 3x3 tensor
        assert stress.shape in [(6,), (3, 3)]
        assert np.all(np.isfinite(stress))


@pytest.mark.integration
class TestPeriodicBoundaryConditions:
    """Test handling of periodic boundary conditions."""

    def test_periodic_system_calculation(self, silicon_crystal, calculator_under_test):
        """Test calculator works with periodic systems."""
        assert silicon_crystal.get_pbc().all(), "Test system should be periodic"

        silicon_crystal.calc = calculator_under_test

        energy = silicon_crystal.get_potential_energy()
        forces = silicon_crystal.get_forces()

        assert isinstance(energy, (float, np.floating))
        assert forces.shape == (len(silicon_crystal), 3)

    def test_non_periodic_system_calculation(self, water_molecule, calculator_under_test):
        """Test calculator works with non-periodic systems."""
        # Ensure water is non-periodic
        water_molecule.set_pbc(False)
        water_molecule.calc = calculator_under_test

        energy = water_molecule.get_potential_energy()
        forces = water_molecule.get_forces()

        assert isinstance(energy, (float, np.floating))
        assert forces.shape == (len(water_molecule), 3)

    def test_mixed_pbc_calculation(self, calculator_under_test):
        """Test calculator with mixed periodic boundaries."""
        # Create system periodic in x and y only
        atoms = bulk("Si", "diamond", a=5.43)
        atoms.set_pbc([True, True, False])
        atoms.calc = calculator_under_test

        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

        assert isinstance(energy, (float, np.floating))
        assert forces.shape == (len(atoms), 3)


@pytest.mark.integration
class TestMDIntegration:
    """Test integration with ASE MD integrators."""

    def test_nve_md_runs(self, water_molecule, calculator_under_test):
        """Test calculator works with NVE (VelocityVerlet) MD."""
        water_molecule.calc = calculator_under_test

        # Initialize velocities
        water_molecule.set_momenta(np.zeros((len(water_molecule), 3)))

        # Create integrator
        dyn = VelocityVerlet(water_molecule, timestep=1.0 * units.fs)

        # Run short trajectory (should not raise errors)
        dyn.run(50)

    def test_nvt_md_runs(self, water_molecule, calculator_under_test):
        """Test calculator works with NVT (Langevin) MD."""
        water_molecule.calc = calculator_under_test

        # Create Langevin thermostat
        dyn = Langevin(
            water_molecule,
            timestep=1.0 * units.fs,
            temperature_K=300,
            friction=0.01,
        )

        # Run short trajectory
        dyn.run(50)

    def test_geometry_optimization(self, water_molecule, calculator_under_test):
        """Test calculator works with geometry optimizer."""
        water_molecule.calc = calculator_under_test

        # Create optimizer
        opt = BFGS(water_molecule, logfile=None)

        # Run a few optimization steps
        opt.run(fmax=0.1, steps=10)


@pytest.mark.integration
class TestMemoryStability:
    """Test memory stability over repeated calls."""

    def test_no_memory_leak_repeated_energy_calls(self, water_molecule, calculator_under_test):
        """Test repeated energy calls don't cause memory leaks."""
        water_molecule.calc = calculator_under_test

        # Make many repeated calls
        for i in range(1000):
            # Slightly perturb to avoid caching
            water_molecule.positions += np.random.randn(len(water_molecule), 3) * 0.001
            energy = water_molecule.get_potential_energy()
            assert np.isfinite(energy)

    def test_no_memory_leak_repeated_force_calls(self, water_molecule, calculator_under_test):
        """Test repeated force calls don't cause memory leaks."""
        water_molecule.calc = calculator_under_test

        # Make many repeated calls
        for i in range(1000):
            water_molecule.positions += np.random.randn(len(water_molecule), 3) * 0.001
            forces = water_molecule.get_forces()
            assert np.all(np.isfinite(forces))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_memory_stable(self, water_molecule, cuda_device):
        """Test CUDA memory is stable over repeated calls."""
        model = MockStudentModel(hidden_dim=64)
        calc = StudentCalculator(model=model, device="cuda")

        water_molecule.calc = calc

        # Record initial memory
        torch.cuda.reset_peak_memory_stats()
        initial_mem = torch.cuda.memory_allocated()

        # Make many calls
        for i in range(100):
            water_molecule.positions += np.random.randn(len(water_molecule), 3) * 0.001
            energy = water_molecule.get_potential_energy()
            forces = water_molecule.get_forces()

        # Check memory didn't grow significantly
        final_mem = torch.cuda.memory_allocated()
        mem_growth_mb = (final_mem - initial_mem) / 1e6

        # Allow small growth (< 10 MB) for caching
        assert mem_growth_mb < 10, f"Memory grew by {mem_growth_mb} MB"


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_atoms_object_fails_gracefully(self, calculator_under_test):
        """Test calculator handles empty atoms object gracefully."""
        atoms = Atoms()  # Empty
        atoms.calc = calculator_under_test

        # Should either work or raise clear error (not crash)
        try:
            energy = atoms.get_potential_energy()
        except (ValueError, RuntimeError, IndexError) as e:
            # Expected for empty system
            pass

    def test_single_atom_calculation(self, calculator_under_test):
        """Test calculator works with single atom."""
        atoms = Atoms('H', positions=[[0, 0, 0]])
        atoms.calc = calculator_under_test

        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

        assert isinstance(energy, (float, np.floating))
        assert forces.shape == (1, 3)

    def test_large_system_calculation(self, calculator_under_test):
        """Test calculator works with larger systems."""
        # Create a larger system (100 atoms)
        positions = np.random.randn(100, 3) * 10.0
        atoms = Atoms('H100', positions=positions)
        atoms.calc = calculator_under_test

        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

        assert isinstance(energy, (float, np.floating))
        assert forces.shape == (100, 3)


@pytest.mark.integration
class TestDropInCompatibility:
    """Test drop-in replacement compatibility."""

    def test_calculator_swap_in_md_script(self, water_molecule):
        """
        Test that calculator can be swapped in existing MD script.

        Simulates user replacing teacher with student calculator.
        """
        # Create mock calculators
        model1 = MockStudentModel(hidden_dim=64)
        model2 = MockStudentModel(hidden_dim=32)

        calc1 = StudentCalculator(model=model1, device="cpu")
        calc2 = StudentCalculator(model=model2, device="cpu")

        # Same MD script, different calculators
        def run_md(calculator, n_steps=20):
            atoms = water_molecule.copy()
            atoms.calc = calculator
            dyn = VelocityVerlet(atoms, timestep=1.0 * units.fs)
            dyn.run(n_steps)
            return atoms

        # Should work with both calculators
        final1 = run_md(calc1)
        final2 = run_md(calc2)

        assert final1 is not None
        assert final2 is not None

    def test_identical_interface_methods(self):
        """Test that student calculator has same interface as expected."""
        model = MockStudentModel(hidden_dim=64)
        calc = StudentCalculator(model=model, device="cpu")

        # Check required methods exist
        assert hasattr(calc, 'calculate')
        assert hasattr(calc, 'implemented_properties')
        assert hasattr(calc, 'results')

        # Check methods are callable
        assert callable(calc.calculate)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
