"""
Integration tests for StudentCalculator.

Tests StudentCalculator in realistic usage scenarios including:
1. MD simulations (NVE, NVT)
2. Drop-in replacement for teacher calculators
3. Batch processing multiple structures
4. Long-running simulations (memory stability)

Author: ML Architecture Designer
Date: 2025-11-23
"""

import pytest
import numpy as np
import torch
from ase import Atoms, units
from ase.build import bulk, molecule
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from mlff_distiller.models import StudentCalculator, MockStudentModel, SimpleMLP


class TestStudentCalculatorMD:
    """Test StudentCalculator in MD simulations."""

    def test_nve_simulation(self):
        """Test NVE MD simulation completes without errors."""
        # Create system
        atoms = molecule("H2O")
        atoms.set_cell([10, 10, 10])
        atoms.center()

        # Create calculator
        model = MockStudentModel()
        calc = StudentCalculator(model=model, device="cpu")
        atoms.calc = calc

        # Set initial velocities
        MaxwellBoltzmannDistribution(atoms, temperature_K=300)

        # Run NVE
        dyn = VelocityVerlet(atoms, timestep=1.0 * units.fs)
        dyn.run(50)

        # Verify completed
        assert calc.n_calls >= 50

    def test_nvt_simulation(self):
        """Test NVT MD simulation with Langevin thermostat."""
        # Create system
        atoms = molecule("CH4")
        atoms.set_cell([10, 10, 10])
        atoms.center()

        # Create calculator
        model = MockStudentModel()
        calc = StudentCalculator(model=model, device="cpu")
        atoms.calc = calc

        # Set initial velocities
        MaxwellBoltzmannDistribution(atoms, temperature_K=300)

        # Run NVT
        dyn = Langevin(
            atoms,
            timestep=1.0 * units.fs,
            temperature_K=300,
            friction=0.01,
        )
        dyn.run(50)

        # Verify completed
        assert calc.n_calls >= 50

    def test_periodic_system_md(self):
        """Test MD with periodic boundary conditions."""
        # Create periodic system
        atoms = bulk("Si", "diamond", a=5.43, cubic=True)

        # Create calculator
        model = MockStudentModel()
        calc = StudentCalculator(model=model, device="cpu")
        atoms.calc = calc

        # Set initial velocities
        MaxwellBoltzmannDistribution(atoms, temperature_K=300)

        # Run MD
        dyn = VelocityVerlet(atoms, timestep=1.0 * units.fs)
        dyn.run(50)

        # Verify stress is calculated (periodic system)
        stress = atoms.get_stress()
        assert stress is not None
        assert len(stress) == 6

    def test_long_simulation_memory_stable(self):
        """Test that long MD runs don't leak memory."""
        atoms = molecule("H2O")
        atoms.set_cell([10, 10, 10])
        atoms.center()

        model = MockStudentModel()
        calc = StudentCalculator(model=model, device="cpu")
        atoms.calc = calc

        MaxwellBoltzmannDistribution(atoms, temperature_K=300)

        # Run longer simulation
        dyn = VelocityVerlet(atoms, timestep=1.0 * units.fs)
        dyn.run(500)

        # If we got here without OOM, test passes
        assert calc.n_calls >= 500


class TestStudentCalculatorBatchProcessing:
    """Test batch processing multiple structures."""

    def test_multiple_structures(self):
        """Test processing multiple structures with same calculator."""
        # Create calculator once
        model = MockStudentModel()
        calc = StudentCalculator(model=model, device="cpu")

        # Create multiple structures
        structures = [
            molecule("H2O"),
            molecule("CH4"),
            molecule("NH3"),
            bulk("Si", "diamond", a=5.43),
        ]

        # Process each
        energies = []
        for atoms in structures:
            atoms.calc = calc
            energy = atoms.get_potential_energy()
            energies.append(energy)

        # Verify all calculated
        assert len(energies) == len(structures)
        assert all(isinstance(e, (float, np.floating)) for e in energies)
        assert calc.n_calls == len(structures)

    def test_variable_system_sizes(self):
        """Test calculator handles variable system sizes."""
        calc = StudentCalculator(model=MockStudentModel(), device="cpu")

        # Different sizes
        sizes = [
            Atoms("H", positions=[[0, 0, 0]]),  # 1 atom
            molecule("H2O"),  # 3 atoms
            molecule("CH4"),  # 5 atoms
            bulk("Si", "diamond", a=5.43),  # 8 atoms
            bulk("Cu", "fcc", a=3.58).repeat((2, 2, 2)),  # 32 atoms
        ]

        for atoms in sizes:
            atoms.calc = calc
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()

            assert isinstance(energy, (float, np.floating))
            assert forces.shape == (len(atoms), 3)


class TestStudentCalculatorDropInReplacement:
    """Test drop-in replacement compatibility."""

    def test_same_interface_as_ase_calculator(self):
        """Test StudentCalculator has same interface as ASE Calculator."""
        from ase.calculators.calculator import Calculator

        calc = StudentCalculator(model=MockStudentModel(), device="cpu")

        # Check inheritance
        assert isinstance(calc, Calculator)

        # Check methods
        assert hasattr(calc, "calculate")
        assert hasattr(calc, "get_potential_energy")
        assert hasattr(calc, "get_forces")
        assert hasattr(calc, "get_stress")

    def test_works_with_ase_atoms(self):
        """Test calculator works with ASE Atoms methods."""
        calc = StudentCalculator(model=MockStudentModel(), device="cpu")
        atoms = molecule("H2O")
        atoms.calc = calc

        # All standard ASE methods should work
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

        assert isinstance(energy, (float, np.floating))
        assert isinstance(forces, np.ndarray)
        assert forces.shape == (len(atoms), 3)

    def test_works_with_md_integrators(self):
        """Test calculator works with various MD integrators."""
        calc = StudentCalculator(model=MockStudentModel(), device="cpu")
        atoms = molecule("H2O")
        atoms.set_cell([10, 10, 10])
        atoms.center()
        atoms.calc = calc

        MaxwellBoltzmannDistribution(atoms, temperature_K=300)

        # Test VelocityVerlet
        dyn1 = VelocityVerlet(atoms, timestep=1.0 * units.fs)
        dyn1.run(10)

        # Test Langevin
        dyn2 = Langevin(
            atoms, timestep=1.0 * units.fs, temperature_K=300, friction=0.01
        )
        dyn2.run(10)

        # Both should work
        assert calc.n_calls >= 20


class TestStudentCalculatorEdgeCases:
    """Test edge cases and error handling."""

    def test_single_atom(self):
        """Test single atom system."""
        calc = StudentCalculator(model=MockStudentModel(), device="cpu")
        atoms = Atoms("H", positions=[[0, 0, 0]])
        atoms.calc = calc

        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

        assert isinstance(energy, (float, np.floating))
        assert forces.shape == (1, 3)

    def test_large_system(self):
        """Test large system (64 atoms)."""
        calc = StudentCalculator(model=MockStudentModel(), device="cpu")
        atoms = bulk("Si", "diamond", a=5.43).repeat((2, 2, 2))
        atoms.calc = calc

        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

        assert isinstance(energy, (float, np.floating))
        assert forces.shape == (len(atoms), 3)

    def test_repeated_calculations_cache(self):
        """Test that ASE caching works correctly."""
        calc = StudentCalculator(model=MockStudentModel(), device="cpu")
        atoms = molecule("H2O")
        atoms.calc = calc

        # First calculation
        energy1 = atoms.get_potential_energy()
        n_calls_1 = calc.n_calls

        # Second calculation (should use cache)
        energy2 = atoms.get_potential_energy()
        n_calls_2 = calc.n_calls

        # Should be cached (same number of calls)
        assert energy1 == energy2
        assert n_calls_1 == n_calls_2

        # Modify atoms to invalidate cache
        atoms.positions[0] += 0.1

        # Third calculation (should recalculate)
        energy3 = atoms.get_potential_energy()
        n_calls_3 = calc.n_calls

        # Should have made new calculation
        assert n_calls_3 > n_calls_2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestStudentCalculatorCUDA:
    """Test StudentCalculator on CUDA device."""

    def test_cuda_calculation(self):
        """Test calculation on CUDA device."""
        calc = StudentCalculator(model=MockStudentModel(), device="cuda")
        atoms = molecule("H2O")
        atoms.calc = calc

        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

        assert isinstance(energy, (float, np.floating))
        assert forces.shape == (3, 3)

    def test_cuda_md_simulation(self):
        """Test MD simulation on CUDA."""
        calc = StudentCalculator(model=MockStudentModel(), device="cuda")
        atoms = molecule("H2O")
        atoms.set_cell([10, 10, 10])
        atoms.center()
        atoms.calc = calc

        MaxwellBoltzmannDistribution(atoms, temperature_K=300)

        dyn = VelocityVerlet(atoms, timestep=1.0 * units.fs)
        dyn.run(50)

        assert calc.n_calls >= 50


class TestStudentCalculatorWithSimpleMLP:
    """Test StudentCalculator with SimpleMLP model."""

    def test_simple_mlp_calculation(self):
        """Test SimpleMLP model works with calculator."""
        model = SimpleMLP(hidden_dim=32, num_layers=2)
        calc = StudentCalculator(model=model, device="cpu")

        atoms = molecule("H2O")
        atoms.calc = calc

        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

        assert isinstance(energy, (float, np.floating))
        assert forces.shape == (3, 3)

    def test_simple_mlp_md(self):
        """Test SimpleMLP in MD simulation."""
        model = SimpleMLP(hidden_dim=32, num_layers=2)
        calc = StudentCalculator(model=model, device="cpu")

        atoms = molecule("H2O")
        atoms.set_cell([10, 10, 10])
        atoms.center()
        atoms.calc = calc

        MaxwellBoltzmannDistribution(atoms, temperature_K=300)

        dyn = VelocityVerlet(atoms, timestep=1.0 * units.fs)
        dyn.run(50)

        assert calc.n_calls >= 50
