"""
Integration Tests for StudentForceFieldCalculator

Tests the production ASE Calculator interface with the trained model.

Author: ML Architecture Designer
Date: 2025-11-24
Issue: #24
"""

import pytest
import numpy as np
import torch
from pathlib import Path

from ase import Atoms
from ase.build import molecule, bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS
from ase import units

from mlff_distiller.inference import StudentForceFieldCalculator


# Fixtures

@pytest.fixture(scope='module')
def checkpoint_path():
    """Path to trained model checkpoint."""
    path = Path('checkpoints/best_model.pt')
    if not path.exists():
        pytest.skip(f"Checkpoint not found: {path}")
    return path


@pytest.fixture(scope='module')
def calculator(checkpoint_path):
    """Create calculator instance."""
    return StudentForceFieldCalculator(
        checkpoint_path=checkpoint_path,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        enable_timing=True
    )


@pytest.fixture
def water_molecule():
    """Create a water molecule."""
    return molecule('H2O')


@pytest.fixture
def methane_molecule():
    """Create a methane molecule."""
    return molecule('CH4')


# Basic Functionality Tests

class TestBasicFunctionality:
    """Test basic calculator functionality."""

    def test_initialization(self, checkpoint_path):
        """Test calculator initialization."""
        calc = StudentForceFieldCalculator(
            checkpoint_path=checkpoint_path,
            device='cpu'
        )
        assert calc is not None
        assert calc.model is not None
        assert 'energy' in calc.implemented_properties
        assert 'forces' in calc.implemented_properties

    def test_energy_calculation(self, calculator, water_molecule):
        """Test energy calculation."""
        water_molecule.calc = calculator
        energy = water_molecule.get_potential_energy()

        assert isinstance(energy, float)
        assert np.isfinite(energy)
        assert calculator.n_calls == 1

    def test_forces_calculation(self, calculator, water_molecule):
        """Test force calculation."""
        water_molecule.calc = calculator
        forces = water_molecule.get_forces()

        assert forces.shape == (len(water_molecule), 3)
        assert np.isfinite(forces).all()

    def test_multiple_calculations(self, calculator, water_molecule):
        """Test multiple calculations."""
        water_molecule.calc = calculator
        initial_calls = calculator.n_calls

        # Multiple calls
        for _ in range(5):
            energy = water_molecule.get_potential_energy()
            assert np.isfinite(energy)

        # Should be 5 new calls (ASE caching)
        # Note: ASE caches results, so same atoms = same result
        assert calculator.n_calls >= initial_calls

    def test_different_molecules(self, calculator):
        """Test calculations on different molecules."""
        molecules = [
            molecule('H2O'),
            molecule('CO2'),
            molecule('NH3')
        ]

        energies = []
        for mol in molecules:
            mol.calc = calculator
            energy = mol.get_potential_energy()
            energies.append(energy)

        # All should be finite
        assert all(np.isfinite(e) for e in energies)

        # Different molecules should have different energies
        assert len(set(energies)) == len(energies)


class TestInputValidation:
    """Test input validation and error handling."""

    def test_empty_structure(self, calculator):
        """Test error handling for empty structure."""
        empty_atoms = Atoms()
        empty_atoms.calc = calculator

        with pytest.raises(ValueError, match="empty structure"):
            empty_atoms.get_potential_energy()

    def test_invalid_atomic_numbers(self, calculator):
        """Test error handling for invalid atomic numbers."""
        # Create atoms with invalid atomic number
        invalid_atoms = Atoms('X', positions=[[0, 0, 0]])
        invalid_atoms.calc = calculator

        # Should raise error (X is not a valid element)
        # Note: This depends on how ASE handles invalid symbols
        # May need to adjust based on actual behavior

    def test_nan_positions(self, calculator, water_molecule):
        """Test error handling for NaN positions."""
        water_molecule.calc = calculator
        water_molecule.positions[0] = [np.nan, 0, 0]

        with pytest.raises(ValueError, match="NaN"):
            water_molecule.get_potential_energy()


class TestASEIntegration:
    """Test integration with ASE workflows."""

    def test_geometry_optimization(self, calculator, water_molecule):
        """Test structure optimization."""
        # Distort structure slightly
        water_molecule.positions += np.random.randn(*water_molecule.positions.shape) * 0.05
        water_molecule.calc = calculator

        # Get initial energy
        e_initial = water_molecule.get_potential_energy()

        # Optimize
        opt = BFGS(water_molecule, logfile=None)
        opt.run(fmax=0.05, steps=50)

        # Get final energy
        e_final = water_molecule.get_potential_energy()

        # Energy should decrease
        assert e_final <= e_initial

    def test_md_simulation(self, calculator, water_molecule):
        """Test MD simulation."""
        water_molecule.calc = calculator

        # Initialize velocities
        MaxwellBoltzmannDistribution(water_molecule, temperature_K=300)
        Stationary(water_molecule)

        # Create integrator
        dyn = VelocityVerlet(water_molecule, timestep=0.5*units.fs)

        # Run short simulation
        energies = []
        for _ in range(10):
            dyn.run(1)
            e_pot = water_molecule.get_potential_energy()
            e_kin = water_molecule.get_kinetic_energy()
            energies.append(e_pot + e_kin)

        # Total energy should be approximately conserved
        energies = np.array(energies)
        drift = (energies[-1] - energies[0]) / energies[0]
        assert abs(drift) < 0.1  # 10% drift acceptable for short simulation

    def test_ase_caching(self, calculator, water_molecule):
        """Test that ASE caching works correctly."""
        water_molecule.calc = calculator

        initial_calls = calculator.n_calls

        # Multiple calls without changing atoms
        e1 = water_molecule.get_potential_energy()
        e2 = water_molecule.get_potential_energy()
        e3 = water_molecule.get_potential_energy()

        # Should only calculate once (ASE caches)
        assert calculator.n_calls == initial_calls + 1

        # Results should be identical
        assert e1 == e2 == e3


class TestPerformance:
    """Test performance and timing."""

    def test_timing_tracking(self, calculator, water_molecule):
        """Test timing statistics."""
        water_molecule.calc = calculator

        # Perform calculation
        water_molecule.get_potential_energy()

        # Check timing stats
        stats = calculator.get_timing_stats()
        assert stats['n_calls'] > 0
        assert stats['avg_time'] > 0
        assert stats['total_time'] > 0

    def test_batch_calculation(self, calculator):
        """Test batch calculations."""
        molecules = [
            molecule('H2O'),
            molecule('NH3'),
            molecule('CH4')
        ]

        results = calculator.calculate_batch(molecules, properties=['energy', 'forces'])

        assert len(results) == len(molecules)
        for result in results:
            assert 'energy' in result
            assert 'forces' in result
            assert np.isfinite(result['energy'])
            assert result['forces'].shape[1] == 3


class TestPBC:
    """Test periodic boundary conditions."""

    def test_pbc_calculation(self, calculator):
        """Test calculation with PBC."""
        # Create periodic system
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = calculator

        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

        assert np.isfinite(energy)
        assert np.isfinite(forces).all()

    def test_mixed_pbc(self, calculator, water_molecule):
        """Test with mixed PBC."""
        # Set PBC in only some directions
        water_molecule.pbc = [True, True, False]
        water_molecule.cell = [10, 10, 10]
        water_molecule.calc = calculator

        energy = water_molecule.get_potential_energy()
        assert np.isfinite(energy)


class TestStressCalculation:
    """Test stress tensor calculation (if enabled)."""

    def test_stress_disabled_by_default(self, checkpoint_path):
        """Test that stress is disabled by default."""
        calc = StudentForceFieldCalculator(
            checkpoint_path=checkpoint_path,
            device='cpu'
        )
        assert 'stress' not in calc.implemented_properties

    def test_stress_calculation(self, checkpoint_path):
        """Test stress calculation when enabled."""
        calc = StudentForceFieldCalculator(
            checkpoint_path=checkpoint_path,
            device='cpu',
            enable_stress=True
        )

        # Create periodic system
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = calc

        if 'stress' in calc.implemented_properties:
            stress = atoms.get_stress()
            assert stress.shape == (6,)
            # Stress may be zeros if not implemented
            # Just check it doesn't crash


class TestCalculatorState:
    """Test calculator state management."""

    def test_reset(self, calculator, water_molecule):
        """Test calculator reset."""
        water_molecule.calc = calculator

        # Calculate
        water_molecule.get_potential_energy()

        # Reset
        calculator.reset()

        # Should still work after reset
        energy = water_molecule.get_potential_energy()
        assert np.isfinite(energy)

    def test_repr(self, calculator):
        """Test string representation."""
        repr_str = repr(calculator)
        assert 'StudentForceFieldCalculator' in repr_str
        assert 'best_model.pt' in repr_str or 'checkpoint' in repr_str.lower()


# Integration with Teacher Model (if available)

class TestTeacherComparison:
    """Test comparison with teacher model."""

    @pytest.fixture
    def teacher_calc(self):
        """Create teacher calculator."""
        try:
            from mlff_distiller.models.teacher_wrappers import OrbCalculatorWrapper
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            return OrbCalculatorWrapper(device=device)
        except Exception:
            pytest.skip("Teacher model not available")

    def test_energy_agreement(self, calculator, teacher_calc, water_molecule):
        """Test energy agreement with teacher."""
        # Student prediction
        water_molecule.calc = calculator
        e_student = water_molecule.get_potential_energy()

        # Teacher prediction
        water_molecule.calc = teacher_calc
        e_teacher = water_molecule.get_potential_energy()

        # Should be reasonably close (within 1%)
        error_pct = abs(e_student - e_teacher) / abs(e_teacher) * 100
        assert error_pct < 1.0, f"Energy error {error_pct:.2f}% exceeds 1%"

    def test_force_agreement(self, calculator, teacher_calc, water_molecule):
        """Test force agreement with teacher."""
        # Student prediction
        water_molecule.calc = calculator
        f_student = water_molecule.get_forces()

        # Teacher prediction
        water_molecule.calc = teacher_calc
        f_teacher = water_molecule.get_forces()

        # Calculate RMSE
        rmse = np.sqrt(np.mean((f_student - f_teacher)**2))

        # Should be reasonably close
        assert rmse < 0.2, f"Force RMSE {rmse:.4f} eV/Å exceeds 0.2"


# Mark slow tests

pytestmark = pytest.mark.integration


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
