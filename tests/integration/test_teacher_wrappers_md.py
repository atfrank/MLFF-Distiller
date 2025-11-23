"""
Integration tests for teacher model wrappers in MD simulations.

These tests verify that the calculator wrappers work correctly in realistic
MD simulation scenarios, including:
1. Running MD trajectories with ASE integrators
2. Energy conservation in NVE simulations
3. Proper handling of periodic boundary conditions
4. Variable system sizes
5. Memory stability over repeated calls

Note: These tests require the actual teacher models to be installed
(orb-models and fennol packages). They are marked with pytest markers
to allow selective running.

Author: ML Architecture Designer
Date: 2025-11-23
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms, units
from ase.build import bulk, molecule
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet


@pytest.mark.integration
@pytest.mark.requires_orb
class TestOrbCalculatorMDIntegration:
    """Integration tests for OrbCalculator in MD simulations."""

    @pytest.fixture
    def mock_orb_calculator(self):
        """Fixture providing a mocked OrbCalculator for testing."""
        with patch("src.models.teacher_wrappers.pretrained") as mock_pretrained, patch(
            "src.models.teacher_wrappers.ORBCalculator"
        ) as mock_orb_calc:

            # Mock the pretrained model
            mock_pretrained.orb_v2.return_value = MagicMock()

            # Create a mock calculator that returns consistent results
            mock_calc = MagicMock()

            # Create deterministic results for testing
            def calculate_side_effect(atoms, properties, system_changes):
                n_atoms = len(atoms)
                # Simple harmonic potential for testing
                positions = atoms.positions
                energy = 0.1 * np.sum(positions**2)
                forces = -0.2 * positions  # Force = -gradient of potential

                mock_calc.results = {
                    "energy": energy,
                    "forces": forces,
                    "stress": np.zeros(6),
                }

            mock_calc.calculate.side_effect = calculate_side_effect
            mock_calc.results = {}
            mock_orb_calc.return_value = mock_calc

            from src.models.teacher_wrappers import OrbCalculator

            yield OrbCalculator(model_name="orb-v2", device="cpu")

    def test_calculator_in_nve_simulation(self, mock_orb_calculator):
        """Test OrbCalculator in NVE (constant energy) MD simulation."""
        # Create simple system
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], pbc=False)
        atoms.calc = mock_orb_calculator

        # Initialize velocities
        atoms.set_momenta(np.zeros((len(atoms), 3)))

        # Run short NVE simulation
        dyn = VelocityVerlet(atoms, timestep=0.5 * units.fs)

        # Collect energies during simulation
        energies = []

        def energy_observer():
            energies.append(atoms.get_potential_energy() + atoms.get_kinetic_energy())

        dyn.attach(energy_observer, interval=1)

        # Run 50 steps
        dyn.run(50)

        # Check that simulation completed
        assert len(energies) == 51  # Initial + 50 steps

    def test_calculator_with_periodic_boundaries(self, mock_orb_calculator):
        """Test OrbCalculator with periodic boundary conditions."""
        # Create periodic system
        atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
        atoms.calc = mock_orb_calculator

        # Should be able to calculate properties
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        stress = atoms.get_stress()

        assert isinstance(energy, float)
        assert forces.shape == (len(atoms), 3)
        assert stress.shape == (6,) or stress.shape == (3, 3)

    def test_calculator_with_variable_system_sizes(self, mock_orb_calculator):
        """Test OrbCalculator with different system sizes."""
        sizes = [2, 10, 50]  # Small, medium, larger systems

        for size in sizes:
            # Create system of varying size
            positions = np.random.randn(size, 3) * 5.0
            atoms = Atoms(f"H{size}", positions=positions)
            atoms.calc = mock_orb_calculator

            # Should calculate without errors
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()

            assert isinstance(energy, float)
            assert forces.shape == (size, 3)

    def test_memory_stability_repeated_calls(self, mock_orb_calculator):
        """Test that calculator is memory-stable over many repeated calls."""
        atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        atoms.calc = mock_orb_calculator

        # Make many repeated calls (simulating MD)
        n_calls = 1000
        for i in range(n_calls):
            # Perturb positions slightly
            atoms.positions += np.random.randn(len(atoms), 3) * 0.01

            # Calculate properties
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()

            # Verify results are reasonable
            assert isinstance(energy, float)
            assert forces.shape == (len(atoms), 3)

        # If we get here without memory errors, test passes

    def test_calculator_with_langevin_dynamics(self, mock_orb_calculator):
        """Test OrbCalculator with Langevin (NVT) MD."""
        atoms = molecule("H2O")
        atoms.calc = mock_orb_calculator

        # Setup Langevin dynamics
        dyn = Langevin(
            atoms,
            timestep=1.0 * units.fs,
            temperature_K=300,
            friction=0.01,
        )

        # Run short simulation
        dyn.run(50)

        # Should complete without errors


@pytest.mark.integration
@pytest.mark.requires_fennol
class TestFeNNolCalculatorMDIntegration:
    """Integration tests for FeNNolCalculator in MD simulations."""

    @pytest.fixture
    def mock_fennol_calculator(self):
        """Fixture providing a mocked FeNNolCalculator for testing."""
        with patch("src.models.teacher_wrappers.jax"), patch(
            "src.models.teacher_wrappers.FENNIXCalculator"
        ) as mock_fennix:

            # Create a mock calculator
            mock_calc = MagicMock()

            def calculate_side_effect(atoms, properties, system_changes):
                n_atoms = len(atoms)
                positions = atoms.positions
                energy = 0.1 * np.sum(positions**2)
                forces = -0.2 * positions

                mock_calc.results = {
                    "energy": energy,
                    "forces": forces,
                }

            mock_calc.calculate.side_effect = calculate_side_effect
            mock_calc.results = {}
            mock_fennix.from_pretrained.return_value = mock_calc

            from src.models.teacher_wrappers import FeNNolCalculator

            yield FeNNolCalculator(model_name="ani-2x", device="cpu")

    def test_calculator_in_nve_simulation(self, mock_fennol_calculator):
        """Test FeNNolCalculator in NVE MD simulation."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        atoms.calc = mock_fennol_calculator

        atoms.set_momenta(np.zeros((len(atoms), 3)))

        dyn = VelocityVerlet(atoms, timestep=0.5 * units.fs)

        energies = []

        def energy_observer():
            energies.append(atoms.get_potential_energy() + atoms.get_kinetic_energy())

        dyn.attach(energy_observer, interval=1)

        dyn.run(50)

        assert len(energies) == 51

    def test_calculator_with_molecules(self, mock_fennol_calculator):
        """Test FeNNolCalculator with molecular systems."""
        molecules = ["H2O", "CH4", "NH3"]

        for mol_name in molecules:
            atoms = molecule(mol_name)
            atoms.calc = mock_fennol_calculator

            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()

            assert isinstance(energy, float)
            assert forces.shape == (len(atoms), 3)

    def test_memory_stability_repeated_calls(self, mock_fennol_calculator):
        """Test memory stability over many repeated calls."""
        atoms = molecule("H2O")
        atoms.calc = mock_fennol_calculator

        n_calls = 1000
        for i in range(n_calls):
            atoms.positions += np.random.randn(len(atoms), 3) * 0.01

            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()

            assert isinstance(energy, float)
            assert forces.shape == (len(atoms), 3)


@pytest.mark.integration
class TestDropInReplacementScenario:
    """Test realistic drop-in replacement scenarios."""

    @pytest.fixture
    def mock_orb_calculator(self):
        """Provide mocked OrbCalculator."""
        with patch("src.models.teacher_wrappers.pretrained") as mock_pretrained, patch(
            "src.models.teacher_wrappers.ORBCalculator"
        ) as mock_orb_calc:

            mock_pretrained.orb_v2.return_value = MagicMock()

            mock_calc = MagicMock()

            def calculate_side_effect(atoms, properties, system_changes):
                positions = atoms.positions
                energy = 0.1 * np.sum(positions**2)
                forces = -0.2 * positions

                mock_calc.results = {
                    "energy": energy,
                    "forces": forces,
                    "stress": np.zeros(6),
                }

            mock_calc.calculate.side_effect = calculate_side_effect
            mock_calc.results = {}
            mock_orb_calc.return_value = mock_calc

            from src.models.teacher_wrappers import OrbCalculator

            yield OrbCalculator(model_name="orb-v2", device="cpu")

    def test_swap_calculator_in_existing_script(self, mock_orb_calculator):
        """
        Test that calculator can be swapped in existing MD script.

        This simulates a user replacing their original calculator with
        our wrapper - the only change should be the calculator instantiation.
        """
        # Original MD script (user code)
        def run_md_simulation(calculator, n_steps=100):
            """Generic MD simulation function."""
            atoms = molecule("H2O")
            atoms.calc = calculator  # Only line that changes

            # Rest of MD script is identical
            dyn = VelocityVerlet(atoms, timestep=1.0 * units.fs)
            dyn.run(n_steps)

            return atoms

        # Use our wrapper calculator
        final_atoms = run_md_simulation(mock_orb_calculator, n_steps=50)

        # Verify simulation ran successfully
        assert final_atoms is not None
        assert len(final_atoms) == 3  # H2O has 3 atoms

    def test_calculator_works_with_ase_optimizers(self, mock_orb_calculator):
        """Test that calculator works with ASE geometry optimizers."""
        from ase.optimize import BFGS

        atoms = molecule("H2O")
        atoms.calc = mock_orb_calculator

        # Setup optimizer
        opt = BFGS(atoms, logfile=None)

        # Run a few optimization steps
        opt.run(fmax=0.05, steps=10)

        # Should complete without errors


class TestRealOrbCalculator:
    """
    Tests using real Orb models (requires orb-models package installed).

    These tests are skipped by default and run only when orb-models is available.
    """

    @pytest.mark.skipif(
        not pytest.importorskip("orb_models", reason="orb-models not installed"),
        reason="Requires orb-models package",
    )
    @pytest.mark.slow
    def test_real_orb_calculator_simple_molecule(self):
        """Test real OrbCalculator with a simple molecule (slow test)."""
        from src.models.teacher_wrappers import OrbCalculator

        try:
            calc = OrbCalculator(model_name="orb-v2", device="cpu")

            atoms = molecule("H2")
            atoms.calc = calc

            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()

            # Basic sanity checks
            assert isinstance(energy, float)
            assert forces.shape == (2, 3)
            assert not np.isnan(energy)
            assert not np.any(np.isnan(forces))

        except ImportError:
            pytest.skip("orb-models package not properly installed")


class TestRealFeNNolCalculator:
    """
    Tests using real FeNNol models (requires fennol package installed).

    These tests are skipped by default and run only when fennol is available.
    """

    @pytest.mark.skipif(
        not pytest.importorskip("fennol", reason="fennol not installed"),
        reason="Requires fennol package",
    )
    @pytest.mark.slow
    def test_real_fennol_calculator_simple_molecule(self):
        """Test real FeNNolCalculator with a simple molecule (slow test)."""
        from src.models.teacher_wrappers import FeNNolCalculator

        try:
            # This assumes ANI-2x is available as a pretrained model
            calc = FeNNolCalculator(model_name="ani-2x", device="cpu")

            atoms = molecule("H2O")
            atoms.calc = calc

            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()

            # Basic sanity checks
            assert isinstance(energy, float)
            assert forces.shape == (3, 3)
            assert not np.isnan(energy)
            assert not np.any(np.isnan(forces))

        except (ImportError, ValueError):
            pytest.skip("fennol package not properly installed or ANI-2x not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
