"""
Unit tests for NVE MD Harness

Tests the NVEMDHarness class for running molecular dynamics simulations.

Author: Testing & Benchmarking Engineer
Date: 2025-11-25
Issue: #37
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.build import molecule

from mlff_distiller.testing import NVEMDHarness


class TestNVEMDHarness:
    """Test suite for NVEMDHarness class."""

    @pytest.fixture
    def simple_atoms(self):
        """Create simple test system."""
        # Use argon dimer for simple test
        atoms = Atoms('Ar2', positions=[[0, 0, 0], [3.5, 0, 0]])
        return atoms

    @pytest.fixture
    def water_molecule(self):
        """Create water molecule."""
        return molecule('H2O')

    @pytest.fixture
    def lj_calculator(self):
        """Create Lennard-Jones calculator for testing."""
        return LennardJones(sigma=3.4, epsilon=0.01)

    def test_initialization(self, simple_atoms, lj_calculator):
        """Test harness initialization."""
        harness = NVEMDHarness(
            atoms=simple_atoms,
            calculator=lj_calculator,
            temperature=300.0,
            timestep=0.5
        )

        assert harness.n_atoms == 2
        assert harness.temperature == 300.0
        assert harness.timestep_fs == 0.5
        assert not harness.is_complete
        assert len(harness.trajectory_data['time']) == 0

    def test_initialization_from_file(self, tmp_path, lj_calculator):
        """Test initialization from structure file."""
        from ase.io import write

        # Create temporary structure file
        atoms = molecule('H2O')
        struct_file = tmp_path / 'test.xyz'
        write(str(struct_file), atoms)

        # Initialize from file
        harness = NVEMDHarness(
            atoms=struct_file,
            calculator=lj_calculator
        )

        assert harness.n_atoms == 3
        symbols = harness.atoms.get_chemical_symbols()
        assert sorted(symbols) == ['H', 'H', 'O']  # Order may vary

    def test_velocity_initialization(self, simple_atoms, lj_calculator):
        """Test velocity initialization."""
        harness = NVEMDHarness(
            atoms=simple_atoms,
            calculator=lj_calculator,
            temperature=300.0
        )

        # Before initialization (may be None or zero velocities)
        velocities_before = harness.atoms.get_velocities()
        if velocities_before is not None:
            assert np.allclose(velocities_before, 0.0)

        # Initialize velocities
        harness.initialize_velocities()

        # After initialization
        velocities = harness.atoms.get_velocities()
        assert velocities is not None
        assert velocities.shape == (2, 3)

        # Check temperature is approximately correct
        temp = harness.atoms.get_temperature()
        # For 2 atoms, temperature can vary significantly
        assert temp > 0  # Just check it's positive

    def test_run_simulation_basic(self, simple_atoms, lj_calculator):
        """Test basic MD simulation run."""
        harness = NVEMDHarness(
            atoms=simple_atoms,
            calculator=lj_calculator,
            temperature=100.0,  # Lower temp for faster convergence
            timestep=1.0
        )

        # Run short simulation
        results = harness.run_simulation(steps=10, initialize_velocities=True)

        # Check results structure
        assert 'trajectory_data' in results
        assert 'energy_drift_pct' in results
        assert 'avg_temperature' in results
        assert 'n_steps' in results

        # Check simulation completed
        assert harness.is_complete
        assert results['n_steps'] == 10

        # Check trajectory data collected (includes initial frame, so n_steps + 1)
        assert len(harness.trajectory_data['time']) == 11
        assert len(harness.trajectory_data['positions']) == 11
        assert len(harness.trajectory_data['forces']) == 11
        assert len(harness.trajectory_data['total_energy']) == 11

    def test_trajectory_data_shapes(self, simple_atoms, lj_calculator):
        """Test trajectory data has correct shapes."""
        harness = NVEMDHarness(
            atoms=simple_atoms,
            calculator=lj_calculator,
            temperature=100.0
        )

        results = harness.run_simulation(steps=5)

        # Check shapes (includes initial frame, so n_steps + 1)
        positions = np.array(harness.trajectory_data['positions'])
        velocities = np.array(harness.trajectory_data['velocities'])
        forces = np.array(harness.trajectory_data['forces'])

        assert positions.shape == (6, 2, 3)
        assert velocities.shape == (6, 2, 3)
        assert forces.shape == (6, 2, 3)

        assert len(harness.trajectory_data['time']) == 6
        assert len(harness.trajectory_data['potential_energy']) == 6
        assert len(harness.trajectory_data['kinetic_energy']) == 6
        assert len(harness.trajectory_data['total_energy']) == 6
        assert len(harness.trajectory_data['temperature']) == 6

    def test_energy_conservation(self, simple_atoms, lj_calculator):
        """Test energy conservation in NVE simulation."""
        harness = NVEMDHarness(
            atoms=simple_atoms,
            calculator=lj_calculator,
            temperature=100.0,
            timestep=1.0
        )

        results = harness.run_simulation(steps=50)

        # Check energy drift is small (LJ is well-behaved)
        # Allow 5% drift for this simple test
        assert abs(results['energy_drift_pct']) < 5.0

    def test_reset(self, simple_atoms, lj_calculator):
        """Test resetting harness state."""
        harness = NVEMDHarness(
            atoms=simple_atoms,
            calculator=lj_calculator
        )

        # Run simulation (includes initial frame)
        harness.run_simulation(steps=5)
        assert harness.is_complete
        assert len(harness.trajectory_data['time']) == 6

        # Reset
        harness.reset()
        assert not harness.is_complete
        assert len(harness.trajectory_data['time']) == 0

    def test_get_trajectory_array(self, simple_atoms, lj_calculator):
        """Test getting trajectory data as arrays."""
        harness = NVEMDHarness(
            atoms=simple_atoms,
            calculator=lj_calculator
        )

        harness.run_simulation(steps=10)

        # Get different trajectory arrays (includes initial frame)
        times = harness.get_trajectory_array('time')
        energies = harness.get_trajectory_array('total_energy')
        positions = harness.get_trajectory_array('positions')

        assert isinstance(times, np.ndarray)
        assert isinstance(energies, np.ndarray)
        assert isinstance(positions, np.ndarray)

        assert times.shape == (11,)
        assert energies.shape == (11,)
        assert positions.shape == (11, 2, 3)

    def test_get_trajectory_array_invalid_key(self, simple_atoms, lj_calculator):
        """Test getting trajectory with invalid key raises error."""
        harness = NVEMDHarness(
            atoms=simple_atoms,
            calculator=lj_calculator
        )

        harness.run_simulation(steps=5)

        with pytest.raises(KeyError):
            harness.get_trajectory_array('invalid_key')

    def test_get_trajectory_array_before_simulation(self, simple_atoms, lj_calculator):
        """Test getting trajectory before simulation raises error."""
        harness = NVEMDHarness(
            atoms=simple_atoms,
            calculator=lj_calculator
        )

        with pytest.raises(RuntimeError):
            harness.get_trajectory_array('time')

    def test_save_trajectory(self, simple_atoms, lj_calculator, tmp_path):
        """Test saving trajectory to file."""
        harness = NVEMDHarness(
            atoms=simple_atoms,
            calculator=lj_calculator
        )

        harness.run_simulation(steps=10)

        # Save trajectory
        traj_file = tmp_path / 'trajectory.traj'
        harness.save_trajectory(traj_file)

        assert traj_file.exists()

        # Read back and check (includes initial frame)
        from ase.io import read
        frames = read(str(traj_file), index=':')
        assert len(frames) == 11

        # Check first frame has correct data
        assert 'time' in frames[0].info
        assert 'total_energy' in frames[0].info
        # Note: Forces may not persist after write/read with ASE trajectory format
        # This is expected ASE behavior

    def test_trajectory_file_writing(self, simple_atoms, lj_calculator, tmp_path):
        """Test automatic trajectory file writing during simulation."""
        traj_file = tmp_path / 'auto_trajectory.traj'

        harness = NVEMDHarness(
            atoms=simple_atoms,
            calculator=lj_calculator,
            trajectory_file=traj_file,
            log_interval=5  # Save every 5 steps
        )

        harness.run_simulation(steps=10, save_interval=5)

        # File should exist
        assert traj_file.exists()

        # Read back (only saved every 5 steps, plus initial)
        from ase.io import read
        frames = read(str(traj_file), index=':')
        # Should have 3 frames: step 0, step 5, step 10
        assert len(frames) == 3

    def test_invalid_steps(self, simple_atoms, lj_calculator):
        """Test that invalid step count raises error."""
        harness = NVEMDHarness(
            atoms=simple_atoms,
            calculator=lj_calculator
        )

        with pytest.raises(ValueError):
            harness.run_simulation(steps=0)

        with pytest.raises(ValueError):
            harness.run_simulation(steps=-10)

    def test_wall_time_tracking(self, simple_atoms, lj_calculator):
        """Test that wall time is tracked."""
        harness = NVEMDHarness(
            atoms=simple_atoms,
            calculator=lj_calculator
        )

        results = harness.run_simulation(steps=10)

        assert 'wall_time_s' in results
        assert results['wall_time_s'] > 0

    def test_multiple_runs_without_reset(self, simple_atoms, lj_calculator):
        """Test that multiple runs accumulate data correctly."""
        harness = NVEMDHarness(
            atoms=simple_atoms,
            calculator=lj_calculator
        )

        # First run (includes initial frame)
        results1 = harness.run_simulation(steps=5, initialize_velocities=True)
        assert len(harness.trajectory_data['time']) == 6

        # Second run (should continue from first, adds 6 more frames including initial)
        results2 = harness.run_simulation(steps=5, initialize_velocities=False)

        # Data should accumulate
        assert len(harness.trajectory_data['time']) == 12


class TestNVEMDHarnessIntegration:
    """Integration tests with more realistic systems."""

    def test_water_molecule_stability(self):
        """Test MD simulation of water molecule."""
        atoms = molecule('H2O')
        calc = LennardJones()

        harness = NVEMDHarness(
            atoms=atoms,
            calculator=calc,
            temperature=100.0,  # Low temp for stability
            timestep=1.0
        )

        results = harness.run_simulation(steps=20)

        # Simulation should complete
        assert harness.is_complete
        assert results['n_steps'] == 20

        # Check data integrity (includes initial frame)
        assert len(harness.trajectory_data['time']) == 21
        assert not np.any(np.isnan(harness.get_trajectory_array('total_energy')))

    def test_summary_statistics(self):
        """Test that summary statistics are computed correctly."""
        atoms = Atoms('Ar2', positions=[[0, 0, 0], [3.5, 0, 0]])
        calc = LennardJones()

        harness = NVEMDHarness(
            atoms=atoms,
            calculator=calc,
            temperature=200.0,
            timestep=1.0
        )

        results = harness.run_simulation(steps=30)

        # Check simulation completed
        assert results['n_steps'] == 30

        # Check all expected summary fields exist
        expected_fields = [
            'trajectory_data',
            'n_steps',
            'total_time_ps',
            'timestep_fs',
            'wall_time_s',
            'energy_drift_pct',
            'energy_drift_abs',
            'energy_std',
            'energy_range',
            'avg_temperature',
            'std_temperature',
            'initial_energy',
            'final_energy',
            'avg_potential_energy',
            'avg_kinetic_energy',
        ]

        for field in expected_fields:
            assert field in results, f"Missing field: {field}"

        # Check values are reasonable
        assert results['n_steps'] == 30
        assert results['total_time_ps'] > 0
        assert not np.isnan(results['avg_temperature'])
        assert not np.isnan(results['energy_drift_pct'])
