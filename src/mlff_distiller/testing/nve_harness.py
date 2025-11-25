"""
NVE (Microcanonical) Molecular Dynamics Harness

This module provides a production-ready harness for running NVE (constant energy)
molecular dynamics simulations using ASE calculators. It collects comprehensive
trajectory data for validation and analysis.

Key Features:
- NVE ensemble (constant N, V, E) for energy conservation testing
- Flexible initialization from ASE Atoms or structure files
- Comprehensive data collection (positions, velocities, energies, forces, temperatures)
- GPU/CPU execution support
- Memory-efficient trajectory storage
- Detailed logging and error handling

Usage:
    from mlff_distiller.testing import NVEMDHarness
    from mlff_distiller.inference import StudentForceFieldCalculator
    from ase.build import molecule

    # Create calculator and atoms
    calc = StudentForceFieldCalculator('checkpoints/best_model.pt')
    atoms = molecule('H2O')

    # Run simulation
    harness = NVEMDHarness(
        atoms=atoms,
        calculator=calc,
        temperature=300.0,
        timestep=0.5,  # fs
        trajectory_file='md_trajectory.traj'
    )

    results = harness.run_simulation(steps=1000)
    print(f"Energy drift: {results['energy_drift_pct']:.3f}%")

Author: Testing & Benchmarking Engineer
Date: 2025-11-25
Issue: #37
"""

from typing import Optional, Dict, List, Union, Any
from pathlib import Path
import logging
import time
import warnings

import numpy as np
from ase import Atoms, units
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.verlet import VelocityVerlet
from ase.calculators.calculator import Calculator

logger = logging.getLogger(__name__)


class NVEMDHarness:
    """
    NVE Molecular Dynamics Harness for model validation.

    This class runs microcanonical (NVE) ensemble MD simulations where
    the number of atoms (N), volume (V), and total energy (E) are conserved.
    Energy conservation is a critical test for force field accuracy.

    Args:
        atoms: ASE Atoms object or path to structure file
        calculator: ASE Calculator for energy/force evaluation
        temperature: Initial temperature in Kelvin (default: 300.0)
        timestep: MD timestep in femtoseconds (default: 0.5)
        trajectory_file: Path to save trajectory (default: None)
        log_interval: Steps between logging (default: 10)
        remove_com_motion: Remove center-of-mass motion (default: True)

    Attributes:
        atoms: The molecular system
        calculator: Force field calculator
        timestep: MD timestep in ASE internal units
        temperature: Target temperature in K
        trajectory_data: Collected MD trajectory data

    Example:
        >>> from ase.build import molecule
        >>> harness = NVEMDHarness(molecule('H2O'), calc, temperature=300)
        >>> results = harness.run_simulation(steps=1000)
        >>> print(f"Total energy drift: {results['energy_drift_pct']:.2f}%")
    """

    def __init__(
        self,
        atoms: Union[Atoms, str, Path],
        calculator: Calculator,
        temperature: float = 300.0,
        timestep: float = 0.5,
        trajectory_file: Optional[Union[str, Path]] = None,
        log_interval: int = 10,
        remove_com_motion: bool = True
    ):
        """Initialize NVE MD harness."""
        # Load atoms if path provided
        if isinstance(atoms, (str, Path)):
            atoms = read(str(atoms))

        self.atoms = atoms.copy()
        self.calculator = calculator
        self.temperature = temperature
        self.timestep_fs = timestep
        self.timestep = timestep * units.fs  # Convert to ASE internal units
        self.trajectory_file = Path(trajectory_file) if trajectory_file else None
        self.log_interval = log_interval
        self.remove_com_motion = remove_com_motion

        # Attach calculator
        self.atoms.calc = self.calculator

        # Trajectory data storage
        self.trajectory_data: Dict[str, List] = {
            'time': [],          # Time in ps
            'positions': [],     # Atomic positions (n_atoms, 3)
            'velocities': [],    # Atomic velocities (n_atoms, 3)
            'forces': [],        # Atomic forces (n_atoms, 3)
            'potential_energy': [],  # Potential energy in eV
            'kinetic_energy': [],    # Kinetic energy in eV
            'total_energy': [],      # Total energy in eV
            'temperature': [],       # Instantaneous temperature in K
        }

        # MD dynamics object (will be created in run_simulation)
        self.dynamics = None

        # Simulation metadata
        self._simulation_complete = False
        self._n_steps = 0
        self._wall_time = 0.0

        logger.info(
            f"Initialized NVE MD harness: {len(self.atoms)} atoms, "
            f"T={temperature}K, dt={timestep}fs"
        )

    def initialize_velocities(
        self,
        temperature: Optional[float] = None,
        remove_com_translation: bool = True,
        remove_com_rotation: bool = True
    ):
        """
        Initialize atomic velocities from Maxwell-Boltzmann distribution.

        Args:
            temperature: Temperature in K (uses self.temperature if None)
            remove_com_translation: Remove center-of-mass translation
            remove_com_rotation: Remove center-of-mass rotation
        """
        temp = temperature if temperature is not None else self.temperature

        # Set velocities from Maxwell-Boltzmann distribution
        MaxwellBoltzmannDistribution(self.atoms, temperature_K=temp)

        # Remove center-of-mass motion
        if remove_com_translation:
            Stationary(self.atoms)

        if remove_com_rotation:
            ZeroRotation(self.atoms)

        # Verify temperature
        actual_temp = self.atoms.get_temperature()
        logger.info(
            f"Initialized velocities: target T={temp}K, actual T={actual_temp:.2f}K"
        )

    def run_simulation(
        self,
        steps: int,
        initialize_velocities: bool = True,
        save_interval: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run NVE molecular dynamics simulation.

        Args:
            steps: Number of MD steps to run
            initialize_velocities: Initialize velocities before running (default: True)
            save_interval: Save trajectory every N steps (default: log_interval)

        Returns:
            Dictionary with simulation results:
                - trajectory_data: Full trajectory data
                - energy_drift_pct: Total energy drift percentage
                - avg_temperature: Average temperature in K
                - std_temperature: Temperature standard deviation in K
                - wall_time: Wall clock time in seconds
                - steps: Number of steps completed

        Raises:
            RuntimeError: If simulation fails
            ValueError: If invalid parameters provided
        """
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")

        save_interval = save_interval or self.log_interval

        logger.info(f"Starting NVE MD simulation: {steps} steps, {steps * self.timestep_fs:.2f} ps")
        start_time = time.perf_counter()

        try:
            # Initialize velocities if requested
            if initialize_velocities:
                self.initialize_velocities()

            # Create VelocityVerlet dynamics
            self.dynamics = VelocityVerlet(self.atoms, self.timestep)

            # Attach trajectory writer if file specified
            if self.trajectory_file:
                from ase.io.trajectory import Trajectory
                traj_writer = Trajectory(str(self.trajectory_file), 'w', self.atoms)
                self.dynamics.attach(traj_writer.write, interval=save_interval)

            # Attach data collection callback
            self.dynamics.attach(
                self._collect_data,
                interval=1  # Collect every step for accurate analysis
            )

            # Attach logging callback
            self.dynamics.attach(
                self._log_progress,
                interval=self.log_interval
            )

            # Run MD
            self.dynamics.run(steps)

            # Close trajectory file
            if self.trajectory_file:
                traj_writer.close()

            self._simulation_complete = True
            self._n_steps = steps
            self._wall_time = time.perf_counter() - start_time

            # Compute summary statistics
            results = self._compute_summary()

            logger.info(
                f"Simulation complete: {steps} steps in {self._wall_time:.2f}s "
                f"({steps/self._wall_time:.1f} steps/s)"
            )
            logger.info(
                f"Energy drift: {results['energy_drift_pct']:.4f}%, "
                f"Avg T: {results['avg_temperature']:.2f}K"
            )

            return results

        except Exception as e:
            logger.error(f"MD simulation failed: {e}", exc_info=True)
            raise RuntimeError(f"NVE MD simulation failed after {len(self.trajectory_data['time'])} steps: {e}") from e

    def _collect_data(self, step=None):
        """Collect trajectory data at current MD step."""
        # Current simulation time
        if self.dynamics is None:
            return

        current_time = self.dynamics.get_time() / (1000 * units.fs)  # Convert to ps

        # Get energies
        epot = self.atoms.get_potential_energy()
        ekin = self.atoms.get_kinetic_energy()
        etot = epot + ekin

        # Get temperature
        temp = self.atoms.get_temperature()

        # Store data
        self.trajectory_data['time'].append(current_time)
        self.trajectory_data['positions'].append(self.atoms.get_positions().copy())
        self.trajectory_data['velocities'].append(self.atoms.get_velocities().copy())
        self.trajectory_data['forces'].append(self.atoms.get_forces().copy())
        self.trajectory_data['potential_energy'].append(epot)
        self.trajectory_data['kinetic_energy'].append(ekin)
        self.trajectory_data['total_energy'].append(etot)
        self.trajectory_data['temperature'].append(temp)

    def _log_progress(self, step=None):
        """Log MD progress."""
        if not self.trajectory_data['time']:
            return

        step = len(self.trajectory_data['time']) - 1
        time_ps = self.trajectory_data['time'][-1]
        etot = self.trajectory_data['total_energy'][-1]
        temp = self.trajectory_data['temperature'][-1]

        # Compute energy drift if we have enough data
        drift_pct = 0.0
        if len(self.trajectory_data['total_energy']) > 1:
            e0 = self.trajectory_data['total_energy'][0]
            drift_pct = 100.0 * (etot - e0) / abs(e0) if e0 != 0 else 0.0

        logger.debug(
            f"Step {step:6d}: t={time_ps:8.3f}ps, "
            f"E={etot:12.6f}eV, T={temp:7.2f}K, "
            f"drift={drift_pct:+.4f}%"
        )

    def _compute_summary(self) -> Dict[str, Any]:
        """
        Compute summary statistics from trajectory.

        Returns:
            Dictionary with simulation summary
        """
        if not self.trajectory_data['time']:
            raise RuntimeError("No trajectory data collected")

        # Convert lists to numpy arrays for analysis
        times = np.array(self.trajectory_data['time'])
        total_energies = np.array(self.trajectory_data['total_energy'])
        temperatures = np.array(self.trajectory_data['temperature'])
        potential_energies = np.array(self.trajectory_data['potential_energy'])
        kinetic_energies = np.array(self.trajectory_data['kinetic_energy'])

        # Energy drift
        e0 = total_energies[0]
        efinal = total_energies[-1]
        energy_drift_pct = 100.0 * (efinal - e0) / abs(e0) if e0 != 0 else 0.0

        # Energy statistics
        energy_std = np.std(total_energies)
        energy_range = np.ptp(total_energies)  # peak-to-peak

        # Temperature statistics
        avg_temp = np.mean(temperatures)
        std_temp = np.std(temperatures)

        # Time statistics
        total_time_ps = times[-1] - times[0]

        return {
            'trajectory_data': self.trajectory_data,
            'n_steps': self._n_steps,  # Use requested steps, not length of trajectory
            'total_time_ps': total_time_ps,
            'timestep_fs': self.timestep_fs,
            'wall_time_s': self._wall_time,
            'energy_drift_pct': energy_drift_pct,
            'energy_drift_abs': efinal - e0,
            'energy_std': energy_std,
            'energy_range': energy_range,
            'avg_temperature': avg_temp,
            'std_temperature': std_temp,
            'initial_energy': e0,
            'final_energy': efinal,
            'avg_potential_energy': np.mean(potential_energies),
            'avg_kinetic_energy': np.mean(kinetic_energies),
        }

    def reset(self):
        """Reset trajectory data and simulation state."""
        self.trajectory_data = {
            'time': [],
            'positions': [],
            'velocities': [],
            'forces': [],
            'potential_energy': [],
            'kinetic_energy': [],
            'total_energy': [],
            'temperature': [],
        }
        self._simulation_complete = False
        self._n_steps = 0
        self._wall_time = 0.0
        self.dynamics = None

    @property
    def is_complete(self) -> bool:
        """Check if simulation has completed."""
        return self._simulation_complete

    @property
    def n_atoms(self) -> int:
        """Number of atoms in system."""
        return len(self.atoms)

    def get_trajectory_array(self, key: str) -> np.ndarray:
        """
        Get trajectory data as numpy array.

        Args:
            key: Data key ('time', 'positions', 'velocities', etc.)

        Returns:
            Numpy array of trajectory data
        """
        if key not in self.trajectory_data:
            raise KeyError(f"Unknown trajectory key: {key}")

        data = self.trajectory_data[key]
        if not data:
            raise RuntimeError("No trajectory data collected")

        return np.array(data)

    def save_trajectory(self, filename: Union[str, Path], format: str = 'traj'):
        """
        Save trajectory to file.

        Args:
            filename: Output file path
            format: File format ('traj', 'xyz', 'pdb', etc.)
        """
        if not self._simulation_complete:
            warnings.warn("Simulation not complete, saving partial trajectory")

        if not self.trajectory_data['time']:
            raise RuntimeError("No trajectory data to save")

        # Create atoms list for each frame
        frames = []
        for i in range(len(self.trajectory_data['time'])):
            atoms = self.atoms.copy()
            atoms.set_positions(self.trajectory_data['positions'][i])
            atoms.set_velocities(self.trajectory_data['velocities'][i])

            # Store energy/force data in info dict
            atoms.info['time'] = self.trajectory_data['time'][i]
            atoms.info['potential_energy'] = self.trajectory_data['potential_energy'][i]
            atoms.info['kinetic_energy'] = self.trajectory_data['kinetic_energy'][i]
            atoms.info['total_energy'] = self.trajectory_data['total_energy'][i]
            atoms.info['temperature'] = self.trajectory_data['temperature'][i]

            # Store forces in arrays
            atoms.arrays['forces'] = self.trajectory_data['forces'][i]

            frames.append(atoms)

        # Write trajectory
        write(str(filename), frames, format=format)
        logger.info(f"Saved trajectory: {len(frames)} frames to {filename}")


__all__ = ['NVEMDHarness']
