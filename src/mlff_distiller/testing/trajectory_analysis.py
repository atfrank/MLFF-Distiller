"""
Trajectory Analysis for MD Validation

This module provides comprehensive analysis tools for MD trajectories including
structural stability, atom displacements, temperature evolution, and summary reports.

Key Features:
- Trajectory stability analysis (RMSD, max displacement)
- Atom displacement statistics
- Temperature evolution tracking
- Bond length monitoring
- Comprehensive summary reports

Usage:
    from mlff_distiller.testing import analyze_trajectory_stability

    # Analyze MD trajectory
    stability = analyze_trajectory_stability(trajectory_data)
    print(f"Max displacement: {stability['max_displacement']:.3f} Å")
    print(f"RMSD: {stability['rmsd_final']:.3f} Å")

Author: Testing & Benchmarking Engineer
Date: 2025-11-25
Issue: #37
"""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_rmsd(
    positions: np.ndarray,
    reference: Optional[np.ndarray] = None,
    align: bool = False
) -> np.ndarray:
    """
    Compute Root Mean Square Deviation of atomic positions.

    Args:
        positions: Positions array [n_frames, n_atoms, 3] or [n_atoms, 3]
        reference: Reference positions [n_atoms, 3] (default: first frame)
        align: Apply optimal rotation alignment before computing RMSD (default: False)

    Returns:
        RMSD values [n_frames] or single value for single frame

    Example:
        >>> positions = np.random.randn(100, 10, 3)
        >>> rmsd = compute_rmsd(positions)
        >>> print(f"Final RMSD: {rmsd[-1]:.3f} Å")
    """
    single_frame = positions.ndim == 2

    if single_frame:
        positions = positions[np.newaxis, :, :]  # Add frame dimension

    n_frames, n_atoms, _ = positions.shape

    # Use first frame as reference if not provided
    if reference is None:
        reference = positions[0]

    if reference.shape != (n_atoms, 3):
        raise ValueError(f"Reference shape must be ({n_atoms}, 3), got {reference.shape}")

    # Compute RMSD for each frame
    rmsd_values = np.zeros(n_frames)

    for i in range(n_frames):
        pos = positions[i]

        if align:
            # Apply optimal rotation alignment (Kabsch algorithm)
            pos_aligned = kabsch_align(pos, reference)
            diff = pos_aligned - reference
        else:
            diff = pos - reference

        rmsd_values[i] = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))

    if single_frame:
        return float(rmsd_values[0])
    else:
        return rmsd_values


def kabsch_align(positions: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Align positions to reference using Kabsch algorithm.

    Args:
        positions: Positions to align [n_atoms, 3]
        reference: Reference positions [n_atoms, 3]

    Returns:
        Aligned positions [n_atoms, 3]
    """
    # Center both structures
    pos_centered = positions - np.mean(positions, axis=0)
    ref_centered = reference - np.mean(reference, axis=0)

    # Compute rotation matrix via SVD
    covariance = np.dot(pos_centered.T, ref_centered)
    U, S, Vt = np.linalg.svd(covariance)

    # Ensure right-handed coordinate system
    d = np.linalg.det(np.dot(U, Vt))
    if d < 0:
        U[:, -1] = -U[:, -1]

    rotation = np.dot(U, Vt)

    # Apply rotation
    aligned = np.dot(pos_centered, rotation.T) + np.mean(reference, axis=0)

    return aligned


def compute_atom_displacements(
    positions: np.ndarray,
    reference: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Compute atom displacement statistics.

    Args:
        positions: Positions array [n_frames, n_atoms, 3]
        reference: Reference positions [n_atoms, 3] (default: first frame)

    Returns:
        Dictionary with displacement metrics:
            - displacements: Displacement vectors [n_frames, n_atoms, 3]
            - magnitudes: Displacement magnitudes [n_frames, n_atoms]
            - max_per_frame: Max displacement per frame [n_frames]
            - mean_per_frame: Mean displacement per frame [n_frames]
            - per_atom_max: Max displacement per atom [n_atoms]

    Example:
        >>> positions = np.random.randn(100, 10, 3)
        >>> displacements = compute_atom_displacements(positions)
        >>> print(f"Max displacement: {displacements['per_atom_max'].max():.3f} Å")
    """
    if positions.ndim != 3:
        raise ValueError(f"Expected 3D array [n_frames, n_atoms, 3], got shape {positions.shape}")

    n_frames, n_atoms, _ = positions.shape

    # Use first frame as reference if not provided
    if reference is None:
        reference = positions[0]

    if reference.shape != (n_atoms, 3):
        raise ValueError(f"Reference shape must be ({n_atoms}, 3), got {reference.shape}")

    # Compute displacement vectors
    displacement_vectors = positions - reference[np.newaxis, :, :]

    # Compute magnitudes
    displacement_magnitudes = np.linalg.norm(displacement_vectors, axis=2)

    # Statistics
    max_per_frame = np.max(displacement_magnitudes, axis=1)
    mean_per_frame = np.mean(displacement_magnitudes, axis=1)
    per_atom_max = np.max(displacement_magnitudes, axis=0)

    return {
        'displacements': displacement_vectors,
        'magnitudes': displacement_magnitudes,
        'max_per_frame': max_per_frame,
        'mean_per_frame': mean_per_frame,
        'per_atom_max': per_atom_max,
    }


def analyze_temperature_evolution(
    temperatures: np.ndarray,
    target_temperature: Optional[float] = None
) -> Dict[str, float]:
    """
    Analyze temperature evolution during MD simulation.

    Args:
        temperatures: Temperature values [n_frames] in Kelvin
        target_temperature: Target temperature for comparison (default: None)

    Returns:
        Dictionary with temperature statistics:
            - mean: Mean temperature
            - std: Standard deviation
            - min: Minimum temperature
            - max: Maximum temperature
            - drift: Temperature drift (final - initial)
            - drift_pct: Drift as percentage of target (if provided)

    Example:
        >>> temps = np.array([300, 305, 295, 300, 310])
        >>> stats = analyze_temperature_evolution(temps, target_temperature=300)
        >>> print(f"Mean T: {stats['mean']:.2f} K")
    """
    if len(temperatures) < 2:
        raise ValueError(f"Need at least 2 temperature values, got {len(temperatures)}")

    mean_temp = np.mean(temperatures)
    std_temp = np.std(temperatures)
    min_temp = np.min(temperatures)
    max_temp = np.max(temperatures)
    drift = temperatures[-1] - temperatures[0]

    result = {
        'mean': float(mean_temp),
        'std': float(std_temp),
        'min': float(min_temp),
        'max': float(max_temp),
        'drift': float(drift),
    }

    if target_temperature is not None:
        drift_pct = 100.0 * drift / target_temperature if target_temperature > 0 else 0.0
        deviation_pct = 100.0 * (mean_temp - target_temperature) / target_temperature if target_temperature > 0 else 0.0
        result['drift_pct'] = float(drift_pct)
        result['deviation_from_target_pct'] = float(deviation_pct)

    return result


def compute_bond_lengths(
    positions: np.ndarray,
    bonds: List[Tuple[int, int]]
) -> np.ndarray:
    """
    Compute bond lengths for specified atom pairs.

    Args:
        positions: Positions array [n_frames, n_atoms, 3] or [n_atoms, 3]
        bonds: List of (atom_i, atom_j) index pairs

    Returns:
        Bond lengths [n_frames, n_bonds] or [n_bonds] for single frame

    Example:
        >>> positions = np.random.randn(100, 10, 3)
        >>> bonds = [(0, 1), (1, 2), (2, 3)]
        >>> bond_lengths = compute_bond_lengths(positions, bonds)
        >>> print(f"Bond 0-1 lengths: {bond_lengths[:, 0]}")
    """
    single_frame = positions.ndim == 2

    if single_frame:
        positions = positions[np.newaxis, :, :]

    n_frames, n_atoms, _ = positions.shape
    n_bonds = len(bonds)

    bond_lengths = np.zeros((n_frames, n_bonds))

    for i, (atom_i, atom_j) in enumerate(bonds):
        if atom_i >= n_atoms or atom_j >= n_atoms:
            raise ValueError(f"Bond ({atom_i}, {atom_j}) exceeds number of atoms {n_atoms}")

        vectors = positions[:, atom_j, :] - positions[:, atom_i, :]
        bond_lengths[:, i] = np.linalg.norm(vectors, axis=1)

    if single_frame:
        return bond_lengths[0]
    else:
        return bond_lengths


def analyze_trajectory_stability(
    trajectory_data: Dict[str, List],
    align_rmsd: bool = False
) -> Dict[str, Any]:
    """
    Comprehensive trajectory stability analysis.

    Args:
        trajectory_data: Trajectory data from NVEMDHarness
        align_rmsd: Apply alignment for RMSD calculation (default: False)

    Returns:
        Dictionary with stability metrics:
            - rmsd_values: RMSD over time [n_frames]
            - rmsd_final: Final RMSD value
            - max_displacement: Maximum atom displacement
            - mean_displacement_final: Mean displacement at final frame
            - displacement_stats: Per-atom displacement statistics
            - positions_stable: Boolean, True if max displacement < 2.0 Å

    Example:
        >>> stability = analyze_trajectory_stability(trajectory_data)
        >>> if stability['positions_stable']:
        >>>     print("Structure remained stable during MD")
    """
    # Extract positions
    positions = np.array(trajectory_data['positions'])  # [n_frames, n_atoms, 3]

    # Compute RMSD
    rmsd_values = compute_rmsd(positions, align=align_rmsd)
    rmsd_final = rmsd_values[-1]

    # Compute displacements
    displacement_stats = compute_atom_displacements(positions)
    max_displacement = float(displacement_stats['per_atom_max'].max())
    mean_displacement_final = float(displacement_stats['mean_per_frame'][-1])

    # Stability criterion (somewhat arbitrary, but reasonable for small molecules)
    # Max displacement < 2.0 Å indicates structure hasn't dramatically changed
    positions_stable = max_displacement < 2.0

    result = {
        'rmsd_values': rmsd_values,
        'rmsd_final': float(rmsd_final),
        'max_displacement': max_displacement,
        'mean_displacement_final': mean_displacement_final,
        'displacement_stats': displacement_stats,
        'positions_stable': positions_stable,
    }

    return result


def generate_trajectory_summary(
    trajectory_data: Dict[str, List],
    target_temperature: Optional[float] = None,
    energy_tolerance_pct: float = 1.0,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Generate comprehensive trajectory summary report.

    This is the main function for high-level trajectory analysis, combining
    all analysis modules.

    Args:
        trajectory_data: Trajectory data from NVEMDHarness
        target_temperature: Target temperature in K (default: None)
        energy_tolerance_pct: Acceptable energy drift percentage (default: 1.0%)
        verbose: Print detailed report (default: True)

    Returns:
        Dictionary with comprehensive summary:
            - simulation_info: Basic simulation metadata
            - energy_summary: Energy statistics
            - temperature_summary: Temperature statistics
            - stability_summary: Structural stability metrics
            - overall_quality: Quality assessment (PASS/FAIL)

    Example:
        >>> summary = generate_trajectory_summary(trajectory_data, target_temperature=300)
        >>> if summary['overall_quality']['passed']:
        >>>     print("Trajectory quality: PASSED")
    """
    # Import energy metrics here to avoid circular imports
    from .energy_metrics import (
        compute_energy_drift,
        compute_energy_conservation_ratio,
        compute_energy_fluctuations,
    )

    # Extract data
    times = np.array(trajectory_data['time'])
    total_energies = np.array(trajectory_data['total_energy'])
    kinetic_energies = np.array(trajectory_data['kinetic_energy'])
    potential_energies = np.array(trajectory_data['potential_energy'])
    temperatures = np.array(trajectory_data['temperature'])
    positions = np.array(trajectory_data['positions'])

    # Simulation info
    n_frames = len(times)
    n_atoms = positions.shape[1]
    total_time_ps = times[-1] - times[0]
    timestep_fs = (times[1] - times[0]) * 1000 if n_frames > 1 else 0.0

    simulation_info = {
        'n_frames': n_frames,
        'n_atoms': n_atoms,
        'total_time_ps': float(total_time_ps),
        'timestep_fs': float(timestep_fs),
    }

    # Energy analysis
    energy_drift_pct = compute_energy_drift(total_energies)
    conservation_ratio = compute_energy_conservation_ratio(total_energies)
    fluctuation_stats = compute_energy_fluctuations(total_energies)

    energy_summary = {
        'initial_energy': float(total_energies[0]),
        'final_energy': float(total_energies[-1]),
        'mean_energy': float(np.mean(total_energies)),
        'drift_pct': energy_drift_pct,
        'conservation_ratio': conservation_ratio,
        'std': fluctuation_stats['std'],
        'range': fluctuation_stats['range'],
    }

    # Temperature analysis
    temperature_summary = analyze_temperature_evolution(temperatures, target_temperature)

    # Stability analysis
    stability_summary = analyze_trajectory_stability(trajectory_data)

    # Quality assessment
    energy_ok = abs(energy_drift_pct) <= energy_tolerance_pct
    conservation_ok = conservation_ratio >= 0.99
    structure_ok = stability_summary['positions_stable']

    if target_temperature is not None:
        temp_deviation_ok = abs(temperature_summary['deviation_from_target_pct']) <= 10.0  # ±10%
    else:
        temp_deviation_ok = True

    overall_passed = energy_ok and conservation_ok and structure_ok and temp_deviation_ok

    overall_quality = {
        'passed': overall_passed,
        'energy_drift_ok': energy_ok,
        'energy_conservation_ok': conservation_ok,
        'structure_stable': structure_ok,
        'temperature_ok': temp_deviation_ok,
    }

    # Build summary
    summary = {
        'simulation_info': simulation_info,
        'energy_summary': energy_summary,
        'temperature_summary': temperature_summary,
        'stability_summary': stability_summary,
        'overall_quality': overall_quality,
    }

    # Verbose report
    if verbose:
        logger.info("=" * 70)
        logger.info("TRAJECTORY SUMMARY REPORT")
        logger.info("=" * 70)
        logger.info("")
        logger.info("Simulation Info:")
        logger.info(f"  Atoms: {n_atoms}")
        logger.info(f"  Frames: {n_frames}")
        logger.info(f"  Total time: {total_time_ps:.3f} ps")
        logger.info(f"  Timestep: {timestep_fs:.3f} fs")
        logger.info("")
        logger.info("Energy:")
        logger.info(f"  Initial: {energy_summary['initial_energy']:.6f} eV")
        logger.info(f"  Final: {energy_summary['final_energy']:.6f} eV")
        logger.info(f"  Drift: {energy_drift_pct:+.4f}% (tolerance: ±{energy_tolerance_pct}%)")
        logger.info(f"  Conservation ratio: {conservation_ratio:.6f} (target: > 0.99)")
        logger.info(f"  Status: {'PASS' if energy_ok and conservation_ok else 'FAIL'}")
        logger.info("")
        logger.info("Temperature:")
        logger.info(f"  Mean: {temperature_summary['mean']:.2f} K")
        logger.info(f"  Std: {temperature_summary['std']:.2f} K")
        logger.info(f"  Range: [{temperature_summary['min']:.2f}, {temperature_summary['max']:.2f}] K")
        if target_temperature is not None:
            logger.info(f"  Deviation from target: {temperature_summary['deviation_from_target_pct']:+.2f}%")
            logger.info(f"  Status: {'PASS' if temp_deviation_ok else 'FAIL'}")
        logger.info("")
        logger.info("Structural Stability:")
        logger.info(f"  Final RMSD: {stability_summary['rmsd_final']:.3f} Å")
        logger.info(f"  Max displacement: {stability_summary['max_displacement']:.3f} Å")
        logger.info(f"  Mean displacement (final): {stability_summary['mean_displacement_final']:.3f} Å")
        logger.info(f"  Status: {'PASS' if structure_ok else 'FAIL'}")
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"OVERALL QUALITY: {'PASSED' if overall_passed else 'FAILED'}")
        logger.info("=" * 70)

    return summary


__all__ = [
    'compute_rmsd',
    'kabsch_align',
    'compute_atom_displacements',
    'analyze_temperature_evolution',
    'compute_bond_lengths',
    'analyze_trajectory_stability',
    'generate_trajectory_summary',
]
