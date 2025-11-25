"""
Energy Conservation Metrics for MD Validation

This module provides comprehensive metrics for analyzing energy conservation
in molecular dynamics simulations. Energy conservation is a critical test
for force field accuracy in NVE (microcanonical) ensemble simulations.

Key Metrics:
- Total energy drift (absolute and percentage)
- Energy conservation ratio
- Energy fluctuations (standard deviation, range)
- Kinetic/potential energy stability
- Time-resolved drift analysis

Target Criteria (Production Quality):
- Total energy drift < 1.0% for 10ps simulation
- Energy conservation ratio > 0.99
- Stable kinetic/potential energy partitioning

Usage:
    from mlff_distiller.testing import NVEMDHarness, compute_energy_drift

    # Run simulation
    harness = NVEMDHarness(atoms, calc)
    results = harness.run_simulation(steps=1000)

    # Analyze energy conservation
    energies = results['trajectory_data']['total_energy']
    drift_pct = compute_energy_drift(energies)
    conservation_ratio = compute_energy_conservation_ratio(energies)

    print(f"Energy drift: {drift_pct:.4f}%")
    print(f"Conservation ratio: {conservation_ratio:.6f}")

Author: Testing & Benchmarking Engineer
Date: 2025-11-25
Issue: #37
"""

from typing import Dict, List, Optional, Tuple, Any
import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_energy_drift(
    energies: np.ndarray,
    method: str = 'relative'
) -> float:
    """
    Compute total energy drift over trajectory.

    Args:
        energies: Array of total energies [n_steps]
        method: Drift computation method:
            - 'relative': (E_final - E_initial) / |E_initial| * 100 (default)
            - 'absolute': E_final - E_initial
            - 'max_deviation': max(|E - E_initial|) / |E_initial| * 100

    Returns:
        Energy drift (percentage for 'relative'/'max_deviation', eV for 'absolute')

    Example:
        >>> energies = np.array([-100.0, -100.1, -100.2, -99.9])
        >>> drift = compute_energy_drift(energies)
        >>> print(f"Drift: {drift:.4f}%")
    """
    if len(energies) < 2:
        raise ValueError(f"Need at least 2 energy values, got {len(energies)}")

    e_initial = energies[0]
    e_final = energies[-1]

    if method == 'relative':
        if abs(e_initial) < 1e-10:
            logger.warning("Initial energy near zero, drift percentage may be unreliable")
            return 0.0
        drift_pct = 100.0 * (e_final - e_initial) / abs(e_initial)
        return drift_pct

    elif method == 'absolute':
        return e_final - e_initial

    elif method == 'max_deviation':
        if abs(e_initial) < 1e-10:
            logger.warning("Initial energy near zero, drift percentage may be unreliable")
            return 0.0
        max_dev = np.max(np.abs(energies - e_initial))
        return 100.0 * max_dev / abs(e_initial)

    else:
        raise ValueError(f"Unknown drift method: {method}")


def compute_energy_conservation_ratio(
    energies: np.ndarray,
    window: Optional[int] = None
) -> float:
    """
    Compute energy conservation ratio.

    This metric quantifies how well energy is conserved by comparing
    energy fluctuations to the mean energy:

        conservation_ratio = 1 - (σ_E / |<E>|)

    A perfect conservation has ratio = 1.0.
    Target for production: ratio > 0.99 (< 1% fluctuation)

    Args:
        energies: Array of total energies [n_steps]
        window: Optional rolling window size for local analysis

    Returns:
        Energy conservation ratio (0 to 1, higher is better)

    Example:
        >>> energies = np.array([-100.0, -100.1, -99.9, -100.0])
        >>> ratio = compute_energy_conservation_ratio(energies)
        >>> print(f"Conservation ratio: {ratio:.6f}")
    """
    if len(energies) < 2:
        raise ValueError(f"Need at least 2 energy values, got {len(energies)}")

    if window is not None:
        # Rolling window analysis
        if window > len(energies):
            logger.warning(f"Window size {window} > trajectory length {len(energies)}, using full trajectory")
            window = None

    if window is None:
        # Global analysis
        mean_energy = np.mean(energies)
        std_energy = np.std(energies)

        if abs(mean_energy) < 1e-10:
            logger.warning("Mean energy near zero, conservation ratio may be unreliable")
            return 1.0

        ratio = 1.0 - (std_energy / abs(mean_energy))
        return max(0.0, ratio)  # Clamp to [0, 1]

    else:
        # Local analysis with rolling window
        ratios = []
        for i in range(len(energies) - window + 1):
            window_energies = energies[i:i+window]
            mean_e = np.mean(window_energies)
            std_e = np.std(window_energies)

            if abs(mean_e) > 1e-10:
                ratio = 1.0 - (std_e / abs(mean_e))
                ratios.append(max(0.0, ratio))

        return np.mean(ratios) if ratios else 1.0


def compute_energy_fluctuations(
    energies: np.ndarray
) -> Dict[str, float]:
    """
    Compute energy fluctuation statistics.

    Args:
        energies: Array of total energies [n_steps]

    Returns:
        Dictionary with fluctuation metrics:
            - std: Standard deviation (eV)
            - range: Peak-to-peak range (eV)
            - mean_abs_dev: Mean absolute deviation from mean (eV)
            - coefficient_of_variation: std / |mean| (dimensionless)

    Example:
        >>> energies = np.array([-100.0, -100.1, -99.9, -100.0])
        >>> stats = compute_energy_fluctuations(energies)
        >>> print(f"Energy std: {stats['std']:.6f} eV")
    """
    if len(energies) < 2:
        raise ValueError(f"Need at least 2 energy values, got {len(energies)}")

    mean_energy = np.mean(energies)
    std_energy = np.std(energies)
    range_energy = np.ptp(energies)  # peak-to-peak
    mean_abs_dev = np.mean(np.abs(energies - mean_energy))

    # Coefficient of variation
    if abs(mean_energy) > 1e-10:
        coef_var = std_energy / abs(mean_energy)
    else:
        coef_var = 0.0

    return {
        'std': std_energy,
        'range': range_energy,
        'mean_abs_dev': mean_abs_dev,
        'coefficient_of_variation': coef_var,
    }


def compute_kinetic_potential_stability(
    kinetic_energies: np.ndarray,
    potential_energies: np.ndarray
) -> Dict[str, float]:
    """
    Analyze kinetic and potential energy stability.

    In NVE simulations, total energy is conserved but kinetic and potential
    energies fluctuate around their equilibrium values. This function
    quantifies these fluctuations.

    Args:
        kinetic_energies: Array of kinetic energies [n_steps]
        potential_energies: Array of potential energies [n_steps]

    Returns:
        Dictionary with stability metrics:
            - ke_mean: Mean kinetic energy (eV)
            - ke_std: Kinetic energy standard deviation (eV)
            - pe_mean: Mean potential energy (eV)
            - pe_std: Potential energy standard deviation (eV)
            - ke_pe_correlation: Correlation coefficient (expect negative)
            - energy_partition_ratio: <KE> / <PE> (expect ~ 0 for condensed phase)

    Example:
        >>> ke = np.array([10.0, 11.0, 9.0, 10.0])
        >>> pe = np.array([-110.0, -111.0, -109.0, -110.0])
        >>> stats = compute_kinetic_potential_stability(ke, pe)
        >>> print(f"KE/PE correlation: {stats['ke_pe_correlation']:.3f}")
    """
    if len(kinetic_energies) != len(potential_energies):
        raise ValueError(
            f"Kinetic and potential energy arrays must have same length: "
            f"{len(kinetic_energies)} vs {len(potential_energies)}"
        )

    if len(kinetic_energies) < 2:
        raise ValueError(f"Need at least 2 energy values, got {len(kinetic_energies)}")

    # Basic statistics
    ke_mean = np.mean(kinetic_energies)
    ke_std = np.std(kinetic_energies)
    pe_mean = np.mean(potential_energies)
    pe_std = np.std(potential_energies)

    # Correlation between KE and PE (should be negative - energy exchange)
    if ke_std > 1e-10 and pe_std > 1e-10:
        correlation = np.corrcoef(kinetic_energies, potential_energies)[0, 1]
    else:
        correlation = 0.0

    # Energy partition ratio
    if abs(pe_mean) > 1e-10:
        partition_ratio = ke_mean / abs(pe_mean)
    else:
        partition_ratio = 0.0

    return {
        'ke_mean': ke_mean,
        'ke_std': ke_std,
        'pe_mean': pe_mean,
        'pe_std': pe_std,
        'ke_pe_correlation': correlation,
        'energy_partition_ratio': partition_ratio,
    }


def compute_time_resolved_drift(
    times: np.ndarray,
    energies: np.ndarray,
    method: str = 'linear_fit'
) -> Dict[str, float]:
    """
    Compute time-resolved energy drift characteristics.

    Args:
        times: Array of simulation times [n_steps] (ps)
        energies: Array of total energies [n_steps] (eV)
        method: Analysis method:
            - 'linear_fit': Linear regression to estimate drift rate
            - 'cumulative': Cumulative drift from initial energy

    Returns:
        Dictionary with time-resolved metrics:
            - drift_rate: Energy drift rate (eV/ps or %/ps)
            - drift_rate_std: Standard error of drift rate
            - r_squared: R² of linear fit (if method='linear_fit')
            - total_drift: Total drift over trajectory (eV or %)

    Example:
        >>> times = np.linspace(0, 10, 100)  # 10 ps
        >>> energies = -100.0 + 0.01 * times + np.random.randn(100) * 0.001
        >>> drift = compute_time_resolved_drift(times, energies)
        >>> print(f"Drift rate: {drift['drift_rate']:.6f} eV/ps")
    """
    if len(times) != len(energies):
        raise ValueError(f"Times and energies must have same length: {len(times)} vs {len(energies)}")

    if len(times) < 3:
        raise ValueError(f"Need at least 3 data points for drift analysis, got {len(times)}")

    if method == 'linear_fit':
        # Fit linear trend: E(t) = E0 + drift_rate * t
        coeffs = np.polyfit(times, energies, deg=1)
        drift_rate = coeffs[0]  # eV/ps
        e0 = coeffs[1]

        # Compute R²
        e_fit = np.polyval(coeffs, times)
        ss_res = np.sum((energies - e_fit) ** 2)
        ss_tot = np.sum((energies - np.mean(energies)) ** 2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-10 else 1.0

        # Standard error of drift rate
        n = len(times)
        residuals = energies - e_fit
        std_residuals = np.std(residuals)
        std_times = np.std(times)
        drift_rate_std = std_residuals / (std_times * np.sqrt(n)) if std_times > 1e-10 else 0.0

        # Total drift
        total_drift = energies[-1] - energies[0]

        return {
            'drift_rate': drift_rate,
            'drift_rate_std': drift_rate_std,
            'r_squared': r_squared,
            'total_drift': total_drift,
            'initial_energy': e0,
        }

    elif method == 'cumulative':
        # Cumulative drift from initial energy
        e0 = energies[0]
        cumulative_drift = energies - e0

        # Average drift rate
        total_time = times[-1] - times[0]
        if total_time > 1e-10:
            avg_drift_rate = (energies[-1] - e0) / total_time
        else:
            avg_drift_rate = 0.0

        return {
            'drift_rate': avg_drift_rate,
            'total_drift': energies[-1] - e0,
            'max_drift': np.max(np.abs(cumulative_drift)),
            'cumulative_drift': cumulative_drift,
        }

    else:
        raise ValueError(f"Unknown method: {method}")


def assess_energy_conservation(
    trajectory_data: Dict[str, List],
    tolerance_pct: float = 1.0,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive energy conservation assessment.

    This is the main function for validating energy conservation in MD simulations.
    It computes all relevant metrics and provides a pass/fail assessment.

    Args:
        trajectory_data: Trajectory data from NVEMDHarness
        tolerance_pct: Acceptable energy drift percentage (default: 1.0%)
        verbose: Print detailed report (default: True)

    Returns:
        Dictionary with comprehensive assessment:
            - passed: Boolean, True if all criteria met
            - energy_drift_pct: Total energy drift percentage
            - conservation_ratio: Energy conservation ratio
            - fluctuation_stats: Energy fluctuation statistics
            - ke_pe_stability: Kinetic/potential energy stability
            - time_resolved_drift: Time-resolved drift analysis
            - criteria: Dictionary of pass/fail for each criterion

    Example:
        >>> assessment = assess_energy_conservation(trajectory_data, tolerance_pct=1.0)
        >>> if assessment['passed']:
        >>>     print("Energy conservation PASSED")
        >>> else:
        >>>     print(f"Energy drift {assessment['energy_drift_pct']:.3f}% exceeds tolerance")
    """
    # Convert to numpy arrays
    times = np.array(trajectory_data['time'])
    total_energies = np.array(trajectory_data['total_energy'])
    kinetic_energies = np.array(trajectory_data['kinetic_energy'])
    potential_energies = np.array(trajectory_data['potential_energy'])

    # Compute all metrics
    energy_drift_pct = compute_energy_drift(total_energies, method='relative')
    energy_drift_max_pct = compute_energy_drift(total_energies, method='max_deviation')
    conservation_ratio = compute_energy_conservation_ratio(total_energies)
    fluctuation_stats = compute_energy_fluctuations(total_energies)
    ke_pe_stability = compute_kinetic_potential_stability(kinetic_energies, potential_energies)
    time_drift = compute_time_resolved_drift(times, total_energies, method='linear_fit')

    # Criteria evaluation
    criteria = {
        'drift_within_tolerance': abs(energy_drift_pct) <= tolerance_pct,
        'max_drift_within_tolerance': energy_drift_max_pct <= tolerance_pct * 1.5,  # Allow 1.5x for max
        'conservation_ratio_good': conservation_ratio >= 0.99,
        'ke_pe_anticorrelated': ke_pe_stability['ke_pe_correlation'] < 0,  # Should be negative
    }

    # Overall pass/fail
    passed = all(criteria.values())

    # Build assessment
    assessment = {
        'passed': passed,
        'energy_drift_pct': energy_drift_pct,
        'energy_drift_max_pct': energy_drift_max_pct,
        'conservation_ratio': conservation_ratio,
        'fluctuation_stats': fluctuation_stats,
        'ke_pe_stability': ke_pe_stability,
        'time_resolved_drift': time_drift,
        'criteria': criteria,
        'tolerance_pct': tolerance_pct,
    }

    # Verbose report
    if verbose:
        logger.info("=" * 70)
        logger.info("ENERGY CONSERVATION ASSESSMENT")
        logger.info("=" * 70)
        logger.info(f"Simulation time: {times[-1]:.3f} ps ({len(times)} steps)")
        logger.info(f"Initial energy: {total_energies[0]:.6f} eV")
        logger.info(f"Final energy: {total_energies[-1]:.6f} eV")
        logger.info("")
        logger.info("Energy Drift:")
        logger.info(f"  Total drift: {energy_drift_pct:+.4f}% (tolerance: ±{tolerance_pct}%)")
        logger.info(f"  Max deviation: {energy_drift_max_pct:.4f}%")
        logger.info(f"  Drift rate: {time_drift['drift_rate']:.6e} eV/ps")
        logger.info(f"  Status: {'PASS' if criteria['drift_within_tolerance'] else 'FAIL'}")
        logger.info("")
        logger.info("Energy Conservation:")
        logger.info(f"  Conservation ratio: {conservation_ratio:.6f} (target: > 0.99)")
        logger.info(f"  Energy std: {fluctuation_stats['std']:.6f} eV")
        logger.info(f"  Status: {'PASS' if criteria['conservation_ratio_good'] else 'FAIL'}")
        logger.info("")
        logger.info("Kinetic/Potential Energy:")
        logger.info(f"  Mean KE: {ke_pe_stability['ke_mean']:.6f} eV")
        logger.info(f"  Mean PE: {ke_pe_stability['pe_mean']:.6f} eV")
        logger.info(f"  KE-PE correlation: {ke_pe_stability['ke_pe_correlation']:+.4f} (expect < 0)")
        logger.info(f"  Status: {'PASS' if criteria['ke_pe_anticorrelated'] else 'FAIL'}")
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"OVERALL: {'PASSED' if passed else 'FAILED'}")
        logger.info("=" * 70)

    return assessment


__all__ = [
    'compute_energy_drift',
    'compute_energy_conservation_ratio',
    'compute_energy_fluctuations',
    'compute_kinetic_potential_stability',
    'compute_time_resolved_drift',
    'assess_energy_conservation',
]
