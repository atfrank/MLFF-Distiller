"""
Force Accuracy Metrics for MD Validation

This module provides comprehensive metrics for analyzing force prediction accuracy
during molecular dynamics simulations. Force accuracy is critical for stable MD
simulations and reliable prediction of dynamical properties.

Key Metrics:
- Force RMSE vs teacher model
- Force MAE (component-wise and magnitude)
- Angular error distribution
- Per-atom force errors
- Time-averaged force accuracy

Target Criteria (Production Quality):
- Force RMSE < 0.2 eV/Å for organic molecules
- Force MAE < 0.15 eV/Å
- Angular error < 15° (median)

Usage:
    from mlff_distiller.testing import compute_force_rmse, compute_force_mae

    # Compare student and teacher forces
    student_forces = student_calc.get_forces(atoms)
    teacher_forces = teacher_calc.get_forces(atoms)

    rmse = compute_force_rmse(student_forces, teacher_forces)
    mae = compute_force_mae(student_forces, teacher_forces)

    print(f"Force RMSE: {rmse:.4f} eV/Å")
    print(f"Force MAE: {mae:.4f} eV/Å")

Author: Testing & Benchmarking Engineer
Date: 2025-11-25
Issue: #37
"""

from typing import Dict, List, Optional, Tuple, Any
import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_force_rmse(
    predicted_forces: np.ndarray,
    reference_forces: np.ndarray,
    per_component: bool = False
) -> float:
    """
    Compute Root Mean Square Error of forces.

    Args:
        predicted_forces: Predicted forces [n_atoms, 3] or [n_frames, n_atoms, 3]
        reference_forces: Reference forces [n_atoms, 3] or [n_frames, n_atoms, 3]
        per_component: Return per-component RMSE instead of overall (default: False)

    Returns:
        Force RMSE in eV/Å (or array of 3 values if per_component=True)

    Example:
        >>> pred = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        >>> ref = np.array([[1.1, 2.1, 2.9], [3.9, 5.1, 6.1]])
        >>> rmse = compute_force_rmse(pred, ref)
        >>> print(f"Force RMSE: {rmse:.4f} eV/Å")
    """
    if predicted_forces.shape != reference_forces.shape:
        raise ValueError(
            f"Force arrays must have same shape: {predicted_forces.shape} vs {reference_forces.shape}"
        )

    # Compute squared errors
    squared_errors = (predicted_forces - reference_forces) ** 2

    if per_component:
        # RMSE per component (x, y, z)
        rmse = np.sqrt(np.mean(squared_errors, axis=tuple(range(squared_errors.ndim - 1))))
        return rmse
    else:
        # Overall RMSE
        rmse = np.sqrt(np.mean(squared_errors))
        return float(rmse)


def compute_force_mae(
    predicted_forces: np.ndarray,
    reference_forces: np.ndarray,
    per_component: bool = False
) -> float:
    """
    Compute Mean Absolute Error of forces.

    Args:
        predicted_forces: Predicted forces [n_atoms, 3] or [n_frames, n_atoms, 3]
        reference_forces: Reference forces [n_atoms, 3] or [n_frames, n_atoms, 3]
        per_component: Return per-component MAE instead of overall (default: False)

    Returns:
        Force MAE in eV/Å (or array of 3 values if per_component=True)

    Example:
        >>> pred = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        >>> ref = np.array([[1.1, 2.1, 2.9], [3.9, 5.1, 6.1]])
        >>> mae = compute_force_mae(pred, ref)
        >>> print(f"Force MAE: {mae:.4f} eV/Å")
    """
    if predicted_forces.shape != reference_forces.shape:
        raise ValueError(
            f"Force arrays must have same shape: {predicted_forces.shape} vs {reference_forces.shape}"
        )

    # Compute absolute errors
    abs_errors = np.abs(predicted_forces - reference_forces)

    if per_component:
        # MAE per component (x, y, z)
        mae = np.mean(abs_errors, axis=tuple(range(abs_errors.ndim - 1)))
        return mae
    else:
        # Overall MAE
        mae = np.mean(abs_errors)
        return float(mae)


def compute_force_magnitude_error(
    predicted_forces: np.ndarray,
    reference_forces: np.ndarray
) -> Dict[str, float]:
    """
    Compute errors in force magnitudes.

    Args:
        predicted_forces: Predicted forces [n_atoms, 3] or [n_frames, n_atoms, 3]
        reference_forces: Reference forces [n_atoms, 3] or [n_frames, n_atoms, 3]

    Returns:
        Dictionary with magnitude error metrics:
            - rmse: RMSE of force magnitudes
            - mae: MAE of force magnitudes
            - relative_error: Mean relative error in magnitudes (%)

    Example:
        >>> pred = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        >>> ref = np.array([[1.1, 2.1, 2.9], [3.9, 5.1, 6.1]])
        >>> errors = compute_force_magnitude_error(pred, ref)
        >>> print(f"Magnitude RMSE: {errors['rmse']:.4f} eV/Å")
    """
    if predicted_forces.shape != reference_forces.shape:
        raise ValueError(
            f"Force arrays must have same shape: {predicted_forces.shape} vs {reference_forces.shape}"
        )

    # Reshape to [n_samples, 3]
    pred_flat = predicted_forces.reshape(-1, 3)
    ref_flat = reference_forces.reshape(-1, 3)

    # Compute magnitudes
    pred_mag = np.linalg.norm(pred_flat, axis=1)
    ref_mag = np.linalg.norm(ref_flat, axis=1)

    # Magnitude errors
    mag_errors = pred_mag - ref_mag
    rmse = np.sqrt(np.mean(mag_errors ** 2))
    mae = np.mean(np.abs(mag_errors))

    # Relative error (avoid division by zero)
    nonzero_mask = ref_mag > 1e-6
    if np.any(nonzero_mask):
        relative_errors = np.abs(mag_errors[nonzero_mask]) / ref_mag[nonzero_mask]
        relative_error = 100.0 * np.mean(relative_errors)
    else:
        relative_error = 0.0

    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'relative_error': float(relative_error),
    }


def compute_angular_error(
    predicted_forces: np.ndarray,
    reference_forces: np.ndarray,
    return_distribution: bool = False
) -> float:
    """
    Compute angular error between predicted and reference force vectors.

    The angular error is the angle (in degrees) between force vectors,
    which measures directional accuracy independent of magnitude.

    Args:
        predicted_forces: Predicted forces [n_atoms, 3] or [n_frames, n_atoms, 3]
        reference_forces: Reference forces [n_atoms, 3] or [n_frames, n_atoms, 3]
        return_distribution: Return full distribution instead of median (default: False)

    Returns:
        Median angular error in degrees (or array of all errors if return_distribution=True)

    Example:
        >>> pred = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        >>> ref = np.array([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]])
        >>> angle = compute_angular_error(pred, ref)
        >>> print(f"Angular error: {angle:.2f}°")
    """
    if predicted_forces.shape != reference_forces.shape:
        raise ValueError(
            f"Force arrays must have same shape: {predicted_forces.shape} vs {reference_forces.shape}"
        )

    # Reshape to [n_samples, 3]
    pred_flat = predicted_forces.reshape(-1, 3)
    ref_flat = reference_forces.reshape(-1, 3)

    # Compute magnitudes
    pred_mag = np.linalg.norm(pred_flat, axis=1)
    ref_mag = np.linalg.norm(ref_flat, axis=1)

    # Filter out near-zero forces (angular error undefined)
    min_magnitude = 1e-6
    valid_mask = (pred_mag > min_magnitude) & (ref_mag > min_magnitude)

    if not np.any(valid_mask):
        logger.warning("No valid force vectors for angular error computation (all forces near zero)")
        return 0.0 if not return_distribution else np.array([])

    pred_valid = pred_flat[valid_mask]
    ref_valid = ref_flat[valid_mask]
    pred_mag_valid = pred_mag[valid_mask]
    ref_mag_valid = ref_mag[valid_mask]

    # Compute cosine of angle: cos(θ) = (F1 · F2) / (|F1| |F2|)
    dot_products = np.sum(pred_valid * ref_valid, axis=1)
    cos_angles = dot_products / (pred_mag_valid * ref_mag_valid)

    # Clamp to [-1, 1] to avoid numerical errors in arccos
    cos_angles = np.clip(cos_angles, -1.0, 1.0)

    # Angular errors in degrees
    angular_errors = np.rad2deg(np.arccos(cos_angles))

    if return_distribution:
        return angular_errors
    else:
        return float(np.median(angular_errors))


def compute_per_atom_force_errors(
    predicted_forces: np.ndarray,
    reference_forces: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute per-atom force errors.

    Args:
        predicted_forces: Predicted forces [n_atoms, 3]
        reference_forces: Reference forces [n_atoms, 3]

    Returns:
        Dictionary with per-atom errors:
            - rmse: RMSE per atom [n_atoms]
            - mae: MAE per atom [n_atoms]
            - magnitude_error: Error in force magnitude per atom [n_atoms]
            - angular_error: Angular error per atom [n_atoms]

    Example:
        >>> pred = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        >>> ref = np.array([[1.1, 2.1, 2.9], [3.9, 5.1, 6.1]])
        >>> errors = compute_per_atom_force_errors(pred, ref)
        >>> print(f"Atom 0 RMSE: {errors['rmse'][0]:.4f} eV/Å")
    """
    if predicted_forces.shape != reference_forces.shape:
        raise ValueError(
            f"Force arrays must have same shape: {predicted_forces.shape} vs {reference_forces.shape}"
        )

    if predicted_forces.ndim != 2 or predicted_forces.shape[1] != 3:
        raise ValueError(f"Expected shape [n_atoms, 3], got {predicted_forces.shape}")

    n_atoms = predicted_forces.shape[0]

    # Per-atom RMSE
    squared_errors = (predicted_forces - reference_forces) ** 2
    rmse_per_atom = np.sqrt(np.mean(squared_errors, axis=1))

    # Per-atom MAE
    abs_errors = np.abs(predicted_forces - reference_forces)
    mae_per_atom = np.mean(abs_errors, axis=1)

    # Per-atom magnitude error
    pred_mag = np.linalg.norm(predicted_forces, axis=1)
    ref_mag = np.linalg.norm(reference_forces, axis=1)
    mag_error_per_atom = np.abs(pred_mag - ref_mag)

    # Per-atom angular error
    angular_errors = np.zeros(n_atoms)
    for i in range(n_atoms):
        pred_vec = predicted_forces[i]
        ref_vec = reference_forces[i]
        pred_mag_i = np.linalg.norm(pred_vec)
        ref_mag_i = np.linalg.norm(ref_vec)

        if pred_mag_i > 1e-6 and ref_mag_i > 1e-6:
            cos_angle = np.dot(pred_vec, ref_vec) / (pred_mag_i * ref_mag_i)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angular_errors[i] = np.rad2deg(np.arccos(cos_angle))
        else:
            angular_errors[i] = 0.0

    return {
        'rmse': rmse_per_atom,
        'mae': mae_per_atom,
        'magnitude_error': mag_error_per_atom,
        'angular_error': angular_errors,
    }


def compute_force_correlation(
    predicted_forces: np.ndarray,
    reference_forces: np.ndarray
) -> Dict[str, float]:
    """
    Compute correlation metrics between predicted and reference forces.

    Args:
        predicted_forces: Predicted forces [n_atoms, 3] or [n_frames, n_atoms, 3]
        reference_forces: Reference forces [n_atoms, 3] or [n_frames, n_atoms, 3]

    Returns:
        Dictionary with correlation metrics:
            - pearson_r: Pearson correlation coefficient
            - r_squared: R² (coefficient of determination)
            - cosine_similarity: Mean cosine similarity

    Example:
        >>> pred = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        >>> ref = np.array([[1.1, 2.1, 2.9], [3.9, 5.1, 6.1]])
        >>> corr = compute_force_correlation(pred, ref)
        >>> print(f"R²: {corr['r_squared']:.4f}")
    """
    if predicted_forces.shape != reference_forces.shape:
        raise ValueError(
            f"Force arrays must have same shape: {predicted_forces.shape} vs {reference_forces.shape}"
        )

    # Flatten to 1D
    pred_flat = predicted_forces.flatten()
    ref_flat = reference_forces.flatten()

    # Pearson correlation
    if len(pred_flat) > 1 and np.std(pred_flat) > 1e-10 and np.std(ref_flat) > 1e-10:
        pearson_r = float(np.corrcoef(pred_flat, ref_flat)[0, 1])
    else:
        pearson_r = 0.0

    # R² (coefficient of determination)
    ss_res = np.sum((ref_flat - pred_flat) ** 2)
    ss_tot = np.sum((ref_flat - np.mean(ref_flat)) ** 2)
    r_squared = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-10 else 0.0

    # Cosine similarity (averaged over force vectors)
    pred_vectors = predicted_forces.reshape(-1, 3)
    ref_vectors = reference_forces.reshape(-1, 3)

    pred_mag = np.linalg.norm(pred_vectors, axis=1)
    ref_mag = np.linalg.norm(ref_vectors, axis=1)

    valid_mask = (pred_mag > 1e-6) & (ref_mag > 1e-6)
    if np.any(valid_mask):
        dot_products = np.sum(pred_vectors[valid_mask] * ref_vectors[valid_mask], axis=1)
        cosine_sims = dot_products / (pred_mag[valid_mask] * ref_mag[valid_mask])
        cosine_similarity = float(np.mean(cosine_sims))
    else:
        cosine_similarity = 0.0

    return {
        'pearson_r': pearson_r,
        'r_squared': r_squared,
        'cosine_similarity': cosine_similarity,
    }


def assess_force_accuracy(
    predicted_forces: np.ndarray,
    reference_forces: np.ndarray,
    rmse_tolerance: float = 0.2,
    mae_tolerance: float = 0.15,
    angular_tolerance: float = 15.0,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive force accuracy assessment.

    This is the main function for validating force accuracy against reference
    (teacher) forces. It computes all relevant metrics and provides pass/fail assessment.

    Args:
        predicted_forces: Predicted forces [n_atoms, 3] or [n_frames, n_atoms, 3]
        reference_forces: Reference forces [n_atoms, 3] or [n_frames, n_atoms, 3]
        rmse_tolerance: Acceptable RMSE in eV/Å (default: 0.2)
        mae_tolerance: Acceptable MAE in eV/Å (default: 0.15)
        angular_tolerance: Acceptable median angular error in degrees (default: 15.0)
        verbose: Print detailed report (default: True)

    Returns:
        Dictionary with comprehensive assessment:
            - passed: Boolean, True if all criteria met
            - rmse: Force RMSE
            - mae: Force MAE
            - magnitude_errors: Magnitude error metrics
            - angular_error_median: Median angular error
            - correlation: Correlation metrics
            - criteria: Dictionary of pass/fail for each criterion

    Example:
        >>> assessment = assess_force_accuracy(student_forces, teacher_forces)
        >>> if assessment['passed']:
        >>>     print("Force accuracy PASSED")
        >>> else:
        >>>     print(f"Force RMSE {assessment['rmse']:.3f} exceeds tolerance")
    """
    # Compute all metrics
    rmse = compute_force_rmse(predicted_forces, reference_forces)
    mae = compute_force_mae(predicted_forces, reference_forces)
    magnitude_errors = compute_force_magnitude_error(predicted_forces, reference_forces)
    angular_error_median = compute_angular_error(predicted_forces, reference_forces)
    angular_errors_dist = compute_angular_error(predicted_forces, reference_forces, return_distribution=True)
    correlation = compute_force_correlation(predicted_forces, reference_forces)

    # Criteria evaluation
    criteria = {
        'rmse_within_tolerance': rmse <= rmse_tolerance,
        'mae_within_tolerance': mae <= mae_tolerance,
        'angular_error_acceptable': angular_error_median <= angular_tolerance,
        'high_correlation': correlation['r_squared'] >= 0.9,
    }

    # Overall pass/fail
    passed = all(criteria.values())

    # Additional statistics
    angular_percentiles = {
        'p50': float(np.median(angular_errors_dist)) if len(angular_errors_dist) > 0 else 0.0,
        'p75': float(np.percentile(angular_errors_dist, 75)) if len(angular_errors_dist) > 0 else 0.0,
        'p90': float(np.percentile(angular_errors_dist, 90)) if len(angular_errors_dist) > 0 else 0.0,
        'p95': float(np.percentile(angular_errors_dist, 95)) if len(angular_errors_dist) > 0 else 0.0,
    }

    # Build assessment
    assessment = {
        'passed': passed,
        'rmse': rmse,
        'mae': mae,
        'magnitude_errors': magnitude_errors,
        'angular_error_median': angular_error_median,
        'angular_percentiles': angular_percentiles,
        'correlation': correlation,
        'criteria': criteria,
        'tolerances': {
            'rmse': rmse_tolerance,
            'mae': mae_tolerance,
            'angular': angular_tolerance,
        },
    }

    # Verbose report
    if verbose:
        logger.info("=" * 70)
        logger.info("FORCE ACCURACY ASSESSMENT")
        logger.info("=" * 70)
        logger.info(f"Number of force vectors: {predicted_forces.reshape(-1, 3).shape[0]}")
        logger.info("")
        logger.info("Component-wise Errors:")
        logger.info(f"  RMSE: {rmse:.4f} eV/Å (tolerance: {rmse_tolerance} eV/Å)")
        logger.info(f"  MAE:  {mae:.4f} eV/Å (tolerance: {mae_tolerance} eV/Å)")
        logger.info(f"  RMSE status: {'PASS' if criteria['rmse_within_tolerance'] else 'FAIL'}")
        logger.info(f"  MAE status: {'PASS' if criteria['mae_within_tolerance'] else 'FAIL'}")
        logger.info("")
        logger.info("Magnitude Errors:")
        logger.info(f"  RMSE: {magnitude_errors['rmse']:.4f} eV/Å")
        logger.info(f"  MAE:  {magnitude_errors['mae']:.4f} eV/Å")
        logger.info(f"  Relative: {magnitude_errors['relative_error']:.2f}%")
        logger.info("")
        logger.info("Angular Errors:")
        logger.info(f"  Median: {angular_error_median:.2f}° (tolerance: {angular_tolerance}°)")
        logger.info(f"  75th percentile: {angular_percentiles['p75']:.2f}°")
        logger.info(f"  90th percentile: {angular_percentiles['p90']:.2f}°")
        logger.info(f"  95th percentile: {angular_percentiles['p95']:.2f}°")
        logger.info(f"  Status: {'PASS' if criteria['angular_error_acceptable'] else 'FAIL'}")
        logger.info("")
        logger.info("Correlation Metrics:")
        logger.info(f"  Pearson r: {correlation['pearson_r']:.6f}")
        logger.info(f"  R²: {correlation['r_squared']:.6f} (target: > 0.9)")
        logger.info(f"  Cosine similarity: {correlation['cosine_similarity']:.6f}")
        logger.info(f"  Status: {'PASS' if criteria['high_correlation'] else 'FAIL'}")
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"OVERALL: {'PASSED' if passed else 'FAILED'}")
        logger.info("=" * 70)

    return assessment


__all__ = [
    'compute_force_rmse',
    'compute_force_mae',
    'compute_force_magnitude_error',
    'compute_angular_error',
    'compute_per_atom_force_errors',
    'compute_force_correlation',
    'assess_force_accuracy',
]
