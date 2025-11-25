"""
Unit tests for Force Accuracy Metrics

Tests force accuracy metrics for MD validation.

Author: Testing & Benchmarking Engineer
Date: 2025-11-25
Issue: #37
"""

import pytest
import numpy as np

from mlff_distiller.testing import (
    compute_force_rmse,
    compute_force_mae,
    compute_force_magnitude_error,
    compute_angular_error,
    compute_per_atom_force_errors,
    compute_force_correlation,
    assess_force_accuracy,
)


class TestForceRMSE:
    """Test force RMSE computation."""

    def test_zero_error(self):
        """Test with identical forces."""
        forces = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        rmse = compute_force_rmse(forces, forces)
        assert rmse < 1e-10

    def test_known_error(self):
        """Test with known error."""
        pred = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        ref = np.array([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]])

        rmse = compute_force_rmse(pred, ref)

        # All errors are 0.1 → RMSE = sqrt(mean(0.1²)) = 0.1
        assert abs(rmse - 0.1) < 1e-10

    def test_3d_trajectory(self):
        """Test with 3D trajectory [n_frames, n_atoms, 3]."""
        np.random.seed(42)
        pred = np.random.randn(10, 5, 3)
        ref = pred + 0.1  # Add constant error

        rmse = compute_force_rmse(pred, ref)
        assert abs(rmse - 0.1) < 1e-6

    def test_per_component(self):
        """Test per-component RMSE."""
        pred = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        ref = np.array([[1.1, 2.0, 3.0], [4.1, 5.0, 6.0]])

        rmse_components = compute_force_rmse(pred, ref, per_component=True)

        # Only x-component has error
        assert abs(rmse_components[0] - 0.1) < 1e-10
        assert rmse_components[1] < 1e-10
        assert rmse_components[2] < 1e-10

    def test_shape_mismatch(self):
        """Test that shape mismatch raises error."""
        pred = np.ones((5, 3))
        ref = np.ones((6, 3))

        with pytest.raises(ValueError):
            compute_force_rmse(pred, ref)


class TestForceMAE:
    """Test force MAE computation."""

    def test_zero_error(self):
        """Test with identical forces."""
        forces = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mae = compute_force_mae(forces, forces)
        assert mae < 1e-10

    def test_known_error(self):
        """Test with known error."""
        pred = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        ref = np.array([[1.2, 2.1, 2.9], [3.9, 5.2, 6.1]])

        mae = compute_force_mae(pred, ref)

        # Mean of absolute errors: |0.2| + |0.1| + |0.1| + |0.1| + |0.2| + |0.1| / 6
        expected_mae = (0.2 + 0.1 + 0.1 + 0.1 + 0.2 + 0.1) / 6
        assert abs(mae - expected_mae) < 1e-10

    def test_per_component(self):
        """Test per-component MAE."""
        pred = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        ref = np.array([[1.2, 2.0, 3.0], [4.2, 5.0, 6.0]])

        mae_components = compute_force_mae(pred, ref, per_component=True)

        # Only x-component has error
        assert abs(mae_components[0] - 0.2) < 1e-10
        assert mae_components[1] < 1e-10
        assert mae_components[2] < 1e-10


class TestForceMagnitudeError:
    """Test force magnitude error computation."""

    def test_zero_magnitude_error(self):
        """Test with forces of same magnitude."""
        pred = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        ref = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

        errors = compute_force_magnitude_error(pred, ref)

        # Magnitudes are same (both norm = 1), just different directions
        assert errors['rmse'] < 1e-10
        assert errors['mae'] < 1e-10

    def test_known_magnitude_error(self):
        """Test with known magnitude difference."""
        # Pred has magnitude 1, ref has magnitude 2
        pred = np.array([[1.0, 0.0, 0.0]])
        ref = np.array([[2.0, 0.0, 0.0]])

        errors = compute_force_magnitude_error(pred, ref)

        # Magnitude error = 1.0
        assert abs(errors['rmse'] - 1.0) < 1e-10
        assert abs(errors['mae'] - 1.0) < 1e-10

        # Relative error = |1.0 - 2.0| / 2.0 = 0.5 / 2.0 = 0.25 = 25%, then *100 = 25%
        # But we're computing |pred_mag - ref_mag| / ref_mag * 100
        # = |1.0 - 2.0| / 2.0 * 100 = 50%
        assert abs(errors['relative_error'] - 50.0) < 1e-6

    def test_relative_error(self):
        """Test relative magnitude error."""
        pred = np.array([[1.5, 0.0, 0.0]])
        ref = np.array([[2.0, 0.0, 0.0]])

        errors = compute_force_magnitude_error(pred, ref)

        # Relative error = 0.5 / 2.0 = 25%
        assert abs(errors['relative_error'] - 25.0) < 1e-6


class TestAngularError:
    """Test angular error computation."""

    def test_zero_angle(self):
        """Test with parallel forces."""
        pred = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        ref = np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])

        angle = compute_angular_error(pred, ref)
        assert angle < 1e-6

    def test_perpendicular_forces(self):
        """Test with perpendicular forces."""
        pred = np.array([[1.0, 0.0, 0.0]])
        ref = np.array([[0.0, 1.0, 0.0]])

        angle = compute_angular_error(pred, ref)

        # Should be 90 degrees
        assert abs(angle - 90.0) < 1e-6

    def test_opposite_forces(self):
        """Test with opposite forces."""
        pred = np.array([[1.0, 0.0, 0.0]])
        ref = np.array([[-1.0, 0.0, 0.0]])

        angle = compute_angular_error(pred, ref)

        # Should be 180 degrees
        assert abs(angle - 180.0) < 1e-6

    def test_45_degree_angle(self):
        """Test with 45 degree angle."""
        pred = np.array([[1.0, 0.0, 0.0]])
        ref = np.array([[1.0, 1.0, 0.0]]) / np.sqrt(2)

        angle = compute_angular_error(pred, ref)

        # Should be 45 degrees
        assert abs(angle - 45.0) < 1e-3

    def test_return_distribution(self):
        """Test returning full distribution."""
        pred = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        ref = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])

        angles = compute_angular_error(pred, ref, return_distribution=True)

        # Should have 3 angles
        assert len(angles) == 3
        assert angles[0] < 1e-6  # Parallel
        assert angles[1] < 1e-6  # Parallel
        assert 40 < angles[2] < 50  # ~45 degrees

    def test_zero_magnitude_forces(self):
        """Test with near-zero forces (undefined angle)."""
        pred = np.array([[1e-10, 0.0, 0.0]])
        ref = np.array([[0.0, 1e-10, 0.0]])

        angle = compute_angular_error(pred, ref)

        # Should return 0 (no valid vectors)
        assert angle == 0.0


class TestPerAtomForceErrors:
    """Test per-atom force error computation."""

    def test_uniform_error(self):
        """Test with uniform error across atoms."""
        pred = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        ref = np.array([[1.1, 0.0, 0.0], [0.0, 1.1, 0.0]])

        errors = compute_per_atom_force_errors(pred, ref)

        # Both atoms have same error
        assert abs(errors['rmse'][0] - errors['rmse'][1]) < 1e-10
        assert abs(errors['mae'][0] - errors['mae'][1]) < 1e-10

    def test_varying_error(self):
        """Test with different errors per atom."""
        pred = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        ref = np.array([[1.5, 0.0, 0.0], [0.0, 1.1, 0.0]])

        errors = compute_per_atom_force_errors(pred, ref)

        # Atom 0 has larger error
        assert errors['rmse'][0] > errors['rmse'][1]
        assert errors['magnitude_error'][0] > errors['magnitude_error'][1]

    def test_shape(self):
        """Test output shape."""
        pred = np.random.randn(10, 3)
        ref = np.random.randn(10, 3)

        errors = compute_per_atom_force_errors(pred, ref)

        assert errors['rmse'].shape == (10,)
        assert errors['mae'].shape == (10,)
        assert errors['magnitude_error'].shape == (10,)
        assert errors['angular_error'].shape == (10,)

    def test_invalid_shape(self):
        """Test with invalid shape raises error."""
        pred = np.random.randn(10, 2)  # Wrong last dimension
        ref = np.random.randn(10, 2)

        with pytest.raises(ValueError):
            compute_per_atom_force_errors(pred, ref)


class TestForceCorrelation:
    """Test force correlation metrics."""

    def test_perfect_correlation(self):
        """Test with perfectly correlated forces."""
        forces = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        corr = compute_force_correlation(forces, forces)

        assert abs(corr['pearson_r'] - 1.0) < 1e-6
        assert abs(corr['r_squared'] - 1.0) < 1e-6
        assert abs(corr['cosine_similarity'] - 1.0) < 1e-6

    def test_scaled_forces(self):
        """Test with scaled forces (high correlation, not perfect R²)."""
        pred = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        ref = pred * 2.0

        corr = compute_force_correlation(pred, ref)

        # Pearson correlation should be 1.0 (linear relationship)
        assert abs(corr['pearson_r'] - 1.0) < 1e-6

        # R² < 1 because not exact match
        assert corr['r_squared'] < 1.0

    def test_uncorrelated_forces(self):
        """Test with uncorrelated forces."""
        np.random.seed(42)
        pred = np.random.randn(100, 3)
        ref = np.random.randn(100, 3)

        corr = compute_force_correlation(pred, ref)

        # Should have low correlation
        assert abs(corr['pearson_r']) < 0.3
        assert corr['r_squared'] < 0.2

    def test_negative_correlation(self):
        """Test with negatively correlated forces."""
        pred = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        ref = -pred

        corr = compute_force_correlation(pred, ref)

        # Pearson correlation should be -1.0
        assert abs(corr['pearson_r'] - (-1.0)) < 1e-6


class TestAssessForceAccuracy:
    """Test comprehensive force accuracy assessment."""

    def test_perfect_accuracy_passes(self):
        """Test that perfect accuracy passes all criteria."""
        forces = np.random.randn(50, 3)

        assessment = assess_force_accuracy(
            forces, forces,
            rmse_tolerance=0.2,
            mae_tolerance=0.15,
            angular_tolerance=15.0,
            verbose=False
        )

        assert assessment['passed']
        assert assessment['criteria']['rmse_within_tolerance']
        assert assessment['criteria']['mae_within_tolerance']
        assert assessment['criteria']['angular_error_acceptable']
        assert assessment['criteria']['high_correlation']

    def test_small_error_passes(self):
        """Test that small errors pass."""
        np.random.seed(42)
        ref = np.random.randn(50, 3)
        pred = ref + np.random.randn(50, 3) * 0.05  # Small noise

        assessment = assess_force_accuracy(
            pred, ref,
            rmse_tolerance=0.2,
            verbose=False
        )

        # Should pass with generous tolerance
        assert assessment['passed']
        assert assessment['rmse'] < 0.2

    def test_large_error_fails(self):
        """Test that large errors fail."""
        np.random.seed(42)
        ref = np.random.randn(50, 3)
        pred = ref + np.random.randn(50, 3) * 2.0  # Large noise

        assessment = assess_force_accuracy(
            pred, ref,
            rmse_tolerance=0.2,
            verbose=False
        )

        # Should fail due to excessive error
        assert not assessment['passed']
        assert not assessment['criteria']['rmse_within_tolerance']

    def test_assessment_fields(self):
        """Test that assessment contains all expected fields."""
        pred = np.random.randn(50, 3)
        ref = np.random.randn(50, 3)

        assessment = assess_force_accuracy(
            pred, ref,
            verbose=False
        )

        expected_fields = [
            'passed',
            'rmse',
            'mae',
            'magnitude_errors',
            'angular_error_median',
            'angular_percentiles',
            'correlation',
            'criteria',
            'tolerances',
        ]

        for field in expected_fields:
            assert field in assessment, f"Missing field: {field}"

    def test_verbose_output(self, caplog):
        """Test that verbose mode produces log output."""
        import logging
        caplog.set_level(logging.INFO)

        pred = np.random.randn(50, 3)
        ref = pred + np.random.randn(50, 3) * 0.1

        assess_force_accuracy(
            pred, ref,
            verbose=True
        )

        # Check that log contains expected strings
        log_text = caplog.text
        assert 'FORCE ACCURACY ASSESSMENT' in log_text
        assert 'Component-wise Errors:' in log_text
        assert 'OVERALL:' in log_text

    def test_custom_tolerances(self):
        """Test with custom tolerances."""
        pred = np.random.randn(50, 3)
        ref = pred + 0.3  # 0.3 error

        # Strict tolerance - should fail
        assessment_strict = assess_force_accuracy(
            pred, ref,
            rmse_tolerance=0.2,
            verbose=False
        )
        assert not assessment_strict['passed']

        # Loose tolerance - should pass
        # Need to check what the actual errors are first
        assessment_loose = assess_force_accuracy(
            pred, ref,
            rmse_tolerance=0.5,
            mae_tolerance=0.5,
            angular_tolerance=180.0,  # Very loose
            verbose=False
        )
        # At minimum RMSE should pass
        assert assessment_loose['criteria']['rmse_within_tolerance']


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_arrays(self):
        """Test with empty arrays."""
        # Empty arrays should either raise or return NaN
        pred = np.array([]).reshape(0, 3)
        ref = np.array([]).reshape(0, 3)
        result = compute_force_rmse(pred, ref)
        # Should be NaN for empty input
        assert np.isnan(result)

    def test_single_atom(self):
        """Test with single atom."""
        pred = np.array([[1.0, 2.0, 3.0]])
        ref = np.array([[1.1, 2.1, 3.1]])

        rmse = compute_force_rmse(pred, ref)
        mae = compute_force_mae(pred, ref)

        # Should work fine
        assert rmse > 0
        assert mae > 0

    def test_nan_handling(self):
        """Test handling of NaN values."""
        pred = np.array([[1.0, np.nan, 3.0]])
        ref = np.array([[1.0, 2.0, 3.0]])

        rmse = compute_force_rmse(pred, ref)
        assert np.isnan(rmse)

    def test_zero_forces(self):
        """Test with all zero forces."""
        pred = np.zeros((10, 3))
        ref = np.zeros((10, 3))

        rmse = compute_force_rmse(pred, ref)
        assert rmse < 1e-10

        # Angular error should be 0 (no valid vectors)
        angle = compute_angular_error(pred, ref)
        assert angle == 0.0
