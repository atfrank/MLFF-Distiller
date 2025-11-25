"""
Unit tests for Energy Conservation Metrics

Tests energy conservation metrics for MD validation.

Author: Testing & Benchmarking Engineer
Date: 2025-11-25
Issue: #37
"""

import pytest
import numpy as np

from mlff_distiller.testing import (
    compute_energy_drift,
    compute_energy_conservation_ratio,
    compute_energy_fluctuations,
    compute_kinetic_potential_stability,
    compute_time_resolved_drift,
    assess_energy_conservation,
)


class TestEnergyDrift:
    """Test energy drift computation."""

    def test_zero_drift(self):
        """Test with perfectly conserved energy."""
        energies = np.ones(100) * -100.0
        drift = compute_energy_drift(energies, method='relative')
        assert abs(drift) < 1e-10

    def test_positive_drift(self):
        """Test with positive drift."""
        energies = np.linspace(-100.0, -99.0, 100)
        drift = compute_energy_drift(energies, method='relative')
        assert abs(drift - 1.0) < 0.01  # ~1% drift

    def test_negative_drift(self):
        """Test with negative drift."""
        energies = np.linspace(-100.0, -101.0, 100)
        drift = compute_energy_drift(energies, method='relative')
        assert abs(drift - (-1.0)) < 0.01  # ~-1% drift

    def test_absolute_drift(self):
        """Test absolute drift method."""
        energies = np.array([-100.0, -100.5, -101.0])
        drift = compute_energy_drift(energies, method='absolute')
        assert abs(drift - (-1.0)) < 1e-10

    def test_max_deviation(self):
        """Test max deviation method."""
        energies = np.array([-100.0, -99.5, -100.0, -100.5])
        drift = compute_energy_drift(energies, method='max_deviation')
        # Max deviation is 0.5, which is 0.5% of 100
        assert abs(drift - 0.5) < 0.01

    def test_invalid_method(self):
        """Test invalid drift method raises error."""
        energies = np.ones(10) * -100.0
        with pytest.raises(ValueError):
            compute_energy_drift(energies, method='invalid')

    def test_too_few_energies(self):
        """Test with too few energy values."""
        with pytest.raises(ValueError):
            compute_energy_drift(np.array([100.0]))


class TestEnergyConservationRatio:
    """Test energy conservation ratio computation."""

    def test_perfect_conservation(self):
        """Test with perfectly conserved energy."""
        energies = np.ones(100) * -100.0
        ratio = compute_energy_conservation_ratio(energies)
        assert ratio > 0.999

    def test_small_fluctuations(self):
        """Test with small fluctuations."""
        np.random.seed(42)
        energies = -100.0 + np.random.randn(100) * 0.01  # 0.01 eV std
        ratio = compute_energy_conservation_ratio(energies)
        # Std ~ 0.01 eV, mean ~ 100 eV → ratio ~ 1 - 0.0001
        assert ratio > 0.999

    def test_large_fluctuations(self):
        """Test with large fluctuations."""
        np.random.seed(42)
        energies = -100.0 + np.random.randn(100) * 5.0  # 5 eV std
        ratio = compute_energy_conservation_ratio(energies)
        # Std ~ 5 eV, mean ~ 100 eV → ratio ~ 1 - 0.05 = 0.95
        assert 0.90 < ratio < 0.96

    def test_window_analysis(self):
        """Test with rolling window analysis."""
        energies = np.ones(100) * -100.0
        ratio = compute_energy_conservation_ratio(energies, window=10)
        assert ratio > 0.999


class TestEnergyFluctuations:
    """Test energy fluctuation statistics."""

    def test_zero_fluctuations(self):
        """Test with constant energy."""
        energies = np.ones(100) * -100.0
        stats = compute_energy_fluctuations(energies)

        assert stats['std'] < 1e-10
        assert stats['range'] < 1e-10
        assert stats['mean_abs_dev'] < 1e-10
        assert stats['coefficient_of_variation'] < 1e-10

    def test_known_fluctuations(self):
        """Test with known fluctuation pattern."""
        np.random.seed(42)
        energies = -100.0 + np.random.randn(1000) * 2.0  # std = 2.0 eV

        stats = compute_energy_fluctuations(energies)

        # Check std is approximately 2.0
        assert 1.9 < stats['std'] < 2.1

        # Coefficient of variation ~ 2/100 = 0.02
        assert 0.015 < stats['coefficient_of_variation'] < 0.025

        # Range should be ~6-8 sigma for 1000 samples
        assert 8.0 < stats['range'] < 16.0


class TestKineticPotentialStability:
    """Test kinetic/potential energy stability analysis."""

    def test_anticorrelated_energies(self):
        """Test with anticorrelated KE and PE (expected in MD)."""
        # KE and PE exchange: when KE increases, PE decreases
        ke = np.sin(np.linspace(0, 10, 100)) * 10 + 50
        pe = -np.sin(np.linspace(0, 10, 100)) * 10 - 150

        stats = compute_kinetic_potential_stability(ke, pe)

        # Should be anticorrelated
        assert stats['ke_pe_correlation'] < 0

        # Check means
        assert 45 < stats['ke_mean'] < 55
        assert -155 < stats['pe_mean'] < -145

    def test_mismatched_lengths(self):
        """Test with mismatched array lengths."""
        ke = np.ones(10)
        pe = np.ones(20)

        with pytest.raises(ValueError):
            compute_kinetic_potential_stability(ke, pe)

    def test_energy_partition_ratio(self):
        """Test energy partition ratio calculation."""
        ke = np.ones(100) * 50
        pe = np.ones(100) * -200

        stats = compute_kinetic_potential_stability(ke, pe)

        # Partition ratio = 50 / 200 = 0.25
        assert abs(stats['energy_partition_ratio'] - 0.25) < 0.01


class TestTimeResolvedDrift:
    """Test time-resolved drift analysis."""

    def test_linear_drift(self):
        """Test with linear energy drift."""
        times = np.linspace(0, 10, 100)  # 10 ps
        energies = -100.0 + 0.1 * times  # 0.1 eV/ps drift

        drift = compute_time_resolved_drift(times, energies, method='linear_fit')

        # Drift rate should be ~0.1 eV/ps
        assert abs(drift['drift_rate'] - 0.1) < 0.01

        # R² should be very high (linear trend)
        assert drift['r_squared'] > 0.99

        # Total drift should be ~1.0 eV
        assert abs(drift['total_drift'] - 1.0) < 0.1

    def test_no_drift(self):
        """Test with no drift."""
        times = np.linspace(0, 10, 100)
        energies = np.ones(100) * -100.0

        drift = compute_time_resolved_drift(times, energies, method='linear_fit')

        # Drift rate should be ~0
        assert abs(drift['drift_rate']) < 0.01

        # Total drift should be ~0
        assert abs(drift['total_drift']) < 0.01

    def test_cumulative_drift(self):
        """Test cumulative drift method."""
        times = np.linspace(0, 10, 100)
        energies = -100.0 + 0.1 * times

        drift = compute_time_resolved_drift(times, energies, method='cumulative')

        # Average drift rate
        assert abs(drift['drift_rate'] - 0.1) < 0.01

        # Total drift
        assert abs(drift['total_drift'] - 1.0) < 0.1

        # Max drift
        assert abs(drift['max_drift'] - 1.0) < 0.1

    def test_invalid_method(self):
        """Test invalid method raises error."""
        times = np.linspace(0, 10, 100)
        energies = np.ones(100)

        with pytest.raises(ValueError):
            compute_time_resolved_drift(times, energies, method='invalid')

    def test_too_few_points(self):
        """Test with too few data points."""
        times = np.array([0, 1])
        energies = np.array([100, 101])

        with pytest.raises(ValueError):
            compute_time_resolved_drift(times, energies)


class TestAssessEnergyConservation:
    """Test comprehensive energy conservation assessment."""

    @pytest.fixture
    def perfect_trajectory(self):
        """Create perfect energy conservation trajectory."""
        n_steps = 100
        # Add small anticorrelated fluctuations in KE and PE
        ke = 50.0 + np.sin(np.linspace(0, 10, n_steps)) * 0.001
        pe = -150.0 - np.sin(np.linspace(0, 10, n_steps)) * 0.001
        trajectory_data = {
            'time': list(np.linspace(0, 10, n_steps)),
            'total_energy': list(ke + pe),  # Should be nearly constant
            'kinetic_energy': list(ke),
            'potential_energy': list(pe),
        }
        return trajectory_data

    @pytest.fixture
    def realistic_trajectory(self):
        """Create realistic trajectory with small fluctuations."""
        np.random.seed(42)
        n_steps = 100

        # Small energy fluctuations around -100 eV
        total_energy = -100.0 + np.random.randn(n_steps) * 0.05

        # Anticorrelated KE and PE
        ke_base = np.sin(np.linspace(0, 10, n_steps)) * 2 + 50
        pe = total_energy - ke_base

        trajectory_data = {
            'time': list(np.linspace(0, 10, n_steps)),
            'total_energy': list(total_energy),
            'kinetic_energy': list(ke_base),
            'potential_energy': list(pe),
        }
        return trajectory_data

    def test_perfect_conservation_passes(self, perfect_trajectory):
        """Test that perfect conservation passes all criteria."""
        assessment = assess_energy_conservation(
            perfect_trajectory,
            tolerance_pct=1.0,
            verbose=False
        )

        assert assessment['passed']
        assert assessment['criteria']['drift_within_tolerance']
        assert assessment['criteria']['conservation_ratio_good']
        assert abs(assessment['energy_drift_pct']) < 1e-6

    def test_realistic_trajectory_passes(self, realistic_trajectory):
        """Test that realistic trajectory passes."""
        assessment = assess_energy_conservation(
            realistic_trajectory,
            tolerance_pct=1.0,
            verbose=False
        )

        # Should pass with 1% tolerance
        assert assessment['passed']
        assert abs(assessment['energy_drift_pct']) < 0.5

    def test_failed_conservation(self):
        """Test trajectory that fails conservation."""
        # Linear drift of 5%
        n_steps = 100
        energies = np.linspace(-100.0, -95.0, n_steps)

        trajectory_data = {
            'time': list(np.linspace(0, 10, n_steps)),
            'total_energy': list(energies),
            'kinetic_energy': list(np.ones(n_steps) * 50.0),
            'potential_energy': list(energies - 50.0),
        }

        assessment = assess_energy_conservation(
            trajectory_data,
            tolerance_pct=1.0,
            verbose=False
        )

        # Should fail due to excessive drift
        assert not assessment['passed']
        assert not assessment['criteria']['drift_within_tolerance']
        assert abs(assessment['energy_drift_pct']) > 4.0

    def test_assessment_fields(self, realistic_trajectory):
        """Test that assessment contains all expected fields."""
        assessment = assess_energy_conservation(
            realistic_trajectory,
            tolerance_pct=1.0,
            verbose=False
        )

        expected_fields = [
            'passed',
            'energy_drift_pct',
            'energy_drift_max_pct',
            'conservation_ratio',
            'fluctuation_stats',
            'ke_pe_stability',
            'time_resolved_drift',
            'criteria',
            'tolerance_pct',
        ]

        for field in expected_fields:
            assert field in assessment, f"Missing field: {field}"

    def test_verbose_output(self, realistic_trajectory, caplog):
        """Test that verbose mode produces log output."""
        import logging
        caplog.set_level(logging.INFO)

        assess_energy_conservation(
            realistic_trajectory,
            tolerance_pct=1.0,
            verbose=True
        )

        # Check that log contains expected strings
        log_text = caplog.text
        assert 'ENERGY CONSERVATION ASSESSMENT' in log_text
        assert 'Energy Drift:' in log_text
        assert 'OVERALL:' in log_text


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_arrays(self):
        """Test with empty arrays."""
        with pytest.raises(ValueError):
            compute_energy_drift(np.array([]))

        with pytest.raises(ValueError):
            compute_energy_conservation_ratio(np.array([]))

    def test_single_value(self):
        """Test with single value."""
        with pytest.raises(ValueError):
            compute_energy_drift(np.array([100.0]))

    def test_nan_values(self):
        """Test handling of NaN values."""
        energies = np.array([100.0, np.nan, 101.0])
        # NaN will propagate through calculations
        drift = compute_energy_drift(energies)
        # Result may be NaN or finite depending on implementation
        # Just check it doesn't crash
        assert drift is not None

    def test_inf_values(self):
        """Test handling of Inf values."""
        energies = np.array([100.0, np.inf, 101.0])
        drift = compute_energy_drift(energies)
        # Result may be inf or finite depending on implementation
        # Just check it doesn't crash
        assert drift is not None
