"""
Unit Tests for MD Benchmark Framework

Tests for the MD trajectory benchmarking utilities including:
1. BenchmarkResults data class and statistics
2. System creation utilities
3. MDTrajectoryBenchmark class
4. Comparison utilities
5. JSON serialization/deserialization

Author: Testing & Benchmark Engineer
Date: 2025-11-23
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
from ase import Atoms, units
from ase.build import bulk, molecule
from ase.calculators.calculator import Calculator
from ase.md.verlet import VelocityVerlet

from mlff_distiller.benchmarks.md_trajectory import (
    BenchmarkResults,
    MDProtocol,
    MDTrajectoryBenchmark,
    create_benchmark_system,
    compare_calculators,
)


class TestBenchmarkResults:
    """Test BenchmarkResults data class."""

    @pytest.fixture
    def sample_step_times(self):
        """Generate realistic step time distribution."""
        # Simulate realistic latency with some outliers
        rng = np.random.RandomState(42)
        base_times = rng.normal(10.0, 1.0, 950)  # Most steps ~10ms
        outliers = rng.normal(15.0, 2.0, 50)  # Some slower steps
        return np.concatenate([base_times, outliers]).tolist()

    @pytest.fixture
    def sample_energies(self):
        """Generate sample energy trajectory."""
        rng = np.random.RandomState(42)
        # Small drift in energy (realistic for NVE)
        return (-100.0 + 0.001 * np.arange(1000) + rng.normal(0, 0.01, 1000)).tolist()

    def test_benchmark_results_initialization(self, sample_step_times):
        """Test BenchmarkResults initialization and statistics computation."""
        results = BenchmarkResults(
            name="test_benchmark",
            protocol="NVE",
            n_steps=1000,
            n_atoms=64,
            system_type="silicon",
            device="cuda",
            step_times_ms=sample_step_times,
            memory_before_gb=2.0,
            memory_after_gb=2.05,
            peak_memory_gb=2.1,
        )

        # Check metadata
        assert results.name == "test_benchmark"
        assert results.protocol == "NVE"
        assert results.n_steps == 1000
        assert results.n_atoms == 64

        # Check computed statistics
        assert results.mean_step_time_ms > 0
        assert results.std_step_time_ms > 0
        assert results.median_step_time_ms > 0
        assert results.p95_step_time_ms > results.median_step_time_ms
        assert results.p99_step_time_ms > results.p95_step_time_ms
        assert results.steps_per_second > 0

        # Check memory (use approximate equality for floating point)
        assert abs(results.memory_delta_gb - 0.05) < 1e-6
        assert results.peak_memory_gb == 2.1

    def test_benchmark_results_energy_statistics(self, sample_step_times, sample_energies):
        """Test energy drift calculation for NVE."""
        results = BenchmarkResults(
            name="test_nve",
            protocol="NVE",
            n_steps=1000,
            n_atoms=64,
            system_type="silicon",
            device="cuda",
            step_times_ms=sample_step_times,
            memory_before_gb=2.0,
            memory_after_gb=2.0,
            peak_memory_gb=2.1,
            energies=sample_energies,
        )

        # Energy statistics should be computed
        assert results.energy_std is not None
        assert results.energy_drift is not None

        # Energy drift should be small for NVE
        assert abs(results.energy_drift) < 0.1  # Less than 10% drift

    def test_benchmark_results_memory_leak_detection(self, sample_step_times):
        """Test memory leak detection."""
        # No leak
        results_no_leak = BenchmarkResults(
            name="no_leak",
            protocol="NVE",
            n_steps=1000,
            n_atoms=64,
            system_type="silicon",
            device="cuda",
            step_times_ms=sample_step_times,
            memory_before_gb=2.0,
            memory_after_gb=2.005,  # 5 MB growth
            peak_memory_gb=2.1,
        )
        assert not results_no_leak.memory_leak_detected

        # With leak
        results_with_leak = BenchmarkResults(
            name="with_leak",
            protocol="NVE",
            n_steps=1000,
            n_atoms=64,
            system_type="silicon",
            device="cuda",
            step_times_ms=sample_step_times,
            memory_before_gb=2.0,
            memory_after_gb=2.05,  # 50 MB growth
            peak_memory_gb=2.1,
        )
        assert results_with_leak.memory_leak_detected

    def test_benchmark_results_summary(self, sample_step_times):
        """Test summary string generation."""
        results = BenchmarkResults(
            name="test_summary",
            protocol="NVE",
            n_steps=1000,
            n_atoms=64,
            system_type="silicon",
            device="cuda",
            step_times_ms=sample_step_times,
            memory_before_gb=2.0,
            memory_after_gb=2.0,
            peak_memory_gb=2.1,
        )

        summary = results.summary()

        # Check that summary contains key information
        assert "test_summary" in summary
        assert "NVE" in summary
        assert "64 atoms" in summary
        assert "Mean:" in summary
        assert "P95:" in summary
        assert "Memory Usage" in summary

    def test_benchmark_results_to_dict(self, sample_step_times):
        """Test dictionary conversion."""
        results = BenchmarkResults(
            name="test_dict",
            protocol="NVE",
            n_steps=1000,
            n_atoms=64,
            system_type="silicon",
            device="cuda",
            step_times_ms=sample_step_times,
            memory_before_gb=2.0,
            memory_after_gb=2.0,
            peak_memory_gb=2.1,
        )

        data = results.to_dict()

        # Check structure
        assert 'metadata' in data
        assert 'performance' in data
        assert 'memory' in data

        # Check content
        assert data['metadata']['name'] == "test_dict"
        assert data['metadata']['n_atoms'] == 64
        assert data['performance']['mean_step_time_ms'] > 0
        assert data['memory']['peak_gb'] == 2.1

    def test_benchmark_results_save_load(self, sample_step_times, temp_dir):
        """Test JSON serialization and deserialization."""
        results = BenchmarkResults(
            name="test_save_load",
            protocol="NVE",
            n_steps=1000,
            n_atoms=64,
            system_type="silicon",
            device="cuda",
            step_times_ms=sample_step_times,
            memory_before_gb=2.0,
            memory_after_gb=2.0,
            peak_memory_gb=2.1,
        )

        # Save
        save_path = temp_dir / "test_results.json"
        results.save(save_path)

        assert save_path.exists()

        # Load
        loaded_results = BenchmarkResults.load(save_path)

        # Check that key fields match
        assert loaded_results.name == results.name
        assert loaded_results.n_atoms == results.n_atoms
        assert loaded_results.n_steps == results.n_steps


class TestCreateBenchmarkSystem:
    """Test benchmark system creation utilities."""

    def test_create_silicon_system(self):
        """Test silicon crystal creation."""
        atoms = create_benchmark_system("silicon", n_atoms=64, temperature_K=300)

        # Check properties
        assert len(atoms) >= 60  # Approximate target
        assert atoms.get_pbc().all()  # Periodic
        assert "Si" in atoms.get_chemical_symbols()

        # Check that velocities were initialized
        velocities = atoms.get_velocities()
        assert velocities is not None
        assert not np.allclose(velocities, 0)

    def test_create_copper_system(self):
        """Test copper crystal creation."""
        atoms = create_benchmark_system("copper", n_atoms=100, temperature_K=300)

        # FCC copper has 4 atoms per unit cell, so we get closest supercell
        assert len(atoms) >= 30  # Relaxed constraint
        assert atoms.get_pbc().all()
        assert "Cu" in atoms.get_chemical_symbols()

    def test_create_water_system(self):
        """Test water box creation."""
        atoms = create_benchmark_system("water", n_atoms=30, temperature_K=300)

        # Water has H and O
        symbols = atoms.get_chemical_symbols()
        assert "H" in symbols
        assert "O" in symbols
        assert atoms.get_pbc().all()

    def test_invalid_system_type(self):
        """Test error handling for invalid system type."""
        with pytest.raises(ValueError, match="Unknown system type"):
            create_benchmark_system("invalid_system", n_atoms=64)


class MockCalculator(Calculator):
    """Mock calculator for testing benchmark framework."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, latency_ms=5.0, **kwargs):
        """
        Initialize mock calculator.

        Args:
            latency_ms: Simulated latency per call (milliseconds)
        """
        Calculator.__init__(self, **kwargs)
        self.latency_ms = latency_ms
        self.call_count = 0

    def calculate(self, atoms=None, properties=['energy'], system_changes=None):
        """Simulate calculation with realistic timing."""
        Calculator.calculate(self, atoms, properties, system_changes)

        # Simulate computation time
        import time
        time.sleep(self.latency_ms / 1000.0)

        self.call_count += 1

        # Return simple harmonic potential
        positions = atoms.positions
        energy = 0.1 * np.sum(positions**2)
        forces = -0.2 * positions

        self.results = {
            'energy': energy,
            'forces': forces,
        }


class TestMDTrajectoryBenchmark:
    """Test MDTrajectoryBenchmark class."""

    @pytest.fixture
    def mock_calculator(self):
        """Provide mock calculator with fast latency."""
        return MockCalculator(latency_ms=1.0)  # 1ms per call

    def test_benchmark_initialization(self, mock_calculator):
        """Test benchmark initialization."""
        benchmark = MDTrajectoryBenchmark(
            calculator=mock_calculator,
            system_type="silicon",
            n_atoms=32,
            protocol=MDProtocol.NVE,
        )

        assert benchmark.calculator is mock_calculator
        assert benchmark.system_type == "silicon"
        assert benchmark.protocol == MDProtocol.NVE
        assert benchmark.atoms is not None
        # Small system size (32 atoms target) may result in unit cell
        assert len(benchmark.atoms) >= 8

    def test_benchmark_run_nve(self, mock_calculator):
        """Test running NVE benchmark."""
        benchmark = MDTrajectoryBenchmark(
            calculator=mock_calculator,
            system_type="silicon",
            n_atoms=32,
            protocol=MDProtocol.NVE,
        )

        results = benchmark.run(n_steps=50, warmup_steps=5)

        # Check results
        assert results.n_steps == 50
        assert len(results.step_times_ms) == 50
        assert results.mean_step_time_ms > 0
        assert results.protocol == MDProtocol.NVE.value

        # Check that calculator was called
        assert mock_calculator.call_count > 0

    def test_benchmark_run_nvt(self, mock_calculator):
        """Test running NVT (Langevin) benchmark."""
        benchmark = MDTrajectoryBenchmark(
            calculator=mock_calculator,
            system_type="silicon",
            n_atoms=32,
            protocol=MDProtocol.NVT,
            temperature_K=300,
        )

        results = benchmark.run(n_steps=50, warmup_steps=5)

        assert results.n_steps == 50
        assert results.protocol == MDProtocol.NVT.value

    def test_benchmark_memory_tracking(self, mock_calculator):
        """Test memory tracking during benchmark."""
        benchmark = MDTrajectoryBenchmark(
            calculator=mock_calculator,
            system_type="silicon",
            n_atoms=32,
            protocol=MDProtocol.NVE,
        )

        results = benchmark.run(
            n_steps=100,
            warmup_steps=10,
            memory_sample_interval=10,
        )

        # Memory should be tracked
        assert results.memory_before_gb >= 0
        assert results.memory_after_gb >= 0
        assert results.peak_memory_gb >= 0

        # Memory samples should be collected
        assert len(results.memory_samples_gb) > 0

    def test_benchmark_energy_tracking(self, mock_calculator):
        """Test energy tracking during benchmark."""
        benchmark = MDTrajectoryBenchmark(
            calculator=mock_calculator,
            system_type="silicon",
            n_atoms=32,
            protocol=MDProtocol.NVE,
        )

        results = benchmark.run(
            n_steps=100,
            warmup_steps=10,
            energy_sample_interval=5,
        )

        # Energies should be tracked
        assert len(results.energies) > 0
        assert results.energy_std is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_benchmark_cuda_memory(self, mock_calculator):
        """Test CUDA memory tracking."""
        # Set device to cuda
        benchmark = MDTrajectoryBenchmark(
            calculator=mock_calculator,
            system_type="silicon",
            n_atoms=32,
            protocol=MDProtocol.NVE,
        )

        # Override device for testing
        benchmark.device = "cuda"

        results = benchmark.run(n_steps=50, warmup_steps=5)

        # Memory tracking should work
        assert results.memory_before_gb >= 0
        assert results.peak_memory_gb >= 0


class TestCompareCalculators:
    """Test calculator comparison utilities."""

    def test_compare_two_calculators(self):
        """Test comparing multiple calculators."""
        calc1 = MockCalculator(latency_ms=1.0)
        calc2 = MockCalculator(latency_ms=2.0)  # Slower

        calculators = {
            "Fast": calc1,
            "Slow": calc2,
        }

        results = compare_calculators(
            calculators=calculators,
            system_type="silicon",
            n_atoms=32,
            protocol=MDProtocol.NVE,
            n_steps=50,
        )

        # Check results
        assert "Fast" in results
        assert "Slow" in results

        # Fast should be faster
        assert results["Fast"].mean_step_time_ms < results["Slow"].mean_step_time_ms

        # Both should have same number of steps
        assert results["Fast"].n_steps == results["Slow"].n_steps == 50


class TestMDProtocol:
    """Test MDProtocol enum."""

    def test_protocol_values(self):
        """Test protocol enum values."""
        assert MDProtocol.NVE.value == "NVE"
        assert MDProtocol.NVT.value == "NVT"
        assert MDProtocol.NPT.value == "NPT"

    def test_protocol_from_string(self):
        """Test creating protocol from string."""
        protocol = MDProtocol["NVE"]
        assert protocol == MDProtocol.NVE


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_benchmark_workflow(self, temp_dir):
        """Test complete benchmark workflow: run -> save -> load -> analyze."""
        # Create calculator
        calculator = MockCalculator(latency_ms=2.0)

        # Run benchmark
        benchmark = MDTrajectoryBenchmark(
            calculator=calculator,
            system_type="silicon",
            n_atoms=32,
            protocol=MDProtocol.NVE,
        )

        results = benchmark.run(n_steps=50, warmup_steps=5)

        # Save results
        save_path = temp_dir / "benchmark_results.json"
        results.save(save_path)

        # Verify file exists and is valid JSON
        assert save_path.exists()
        with open(save_path) as f:
            data = json.load(f)
            assert 'metadata' in data
            assert 'performance' in data

        # Load and verify
        loaded_results = BenchmarkResults.load(save_path)
        assert loaded_results.name == results.name
        assert loaded_results.n_steps == results.n_steps

    def test_benchmark_with_different_system_sizes(self):
        """Test benchmark scales correctly with system size."""
        calculator = MockCalculator(latency_ms=1.0)

        sizes = [32, 64]
        results = {}

        for size in sizes:
            benchmark = MDTrajectoryBenchmark(
                calculator=calculator,
                system_type="silicon",
                n_atoms=size,
                protocol=MDProtocol.NVE,
            )

            result = benchmark.run(n_steps=20, warmup_steps=5)
            results[size] = result

        # Both should complete successfully
        assert results[32].n_steps == 20
        assert results[64].n_steps == 20

        # Actual atoms counts should differ
        assert results[32].n_atoms < results[64].n_atoms


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
