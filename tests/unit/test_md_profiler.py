"""
Unit tests for MD profiler module.

Tests the MDProfiler class and related utilities for profiling
MD workloads with ASE calculators.
"""

import json
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes

from mlff_distiller.cuda.md_profiler import (
    MDProfileResult,
    MDProfiler,
    identify_hotspots,
    profile_md_trajectory,
)


class DummyCalculator(Calculator):
    """Dummy calculator for testing."""

    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, delay_ms: float = 1.0, **kwargs):
        """Initialize with configurable delay."""
        Calculator.__init__(self, **kwargs)
        self.delay_ms = delay_ms
        self.call_count = 0

    def calculate(
        self,
        atoms=None,
        properties: List[str] = ['energy'],
        system_changes: List[str] = all_changes,
    ):
        """Calculate properties with artificial delay."""
        Calculator.calculate(self, atoms, properties, system_changes)

        # Simulate computation time
        import time
        time.sleep(self.delay_ms / 1000.0)

        self.call_count += 1

        # Return dummy results
        self.results = {
            'energy': 0.0,
            'forces': np.zeros((len(atoms), 3)),
            'stress': np.zeros(6),
        }


@pytest.fixture
def dummy_calculator():
    """Fixture providing dummy calculator."""
    return DummyCalculator(delay_ms=1.0)


@pytest.fixture
def short_trajectory():
    """Fixture providing short MD trajectory."""
    trajectory = []
    base_atoms = bulk("Si", "diamond", a=5.43)

    for i in range(10):
        atoms = base_atoms.copy()
        # Add small perturbation
        positions = atoms.get_positions()
        positions += np.random.randn(*positions.shape) * 0.01
        atoms.set_positions(positions)
        trajectory.append(atoms)

    return trajectory


@pytest.fixture
def device():
    """Fixture providing available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class TestMDProfileResult:
    """Tests for MDProfileResult data class."""

    def test_initialization(self):
        """Test basic initialization."""
        latencies = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = MDProfileResult(
            name="Test",
            model_name="DummyModel",
            n_steps=5,
            device="cpu",
            latencies_ms=latencies,
            mean_latency_ms=3.0,
            median_latency_ms=3.0,
            std_latency_ms=1.41,
            min_latency_ms=1.0,
            max_latency_ms=5.0,
            p95_latency_ms=4.8,
            p99_latency_ms=4.96,
        )

        assert result.name == "Test"
        assert result.n_steps == 5
        assert result.mean_latency_ms == 3.0

    def test_memory_delta(self):
        """Test memory delta calculation."""
        result = MDProfileResult(
            name="Test",
            model_name="DummyModel",
            n_steps=10,
            device="cpu",
            latencies_ms=[1.0] * 10,
            mean_latency_ms=1.0,
            median_latency_ms=1.0,
            std_latency_ms=0.0,
            min_latency_ms=1.0,
            max_latency_ms=1.0,
            p95_latency_ms=1.0,
            p99_latency_ms=1.0,
            memory_initial_gb=1.0,
            memory_final_gb=1.05,
        )

        assert abs(result.memory_delta_gb - 0.05) < 1e-6

    def test_memory_stable(self):
        """Test memory stability check."""
        # Stable memory
        result = MDProfileResult(
            name="Test",
            model_name="DummyModel",
            n_steps=10,
            device="cpu",
            latencies_ms=[1.0] * 10,
            mean_latency_ms=1.0,
            median_latency_ms=1.0,
            std_latency_ms=0.0,
            min_latency_ms=1.0,
            max_latency_ms=1.0,
            p95_latency_ms=1.0,
            p99_latency_ms=1.0,
            memory_initial_gb=1.0,
            memory_final_gb=1.005,  # 5 MB increase
        )
        assert result.memory_stable

        # Unstable memory (leak)
        result.memory_final_gb = 1.02  # 20 MB increase
        assert not result.memory_stable

    def test_us_per_atom(self):
        """Test per-atom timing calculation."""
        result = MDProfileResult(
            name="Test",
            model_name="DummyModel",
            n_steps=10,
            device="cpu",
            latencies_ms=[1.0] * 10,
            mean_latency_ms=1.0,
            median_latency_ms=1.0,
            std_latency_ms=0.0,
            min_latency_ms=1.0,
            max_latency_ms=1.0,
            p95_latency_ms=1.0,
            p99_latency_ms=1.0,
            n_atoms=100,
        )

        # 1 ms / 100 atoms = 10 µs/atom
        assert result.us_per_atom == 10.0

    def test_summary(self):
        """Test summary generation."""
        result = MDProfileResult(
            name="Test Profile",
            model_name="DummyModel",
            n_steps=10,
            device="cpu",
            latencies_ms=[1.0] * 10,
            mean_latency_ms=1.0,
            median_latency_ms=1.0,
            std_latency_ms=0.1,
            min_latency_ms=0.9,
            max_latency_ms=1.1,
            p95_latency_ms=1.05,
            p99_latency_ms=1.08,
            n_atoms=100,
        )

        summary = result.summary()

        assert "Test Profile" in summary
        assert "DummyModel" in summary
        assert "Mean:" in summary
        assert "1.0000 ms" in summary

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = MDProfileResult(
            name="Test",
            model_name="DummyModel",
            n_steps=5,
            device="cpu",
            latencies_ms=[1.0, 2.0, 3.0, 4.0, 5.0],
            mean_latency_ms=3.0,
            median_latency_ms=3.0,
            std_latency_ms=1.41,
            min_latency_ms=1.0,
            max_latency_ms=5.0,
            p95_latency_ms=4.8,
            p99_latency_ms=4.96,
        )

        data = result.to_dict()

        assert isinstance(data, dict)
        assert data['name'] == "Test"
        assert data['n_steps'] == 5
        assert isinstance(data['latencies_ms'], list)

    def test_save_load_json(self):
        """Test JSON serialization."""
        result = MDProfileResult(
            name="Test",
            model_name="DummyModel",
            n_steps=5,
            device="cpu",
            latencies_ms=[1.0, 2.0, 3.0, 4.0, 5.0],
            mean_latency_ms=3.0,
            median_latency_ms=3.0,
            std_latency_ms=1.41,
            min_latency_ms=1.0,
            max_latency_ms=5.0,
            p95_latency_ms=4.8,
            p99_latency_ms=4.96,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_profile.json"

            # Save
            result.save_json(filepath)
            assert filepath.exists()

            # Load
            loaded = MDProfileResult.load_json(filepath)
            assert loaded.name == result.name
            assert loaded.n_steps == result.n_steps
            assert loaded.mean_latency_ms == result.mean_latency_ms


class TestMDProfiler:
    """Tests for MDProfiler class."""

    def test_initialization(self, device):
        """Test profiler initialization."""
        profiler = MDProfiler(device=device)

        assert profiler.device == device
        assert profiler.profile_memory is True
        assert profiler.warmup_steps == 10

    def test_profile_calculator(self, dummy_calculator, short_trajectory, device):
        """Test profiling a calculator."""
        profiler = MDProfiler(device=device, warmup_steps=2)

        result = profiler.profile_calculator(
            dummy_calculator,
            short_trajectory,
            properties=['energy', 'forces'],
            name="Dummy Test",
        )

        assert isinstance(result, MDProfileResult)
        assert result.name == "Dummy Test"
        assert result.n_steps == len(short_trajectory)
        assert result.mean_latency_ms > 0
        assert len(result.latencies_ms) == len(short_trajectory)

        # Check that calculator was called for each step
        assert dummy_calculator.call_count >= len(short_trajectory)

    def test_profile_calculator_memory_tracking(self, dummy_calculator, short_trajectory, device):
        """Test memory tracking during profiling."""
        profiler = MDProfiler(device=device, profile_memory=True)

        result = profiler.profile_calculator(
            dummy_calculator,
            short_trajectory,
            properties=['energy'],
        )

        # Memory should be tracked
        assert result.memory_initial_gb >= 0
        assert result.memory_final_gb >= 0
        assert result.memory_peak_gb >= 0

    def test_compare_calculators(self, short_trajectory, device):
        """Test comparing multiple calculators."""
        calc1 = DummyCalculator(delay_ms=1.0)
        calc2 = DummyCalculator(delay_ms=2.0)

        calculators = {
            "Fast Calculator": calc1,
            "Slow Calculator": calc2,
        }

        profiler = MDProfiler(device=device, warmup_steps=2)

        results = profiler.compare_calculators(
            calculators,
            short_trajectory,
            properties=['energy'],
        )

        assert len(results) == 2
        assert "Fast Calculator" in results
        assert "Slow Calculator" in results

        # Fast should be faster
        assert results["Fast Calculator"].mean_latency_ms < results["Slow Calculator"].mean_latency_ms


class TestIdentifyHotspots:
    """Tests for hotspot identification."""

    def test_identify_hotspots_basic(self):
        """Test basic hotspot identification."""
        result = MDProfileResult(
            name="Test",
            model_name="DummyModel",
            n_steps=10,
            device="cpu",
            latencies_ms=[1.0] * 10,
            mean_latency_ms=10.0,
            median_latency_ms=10.0,
            std_latency_ms=1.0,
            min_latency_ms=9.0,
            max_latency_ms=11.0,
            p95_latency_ms=10.5,
            p99_latency_ms=10.8,
            energy_time_ms=2.0,  # 20% of total
            forces_time_ms=7.0,  # 70% of total (hotspot!)
        )

        hotspots = identify_hotspots(result, threshold_pct=10.0)

        assert 'components' in hotspots
        assert 'forces' in hotspots['components']
        assert hotspots['components']['forces']['is_hotspot'] is True
        assert hotspots['components']['energy']['is_hotspot'] is True

    def test_identify_hotspots_memory_leak(self):
        """Test memory leak detection in hotspots."""
        result = MDProfileResult(
            name="Test",
            model_name="DummyModel",
            n_steps=10,
            device="cpu",
            latencies_ms=[1.0] * 10,
            mean_latency_ms=1.0,
            median_latency_ms=1.0,
            std_latency_ms=0.0,
            min_latency_ms=1.0,
            max_latency_ms=1.0,
            p95_latency_ms=1.0,
            p99_latency_ms=1.0,
            memory_initial_gb=1.0,
            memory_final_gb=1.02,  # 20 MB leak
        )

        hotspots = identify_hotspots(result)

        assert any("Memory leak" in rec for rec in hotspots['recommendations'])

    def test_identify_hotspots_variance(self):
        """Test high variance detection."""
        result = MDProfileResult(
            name="Test",
            model_name="DummyModel",
            n_steps=10,
            device="cpu",
            latencies_ms=[1.0] * 10,
            mean_latency_ms=10.0,
            median_latency_ms=10.0,
            std_latency_ms=3.0,  # 30% CV - high variance
            min_latency_ms=5.0,
            max_latency_ms=15.0,
            p95_latency_ms=14.0,
            p99_latency_ms=14.8,
        )

        hotspots = identify_hotspots(result)

        assert any("variance" in rec.lower() for rec in hotspots['recommendations'])


class TestProfileMDTrajectory:
    """Tests for convenience function."""

    def test_profile_md_trajectory(self, dummy_calculator, short_trajectory, device):
        """Test convenience function."""
        result = profile_md_trajectory(
            dummy_calculator,
            short_trajectory,
            device=device,
            properties=['energy'],
        )

        assert isinstance(result, MDProfileResult)
        assert result.n_steps == len(short_trajectory)

    def test_profile_md_trajectory_with_output(self, dummy_calculator, short_trajectory, device):
        """Test with JSON output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "profile.json"

            result = profile_md_trajectory(
                dummy_calculator,
                short_trajectory,
                device=device,
                output_file=output_file,
            )

            assert output_file.exists()

            # Load and verify
            with open(output_file) as f:
                data = json.load(f)
                assert data['n_steps'] == len(short_trajectory)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestMDProfilerCUDA:
    """CUDA-specific tests."""

    def test_profile_on_cuda(self, dummy_calculator, short_trajectory):
        """Test profiling on CUDA device."""
        device = torch.device('cuda')
        profiler = MDProfiler(device=device)

        result = profiler.profile_calculator(
            dummy_calculator,
            short_trajectory,
            properties=['energy'],
        )

        assert result.device == 'cuda'
        assert result.mean_latency_ms > 0

    def test_memory_tracking_cuda(self, dummy_calculator, short_trajectory):
        """Test CUDA memory tracking."""
        device = torch.device('cuda')
        profiler = MDProfiler(device=device, profile_memory=True)

        result = profiler.profile_calculator(
            dummy_calculator,
            short_trajectory,
            properties=['energy'],
        )

        # CUDA memory should be tracked
        assert result.memory_peak_gb >= 0
