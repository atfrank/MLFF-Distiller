"""
Unit tests for molecular structure sampling strategies.

Tests:
- SamplingConfig validation and defaults
- DiversityMetrics computation
- StratifiedSampler sampling logic

Author: Data Pipeline Engineer
Date: 2025-11-23
"""

import numpy as np
import pytest
from ase import Atoms

from mlff_distiller.data.sampling import (
    DiversityMetrics,
    SamplingConfig,
    StratifiedSampler,
    SystemType,
)


class TestSamplingConfig:
    """Test SamplingConfig class."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = SamplingConfig()

        assert config.total_samples == 120000
        assert config.seed == 42
        assert "H" in config.element_set
        assert "C" in config.element_set
        assert len(config.element_set) == 9

        # Check distribution sums to 1.0
        total_frac = sum(config.system_distribution.values())
        assert np.isclose(total_frac, 1.0)

    def test_custom_configuration(self):
        """Test custom configuration."""
        element_set = {"H", "C", "N", "O"}
        system_dist = {
            SystemType.MOLECULE: 0.7,
            SystemType.CRYSTAL: 0.3,
        }
        size_ranges = {
            SystemType.MOLECULE: (5, 50),
            SystemType.CRYSTAL: (20, 100),
        }

        config = SamplingConfig(
            total_samples=10000,
            seed=123,
            element_set=element_set,
            system_distribution=system_dist,
            size_ranges=size_ranges,
        )

        assert config.total_samples == 10000
        assert config.seed == 123
        assert config.element_set == element_set
        assert config.system_distribution == system_dist
        assert config.size_ranges == size_ranges

    def test_invalid_distribution(self):
        """Test that invalid distribution raises error."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            SamplingConfig(
                system_distribution={
                    SystemType.MOLECULE: 0.5,
                    SystemType.CRYSTAL: 0.3,
                }
            )

    def test_invalid_size_range(self):
        """Test that invalid size range raises error."""
        with pytest.raises(ValueError, match="max_size < min_size"):
            SamplingConfig(
                system_distribution={
                    SystemType.MOLECULE: 0.6,
                    SystemType.CRYSTAL: 0.4,
                },
                size_ranges={
                    SystemType.MOLECULE: (100, 10),  # Invalid: max < min
                    SystemType.CRYSTAL: (50, 500),
                }
            )

    def test_get_sample_counts(self):
        """Test sample count calculation."""
        config = SamplingConfig(
            total_samples=1000,
            system_distribution={
                SystemType.MOLECULE: 0.5,
                SystemType.CRYSTAL: 0.3,
                SystemType.CLUSTER: 0.2,
            },
            size_ranges={
                SystemType.MOLECULE: (10, 100),
                SystemType.CRYSTAL: (50, 500),
                SystemType.CLUSTER: (20, 200),
            },
        )

        counts = config.get_sample_counts()

        # Check total
        assert sum(counts.values()) == 1000

        # Check approximate proportions (within rounding)
        assert counts[SystemType.MOLECULE] >= 450
        assert counts[SystemType.MOLECULE] <= 550
        assert counts[SystemType.CRYSTAL] >= 250
        assert counts[SystemType.CRYSTAL] <= 350


class TestDiversityMetrics:
    """Test DiversityMetrics class."""

    def test_element_coverage(self):
        """Test element coverage metric."""
        # Create test structures
        atoms1 = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        atoms2 = Atoms("CO2", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])

        structures = [atoms1, atoms2]
        target_elements = {"H", "C", "O", "N"}

        coverage = DiversityMetrics.element_coverage(structures, target_elements)

        # Should have H, C, O (3 out of 4)
        assert coverage == 0.75

    def test_element_coverage_full(self):
        """Test full element coverage."""
        atoms = Atoms("HCON", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])

        coverage = DiversityMetrics.element_coverage(
            [atoms], {"H", "C", "O", "N"}
        )

        assert coverage == 1.0

    def test_composition_diversity(self):
        """Test composition diversity (entropy)."""
        # Uniform distribution should give higher entropy
        atoms1 = Atoms("HHHCCCNNNOOOFFF")

        # Skewed distribution should give lower entropy
        atoms2 = Atoms("HHHHHHHHHHHHHCC")

        entropy1 = DiversityMetrics.composition_diversity([atoms1])
        entropy2 = DiversityMetrics.composition_diversity([atoms2])

        assert entropy1 > entropy2
        assert entropy1 > 2.0  # At least 2 bits for 5 elements
        assert entropy2 > 0.0  # Non-zero entropy

    def test_size_diversity(self):
        """Test size diversity metrics."""
        # Create structures of different sizes
        structures = [
            Atoms("H" * 10, positions=np.random.randn(10, 3)),
            Atoms("H" * 20, positions=np.random.randn(20, 3)),
            Atoms("H" * 30, positions=np.random.randn(30, 3)),
        ]

        mean, std, cv = DiversityMetrics.size_diversity(structures)

        assert mean == 20.0
        assert std > 0
        assert cv > 0

    def test_system_type_balance(self):
        """Test system type balance metric."""
        # Create balanced dataset
        structures = [Atoms("H2")] * 30

        system_types_balanced = (
            [SystemType.MOLECULE] * 10
            + [SystemType.CRYSTAL] * 10
            + [SystemType.CLUSTER] * 10
        )

        # Create imbalanced dataset
        system_types_imbalanced = (
            [SystemType.MOLECULE] * 25
            + [SystemType.CRYSTAL] * 3
            + [SystemType.CLUSTER] * 2
        )

        cv_balanced = DiversityMetrics.system_type_balance(
            structures, system_types_balanced
        )
        cv_imbalanced = DiversityMetrics.system_type_balance(
            structures, system_types_imbalanced
        )

        # Balanced should have lower CV
        assert cv_balanced < cv_imbalanced
        assert cv_balanced < 0.1  # Nearly perfect balance

    def test_compute_all_metrics(self):
        """Test computing all metrics at once."""
        structures = [
            Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            Atoms("CO2", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            Atoms("NH3", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]),
        ]
        system_types = [SystemType.MOLECULE] * 3
        target_elements = {"H", "C", "O", "N"}

        metrics = DiversityMetrics.compute_all_metrics(
            structures, system_types, target_elements
        )

        # Check all keys present
        assert "element_coverage" in metrics
        assert "composition_entropy" in metrics
        assert "mean_size" in metrics
        assert "std_size" in metrics
        assert "size_cv" in metrics
        assert "type_balance_cv" in metrics

        # Check reasonable values
        assert 0 <= metrics["element_coverage"] <= 1.0
        assert metrics["composition_entropy"] > 0
        assert metrics["mean_size"] > 0


class TestStratifiedSampler:
    """Test StratifiedSampler class."""

    def test_initialization(self):
        """Test sampler initialization."""
        config = SamplingConfig(seed=42)
        sampler = StratifiedSampler(config)

        assert sampler.config == config
        assert sampler.rng is not None
        assert len(sampler.target_counts) > 0

    def test_sample_size(self):
        """Test size sampling."""
        config = SamplingConfig(seed=42)
        sampler = StratifiedSampler(config)

        # Sample multiple sizes
        sizes = [sampler.sample_size(SystemType.MOLECULE) for _ in range(100)]

        # Check all within range
        min_size, max_size = config.size_ranges[SystemType.MOLECULE]
        assert all(min_size <= s <= max_size for s in sizes)

        # Check some diversity
        assert len(set(sizes)) > 10

    def test_sample_elements(self):
        """Test element sampling."""
        config = SamplingConfig(seed=42, element_set={"H", "C", "N", "O", "F"})
        sampler = StratifiedSampler(config)

        elements = sampler.sample_elements(3)

        assert len(elements) == 3
        assert len(set(elements)) == 3  # All unique
        assert all(e in config.element_set for e in elements)

    def test_sample_elements_with_exclusion(self):
        """Test element sampling with exclusions."""
        config = SamplingConfig(seed=42, element_set={"H", "C", "N", "O", "F"})
        sampler = StratifiedSampler(config)

        elements = sampler.sample_elements(2, exclude={"H", "C"})

        assert len(elements) == 2
        assert "H" not in elements
        assert "C" not in elements
        assert all(e in {"N", "O", "F"} for e in elements)

    def test_sample_elements_insufficient(self):
        """Test element sampling with insufficient elements."""
        config = SamplingConfig(seed=42, element_set={"H", "C"})
        sampler = StratifiedSampler(config)

        with pytest.raises(ValueError, match="Cannot sample"):
            sampler.sample_elements(5)

    def test_get_sampling_plan(self):
        """Test sampling plan generation."""
        config = SamplingConfig(total_samples=100, seed=42)
        sampler = StratifiedSampler(config)

        plan = sampler.get_sampling_plan()

        # Check all system types present
        assert SystemType.MOLECULE in plan
        assert SystemType.CRYSTAL in plan

        # Check total count
        total_samples = sum(len(sizes) for sizes in plan.values())
        assert total_samples == 100

        # Check all sizes valid
        for sys_type, sizes in plan.items():
            min_size, max_size = config.size_ranges[sys_type]
            assert all(min_size <= s <= max_size for s in sizes)

    def test_reproducibility(self):
        """Test that same seed gives same results."""
        config1 = SamplingConfig(seed=42)
        config2 = SamplingConfig(seed=42)

        sampler1 = StratifiedSampler(config1)
        sampler2 = StratifiedSampler(config2)

        plan1 = sampler1.get_sampling_plan()
        plan2 = sampler2.get_sampling_plan()

        # Should generate identical plans
        for sys_type in plan1:
            assert plan1[sys_type] == plan2[sys_type]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
