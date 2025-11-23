"""
Unit tests for dataset validation framework.

Tests all validation components:
- StructureValidator
- LabelValidator
- DiversityValidator
- DatasetValidator
- ValidationIssue and ValidationReport
"""

import numpy as np
import pytest
import torch
from ase import Atoms

from mlff_distiller.data.validation import (
    DatasetValidator,
    DiversityValidator,
    LabelValidator,
    StructureValidator,
    ValidationIssue,
    ValidationReport,
)


class TestValidationIssue:
    """Test ValidationIssue class."""

    def test_creation(self):
        """Test creating a validation issue."""
        issue = ValidationIssue(
            severity='error',
            category='structure',
            message='Test issue',
            sample_idx=42,
            details={'key': 'value'}
        )

        assert issue.severity == 'error'
        assert issue.category == 'structure'
        assert issue.message == 'Test issue'
        assert issue.sample_idx == 42
        assert issue.details == {'key': 'value'}

    def test_string_representation(self):
        """Test string representation of issue."""
        issue = ValidationIssue(
            severity='warning',
            category='label',
            message='Large force detected',
            sample_idx=10,
        )

        str_repr = str(issue)
        assert 'WARNING' in str_repr
        assert 'label' in str_repr
        assert 'Sample 10' in str_repr
        assert 'Large force detected' in str_repr

    def test_string_representation_no_index(self):
        """Test string representation without sample index."""
        issue = ValidationIssue(
            severity='info',
            category='diversity',
            message='Limited diversity',
        )

        str_repr = str(issue)
        assert 'INFO' in str_repr
        assert 'diversity' in str_repr
        assert 'Sample' not in str_repr
        assert 'Limited diversity' in str_repr


class TestValidationReport:
    """Test ValidationReport class."""

    def test_creation(self):
        """Test creating a validation report."""
        issues = [
            ValidationIssue('error', 'structure', 'Error 1'),
            ValidationIssue('warning', 'label', 'Warning 1'),
            ValidationIssue('info', 'diversity', 'Info 1'),
        ]

        report = ValidationReport(
            passed=False,
            issues=issues,
            statistics={'mean': 1.0},
            num_samples=100,
            num_errors=1,
            num_warnings=1,
            num_info=1,
        )

        assert not report.passed
        assert len(report.issues) == 3
        assert report.num_errors == 1
        assert report.num_warnings == 1
        assert report.num_info == 1

    def test_summary(self):
        """Test generating summary text."""
        issues = [
            ValidationIssue('error', 'structure', 'Test error'),
        ]

        report = ValidationReport(
            passed=False,
            issues=issues,
            statistics={'energy_mean': 1.234},
            num_samples=50,
            num_errors=1,
            num_warnings=0,
            num_info=0,
        )

        summary = report.summary()
        assert 'FAILED' in summary
        assert 'Total samples: 50' in summary
        assert 'Errors: 1' in summary
        assert 'Test error' in summary
        assert 'energy_mean' in summary

    def test_to_dict(self):
        """Test converting report to dictionary."""
        issues = [
            ValidationIssue('error', 'structure', 'Error', sample_idx=5),
        ]

        report = ValidationReport(
            passed=True,
            issues=issues,
            statistics={'key': 'value'},
            num_samples=10,
            num_errors=0,
            num_warnings=0,
            num_info=1,
        )

        data = report.to_dict()
        assert data['passed'] is True
        assert data['num_samples'] == 10
        assert len(data['issues']) == 1
        assert data['issues'][0]['severity'] == 'error'
        assert data['issues'][0]['sample_idx'] == 5


class TestStructureValidator:
    """Test StructureValidator class."""

    def test_valid_structure(self):
        """Test validation of a valid structure."""
        # Use smaller min_atoms to match our 3-atom H2O structure
        validator = StructureValidator(min_atoms=1, max_atoms=100)

        # Create simple valid structure
        atoms = Atoms('H2O', positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])
        atoms.center(vacuum=5.0)
        atoms.set_cell([10, 10, 10])
        atoms.set_pbc(True)

        issues = validator.validate_geometry(atoms)
        # Should have no errors (warnings are ok for this test)
        errors = [i for i in issues if i.severity == 'error']
        assert len(errors) == 0

    def test_overlapping_atoms(self):
        """Test detection of overlapping atoms."""
        validator = StructureValidator(min_distance=0.5)

        # Create structure with overlapping atoms
        atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.1]])  # Too close

        issues = validator.validate_geometry(atoms)
        errors = [i for i in issues if i.severity == 'error']
        assert len(errors) > 0
        assert any('Overlapping' in issue.message for issue in errors)

    def test_nan_positions(self):
        """Test detection of NaN in positions."""
        validator = StructureValidator()

        atoms = Atoms('H2', positions=[[0, 0, 0], [np.nan, 0, 0]])

        issues = validator.validate_geometry(atoms)
        errors = [i for i in issues if i.severity == 'error']
        assert len(errors) > 0
        assert any('Non-finite' in issue.message for issue in errors)

    def test_system_size_warnings(self):
        """Test warnings for system size."""
        validator = StructureValidator(min_atoms=10, max_atoms=20)

        # Too small
        small_atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1]])
        issues = validator.validate_geometry(small_atoms)
        warnings = [i for i in issues if i.severity == 'warning']
        assert any('too small' in issue.message for issue in warnings)

        # Too large
        large_positions = np.random.randn(30, 3) * 10
        large_atoms = Atoms('H30', positions=large_positions)
        issues = validator.validate_geometry(large_atoms)
        warnings = [i for i in issues if i.severity == 'warning']
        assert any('too large' in issue.message for issue in warnings)

    def test_invalid_pbc_cell(self):
        """Test detection of invalid PBC cell."""
        validator = StructureValidator(check_pbc=True)

        # PBC enabled but cell not set
        atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1]])
        atoms.set_pbc(True)
        atoms.set_cell([0, 0, 0])  # Invalid cell

        issues = validator.validate_geometry(atoms)
        errors = [i for i in issues if i.severity == 'error']
        assert len(errors) > 0
        assert any('cell' in issue.message.lower() for issue in errors)

    def test_small_cell_warning(self):
        """Test warning for very small cell dimensions."""
        validator = StructureValidator(check_pbc=True)

        atoms = Atoms('H', positions=[[0, 0, 0]])
        atoms.set_pbc(True)
        atoms.set_cell([0.5, 0.5, 0.5])  # Very small cell

        issues = validator.validate_geometry(atoms)
        warnings = [i for i in issues if i.severity == 'warning']
        assert any('small cell' in issue.message.lower() for issue in warnings)

    def test_check_distances(self):
        """Test distance checking method."""
        validator = StructureValidator(min_distance=0.8)

        atoms = Atoms('H3', positions=[[0, 0, 0], [0, 0, 1.5], [0, 0, 0.5]])

        issues = validator.check_distances(atoms)
        # Should detect close atoms (0.5 Å apart)
        errors = [i for i in issues if i.severity == 'error']
        assert len(errors) > 0

    def test_close_atoms_warning(self):
        """Test warning for close (but not overlapping) atoms."""
        validator = StructureValidator(min_distance=0.5)

        # Atoms at 0.7 Å apart (between min_distance and 1.0 Å)
        atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.7]])

        issues = validator.check_distances(atoms)
        warnings = [i for i in issues if i.severity == 'warning']
        assert len(warnings) > 0
        assert any('close' in issue.message.lower() for issue in warnings)


class TestLabelValidator:
    """Test LabelValidator class."""

    def test_valid_energy(self):
        """Test validation of valid energy."""
        validator = LabelValidator()

        issues = validator.validate_energy(energy=-100.0, natoms=10)
        assert len(issues) == 0

    def test_nan_energy(self):
        """Test detection of NaN energy."""
        validator = LabelValidator()

        issues = validator.validate_energy(energy=np.nan, natoms=10)
        errors = [i for i in issues if i.severity == 'error']
        assert len(errors) > 0
        assert any('Non-finite' in issue.message for issue in errors)

    def test_energy_out_of_range(self):
        """Test detection of energy outside acceptable range."""
        validator = LabelValidator(energy_range=(-10.0, 10.0))

        # Too low
        issues_low = validator.validate_energy(energy=-200.0, natoms=10)
        warnings = [i for i in issues_low if i.severity == 'warning']
        assert any('too low' in issue.message for issue in warnings)

        # Too high
        issues_high = validator.validate_energy(energy=200.0, natoms=10)
        warnings = [i for i in issues_high if i.severity == 'warning']
        assert any('too high' in issue.message for issue in warnings)

    def test_valid_forces(self):
        """Test validation of valid forces."""
        validator = LabelValidator()

        forces = np.random.randn(10, 3) * 2.0  # Reasonable forces
        issues = validator.validate_forces(forces)

        # May have warnings but no errors
        errors = [i for i in issues if i.severity == 'error']
        assert len(errors) == 0

    def test_nan_forces(self):
        """Test detection of NaN in forces."""
        validator = LabelValidator()

        forces = np.array([[1, 2, 3], [np.nan, 0, 0]])
        issues = validator.validate_forces(forces)
        errors = [i for i in issues if i.severity == 'error']
        assert len(errors) > 0
        assert any('Non-finite' in issue.message for issue in errors)

    def test_excessive_forces(self):
        """Test detection of excessive forces."""
        validator = LabelValidator(max_force=10.0)

        forces = np.array([[0, 0, 0], [0, 0, 50.0]])  # One very large force
        issues = validator.validate_forces(forces)
        errors = [i for i in issues if i.severity == 'error']
        assert len(errors) > 0
        assert any('Excessive force' in issue.message for issue in errors)

    def test_large_force_warning(self):
        """Test warning for large (but not excessive) forces."""
        validator = LabelValidator(max_force=100.0)

        # Force magnitude = 60 (> 50 warning threshold, < 100 error threshold)
        forces = np.array([[0, 0, 60.0]])
        issues = validator.validate_forces(forces)
        warnings = [i for i in issues if i.severity == 'warning']
        assert len(warnings) > 0
        assert any('Large force' in issue.message for issue in warnings)

    def test_detect_outliers(self):
        """Test outlier detection."""
        validator = LabelValidator(outlier_threshold=2.0)

        # Create data with outliers
        values = np.concatenate([
            np.random.randn(100),  # Normal data
            np.array([10.0, -10.0])  # Outliers
        ])

        issues = validator.detect_outliers(values, 'test_values')
        assert len(issues) > 0
        assert any('outlier' in issue.message.lower() for issue in issues)

    def test_detect_outliers_few_samples(self):
        """Test outlier detection with insufficient samples."""
        validator = LabelValidator()

        # Too few samples for statistics
        values = np.array([1.0, 2.0])
        issues = validator.detect_outliers(values, 'test_values')
        assert len(issues) == 0  # Should not attempt outlier detection

    def test_detect_outliers_severity(self):
        """Test outlier severity levels based on percentage."""
        validator = LabelValidator(outlier_threshold=2.0)

        # Many outliers (>1%) -> error
        values = np.concatenate([
            np.random.randn(50),
            np.ones(10) * 10.0  # 20% outliers
        ])
        issues = validator.detect_outliers(values, 'test')
        errors = [i for i in issues if i.severity == 'error']
        assert len(errors) > 0


class TestDiversityValidator:
    """Test DiversityValidator class."""

    def test_element_coverage_good(self):
        """Test good element coverage."""
        validator = DiversityValidator(min_element_samples=5)

        element_counts = {1: 100, 6: 50, 8: 75}  # H, C, O
        issues = validator.check_element_coverage(element_counts, total_samples=225)

        # Should have no warnings about rare elements
        warnings = [i for i in issues if 'rare' in issue.message.lower()]
        assert len(warnings) == 0

    def test_rare_elements(self):
        """Test detection of rare elements."""
        validator = DiversityValidator(min_element_samples=10)

        element_counts = {1: 100, 6: 5, 8: 3}  # C and O are rare
        issues = validator.check_element_coverage(element_counts, total_samples=108)

        # Check for warning about elements with < min_element_samples
        warnings = [i for i in issues if i.severity == 'warning' and 'elements have <' in i.message]
        assert len(warnings) > 0

    def test_dominated_by_single_element(self):
        """Test detection of single element dominance."""
        validator = DiversityValidator()

        element_counts = {1: 950, 6: 50}  # 95% hydrogen
        issues = validator.check_element_coverage(element_counts, total_samples=1000)

        warnings = [i for i in issues if 'dominated' in i.message.lower()]
        assert len(warnings) > 0

    def test_size_distribution_good(self):
        """Test good size distribution."""
        validator = DiversityValidator(min_size_bins=3)

        sizes = np.random.randint(10, 100, size=100)
        issues = validator.check_size_distribution(sizes)

        # Should have good diversity
        warnings = [i for i in issues if i.severity == 'warning']
        assert len(warnings) == 0

    def test_limited_size_diversity(self):
        """Test detection of limited size diversity."""
        validator = DiversityValidator(min_size_bins=5)

        sizes = np.array([10, 10, 10, 20, 20, 20])  # Only 2 unique sizes
        issues = validator.check_size_distribution(sizes)

        warnings = [i for i in issues if 'size diversity' in i.message.lower()]
        assert len(warnings) > 0

    def test_narrow_size_range(self):
        """Test detection of narrow size range."""
        validator = DiversityValidator()

        sizes = np.array([100, 101, 102, 103, 104])  # Very narrow range
        issues = validator.check_size_distribution(sizes)

        # Should have info message about narrow distribution
        info = [i for i in issues if 'narrow' in i.message.lower()]
        assert len(info) > 0

    def test_composition_variety_good(self):
        """Test good composition variety."""
        validator = DiversityValidator()

        compositions = [
            (1, 1, 8),  # H2O
            (1, 6, 6),  # CH2
            (1, 1, 1, 6, 8, 8),  # Different
        ]
        issues = validator.check_composition_variety(compositions)

        # Should have no warnings
        warnings = [i for i in issues if i.severity == 'warning']
        assert len(warnings) == 0

    def test_identical_compositions(self):
        """Test detection of identical compositions."""
        validator = DiversityValidator()

        compositions = [(1, 1, 8)] * 10  # All H2O
        issues = validator.check_composition_variety(compositions)

        warnings = [i for i in issues if 'identical' in i.message.lower()]
        assert len(warnings) > 0

    def test_limited_composition_variety(self):
        """Test detection of limited composition variety."""
        validator = DiversityValidator()

        # 100 structures, only 5 unique compositions
        compositions = [(1, 1, 8)] * 20 + [(1, 6, 6)] * 20 + [(6, 8, 8)] * 20
        compositions += [(1, 1, 6)] * 20 + [(8, 8)] * 20

        issues = validator.check_composition_variety(compositions)

        # Should have info about limited variety
        info = [i for i in issues if 'variety' in i.message.lower()]
        assert len(info) > 0


class TestDatasetValidator:
    """Test DatasetValidator class."""

    def test_initialization(self):
        """Test creating a DatasetValidator."""
        validator = DatasetValidator()

        assert validator.structure_validator is not None
        assert validator.label_validator is not None
        assert validator.diversity_validator is not None
        assert validator.fail_on_error is True
        assert validator.verbose is True

    def test_initialization_custom(self):
        """Test creating validator with custom components."""
        struct_val = StructureValidator(min_distance=0.3)
        label_val = LabelValidator(max_force=50.0)
        div_val = DiversityValidator(min_element_samples=5)

        validator = DatasetValidator(
            structure_validator=struct_val,
            label_validator=label_val,
            diversity_validator=div_val,
            fail_on_error=False,
            verbose=False,
        )

        assert validator.structure_validator == struct_val
        assert validator.label_validator == label_val
        assert validator.diversity_validator == div_val
        assert validator.fail_on_error is False
        assert validator.verbose is False

    def test_validate_sample(self):
        """Test validating a single sample."""
        validator = DatasetValidator()

        # Create valid sample
        atoms = Atoms('H2O', positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])
        atoms.center(vacuum=5.0)
        atoms.set_cell([10, 10, 10])
        atoms.set_pbc(True)

        sample = {
            'atoms': atoms,
            'energy': torch.tensor(-10.0),
            'forces': torch.tensor([[0, 0, 0.1], [0, 0, -0.1], [0, 0.05, 0]]),
            'natoms': 3,
        }

        issues = validator.validate_sample(sample)

        # Should be valid
        errors = [i for i in issues if i.severity == 'error']
        assert len(errors) == 0

    def test_validate_sample_with_issues(self):
        """Test validating sample with issues."""
        validator = DatasetValidator()

        # Create sample with overlapping atoms
        atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.1]])  # Too close

        sample = {
            'atoms': atoms,
            'energy': torch.tensor(np.nan),  # Invalid energy
            'forces': torch.tensor([[100, 0, 0], [0, 200, 0]]),  # Excessive forces
            'natoms': 2,
        }

        issues = validator.validate_sample(sample)

        # Should have multiple errors
        errors = [i for i in issues if i.severity == 'error']
        assert len(errors) >= 3  # overlapping, nan energy, excessive forces

    def test_compute_statistics(self):
        """Test statistics computation."""
        validator = DatasetValidator()

        energies = [-10.0, -15.0, -12.0]
        energy_per_atom = [-5.0, -7.5, -6.0]
        force_mags = [1.0, 2.0, 3.0, 1.5]
        sizes = [2, 2, 2]
        element_counts = {1: 6}

        stats = validator._compute_statistics(
            energies, energy_per_atom, force_mags, sizes, element_counts
        )

        assert 'energy_mean' in stats
        assert 'energy_std' in stats
        assert 'force_magnitude_mean' in stats
        assert 'system_size_mean' in stats
        assert 'num_unique_elements' in stats
        assert stats['num_unique_elements'] == 1

    def test_compute_statistics_empty(self):
        """Test statistics computation with empty data."""
        validator = DatasetValidator()

        stats = validator._compute_statistics([], [], [], [], {})

        # Should return empty or minimal stats
        assert isinstance(stats, dict)


@pytest.mark.integration
class TestDatasetValidatorIntegration:
    """Integration tests requiring MolecularDataset."""

    def test_validate_dataset_mock(self, tmp_path):
        """Test validating a mock dataset."""
        # This will be expanded with actual dataset testing in integration tests
        pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
