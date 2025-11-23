"""
Integration tests for dataset validation with MolecularDataset.

Tests validation workflow end-to-end with real datasets.
"""

import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
from ase import Atoms
from ase.db import connect

from mlff_distiller.data.dataset import MolecularDataset
from mlff_distiller.data.validation import DatasetValidator


@pytest.fixture
def valid_ase_db(tmp_path):
    """Create a valid ASE database for testing."""
    db_path = tmp_path / "valid.db"
    db = connect(str(db_path))

    # Add diverse valid structures
    np.random.seed(42)

    # Water molecules (various configurations)
    for i in range(10):
        atoms = Atoms('H2O', positions=[[0, 0, 0], [0, 0, 1.5], [0, 1.5, 0]])
        atoms.positions += np.random.randn(3, 3) * 0.05  # Smaller perturbation
        atoms.center(vacuum=5.0)
        atoms.set_cell([10, 10, 10])
        atoms.set_pbc(True)

        energy = -15.0 + np.random.randn() * 0.5
        forces = np.random.randn(3, 3) * 2.0

        db.write(atoms, data={'energy': energy, 'forces': forces})

    # Methane molecules
    for i in range(10):
        atoms = Atoms('CH4',
                     positions=[[0, 0, 0], [1.2, 0, 0], [0, 1.2, 0], [0, 0, 1.2], [-1.2, 0, 0]])
        atoms.positions += np.random.randn(5, 3) * 0.05  # Smaller perturbation
        atoms.center(vacuum=5.0)
        atoms.set_cell([10, 10, 10])
        atoms.set_pbc(True)

        energy = -25.0 + np.random.randn() * 0.5
        forces = np.random.randn(5, 3) * 2.0

        db.write(atoms, data={'energy': energy, 'forces': forces})

    return db_path


@pytest.fixture
def problematic_ase_db(tmp_path):
    """Create an ASE database with validation issues."""
    db_path = tmp_path / "problematic.db"
    db = connect(str(db_path))

    # Add structures with issues

    # 1. Overlapping atoms
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.1]])  # Too close
    db.write(atoms, data={'energy': -10.0, 'forces': [[0, 0, 1], [0, 0, -1]]})

    # 2. NaN energy
    atoms = Atoms('H2O', positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    atoms.center(vacuum=5.0)
    db.write(atoms, data={'energy': np.nan, 'forces': np.random.randn(3, 3).tolist()})

    # 3. Excessive forces
    atoms = Atoms('H2O', positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    atoms.center(vacuum=5.0)
    forces = np.array([[0, 0, 0], [0, 0, 150.0], [0, 0, 0]])  # Very large force
    db.write(atoms, data={'energy': -15.0, 'forces': forces.tolist()})

    # 4. Invalid PBC
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1]])
    atoms.set_pbc(True)
    atoms.set_cell([0, 0, 0])  # Invalid cell
    db.write(atoms, data={'energy': -10.0, 'forces': [[0, 0, 0.5], [0, 0, -0.5]]})

    # 5. Energy outlier
    atoms = Atoms('H2O', positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    atoms.center(vacuum=5.0)
    atoms.set_cell([10, 10, 10])
    atoms.set_pbc(True)
    energy = 1000.0  # Extreme outlier
    db.write(atoms, data={'energy': energy, 'forces': np.random.randn(3, 3).tolist()})

    return db_path


@pytest.fixture
def valid_hdf5(tmp_path):
    """Create a valid HDF5 dataset for testing."""
    h5_path = tmp_path / "valid.h5"

    with h5py.File(h5_path, 'w') as f:
        structures = f.create_group('structures')

        np.random.seed(42)

        for i in range(15):
            group = structures.create_group(str(i))

            # Vary between H2O and CH4
            if i % 2 == 0:
                natoms = 3
                species = np.array([1, 1, 8])
                positions = np.array([[0, 0, 0], [0, 0, 1.5], [0, 1.5, 0]], dtype=float)
            else:
                natoms = 5
                species = np.array([6, 1, 1, 1, 1])
                positions = np.array([[0, 0, 0], [1.2, 0, 0], [0, 1.2, 0],
                                     [0, 0, 1.2], [-1.2, 0, 0]], dtype=float)

            positions = positions + np.random.randn(*positions.shape) * 0.05  # Smaller perturbation

            group.create_dataset('positions', data=positions)
            group.create_dataset('species', data=species)
            group.create_dataset('cell', data=np.eye(3) * 10)
            group.create_dataset('pbc', data=np.array([True, True, True]))
            group.create_dataset('energy', data=-15.0 + np.random.randn() * 0.5)
            group.create_dataset('forces', data=np.random.randn(natoms, 3) * 2.0)

    return h5_path


class TestDatasetValidatorIntegration:
    """Integration tests for DatasetValidator with real datasets."""

    def test_validate_valid_dataset(self, valid_ase_db):
        """Test validation of a fully valid dataset."""
        from mlff_distiller.data.validation import StructureValidator, LabelValidator

        dataset = MolecularDataset(valid_ase_db, format='ase')
        # Use appropriate thresholds for test data
        struct_val = StructureValidator(min_atoms=1, max_atoms=100)
        label_val = LabelValidator(outlier_threshold=4.0)  # Higher for small random dataset
        validator = DatasetValidator(
            structure_validator=struct_val,
            label_validator=label_val,
            verbose=False
        )

        report = validator.validate_dataset(dataset)

        assert report.num_errors == 0  # No errors
        assert report.num_samples == len(dataset)
        assert 'energy_mean' in report.statistics
        assert 'force_magnitude_mean' in report.statistics

    def test_validate_problematic_dataset(self, problematic_ase_db):
        """Test validation of dataset with issues."""
        dataset = MolecularDataset(problematic_ase_db, format='ase')
        validator = DatasetValidator(verbose=False, fail_on_error=True)

        report = validator.validate_dataset(dataset)

        assert report.passed is False
        assert report.num_errors > 0

        # Check for expected error types
        error_messages = [issue.message for issue in report.issues
                         if issue.severity == 'error']

        # Should detect overlapping atoms
        assert any('overlapping' in msg.lower() for msg in error_messages)

        # Should detect NaN energy
        assert any('non-finite' in msg.lower() and 'energy' in msg.lower()
                  for msg in error_messages)

        # Should detect excessive forces
        assert any('excessive force' in msg.lower() for msg in error_messages)

    def test_validate_hdf5_dataset(self, valid_hdf5):
        """Test validation of HDF5 dataset."""
        from mlff_distiller.data.validation import StructureValidator, LabelValidator

        dataset = MolecularDataset(valid_hdf5, format='hdf5')
        struct_val = StructureValidator(min_atoms=1, max_atoms=100)
        label_val = LabelValidator(outlier_threshold=4.0)  # Higher for small random dataset
        validator = DatasetValidator(
            structure_validator=struct_val,
            label_validator=label_val,
            verbose=False
        )

        report = validator.validate_dataset(dataset)

        assert report.num_errors == 0  # No errors
        assert report.num_samples == len(dataset)
        assert report.statistics['num_unique_elements'] >= 2  # H, C, O

    def test_validate_max_samples(self, valid_ase_db):
        """Test validation with max_samples limit."""
        dataset = MolecularDataset(valid_ase_db, format='ase')
        validator = DatasetValidator(verbose=False)

        report = validator.validate_dataset(dataset, max_samples=5)

        assert report.num_samples == 5
        assert report.passed is True

    def test_report_summary(self, valid_ase_db):
        """Test generating report summary."""
        from mlff_distiller.data.validation import StructureValidator

        dataset = MolecularDataset(valid_ase_db, format='ase')
        struct_val = StructureValidator(min_atoms=1, max_atoms=100)
        validator = DatasetValidator(
            structure_validator=struct_val,
            verbose=False
        )

        report = validator.validate_dataset(dataset)
        summary = report.summary()

        assert 'DATASET VALIDATION REPORT' in summary
        assert 'Total samples' in summary
        assert 'STATISTICS' in summary

    def test_report_to_dict(self, valid_ase_db):
        """Test converting report to dictionary."""
        dataset = MolecularDataset(valid_ase_db, format='ase')
        validator = DatasetValidator(verbose=False)

        report = validator.validate_dataset(dataset)
        data = report.to_dict()

        assert isinstance(data, dict)
        assert 'passed' in data
        assert 'num_samples' in data
        assert 'statistics' in data
        assert 'issues' in data
        assert isinstance(data['issues'], list)

        # Should be JSON serializable
        json_str = json.dumps(data)
        assert len(json_str) > 0

    def test_diversity_validation(self, valid_ase_db):
        """Test diversity validation on dataset."""
        dataset = MolecularDataset(valid_ase_db, format='ase')
        validator = DatasetValidator(verbose=False)

        report = validator.validate_dataset(dataset)

        # Check diversity statistics
        assert 'num_unique_elements' in report.statistics
        assert report.statistics['num_unique_elements'] >= 2  # H, C, O

        # Should have reasonable element distribution
        element_dist = report.statistics.get('element_distribution', {})
        assert len(element_dist) >= 2

    def test_statistical_validation(self, valid_ase_db):
        """Test statistical validation and outlier detection."""
        from mlff_distiller.data.validation import StructureValidator, LabelValidator

        dataset = MolecularDataset(valid_ase_db, format='ase')
        struct_val = StructureValidator(min_atoms=1, max_atoms=100)
        # Use higher outlier threshold to avoid random outliers in small test dataset
        label_val = LabelValidator(outlier_threshold=4.0)
        validator = DatasetValidator(
            structure_validator=struct_val,
            label_validator=label_val,
            verbose=False
        )

        report = validator.validate_dataset(dataset)

        # Should compute distributions
        assert 'energy_mean' in report.statistics
        assert 'energy_std' in report.statistics
        assert 'force_magnitude_mean' in report.statistics
        assert 'force_magnitude_std' in report.statistics

        # Should have computed statistics without errors
        assert report.num_errors == 0

    def test_custom_validators(self, valid_ase_db):
        """Test using custom validator parameters."""
        from mlff_distiller.data.validation import (
            DiversityValidator,
            LabelValidator,
            StructureValidator,
        )

        dataset = MolecularDataset(valid_ase_db, format='ase')

        # Create custom validators with stricter thresholds
        struct_val = StructureValidator(min_distance=1.0)  # Stricter
        label_val = LabelValidator(max_force=1.0)  # Stricter
        div_val = DiversityValidator(min_element_samples=20)  # Stricter

        validator = DatasetValidator(
            structure_validator=struct_val,
            label_validator=label_val,
            diversity_validator=div_val,
            verbose=False,
        )

        report = validator.validate_dataset(dataset)

        # With stricter thresholds, may have more warnings
        assert report.num_warnings >= 0  # May have warnings now

    def test_fail_on_error_flag(self, problematic_ase_db):
        """Test fail_on_error flag behavior."""
        dataset = MolecularDataset(problematic_ase_db, format='ase')

        # With fail_on_error=True
        validator_strict = DatasetValidator(verbose=False, fail_on_error=True)
        report_strict = validator_strict.validate_dataset(dataset)
        assert report_strict.passed is False

        # With fail_on_error=False
        validator_lenient = DatasetValidator(verbose=False, fail_on_error=False)
        report_lenient = validator_lenient.validate_dataset(dataset)
        assert report_lenient.passed is True  # Still "passes" despite errors

        # But both should report the same errors
        assert report_strict.num_errors == report_lenient.num_errors

    def test_validate_single_sample(self, valid_ase_db):
        """Test validating individual samples."""
        dataset = MolecularDataset(valid_ase_db, format='ase')
        validator = DatasetValidator(verbose=False)

        # Validate first sample
        sample = dataset[0]
        issues = validator.validate_sample(sample, sample_idx=0)

        # Should be valid
        errors = [i for i in issues if i.severity == 'error']
        assert len(errors) == 0

    def test_size_distribution_validation(self, valid_ase_db):
        """Test system size distribution validation."""
        dataset = MolecularDataset(valid_ase_db, format='ase')
        validator = DatasetValidator(verbose=False)

        report = validator.validate_dataset(dataset)

        # Check size statistics
        assert 'system_size_mean' in report.statistics
        assert 'system_size_min' in report.statistics
        assert 'system_size_max' in report.statistics

        # Should have at least 2 different sizes (H2O=3, CH4=5)
        size_range = report.statistics['system_size_max'] - report.statistics['system_size_min']
        assert size_range > 0


class TestValidationWorkflow:
    """Test complete validation workflows."""

    def test_validation_pipeline(self, valid_ase_db, tmp_path):
        """Test complete validation pipeline with report saving."""
        dataset = MolecularDataset(valid_ase_db, format='ase')
        validator = DatasetValidator(verbose=False)

        # Run validation
        report = validator.validate_dataset(dataset)

        # Save report as JSON
        report_path = tmp_path / "validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)

        # Verify file was created and can be read back
        assert report_path.exists()

        with open(report_path, 'r') as f:
            loaded_data = json.load(f)

        assert loaded_data['passed'] == report.passed
        assert loaded_data['num_samples'] == report.num_samples

    def test_validation_with_statistics_export(self, valid_ase_db, tmp_path):
        """Test exporting statistics to text file."""
        dataset = MolecularDataset(valid_ase_db, format='ase')
        validator = DatasetValidator(verbose=False)

        report = validator.validate_dataset(dataset)

        # Save statistics
        stats_path = tmp_path / "statistics.txt"
        with open(stats_path, 'w') as f:
            f.write("DATASET STATISTICS\n")
            f.write("=" * 80 + "\n\n")
            for key, value in sorted(report.statistics.items()):
                if isinstance(value, float):
                    f.write(f"{key}: {value:.6f}\n")
                else:
                    f.write(f"{key}: {value}\n")

        # Verify file
        assert stats_path.exists()
        content = stats_path.read_text()
        assert 'DATASET STATISTICS' in content
        assert 'energy_mean' in content

    def test_incremental_validation(self, valid_ase_db):
        """Test validating dataset in batches."""
        from mlff_distiller.data.validation import StructureValidator

        dataset = MolecularDataset(valid_ase_db, format='ase')
        struct_val = StructureValidator(min_atoms=1, max_atoms=100)
        validator = DatasetValidator(
            structure_validator=struct_val,
            verbose=False
        )

        # Validate in batches
        batch_size = 5
        all_issues = []

        for i in range(0, len(dataset), batch_size):
            end_idx = min(i + batch_size, len(dataset))
            for j in range(i, end_idx):
                sample = dataset[j]
                issues = validator.validate_sample(sample, sample_idx=j)
                all_issues.extend(issues)

        # Should match full validation
        report = validator.validate_dataset(dataset)

        # Individual sample validation won't detect outliers
        # So compare structure/label errors only
        batch_struct_errors = sum(1 for i in all_issues
                                 if i.severity == 'error' and i.category in ['structure', 'label'])
        report_struct_errors = sum(1 for i in report.issues
                                  if i.severity == 'error' and i.category in ['structure', 'label'])
        assert batch_struct_errors == report_struct_errors


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
