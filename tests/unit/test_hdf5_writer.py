"""
Unit tests for HDF5DatasetWriter

Tests cover:
- Basic write operations
- Append mode
- Compression
- Validation
- Error handling
- Edge cases

Author: Data Pipeline Engineer
Date: 2025-11-23
"""

import pytest
import numpy as np
import h5py
from pathlib import Path
import tempfile
import shutil

from ase import Atoms
from mlff_distiller.data.hdf5_writer import HDF5DatasetWriter, convert_pickle_to_hdf5


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_atoms():
    """Create sample ASE Atoms objects for testing."""
    # Water molecule
    h2o = Atoms(
        'H2O',
        positions=[[0, 0, 0], [0.96, 0, 0], [0.24, 0.93, 0]],
        cell=[10, 10, 10],
        pbc=[False, False, False]
    )

    # Methane molecule
    ch4 = Atoms(
        'CH4',
        positions=[
            [0, 0, 0],
            [0.63, 0.63, 0.63],
            [-0.63, -0.63, 0.63],
            [-0.63, 0.63, -0.63],
            [0.63, -0.63, -0.63]
        ],
        cell=[12, 12, 12],
        pbc=[False, False, False]
    )

    # Periodic crystal (diamond)
    diamond = Atoms(
        'C8',
        positions=[
            [0, 0, 0],
            [0.89, 0.89, 0.89],
            [1.78, 1.78, 0],
            [2.67, 2.67, 0.89],
            [1.78, 0, 1.78],
            [2.67, 0.89, 2.67],
            [0, 1.78, 1.78],
            [0.89, 2.67, 2.67]
        ],
        cell=[[3.567, 0, 0], [0, 3.567, 0], [0, 0, 3.567]],
        pbc=[True, True, True]
    )

    return [h2o, ch4, diamond]


@pytest.fixture
def sample_labels(sample_atoms):
    """Create sample labels for test atoms."""
    energies = [-10.5, -23.4, -65.2]
    forces = [
        np.random.randn(len(atoms), 3) * 0.1
        for atoms in sample_atoms
    ]
    stresses = [
        None,  # Non-periodic
        None,  # Non-periodic
        np.random.randn(6) * 0.01  # Periodic
    ]
    return energies, forces, stresses


class TestHDF5DatasetWriter:
    """Test suite for HDF5DatasetWriter."""

    def test_basic_write(self, temp_dir, sample_atoms, sample_labels):
        """Test basic write functionality."""
        output_path = temp_dir / "test_basic.h5"
        energies, forces, stresses = sample_labels

        # Write data
        writer = HDF5DatasetWriter(output_path, compression="gzip")

        for atoms, energy, force, stress in zip(sample_atoms, energies, forces, stresses):
            writer.add_structure(atoms, energy, force, stress)

        writer.finalize()

        # Verify file exists
        assert output_path.exists()

        # Verify contents
        with h5py.File(output_path, 'r') as f:
            assert 'structures' in f
            assert 'labels' in f
            assert 'metadata' in f

            # Check structure count
            assert f['labels']['energy'].shape[0] == 3

            # Check energy values
            np.testing.assert_allclose(
                f['labels']['energy'][:],
                energies,
                rtol=1e-6
            )

            # Check total atoms
            total_atoms = sum(len(atoms) for atoms in sample_atoms)
            assert f['structures']['atomic_numbers'].shape[0] == total_atoms

    def test_batch_write(self, temp_dir, sample_atoms, sample_labels):
        """Test batch writing."""
        output_path = temp_dir / "test_batch.h5"
        energies, forces, stresses = sample_labels

        # Write in batch
        with HDF5DatasetWriter(output_path, compression="gzip") as writer:
            writer.add_batch(
                structures=sample_atoms,
                energies=energies,
                forces=forces,
                stresses=stresses
            )

        # Verify
        with h5py.File(output_path, 'r') as f:
            assert f['labels']['energy'].shape[0] == len(sample_atoms)

    def test_append_mode(self, temp_dir, sample_atoms, sample_labels):
        """Test append mode."""
        output_path = temp_dir / "test_append.h5"
        energies, forces, stresses = sample_labels

        # Write first batch
        with HDF5DatasetWriter(output_path, mode="w") as writer:
            writer.add_structure(
                sample_atoms[0],
                energies[0],
                forces[0],
                stresses[0]
            )

        # Check first write
        with h5py.File(output_path, 'r') as f:
            assert f['labels']['energy'].shape[0] == 1

        # Append second batch
        with HDF5DatasetWriter(output_path, mode="a") as writer:
            writer.add_batch(
                structures=sample_atoms[1:],
                energies=energies[1:],
                forces=forces[1:],
                stresses=stresses[1:]
            )

        # Verify final count
        with h5py.File(output_path, 'r') as f:
            assert f['labels']['energy'].shape[0] == 3

            # Verify all energies
            np.testing.assert_allclose(
                f['labels']['energy'][:],
                energies,
                rtol=1e-6
            )

    def test_compression(self, temp_dir, sample_atoms, sample_labels):
        """Test compression effectiveness."""
        energies, forces, stresses = sample_labels

        # Create many copies for better compression test
        many_atoms = sample_atoms * 50
        many_energies = energies * 50
        many_forces = forces * 50
        many_stresses = stresses * 50

        # Write without compression
        path_no_comp = temp_dir / "no_compression.h5"
        with HDF5DatasetWriter(path_no_comp, compression=None) as writer:
            writer.add_batch(many_atoms, many_energies, many_forces, many_stresses)

        # Write with gzip compression
        path_gzip = temp_dir / "gzip_compression.h5"
        with HDF5DatasetWriter(path_gzip, compression="gzip") as writer:
            writer.add_batch(many_atoms, many_energies, many_forces, many_stresses)

        # Check compression effectiveness
        size_no_comp = path_no_comp.stat().st_size
        size_gzip = path_gzip.stat().st_size

        compression_ratio = size_gzip / size_no_comp
        print(f"Compression ratio: {compression_ratio:.2%}")

        # Gzip should reduce size (typically 30-70% of original)
        assert compression_ratio < 0.95  # At least some compression

    def test_validation_invalid_energy(self, temp_dir, sample_atoms):
        """Test validation catches invalid energy."""
        output_path = temp_dir / "test_validation.h5"

        writer = HDF5DatasetWriter(output_path, validate=True)

        # Try to add structure with NaN energy
        with pytest.raises(ValueError, match="Invalid energy"):
            writer.add_structure(
                sample_atoms[0],
                energy=np.nan,
                forces=np.zeros((3, 3))
            )

        writer.finalize()

    def test_validation_invalid_forces(self, temp_dir, sample_atoms):
        """Test validation catches invalid forces."""
        output_path = temp_dir / "test_validation.h5"

        writer = HDF5DatasetWriter(output_path, validate=True)

        # Try to add structure with wrong force shape
        with pytest.raises(ValueError, match="Forces shape"):
            writer.add_structure(
                sample_atoms[0],
                energy=-10.0,
                forces=np.zeros((5, 3))  # Wrong shape
            )

        writer.finalize()

    def test_validation_nan_forces(self, temp_dir, sample_atoms):
        """Test validation catches NaN in forces."""
        output_path = temp_dir / "test_validation.h5"

        writer = HDF5DatasetWriter(output_path, validate=True)

        forces = np.zeros((3, 3))
        forces[0, 0] = np.nan

        with pytest.raises(ValueError, match="Forces contain NaN"):
            writer.add_structure(
                sample_atoms[0],
                energy=-10.0,
                forces=forces
            )

        writer.finalize()

    def test_validation_disabled(self, temp_dir, sample_atoms):
        """Test that validation can be disabled."""
        output_path = temp_dir / "test_no_validation.h5"

        # With validation disabled, this should work (though produce invalid data)
        writer = HDF5DatasetWriter(output_path, validate=False)

        # This would normally fail validation
        forces = np.zeros((3, 3))
        forces[0, 0] = np.inf

        # Should not raise error with validation disabled
        writer.add_structure(
            sample_atoms[0],
            energy=-10.0,
            forces=forces
        )

        writer.finalize()

        # Verify file was created
        assert output_path.exists()

    def test_context_manager(self, temp_dir, sample_atoms, sample_labels):
        """Test context manager usage."""
        output_path = temp_dir / "test_context.h5"
        energies, forces, stresses = sample_labels

        # Use context manager
        with HDF5DatasetWriter(output_path) as writer:
            writer.add_batch(sample_atoms, energies, forces, stresses)
            # finalize() should be called automatically

        # Verify file is properly closed
        with h5py.File(output_path, 'r') as f:
            assert f['labels']['energy'].shape[0] == 3

    def test_metadata(self, temp_dir, sample_atoms, sample_labels):
        """Test metadata storage."""
        output_path = temp_dir / "test_metadata.h5"
        energies, forces, stresses = sample_labels

        extra_metadata = {
            "teacher_model": "orb-v2",
            "generation_config": {"batch_size": 32, "device": "cuda"},
            "source": "mattergen"
        }

        with HDF5DatasetWriter(output_path) as writer:
            writer.add_batch(sample_atoms, energies, forces, stresses)
            writer.finalize(extra_metadata=extra_metadata)

        # Verify metadata
        with h5py.File(output_path, 'r') as f:
            meta = f['metadata']
            assert 'teacher_model' in meta.attrs
            assert meta.attrs['teacher_model'] == "orb-v2"
            assert 'num_structures' in meta.attrs
            assert meta.attrs['num_structures'] == 3

    def test_splits_consistency(self, temp_dir, sample_atoms, sample_labels):
        """Test that split arrays are consistent."""
        output_path = temp_dir / "test_splits.h5"
        energies, forces, stresses = sample_labels

        with HDF5DatasetWriter(output_path) as writer:
            writer.add_batch(sample_atoms, energies, forces, stresses)

        with h5py.File(output_path, 'r') as f:
            # Check atomic_numbers_splits
            splits = f['structures']['atomic_numbers_splits'][:]
            atomic_numbers = f['structures']['atomic_numbers'][:]

            # Splits should have n_structures + 1 elements
            assert len(splits) == len(sample_atoms) + 1

            # First split should be 0
            assert splits[0] == 0

            # Last split should equal total atoms
            assert splits[-1] == len(atomic_numbers)

            # Check individual splits match atom counts
            for i, atoms in enumerate(sample_atoms):
                start = splits[i]
                end = splits[i + 1]
                assert end - start == len(atoms)

    def test_stress_handling(self, temp_dir, sample_atoms, sample_labels):
        """Test stress tensor handling for periodic/non-periodic systems."""
        output_path = temp_dir / "test_stress.h5"
        energies, forces, stresses = sample_labels

        with HDF5DatasetWriter(output_path) as writer:
            writer.add_batch(sample_atoms, energies, forces, stresses)

        with h5py.File(output_path, 'r') as f:
            stress_mask = f['labels']['stress_mask'][:]

            # First two structures are non-periodic (no stress)
            assert stress_mask[0] == False
            assert stress_mask[1] == False

            # Third structure is periodic (has stress)
            assert stress_mask[2] == True

            # Check stress array shape
            stress = f['labels']['stress'][:]
            assert stress.shape == (3, 6)

    def test_empty_dataset(self, temp_dir):
        """Test creating empty dataset (edge case)."""
        output_path = temp_dir / "test_empty.h5"

        with HDF5DatasetWriter(output_path) as writer:
            pass  # Don't add anything

        # Should still create valid file structure
        assert output_path.exists()

        with h5py.File(output_path, 'r') as f:
            assert 'metadata' in f
            meta = f['metadata']
            assert meta.attrs['num_structures'] == 0

    def test_single_atom_structure(self, temp_dir):
        """Test single atom structure (edge case)."""
        output_path = temp_dir / "test_single_atom.h5"

        # Single hydrogen atom
        single_atom = Atoms('H', positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=False)
        energy = -1.0
        forces = np.array([[0.0, 0.0, 0.0]])

        with HDF5DatasetWriter(output_path) as writer:
            writer.add_structure(single_atom, energy, forces)

        with h5py.File(output_path, 'r') as f:
            assert f['structures']['atomic_numbers'].shape[0] == 1
            assert f['labels']['energy'].shape[0] == 1

    def test_large_structure(self, temp_dir):
        """Test large structure (100 atoms)."""
        output_path = temp_dir / "test_large.h5"

        # Create random 100-atom structure
        large_structure = Atoms(
            'H100',
            positions=np.random.randn(100, 3) * 5,
            cell=[20, 20, 20],
            pbc=False
        )
        energy = -100.0
        forces = np.random.randn(100, 3) * 0.1

        with HDF5DatasetWriter(output_path) as writer:
            writer.add_structure(large_structure, energy, forces)

        with h5py.File(output_path, 'r') as f:
            assert f['structures']['atomic_numbers'].shape[0] == 100
            assert f['labels']['forces'].shape == (100, 3)

    def test_mixed_structure_sizes(self, temp_dir):
        """Test dataset with varying structure sizes."""
        output_path = temp_dir / "test_mixed.h5"

        # Create structures with different sizes
        structures = [
            Atoms('H', positions=[[0, 0, 0]]),
            Atoms('H2O', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            Atoms('C6H12', positions=np.random.randn(18, 3)),
            Atoms('H50', positions=np.random.randn(50, 3))
        ]

        energies = [-1.0, -10.0, -50.0, -100.0]
        forces = [np.random.randn(len(s), 3) * 0.1 for s in structures]

        with HDF5DatasetWriter(output_path) as writer:
            writer.add_batch(structures, energies, forces)

        with h5py.File(output_path, 'r') as f:
            splits = f['structures']['atomic_numbers_splits'][:]

            # Verify each structure size
            expected_sizes = [1, 3, 18, 50]
            for i, expected in enumerate(expected_sizes):
                actual = splits[i + 1] - splits[i]
                assert actual == expected

    def test_dtype_consistency(self, temp_dir, sample_atoms, sample_labels):
        """Test that dtypes match expected format."""
        output_path = temp_dir / "test_dtypes.h5"
        energies, forces, stresses = sample_labels

        with HDF5DatasetWriter(output_path) as writer:
            writer.add_batch(sample_atoms, energies, forces, stresses)

        with h5py.File(output_path, 'r') as f:
            # Check dtypes match specification
            assert f['structures']['atomic_numbers'].dtype == np.int64
            assert f['structures']['positions'].dtype == np.float64
            assert f['structures']['cells'].dtype == np.float64
            assert f['structures']['pbc'].dtype == bool

            assert f['labels']['energy'].dtype == np.float64
            assert f['labels']['forces'].dtype == np.float32  # Note: float32 for memory
            assert f['labels']['structure_indices'].dtype == np.int64

    def test_finalize_twice(self, temp_dir, sample_atoms, sample_labels):
        """Test that calling finalize() twice doesn't cause errors."""
        output_path = temp_dir / "test_finalize_twice.h5"
        energies, forces, stresses = sample_labels

        writer = HDF5DatasetWriter(output_path)
        writer.add_batch(sample_atoms, energies, forces, stresses)

        writer.finalize()
        writer.finalize()  # Should log warning but not crash

    def test_add_after_finalize(self, temp_dir, sample_atoms, sample_labels):
        """Test that adding structures after finalize raises error."""
        output_path = temp_dir / "test_add_after_finalize.h5"
        energies, forces, stresses = sample_labels

        writer = HDF5DatasetWriter(output_path)
        writer.add_structure(sample_atoms[0], energies[0], forces[0], stresses[0])
        writer.finalize()

        # Should raise error
        with pytest.raises(RuntimeError, match="Cannot add structures after finalize"):
            writer.add_structure(sample_atoms[1], energies[1], forces[1], stresses[1])


class TestConvertPickleToHDF5:
    """Test suite for pickle conversion utility."""

    def test_convert_with_labels(self, temp_dir, sample_atoms, sample_labels):
        """Test converting pickle with provided labels."""
        import pickle

        pickle_path = temp_dir / "structures.pkl"
        hdf5_path = temp_dir / "dataset.h5"

        energies, forces, stresses = sample_labels

        # Save structures to pickle
        with open(pickle_path, 'wb') as f:
            pickle.dump(sample_atoms, f)

        # Convert to HDF5
        convert_pickle_to_hdf5(
            pickle_path,
            hdf5_path,
            energies=energies,
            forces=forces,
            stresses=stresses,
            compression="gzip",
            show_progress=False
        )

        # Verify
        assert hdf5_path.exists()

        with h5py.File(hdf5_path, 'r') as f:
            assert f['labels']['energy'].shape[0] == len(sample_atoms)
            np.testing.assert_allclose(
                f['labels']['energy'][:],
                energies,
                rtol=1e-6
            )


def test_format_compatibility():
    """
    Test that the format matches the existing all_labels_orb_v2.h5 format.

    This test verifies that our writer produces HDF5 files with the same
    structure as the existing label generation pipeline.
    """
    # This test would compare against the actual file
    # For now, we just verify the expected groups and datasets exist

    import tempfile
    temp_dir = Path(tempfile.mkdtemp())

    try:
        output_path = temp_dir / "format_test.h5"

        # Create test data
        atoms = Atoms('H2O', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        energy = -10.0
        forces = np.random.randn(3, 3)

        with HDF5DatasetWriter(output_path) as writer:
            writer.add_structure(atoms, energy, forces)

        # Verify format
        with h5py.File(output_path, 'r') as f:
            # Check required groups
            assert 'structures' in f
            assert 'labels' in f
            assert 'metadata' in f

            # Check structures datasets
            struct_datasets = set(f['structures'].keys())
            expected_struct = {
                'atomic_numbers',
                'atomic_numbers_splits',
                'positions',
                'cells',
                'pbc'
            }
            assert struct_datasets == expected_struct

            # Check labels datasets
            label_datasets = set(f['labels'].keys())
            expected_labels = {
                'energy',
                'forces',
                'forces_splits',
                'structure_indices'
            }
            assert expected_labels.issubset(label_datasets)

    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
