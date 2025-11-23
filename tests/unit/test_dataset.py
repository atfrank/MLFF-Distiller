"""
Unit tests for MolecularDataset and related classes.

Tests cover:
- Dataset initialization and loading from multiple formats
- ASE Atoms compatibility
- Variable system sizes
- Batching with padding and masking
- Data augmentation and transforms
"""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from ase import Atoms
from ase.db import connect
from ase.io import write

from mlff_distiller.data import (
    MolecularDataLoader,
    MolecularDataset,
    RandomRotation,
    AddNoise,
    NormalizeEnergy,
    train_test_split,
    molecular_collate_fn,
    molecular_collate_fn_no_padding,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_atoms_list():
    """Create a list of ASE Atoms objects with varying sizes."""
    atoms_list = []

    # Small molecule (10 atoms) - water cluster
    positions = np.random.randn(10, 3) * 2.0
    atoms = Atoms('H5O5', positions=positions)
    atoms.info['energy'] = -100.5
    atoms.arrays['forces'] = np.random.randn(10, 3) * 0.1
    atoms_list.append(atoms)

    # Medium molecule (50 atoms)
    positions = np.random.randn(50, 3) * 5.0
    atoms = Atoms('C20H30', positions=positions)
    atoms.info['energy'] = -500.3
    atoms.arrays['forces'] = np.random.randn(50, 3) * 0.1
    atoms_list.append(atoms)

    # Large molecule (100 atoms)
    positions = np.random.randn(100, 3) * 8.0
    atoms = Atoms('C50H50', positions=positions)
    atoms.info['energy'] = -1200.7
    atoms.arrays['forces'] = np.random.randn(100, 3) * 0.1
    atoms_list.append(atoms)

    # Periodic crystal (32 atoms)
    positions = np.random.randn(32, 3) * 5.0
    cell = np.eye(3) * 10.0
    atoms = Atoms('Si32', positions=positions, cell=cell, pbc=True)
    atoms.info['energy'] = -800.2
    atoms.arrays['forces'] = np.random.randn(32, 3) * 0.1
    atoms.info['stress'] = np.random.randn(6) * 0.01
    atoms_list.append(atoms)

    return atoms_list


@pytest.fixture
def ase_db_file(temp_dir, sample_atoms_list):
    """Create an ASE database file with sample data."""
    db_path = temp_dir / 'test.db'
    db = connect(str(db_path))

    for atoms in sample_atoms_list:
        # Store energy and other data
        energy = atoms.info.get('energy')
        forces = atoms.arrays.get('forces')
        stress = atoms.info.get('stress')

        data = {'energy': energy}  # Store energy in data dict
        if forces is not None:
            data['forces'] = forces.tolist()
        if stress is not None:
            data['stress'] = stress.tolist()

        db.write(atoms, data=data)

    return db_path


@pytest.fixture
def hdf5_file(temp_dir, sample_atoms_list):
    """Create an HDF5 file with sample data."""
    h5_path = temp_dir / 'test.h5'

    with h5py.File(h5_path, 'w') as f:
        structures = f.create_group('structures')

        for i, atoms in enumerate(sample_atoms_list):
            group = structures.create_group(str(i))

            # Store atomic data
            group.create_dataset('positions', data=atoms.positions)
            group.create_dataset('species', data=atoms.numbers)
            group.create_dataset('cell', data=atoms.cell.array)
            group.create_dataset('pbc', data=atoms.pbc)

            # Store properties
            if 'energy' in atoms.info:
                group.create_dataset('energy', data=atoms.info['energy'])
            if 'forces' in atoms.arrays:
                group.create_dataset('forces', data=atoms.arrays['forces'])
            if 'stress' in atoms.info:
                group.create_dataset('stress', data=atoms.info['stress'])

    return h5_path


@pytest.fixture
def xyz_file(temp_dir, sample_atoms_list):
    """Create an XYZ file with sample data."""
    xyz_path = temp_dir / 'test.xyz'

    # Write all structures to XYZ file
    write(str(xyz_path), sample_atoms_list)

    return xyz_path


class TestMolecularDataset:
    """Test MolecularDataset class."""

    def test_init_ase_format(self, ase_db_file):
        """Test initialization with ASE database format."""
        dataset = MolecularDataset(ase_db_file, format='ase')
        assert len(dataset) == 4
        assert dataset.format == 'ase'

    def test_init_hdf5_format(self, hdf5_file):
        """Test initialization with HDF5 format."""
        dataset = MolecularDataset(hdf5_file, format='hdf5')
        assert len(dataset) == 4
        assert dataset.format == 'hdf5'

    def test_init_xyz_format(self, xyz_file):
        """Test initialization with XYZ format."""
        dataset = MolecularDataset(xyz_file, format='xyz')
        assert len(dataset) == 4
        assert dataset.format == 'xyz'

    def test_invalid_format(self, temp_dir):
        """Test that invalid format raises error."""
        dummy_file = temp_dir / 'dummy.txt'
        dummy_file.touch()
        with pytest.raises(ValueError, match="Unsupported format"):
            MolecularDataset(dummy_file, format='invalid')

    def test_nonexistent_file(self):
        """Test that nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            MolecularDataset('/nonexistent/path.db', format='ase')

    def test_getitem_returns_dict(self, ase_db_file):
        """Test that __getitem__ returns correct dictionary structure."""
        dataset = MolecularDataset(ase_db_file, format='ase')
        sample = dataset[0]

        # Check required keys
        assert 'positions' in sample
        assert 'species' in sample
        assert 'cell' in sample
        assert 'pbc' in sample
        assert 'natoms' in sample

        # Check types
        assert isinstance(sample['positions'], torch.Tensor)
        assert isinstance(sample['species'], torch.Tensor)
        assert isinstance(sample['natoms'], int)

    def test_getitem_ase_atoms_included(self, ase_db_file):
        """Test that ASE Atoms object is included when requested."""
        dataset = MolecularDataset(ase_db_file, format='ase', return_atoms=True)
        sample = dataset[0]

        assert 'atoms' in sample
        assert isinstance(sample['atoms'], Atoms)

    def test_getitem_ase_atoms_excluded(self, ase_db_file):
        """Test that ASE Atoms can be excluded."""
        dataset = MolecularDataset(ase_db_file, format='ase', return_atoms=False)
        sample = dataset[0]

        assert 'atoms' not in sample

    def test_variable_system_sizes(self, ase_db_file):
        """Test handling of variable system sizes."""
        dataset = MolecularDataset(ase_db_file, format='ase')

        sizes = [dataset[i]['natoms'] for i in range(len(dataset))]
        assert len(set(sizes)) > 1  # Multiple different sizes
        assert min(sizes) >= 10  # Minimum size
        assert max(sizes) <= 100  # Maximum size

    def test_tensor_shapes(self, ase_db_file):
        """Test that tensor shapes are correct."""
        dataset = MolecularDataset(ase_db_file, format='ase')
        sample = dataset[0]

        n_atoms = sample['natoms']
        assert sample['positions'].shape == (n_atoms, 3)
        assert sample['species'].shape == (n_atoms,)
        assert sample['cell'].shape == (3, 3)
        assert sample['pbc'].shape == (3,)

        if 'forces' in sample:
            assert sample['forces'].shape == (n_atoms, 3)

    def test_energy_forces_stress(self, ase_db_file):
        """Test that energies, forces, and stress are loaded."""
        dataset = MolecularDataset(ase_db_file, format='ase')

        # Check that at least some samples have these properties
        has_energy = any('energy' in dataset[i] for i in range(len(dataset)))
        has_forces = any('forces' in dataset[i] for i in range(len(dataset)))

        assert has_energy
        assert has_forces

    def test_caching(self, ase_db_file):
        """Test that caching works correctly."""
        dataset = MolecularDataset(ase_db_file, format='ase', cache=True)

        # Access same sample twice
        sample1 = dataset[0]
        sample2 = dataset[0]

        # Should be identical (from cache)
        assert torch.allclose(sample1['positions'], sample2['positions'])

    def test_get_atoms_method(self, ase_db_file):
        """Test get_atoms convenience method."""
        dataset = MolecularDataset(ase_db_file, format='ase', return_atoms=True)
        atoms = dataset.get_atoms(0)

        assert isinstance(atoms, Atoms)

    def test_get_statistics(self, ase_db_file):
        """Test dataset statistics computation."""
        dataset = MolecularDataset(ase_db_file, format='ase')
        stats = dataset.get_statistics()

        # Check that statistics are computed
        assert 'num_samples' in stats
        assert 'natoms_mean' in stats
        assert 'natoms_std' in stats
        assert 'natoms_min' in stats
        assert 'natoms_max' in stats

        assert stats['num_samples'] == len(dataset)
        assert stats['natoms_min'] > 0
        assert stats['natoms_max'] >= stats['natoms_min']

    def test_different_formats_same_data(self, ase_db_file, hdf5_file, xyz_file):
        """Test that different formats load equivalent data."""
        dataset_ase = MolecularDataset(ase_db_file, format='ase')
        dataset_hdf5 = MolecularDataset(hdf5_file, format='hdf5')
        dataset_xyz = MolecularDataset(xyz_file, format='xyz')

        assert len(dataset_ase) == len(dataset_hdf5) == len(dataset_xyz)

        # Check that system sizes match
        sizes_ase = [dataset_ase[i]['natoms'] for i in range(len(dataset_ase))]
        sizes_hdf5 = [dataset_hdf5[i]['natoms'] for i in range(len(dataset_hdf5))]

        assert sizes_ase == sizes_hdf5


class TestTrainTestSplit:
    """Test train_test_split function."""

    def test_split_ratios(self, ase_db_file):
        """Test that split ratios are correct."""
        dataset = MolecularDataset(ase_db_file, format='ase')
        train, val, test = train_test_split(
            dataset, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25
        )

        total = len(train) + len(val) + len(test)
        assert total == len(dataset)

        # Check approximate ratios (may not be exact due to rounding)
        assert abs(len(train) / total - 0.5) < 0.2
        assert abs(len(val) / total - 0.25) < 0.2
        assert abs(len(test) / total - 0.25) < 0.2

    def test_split_shuffle(self, ase_db_file):
        """Test that shuffling works."""
        dataset = MolecularDataset(ase_db_file, format='ase')

        # Split with different seeds
        train1, _, _ = train_test_split(dataset, random_seed=42)
        train2, _, _ = train_test_split(dataset, random_seed=123)

        # Indices should be different
        assert train1.indices != train2.indices

    def test_split_no_overlap(self, ase_db_file):
        """Test that splits have no overlap."""
        dataset = MolecularDataset(ase_db_file, format='ase')
        train, val, test = train_test_split(dataset)

        train_set = set(train.indices)
        val_set = set(val.indices)
        test_set = set(test.indices)

        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0

    def test_invalid_ratios(self, ase_db_file):
        """Test that invalid ratios raise error."""
        dataset = MolecularDataset(ase_db_file, format='ase')

        with pytest.raises(ValueError, match="must sum to 1.0"):
            train_test_split(dataset, train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)


class TestCollateFunctions:
    """Test collate functions for batching."""

    def test_molecular_collate_fn_padding(self, ase_db_file):
        """Test padded batching."""
        dataset = MolecularDataset(ase_db_file, format='ase')
        samples = [dataset[i] for i in range(3)]

        batch = molecular_collate_fn(samples)

        # Check batch structure
        assert 'positions' in batch
        assert 'species' in batch
        assert 'mask' in batch
        assert 'natoms' in batch
        assert 'batch_size' in batch
        assert 'max_atoms' in batch

        assert batch['batch_size'] == 3
        assert batch['max_atoms'] == max(s['natoms'] for s in samples)

        # Check padding
        assert batch['positions'].shape == (3, batch['max_atoms'], 3)
        assert batch['species'].shape == (3, batch['max_atoms'])
        assert batch['mask'].shape == (3, batch['max_atoms'])

    def test_molecular_collate_fn_mask(self, ase_db_file):
        """Test that padding mask is correct."""
        dataset = MolecularDataset(ase_db_file, format='ase')
        samples = [dataset[i] for i in range(3)]

        batch = molecular_collate_fn(samples)

        # Check mask correctness
        for i, sample in enumerate(samples):
            n_atoms = sample['natoms']
            assert batch['mask'][i, :n_atoms].all()  # True for real atoms
            assert not batch['mask'][i, n_atoms:].any()  # False for padding

    def test_molecular_collate_fn_no_padding(self, ase_db_file):
        """Test graph-based batching without padding."""
        dataset = MolecularDataset(ase_db_file, format='ase')
        samples = [dataset[i] for i in range(3)]

        batch = molecular_collate_fn_no_padding(samples)

        # Check batch structure
        assert 'positions' in batch
        assert 'species' in batch
        assert 'batch' in batch
        assert 'natoms' in batch
        assert 'batch_size' in batch
        assert 'total_atoms' in batch

        total_atoms = sum(s['natoms'] for s in samples)
        assert batch['total_atoms'] == total_atoms
        assert batch['positions'].shape == (total_atoms, 3)
        assert batch['species'].shape == (total_atoms,)
        assert batch['batch'].shape == (total_atoms,)

    def test_molecular_collate_fn_batch_indices(self, ase_db_file):
        """Test that batch indices are correct in graph batching."""
        dataset = MolecularDataset(ase_db_file, format='ase')
        samples = [dataset[i] for i in range(3)]

        batch = molecular_collate_fn_no_padding(samples)

        # Check batch indices
        offset = 0
        for i, sample in enumerate(samples):
            n_atoms = sample['natoms']
            batch_slice = batch['batch'][offset:offset + n_atoms]
            assert (batch_slice == i).all()
            offset += n_atoms


class TestMolecularDataLoader:
    """Test MolecularDataLoader class."""

    def test_dataloader_initialization(self, ase_db_file):
        """Test DataLoader initialization."""
        dataset = MolecularDataset(ase_db_file, format='ase')
        loader = MolecularDataLoader(dataset, batch_size=2, shuffle=False)

        assert loader.batch_size == 2

    def test_dataloader_iteration(self, ase_db_file):
        """Test iterating through DataLoader."""
        dataset = MolecularDataset(ase_db_file, format='ase')
        loader = MolecularDataLoader(dataset, batch_size=2, shuffle=False)

        batches = list(loader)
        assert len(batches) > 0

        # Check first batch
        batch = batches[0]
        assert 'positions' in batch
        assert 'species' in batch
        assert batch['batch_size'] <= 2

    def test_dataloader_padding_mode(self, ase_db_file):
        """Test DataLoader with padding."""
        dataset = MolecularDataset(ase_db_file, format='ase')
        loader = MolecularDataLoader(dataset, batch_size=2, use_padding=True)

        batch = next(iter(loader))
        assert 'mask' in batch
        assert 'max_atoms' in batch

    def test_dataloader_graph_mode(self, ase_db_file):
        """Test DataLoader with graph concatenation."""
        dataset = MolecularDataset(ase_db_file, format='ase')
        loader = MolecularDataLoader(dataset, batch_size=2, use_padding=False)

        batch = next(iter(loader))
        assert 'batch' in batch
        assert 'total_atoms' in batch


class TestTransforms:
    """Test data transformation classes."""

    def test_random_rotation(self, ase_db_file):
        """Test RandomRotation transform."""
        dataset = MolecularDataset(ase_db_file, format='ase')
        sample = dataset[0]

        original_positions = sample['positions'].clone()
        transform = RandomRotation(p=1.0, random_state=42)
        transformed = transform(sample)

        # Positions should be different after rotation
        assert not torch.allclose(transformed['positions'], original_positions)

        # But distances should be preserved
        center = original_positions.mean(dim=0)
        original_dists = torch.sqrt(((original_positions - center) ** 2).sum(dim=1))
        transformed_dists = torch.sqrt(
            ((transformed['positions'] - transformed['positions'].mean(dim=0)) ** 2).sum(dim=1)
        )
        assert torch.allclose(original_dists, transformed_dists, atol=1e-5)

    def test_add_noise(self, ase_db_file):
        """Test AddNoise transform."""
        dataset = MolecularDataset(ase_db_file, format='ase')
        sample = dataset[0]

        original_positions = sample['positions'].clone()
        transform = AddNoise(std=0.01, p=1.0, random_state=42)
        transformed = transform(sample)

        # Positions should be different but close
        assert not torch.allclose(transformed['positions'], original_positions)

        # Difference should be small
        diff = (transformed['positions'] - original_positions).abs().max()
        assert diff < 0.1  # Should be small noise

    def test_normalize_energy(self, ase_db_file):
        """Test NormalizeEnergy transform."""
        dataset = MolecularDataset(ase_db_file, format='ase')
        sample = dataset[0]

        if 'energy' not in sample:
            pytest.skip("Sample doesn't have energy")

        original_energy = sample['energy'].clone()
        transform = NormalizeEnergy(mean=-500.0, std=100.0, per_atom=False)
        transformed = transform(sample)

        # Energy should be normalized
        expected = (original_energy - (-500.0)) / 100.0
        assert torch.allclose(transformed['energy'], expected)

    def test_transform_preserves_keys(self, ase_db_file):
        """Test that transforms preserve dictionary keys."""
        dataset = MolecularDataset(ase_db_file, format='ase')
        sample = dataset[0]

        original_keys = set(sample.keys())
        transform = RandomRotation(p=1.0)
        transformed = transform(sample)

        assert set(transformed.keys()) == original_keys


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_pipeline_ase(self, ase_db_file):
        """Test complete pipeline with ASE database."""
        # Create dataset
        dataset = MolecularDataset(ase_db_file, format='ase')

        # Split data
        train, val, test = train_test_split(dataset)

        # Create loaders
        train_loader = MolecularDataLoader(train, batch_size=2)
        val_loader = MolecularDataLoader(val, batch_size=2)

        # Iterate through batches
        for batch in train_loader:
            assert batch['positions'].shape[0] <= 2
            assert 'energy' in batch or 'forces' in batch

    def test_full_pipeline_with_transforms(self, ase_db_file):
        """Test pipeline with data augmentation."""
        from mlff_distiller.data import Compose

        # Create dataset with transforms
        transform = Compose([
            RandomRotation(p=0.5),
            AddNoise(std=0.01),
        ])

        dataset = MolecularDataset(ase_db_file, format='ase', transform=transform)
        loader = MolecularDataLoader(dataset, batch_size=2)

        # Should work without errors
        batch = next(iter(loader))
        assert batch is not None

    def test_compatibility_with_ase_atoms(self, ase_db_file):
        """Test that loaded data is compatible with ASE Atoms."""
        dataset = MolecularDataset(ase_db_file, format='ase', return_atoms=True)
        sample = dataset[0]

        atoms = sample['atoms']
        assert isinstance(atoms, Atoms)

        # Verify data consistency
        assert len(atoms) == sample['natoms']
        assert np.allclose(atoms.positions, sample['positions'].numpy())
        assert np.array_equal(atoms.numbers, sample['species'].numpy())
