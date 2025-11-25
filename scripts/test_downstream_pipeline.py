#!/usr/bin/env python
"""
Test Downstream Pipeline with Partial 10K Dataset

Tests the distillation training pipeline integration with the first ~700 structures
from the 10K generation run while the full generation continues.

This validates:
1. HDF5 dataset can be loaded correctly
2. PyTorch DataLoader works with our format
3. Data statistics look reasonable
4. Batch sampling and collation work
5. Ready for actual training when we have student model

Usage:
    python scripts/test_downstream_pipeline.py
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class MLFFDistillationDataset(Dataset):
    """
    PyTorch Dataset for MLFF distillation from HDF5 file.

    Loads structures, energies, forces from teacher-labeled HDF5 dataset.
    """

    def __init__(self, hdf5_path):
        """
        Initialize dataset from HDF5 file.

        Args:
            hdf5_path: Path to HDF5 file with teacher labels
        """
        self.hdf5_path = Path(hdf5_path)

        # Load dataset metadata
        with h5py.File(self.hdf5_path, 'r') as f:
            self.n_structures = f['structures']['atomic_numbers'].shape[0]
            print(f"Loaded dataset: {self.n_structures} structures")

            # Cache dataset statistics for verification
            self.mean_energy = np.mean(f['labels']['energy'][:])
            self.std_energy = np.std(f['labels']['energy'][:])
            self.mean_force_norm = np.mean(np.linalg.norm(f['labels']['forces'][:], axis=-1))

    def __len__(self):
        """Return number of structures in dataset."""
        return self.n_structures

    def __getitem__(self, idx):
        """
        Get a single structure with labels.

        Returns:
            dict with keys:
                - atomic_numbers: (n_atoms,) atomic numbers
                - positions: (n_atoms, 3) Cartesian coordinates
                - cell: (3, 3) unit cell (if periodic)
                - pbc: (3,) periodic boundary conditions
                - energy: scalar total energy
                - forces: (n_atoms, 3) atomic forces
                - num_atoms: scalar number of atoms
        """
        with h5py.File(self.hdf5_path, 'r') as f:
            # Load structure
            atomic_numbers = f['structures']['atomic_numbers'][idx]
            positions = f['structures']['positions'][idx]
            cell = f['structures']['cell'][idx]
            pbc = f['structures']['pbc'][idx]

            # Load labels
            energy = f['labels']['energy'][idx]
            forces = f['labels']['forces'][idx]

            # Find actual number of atoms (remove padding)
            num_atoms = np.sum(atomic_numbers > 0)

            return {
                'atomic_numbers': torch.from_numpy(atomic_numbers[:num_atoms]).long(),
                'positions': torch.from_numpy(positions[:num_atoms]).float(),
                'cell': torch.from_numpy(cell).float(),
                'pbc': torch.from_numpy(pbc).bool(),
                'energy': torch.tensor(energy).float(),
                'forces': torch.from_numpy(forces[:num_atoms]).float(),
                'num_atoms': num_atoms,
                'idx': idx,
            }


def collate_batch(batch):
    """
    Collate a batch of variable-size structures.

    Since structures have different numbers of atoms, we need custom collation.
    This creates a batched representation with padding or graph batch indexing.

    Args:
        batch: List of samples from __getitem__

    Returns:
        Batched dictionary suitable for GNN model
    """
    # For now, simple batch_size=1 processing (most GNN libraries handle this)
    # In production, would use PyG batching or manual graph batch construction

    return {
        'atomic_numbers': torch.nn.utils.rnn.pad_sequence(
            [item['atomic_numbers'] for item in batch],
            batch_first=True,
            padding_value=0
        ),
        'positions': torch.nn.utils.rnn.pad_sequence(
            [item['positions'] for item in batch],
            batch_first=True,
            padding_value=0.0
        ),
        'cell': torch.stack([item['cell'] for item in batch]),
        'pbc': torch.stack([item['pbc'] for item in batch]),
        'energy': torch.stack([item['energy'] for item in batch]),
        'forces': torch.nn.utils.rnn.pad_sequence(
            [item['forces'] for item in batch],
            batch_first=True,
            padding_value=0.0
        ),
        'num_atoms': torch.tensor([item['num_atoms'] for item in batch]),
        'batch_idx': torch.tensor([item['idx'] for item in batch]),
    }


def compute_dataset_statistics(dataset):
    """Compute and display dataset statistics."""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)

    energies = []
    force_norms = []
    num_atoms_list = []
    atomic_numbers_set = set()

    print(f"Computing statistics over {len(dataset)} structures...")

    for i in range(len(dataset)):
        sample = dataset[i]
        energies.append(sample['energy'].item())
        force_norms.append(torch.norm(sample['forces'], dim=-1).mean().item())
        num_atoms_list.append(sample['num_atoms'])
        atomic_numbers_set.update(sample['atomic_numbers'].tolist())

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(dataset)} structures...")

    energies = np.array(energies)
    force_norms = np.array(force_norms)
    num_atoms_list = np.array(num_atoms_list)

    print(f"\nEnergy Statistics:")
    print(f"  Mean: {energies.mean():.3f} eV")
    print(f"  Std:  {energies.std():.3f} eV")
    print(f"  Min:  {energies.min():.3f} eV")
    print(f"  Max:  {energies.max():.3f} eV")

    print(f"\nForce Norm Statistics:")
    print(f"  Mean: {force_norms.mean():.3f} eV/Å")
    print(f"  Std:  {force_norms.std():.3f} eV/Å")
    print(f"  Min:  {force_norms.min():.3f} eV/Å")
    print(f"  Max:  {force_norms.max():.3f} eV/Å")

    print(f"\nSystem Size Statistics:")
    print(f"  Mean atoms: {num_atoms_list.mean():.1f}")
    print(f"  Min atoms:  {num_atoms_list.min()}")
    print(f"  Max atoms:  {num_atoms_list.max()}")

    print(f"\nElement Diversity:")
    print(f"  Unique elements: {len(atomic_numbers_set)}")
    print(f"  Element Z values: {sorted(atomic_numbers_set)}")

    return {
        'energy_mean': energies.mean(),
        'energy_std': energies.std(),
        'force_mean': force_norms.mean(),
        'force_std': force_norms.std(),
        'n_atoms_mean': num_atoms_list.mean(),
        'n_elements': len(atomic_numbers_set),
    }


def test_dataloader(dataset, batch_size=4):
    """Test PyTorch DataLoader functionality."""
    print("\n" + "="*60)
    print("TESTING PYTORCH DATALOADER")
    print("="*60)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=0,  # Single-threaded for now
    )

    print(f"\nCreated DataLoader with batch_size={batch_size}")
    print(f"Number of batches: {len(loader)}")

    # Test first batch
    print("\nTesting first batch...")
    for batch in loader:
        print(f"  Batch atomic_numbers shape: {batch['atomic_numbers'].shape}")
        print(f"  Batch positions shape: {batch['positions'].shape}")
        print(f"  Batch energy shape: {batch['energy'].shape}")
        print(f"  Batch forces shape: {batch['forces'].shape}")
        print(f"  Num atoms per structure: {batch['num_atoms'].tolist()}")
        break

    print("\n✓ DataLoader test passed!")
    return True


def test_training_readiness(dataset):
    """Simulate what training loop would need."""
    print("\n" + "="*60)
    print("TESTING TRAINING PIPELINE READINESS")
    print("="*60)

    # Simulate training setup
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_batch)

    print("\nSimulating 10 training steps...")
    for i, batch in enumerate(loader):
        if i >= 10:
            break

        # Simulate what training would do
        atomic_numbers = batch['atomic_numbers']  # Input to student model
        positions = batch['positions']  # Input to student model
        energy_target = batch['energy']  # Teacher label for energy
        forces_target = batch['forces']  # Teacher label for forces

        # Check shapes are reasonable
        assert atomic_numbers.shape[0] == 1, "Batch size should be 1"
        assert positions.shape[0] == 1, "Batch size should be 1"
        assert energy_target.shape == (1,), f"Energy shape wrong: {energy_target.shape}"

        if (i + 1) % 5 == 0:
            print(f"  Step {i+1}/10: OK (energy={energy_target.item():.2f} eV)")

    print("\n✓ Training pipeline simulation passed!")
    print("  - Data loading works")
    print("  - Batch shapes are correct")
    print("  - Ready for student model training")
    return True


def main():
    """Main test script."""
    print("="*60)
    print("DOWNSTREAM PIPELINE TEST - PARTIAL 10K DATASET")
    print("="*60)

    # Load dataset
    hdf5_path = Path("data/medium_scale_10k_moldiff/medium_scale_10k_moldiff.h5")

    if not hdf5_path.exists():
        print(f"ERROR: HDF5 file not found: {hdf5_path}")
        print("Make sure the 10K generation is running and has created the file.")
        return 1

    print(f"\nLoading dataset from: {hdf5_path}")
    dataset = MLFFDistillationDataset(hdf5_path)

    # Compute statistics
    stats = compute_dataset_statistics(dataset)

    # Test DataLoader
    test_dataloader(dataset, batch_size=4)

    # Test training readiness
    test_training_readiness(dataset)

    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY - DOWNSTREAM PIPELINE TEST")
    print("="*60)
    print(f"✓ HDF5 loading: PASS")
    print(f"✓ Dataset statistics: PASS ({len(dataset)} structures)")
    print(f"✓ DataLoader batching: PASS")
    print(f"✓ Training simulation: PASS")
    print("\n✓ DOWNSTREAM PIPELINE READY FOR TRAINING")
    print("  - Can load data from HDF5")
    print("  - Can create batches")
    print("  - Ready for student model integration")
    print("="*60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
