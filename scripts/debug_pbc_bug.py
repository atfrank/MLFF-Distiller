#!/usr/bin/env python3
"""
Debug script to identify the PBC bug in radius_graph_native.

This script loads a structure and attempts to reproduce the tensor dimension error.
"""

import sys
from pathlib import Path
import torch
import h5py
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from mlff_distiller.models.student_model import StudentForceField, radius_graph_native


def load_structure(hdf5_path: Path, structure_idx: int = 10):
    """Load a single structure from HDF5."""
    with h5py.File(hdf5_path, 'r') as f:
        structures_group = f['structures']
        labels_group = f['labels']

        atom_splits = structures_group['atomic_numbers_splits'][:]
        forces_splits = labels_group['forces_splits'][:]

        atom_start_idx = atom_splits[structure_idx]
        atom_end_idx = atom_splits[structure_idx + 1]

        forces_start_idx = forces_splits[structure_idx]
        forces_end_idx = forces_splits[structure_idx + 1]

        structure = {
            'atomic_numbers': structures_group['atomic_numbers'][atom_start_idx:atom_end_idx],
            'positions': structures_group['positions'][atom_start_idx:atom_end_idx],  # Already (N, 3)
            'energy': labels_group['energy'][structure_idx],
            'forces': labels_group['forces'][forces_start_idx:forces_end_idx].reshape(-1, 3),
            'cell': structures_group['cells'][structure_idx],
            'pbc': structures_group['pbc'][structure_idx],
        }

        return structure


def debug_radius_graph():
    """Debug the radius_graph_native function."""
    print("=" * 80)
    print("PBC Bug Debugging")
    print("=" * 80)

    # Load structure
    hdf5_path = Path('/home/aaron/ATX/software/MLFF_Distiller/data/merged_dataset_4883/merged_dataset.h5')
    print(f"\nLoading structure from: {hdf5_path}")

    structure = load_structure(hdf5_path, structure_idx=10)

    print(f"\nStructure Info:")
    print(f"  N atoms: {len(structure['atomic_numbers'])}")
    print(f"  Atomic numbers: {structure['atomic_numbers']}")
    print(f"  Positions shape: {structure['positions'].shape}")
    print(f"  PBC: {structure['pbc']}")

    # Convert to tensors
    positions = torch.tensor(structure['positions'], dtype=torch.float32)
    atomic_numbers = torch.tensor(structure['atomic_numbers'], dtype=torch.long)
    batch = torch.zeros(len(atomic_numbers), dtype=torch.long)

    print(f"\nTensor shapes:")
    print(f"  positions: {positions.shape}")
    print(f"  atomic_numbers: {atomic_numbers.shape}")
    print(f"  batch: {batch.shape}")

    # Test radius_graph_native with detailed debugging
    print(f"\n" + "=" * 80)
    print("Testing radius_graph_native function")
    print("=" * 80)

    cutoff = 5.0
    num_atoms = positions.shape[0]

    print(f"\nStep 1: Compute pairwise distances")
    print(f"  positions.unsqueeze(0).shape: {positions.unsqueeze(0).shape}")
    print(f"  positions.unsqueeze(1).shape: {positions.unsqueeze(1).shape}")

    diff = positions.unsqueeze(0) - positions.unsqueeze(1)
    print(f"  diff.shape: {diff.shape}")

    distances = torch.norm(diff, dim=2)
    print(f"  distances.shape: {distances.shape}")

    print(f"\nStep 2: Create masks")
    print(f"  batch.shape: {batch.shape}")
    print(f"  batch.unsqueeze(0).shape: {batch.unsqueeze(0).shape}")
    print(f"  batch.unsqueeze(1).shape: {batch.unsqueeze(1).shape}")

    batch_mask = batch.unsqueeze(0) == batch.unsqueeze(1)
    print(f"  batch_mask.shape: {batch_mask.shape}")

    distance_mask = distances <= cutoff
    print(f"  distance_mask.shape: {distance_mask.shape}")

    print(f"\nStep 3: Combine masks")
    print(f"  Attempting: batch_mask & distance_mask")
    print(f"  batch_mask.shape: {batch_mask.shape}")
    print(f"  distance_mask.shape: {distance_mask.shape}")

    try:
        mask = batch_mask & distance_mask
        print(f"  SUCCESS! mask.shape: {mask.shape}")

        # Remove self-loops
        mask = mask & ~torch.eye(num_atoms, dtype=torch.bool)
        print(f"  After self-loop removal: {mask.shape}")

        # Get edges
        src, dst = torch.where(mask)
        print(f"\nEdge info:")
        print(f"  Number of edges: {len(src)}")
        print(f"  src.shape: {src.shape}")
        print(f"  dst.shape: {dst.shape}")

        return True

    except RuntimeError as e:
        print(f"  ERROR: {e}")
        print(f"\n  batch_mask dtype: {batch_mask.dtype}")
        print(f"  distance_mask dtype: {distance_mask.dtype}")
        print(f"  batch_mask device: {batch_mask.device}")
        print(f"  distance_mask device: {distance_mask.device}")
        return False


def test_with_model():
    """Test with actual StudentForceField model."""
    print("\n" + "=" * 80)
    print("Testing with StudentForceField model")
    print("=" * 80)

    # Load structure
    hdf5_path = Path('/home/aaron/ATX/software/MLFF_Distiller/data/merged_dataset_4883/merged_dataset.h5')
    structure = load_structure(hdf5_path, structure_idx=10)

    # Convert to tensors
    positions = torch.tensor(structure['positions'], dtype=torch.float32)
    atomic_numbers = torch.tensor(structure['atomic_numbers'], dtype=torch.long)
    cell = torch.tensor(structure['cell'], dtype=torch.float32).unsqueeze(0)
    pbc = torch.tensor(structure['pbc'], dtype=torch.bool).unsqueeze(0)
    batch = torch.zeros(len(atomic_numbers), dtype=torch.long)

    print(f"\nInput shapes:")
    print(f"  positions: {positions.shape}")
    print(f"  atomic_numbers: {atomic_numbers.shape}")
    print(f"  cell: {cell.shape}")
    print(f"  pbc: {pbc.shape}")
    print(f"  batch: {batch.shape}")

    # Create model
    print(f"\nCreating StudentForceField model...")
    model = StudentForceField(
        num_interactions=3,
        hidden_dim=128,
        num_rbf=20,
        cutoff=5.0
    )
    model.eval()

    print(f"Model created successfully")
    print(f"  Parameters: {model.num_parameters():,}")

    # Forward pass
    print(f"\nAttempting forward pass...")
    try:
        with torch.no_grad():
            energy = model(
                atomic_numbers=atomic_numbers,
                positions=positions,
                cell=cell,
                pbc=pbc,
                batch=batch
            )
        print(f"  SUCCESS! energy: {energy.item():.4f} eV")
        return True

    except RuntimeError as e:
        print(f"  ERROR during forward pass:")
        print(f"  {e}")
        return False


if __name__ == '__main__':
    print("\n")

    # Test radius_graph_native directly
    success_1 = debug_radius_graph()

    # Test with model
    success_2 = test_with_model()

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"radius_graph_native: {'✓ PASS' if success_1 else '✗ FAIL'}")
    print(f"StudentForceField:   {'✓ PASS' if success_2 else '✗ FAIL'}")
    print()
