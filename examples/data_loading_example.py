"""
Example script demonstrating data loading infrastructure usage.

This script shows how to:
1. Load molecular data from different formats
2. Split into train/val/test sets
3. Apply data augmentation and normalization
4. Create dataloaders for training
5. Iterate through batches

Run this example after creating sample data or use your own dataset.
"""

import tempfile
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.db import connect
from ase.io import write

from mlff_distiller.data import (
    AddNoise,
    Compose,
    MolecularDataLoader,
    MolecularDataset,
    NormalizeEnergy,
    NormalizeForces,
    RandomRotation,
    create_dataloaders,
    train_test_split,
)


def create_sample_database(db_path: Path, n_samples: int = 100):
    """
    Create a sample ASE database with random molecular structures.

    Args:
        db_path: Path to database file
        n_samples: Number of structures to generate
    """
    print(f"Creating sample database with {n_samples} structures...")

    db = connect(str(db_path))

    for i in range(n_samples):
        # Random system size (10-100 atoms)
        n_atoms = np.random.randint(10, 101)

        # Random atomic numbers (C, H, O, N)
        species = np.random.choice([1, 6, 7, 8], size=n_atoms)

        # Random positions
        positions = np.random.randn(n_atoms, 3) * 5.0

        # Create Atoms object
        atoms = Atoms(numbers=species, positions=positions)

        # Generate random properties
        energy = np.random.randn() * 100.0 - 500.0  # Random energy around -500 eV
        forces = np.random.randn(n_atoms, 3) * 0.1  # Small random forces

        # Add periodic boundary conditions for some samples
        if i % 3 == 0:
            cell = np.eye(3) * 10.0
            atoms.set_cell(cell)
            atoms.set_pbc(True)
            stress = np.random.randn(6) * 0.01
        else:
            stress = None

        # Write to database
        data = {
            'energy': energy,
            'forces': forces.tolist()
        }
        if stress is not None:
            data['stress'] = stress.tolist()

        db.write(atoms, data=data)

    print(f"Created database at: {db_path}")
    return db_path


def example_basic_loading():
    """Example 1: Basic dataset loading and inspection."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Dataset Loading")
    print("=" * 70)

    # Create temporary database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / 'structures.db'
        create_sample_database(db_path, n_samples=50)

        # Load dataset
        dataset = MolecularDataset(db_path, format='ase')
        print(f"\nLoaded dataset with {len(dataset)} samples")

        # Inspect first sample
        sample = dataset[0]
        print(f"\nFirst sample:")
        print(f"  Number of atoms: {sample['natoms']}")
        print(f"  Positions shape: {sample['positions'].shape}")
        print(f"  Species shape: {sample['species'].shape}")
        print(f"  Has energy: {'energy' in sample}")
        print(f"  Has forces: {'forces' in sample}")
        print(f"  Has ASE Atoms: {'atoms' in sample}")

        # Get dataset statistics
        print("\nComputing dataset statistics...")
        stats = dataset.get_statistics()
        print(f"  Samples: {stats['num_samples']}")
        print(f"  Atoms per sample: {stats['natoms_mean']:.1f} ± {stats['natoms_std']:.1f}")
        print(f"  Range: {stats['natoms_min']} - {stats['natoms_max']} atoms")
        print(f"  Energy per atom: {stats['energy_per_atom_mean']:.3f} ± "
              f"{stats['energy_per_atom_std']:.3f} eV")
        print(f"  Forces RMS: {stats['forces_rms']:.3f} eV/Å")


def example_train_val_test_split():
    """Example 2: Splitting data into train/val/test sets."""
    print("\n" + "=" * 70)
    print("Example 2: Train/Val/Test Split")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / 'structures.db'
        create_sample_database(db_path, n_samples=100)

        # Load dataset
        dataset = MolecularDataset(db_path, format='ase')

        # Split data
        train_dataset, val_dataset, test_dataset = train_test_split(
            dataset,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            shuffle=True,
            random_seed=42
        )

        print(f"\nDataset split:")
        print(f"  Training: {len(train_dataset)} samples")
        print(f"  Validation: {len(val_dataset)} samples")
        print(f"  Test: {len(test_dataset)} samples")


def example_data_augmentation():
    """Example 3: Data augmentation with transforms."""
    print("\n" + "=" * 70)
    print("Example 3: Data Augmentation")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / 'structures.db'
        create_sample_database(db_path, n_samples=50)

        # Get statistics for normalization
        base_dataset = MolecularDataset(db_path, format='ase')
        stats = base_dataset.get_statistics()

        # Create augmentation pipeline
        transform = Compose([
            RandomRotation(p=0.5),  # 50% chance of rotation
            AddNoise(std=0.01, p=0.3),  # 30% chance of position noise
            NormalizeEnergy(
                mean=stats['energy_per_atom_mean'],
                std=stats['energy_per_atom_std'],
                per_atom=True
            ),
            NormalizeForces(rms=stats['forces_rms']),
        ])

        # Load dataset with transforms
        dataset = MolecularDataset(db_path, format='ase', transform=transform)

        # Compare original and transformed
        print("\nTransform pipeline:")
        print("  1. Random rotation (p=0.5)")
        print("  2. Position noise (std=0.01 Å, p=0.3)")
        print("  3. Energy normalization (per-atom)")
        print("  4. Force normalization (RMS)")

        sample = dataset[0]
        print(f"\nTransformed sample:")
        print(f"  Positions shape: {sample['positions'].shape}")
        print(f"  Normalized energy: {sample['energy'].item():.3f}")
        if 'forces' in sample:
            print(f"  Normalized forces RMS: {torch.sqrt((sample['forces']**2).mean()).item():.3f}")


def example_batching_strategies():
    """Example 4: Different batching strategies."""
    print("\n" + "=" * 70)
    print("Example 4: Batching Strategies")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / 'structures.db'
        create_sample_database(db_path, n_samples=50)

        dataset = MolecularDataset(db_path, format='ase')

        # Strategy 1: Padded batching
        print("\nStrategy 1: Padded Batching (use_padding=True)")
        loader_padded = MolecularDataLoader(
            dataset,
            batch_size=8,
            shuffle=False,
            use_padding=True
        )

        batch = next(iter(loader_padded))
        print(f"  Batch size: {batch['batch_size']}")
        max_atoms = batch.get('max_atoms', batch['positions'].shape[1])
        print(f"  Max atoms: {max_atoms}")
        print(f"  Positions shape: {batch['positions'].shape}")
        print(f"  Mask shape: {batch['mask'].shape}")
        print(f"  Total elements: {batch['positions'].numel()}")

        # Calculate padding ratio
        real_atoms = batch['natoms'].sum().item()
        total_atoms = batch['batch_size'] * max_atoms
        padding_ratio = 1.0 - (real_atoms / total_atoms)
        print(f"  Padding ratio: {padding_ratio:.1%}")

        # Strategy 2: Graph-based batching
        print("\nStrategy 2: Graph-Based Batching (use_padding=False)")
        loader_graph = MolecularDataLoader(
            dataset,
            batch_size=8,
            shuffle=False,
            use_padding=False
        )

        batch = next(iter(loader_graph))
        print(f"  Batch size: {batch['batch_size']}")
        print(f"  Total atoms: {batch['total_atoms']}")
        print(f"  Positions shape: {batch['positions'].shape}")
        print(f"  Batch indices shape: {batch['batch'].shape}")
        print(f"  Total elements: {batch['positions'].numel()}")

        # Memory comparison
        padded_elements = loader_padded.batch_size * max_atoms * 3
        graph_elements = batch['total_atoms'] * 3
        memory_saving = 1.0 - (graph_elements / padded_elements)
        print(f"  Memory saving vs padded: {memory_saving:.1%}")


def example_complete_training_pipeline():
    """Example 5: Complete training pipeline setup."""
    print("\n" + "=" * 70)
    print("Example 5: Complete Training Pipeline")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / 'structures.db'
        create_sample_database(db_path, n_samples=100)

        # Step 1: Load and analyze data
        print("\nStep 1: Loading and analyzing data...")
        full_dataset = MolecularDataset(db_path, format='ase')
        stats = full_dataset.get_statistics()
        print(f"  Dataset: {stats['num_samples']} samples")

        # Step 2: Create transforms
        print("\nStep 2: Creating transforms...")
        train_transform = Compose([
            RandomRotation(p=0.5),
            AddNoise(std=0.01, p=0.3),
            NormalizeEnergy(
                mean=stats['energy_per_atom_mean'],
                std=stats['energy_per_atom_std'],
                per_atom=True
            ),
            NormalizeForces(rms=stats['forces_rms']),
        ])

        eval_transform = Compose([
            NormalizeEnergy(
                mean=stats['energy_per_atom_mean'],
                std=stats['energy_per_atom_std'],
                per_atom=True
            ),
            NormalizeForces(rms=stats['forces_rms']),
        ])
        print("  Training: augmentation + normalization")
        print("  Evaluation: normalization only")

        # Step 3: Split data
        print("\nStep 3: Splitting data...")
        train_dataset, val_dataset, test_dataset = train_test_split(
            full_dataset,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            shuffle=True,
            random_seed=42
        )
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Val: {len(val_dataset)} samples")
        print(f"  Test: {len(test_dataset)} samples")

        # Step 4: Create dataloaders
        print("\nStep 4: Creating dataloaders...")
        loaders = create_dataloaders(
            train_dataset,
            val_dataset,
            test_dataset,
            batch_size=16,
            num_workers=0,  # Use 0 for example, increase for real training
            use_padding=True
        )
        print("  Created train, val, and test loaders")

        # Step 5: Simulate training loop
        print("\nStep 5: Simulating training loop...")
        train_loader = loaders['train']

        for epoch in range(2):  # Just 2 epochs for demonstration
            print(f"\n  Epoch {epoch + 1}:")
            total_energy_loss = 0.0
            n_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                # Simulate forward pass
                positions = batch['positions']
                species = batch['species']
                mask = batch['mask']
                energy = batch['energy']
                forces = batch['forces']

                # Simulate predictions (random for this example)
                pred_energy = torch.randn_like(energy)

                # Compute loss
                energy_loss = torch.nn.functional.mse_loss(pred_energy, energy)
                total_energy_loss += energy_loss.item()
                n_batches += 1

                if batch_idx == 0:
                    print(f"    Batch shape: {positions.shape}")
                    print(f"    Mask shape: {mask.shape}")

            avg_loss = total_energy_loss / n_batches
            print(f"    Average loss: {avg_loss:.4f}")
            print(f"    Processed {n_batches} batches")


def example_ase_compatibility():
    """Example 6: ASE Atoms compatibility demonstration."""
    print("\n" + "=" * 70)
    print("Example 6: ASE Atoms Compatibility")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / 'structures.db'
        create_sample_database(db_path, n_samples=10)

        # Load dataset with ASE Atoms
        dataset = MolecularDataset(db_path, format='ase', return_atoms=True)

        sample = dataset[0]
        atoms = sample['atoms']

        print(f"\nASE Atoms object:")
        print(f"  Type: {type(atoms)}")
        print(f"  Formula: {atoms.get_chemical_formula()}")
        print(f"  Number of atoms: {len(atoms)}")
        print(f"  Positions shape: {atoms.positions.shape}")
        print(f"  Periodic: {atoms.pbc.any()}")

        # ASE Atoms can be used with ASE calculators
        print("\nCompatibility with ASE:")
        print("  ✓ Can be used with ASE calculators")
        print("  ✓ Compatible with ASE MD integrators")
        print("  ✓ Can use ASE analysis tools")
        print("  ✓ Drop-in replacement in MD workflows")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("MLFF Distiller - Data Loading Examples")
    print("=" * 70)

    examples = [
        ("Basic Loading", example_basic_loading),
        ("Train/Val/Test Split", example_train_val_test_split),
        ("Data Augmentation", example_data_augmentation),
        ("Batching Strategies", example_batching_strategies),
        ("Complete Pipeline", example_complete_training_pipeline),
        ("ASE Compatibility", example_ase_compatibility),
    ]

    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\nError in {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)


if __name__ == '__main__':
    main()
