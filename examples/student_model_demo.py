"""
Student Model Demo: PaiNN-based Force Field

This example demonstrates how to:
1. Load structures from the merged HDF5 dataset
2. Create and initialize a student model
3. Compute energy and forces
4. Compare with teacher predictions (if available)
5. Benchmark inference time
6. Save and load model checkpoints

Author: ML Architecture Specialist
Date: 2025-11-24
Issue: M3 #19
"""

import sys
from pathlib import Path
import time

import torch
import h5py
import numpy as np
from ase import Atoms

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from mlff_distiller.models.student_model import StudentForceField


def load_structure_from_hdf5(hdf5_path: Path, structure_idx: int = 0) -> tuple:
    """
    Load a single structure from the merged HDF5 dataset.

    Args:
        hdf5_path: Path to HDF5 file
        structure_idx: Index of structure to load

    Returns:
        Tuple of (atomic_numbers, positions, cell, pbc, energy, forces)
    """
    with h5py.File(hdf5_path, 'r') as f:
        # Get splits
        splits = f['structures']['atomic_numbers_splits'][:]

        # Get atom range for this structure
        start_idx = splits[structure_idx]
        end_idx = splits[structure_idx + 1]
        num_atoms = end_idx - start_idx

        # Load structure data
        atomic_numbers = f['structures']['atomic_numbers'][start_idx:end_idx]
        positions = f['structures']['positions'][start_idx:end_idx]
        cell = f['structures']['cells'][structure_idx]
        pbc = f['structures']['pbc'][structure_idx]

        # Load labels
        energy = f['labels']['energy'][structure_idx]
        forces = f['labels']['forces'][start_idx:end_idx]

        print(f"Loaded structure {structure_idx}:")
        print(f"  Atoms: {num_atoms}")
        print(f"  Elements: {np.unique(atomic_numbers)}")
        print(f"  Energy (teacher): {energy:.4f} eV")
        print(f"  Max force (teacher): {np.abs(forces).max():.4f} eV/Å")

        return atomic_numbers, positions, cell, pbc, energy, forces


def demo_basic_usage():
    """Demonstrate basic student model usage."""
    print("\n" + "="*70)
    print("Demo 1: Basic Student Model Usage")
    print("="*70)

    # Create student model
    model = StudentForceField(
        hidden_dim=128,
        num_interactions=3,
        num_rbf=20,
        cutoff=5.0,
        max_z=118
    )

    print(f"\nStudent Model Initialized:")
    print(f"  Parameters: {model.num_parameters():,}")
    print(f"  Hidden dim: {model.hidden_dim}")
    print(f"  Interactions: {model.num_interactions}")
    print(f"  Cutoff: {model.cutoff} Å")

    # Load a structure from dataset
    dataset_path = project_root / "data" / "merged_dataset_4883" / "merged_dataset.h5"

    if not dataset_path.exists():
        print(f"\nWarning: Dataset not found at {dataset_path}")
        print("Using synthetic water molecule instead...")

        # Create synthetic water molecule
        atomic_numbers = torch.tensor([8, 1, 1], dtype=torch.long)
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [-0.24, 0.93, 0.0]
        ], dtype=torch.float32)
        energy_teacher = None
        forces_teacher = None
    else:
        # Load from dataset
        atomic_numbers_np, positions_np, cell_np, pbc_np, energy_teacher, forces_teacher = \
            load_structure_from_hdf5(dataset_path, structure_idx=0)

        atomic_numbers = torch.from_numpy(atomic_numbers_np).long()
        positions = torch.from_numpy(positions_np).float()

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    model = model.to(device)
    atomic_numbers = atomic_numbers.to(device)
    positions = positions.to(device)

    # Predict energy and forces
    print("\nComputing energy and forces...")
    energy, forces = model.predict_energy_and_forces(atomic_numbers, positions)

    print(f"\nStudent Model Predictions:")
    print(f"  Energy: {energy.item():.4f} eV")
    print(f"  Max force: {forces.abs().max().item():.4f} eV/Å")
    print(f"  Force norm: {forces.norm().item():.4f} eV/Å")

    if energy_teacher is not None:
        print(f"\nComparison with Teacher:")
        print(f"  Energy error: {abs(energy.item() - energy_teacher):.4f} eV")
        forces_mae = np.abs(forces.cpu().detach().numpy() - forces_teacher).mean()
        print(f"  Force MAE: {forces_mae:.4f} eV/Å")
        print(f"  (Note: Untrained model, large errors expected)")


def demo_benchmark():
    """Benchmark inference speed on various system sizes."""
    print("\n" + "="*70)
    print("Demo 2: Inference Speed Benchmark")
    print("="*70)

    # Create model
    model = StudentForceField(hidden_dim=128, num_interactions=3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    print(f"\nBenchmarking on {device}...")
    print(f"{'System Size':<15} {'Time (ms)':<15} {'Atoms/sec':<15}")
    print("-" * 50)

    sizes = [10, 20, 50, 100, 200]

    with torch.no_grad():
        for size in sizes:
            # Create random structure
            atomic_numbers = torch.randint(1, 10, (size,), device=device)
            positions = torch.randn(size, 3, device=device) * 5.0

            # Warmup
            for _ in range(10):
                _ = model(atomic_numbers, positions)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            # Benchmark
            num_runs = 100
            start = time.time()

            for _ in range(num_runs):
                _ = model(atomic_numbers, positions)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            elapsed = time.time() - start
            time_per_structure = (elapsed / num_runs) * 1000  # ms
            atoms_per_sec = size * num_runs / elapsed

            print(f"{size:<15} {time_per_structure:<15.2f} {atoms_per_sec:<15.0f}")


def demo_save_load():
    """Demonstrate model checkpointing."""
    print("\n" + "="*70)
    print("Demo 3: Model Save/Load")
    print("="*70)

    # Create model
    model = StudentForceField(hidden_dim=64, num_interactions=2)

    print(f"\nOriginal Model: {model.num_parameters():,} parameters")

    # Create test input
    atomic_numbers = torch.tensor([6, 8, 1, 1], dtype=torch.long)
    positions = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.2, 0.0, 0.0],
        [1.8, 0.8, 0.0],
        [1.8, -0.8, 0.0]
    ], dtype=torch.float32)

    # Predict with original model
    energy1 = model(atomic_numbers, positions)
    print(f"Energy (original): {energy1.item():.6f} eV")

    # Save model
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "student_model.pt"
        model.save(save_path)
        print(f"\nModel saved to: {save_path}")

        # Load model
        loaded_model = StudentForceField.load(save_path)
        print(f"Model loaded: {loaded_model.num_parameters():,} parameters")

        # Predict with loaded model
        energy2 = loaded_model(atomic_numbers, positions)
        print(f"Energy (loaded): {energy2.item():.6f} eV")

        # Check consistency
        assert torch.allclose(energy1, energy2, atol=1e-6)
        print("\n✓ Save/load consistency verified")


def demo_batch_processing():
    """Demonstrate batch processing of multiple structures."""
    print("\n" + "="*70)
    print("Demo 4: Batch Processing")
    print("="*70)

    # Create model
    model = StudentForceField(hidden_dim=128, num_interactions=3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Create batch of structures with different sizes
    # Structure 1: 5 atoms
    atomic_numbers_1 = torch.randint(1, 10, (5,), device=device)
    positions_1 = torch.randn(5, 3, device=device) * 3.0

    # Structure 2: 8 atoms
    atomic_numbers_2 = torch.randint(1, 10, (8,), device=device)
    positions_2 = torch.randn(8, 3, device=device) * 3.0

    # Concatenate into batch
    atomic_numbers = torch.cat([atomic_numbers_1, atomic_numbers_2])
    positions = torch.cat([positions_1, positions_2])
    batch = torch.cat([
        torch.zeros(5, dtype=torch.long, device=device),
        torch.ones(8, dtype=torch.long, device=device)
    ])

    print(f"\nBatch configuration:")
    print(f"  Structure 1: {len(atomic_numbers_1)} atoms")
    print(f"  Structure 2: {len(atomic_numbers_2)} atoms")
    print(f"  Total atoms: {len(atomic_numbers)}")

    # Predict energies for batch
    energies = model(atomic_numbers, positions, batch=batch)

    print(f"\nPredicted energies:")
    print(f"  Structure 1: {energies[0].item():.4f} eV")
    print(f"  Structure 2: {energies[1].item():.4f} eV")


def demo_ase_integration():
    """Demonstrate ASE integration (requires ASE Calculator wrapper)."""
    print("\n" + "="*70)
    print("Demo 5: ASE Integration")
    print("="*70)

    print("\nNote: Full ASE Calculator integration requires updating")
    print("student_calculator.py with the StudentForceField model.")
    print("This is a placeholder for future integration.\n")

    # Create ASE Atoms object
    from ase import Atoms

    atoms = Atoms(
        'H2O',
        positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [-0.24, 0.93, 0.0]
        ],
        pbc=False
    )

    print(f"Created ASE Atoms object:")
    print(f"  Formula: {atoms.get_chemical_formula()}")
    print(f"  Number of atoms: {len(atoms)}")

    # Direct model usage
    model = StudentForceField()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    atomic_numbers = torch.from_numpy(atoms.numbers).long().to(device)
    positions = torch.from_numpy(atoms.positions).float().to(device)

    energy, forces = model.predict_energy_and_forces(atomic_numbers, positions)

    print(f"\nDirect model predictions:")
    print(f"  Energy: {energy.item():.4f} eV")
    print(f"  Forces:\n{forces.detach().cpu().numpy()}")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("STUDENT MODEL DEMONSTRATION")
    print("PaiNN-based Force Field for ML Distillation")
    print("="*70)

    # Run demos
    demo_basic_usage()
    demo_benchmark()
    demo_save_load()
    demo_batch_processing()
    demo_ase_integration()

    print("\n" + "="*70)
    print("All demos completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
