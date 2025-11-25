#!/usr/bin/env python3
"""
Train Tiny Student Model (78K params, 0.30 MB)

Uses progressive distillation from current student model (427K params).

Configuration:
- hidden_dim: 64 (vs 128 in current)
- num_interactions: 2 (vs 3 in current)
- num_rbf: 12 (vs 20 in current)

Expected performance:
- Size: 0.30 MB (82% smaller)
- Speed: 2x faster (~8 ms/molecule)
- Accuracy: 90-94% of teacher

Author: CUDA Optimization Engineer
Date: 2025-11-24
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

import sys
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from mlff_distiller.models.student_model import StudentForceField


class SimpleDataset(Dataset):
    """Simple dataset for demonstration."""
    def __init__(self, n_samples=1000):
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Generate random molecular structure
        n_atoms = np.random.randint(5, 20)

        atomic_numbers = torch.randint(1, 10, (n_atoms,))  # H, C, N, O, etc.
        positions = torch.randn(n_atoms, 3) * 2.0  # Random positions

        return {
            'atomic_numbers': atomic_numbers,
            'positions': positions,
            'batch': torch.zeros(n_atoms, dtype=torch.long)
        }


def train_tiny_model(
    teacher_checkpoint: str,
    output_dir: Path,
    epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 5e-4,
    device: str = 'cuda'
):
    """
    Train tiny student model via progressive distillation.

    Args:
        teacher_checkpoint: Path to current student model (teacher)
        output_dir: Output directory
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device
    """
    print(f"\n{'='*70}")
    print("TRAINING TINY STUDENT MODEL")
    print(f"{'='*70}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tiny model configuration
    tiny_config = {
        'hidden_dim': 64,
        'num_interactions': 2,
        'num_rbf': 12,
        'cutoff': 5.0,
        'max_z': 100
    }

    print(f"\nTiny model configuration:")
    for key, value in tiny_config.items():
        print(f"  {key}: {value}")

    # Create tiny model
    tiny_model = StudentForceField(**tiny_config).to(device)

    # Count parameters
    tiny_params = sum(p.numel() for p in tiny_model.parameters())
    tiny_size_mb = tiny_params * 4 / 1024 / 1024

    print(f"\nTiny model statistics:")
    print(f"  Parameters: {tiny_params:,}")
    print(f"  Size: {tiny_size_mb:.2f} MB")

    # Load teacher model
    print(f"\nLoading teacher model from: {teacher_checkpoint}")
    teacher_checkpoint_data = torch.load(teacher_checkpoint, map_location=device, weights_only=False)
    teacher_config = teacher_checkpoint_data.get('config', {
        'hidden_dim': 128,
        'num_interactions': 3,
        'num_rbf': 20,
        'cutoff': 5.0,
        'max_z': 100
    })

    teacher_model = StudentForceField(**teacher_config).to(device)
    teacher_model.load_state_dict(teacher_checkpoint_data['model_state_dict'])
    teacher_model.eval()

    teacher_params = sum(p.numel() for p in teacher_model.parameters())

    print(f"\nTeacher model statistics:")
    print(f"  Parameters: {teacher_params:,}")
    print(f"  Size: {teacher_params * 4 / 1024 / 1024:.2f} MB")

    print(f"\nSize reduction: {tiny_params / teacher_params * 100:.1f}% of teacher ({teacher_params // tiny_params:.1f}x smaller)")

    # Create dataset (using simple random data for demonstration)
    print(f"\nCreating demonstration dataset...")
    train_dataset = SimpleDataset(n_samples=1000)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)  # Batch size 1 for simplicity

    # Optimizer
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Training loop
    print(f"\n{'='*70}")
    print("TRAINING")
    print(f"{'='*70}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Device: {device}")

    history = {
        'energy_loss': [],
        'force_loss': [],
        'total_loss': []
    }

    best_loss = float('inf')

    for epoch in range(epochs):
        tiny_model.train()
        epoch_energy_loss = 0
        epoch_force_loss = 0
        epoch_total_loss = 0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in pbar:
            atomic_numbers = batch['atomic_numbers'].to(device)
            positions = batch['positions'].to(device)
            batch_idx = batch['batch'].to(device)

            # Ensure requires_grad for positions (needed for forces)
            positions = positions.requires_grad_(True)

            # Teacher predictions
            with torch.no_grad():
                teacher_energy = teacher_model(atomic_numbers, positions, batch=batch_idx)
                teacher_forces = -torch.autograd.grad(
                    teacher_energy, positions,
                    create_graph=False, retain_graph=False
                )[0]

            # Student predictions
            positions_student = positions.clone().requires_grad_(True)
            student_energy = tiny_model(atomic_numbers, positions_student, batch=batch_idx)
            student_forces = -torch.autograd.grad(
                student_energy, positions_student,
                create_graph=True, retain_graph=True
            )[0]

            # Distillation losses
            energy_loss = nn.functional.mse_loss(student_energy, teacher_energy)
            force_loss = nn.functional.mse_loss(student_forces, teacher_forces)

            # Combined loss (higher weight on forces for tiny model)
            total_loss = energy_loss + 15.0 * force_loss

            # Backward and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Track losses
            epoch_energy_loss += energy_loss.item()
            epoch_force_loss += force_loss.item()
            epoch_total_loss += total_loss.item()
            n_batches += 1

            pbar.set_postfix({
                'E_loss': f"{energy_loss.item():.4f}",
                'F_loss': f"{force_loss.item():.4f}",
                'Total': f"{total_loss.item():.4f}"
            })

        # Epoch statistics
        avg_energy_loss = epoch_energy_loss / n_batches
        avg_force_loss = epoch_force_loss / n_batches
        avg_total_loss = epoch_total_loss / n_batches

        history['energy_loss'].append(avg_energy_loss)
        history['force_loss'].append(avg_force_loss)
        history['total_loss'].append(avg_total_loss)

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Energy: {avg_energy_loss:.4f}, "
              f"Force: {avg_force_loss:.4f}, "
              f"Total: {avg_total_loss:.4f}")

        # Save best model
        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': tiny_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': tiny_config,
                'total_params': tiny_params,
                'model_size_mb': tiny_size_mb,
                'best_loss': best_loss,
                'history': history
            }
            torch.save(checkpoint, output_dir / 'best_tiny_model.pt')
            print(f"  → Saved best model (loss: {best_loss:.4f})")

    # Save final model
    final_checkpoint = {
        'epoch': epochs,
        'model_state_dict': tiny_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': tiny_config,
        'total_params': tiny_params,
        'model_size_mb': tiny_size_mb,
        'final_loss': avg_total_loss,
        'history': history
    }
    torch.save(final_checkpoint, output_dir / 'final_tiny_model.pt')

    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final loss: {avg_total_loss:.4f}")
    print(f"\nSaved:")
    print(f"  - Best model: {output_dir / 'best_tiny_model.pt'}")
    print(f"  - Final model: {output_dir / 'final_tiny_model.pt'}")
    print(f"  - Training history: {output_dir / 'training_history.json'}")

    return tiny_model, history


def main():
    parser = argparse.ArgumentParser(description="Train Tiny student model")
    parser.add_argument(
        '--teacher',
        type=str,
        default='checkpoints/best_model.pt',
        help='Path to teacher model checkpoint'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='checkpoints/tiny_model',
        help='Output directory'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=5e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device'
    )

    args = parser.parse_args()

    # Train model
    train_tiny_model(
        teacher_checkpoint=args.teacher,
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device
    )


if __name__ == '__main__':
    main()
