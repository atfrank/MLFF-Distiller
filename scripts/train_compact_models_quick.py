#!/usr/bin/env python3
"""
Quick Training Script for Tiny and Ultra-tiny Models

Trains both models using progressive distillation from current student.
Optimized for fast training with demonstration dataset.

Author: CUDA Optimization Engineer
Date: 2025-11-24
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import argparse

import sys
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from mlff_distiller.models.student_model import StudentForceField


# Model configurations
CONFIGS = {
    'tiny': {
        'hidden_dim': 64,
        'num_interactions': 2,
        'num_rbf': 12,
        'cutoff': 5.0,
        'max_z': 100,
        'description': 'Tiny model - 78K params, 0.30 MB'
    },
    'ultra_tiny': {
        'hidden_dim': 32,
        'num_interactions': 2,
        'num_rbf': 10,
        'cutoff': 5.0,
        'max_z': 100,
        'description': 'Ultra-tiny model - 22K params, 0.08 MB'
    }
}


class RandomMoleculeDataset(Dataset):
    """Generate random molecular structures for training."""

    def __init__(self, n_samples=500, min_atoms=5, max_atoms=20):
        self.n_samples = n_samples
        self.min_atoms = min_atoms
        self.max_atoms = max_atoms

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        n_atoms = np.random.randint(self.min_atoms, self.max_atoms + 1)

        # Common elements: H, C, N, O, F, S, Cl
        elements = [1, 6, 7, 8, 9, 16, 17]
        atomic_numbers = torch.tensor([np.random.choice(elements) for _ in range(n_atoms)])

        # Random positions in a box
        positions = torch.randn(n_atoms, 3) * 2.5

        return {
            'atomic_numbers': atomic_numbers,
            'positions': positions,
            'n_atoms': n_atoms
        }


def train_model(
    config_name: str,
    config: dict,
    teacher_model: nn.Module,
    output_dir: Path,
    epochs: int = 50,
    n_samples: int = 500,
    learning_rate: float = 5e-4,
    device: str = 'cuda'
):
    """Train a single compact model."""

    print(f"\n{'='*70}")
    print(f"TRAINING {config_name.upper().replace('_', '-')} MODEL")
    print(f"{'='*70}")
    print(f"{config['description']}")

    # Create model
    model = StudentForceField(
        hidden_dim=config['hidden_dim'],
        num_interactions=config['num_interactions'],
        num_rbf=config['num_rbf'],
        cutoff=config['cutoff'],
        max_z=config.get('max_z', 100)
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / 1024 / 1024

    print(f"\nModel statistics:")
    print(f"  Parameters: {total_params:,}")
    print(f"  Size: {model_size_mb:.3f} MB")

    # Create dataset
    dataset = RandomMoleculeDataset(n_samples=n_samples)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Training history
    history = {
        'energy_loss': [],
        'force_loss': [],
        'total_loss': []
    }

    best_loss = float('inf')
    model_dir = output_dir / config_name
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        epoch_losses = {'energy': 0, 'force': 0, 'total': 0}
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for batch in pbar:
            atomic_numbers = batch['atomic_numbers'][0].to(device)
            positions = batch['positions'][0].to(device)

            # Teacher predictions (compute with grad for forces, then detach)
            positions_teacher = positions.clone().requires_grad_(True)
            teacher_energy = teacher_model(atomic_numbers, positions_teacher)
            teacher_forces = -torch.autograd.grad(
                teacher_energy, positions_teacher,
                create_graph=False, retain_graph=False
            )[0]

            # Detach teacher outputs
            teacher_energy = teacher_energy.detach()
            teacher_forces = teacher_forces.detach()

            # Student predictions
            positions_student = positions.clone().requires_grad_(True)
            student_energy = model(atomic_numbers, positions_student)
            student_forces = -torch.autograd.grad(
                student_energy, positions_student,
                create_graph=True, retain_graph=True
            )[0]

            # Losses
            energy_loss = nn.functional.mse_loss(student_energy, teacher_energy)
            force_loss = nn.functional.mse_loss(student_forces, teacher_forces)
            total_loss = energy_loss + 15.0 * force_loss

            # Optimize
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Track
            epoch_losses['energy'] += energy_loss.item()
            epoch_losses['force'] += force_loss.item()
            epoch_losses['total'] += total_loss.item()
            n_batches += 1

            pbar.set_postfix({
                'E': f"{energy_loss.item():.4f}",
                'F': f"{force_loss.item():.4f}"
            })

        # Epoch stats
        avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}
        history['energy_loss'].append(avg_losses['energy'])
        history['force_loss'].append(avg_losses['force'])
        history['total_loss'].append(avg_losses['total'])

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Energy: {avg_losses['energy']:.4f}, "
                  f"Force: {avg_losses['force']:.4f}, "
                  f"Total: {avg_losses['total']:.4f}")

        # Save best
        if avg_losses['total'] < best_loss:
            best_loss = avg_losses['total']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config,
                'total_params': total_params,
                'model_size_mb': model_size_mb,
                'best_loss': best_loss
            }
            torch.save(checkpoint, model_dir / 'best_model.pt')

    # Save final
    final_checkpoint = {
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'config': config,
        'total_params': total_params,
        'model_size_mb': model_size_mb,
        'history': history
    }
    torch.save(final_checkpoint, model_dir / 'final_model.pt')

    with open(model_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ {config_name.upper().replace('_', '-')} training complete!")
    print(f"   Best loss: {best_loss:.4f}")
    print(f"   Saved to: {model_dir}")

    return model, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher', default='checkpoints/best_model.pt')
    parser.add_argument('--output-dir', default='checkpoints/compact_models')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--n-samples', type=int, default=500)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--models', nargs='+', choices=['tiny', 'ultra_tiny', 'both'], default=['both'])

    args = parser.parse_args()

    device = args.device
    output_dir = Path(args.output_dir)

    print(f"\n{'='*70}")
    print("COMPACT MODEL TRAINING")
    print(f"{'='*70}")
    print(f"Teacher: {args.teacher}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Samples: {args.n_samples}")

    # Load teacher
    print(f"\nLoading teacher model...")
    teacher_ckpt = torch.load(args.teacher, map_location=device, weights_only=False)
    teacher_config = teacher_ckpt.get('config', {
        'hidden_dim': 128,
        'num_interactions': 3,
        'num_rbf': 20,
        'cutoff': 5.0,
        'max_z': 100
    })

    teacher = StudentForceField(**teacher_config).to(device)
    teacher.load_state_dict(teacher_ckpt['model_state_dict'])
    teacher.eval()

    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"Teacher: {teacher_params:,} params ({teacher_params * 4 / 1024 / 1024:.2f} MB)")

    # Determine which models to train
    if 'both' in args.models:
        models_to_train = ['tiny', 'ultra_tiny']
    else:
        models_to_train = args.models

    # Train models
    results = {}
    for model_name in models_to_train:
        config = CONFIGS[model_name]
        model, history = train_model(
            model_name,
            config,
            teacher,
            output_dir,
            epochs=args.epochs,
            n_samples=args.n_samples,
            device=device
        )
        results[model_name] = {
            'config': config,
            'history': history
        }

    # Summary
    print(f"\n{'='*70}")
    print("TRAINING SUMMARY")
    print(f"{'='*70}")

    for model_name in models_to_train:
        config = CONFIGS[model_name]
        model_dir = output_dir / model_name

        # Load best model to get stats
        ckpt = torch.load(model_dir / 'best_model.pt', weights_only=False)

        print(f"\n{model_name.upper().replace('_', '-')}:")
        print(f"  Parameters: {ckpt['total_params']:,}")
        print(f"  Size: {ckpt['model_size_mb']:.3f} MB")
        print(f"  Best loss: {ckpt['best_loss']:.4f}")
        print(f"  Checkpoint: {model_dir / 'best_model.pt'}")

    print(f"\n{'='*70}")
    print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
