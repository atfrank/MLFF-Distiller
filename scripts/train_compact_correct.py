#!/usr/bin/env python3
"""
CORRECT Training Script for Tiny and Ultra-tiny Models

Uses REAL molecular structures with pre-computed Orb labels from the HDF5 dataset.
Follows the same pattern as the original student training.

Author: CUDA Optimization Engineer
Date: 2025-11-24
"""

import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import json
import argparse
import sys

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from mlff_distiller.models.student_model import StudentForceField
from mlff_distiller.data.distillation_dataset import DistillationDataset, distillation_collate_fn
from torch.utils.data import DataLoader


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


def train_epoch(model, dataloader, optimizer, device, force_weight=15.0):
    """Train for one epoch."""
    model.train()
    total_energy_loss = 0.0
    total_force_loss = 0.0
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)

    for batch in pbar:
        # Move batch to device
        positions = batch['positions'].to(device)
        atomic_numbers = batch['atomic_numbers'].to(device)
        cell = batch['cell'].to(device)
        pbc = batch['pbc'].to(device)
        batch_idx = batch['batch'].to(device)

        # Teacher labels (pre-computed in dataset)
        teacher_energy = batch['energy'].to(device)
        teacher_forces = batch['forces'].to(device)

        # Student predictions
        # Clone positions and enable gradients for force computation
        positions_student = positions.clone().requires_grad_(True)

        # Forward pass through student model
        student_energy = model(
            atomic_numbers=atomic_numbers,
            positions=positions_student,
            cell=cell,
            pbc=pbc,
            batch=batch_idx
        )

        # Compute forces via automatic differentiation
        student_forces = -torch.autograd.grad(
            outputs=student_energy.sum(),
            inputs=positions_student,
            create_graph=True,  # Need graph for backprop
            retain_graph=False,
        )[0]

        # Compute losses
        energy_loss = nn.functional.mse_loss(student_energy, teacher_energy)
        force_loss = nn.functional.mse_loss(student_forces, teacher_forces)
        loss = energy_loss + force_weight * force_loss

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Track
        total_energy_loss += energy_loss.item()
        total_force_loss += force_loss.item()
        total_loss += loss.item()
        n_batches += 1

        pbar.set_postfix({
            'E': f"{energy_loss.item():.4f}",
            'F': f"{force_loss.item():.4f}"
        })

    return {
        'energy': total_energy_loss / n_batches,
        'force': total_force_loss / n_batches,
        'total': total_loss / n_batches
    }


def validate(model, dataloader, device, force_weight=15.0):
    """Validate model."""
    model.eval()
    total_energy_loss = 0.0
    total_force_loss = 0.0
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            positions = batch['positions'].to(device)
            atomic_numbers = batch['atomic_numbers'].to(device)
            cell = batch['cell'].to(device)
            pbc = batch['pbc'].to(device)
            batch_idx = batch['batch'].to(device)

            # Teacher labels
            teacher_energy = batch['energy'].to(device)
            teacher_forces = batch['forces'].to(device)

            # Student predictions - still need grad for forces
            positions_student = positions.clone().requires_grad_(True)

            with torch.set_grad_enabled(True):
                student_energy = model(
                    atomic_numbers=atomic_numbers,
                    positions=positions_student,
                    cell=cell,
                    pbc=pbc,
                    batch=batch_idx
                )

                student_forces = -torch.autograd.grad(
                    outputs=student_energy.sum(),
                    inputs=positions_student,
                    create_graph=False,
                    retain_graph=False,
                )[0]

            # Compute losses
            energy_loss = nn.functional.mse_loss(student_energy, teacher_energy)
            force_loss = nn.functional.mse_loss(student_forces, teacher_forces)
            loss = energy_loss + force_weight * force_loss

            total_energy_loss += energy_loss.item()
            total_force_loss += force_loss.item()
            total_loss += loss.item()
            n_batches += 1

    return {
        'energy': total_energy_loss / n_batches,
        'force': total_force_loss / n_batches,
        'total': total_loss / n_batches
    }


def train_model(
    config_name: str,
    config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Path,
    epochs: int = 50,
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

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Training history
    history = {
        'train_energy': [],
        'train_force': [],
        'train_total': [],
        'val_energy': [],
        'val_force': [],
        'val_total': []
    }

    best_val_loss = float('inf')
    model_dir = output_dir / config_name
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining for {epochs} epochs...")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    for epoch in range(epochs):
        # Train
        train_losses = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_losses = validate(model, val_loader, device)

        # Track history
        history['train_energy'].append(train_losses['energy'])
        history['train_force'].append(train_losses['force'])
        history['train_total'].append(train_losses['total'])
        history['val_energy'].append(val_losses['energy'])
        history['val_force'].append(val_losses['force'])
        history['val_total'].append(val_losses['total'])

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train - E: {train_losses['energy']:.4f}, F: {train_losses['force']:.4f}, Total: {train_losses['total']:.4f}")
            print(f"  Val   - E: {val_losses['energy']:.4f}, F: {val_losses['force']:.4f}, Total: {val_losses['total']:.4f}")

        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config,
                'total_params': total_params,
                'model_size_mb': model_size_mb,
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            torch.save(checkpoint, model_dir / 'best_model.pt')

    # Save final model
    final_checkpoint = {
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'config': config,
        'total_params': total_params,
        'model_size_mb': model_size_mb,
        'history': history
    }
    torch.save(final_checkpoint, model_dir / 'final_model.pt')

    # Save history
    with open(model_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ {config_name.upper().replace('_', '-')} training complete!")
    print(f"   Best val loss: {best_val_loss:.4f}")
    print(f"   Saved to: {model_dir}")

    return model, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='data/merged_dataset_4883/merged_dataset.h5')
    parser.add_argument('--output-dir', default='checkpoints/compact_models')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--models', nargs='+', choices=['tiny', 'ultra_tiny', 'both'], default=['both'])
    parser.add_argument('--num-workers', type=int, default=4)

    args = parser.parse_args()

    device = args.device
    output_dir = Path(args.output_dir)

    print(f"\n{'='*70}")
    print("COMPACT MODEL TRAINING - CORRECT APPROACH")
    print(f"{'='*70}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")

    # Load dataset
    print(f"\nLoading dataset from {args.dataset}...")
    full_dataset = DistillationDataset(args.dataset)
    n_total = len(full_dataset)

    # Create train/val split
    n_train = int(0.9 * n_total)
    train_indices = list(range(n_train))
    val_indices = list(range(n_train, n_total))

    train_dataset = DistillationDataset(args.dataset, indices=train_indices)
    val_dataset = DistillationDataset(args.dataset, indices=val_indices)

    print(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} val")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=distillation_collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=distillation_collate_fn,
        pin_memory=True
    )

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
            train_loader,
            val_loader,
            output_dir,
            epochs=args.epochs,
            learning_rate=args.lr,
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
        model_dir = output_dir / model_name
        ckpt = torch.load(model_dir / 'best_model.pt', weights_only=False)

        print(f"\n{model_name.upper().replace('_', '-')}:")
        print(f"  Parameters: {ckpt['total_params']:,}")
        print(f"  Size: {ckpt['model_size_mb']:.3f} MB")
        print(f"  Best val loss: {ckpt['best_val_loss']:.4f}")
        print(f"  Checkpoint: {model_dir / 'best_model.pt'}")

    print(f"\n{'='*70}")
    print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
