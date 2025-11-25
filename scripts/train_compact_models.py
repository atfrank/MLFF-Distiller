#!/usr/bin/env python3
"""
Progressive Distillation Training for Compact Student Models

Trains Tiny and Ultra-tiny student models via progressive distillation from
the current student model (427K params). This provides better stability and
accuracy than distilling directly from Orb.

Architecture configs:
- Tiny: 78K params (18% of current), hidden_dim=64, num_interactions=2, num_rbf=12
- Ultra-tiny: 22K params (5% of current), hidden_dim=32, num_interactions=2, num_rbf=10

Progressive distillation chain:
    Orb teacher (187M) → Current student (427K) → Tiny (78K) / Ultra-tiny (22K)

Uses the actual training dataset:
- Structures: data/merged_dataset_4883/merged_dataset.h5
- Labels: Teacher predictions already in the HDF5 file

Author: Lead Coordinator
Date: 2025-11-24
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json

# Add src to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from mlff_distiller.models.student_model import StudentForceField
from mlff_distiller.data.distillation_dataset import DistillationDataset, distillation_collate_fn


# Model configurations
CONFIGS = {
    'tiny': {
        'hidden_dim': 64,
        'num_interactions': 2,
        'num_rbf': 12,
        'cutoff': 5.0,
        'max_z': 100,
        'description': 'Tiny model - 78K params, 0.30 MB',
        'target_accuracy': '90-94% of current student',
        'target_speedup': '2x faster',
    },
    'ultra_tiny': {
        'hidden_dim': 32,
        'num_interactions': 2,
        'num_rbf': 10,
        'cutoff': 5.0,
        'max_z': 100,
        'description': 'Ultra-tiny model - 22K params, 0.08 MB',
        'target_accuracy': '80-88% of current student',
        'target_speedup': '3x faster',
    }
}


def setup_logging(log_dir: Path, model_name: str) -> logging.Logger:
    """Setup logging for training."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{model_name}_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def load_teacher_model(
    checkpoint_path: Path,
    device: str
) -> Tuple[nn.Module, Dict]:
    """Load the current student model to use as teacher for progressive distillation."""
    logger = logging.getLogger(__name__)

    logger.info(f"Loading teacher model from: {checkpoint_path}")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get teacher config
    if 'config' in ckpt:
        teacher_config = ckpt['config']
    else:
        # Default to current student config
        teacher_config = {
            'hidden_dim': 128,
            'num_interactions': 3,
            'num_rbf': 20,
            'cutoff': 5.0,
            'max_z': 100
        }
        logger.warning("No config found in checkpoint, using default student config")

    # Create model
    teacher = StudentForceField(**teacher_config).to(device)

    # Load weights
    if 'model_state_dict' in ckpt:
        teacher.load_state_dict(ckpt['model_state_dict'])
    elif 'state_dict' in ckpt:
        teacher.load_state_dict(ckpt['state_dict'])
    else:
        # Try loading checkpoint directly as state dict
        teacher.load_state_dict(ckpt)

    teacher.eval()

    # Get teacher stats
    teacher_params = sum(p.numel() for p in teacher.parameters())
    teacher_size_mb = teacher_params * 4 / 1024 / 1024

    logger.info(f"Teacher model loaded successfully:")
    logger.info(f"  Parameters: {teacher_params:,}")
    logger.info(f"  Size: {teacher_size_mb:.2f} MB")
    logger.info(f"  Config: {teacher_config}")

    return teacher, teacher_config


def create_student_model(
    config: Dict,
    device: str
) -> Tuple[nn.Module, int, float]:
    """Create student model from config."""
    logger = logging.getLogger(__name__)

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

    logger.info(f"Created student model:")
    logger.info(f"  Parameters: {total_params:,}")
    logger.info(f"  Size: {model_size_mb:.3f} MB")
    logger.info(f"  Hidden dim: {config['hidden_dim']}")
    logger.info(f"  Interactions: {config['num_interactions']}")
    logger.info(f"  RBF functions: {config['num_rbf']}")

    return model, total_params, model_size_mb


def create_dataloaders(
    data_path: Path,
    batch_size: int,
    train_ratio: float,
    num_workers: int,
    seed: int
) -> Tuple[DataLoader, DataLoader]:
    """Create train/val dataloaders from HDF5 dataset."""
    logger = logging.getLogger(__name__)

    logger.info(f"Loading dataset from: {data_path}")

    # Load full dataset
    full_dataset = DistillationDataset(data_path)
    n_total = len(full_dataset)

    logger.info(f"Total structures: {n_total}")

    # Create train/val split
    indices = list(range(n_total))
    np.random.seed(seed)
    np.random.shuffle(indices)

    n_train = int(n_total * train_ratio)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    logger.info(f"Split: {n_train} train, {len(val_indices)} val ({train_ratio*100:.0f}/{(1-train_ratio)*100:.0f})")

    # Create datasets
    train_dataset = DistillationDataset(data_path, indices=train_indices)
    val_dataset = DistillationDataset(data_path, indices=val_indices)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=distillation_collate_fn,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=distillation_collate_fn,
        persistent_workers=num_workers > 0,
    )

    logger.info(f"Created dataloaders: {len(train_loader)} train batches, {len(val_loader)} val batches")

    return train_loader, val_loader


def compute_batch_predictions(
    model: nn.Module,
    batch: Dict,
    device: str,
    compute_forces: bool = True,
    create_graph: bool = True
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute energy and force predictions for a batch.

    Handles batched graph data properly using batch indices.

    Args:
        model: The model to use for predictions
        batch: Batch dictionary with positions, atomic_numbers, etc.
        device: Device to use
        compute_forces: Whether to compute forces
        create_graph: Whether to create computation graph for forces (needed for student, not for teacher)
    """
    positions = batch['positions'].to(device)
    atomic_numbers = batch['atomic_numbers'].to(device)
    batch_idx = batch['batch'].to(device)
    n_atoms = batch['n_atoms'].to(device)
    batch_size = batch['batch_size']

    if compute_forces:
        positions.requires_grad_(True)

    # Compute energies for all structures in batch
    # The model needs to handle batched data
    energies = []
    forces = []

    atom_splits = batch['atom_splits']
    for i in range(batch_size):
        start = atom_splits[i]
        end = atom_splits[i + 1]

        z_i = atomic_numbers[start:end]

        if compute_forces:
            # Clone positions to enable gradient tracking
            pos_i = positions[start:end].clone().requires_grad_(True)
        else:
            pos_i = positions[start:end]

        # Forward pass
        energy_i = model(z_i, pos_i)
        energies.append(energy_i)

        if compute_forces:
            # Compute forces
            # Use retain_graph=True for all but the last iteration
            is_last = (i == batch_size - 1)
            force_i = -torch.autograd.grad(
                energy_i,
                pos_i,
                create_graph=create_graph,
                retain_graph=(not is_last and create_graph)
            )[0]
            forces.append(force_i)

    # Stack energies
    energies = torch.stack(energies)

    # Concatenate forces
    if compute_forces:
        forces = torch.cat(forces, dim=0)
        return energies, forces
    else:
        return energies, None


def train_epoch(
    model: nn.Module,
    teacher: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    energy_weight: float,
    force_weight: float,
    grad_clip: float,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch using progressive distillation."""
    model.train()
    teacher.eval()

    epoch_losses = {'energy': 0.0, 'force': 0.0, 'total': 0.0}
    n_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

    for batch in pbar:
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Get teacher predictions (no graph, then detach)
        teacher_energies, teacher_forces = compute_batch_predictions(
            teacher, batch, device, compute_forces=True, create_graph=False
        )
        teacher_energies = teacher_energies.detach()
        teacher_forces = teacher_forces.detach()

        # Get student predictions (with graph for backprop)
        student_energies, student_forces = compute_batch_predictions(
            model, batch, device, compute_forces=True, create_graph=True
        )

        # Compute losses
        energy_loss = nn.functional.mse_loss(student_energies, teacher_energies)
        force_loss = nn.functional.mse_loss(student_forces, teacher_forces)
        total_loss = energy_weight * energy_loss + force_weight * force_loss

        # Backward
        optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # Track losses
        epoch_losses['energy'] += energy_loss.item()
        epoch_losses['force'] += force_loss.item()
        epoch_losses['total'] += total_loss.item()
        n_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'E': f"{energy_loss.item():.4f}",
            'F': f"{force_loss.item():.4f}",
            'Total': f"{total_loss.item():.4f}"
        })

    # Average losses
    avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}

    return avg_losses


def validate(
    model: nn.Module,
    teacher: nn.Module,
    val_loader: DataLoader,
    device: str,
    energy_weight: float,
    force_weight: float,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    teacher.eval()

    val_losses = {'energy': 0.0, 'force': 0.0, 'total': 0.0}
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Get predictions
            teacher_energies, teacher_forces = compute_batch_predictions(
                teacher, batch, device, compute_forces=True
            )
            student_energies, student_forces = compute_batch_predictions(
                model, batch, device, compute_forces=True
            )

            # Compute losses
            energy_loss = nn.functional.mse_loss(student_energies, teacher_energies)
            force_loss = nn.functional.mse_loss(student_forces, teacher_forces)
            total_loss = energy_weight * energy_loss + force_weight * force_loss

            val_losses['energy'] += energy_loss.item()
            val_losses['force'] += force_loss.item()
            val_losses['total'] += total_loss.item()
            n_batches += 1

    # Average losses
    avg_losses = {k: v / n_batches for k, v in val_losses.items()}

    return avg_losses


def train_model(
    model_name: str,
    config: Dict,
    teacher: nn.Module,
    data_path: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    energy_weight: float,
    force_weight: float,
    grad_clip: float,
    train_ratio: float,
    num_workers: int,
    device: str,
    seed: int,
) -> Tuple[nn.Module, Dict]:
    """Train a compact model via progressive distillation."""
    logger = logging.getLogger(__name__)

    logger.info("="*70)
    logger.info(f"TRAINING {model_name.upper().replace('_', '-')} MODEL")
    logger.info("="*70)
    logger.info(f"{config['description']}")
    logger.info(f"Target accuracy: {config['target_accuracy']}")
    logger.info(f"Target speedup: {config['target_speedup']}")
    logger.info("")

    # Create student model
    model, total_params, model_size_mb = create_student_model(config, device)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_path, batch_size, train_ratio, num_workers, seed
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5
    )

    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-7
    )

    # Training history
    history = {
        'train_energy_loss': [],
        'train_force_loss': [],
        'train_total_loss': [],
        'val_energy_loss': [],
        'val_force_loss': [],
        'val_total_loss': [],
        'learning_rate': [],
    }

    best_val_loss = float('inf')
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nTraining configuration:")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Energy weight: {energy_weight}")
    logger.info(f"  Force weight: {force_weight}")
    logger.info(f"  Gradient clip: {grad_clip}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Output: {model_dir}")
    logger.info("")

    logger.info("Starting training...")

    for epoch in range(1, epochs + 1):
        # Train
        train_losses = train_epoch(
            model, teacher, train_loader, optimizer, device,
            energy_weight, force_weight, grad_clip, epoch
        )

        # Validate
        val_losses = validate(
            model, teacher, val_loader, device,
            energy_weight, force_weight
        )

        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Record history
        history['train_energy_loss'].append(train_losses['energy'])
        history['train_force_loss'].append(train_losses['force'])
        history['train_total_loss'].append(train_losses['total'])
        history['val_energy_loss'].append(val_losses['energy'])
        history['val_force_loss'].append(val_losses['force'])
        history['val_total_loss'].append(val_losses['total'])
        history['learning_rate'].append(current_lr)

        # Log progress
        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch}/{epochs} - "
                f"Train: E={train_losses['energy']:.4f}, F={train_losses['force']:.4f}, "
                f"Total={train_losses['total']:.4f} | "
                f"Val: E={val_losses['energy']:.4f}, F={val_losses['force']:.4f}, "
                f"Total={val_losses['total']:.4f} | "
                f"LR={current_lr:.2e}"
            )

        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'total_params': total_params,
                'model_size_mb': model_size_mb,
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
            }
            torch.save(checkpoint, model_dir / 'best_model.pt')
            logger.info(f"  → Saved best model (val_loss={best_val_loss:.4f})")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'total_params': total_params,
                'model_size_mb': model_size_mb,
                'best_val_loss': best_val_loss,
                'history': history,
            }
            torch.save(checkpoint, model_dir / f'checkpoint_epoch_{epoch}.pt')

    # Save final model
    final_checkpoint = {
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'total_params': total_params,
        'model_size_mb': model_size_mb,
        'best_val_loss': best_val_loss,
        'history': history,
    }
    torch.save(final_checkpoint, model_dir / 'final_model.pt')

    # Save history
    with open(model_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    logger.info("")
    logger.info(f"Training complete for {model_name.upper().replace('_', '-')}!")
    logger.info(f"  Best val loss: {best_val_loss:.4f}")
    logger.info(f"  Models saved to: {model_dir}")
    logger.info("="*70)
    logger.info("")

    return model, history


def main():
    parser = argparse.ArgumentParser(
        description="Train compact student models via progressive distillation"
    )

    # Model selection
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['tiny', 'ultra_tiny', 'both'],
        default=['both'],
        help='Which models to train'
    )

    # Paths
    parser.add_argument(
        '--teacher',
        default='checkpoints/best_model.pt',
        help='Path to current student model (to use as teacher)'
    )
    parser.add_argument(
        '--data',
        default='data/merged_dataset_4883/merged_dataset.h5',
        help='Path to training dataset'
    )
    parser.add_argument(
        '--output-dir',
        default='checkpoints/compact_models',
        help='Output directory for trained models'
    )
    parser.add_argument(
        '--log-dir',
        default='logs',
        help='Directory for log files'
    )

    # Training config
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--energy-weight', type=float, default=1.0, help='Energy loss weight')
    parser.add_argument('--force-weight', type=float, default=15.0, help='Force loss weight')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--train-ratio', type=float, default=0.9, help='Train/val split ratio')

    # System config
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Convert paths
    teacher_path = Path(args.teacher) if not Path(args.teacher).is_absolute() else Path(args.teacher)
    if not teacher_path.exists():
        teacher_path = REPO_ROOT / args.teacher

    data_path = Path(args.data) if not Path(args.data).is_absolute() else Path(args.data)
    if not data_path.exists():
        data_path = REPO_ROOT / args.data

    output_dir = Path(args.output_dir) if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / args.output_dir

    log_dir = Path(args.log_dir) if not Path(args.log_dir).is_absolute() else Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = REPO_ROOT / args.log_dir

    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    else:
        device = args.device

    # Setup logging
    logger = setup_logging(log_dir, 'compact_models')

    logger.info("="*70)
    logger.info("PROGRESSIVE DISTILLATION: COMPACT MODEL TRAINING")
    logger.info("="*70)
    logger.info(f"Teacher model: {teacher_path}")
    logger.info(f"Training data: {data_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Random seed: {args.seed}")
    logger.info("")

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed(args.seed)

    # Check paths exist
    if not teacher_path.exists():
        logger.error(f"Teacher model not found: {teacher_path}")
        sys.exit(1)

    if not data_path.exists():
        logger.error(f"Dataset not found: {data_path}")
        sys.exit(1)

    # Load teacher model
    teacher, teacher_config = load_teacher_model(teacher_path, device)

    # Determine which models to train
    if 'both' in args.models:
        models_to_train = ['tiny', 'ultra_tiny']
    else:
        models_to_train = args.models

    logger.info(f"Models to train: {', '.join(models_to_train)}")
    logger.info("")

    # Train each model
    results = {}
    for model_name in models_to_train:
        config = CONFIGS[model_name]

        model, history = train_model(
            model_name=model_name,
            config=config,
            teacher=teacher,
            data_path=data_path,
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            energy_weight=args.energy_weight,
            force_weight=args.force_weight,
            grad_clip=args.grad_clip,
            train_ratio=args.train_ratio,
            num_workers=args.num_workers,
            device=device,
            seed=args.seed,
        )

        results[model_name] = {
            'config': config,
            'history': history,
        }

    # Print summary
    logger.info("="*70)
    logger.info("TRAINING SUMMARY")
    logger.info("="*70)
    logger.info("")

    for model_name in models_to_train:
        model_dir = output_dir / model_name
        ckpt = torch.load(model_dir / 'best_model.pt', weights_only=False)

        logger.info(f"{model_name.upper().replace('_', '-')}:")
        logger.info(f"  Parameters: {ckpt['total_params']:,}")
        logger.info(f"  Size: {ckpt['model_size_mb']:.3f} MB")
        logger.info(f"  Best val loss: {ckpt['best_val_loss']:.4f}")
        logger.info(f"  Checkpoint: {model_dir / 'best_model.pt'}")
        logger.info("")

    logger.info("="*70)
    logger.info("ALL MODELS TRAINED SUCCESSFULLY!")
    logger.info("="*70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
