"""
Example training script for MLFF distillation.

This example demonstrates how to use the training framework to train
a student model. It includes:
- Creating a simple dataset
- Configuring the trainer
- Running training with monitoring
- Saving and loading checkpoints

For production use, replace the dummy model and dataset with real ones.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training import (
    ForceFieldLoss,
    Trainer,
    TrainingConfig,
    create_default_config,
    load_config,
    save_config,
)
from training.config import CheckpointConfig, LoggingConfig, LossConfig, OptimizerConfig, SchedulerConfig


class SimpleStudentModel(nn.Module):
    """
    Simple student model for demonstration.

    In production, replace with actual distilled model architecture.
    """

    def __init__(self, n_atoms=20, hidden_dim=128):
        super().__init__()
        self.n_atoms = n_atoms

        # Simple MLP for demonstration
        input_dim = n_atoms * 3  # Flattened positions

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.force_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, batch):
        """
        Forward pass.

        Args:
            batch: Dictionary with 'positions' key

        Returns:
            Dictionary with 'energy' and 'forces' predictions
        """
        positions = batch["positions"]  # (batch, n_atoms, 3)
        batch_size = positions.shape[0]

        # Flatten positions
        x = positions.view(batch_size, -1)

        # Shared features
        features = self.shared(x)

        # Predict energy and forces
        energy = self.energy_head(features).squeeze(-1)
        forces_flat = self.force_head(features)
        forces = forces_flat.view(batch_size, self.n_atoms, 3)

        return {
            "energy": energy,
            "forces": forces,
        }


def create_synthetic_dataset(n_samples=1000, n_atoms=20):
    """
    Create synthetic dataset for demonstration.

    In production, replace with real data from teacher model or DFT.

    Args:
        n_samples: Number of samples
        n_atoms: Number of atoms per structure

    Returns:
        TensorDataset with positions, energy, and forces
    """
    print(f"Creating synthetic dataset with {n_samples} samples, {n_atoms} atoms each...")

    # Random atomic positions
    positions = torch.randn(n_samples, n_atoms, 3)

    # Synthetic energy (e.g., sum of pairwise distances)
    # In reality, this would be DFT or teacher model predictions
    energy = torch.randn(n_samples)

    # Synthetic forces (gradient of energy w.r.t. positions)
    forces = torch.randn(n_samples, n_atoms, 3)

    class DictDataset(TensorDataset):
        """Dataset that returns dictionaries."""

        def __getitem__(self, idx):
            return {
                "positions": positions[idx],
                "energy": energy[idx],
                "forces": forces[idx],
            }

    return DictDataset(positions, energy, forces)


def main(args):
    """Main training function."""
    print("=" * 80)
    print("MLFF Distiller - Training Example")
    print("=" * 80)

    # Load or create configuration
    if args.config:
        print(f"\nLoading configuration from {args.config}")
        config = load_config(args.config)
    else:
        print("\nCreating default configuration")
        config = create_default_config()

        # Override with command line arguments
        if args.epochs:
            config.max_epochs = args.epochs
        if args.batch_size:
            config.batch_size = args.batch_size
        if args.lr:
            config.optimizer.learning_rate = args.lr

        # Set checkpoint and logging directories
        config.checkpoint.checkpoint_dir = Path(args.checkpoint_dir)
        config.logging.tensorboard_dir = Path(args.log_dir)
        config.logging.use_tensorboard = args.use_tensorboard
        config.logging.use_wandb = args.use_wandb

        if args.use_wandb:
            config.logging.wandb_project = args.wandb_project
            config.logging.wandb_run_name = args.wandb_run_name

    # Save configuration for reproducibility
    config_save_path = Path(args.checkpoint_dir) / "config.json"
    save_config(config, config_save_path)
    print(f"Configuration saved to {config_save_path}")

    # Print configuration summary
    print("\n" + "=" * 80)
    print("Training Configuration:")
    print("=" * 80)
    print(f"Max Epochs: {config.max_epochs}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.optimizer.learning_rate}")
    print(f"Optimizer: {config.optimizer.name}")
    print(f"LR Scheduler: {config.scheduler.name}")
    print(f"Device: {config.device}")
    print(f"Mixed Precision: {config.mixed_precision}")
    print("\nLoss Weights:")
    print(f"  Energy: {config.loss.energy_weight}")
    print(f"  Force: {config.loss.force_weight} (CRITICAL for MD)")
    print(f"  Stress: {config.loss.stress_weight}")
    print("=" * 80)

    # Create dataset
    print("\n" + "=" * 80)
    print("Data Preparation")
    print("=" * 80)

    dataset = create_synthetic_dataset(
        n_samples=args.n_samples,
        n_atoms=args.n_atoms,
    )

    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    # Create model
    print("\n" + "=" * 80)
    print("Model Creation")
    print("=" * 80)

    model = SimpleStudentModel(n_atoms=args.n_atoms, hidden_dim=args.hidden_dim)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {n_params:,} parameters")

    # Create loss function
    loss_fn = ForceFieldLoss(
        energy_weight=config.loss.energy_weight,
        force_weight=config.loss.force_weight,
        stress_weight=config.loss.stress_weight,
        energy_loss_type=config.loss.energy_loss_type,
        force_loss_type=config.loss.force_loss_type,
    )

    # Create trainer
    print("\n" + "=" * 80)
    print("Trainer Initialization")
    print("=" * 80)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        loss_fn=loss_fn,
    )

    print(f"Training on device: {trainer.device}")

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(Path(args.resume))
        print(f"Resumed from epoch {trainer.current_epoch}")

    # Train
    print("\n" + "=" * 80)
    print("Training")
    print("=" * 80)

    history = trainer.fit()

    # Print final results
    print("\n" + "=" * 80)
    print("Training Complete")
    print("=" * 80)
    print(f"Final epoch: {trainer.current_epoch}")
    print(f"Best validation loss: {trainer.best_val_loss:.6f}")

    if history["val"]:
        final_val = history["val"][-1]
        print("\nFinal Validation Metrics:")
        print(f"  Total Loss: {final_val['total']:.6f}")
        if "energy_mae" in final_val:
            print(f"  Energy MAE: {final_val['energy_mae']:.6f} eV")
        if "force_rmse" in final_val:
            print(f"  Force RMSE: {final_val['force_rmse']:.6f} eV/Å (CRITICAL)")

    print(f"\nBest model saved to: {config.checkpoint.checkpoint_dir / 'best_model.pt'}")
    print(f"TensorBoard logs: {config.logging.tensorboard_dir}")

    print("\n" + "=" * 80)
    print("To view training progress in TensorBoard:")
    print(f"  tensorboard --logdir {config.logging.tensorboard_dir}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train MLFF student model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of training samples",
    )
    parser.add_argument(
        "--n-atoms",
        type=int,
        default=20,
        help="Number of atoms per structure",
    )

    # Model arguments
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension for model",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate",
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (JSON/YAML)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="runs",
        help="Directory for TensorBoard logs",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    # Logging
    parser.add_argument(
        "--use-tensorboard",
        action="store_true",
        default=True,
        help="Enable TensorBoard logging",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        default=False,
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="mlff-distiller",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name",
    )

    args = parser.parse_args()

    # Run training
    main(args)
