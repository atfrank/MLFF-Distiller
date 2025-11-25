#!/usr/bin/env python3
"""
Train Student Force Field Model via Knowledge Distillation

This script trains a compact student model to mimic a teacher force field
(Orb-v2) using knowledge distillation on teacher-labeled molecular structures.

Usage:
    # Train with default config
    python scripts/train_student.py

    # Train with custom config
    python scripts/train_student.py --config configs/train_student.yaml

    # Override specific parameters
    python scripts/train_student.py --epochs 200 --batch-size 64 --lr 1e-4

    # Resume from checkpoint
    python scripts/train_student.py --resume checkpoints/checkpoint_epoch_50.pt

Author: ML Distillation Project
Date: 2025-11-24
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# Add src to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from mlff_distiller.models.student_model import StudentForceField
from mlff_distiller.models.distillation_wrapper import DistillationWrapper
from mlff_distiller.data.distillation_dataset import create_train_val_dataloaders
from mlff_distiller.training.trainer import Trainer
from mlff_distiller.training.config import TrainingConfig, create_default_config, load_config, save_config
from mlff_distiller.training.losses import ForceFieldLoss


def setup_logging(log_dir: Path, verbose: bool = False):
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "training.log"

    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train student force field model via knowledge distillation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument(
        "--data",
        type=str,
        default=str(REPO_ROOT / "data/merged_dataset_4883/merged_dataset.h5"),
        help="Path to HDF5 training dataset"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Fraction of data for training (rest for validation)"
    )

    # Model arguments
    parser.add_argument(
        "--num-interactions",
        type=int,
        default=3,
        help="Number of interaction blocks in student model"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension size"
    )
    parser.add_argument(
        "--num-rbf",
        type=int,
        default=20,
        help="Number of radial basis functions"
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=5.0,
        help="Cutoff radius in Angstrom"
    )

    # Training arguments
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training config file (YAML/JSON)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size (overrides config)"
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        type=float,
        default=None,
        dest="learning_rate",
        help="Learning rate (overrides config)"
    )
    parser.add_argument(
        "--energy-weight",
        type=float,
        default=None,
        help="Weight for energy loss (overrides config)"
    )
    parser.add_argument(
        "--force-weight",
        type=float,
        default=None,
        help="Weight for force loss (overrides config)"
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=None,
        help="Gradient clipping value (overrides config)"
    )

    # Checkpoint arguments
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(REPO_ROOT / "checkpoints"),
        help="Directory for saving checkpoints"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="Save checkpoint every N epochs (overrides config)"
    )

    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cuda", "cpu", "auto"],
        help="Device to train on"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader worker processes"
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Use mixed precision training"
    )

    # Logging arguments
    parser.add_argument(
        "--log-dir",
        type=str,
        default=str(REPO_ROOT / "logs"),
        help="Directory for log files"
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=str,
        default=str(REPO_ROOT / "runs"),
        help="TensorBoard log directory"
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="mlff-distillation",
        help="W&B project name"
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging (DEBUG level)"
    )

    # Misc arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic mode (may reduce performance)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Setup everything but don't train (for testing)"
    )

    return parser.parse_args()


def create_model(args) -> nn.Module:
    """Create student model from arguments."""
    student = StudentForceField(
        num_interactions=args.num_interactions,
        hidden_dim=args.hidden_dim,
        num_rbf=args.num_rbf,
        cutoff=args.cutoff,
        max_z=100,  # Support up to Fermium
    )

    # Wrap for training interface
    model = DistillationWrapper(student)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Created student model with {n_params:,} parameters")
    logging.info(f"  Interactions: {args.num_interactions}")
    logging.info(f"  Hidden dim: {args.hidden_dim}")
    logging.info(f"  RBF functions: {args.num_rbf}")
    logging.info(f"  Cutoff: {args.cutoff} Å")

    return model


def create_config(args) -> TrainingConfig:
    """Create training configuration from arguments."""
    # Load config from file if provided
    if args.config is not None:
        config = load_config(args.config)
        logging.info(f"Loaded config from: {args.config}")
    else:
        config = create_default_config()
        logging.info("Using default training configuration")

    # Override with command line arguments
    if args.epochs is not None:
        config.max_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.optimizer.learning_rate = args.learning_rate
    if args.energy_weight is not None:
        config.loss.energy_weight = args.energy_weight
    if args.force_weight is not None:
        config.loss.force_weight = args.force_weight
    if args.grad_clip is not None:
        config.grad_clip = args.grad_clip
    if args.save_interval is not None:
        config.checkpoint.save_interval = args.save_interval
    if args.mixed_precision:
        config.mixed_precision = True

    # Update directories
    config.checkpoint.checkpoint_dir = Path(args.checkpoint_dir)
    config.logging.tensorboard_dir = Path(args.tensorboard_dir)

    # Logging settings
    config.logging.use_wandb = args.use_wandb
    if args.use_wandb:
        config.logging.wandb_project = args.wandb_project
        config.logging.wandb_run_name = args.wandb_run_name

    # Device and workers
    config.device = args.device
    config.num_workers = args.num_workers

    # Reproducibility
    config.seed = args.seed
    config.deterministic = args.deterministic

    return config


def main():
    """Main training function."""
    args = parse_args()

    # Setup logging
    logger = setup_logging(Path(args.log_dir), verbose=args.verbose)
    logger.info("="*60)
    logger.info("ML FORCE FIELD DISTILLATION - STUDENT TRAINING")
    logger.info("="*60)

    # Check data file exists
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)
    logger.info(f"Data file: {data_path}")

    # Create configuration
    config = create_config(args)
    logger.info(f"\nTraining Configuration:")
    logger.info(f"  Max epochs: {config.max_epochs}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Learning rate: {config.optimizer.learning_rate}")
    logger.info(f"  Energy weight: {config.loss.energy_weight}")
    logger.info(f"  Force weight: {config.loss.force_weight}")
    logger.info(f"  Gradient clip: {config.grad_clip}")
    logger.info(f"  Device: {config.device}")
    logger.info(f"  Mixed precision: {config.mixed_precision}")
    logger.info(f"  Random seed: {config.seed}")

    # Save config
    config_save_path = config.checkpoint.checkpoint_dir / "training_config.json"
    save_config(config, config_save_path)
    logger.info(f"Saved config to: {config_save_path}")

    # Create data loaders
    logger.info(f"\nCreating data loaders...")
    train_loader, val_loader = create_train_val_dataloaders(
        hdf5_path=data_path,
        batch_size=config.batch_size,
        val_batch_size=config.val_batch_size,
        train_ratio=args.train_ratio,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        random_seed=config.seed,
    )
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")

    # Create model
    logger.info(f"\nCreating student model...")
    model = create_model(args)

    # Create loss function
    loss_fn = ForceFieldLoss(
        energy_weight=config.loss.energy_weight,
        force_weight=config.loss.force_weight,
        stress_weight=config.loss.stress_weight,
        energy_loss_type=config.loss.energy_loss_type,
        force_loss_type=config.loss.force_loss_type,
        stress_loss_type=config.loss.stress_loss_type,
        huber_delta=config.loss.huber_delta,
    )
    logger.info(f"Loss function: ForceFieldLoss")
    logger.info(f"  Energy weight: {config.loss.energy_weight}")
    logger.info(f"  Force weight: {config.loss.force_weight}")

    # Create trainer
    logger.info(f"\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        loss_fn=loss_fn,
    )

    # Resume from checkpoint if specified
    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            logger.error(f"Checkpoint not found: {resume_path}")
            sys.exit(1)
        logger.info(f"Resuming from checkpoint: {resume_path}")
        trainer.load_checkpoint(resume_path, load_optimizer=True)
        logger.info(f"  Resuming from epoch {trainer.current_epoch}")
        logger.info(f"  Best val loss: {trainer.best_val_loss:.4f}")

    # Dry run - exit before training
    if args.dry_run:
        logger.info("\n" + "="*60)
        logger.info("DRY RUN - Setup complete, exiting before training")
        logger.info("="*60)
        return 0

    # Train
    logger.info("\n" + "="*60)
    logger.info("STARTING TRAINING")
    logger.info("="*60)

    try:
        history = trainer.fit()
        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETE")
        logger.info("="*60)
        logger.info(f"Final epoch: {trainer.current_epoch}")
        logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
        logger.info(f"Checkpoints saved to: {config.checkpoint.checkpoint_dir}")

        return 0

    except KeyboardInterrupt:
        logger.info("\n" + "="*60)
        logger.info("TRAINING INTERRUPTED")
        logger.info("="*60)
        logger.info(f"Stopped at epoch {trainer.current_epoch}")
        logger.info("Saving checkpoint...")
        trainer.save_checkpoint(
            filepath=config.checkpoint.checkpoint_dir / "checkpoint_interrupted.pt"
        )
        logger.info("Checkpoint saved. You can resume with --resume")
        return 1

    except Exception as e:
        logger.exception("Training failed with exception:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
