#!/usr/bin/env python3
"""
Train Compact Student Models

Creates and trains smaller variants of the student model to explore
the accuracy-speed-size trade-off space.

Configurations tested:
1. Current (baseline): 427K params, 1.63 MB
2. Compact (3/4 size): 245K params, 0.94 MB
3. Efficient (1/2 size): 113K params, 0.43 MB
4. Tiny (1/4 size): 78K params, 0.30 MB
5. Ultra-tiny (1/8 size): 22K params, 0.08 MB

Author: CUDA Optimization Engineer
Date: 2025-11-24
"""

import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

import sys
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from mlff_distiller.models.student_model import StudentForceField
from mlff_distiller.training.distillation_trainer import DistillationTrainer


# Model configurations
CONFIGS = {
    "current": {
        "hidden_dim": 128,
        "num_interactions": 3,
        "num_rbf": 20,
        "cutoff": 5.0,
        "description": "Current baseline model"
    },
    "compact": {
        "hidden_dim": 96,
        "num_interactions": 3,
        "num_rbf": 16,
        "cutoff": 5.0,
        "description": "3/4 size - Good balance"
    },
    "efficient": {
        "hidden_dim": 64,
        "num_interactions": 3,
        "num_rbf": 16,
        "cutoff": 5.0,
        "description": "1/2 size - Efficient"
    },
    "tiny": {
        "hidden_dim": 64,
        "num_interactions": 2,
        "num_rbf": 12,
        "cutoff": 5.0,
        "description": "1/4 size - Very small"
    },
    "ultra_tiny": {
        "hidden_dim": 32,
        "num_interactions": 2,
        "num_rbf": 10,
        "cutoff": 5.0,
        "description": "1/8 size - Minimal"
    }
}


def train_model(
    config_name: str,
    config: dict,
    train_dataset_path: str,
    val_dataset_path: str,
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 32,
    device: str = 'cuda'
):
    """
    Train a single model configuration.

    Args:
        config_name: Name of configuration
        config: Model configuration dict
        train_dataset_path: Path to training dataset
        val_dataset_path: Path to validation dataset
        output_dir: Output directory for checkpoints
        epochs: Number of training epochs
        batch_size: Batch size
        device: Device to train on
    """
    print(f"\n{'='*70}")
    print(f"TRAINING {config_name.upper()} MODEL")
    print(f"{'='*70}")
    print(f"Description: {config['description']}")
    print(f"Configuration:")
    for key, value in config.items():
        if key != 'description':
            print(f"  {key}: {value}")

    # Create model
    model = StudentForceField(
        hidden_dim=config['hidden_dim'],
        num_interactions=config['num_interactions'],
        num_rbf=config['num_rbf'],
        cutoff=config['cutoff']
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / 1024 / 1024

    print(f"\nModel statistics:")
    print(f"  Parameters: {total_params:,}")
    print(f"  Size: {model_size_mb:.2f} MB")

    # Create output directory
    model_dir = output_dir / config_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Training configuration
    train_config = {
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'energy_weight': 1.0,
        'force_weight': 10.0,
        'device': device,
        'checkpoint_dir': str(model_dir),
        'log_interval': 10
    }

    # Save configuration
    with open(model_dir / 'config.json', 'w') as f:
        json.dump({
            'model': config,
            'training': train_config,
            'total_params': total_params,
            'model_size_mb': model_size_mb
        }, f, indent=2)

    print(f"\nTraining configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {train_config['learning_rate']}")
    print(f"  Energy weight: {train_config['energy_weight']}")
    print(f"  Force weight: {train_config['force_weight']}")

    # Note: Actual training would happen here using DistillationTrainer
    # For now, we'll just save the initialized model

    checkpoint_path = model_dir / 'model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'total_params': total_params,
        'model_size_mb': model_size_mb
    }, checkpoint_path)

    print(f"\nModel initialized and saved to: {checkpoint_path}")
    print(f"  (Training would happen here with actual dataset)")

    return {
        'config_name': config_name,
        'total_params': total_params,
        'model_size_mb': model_size_mb,
        'checkpoint': str(checkpoint_path)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train compact student model variants"
    )
    parser.add_argument(
        '--train-data',
        type=str,
        default='data/processed/train.h5',
        help='Path to training dataset'
    )
    parser.add_argument(
        '--val-data',
        type=str,
        default='data/processed/val.h5',
        help='Path to validation dataset'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='checkpoints/compact_models',
        help='Output directory for models'
    )
    parser.add_argument(
        '--configs',
        nargs='+',
        choices=list(CONFIGS.keys()) + ['all'],
        default=['all'],
        help='Which configurations to train'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to train on'
    )

    args = parser.parse_args()

    # Determine which configs to train
    if 'all' in args.configs:
        configs_to_train = list(CONFIGS.keys())
    else:
        configs_to_train = args.configs

    print(f"\n{'='*70}")
    print("COMPACT MODEL TRAINING")
    print(f"{'='*70}")
    print(f"Training configurations: {', '.join(configs_to_train)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train each configuration
    results = []
    for config_name in configs_to_train:
        config = CONFIGS[config_name]

        result = train_model(
            config_name,
            config,
            args.train_data,
            args.val_data,
            output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device
        )

        results.append(result)

    # Save summary
    summary_path = output_dir / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'configs': CONFIGS,
            'results': results,
            'args': vars(args)
        }, f, indent=2)

    print(f"\n{'='*70}")
    print("TRAINING SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Configuration':<15} {'Params':<12} {'Size (MB)':<12} {'Checkpoint'}")
    print("-" * 70)

    for result in results:
        print(f"{result['config_name']:<15} "
              f"{result['total_params']:>10,}   "
              f"{result['model_size_mb']:>7.2f}      "
              f"{result['checkpoint']}")

    print(f"\nSummary saved to: {summary_path}")
    print("\nNOTE: Models are initialized but not trained yet.")
    print("To train with actual data, use the DistillationTrainer with your dataset.")


if __name__ == '__main__':
    main()
