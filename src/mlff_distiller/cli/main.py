#!/usr/bin/env python3
"""
CLI entry points for MLFF Distiller.

This module provides the main entry point functions for CLI commands:
    - mlff-train: Train a student model
    - mlff-validate: Validate a trained model
    - mlff-benchmark: Benchmark model performance

Each function can be run directly or via the CLI entry points defined
in pyproject.toml.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import torch


def setup_logging(verbose: bool = False, log_file: Optional[Path] = None):
    """Configure logging for CLI commands."""
    level = logging.DEBUG if verbose else logging.INFO

    handlers = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def train():
    """
    Train a student force field model via knowledge distillation.

    This command trains a compact PaiNN-based student model to replicate
    the predictions of a larger teacher model (e.g., Orb-v2).

    Usage:
        mlff-train --config configs/train.yaml
        mlff-train --dataset data/training.h5 --epochs 100
        mlff-train --resume checkpoints/checkpoint_epoch_50.pt
    """
    parser = argparse.ArgumentParser(
        prog='mlff-train',
        description='Train a student force field model via knowledge distillation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with default configuration
    mlff-train --dataset data/training.h5

    # Train with custom config file
    mlff-train --config configs/train.yaml

    # Resume from checkpoint
    mlff-train --resume checkpoints/checkpoint_epoch_50.pt

    # Custom training parameters
    mlff-train --dataset data/training.h5 --epochs 200 --batch-size 64 --lr 1e-4
        """
    )

    # Data arguments
    parser.add_argument('--dataset', type=Path,
                        help='Path to HDF5 training dataset')
    parser.add_argument('--config', type=Path,
                        help='Path to YAML configuration file')

    # Model arguments
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension for student model (default: 128)')
    parser.add_argument('--num-interactions', type=int, default=3,
                        help='Number of message passing layers (default: 3)')
    parser.add_argument('--cutoff', type=float, default=5.0,
                        help='Cutoff distance in Angstroms (default: 5.0)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to train on (default: cuda)')

    # Checkpoint arguments
    parser.add_argument('--resume', type=Path,
                        help='Resume from checkpoint')
    parser.add_argument('--checkpoint-dir', type=Path, default=Path('checkpoints'),
                        help='Directory to save checkpoints (default: checkpoints)')

    # Logging arguments
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')

    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger('mlff-train')

    # Check for required arguments
    if not args.dataset and not args.config and not args.resume:
        parser.error("One of --dataset, --config, or --resume is required")

    logger.info("MLFF Distiller Training")
    logger.info("=" * 50)

    # Import training components
    from mlff_distiller.models import StudentForceField
    from mlff_distiller.training import Trainer, TrainingConfig

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    logger.info(f"Device: {args.device}")

    # Load or create configuration
    if args.config:
        from mlff_distiller.training import load_config
        config = load_config(args.config)
        logger.info(f"Loaded config from: {args.config}")
    else:
        config = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=args.device,
        )

    # Create model
    model = StudentForceField(
        hidden_dim=args.hidden_dim,
        num_interactions=args.num_interactions,
        cutoff=args.cutoff,
    )
    logger.info(f"Created student model: {model.num_parameters():,} parameters")

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from: {args.resume}")
        model = StudentForceField.load(args.resume, device=args.device)

    # Setup data loaders
    if args.dataset:
        from mlff_distiller.data import create_dataloaders
        train_loader, val_loader = create_dataloaders(
            args.dataset,
            batch_size=args.batch_size,
        )
        logger.info(f"Loaded dataset from: {args.dataset}")
    else:
        logger.error("No dataset specified. Use --dataset to provide training data.")
        sys.exit(1)

    # Create trainer and train
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=args.checkpoint_dir,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Training complete!")
    return 0


def validate():
    """
    Validate a trained student model with MD simulations.

    This command runs NVE molecular dynamics simulations to validate
    energy conservation and force accuracy of a trained model.

    Usage:
        mlff-validate --checkpoint checkpoints/best_model.pt
        mlff-validate --checkpoint model.pt --molecule H2O --steps 1000
    """
    parser = argparse.ArgumentParser(
        prog='mlff-validate',
        description='Validate a trained student model with MD simulations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Validate with default water molecule
    mlff-validate --checkpoint checkpoints/best_model.pt

    # Validate with custom molecule
    mlff-validate --checkpoint model.pt --molecule CH4

    # Run longer simulation
    mlff-validate --checkpoint model.pt --steps 5000 --temperature 300
        """
    )

    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--molecule', type=str, default='H2O',
                        help='Molecule to simulate (default: H2O)')
    parser.add_argument('--steps', type=int, default=1000,
                        help='Number of MD steps (default: 1000)')
    parser.add_argument('--temperature', type=float, default=300.0,
                        help='Temperature in Kelvin (default: 300)')
    parser.add_argument('--timestep', type=float, default=0.5,
                        help='Timestep in fs (default: 0.5)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device (default: cuda)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--output', '-o', type=Path,
                        help='Output file for results (JSON)')

    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger('mlff-validate')

    logger.info("MLFF Distiller Validation")
    logger.info("=" * 50)

    # Check checkpoint exists
    if not args.checkpoint.exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    # Import components
    from ase.build import molecule
    from mlff_distiller.inference import StudentForceFieldCalculator
    from mlff_distiller.testing import NVEMDHarness, assess_energy_conservation

    # Create calculator
    logger.info(f"Loading model from: {args.checkpoint}")
    calc = StudentForceFieldCalculator(
        args.checkpoint,
        device=args.device
    )

    # Create test molecule
    logger.info(f"Creating test molecule: {args.molecule}")
    atoms = molecule(args.molecule)
    atoms.calc = calc

    # Run NVE MD
    logger.info(f"Running NVE MD: {args.steps} steps, T={args.temperature}K, dt={args.timestep}fs")
    harness = NVEMDHarness(
        atoms=atoms,
        calculator=calc,
        temperature=args.temperature,
        timestep=args.timestep
    )

    results = harness.run_simulation(steps=args.steps)

    # Analyze energy conservation
    assessment = assess_energy_conservation(
        results['trajectory_data'],
        tolerance_pct=1.0,
        verbose=True
    )

    # Print results
    logger.info("")
    logger.info("Validation Results")
    logger.info("-" * 30)
    logger.info(f"Energy Drift: {assessment['drift_pct']:.4f}%")
    logger.info(f"Conservation Ratio: {assessment['conservation_ratio']:.6f}")
    logger.info(f"Status: {'PASSED' if assessment['passed'] else 'FAILED'}")

    # Save results if requested
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump({
                'molecule': args.molecule,
                'steps': args.steps,
                'temperature': args.temperature,
                'timestep': args.timestep,
                'drift_pct': assessment['drift_pct'],
                'conservation_ratio': assessment['conservation_ratio'],
                'passed': assessment['passed']
            }, f, indent=2)
        logger.info(f"Results saved to: {args.output}")

    return 0 if assessment['passed'] else 1


def benchmark():
    """
    Benchmark model inference performance.

    This command measures inference speed on molecules of various sizes
    and reports timing statistics.

    Usage:
        mlff-benchmark --checkpoint checkpoints/best_model.pt
        mlff-benchmark --checkpoint model.pt --device cuda --warmup 10
    """
    parser = argparse.ArgumentParser(
        prog='mlff-benchmark',
        description='Benchmark model inference performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run benchmark with default settings
    mlff-benchmark --checkpoint checkpoints/best_model.pt

    # Benchmark on CPU
    mlff-benchmark --checkpoint model.pt --device cpu

    # More warmup iterations for stable timing
    mlff-benchmark --checkpoint model.pt --warmup 50 --iterations 200
        """
    )

    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device (default: cuda)')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Warmup iterations (default: 10)')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Benchmark iterations (default: 100)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--output', '-o', type=Path,
                        help='Output file for results (JSON)')

    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger('mlff-benchmark')

    logger.info("MLFF Distiller Benchmark")
    logger.info("=" * 50)

    # Check checkpoint exists
    if not args.checkpoint.exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    # Import components
    import time
    import numpy as np
    from ase.build import molecule
    from mlff_distiller.inference import StudentForceFieldCalculator

    # Create calculator
    logger.info(f"Loading model from: {args.checkpoint}")
    calc = StudentForceFieldCalculator(
        args.checkpoint,
        device=args.device,
        enable_timing=True
    )

    # Test molecules
    test_molecules = ['H2O', 'CH4', 'C2H6', 'C6H6']
    results = {}

    for mol_name in test_molecules:
        logger.info(f"\nBenchmarking {mol_name}...")
        atoms = molecule(mol_name)
        atoms.calc = calc

        # Warmup
        for _ in range(args.warmup):
            atoms.get_potential_energy()
            atoms.get_forces()

        # Benchmark
        times = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            atoms.get_potential_energy()
            atoms.get_forces()
            if args.device == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        times = np.array(times) * 1000  # Convert to ms
        results[mol_name] = {
            'n_atoms': len(atoms),
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
        }

        logger.info(f"  {mol_name} ({len(atoms)} atoms): "
                   f"{np.mean(times):.2f} +/- {np.std(times):.2f} ms")

    # Summary
    logger.info("")
    logger.info("Benchmark Summary")
    logger.info("-" * 50)
    logger.info(f"{'Molecule':<10} {'Atoms':<8} {'Mean (ms)':<12} {'Std (ms)':<10}")
    logger.info("-" * 50)
    for mol_name, data in results.items():
        logger.info(f"{mol_name:<10} {data['n_atoms']:<8} "
                   f"{data['mean_ms']:<12.3f} {data['std_ms']:<10.3f}")

    # Save results if requested
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump({
                'device': args.device,
                'warmup': args.warmup,
                'iterations': args.iterations,
                'results': results
            }, f, indent=2)
        logger.info(f"\nResults saved to: {args.output}")

    return 0


if __name__ == '__main__':
    # Allow running this module directly for testing
    import sys
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        sys.argv = sys.argv[1:]  # Remove the command name
        if cmd == 'train':
            sys.exit(train())
        elif cmd == 'validate':
            sys.exit(validate())
        elif cmd == 'benchmark':
            sys.exit(benchmark())

    print("Usage: python -m mlff_distiller.cli.main [train|validate|benchmark]")
    sys.exit(1)
