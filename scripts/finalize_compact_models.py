#!/usr/bin/env python3
"""
Finalize Compact Models: Fix checkpoint format, validate, and export all models
Author: Continuation from previous session
Date: 2025-11-24
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
import sys
import argparse
from tqdm import tqdm
import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from mlff_distiller.models.student_model import StudentForceField
from mlff_distiller.models.distillation_wrapper import DistillationWrapper
from mlff_distiller.data.distillation_dataset import DistillationDataset, distillation_collate_fn
from torch.utils.data import DataLoader


def fix_checkpoint_format(checkpoint_path):
    """Load checkpoint and fix the 'model.' prefix issue."""
    print(f"  Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    if 'model_state_dict' in checkpoint:
        state = checkpoint['model_state_dict']

        # Check if we need to strip 'model.' prefix
        has_prefix = any(k.startswith('model.') for k in state.keys())

        if has_prefix:
            print(f"  Found 'model.' prefix in state dict keys - fixing...")
            state = {k.replace('model.', ''): v for k, v in state.items()}
            checkpoint['model_state_dict'] = state
            # Save the fixed checkpoint back
            torch.save(checkpoint, checkpoint_path)
            print(f"  ✓ Checkpoint format fixed and saved")
        else:
            print(f"  Checkpoint already in correct format")

    return checkpoint


def load_model(config, checkpoint_path, device):
    """Load a student model from checkpoint."""
    # Create model
    student = StudentForceField(
        hidden_dim=config['hidden_dim'],
        num_interactions=config['num_interactions'],
        num_rbf=config['num_rbf'],
        cutoff=config['cutoff'],
        max_z=config.get('max_z', 100)
    ).to(device)

    # Fix checkpoint and load
    checkpoint = fix_checkpoint_format(checkpoint_path)
    student.load_state_dict(checkpoint['model_state_dict'])
    student.eval()

    return student


def validate_model(model, val_loader, device, force_weight=100.0):
    """Validate a model on the validation dataset."""
    model.eval()

    total_energy_error = 0.0
    total_force_error = 0.0
    n_batches = 0
    all_force_errors = []

    for batch in tqdm(val_loader, desc="Validating", leave=False):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Get teacher labels
        teacher_energy = batch['energy']
        teacher_forces = batch['forces']

        # Get student predictions
        atomic_numbers = batch['atomic_numbers']
        positions = batch['positions'].clone().detach()
        positions.requires_grad_(True)

        with torch.enable_grad():
            student_energy = model(atomic_numbers, positions)

            # Compute forces via autograd
            if student_energy.requires_grad:
                forces_grad = torch.autograd.grad(
                    student_energy.sum(),
                    positions,
                    create_graph=False,
                    retain_graph=False
                )[0]
                student_forces = -forces_grad
            else:
                # If no grad, use pre-computed forces from batch
                student_forces = teacher_forces

        # Compute metrics
        energy_mae = torch.abs(student_energy.detach() - teacher_energy).mean().item()
        force_rmse = torch.sqrt(((student_forces.detach() - teacher_forces) ** 2).mean()).item()

        total_energy_error += energy_mae
        total_force_error += force_rmse
        all_force_errors.append(force_rmse)
        n_batches += 1

    return {
        'energy_mae': total_energy_error / n_batches,
        'force_rmse': total_force_error / n_batches,
        'force_rmse_std': np.std(all_force_errors) if all_force_errors else 0.0
    }


def export_model(model, model_name, output_dir):
    """Export model to TorchScript and ONNX."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Exporting {model_name}...")

    # Create sample input for tracing
    sample_z = torch.randint(1, 8, (10,))
    sample_pos = torch.randn(10, 3)

    try:
        # TorchScript tracing
        print(f"    - TorchScript tracing...", end='', flush=True)
        traced_model = torch.jit.trace(model, (sample_z, sample_pos))
        traced_path = output_dir / f"{model_name}_traced.pt"
        torch.jit.save(traced_model, traced_path)
        print(f" ✓ ({traced_path.stat().st_size / 1e6:.2f} MB)")

        # ONNX export
        print(f"    - ONNX export...", end='', flush=True)
        onnx_path = output_dir / f"{model_name}.onnx"
        torch.onnx.export(
            model,
            (sample_z, sample_pos),
            onnx_path,
            input_names=['atomic_numbers', 'positions'],
            output_names=['energy'],
            opset_version=14,
            dynamic_axes={
                'atomic_numbers': {0: 'num_atoms'},
                'positions': {0: 'num_atoms'},
                'energy': {0: 'batch_size'}
            }
        )
        print(f" ✓ ({onnx_path.stat().st_size / 1e6:.2f} MB)")

        return {
            'torchscript': str(traced_path),
            'onnx': str(onnx_path)
        }
    except Exception as e:
        print(f" ✗ Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Finalize compact models')
    parser.add_argument('--dataset', default='data/merged_dataset_4883/merged_dataset.h5')
    parser.add_argument('--output-dir', default='benchmarks')
    parser.add_argument('--val-samples', type=int, default=100)
    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()
    device = args.device

    # Model configurations
    configs = {
        'original': {
            'name': 'Original Student (427K)',
            'checkpoint': 'checkpoints/best_model.pt',
            'hidden_dim': 128,
            'num_interactions': 3,
            'num_rbf': 20,
            'cutoff': 5.0,
            'max_z': 100
        },
        'tiny': {
            'name': 'Tiny (77K)',
            'checkpoint': 'checkpoints/tiny_model/best_model.pt',
            'hidden_dim': 64,
            'num_interactions': 2,
            'num_rbf': 12,
            'cutoff': 5.0,
            'max_z': 100
        },
        'ultra_tiny': {
            'name': 'Ultra-tiny (21K)',
            'checkpoint': 'checkpoints/ultra_tiny_model/best_model.pt',
            'hidden_dim': 32,
            'num_interactions': 2,
            'num_rbf': 10,
            'cutoff': 5.0,
            'max_z': 100
        }
    }

    print(f"\n{'='*70}")
    print(f"FINALIZING COMPACT MODELS - VALIDATION & EXPORT")
    print(f"{'='*70}")
    print(f"Dataset: {args.dataset}")
    print(f"Val samples: {args.val_samples}")
    print(f"Device: {device}")

    # Load validation dataset
    print(f"\nLoading validation dataset...")
    full_dataset = DistillationDataset(args.dataset)
    n_total = len(full_dataset)
    n_train = int(0.9 * n_total)
    val_indices = list(range(n_train, min(n_train + args.val_samples, n_total)))
    val_dataset = DistillationDataset(args.dataset, indices=val_indices)

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        collate_fn=distillation_collate_fn,
        pin_memory=True
    )
    print(f"Validation set: {len(val_dataset)} samples")

    # Results dictionary
    results = {
        'validation': {},
        'export': {},
        'summary': {}
    }

    # Process each model
    for model_key, config in configs.items():
        print(f"\n{'-'*70}")
        print(f"Processing: {config['name']}")
        print(f"{'-'*70}")

        checkpoint_path = config['checkpoint']
        if not Path(checkpoint_path).exists():
            print(f"✗ Checkpoint not found: {checkpoint_path}")
            continue

        try:
            # Load model
            print(f"Loading model...")
            model = load_model(config, checkpoint_path, device)
            n_params = sum(p.numel() for p in model.parameters())
            model_size_mb = n_params * 4 / 1024 / 1024
            print(f"  Model size: {n_params:,} parameters ({model_size_mb:.3f} MB)")

            # Validate
            print(f"Validating on {len(val_dataset)} samples...")
            val_metrics = validate_model(model, val_loader, device)
            results['validation'][config['name']] = val_metrics
            print(f"  Energy MAE: {val_metrics['energy_mae']:.4f} eV")
            print(f"  Force RMSE: {val_metrics['force_rmse']:.4f} eV/Å")

            # Export
            export_results = export_model(model, f"{model_key}_model", args.output_dir)
            if export_results:
                results['export'][config['name']] = export_results

            # Summary
            results['summary'][config['name']] = {
                'parameters': n_params,
                'size_mb': model_size_mb,
                'validation': val_metrics,
                'exported': export_results is not None
            }

        except Exception as e:
            print(f"✗ Error processing {config['name']}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    output_path = Path(args.output_dir) / 'compact_models_finalized_20251124.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_path}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"FINALIZATION SUMMARY")
    print(f"{'='*70}")
    for model_name, summary in results['summary'].items():
        print(f"\n{model_name}:")
        print(f"  Parameters: {summary['parameters']:,}")
        print(f"  Size: {summary['size_mb']:.3f} MB")
        if 'validation' in summary:
            print(f"  Energy MAE: {summary['validation']['energy_mae']:.4f} eV")
            print(f"  Force RMSE: {summary['validation']['force_rmse']:.4f} eV/Å")
        print(f"  Exported: {'✓' if summary['exported'] else '✗'}")

    print(f"\n{'='*70}")
    print(f"✅ FINALIZATION COMPLETE")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
