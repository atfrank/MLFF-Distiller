#!/usr/bin/env python3
"""
Task 2: Validation Suite for Compact Models
Tests accuracy on validation dataset and compares with teacher model
"""

import os
import json
import torch
import numpy as np
import h5py
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

# Add src to path
import sys
sys.path.insert(0, '/home/aaron/ATX/software/MLFF_Distiller/src')

from mlff_distiller.models.student_model import StudentForceField


def load_model(checkpoint_path: str, device: torch.device):
    """Load a model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model = StudentForceField(hidden_dim=128, max_z=100)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = checkpoint if isinstance(checkpoint, StudentForceField) else torch.jit.load(checkpoint)

    model.to(device)
    model.eval()
    return model


def load_validation_data(dataset_path: str, max_samples: int = None) -> List[Dict]:
    """Load validation data from HDF5 dataset."""
    data = []

    try:
        with h5py.File(dataset_path, 'r') as f:
            # New format: structures, labels, metadata
            if 'structures' in f and 'labels' in f:
                structures = f['structures']
                labels = f['labels']

                # Get number of structures
                num_structures = len(structures['atomic_numbers'][:])

                # Load data
                atomic_numbers_all = structures['atomic_numbers'][:]
                positions_all = structures['positions'][:]

                energy_all = labels['energy'][:]
                forces_all = labels['forces'][:]
                forces_splits = labels['forces_splits'][:]

                # Process each structure
                start_idx = 0
                for struct_idx in range(num_structures):
                    if max_samples and len(data) >= max_samples:
                        break

                    # Get forces for this structure
                    if struct_idx < len(forces_splits) - 1:
                        force_start = forces_splits[struct_idx]
                        force_end = forces_splits[struct_idx + 1]
                    else:
                        force_start = forces_splits[struct_idx] if struct_idx < len(forces_splits) else 0
                        force_end = len(forces_all)

                    positions = torch.tensor(positions_all[struct_idx], dtype=torch.float32)
                    atom_numbers = torch.tensor(atomic_numbers_all[struct_idx], dtype=torch.long)
                    forces = torch.tensor(forces_all[force_start:force_end], dtype=torch.float32)
                    energy = torch.tensor(energy_all[struct_idx], dtype=torch.float32)

                    # Skip if shapes don't match
                    if forces.shape[0] == atom_numbers.shape[0]:
                        data.append({
                            'positions': positions,
                            'atom_types': atom_numbers,
                            'forces': forces,
                            'energy': energy,
                            'mol_id': f'structure_{struct_idx}',
                            'frame_id': 'frame_0'
                        })

    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        # Return empty list if dataset not found
        return []

    return data


def compute_metrics(predictions: Dict, targets: Dict) -> Dict:
    """Compute accuracy metrics."""
    metrics = {}

    # Energy MAE
    energy_mae = torch.nn.functional.l1_loss(predictions['energy'], targets['energy'])
    metrics['energy_mae'] = float(energy_mae.item())

    # Force RMSE
    force_diff = predictions['forces'] - targets['forces']
    force_rmse = torch.sqrt(torch.mean(force_diff ** 2))
    metrics['force_rmse'] = float(force_rmse.item())

    # Per-atom force RMSE
    force_rmse_per_atom = torch.sqrt(torch.mean(force_diff ** 2, dim=1))
    metrics['force_rmse_per_atom_mean'] = float(force_rmse_per_atom.mean().item())
    metrics['force_rmse_per_atom_std'] = float(force_rmse_per_atom.std().item())

    # Angular error (angle between predicted and target force vectors)
    force_dots = torch.sum(predictions['forces'] * targets['forces'], dim=1)
    force_norms_pred = torch.norm(predictions['forces'], dim=1)
    force_norms_target = torch.norm(targets['forces'], dim=1)

    # Avoid division by zero
    valid_mask = (force_norms_pred > 1e-6) & (force_norms_target > 1e-6)
    if valid_mask.sum() > 0:
        cos_angles = force_dots[valid_mask] / (force_norms_pred[valid_mask] * force_norms_target[valid_mask])
        cos_angles = torch.clamp(cos_angles, -1, 1)
        angles = torch.acos(cos_angles) * 180 / np.pi
        metrics['angular_error_deg'] = float(angles.mean().item())
        metrics['angular_error_std'] = float(angles.std().item())
    else:
        metrics['angular_error_deg'] = 0.0
        metrics['angular_error_std'] = 0.0

    # Total MAE
    total_mae = (metrics['energy_mae'] + metrics['force_rmse']) / 2
    metrics['total_mae'] = total_mae

    return metrics


def validate_models(models_config: Dict, dataset_path: str, device: torch.device,
                   max_samples: int = 500) -> Dict:
    """Validate all models on the validation dataset."""
    print(f"\nLoading validation data from {dataset_path}")
    validation_data = load_validation_data(dataset_path, max_samples=max_samples)
    print(f"Loaded {len(validation_data)} validation samples")

    if len(validation_data) == 0:
        print("Warning: No validation data loaded!")
        return {}

    results = {}

    for model_name, config in models_config.items():
        checkpoint_path = config['checkpoint']

        if not os.path.exists(checkpoint_path):
            print(f"Warning: {checkpoint_path} not found. Skipping {model_name}")
            continue

        print(f"\n{model_name}")
        print("-" * 60)

        model = load_model(checkpoint_path, device)
        print(f"Model loaded. Validating on {len(validation_data)} samples...")

        all_metrics = []
        error_data = {
            'energy_errors': [],
            'force_rmse_errors': [],
            'angular_errors': [],
            'force_magnitude_pred': [],
            'force_magnitude_target': []
        }

        with torch.no_grad():
            for idx, sample in enumerate(validation_data):
                positions = sample['positions'].unsqueeze(0).to(device)
                atom_numbers = sample['atom_types'].unsqueeze(0).to(device).long()
                target_forces = sample['forces'].to(device)
                target_energy = sample['energy'].to(device)

                # Get predictions
                pred_energy, pred_forces = model(positions, atom_numbers)
                pred_energy = pred_energy.squeeze()
                pred_forces = pred_forces.squeeze(0)

                # Compute metrics
                sample_metrics = compute_metrics(
                    {'energy': pred_energy, 'forces': pred_forces},
                    {'energy': target_energy, 'forces': target_forces}
                )

                all_metrics.append(sample_metrics)

                # Collect error data
                error_data['energy_errors'].append(float((pred_energy - target_energy).abs().item()))
                force_diff = pred_forces - target_forces
                error_data['force_rmse_errors'].append(float(torch.sqrt(torch.mean(force_diff ** 2)).item()))

                # Angular error
                if sample_metrics.get('angular_error_deg') is not None:
                    error_data['angular_errors'].append(sample_metrics['angular_error_deg'])

                error_data['force_magnitude_pred'].append(float(torch.norm(pred_forces).item()))
                error_data['force_magnitude_target'].append(float(torch.norm(target_forces).item()))

                if (idx + 1) % 100 == 0:
                    print(f"  Processed {idx + 1}/{len(validation_data)} samples")

        # Aggregate metrics
        energy_maes = np.array([m['energy_mae'] for m in all_metrics])
        force_rmses = np.array([m['force_rmse'] for m in all_metrics])
        angular_errors = np.array([m.get('angular_error_deg', 0) for m in all_metrics])

        results[model_name] = {
            'energy_mae': {
                'mean': float(energy_maes.mean()),
                'std': float(energy_maes.std()),
                'min': float(energy_maes.min()),
                'max': float(energy_maes.max()),
                'median': float(np.median(energy_maes))
            },
            'force_rmse': {
                'mean': float(force_rmses.mean()),
                'std': float(force_rmses.std()),
                'min': float(force_rmses.min()),
                'max': float(force_rmses.max()),
                'median': float(np.median(force_rmses))
            },
            'angular_error': {
                'mean': float(angular_errors.mean()) if len(angular_errors) > 0 else 0,
                'std': float(angular_errors.std()) if len(angular_errors) > 0 else 0,
                'min': float(angular_errors.min()) if len(angular_errors) > 0 else 0,
                'max': float(angular_errors.max()) if len(angular_errors) > 0 else 0,
            },
            'num_samples': len(validation_data),
            'error_distributions': error_data,
            'checkpoint_path': checkpoint_path
        }

        # Print summary
        print(f"\nValidation Summary:")
        print(f"  Energy MAE: {results[model_name]['energy_mae']['mean']:.4f} ± {results[model_name]['energy_mae']['std']:.4f} eV")
        print(f"  Force RMSE: {results[model_name]['force_rmse']['mean']:.4f} ± {results[model_name]['force_rmse']['std']:.4f} eV/Å")
        print(f"  Angular Error: {results[model_name]['angular_error']['mean']:.2f} ± {results[model_name]['angular_error']['std']:.2f}°")

    return results


def create_validation_plots(validation_results: Dict, output_dir: str):
    """Create validation comparison plots."""
    os.makedirs(output_dir, exist_ok=True)

    models = list(validation_results.keys())

    # Plot 1: Accuracy Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Model Validation: Accuracy Metrics Comparison', fontsize=14, fontweight='bold')

    # Energy MAE
    ax = axes[0]
    energy_means = [validation_results[m]['energy_mae']['mean'] for m in models]
    energy_stds = [validation_results[m]['energy_mae']['std'] for m in models]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    ax.bar(range(len(models)), energy_means, yerr=energy_stds, color=colors, alpha=0.7, capsize=5)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylabel('MAE (eV)')
    ax.set_title('Energy MAE')
    ax.grid(True, alpha=0.3, axis='y')

    # Force RMSE
    ax = axes[1]
    force_means = [validation_results[m]['force_rmse']['mean'] for m in models]
    force_stds = [validation_results[m]['force_rmse']['std'] for m in models]
    ax.bar(range(len(models)), force_means, yerr=force_stds, color=colors, alpha=0.7, capsize=5)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylabel('RMSE (eV/Å)')
    ax.set_title('Force RMSE')
    ax.grid(True, alpha=0.3, axis='y')

    # Angular Error
    ax = axes[2]
    angle_means = [validation_results[m]['angular_error']['mean'] for m in models]
    angle_stds = [validation_results[m]['angular_error']['std'] for m in models]
    ax.bar(range(len(models)), angle_means, yerr=angle_stds, color=colors, alpha=0.7, capsize=5)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylabel('Error (degrees)')
    ax.set_title('Angular Error')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/validation_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/validation_accuracy_comparison.png")

    # Plot 2: Error distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Error Distributions on Validation Set', fontsize=14, fontweight='bold')

    for idx, model in enumerate(models):
        # Energy errors
        ax = axes[0]
        energy_errors = validation_results[model]['error_distributions']['energy_errors']
        ax.hist(energy_errors, bins=30, alpha=0.6, label=model)

        # Force RMSE errors
        ax = axes[1]
        force_errors = validation_results[model]['error_distributions']['force_rmse_errors']
        ax.hist(force_errors, bins=30, alpha=0.6, label=model)

        # Angular errors
        ax = axes[2]
        angular_errors = validation_results[model]['error_distributions']['angular_errors']
        if len(angular_errors) > 0:
            ax.hist(angular_errors, bins=30, alpha=0.6, label=model)

    axes[0].set_xlabel('Energy Error (eV)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Energy Error Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Force RMSE Error (eV/Å)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Force RMSE Error Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel('Angular Error (degrees)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Angular Error Distribution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_distributions.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/error_distributions.png")


def main():
    print("="*80)
    print("TASK 2: VALIDATION SUITE FOR COMPACT MODELS")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Model configurations
    models_config = {
        'Original Student (427K)': {
            'checkpoint': '/home/aaron/ATX/software/MLFF_Distiller/checkpoints/best_model.pt',
            'arch_params': {'hidden_dims': [128, 128, 128], 'output_dim': 1}
        },
        'Tiny Model (77K)': {
            'checkpoint': '/home/aaron/ATX/software/MLFF_Distiller/checkpoints/tiny_model/best_model.pt',
            'arch_params': {'hidden_dims': [32, 32], 'output_dim': 1}
        },
        'Ultra-tiny Model (21K)': {
            'checkpoint': '/home/aaron/ATX/software/MLFF_Distiller/checkpoints/ultra_tiny_model/best_model.pt',
            'arch_params': {'hidden_dims': [16], 'output_dim': 1}
        }
    }

    # Dataset path
    dataset_path = '/home/aaron/ATX/software/MLFF_Distiller/data/merged_dataset_4883/merged_dataset.h5'

    # Run validation
    validation_results = validate_models(models_config, dataset_path, device, max_samples=500)

    # Save results
    output_dir = '/home/aaron/ATX/software/MLFF_Distiller/validation_results'
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'{output_dir}/compact_models_accuracy_{timestamp}.json'

    # Prepare JSON-serializable results
    serializable_results = {}
    for model_name, results in validation_results.items():
        serializable_results[model_name] = {
            'energy_mae': results['energy_mae'],
            'force_rmse': results['force_rmse'],
            'angular_error': results['angular_error'],
            'num_samples': results['num_samples'],
            'checkpoint_path': results['checkpoint_path']
        }

    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\n✓ Saved validation results: {output_file}")

    # Create plots
    try:
        create_validation_plots(validation_results, output_dir)
    except Exception as e:
        print(f"Warning: Could not create validation plots: {e}")

    print("\n" + "="*80)
    print("TASK 2 COMPLETE: Validation Results Saved")
    print("="*80)


if __name__ == '__main__':
    main()
