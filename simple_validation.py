#!/usr/bin/env python3
"""
Simplified validation script - loads actual dataset from HDF5
"""

import os
import json
import torch
import numpy as np
import h5py
from datetime import datetime
from pathlib import Path

def load_and_validate_models(dataset_path, max_samples=100):
    """Load models and validate on dataset."""

    import sys
    sys.path.insert(0, '/home/aaron/ATX/software/MLFF_Distiller/src')
    from mlff_distiller.models.student_model import StudentForceField

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    with h5py.File(dataset_path, 'r') as f:
        print(f"  Available groups: {list(f.keys())}")

        # Check structure
        if 'structures' in f and 'labels' in f:
            structures = f['structures']
            labels = f['labels']

            print(f"  Structures: {list(structures.keys())}")
            print(f"  Labels: {list(labels.keys())}")

            # Load some sample data
            atomic_numbers_data = structures['atomic_numbers'][:]
            positions_data = structures['positions'][:]
            energy_data = labels['energy'][:]
            forces_data = labels['forces'][:]

            print(f"  Atomic numbers shape: {atomic_numbers_data.shape}")
            print(f"  Positions shape: {positions_data.shape}")
            print(f"  Energy shape: {energy_data.shape}")
            print(f"  Forces shape: {forces_data.shape}")
        else:
            print("  Error: Unexpected dataset structure")
            return {}

    # Load models
    models_config = {
        'Original Student (427K)': 'checkpoints/best_model.pt',
        'Tiny Model (77K)': 'checkpoints/tiny_model/best_model.pt',
        'Ultra-tiny Model (21K)': 'checkpoints/ultra_tiny_model/best_model.pt'
    }

    results = {}

    for model_name, checkpoint_path in models_config.items():
        if not os.path.exists(checkpoint_path):
            print(f"Not found: {checkpoint_path}")
            continue

        print(f"\nValidating {model_name}...")

        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

            # Create and load model
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model = StudentForceField(hidden_dim=128, max_z=100)
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model = checkpoint

            model = model.to(device)
            model.eval()

            num_params = sum(p.numel() for p in model.parameters())
            print(f"  Params: {num_params:,}")

            # Run validation on subset
            energy_errors = []
            force_errors = []
            num_validated = 0

            with torch.no_grad():
                for idx in range(min(max_samples, len(atomic_numbers_data))):
                    # Get sample data
                    atomic_numbers = torch.tensor(atomic_numbers_data[idx], dtype=torch.long, device=device)
                    positions = torch.tensor(positions_data[idx], dtype=torch.float32, device=device)
                    target_energy = torch.tensor(energy_data[idx], dtype=torch.float32, device=device)

                    # Find forces for this sample (using simple indexing)
                    start_idx = sum(len(atomic_numbers_data[j]) for j in range(idx))
                    end_idx = start_idx + len(atomic_numbers)

                    if end_idx <= len(forces_data):
                        target_forces = torch.tensor(forces_data[start_idx:end_idx], dtype=torch.float32, device=device)

                        # Predict
                        pred_energy = model(atomic_numbers, positions)

                        # Compute force via autograd
                        positions_req = positions.clone().detach().requires_grad_(True)
                        energy_req = model(atomic_numbers, positions_req)
                        pred_forces = -torch.autograd.grad(energy_req.sum(), positions_req)[0]

                        # Metrics
                        energy_error = float((pred_energy - target_energy).abs().item())
                        force_rmse = float(torch.sqrt(torch.mean((pred_forces - target_forces) ** 2)).item())

                        energy_errors.append(energy_error)
                        force_errors.append(force_rmse)
                        num_validated += 1

                    if num_validated >= max_samples:
                        break

            if len(energy_errors) > 0:
                results[model_name] = {
                    'num_parameters': int(num_params),
                    'num_validated': num_validated,
                    'energy_mae': {
                        'mean': float(np.mean(energy_errors)),
                        'std': float(np.std(energy_errors)),
                        'median': float(np.median(energy_errors))
                    },
                    'force_rmse': {
                        'mean': float(np.mean(force_errors)),
                        'std': float(np.std(force_errors)),
                        'median': float(np.median(force_errors))
                    }
                }

                print(f"  Validated {num_validated} samples")
                print(f"  Energy MAE: {results[model_name]['energy_mae']['mean']:.4f} eV")
                print(f"  Force RMSE: {results[model_name]['force_rmse']['mean']:.4f} eV/Å")

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    return results


def main():
    print("="*80)
    print("SIMPLIFIED VALIDATION")
    print("="*80 + "\n")

    dataset_path = 'data/merged_dataset_4883/merged_dataset.h5'

    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return

    results = load_and_validate_models(dataset_path, max_samples=50)

    # Save results
    output_dir = 'validation_results'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'{output_dir}/compact_models_accuracy_{timestamp}.json'

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\nSaved: {output_file}")


if __name__ == '__main__':
    main()
