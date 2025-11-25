#!/usr/bin/env python3
"""
Simplified export script for compact models
"""

import os
import json
import torch
from pathlib import Path
from datetime import datetime

# Import early to ensure torch is available
torch.set_printoptions(sci_mode=False)

def export_models_simple():
    """Export models to ONNX and TorchScript."""

    import sys
    sys.path.insert(0, '/home/aaron/ATX/software/MLFF_Distiller/src')
    from mlff_distiller.models.student_model import StudentForceField

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    models_config = {
        'original': {
            'checkpoint': 'checkpoints/best_model.pt',
            'name': 'Original Student (427K)'
        },
        'tiny': {
            'checkpoint': 'checkpoints/tiny_model/best_model.pt',
            'name': 'Tiny Model (77K)'
        },
        'ultra_tiny': {
            'checkpoint': 'checkpoints/ultra_tiny_model/best_model.pt',
            'name': 'Ultra-tiny Model (21K)'
        }
    }

    export_results = {}
    os.makedirs('models', exist_ok=True)

    for key, config in models_config.items():
        checkpoint_path = config['checkpoint']
        model_name = config['name']

        if not os.path.exists(checkpoint_path):
            print(f"Not found: {checkpoint_path}")
            continue

        print(f"Exporting {model_name}...")

        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

            # Create model
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model = StudentForceField(hidden_dim=128, max_z=100)
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model = checkpoint

            model = model.to(device)
            model.eval()

            checkpoint_size = os.path.getsize(checkpoint_path) / 1e6
            num_params = sum(p.numel() for p in model.parameters())

            export_paths = {
                'checkpoint': checkpoint_path,
                'checkpoint_size_mb': float(checkpoint_size),
                'num_parameters': int(num_params)
            }

            # Try TorchScript export
            try:
                print(f"  Exporting TorchScript...")
                atomic_numbers = torch.randint(1, 8, (16,), device=device, dtype=torch.long)
                positions = torch.randn(16, 3, device=device, dtype=torch.float32)

                ts_model = torch.jit.trace(model, (atomic_numbers, positions))
                ts_path = f'models/{key}_model_traced.pt'
                ts_model.save(ts_path)
                export_paths['torchscript_traced'] = ts_path
                print(f"    Saved: {ts_path}")
            except Exception as e:
                print(f"    Error: {e}")

            # Try ONNX export
            try:
                print(f"  Exporting ONNX...")

                atomic_numbers_dummy = torch.randint(1, 8, (16,), device=device, dtype=torch.long)
                positions_dummy = torch.randn(16, 3, device=device, dtype=torch.float32)

                onnx_path = f'models/{key}_model.onnx'

                torch.onnx.export(
                    model,
                    (atomic_numbers_dummy, positions_dummy),
                    onnx_path,
                    input_names=['atomic_numbers', 'positions'],
                    output_names=['energy'],
                    dynamic_axes={
                        'atomic_numbers': {0: 'num_atoms'},
                        'positions': {0: 'num_atoms'},
                        'energy': {}
                    },
                    opset_version=14,
                    do_constant_folding=True,
                    verbose=False
                )
                export_paths['onnx'] = onnx_path
                print(f"    Saved: {onnx_path}")
            except Exception as e:
                print(f"    Error: {e}")

            export_results[model_name] = export_paths
            print()

        except Exception as e:
            print(f"  Error: {e}\n")

    # Save summary
    output_dir = 'benchmarks'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = f'{output_dir}/export_summary_{timestamp}.json'

    with open(summary_file, 'w') as f:
        json.dump(export_results, f, indent=2)

    print(f"Saved: {summary_file}")


def main():
    print("="*80)
    print("SIMPLIFIED MODEL EXPORT")
    print("="*80 + "\n")

    export_models_simple()
    print("\nExport complete!")


if __name__ == '__main__':
    main()
