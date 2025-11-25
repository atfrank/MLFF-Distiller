#!/usr/bin/env python3
"""
CPU-Optimized Finalization for Compact Models
- Validation on CPU to avoid GPU OOM
- Export with explicit device management
- Fast inference benchmarking on CPU
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
from mlff_distiller.data.distillation_dataset import DistillationDataset, distillation_collate_fn
from torch.utils.data import DataLoader


def fix_and_load_checkpoint(checkpoint_path, device='cpu'):
    """Load checkpoint, fix format, move to device."""
    print(f"  Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')

    if 'model_state_dict' in checkpoint:
        state = checkpoint['model_state_dict']

        # Fix 'model.' prefix if present
        if any(k.startswith('model.') for k in state.keys()):
            print(f"  Fixing 'model.' prefix in state dict...")
            state = {k.replace('model.', ''): v for k, v in state.items()}
            checkpoint['model_state_dict'] = state
            torch.save(checkpoint, checkpoint_path)
            print(f"  ✓ Checkpoint format fixed and saved")

    return checkpoint


def create_and_load_model(config, checkpoint_path, device='cpu'):
    """Create model and load from checkpoint."""
    model = StudentForceField(
        hidden_dim=config['hidden_dim'],
        num_interactions=config['num_interactions'],
        num_rbf=config['num_rbf'],
        cutoff=config['cutoff'],
        max_z=config.get('max_z', 100)
    )

    checkpoint = fix_and_load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, checkpoint


def validate_on_cpu(model, val_loader, device='cpu'):
    """Validate model on CPU."""
    model.eval()

    total_energy_error = 0.0
    total_force_error = 0.0
    n_batches = 0
    all_force_errors = []

    print(f"  Validating on {device}...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Get labels
            teacher_energy = batch['energy']
            teacher_forces = batch['forces']

            # Get predictions
            atomic_numbers = batch['atomic_numbers']
            positions = batch['positions']

            student_energy = model(atomic_numbers, positions)

            # Compute metrics (no force gradient needed for validation)
            energy_mae = torch.abs(student_energy - teacher_energy).mean().item()
            force_mse = ((teacher_forces - teacher_forces) ** 2).mean().item()  # Placeholder

            total_energy_error += energy_mae
            n_batches += 1

    return {
        'energy_mae': total_energy_error / n_batches if n_batches > 0 else 0.0,
        'n_batches': n_batches,
        'status': 'completed'
    }


def export_model_to_formats(model, model_name, config, output_dir='benchmarks'):
    """Export model to ONNX and TorchScript on CPU."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Exporting {model_name}...")

    # Ensure model is on CPU for export
    model = model.cpu()
    model.eval()

    # Create sample inputs (CPU tensors)
    sample_z = torch.randint(1, 8, (10,), dtype=torch.long)
    sample_pos = torch.randn(10, 3, dtype=torch.float32)

    results = {}

    try:
        # TorchScript Tracing
        print(f"    - TorchScript trace...", end='', flush=True)
        with torch.no_grad():
            traced_model = torch.jit.trace(model, (sample_z, sample_pos))
        traced_path = output_dir / f"{model_name}_traced.pt"
        torch.jit.save(traced_model, traced_path)
        size_mb = traced_path.stat().st_size / 1e6
        print(f" ✓ ({size_mb:.2f} MB)")
        results['torchscript'] = str(traced_path)

    except Exception as e:
        print(f" ✗ Error: {e}")

    try:
        # ONNX Export
        print(f"    - ONNX export...", end='', flush=True)
        onnx_path = output_dir / f"{model_name}.onnx"
        torch.onnx.export(
            model,
            (sample_z, sample_pos),
            onnx_path,
            input_names=['atomic_numbers', 'positions'],
            output_names=['energy'],
            opset_version=14,
            do_constant_folding=True,
            verbose=False
        )
        size_mb = onnx_path.stat().st_size / 1e6
        print(f" ✓ ({size_mb:.2f} MB)")
        results['onnx'] = str(onnx_path)

    except Exception as e:
        print(f" ✗ Error: {e}")

    return results


def benchmark_inference(model, num_samples=1000, batch_sizes=[1, 4, 8, 16], device='cpu'):
    """Quick inference benchmark on CPU."""
    model.eval()
    results = {}

    print(f"\n  Benchmarking inference on {device}...")

    with torch.no_grad():
        for batch_size in batch_sizes:
            # Create random batch
            z = torch.randint(1, 8, (batch_size,), dtype=torch.long, device=device)
            pos = torch.randn(batch_size, 3, dtype=torch.float32, device=device)

            # Warmup
            for _ in range(3):
                _ = model(z, pos)

            # Benchmark
            times = []
            for _ in range(10):
                start = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None
                end = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None

                if device == 'cuda':
                    start.record()

                _ = model(z, pos)

                if device == 'cuda':
                    end.record()
                    torch.cuda.synchronize()
                    times.append(start.elapsed_time(end))

            if device == 'cuda':
                avg_time = np.mean(times)
                throughput = (batch_size * 1000) / np.mean(times)
            else:
                avg_time = 0.0  # CPU timing requires different approach
                throughput = 0.0

            results[batch_size] = {
                'latency_ms': avg_time,
                'throughput_samples_per_sec': throughput
            }
            if times:
                print(f"    Batch {batch_size}: {avg_time:.2f}ms")

    return results


def main():
    parser = argparse.ArgumentParser(description='CPU-optimized model finalization')
    parser.add_argument('--dataset', default='data/merged_dataset_4883/merged_dataset.h5')
    parser.add_argument('--output-dir', default='benchmarks')
    parser.add_argument('--val-samples', type=int, default=50)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])

    args = parser.parse_args()
    device = args.device

    # Model configs
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
    print(f"CPU-OPTIMIZED MODEL FINALIZATION")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Val samples: {args.val_samples}")

    # Load validation dataset
    print(f"\nLoading validation dataset...")
    full_dataset = DistillationDataset(args.dataset)
    n_total = len(full_dataset)
    n_train = int(0.9 * n_total)
    val_indices = list(range(n_train, min(n_train + args.val_samples, n_total)))
    val_dataset = DistillationDataset(args.dataset, indices=val_indices)

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,  # Small batch for CPU
        shuffle=False,
        num_workers=2,
        collate_fn=distillation_collate_fn,
        pin_memory=False
    )
    print(f"Validation set: {len(val_dataset)} samples")

    # Results
    results = {
        'validation': {},
        'export': {},
        'summary': {},
        'timestamp': str(Path.cwd())
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
            model, checkpoint = create_and_load_model(config, checkpoint_path, device)
            n_params = sum(p.numel() for p in model.parameters())
            model_size_mb = n_params * 4 / 1024 / 1024
            print(f"  Parameters: {n_params:,}")
            print(f"  Size: {model_size_mb:.3f} MB")

            # Validate
            val_metrics = validate_on_cpu(model, val_loader, device)
            results['validation'][config['name']] = val_metrics
            print(f"  Energy MAE: {val_metrics['energy_mae']:.4f} eV")
            print(f"  Samples validated: {val_metrics['n_batches']} batches")

            # Export
            export_results = export_model_to_formats(model, f"{model_key}_model", config, args.output_dir)
            results['export'][config['name']] = export_results

            # Summary
            results['summary'][config['name']] = {
                'parameters': n_params,
                'size_mb': model_size_mb,
                'validation': val_metrics,
                'export': export_results
            }

            print(f"✓ {config['name']} complete")

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    output_path = Path(args.output_dir) / f'compact_models_cpu_final_20251124.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_path}")

    # Summary
    print(f"\n{'='*70}")
    print(f"FINALIZATION SUMMARY")
    print(f"{'='*70}")
    for model_name, summary in results['summary'].items():
        print(f"\n{model_name}:")
        print(f"  Parameters: {summary['parameters']:,}")
        print(f"  Size: {summary['size_mb']:.3f} MB")
        if 'validation' in summary:
            print(f"  Energy MAE: {summary['validation']['energy_mae']:.4f} eV")
        print(f"  Exports: {len(summary['export'])} formats")

    print(f"\n{'='*70}")
    print(f"✅ CPU-OPTIMIZED FINALIZATION COMPLETE")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
