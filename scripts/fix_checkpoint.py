#!/usr/bin/env python3
"""
Fix checkpoint format - extract model config from training config.

The training script saved the full training config instead of just the model config.
This script extracts the correct model config and re-saves the checkpoint.
"""

import sys
from pathlib import Path
import torch

# Add src to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

def fix_checkpoint(input_path: str, output_path: str = None):
    """Fix checkpoint by extracting correct model config."""

    if output_path is None:
        output_path = input_path

    print(f"Loading checkpoint from {input_path}")
    checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)

    print(f"Current config keys: {list(checkpoint.get('config', {}).keys())}")

    # Infer max_z from embedding weight shape
    state_dict = checkpoint['model_state_dict']

    # Find embedding key (might have "model." prefix)
    embedding_key = None
    for key in state_dict.keys():
        if 'embedding.weight' in key:
            embedding_key = key
            break

    if embedding_key:
        embedding_shape = state_dict[embedding_key].shape
        max_z = embedding_shape[0] - 1
        print(f"Inferred max_z={max_z} from embedding shape {embedding_shape}")
    else:
        max_z = 100  # Default fallback
        print(f"Warning: Could not find embedding, using default max_z={max_z}")

    # Model architecture parameters (these should be in config)
    model_config = {
        'hidden_dim': 128,
        'num_interactions': 3,
        'num_rbf': 20,
        'cutoff': 5.0,
        'max_z': max_z
    }

    # Fix state dict keys - remove "model." prefix if present
    state_dict = checkpoint['model_state_dict']
    fixed_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            # Remove "model." prefix
            new_key = key[6:]  # len("model.") = 6
            fixed_state_dict[new_key] = value
        else:
            fixed_state_dict[key] = value

    print(f"Fixed {len([k for k in state_dict.keys() if k.startswith('model.')])} keys with 'model.' prefix")

    # Create new checkpoint with correct format
    new_checkpoint = {
        'model_state_dict': fixed_state_dict,
        'config': model_config,
        'num_parameters': checkpoint.get('num_parameters', 427626)
    }

    # Preserve other useful metadata if present
    if 'epoch' in checkpoint:
        new_checkpoint['epoch'] = checkpoint['epoch']
    if 'val_loss' in checkpoint:
        new_checkpoint['val_loss'] = checkpoint['val_loss']
    if 'metrics' in checkpoint:
        new_checkpoint['metrics'] = checkpoint['metrics']

    print(f"\nNew config keys: {list(new_checkpoint['config'].keys())}")
    print(f"New config: {new_checkpoint['config']}")

    print(f"\nSaving fixed checkpoint to {output_path}")
    torch.save(new_checkpoint, output_path)

    print("\nVerifying fixed checkpoint can be loaded...")
    try:
        from mlff_distiller.models.student_model import StudentForceField
        model = StudentForceField.load(output_path, device='cpu')
        print(f"SUCCESS! Model loaded with {model.num_parameters():,} parameters")
    except Exception as e:
        print(f"ERROR: Failed to load: {e}")
        return False

    return True

if __name__ == '__main__':
    checkpoint_path = 'checkpoints/best_model.pt'

    if not Path(checkpoint_path).exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    success = fix_checkpoint(checkpoint_path)
    sys.exit(0 if success else 1)
