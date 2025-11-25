#!/usr/bin/env python3
"""
Debug - check for model buffers.
"""

import sys
from pathlib import Path

# Add src to path
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from mlff_distiller.inference import StudentForceFieldCalculator

# Initialize calculator
checkpoint_path = REPO_ROOT / 'checkpoints' / 'best_model.pt'
calc = StudentForceFieldCalculator(
    checkpoint_path=checkpoint_path,
    device='cuda',
    enable_timing=True
)

print("Model buffers:")
for name, buffer in calc.model.named_buffers():
    print(f"  {name}: {buffer.shape}, requires_grad={buffer.requires_grad}")

print("\nModel parameters (first few):")
for i, (name, param) in enumerate(calc.model.named_parameters()):
    if i < 5:
        print(f"  {name}: {param.shape}, requires_grad={param.requires_grad}")

print("\nModel modules:")
for name, module in calc.model.named_modules():
    if name:  # Skip root
        print(f"  {name}: {type(module).__name__}")
