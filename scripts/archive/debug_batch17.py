#!/usr/bin/env python3
"""
Debug - test CH4 alone vs in batch.
"""

import sys
from pathlib import Path
import numpy as np
from ase.build import molecule

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

# Create molecules
molecules = [
    molecule('H2O'),
    molecule('NH3'),
    molecule('CH4'),
    molecule('C2H6'),
]

# Test CH4 alone
print("CH4 alone in batch:")
ch4_alone = [molecules[2]]
results_alone = calc.calculate_batch(ch4_alone)
print(f"  Energy: {results_alone[0]['energy']:.8f}")
print(f"  Forces norm: {np.linalg.norm(results_alone[0]['forces']):.8f}")
print(f"  Forces:\n{results_alone[0]['forces']}")

# Test CH4 in batch with all 4 molecules
print("\nCH4 in batch with H2O, NH3, CH4, C2H6:")
results_batch = calc.calculate_batch(molecules)
print(f"  Energy: {results_batch[2]['energy']:.8f}")
print(f"  Forces norm: {np.linalg.norm(results_batch[2]['forces']):.8f}")
print(f"  Forces:\n{results_batch[2]['forces']}")

# Compare
energy_diff = abs(results_alone[0]['energy'] - results_batch[2]['energy'])
forces_diff = np.max(np.abs(results_alone[0]['forces'] - results_batch[2]['forces']))
print(f"\nDifference:")
print(f"  Energy: {energy_diff:.2e}")
print(f"  Forces: {forces_diff:.2e}")

if forces_diff < 1e-5:
    print("  Status: PASS (batch size doesn't affect results)")
else:
    print("  Status: FAIL (batch size affects results!)")
