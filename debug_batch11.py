#!/usr/bin/env python3
"""
Debug - use the SAME molecule object, just compute in different ways.
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

# Create ONE molecule
mol = molecule('CH4')
print(f"Testing with single {mol.get_chemical_formula()} molecule")
print(f"Positions:\n{mol.get_positions()}")

# Sequential
print("\nSequential:")
mol.calc = calc
energy_seq = mol.get_potential_energy()
forces_seq = mol.get_forces()
print(f"  Energy: {energy_seq:.8f}")
print(f"  Forces norm: {np.linalg.norm(forces_seq):.8f}")
print(f"  Forces:\n{forces_seq}")

# Batch with single molecule
print("\nBatch (single molecule):")
batch_results = calc.calculate_batch([mol], properties=['energy', 'forces'])
energy_batch = batch_results[0]['energy']
forces_batch = batch_results[0]['forces']
print(f"  Energy: {energy_batch:.8f}")
print(f"  Forces norm: {np.linalg.norm(forces_batch):.8f}")
print(f"  Forces:\n{forces_batch}")

# Compare
energy_diff = abs(energy_seq - energy_batch)
forces_diff = np.max(np.abs(forces_seq - forces_batch))
print(f"\nDifference:")
print(f"  Energy: {energy_diff:.2e}")
print(f"  Forces: {forces_diff:.2e}")

if forces_diff < 1e-5:
    print("  Status: PASS")
else:
    print("  Status: FAIL")
    print(f"\n  Force differences:\n{forces_seq - forces_batch}")
