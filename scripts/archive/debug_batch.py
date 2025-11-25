#!/usr/bin/env python3
"""
Debug batch processing issue.
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

# Test single problematic molecule
mol = molecule('CH4')
print(f"Testing {mol.get_chemical_formula()} with {len(mol)} atoms")

# Sequential
mol.calc = calc
energy_seq = mol.get_potential_energy()
forces_seq = mol.get_forces()

print(f"\nSequential:")
print(f"  Energy: {energy_seq:.6f} eV")
print(f"  Forces shape: {forces_seq.shape}")
print(f"  Forces:\n{forces_seq}")

# Batch
batch_results = calc.calculate_batch([mol], properties=['energy', 'forces'])
energy_batch = batch_results[0]['energy']
forces_batch = batch_results[0]['forces']

print(f"\nBatch:")
print(f"  Energy: {energy_batch:.6f} eV")
print(f"  Forces shape: {forces_batch.shape}")
print(f"  Forces:\n{forces_batch}")

# Compare
print(f"\nDifference:")
print(f"  Energy: {abs(energy_seq - energy_batch):.2e} eV")
print(f"  Forces max diff: {np.max(np.abs(forces_seq - forces_batch)):.2e} eV/A")
print(f"  Forces:\n{forces_seq - forces_batch}")
