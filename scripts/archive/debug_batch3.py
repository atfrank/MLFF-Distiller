#!/usr/bin/env python3
"""
Debug batch processing - check if positions are being reused.
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

# Create FRESH molecules - ensure no caching
def make_fresh_molecules():
    return [
        molecule('H2O'),
        molecule('NH3'),
        molecule('CH4'),
    ]

# Sequential with fresh molecules
print("Sequential with fresh molecules:")
seq_results = []
molecules_seq = make_fresh_molecules()
for i, mol in enumerate(molecules_seq):
    mol.calc = calc
    energy = mol.get_potential_energy()
    forces = mol.get_forces()
    seq_results.append({'energy': energy, 'forces': forces})
    print(f"  {i}: energy={energy:.6f}, forces_norm={np.linalg.norm(forces):.4f}")

# Batch with fresh molecules
print("\nBatch with fresh molecules:")
molecules_batch = make_fresh_molecules()
batch_results = calc.calculate_batch(molecules_batch, properties=['energy', 'forces'])
for i, result in enumerate(batch_results):
    print(f"  {i}: energy={result['energy']:.6f}, forces_norm={np.linalg.norm(result['forces']):.4f}")

# Compare
print("\nComparison:")
for i in range(len(seq_results)):
    energy_diff = abs(seq_results[i]['energy'] - batch_results[i]['energy'])
    forces_diff = np.max(np.abs(seq_results[i]['forces'] - batch_results[i]['forces']))
    status = "PASS" if forces_diff < 1e-5 else "FAIL"
    print(f"  {i}: {status} (energy_diff={energy_diff:.2e}, forces_diff={forces_diff:.2e})")

    if forces_diff > 1e-5:
        print(f"    Sequential forces sum: {np.sum(seq_results[i]['forces'], axis=0)}")
        print(f"    Batch forces sum:      {np.sum(batch_results[i]['forces'], axis=0)}")
