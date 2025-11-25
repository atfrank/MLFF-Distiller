#!/usr/bin/env python3
"""
Debug - test with offset molecules.
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

# Create 3 CH4 molecules with different centers
molecules = []
for i in range(3):
    mol = molecule('CH4')
    # Offset each molecule by 20 Angstroms (well beyond cutoff)
    mol.positions += np.array([i * 20.0, 0, 0])
    molecules.append(mol)

print("Molecules with offsets:")
for i, mol in enumerate(molecules):
    print(f"  {i}: center at {np.mean(mol.positions, axis=0)}")

# Sequential
print("\nSequential:")
seq_results = []
for i, mol in enumerate(molecules):
    mol.calc = calc
    energy = mol.get_potential_energy()
    forces = mol.get_forces()
    seq_results.append({'energy': energy, 'forces': forces})
    print(f"  {i}: energy={energy:.8f}, forces_norm={np.linalg.norm(forces):.8f}")

# Batch
print("\nBatch:")
batch_results = calc.calculate_batch(molecules, properties=['energy', 'forces'])
for i, result in enumerate(batch_results):
    print(f"  {i}: energy={result['energy']:.8f}, forces_norm={np.linalg.norm(result['forces']):.8f}")

# Compare
print("\nComparison:")
all_match = True
for i in range(len(molecules)):
    energy_diff = abs(seq_results[i]['energy'] - batch_results[i]['energy'])
    forces_diff = np.max(np.abs(seq_results[i]['forces'] - batch_results[i]['forces']))
    status = "PASS" if forces_diff < 1e-5 else "FAIL"
    print(f"  {i}: {status} (energy_diff={energy_diff:.2e}, forces_diff={forces_diff:.2e})")
    if forces_diff >= 1e-5:
        all_match = False

if all_match:
    print("\nSUCCESS: All tests passed!")
else:
    print("\nFAILURE: Some tests failed!")
