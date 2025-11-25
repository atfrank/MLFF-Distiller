#!/usr/bin/env python3
"""
Debug - test if sequential calculate() is deterministic.
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

# Create ONE CH4 molecule
mol = molecule('CH4')

print("Testing determinism of calculate() method:")
print(f"Molecule: {mol.get_chemical_formula()} ({len(mol)} atoms)")

# Run 5 times
results = []
for i in range(5):
    mol.calc = calc
    energy = mol.get_potential_energy()
    forces = mol.get_forces()
    forces_norm = np.linalg.norm(forces)
    results.append({'energy': energy, 'forces': forces, 'forces_norm': forces_norm})
    print(f"  Run {i}: energy={energy:.8f}, forces_norm={forces_norm:.8f}")

# Check if all identical
all_identical = True
for i in range(1, len(results)):
    energy_diff = abs(results[i]['energy'] - results[0]['energy'])
    forces_diff = np.max(np.abs(results[i]['forces'] - results[0]['forces']))
    if energy_diff > 1e-10 or forces_diff > 1e-10:
        all_identical = False
        print(f"\nRun {i} differs from Run 0:")
        print(f"  Energy diff: {energy_diff:.2e}")
        print(f"  Forces diff: {forces_diff:.2e}")

if all_identical:
    print("\nSUCCESS: All runs are identical!")
else:
    print("\nFAILURE: Runs are not deterministic!")
