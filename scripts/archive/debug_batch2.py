#!/usr/bin/env python3
"""
Debug batch processing issue with multiple molecules.
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

# Test with 3 molecules
molecules = [
    molecule('H2O'),   # 3 atoms - index 0
    molecule('NH3'),   # 4 atoms - index 1
    molecule('CH4'),   # 5 atoms - index 2
]

print("Testing batch with multiple molecules:")
for i, mol in enumerate(molecules):
    print(f"  {i}: {mol.get_chemical_formula()} ({len(mol)} atoms)")

# Sequential
seq_results = []
for i, mol in enumerate(molecules):
    mol.calc = calc
    energy = mol.get_potential_energy()
    forces = mol.get_forces()
    seq_results.append({'energy': energy, 'forces': forces})
    print(f"\nSequential {i} ({mol.get_chemical_formula()}):")
    print(f"  Energy: {energy:.6f} eV")
    print(f"  Forces shape: {forces.shape}")
    print(f"  Forces sum: {np.sum(forces, axis=0)}")

# Batch
batch_results = calc.calculate_batch(molecules, properties=['energy', 'forces'])

print("\n" + "=" * 80)
print("BATCH RESULTS:")
for i, (mol, result) in enumerate(zip(molecules, batch_results)):
    print(f"\nBatch {i} ({mol.get_chemical_formula()}):")
    print(f"  Energy: {result['energy']:.6f} eV")
    print(f"  Forces shape: {result['forces'].shape}")
    print(f"  Forces sum: {np.sum(result['forces'], axis=0)}")

print("\n" + "=" * 80)
print("COMPARISON:")
for i, (mol, seq, batch) in enumerate(zip(molecules, seq_results, batch_results)):
    energy_diff = abs(seq['energy'] - batch['energy'])
    forces_diff = np.max(np.abs(seq['forces'] - batch['forces']))

    print(f"\nMolecule {i} ({mol.get_chemical_formula()}):")
    print(f"  Energy diff: {energy_diff:.2e} eV")
    print(f"  Forces diff: {forces_diff:.2e} eV/A")

    if forces_diff > 1e-5:
        print(f"  FAILED! Forces don't match!")
        print(f"  Sequential forces:\n{seq['forces']}")
        print(f"  Batch forces:\n{batch['forces']}")
