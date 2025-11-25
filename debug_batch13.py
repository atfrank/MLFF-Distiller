#!/usr/bin/env python3
"""
Debug - check atomic numbers buffer.
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

# Create molecules with different sizes
molecules = [
    molecule('H2O'),   # 3 atoms
    molecule('NH3'),   # 4 atoms
    molecule('CH4'),   # 5 atoms
]

print("Molecule sizes:")
for i, mol in enumerate(molecules):
    print(f"  {i}: {mol.get_chemical_formula()} ({len(mol)} atoms)")

# H2O: 3 atoms [8, 1, 1]
print("\nCalculating H2O:")
mol_h2o = molecules[0]
mol_h2o.calc = calc
energy_h2o = mol_h2o.get_potential_energy()
print(f"  Energy: {energy_h2o:.6f}")
print(f"  Numbers buffer size after: {calc._numbers_buffer.shape if calc._numbers_buffer is not None else 'None'}")

# NH3: 4 atoms [7, 1, 1, 1]
print("\nCalculating NH3:")
mol_nh3 = molecules[1]
mol_nh3.calc = calc
energy_nh3 = mol_nh3.get_potential_energy()
print(f"  Energy: {energy_nh3:.6f}")
print(f"  Numbers buffer size after: {calc._numbers_buffer.shape if calc._numbers_buffer is not None else 'None'}")

# CH4: 5 atoms [6, 1, 1, 1, 1]
print("\nCalculating CH4:")
mol_ch4 = molecules[2]
mol_ch4.calc = calc
energy_ch4 = mol_ch4.get_potential_energy()
forces_ch4 = mol_ch4.get_forces()
print(f"  Energy: {energy_ch4:.6f}")
print(f"  Numbers buffer size after: {calc._numbers_buffer.shape if calc._numbers_buffer is not None else 'None'}")
print(f"  Actual atomic numbers: {mol_ch4.get_atomic_numbers()}")
if calc._numbers_buffer is not None:
    print(f"  Buffer contents: {calc._numbers_buffer.cpu().numpy()}")
