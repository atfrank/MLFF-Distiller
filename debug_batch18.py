#!/usr/bin/env python3
"""
Debug - check molecule positions.
"""

import numpy as np
from ase.build import molecule

molecules = [
    molecule('H2O'),
    molecule('NH3'),
    molecule('CH4'),
    molecule('C2H6'),
]

print("Molecule centers:")
for i, mol in enumerate(molecules):
    center = np.mean(mol.positions, axis=0)
    print(f"  {i} ({mol.get_chemical_formula()}): {center}")
