#!/usr/bin/env python3
"""
Debug - check if ASE molecule() gives identical positions.
"""

import numpy as np
from ase.build import molecule

# Create 3 CH4 molecules
mol1 = molecule('CH4')
mol2 = molecule('CH4')
mol3 = molecule('CH4')

print("CH4 molecule positions:")
print("\nMol 1:")
print(mol1.get_positions())
print("\nMol 2:")
print(mol2.get_positions())
print("\nMol 3:")
print(mol3.get_positions())

# Check if they're identical
diff_12 = np.max(np.abs(mol1.get_positions() - mol2.get_positions()))
diff_13 = np.max(np.abs(mol1.get_positions() - mol3.get_positions()))

print(f"\nMax difference mol1 vs mol2: {diff_12}")
print(f"Max difference mol1 vs mol3: {diff_13}")

if diff_12 < 1e-10 and diff_13 < 1e-10:
    print("Positions are IDENTICAL")
else:
    print("Positions are DIFFERENT!")
