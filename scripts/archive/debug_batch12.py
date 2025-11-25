#!/usr/bin/env python3
"""
Debug - check if ASE caching is interfering.
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
]

print("Testing CH4 (molecule 2) in isolation vs after other calculations:")

# Test CH4 in isolation
mol_ch4 = molecules[2]
mol_ch4.calc = calc
energy_isolated = mol_ch4.get_potential_energy()
forces_isolated = mol_ch4.get_forces()
print(f"\nCH4 in isolation:")
print(f"  Energy: {energy_isolated:.8f}")
print(f"  Forces:\n{forces_isolated}")

# Now run H2O and NH3 first, then CH4
print(f"\nRunning H2O first:")
mol_h2o = molecules[0]
mol_h2o.calc = calc
_ = mol_h2o.get_potential_energy()
_ = mol_h2o.get_forces()

print(f"Running NH3 second:")
mol_nh3 = molecules[1]
mol_nh3.calc = calc
_ = mol_nh3.get_potential_energy()
_ = mol_nh3.get_forces()

print(f"Running CH4 third (after H2O and NH3):")
mol_ch4_2 = molecules[2]
mol_ch4_2.calc = calc
energy_after = mol_ch4_2.get_potential_energy()
forces_after = mol_ch4_2.get_forces()
print(f"  Energy: {energy_after:.8f}")
print(f"  Forces:\n{forces_after}")

# Compare
energy_diff = abs(energy_isolated - energy_after)
forces_diff = np.max(np.abs(forces_isolated - forces_after))
print(f"\nDifference:")
print(f"  Energy: {energy_diff:.2e}")
print(f"  Forces: {forces_diff:.2e}")

if forces_diff < 1e-5:
    print("  Status: SAME (no interference)")
else:
    print(f"  Status: DIFFERENT (interference detected!)")
    print(f"  Forces difference:\n{forces_isolated - forces_after}")
