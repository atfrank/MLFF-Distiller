#!/usr/bin/env python3
"""
Debug - test on CPU to see if it's a CUDA issue.
"""

import sys
from pathlib import Path
import torch
from ase.build import molecule

# Add src to path
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from mlff_distiller.inference import StudentForceFieldCalculator

# Initialize calculator on CPU
checkpoint_path = REPO_ROOT / 'checkpoints' / 'best_model.pt'
calc = StudentForceFieldCalculator(
    checkpoint_path=checkpoint_path,
    device='cpu',
    enable_timing=True
)

print(f"Model device: {next(calc.model.parameters()).device}")
print(f"Model training mode: {calc.model.training}")

# Create molecules
mol_h2o = molecule('H2O')
mol_nh3 = molecule('NH3')
mol_ch4 = molecule('CH4')

# Convert CH4 to tensors
atomic_numbers_ch4 = torch.tensor(mol_ch4.get_atomic_numbers(), dtype=torch.long)
positions_ch4 = torch.tensor(mol_ch4.get_positions(), dtype=torch.float32)

print("\nTest 1: Calculate CH4 in isolation")
energy1, forces1 = calc.model.predict_energy_and_forces(
    atomic_numbers_ch4, positions_ch4.clone()
)
print(f"  Energy: {energy1.item():.8f}")
print(f"  Forces norm: {torch.norm(forces1).item():.8f}")

print("\nTest 2: Calculate H2O, then CH4")
atomic_numbers_h2o = torch.tensor(mol_h2o.get_atomic_numbers(), dtype=torch.long)
positions_h2o = torch.tensor(mol_h2o.get_positions(), dtype=torch.float32)
_, _ = calc.model.predict_energy_and_forces(atomic_numbers_h2o, positions_h2o.clone())

energy2, forces2 = calc.model.predict_energy_and_forces(
    atomic_numbers_ch4, positions_ch4.clone()
)
print(f"  Energy: {energy2.item():.8f}")
print(f"  Forces norm: {torch.norm(forces2).item():.8f}")

print("\nTest 3: Calculate NH3, then CH4")
atomic_numbers_nh3 = torch.tensor(mol_nh3.get_atomic_numbers(), dtype=torch.long)
positions_nh3 = torch.tensor(mol_nh3.get_positions(), dtype=torch.float32)
_, _ = calc.model.predict_energy_and_forces(atomic_numbers_nh3, positions_nh3.clone())

energy3, forces3 = calc.model.predict_energy_and_forces(
    atomic_numbers_ch4, positions_ch4.clone()
)
print(f"  Energy: {energy3.item():.8f}")
print(f"  Forces norm: {torch.norm(forces3).item():.8f}")

# Compare
print("\nComparison:")
diff_12 = torch.max(torch.abs(forces1 - forces2)).item()
diff_13 = torch.max(torch.abs(forces1 - forces3)).item()
print(f"  Test 1 vs Test 2: energy_diff={abs(energy1.item() - energy2.item()):.2e}, forces_diff={diff_12:.2e}")
print(f"  Test 1 vs Test 3: energy_diff={abs(energy1.item() - energy3.item()):.2e}, forces_diff={diff_13:.2e}")

if diff_12 < 1e-6 and diff_13 < 1e-6:
    print("\nCPU: DETERMINISTIC")
else:
    print("\nCPU: NON-DETERMINISTIC (bug in model)")
