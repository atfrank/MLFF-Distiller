#!/usr/bin/env python3
"""
Debug batch processing - check tensor contents.
"""

import sys
from pathlib import Path
import numpy as np
import torch
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
    molecule('H2O'),   # 3 atoms
    molecule('NH3'),   # 4 atoms
    molecule('CH4'),   # 5 atoms
]

print("Molecules:")
for i, mol in enumerate(molecules):
    print(f"  {i}: {mol.get_chemical_formula()} ({len(mol)} atoms)")
    print(f"      Positions shape: {mol.positions.shape}")
    print(f"      Atomic numbers: {mol.get_atomic_numbers()}")

# Manually call _prepare_batch to see what it produces
batch_data = calc._prepare_batch(molecules)

print("\nBatch data:")
print(f"  atomic_numbers shape: {batch_data['atomic_numbers'].shape}")
print(f"  atomic_numbers: {batch_data['atomic_numbers']}")
print(f"  positions shape: {batch_data['positions'].shape}")
print(f"  batch shape: {batch_data['batch'].shape}")
print(f"  batch: {batch_data['batch']}")
print(f"  atom_counts: {batch_data['atom_counts']}")

# Verify batch assignments
print("\nBatch assignments:")
for i in range(len(molecules)):
    mask = batch_data['batch'] == i
    n_atoms = mask.sum().item()
    print(f"  Structure {i}: {n_atoms} atoms (expected {len(molecules[i])})")

# Now run forward pass
print("\nRunning forward pass...")
with torch.enable_grad():
    batch_results = calc._batch_forward(batch_data)

print(f"\nBatch results:")
print(f"  energies shape: {batch_results['energies'].shape}")
print(f"  energies: {batch_results['energies']}")
print(f"  forces shape: {batch_results['forces'].shape}")
print(f"  atom_counts: {batch_results['atom_counts']}")

# Unstack forces manually
print("\nUnstacking forces:")
forces = batch_results['forces'].detach().cpu().numpy()
atom_offset = 0
for i, mol in enumerate(molecules):
    n_atoms = batch_results['atom_counts'][i]
    structure_forces = forces[atom_offset:atom_offset + n_atoms]
    print(f"  Structure {i}: offset={atom_offset}, n_atoms={n_atoms}, shape={structure_forces.shape}")
    print(f"    Forces:\n{structure_forces}")
    atom_offset += n_atoms
