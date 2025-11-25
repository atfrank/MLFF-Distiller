#!/usr/bin/env python3
"""
Debug - manually trace through batch forward pass.
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

# Create 3 identical CH4 molecules
molecules = [molecule('CH4'), molecule('CH4'), molecule('CH4')]

# Prepare batch
batch_data = calc._prepare_batch(molecules)

print("Batch data prepared:")
print(f"  Positions:\n{batch_data['positions']}")
print(f"  Batch indices: {batch_data['batch']}")

# Run forward pass WITH gradients
positions = batch_data['positions']
positions.requires_grad_(True)

print("\nRunning forward pass...")
energies = calc.model(
    atomic_numbers=batch_data['atomic_numbers'],
    positions=positions,
    cell=None,
    pbc=None,
    batch=batch_data['batch']
)

print(f"Energies: {energies}")

# Compute forces
print("\nComputing forces via autograd...")
forces = -torch.autograd.grad(
    energies.sum(),
    positions,
    create_graph=False,
    retain_graph=False
)[0]

print(f"Forces shape: {forces.shape}")
print(f"Forces:\n{forces}")

# Unstack per structure
print("\nForces per structure:")
atom_counts = [5, 5, 5]
atom_offset = 0
for i in range(3):
    n_atoms = atom_counts[i]
    structure_forces = forces[atom_offset:atom_offset + n_atoms]
    print(f"\nStructure {i}:")
    print(f"  Energy: {energies[i].item():.8f}")
    print(f"  Forces norm: {torch.norm(structure_forces).item():.8f}")
    print(f"  Forces:\n{structure_forces}")
    atom_offset += n_atoms
