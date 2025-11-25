#!/usr/bin/env python3
"""
Debug - check if results are deterministic.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from ase.build import molecule

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

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

print(f"Model training mode: {calc.model.training}")

# Create CH4 molecule
mol = molecule('CH4')
print(f"\nTesting {mol.get_chemical_formula()} ({len(mol)} atoms)")

# Run it multiple times sequentially
print("\nSequential runs:")
for i in range(3):
    mol.calc = calc
    energy = mol.get_potential_energy()
    forces = mol.get_forces()
    print(f"  Run {i}: energy={energy:.8f}, forces_norm={np.linalg.norm(forces):.8f}")

# Run it multiple times in batch
print("\nBatch runs (single molecule):")
for i in range(3):
    results = calc.calculate_batch([mol], properties=['energy', 'forces'])
    energy = results[0]['energy']
    forces = results[0]['forces']
    print(f"  Run {i}: energy={energy:.8f}, forces_norm={np.linalg.norm(forces):.8f}")

# Run with 3 copies of the same molecule
print("\nBatch with 3 identical molecules:")
molecules = [molecule('CH4'), molecule('CH4'), molecule('CH4')]
results = calc.calculate_batch(molecules, properties=['energy', 'forces'])
for i, result in enumerate(results):
    energy = result['energy']
    forces = result['forces']
    print(f"  Mol {i}: energy={energy:.8f}, forces_norm={np.linalg.norm(forces):.8f}")
