"""
Validate MD Stability with Optimized Inference

Quick test to ensure MD simulations remain stable with TorchScript optimization.

Author: CUDA Optimization Engineer
Date: 2025-11-24
"""

import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mlff_distiller.inference.ase_calculator import StudentForceFieldCalculator

print("Testing MD stability with TorchScript optimization...")

# Create test molecule (H2O)
atoms = Atoms(
    'H2O',
    positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.24, 0.93, 0.0]]
)

# Attach TorchScript calculator
calc = StudentForceFieldCalculator(
    checkpoint_path='checkpoints/best_model.pt',
    jit_path='models/student_model_jit.pt',
    use_jit=True,
    device='cuda'
)
atoms.calc = calc

# Initialize velocities (300K)
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# Run short MD
dyn = VelocityVerlet(atoms, timestep=0.5*units.fs)

initial_energy = atoms.get_potential_energy()
print(f"Initial energy: {initial_energy:.4f} eV")

energies = [initial_energy]

for i in range(100):
    dyn.run(1)
    energy = atoms.get_potential_energy()
    energies.append(energy)

    if i % 20 == 0:
        print(f"Step {i}: E = {energy:.4f} eV")

final_energy = energies[-1]
energy_drift = abs(final_energy - initial_energy)

print(f"\nFinal energy: {final_energy:.4f} eV")
print(f"Energy drift: {energy_drift:.6f} eV ({energy_drift/abs(initial_energy)*100:.4f}%)")

if energy_drift / abs(initial_energy) < 0.01:  # <1% drift
    print("\nMD simulation is STABLE with TorchScript optimization!")
else:
    print("\nWARNING: Large energy drift detected!")

print(f"Energy std: {np.std(energies):.6f} eV")
