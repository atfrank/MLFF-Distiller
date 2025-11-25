#!/usr/bin/env python3
"""
Extract MD trajectory for PyMOL visualization

Runs a short MD simulation and saves the trajectory in XYZ format
that can be loaded directly into PyMOL.

Usage:
    python scripts/extract_trajectory_for_pymol.py \
        --checkpoint checkpoints/best_model.pt \
        --duration 5 \
        --output md_trajectory_5ps.xyz
"""

import sys
from pathlib import Path
import argparse
import numpy as np
from ase.build import molecule
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
from ase.io import write

# Add src to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from mlff_distiller.inference import StudentForceFieldCalculator

def run_md_for_pymol(
    checkpoint_path,
    duration_ps=5,
    timestep_fs=0.5,
    temperature_K=300,
    output_file='trajectory.xyz',
    save_interval=10
):
    """
    Run MD simulation and save trajectory for PyMOL.

    Args:
        checkpoint_path: Path to model checkpoint
        duration_ps: Simulation duration in picoseconds
        timestep_fs: Timestep in femtoseconds
        temperature_K: Initial temperature in Kelvin
        output_file: Output XYZ trajectory file
        save_interval: Save every N steps
    """
    print("="*60)
    print("MD Trajectory Generation for PyMOL")
    print("="*60)

    # Create test system
    print("\n[1/4] Creating molecular system...")
    atoms = molecule('C6H6')  # Benzene
    print(f"  System: Benzene (C6H6)")
    print(f"  Atoms: {len(atoms)}")

    # Set up calculator
    print("\n[2/4] Loading force field...")
    calc = StudentForceFieldCalculator(
        checkpoint_path=checkpoint_path,
        device='cuda'
    )
    atoms.calc = calc
    print("  ✓ Force field loaded")

    # Initialize velocities
    print(f"\n[3/4] Initializing MD (T={temperature_K}K)...")
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
    print(f"  Initial temperature: {atoms.get_temperature():.1f} K")

    # Set up dynamics
    n_steps = int(duration_ps * 1000 / timestep_fs)
    dyn = VelocityVerlet(atoms, timestep=timestep_fs * units.fs)

    print(f"  Timestep: {timestep_fs} fs")
    print(f"  Duration: {duration_ps} ps")
    print(f"  Total steps: {n_steps:,}")
    print(f"  Save interval: every {save_interval} steps")
    print(f"  Total frames: {n_steps//save_interval + 1}")

    # Storage for trajectory
    trajectory = []
    trajectory.append(atoms.copy())  # Initial frame

    # Progress tracking
    energies = []
    temperatures = []

    def save_frame():
        trajectory.append(atoms.copy())
        energies.append(atoms.get_potential_energy() + atoms.get_kinetic_energy())
        temperatures.append(atoms.get_temperature())

        if len(trajectory) % 10 == 0:
            print(f"  Frame {len(trajectory)}/{n_steps//save_interval + 1} "
                  f"(t={dyn.get_time()/units.fs:.1f} fs, "
                  f"T={atoms.get_temperature():.1f} K, "
                  f"E={energies[-1]:.3f} eV)")

    # Attach observer
    dyn.attach(save_frame, interval=save_interval)

    # Run simulation
    print(f"\n[4/4] Running MD simulation...")
    dyn.run(n_steps)

    # Save trajectory
    print(f"\n✓ Simulation complete!")
    print(f"  Total frames: {len(trajectory)}")
    print(f"  Energy drift: {(energies[-1] - energies[0])/energies[0]*100:.3f}%")
    print(f"  Temperature: {np.mean(temperatures):.1f} ± {np.std(temperatures):.1f} K")

    output_path = Path(output_file)
    write(output_path, trajectory)
    print(f"\n✓ Trajectory saved: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"  Format: XYZ (PyMOL compatible)")

    # Print PyMOL commands
    print("\n" + "="*60)
    print("To visualize in PyMOL:")
    print("="*60)
    print(f"  pymol {output_path}")
    print("\nOr in PyMOL console:")
    print(f"  load {output_path}")
    print("  set movie_loop, 1")
    print("  mplay")
    print("\nUseful PyMOL commands:")
    print("  hide everything; show sticks")
    print("  color atomic")
    print("  set stick_radius, 0.15")
    print("  set sphere_scale, 0.3")
    print("  orient")
    print("="*60)

    return len(trajectory)

def main():
    parser = argparse.ArgumentParser(
        description="Generate MD trajectory for PyMOL visualization"
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default='checkpoints/best_model.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=5.0,
        help='Simulation duration in picoseconds (default: 5)'
    )
    parser.add_argument(
        '--timestep',
        type=float,
        default=0.5,
        help='Timestep in femtoseconds (default: 0.5)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=300.0,
        help='Initial temperature in Kelvin (default: 300)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='md_trajectory.xyz',
        help='Output XYZ file (default: md_trajectory.xyz)'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=10,
        help='Save every N steps (default: 10)'
    )

    args = parser.parse_args()

    n_frames = run_md_for_pymol(
        checkpoint_path=args.checkpoint,
        duration_ps=args.duration,
        timestep_fs=args.timestep,
        temperature_K=args.temperature,
        output_file=args.output,
        save_interval=args.interval
    )

    return 0

if __name__ == '__main__':
    sys.exit(main())
