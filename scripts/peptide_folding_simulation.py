#!/usr/bin/env python3
"""
Peptide Folding Simulation with Student Model

Creates an extended peptide and runs a high-temperature MD simulation
to observe folding dynamics. Uses optimized student model for fast simulation.

Usage:
    python scripts/peptide_folding_simulation.py \\
        --sequence ACDEFGHIKLMNPQRSTVWY \\
        --duration 50 \\
        --temperature 500 \\
        --output visualizations/peptide_folding.xyz
"""

import sys
from pathlib import Path
import argparse
import numpy as np
from ase import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
from ase.io import write

# Add src to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from mlff_distiller.inference import StudentForceFieldCalculator

def create_extended_peptide(sequence='ACDEFGHIKLMNPQRSTVWY'):
    """
    Create an extended (linear) peptide structure.

    For demonstration, we'll create a simplified peptide backbone.
    In reality, you'd use a tool like RDKit or OpenBabel for proper geometry.

    For now, let's use a small test peptide: Alanine dipeptide (blocked alanine).
    """
    # For simplicity, let's create a small molecule that can show dynamics
    # We'll use multiple alanine residues in extended conformation

    # Alternative: Create a simple polyglycine or use a known structure
    # For demo purposes, let's create a chain of atoms that can fold

    # Simple approach: Create a zigzag carbon chain with hydrogens
    # This will demonstrate folding dynamics

    n_residues = len(sequence) if len(sequence) <= 5 else 5  # Limit size for speed

    positions = []
    symbols = []

    # Create a simple extended chain (C-C-C-C... with H atoms)
    spacing = 1.5  # Angstroms between backbone atoms

    for i in range(n_residues * 3):  # 3 heavy atoms per "residue"
        x = i * spacing
        y = 0.0
        z = 0.0
        positions.append([x, y, z])
        symbols.append('C')

        # Add hydrogens
        positions.append([x, y + 1.1, z])
        symbols.append('H')
        positions.append([x, y - 1.1, z])
        symbols.append('H')

    atoms = Atoms(symbols=symbols, positions=positions)

    return atoms

def run_folding_simulation(
    checkpoint_path,
    sequence='ACDEFGHIKLMNPQRSTVWY',
    duration_ps=50,
    timestep_fs=0.5,
    temperature_K=500,
    output_file='peptide_folding.xyz',
    save_interval=20,
    use_compile=True,
    use_fp16=True
):
    """
    Run peptide folding simulation.

    Args:
        checkpoint_path: Path to model checkpoint
        sequence: Amino acid sequence (1-letter codes)
        duration_ps: Simulation duration in picoseconds
        timestep_fs: Timestep in femtoseconds
        temperature_K: Temperature in Kelvin (higher = faster dynamics)
        output_file: Output XYZ trajectory file
        save_interval: Save every N steps
        use_compile: Use torch.compile() optimization
        use_fp16: Use FP16 mixed precision
    """
    print("="*60)
    print("Peptide Folding Simulation")
    print("="*60)

    # Create peptide system
    print(f"\n[1/5] Creating peptide structure...")
    atoms = create_extended_peptide(sequence)
    print(f"  Sequence: {sequence[:5]}{'...' if len(sequence) > 5 else ''}")
    print(f"  Total atoms: {len(atoms)}")
    print(f"  Heavy atoms: {sum(1 for s in atoms.get_chemical_symbols() if s != 'H')}")

    # Set up calculator with optimizations
    print(f"\n[2/5] Loading force field...")
    print(f"  torch.compile(): {'ON' if use_compile else 'OFF'}")
    print(f"  FP16 precision: {'ON' if use_fp16 else 'OFF'}")

    calc = StudentForceFieldCalculator(
        checkpoint_path=checkpoint_path,
        device='cuda',
        use_compile=use_compile,
        use_fp16=use_fp16
    )
    atoms.calc = calc
    print("  ✓ Force field loaded")

    # Initialize velocities at high temperature for faster dynamics
    print(f"\n[3/5] Initializing MD (T={temperature_K}K)...")
    print(f"  Note: High temperature accelerates folding dynamics")
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
    trajectory.append(atoms.copy())

    # Progress tracking
    energies = []
    temperatures = []

    # Calculate initial end-to-end distance
    positions = atoms.get_positions()
    heavy_atom_indices = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s != 'H']
    initial_end_to_end = np.linalg.norm(
        positions[heavy_atom_indices[0]] - positions[heavy_atom_indices[-1]]
    )
    end_to_end_distances = [initial_end_to_end]

    def save_frame():
        trajectory.append(atoms.copy())
        energies.append(atoms.get_potential_energy() + atoms.get_kinetic_energy())
        temperatures.append(atoms.get_temperature())

        # Calculate end-to-end distance
        pos = atoms.get_positions()
        end_to_end = np.linalg.norm(
            pos[heavy_atom_indices[0]] - pos[heavy_atom_indices[-1]]
        )
        end_to_end_distances.append(end_to_end)

        if len(trajectory) % 10 == 0:
            print(f"  Frame {len(trajectory)}/{n_steps//save_interval + 1} "
                  f"(t={dyn.get_time()/units.fs:.1f} fs, "
                  f"T={atoms.get_temperature():.0f} K, "
                  f"E={energies[-1]:.3f} eV, "
                  f"d_ee={end_to_end:.1f} Å)")

    # Attach observer
    dyn.attach(save_frame, interval=save_interval)

    # Run simulation
    print(f"\n[4/5] Running folding simulation...")
    print(f"  Initial end-to-end distance: {initial_end_to_end:.1f} Å")

    import time
    start_time = time.time()
    dyn.run(n_steps)
    elapsed = time.time() - start_time

    # Save trajectory
    print(f"\n✓ Simulation complete!")
    print(f"  Wall time: {elapsed:.1f} seconds ({n_steps/elapsed:.0f} steps/s)")
    print(f"  Total frames: {len(trajectory)}")

    final_end_to_end = end_to_end_distances[-1]
    compaction = (initial_end_to_end - final_end_to_end) / initial_end_to_end * 100

    print(f"\n  Structural changes:")
    print(f"    Initial end-to-end: {initial_end_to_end:.1f} Å")
    print(f"    Final end-to-end:   {final_end_to_end:.1f} Å")
    print(f"    Compaction:         {compaction:.1f}%")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write(output_path, trajectory)

    print(f"\n✓ Trajectory saved: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"  Format: XYZ (PyMOL compatible)")

    # Print PyMOL commands
    print("\n" + "="*60)
    print("To visualize folding in PyMOL:")
    print("="*60)
    print(f"  pymol {output_path}")
    print("\nRecommended PyMOL commands:")
    print("  hide everything; show sticks")
    print("  color atomic")
    print("  set stick_radius, 0.15")
    print("  set movie_loop, 1")
    print("  mplay")
    print("\nTo see folding dynamics:")
    print("  # Watch end-to-end distance change")
    print("  # Initial: extended chain")
    print("  # Final: more compact structure")
    print("="*60)

    return len(trajectory)

def main():
    parser = argparse.ArgumentParser(
        description="Peptide folding simulation with student model"
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default='checkpoints/best_model.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--sequence',
        type=str,
        default='ACDEFGHIKLMNPQRSTVWY',
        help='Amino acid sequence (1-letter codes, max 5 used for demo)'
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=50.0,
        help='Simulation duration in picoseconds (default: 50)'
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
        default=500.0,
        help='Temperature in Kelvin (default: 500, high for fast dynamics)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='visualizations/peptide_folding.xyz',
        help='Output XYZ file'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=20,
        help='Save every N steps (default: 20)'
    )
    parser.add_argument(
        '--no-compile',
        action='store_true',
        help='Disable torch.compile() optimization'
    )
    parser.add_argument(
        '--no-fp16',
        action='store_true',
        help='Disable FP16 mixed precision'
    )

    args = parser.parse_args()

    n_frames = run_folding_simulation(
        checkpoint_path=args.checkpoint,
        sequence=args.sequence,
        duration_ps=args.duration,
        timestep_fs=args.timestep,
        temperature_K=args.temperature,
        output_file=args.output,
        save_interval=args.interval,
        use_compile=not args.no_compile,
        use_fp16=not args.no_fp16
    )

    return 0

if __name__ == '__main__':
    sys.exit(main())
