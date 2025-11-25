"""
ASE Calculator Usage Examples

This script demonstrates how to use the StudentForceFieldCalculator
for various computational chemistry tasks.

Examples:
1. Basic energy and force calculation
2. Structure optimization
3. MD simulation (NVE ensemble)
4. Batch calculations
5. Comparison with teacher model

Author: ML Architecture Designer
Date: 2025-11-24
Issue: #24
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mlff_distiller.inference import StudentForceFieldCalculator
from ase import Atoms
from ase.build import molecule, bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS
from ase import units
from ase.io import read


def example_1_basic_calculation():
    """Example 1: Basic energy and force calculation."""
    print("\n" + "="*60)
    print("Example 1: Basic Energy and Force Calculation")
    print("="*60)

    # Create calculator
    calc = StudentForceFieldCalculator(
        checkpoint_path='checkpoints/best_model.pt',
        device='cuda',
        enable_timing=True
    )

    # Create a water molecule
    atoms = molecule('H2O')
    atoms.calc = calc

    # Calculate properties
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    print(f"System: H2O molecule")
    print(f"Number of atoms: {len(atoms)}")
    print(f"Energy: {energy:.4f} eV")
    print(f"Forces (eV/Å):")
    for i, (symbol, force) in enumerate(zip(atoms.get_chemical_symbols(), forces)):
        print(f"  {symbol}{i}: [{force[0]:7.4f}, {force[1]:7.4f}, {force[2]:7.4f}]")
    print(f"Max force: {np.max(np.abs(forces)):.4f} eV/Å")

    # Performance stats
    stats = calc.get_timing_stats()
    print(f"\nTiming: {stats['avg_time']*1000:.3f} ms per calculation")


def example_2_structure_optimization():
    """Example 2: Geometry optimization."""
    print("\n" + "="*60)
    print("Example 2: Structure Optimization")
    print("="*60)

    # Create calculator
    calc = StudentForceFieldCalculator(
        checkpoint_path='checkpoints/best_model.pt',
        device='cuda'
    )

    # Create a distorted water molecule
    atoms = molecule('H2O')
    # Slightly distort positions
    atoms.positions += np.random.randn(*atoms.positions.shape) * 0.1
    atoms.calc = calc

    # Get initial energy
    e_initial = atoms.get_potential_energy()
    print(f"Initial energy: {e_initial:.4f} eV")

    # Optimize geometry
    print("Running BFGS optimization...")
    opt = BFGS(atoms, trajectory='opt.traj', logfile='opt.log')
    opt.run(fmax=0.01)  # Converge to max force < 0.01 eV/Å

    # Get final energy
    e_final = atoms.get_potential_energy()
    print(f"Final energy: {e_final:.4f} eV")
    print(f"Energy change: {e_final - e_initial:.4f} eV")
    print(f"Optimization steps: {opt.get_number_of_steps()}")


def example_3_md_simulation():
    """Example 3: MD simulation in NVE ensemble."""
    print("\n" + "="*60)
    print("Example 3: Molecular Dynamics Simulation (NVE)")
    print("="*60)

    # Create calculator
    calc = StudentForceFieldCalculator(
        checkpoint_path='checkpoints/best_model.pt',
        device='cuda',
        enable_timing=True
    )

    # Create molecule
    atoms = molecule('H2O')
    atoms.calc = calc

    # Initialize velocities for 300K
    MaxwellBoltzmannDistribution(atoms, temperature_K=300)
    Stationary(atoms)  # Remove center of mass motion

    # Create MD integrator (0.5 fs timestep)
    dyn = VelocityVerlet(atoms, timestep=0.5*units.fs)

    # Track energy over time
    energies = []
    temperatures = []
    times = []

    def track_energy():
        """Callback to track energy."""
        e_pot = atoms.get_potential_energy()
        e_kin = atoms.get_kinetic_energy()
        e_tot = e_pot + e_kin
        temp = atoms.get_temperature()

        energies.append([e_pot, e_kin, e_tot])
        temperatures.append(temp)
        times.append(dyn.get_time() / units.fs)

    # Attach callback
    dyn.attach(track_energy, interval=10)

    # Run MD for 1000 steps (0.5 ps)
    print("Running MD simulation (1000 steps, 0.5 ps)...")
    dyn.run(1000)

    # Analysis
    energies = np.array(energies)
    temperatures = np.array(temperatures)
    times = np.array(times)

    # Calculate energy drift
    e_total = energies[:, 2]
    drift = (e_total[-1] - e_total[0]) / e_total[0] * 100

    print(f"\nMD Results:")
    print(f"  Total time: {times[-1]:.2f} fs")
    print(f"  Average temperature: {np.mean(temperatures):.2f} K")
    print(f"  Energy drift: {drift:.4f}%")
    print(f"  Avg calculation time: {calc.avg_time*1000:.3f} ms")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Energy plot
    ax1.plot(times, energies[:, 0], label='Potential', alpha=0.7)
    ax1.plot(times, energies[:, 1], label='Kinetic', alpha=0.7)
    ax1.plot(times, energies[:, 2], label='Total', linewidth=2, color='black')
    ax1.set_xlabel('Time (fs)')
    ax1.set_ylabel('Energy (eV)')
    ax1.set_title('NVE Energy Conservation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Temperature plot
    ax2.plot(times, temperatures, alpha=0.7, color='red')
    ax2.axhline(300, color='black', linestyle='--', label='Target (300 K)')
    ax2.set_xlabel('Time (fs)')
    ax2.set_ylabel('Temperature (K)')
    ax2.set_title('Temperature Fluctuations')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('md_nve_example.png', dpi=150)
    print(f"\nPlot saved to: md_nve_example.png")


def example_4_batch_calculation():
    """Example 4: Batch calculations."""
    print("\n" + "="*60)
    print("Example 4: Batch Calculations")
    print("="*60)

    # Create calculator
    calc = StudentForceFieldCalculator(
        checkpoint_path='checkpoints/best_model.pt',
        device='cuda',
        enable_timing=True
    )

    # Create multiple structures
    molecules = [
        molecule('H2O'),
        molecule('CO2'),
        molecule('NH3'),
        molecule('CH4')
    ]

    print(f"Calculating properties for {len(molecules)} molecules...")

    # Batch calculation
    results = calc.calculate_batch(molecules, properties=['energy', 'forces'])

    # Print results
    print("\nResults:")
    for mol, result in zip(molecules, results):
        formula = mol.get_chemical_formula()
        energy = result['energy']
        max_force = np.max(np.abs(result['forces']))
        print(f"  {formula:6s}: E = {energy:8.4f} eV, max|F| = {max_force:.4f} eV/Å")

    # Timing stats
    stats = calc.get_timing_stats()
    print(f"\nTiming statistics:")
    print(f"  Total calls: {stats['n_calls']}")
    print(f"  Average time: {stats['avg_time']*1000:.3f} ms")
    print(f"  Min time: {stats['min_time']*1000:.3f} ms")
    print(f"  Max time: {stats['max_time']*1000:.3f} ms")


def example_5_teacher_comparison():
    """Example 5: Compare student vs teacher model."""
    print("\n" + "="*60)
    print("Example 5: Student vs Teacher Comparison")
    print("="*60)

    try:
        from mlff_distiller.models.teacher_wrappers import OrbCalculatorWrapper

        # Create both calculators
        student_calc = StudentForceFieldCalculator(
            checkpoint_path='checkpoints/best_model.pt',
            device='cuda',
            enable_timing=True
        )

        teacher_calc = OrbCalculatorWrapper(device='cuda')

        # Test molecule
        atoms = molecule('H2O')

        # Student prediction
        atoms.calc = student_calc
        e_student = atoms.get_potential_energy()
        f_student = atoms.get_forces()

        # Teacher prediction
        atoms.calc = teacher_calc
        e_teacher = atoms.get_potential_energy()
        f_teacher = atoms.get_forces()

        # Compare
        e_error = abs(e_student - e_teacher)
        e_error_pct = (e_error / abs(e_teacher)) * 100
        f_rmse = np.sqrt(np.mean((f_student - f_teacher)**2))

        print(f"\nComparison for H2O:")
        print(f"  Student energy: {e_student:.4f} eV")
        print(f"  Teacher energy: {e_teacher:.4f} eV")
        print(f"  Energy error: {e_error:.4f} eV ({e_error_pct:.2f}%)")
        print(f"  Force RMSE: {f_rmse:.4f} eV/Å")

        # Timing comparison
        student_stats = student_calc.get_timing_stats()
        print(f"\nTiming:")
        print(f"  Student: {student_stats['avg_time']*1000:.3f} ms")
        # Note: Teacher timing would need to be measured similarly

    except ImportError as e:
        print(f"Skipping teacher comparison (teacher model not available): {e}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print(" StudentForceFieldCalculator Usage Examples")
    print("="*70)

    # Check if checkpoint exists
    checkpoint_path = Path('checkpoints/best_model.pt')
    if not checkpoint_path.exists():
        print(f"\nERROR: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first or update the checkpoint path.")
        return

    try:
        # Run examples
        example_1_basic_calculation()
        example_2_structure_optimization()
        example_3_md_simulation()
        example_4_batch_calculation()
        example_5_teacher_comparison()

        print("\n" + "="*70)
        print(" All examples completed successfully!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
