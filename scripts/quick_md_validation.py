#!/usr/bin/env python3
"""
Quick MD Validation - 100ps NVE trajectory

Verifies that the student model is stable for MD simulations
before proceeding with heavy optimization work.

Usage:
    python scripts/quick_md_validation.py \
        --checkpoint checkpoints/best_model.pt \
        --timestep 0.5 \
        --duration 100 \
        --output validation_results/quick_nve
"""

import sys
from pathlib import Path
import numpy as np
from ase.build import molecule
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
import matplotlib.pyplot as plt

# Add src to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from mlff_distiller.inference import StudentForceFieldCalculator

def run_nve_validation(
    checkpoint_path,
    timestep_fs=0.5,
    duration_ps=100,
    temperature_K=300,
    output_dir='validation_results/quick_nve'
):
    """Run quick NVE validation trajectory."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Quick MD Validation - 100ps NVE")
    print("="*60)

    # Create test system
    print("\n[1/5] Creating test system (benzene)...")
    atoms = molecule('C6H6')
    print(f"  Atoms: {len(atoms)}")
    print(f"  Formula: {atoms.get_chemical_formula()}")

    # Set up calculator
    print("\n[2/5] Loading student model...")
    calc = StudentForceFieldCalculator(
        checkpoint_path=checkpoint_path,
        device='cuda',
        use_compile=False,  # No optimizations for validation
        use_fp16=False
    )
    atoms.calc = calc
    print("  ✓ Calculator ready")

    # Initialize velocities
    print(f"\n[3/5] Initializing velocities (T={temperature_K}K)...")
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
    print(f"  Initial temperature: {atoms.get_temperature():.1f} K")

    # Set up NVE dynamics
    print(f"\n[4/5] Setting up NVE dynamics...")
    print(f"  Timestep: {timestep_fs} fs")
    print(f"  Duration: {duration_ps} ps")
    n_steps = int(duration_ps * 1000 / timestep_fs)
    print(f"  Total steps: {n_steps:,}")

    dyn = VelocityVerlet(atoms, timestep=timestep_fs * units.fs)

    # Storage for analysis
    energies = []
    temperatures = []
    times = []

    def record_state():
        energies.append(atoms.get_potential_energy() + atoms.get_kinetic_energy())
        temperatures.append(atoms.get_temperature())
        times.append(dyn.get_time() / units.fs)

        # Progress update
        if len(times) % 1000 == 0:
            progress = len(times) / (n_steps / 10) * 100
            print(f"  Progress: {progress:.1f}% ({len(times)*10:,}/{n_steps:,} steps)", end='\r')

    # Attach observer (record every 10 steps)
    dyn.attach(record_state, interval=10)

    # Run trajectory
    print("\n[5/5] Running MD trajectory...")
    dyn.run(n_steps)
    print("\n  ✓ Trajectory complete")

    # Convert to arrays
    energies = np.array(energies)
    temperatures = np.array(temperatures)
    times = np.array(times)

    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)

    energy_drift = (energies[-1] - energies[0]) / energies[0] * 100
    energy_std = energies.std() / energies.mean() * 100
    temp_mean = temperatures.mean()
    temp_std = temperatures.std()

    print(f"\nEnergy Conservation:")
    print(f"  Initial energy: {energies[0]:.6f} eV")
    print(f"  Final energy:   {energies[-1]:.6f} eV")
    print(f"  Drift:          {energy_drift:+.3f}%")
    print(f"  Fluctuations:   {energy_std:.3f}%")

    print(f"\nTemperature Stability:")
    print(f"  Target:         {temperature_K:.1f} K")
    print(f"  Mean:           {temp_mean:.1f} K")
    print(f"  Std:            {temp_std:.1f} K")
    print(f"  Deviation:      {abs(temp_mean - temperature_K):.1f} K")

    # Pass/fail criteria
    energy_drift_pass = abs(energy_drift) < 1.0
    energy_std_pass = energy_std < 0.5
    temp_pass = abs(temp_mean - temperature_K) < 20

    passed = energy_drift_pass and energy_std_pass and temp_pass

    print(f"\n" + "="*60)
    print("RESULT: {'PASS ✓' if passed else 'FAIL ✗'}")
    print("="*60)

    print(f"\nCriteria:")
    print(f"  Energy drift <1%:           {'✓ PASS' if energy_drift_pass else '✗ FAIL'}")
    print(f"  Energy fluctuations <0.5%:  {'✓ PASS' if energy_std_pass else '✗ FAIL'}")
    print(f"  Temperature within 20K:     {'✓ PASS' if temp_pass else '✗ FAIL'}")

    # Generate plots
    print(f"\nGenerating plots...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Energy conservation
    ax1.plot(times / 1000, energies - energies[0], 'b-', linewidth=1)
    ax1.set_xlabel('Time (ps)')
    ax1.set_ylabel('Energy Drift (eV)')
    ax1.set_title(f'Energy Conservation (Drift: {energy_drift:+.3f}%)')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='r', linestyle='--', alpha=0.5)

    # Temperature
    ax2.plot(times / 1000, temperatures, 'r-', linewidth=1)
    ax2.axhline(temperature_K, color='k', linestyle='--', alpha=0.5, label='Target')
    ax2.set_xlabel('Time (ps)')
    ax2.set_ylabel('Temperature (K)')
    ax2.set_title(f'Temperature Stability (Mean: {temp_mean:.1f}±{temp_std:.1f}K)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plot_path = output_dir / 'quick_md_validation.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  ✓ Plot saved: {plot_path}")

    # Write report
    report = f"""
Quick MD Validation Results
===========================

System: Benzene (C6H6, 12 atoms)
Duration: {duration_ps} ps
Timestep: {timestep_fs} fs
Temperature: {temperature_K} K

Energy Conservation:
  Initial energy: {energies[0]:.6f} eV
  Final energy:   {energies[-1]:.6f} eV
  Drift:          {energy_drift:+.3f}%
  Fluctuations:   {energy_std:.3f}%

Temperature:
  Target:         {temperature_K:.1f} K
  Mean:           {temp_mean:.1f} K
  Std:            {temp_std:.1f} K
  Deviation:      {abs(temp_mean - temperature_K):.1f} K

Result: {'PASS ✓' if passed else 'FAIL ✗'}

Criteria:
  Energy drift <1%:           {'✓ PASS' if energy_drift_pass else '✗ FAIL'}
  Energy fluctuations <0.5%:  {'✓ PASS' if energy_std_pass else '✗ FAIL'}
  Temperature within 20K:     {'✓ PASS' if temp_pass else '✗ FAIL'}

{'Decision: Proceed with Phase 1 optimizations' if passed else 'Decision: Model requires fixes before optimization'}
"""

    report_path = output_dir / 'validation_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  ✓ Report saved: {report_path}")

    # Save raw data
    np.savez(
        output_dir / 'trajectory_data.npz',
        times=times,
        energies=energies,
        temperatures=temperatures
    )
    print(f"  ✓ Data saved: {output_dir / 'trajectory_data.npz'}")

    return passed

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Quick MD validation for student model')
    parser.add_argument('--checkpoint', type=Path, default='checkpoints/best_model.pt',
                        help='Path to student model checkpoint')
    parser.add_argument('--timestep', type=float, default=0.5,
                        help='MD timestep in femtoseconds (default: 0.5)')
    parser.add_argument('--duration', type=float, default=100,
                        help='Trajectory duration in picoseconds (default: 100)')
    parser.add_argument('--temperature', type=float, default=300,
                        help='Initial temperature in Kelvin (default: 300)')
    parser.add_argument('--output', type=Path, default='validation_results/quick_nve',
                        help='Output directory (default: validation_results/quick_nve)')
    args = parser.parse_args()

    passed = run_nve_validation(
        args.checkpoint,
        args.timestep,
        args.duration,
        args.temperature,
        args.output
    )

    sys.exit(0 if passed else 1)
