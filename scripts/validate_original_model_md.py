#!/usr/bin/env python3
"""
Original Student Model (427K) - Production MD Validation
Issue #33: Comprehensive molecular dynamics testing for production approval

This script validates the Original Student Model meets production requirements:
1. 10+ picosecond NVE simulations without crashes
2. Energy drift <1% over trajectory
3. Average force RMSE <0.2 eV/Å during MD
4. No NaN values or numerical instabilities

Author: Testing & Benchmarking Engineer (Agent 5)
Date: 2025-11-25
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns
from ase.io import read, write
from ase import Atoms

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))

from mlff_distiller.inference import StudentForceFieldCalculator
from mlff_distiller.testing import (
    NVEMDHarness,
    assess_energy_conservation,
    assess_force_accuracy,
    generate_trajectory_summary,
    compute_force_rmse,
    compute_force_mae,
    compute_angular_error,
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configuration
CHECKPOINT_PATH = '/home/aaron/ATX/software/MLFF_Distiller/checkpoints/best_model.pt'
TEST_DATA_DIR = '/home/aaron/ATX/software/MLFF_Distiller/data/generative_test/moldiff/test_10mols_20251123_181225_SDF'
OUTPUT_DIR = '/home/aaron/ATX/software/MLFF_Distiller/validation_results/original_model'
TRAJECTORIES_DIR = OUTPUT_DIR + '/trajectories'
PLOTS_DIR = OUTPUT_DIR + '/plots'

# MD Parameters
TEMPERATURE = 300.0  # Kelvin
TIMESTEP = 0.5  # fs
SIMULATION_STEPS = 20000  # 20k steps @ 0.5fs = 10 picoseconds
FORCE_COMPARISON_INTERVAL = 200  # Compare forces every 200 steps

# Success Criteria
ENERGY_DRIFT_THRESHOLD = 1.0  # percent
FORCE_RMSE_THRESHOLD = 0.2  # eV/Å

# Visualization settings
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def setup_directories():
    """Create output directory structure."""
    for directory in [OUTPUT_DIR, TRAJECTORIES_DIR, PLOTS_DIR]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print(f"Output directories created at: {OUTPUT_DIR}")


def load_student_calculator(checkpoint_path: str, device: str = 'cuda') -> StudentForceFieldCalculator:
    """Load the student model calculator."""
    print(f"\nLoading Original Student Model from: {checkpoint_path}")

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    calc = StudentForceFieldCalculator(checkpoint_path, device=device)

    # Get model info
    checkpoint = calc.model.state_dict()
    total_params = sum(p.numel() for p in calc.model.parameters())

    print(f"Model loaded successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Device: {device}")

    return calc


def load_orb_teacher():
    """Load Orb-v2 teacher model for force comparison."""
    print("\nLoading Orb-v2 Teacher model for force comparison...")

    try:
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        orb_model = pretrained.orb_v2()
        teacher_calc = ORBCalculator(orb_model, device='cuda')

        print("Orb-v2 Teacher loaded successfully!")
        return teacher_calc

    except Exception as e:
        print(f"Warning: Could not load Orb teacher: {e}")
        print("Force comparison will be skipped.")
        return None


def select_test_molecules(data_dir: str, num_molecules: int = 5) -> List[str]:
    """Select diverse test molecules from the dataset."""
    sdf_files = sorted(Path(data_dir).glob('*.sdf'))

    if len(sdf_files) < num_molecules:
        print(f"Warning: Only {len(sdf_files)} molecules available, using all")
        selected = sdf_files
    else:
        # Select evenly spaced molecules for diversity
        indices = np.linspace(0, len(sdf_files)-1, num_molecules, dtype=int)
        selected = [sdf_files[i] for i in indices]

    print(f"\nSelected {len(selected)} test molecules:")
    for i, path in enumerate(selected, 1):
        mol = read(str(path))
        print(f"  {i}. {path.name}: {len(mol)} atoms, formula: {mol.get_chemical_formula()}")

    return [str(p) for p in selected]


def run_nve_simulation(atoms: Atoms, calculator: StudentForceFieldCalculator,
                       mol_name: str, steps: int) -> Dict[str, Any]:
    """Run a single NVE MD simulation."""
    print(f"\n{'='*70}")
    print(f"Running NVE MD simulation: {mol_name}")
    print(f"  Atoms: {len(atoms)}, Formula: {atoms.get_chemical_formula()}")
    print(f"  Steps: {steps:,}, Timestep: {TIMESTEP} fs")
    print(f"  Duration: {steps * TIMESTEP / 1000:.1f} picoseconds")
    print(f"{'='*70}")

    # Create harness
    harness = NVEMDHarness(
        atoms=atoms,
        calculator=calculator,
        temperature=TEMPERATURE,
        timestep=TIMESTEP
    )

    # Run simulation
    start_time = time.time()
    results = harness.run_simulation(steps=steps, save_interval=2000)
    elapsed = time.time() - start_time

    print(f"\nSimulation completed in {elapsed:.1f} seconds")
    print(f"Performance: {steps/elapsed:.1f} steps/second")

    # Check for NaN or instabilities
    traj_data = results['trajectory_data']
    total_energies = np.array(traj_data['total_energy'])
    forces = np.array(traj_data['forces'])
    has_nan = (np.isnan(total_energies).any() or
               np.isnan(forces).any())

    if has_nan:
        print("ERROR: NaN values detected in trajectory!")
        results['stability_check'] = 'FAILED - NaN detected'
    else:
        print("Stability check: PASSED (no NaN values)")
        results['stability_check'] = 'PASSED'

    return results


def analyze_energy_conservation(trajectory_data: Dict, mol_name: str) -> Dict[str, Any]:
    """Analyze energy conservation for a trajectory."""
    print(f"\n--- Energy Conservation Analysis: {mol_name} ---")

    assessment = assess_energy_conservation(
        trajectory_data,
        tolerance_pct=ENERGY_DRIFT_THRESHOLD,
        verbose=True
    )

    # Extract key metrics
    metrics = {
        'total_energy_drift_pct': assessment['energy_drift_pct'],
        'max_drift_pct': assessment['energy_drift_max_pct'],
        'energy_conservation_ratio': assessment['conservation_ratio'],
        'kinetic_std': assessment['fluctuation_stats']['std'],
        'potential_std': assessment['ke_pe_stability']['pe_std'],
        'passed': assessment['passed'],
        'threshold': ENERGY_DRIFT_THRESHOLD
    }

    # Print summary
    print(f"\nEnergy Conservation Summary:")
    print(f"  Total energy drift: {metrics['total_energy_drift_pct']:.3f}% (threshold: {ENERGY_DRIFT_THRESHOLD}%)")
    print(f"  Maximum drift: {metrics['max_drift_pct']:.3f}%")
    print(f"  Conservation ratio: {metrics['energy_conservation_ratio']:.6f}")
    print(f"  Status: {'PASSED' if metrics['passed'] else 'FAILED'}")

    return metrics


def compare_forces_during_md(trajectory_data: Dict, original_atoms: Atoms,
                             student_calc: StudentForceFieldCalculator,
                             teacher_calc, mol_name: str) -> Dict[str, Any]:
    """Compare student and teacher forces at multiple trajectory frames."""

    if teacher_calc is None:
        print(f"\nSkipping force comparison for {mol_name} (no teacher available)")
        return {'skipped': True}

    print(f"\n--- Force Comparison vs Teacher: {mol_name} ---")

    positions = trajectory_data['positions']
    student_forces = trajectory_data['forces']

    # Sample frames for comparison
    num_frames = len(positions)
    sample_indices = range(0, num_frames, FORCE_COMPARISON_INTERVAL)

    print(f"Comparing forces at {len(list(sample_indices))} frames (every {FORCE_COMPARISON_INTERVAL} steps)")

    force_rmse_list = []
    force_mae_list = []
    angular_error_list = []

    for idx in sample_indices:
        # Create atoms object at this frame
        atoms_frame = original_atoms.copy()
        atoms_frame.set_positions(positions[idx])

        # Get teacher forces
        atoms_frame.calc = teacher_calc
        try:
            teacher_forces = atoms_frame.get_forces()
        except Exception as e:
            print(f"Warning: Teacher force computation failed at frame {idx}: {e}")
            continue

        # Student forces already computed
        student_forces_frame = student_forces[idx]

        # Compute metrics
        rmse = compute_force_rmse(student_forces_frame, teacher_forces)
        mae = compute_force_mae(student_forces_frame, teacher_forces)
        ang_err = compute_angular_error(student_forces_frame, teacher_forces)

        force_rmse_list.append(rmse)
        force_mae_list.append(mae)
        angular_error_list.append(ang_err)

    if not force_rmse_list:
        print("Error: No valid force comparisons!")
        return {'skipped': True, 'error': 'No valid comparisons'}

    # Compute statistics
    metrics = {
        'avg_rmse': np.mean(force_rmse_list),
        'std_rmse': np.std(force_rmse_list),
        'max_rmse': np.max(force_rmse_list),
        'min_rmse': np.min(force_rmse_list),
        'avg_mae': np.mean(force_mae_list),
        'avg_angular_error': np.mean(angular_error_list),
        'rmse_evolution': force_rmse_list,
        'mae_evolution': force_mae_list,
        'angular_evolution': angular_error_list,
        'sample_frames': list(sample_indices)[:len(force_rmse_list)],
        'passed': np.mean(force_rmse_list) < FORCE_RMSE_THRESHOLD,
        'threshold': FORCE_RMSE_THRESHOLD
    }

    print(f"\nForce Accuracy Summary:")
    print(f"  Average RMSE: {metrics['avg_rmse']:.4f} eV/Å (threshold: {FORCE_RMSE_THRESHOLD} eV/Å)")
    print(f"  RMSE range: [{metrics['min_rmse']:.4f}, {metrics['max_rmse']:.4f}] eV/Å")
    print(f"  Average MAE: {metrics['avg_mae']:.4f} eV/Å")
    print(f"  Average angular error: {metrics['avg_angular_error']:.2f}°")
    print(f"  Status: {'PASSED' if metrics['passed'] else 'FAILED'}")

    return metrics


def plot_energy_evolution(trajectory_data: Dict, mol_name: str, save_path: str):
    """Plot energy components over time."""
    times = np.array(trajectory_data['time'])
    ke = np.array(trajectory_data['kinetic_energy'])
    pe = np.array(trajectory_data['potential_energy'])
    te = np.array(trajectory_data['total_energy'])

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: All energy components
    ax = axes[0]
    ax.plot(times, te, 'k-', linewidth=1.5, label='Total Energy', alpha=0.8)
    ax.plot(times, ke, 'r-', linewidth=1.0, label='Kinetic Energy', alpha=0.7)
    ax.plot(times, pe, 'b-', linewidth=1.0, label='Potential Energy', alpha=0.7)
    ax.set_xlabel('Time (ps)', fontsize=12)
    ax.set_ylabel('Energy (eV)', fontsize=12)
    ax.set_title(f'Energy Evolution - {mol_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: Total energy drift
    ax = axes[1]
    te_drift = (te - te[0]) / np.abs(te[0]) * 100
    ax.plot(times, te_drift, 'k-', linewidth=1.5, alpha=0.8)
    ax.axhline(y=ENERGY_DRIFT_THRESHOLD, color='r', linestyle='--',
               linewidth=2, label=f'Threshold: ±{ENERGY_DRIFT_THRESHOLD}%')
    ax.axhline(y=-ENERGY_DRIFT_THRESHOLD, color='r', linestyle='--', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Time (ps)', fontsize=12)
    ax.set_ylabel('Total Energy Drift (%)', fontsize=12)
    ax.set_title('Energy Conservation', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Energy plot saved: {save_path}")


def plot_force_accuracy(force_metrics: Dict, mol_name: str, save_path: str):
    """Plot force accuracy evolution during MD."""

    if force_metrics.get('skipped'):
        print(f"Skipping force accuracy plot for {mol_name}")
        return

    frames = force_metrics['sample_frames']
    times = np.array(frames) * TIMESTEP / 1000  # Convert to ps

    rmse = force_metrics['rmse_evolution']
    mae = force_metrics['mae_evolution']
    angular = force_metrics['angular_evolution']

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: RMSE and MAE
    ax = axes[0]
    ax.plot(times, rmse, 'o-', linewidth=2, markersize=4, label='RMSE', color='steelblue')
    ax.plot(times, mae, 's-', linewidth=2, markersize=4, label='MAE', color='coral')
    ax.axhline(y=FORCE_RMSE_THRESHOLD, color='r', linestyle='--',
               linewidth=2, label=f'Threshold: {FORCE_RMSE_THRESHOLD} eV/Å')
    ax.axhline(y=force_metrics['avg_rmse'], color='steelblue', linestyle=':',
               linewidth=1.5, alpha=0.7, label=f'Avg RMSE: {force_metrics["avg_rmse"]:.4f} eV/Å')
    ax.set_xlabel('Time (ps)', fontsize=12)
    ax.set_ylabel('Force Error (eV/Å)', fontsize=12)
    ax.set_title(f'Force Accuracy Evolution - {mol_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: Angular error
    ax = axes[1]
    ax.plot(times, angular, 'o-', linewidth=2, markersize=4, color='green')
    ax.axhline(y=force_metrics['avg_angular_error'], color='green', linestyle=':',
               linewidth=1.5, alpha=0.7, label=f'Avg: {force_metrics["avg_angular_error"]:.2f}°')
    ax.set_xlabel('Time (ps)', fontsize=12)
    ax.set_ylabel('Angular Error (degrees)', fontsize=12)
    ax.set_title('Force Direction Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Force accuracy plot saved: {save_path}")


def plot_temperature_evolution(trajectory_data: Dict, mol_name: str, save_path: str):
    """Plot temperature evolution during MD."""
    times = np.array(trajectory_data['time'])
    temps = np.array(trajectory_data['temperature'])

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(times, temps, 'b-', linewidth=1.5, alpha=0.8)
    ax.axhline(y=TEMPERATURE, color='r', linestyle='--', linewidth=2,
               label=f'Target: {TEMPERATURE} K')
    ax.axhline(y=np.mean(temps), color='green', linestyle=':', linewidth=1.5,
               label=f'Average: {np.mean(temps):.1f} K')

    ax.set_xlabel('Time (ps)', fontsize=12)
    ax.set_ylabel('Temperature (K)', fontsize=12)
    ax.set_title(f'Temperature Evolution - {mol_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Temperature plot saved: {save_path}")


def save_trajectory(trajectory_data: Dict, original_atoms: Atoms, save_path: str):
    """Save trajectory to file for inspection."""
    positions = trajectory_data['positions']
    total_energies = trajectory_data['total_energy']
    forces = trajectory_data['forces']

    trajectory = []
    for i, (pos, energy, force) in enumerate(zip(positions, total_energies, forces)):
        atoms = original_atoms.copy()
        atoms.set_positions(pos)
        atoms.info['energy'] = float(energy)
        atoms.arrays['forces'] = force
        trajectory.append(atoms)

    write(save_path, trajectory)
    print(f"Trajectory saved: {save_path} ({len(trajectory)} frames)")


def generate_summary_report(all_results: Dict, save_path: str):
    """Generate comprehensive markdown report."""

    report = []
    report.append("# Original Student Model (427K) - MD Validation Report")
    report.append("")
    report.append("**Issue #33: Production Approval Testing**")
    report.append("")
    report.append(f"**Date**: 2025-11-25")
    report.append(f"**Model**: Original Student (427K parameters)")
    report.append(f"**Checkpoint**: `{CHECKPOINT_PATH}`")
    report.append("")
    report.append("---")
    report.append("")

    # Executive Summary
    report.append("## Executive Summary")
    report.append("")

    # Count passes/fails
    energy_passes = sum(1 for r in all_results.values() if r['energy_metrics']['passed'])
    force_passes = sum(1 for r in all_results.values()
                      if not r['force_metrics'].get('skipped') and r['force_metrics']['passed'])
    stability_passes = sum(1 for r in all_results.values() if r['stability_check'] == 'PASSED')

    total_tests = len(all_results)

    report.append(f"**Test Molecules**: {total_tests}")
    report.append(f"**Simulation Duration**: {SIMULATION_STEPS * TIMESTEP / 1000:.1f} picoseconds per molecule")
    report.append(f"**Total MD Time**: {total_tests * SIMULATION_STEPS * TIMESTEP / 1000:.1f} picoseconds")
    report.append("")

    report.append("### Test Results")
    report.append("")
    report.append(f"- **Stability Tests**: {stability_passes}/{total_tests} PASSED")
    report.append(f"- **Energy Conservation (<{ENERGY_DRIFT_THRESHOLD}% drift)**: {energy_passes}/{total_tests} PASSED")
    report.append(f"- **Force Accuracy (<{FORCE_RMSE_THRESHOLD} eV/Å RMSE)**: {force_passes}/{total_tests} PASSED")
    report.append("")

    # Overall decision
    all_passed = (stability_passes == total_tests and
                  energy_passes == total_tests and
                  force_passes == total_tests)

    if all_passed:
        report.append("### Production Approval Decision")
        report.append("")
        report.append("**STATUS: APPROVED FOR PRODUCTION**")
        report.append("")
        report.append("The Original Student Model (427K parameters) has successfully passed all ")
        report.append("production validation criteria:")
        report.append("")
        report.append("- All simulations completed without crashes or NaN values")
        report.append("- Energy conservation maintained within specifications")
        report.append("- Force accuracy meets production requirements")
        report.append("- Numerical stability confirmed across diverse molecular systems")
        report.append("")
        report.append("**Recommendation**: Deploy to production for molecular dynamics applications.")
    else:
        report.append("### Production Approval Decision")
        report.append("")
        report.append("**STATUS: NOT APPROVED FOR PRODUCTION**")
        report.append("")
        report.append("The model failed one or more validation criteria. Review detailed ")
        report.append("results below and address issues before deployment.")

    report.append("")
    report.append("---")
    report.append("")

    # Detailed Results
    report.append("## Detailed Results")
    report.append("")

    for mol_name, results in all_results.items():
        report.append(f"### {mol_name}")
        report.append("")

        # Basic info
        report.append(f"**Formula**: {results['formula']}")
        report.append(f"**Atoms**: {results['num_atoms']}")
        report.append(f"**Simulation Steps**: {results['simulation_steps']:,}")
        report.append(f"**Duration**: {results['simulation_time_ps']:.2f} ps")
        report.append("")

        # Stability
        report.append(f"**Stability**: {results['stability_check']}")
        report.append("")

        # Energy conservation
        em = results['energy_metrics']
        status = "PASSED" if em['passed'] else "FAILED"
        report.append(f"**Energy Conservation**: {status}")
        report.append(f"- Total energy drift: {em['total_energy_drift_pct']:.3f}%")
        report.append(f"- Maximum drift: {em['max_drift_pct']:.3f}%")
        report.append(f"- Conservation ratio: {em['energy_conservation_ratio']:.6f}")
        report.append(f"- Kinetic energy std: {em['kinetic_std']:.4f} eV")
        report.append(f"- Potential energy std: {em['potential_std']:.4f} eV")
        report.append("")

        # Force accuracy
        fm = results['force_metrics']
        if not fm.get('skipped'):
            status = "PASSED" if fm['passed'] else "FAILED"
            report.append(f"**Force Accuracy vs Teacher**: {status}")
            report.append(f"- Average RMSE: {fm['avg_rmse']:.4f} eV/Å")
            report.append(f"- RMSE range: [{fm['min_rmse']:.4f}, {fm['max_rmse']:.4f}] eV/Å")
            report.append(f"- Average MAE: {fm['avg_mae']:.4f} eV/Å")
            report.append(f"- Average angular error: {fm['avg_angular_error']:.2f}°")
            report.append(f"- Frames compared: {len(fm['rmse_evolution'])}")
        else:
            report.append(f"**Force Accuracy vs Teacher**: SKIPPED")
        report.append("")

        # Visualizations
        report.append("**Visualizations**:")
        report.append(f"- Energy evolution: `plots/{mol_name}_energy.png`")
        report.append(f"- Force accuracy: `plots/{mol_name}_forces.png`")
        report.append(f"- Temperature: `plots/{mol_name}_temperature.png`")
        report.append(f"- Trajectory: `trajectories/{mol_name}.traj`")
        report.append("")
        report.append("---")
        report.append("")

    # Test Configuration
    report.append("## Test Configuration")
    report.append("")
    report.append("```")
    report.append(f"Temperature: {TEMPERATURE} K")
    report.append(f"Timestep: {TIMESTEP} fs")
    report.append(f"Simulation steps: {SIMULATION_STEPS:,}")
    report.append(f"Duration per molecule: {SIMULATION_STEPS * TIMESTEP / 1000:.1f} ps")
    report.append(f"Force comparison interval: {FORCE_COMPARISON_INTERVAL} steps")
    report.append("")
    report.append("Success Criteria:")
    report.append(f"  - Energy drift threshold: <{ENERGY_DRIFT_THRESHOLD}%")
    report.append(f"  - Force RMSE threshold: <{FORCE_RMSE_THRESHOLD} eV/Å")
    report.append(f"  - No NaN or stability issues")
    report.append("```")
    report.append("")

    # Save report
    report_text = "\n".join(report)
    with open(save_path, 'w') as f:
        f.write(report_text)

    print(f"\nValidation report saved: {save_path}")

    return all_passed


def main():
    """Main validation workflow."""
    print("="*80)
    print("ORIGINAL STUDENT MODEL (427K) - PRODUCTION MD VALIDATION")
    print("Issue #33: Comprehensive Testing for Production Approval")
    print("="*80)

    # Setup
    setup_directories()

    # Load models
    student_calc = load_student_calculator(CHECKPOINT_PATH, device='cuda')
    teacher_calc = load_orb_teacher()

    # Select test molecules
    test_molecules = select_test_molecules(TEST_DATA_DIR, num_molecules=5)

    # Run validation on each molecule
    all_results = {}

    for mol_path in test_molecules:
        mol_name = Path(mol_path).stem

        # Load molecule
        atoms = read(mol_path)
        atoms.calc = student_calc

        # Run MD simulation
        sim_results = run_nve_simulation(
            atoms, student_calc, mol_name, SIMULATION_STEPS
        )

        # Analyze energy conservation
        energy_metrics = analyze_energy_conservation(
            sim_results['trajectory_data'], mol_name
        )

        # Compare forces vs teacher
        force_metrics = compare_forces_during_md(
            sim_results['trajectory_data'],
            atoms,
            student_calc,
            teacher_calc,
            mol_name
        )

        # Generate visualizations
        plot_energy_evolution(
            sim_results['trajectory_data'],
            mol_name,
            f"{PLOTS_DIR}/{mol_name}_energy.png"
        )

        plot_force_accuracy(
            force_metrics,
            mol_name,
            f"{PLOTS_DIR}/{mol_name}_forces.png"
        )

        plot_temperature_evolution(
            sim_results['trajectory_data'],
            mol_name,
            f"{PLOTS_DIR}/{mol_name}_temperature.png"
        )

        # Save trajectory
        save_trajectory(
            sim_results['trajectory_data'],
            atoms,
            f"{TRAJECTORIES_DIR}/{mol_name}.traj"
        )

        # Store results
        all_results[mol_name] = {
            'formula': atoms.get_chemical_formula(),
            'num_atoms': len(atoms),
            'simulation_steps': SIMULATION_STEPS,
            'simulation_time_ps': SIMULATION_STEPS * TIMESTEP / 1000,
            'stability_check': sim_results['stability_check'],
            'energy_metrics': energy_metrics,
            'force_metrics': force_metrics,
        }

        print(f"\n{'='*70}")
        print(f"MOLECULE {mol_name} COMPLETE")
        print(f"{'='*70}\n")

    # Generate summary report
    report_path = f"{OUTPUT_DIR}/original_model_md_report.md"
    approved = generate_summary_report(all_results, report_path)

    # Save raw results as JSON
    results_json_path = f"{OUTPUT_DIR}/validation_results.json"
    with open(results_json_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for mol_name, results in all_results.items():
            json_results[mol_name] = {
                'formula': results['formula'],
                'num_atoms': results['num_atoms'],
                'simulation_steps': results['simulation_steps'],
                'simulation_time_ps': results['simulation_time_ps'],
                'stability_check': results['stability_check'],
                'energy_metrics': {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in results['energy_metrics'].items()
                },
                'force_metrics': {
                    k: (float(v) if isinstance(v, (np.floating, np.integer)) else
                        [float(x) for x in v] if isinstance(v, (list, np.ndarray)) else v)
                    for k, v in results['force_metrics'].items()
                } if not results['force_metrics'].get('skipped') else {'skipped': True}
            }

        json.dump(json_results, f, indent=2)

    print(f"\nRaw results saved: {results_json_path}")

    # Final summary
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print(f"\nTest molecules: {len(all_results)}")
    print(f"Total simulation time: {len(all_results) * SIMULATION_STEPS * TIMESTEP / 1000:.1f} ps")
    print(f"\nResults directory: {OUTPUT_DIR}")
    print(f"Validation report: {report_path}")
    print(f"\nProduction Approval: {'APPROVED' if approved else 'NOT APPROVED'}")
    print("="*80)

    return 0 if approved else 1


if __name__ == '__main__':
    sys.exit(main())
