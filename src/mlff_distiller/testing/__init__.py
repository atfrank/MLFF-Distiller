"""
MD Testing & Validation Framework

This package provides comprehensive tools for validating molecular dynamics
simulations using student force field models. It includes NVE MD harness,
energy conservation metrics, force accuracy metrics, and trajectory analysis.

Main Components:
    - NVEMDHarness: Run NVE (microcanonical) MD simulations
    - Energy metrics: Analyze energy conservation
    - Force metrics: Validate force accuracy vs teacher
    - Trajectory analysis: Structural stability and dynamics

Quick Start:
    from mlff_distiller.testing import NVEMDHarness, assess_energy_conservation
    from mlff_distiller.inference import StudentForceFieldCalculator
    from ase.build import molecule

    # Setup
    calc = StudentForceFieldCalculator('checkpoints/best_model.pt')
    atoms = molecule('H2O')

    # Run MD
    harness = NVEMDHarness(atoms, calc, temperature=300.0, timestep=0.5)
    results = harness.run_simulation(steps=1000)

    # Analyze
    assessment = assess_energy_conservation(
        results['trajectory_data'],
        tolerance_pct=1.0,
        verbose=True
    )

    if assessment['passed']:
        print("Energy conservation PASSED")

Author: Testing & Benchmarking Engineer
Date: 2025-11-25
Issue: #37
"""

# NVE MD Harness
from .nve_harness import NVEMDHarness

# Energy Conservation Metrics
from .energy_metrics import (
    compute_energy_drift,
    compute_energy_conservation_ratio,
    compute_energy_fluctuations,
    compute_kinetic_potential_stability,
    compute_time_resolved_drift,
    assess_energy_conservation,
)

# Force Accuracy Metrics
from .force_metrics import (
    compute_force_rmse,
    compute_force_mae,
    compute_force_magnitude_error,
    compute_angular_error,
    compute_per_atom_force_errors,
    compute_force_correlation,
    assess_force_accuracy,
)

# Trajectory Analysis
from .trajectory_analysis import (
    compute_rmsd,
    kabsch_align,
    compute_atom_displacements,
    analyze_temperature_evolution,
    compute_bond_lengths,
    analyze_trajectory_stability,
    generate_trajectory_summary,
)

__all__ = [
    # NVE MD Harness
    'NVEMDHarness',

    # Energy Metrics
    'compute_energy_drift',
    'compute_energy_conservation_ratio',
    'compute_energy_fluctuations',
    'compute_kinetic_potential_stability',
    'compute_time_resolved_drift',
    'assess_energy_conservation',

    # Force Metrics
    'compute_force_rmse',
    'compute_force_mae',
    'compute_force_magnitude_error',
    'compute_angular_error',
    'compute_per_atom_force_errors',
    'compute_force_correlation',
    'assess_force_accuracy',

    # Trajectory Analysis
    'compute_rmsd',
    'kabsch_align',
    'compute_atom_displacements',
    'analyze_temperature_evolution',
    'compute_bond_lengths',
    'analyze_trajectory_stability',
    'generate_trajectory_summary',
]

__version__ = '1.0.0'
