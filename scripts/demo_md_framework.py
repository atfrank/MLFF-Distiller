"""
Demo script for MD test framework with real student model.

This demonstrates the complete workflow for Issue #37.

Usage:
    python scripts/demo_md_framework.py
"""

from pathlib import Path
from ase.build import molecule

from mlff_distiller.inference import StudentForceFieldCalculator
from mlff_distiller.testing import (
    NVEMDHarness,
    assess_energy_conservation,
    generate_trajectory_summary,
)


def main():
    # Setup paths
    checkpoint_path = Path('checkpoints/best_model.pt')

    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Please ensure the original 427K model is trained.")
        return

    print("="*70)
    print("MD TEST FRAMEWORK DEMO - Issue #37")
    print("="*70)
    print()

    # Load calculator
    print(f"Loading student model from {checkpoint_path}...")
    calc = StudentForceFieldCalculator(
        checkpoint_path=checkpoint_path,
        device='cuda',
        enable_timing=True
    )
    print(f"Model loaded: {calc.model.num_parameters():,} parameters")
    print()

    # Create test molecule
    atoms = molecule('H2O')
    print(f"Test system: H2O ({len(atoms)} atoms)")
    print()

    # Run MD simulation
    print("Running NVE MD simulation (100 steps, 50 fs)...")
    harness = NVEMDHarness(
        atoms=atoms,
        calculator=calc,
        temperature=300.0,
        timestep=0.5,  # fs
        log_interval=20
    )

    results = harness.run_simulation(steps=100, initialize_velocities=True)

    print(f"Simulation complete: {results['wall_time_s']:.2f}s")
    print()

    # Energy conservation assessment
    print("="*70)
    print("ENERGY CONSERVATION ASSESSMENT")
    print("="*70)
    energy_assessment = assess_energy_conservation(
        results['trajectory_data'],
        tolerance_pct=1.0,
        verbose=True
    )
    print()

    # Trajectory summary
    print("="*70)
    print("TRAJECTORY SUMMARY")
    print("="*70)
    summary = generate_trajectory_summary(
        results['trajectory_data'],
        target_temperature=300.0,
        energy_tolerance_pct=1.0,
        verbose=True
    )
    print()

    # Calculator performance
    if hasattr(calc, 'get_timing_stats'):
        print("="*70)
        print("CALCULATOR PERFORMANCE")
        print("="*70)
        timing = calc.get_timing_stats()
        print(f"Total calls: {timing['n_calls']}")
        print(f"Average time: {timing['avg_time']*1000:.3f} ms/call")
        print(f"Median time: {timing['median_time']*1000:.3f} ms/call")
        print(f"Total time: {timing['total_time']:.3f} s")
        print()

    # Final verdict
    print("="*70)
    print("FINAL VERDICT")
    print("="*70)
    if energy_assessment['passed'] and summary['overall_quality']['passed']:
        print("MD TEST FRAMEWORK: PASSED")
        print("Framework is ready for use in Issues #33, #34, #35, #36")
    else:
        print("MD TEST FRAMEWORK: FAILED")
        if not energy_assessment['passed']:
            print(f"  - Energy drift {energy_assessment['energy_drift_pct']:.3f}% exceeds tolerance")
        if not summary['overall_quality']['passed']:
            print(f"  - Trajectory quality issues detected")
    print("="*70)


if __name__ == '__main__':
    main()
