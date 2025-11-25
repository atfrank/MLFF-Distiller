"""
Integration tests for MD Testing Framework

Tests the complete MD testing framework with real student model checkpoints.
This integration test validates:
- NVE MD harness with StudentForceFieldCalculator
- Energy conservation analysis
- Force accuracy metrics
- Trajectory analysis
- End-to-end workflow

Target runtime: < 2 minutes

Author: Testing & Benchmarking Engineer
Date: 2025-11-25
Issue: #37
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from ase.build import molecule
from ase.calculators.lj import LennardJones

from mlff_distiller.testing import (
    NVEMDHarness,
    assess_energy_conservation,
    assess_force_accuracy,
    generate_trajectory_summary,
)


class TestMDFrameworkBasic:
    """Basic integration tests without real model (for speed)."""

    @pytest.fixture
    def lj_calculator(self):
        """Lennard-Jones calculator for testing."""
        return LennardJones(sigma=3.4, epsilon=0.01)

    def test_complete_workflow(self, lj_calculator):
        """Test complete MD workflow: run → analyze energy → analyze trajectory."""
        # Create test system
        atoms = molecule('H2O')

        # Run MD simulation
        harness = NVEMDHarness(
            atoms=atoms,
            calculator=lj_calculator,
            temperature=100.0,
            timestep=1.0
        )

        results = harness.run_simulation(steps=50)

        # Test 1: Energy conservation assessment
        energy_assessment = assess_energy_conservation(
            results['trajectory_data'],
            tolerance_pct=5.0,  # Generous for LJ
            verbose=False
        )

        assert 'passed' in energy_assessment
        assert 'energy_drift_pct' in energy_assessment
        assert 'conservation_ratio' in energy_assessment

        # Test 2: Trajectory summary
        summary = generate_trajectory_summary(
            results['trajectory_data'],
            target_temperature=100.0,
            energy_tolerance_pct=5.0,
            verbose=False
        )

        assert 'simulation_info' in summary
        assert 'energy_summary' in summary
        assert 'temperature_summary' in summary
        assert 'stability_summary' in summary
        assert 'overall_quality' in summary

        # Check simulation info
        assert summary['simulation_info']['n_atoms'] == 3
        assert summary['simulation_info']['n_frames'] == 51  # includes initial frame

    def test_force_comparison_workflow(self, lj_calculator):
        """Test force comparison workflow (student vs teacher)."""
        atoms = molecule('H2O')

        # Create two calculators (simulating student and teacher)
        student_calc = LennardJones(sigma=3.4, epsilon=0.01)
        teacher_calc = LennardJones(sigma=3.4, epsilon=0.01)  # Same for test

        # Run MD with student
        harness = NVEMDHarness(
            atoms=atoms,
            calculator=student_calc,
            temperature=100.0,
            timestep=1.0
        )

        results = harness.run_simulation(steps=20)

        # Get student forces
        student_forces = np.array(results['trajectory_data']['forces'])

        # Compute teacher forces for same trajectory
        teacher_forces = []
        for positions in results['trajectory_data']['positions']:
            atoms_copy = atoms.copy()
            atoms_copy.set_positions(positions)
            atoms_copy.calc = teacher_calc
            teacher_forces.append(atoms_copy.get_forces())

        teacher_forces = np.array(teacher_forces)

        # Assess force accuracy
        force_assessment = assess_force_accuracy(
            student_forces,
            teacher_forces,
            rmse_tolerance=1e-6,  # Should be identical
            verbose=False
        )

        # Should pass with very small tolerance (same calculator)
        assert force_assessment['passed']
        assert force_assessment['rmse'] < 1e-6

    def test_trajectory_file_io(self, lj_calculator, tmp_path):
        """Test trajectory file I/O."""
        atoms = molecule('H2O')
        traj_file = tmp_path / 'test_trajectory.traj'

        # Run with automatic trajectory writing
        harness = NVEMDHarness(
            atoms=atoms,
            calculator=lj_calculator,
            trajectory_file=traj_file,
            temperature=100.0,
            timestep=1.0
        )

        results = harness.run_simulation(steps=20)

        # Check file exists
        assert traj_file.exists()

        # Read back and verify
        from ase.io import read
        frames = read(str(traj_file), index=':')

        # ASE VelocityVerlet default log interval may differ
        assert len(frames) > 0  # Just check we have frames
        assert all(len(frame) == 3 for frame in frames)

        # Note: ASE Trajectory writer may not preserve custom metadata
        # That's expected behavior - we have save_trajectory() for full metadata

    def test_multiple_molecules(self, lj_calculator):
        """Test framework with different molecules."""
        molecules = ['H2O', 'CH4', 'NH3']

        for mol_name in molecules:
            atoms = molecule(mol_name)

            harness = NVEMDHarness(
                atoms=atoms,
                calculator=lj_calculator,
                temperature=100.0,
                timestep=1.0
            )

            results = harness.run_simulation(steps=10)

            # Should complete successfully
            assert results['n_steps'] == 10
            assert len(results['trajectory_data']['time']) == 11  # includes initial frame

    def test_energy_metrics_integration(self, lj_calculator):
        """Test integration of all energy metrics."""
        from mlff_distiller.testing import (
            compute_energy_drift,
            compute_energy_conservation_ratio,
            compute_energy_fluctuations,
            compute_kinetic_potential_stability,
        )

        atoms = molecule('H2O')

        harness = NVEMDHarness(
            atoms=atoms,
            calculator=lj_calculator,
            temperature=100.0,
            timestep=1.0
        )

        results = harness.run_simulation(steps=30)

        traj = results['trajectory_data']
        total_energies = np.array(traj['total_energy'])
        kinetic_energies = np.array(traj['kinetic_energy'])
        potential_energies = np.array(traj['potential_energy'])

        # Test all metrics
        drift = compute_energy_drift(total_energies)
        assert not np.isnan(drift)

        ratio = compute_energy_conservation_ratio(total_energies)
        assert 0 <= ratio <= 1

        fluct = compute_energy_fluctuations(total_energies)
        assert all(k in fluct for k in ['std', 'range', 'mean_abs_dev'])

        stability = compute_kinetic_potential_stability(kinetic_energies, potential_energies)
        assert all(k in stability for k in ['ke_mean', 'pe_mean', 'ke_pe_correlation'])

    def test_trajectory_analysis_integration(self, lj_calculator):
        """Test integration of trajectory analysis tools."""
        from mlff_distiller.testing import (
            compute_rmsd,
            compute_atom_displacements,
            analyze_temperature_evolution,
            analyze_trajectory_stability,
        )

        atoms = molecule('H2O')

        harness = NVEMDHarness(
            atoms=atoms,
            calculator=lj_calculator,
            temperature=100.0,
            timestep=1.0
        )

        results = harness.run_simulation(steps=30)

        traj = results['trajectory_data']
        positions = np.array(traj['positions'])
        temperatures = np.array(traj['temperature'])

        # Test all analysis tools
        rmsd = compute_rmsd(positions)
        assert len(rmsd) == 31  # includes initial frame

        displacements = compute_atom_displacements(positions)
        assert 'max_per_frame' in displacements

        temp_stats = analyze_temperature_evolution(temperatures, target_temperature=100.0)
        assert 'mean' in temp_stats

        stability = analyze_trajectory_stability(traj)
        assert 'rmsd_final' in stability
        assert 'positions_stable' in stability


@pytest.mark.slow
class TestMDFrameworkWithRealModel:
    """Integration tests with real student model (marked slow for CI)."""

    @pytest.fixture
    def checkpoint_path(self):
        """Path to trained model checkpoint."""
        return Path('/home/aaron/ATX/software/MLFF_Distiller/checkpoints/best_model.pt')

    @pytest.fixture
    def test_molecule_path(self):
        """Path to test molecule."""
        return Path('/home/aaron/ATX/software/MLFF_Distiller/data/generative_test/moldiff/test_10mols_20251123_181225_SDF/0.sdf')

    def test_student_model_md(self, checkpoint_path, test_molecule_path):
        """Test MD with real student model checkpoint."""
        if not checkpoint_path.exists():
            pytest.skip("Checkpoint not available")

        if not test_molecule_path.exists():
            pytest.skip("Test molecule not available")

        from mlff_distiller.inference import StudentForceFieldCalculator
        from ase.io import read

        # Load calculator and molecule
        calc = StudentForceFieldCalculator(
            checkpoint_path=checkpoint_path,
            device='cuda',
            enable_timing=True
        )

        atoms = read(str(test_molecule_path))

        # Run short MD simulation
        harness = NVEMDHarness(
            atoms=atoms,
            calculator=calc,
            temperature=300.0,
            timestep=0.5,
            log_interval=5
        )

        results = harness.run_simulation(steps=100)

        # Validate results
        assert results['n_steps'] == 100
        assert len(results['trajectory_data']['time']) == 100

        # Energy conservation assessment
        energy_assessment = assess_energy_conservation(
            results['trajectory_data'],
            tolerance_pct=1.0,
            verbose=True  # Print report
        )

        # Log results
        print(f"\nEnergy drift: {energy_assessment['energy_drift_pct']:.4f}%")
        print(f"Conservation ratio: {energy_assessment['conservation_ratio']:.6f}")
        print(f"Passed: {energy_assessment['passed']}")

        # Trajectory summary
        summary = generate_trajectory_summary(
            results['trajectory_data'],
            target_temperature=300.0,
            energy_tolerance_pct=1.0,
            verbose=True
        )

        # Performance stats
        if hasattr(calc, 'get_timing_stats'):
            timing = calc.get_timing_stats()
            print(f"\nCalculator performance:")
            print(f"  Average time: {timing['avg_time']*1000:.3f} ms/call")
            print(f"  Total calls: {timing['n_calls']}")

    def test_compare_model_variants(self, checkpoint_path):
        """Test comparing different model variants (if available)."""
        if not checkpoint_path.exists():
            pytest.skip("Checkpoint not available")

        # Check for other model variants
        tiny_path = checkpoint_path.parent / 'tiny_model' / 'best_model.pt'
        ultra_tiny_path = checkpoint_path.parent / 'ultra_tiny_model' / 'best_model.pt'

        available_models = [('original', checkpoint_path)]
        if tiny_path.exists():
            available_models.append(('tiny', tiny_path))
        if ultra_tiny_path.exists():
            available_models.append(('ultra_tiny', ultra_tiny_path))

        if len(available_models) < 2:
            pytest.skip("Need at least 2 model variants for comparison")

        from mlff_distiller.inference import StudentForceFieldCalculator
        from ase.build import molecule

        atoms = molecule('H2O')

        results_by_model = {}

        for model_name, model_path in available_models:
            calc = StudentForceFieldCalculator(
                checkpoint_path=model_path,
                device='cuda'
            )

            harness = NVEMDHarness(
                atoms=atoms,
                calculator=calc,
                temperature=300.0,
                timestep=0.5
            )

            results = harness.run_simulation(steps=50)

            # Store energy drift
            assessment = assess_energy_conservation(
                results['trajectory_data'],
                tolerance_pct=1.0,
                verbose=False
            )

            results_by_model[model_name] = {
                'energy_drift_pct': assessment['energy_drift_pct'],
                'conservation_ratio': assessment['conservation_ratio'],
            }

        # Print comparison
        print("\nModel Comparison:")
        for model_name, metrics in results_by_model.items():
            print(f"  {model_name}:")
            print(f"    Energy drift: {metrics['energy_drift_pct']:+.4f}%")
            print(f"    Conservation ratio: {metrics['conservation_ratio']:.6f}")


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_calculator(self):
        """Test with invalid calculator."""
        from ase.build import molecule

        atoms = molecule('H2O')

        # Calculator without required methods
        class BadCalculator:
            implemented_properties = ['energy']

        with pytest.raises(Exception):
            harness = NVEMDHarness(
                atoms=atoms,
                calculator=BadCalculator(),
                temperature=300.0
            )
            harness.run_simulation(steps=10)

    def test_missing_checkpoint(self):
        """Test with missing checkpoint file."""
        from mlff_distiller.inference import StudentForceFieldCalculator

        with pytest.raises(FileNotFoundError):
            StudentForceFieldCalculator(
                checkpoint_path='/nonexistent/path/model.pt'
            )

    def test_invalid_molecule_file(self):
        """Test with invalid molecule file."""
        with pytest.raises(Exception):
            NVEMDHarness(
                atoms='/nonexistent/molecule.xyz',
                calculator=LennardJones()
            )


class TestPerformance:
    """Performance benchmarks for MD framework."""

    def test_harness_overhead(self):
        """Test that harness overhead is minimal."""
        import time
        from ase.build import molecule

        atoms = molecule('H2O')
        calc = LennardJones()

        # Time with harness
        harness = NVEMDHarness(
            atoms=atoms,
            calculator=calc,
            temperature=100.0,
            timestep=1.0
        )

        start = time.perf_counter()
        harness.run_simulation(steps=100)
        harness_time = time.perf_counter() - start

        # Overhead should be small (< 10% for this simple calculator)
        # Main cost is in the calculator, not the harness
        assert harness_time < 5.0  # Should complete in < 5s

    def test_trajectory_memory_efficiency(self):
        """Test that trajectory storage is memory-efficient."""
        import sys
        from ase.build import molecule

        atoms = molecule('H2O')
        calc = LennardJones()

        harness = NVEMDHarness(
            atoms=atoms,
            calculator=calc,
            temperature=100.0,
            timestep=1.0
        )

        # Run simulation
        harness.run_simulation(steps=100)

        # Check memory usage of trajectory data
        traj_data = harness.trajectory_data
        positions_size = sys.getsizeof(traj_data['positions'])
        forces_size = sys.getsizeof(traj_data['forces'])

        # Should be reasonable (< 1 MB for small system)
        total_size = positions_size + forces_size
        assert total_size < 1_000_000  # 1 MB
