#!/usr/bin/env python3
"""
Tiny Model MD Validation Script (Issue #34)

Validates the Tiny Model (77K parameters) for molecular dynamics stability.
Given the lower force accuracy (R^2 = 0.3787), we expect higher energy drift
but need to characterize the behavior for potential screening applications.

Test Objectives:
1. Run 10ps NVE simulations on 2-3 test molecules
2. Measure energy conservation (expect >1% drift)
3. Document stability/instability behavior
4. Determine use case suitability

Author: Testing & Benchmarking Engineer
Date: 2025-11-25
Issue: #34
"""

import sys
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from ase.io import read

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mlff_distiller.inference import StudentForceFieldCalculator
from mlff_distiller.testing import (
    NVEMDHarness,
    assess_energy_conservation,
    generate_trajectory_summary,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


class TinyModelMDValidator:
    """
    Validates Tiny Model performance in MD simulations.

    Focuses on characterizing behavior rather than strict pass/fail,
    since we expect degraded performance vs original model.
    """

    def __init__(
        self,
        checkpoint_path: str,
        output_dir: str,
        device: str = 'cuda',
        temperature: float = 300.0,
        timestep: float = 0.5,
    ):
        """
        Initialize validator.

        Args:
            checkpoint_path: Path to tiny model checkpoint
            output_dir: Directory for results
            device: Computation device
            temperature: Simulation temperature in K
            timestep: MD timestep in fs
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir)
        self.device = device
        self.temperature = temperature
        self.timestep = timestep

        # Verify checkpoint exists
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'plots').mkdir(exist_ok=True)
        (self.output_dir / 'trajectories').mkdir(exist_ok=True)

        # Results storage
        self.results: Dict[str, Any] = {}

        # File logging
        log_file = self.output_dir / 'validation_log.txt'
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        logger.addHandler(file_handler)

        logger.info("=" * 80)
        logger.info("TINY MODEL MD VALIDATION - Issue #34")
        logger.info("=" * 80)
        logger.info(f"Checkpoint: {self.checkpoint_path}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Temperature: {self.temperature} K")
        logger.info(f"Timestep: {self.timestep} fs")

    def load_calculator(self) -> StudentForceFieldCalculator:
        """Load the student model calculator."""
        logger.info("Loading Tiny Model calculator...")

        calc = StudentForceFieldCalculator(
            checkpoint_path=str(self.checkpoint_path),
            device=self.device,
            enable_timing=True
        )

        # Get model info
        if hasattr(calc.model, 'num_parameters'):
            n_params = calc.model.num_parameters()
            logger.info(f"Model loaded: {n_params:,} parameters")

        return calc

    def run_validation(
        self,
        molecule_paths: List[str],
        simulation_steps: int = 20000,  # 10ps at 0.5fs timestep
        short_simulation_steps: int = 10000,  # 5ps for initial test
        energy_threshold: float = 1.0,  # 1% for original model
        extended_threshold: float = 5.0,  # 5% extended tolerance for tiny model
    ) -> Dict[str, Any]:
        """
        Run MD validation on test molecules.

        Args:
            molecule_paths: List of paths to test molecules
            simulation_steps: Number of MD steps (20000 = 10ps at 0.5fs)
            short_simulation_steps: Steps for initial short test
            energy_threshold: Standard energy drift threshold (%)
            extended_threshold: Extended threshold for tiny model (%)

        Returns:
            Dictionary with validation results
        """
        logger.info(f"Starting validation on {len(molecule_paths)} molecules")
        logger.info(f"Standard threshold: {energy_threshold}%")
        logger.info(f"Extended threshold: {extended_threshold}%")

        # Load calculator
        calc = self.load_calculator()

        # Track results
        all_results = {}
        passed_standard = 0
        passed_extended = 0
        stable_simulations = 0

        for mol_idx, mol_path in enumerate(molecule_paths):
            mol_path = Path(mol_path)
            mol_name = mol_path.stem

            logger.info("=" * 60)
            logger.info(f"MOLECULE: {mol_name}")
            logger.info("=" * 60)

            try:
                # Load molecule
                atoms = read(str(mol_path))
                formula = atoms.get_chemical_formula()
                n_atoms = len(atoms)

                logger.info(f"Formula: {formula}")
                logger.info(f"Atoms: {n_atoms}")

                # First run a short simulation to check stability
                logger.info(f"Running short test ({short_simulation_steps} steps = {short_simulation_steps * self.timestep / 1000:.1f}ps)...")

                short_result = self._run_single_simulation(
                    atoms=atoms.copy(),
                    calc=calc,
                    steps=short_simulation_steps,
                    mol_name=f"{mol_name}_short",
                )

                # Check if short simulation was stable
                if short_result['stability'] == 'CRASHED':
                    logger.warning(f"Short simulation CRASHED - skipping full simulation")
                    all_results[mol_name] = {
                        'formula': formula,
                        'num_atoms': n_atoms,
                        'short_test': short_result,
                        'full_test': None,
                        'stability': 'CRASHED',
                        'use_case': 'UNSUITABLE - crashes in short simulation',
                    }
                    continue

                # If short simulation passed, run full simulation
                logger.info(f"Running full simulation ({simulation_steps} steps = {simulation_steps * self.timestep / 1000:.1f}ps)...")

                full_result = self._run_single_simulation(
                    atoms=atoms.copy(),
                    calc=calc,
                    steps=simulation_steps,
                    mol_name=mol_name,
                    save_trajectory=True,
                    create_plots=True,
                )

                # Analyze results
                energy_drift = abs(full_result['energy_drift_pct'])

                # Determine pass/fail status
                passed_std = energy_drift < energy_threshold
                passed_ext = energy_drift < extended_threshold
                stable = full_result['stability'] == 'STABLE'

                if passed_std:
                    passed_standard += 1
                if passed_ext:
                    passed_extended += 1
                if stable:
                    stable_simulations += 1

                # Determine use case
                if passed_std:
                    use_case = "SUITABLE - production quality"
                elif passed_ext:
                    use_case = "MARGINAL - fast screening only"
                else:
                    use_case = "UNSUITABLE - excessive energy drift"

                all_results[mol_name] = {
                    'formula': formula,
                    'num_atoms': n_atoms,
                    'short_test': short_result,
                    'full_test': full_result,
                    'energy_drift_pct': energy_drift,
                    'passed_standard': passed_std,
                    'passed_extended': passed_ext,
                    'stability': full_result['stability'],
                    'use_case': use_case,
                }

                logger.info(f"Energy drift: {energy_drift:.2f}%")
                logger.info(f"Standard pass (<{energy_threshold}%): {'YES' if passed_std else 'NO'}")
                logger.info(f"Extended pass (<{extended_threshold}%): {'YES' if passed_ext else 'NO'}")
                logger.info(f"Stability: {full_result['stability']}")
                logger.info(f"Use case: {use_case}")

            except Exception as e:
                logger.error(f"Error validating {mol_name}: {e}", exc_info=True)
                all_results[mol_name] = {
                    'error': str(e),
                    'stability': 'ERROR',
                    'use_case': 'UNSUITABLE - validation error',
                }

        # Summary
        n_molecules = len(molecule_paths)

        summary = {
            'timestamp': datetime.now().isoformat(),
            'model': 'Tiny Model (77K params)',
            'checkpoint': str(self.checkpoint_path),
            'n_molecules': n_molecules,
            'passed_standard': passed_standard,
            'passed_extended': passed_extended,
            'stable_simulations': stable_simulations,
            'standard_threshold': energy_threshold,
            'extended_threshold': extended_threshold,
            'simulation_time_ps': simulation_steps * self.timestep / 1000,
            'temperature_K': self.temperature,
            'timestep_fs': self.timestep,
            'results': all_results,
        }

        # Calculate aggregate statistics
        drift_values = [
            r['energy_drift_pct'] for r in all_results.values()
            if 'energy_drift_pct' in r
        ]
        if drift_values:
            summary['mean_energy_drift_pct'] = float(np.mean(drift_values))
            summary['max_energy_drift_pct'] = float(np.max(drift_values))
            summary['min_energy_drift_pct'] = float(np.min(drift_values))
            summary['std_energy_drift_pct'] = float(np.std(drift_values))

        # Overall recommendation
        if passed_standard == n_molecules:
            summary['recommendation'] = 'PRODUCTION_READY'
            summary['recommendation_detail'] = 'All molecules passed standard criteria'
        elif passed_extended == n_molecules:
            summary['recommendation'] = 'SCREENING_ONLY'
            summary['recommendation_detail'] = f'All molecules pass extended criteria ({extended_threshold}%), but not standard ({energy_threshold}%)'
        elif stable_simulations == n_molecules:
            summary['recommendation'] = 'LIMITED_USE'
            summary['recommendation_detail'] = 'Simulations stable but energy drift exceeds thresholds'
        else:
            summary['recommendation'] = 'NOT_RECOMMENDED'
            summary['recommendation_detail'] = 'Some simulations crashed or showed instabilities'

        self.results = summary
        return summary

    def _run_single_simulation(
        self,
        atoms,
        calc: StudentForceFieldCalculator,
        steps: int,
        mol_name: str,
        save_trajectory: bool = False,
        create_plots: bool = False,
    ) -> Dict[str, Any]:
        """
        Run a single MD simulation.

        Args:
            atoms: ASE Atoms object
            calc: StudentForceFieldCalculator
            steps: Number of MD steps
            mol_name: Name for output files
            save_trajectory: Whether to save trajectory
            create_plots: Whether to create plots

        Returns:
            Dictionary with simulation results
        """
        # Setup trajectory file if saving
        traj_file = None
        if save_trajectory:
            traj_file = self.output_dir / 'trajectories' / f'{mol_name}.traj'

        # Create harness
        harness = NVEMDHarness(
            atoms=atoms,
            calculator=calc,
            temperature=self.temperature,
            timestep=self.timestep,
            trajectory_file=traj_file,
            log_interval=max(steps // 100, 10),
        )

        # Run simulation
        start_time = time.perf_counter()

        try:
            results = harness.run_simulation(steps=steps)
            wall_time = time.perf_counter() - start_time

            # Assess energy conservation
            assessment = assess_energy_conservation(
                results['trajectory_data'],
                tolerance_pct=1.0,
                verbose=False
            )

            # Generate trajectory summary
            summary = generate_trajectory_summary(
                results['trajectory_data'],
                target_temperature=self.temperature,
                energy_tolerance_pct=1.0,
                verbose=False
            )

            # Check for instabilities
            total_energies = np.array(results['trajectory_data']['total_energy'])

            # Check for NaN/Inf
            has_nan = np.any(np.isnan(total_energies)) or np.any(np.isinf(total_energies))

            # Check for explosive drift (>50% change)
            max_drift = abs(assessment['energy_drift_max_pct'])
            explosive = max_drift > 50.0

            # Determine stability
            if has_nan:
                stability = 'CRASHED'
            elif explosive:
                stability = 'UNSTABLE'
            else:
                stability = 'STABLE'

            result = {
                'steps': steps,
                'simulation_time_ps': steps * self.timestep / 1000,
                'wall_time_s': wall_time,
                'steps_per_second': steps / wall_time,
                'energy_drift_pct': assessment['energy_drift_pct'],
                'energy_drift_max_pct': assessment['energy_drift_max_pct'],
                'conservation_ratio': assessment['conservation_ratio'],
                'avg_temperature': results['avg_temperature'],
                'std_temperature': results['std_temperature'],
                'initial_energy': results['initial_energy'],
                'final_energy': results['final_energy'],
                'stability': stability,
                'passed': assessment['passed'],
            }

            # Create plots if requested
            if create_plots:
                self._create_plots(
                    results['trajectory_data'],
                    mol_name,
                    assessment
                )

            return result

        except Exception as e:
            logger.error(f"Simulation crashed: {e}")
            return {
                'steps': steps,
                'simulation_time_ps': steps * self.timestep / 1000,
                'wall_time_s': time.perf_counter() - start_time,
                'stability': 'CRASHED',
                'error': str(e),
            }

    def _create_plots(
        self,
        trajectory_data: Dict,
        mol_name: str,
        assessment: Dict
    ):
        """Create diagnostic plots for the simulation."""
        times = np.array(trajectory_data['time'])
        total_energies = np.array(trajectory_data['total_energy'])
        kinetic_energies = np.array(trajectory_data['kinetic_energy'])
        potential_energies = np.array(trajectory_data['potential_energy'])
        temperatures = np.array(trajectory_data['temperature'])

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Total Energy
        ax1 = axes[0, 0]
        ax1.plot(times, total_energies, 'b-', alpha=0.7, linewidth=0.5)
        ax1.set_xlabel('Time (ps)')
        ax1.set_ylabel('Total Energy (eV)')
        ax1.set_title(f'Total Energy Evolution\nDrift: {assessment["energy_drift_pct"]:.3f}%')
        ax1.axhline(total_energies[0], color='r', linestyle='--', alpha=0.5, label='Initial')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # KE and PE
        ax2 = axes[0, 1]
        ax2.plot(times, kinetic_energies, 'r-', alpha=0.7, linewidth=0.5, label='KE')
        ax2.plot(times, potential_energies, 'b-', alpha=0.7, linewidth=0.5, label='PE')
        ax2.set_xlabel('Time (ps)')
        ax2.set_ylabel('Energy (eV)')
        ax2.set_title('Kinetic and Potential Energy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Temperature
        ax3 = axes[1, 0]
        ax3.plot(times, temperatures, 'g-', alpha=0.7, linewidth=0.5)
        ax3.axhline(self.temperature, color='r', linestyle='--', alpha=0.5, label=f'Target ({self.temperature}K)')
        ax3.axhline(np.mean(temperatures), color='b', linestyle='--', alpha=0.5, label=f'Mean ({np.mean(temperatures):.1f}K)')
        ax3.set_xlabel('Time (ps)')
        ax3.set_ylabel('Temperature (K)')
        ax3.set_title(f'Temperature Evolution\nStd: {np.std(temperatures):.1f}K')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Energy drift over time
        ax4 = axes[1, 1]
        e0 = total_energies[0]
        drift_pct = 100.0 * (total_energies - e0) / abs(e0)
        ax4.plot(times, drift_pct, 'purple', alpha=0.7, linewidth=0.5)
        ax4.axhline(0, color='k', linestyle='-', alpha=0.3)
        ax4.axhline(1.0, color='g', linestyle='--', alpha=0.5, label='1% threshold')
        ax4.axhline(-1.0, color='g', linestyle='--', alpha=0.5)
        ax4.axhline(5.0, color='orange', linestyle='--', alpha=0.5, label='5% extended')
        ax4.axhline(-5.0, color='orange', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Time (ps)')
        ax4.set_ylabel('Energy Drift (%)')
        ax4.set_title('Cumulative Energy Drift')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.suptitle(f'Tiny Model MD Validation: {mol_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        plot_path = self.output_dir / 'plots' / f'{mol_name}_validation.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved plot: {plot_path}")

    def save_results(self):
        """Save validation results to files."""
        # Save JSON results
        json_path = self.output_dir / 'validation_results.json'
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Saved results: {json_path}")

        # Generate markdown report
        self._generate_report()

    def _generate_report(self):
        """Generate markdown validation report."""
        r = self.results

        report = f"""# Tiny Model (77K) MD Validation Report

**Issue #34: Tiny Model MD Validation**

**Date**: {r.get('timestamp', 'N/A')}
**Model**: {r.get('model', 'Tiny Model')}
**Checkpoint**: `{r.get('checkpoint', 'N/A')}`

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Test Molecules | {r.get('n_molecules', 0)} |
| Simulation Duration | {r.get('simulation_time_ps', 0):.1f} ps |
| Temperature | {r.get('temperature_K', 300)} K |
| Timestep | {r.get('timestep_fs', 0.5)} fs |

### Pass Rates

| Criterion | Passed | Total | Rate |
|-----------|--------|-------|------|
| Standard (<{r.get('standard_threshold', 1.0)}% drift) | {r.get('passed_standard', 0)} | {r.get('n_molecules', 0)} | {100 * r.get('passed_standard', 0) / max(r.get('n_molecules', 1), 1):.0f}% |
| Extended (<{r.get('extended_threshold', 5.0)}% drift) | {r.get('passed_extended', 0)} | {r.get('n_molecules', 0)} | {100 * r.get('passed_extended', 0) / max(r.get('n_molecules', 1), 1):.0f}% |
| Stable Simulations | {r.get('stable_simulations', 0)} | {r.get('n_molecules', 0)} | {100 * r.get('stable_simulations', 0) / max(r.get('n_molecules', 1), 1):.0f}% |

### Energy Drift Statistics

| Metric | Value |
|--------|-------|
| Mean Drift | {r.get('mean_energy_drift_pct', 0):.2f}% |
| Max Drift | {r.get('max_energy_drift_pct', 0):.2f}% |
| Min Drift | {r.get('min_energy_drift_pct', 0):.2f}% |
| Std Drift | {r.get('std_energy_drift_pct', 0):.2f}% |

---

## Overall Recommendation

**STATUS: {r.get('recommendation', 'UNKNOWN')}**

{r.get('recommendation_detail', 'No recommendation available.')}

---

## Detailed Results by Molecule

"""
        # Add per-molecule results
        for mol_name, mol_result in r.get('results', {}).items():
            if 'error' in mol_result and mol_result.get('stability') == 'ERROR':
                report += f"""### {mol_name}

**Status**: ERROR
**Error**: {mol_result.get('error', 'Unknown error')}

---

"""
            elif 'full_test' in mol_result and mol_result['full_test']:
                ft = mol_result['full_test']
                report += f"""### {mol_name}

**Formula**: {mol_result.get('formula', 'N/A')}
**Atoms**: {mol_result.get('num_atoms', 0)}
**Stability**: {mol_result.get('stability', 'UNKNOWN')}
**Use Case**: {mol_result.get('use_case', 'N/A')}

| Metric | Value |
|--------|-------|
| Energy Drift | {mol_result.get('energy_drift_pct', 0):.3f}% |
| Conservation Ratio | {ft.get('conservation_ratio', 0):.6f} |
| Average Temperature | {ft.get('avg_temperature', 0):.1f} K |
| Temperature Std | {ft.get('std_temperature', 0):.1f} K |
| Wall Time | {ft.get('wall_time_s', 0):.1f} s |
| Performance | {ft.get('steps_per_second', 0):.0f} steps/s |

**Pass Standard (<{r.get('standard_threshold', 1.0)}%)**: {'YES' if mol_result.get('passed_standard', False) else 'NO'}
**Pass Extended (<{r.get('extended_threshold', 5.0)}%)**: {'YES' if mol_result.get('passed_extended', False) else 'NO'}

---

"""
            else:
                report += f"""### {mol_name}

**Status**: {mol_result.get('stability', 'UNKNOWN')}
**Use Case**: {mol_result.get('use_case', 'N/A')}

---

"""

        # Add use case recommendations
        report += """## Use Case Recommendations

Based on the validation results, the Tiny Model (77K parameters) is recommended for:

"""

        rec = r.get('recommendation', 'UNKNOWN')
        if rec == 'PRODUCTION_READY':
            report += """**Production MD Simulations**: The model maintains excellent energy conservation
and can be used for production molecular dynamics simulations.

Recommended applications:
- Long timescale MD simulations
- Free energy calculations
- Property predictions requiring accurate dynamics
"""
        elif rec == 'SCREENING_ONLY':
            report += """**Fast Screening Applications**: The model shows acceptable stability but elevated
energy drift. Suitable for rapid screening applications where speed is prioritized
over long-term accuracy.

Recommended applications:
- Initial structure screening
- Rough energy ranking
- Quick conformational searches
- Pre-filtering before higher-accuracy calculations

NOT recommended for:
- Long timescale dynamics
- Quantitative free energy calculations
- Property predictions requiring strict energy conservation
"""
        elif rec == 'LIMITED_USE':
            report += """**Very Limited Use**: Simulations are stable but energy drift exceeds acceptable
thresholds. Use only when 5.5x model compression is critical.

Potentially acceptable for:
- Very short simulations (<1ps)
- Qualitative structure analysis
- Proof-of-concept testing

NOT recommended for:
- Any quantitative analysis
- Simulations longer than 1ps
"""
        else:
            report += """**NOT RECOMMENDED**: The model shows instabilities or excessive energy drift
that make it unsuitable for any production use.

Consider:
- Using the larger student model (427K parameters)
- Retraining with different hyperparameters
- Using for non-dynamics applications only (e.g., single-point energies)
"""

        report += f"""
---

## Comparison with Original Model (427K)

The original student model (Issue #33) achieved:
- Energy drift: 0.02% - 0.40% (all passed <1%)
- 100% pass rate on standard criteria

The Tiny Model (77K parameters, 5.5x compression) shows:
- Mean energy drift: {r.get('mean_energy_drift_pct', 0):.2f}%
- Pass rate (standard): {100 * r.get('passed_standard', 0) / max(r.get('n_molecules', 1), 1):.0f}%
- Pass rate (extended): {100 * r.get('passed_extended', 0) / max(r.get('n_molecules', 1), 1):.0f}%

This degradation is expected given the force R^2 = 0.3787 (vs ~0.9+ for original).

---

*Report generated by Agent 5 (Testing & Benchmarking Engineer)*
*Issue #34: Tiny Model MD Validation*
"""

        report_path = self.output_dir / 'tiny_model_md_report.md'
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Saved report: {report_path}")


def main():
    """Main entry point."""
    # Paths
    project_root = Path(__file__).parent.parent
    checkpoint_path = project_root / 'checkpoints' / 'tiny_model' / 'best_model_fixed.pt'
    output_dir = project_root / 'validation_results' / 'tiny_model'
    mol_dir = project_root / 'data' / 'generative_test' / 'moldiff' / 'test_10mols_20251123_181225_SDF'

    # Select test molecules (use subset for faster validation)
    # Using molecules 0, 2, 4 which were tested with original model
    test_molecules = [
        mol_dir / '0.sdf',
        mol_dir / '2.sdf',
        mol_dir / '4.sdf',
    ]

    # Verify all molecules exist
    for mol_path in test_molecules:
        if not mol_path.exists():
            logger.error(f"Test molecule not found: {mol_path}")
            sys.exit(1)

    logger.info(f"Validating on {len(test_molecules)} molecules")

    # Create validator
    validator = TinyModelMDValidator(
        checkpoint_path=str(checkpoint_path),
        output_dir=str(output_dir),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        temperature=300.0,
        timestep=0.5,
    )

    # Run validation
    # Using 20000 steps = 10ps at 0.5fs timestep
    # Matching original model validation parameters
    results = validator.run_validation(
        molecule_paths=[str(p) for p in test_molecules],
        simulation_steps=20000,
        short_simulation_steps=2000,  # 1ps initial stability check
        energy_threshold=1.0,
        extended_threshold=5.0,
    )

    # Save results
    validator.save_results()

    # Print summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Recommendation: {results['recommendation']}")
    logger.info(f"Detail: {results['recommendation_detail']}")
    logger.info(f"Mean energy drift: {results.get('mean_energy_drift_pct', 0):.2f}%")
    logger.info(f"Passed standard (<1%): {results['passed_standard']}/{results['n_molecules']}")
    logger.info(f"Passed extended (<5%): {results['passed_extended']}/{results['n_molecules']}")
    logger.info(f"Results saved to: {output_dir}")

    return results


if __name__ == '__main__':
    main()
