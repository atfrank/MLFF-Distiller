#!/usr/bin/env python3
"""
Validate Analytical Force Computation

Compare analytical forces vs autograd to ensure correctness.
Maximum acceptable error: 1e-4 eV/Å (typical MD tolerance)

Usage:
    python scripts/validate_analytical_forces.py --device cuda
    python scripts/validate_analytical_forces.py --verbose
"""

import sys
import argparse
from pathlib import Path
import logging

import numpy as np
import torch
from ase.build import molecule, bulk
from ase import Atoms

# Add src to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from mlff_distiller.models.student_model import StudentForceField

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_cases():
    """
    Create comprehensive test cases for validation.

    Returns:
        List of (name, atoms) tuples
    """
    test_cases = []

    # 1. Diatomic molecule (simplest case)
    h2 = Atoms('H2', positions=[[0, 0, 0], [0.74, 0, 0]])
    test_cases.append(('H2', h2))

    # 2. Water (bent geometry, different species)
    water = molecule('H2O')
    test_cases.append(('H2O', water))

    # 3. Methane (tetrahedral)
    methane = molecule('CH4')
    test_cases.append(('CH4', methane))

    # 4. Ammonia (pyramidal)
    ammonia = molecule('NH3')
    test_cases.append(('NH3', ammonia))

    # 5. Ethane (larger molecule)
    ethane = molecule('C2H6')
    test_cases.append(('C2H6', ethane))

    # 6. Benzene (planar, aromatic)
    benzene = molecule('C6H6')
    test_cases.append(('C6H6', benzene))

    # 7. Random perturbation of water (test different geometry)
    water_perturbed = molecule('H2O')
    water_perturbed.positions += np.random.randn(3, 3) * 0.1
    test_cases.append(('H2O_perturbed', water_perturbed))

    # 8. Linear molecule
    co2 = molecule('CO2')
    test_cases.append(('CO2', co2))

    logger.info(f"Created {len(test_cases)} test cases")
    return test_cases


def validate_single_molecule(
    model: StudentForceField,
    name: str,
    atoms: Atoms,
    tolerance: float = 1e-4,
    verbose: bool = False
) -> dict:
    """
    Validate analytical forces against autograd for a single molecule.

    Args:
        model: StudentForceField model
        name: Molecule name
        atoms: ASE Atoms object
        tolerance: Maximum allowed force error (eV/Å)
        verbose: Print detailed comparison

    Returns:
        dict with validation results
    """
    device = model.embedding.weight.device

    # Prepare tensors
    atomic_numbers = torch.tensor(
        atoms.get_atomic_numbers(),
        dtype=torch.long,
        device=device
    )
    positions = torch.tensor(
        atoms.get_positions(),
        dtype=torch.float32,
        device=device
    )

    # 1. Compute forces with autograd (ground truth)
    energy_auto, forces_auto = model.predict_energy_and_forces(
        atomic_numbers, positions
    )

    # 2. Compute forces analytically
    energy_analytical, forces_analytical = model.forward_with_analytical_forces(
        atomic_numbers, positions
    )

    # 3. Compare
    energy_auto_val = energy_auto.item()
    energy_analytical_val = energy_analytical.item()
    energy_error = abs(energy_auto_val - energy_analytical_val)

    forces_auto_np = forces_auto.detach().cpu().numpy()
    forces_analytical_np = forces_analytical.detach().cpu().numpy()

    force_errors = np.abs(forces_auto_np - forces_analytical_np)
    max_error = np.max(force_errors)
    mean_error = np.mean(force_errors)
    rmse = np.sqrt(np.mean(force_errors**2))

    # Check if passes tolerance
    passes = max_error < tolerance

    results = {
        'name': name,
        'n_atoms': len(atoms),
        'energy_auto': energy_auto_val,
        'energy_analytical': energy_analytical_val,
        'energy_error': energy_error,
        'max_force_error': max_error,
        'mean_force_error': mean_error,
        'rmse_force_error': rmse,
        'passes': passes,
        'tolerance': tolerance
    }

    # Print results
    status = "✓ PASS" if passes else "✗ FAIL"
    logger.info(
        f"{status} | {name:<15} | {len(atoms):>3} atoms | "
        f"Max error: {max_error:.2e} eV/Å | "
        f"RMSE: {rmse:.2e} eV/Å"
    )

    if verbose or not passes:
        print(f"\n  Detailed results for {name}:")
        print(f"    Energy (autograd):    {energy_auto_val:.6f} eV")
        print(f"    Energy (analytical):  {energy_analytical_val:.6f} eV")
        print(f"    Energy error:         {energy_error:.2e} eV")
        print(f"    Max force error:      {max_error:.2e} eV/Å")
        print(f"    Mean force error:     {mean_error:.2e} eV/Å")
        print(f"    RMSE force error:     {rmse:.2e} eV/Å")

        if not passes:
            print(f"\n    Per-atom force comparison:")
            print(f"    {'Atom':<6} {'Autograd Forces':<40} {'Analytical Forces':<40} {'Error':<15}")
            print(f"    {'-'*110}")
            for i in range(len(atoms)):
                f_auto = forces_auto_np[i]
                f_analytical = forces_analytical_np[i]
                f_error = force_errors[i]
                print(f"    {i:<6} "
                      f"[{f_auto[0]:>10.6f}, {f_auto[1]:>10.6f}, {f_auto[2]:>10.6f}]  "
                      f"[{f_analytical[0]:>10.6f}, {f_analytical[1]:>10.6f}, {f_analytical[2]:>10.6f}]  "
                      f"[{f_error[0]:.2e}, {f_error[1]:.2e}, {f_error[2]:.2e}]")
        print()

    return results


def run_validation_suite(
    model: StudentForceField,
    test_cases: list,
    tolerance: float = 1e-4,
    verbose: bool = False
) -> dict:
    """
    Run full validation suite.

    Args:
        model: StudentForceField model
        test_cases: List of (name, atoms) tuples
        tolerance: Maximum allowed force error
        verbose: Print detailed output

    Returns:
        dict with aggregated results
    """
    logger.info("=" * 80)
    logger.info("ANALYTICAL FORCES VALIDATION SUITE")
    logger.info("=" * 80)
    logger.info(f"Tolerance: {tolerance} eV/Å")
    logger.info(f"Test cases: {len(test_cases)}")
    logger.info("")

    all_results = []
    n_passed = 0
    n_failed = 0

    for name, atoms in test_cases:
        result = validate_single_molecule(
            model, name, atoms, tolerance, verbose
        )
        all_results.append(result)

        if result['passes']:
            n_passed += 1
        else:
            n_failed += 1

    # Summary statistics
    max_errors = [r['max_force_error'] for r in all_results]
    mean_errors = [r['mean_force_error'] for r in all_results]
    rmse_errors = [r['rmse_force_error'] for r in all_results]

    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total test cases: {len(test_cases)}")
    logger.info(f"Passed: {n_passed} ({n_passed/len(test_cases)*100:.1f}%)")
    logger.info(f"Failed: {n_failed} ({n_failed/len(test_cases)*100:.1f}%)")
    logger.info("")
    logger.info(f"Force Error Statistics:")
    logger.info(f"  Max error (worst case):  {max(max_errors):.2e} eV/Å")
    logger.info(f"  Max error (average):     {np.mean(max_errors):.2e} eV/Å")
    logger.info(f"  RMSE (average):          {np.mean(rmse_errors):.2e} eV/Å")
    logger.info("")

    if n_failed > 0:
        logger.warning(f"⚠ {n_failed} test cases failed!")
        logger.warning("Analytical forces may have correctness issues.")
        logger.warning("Review failed cases above for details.")
    else:
        logger.info("✓ All test cases passed!")
        logger.info("Analytical forces are numerically correct.")

    return {
        'n_total': len(test_cases),
        'n_passed': n_passed,
        'n_failed': n_failed,
        'pass_rate': n_passed / len(test_cases),
        'max_error_worst': max(max_errors),
        'max_error_avg': np.mean(max_errors),
        'rmse_avg': np.mean(rmse_errors),
        'all_results': all_results
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate analytical force computation"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=1e-4,
        help='Maximum allowed force error (eV/Å)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed per-molecule results'
    )

    args = parser.parse_args()

    # Load model
    logger.info(f"Loading model from {args.checkpoint}...")
    model = StudentForceField.load(args.checkpoint, device=args.device)
    model.eval()
    logger.info(f"Loaded model: {model.num_parameters():,} parameters\n")

    # Create test cases
    test_cases = create_test_cases()

    # Run validation
    results = run_validation_suite(
        model, test_cases, args.tolerance, args.verbose
    )

    # Exit code
    if results['n_failed'] > 0:
        logger.error("\nValidation FAILED ❌")
        return 1
    else:
        logger.info("\nValidation PASSED ✅")
        return 0


if __name__ == '__main__':
    sys.exit(main())
