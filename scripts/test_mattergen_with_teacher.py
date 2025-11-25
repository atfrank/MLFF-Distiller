#!/usr/bin/env python
"""
Test MatterGen-generated structures with orb-v2 teacher model.

This script validates that:
1. MatterGen structures can be loaded as ASE Atoms
2. Structures are valid (periodic, reasonable cell parameters)
3. Teacher model can compute labels without errors
4. Validation success rate meets >95% threshold

Usage:
    python scripts/test_mattergen_with_teacher.py /path/to/generated_crystals.extxyz

Author: Lead Project Coordinator
Date: 2025-11-23
"""

import sys
from pathlib import Path
from typing import List

import numpy as np
from ase import Atoms
from ase.io import read

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlff_distiller.data.label_generation import OrbV2Teacher


def load_structures(extxyz_path: Path) -> List[Atoms]:
    """
    Load structures from .extxyz file.

    Args:
        extxyz_path: Path to .extxyz file

    Returns:
        List of ASE Atoms objects
    """
    if not extxyz_path.exists():
        raise FileNotFoundError(f"File not found: {extxyz_path}")

    structures = read(str(extxyz_path), index=":")
    if not isinstance(structures, list):
        structures = [structures]

    return structures


def validate_structure(atoms: Atoms) -> tuple[bool, str]:
    """
    Validate structure is physically reasonable.

    Args:
        atoms: ASE Atoms object

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check has atoms
    if len(atoms) == 0:
        return False, "Empty structure"

    # Check positions are finite
    if not np.isfinite(atoms.positions).all():
        return False, "Non-finite positions (NaN or Inf)"

    # Check periodic boundary conditions
    if not atoms.pbc.any():
        return False, "No periodic boundary conditions"

    # Check cell is valid
    try:
        volume = atoms.get_volume()
        if volume <= 0 or not np.isfinite(volume):
            return False, f"Invalid cell volume: {volume}"
    except Exception as e:
        return False, f"Cell error: {e}"

    # Check for overlapping atoms (minimum distance)
    distances = atoms.get_all_distances(mic=True)
    np.fill_diagonal(distances, np.inf)
    min_dist = distances.min()
    if min_dist < 0.5:  # Angstrom
        return False, f"Atoms too close: {min_dist:.3f} A"

    return True, "Valid"


def test_teacher_labeling(structures: List[Atoms], device: str = "cuda") -> dict:
    """
    Test teacher model labeling on structures.

    Args:
        structures: List of ASE Atoms
        device: Device for teacher model

    Returns:
        Dictionary with results
    """
    print(f"\nInitializing orb-v2 teacher model on {device}...")
    teacher = OrbV2Teacher(device=device)
    print("Teacher model ready\n")

    results = {
        "total": len(structures),
        "valid_structure": 0,
        "teacher_success": 0,
        "teacher_failed": 0,
        "invalid_structure": 0,
        "errors": [],
        "energies": [],
        "max_forces": [],
    }

    for i, atoms in enumerate(structures):
        print(f"Structure {i+1}/{len(structures)}:")
        print(f"  Formula: {atoms.get_chemical_formula()}")
        print(f"  Atoms: {len(atoms)}")
        print(f"  Volume: {atoms.get_volume():.2f} A^3")

        # Validate structure
        is_valid, msg = validate_structure(atoms)
        if not is_valid:
            print(f"  Status: INVALID - {msg}")
            results["invalid_structure"] += 1
            results["errors"].append(f"Structure {i+1}: {msg}")
            continue

        results["valid_structure"] += 1

        # Test teacher labeling
        try:
            label = teacher.label_structure(atoms)
            energy = label["energy"]
            forces = label["forces"]
            max_force = np.abs(forces).max()

            print(f"  Energy: {energy:.4f} eV")
            print(f"  Max force: {max_force:.4f} eV/A")
            print(f"  Status: SUCCESS")

            results["teacher_success"] += 1
            results["energies"].append(energy)
            results["max_forces"].append(max_force)

        except Exception as e:
            print(f"  Status: FAILED - {e}")
            results["teacher_failed"] += 1
            results["errors"].append(f"Structure {i+1}: {e}")

        print()

    return results


def print_summary(results: dict):
    """Print summary of validation results."""
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    total = results["total"]
    valid = results["valid_structure"]
    success = results["teacher_success"]
    failed = results["teacher_failed"]
    invalid = results["invalid_structure"]

    print(f"\nTotal structures tested: {total}")
    print(f"  Valid structures: {valid} ({valid/total*100:.1f}%)")
    print(f"  Invalid structures: {invalid} ({invalid/total*100:.1f}%)")
    print(f"\nTeacher model labeling:")
    print(f"  Success: {success} ({success/total*100:.1f}%)")
    print(f"  Failed: {failed} ({failed/total*100:.1f}%)")

    if results["energies"]:
        energies = np.array(results["energies"])
        forces = np.array(results["max_forces"])
        print(f"\nEnergy statistics:")
        print(f"  Mean: {energies.mean():.4f} eV")
        print(f"  Std: {energies.std():.4f} eV")
        print(f"  Range: [{energies.min():.4f}, {energies.max():.4f}] eV")
        print(f"\nMax force statistics:")
        print(f"  Mean: {forces.mean():.4f} eV/A")
        print(f"  Std: {forces.std():.4f} eV/A")
        print(f"  Range: [{forces.min():.4f}, {forces.max():.4f}] eV/A")

    # Go/No-Go decision
    success_rate = success / total * 100
    threshold = 95.0

    print("\n" + "=" * 70)
    print("GO/NO-GO DECISION")
    print("=" * 70)
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Threshold: {threshold:.1f}%")

    if success_rate >= threshold:
        print("\nSTATUS: PASS")
        print("Recommendation: Proceed to wrapper implementation")
    else:
        print("\nSTATUS: FAIL")
        print("Recommendation: Investigate errors before proceeding")

    if results["errors"]:
        print(f"\nErrors encountered ({len(results['errors'])}):")
        for error in results["errors"][:10]:  # Show first 10
            print(f"  - {error}")
        if len(results["errors"]) > 10:
            print(f"  ... and {len(results['errors']) - 10} more")

    print("=" * 70)


def main():
    """Main execution."""
    if len(sys.argv) < 2:
        # Default path for testing
        extxyz_path = Path("/tmp/mattergen_test_001/generated_crystals.extxyz")
        print(f"No path provided, using default: {extxyz_path}")
    else:
        extxyz_path = Path(sys.argv[1])

    print("=" * 70)
    print("MATTERGEN + ORB-V2 VALIDATION TEST")
    print("=" * 70)
    print(f"Input file: {extxyz_path}")

    # Load structures
    try:
        structures = load_structures(extxyz_path)
        print(f"Loaded {len(structures)} structures")
    except Exception as e:
        print(f"ERROR: Failed to load structures - {e}")
        return 1

    # Test with teacher
    try:
        results = test_teacher_labeling(structures, device="cuda")
    except Exception as e:
        print(f"ERROR: Teacher validation failed - {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Print summary
    print_summary(results)

    # Return exit code
    success_rate = results["teacher_success"] / results["total"] * 100
    return 0 if success_rate >= 95.0 else 1


if __name__ == "__main__":
    sys.exit(main())
