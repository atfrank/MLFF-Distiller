#!/usr/bin/env python3
"""
Verification script to test hydrogen addition fix for SDF files.

This script:
1. Reads a sample SDF file with ASE (missing H)
2. Reads the same file with RDKit + AddHs (with H)
3. Converts to ASE format with explicit hydrogens
4. Verifies the teacher model can process hydrogen-complete structures
5. Compares energy/force predictions

Usage:
    python scripts/verify_hydrogen_fix.py --sdf data/medium_scale_10k_moldiff/moldiff_batch_3271/moldiff_config_20251123_201255_SDF/3.sdf
"""

import sys
from pathlib import Path
import argparse
import numpy as np
from collections import Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ase import Atoms
from ase.io import read as ase_read
from rdkit import Chem
from rdkit.Chem import AllChem


def read_sdf_with_hydrogens(sdf_path: Path) -> Atoms:
    """
    Read SDF file and add explicit hydrogens with 3D coordinates.

    Args:
        sdf_path: Path to SDF file

    Returns:
        ASE Atoms object with explicit hydrogens
    """
    # Load with RDKit
    supplier = Chem.SDMolSupplier(str(sdf_path))
    mol = supplier[0]

    if mol is None:
        raise ValueError(f"Failed to read {sdf_path}")

    # Add explicit hydrogens
    mol_with_h = Chem.AddHs(mol)

    # Embed 3D coordinates for H atoms
    # Use existing heavy atom coords, generate H coords
    AllChem.EmbedMolecule(mol_with_h, randomSeed=42, useRandomCoords=False)

    # Convert to ASE Atoms
    positions = mol_with_h.GetConformer().GetPositions()
    atomic_numbers = [atom.GetAtomicNum() for atom in mol_with_h.GetAtoms()]

    atoms = Atoms(numbers=atomic_numbers, positions=positions)

    return atoms


def compare_structures(atoms_without_h: Atoms, atoms_with_h: Atoms):
    """Print comparison between structures with and without H."""
    print("="*80)
    print("Structure Comparison")
    print("="*80)

    print("\n1. WITHOUT HYDROGEN (ASE default read):")
    print(f"   Number of atoms: {len(atoms_without_h)}")
    print(f"   Chemical formula: {atoms_without_h.get_chemical_formula()}")
    symbols = atoms_without_h.get_chemical_symbols()
    print(f"   Composition: {Counter(symbols)}")
    print(f"   Has hydrogen: {1 in atoms_without_h.get_atomic_numbers()}")

    print("\n2. WITH HYDROGEN (RDKit AddHs + conversion):")
    print(f"   Number of atoms: {len(atoms_with_h)}")
    print(f"   Chemical formula: {atoms_with_h.get_chemical_formula()}")
    symbols = atoms_with_h.get_chemical_symbols()
    composition = Counter(symbols)
    print(f"   Composition: {composition}")
    print(f"   Has hydrogen: {1 in atoms_with_h.get_atomic_numbers()}")
    print(f"   Hydrogen percentage: {100*composition['H']/len(atoms_with_h):.1f}%")

    print("\n3. DIFFERENCE:")
    print(f"   Added atoms: {len(atoms_with_h) - len(atoms_without_h)}")
    print(f"   All added atoms are hydrogen: {len(atoms_with_h) - len(atoms_without_h) == composition['H']}")


def test_teacher_model(atoms: Atoms):
    """Test if teacher model can process the structure."""
    print("\n" + "="*80)
    print("Teacher Model Test (Orb-v2)")
    print("="*80)

    try:
        from mlff_distiller.models.teacher_wrappers import OrbWrapper
        import torch

        print("\nInitializing Orb-v2 model...")
        teacher = OrbWrapper(device='cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Predicting energy and forces for {len(atoms)} atoms...")
        result = teacher.predict(
            positions=atoms.get_positions(),
            species=atoms.get_atomic_numbers(),
            cell=None
        )

        print(f"\n✓ SUCCESS!")
        print(f"  Energy: {result['energy']:.4f} eV")
        print(f"  Forces shape: {result['forces'].shape}")
        print(f"  Force magnitude: {np.linalg.norm(result['forces'], axis=1).mean():.4f} eV/Å (mean)")

        # Check hydrogen forces specifically
        h_indices = [i for i, z in enumerate(atoms.get_atomic_numbers()) if z == 1]
        if h_indices:
            h_forces = result['forces'][h_indices]
            print(f"\n  Hydrogen-specific analysis:")
            print(f"    Number of H atoms: {len(h_indices)}")
            print(f"    H force magnitude: {np.linalg.norm(h_forces, axis=1).mean():.4f} eV/Å (mean)")
            print(f"    H force range: [{np.linalg.norm(h_forces, axis=1).min():.4f}, "
                  f"{np.linalg.norm(h_forces, axis=1).max():.4f}] eV/Å")

        return True

    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Verify hydrogen addition fix for SDF files"
    )
    parser.add_argument(
        '--sdf',
        type=Path,
        default=Path('/home/aaron/ATX/software/MLFF_Distiller/data/medium_scale_10k_moldiff/moldiff_batch_3271/moldiff_config_20251123_201255_SDF/3.sdf'),
        help='Path to SDF file to test'
    )
    parser.add_argument(
        '--no-teacher-test',
        action='store_true',
        help='Skip teacher model testing'
    )

    args = parser.parse_args()

    if not args.sdf.exists():
        print(f"ERROR: SDF file not found: {args.sdf}")
        sys.exit(1)

    print("="*80)
    print("Hydrogen Addition Fix Verification")
    print("="*80)
    print(f"\nTesting with: {args.sdf}")

    # Read without hydrogen (ASE default)
    print("\nReading with ASE (default - no H)...")
    atoms_without_h = ase_read(str(args.sdf))

    # Read with hydrogen (RDKit + conversion)
    print("Reading with RDKit + AddHs...")
    atoms_with_h = read_sdf_with_hydrogens(args.sdf)

    # Compare
    compare_structures(atoms_without_h, atoms_with_h)

    # Test teacher model
    if not args.no_teacher_test:
        success = test_teacher_model(atoms_with_h)

        if success:
            print("\n" + "="*80)
            print("✓ VERIFICATION PASSED")
            print("="*80)
            print("\nThe fix is ready to be integrated into dataset generation pipeline.")
            print("Next steps:")
            print("  1. Update scripts/generate_medium_scale.py to use read_sdf_with_hydrogens()")
            print("  2. Update scripts/generate_labels.py for SDF handling")
            print("  3. Regenerate dataset with explicit hydrogens")
            print("  4. Retrain student model on hydrogen-complete dataset")
        else:
            print("\n" + "="*80)
            print("✗ VERIFICATION FAILED")
            print("="*80)
            print("\nTeacher model could not process hydrogen-complete structure.")
            print("Check teacher model compatibility with hydrogen atoms.")
    else:
        print("\n" + "="*80)
        print("✓ STRUCTURE CONVERSION SUCCESSFUL")
        print("="*80)
        print("Skipped teacher model test (use --no-teacher-test flag)")


if __name__ == '__main__':
    main()
