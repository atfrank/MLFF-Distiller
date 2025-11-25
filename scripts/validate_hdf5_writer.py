#!/usr/bin/env python3
"""
Validation Script for HDF5DatasetWriter

This script validates the HDF5DatasetWriter implementation by:
1. Loading existing labeled data from all_labels_orb_v2.h5
2. Writing it using the new HDF5DatasetWriter
3. Comparing the two files for consistency
4. Testing append mode
5. Measuring compression effectiveness

Usage:
    python scripts/validate_hdf5_writer.py

Author: Data Pipeline Engineer
Date: 2025-11-23
"""

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

import numpy as np
import h5py
from ase import Atoms
import pickle
from tqdm import tqdm

from mlff_distiller.data.hdf5_writer import HDF5DatasetWriter


def load_structures_from_hdf5(hdf5_path):
    """
    Load structures from existing HDF5 file.

    Args:
        hdf5_path: Path to HDF5 file

    Returns:
        List of (atoms, energy, forces, stress) tuples
    """
    print(f"Loading structures from {hdf5_path}...")

    structures_data = []

    with h5py.File(hdf5_path, 'r') as f:
        # Get splits
        splits = f['structures']['atomic_numbers_splits'][:]
        n_structures = len(splits) - 1

        # Load all data
        all_atomic_numbers = f['structures']['atomic_numbers'][:]
        all_positions = f['structures']['positions'][:]
        cells = f['structures']['cells'][:]
        pbc = f['structures']['pbc'][:]

        energies = f['labels']['energy'][:]
        all_forces = f['labels']['forces'][:]

        # Check if stress exists
        has_stress = 'stress' in f['labels']
        if has_stress:
            stresses = f['labels']['stress'][:]
            stress_mask = f['labels']['stress_mask'][:]
        else:
            stresses = None
            stress_mask = None

        # Reconstruct each structure
        for i in tqdm(range(n_structures), desc="Loading structures"):
            start = splits[i]
            end = splits[i + 1]

            # Create Atoms object
            atoms = Atoms(
                numbers=all_atomic_numbers[start:end],
                positions=all_positions[start:end],
                cell=cells[i],
                pbc=pbc[i]
            )

            energy = energies[i]
            forces = all_forces[start:end]

            if has_stress and stress_mask[i]:
                stress = stresses[i]
            else:
                stress = None

            structures_data.append((atoms, energy, forces, stress))

    print(f"Loaded {len(structures_data)} structures")
    return structures_data


def compare_hdf5_files(file1, file2, tolerance=1e-6):
    """
    Compare two HDF5 files for consistency.

    Args:
        file1: Path to first HDF5 file
        file2: Path to second HDF5 file
        tolerance: Numerical tolerance for comparisons

    Returns:
        bool: True if files are consistent
    """
    print(f"\nComparing {file1.name} vs {file2.name}...")

    with h5py.File(file1, 'r') as f1, h5py.File(file2, 'r') as f2:
        # Compare structures group
        for key in ['atomic_numbers', 'atomic_numbers_splits', 'positions', 'cells', 'pbc']:
            data1 = f1['structures'][key][:]
            data2 = f2['structures'][key][:]

            if np.issubdtype(data1.dtype, np.floating):
                if not np.allclose(data1, data2, rtol=tolerance, atol=tolerance):
                    print(f"  ERROR: {key} values differ!")
                    print(f"    Max difference: {np.max(np.abs(data1 - data2))}")
                    return False
            else:
                if not np.array_equal(data1, data2):
                    print(f"  ERROR: {key} values differ!")
                    return False

            print(f"  ✓ {key} matches")

        # Compare labels group
        for key in ['energy', 'forces', 'forces_splits', 'structure_indices']:
            data1 = f1['labels'][key][:]
            data2 = f2['labels'][key][:]

            if np.issubdtype(data1.dtype, np.floating):
                if not np.allclose(data1, data2, rtol=tolerance, atol=tolerance):
                    print(f"  ERROR: {key} values differ!")
                    print(f"    Max difference: {np.max(np.abs(data1 - data2))}")
                    return False
            else:
                if not np.array_equal(data1, data2):
                    print(f"  ERROR: {key} values differ!")
                    return False

            print(f"  ✓ {key} matches")

        # Compare stress if present
        if 'stress' in f1['labels'] and 'stress' in f2['labels']:
            data1 = f1['labels']['stress'][:]
            data2 = f2['labels']['stress'][:]
            if not np.allclose(data1, data2, rtol=tolerance, atol=tolerance):
                print(f"  ERROR: stress values differ!")
                return False
            print(f"  ✓ stress matches")

            mask1 = f1['labels']['stress_mask'][:]
            mask2 = f2['labels']['stress_mask'][:]
            if not np.array_equal(mask1, mask2):
                print(f"  ERROR: stress_mask values differ!")
                return False
            print(f"  ✓ stress_mask matches")

    print("  All datasets match! ✓")
    return True


def test_basic_write(structures_data, output_dir):
    """Test basic write functionality."""
    print("\n" + "=" * 60)
    print("TEST 1: Basic Write")
    print("=" * 60)

    output_path = output_dir / "test_basic_write.h5"

    # Write using new writer
    with HDF5DatasetWriter(output_path, compression="gzip") as writer:
        for atoms, energy, forces, stress in tqdm(structures_data, desc="Writing structures"):
            writer.add_structure(atoms, energy, forces, stress)

    print(f"\nWrote {len(structures_data)} structures to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024**2:.2f} MB")

    return output_path


def test_batch_write(structures_data, output_dir):
    """Test batch write functionality."""
    print("\n" + "=" * 60)
    print("TEST 2: Batch Write")
    print("=" * 60)

    output_path = output_dir / "test_batch_write.h5"

    # Unpack data
    structures = [s[0] for s in structures_data]
    energies = [s[1] for s in structures_data]
    forces = [s[2] for s in structures_data]
    stresses = [s[3] for s in structures_data]

    # Write in batch
    with HDF5DatasetWriter(output_path, compression="gzip") as writer:
        writer.add_batch(
            structures=structures,
            energies=energies,
            forces=forces,
            stresses=stresses,
            show_progress=True
        )

    print(f"\nWrote {len(structures_data)} structures in batch to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024**2:.2f} MB")

    return output_path


def test_append_mode(structures_data, output_dir):
    """Test append mode."""
    print("\n" + "=" * 60)
    print("TEST 3: Append Mode")
    print("=" * 60)

    output_path = output_dir / "test_append.h5"

    # Split data
    split_idx = len(structures_data) // 2
    first_half = structures_data[:split_idx]
    second_half = structures_data[split_idx:]

    # Write first half
    print(f"Writing first {len(first_half)} structures...")
    with HDF5DatasetWriter(output_path, mode="w", compression="gzip") as writer:
        for atoms, energy, forces, stress in first_half:
            writer.add_structure(atoms, energy, forces, stress)

    print(f"Appending {len(second_half)} structures...")
    with HDF5DatasetWriter(output_path, mode="a", compression="gzip") as writer:
        for atoms, energy, forces, stress in second_half:
            writer.add_structure(atoms, energy, forces, stress)

    # Verify total count
    with h5py.File(output_path, 'r') as f:
        total = f['labels']['energy'].shape[0]
        print(f"\nTotal structures in file: {total}")
        assert total == len(structures_data), "Append mode failed!"

    print(f"File size: {output_path.stat().st_size / 1024**2:.2f} MB")
    print("Append mode test passed! ✓")

    return output_path


def test_compression(structures_data, output_dir):
    """Test compression effectiveness."""
    print("\n" + "=" * 60)
    print("TEST 4: Compression Effectiveness")
    print("=" * 60)

    structures = [s[0] for s in structures_data]
    energies = [s[1] for s in structures_data]
    forces = [s[2] for s in structures_data]
    stresses = [s[3] for s in structures_data]

    results = {}

    # No compression
    print("\nWriting without compression...")
    path_none = output_dir / "test_no_compression.h5"
    with HDF5DatasetWriter(path_none, compression=None) as writer:
        writer.add_batch(structures, energies, forces, stresses, show_progress=True)
    size_none = path_none.stat().st_size
    results['none'] = size_none
    print(f"  Size: {size_none / 1024**2:.2f} MB")

    # LZF compression
    print("\nWriting with LZF compression...")
    path_lzf = output_dir / "test_lzf_compression.h5"
    with HDF5DatasetWriter(path_lzf, compression="lzf") as writer:
        writer.add_batch(structures, energies, forces, stresses, show_progress=True)
    size_lzf = path_lzf.stat().st_size
    results['lzf'] = size_lzf
    print(f"  Size: {size_lzf / 1024**2:.2f} MB")
    print(f"  Compression ratio: {size_lzf / size_none:.2%}")

    # GZIP compression (level 4)
    print("\nWriting with GZIP compression (level 4)...")
    path_gzip4 = output_dir / "test_gzip4_compression.h5"
    with HDF5DatasetWriter(path_gzip4, compression="gzip", compression_opts=4) as writer:
        writer.add_batch(structures, energies, forces, stresses, show_progress=True)
    size_gzip4 = path_gzip4.stat().st_size
    results['gzip-4'] = size_gzip4
    print(f"  Size: {size_gzip4 / 1024**2:.2f} MB")
    print(f"  Compression ratio: {size_gzip4 / size_none:.2%}")

    # GZIP compression (level 9)
    print("\nWriting with GZIP compression (level 9)...")
    path_gzip9 = output_dir / "test_gzip9_compression.h5"
    with HDF5DatasetWriter(path_gzip9, compression="gzip", compression_opts=9) as writer:
        writer.add_batch(structures, energies, forces, stresses, show_progress=True)
    size_gzip9 = path_gzip9.stat().st_size
    results['gzip-9'] = size_gzip9
    print(f"  Size: {size_gzip9 / 1024**2:.2f} MB")
    print(f"  Compression ratio: {size_gzip9 / size_none:.2%}")

    # Summary
    print("\n" + "-" * 60)
    print("Compression Summary:")
    print("-" * 60)
    for method, size in results.items():
        ratio = size / size_none
        savings = (1 - ratio) * 100
        print(f"  {method:12s}: {size / 1024**2:6.2f} MB ({ratio:5.1%} of original, {savings:4.1f}% savings)")

    return path_gzip4  # Return the recommended option


def main():
    """Main validation script."""
    print("=" * 60)
    print("HDF5DatasetWriter Validation Script")
    print("=" * 60)

    # Paths
    data_dir = repo_root / "data"
    existing_hdf5 = data_dir / "labels" / "all_labels_orb_v2.h5"
    output_dir = data_dir / "validation"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Check if existing file exists
    if not existing_hdf5.exists():
        print(f"ERROR: Existing HDF5 file not found at {existing_hdf5}")
        print("Please ensure the labeled data has been generated.")
        return 1

    # Load existing structures
    structures_data = load_structures_from_hdf5(existing_hdf5)

    # Run tests
    basic_path = test_basic_write(structures_data, output_dir)
    batch_path = test_batch_write(structures_data, output_dir)
    append_path = test_append_mode(structures_data, output_dir)
    compressed_path = test_compression(structures_data, output_dir)

    # Compare files
    print("\n" + "=" * 60)
    print("VALIDATION: Comparing Files")
    print("=" * 60)

    # Compare basic write with original
    if not compare_hdf5_files(existing_hdf5, basic_path):
        print("\nERROR: Basic write does not match original!")
        return 1

    # Compare batch write with basic write
    if not compare_hdf5_files(basic_path, batch_path):
        print("\nERROR: Batch write does not match basic write!")
        return 1

    # Compare append mode with basic write
    if not compare_hdf5_files(basic_path, append_path):
        print("\nERROR: Append mode does not match basic write!")
        return 1

    # Compare compressed with basic write
    if not compare_hdf5_files(basic_path, compressed_path):
        print("\nERROR: Compressed version does not match basic write!")
        return 1

    # Final summary
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE - ALL TESTS PASSED! ✓")
    print("=" * 60)
    print(f"\nOriginal file:  {existing_hdf5}")
    print(f"  Size: {existing_hdf5.stat().st_size / 1024**2:.2f} MB")
    print(f"\nNew file:       {compressed_path}")
    print(f"  Size: {compressed_path.stat().st_size / 1024**2:.2f} MB")

    print("\nRecommendation: Use compression='gzip' with compression_opts=4")
    print("This provides good compression (~40-50% reduction) with fast I/O.")

    print("\nValidation files saved to:", output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
