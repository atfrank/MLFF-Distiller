#!/usr/bin/env python3
"""
Regenerate Dataset with Explicit Hydrogens

This script regenerates the merged dataset by:
1. Reading original SDF files from MolDiff with RDKit AddHs()
2. Generating teacher labels with Orb-v2 on H-complete structures
3. Creating new HDF5 dataset with explicit hydrogens

Usage:
    python scripts/regenerate_dataset_with_hydrogens.py \\
        --source-dir data/medium_scale_10k_moldiff \\
        --output data/merged_dataset_with_H/dataset.h5 \\
        --batch-size 32

Author: ML Distillation Project
Date: 2025-11-24
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List
import time

import h5py
import numpy as np
import torch
from tqdm import tqdm

# Add src to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from mlff_distiller.data.sdf_utils import read_structure_with_hydrogen_support, check_hydrogen_content
from mlff_distiller.models.teacher_wrappers import OrbCalculator


def setup_logging(log_file: Path):
    """Setup logging configuration."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def find_sdf_files(source_dir: Path) -> List[Path]:
    """Find all SDF files in source directory."""
    sdf_files = []

    # Search recursively for SDF files
    for sdf_path in source_dir.rglob("*.sdf"):
        sdf_files.append(sdf_path)

    return sorted(sdf_files)


def create_hdf5_dataset(output_path: Path, compression: str = 'gzip'):
    """Create HDF5 file with proper structure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    f = h5py.File(output_path, 'w')

    # Create groups
    structures_group = f.create_group('structures')
    labels_group = f.create_group('labels')

    # Create datasets with chunking for efficient access
    # We'll resize these as we add data
    structures_group.create_dataset(
        'atomic_numbers',
        shape=(0,),
        maxshape=(None,),
        dtype='int32',
        compression=compression
    )
    structures_group.create_dataset(
        'positions',
        shape=(0, 3),
        maxshape=(None, 3),
        dtype='float32',
        compression=compression
    )
    structures_group.create_dataset(
        'cells',
        shape=(0, 3, 3),
        maxshape=(None, 3, 3),
        dtype='float32',
        compression=compression
    )
    structures_group.create_dataset(
        'pbc',
        shape=(0, 3),
        maxshape=(None, 3),
        dtype='bool',
        compression=compression
    )
    structures_group.create_dataset(
        'atomic_numbers_splits',
        shape=(1,),  # Start with [0]
        maxshape=(None,),
        dtype='int64',
        compression=compression
    )
    structures_group['atomic_numbers_splits'][0] = 0

    # Labels
    labels_group.create_dataset(
        'energy',
        shape=(0,),
        maxshape=(None,),
        dtype='float32',
        compression=compression
    )
    labels_group.create_dataset(
        'forces',
        shape=(0, 3),
        maxshape=(None, 3),
        dtype='float32',
        compression=compression
    )
    labels_group.create_dataset(
        'forces_splits',
        shape=(1,),  # Start with [0]
        maxshape=(None,),
        dtype='int64',
        compression=compression
    )
    labels_group['forces_splits'][0] = 0

    return f


def add_structure_to_hdf5(
    f: h5py.File,
    atoms,
    energy: float,
    forces: np.ndarray,
    logger: logging.Logger
):
    """Add a single structure to HDF5 file."""
    structures_group = f['structures']
    labels_group = f['labels']

    # Get current sizes
    current_n_atoms = structures_group['atomic_numbers'].shape[0]
    current_n_structures = structures_group['cells'].shape[0]

    n_atoms = len(atoms)

    # Resize datasets
    structures_group['atomic_numbers'].resize((current_n_atoms + n_atoms,))
    structures_group['positions'].resize((current_n_atoms + n_atoms, 3))
    structures_group['cells'].resize((current_n_structures + 1, 3, 3))
    structures_group['pbc'].resize((current_n_structures + 1, 3))
    structures_group['atomic_numbers_splits'].resize((current_n_structures + 2,))

    labels_group['energy'].resize((current_n_structures + 1,))
    labels_group['forces'].resize((current_n_atoms + n_atoms, 3))
    labels_group['forces_splits'].resize((current_n_structures + 2,))

    # Add data
    structures_group['atomic_numbers'][current_n_atoms:current_n_atoms + n_atoms] = atoms.get_atomic_numbers()
    structures_group['positions'][current_n_atoms:current_n_atoms + n_atoms] = atoms.get_positions()

    # Non-periodic molecules
    structures_group['cells'][current_n_structures] = np.zeros((3, 3))
    structures_group['pbc'][current_n_structures] = [False, False, False]

    structures_group['atomic_numbers_splits'][current_n_structures + 1] = current_n_atoms + n_atoms

    labels_group['energy'][current_n_structures] = energy
    labels_group['forces'][current_n_atoms:current_n_atoms + n_atoms] = forces
    labels_group['forces_splits'][current_n_structures + 1] = current_n_atoms + n_atoms


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate dataset with explicit hydrogens"
    )
    parser.add_argument(
        '--source-dir',
        type=Path,
        default=REPO_ROOT / 'data/medium_scale_10k_moldiff',
        help='Directory containing original SDF files'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=REPO_ROOT / 'data/merged_dataset_with_H/dataset.h5',
        help='Output HDF5 file path'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for teacher inference (use 1 for Orb models)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device for teacher model'
    )
    parser.add_argument(
        '--max-structures',
        type=int,
        default=None,
        help='Maximum number of structures to process (for testing)'
    )
    parser.add_argument(
        '--log-file',
        type=Path,
        default=REPO_ROOT / 'logs/dataset_regeneration.log',
        help='Log file path'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_file)

    logger.info("="*80)
    logger.info("Dataset Regeneration with Explicit Hydrogens")
    logger.info("="*80)
    logger.info(f"Source directory: {args.source_dir}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Device: {args.device}")

    # Find SDF files
    logger.info("\nSearching for SDF files...")
    sdf_files = find_sdf_files(args.source_dir)
    logger.info(f"Found {len(sdf_files)} SDF files")

    if len(sdf_files) == 0:
        logger.error("No SDF files found! Check source directory.")
        return 1

    # Limit if requested
    if args.max_structures is not None:
        sdf_files = sdf_files[:args.max_structures]
        logger.info(f"Limited to {len(sdf_files)} structures for testing")

    # Initialize teacher model
    logger.info("\nInitializing Orb-v2 teacher model...")
    try:
        teacher = OrbCalculator(device=args.device)
        logger.info("✓ Teacher model loaded")
    except Exception as e:
        logger.error(f"Failed to load teacher model: {e}")
        return 1

    # Create HDF5 file
    logger.info(f"\nCreating HDF5 file: {args.output}")
    hdf5_file = create_hdf5_dataset(args.output)

    # Statistics
    stats = {
        'total': len(sdf_files),
        'success': 0,
        'failed': 0,
        'total_atoms': 0,
        'total_hydrogen': 0,
        'start_time': time.time()
    }

    # Process structures
    logger.info("\nProcessing structures...")
    logger.info("-"*80)

    with tqdm(total=len(sdf_files), desc="Generating labels") as pbar:
        for idx, sdf_path in enumerate(sdf_files):
            try:
                # Read with hydrogen addition
                atoms = read_structure_with_hydrogen_support(sdf_path)

                # Check hydrogen content
                h_stats = check_hydrogen_content(atoms)

                if not h_stats['has_hydrogen']:
                    logger.warning(f"Structure {idx} has NO hydrogen: {sdf_path.name}")

                # Get teacher predictions using ASE Calculator interface
                atoms.calc = teacher
                energy = atoms.get_potential_energy()  # eV
                forces = atoms.get_forces()            # eV/Angstrom

                # Add to HDF5
                add_structure_to_hdf5(
                    hdf5_file,
                    atoms,
                    energy=energy,
                    forces=forces,
                    logger=logger
                )

                # Update stats
                stats['success'] += 1
                stats['total_atoms'] += h_stats['n_atoms']
                stats['total_hydrogen'] += h_stats['n_hydrogen']

                # Log progress
                if (idx + 1) % 100 == 0:
                    avg_h_pct = 100.0 * stats['total_hydrogen'] / max(stats['total_atoms'], 1)
                    logger.info(
                        f"Processed {idx + 1}/{len(sdf_files)} | "
                        f"Success: {stats['success']} | "
                        f"Avg H: {avg_h_pct:.1f}% | "
                        f"Atoms: {stats['total_atoms']:,}"
                    )

                pbar.update(1)

            except Exception as e:
                logger.error(f"Failed on {sdf_path.name}: {e}")
                stats['failed'] += 1
                pbar.update(1)
                continue

    # Close HDF5
    hdf5_file.close()

    # Final statistics
    elapsed = time.time() - stats['start_time']
    avg_h_pct = 100.0 * stats['total_hydrogen'] / max(stats['total_atoms'], 1)

    logger.info("\n" + "="*80)
    logger.info("Dataset Regeneration Complete!")
    logger.info("="*80)
    logger.info(f"Total structures: {stats['total']}")
    logger.info(f"Successfully processed: {stats['success']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Success rate: {100.0 * stats['success'] / stats['total']:.1f}%")
    logger.info(f"\nAtom statistics:")
    logger.info(f"  Total atoms: {stats['total_atoms']:,}")
    logger.info(f"  Hydrogen atoms: {stats['total_hydrogen']:,}")
    logger.info(f"  Hydrogen percentage: {avg_h_pct:.1f}%")
    logger.info(f"\nElapsed time: {elapsed/3600:.2f} hours")
    logger.info(f"Structures/hour: {stats['success'] / (elapsed/3600):.1f}")
    logger.info(f"\nOutput: {args.output}")
    logger.info("="*80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
