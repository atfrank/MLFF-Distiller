#!/usr/bin/env python3
"""
Merge MolDiff and RNA Datasets

Combines the MolDiff-only dataset (3,883 structures) with the RNA dataset
(1,000 structures) into a single unified training dataset.

Author: Lead Coordinator
Date: 2025-11-24
"""

import sys
import logging
from pathlib import Path
import h5py
import numpy as np

REPO_ROOT = Path("/home/aaron/ATX/software/MLFF_Distiller")
sys.path.insert(0, str(REPO_ROOT / "src"))


def setup_logging():
    """Configure logging."""
    log_file = REPO_ROOT / "logs/merge_datasets.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def load_dataset_info(hdf5_path: Path):
    """Load dataset information."""
    with h5py.File(hdf5_path, 'r') as f:
        n_structures = len(f['labels']['energy'][:])
        n_atoms = f['structures']['atomic_numbers'].shape[0]
        return n_structures, n_atoms


def merge_datasets(source1: Path, source2: Path, output: Path):
    """Merge two HDF5 datasets into one."""

    logging.info("="*60)
    logging.info("DATASET MERGING")
    logging.info("="*60)

    # Load source datasets info
    n1, atoms1 = load_dataset_info(source1)
    n2, atoms2 = load_dataset_info(source2)

    logging.info(f"\nSource 1 ({source1.name}):")
    logging.info(f"  Structures: {n1}")
    logging.info(f"  Total atoms: {atoms1}")

    logging.info(f"\nSource 2 ({source2.name}):")
    logging.info(f"  Structures: {n2}")
    logging.info(f"  Total atoms: {atoms2}")

    total_structures = n1 + n2
    total_atoms = atoms1 + atoms2

    logging.info(f"\nMerged dataset will have:")
    logging.info(f"  Structures: {total_structures}")
    logging.info(f"  Total atoms: {total_atoms}")

    # Create output file
    logging.info(f"\nCreating merged dataset: {output}")

    with h5py.File(source1, 'r') as f1, h5py.File(source2, 'r') as f2, h5py.File(output, 'w') as out:

        # Create groups
        out.create_group('structures')
        out.create_group('labels')
        out.create_group('metadata')

        # Merge structures
        logging.info("\nMerging /structures/ datasets...")

        # Concatenate atomic numbers
        atomic_numbers1 = f1['structures']['atomic_numbers'][:]
        atomic_numbers2 = f2['structures']['atomic_numbers'][:]
        out['structures'].create_dataset(
            'atomic_numbers',
            data=np.concatenate([atomic_numbers1, atomic_numbers2]),
            compression='gzip',
            compression_opts=4
        )

        # Merge splits (need to offset second dataset)
        splits1 = f1['structures']['atomic_numbers_splits'][:]
        splits2 = f2['structures']['atomic_numbers_splits'][:]
        # Remove first element of splits2 (which is 0) and add offset
        splits2_offset = splits2[1:] + splits1[-1]
        merged_splits = np.concatenate([splits1, splits2_offset])
        out['structures'].create_dataset('atomic_numbers_splits', data=merged_splits)

        # Concatenate positions
        positions1 = f1['structures']['positions'][:]
        positions2 = f2['structures']['positions'][:]
        out['structures'].create_dataset(
            'positions',
            data=np.concatenate([positions1, positions2]),
            compression='gzip',
            compression_opts=4
        )

        # Concatenate cells
        cells1 = f1['structures']['cells'][:]
        cells2 = f2['structures']['cells'][:]
        out['structures'].create_dataset(
            'cells',
            data=np.concatenate([cells1, cells2]),
            compression='gzip',
            compression_opts=4
        )

        # Concatenate pbc
        pbc1 = f1['structures']['pbc'][:]
        pbc2 = f2['structures']['pbc'][:]
        out['structures'].create_dataset(
            'pbc',
            data=np.concatenate([pbc1, pbc2])
        )

        # Merge labels
        logging.info("Merging /labels/ datasets...")

        # Concatenate energies
        energy1 = f1['labels']['energy'][:]
        energy2 = f2['labels']['energy'][:]
        out['labels'].create_dataset(
            'energy',
            data=np.concatenate([energy1, energy2]),
            compression='gzip',
            compression_opts=4
        )

        # Concatenate forces
        forces1 = f1['labels']['forces'][:]
        forces2 = f2['labels']['forces'][:]
        out['labels'].create_dataset(
            'forces',
            data=np.concatenate([forces1, forces2]),
            compression='gzip',
            compression_opts=4
        )

        # Merge force splits
        forces_splits1 = f1['labels']['forces_splits'][:]
        forces_splits2 = f2['labels']['forces_splits'][:]
        forces_splits2_offset = forces_splits2[1:] + forces_splits1[-1]
        merged_forces_splits = np.concatenate([forces_splits1, forces_splits2_offset])
        out['labels'].create_dataset('forces_splits', data=merged_forces_splits)

        # Concatenate stress
        stress1 = f1['labels']['stress'][:]
        stress2 = f2['labels']['stress'][:]
        out['labels'].create_dataset(
            'stress',
            data=np.concatenate([stress1, stress2])
        )

        # Concatenate stress_mask
        stress_mask1 = f1['labels']['stress_mask'][:]
        stress_mask2 = f2['labels']['stress_mask'][:]
        out['labels'].create_dataset(
            'stress_mask',
            data=np.concatenate([stress_mask1, stress_mask2])
        )

        # Create new structure indices
        indices = np.arange(total_structures, dtype=np.int64)
        out['labels'].create_dataset('structure_indices', data=indices)

        # Metadata
        logging.info("Creating metadata...")
        out['metadata'].attrs['num_structures'] = total_structures
        out['metadata'].attrs['total_atoms'] = total_atoms
        out['metadata'].attrs['avg_atoms_per_structure'] = total_atoms / total_structures
        out['metadata'].attrs['source1_name'] = str(source1.name)
        out['metadata'].attrs['source1_structures'] = n1
        out['metadata'].attrs['source2_name'] = str(source2.name)
        out['metadata'].attrs['source2_structures'] = n2
        out['metadata'].attrs['creation_time'] = str(np.datetime64('now'))

    logging.info("\n" + "="*60)
    logging.info("MERGE COMPLETE")
    logging.info("="*60)
    logging.info(f"Output file: {output}")
    logging.info(f"File size: {output.stat().st_size / 1024 / 1024:.2f} MB")
    logging.info(f"Total structures: {total_structures}")
    logging.info(f"Total atoms: {total_atoms}")
    logging.info(f"Avg atoms/structure: {total_atoms/total_structures:.1f}")
    logging.info("="*60)


def main():
    """Main entry point."""
    setup_logging()

    # Paths
    source1 = REPO_ROOT / "data/medium_scale_10k_moldiff/medium_scale_10k_moldiff.h5"
    source2 = REPO_ROOT / "data/medium_scale_10k_hybrid/medium_scale_10k_hybrid.h5"
    output = REPO_ROOT / "data/merged_dataset_4883/merged_dataset.h5"

    # Create output directory
    output.parent.mkdir(parents=True, exist_ok=True)

    # Merge
    merge_datasets(source1, source2, output)

    return 0


if __name__ == '__main__':
    sys.exit(main())
