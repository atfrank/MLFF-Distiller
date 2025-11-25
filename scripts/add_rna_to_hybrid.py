#!/usr/bin/env python3
"""
Add RNA Structures to Existing Hybrid Dataset

Adds 1,000 RNA structures to the existing hybrid HDF5 file that already contains
8,002 MolDiff molecules.

Author: Lead Coordinator
Date: 2025-11-24
"""

import os
import sys
import gzip
import random
import logging
import time
from pathlib import Path
from typing import List
import numpy as np
from ase.io import read

# Add src to path
REPO_ROOT = Path("/home/aaron/ATX/software/MLFF_Distiller")
sys.path.insert(0, str(REPO_ROOT / "src"))

from mlff_distiller.data.hdf5_writer import HDF5DatasetWriter
from mlff_distiller.models.teacher_wrappers import OrbCalculator


def setup_logging():
    """Configure logging."""
    log_file = REPO_ROOT / "logs/add_rna_to_hybrid.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def get_rna_files(rna_source: Path, num_structures: int = 1000) -> List[Path]:
    """Sample RNA PDB files."""
    logging.info(f"Scanning RNA structures from: {rna_source}")

    # Get all RNA systems
    rna_systems = [d for d in rna_source.iterdir() if d.is_dir() and not d.name.startswith('.')]
    logging.info(f"Found {len(rna_systems)} RNA systems")

    # Collect all PDB files
    all_pdb_files = []
    for system_dir in rna_systems:
        coord_dir = system_dir / "coordinates"
        if coord_dir.exists():
            pdb_files = list(coord_dir.glob("*.pdb.gz"))
            all_pdb_files.extend(pdb_files)

    logging.info(f"Found {len(all_pdb_files)} total RNA structures")

    # Sample desired number
    if len(all_pdb_files) > num_structures:
        sampled_files = random.sample(all_pdb_files, num_structures)
    else:
        sampled_files = all_pdb_files

    logging.info(f"Sampled {len(sampled_files)} RNA structures")
    return sampled_files


def add_rna_structures(hdf5_path: Path, rna_files: List[Path], teacher_calc):
    """Add RNA structures to existing HDF5 file."""
    logging.info(f"Opening HDF5 file: {hdf5_path}")

    # Open existing file for appending
    writer = HDF5DatasetWriter(str(hdf5_path), mode='append')

    success_count = 0
    failed_count = 0
    start_time = time.time()

    for i, file_path in enumerate(rna_files):
        try:
            # Load structure (decompress and read)
            with gzip.open(file_path, 'rt') as f:
                content = f.read()

            # Write to temporary file and read with ASE
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            try:
                atoms = read(tmp_path)
            finally:
                Path(tmp_path).unlink()

            # Label with teacher
            atoms.calc = teacher_calc
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()

            # Validate
            if np.isnan(energy) or np.isinf(energy):
                logging.warning(f"Invalid energy for {file_path.name}, skipping")
                failed_count += 1
                continue

            if np.any(np.isnan(forces)) or np.any(np.isinf(forces)):
                logging.warning(f"Invalid forces for {file_path.name}, skipping")
                failed_count += 1
                continue

            # Write to HDF5
            writer.add_structure(
                atoms=atoms,
                energy=energy,
                forces=forces,
                metadata={
                    'source': 'rna',
                    'source_file': str(file_path.name),
                    'num_atoms': len(atoms)
                }
            )

            success_count += 1

            # Progress logging
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = success_count / elapsed if elapsed > 0 else 0
                remaining = len(rna_files) - (i + 1)
                eta = remaining / rate if rate > 0 else 0
                logging.info(f"Progress: {i+1}/{len(rna_files)} processed, "
                           f"{success_count} successful, {failed_count} failed, "
                           f"Rate: {rate:.2f} struct/s, ETA: {eta/60:.1f} min")

        except Exception as e:
            logging.error(f"Failed to label {file_path.name}: {e}")
            failed_count += 1

    # Finalize
    writer.finalize()

    elapsed = time.time() - start_time
    logging.info("="*60)
    logging.info("RNA ADDITION COMPLETE")
    logging.info("="*60)
    logging.info(f"Successful: {success_count}")
    logging.info(f"Failed: {failed_count}")
    logging.info(f"Success rate: {100*success_count/(success_count+failed_count):.1f}%")
    logging.info(f"Total time: {elapsed/60:.1f} minutes")
    logging.info("="*60)

    return success_count, failed_count


def main():
    """Main entry point."""
    setup_logging()

    logging.info("="*60)
    logging.info("ADDING RNA STRUCTURES TO HYBRID DATASET")
    logging.info("="*60)

    # Paths
    hdf5_path = REPO_ROOT / "data/medium_scale_10k_hybrid/medium_scale_10k_hybrid.h5"
    rna_source = Path("/tmp/RNA-NMR-Decoys")

    # Initialize teacher calculator
    logging.info("Initializing orb-v2 teacher calculator...")
    teacher_calc = OrbCalculator(model_name="orb-v2", device="cuda")

    # Get RNA files
    rna_files = get_rna_files(rna_source, num_structures=1000)

    # Add to HDF5
    success, failed = add_rna_structures(hdf5_path, rna_files, teacher_calc)

    logging.info(f"\nFinal HDF5 file: {hdf5_path}")
    logging.info(f"File size: {hdf5_path.stat().st_size / 1024 / 1024:.1f} MB")

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
