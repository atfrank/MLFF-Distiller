#!/usr/bin/env python3
"""
CLI Utility for Converting Datasets to HDF5

This script provides a command-line interface for converting various dataset
formats (pickle, XYZ, etc.) to the HDF5 format used by MLFF_Distiller.

Usage:
    # Convert pickle file with separate labels
    python scripts/convert_to_hdf5.py \\
        --input data/raw/structures.pkl \\
        --output data/dataset.h5 \\
        --labels data/labels.pkl \\
        --compression gzip

    # Convert pickle file with labels in Atoms.calc
    python scripts/convert_to_hdf5.py \\
        --input structures.pkl \\
        --output dataset.h5

    # Convert XYZ file
    python scripts/convert_to_hdf5.py \\
        --input structures.xyz \\
        --output dataset.h5 \\
        --format xyz

Author: Data Pipeline Engineer
Date: 2025-11-23
"""

import sys
from pathlib import Path
import argparse
import pickle
import logging

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

import numpy as np
from ase.io import read
from mlff_distiller.data.hdf5_writer import HDF5DatasetWriter

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_structures(input_path, format='auto'):
    """
    Load structures from file.

    Args:
        input_path: Path to input file
        format: File format ('pickle', 'xyz', 'auto')

    Returns:
        List of ASE Atoms objects
    """
    input_path = Path(input_path)

    if format == 'auto':
        # Detect format from extension
        ext = input_path.suffix.lower()
        if ext == '.pkl':
            format = 'pickle'
        elif ext in ['.xyz', '.extxyz']:
            format = 'xyz'
        else:
            format = 'auto'

    if format == 'pickle':
        logger.info(f"Loading pickle file: {input_path}")
        with open(input_path, 'rb') as f:
            structures = pickle.load(f)
        logger.info(f"Loaded {len(structures)} structures")
        return structures

    else:
        # Use ASE to read
        logger.info(f"Loading {format} file: {input_path}")
        structures = read(input_path, index=':')
        if not isinstance(structures, list):
            structures = [structures]
        logger.info(f"Loaded {len(structures)} structures")
        return structures


def load_labels(labels_path):
    """
    Load labels from pickle file.

    Expected format: dict with 'energies', 'forces', 'stresses' keys
    Or: tuple of (energies, forces, stresses)

    Args:
        labels_path: Path to labels pickle file

    Returns:
        Tuple of (energies, forces, stresses)
    """
    logger.info(f"Loading labels from: {labels_path}")

    with open(labels_path, 'rb') as f:
        labels = pickle.load(f)

    if isinstance(labels, dict):
        energies = labels.get('energies')
        forces = labels.get('forces')
        stresses = labels.get('stresses', None)
    elif isinstance(labels, (tuple, list)):
        energies = labels[0]
        forces = labels[1]
        stresses = labels[2] if len(labels) > 2 else None
    else:
        raise ValueError(f"Unknown labels format: {type(labels)}")

    logger.info(f"Loaded {len(energies)} energies, {len(forces)} forces")
    return energies, forces, stresses


def extract_labels_from_calc(structures):
    """
    Extract labels from Atoms.calc if available.

    Args:
        structures: List of ASE Atoms objects

    Returns:
        Tuple of (energies, forces, stresses)
    """
    logger.info("Extracting labels from Atoms calculators...")

    energies = []
    forces = []
    stresses = []

    for i, atoms in enumerate(structures):
        if atoms.calc is None:
            raise ValueError(
                f"Structure {i} has no calculator. "
                "Please provide labels explicitly with --labels."
            )

        try:
            energies.append(atoms.get_potential_energy())
            forces.append(atoms.get_forces())

            if np.any(atoms.pbc):
                try:
                    stresses.append(atoms.get_stress())
                except:
                    stresses.append(None)
            else:
                stresses.append(None)

        except Exception as e:
            raise ValueError(f"Failed to extract labels from structure {i}: {e}")

    logger.info(f"Extracted {len(energies)} labels from calculators")
    return energies, forces, stresses


def convert_to_hdf5(
    input_path,
    output_path,
    labels_path=None,
    format='auto',
    compression='gzip',
    compression_level=4,
    metadata=None
):
    """
    Convert dataset to HDF5 format.

    Args:
        input_path: Path to input file
        output_path: Path to output HDF5 file
        labels_path: Optional path to labels pickle file
        format: Input file format
        compression: Compression algorithm
        compression_level: Compression level (1-9 for gzip)
        metadata: Optional metadata dict
    """
    # Load structures
    structures = load_structures(input_path, format=format)

    # Load or extract labels
    if labels_path is not None:
        energies, forces, stresses = load_labels(labels_path)

        # Validate lengths
        if len(energies) != len(structures):
            raise ValueError(
                f"Number of energies ({len(energies)}) != "
                f"number of structures ({len(structures)})"
            )
        if len(forces) != len(structures):
            raise ValueError(
                f"Number of forces ({len(forces)}) != "
                f"number of structures ({len(structures)})"
            )
    else:
        energies, forces, stresses = extract_labels_from_calc(structures)

    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to HDF5
    logger.info(f"Writing to HDF5: {output_path}")
    logger.info(f"Compression: {compression} (level {compression_level if compression == 'gzip' else 'default'})")

    with HDF5DatasetWriter(
        output_path,
        compression=compression,
        compression_opts=compression_level if compression == 'gzip' else None
    ) as writer:
        writer.add_batch(
            structures=structures,
            energies=energies,
            forces=forces,
            stresses=stresses,
            show_progress=True
        )

        # Add metadata if provided
        if metadata:
            writer.finalize(extra_metadata=metadata)
        else:
            writer.finalize()

    # Report statistics
    file_size = output_path.stat().st_size
    logger.info(f"\nConversion complete!")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Structures: {len(structures)}")
    logger.info(f"  Total atoms: {sum(len(atoms) for atoms in structures)}")
    logger.info(f"  File size: {file_size / 1024**2:.2f} MB")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert molecular/materials datasets to HDF5 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert pickle file with labels in Atoms.calc
  python scripts/convert_to_hdf5.py -i structures.pkl -o dataset.h5

  # Convert pickle with separate labels file
  python scripts/convert_to_hdf5.py -i structures.pkl -o dataset.h5 -l labels.pkl

  # Convert XYZ file
  python scripts/convert_to_hdf5.py -i data.xyz -o dataset.h5 -f xyz

  # Use LZF compression instead of GZIP
  python scripts/convert_to_hdf5.py -i structures.pkl -o dataset.h5 -c lzf

  # Add custom metadata
  python scripts/convert_to_hdf5.py -i structures.pkl -o dataset.h5 \\
      -m source=mattergen -m version=1.0
        """
    )

    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input file path (pickle, XYZ, etc.)'
    )

    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output HDF5 file path'
    )

    parser.add_argument(
        '-l', '--labels',
        help='Optional labels pickle file (if not in Atoms.calc)'
    )

    parser.add_argument(
        '-f', '--format',
        default='auto',
        choices=['auto', 'pickle', 'xyz', 'extxyz'],
        help='Input file format (default: auto-detect)'
    )

    parser.add_argument(
        '-c', '--compression',
        default='gzip',
        choices=['gzip', 'lzf', 'none'],
        help='Compression algorithm (default: gzip)'
    )

    parser.add_argument(
        '--compression-level',
        type=int,
        default=4,
        choices=range(1, 10),
        help='GZIP compression level 1-9 (default: 4)'
    )

    parser.add_argument(
        '-m', '--metadata',
        action='append',
        help='Metadata key=value pairs (can be used multiple times)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse metadata
    metadata = {}
    if args.metadata:
        for item in args.metadata:
            if '=' not in item:
                logger.warning(f"Invalid metadata format: {item} (expected key=value)")
                continue
            key, value = item.split('=', 1)
            metadata[key] = value

    # Convert compression
    compression = None if args.compression == 'none' else args.compression

    try:
        # Run conversion
        convert_to_hdf5(
            input_path=args.input,
            output_path=args.output,
            labels_path=args.labels,
            format=args.format,
            compression=compression,
            compression_level=args.compression_level,
            metadata=metadata if metadata else None
        )

        return 0

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
