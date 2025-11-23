#!/usr/bin/env python
"""
Production CLI Tool for Teacher Model Label Generation

This script provides a command-line interface for generating energy, force, and stress
labels from teacher models for knowledge distillation.

Features:
- Supports multiple input formats (XYZ, EXTXYZ, LMDB, HDF5)
- GPU acceleration with configurable device
- Batch processing for large datasets
- Progress tracking and logging
- Error handling and recovery
- HDF5 output compatible with distillation pipeline
- Resume capability for interrupted runs

Usage:
    # Generate labels from XYZ file using Orb-v2
    python scripts/generate_labels.py \\
        --input data/structures.xyz \\
        --output data/labels.h5 \\
        --teacher-model orb-v2 \\
        --device cuda \\
        --batch-size 32

    # Resume interrupted run
    python scripts/generate_labels.py \\
        --input data/structures.xyz \\
        --output data/labels.h5 \\
        --teacher-model orb-v2 \\
        --device cuda \\
        --resume

    # Process LMDB dataset
    python scripts/generate_labels.py \\
        --input data/structures.lmdb \\
        --output data/labels.h5 \\
        --teacher-model orb-v3 \\
        --device cuda:0

Examples:
    # Process 120K structures from MatBench
    python scripts/generate_labels.py \\
        --input data/matbench_discovery/structures.lmdb \\
        --output data/matbench_labels.h5 \\
        --teacher-model orb-v2 \\
        --device cuda \\
        --progress

Author: ML Architecture Designer
Date: 2025-11-23
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from ase import Atoms
from ase.io import read, iread

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mlff_distiller.data.label_generation import LabelGenerator, LabelResult


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('label_generation.log')
    ]
)
logger = logging.getLogger(__name__)


def load_structures(
    input_path: Path,
    input_format: Optional[str] = None,
    max_structures: Optional[int] = None,
    resume_from: int = 0
) -> List[Atoms]:
    """
    Load atomic structures from various file formats.

    Args:
        input_path: Path to input file
        input_format: Format string for ASE reader (auto-detected if None)
        max_structures: Maximum number of structures to load
        resume_from: Skip first N structures (for resuming)

    Returns:
        List of ASE Atoms objects
    """
    logger.info(f"Loading structures from {input_path}")

    # Handle different formats
    if input_path.suffix == '.lmdb':
        # LMDB format (e.g., from Open Catalyst)
        try:
            from lmdb_reader import LMDBReader  # Custom reader if needed
            structures = LMDBReader(input_path).read_all()
        except ImportError:
            logger.error("LMDB support requires custom reader. Use HDF5 or XYZ instead.")
            raise
    else:
        # Use ASE reader for standard formats (XYZ, EXTXYZ, CIF, etc.)
        structures = []
        for i, atoms in enumerate(iread(str(input_path), format=input_format)):
            # Skip if resuming
            if i < resume_from:
                continue

            structures.append(atoms)

            # Check limit
            if max_structures is not None and len(structures) >= max_structures:
                break

            # Progress logging
            if (i + 1) % 1000 == 0:
                logger.info(f"Loaded {len(structures)} structures...")

    logger.info(f"Successfully loaded {len(structures)} structures")
    return structures


def check_cuda_availability(device: str) -> str:
    """
    Check if requested CUDA device is available.

    Args:
        device: Requested device string

    Returns:
        Valid device string (may fall back to CPU)
    """
    if 'cuda' in device:
        import torch
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return 'cpu'

        # Check specific device
        if ':' in device:
            device_id = int(device.split(':')[1])
            if device_id >= torch.cuda.device_count():
                logger.warning(
                    f"CUDA device {device_id} not available. "
                    f"Using cuda:0 instead."
                )
                return 'cuda:0'

    return device


def check_existing_output(output_path: Path, resume: bool) -> Optional[int]:
    """
    Check if output file exists and handle resume logic.

    Args:
        output_path: Path to output HDF5 file
        resume: Whether to resume from existing file

    Returns:
        Number of structures already processed (for resume), or None for fresh start
    """
    if not output_path.exists():
        return None

    if not resume:
        logger.warning(f"Output file {output_path} already exists!")
        response = input("Overwrite? (y/N): ").strip().lower()
        if response != 'y':
            logger.info("Aborting to avoid overwriting existing file.")
            sys.exit(0)
        return None

    # Resume mode: read existing file
    try:
        import h5py
        with h5py.File(output_path, 'r') as f:
            if 'labels' in f and 'structure_indices' in f['labels']:
                indices = f['labels']['structure_indices'][:]
                n_processed = len(indices)
                logger.info(f"Resuming from existing file with {n_processed} structures")
                return n_processed
    except Exception as e:
        logger.warning(f"Could not read existing file: {e}. Starting fresh.")

    return None


def main():
    """Main entry point for label generation CLI."""
    parser = argparse.ArgumentParser(
        description="Generate labels from teacher models for knowledge distillation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input/output arguments
    parser.add_argument(
        '--input', '-i',
        type=Path,
        required=True,
        help='Path to input structures (XYZ, EXTXYZ, LMDB, HDF5)'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        required=True,
        help='Path to output HDF5 file for labels'
    )
    parser.add_argument(
        '--input-format',
        type=str,
        default=None,
        help='Format for ASE reader (auto-detected if not specified)'
    )

    # Model arguments
    parser.add_argument(
        '--teacher-model', '-m',
        type=str,
        default='orb-v2',
        choices=['orb-v1', 'orb-v2', 'orb-v3', 'fennol'],
        help='Teacher model to use for label generation'
    )
    parser.add_argument(
        '--device', '-d',
        type=str,
        default='cuda',
        help='Device to run on (cpu, cuda, cuda:0, etc.)'
    )
    parser.add_argument(
        '--precision',
        type=str,
        default='float32-high',
        choices=['float32-high', 'float32-highest', 'float64'],
        help='Precision mode for Orb models'
    )

    # Processing arguments
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=1,
        help='Batch size for processing (most models use batch_size=1)'
    )
    parser.add_argument(
        '--max-structures', '-n',
        type=int,
        default=None,
        help='Maximum number of structures to process'
    )
    parser.add_argument(
        '--resume', '-r',
        action='store_true',
        help='Resume from existing output file'
    )
    parser.add_argument(
        '--skip-errors',
        action='store_true',
        default=True,
        help='Continue processing if a structure fails (default: True)'
    )
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bar'
    )
    parser.add_argument(
        '--compression',
        type=str,
        default='gzip',
        choices=['gzip', 'lzf', 'none'],
        help='HDF5 compression algorithm'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Check CUDA availability
    device = check_cuda_availability(args.device)

    # Check for existing output and handle resume
    resume_from = check_existing_output(args.output, args.resume)
    if resume_from is not None and resume_from > 0:
        logger.info(f"Resuming: skipping first {resume_from} structures")

    # Load structures
    try:
        structures = load_structures(
            args.input,
            input_format=args.input_format,
            max_structures=args.max_structures,
            resume_from=resume_from or 0
        )
    except Exception as e:
        logger.error(f"Failed to load structures: {e}")
        sys.exit(1)

    if len(structures) == 0:
        logger.error("No structures to process!")
        sys.exit(1)

    # Initialize label generator
    try:
        logger.info(f"Initializing {args.teacher_model} on {device}")
        generator = LabelGenerator(
            teacher_model=args.teacher_model,
            device=device,
            batch_size=args.batch_size,
            precision=args.precision
        )
    except Exception as e:
        logger.error(f"Failed to initialize teacher model: {e}")
        sys.exit(1)

    # Generate labels
    logger.info(f"Starting label generation for {len(structures)} structures")
    try:
        results = generator.generate_labels(
            structures,
            progress=not args.no_progress,
            skip_errors=args.skip_errors
        )
    except KeyboardInterrupt:
        logger.warning("Label generation interrupted by user")
        logger.info("Partial results will be saved")
        # Continue to save partial results
        results = [LabelResult(success=False) for _ in structures]
    except Exception as e:
        logger.error(f"Label generation failed: {e}")
        sys.exit(1)

    # Save results
    try:
        compression = None if args.compression == 'none' else args.compression
        generator.save_results(
            args.output,
            results,
            structures=structures,
            compression=compression
        )
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        sys.exit(1)

    # Print summary
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    logger.info(f"\n{'='*60}")
    logger.info(f"Label Generation Complete!")
    logger.info(f"  Total structures: {len(results)}")
    logger.info(f"  Successful: {successful} ({100*successful/len(results):.1f}%)")
    logger.info(f"  Failed: {failed} ({100*failed/len(results):.1f}%)")
    logger.info(f"  Output saved to: {args.output}")
    logger.info(f"{'='*60}\n")


if __name__ == '__main__':
    main()
