#!/usr/bin/env python3
"""
Command-line tool for validating molecular dynamics datasets.

This script provides comprehensive validation of dataset quality including:
- Structure validation (geometry, distances, cells)
- Label validation (energies, forces, outliers)
- Diversity validation (element coverage, size distribution)
- Statistical validation (distributions, outliers)

Example usage:
    # Validate ASE database
    python validate_dataset.py structures.db --format ase

    # Validate HDF5 file with custom thresholds
    python validate_dataset.py data.h5 --format hdf5 --max-force 50.0

    # Validate and save detailed report
    python validate_dataset.py data.db --output report.json --save-stats

    # Quick validation (100 samples)
    python validate_dataset.py large_dataset.db --max-samples 100
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from mlff_distiller.data.dataset import MolecularDataset
from mlff_distiller.data.validation import (
    DatasetValidator,
    DiversityValidator,
    LabelValidator,
    StructureValidator,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Validate molecular dynamics dataset quality',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic validation
  %(prog)s structures.db --format ase

  # Custom thresholds
  %(prog)s data.h5 --format hdf5 --min-distance 0.3 --max-force 80.0

  # Save detailed report
  %(prog)s data.db --output report.json --save-stats stats.txt

  # Quick check (first 100 samples)
  %(prog)s large_data.db --max-samples 100
        """
    )

    # Required arguments
    parser.add_argument(
        'data_path',
        type=str,
        help='Path to dataset file (ASE database, HDF5, or XYZ)'
    )

    # Dataset format
    parser.add_argument(
        '--format',
        type=str,
        choices=['ase', 'hdf5', 'xyz'],
        default='ase',
        help='Dataset format (default: ase)'
    )

    # Validation options
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to validate (default: all)'
    )

    parser.add_argument(
        '--no-fail-on-error',
        action='store_true',
        help='Do not fail validation if errors are found (still report them)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )

    # Structure validation parameters
    struct_group = parser.add_argument_group('structure validation')
    struct_group.add_argument(
        '--min-distance',
        type=float,
        default=0.5,
        help='Minimum allowed interatomic distance in Angstrom (default: 0.5)'
    )

    struct_group.add_argument(
        '--min-atoms',
        type=int,
        default=10,
        help='Minimum number of atoms in a system (default: 10)'
    )

    struct_group.add_argument(
        '--max-atoms',
        type=int,
        default=500,
        help='Maximum number of atoms in a system (default: 500)'
    )

    # Label validation parameters
    label_group = parser.add_argument_group('label validation')
    label_group.add_argument(
        '--max-force',
        type=float,
        default=100.0,
        help='Maximum allowed force magnitude in eV/Angstrom (default: 100.0)'
    )

    label_group.add_argument(
        '--energy-min',
        type=float,
        default=-50.0,
        help='Minimum energy per atom in eV (default: -50.0)'
    )

    label_group.add_argument(
        '--energy-max',
        type=float,
        default=50.0,
        help='Maximum energy per atom in eV (default: 50.0)'
    )

    label_group.add_argument(
        '--outlier-threshold',
        type=float,
        default=3.0,
        help='Number of standard deviations for outlier detection (default: 3.0)'
    )

    # Diversity validation parameters
    diversity_group = parser.add_argument_group('diversity validation')
    diversity_group.add_argument(
        '--min-element-samples',
        type=int,
        default=10,
        help='Minimum samples per element type (default: 10)'
    )

    diversity_group.add_argument(
        '--min-size-bins',
        type=int,
        default=3,
        help='Minimum number of different system size bins (default: 3)'
    )

    # Output options
    output_group = parser.add_argument_group('output options')
    output_group.add_argument(
        '--output',
        '-o',
        type=str,
        default=None,
        help='Path to save JSON validation report'
    )

    output_group.add_argument(
        '--save-stats',
        type=str,
        default=None,
        help='Path to save detailed statistics text file'
    )

    output_group.add_argument(
        '--show-all-issues',
        action='store_true',
        help='Show all issues (including info level)'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Validate input path
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"ERROR: Data path not found: {data_path}", file=sys.stderr)
        return 1

    # Print header
    if not args.quiet:
        print("=" * 80)
        print("DATASET VALIDATION")
        print("=" * 80)
        print(f"Data path: {data_path}")
        print(f"Format: {args.format}")
        print(f"Max samples: {args.max_samples or 'all'}")
        print()

    # Load dataset
    try:
        if not args.quiet:
            print("Loading dataset...")
        dataset = MolecularDataset(
            data_path,
            format=args.format,
            cache=False,  # Don't cache during validation
            return_atoms=True,
        )
        if not args.quiet:
            print(f"  Loaded {len(dataset)} samples")
            print()
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}", file=sys.stderr)
        return 1

    # Create validators with custom parameters
    structure_validator = StructureValidator(
        min_distance=args.min_distance,
        min_atoms=args.min_atoms,
        max_atoms=args.max_atoms,
    )

    label_validator = LabelValidator(
        max_force=args.max_force,
        energy_range=(args.energy_min, args.energy_max),
        outlier_threshold=args.outlier_threshold,
    )

    diversity_validator = DiversityValidator(
        min_element_samples=args.min_element_samples,
        min_size_bins=args.min_size_bins,
    )

    # Create main validator
    validator = DatasetValidator(
        structure_validator=structure_validator,
        label_validator=label_validator,
        diversity_validator=diversity_validator,
        fail_on_error=not args.no_fail_on_error,
        verbose=not args.quiet,
    )

    # Run validation
    try:
        report = validator.validate_dataset(dataset, max_samples=args.max_samples)
    except Exception as e:
        print(f"ERROR: Validation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    # Print report summary
    if not args.quiet:
        print()
        print(report.summary())
    else:
        # Print minimal summary even in quiet mode
        status = "PASSED" if report.passed else "FAILED"
        print(f"Validation {status}: {report.num_errors} errors, "
              f"{report.num_warnings} warnings, {report.num_info} info")

    # Print issues
    if report.issues:
        if args.show_all_issues:
            print("\nALL ISSUES:")
            print("-" * 80)
            for issue in report.issues:
                print(f"  {issue}")
        else:
            # Show only errors and warnings
            errors_warnings = [i for i in report.issues
                             if i.severity in ['error', 'warning']]
            if errors_warnings:
                print("\nERRORS AND WARNINGS:")
                print("-" * 80)
                for issue in errors_warnings:
                    print(f"  {issue}")

    # Save JSON report if requested
    if args.output:
        output_path = Path(args.output)
        if not args.quiet:
            print(f"\nSaving JSON report to {output_path}...")

        try:
            with open(output_path, 'w') as f:
                json.dump(report.to_dict(), f, indent=2)
            if not args.quiet:
                print("  Report saved successfully")
        except Exception as e:
            print(f"ERROR: Failed to save report: {e}", file=sys.stderr)

    # Save detailed statistics if requested
    if args.save_stats:
        stats_path = Path(args.save_stats)
        if not args.quiet:
            print(f"\nSaving statistics to {stats_path}...")

        try:
            with open(stats_path, 'w') as f:
                f.write("DATASET STATISTICS\n")
                f.write("=" * 80 + "\n\n")

                for key, value in sorted(report.statistics.items()):
                    if isinstance(value, dict):
                        f.write(f"{key}:\n")
                        for k, v in value.items():
                            f.write(f"  {k}: {v}\n")
                    elif isinstance(value, float):
                        f.write(f"{key}: {value:.6f}\n")
                    else:
                        f.write(f"{key}: {value}\n")

            if not args.quiet:
                print("  Statistics saved successfully")
        except Exception as e:
            print(f"ERROR: Failed to save statistics: {e}", file=sys.stderr)

    # Return exit code
    if report.passed:
        if not args.quiet:
            print("\nValidation PASSED")
        return 0
    else:
        if not args.quiet:
            print(f"\nValidation FAILED ({report.num_errors} errors)")
        return 1


if __name__ == '__main__':
    sys.exit(main())
