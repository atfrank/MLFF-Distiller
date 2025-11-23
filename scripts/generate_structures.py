#!/usr/bin/env python
"""
Structure Generation CLI Tool

Command-line interface for generating diverse molecular and materials structures
for ML force field training.

Usage:
    # Generate 1000 test structures
    python scripts/generate_structures.py --output data/raw/test_structures --num-samples 1000

    # Generate full 120K dataset
    python scripts/generate_structures.py --output data/raw/full_dataset

    # Custom configuration
    python scripts/generate_structures.py --output data/raw/custom \\
        --num-samples 10000 --seed 123 --molecules 0.6 --crystals 0.3 --clusters 0.1

Author: Data Pipeline Engineer
Date: 2025-11-23
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlff_distiller.data.sampling import DiversityMetrics, SamplingConfig, SystemType
from mlff_distiller.data.structure_generation import StructureGenerator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate diverse molecular and materials structures"
    )

    # Output configuration
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for generated structures",
    )

    # Sampling configuration
    parser.add_argument(
        "--num-samples",
        type=int,
        default=120000,
        help="Total number of structures to generate (default: 120000)",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    # System type distribution
    parser.add_argument(
        "--molecules",
        type=float,
        default=0.50,
        help="Fraction of molecules (default: 0.50)",
    )

    parser.add_argument(
        "--crystals",
        type=float,
        default=0.33,
        help="Fraction of crystals (default: 0.33)",
    )

    parser.add_argument(
        "--clusters",
        type=float,
        default=0.10,
        help="Fraction of clusters (default: 0.10)",
    )

    parser.add_argument(
        "--surfaces",
        type=float,
        default=0.07,
        help="Fraction of surfaces (default: 0.07)",
    )

    # Element set
    parser.add_argument(
        "--elements",
        type=str,
        nargs="+",
        default=None,
        help="Allowed chemical elements (default: H C N O F S P Cl Si)",
    )

    # Size ranges
    parser.add_argument(
        "--molecule-size-range",
        type=int,
        nargs=2,
        default=[10, 100],
        help="Size range for molecules (default: 10 100)",
    )

    parser.add_argument(
        "--crystal-size-range",
        type=int,
        nargs=2,
        default=[50, 500],
        help="Size range for crystals (default: 50 500)",
    )

    parser.add_argument(
        "--cluster-size-range",
        type=int,
        nargs=2,
        default=[20, 200],
        help="Size range for clusters (default: 20 200)",
    )

    parser.add_argument(
        "--surface-size-range",
        type=int,
        nargs=2,
        default=[50, 300],
        help="Size range for surfaces (default: 50 300)",
    )

    # Validation options
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip diversity validation after generation",
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Print verbose output"
    )

    return parser.parse_args()


def create_config(args) -> SamplingConfig:
    """
    Create sampling configuration from command line arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        SamplingConfig object
    """
    # System distribution
    system_distribution = {
        SystemType.MOLECULE: args.molecules,
        SystemType.CRYSTAL: args.crystals,
        SystemType.CLUSTER: args.clusters,
        SystemType.SURFACE: args.surfaces,
    }

    # Size ranges
    size_ranges = {
        SystemType.MOLECULE: tuple(args.molecule_size_range),
        SystemType.CRYSTAL: tuple(args.crystal_size_range),
        SystemType.CLUSTER: tuple(args.cluster_size_range),
        SystemType.SURFACE: tuple(args.surface_size_range),
    }

    # Element set
    element_set = (
        set(args.elements) if args.elements else None
    )  # None uses default

    config = SamplingConfig(
        total_samples=args.num_samples,
        seed=args.seed,
        element_set=element_set,
        system_distribution=system_distribution,
        size_ranges=size_ranges,
    )

    return config


def validate_diversity(all_structures, config, output_dir):
    """
    Compute and save diversity metrics.

    Args:
        all_structures: Dictionary mapping SystemType to list of Atoms
        config: SamplingConfig
        output_dir: Path to output directory
    """
    print("\n" + "=" * 60)
    print("DIVERSITY VALIDATION")
    print("=" * 60)

    # Flatten structures
    all_atoms = []
    all_types = []
    for sys_type, structures in all_structures.items():
        all_atoms.extend(structures)
        all_types.extend([sys_type] * len(structures))

    # Compute metrics
    metrics = DiversityMetrics.compute_all_metrics(
        all_atoms, all_types, config.element_set
    )

    # Print metrics
    print("\nDiversity Metrics:")
    print(f"  Total structures: {len(all_atoms)}")
    print(f"  Element coverage: {metrics['element_coverage']:.2%}")
    print(f"  Composition entropy: {metrics['composition_entropy']:.2f} bits")
    print(f"  Mean system size: {metrics['mean_size']:.1f} atoms")
    print(f"  Size std dev: {metrics['std_size']:.1f} atoms")
    print(f"  Size CV: {metrics['size_cv']:.3f}")
    print(f"  Type balance CV: {metrics['type_balance_cv']:.3f}")

    # Per-type statistics
    print("\nPer-Type Statistics:")
    for sys_type, structures in all_structures.items():
        sizes = [len(atoms) for atoms in structures]
        if sizes:
            print(
                f"  {sys_type.value:12s}: {len(structures):6d} samples, "
                f"size = {min(sizes):3d}-{max(sizes):3d} atoms "
                f"(mean: {sum(sizes) / len(sizes):.1f})"
            )

    # Element distribution
    print("\nElement Distribution:")
    element_counts = {}
    for atoms in all_atoms:
        for symbol in atoms.get_chemical_symbols():
            element_counts[symbol] = element_counts.get(symbol, 0) + 1

    total_atoms = sum(element_counts.values())
    for elem in sorted(element_counts.keys()):
        count = element_counts[elem]
        frac = count / total_atoms
        print(f"  {elem:3s}: {count:8d} atoms ({frac:6.2%})")

    # Save metrics to JSON
    metrics_file = Path(output_dir) / "diversity_metrics.json"
    metrics_serializable = {k: float(v) for k, v in metrics.items()}
    metrics_serializable["total_structures"] = len(all_atoms)
    metrics_serializable["element_counts"] = element_counts

    with open(metrics_file, "w") as f:
        json.dump(metrics_serializable, f, indent=2)

    print(f"\nMetrics saved to: {metrics_file}")

    # Check if metrics meet targets
    warnings = []
    if metrics["element_coverage"] < 0.8:
        warnings.append(
            f"Low element coverage: {metrics['element_coverage']:.2%} < 80%"
        )
    if metrics["type_balance_cv"] > 0.3:
        warnings.append(
            f"Imbalanced system types: CV = {metrics['type_balance_cv']:.3f} > 0.3"
        )

    if warnings:
        print("\nWARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("\nAll diversity checks passed!")


def main():
    """Main entry point."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STRUCTURE GENERATION")
    print("=" * 60)

    # Create configuration
    config = create_config(args)

    print(f"\nConfiguration:")
    print(f"  Output directory: {output_dir}")
    print(f"  Total samples: {config.total_samples}")
    print(f"  Random seed: {config.seed}")
    print(f"  Elements: {sorted(config.element_set)}")

    print(f"\nSystem Distribution:")
    target_counts = config.get_sample_counts()
    for sys_type, count in target_counts.items():
        frac = config.system_distribution[sys_type]
        print(f"  {sys_type.value:12s}: {count:6d} samples ({frac:.1%})")

    print(f"\nSize Ranges:")
    for sys_type, (min_size, max_size) in config.size_ranges.items():
        print(f"  {sys_type.value:12s}: {min_size}-{max_size} atoms")

    # Save configuration
    config_file = output_dir / "sampling_config.json"
    config_dict = {
        "total_samples": config.total_samples,
        "seed": config.seed,
        "element_set": sorted(config.element_set),
        "system_distribution": {
            k.value: v for k, v in config.system_distribution.items()
        },
        "size_ranges": {
            k.value: list(v) for k, v in config.size_ranges.items()
        },
    }
    with open(config_file, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"\nConfiguration saved to: {config_file}")

    # Initialize generator
    print("\n" + "=" * 60)
    print("GENERATING STRUCTURES")
    print("=" * 60)

    generator = StructureGenerator(config)

    # Generate dataset
    all_structures = generator.generate_dataset(output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    total_generated = sum(len(structs) for structs in all_structures.values())
    print(f"\nGenerated {total_generated} structures:")
    for sys_type, structures in all_structures.items():
        print(f"  {sys_type.value:12s}: {len(structures)} structures")

    # Validate diversity
    if not args.skip_validation:
        validate_diversity(all_structures, config, output_dir)

    print("\n" + "=" * 60)
    print("SUCCESS")
    print("=" * 60)
    print(f"\nStructures saved to: {output_dir}")
    print(
        "\nNext steps:"
    )
    print(
        "  1. Run teacher model inference: python scripts/run_teacher_inference.py"
    )
    print(
        "  2. Create training dataset: python scripts/create_training_dataset.py"
    )


if __name__ == "__main__":
    main()
