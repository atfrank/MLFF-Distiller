#!/usr/bin/env python
"""
Structure Inspection Tool

Utility script to inspect and validate generated molecular structures.

Usage:
    python scripts/inspect_structures.py data/raw/test_structures

Author: Data Pipeline Engineer
Date: 2025-11-23
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.visualize import view


def load_structures(data_dir: Path):
    """Load all structure files from directory."""
    structures_by_type = {}

    for pkl_file in data_dir.glob("*_structures.pkl"):
        sys_type = pkl_file.stem.replace("_structures", "")
        with open(pkl_file, "rb") as f:
            structures = pickle.load(f)
        structures_by_type[sys_type] = structures
        print(f"Loaded {len(structures)} {sys_type} structures")

    return structures_by_type


def analyze_structure(atoms: Atoms):
    """Analyze single structure."""
    info = {
        "n_atoms": len(atoms),
        "elements": list(set(atoms.get_chemical_symbols())),
        "formula": atoms.get_chemical_formula(),
        "periodic": atoms.pbc.tolist(),
        "volume": atoms.get_volume() if atoms.pbc.any() else None,
    }

    # Compute interatomic distances
    positions = atoms.get_positions()
    if len(positions) > 1:
        dists = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                dists.append(dist)

        info["min_distance"] = min(dists)
        info["max_distance"] = max(dists)
        info["mean_distance"] = np.mean(dists)

    return info


def print_summary(structures_by_type):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("STRUCTURE SUMMARY")
    print("=" * 70)

    total_structures = sum(len(s) for s in structures_by_type.values())
    total_atoms = sum(sum(len(a) for a in s) for s in structures_by_type.values())

    print(f"\nTotal structures: {total_structures}")
    print(f"Total atoms: {total_atoms:,}")

    for sys_type, structures in structures_by_type.items():
        print(f"\n{sys_type.upper()} Structures:")
        print(f"  Count: {len(structures)}")

        if len(structures) > 0:
            sizes = [len(a) for a in structures]
            print(f"  Size range: {min(sizes)}-{max(sizes)} atoms")
            print(f"  Mean size: {np.mean(sizes):.1f} atoms")

            # Element diversity
            all_elements = set()
            for atoms in structures:
                all_elements.update(atoms.get_chemical_symbols())
            print(f"  Elements present: {sorted(all_elements)}")

            # Periodicity
            n_periodic = sum(a.pbc.any() for a in structures)
            print(f"  Periodic: {n_periodic}/{len(structures)}")


def inspect_structure(atoms: Atoms, idx: int, sys_type: str):
    """Print detailed info about a structure."""
    info = analyze_structure(atoms)

    print(f"\n{sys_type.upper()} Structure #{idx}:")
    print(f"  Formula: {info['formula']}")
    print(f"  Atoms: {info['n_atoms']}")
    print(f"  Elements: {info['elements']}")
    print(f"  Periodic: {info['periodic']}")

    if info["volume"] is not None:
        print(f"  Volume: {info['volume']:.2f} Ų")

    if "min_distance" in info:
        print(f"  Min distance: {info['min_distance']:.3f} Å")
        print(f"  Max distance: {info['max_distance']:.3f} Å")
        print(f"  Mean distance: {info['mean_distance']:.3f} Å")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect generated molecular structures"
    )
    parser.add_argument("data_dir", type=str, help="Directory containing structure files")
    parser.add_argument(
        "--detailed",
        type=int,
        default=3,
        help="Number of structures to inspect in detail (default: 3)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize structures using ASE viewer",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["molecule", "crystal", "cluster", "surface"],
        help="Specific structure type to inspect",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"Error: Directory {data_dir} not found")
        return

    # Load structures
    print("Loading structures...")
    structures_by_type = load_structures(data_dir)

    # Print summary
    print_summary(structures_by_type)

    # Filter by type if specified
    if args.type:
        structures_by_type = {args.type: structures_by_type[args.type]}

    # Detailed inspection
    if args.detailed > 0:
        print("\n" + "=" * 70)
        print("DETAILED INSPECTION")
        print("=" * 70)

        for sys_type, structures in structures_by_type.items():
            n_inspect = min(args.detailed, len(structures))
            for i in range(n_inspect):
                inspect_structure(structures[i], i, sys_type)

    # Visualization
    if args.visualize:
        print("\nLaunching ASE viewer...")
        for sys_type, structures in structures_by_type.items():
            if len(structures) > 0:
                print(f"\nViewing {sys_type} structure #0")
                view(structures[0])


if __name__ == "__main__":
    main()
