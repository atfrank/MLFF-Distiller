#!/usr/bin/env python3
"""
Teacher Model Profiling Script for MLFF Distiller

Profiles Orb-v2 and FeNNol teacher models on realistic MD trajectories to:
1. Establish baseline performance metrics
2. Identify computational hotspots
3. Measure memory stability over long runs
4. Generate profiling reports for optimization planning

This profiling focuses on MD-relevant metrics:
- Per-call latency (not batch throughput)
- Memory stability over 1000+ steps
- Component-level timing breakdown
- Sustained performance characteristics

Usage:
    # Profile Orb-v2 with default settings
    python benchmarks/profile_teachers.py --model orb-v2

    # Profile with 1000-step trajectory
    python benchmarks/profile_teachers.py --model orb-v2 --n-steps 1000

    # Profile multiple system sizes
    python benchmarks/profile_teachers.py --model orb-v2 --system-sizes 32,64,128,256

    # Compare models
    python benchmarks/profile_teachers.py --compare-all

    # Export PyTorch profiler traces
    python benchmarks/profile_teachers.py --model orb-v2 --export-traces

Results are saved to: benchmarks/profiling_reports/
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from ase import Atoms
from ase.build import bulk, molecule

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlff_distiller.cuda.device_utils import (
    get_device,
    print_device_summary,
    warmup_cuda,
)
from mlff_distiller.cuda.md_profiler import (
    MDProfiler,
    MDProfileResult,
    identify_hotspots,
)

# Import teacher calculators
try:
    from mlff_distiller.models.teacher_wrappers import OrbCalculator
    HAS_ORB = True
except ImportError as e:
    HAS_ORB = False
    print(f"Warning: orb-models not available: {e}")

try:
    from mlff_distiller.models.teacher_wrappers import FeNNolCalculator
    HAS_FENNOL = True
except ImportError as e:
    HAS_FENNOL = False
    print(f"Warning: FeNNol not available: {e}")


def generate_md_trajectory(
    system_type: str = "water",
    n_steps: int = 100,
    n_atoms: Optional[int] = None,
    temperature: float = 300.0,
) -> List[Atoms]:
    """
    Generate synthetic MD trajectory for profiling.

    In production, this would be replaced with real MD trajectories.
    For profiling purposes, we generate configurations with realistic
    atomic perturbations.

    Args:
        system_type: Type of system ('water', 'silicon', 'aluminum', 'iron')
        n_steps: Number of trajectory steps
        n_atoms: Target number of atoms (will create supercell)
        temperature: Temperature in Kelvin (affects perturbation magnitude)

    Returns:
        List of Atoms objects representing trajectory
    """
    # Create base structure
    if system_type == "water":
        base_atoms = molecule("H2O")
        # Replicate to get desired size
        if n_atoms is not None and n_atoms > 3:
            n_molecules = max(1, n_atoms // 3)
            # Create water box
            from ase import Atoms as AtomsBuilder
            positions = []
            symbols = []
            for i in range(n_molecules):
                # Simple cubic packing
                x = (i % 3) * 3.0
                y = ((i // 3) % 3) * 3.0
                z = (i // 9) * 3.0
                mol_pos = base_atoms.get_positions() + np.array([x, y, z])
                positions.extend(mol_pos)
                symbols.extend(['O', 'H', 'H'])
            base_atoms = Atoms(symbols=symbols, positions=positions)
            base_atoms.set_cell([9.0, 9.0, 9.0])
            base_atoms.set_pbc(True)
    elif system_type == "silicon":
        base_atoms = bulk("Si", "diamond", a=5.43, cubic=True)
        if n_atoms is not None and n_atoms > 8:
            # Calculate supercell size
            n = max(1, int(np.ceil((n_atoms / 8) ** (1/3))))
            base_atoms = base_atoms.repeat((n, n, n))
    elif system_type == "aluminum":
        base_atoms = bulk("Al", "fcc", a=4.05)
        if n_atoms is not None and n_atoms > 4:
            n = max(1, int(np.ceil((n_atoms / 4) ** (1/3))))
            base_atoms = base_atoms.repeat((n, n, n))
    elif system_type == "iron":
        base_atoms = bulk("Fe", "bcc", a=2.87)
        if n_atoms is not None and n_atoms > 2:
            n = max(1, int(np.ceil((n_atoms / 2) ** (1/3))))
            base_atoms = base_atoms.repeat((n, n, n))
    else:
        raise ValueError(f"Unknown system type: {system_type}")

    # Generate trajectory with thermal perturbations
    trajectory = []
    positions = base_atoms.get_positions()

    # Thermal displacement amplitude (rough estimate from equipartition)
    # sqrt(k_B * T / m) in Angstroms
    # Using rough average mass of 20 amu
    thermal_amplitude = 0.1 * np.sqrt(temperature / 300.0)

    for step in range(n_steps):
        # Add random thermal displacements
        displacement = np.random.randn(*positions.shape) * thermal_amplitude

        # Create perturbed atoms
        atoms = base_atoms.copy()
        atoms.set_positions(positions + displacement)

        trajectory.append(atoms)

    print(f"Generated {len(trajectory)}-step trajectory")
    print(f"  System: {system_type}")
    print(f"  Atoms: {len(trajectory[0])}")
    print(f"  Temperature: {temperature} K")

    return trajectory


def profile_orb_model(
    model_name: str,
    trajectory: List[Atoms],
    device: torch.device,
    output_dir: Path,
) -> Optional[MDProfileResult]:
    """Profile Orb model on trajectory."""
    if not HAS_ORB:
        print(f"\nSkipping {model_name}: orb-models not installed")
        return None

    print(f"\n{'=' * 80}")
    print(f"Profiling {model_name}")
    print(f"{'=' * 80}")

    try:
        # Create calculator
        calc = OrbCalculator(
            model_name=model_name,
            device=str(device),
        )

        # Create profiler
        profiler = MDProfiler(device=device)

        # Profile
        result = profiler.profile_calculator(
            calc,
            trajectory,
            properties=['energy', 'forces'],
            name=f"{model_name} on {len(trajectory[0])}-atom system",
        )

        # Print summary
        print(result.summary())

        # Identify hotspots
        hotspots = identify_hotspots(result)
        print("\nHotspot Analysis:")
        for component, info in hotspots.get('components', {}).items():
            status = "HOTSPOT" if info['is_hotspot'] else "ok"
            print(f"  {component:10s}: {info['time_ms']:8.4f} ms ({info['percentage']:5.1f}%) [{status}]")

        if hotspots['recommendations']:
            print("\nRecommendations:")
            for rec in hotspots['recommendations']:
                print(f"  - {rec}")

        # Save results
        output_file = output_dir / f"{model_name.replace('-', '_')}_profile.json"
        result.save_json(output_file)

        return result

    except Exception as e:
        print(f"Error profiling {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def profile_fennol_model(
    model_name: str,
    trajectory: List[Atoms],
    device: torch.device,
    output_dir: Path,
) -> Optional[MDProfileResult]:
    """Profile FeNNol model on trajectory."""
    if not HAS_FENNOL:
        print(f"\nSkipping {model_name}: FeNNol not installed")
        return None

    print(f"\n{'=' * 80}")
    print(f"Profiling {model_name}")
    print(f"{'=' * 80}")

    try:
        # Create calculator
        calc = FeNNolCalculator(
            model_name=model_name,
            device=str(device),
        )

        # Create profiler
        profiler = MDProfiler(device=device)

        # Profile
        result = profiler.profile_calculator(
            calc,
            trajectory,
            properties=['energy', 'forces'],
            name=f"{model_name} on {len(trajectory[0])}-atom system",
        )

        # Print summary
        print(result.summary())

        # Identify hotspots
        hotspots = identify_hotspots(result)
        print("\nHotspot Analysis:")
        for component, info in hotspots.get('components', {}).items():
            status = "HOTSPOT" if info['is_hotspot'] else "ok"
            print(f"  {component:10s}: {info['time_ms']:8.4f} ms ({info['percentage']:5.1f}%) [{status}]")

        if hotspots['recommendations']:
            print("\nRecommendations:")
            for rec in hotspots['recommendations']:
                print(f"  - {rec}")

        # Save results
        output_file = output_dir / f"{model_name.replace('-', '_')}_profile.json"
        result.save_json(output_file)

        return result

    except Exception as e:
        print(f"Error profiling {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def profile_system_size_scaling(
    model_type: str,
    model_name: str,
    system_sizes: List[int],
    n_steps: int,
    device: torch.device,
    output_dir: Path,
) -> None:
    """Profile model across different system sizes."""
    print(f"\n{'=' * 80}")
    print(f"System Size Scaling Analysis: {model_name}")
    print(f"{'=' * 80}")

    results = []

    for n_atoms in system_sizes:
        print(f"\n\nProfiling {n_atoms}-atom system...")

        # Generate trajectory
        trajectory = generate_md_trajectory(
            system_type="silicon",
            n_steps=n_steps,
            n_atoms=n_atoms,
        )

        # Profile based on model type
        if model_type == "orb":
            result = profile_orb_model(model_name, trajectory, device, output_dir)
        elif model_type == "fennol":
            result = profile_fennol_model(model_name, trajectory, device, output_dir)
        else:
            continue

        if result is not None:
            results.append((n_atoms, result))

    # Print scaling summary
    if results:
        print("\n" + "=" * 100)
        print("System Size Scaling Summary")
        print("=" * 100)
        print(f"{'Atoms':>10} {'Mean (ms)':>12} {'µs/atom':>12} {'Memory (GB)':>12} {'Steps/s':>12}")
        print("-" * 100)

        for n_atoms, result in results:
            us_per_atom = (result.mean_latency_ms * 1000) / n_atoms if n_atoms > 0 else 0
            print(f"{n_atoms:>10} {result.mean_latency_ms:>12.4f} {us_per_atom:>12.2f} "
                  f"{result.memory_peak_gb:>12.4f} {result.steps_per_second:>12.1f}")

        print("=" * 100 + "\n")


def compare_all_models(
    trajectory: List[Atoms],
    device: torch.device,
    output_dir: Path,
) -> None:
    """Compare all available teacher models."""
    print(f"\n{'=' * 80}")
    print("Comparing All Teacher Models")
    print(f"{'=' * 80}")

    results = {}

    # Profile Orb models
    if HAS_ORB:
        for model_name in ["orb-v2"]:  # Can add orb-v1, orb-v3 if needed
            result = profile_orb_model(model_name, trajectory, device, output_dir)
            if result is not None:
                results[model_name] = result

    # Profile FeNNol models
    if HAS_FENNOL:
        for model_name in ["ani-2x"]:  # Add other FeNNol models if available
            result = profile_fennol_model(model_name, trajectory, device, output_dir)
            if result is not None:
                results[model_name] = result

    # Print comparison
    if len(results) > 1:
        print("\n" + "=" * 100)
        print("Model Comparison")
        print("=" * 100)
        print(f"{'Model':<20} {'Mean (ms)':>12} {'P95 (ms)':>12} {'Steps/s':>12} "
              f"{'Memory (GB)':>12} {'Speedup':>10}")
        print("-" * 100)

        # Use first model as baseline
        baseline_name = list(results.keys())[0]
        baseline_mean = results[baseline_name].mean_latency_ms

        for name, result in results.items():
            speedup = baseline_mean / result.mean_latency_ms if result.mean_latency_ms > 0 else 0
            speedup_str = f"{speedup:.2f}x"

            print(f"{name:<20} {result.mean_latency_ms:>12.4f} {result.p95_latency_ms:>12.4f} "
                  f"{result.steps_per_second:>12.1f} {result.memory_peak_gb:>12.4f} {speedup_str:>10}")

        print("=" * 100 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Profile teacher models on MD trajectories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile Orb-v2 with default settings
  python profile_teachers.py --model orb-v2

  # Profile with long trajectory
  python profile_teachers.py --model orb-v2 --n-steps 1000

  # System size scaling
  python profile_teachers.py --model orb-v2 --system-sizes 32,64,128,256

  # Compare all models
  python profile_teachers.py --compare-all

Results saved to: benchmarks/profiling_reports/
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["orb-v2", "orb-v1", "orb-v3", "ani-2x"],
        help="Model to profile (default: orb-v2)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=100,
        help="Number of trajectory steps (default: 100)",
    )
    parser.add_argument(
        "--system",
        type=str,
        default="silicon",
        choices=["water", "silicon", "aluminum", "iron"],
        help="System type (default: silicon)",
    )
    parser.add_argument(
        "--n-atoms",
        type=int,
        default=64,
        help="Target number of atoms (default: 64)",
    )
    parser.add_argument(
        "--system-sizes",
        type=str,
        help="Comma-separated list of system sizes for scaling analysis (e.g., 32,64,128)",
    )
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Compare all available teacher models",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/profiling_reports",
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )

    args = parser.parse_args()

    # Setup
    print_device_summary()

    device = get_device(device=args.device)
    print(f"\nUsing device: {device}")

    if device.type == 'cuda':
        warmup_cuda(device=device)
        print("CUDA warmup complete")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults will be saved to: {output_dir}")

    try:
        # System size scaling analysis
        if args.system_sizes:
            system_sizes = [int(x.strip()) for x in args.system_sizes.split(',')]
            model_type = "orb" if args.model and "orb" in args.model else "fennol"
            model_name = args.model or "orb-v2"

            profile_system_size_scaling(
                model_type,
                model_name,
                system_sizes,
                args.n_steps,
                device,
                output_dir,
            )

        # Compare all models
        elif args.compare_all:
            trajectory = generate_md_trajectory(
                system_type=args.system,
                n_steps=args.n_steps,
                n_atoms=args.n_atoms,
            )
            compare_all_models(trajectory, device, output_dir)

        # Profile single model
        elif args.model:
            trajectory = generate_md_trajectory(
                system_type=args.system,
                n_steps=args.n_steps,
                n_atoms=args.n_atoms,
            )

            if "orb" in args.model:
                profile_orb_model(args.model, trajectory, device, output_dir)
            elif "ani" in args.model:
                profile_fennol_model(args.model, trajectory, device, output_dir)
            else:
                print(f"Unknown model: {args.model}")

        else:
            print("No profiling task specified. Use --model, --system-sizes, or --compare-all")
            parser.print_help()
            return

        print("\n" + "=" * 80)
        print("Profiling Complete!")
        print("=" * 80)
        print(f"\nResults saved to: {output_dir}")
        print("\nNext Steps:")
        print("1. Review profiling reports in profiling_reports/")
        print("2. Identify optimization opportunities")
        print("3. Use insights to guide CUDA kernel development (M4-M5)")
        print("4. Set baseline targets for student model performance")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\nError during profiling: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
