#!/usr/bin/env python3
"""
MD Trajectory Benchmarking Script

This script provides a command-line interface for benchmarking teacher and student
calculators on molecular dynamics trajectories. It measures the performance metrics
critical for production MD use cases:

1. Per-call latency (mean, median, P95, P99)
2. Memory usage and stability
3. Energy conservation (for NVE)
4. Total trajectory execution time

Usage:
    # Benchmark single calculator
    python md_benchmark.py --calculator orb-v2 --system silicon --atoms 64 --steps 1000

    # Compare multiple calculators
    python md_benchmark.py --compare orb-v2 orb-v3 --system silicon --atoms 128

    # Run full benchmark suite
    python md_benchmark.py --suite baseline --output results/baseline_2025-11-23

    # Load and analyze existing results
    python md_benchmark.py --analyze results/baseline_2025-11-23/silicon_64_NVE.json

Author: Testing & Benchmark Engineer
Date: 2025-11-23
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlff_distiller.benchmarks import (
    MDProtocol,
    MDTrajectoryBenchmark,
    BenchmarkResults,
    compare_calculators,
    create_benchmark_report,
    plot_latency_distribution,
    plot_energy_conservation,
    plot_performance_comparison,
)


def load_calculator(calculator_name: str, device: str = "cuda"):
    """
    Load calculator by name.

    Args:
        calculator_name: Name of calculator ('orb-v1', 'orb-v2', 'orb-v3', etc.)
        device: Device to use ('cuda' or 'cpu')

    Returns:
        ASE Calculator instance
    """
    from mlff_distiller.models.teacher_wrappers import OrbCalculator, FeNNolCalculator

    # Map calculator names to implementations
    if calculator_name.startswith('orb'):
        return OrbCalculator(model_name=calculator_name, device=device)
    elif calculator_name.startswith('fennol') or calculator_name.startswith('ani'):
        return FeNNolCalculator(model_name=calculator_name, device=device)
    else:
        raise ValueError(f"Unknown calculator: {calculator_name}")


def run_single_benchmark(
    calculator_name: str,
    system_type: str,
    n_atoms: int,
    protocol: MDProtocol,
    n_steps: int,
    device: str,
    output_path: Optional[Path] = None,
) -> BenchmarkResults:
    """
    Run benchmark on single calculator.

    Args:
        calculator_name: Name of calculator
        system_type: Type of system ('silicon', 'water', etc.)
        n_atoms: Number of atoms
        protocol: MD protocol
        n_steps: Number of MD steps
        device: Device to use
        output_path: Optional path to save results

    Returns:
        BenchmarkResults object
    """
    print(f"\n{'=' * 80}")
    print(f"Running MD Trajectory Benchmark")
    print(f"{'=' * 80}")
    print(f"Calculator: {calculator_name}")
    print(f"System: {system_type} ({n_atoms} atoms)")
    print(f"Protocol: {protocol.value}")
    print(f"Steps: {n_steps}")
    print(f"Device: {device}")
    print(f"{'=' * 80}\n")

    # Load calculator
    calculator = load_calculator(calculator_name, device=device)

    # Create benchmark
    benchmark = MDTrajectoryBenchmark(
        calculator=calculator,
        system_type=system_type,
        n_atoms=n_atoms,
        protocol=protocol,
    )

    # Run benchmark
    results = benchmark.run(n_steps=n_steps)

    # Print results
    print(results.summary())

    # Save if requested
    if output_path:
        output_path = Path(output_path)
        results.save(output_path)
        print(f"\nResults saved to: {output_path}")

    return results


def run_comparison_benchmark(
    calculator_names: List[str],
    system_type: str,
    n_atoms: int,
    protocol: MDProtocol,
    n_steps: int,
    device: str,
    output_dir: Optional[Path] = None,
) -> Dict[str, BenchmarkResults]:
    """
    Compare multiple calculators on same trajectory.

    Args:
        calculator_names: List of calculator names
        system_type: Type of system
        n_atoms: Number of atoms
        protocol: MD protocol
        n_steps: Number of MD steps
        device: Device to use
        output_dir: Optional directory to save results

    Returns:
        Dictionary mapping calculator names to results
    """
    print(f"\n{'=' * 80}")
    print(f"Running Comparison Benchmark")
    print(f"{'=' * 80}")
    print(f"Calculators: {', '.join(calculator_names)}")
    print(f"System: {system_type} ({n_atoms} atoms)")
    print(f"Protocol: {protocol.value}")
    print(f"Steps: {n_steps}")
    print(f"Device: {device}")
    print(f"{'=' * 80}\n")

    # Load all calculators
    calculators = {
        name: load_calculator(name, device=device)
        for name in calculator_names
    }

    # Run comparison
    results = compare_calculators(
        calculators=calculators,
        system_type=system_type,
        n_atoms=n_atoms,
        protocol=protocol,
        n_steps=n_steps,
    )

    # Create comparison plots
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save individual results
        for name, result in results.items():
            result.save(output_dir / f"{name.replace(' ', '_')}.json")

        # Create comprehensive report
        create_benchmark_report(
            results=results,
            output_dir=output_dir,
            title=f"MD Benchmark Comparison: {system_type} ({n_atoms} atoms)",
        )

        print(f"\nComparison report saved to: {output_dir}")

    return results


def run_benchmark_suite(
    suite_name: str,
    device: str,
    output_dir: Path,
):
    """
    Run comprehensive benchmark suite.

    Args:
        suite_name: Name of suite ('baseline', 'quick', 'comprehensive')
        device: Device to use
        output_dir: Directory to save all results
    """
    print(f"\n{'=' * 80}")
    print(f"Running Benchmark Suite: {suite_name}")
    print(f"{'=' * 80}\n")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if suite_name == "quick":
        # Quick tests for CI/development
        configs = [
            ("silicon", 32, 100),
        ]
        calculators = ["orb-v2"]
        protocols = [MDProtocol.NVE]

    elif suite_name == "baseline":
        # Baseline characterization
        configs = [
            ("silicon", 64, 1000),
            ("silicon", 128, 1000),
            ("copper", 108, 1000),
        ]
        calculators = ["orb-v2"]
        protocols = [MDProtocol.NVE, MDProtocol.NVT]

    elif suite_name == "comprehensive":
        # Full benchmark suite
        configs = [
            ("silicon", 64, 1000),
            ("silicon", 128, 1000),
            ("silicon", 256, 1000),
            ("copper", 108, 1000),
            ("aluminum", 108, 1000),
        ]
        calculators = ["orb-v2", "orb-v3"]
        protocols = [MDProtocol.NVE, MDProtocol.NVT]

    else:
        raise ValueError(f"Unknown suite: {suite_name}")

    # Run all combinations
    all_results = {}

    for system_type, n_atoms, n_steps in configs:
        for protocol in protocols:
            suite_key = f"{system_type}_{n_atoms}_{protocol.value}"
            suite_dir = output_dir / suite_key

            print(f"\n--- Running: {suite_key} ---\n")

            results = run_comparison_benchmark(
                calculator_names=calculators,
                system_type=system_type,
                n_atoms=n_atoms,
                protocol=protocol,
                n_steps=n_steps,
                device=device,
                output_dir=suite_dir,
            )

            all_results[suite_key] = results

    print(f"\n{'=' * 80}")
    print(f"Benchmark Suite Complete!")
    print(f"{'=' * 80}")
    print(f"Results saved to: {output_dir}")
    print(f"Total benchmarks run: {len(all_results)}")


def analyze_results(results_path: Path):
    """
    Load and analyze saved benchmark results.

    Args:
        results_path: Path to saved results JSON file or directory
    """
    results_path = Path(results_path)

    if results_path.is_file():
        # Single result file
        result = BenchmarkResults.load(results_path)
        print(result.summary())

    elif results_path.is_dir():
        # Directory with multiple results
        json_files = list(results_path.glob("*.json"))

        if not json_files:
            print(f"No JSON files found in {results_path}")
            return

        results = {}
        for json_file in json_files:
            name = json_file.stem
            results[name] = BenchmarkResults.load(json_file)

        # Create comparison report
        create_benchmark_report(
            results=results,
            output_dir=results_path / "analysis",
            title="Benchmark Analysis",
        )

        print(f"\nAnalysis complete. Report saved to: {results_path / 'analysis'}")

    else:
        print(f"Path not found: {results_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MD Trajectory Benchmark Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single benchmark
  python md_benchmark.py --calculator orb-v2 --system silicon --atoms 64 --steps 1000

  # Compare calculators
  python md_benchmark.py --compare orb-v2 orb-v3 --system silicon --atoms 128

  # Run baseline suite
  python md_benchmark.py --suite baseline --output results/baseline

  # Analyze results
  python md_benchmark.py --analyze results/baseline/silicon_64_NVE
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--calculator",
        type=str,
        help="Run single calculator benchmark"
    )
    mode_group.add_argument(
        "--compare",
        nargs='+',
        metavar="CALC",
        help="Compare multiple calculators"
    )
    mode_group.add_argument(
        "--suite",
        choices=["quick", "baseline", "comprehensive"],
        help="Run benchmark suite"
    )
    mode_group.add_argument(
        "--analyze",
        type=Path,
        metavar="PATH",
        help="Analyze existing results"
    )

    # Benchmark parameters
    parser.add_argument(
        "--system",
        type=str,
        default="silicon",
        choices=["silicon", "copper", "aluminum", "water"],
        help="System type (default: silicon)"
    )
    parser.add_argument(
        "--atoms",
        type=int,
        default=64,
        help="Number of atoms (default: 64)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of MD steps (default: 1000)"
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default="NVE",
        choices=["NVE", "NVT", "NPT"],
        help="MD protocol (default: NVE)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for results"
    )

    args = parser.parse_args()

    # Convert protocol string to enum
    protocol = MDProtocol[args.protocol]

    # Execute based on mode
    if args.analyze:
        analyze_results(args.analyze)

    elif args.suite:
        if not args.output:
            args.output = Path(f"results/{args.suite}_benchmark")
        run_benchmark_suite(args.suite, args.device, args.output)

    elif args.compare:
        if not args.output:
            args.output = Path(f"results/comparison_{args.system}_{args.atoms}")
        run_comparison_benchmark(
            calculator_names=args.compare,
            system_type=args.system,
            n_atoms=args.atoms,
            protocol=protocol,
            n_steps=args.steps,
            device=args.device,
            output_dir=args.output,
        )

    elif args.calculator:
        if not args.output:
            args.output = Path(f"results/{args.calculator}_{args.system}_{args.atoms}.json")
        run_single_benchmark(
            calculator_name=args.calculator,
            system_type=args.system,
            n_atoms=args.atoms,
            protocol=protocol,
            n_steps=args.steps,
            device=args.device,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
