"""
Comprehensive Benchmark of All Optimization Strategies

This script benchmarks the StudentForceField model with various optimization
strategies to measure speedup and accuracy:

1. Baseline: Standard PyTorch inference (FP32)
2. FP16: Mixed precision inference (autocast)
3. TorchScript: JIT-compiled model
4. TorchScript + FP16: Combined optimization
5. torch.compile: PyTorch 2.0 compiler (if supported)

Metrics:
- Inference latency (ms)
- Throughput (structures/sec)
- Memory usage (MB)
- Accuracy (energy/force errors vs baseline)

Usage:
    python scripts/benchmark_optimizations.py \\
        --checkpoint checkpoints/best_model.pt \\
        --jit-model models/student_model_jit.pt \\
        --num-iterations 100 \\
        --system-sizes 10,20,50,100

Author: CUDA Optimization Engineer
Date: 2025-11-24
Issue: M3 #24 - TensorRT Optimization (Comprehensive Benchmarking)
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import time
import json

import torch
import numpy as np
from ase import Atoms
from ase.build import molecule

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mlff_distiller.inference.ase_calculator import StudentForceFieldCalculator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_molecules(system_sizes: List[int]) -> Dict[int, Atoms]:
    """
    Create test molecules of various sizes.

    Args:
        system_sizes: List of system sizes (number of atoms)

    Returns:
        Dictionary mapping size to Atoms object
    """
    molecules = {}

    for size in system_sizes:
        # Create random molecule
        # Use simple approach: random atoms in a box
        symbols = ['H', 'C', 'N', 'O'] * (size // 4 + 1)
        symbols = symbols[:size]

        positions = np.random.randn(size, 3) * 3.0  # Random positions in ~3Å radius

        atoms = Atoms(symbols=symbols, positions=positions)
        molecules[size] = atoms

    return molecules


def benchmark_configuration(
    config_name: str,
    calculator: StudentForceFieldCalculator,
    molecules: Dict[int, Atoms],
    num_iterations: int = 100,
    warmup: int = 10
) -> Dict[str, any]:
    """
    Benchmark a specific calculator configuration.

    Args:
        config_name: Name of configuration
        calculator: Configured calculator
        molecules: Test molecules
        num_iterations: Number of iterations per molecule
        warmup: Number of warmup iterations

    Returns:
        Benchmark results dictionary
    """
    logger.info(f"\nBenchmarking: {config_name}")
    logger.info("=" * 60)

    results = {
        'config': config_name,
        'system_results': {},
        'total_time': 0.0,
        'total_calls': 0,
    }

    for size, atoms in sorted(molecules.items()):
        logger.info(f"  Testing {size} atoms...")

        # Attach calculator
        atoms.calc = calculator

        # Warmup
        for _ in range(warmup):
            _ = atoms.get_potential_energy()
            _ = atoms.get_forces()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark
        times = []
        energies = []
        forces_list = []

        for _ in range(num_iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.perf_counter()

            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end = time.perf_counter()

            times.append((end - start) * 1000)  # Convert to ms
            energies.append(energy)
            forces_list.append(forces)

        # Compute statistics
        mean_time = np.mean(times)
        std_time = np.std(times)
        median_time = np.median(times)
        min_time = np.min(times)

        results['system_results'][size] = {
            'mean_time_ms': mean_time,
            'std_time_ms': std_time,
            'median_time_ms': median_time,
            'min_time_ms': min_time,
            'throughput': 1000.0 / mean_time,  # structures/second
            'sample_energy': float(energies[0]),
            'sample_forces_shape': forces_list[0].shape,
        }

        results['total_time'] += sum(times)
        results['total_calls'] += num_iterations

        logger.info(f"    {mean_time:.3f} ± {std_time:.3f} ms (median: {median_time:.3f} ms)")
        logger.info(f"    Throughput: {1000.0/mean_time:.1f} structures/sec")

    # Compute overall statistics
    results['avg_time_per_call'] = results['total_time'] / results['total_calls']

    return results


def compare_accuracy(
    baseline_results: Dict,
    test_results: Dict,
    molecules: Dict[int, Atoms],
    baseline_calc: StudentForceFieldCalculator,
    test_calc: StudentForceFieldCalculator
) -> Dict[str, float]:
    """
    Compare accuracy of test configuration against baseline.

    Args:
        baseline_results: Baseline benchmark results
        test_results: Test configuration results
        molecules: Test molecules
        baseline_calc: Baseline calculator
        test_calc: Test calculator

    Returns:
        Accuracy metrics
    """
    logger.info("\nComparing accuracy...")

    energy_errors = []
    force_errors = []

    for size, atoms in sorted(molecules.items()):
        # Baseline
        atoms.calc = baseline_calc
        baseline_energy = atoms.get_potential_energy()
        baseline_forces = atoms.get_forces()

        # Test
        atoms.calc = test_calc
        test_energy = atoms.get_potential_energy()
        test_forces = atoms.get_forces()

        # Compute errors
        energy_error = abs(baseline_energy - test_energy)
        force_error = np.sqrt(np.mean((baseline_forces - test_forces) ** 2))  # RMSE

        energy_errors.append(energy_error)
        force_errors.append(force_error)

        logger.info(f"  {size} atoms: E_err={energy_error:.6f} eV, F_RMSE={force_error:.6f} eV/Å")

    return {
        'mean_energy_error_eV': float(np.mean(energy_errors)),
        'max_energy_error_eV': float(np.max(energy_errors)),
        'mean_force_rmse': float(np.mean(force_errors)),
        'max_force_rmse': float(np.max(force_errors)),
    }


def get_gpu_memory_usage() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive benchmark of optimization strategies'
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=Path('checkpoints/best_model.pt'),
        help='Path to PyTorch checkpoint'
    )
    parser.add_argument(
        '--jit-model',
        type=Path,
        default=Path('models/student_model_jit.pt'),
        help='Path to TorchScript model'
    )
    parser.add_argument(
        '--system-sizes',
        type=str,
        default='10,20,50,100',
        help='Comma-separated list of system sizes to test'
    )
    parser.add_argument(
        '--num-iterations',
        type=int,
        default=100,
        help='Number of iterations per configuration'
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=10,
        help='Number of warmup iterations'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('benchmarks/optimization_results.json'),
        help='Output file for results'
    )
    parser.add_argument(
        '--device',
        default='cuda',
        help='Device for inference'
    )

    args = parser.parse_args()

    # Parse system sizes
    system_sizes = [int(x) for x in args.system_sizes.split(',')]

    logger.info("="*80)
    logger.info("COMPREHENSIVE OPTIMIZATION BENCHMARK")
    logger.info("="*80)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"TorchScript model: {args.jit_model}")
    logger.info(f"System sizes: {system_sizes}")
    logger.info(f"Iterations: {args.num_iterations}")
    logger.info(f"Device: {args.device}")
    logger.info("="*80)

    # Create test molecules
    logger.info("\nCreating test molecules...")
    molecules = create_test_molecules(system_sizes)
    logger.info(f"Created {len(molecules)} test systems")

    # Define configurations to test
    configurations = []

    # 1. Baseline (FP32)
    configurations.append({
        'name': 'Baseline (FP32)',
        'use_fp16': False,
        'use_jit': False,
        'use_compile': False,
    })

    # 2. FP16
    if args.device == 'cuda':
        configurations.append({
            'name': 'FP16',
            'use_fp16': True,
            'use_jit': False,
            'use_compile': False,
        })

    # 3. TorchScript
    if args.jit_model.exists():
        configurations.append({
            'name': 'TorchScript',
            'use_fp16': False,
            'use_jit': True,
            'use_compile': False,
        })

    # 4. TorchScript + FP16
    if args.jit_model.exists() and args.device == 'cuda':
        configurations.append({
            'name': 'TorchScript + FP16',
            'use_fp16': True,
            'use_jit': True,
            'use_compile': False,
        })

    # 5. torch.compile (if supported)
    # Note: torch.compile() not supported on Python 3.13+
    try:
        import sys
        if sys.version_info < (3, 13):
            configurations.append({
                'name': 'torch.compile',
                'use_fp16': False,
                'use_jit': False,
                'use_compile': True,
            })
    except:
        pass

    # Run benchmarks
    all_results = {}
    calculators = {}

    for config in configurations:
        config_name = config['name']

        # Create calculator
        calc_kwargs = {
            'checkpoint_path': args.checkpoint,
            'device': args.device,
            'use_fp16': config['use_fp16'],
            'use_compile': config['use_compile'],
        }

        if config['use_jit']:
            calc_kwargs['use_jit'] = True
            calc_kwargs['jit_path'] = args.jit_model

        try:
            calculator = StudentForceFieldCalculator(**calc_kwargs)
            calculators[config_name] = calculator

            # Reset GPU memory tracking
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            # Run benchmark
            results = benchmark_configuration(
                config_name=config_name,
                calculator=calculator,
                molecules=molecules,
                num_iterations=args.num_iterations,
                warmup=args.warmup
            )

            # Add memory usage
            results['gpu_memory_mb'] = get_gpu_memory_usage()

            all_results[config_name] = results

        except Exception as e:
            logger.error(f"Failed to benchmark {config_name}: {e}")
            continue

    # Compute speedups
    if 'Baseline (FP32)' in all_results:
        baseline_results = all_results['Baseline (FP32)']
        baseline_calc = calculators['Baseline (FP32)']

        logger.info("\n" + "="*80)
        logger.info("SPEEDUP ANALYSIS")
        logger.info("="*80)

        for config_name, results in all_results.items():
            if config_name == 'Baseline (FP32)':
                continue

            # Compute speedup
            speedup = baseline_results['avg_time_per_call'] / results['avg_time_per_call']
            results['speedup_vs_baseline'] = speedup

            logger.info(f"\n{config_name}:")
            logger.info(f"  Overall speedup: {speedup:.2f}x")

            # Per-system speedup
            logger.info(f"  Per-system speedups:")
            for size in sorted(molecules.keys()):
                baseline_time = baseline_results['system_results'][size]['mean_time_ms']
                test_time = results['system_results'][size]['mean_time_ms']
                system_speedup = baseline_time / test_time
                logger.info(f"    {size:3d} atoms: {system_speedup:.2f}x")

            # Accuracy comparison
            if config_name in calculators:
                accuracy = compare_accuracy(
                    baseline_results,
                    results,
                    molecules,
                    baseline_calc,
                    calculators[config_name]
                )
                results['accuracy'] = accuracy

                logger.info(f"  Accuracy:")
                logger.info(f"    Energy error: {accuracy['mean_energy_error_eV']:.6f} ± {accuracy['max_energy_error_eV']:.6f} eV")
                logger.info(f"    Force RMSE:   {accuracy['mean_force_rmse']:.6f} ± {accuracy['max_force_rmse']:.6f} eV/Å")

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to native Python types for JSON serialization
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        return obj

    serializable_results = convert_to_json_serializable(all_results)

    with open(args.output, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    logger.info(f"\nResults saved to: {args.output}")

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)

    if 'Baseline (FP32)' in all_results:
        baseline_time = baseline_results['avg_time_per_call']
        logger.info(f"Baseline: {baseline_time:.3f} ms/call")

        best_config = max(
            [name for name in all_results.keys() if name != 'Baseline (FP32)'],
            key=lambda name: all_results[name].get('speedup_vs_baseline', 0),
            default=None
        )

        if best_config:
            best_speedup = all_results[best_config]['speedup_vs_baseline']
            best_time = all_results[best_config]['avg_time_per_call']
            logger.info(f"\nBest configuration: {best_config}")
            logger.info(f"  Speedup: {best_speedup:.2f}x")
            logger.info(f"  Time: {best_time:.3f} ms/call")

            if 'accuracy' in all_results[best_config]:
                acc = all_results[best_config]['accuracy']
                logger.info(f"  Energy error: {acc['mean_energy_error_eV']:.6f} eV")
                logger.info(f"  Force RMSE: {acc['mean_force_rmse']:.6f} eV/Å")

    logger.info("\n" + "="*80)
    logger.info("Benchmark complete!")
    logger.info("="*80)


if __name__ == '__main__':
    main()
