#!/usr/bin/env python3
"""
Performance Baseline Benchmarks for Student Model

Comprehensive benchmarking suite to establish baseline performance
and identify optimization opportunities.

Usage:
    python scripts/benchmark_inference.py --device cuda --output benchmarks/
    python scripts/benchmark_inference.py --device cpu --no-profile
    python scripts/benchmark_inference.py --help
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

import numpy as np
import torch
from ase.build import molecule, bulk
from ase.io import read
from ase import Atoms

# Add src to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from mlff_distiller.inference import StudentForceFieldCalculator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_molecules() -> Dict[str, List[Atoms]]:
    """Create test molecules of various sizes."""
    molecules = {}

    # Small molecules (5-15 atoms)
    small = [
        molecule('H2O'),      # 3 atoms
        molecule('NH3'),      # 4 atoms
        molecule('CH4'),      # 5 atoms
        molecule('C2H6'),     # 8 atoms
        molecule('C3H8'),     # 11 atoms
    ]
    molecules['small'] = small
    logger.info(f"Created {len(small)} small molecules (3-11 atoms)")

    # Medium molecules (15-50 atoms)
    medium = [
        molecule('C6H6'),     # Benzene - 12 atoms
        molecule('C3H8'),     # Propane - 11 atoms
        molecule('C2H6CHOH'), # Ethanol derivative - 9 atoms
    ]
    # Add some larger molecules by replicating
    for i in range(2):
        atoms = molecule('C6H6').copy()  # Benzene
        atoms.positions += np.array([i * 6.0, 0, 0])  # Offset
        medium.append(atoms)
    molecules['medium'] = medium
    logger.info(f"Created {len(medium)} medium molecules (9-12 atoms)")

    # Large molecules (50-200 atoms) - create by replicating
    large = []
    base = molecule('C6H6')  # Benzene - 12 atoms
    # Create larger systems by replication
    for i in range(5):
        atoms = base.copy()
        atoms.positions += np.array([i * 6.0, 0, 0])  # Offset in x
        large.append(atoms)
    # Combine first 5 into one large molecule
    combined = large[0].copy()
    for atoms in large[1:]:
        combined += atoms
    molecules['large'] = [combined]  # One large molecule with 60 atoms
    logger.info(f"Created {len(molecules['large'])} large molecules (~60 atoms each)")

    return molecules


def benchmark_single_inference(
    calc: StudentForceFieldCalculator,
    molecules: List[Atoms],
    warmup: int = 5,
    iterations: int = 20
) -> Dict:
    """
    Benchmark single structure inference.

    Args:
        calc: Calculator instance
        molecules: List of molecules to test
        warmup: Number of warmup iterations
        iterations: Number of benchmark iterations

    Returns:
        Dictionary with timing statistics
    """
    logger.info(f"Running single inference benchmark (warmup={warmup}, iterations={iterations})")

    # Warmup
    for i in range(min(warmup, len(molecules))):
        molecules[i].calc = calc
        _ = molecules[i].get_potential_energy()
        _ = molecules[i].get_forces()

    # Benchmark each molecule
    results_by_size = {}
    all_times = []

    for mol in molecules[:iterations]:
        mol.calc = calc
        n_atoms = len(mol)

        start = time.perf_counter()
        energy = mol.get_potential_energy()
        forces = mol.get_forces()
        end = time.perf_counter()

        elapsed = (end - start) * 1000  # Convert to ms
        all_times.append(elapsed)

        if n_atoms not in results_by_size:
            results_by_size[n_atoms] = []
        results_by_size[n_atoms].append(elapsed)

    # Aggregate statistics
    stats = {
        'mean_time_ms': float(np.mean(all_times)),
        'std_time_ms': float(np.std(all_times)),
        'min_time_ms': float(np.min(all_times)),
        'max_time_ms': float(np.max(all_times)),
        'median_time_ms': float(np.median(all_times)),
        'p95_time_ms': float(np.percentile(all_times, 95)),
        'n_samples': len(all_times),
        'by_size': {}
    }

    for size, times in results_by_size.items():
        stats['by_size'][size] = {
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'n_samples': len(times)
        }

    logger.info(f"Single inference: {stats['mean_time_ms']:.2f} ± {stats['std_time_ms']:.2f} ms")
    return stats


def benchmark_batch_inference(
    calc: StudentForceFieldCalculator,
    molecules: List[Atoms],
    batch_sizes: List[int] = [1, 2, 4, 8, 16]
) -> Dict:
    """
    Benchmark batch inference with various batch sizes.

    Args:
        calc: Calculator instance
        molecules: List of molecules to test
        batch_sizes: List of batch sizes to test

    Returns:
        Dictionary with batch performance statistics
    """
    logger.info(f"Running batch inference benchmark (sizes={batch_sizes})")

    results = {}

    for batch_size in batch_sizes:
        if len(molecules) < batch_size:
            logger.warning(f"Skipping batch size {batch_size} (not enough molecules)")
            continue

        batch = molecules[:batch_size]

        # Warmup
        try:
            _ = calc.calculate_batch(batch)
        except AttributeError:
            logger.warning("Batch calculation not implemented, skipping batch benchmarks")
            return {}

        # Benchmark
        times = []
        for _ in range(5):  # 5 iterations
            start = time.perf_counter()
            results_batch = calc.calculate_batch(batch)
            end = time.perf_counter()
            times.append(end - start)

        elapsed = np.mean(times) * 1000  # ms
        throughput = batch_size / (np.mean(times))  # structures/sec

        results[batch_size] = {
            'mean_time_ms': float(elapsed),
            'std_time_ms': float(np.std(times) * 1000),
            'time_per_structure_ms': float(elapsed / batch_size),
            'throughput_struct_per_sec': float(throughput),
            'speedup_vs_batch1': 1.0  # Will update later
        }

        logger.info(f"Batch {batch_size}: {elapsed:.2f} ms ({throughput:.2f} struct/sec)")

    # Calculate speedup vs batch size 1
    if 1 in results:
        baseline_time_per_struct = results[1]['time_per_structure_ms']
        for batch_size, data in results.items():
            if batch_size > 1:
                data['speedup_vs_batch1'] = baseline_time_per_struct / data['time_per_structure_ms']

    return results


def profile_inference(
    calc: StudentForceFieldCalculator,
    molecule: Atoms,
    output_dir: Path
) -> Dict:
    """
    Profile inference using PyTorch profiler.

    Args:
        calc: Calculator instance
        molecule: Test molecule
        output_dir: Directory to save profiling results

    Returns:
        Dictionary with profiling statistics
    """
    logger.info("Running PyTorch profiler...")

    output_dir.mkdir(parents=True, exist_ok=True)
    profile_file = output_dir / 'profiling_results.txt'

    molecule.calc = calc

    # Warmup
    for _ in range(5):
        _ = molecule.get_potential_energy()

    # Profile
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(10):  # Profile 10 iterations
            energy = molecule.get_potential_energy()
            forces = molecule.get_forces()

    # Save detailed profiling results
    with open(profile_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PyTorch Profiler Results - Sorted by CUDA Time\n")
        f.write("=" * 80 + "\n\n")
        f.write(prof.key_averages().table(
            sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
            row_limit=50
        ))
        f.write("\n\n")
        f.write("=" * 80 + "\n")
        f.write("PyTorch Profiler Results - Sorted by CPU Time\n")
        f.write("=" * 80 + "\n\n")
        f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))

    logger.info(f"Profiling results saved to {profile_file}")

    # Extract key statistics
    key_averages = prof.key_averages()

    stats = {
        'total_events': len(key_averages),
        'profiling_file': str(profile_file)
    }

    # Find top operations
    # Note: In newer PyTorch versions, use self_cpu_time_total instead of cpu_time_total
    sorted_events = sorted(
        key_averages,
        key=lambda x: (x.self_cuda_time_total if hasattr(x, 'self_cuda_time_total') and torch.cuda.is_available()
                      else x.self_cpu_time_total),
        reverse=True
    )

    stats['top_5_operations'] = []
    for event in sorted_events[:5]:
        stats['top_5_operations'].append({
            'name': event.key,
            'cpu_time_us': event.self_cpu_time_total,
            'cuda_time_us': (event.self_cuda_time_total
                           if hasattr(event, 'self_cuda_time_total') and torch.cuda.is_available()
                           else 0),
            'calls': event.count
        })

    return stats


def measure_memory_usage(
    calc: StudentForceFieldCalculator,
    molecules: List[Atoms]
) -> Dict:
    """
    Measure memory usage during inference.

    Args:
        calc: Calculator instance
        molecules: Test molecules

    Returns:
        Dictionary with memory statistics
    """
    logger.info("Measuring memory usage...")

    if not torch.cuda.is_available():
        logger.warning("CUDA not available, skipping GPU memory measurement")
        return {'gpu_memory': 'N/A'}

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Measure baseline
    baseline = torch.cuda.memory_allocated() / 1024**2  # MB

    # Run inference
    for mol in molecules[:10]:
        mol.calc = calc
        _ = mol.get_potential_energy()
        _ = mol.get_forces()

    # Measure peak
    peak = torch.cuda.max_memory_allocated() / 1024**2  # MB
    current = torch.cuda.memory_allocated() / 1024**2  # MB

    stats = {
        'baseline_mb': float(baseline),
        'peak_mb': float(peak),
        'current_mb': float(current),
        'inference_overhead_mb': float(peak - baseline)
    }

    logger.info(f"Memory: baseline={baseline:.2f} MB, peak={peak:.2f} MB, overhead={peak-baseline:.2f} MB")
    return stats


def analyze_scaling(
    calc: StudentForceFieldCalculator,
    molecules_by_size: Dict[str, List[Atoms]]
) -> Dict:
    """
    Analyze how inference time scales with system size.

    Args:
        calc: Calculator instance
        molecules_by_size: Dictionary of molecules grouped by size

    Returns:
        Dictionary with scaling analysis
    """
    logger.info("Analyzing scaling with system size...")

    scaling_data = []

    for category, molecules in molecules_by_size.items():
        for mol in molecules[:3]:  # Test first 3 of each category
            mol.calc = calc
            n_atoms = len(mol)

            # Time inference
            times = []
            for _ in range(5):
                start = time.perf_counter()
                _ = mol.get_potential_energy()
                _ = mol.get_forces()
                end = time.perf_counter()
                times.append((end - start) * 1000)

            scaling_data.append({
                'n_atoms': n_atoms,
                'category': category,
                'mean_time_ms': float(np.mean(times)),
                'std_time_ms': float(np.std(times))
            })

    # Sort by size
    scaling_data.sort(key=lambda x: x['n_atoms'])

    # Calculate time per atom
    for data in scaling_data:
        data['ms_per_atom'] = data['mean_time_ms'] / data['n_atoms']

    logger.info(f"Scaling analysis complete ({len(scaling_data)} data points)")
    return {'scaling_data': scaling_data}


def generate_summary_report(
    results: Dict,
    output_file: Path
):
    """
    Generate human-readable summary report.

    Args:
        results: All benchmark results
        output_file: Path to save markdown report
    """
    logger.info(f"Generating summary report: {output_file}")

    with open(output_file, 'w') as f:
        f.write("# Student Model Performance Baseline Report\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Device**: {results['config']['device']}\n")
        f.write(f"**Model**: {results['config']['checkpoint']}\n\n")

        f.write("---\n\n")

        # Single inference summary
        if 'single_inference' in results:
            single = results['single_inference']
            f.write("## Single Inference Performance\n\n")
            f.write(f"- **Mean time**: {single['mean_time_ms']:.2f} ± {single['std_time_ms']:.2f} ms\n")
            f.write(f"- **Median time**: {single['median_time_ms']:.2f} ms\n")
            f.write(f"- **95th percentile**: {single['p95_time_ms']:.2f} ms\n")
            f.write(f"- **Throughput**: {1000/single['mean_time_ms']:.2f} structures/second\n\n")

            f.write("### Performance by System Size\n\n")
            f.write("| Atoms | Mean (ms) | Std (ms) | Samples |\n")
            f.write("|-------|-----------|----------|----------|\n")
            for size, stats in sorted(single['by_size'].items()):
                f.write(f"| {size} | {stats['mean_ms']:.2f} | {stats['std_ms']:.2f} | {stats['n_samples']} |\n")
            f.write("\n")

        # Batch inference summary
        if 'batch_inference' in results and results['batch_inference']:
            batch = results['batch_inference']
            f.write("## Batch Inference Performance\n\n")
            f.write("| Batch Size | Total Time (ms) | Time/Structure (ms) | Throughput (struct/s) | Speedup vs Batch=1 |\n")
            f.write("|------------|------------------|---------------------|------------------------|--------------------|\n")
            for size, stats in sorted(batch.items()):
                f.write(f"| {size} | {stats['mean_time_ms']:.2f} | "
                       f"{stats['time_per_structure_ms']:.2f} | "
                       f"{stats['throughput_struct_per_sec']:.2f} | "
                       f"{stats['speedup_vs_batch1']:.2f}x |\n")
            f.write("\n")

        # Memory usage
        if 'memory_usage' in results:
            mem = results['memory_usage']
            if isinstance(mem.get('peak_mb'), (int, float)):
                f.write("## Memory Usage\n\n")
                f.write(f"- **Baseline**: {mem['baseline_mb']:.2f} MB\n")
                f.write(f"- **Peak**: {mem['peak_mb']:.2f} MB\n")
                f.write(f"- **Inference overhead**: {mem['inference_overhead_mb']:.2f} MB\n\n")

        # Scaling analysis
        if 'scaling_analysis' in results:
            scaling = results['scaling_analysis']['scaling_data']
            f.write("## Scaling with System Size\n\n")
            f.write("| Atoms | Category | Time (ms) | ms/atom |\n")
            f.write("|-------|----------|-----------|----------|\n")
            for data in scaling:
                f.write(f"| {data['n_atoms']} | {data['category']} | "
                       f"{data['mean_time_ms']:.2f} ± {data['std_time_ms']:.2f} | "
                       f"{data['ms_per_atom']:.3f} |\n")
            f.write("\n")

        # Profiling summary
        if 'profiling' in results:
            prof = results['profiling']
            f.write("## Profiling Analysis\n\n")
            f.write(f"Top 5 most expensive operations:\n\n")
            for i, op in enumerate(prof['top_5_operations'], 1):
                cpu_ms = op['cpu_time_us'] / 1000
                cuda_ms = op['cuda_time_us'] / 1000
                f.write(f"{i}. **{op['name']}**\n")
                f.write(f"   - CPU time: {cpu_ms:.2f} ms\n")
                if cuda_ms > 0:
                    f.write(f"   - CUDA time: {cuda_ms:.2f} ms\n")
                f.write(f"   - Calls: {op['calls']}\n\n")

            f.write(f"Full profiling results: `{prof['profiling_file']}`\n\n")

        f.write("---\n\n")
        f.write("## Key Findings\n\n")

        # Calculate some key metrics
        if 'single_inference' in results:
            mean_time = results['single_inference']['mean_time_ms']
            throughput = 1000 / mean_time
            f.write(f"- Current inference speed: **{mean_time:.2f} ms/structure**\n")
            f.write(f"- Current throughput: **{throughput:.2f} structures/second**\n")

        if 'memory_usage' in results and isinstance(results['memory_usage'].get('peak_mb'), (int, float)):
            f.write(f"- Memory footprint: **{results['memory_usage']['peak_mb']:.2f} MB**\n")

        f.write("\n## Next Steps\n\n")
        f.write("1. Analyze profiling results to identify bottlenecks\n")
        f.write("2. Review optimization roadmap (see `OPTIMIZATION_ROADMAP.md`)\n")
        f.write("3. Implement high-priority optimizations\n")
        f.write("4. Re-benchmark after optimizations\n")

    logger.info(f"Report saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark student model inference performance"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda/cpu)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='benchmarks',
        help='Output directory for results'
    )
    parser.add_argument(
        '--no-profile',
        action='store_true',
        help='Skip profiling (faster)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick benchmark with fewer iterations'
    )
    parser.add_argument(
        '--use-compile',
        action='store_true',
        help='Enable torch.compile() optimization (Phase 1A)'
    )
    parser.add_argument(
        '--use-fp16',
        action='store_true',
        help='Enable FP16 mixed precision (Phase 1B)'
    )

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Student Model Performance Benchmarks")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Output: {output_dir}")
    logger.info("")

    # Load calculator
    logger.info("Loading calculator...")
    opt_info = []
    if args.use_compile:
        opt_info.append("torch.compile()")
    if args.use_fp16:
        opt_info.append("FP16")
    if opt_info:
        logger.info(f"Optimizations enabled: {', '.join(opt_info)}")

    calc = StudentForceFieldCalculator(
        checkpoint_path=args.checkpoint,
        device=args.device,
        enable_timing=True,
        use_compile=args.use_compile,
        use_fp16=args.use_fp16
    )

    # Create test molecules
    logger.info("Creating test molecules...")
    molecules_by_size = create_test_molecules()
    all_molecules = []
    for mols in molecules_by_size.values():
        all_molecules.extend(mols)

    # Store results
    results = {
        'config': {
            'checkpoint': args.checkpoint,
            'device': args.device,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'use_compile': args.use_compile,
            'use_fp16': args.use_fp16,
            'optimizations': opt_info if opt_info else ['none']
        }
    }

    if torch.cuda.is_available():
        results['config']['cuda_device'] = torch.cuda.get_device_name(0)

    # Run benchmarks
    logger.info("\n" + "=" * 80)
    logger.info("Running Benchmarks")
    logger.info("=" * 80 + "\n")

    # 1. Single inference
    iterations = 10 if args.quick else 20
    results['single_inference'] = benchmark_single_inference(
        calc, all_molecules, warmup=5, iterations=iterations
    )

    # 2. Batch inference
    batch_sizes = [1, 2, 4] if args.quick else [1, 2, 4, 8, 16]
    results['batch_inference'] = benchmark_batch_inference(
        calc, all_molecules, batch_sizes=batch_sizes
    )

    # 3. Memory usage
    results['memory_usage'] = measure_memory_usage(calc, all_molecules)

    # 4. Scaling analysis
    results['scaling_analysis'] = analyze_scaling(calc, molecules_by_size)

    # 5. Profiling
    if not args.no_profile:
        test_mol = molecules_by_size['medium'][0]
        results['profiling'] = profile_inference(calc, test_mol, output_dir)

    # Save results
    logger.info("\n" + "=" * 80)
    logger.info("Saving Results")
    logger.info("=" * 80 + "\n")

    json_file = output_dir / 'baseline_performance.json'
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"JSON results saved to {json_file}")

    # Generate report
    report_file = output_dir / 'BASELINE_REPORT.md'
    generate_summary_report(results, report_file)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("Summary")
    logger.info("=" * 80 + "\n")

    if 'single_inference' in results:
        mean_time = results['single_inference']['mean_time_ms']
        throughput = 1000 / mean_time
        logger.info(f"Mean inference time: {mean_time:.2f} ms")
        logger.info(f"Throughput: {throughput:.2f} structures/second")

    if 'memory_usage' in results and isinstance(results['memory_usage'].get('peak_mb'), (int, float)):
        logger.info(f"Peak memory: {results['memory_usage']['peak_mb']:.2f} MB")

    logger.info(f"\nResults saved to: {output_dir}/")
    logger.info("\nNext steps:")
    logger.info("1. Review the baseline report")
    logger.info("2. Check profiling results for bottlenecks")
    logger.info("3. Create optimization roadmap")

    return 0


if __name__ == '__main__':
    sys.exit(main())
