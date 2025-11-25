#!/usr/bin/env python3
"""
Comprehensive Performance Benchmarking for All Student Models

This script benchmarks inference performance of Original, Tiny, and Ultra-tiny
student models across multiple dimensions:
- Single-structure inference at varying system sizes
- Batch processing efficiency
- Memory usage and profiling
- MD simulation performance
- GPU vs CPU comparison
- Speedup analysis

Author: Testing & Benchmarking Engineer
Date: 2025-11-25
Issue: #36
"""

import json
import logging
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from ase import Atoms
from ase.build import molecule

# Suppress LAMMPS warnings
warnings.filterwarnings("ignore")

# Import MLFF Distiller components
from mlff_distiller.inference import StudentForceFieldCalculator
from mlff_distiller.testing import NVEMDHarness

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress verbose library logging
logging.getLogger('mlff_distiller').setLevel(logging.WARNING)

# Configure matplotlib
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Model specifications
MODEL_CONFIGS = {
    'Original (427K)': {
        'checkpoint': 'checkpoints/best_model.pt',
        'expected_params': 427_000,
        'color': '#1f77b4',  # Blue
        'marker': 'o'
    },
    'Tiny (77K)': {
        'checkpoint': 'checkpoints/tiny_model/best_model.pt',
        'expected_params': 77_000,
        'color': '#ff7f0e',  # Orange
        'marker': 's'
    },
    'Ultra-tiny (21K)': {
        'checkpoint': 'checkpoints/ultra_tiny_model/best_model.pt',
        'expected_params': 21_000,
        'color': '#2ca02c',  # Green
        'marker': '^'
    }
}


def create_test_molecules() -> Dict[int, Atoms]:
    """
    Create test molecules of varying sizes for benchmarking.

    Returns:
        Dictionary mapping number of atoms to ASE Atoms object
    """
    logger.info("Creating test molecules for benchmarking...")

    molecules = {}

    # Small molecules (from ASE database)
    test_mols = [
        'H2O',      # 3 atoms - water
        'NH3',      # 4 atoms - ammonia
        'CH4',      # 5 atoms - methane
        'CH3OH',    # 6 atoms - methanol
        'C2H6',     # 8 atoms - ethane
        'CH3CH2OH', # 9 atoms - ethanol
        'C3H8',     # 11 atoms - propane
        'C6H6',     # 12 atoms - benzene
        'C4H10',    # 14 atoms - butane
    ]

    for name in test_mols:
        try:
            mol = molecule(name)
            n_atoms = len(mol)
            if n_atoms not in molecules:
                molecules[n_atoms] = mol
                logger.debug(f"  Added {name}: {n_atoms} atoms")
        except Exception as e:
            logger.debug(f"  Could not create {name}: {e}")

    # Additional manual molecules if needed
    if 10 not in molecules and 9 in molecules:
        # Create a 10-atom molecule by extending ethanol
        mol = molecules[9].copy()
        # Add an extra H
        pos = mol.get_positions()
        mol.append('H')
        new_pos = np.vstack([pos, pos[-1] + [1.0, 0, 0]])
        mol.set_positions(new_pos)
        molecules[10] = mol

    # Sort by size
    sorted_molecules = {k: molecules[k] for k in sorted(molecules.keys())}

    logger.info(f"Created {len(sorted_molecules)} test molecules: {list(sorted_molecules.keys())} atoms")

    return sorted_molecules


def load_model(model_name: str, device: str = 'cuda') -> StudentForceFieldCalculator:
    """
    Load a model and return the calculator.

    Args:
        model_name: Name of model (key in MODEL_CONFIGS)
        device: Device to load on ('cuda' or 'cpu')

    Returns:
        StudentForceFieldCalculator instance
    """
    config = MODEL_CONFIGS[model_name]
    checkpoint_path = Path(config['checkpoint'])

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading {model_name} from {checkpoint_path}")

    calc = StudentForceFieldCalculator(
        checkpoint_path=checkpoint_path,
        device=device,
        enable_timing=False  # We'll time manually
    )

    # Count parameters
    n_params = sum(p.numel() for p in calc.model.parameters())
    logger.info(f"  Loaded {model_name}: {n_params:,} parameters")

    return calc


def benchmark_single_inference(
    calc: StudentForceFieldCalculator,
    atoms: Atoms,
    n_warmup: int = 10,
    n_trials: int = 100,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Benchmark single-structure inference time.

    Args:
        calc: Calculator instance
        atoms: Atoms object to benchmark
        n_warmup: Number of warmup calls
        n_trials: Number of timing trials
        device: Device being used

    Returns:
        Dictionary with timing statistics
    """
    atoms = atoms.copy()
    atoms.calc = calc

    # Warmup
    for _ in range(n_warmup):
        _ = atoms.get_forces()

    # Synchronize GPU if using CUDA
    if device == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        forces = atoms.get_forces()

        # Synchronize GPU to ensure completion
        if device == 'cuda':
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append(end - start)

    times = np.array(times)

    return {
        'mean_time': float(np.mean(times)),
        'std_time': float(np.std(times)),
        'median_time': float(np.median(times)),
        'min_time': float(np.min(times)),
        'max_time': float(np.max(times)),
        'mean_time_ms': float(np.mean(times) * 1000),
        'std_time_ms': float(np.std(times) * 1000),
    }


def benchmark_batch_inference(
    calc: StudentForceFieldCalculator,
    atoms: Atoms,
    batch_sizes: List[int] = [1, 4, 8, 16],
    n_warmup: int = 5,
    n_trials: int = 50,
    device: str = 'cuda'
) -> Dict[int, Dict[str, float]]:
    """
    Benchmark batch processing at different batch sizes.

    Args:
        calc: Calculator instance
        atoms: Base atoms object to replicate
        batch_sizes: List of batch sizes to test
        n_warmup: Number of warmup calls
        n_trials: Number of timing trials
        device: Device being used

    Returns:
        Dictionary mapping batch size to timing statistics
    """
    results = {}

    for batch_size in batch_sizes:
        logger.debug(f"    Batch size {batch_size}...")

        # Create batch
        atoms_list = [atoms.copy() for _ in range(batch_size)]

        # Warmup
        for _ in range(n_warmup):
            _ = calc.calculate_batch(atoms_list)

        # Synchronize GPU
        if device == 'cuda':
            torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(n_trials):
            start = time.perf_counter()
            batch_results = calc.calculate_batch(atoms_list)

            if device == 'cuda':
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append(end - start)

        times = np.array(times)

        results[batch_size] = {
            'total_time': float(np.mean(times)),
            'total_time_std': float(np.std(times)),
            'per_structure': float(np.mean(times) / batch_size),
            'per_structure_ms': float(np.mean(times) / batch_size * 1000),
            'throughput': float(batch_size / np.mean(times)),  # structures/second
        }

    return results


def profile_memory(
    calc: StudentForceFieldCalculator,
    atoms: Atoms,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Profile memory usage during inference.

    Args:
        calc: Calculator instance
        atoms: Atoms object for testing
        device: Device being used

    Returns:
        Dictionary with memory statistics
    """
    if device != 'cuda':
        return {
            'peak_memory_mb': 0.0,
            'allocated_memory_mb': 0.0,
            'reserved_memory_mb': 0.0,
        }

    atoms = atoms.copy()
    atoms.calc = calc

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Measure baseline
    baseline_allocated = torch.cuda.memory_allocated() / 1024**2
    baseline_reserved = torch.cuda.memory_reserved() / 1024**2

    # Run inference
    _ = atoms.get_forces()
    torch.cuda.synchronize()

    # Measure peak
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2
    allocated_memory = torch.cuda.memory_allocated() / 1024**2
    reserved_memory = torch.cuda.memory_reserved() / 1024**2

    # Count parameters
    n_params = sum(p.numel() for p in calc.model.parameters())
    model_size_mb = n_params * 4 / 1024**2  # float32

    return {
        'n_parameters': n_params,
        'model_size_mb': float(model_size_mb),
        'peak_memory_mb': float(peak_memory),
        'allocated_memory_mb': float(allocated_memory),
        'reserved_memory_mb': float(reserved_memory),
        'memory_overhead_mb': float(peak_memory - baseline_allocated),
    }


def benchmark_md_simulation(
    calc: StudentForceFieldCalculator,
    atoms: Atoms,
    steps: int = 1000,
    temperature: float = 300.0,
    timestep: float = 0.5,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Benchmark MD simulation performance.

    Args:
        calc: Calculator instance
        atoms: Atoms object for simulation
        steps: Number of MD steps
        temperature: Temperature in K
        timestep: Timestep in fs
        device: Device being used

    Returns:
        Dictionary with MD performance metrics
    """
    logger.debug(f"    Running {steps} MD steps...")

    atoms = atoms.copy()
    atoms.calc = calc

    # Create MD harness
    harness = NVEMDHarness(
        atoms=atoms,
        calculator=calc,
        temperature=temperature,
        timestep=timestep
    )

    # Warmup (short run)
    _ = harness.run_simulation(steps=10)

    # Synchronize GPU
    if device == 'cuda':
        torch.cuda.synchronize()

    # Benchmark full run
    start = time.perf_counter()
    sim_results = harness.run_simulation(steps=steps)

    if device == 'cuda':
        torch.cuda.synchronize()

    end = time.perf_counter()

    wall_time = end - start
    steps_per_second = steps / wall_time
    time_per_step_ms = (wall_time / steps) * 1000

    return {
        'steps': steps,
        'wall_time': float(wall_time),
        'steps_per_second': float(steps_per_second),
        'time_per_step_ms': float(time_per_step_ms),
        'simulation_time_ps': float(steps * timestep / 1000),  # timestep in fs
    }


def run_full_benchmark(device: str = 'cuda', output_dir: Path = None) -> Dict:
    """
    Run comprehensive benchmark of all models.

    Args:
        device: Device to benchmark on
        output_dir: Directory to save results

    Returns:
        Dictionary with all benchmark results
    """
    if output_dir is None:
        output_dir = Path('benchmarks')
    output_dir.mkdir(exist_ok=True)

    logger.info(f"Starting comprehensive benchmark on {device}...")
    logger.info("=" * 80)

    # Create test molecules
    test_molecules = create_test_molecules()

    # Use a representative molecule for most tests
    # Choose 20-atom molecule if available, else closest
    sizes = sorted(test_molecules.keys())
    target_size = 20
    closest_size = min(sizes, key=lambda x: abs(x - target_size))
    representative_molecule = test_molecules[closest_size]

    logger.info(f"\nUsing {closest_size}-atom molecule as representative structure")
    logger.info(f"Available molecule sizes: {sizes}")

    results = {
        'device': device,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'test_molecule_sizes': sizes,
        'representative_size': closest_size,
        'models': {}
    }

    # Benchmark each model
    for model_name in MODEL_CONFIGS.keys():
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Benchmarking: {model_name}")
        logger.info(f"{'=' * 80}")

        try:
            # Load model
            calc = load_model(model_name, device=device)

            model_results = {
                'checkpoint': MODEL_CONFIGS[model_name]['checkpoint'],
                'single_inference': {},
                'batch_inference': {},
                'memory': {},
                'md_simulation': {},
            }

            # 1. Single-structure inference across system sizes
            logger.info(f"\n1. Single-structure inference benchmarking...")
            for size, atoms in test_molecules.items():
                logger.info(f"  Testing {size} atoms...")
                timing = benchmark_single_inference(calc, atoms, device=device)
                model_results['single_inference'][size] = timing
                logger.info(f"    Mean: {timing['mean_time_ms']:.3f} ± {timing['std_time_ms']:.3f} ms")

            # 2. Batch inference
            logger.info(f"\n2. Batch inference benchmarking...")
            batch_results = benchmark_batch_inference(
                calc, representative_molecule, device=device
            )
            model_results['batch_inference'] = batch_results

            for batch_size, timing in batch_results.items():
                logger.info(
                    f"  Batch {batch_size}: {timing['per_structure_ms']:.3f} ms/structure, "
                    f"{timing['throughput']:.1f} structures/s"
                )

            # 3. Memory profiling
            logger.info(f"\n3. Memory profiling...")
            memory_stats = profile_memory(calc, representative_molecule, device=device)
            model_results['memory'] = memory_stats

            if device == 'cuda':
                logger.info(
                    f"  Parameters: {memory_stats['n_parameters']:,}, "
                    f"Model size: {memory_stats['model_size_mb']:.1f} MB, "
                    f"Peak memory: {memory_stats['peak_memory_mb']:.1f} MB"
                )

            # 4. MD simulation performance
            logger.info(f"\n4. MD simulation benchmarking...")
            md_stats = benchmark_md_simulation(
                calc, representative_molecule, steps=1000, device=device
            )
            model_results['md_simulation'] = md_stats

            logger.info(
                f"  {md_stats['steps_per_second']:.1f} steps/s, "
                f"{md_stats['time_per_step_ms']:.3f} ms/step"
            )

            results['models'][model_name] = model_results

            logger.info(f"\n✓ {model_name} benchmarking complete")

        except Exception as e:
            logger.error(f"Failed to benchmark {model_name}: {e}", exc_info=True)
            results['models'][model_name] = {'error': str(e)}

    # Compute speedup metrics
    logger.info(f"\n{'=' * 80}")
    logger.info("Computing speedup metrics...")
    logger.info(f"{'=' * 80}")

    results['speedup_analysis'] = compute_speedup_metrics(results)

    # Save results
    output_file = output_dir / f'm6_performance_results_{device}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ Results saved to {output_file}")

    return results


def compute_speedup_metrics(results: Dict) -> Dict:
    """
    Compute speedup ratios between models.

    Args:
        results: Full benchmark results

    Returns:
        Dictionary with speedup metrics
    """
    baseline_name = 'Original (427K)'

    if baseline_name not in results['models']:
        logger.warning(f"Baseline model {baseline_name} not found in results")
        return {}

    baseline = results['models'][baseline_name]
    rep_size = results['representative_size']

    speedup_metrics = {}

    for model_name, model_results in results['models'].items():
        if model_name == baseline_name or 'error' in model_results:
            continue

        metrics = {}

        # Inference speedup
        if rep_size in baseline['single_inference'] and rep_size in model_results['single_inference']:
            baseline_time = baseline['single_inference'][rep_size]['mean_time']
            model_time = model_results['single_inference'][rep_size]['mean_time']
            metrics['inference_speedup'] = float(baseline_time / model_time)

        # MD speedup
        if 'steps_per_second' in baseline['md_simulation'] and 'steps_per_second' in model_results['md_simulation']:
            baseline_sps = baseline['md_simulation']['steps_per_second']
            model_sps = model_results['md_simulation']['steps_per_second']
            metrics['md_speedup'] = float(model_sps / baseline_sps)

        # Memory reduction
        if results['device'] == 'cuda':
            baseline_mem = baseline['memory']['peak_memory_mb']
            model_mem = model_results['memory']['peak_memory_mb']
            metrics['memory_reduction'] = float(baseline_mem / model_mem)

        # Compression ratio
        baseline_params = baseline['memory']['n_parameters']
        model_params = model_results['memory']['n_parameters']
        metrics['compression_ratio'] = float(baseline_params / model_params)

        speedup_metrics[model_name] = metrics

    return speedup_metrics


def visualize_results(results: Dict, output_dir: Path):
    """
    Generate visualization plots from benchmark results.

    Args:
        results: Benchmark results dictionary
        output_dir: Directory to save plots
    """
    logger.info("\nGenerating visualizations...")

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    device = results['device']

    # Filter out models with errors
    valid_models = {
        name: data for name, data in results['models'].items()
        if 'error' not in data
    }

    if not valid_models:
        logger.warning("No valid model results to visualize")
        return

    # 1. Inference time vs system size
    logger.info("  Creating inference time vs system size plot...")
    fig, ax = plt.subplots(figsize=(12, 7))

    for model_name, model_data in valid_models.items():
        config = MODEL_CONFIGS[model_name]

        sizes = []
        times = []
        errors = []

        for size, timing in sorted(model_data['single_inference'].items()):
            sizes.append(size)
            times.append(timing['mean_time_ms'])
            errors.append(timing['std_time_ms'])

        ax.errorbar(
            sizes, times, yerr=errors,
            marker=config['marker'],
            color=config['color'],
            label=model_name,
            linewidth=2,
            markersize=8,
            capsize=4
        )

    ax.set_xlabel('Number of Atoms', fontsize=13, fontweight='bold')
    ax.set_ylabel('Inference Time (ms)', fontsize=13, fontweight='bold')
    ax.set_title(f'Inference Time vs System Size ({device.upper()})', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'inference_time_vs_size_{device}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Speedup comparison
    logger.info("  Creating speedup comparison plot...")
    if 'speedup_analysis' in results and results['speedup_analysis']:
        fig, ax = plt.subplots(figsize=(10, 7))

        models = []
        inference_speedups = []
        md_speedups = []

        for model_name, metrics in results['speedup_analysis'].items():
            models.append(model_name.replace(' ', '\n'))
            inference_speedups.append(metrics.get('inference_speedup', 0))
            md_speedups.append(metrics.get('md_speedup', 0))

        x = np.arange(len(models))
        width = 0.35

        ax.bar(x - width/2, inference_speedups, width, label='Inference', color='#1f77b4')
        ax.bar(x + width/2, md_speedups, width, label='MD Simulation', color='#ff7f0e')

        # Add baseline line
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Baseline (Original)')

        ax.set_xlabel('Model', fontsize=13, fontweight='bold')
        ax.set_ylabel('Speedup Factor', fontsize=13, fontweight='bold')
        ax.set_title(f'Speedup vs Original Model ({device.upper()})', fontsize=15, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, (inf_sp, md_sp) in enumerate(zip(inference_speedups, md_speedups)):
            ax.text(i - width/2, inf_sp + 0.3, f'{inf_sp:.1f}x', ha='center', fontsize=10, fontweight='bold')
            ax.text(i + width/2, md_sp + 0.3, f'{md_sp:.1f}x', ha='center', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_dir / f'speedup_comparison_{device}.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 3. Memory usage comparison
    if device == 'cuda':
        logger.info("  Creating memory usage comparison plot...")
        fig, ax = plt.subplots(figsize=(10, 7))

        models = []
        model_sizes = []
        peak_memories = []
        colors = []

        for model_name, model_data in valid_models.items():
            models.append(model_name.replace(' ', '\n'))
            model_sizes.append(model_data['memory']['model_size_mb'])
            peak_memories.append(model_data['memory']['peak_memory_mb'])
            colors.append(MODEL_CONFIGS[model_name]['color'])

        x = np.arange(len(models))
        width = 0.35

        ax.bar(x - width/2, model_sizes, width, label='Model Size', color='#1f77b4')
        ax.bar(x + width/2, peak_memories, width, label='Peak GPU Memory', color='#ff7f0e')

        ax.set_xlabel('Model', fontsize=13, fontweight='bold')
        ax.set_ylabel('Memory (MB)', fontsize=13, fontweight='bold')
        ax.set_title('Memory Usage Comparison (GPU)', fontsize=15, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for i, (ms, pm) in enumerate(zip(model_sizes, peak_memories)):
            ax.text(i - width/2, ms + 5, f'{ms:.1f}', ha='center', fontsize=9)
            ax.text(i + width/2, pm + 5, f'{pm:.1f}', ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_dir / f'memory_usage_{device}.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 4. Batch throughput
    logger.info("  Creating batch throughput plot...")
    fig, ax = plt.subplots(figsize=(12, 7))

    for model_name, model_data in valid_models.items():
        config = MODEL_CONFIGS[model_name]

        batch_sizes = []
        throughputs = []

        for batch_size, timing in sorted(model_data['batch_inference'].items()):
            batch_sizes.append(batch_size)
            throughputs.append(timing['throughput'])

        ax.plot(
            batch_sizes, throughputs,
            marker=config['marker'],
            color=config['color'],
            label=model_name,
            linewidth=2,
            markersize=8
        )

    ax.set_xlabel('Batch Size', fontsize=13, fontweight='bold')
    ax.set_ylabel('Throughput (structures/second)', fontsize=13, fontweight='bold')
    ax.set_title(f'Batch Processing Throughput ({device.upper()})', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'batch_throughput_{device}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Speedup-Accuracy Tradeoff (if accuracy data available)
    logger.info("  Creating speedup-accuracy tradeoff plot...")

    # These values should be from Issue #35 (accuracy validation)
    # For now, using placeholder values - should be updated with actual accuracy results
    accuracy_data = {
        'Original (427K)': {'r2': 0.996, 'mae': 0.015},  # From Issue #33
        'Tiny (77K)': {'r2': 0.379, 'mae': 0.150},       # Placeholder - update with actual
        'Ultra-tiny (21K)': {'r2': 0.150, 'mae': 0.300}, # Placeholder - update with actual
    }

    if 'speedup_analysis' in results and results['speedup_analysis']:
        fig, ax = plt.subplots(figsize=(10, 8))

        speedups = []
        r2_scores = []
        labels = []

        # Add baseline
        speedups.append(1.0)
        r2_scores.append(accuracy_data['Original (427K)']['r2'])
        labels.append('Original')

        for model_name, metrics in results['speedup_analysis'].items():
            if model_name in accuracy_data:
                speedups.append(metrics.get('inference_speedup', 0))
                r2_scores.append(accuracy_data[model_name]['r2'])

                # Extract short label
                if 'Tiny' in model_name and 'Ultra' not in model_name:
                    labels.append('Tiny')
                elif 'Ultra' in model_name:
                    labels.append('Ultra-tiny')
                else:
                    labels.append(model_name)

        # Plot points
        for i, (sp, r2, label) in enumerate(zip(speedups, r2_scores, labels)):
            config = MODEL_CONFIGS[list(MODEL_CONFIGS.keys())[i]]
            ax.scatter(sp, r2, s=200, color=config['color'], marker=config['marker'],
                      label=label, zorder=3, edgecolors='black', linewidth=1.5)

        # Connect points to show Pareto frontier
        ax.plot(speedups, r2_scores, 'k--', alpha=0.3, linewidth=1, zorder=1)

        # Add annotations
        for i, (sp, r2, label) in enumerate(zip(speedups, r2_scores, labels)):
            ax.annotate(
                f'{label}\n{sp:.1f}x, R²={r2:.3f}',
                (sp, r2),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
            )

        ax.set_xlabel('Speedup Factor', fontsize=13, fontweight='bold')
        ax.set_ylabel('Force R² Score', fontsize=13, fontweight='bold')
        ax.set_title(f'Speedup-Accuracy Tradeoff ({device.upper()})', fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

        # Set reasonable axis limits
        ax.set_xlim(0.8, max(speedups) * 1.3)
        ax.set_ylim(0, 1.05)

        plt.tight_layout()
        plt.savefig(output_dir / f'speedup_accuracy_tradeoff_{device}.png', dpi=300, bbox_inches='tight')
        plt.close()

    logger.info(f"✓ Visualizations saved to {output_dir}/")


def generate_summary_table(results: Dict) -> str:
    """
    Generate a markdown summary table from benchmark results.

    Args:
        results: Benchmark results dictionary

    Returns:
        Markdown formatted table string
    """
    device = results['device']
    rep_size = results['representative_size']

    # Filter valid models
    valid_models = {
        name: data for name, data in results['models'].items()
        if 'error' not in data
    }

    if not valid_models:
        return "No valid results to summarize.\n"

    # Build table
    table = f"\n## Performance Summary ({device.upper()}, {rep_size} atoms)\n\n"
    table += "| Model | Params | Inference (ms) | MD (steps/s) | Memory (MB) | Speedup |\n"
    table += "|-------|--------|----------------|--------------|-------------|---------|\n"

    baseline_name = 'Original (427K)'

    for model_name, model_data in valid_models.items():
        # Extract metrics
        n_params = model_data['memory']['n_parameters']

        if rep_size in model_data['single_inference']:
            inf_time = model_data['single_inference'][rep_size]['mean_time_ms']
        else:
            inf_time = 0.0

        md_sps = model_data['md_simulation'].get('steps_per_second', 0)

        if device == 'cuda':
            memory = model_data['memory']['peak_memory_mb']
        else:
            memory = model_data['memory']['model_size_mb']

        # Get speedup
        if model_name in results.get('speedup_analysis', {}):
            speedup = results['speedup_analysis'][model_name].get('inference_speedup', 1.0)
        else:
            speedup = 1.0

        # Format row
        table += f"| {model_name} | {n_params/1000:.0f}K | {inf_time:.2f} | {md_sps:.1f} | {memory:.0f} | {speedup:.1f}x |\n"

    return table


def main():
    """Main benchmarking routine."""
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark all student models')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to benchmark on'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='benchmarks',
        help='Output directory for results'
    )
    parser.add_argument(
        '--skip-visualizations',
        action='store_true',
        help='Skip generating plots'
    )

    args = parser.parse_args()

    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    output_dir = Path(args.output_dir)

    # Run benchmarks
    results = run_full_benchmark(device=args.device, output_dir=output_dir)

    # Generate visualizations
    if not args.skip_visualizations:
        visualize_results(results, output_dir)

    # Print summary table
    summary = generate_summary_table(results)
    print(summary)

    # Save summary to file
    summary_file = output_dir / f'm6_summary_{args.device}.txt'
    with open(summary_file, 'w') as f:
        f.write(summary)

    logger.info(f"\n✓ All benchmarks complete! Results in {output_dir}/")

    return 0


if __name__ == '__main__':
    exit(main())
