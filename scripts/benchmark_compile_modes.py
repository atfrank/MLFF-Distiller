#!/usr/bin/env python3
"""
Comprehensive Torch.compile() Optimization Benchmark

This script tests all available compilation modes and configurations to find
the optimal settings for maximum inference speedup.

Week 2 Quick Wins Strategy:
- Test torch.compile() modes (if supported)
- Test CUDA graphs
- Test FP16 autocast
- Test combinations
- Find best configuration for 2-3x speedup

Author: CUDA Optimization Engineer
Date: 2025-11-24
Target: 10-15x total speedup (2-3x over current 5x baseline)
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import warnings

import numpy as np
import torch
from ase import Atoms
from ase.build import molecule

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from mlff_distiller.models.student_model import StudentForceField
from mlff_distiller.inference.ase_calculator import StudentForceFieldCalculator


def check_torch_compile_support() -> bool:
    """Check if torch.compile() is supported on current Python version."""
    try:
        import torch._dynamo
        return True
    except (ImportError, AttributeError):
        return False


def create_test_molecules() -> List[Tuple[str, Atoms]]:
    """Create test molecules of varying sizes."""
    molecules = [
        ("H2", molecule('H2')),
        ("H2O", molecule('H2O')),
        ("CH4", molecule('CH4')),
        ("Benzene", molecule('C6H6')),
    ]
    return molecules


class CompilationBenchmark:
    """Benchmark different compilation configurations."""

    def __init__(
        self,
        checkpoint_path: Path,
        device: str = 'cuda',
        n_warmup: int = 5,
        n_trials: int = 50
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.n_warmup = n_warmup
        self.n_trials = n_trials
        self.torch_compile_supported = check_torch_compile_support()

        if not self.torch_compile_supported:
            warnings.warn(
                "torch.compile() is NOT supported on this Python version. "
                "Only testing alternative optimizations (CUDA graphs, FP16)."
            )

    def benchmark_configuration(
        self,
        config_name: str,
        use_compile: bool = False,
        compile_mode: Optional[str] = None,
        compile_options: Optional[Dict] = None,
        use_fp16: bool = False,
        use_cuda_graphs: bool = False,
        use_jit: bool = False,
        molecules: Optional[List[Tuple[str, Atoms]]] = None
    ) -> Dict:
        """
        Benchmark a specific configuration.

        Args:
            config_name: Name of configuration
            use_compile: Use torch.compile()
            compile_mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune')
            compile_options: Additional compilation options
            use_fp16: Use FP16 autocast
            use_cuda_graphs: Use CUDA graphs
            use_jit: Use TorchScript JIT
            molecules: Test molecules

        Returns:
            Dictionary with benchmark results
        """
        print(f"\n{'='*60}")
        print(f"Benchmarking: {config_name}")
        print(f"{'='*60}")

        if molecules is None:
            molecules = create_test_molecules()

        # Create calculator with specified configuration
        try:
            if use_jit:
                # For JIT, we need pre-exported model
                jit_path = self.checkpoint_path.parent / "student_model_jit.pt"
                if not jit_path.exists():
                    print(f"WARNING: JIT model not found at {jit_path}, skipping")
                    return None

                calc = StudentForceFieldCalculator(
                    checkpoint_path=self.checkpoint_path,
                    device=self.device,
                    use_jit=True,
                    jit_path=jit_path,
                    use_fp16=use_fp16,
                    enable_timing=True
                )
            else:
                calc = StudentForceFieldCalculator(
                    checkpoint_path=self.checkpoint_path,
                    device=self.device,
                    use_compile=use_compile and self.torch_compile_supported,
                    use_fp16=use_fp16,
                    enable_timing=True
                )

                # Apply torch.compile() with custom settings if requested
                if use_compile and self.torch_compile_supported and compile_mode:
                    print(f"Compiling with mode={compile_mode}, options={compile_options}")
                    calc.model = torch.compile(
                        calc.model,
                        mode=compile_mode,
                        fullgraph=True,
                        options=compile_options or {}
                    )

        except Exception as e:
            print(f"ERROR: Failed to create calculator: {e}")
            return None

        # Benchmark each molecule
        results = {}

        for mol_name, atoms in molecules:
            print(f"\nMolecule: {mol_name} ({len(atoms)} atoms)")
            atoms.calc = calc

            # Warmup
            print(f"  Warmup ({self.n_warmup} iterations)...", end='', flush=True)
            for _ in range(self.n_warmup):
                try:
                    _ = atoms.get_potential_energy()
                    _ = atoms.get_forces()
                except Exception as e:
                    print(f"\n  ERROR during warmup: {e}")
                    return None
            print(" done")

            # Benchmark
            print(f"  Benchmarking ({self.n_trials} trials)...", end='', flush=True)
            times = []
            for _ in range(self.n_trials):
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start = time.perf_counter()

                try:
                    energy = atoms.get_potential_energy()
                    forces = atoms.get_forces()
                except Exception as e:
                    print(f"\n  ERROR during benchmark: {e}")
                    return None

                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms

            times = np.array(times)
            print(" done")

            results[mol_name] = {
                'n_atoms': len(atoms),
                'mean_ms': float(np.mean(times)),
                'std_ms': float(np.std(times)),
                'median_ms': float(np.median(times)),
                'min_ms': float(np.min(times)),
                'max_ms': float(np.max(times)),
                'p95_ms': float(np.percentile(times, 95)),
            }

            print(f"  Results: {results[mol_name]['mean_ms']:.3f} ± {results[mol_name]['std_ms']:.3f} ms")

        # Compute aggregate statistics
        all_times = [r['mean_ms'] for r in results.values()]
        aggregate = {
            'mean_across_molecules_ms': float(np.mean(all_times)),
            'std_across_molecules_ms': float(np.std(all_times)),
        }

        return {
            'config_name': config_name,
            'config': {
                'use_compile': use_compile and self.torch_compile_supported,
                'compile_mode': compile_mode,
                'compile_options': compile_options,
                'use_fp16': use_fp16,
                'use_cuda_graphs': use_cuda_graphs,
                'use_jit': use_jit,
            },
            'per_molecule': results,
            'aggregate': aggregate,
        }

    def run_all_configurations(self) -> Dict:
        """Run benchmarks for all configurations."""
        all_results = {}

        # Configuration 1: Baseline (no optimizations)
        result = self.benchmark_configuration(
            config_name="Baseline (No optimizations)",
            use_compile=False,
            use_fp16=False,
            use_cuda_graphs=False,
        )
        if result:
            all_results['baseline'] = result

        # Configuration 2: torch.compile() default (if supported)
        if self.torch_compile_supported:
            result = self.benchmark_configuration(
                config_name="torch.compile() - default mode",
                use_compile=True,
                compile_mode='default',
            )
            if result:
                all_results['compile_default'] = result

        # Configuration 3: torch.compile() reduce-overhead (if supported)
        if self.torch_compile_supported:
            result = self.benchmark_configuration(
                config_name="torch.compile() - reduce-overhead mode",
                use_compile=True,
                compile_mode='reduce-overhead',
            )
            if result:
                all_results['compile_reduce_overhead'] = result

        # Configuration 4: torch.compile() max-autotune (if supported)
        if self.torch_compile_supported:
            result = self.benchmark_configuration(
                config_name="torch.compile() - max-autotune mode",
                use_compile=True,
                compile_mode='max-autotune',
            )
            if result:
                all_results['compile_max_autotune'] = result

        # Configuration 5: torch.compile() with CUDA graphs (if supported)
        if self.torch_compile_supported:
            result = self.benchmark_configuration(
                config_name="torch.compile() - reduce-overhead + CUDA graphs",
                use_compile=True,
                compile_mode='reduce-overhead',
                compile_options={'triton.cudagraphs': True},
            )
            if result:
                all_results['compile_cudagraphs'] = result

        # Configuration 6: torch.compile() with max-autotune + CUDA graphs (if supported)
        if self.torch_compile_supported:
            result = self.benchmark_configuration(
                config_name="torch.compile() - max-autotune + CUDA graphs",
                use_compile=True,
                compile_mode='max-autotune',
                compile_options={
                    'max_autotune': True,
                    'triton.cudagraphs': True,
                },
            )
            if result:
                all_results['compile_max_cudagraphs'] = result

        # Configuration 7: FP16 autocast only
        result = self.benchmark_configuration(
            config_name="FP16 autocast (no compile)",
            use_compile=False,
            use_fp16=True,
        )
        if result:
            all_results['fp16_only'] = result

        # Configuration 8: torch.compile() + FP16 (if supported)
        if self.torch_compile_supported:
            result = self.benchmark_configuration(
                config_name="torch.compile() reduce-overhead + FP16",
                use_compile=True,
                compile_mode='reduce-overhead',
                use_fp16=True,
            )
            if result:
                all_results['compile_fp16'] = result

        # Configuration 9: torch.compile() + FP16 + CUDA graphs (if supported)
        if self.torch_compile_supported:
            result = self.benchmark_configuration(
                config_name="torch.compile() max-autotune + FP16 + CUDA graphs",
                use_compile=True,
                compile_mode='max-autotune',
                compile_options={
                    'max_autotune': True,
                    'triton.cudagraphs': True,
                },
                use_fp16=True,
            )
            if result:
                all_results['compile_fp16_cudagraphs'] = result

        # Configuration 10: TorchScript JIT (if available)
        jit_path = self.checkpoint_path.parent / "student_model_jit.pt"
        if jit_path.exists():
            result = self.benchmark_configuration(
                config_name="TorchScript JIT",
                use_jit=True,
            )
            if result:
                all_results['jit'] = result

            # TorchScript + FP16
            result = self.benchmark_configuration(
                config_name="TorchScript JIT + FP16",
                use_jit=True,
                use_fp16=True,
            )
            if result:
                all_results['jit_fp16'] = result

        return all_results

    def compare_results(self, all_results: Dict) -> Dict:
        """Compare all results and compute speedups."""
        if 'baseline' not in all_results:
            print("WARNING: No baseline results, cannot compute speedups")
            return {}

        baseline_time = all_results['baseline']['aggregate']['mean_across_molecules_ms']

        comparisons = {}
        for config_name, result in all_results.items():
            if config_name == 'baseline':
                continue

            config_time = result['aggregate']['mean_across_molecules_ms']
            speedup = baseline_time / config_time

            comparisons[config_name] = {
                'mean_time_ms': config_time,
                'baseline_time_ms': baseline_time,
                'speedup': speedup,
                'config': result['config'],
            }

        # Sort by speedup (best first)
        comparisons = dict(sorted(
            comparisons.items(),
            key=lambda x: x[1]['speedup'],
            reverse=True
        ))

        return comparisons


def print_comparison_table(comparisons: Dict):
    """Print formatted comparison table."""
    print("\n" + "="*100)
    print("SPEEDUP COMPARISON (sorted by speedup)")
    print("="*100)
    print(f"{'Configuration':<50} {'Time (ms)':<15} {'Speedup':<10} {'vs Baseline'}")
    print("-"*100)

    for config_name, data in comparisons.items():
        time_ms = data['mean_time_ms']
        speedup = data['speedup']
        baseline_ms = data['baseline_time_ms']

        print(f"{config_name:<50} {time_ms:>10.3f} ms   {speedup:>6.2f}x    {baseline_ms:.3f} ms")

    print("="*100)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark torch.compile() modes and optimization configurations"
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=Path('checkpoints/best_model.pt'),
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )
    parser.add_argument(
        '--n-warmup',
        type=int,
        default=5,
        help='Number of warmup iterations'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=50,
        help='Number of benchmark trials'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('benchmarks/compile_modes_results.json'),
        help='Output JSON file for results'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick benchmark (fewer trials)'
    )

    args = parser.parse_args()

    if args.quick:
        args.n_warmup = 3
        args.n_trials = 20

    # Check checkpoint exists
    if not args.checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = 'cpu'

    print("="*100)
    print("TORCH.COMPILE() OPTIMIZATION BENCHMARK")
    print("="*100)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version.split()[0]}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    print(f"torch.compile() supported: {check_torch_compile_support()}")
    print(f"Warmup iterations: {args.n_warmup}")
    print(f"Benchmark trials: {args.n_trials}")
    print("="*100)

    # Run benchmarks
    benchmark = CompilationBenchmark(
        checkpoint_path=args.checkpoint,
        device=args.device,
        n_warmup=args.n_warmup,
        n_trials=args.n_trials
    )

    all_results = benchmark.run_all_configurations()

    # Compare results
    comparisons = benchmark.compare_results(all_results)

    # Print comparison table
    print_comparison_table(comparisons)

    # Find best configuration
    if comparisons:
        best_config = list(comparisons.keys())[0]
        best_speedup = comparisons[best_config]['speedup']
        print(f"\nBEST CONFIGURATION: {best_config}")
        print(f"SPEEDUP: {best_speedup:.2f}x over baseline")
        print(f"Time: {comparisons[best_config]['mean_time_ms']:.3f} ms")

    # Save results
    output_data = {
        'metadata': {
            'checkpoint': str(args.checkpoint),
            'device': args.device,
            'pytorch_version': torch.__version__,
            'python_version': sys.version.split()[0],
            'cuda_device': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            'torch_compile_supported': check_torch_compile_support(),
            'n_warmup': args.n_warmup,
            'n_trials': args.n_trials,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
        'all_results': all_results,
        'comparisons': comparisons,
        'best_config': list(comparisons.keys())[0] if comparisons else None,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {args.output}")
    print("\nDone!")


if __name__ == '__main__':
    main()
