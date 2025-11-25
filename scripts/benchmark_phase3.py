#!/usr/bin/env python3
"""
Phase 3 CUDA Optimization Benchmark Suite

Comprehensive benchmarking script for Phase 3 optimization work.
Tests various optimization configurations and tracks performance improvements.

Author: Testing & Benchmarking Engineer
Date: 2025-11-24
Issue: #28
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import torch
import numpy as np
from ase import Atoms
from ase.build import molecule

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mlff_distiller.models.student_model import StudentForceField

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Phase3Benchmarker:
    """
    Comprehensive benchmarking for Phase 3 optimizations.

    Tests:
    - Baseline (PyTorch eager)
    - TorchScript JIT
    - TorchScript + torch-cluster
    - torch.compile() (if available)
    - FP16 mixed precision
    - Combined optimizations
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        warmup_runs: int = 10,
        benchmark_runs: int = 100
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs

        # Results storage
        self.results: Dict = {
            'config': {
                'checkpoint': str(checkpoint_path),
                'device': device,
                'warmup_runs': warmup_runs,
                'benchmark_runs': benchmark_runs,
                'pytorch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'optimizations': {}
        }

        # Test molecules
        self.test_molecules = self._create_test_molecules()

        logger.info(f"Initialized Phase3Benchmarker")
        logger.info(f"Device: {device}")
        logger.info(f"Test molecules: {len(self.test_molecules)}")

    def _create_test_molecules(self) -> List[Atoms]:
        """Create test molecules of various sizes."""
        molecules = []

        # Small molecules (3-10 atoms)
        for name in ['H2O', 'NH3', 'CH4']:
            try:
                mol = molecule(name)
                molecules.append(mol)
            except:
                pass

        # Medium molecules (10-50 atoms)
        for name in ['C6H6', 'C8H10', 'caffeine']:
            try:
                mol = molecule(name)
                molecules.append(mol)
            except:
                pass

        # Create synthetic larger molecules
        for n_atoms in [20, 50, 100]:
            atoms = Atoms(
                symbols=['C'] * n_atoms,
                positions=np.random.randn(n_atoms, 3) * 3.0
            )
            molecules.append(atoms)

        logger.info(f"Created {len(molecules)} test molecules")
        for i, mol in enumerate(molecules):
            logger.info(f"  Molecule {i}: {len(mol)} atoms, {mol.get_chemical_formula()}")

        return molecules

    def _molecule_to_tensors(
        self,
        atoms: Atoms,
        requires_grad: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert ASE Atoms to tensors."""
        atomic_numbers = torch.tensor(
            atoms.get_atomic_numbers(),
            dtype=torch.long,
            device=self.device
        )

        positions = torch.tensor(
            atoms.get_positions(),
            dtype=torch.float32,
            device=self.device,
            requires_grad=requires_grad
        )

        return atomic_numbers, positions

    def benchmark_single_config(
        self,
        model: torch.nn.Module,
        config_name: str,
        test_energy_forces: bool = True
    ) -> Dict:
        """
        Benchmark a single optimization configuration.

        Args:
            model: Model to benchmark
            config_name: Name of configuration (e.g., "baseline", "jit", "jit+torch-cluster")
            test_energy_forces: Whether to test force computation

        Returns:
            Dictionary of benchmark results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmarking: {config_name}")
        logger.info(f"{'='*60}")

        results = {
            'config_name': config_name,
            'by_size': {},
            'aggregated': {
                'all_times_ms': [],
                'mean_ms': None,
                'std_ms': None,
                'median_ms': None,
                'p95_ms': None,
                'p99_ms': None
            }
        }

        for mol_idx, mol in enumerate(self.test_molecules):
            n_atoms = len(mol)
            logger.info(f"\nMolecule {mol_idx + 1}/{len(self.test_molecules)}: {n_atoms} atoms")

            atomic_numbers, positions = self._molecule_to_tensors(
                mol,
                requires_grad=test_energy_forces
            )

            # Warmup
            logger.info(f"  Warmup ({self.warmup_runs} runs)...")
            for _ in range(self.warmup_runs):
                with torch.no_grad():
                    _ = model(atomic_numbers, positions)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            # Benchmark energy only
            logger.info(f"  Benchmarking energy ({self.benchmark_runs} runs)...")
            times_ms = []

            for _ in range(self.benchmark_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                start = time.perf_counter()

                with torch.no_grad():
                    energy = model(atomic_numbers, positions)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
                times_ms.append(elapsed)

            # Statistics
            mean_ms = np.mean(times_ms)
            std_ms = np.std(times_ms)
            median_ms = np.median(times_ms)
            p95_ms = np.percentile(times_ms, 95)
            p99_ms = np.percentile(times_ms, 99)

            logger.info(f"  Energy inference: {mean_ms:.4f} ± {std_ms:.4f} ms")
            logger.info(f"    Median: {median_ms:.4f} ms, P95: {p95_ms:.4f} ms, P99: {p99_ms:.4f} ms")

            results['by_size'][n_atoms] = {
                'n_atoms': n_atoms,
                'mean_ms': float(mean_ms),
                'std_ms': float(std_ms),
                'median_ms': float(median_ms),
                'p95_ms': float(p95_ms),
                'p99_ms': float(p99_ms),
                'times_ms': [float(t) for t in times_ms]
            }

            results['aggregated']['all_times_ms'].extend(times_ms)

            # Test force computation (more expensive)
            if test_energy_forces:
                logger.info(f"  Benchmarking energy+forces (10 runs)...")
                force_times_ms = []

                for _ in range(10):
                    atomic_numbers, positions = self._molecule_to_tensors(mol, requires_grad=True)

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    start = time.perf_counter()

                    energy = model(atomic_numbers, positions)
                    forces = -torch.autograd.grad(
                        energy,
                        positions,
                        create_graph=False,
                        retain_graph=False
                    )[0]

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    elapsed = (time.perf_counter() - start) * 1000
                    force_times_ms.append(elapsed)

                force_mean_ms = np.mean(force_times_ms)
                logger.info(f"  Energy+Forces: {force_mean_ms:.4f} ms")

                results['by_size'][n_atoms]['force_mean_ms'] = float(force_mean_ms)

        # Compute aggregated statistics
        all_times = results['aggregated']['all_times_ms']
        results['aggregated']['mean_ms'] = float(np.mean(all_times))
        results['aggregated']['std_ms'] = float(np.std(all_times))
        results['aggregated']['median_ms'] = float(np.median(all_times))
        results['aggregated']['p95_ms'] = float(np.percentile(all_times, 95))
        results['aggregated']['p99_ms'] = float(np.percentile(all_times, 99))

        # Don't store all individual times in final output (too large)
        del results['aggregated']['all_times_ms']

        logger.info(f"\n{config_name} Overall Performance:")
        logger.info(f"  Mean: {results['aggregated']['mean_ms']:.4f} ms")
        logger.info(f"  Median: {results['aggregated']['median_ms']:.4f} ms")
        logger.info(f"  P95: {results['aggregated']['p95_ms']:.4f} ms")

        return results

    def benchmark_baseline(self) -> Dict:
        """Benchmark baseline PyTorch eager model."""
        logger.info("\n" + "="*60)
        logger.info("CONFIGURATION: Baseline (PyTorch Eager)")
        logger.info("="*60)

        model = StudentForceField.load(self.checkpoint_path, device=self.device)
        model.eval()

        return self.benchmark_single_config(model, "baseline")

    def benchmark_torchscript(self) -> Dict:
        """Benchmark TorchScript JIT compiled model."""
        logger.info("\n" + "="*60)
        logger.info("CONFIGURATION: TorchScript JIT")
        logger.info("="*60)

        # Check if JIT model exists
        jit_path = self.checkpoint_path.parent / 'student_model_jit.pt'

        if jit_path.exists():
            logger.info(f"Loading pre-compiled JIT model from {jit_path}")
            model = torch.jit.load(str(jit_path), map_location=self.device)
        else:
            logger.info("Compiling model with TorchScript...")
            model = StudentForceField.load(self.checkpoint_path, device=self.device)
            model.eval()

            # Create example inputs
            example_atoms = self.test_molecules[0]
            atomic_numbers, positions = self._molecule_to_tensors(example_atoms)

            # Trace the model
            model = torch.jit.trace(model, (atomic_numbers, positions))

            # Save for future use
            torch.jit.save(model, str(jit_path))
            logger.info(f"Saved JIT model to {jit_path}")

        return self.benchmark_single_config(model, "torchscript")

    def benchmark_torch_cluster(self) -> Dict:
        """Benchmark model with torch-cluster neighbor search."""
        logger.info("\n" + "="*60)
        logger.info("CONFIGURATION: TorchScript + torch-cluster")
        logger.info("="*60)

        try:
            import torch_cluster
            logger.info("torch-cluster available!")
        except ImportError:
            logger.warning("torch-cluster not installed, skipping")
            return None

        # Load model with torch-cluster enabled
        model = StudentForceField.load(self.checkpoint_path, device=self.device)

        # TODO: Modify model to use torch-cluster
        # This will be implemented after torch-cluster integration
        logger.warning("torch-cluster integration not yet complete")

        return None

    def benchmark_fp16(self) -> Dict:
        """Benchmark model with FP16 mixed precision."""
        logger.info("\n" + "="*60)
        logger.info("CONFIGURATION: FP16 Mixed Precision")
        logger.info("="*60)

        model = StudentForceField.load(self.checkpoint_path, device=self.device)
        model.eval()

        # Wrap model with autocast
        class FP16Model(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, atomic_numbers, positions, cell=None, pbc=None, batch=None):
                with torch.cuda.amp.autocast():
                    return self.model(atomic_numbers, positions, cell, pbc, batch)

        fp16_model = FP16Model(model)

        return self.benchmark_single_config(fp16_model, "fp16")

    def benchmark_torch_compile(self) -> Optional[Dict]:
        """Benchmark model with torch.compile()."""
        logger.info("\n" + "="*60)
        logger.info("CONFIGURATION: torch.compile()")
        logger.info("="*60)

        # Check if torch.compile is available
        if not hasattr(torch, 'compile'):
            logger.warning("torch.compile() not available (requires PyTorch 2.0+)")
            return None

        try:
            model = StudentForceField.load(self.checkpoint_path, device=self.device)
            model.eval()

            # Compile with default mode
            logger.info("Compiling with mode='default'...")
            compiled_model = torch.compile(model, mode='default')

            return self.benchmark_single_config(compiled_model, "torch_compile_default")

        except Exception as e:
            logger.error(f"torch.compile() failed: {e}")
            return None

    def run_all_benchmarks(self) -> Dict:
        """Run all available benchmarks and compare results."""
        logger.info("\n" + "="*80)
        logger.info("PHASE 3 COMPREHENSIVE BENCHMARK SUITE")
        logger.info("="*80)

        # Run each configuration
        baseline_results = self.benchmark_baseline()
        self.results['optimizations']['baseline'] = baseline_results

        jit_results = self.benchmark_torchscript()
        self.results['optimizations']['torchscript'] = jit_results

        fp16_results = self.benchmark_fp16()
        if fp16_results:
            self.results['optimizations']['fp16'] = fp16_results

        compile_results = self.benchmark_torch_compile()
        if compile_results:
            self.results['optimizations']['torch_compile'] = compile_results

        cluster_results = self.benchmark_torch_cluster()
        if cluster_results:
            self.results['optimizations']['torch_cluster'] = cluster_results

        # Compute speedups
        self._compute_speedups()

        # Print summary
        self._print_summary()

        return self.results

    def _compute_speedups(self):
        """Compute speedup ratios relative to baseline."""
        baseline_time = self.results['optimizations']['baseline']['aggregated']['mean_ms']

        speedups = {}
        for config_name, config_results in self.results['optimizations'].items():
            if config_results is None:
                continue

            config_time = config_results['aggregated']['mean_ms']
            speedup = baseline_time / config_time
            speedups[config_name] = {
                'time_ms': config_time,
                'speedup_vs_baseline': speedup
            }

        self.results['speedups'] = speedups

    def _print_summary(self):
        """Print comprehensive summary of results."""
        logger.info("\n" + "="*80)
        logger.info("BENCHMARK SUMMARY")
        logger.info("="*80)

        baseline_time = self.results['optimizations']['baseline']['aggregated']['mean_ms']

        print(f"\nBaseline Performance: {baseline_time:.4f} ms")
        print(f"\n{'Configuration':<30} {'Time (ms)':<15} {'Speedup':<10}")
        print("-" * 55)

        for config_name, speedup_data in self.results['speedups'].items():
            time_ms = speedup_data['time_ms']
            speedup = speedup_data['speedup_vs_baseline']
            print(f"{config_name:<30} {time_ms:<15.4f} {speedup:<10.2f}x")

        print("\n" + "="*80)

        # Check if target achieved
        best_speedup = max(
            data['speedup_vs_baseline']
            for data in self.results['speedups'].values()
        )

        print(f"\nBest Speedup: {best_speedup:.2f}x")

        if best_speedup >= 5.0:
            print("STATUS: TARGET ACHIEVED (5x speedup)")
        elif best_speedup >= 3.0:
            print("STATUS: GOOD PROGRESS (3x speedup)")
        else:
            print("STATUS: MORE WORK NEEDED (<3x speedup)")

        print("="*80 + "\n")

    def save_results(self, output_path: Path):
        """Save benchmark results to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 3 CUDA Optimization Benchmarks")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='benchmarks/phase3_results.json',
        help='Output JSON file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda/cpu)'
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=10,
        help='Number of warmup runs'
    )
    parser.add_argument(
        '--runs',
        type=int,
        default=100,
        help='Number of benchmark runs'
    )

    args = parser.parse_args()

    # Run benchmarks
    benchmarker = Phase3Benchmarker(
        checkpoint_path=args.checkpoint,
        device=args.device,
        warmup_runs=args.warmup,
        benchmark_runs=args.runs
    )

    results = benchmarker.run_all_benchmarks()

    # Save results
    benchmarker.save_results(Path(args.output))


if __name__ == '__main__':
    main()
