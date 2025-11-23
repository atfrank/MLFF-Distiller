"""
MD-Specific Profiling Framework for MLFF Distiller

This module provides profiling utilities specifically designed for molecular dynamics
workloads where models are called millions of times sequentially. Unlike standard ML
benchmarking which focuses on throughput, MD profiling emphasizes:

1. Per-call latency (not batch throughput)
2. Memory stability over long trajectories (1000+ steps)
3. Hotspot identification in forward pass components
4. GPU utilization patterns
5. CPU-GPU transfer overhead

Key Philosophy:
- MD simulations care about LATENCY, not throughput
- Memory leaks are critical failures (will crash long simulations)
- Variance/outliers in timing cause MD instability
- Need to understand WHERE time is spent to optimize effectively

Usage:
    from mlff_distiller.cuda.md_profiler import MDProfiler, profile_md_trajectory
    from mlff_distiller.models import OrbCalculator

    # Create profiler
    profiler = MDProfiler(device='cuda')

    # Profile calculator on trajectory
    calc = OrbCalculator(model_name='orb-v2', device='cuda')
    trajectory = [generate_atoms() for _ in range(1000)]

    results = profiler.profile_calculator(calc, trajectory)
    print(results.summary())

    # Save detailed report
    results.save_json('profiling_reports/orb_v2_profile.json')
"""

import json
import time
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator

from .benchmark_utils import CUDATimer, benchmark_md_trajectory
from .device_utils import (
    empty_cache,
    get_gpu_memory_info,
    get_peak_memory_allocated,
    reset_peak_memory_stats,
    synchronize_device,
)


@dataclass
class MDProfileResult:
    """
    Container for MD profiling results with comprehensive statistics.

    This extends BenchmarkResult with MD-specific metrics:
    - Per-call latency distributions
    - Component-level timing (energy, forces, stress separately)
    - Memory allocation patterns over trajectory
    - Hotspot identification
    """

    name: str
    model_name: str
    n_steps: int
    device: str

    # Latency statistics (ms per call)
    latencies_ms: List[float]
    mean_latency_ms: float
    median_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float

    # Component timing (if available)
    energy_time_ms: Optional[float] = None
    forces_time_ms: Optional[float] = None
    stress_time_ms: Optional[float] = None

    # Memory statistics (GB)
    memory_initial_gb: float = 0.0
    memory_final_gb: float = 0.0
    memory_peak_gb: float = 0.0
    memory_per_step_gb: List[float] = field(default_factory=list)

    # System information
    n_atoms: Optional[int] = None
    system_size: Optional[str] = None

    # Performance metrics
    total_time_s: float = 0.0
    steps_per_second: float = 0.0

    # Metadata
    timestamp: str = ""
    notes: str = ""

    @property
    def memory_delta_gb(self) -> float:
        """Memory change over trajectory."""
        return self.memory_final_gb - self.memory_initial_gb

    @property
    def memory_stable(self) -> bool:
        """Check if memory is stable (< 10 MB increase)."""
        return abs(self.memory_delta_gb * 1024) < 10.0

    @property
    def us_per_atom(self) -> Optional[float]:
        """Microseconds per atom per call."""
        if self.n_atoms is None:
            return None
        return (self.mean_latency_ms * 1000) / self.n_atoms

    def summary(self) -> str:
        """Generate formatted summary report."""
        lines = [
            f"\n{'=' * 80}",
            f"MD Profile Results: {self.name}",
            f"{'=' * 80}",
            f"Model: {self.model_name}",
            f"Device: {self.device}",
            f"Trajectory Steps: {self.n_steps}",
        ]

        if self.n_atoms is not None:
            lines.append(f"System Size: {self.n_atoms} atoms")

        lines.extend([
            f"",
            f"Latency Statistics (ms per step):",
            f"  Mean:     {self.mean_latency_ms:8.4f} ms ± {self.std_latency_ms:.4f}",
            f"  Median:   {self.median_latency_ms:8.4f} ms",
            f"  Min:      {self.min_latency_ms:8.4f} ms",
            f"  Max:      {self.max_latency_ms:8.4f} ms",
            f"  P95:      {self.p95_latency_ms:8.4f} ms",
            f"  P99:      {self.p99_latency_ms:8.4f} ms",
        ])

        if self.us_per_atom is not None:
            lines.append(f"  Per Atom: {self.us_per_atom:8.2f} µs/atom")

        # Component timing
        if self.energy_time_ms is not None:
            lines.extend([
                f"",
                f"Component Timing:",
                f"  Energy:   {self.energy_time_ms:8.4f} ms",
            ])
            if self.forces_time_ms is not None:
                lines.append(f"  Forces:   {self.forces_time_ms:8.4f} ms")
            if self.stress_time_ms is not None:
                lines.append(f"  Stress:   {self.stress_time_ms:8.4f} ms")

        lines.extend([
            f"",
            f"Throughput:",
            f"  Steps/sec: {self.steps_per_second:,.1f}",
            f"  Total time: {self.total_time_s:.2f} s",
            f"",
            f"Memory Usage (GB):",
            f"  Initial:   {self.memory_initial_gb:.4f}",
            f"  Final:     {self.memory_final_gb:.4f}",
            f"  Delta:     {self.memory_delta_gb:+.4f}",
            f"  Peak:      {self.memory_peak_gb:.4f}",
        ])

        # Memory stability warning
        if not self.memory_stable:
            lines.extend([
                f"",
                f"WARNING: Memory increased by {self.memory_delta_gb * 1024:.2f} MB!",
                f"This indicates a potential memory leak that will cause issues in long MD runs.",
            ])
        else:
            lines.append(f"  Status:    STABLE (no leak detected)")

        lines.append(f"{'=' * 80}\n")

        return '\n'.join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert lists to regular Python lists (not numpy)
        if isinstance(result['latencies_ms'], np.ndarray):
            result['latencies_ms'] = result['latencies_ms'].tolist()
        if isinstance(result['memory_per_step_gb'], np.ndarray):
            result['memory_per_step_gb'] = result['memory_per_step_gb'].tolist()
        return result

    def save_json(self, filepath: Union[str, Path]) -> None:
        """Save results to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        print(f"Profile results saved to: {filepath}")

    @classmethod
    def load_json(cls, filepath: Union[str, Path]) -> 'MDProfileResult':
        """Load results from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)


class MDProfiler:
    """
    Comprehensive profiler for MD workloads.

    Provides detailed profiling of force field calculators on realistic MD trajectories:
    - Per-step latency measurement
    - Component-level timing (energy, forces, stress)
    - Memory allocation tracking
    - Hotspot identification
    - PyTorch profiler integration

    Args:
        device: Device to profile on ('cuda' or 'cpu')
        profile_memory: Whether to track detailed memory allocation
        warmup_steps: Number of warmup steps before profiling

    Example:
        >>> profiler = MDProfiler(device='cuda')
        >>> calc = OrbCalculator(model_name='orb-v2', device='cuda')
        >>> trajectory = [generate_atoms() for _ in range(1000)]
        >>> results = profiler.profile_calculator(calc, trajectory)
        >>> print(results.summary())
    """

    def __init__(
        self,
        device: Union[str, torch.device] = 'cuda',
        profile_memory: bool = True,
        warmup_steps: int = 10,
    ):
        """Initialize MD profiler."""
        self.device = torch.device(device) if isinstance(device, str) else device
        self.profile_memory = profile_memory
        self.warmup_steps = warmup_steps

        # CUDA timer for precise measurements
        self.timer = CUDATimer(self.device)

    def profile_calculator(
        self,
        calculator: Calculator,
        trajectory: List[Atoms],
        properties: List[str] = ['energy', 'forces'],
        check_memory_leak: bool = True,
        name: Optional[str] = None,
    ) -> MDProfileResult:
        """
        Profile an ASE calculator on MD trajectory.

        Args:
            calculator: ASE Calculator instance (e.g., OrbCalculator)
            trajectory: List of Atoms objects (MD trajectory)
            properties: Properties to compute at each step
            check_memory_leak: Whether to check for memory leaks
            name: Optional name for this profiling run

        Returns:
            MDProfileResult with comprehensive statistics
        """
        if name is None:
            name = f"{calculator.__class__.__name__} Profile"

        n_steps = len(trajectory)

        # Get model name if available
        model_name = getattr(calculator, 'model_name', calculator.__class__.__name__)

        # Get system size from first frame
        n_atoms = len(trajectory[0]) if trajectory else None

        print(f"\nProfiling {name}")
        print(f"  Model: {model_name}")
        print(f"  Steps: {n_steps}")
        print(f"  Atoms: {n_atoms}")
        print(f"  Properties: {properties}")

        # Warmup
        print(f"\nWarmup ({self.warmup_steps} steps)...")
        for i in range(min(self.warmup_steps, n_steps)):
            atoms = trajectory[i].copy()
            atoms.calc = calculator
            for prop in properties:
                if prop == 'energy':
                    _ = atoms.get_potential_energy()
                elif prop == 'forces':
                    _ = atoms.get_forces()
                elif prop == 'stress':
                    _ = atoms.get_stress()

        # Synchronize and clear cache
        if self.device.type == 'cuda':
            synchronize_device(self.device)
            empty_cache()

        # Record initial memory
        mem_initial = get_gpu_memory_info(self.device)['allocated']
        reset_peak_memory_stats(self.device)

        # Profile trajectory
        print(f"\nProfiling trajectory...")
        latencies_ms = []
        memory_per_step_gb = []

        # Component timing (if we can measure separately)
        energy_times = []
        forces_times = []
        stress_times = []

        start_time = time.perf_counter()

        for i, atoms_template in enumerate(trajectory):
            # Copy atoms to avoid caching issues
            atoms = atoms_template.copy()
            atoms.calc = calculator

            # Time this step
            self.timer.start()

            # Compute properties
            for prop in properties:
                if prop == 'energy':
                    comp_start = time.perf_counter()
                    _ = atoms.get_potential_energy()
                    comp_time = (time.perf_counter() - comp_start) * 1000
                    energy_times.append(comp_time)
                elif prop == 'forces':
                    comp_start = time.perf_counter()
                    _ = atoms.get_forces()
                    comp_time = (time.perf_counter() - comp_start) * 1000
                    forces_times.append(comp_time)
                elif prop == 'stress':
                    comp_start = time.perf_counter()
                    _ = atoms.get_stress()
                    comp_time = (time.perf_counter() - comp_start) * 1000
                    stress_times.append(comp_time)

            # Record total step time
            step_time = self.timer.stop()
            latencies_ms.append(step_time)

            # Track memory if enabled
            if self.profile_memory and i % 10 == 0:  # Every 10th step
                mem_info = get_gpu_memory_info(self.device)
                memory_per_step_gb.append(mem_info['allocated'])

            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"  Completed {i + 1}/{n_steps} steps...")

        total_time = time.perf_counter() - start_time

        # Synchronize before final memory measurement
        if self.device.type == 'cuda':
            synchronize_device(self.device)

        # Record final memory
        mem_final = get_gpu_memory_info(self.device)['allocated']
        mem_peak = get_peak_memory_allocated(self.device)

        # Check for memory leak
        if check_memory_leak:
            mem_delta_mb = (mem_final - mem_initial) * 1024
            if mem_delta_mb > 10.0:
                warnings.warn(
                    f"Memory leak detected: {mem_delta_mb:.2f} MB increase over {n_steps} steps. "
                    f"This will cause OOM errors in long MD runs.",
                    RuntimeWarning,
                )

        # Compute statistics
        latencies_array = np.array(latencies_ms)

        # Create result object
        result = MDProfileResult(
            name=name,
            model_name=model_name,
            n_steps=n_steps,
            device=str(self.device),
            latencies_ms=latencies_ms,
            mean_latency_ms=float(np.mean(latencies_array)),
            median_latency_ms=float(np.median(latencies_array)),
            std_latency_ms=float(np.std(latencies_array)),
            min_latency_ms=float(np.min(latencies_array)),
            max_latency_ms=float(np.max(latencies_array)),
            p95_latency_ms=float(np.percentile(latencies_array, 95)),
            p99_latency_ms=float(np.percentile(latencies_array, 99)),
            energy_time_ms=float(np.mean(energy_times)) if energy_times else None,
            forces_time_ms=float(np.mean(forces_times)) if forces_times else None,
            stress_time_ms=float(np.mean(stress_times)) if stress_times else None,
            memory_initial_gb=mem_initial,
            memory_final_gb=mem_final,
            memory_peak_gb=mem_peak,
            memory_per_step_gb=memory_per_step_gb,
            n_atoms=n_atoms,
            system_size=f"{n_atoms} atoms" if n_atoms else None,
            total_time_s=total_time,
            steps_per_second=n_steps / total_time if total_time > 0 else 0.0,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        print(f"\nProfiling complete!")

        return result

    def compare_calculators(
        self,
        calculators: Dict[str, Calculator],
        trajectory: List[Atoms],
        properties: List[str] = ['energy', 'forces'],
    ) -> Dict[str, MDProfileResult]:
        """
        Compare multiple calculators on same trajectory.

        Args:
            calculators: Dict mapping names to Calculator instances
            trajectory: List of Atoms objects
            properties: Properties to compute

        Returns:
            Dict mapping calculator names to MDProfileResults
        """
        results = {}

        print(f"\n{'=' * 80}")
        print(f"Comparing {len(calculators)} calculators on {len(trajectory)}-step trajectory")
        print(f"{'=' * 80}")

        for name, calc in calculators.items():
            print(f"\n\nProfiling: {name}")
            print(f"{'=' * 80}")

            result = self.profile_calculator(
                calc,
                trajectory,
                properties=properties,
                name=name,
            )
            results[name] = result

        # Print comparison table
        self._print_comparison_table(results)

        return results

    def _print_comparison_table(self, results: Dict[str, MDProfileResult]) -> None:
        """Print formatted comparison table."""
        if not results:
            return

        print("\n" + "=" * 100)
        print("Calculator Comparison")
        print("=" * 100)

        # Header
        print(f"{'Calculator':<30} {'Mean (ms)':>12} {'P95 (ms)':>12} {'Steps/s':>12} "
              f"{'Memory (GB)':>12} {'Speedup':>10}")
        print("-" * 100)

        # Baseline (first entry)
        baseline_name = list(results.keys())[0]
        baseline_mean = results[baseline_name].mean_latency_ms

        # Rows
        for name, result in results.items():
            speedup = baseline_mean / result.mean_latency_ms if result.mean_latency_ms > 0 else 0
            speedup_str = f"{speedup:.2f}x"

            print(f"{name:<30} {result.mean_latency_ms:12.4f} {result.p95_latency_ms:12.4f} "
                  f"{result.steps_per_second:12.1f} {result.memory_peak_gb:12.4f} {speedup_str:>10}")

        print("=" * 100 + "\n")


def profile_md_trajectory(
    calculator: Calculator,
    trajectory: List[Atoms],
    device: Union[str, torch.device] = 'cuda',
    properties: List[str] = ['energy', 'forces'],
    output_file: Optional[Union[str, Path]] = None,
) -> MDProfileResult:
    """
    Convenience function to profile a calculator on MD trajectory.

    Args:
        calculator: ASE Calculator instance
        trajectory: List of Atoms objects
        device: Device to use for profiling
        properties: Properties to compute
        output_file: Optional JSON file to save results

    Returns:
        MDProfileResult

    Example:
        >>> from mlff_distiller.models import OrbCalculator
        >>> calc = OrbCalculator(model_name='orb-v2', device='cuda')
        >>> trajectory = [generate_atoms() for _ in range(1000)]
        >>> results = profile_md_trajectory(calc, trajectory, output_file='results.json')
    """
    profiler = MDProfiler(device=device)
    results = profiler.profile_calculator(calculator, trajectory, properties=properties)

    if output_file is not None:
        results.save_json(output_file)

    return results


def identify_hotspots(
    result: MDProfileResult,
    threshold_pct: float = 10.0,
) -> Dict[str, Any]:
    """
    Identify computational hotspots from profiling results.

    Args:
        result: MDProfileResult from profiling
        threshold_pct: Threshold percentage for hotspot identification

    Returns:
        Dictionary with hotspot analysis and recommendations
    """
    hotspots = {
        'summary': f"Hotspot analysis for {result.name}",
        'total_time_ms': result.mean_latency_ms,
        'components': {},
        'recommendations': [],
    }

    # Analyze component timing
    total_time = result.mean_latency_ms

    if result.energy_time_ms is not None:
        energy_pct = (result.energy_time_ms / total_time) * 100
        hotspots['components']['energy'] = {
            'time_ms': result.energy_time_ms,
            'percentage': energy_pct,
            'is_hotspot': energy_pct > threshold_pct,
        }

        if energy_pct > threshold_pct:
            hotspots['recommendations'].append(
                f"Energy computation is {energy_pct:.1f}% of total time - consider optimization"
            )

    if result.forces_time_ms is not None:
        forces_pct = (result.forces_time_ms / total_time) * 100
        hotspots['components']['forces'] = {
            'time_ms': result.forces_time_ms,
            'percentage': forces_pct,
            'is_hotspot': forces_pct > threshold_pct,
        }

        if forces_pct > threshold_pct:
            hotspots['recommendations'].append(
                f"Forces computation is {forces_pct:.1f}% of total time - primary optimization target"
            )

    # Memory analysis
    if not result.memory_stable:
        hotspots['recommendations'].append(
            f"Memory leak detected ({result.memory_delta_gb * 1024:.2f} MB) - investigate allocation patterns"
        )

    # Latency variance analysis
    cv = (result.std_latency_ms / result.mean_latency_ms) * 100  # Coefficient of variation
    if cv > 20:
        hotspots['recommendations'].append(
            f"High latency variance (CV={cv:.1f}%) - investigate outliers and GPU utilization"
        )

    # P99 latency analysis
    p99_ratio = result.p99_latency_ms / result.mean_latency_ms
    if p99_ratio > 2.0:
        hotspots['recommendations'].append(
            f"P99 latency is {p99_ratio:.1f}x mean - investigate tail latencies"
        )

    return hotspots


__all__ = [
    'MDProfiler',
    'MDProfileResult',
    'profile_md_trajectory',
    'identify_hotspots',
]
