"""
MD Trajectory Benchmarking Framework

This module provides tools for benchmarking molecular dynamics trajectories with
focus on metrics critical for production MD use cases:

1. Per-call latency (not just total time)
2. Memory stability over long runs
3. Energy conservation (correctness check for NVE)
4. Trajectory execution time

The benchmarks measure performance on realistic MD workloads (1000+ steps) to
capture overhead, memory behavior, and sustained performance that single-inference
benchmarks miss.

Author: Testing & Benchmark Engineer
Date: 2025-11-23
"""

import json
import time
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from ase import Atoms, units
from ase.build import bulk, molecule
from ase.calculators.calculator import Calculator
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.verlet import VelocityVerlet


class MDProtocol(str, Enum):
    """MD protocol/ensemble types."""
    NVE = "NVE"  # Microcanonical (constant N, V, E) - tests energy conservation
    NVT = "NVT"  # Canonical (constant N, V, T) - Langevin dynamics
    NPT = "NPT"  # Isothermal-isobaric (constant N, P, T) - pressure coupling


@dataclass
class BenchmarkResults:
    """Container for MD trajectory benchmark results with statistical analysis."""

    # Metadata
    name: str
    protocol: str
    n_steps: int
    n_atoms: int
    system_type: str
    device: str

    # Timing results (milliseconds per step)
    step_times_ms: List[float]

    # Memory tracking (GB)
    memory_before_gb: float
    memory_after_gb: float
    peak_memory_gb: float
    memory_samples_gb: List[float] = field(default_factory=list)

    # Energy tracking (eV)
    energies: List[float] = field(default_factory=list)

    # Computed statistics (populated in __post_init__)
    total_time_s: float = field(init=False)
    mean_step_time_ms: float = field(init=False)
    std_step_time_ms: float = field(init=False)
    median_step_time_ms: float = field(init=False)
    min_step_time_ms: float = field(init=False)
    max_step_time_ms: float = field(init=False)
    p95_step_time_ms: float = field(init=False)
    p99_step_time_ms: float = field(init=False)
    steps_per_second: float = field(init=False)
    energy_drift: Optional[float] = field(init=False, default=None)
    energy_std: Optional[float] = field(init=False, default=None)

    def __post_init__(self):
        """Compute statistics from raw data."""
        times_array = np.array(self.step_times_ms)

        self.total_time_s = np.sum(times_array) / 1000.0
        self.mean_step_time_ms = float(np.mean(times_array))
        self.std_step_time_ms = float(np.std(times_array))
        self.median_step_time_ms = float(np.median(times_array))
        self.min_step_time_ms = float(np.min(times_array))
        self.max_step_time_ms = float(np.max(times_array))
        self.p95_step_time_ms = float(np.percentile(times_array, 95))
        self.p99_step_time_ms = float(np.percentile(times_array, 99))
        self.steps_per_second = self.n_steps / self.total_time_s if self.total_time_s > 0 else 0.0

        # Energy statistics
        if self.energies:
            energies_array = np.array(self.energies)
            self.energy_std = float(np.std(energies_array))

            # Energy drift for NVE (should be near zero)
            if self.protocol == MDProtocol.NVE:
                self.energy_drift = (energies_array[-1] - energies_array[0]) / abs(energies_array[0])

    @property
    def memory_delta_gb(self) -> float:
        """Memory change during benchmark."""
        return self.memory_after_gb - self.memory_before_gb

    @property
    def memory_leak_detected(self) -> bool:
        """Check if potential memory leak detected (>10 MB growth)."""
        return self.memory_delta_gb > 0.01  # 10 MB threshold

    def summary(self) -> str:
        """Return formatted summary string."""
        lines = [
            f"\n{'=' * 80}",
            f"MD Trajectory Benchmark Results: {self.name}",
            f"{'=' * 80}",
            f"System: {self.system_type} ({self.n_atoms} atoms)",
            f"Protocol: {self.protocol}",
            f"Device: {self.device}",
            f"Steps: {self.n_steps}",
            f"",
            f"Trajectory Performance:",
            f"  Total time:     {self.total_time_s:8.2f} s",
            f"  Steps/second:   {self.steps_per_second:8.1f}",
            f"",
            f"Per-Step Latency (ms):",
            f"  Mean:     {self.mean_step_time_ms:8.4f} ms ± {self.std_step_time_ms:.4f}",
            f"  Median:   {self.median_step_time_ms:8.4f} ms",
            f"  Min:      {self.min_step_time_ms:8.4f} ms",
            f"  Max:      {self.max_step_time_ms:8.4f} ms",
            f"  P95:      {self.p95_step_time_ms:8.4f} ms",
            f"  P99:      {self.p99_step_time_ms:8.4f} ms",
            f"",
            f"Memory Usage (GB):",
            f"  Before:    {self.memory_before_gb:.4f}",
            f"  After:     {self.memory_after_gb:.4f}",
            f"  Delta:     {self.memory_delta_gb:+.4f}",
            f"  Peak:      {self.peak_memory_gb:.4f}",
        ]

        # Add energy statistics if available
        if self.energies:
            lines.extend([
                f"",
                f"Energy Statistics:",
                f"  Std:       {self.energy_std:.6f} eV",
            ])

            if self.protocol == MDProtocol.NVE and self.energy_drift is not None:
                lines.append(f"  Drift:     {self.energy_drift:.6e} (NVE - should be ~0)")

                # Warn if energy drift is large
                if abs(self.energy_drift) > 1e-4:
                    lines.append(f"  WARNING: Large energy drift detected!")

        if self.memory_leak_detected:
            lines.extend([
                f"",
                f"WARNING: Potential memory leak detected!",
                f"  Memory grew by {self.memory_delta_gb * 1024:.2f} MB over {self.n_steps} steps",
            ])

        lines.append(f"{'=' * 80}\n")
        return '\n'.join(lines)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        # Use dataclass asdict but exclude raw lists to save space
        data = {
            'metadata': {
                'name': self.name,
                'protocol': self.protocol,
                'n_steps': self.n_steps,
                'n_atoms': self.n_atoms,
                'system_type': self.system_type,
                'device': self.device,
            },
            'performance': {
                'total_time_s': self.total_time_s,
                'steps_per_second': self.steps_per_second,
                'mean_step_time_ms': self.mean_step_time_ms,
                'std_step_time_ms': self.std_step_time_ms,
                'median_step_time_ms': self.median_step_time_ms,
                'min_step_time_ms': self.min_step_time_ms,
                'max_step_time_ms': self.max_step_time_ms,
                'p95_step_time_ms': self.p95_step_time_ms,
                'p99_step_time_ms': self.p99_step_time_ms,
            },
            'memory': {
                'before_gb': self.memory_before_gb,
                'after_gb': self.memory_after_gb,
                'delta_gb': self.memory_delta_gb,
                'peak_gb': self.peak_memory_gb,
                'leak_detected': self.memory_leak_detected,
            },
        }

        if self.energies:
            data['energy'] = {
                'std': self.energy_std,
            }
            if self.energy_drift is not None:
                data['energy']['drift'] = self.energy_drift

        return data

    def save(self, filepath: Union[str, Path]):
        """Save results to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'BenchmarkResults':
        """Load results from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Reconstruct from simplified dict (without raw arrays)
        # Note: This will not restore raw step_times_ms and energies
        return cls(
            name=data['metadata']['name'],
            protocol=data['metadata']['protocol'],
            n_steps=data['metadata']['n_steps'],
            n_atoms=data['metadata']['n_atoms'],
            system_type=data['metadata']['system_type'],
            device=data['metadata']['device'],
            step_times_ms=[data['performance']['mean_step_time_ms']] * data['metadata']['n_steps'],  # Placeholder
            memory_before_gb=data['memory']['before_gb'],
            memory_after_gb=data['memory']['after_gb'],
            peak_memory_gb=data['memory']['peak_gb'],
        )


def create_benchmark_system(
    system_type: str,
    n_atoms: int,
    temperature_K: float = 300.0,
) -> Atoms:
    """
    Create a benchmark system for MD testing.

    Args:
        system_type: Type of system ('silicon', 'water', 'cu', 'al', etc.)
        n_atoms: Target number of atoms (will be approximate for crystals)
        temperature_K: Initial temperature in Kelvin

    Returns:
        ASE Atoms object ready for MD simulation

    Example:
        >>> atoms = create_benchmark_system('silicon', 64, temperature_K=300)
        >>> print(len(atoms), atoms.get_pbc())
        64 [True True True]
    """
    # Molecule systems
    if system_type.lower() == "water":
        # Create water box
        mol = molecule("H2O")
        # Simple approach: replicate and offset
        n_molecules = max(1, n_atoms // 3)
        cell_size = (n_molecules ** (1/3)) * 3.0  # Rough spacing

        atoms = Atoms(
            symbols='H' * (n_molecules * 2) + 'O' * n_molecules,
            positions=np.random.rand(n_molecules * 3, 3) * cell_size,
            cell=[cell_size, cell_size, cell_size],
            pbc=True
        )

    # Crystal systems
    elif system_type.lower() in ["silicon", "si"]:
        si = bulk("Si", "diamond", a=5.43, cubic=True)
        # Determine supercell size to get close to n_atoms
        n_per_cell = len(si)
        repeat = max(1, int((n_atoms / n_per_cell) ** (1/3)))
        atoms = si.repeat((repeat, repeat, repeat))

    elif system_type.lower() in ["copper", "cu"]:
        cu = bulk("Cu", "fcc", a=3.6, cubic=True)
        n_per_cell = len(cu)
        repeat = max(1, int((n_atoms / n_per_cell) ** (1/3)))
        atoms = cu.repeat((repeat, repeat, repeat))

    elif system_type.lower() in ["aluminum", "al"]:
        al = bulk("Al", "fcc", a=4.05, cubic=True)
        n_per_cell = len(al)
        repeat = max(1, int((n_atoms / n_per_cell) ** (1/3)))
        atoms = al.repeat((repeat, repeat, repeat))

    else:
        raise ValueError(
            f"Unknown system type: {system_type}. "
            f"Available: silicon, water, copper, aluminum"
        )

    # Initialize velocities for given temperature
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)

    return atoms


class MDTrajectoryBenchmark:
    """
    Benchmark framework for MD trajectory performance measurement.

    This class runs MD simulations and collects detailed performance metrics:
    - Per-step latency distribution
    - Memory usage over time
    - Energy conservation (for NVE)
    - Total trajectory time

    Example:
        >>> from mlff_distiller.models.teacher_wrappers import OrbCalculator
        >>> calc = OrbCalculator(model_name="orb-v2", device="cuda")
        >>>
        >>> benchmark = MDTrajectoryBenchmark(
        ...     calculator=calc,
        ...     system_type="silicon",
        ...     n_atoms=64,
        ...     protocol=MDProtocol.NVE
        ... )
        >>>
        >>> results = benchmark.run(n_steps=1000)
        >>> print(results.summary())
        >>> results.save("baseline_si64_nve.json")
    """

    def __init__(
        self,
        calculator: Calculator,
        system_type: str,
        n_atoms: int,
        protocol: MDProtocol = MDProtocol.NVE,
        timestep_fs: float = 1.0,
        temperature_K: float = 300.0,
        name: Optional[str] = None,
    ):
        """
        Initialize MD trajectory benchmark.

        Args:
            calculator: ASE Calculator to benchmark
            system_type: Type of system ('silicon', 'water', 'cu', etc.)
            n_atoms: Target number of atoms
            protocol: MD protocol (NVE, NVT, or NPT)
            timestep_fs: MD timestep in femtoseconds
            temperature_K: Temperature in Kelvin
            name: Optional name for benchmark (auto-generated if None)
        """
        self.calculator = calculator
        self.system_type = system_type
        self.n_atoms = n_atoms
        self.protocol = protocol
        self.timestep_fs = timestep_fs
        self.temperature_K = temperature_K

        # Auto-generate name if not provided
        if name is None:
            calc_name = calculator.__class__.__name__
            self.name = f"{calc_name}_{system_type}_{n_atoms}atoms_{protocol.value}"
        else:
            self.name = name

        # Device detection
        self.device = self._detect_device()

        # Setup system
        self.atoms = create_benchmark_system(
            system_type=system_type,
            n_atoms=n_atoms,
            temperature_K=temperature_K,
        )
        self.atoms.calc = self.calculator

        # Storage for measurements
        self.step_times_ms = []
        self.energies = []
        self.memory_samples_gb = []

    def _detect_device(self) -> str:
        """Detect device from calculator."""
        if hasattr(self.calculator, 'device'):
            return str(self.calculator.device)
        return "unknown"

    def _get_memory_usage_gb(self) -> float:
        """Get current GPU memory usage in GB."""
        if torch.cuda.is_available() and 'cuda' in self.device:
            return torch.cuda.memory_allocated() / 1e9
        return 0.0

    def _setup_md_integrator(self):
        """Setup MD integrator based on protocol."""
        timestep = self.timestep_fs * units.fs

        if self.protocol == MDProtocol.NVE:
            return VelocityVerlet(self.atoms, timestep=timestep)

        elif self.protocol == MDProtocol.NVT:
            return Langevin(
                self.atoms,
                timestep=timestep,
                temperature_K=self.temperature_K,
                friction=0.01,  # Standard friction coefficient
            )

        elif self.protocol == MDProtocol.NPT:
            # NPT requires different setup
            return NPT(
                self.atoms,
                timestep=timestep,
                temperature_K=self.temperature_K,
                externalstress=0.0,  # Target pressure (bar)
                ttime=25 * units.fs,  # Temperature coupling time
                pfactor=75 * units.fs ** 2,  # Pressure coupling time
            )

        else:
            raise ValueError(f"Unknown protocol: {self.protocol}")

    def run(
        self,
        n_steps: int = 1000,
        warmup_steps: int = 10,
        memory_sample_interval: int = 100,
        energy_sample_interval: int = 1,
    ) -> BenchmarkResults:
        """
        Run MD trajectory benchmark.

        Args:
            n_steps: Number of MD steps to run
            warmup_steps: Number of warmup steps before timing
            memory_sample_interval: Interval for memory sampling
            energy_sample_interval: Interval for energy sampling

        Returns:
            BenchmarkResults object with comprehensive statistics
        """
        # Warmup
        if warmup_steps > 0:
            dyn = self._setup_md_integrator()
            dyn.run(warmup_steps)

        # Synchronize and record initial memory
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        memory_before = self._get_memory_usage_gb()

        # Reset storage
        self.step_times_ms = []
        self.energies = []
        self.memory_samples_gb = []

        # Setup fresh integrator for benchmark
        dyn = self._setup_md_integrator()

        # Run trajectory with per-step timing
        for i in range(n_steps):
            # Time single step
            start_time = time.perf_counter()
            dyn.run(1)
            step_time = (time.perf_counter() - start_time) * 1000  # Convert to ms

            self.step_times_ms.append(step_time)

            # Sample energy
            if i % energy_sample_interval == 0:
                try:
                    energy = self.atoms.get_potential_energy()
                    self.energies.append(energy)
                except Exception as e:
                    warnings.warn(f"Failed to get energy at step {i}: {e}")

            # Sample memory
            if i % memory_sample_interval == 0:
                mem = self._get_memory_usage_gb()
                self.memory_samples_gb.append(mem)

        # Synchronize and record final memory
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        memory_after = self._get_memory_usage_gb()
        peak_memory = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else memory_after

        # Create results
        results = BenchmarkResults(
            name=self.name,
            protocol=self.protocol.value,
            n_steps=n_steps,
            n_atoms=len(self.atoms),
            system_type=self.system_type,
            device=self.device,
            step_times_ms=self.step_times_ms,
            memory_before_gb=memory_before,
            memory_after_gb=memory_after,
            peak_memory_gb=peak_memory,
            memory_samples_gb=self.memory_samples_gb,
            energies=self.energies,
        )

        # Check for memory leaks
        if results.memory_leak_detected:
            warnings.warn(
                f"Potential memory leak detected in {self.name}: "
                f"{results.memory_delta_gb * 1024:.2f} MB increase over {n_steps} steps",
                RuntimeWarning,
            )

        # Check energy conservation for NVE
        if self.protocol == MDProtocol.NVE and results.energy_drift is not None:
            if abs(results.energy_drift) > 1e-4:
                warnings.warn(
                    f"Large energy drift in NVE simulation: {results.energy_drift:.6e}. "
                    f"This may indicate numerical instability.",
                    RuntimeWarning,
                )

        return results


def compare_calculators(
    calculators: Dict[str, Calculator],
    system_type: str = "silicon",
    n_atoms: int = 64,
    protocol: MDProtocol = MDProtocol.NVE,
    n_steps: int = 1000,
    **kwargs
) -> Dict[str, BenchmarkResults]:
    """
    Compare multiple calculators on same MD trajectory.

    Args:
        calculators: Dict mapping names to Calculator instances
        system_type: Type of system to benchmark
        n_atoms: Number of atoms
        protocol: MD protocol
        n_steps: Number of MD steps
        **kwargs: Additional arguments for MDTrajectoryBenchmark

    Returns:
        Dictionary mapping calculator names to BenchmarkResults

    Example:
        >>> from mlff_distiller.models.teacher_wrappers import OrbCalculator
        >>>
        >>> calculators = {
        ...     "Orb-v2": OrbCalculator(model_name="orb-v2", device="cuda"),
        ...     "Orb-v3": OrbCalculator(model_name="orb-v3", device="cuda"),
        ... }
        >>>
        >>> results = compare_calculators(calculators, n_atoms=128, n_steps=1000)
        >>>
        >>> for name, result in results.items():
        ...     print(f"{name}: {result.mean_step_time_ms:.2f} ms/step")
    """
    results = {}

    for name, calculator in calculators.items():
        print(f"\nBenchmarking: {name}")
        print(f"{'-' * 40}")

        benchmark = MDTrajectoryBenchmark(
            calculator=calculator,
            system_type=system_type,
            n_atoms=n_atoms,
            protocol=protocol,
            name=name,
            **kwargs
        )

        result = benchmark.run(n_steps=n_steps)
        results[name] = result

        print(result.summary())

    return results
