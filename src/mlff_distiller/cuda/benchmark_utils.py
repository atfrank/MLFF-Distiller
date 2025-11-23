"""
CUDA Benchmarking Utilities for Molecular Dynamics

Specialized benchmarking tools focused on MD simulation performance:
- Per-call latency measurement (critical for MD where model called millions of times)
- Memory stability tracking over long runs
- Statistical analysis of timing distributions
- GPU utilization monitoring
- Profiling integration (torch.profiler, nsys)

Key Philosophy:
MD simulations care about LATENCY, not throughput. We measure:
1. Individual inference time (µs/call)
2. Memory stability over millions of calls
3. Variance and outliers in timing
4. Sustained performance (not just peak)
"""

import gc
import time
import warnings
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .device_utils import (
    empty_cache,
    get_gpu_memory_info,
    get_peak_memory_allocated,
    reset_peak_memory_stats,
    synchronize_device,
    warmup_cuda,
)


@dataclass
class BenchmarkResult:
    """Container for benchmark results with statistical analysis."""

    name: str
    n_calls: int
    times_ms: List[float]
    memory_before_gb: float
    memory_after_gb: float
    peak_memory_gb: float
    device: str

    # Computed statistics
    mean_ms: float = field(init=False)
    std_ms: float = field(init=False)
    median_ms: float = field(init=False)
    min_ms: float = field(init=False)
    max_ms: float = field(init=False)
    p95_ms: float = field(init=False)
    p99_ms: float = field(init=False)

    def __post_init__(self):
        """Compute statistics from timing data."""
        times_array = np.array(self.times_ms)

        self.mean_ms = float(np.mean(times_array))
        self.std_ms = float(np.std(times_array))
        self.median_ms = float(np.median(times_array))
        self.min_ms = float(np.min(times_array))
        self.max_ms = float(np.max(times_array))
        self.p95_ms = float(np.percentile(times_array, 95))
        self.p99_ms = float(np.percentile(times_array, 99))

    @property
    def memory_delta_gb(self) -> float:
        """Memory change during benchmark."""
        return self.memory_after_gb - self.memory_before_gb

    @property
    def total_time_s(self) -> float:
        """Total time for all calls in seconds."""
        return sum(self.times_ms) / 1000.0

    @property
    def calls_per_second(self) -> float:
        """Average throughput in calls/second."""
        return self.n_calls / self.total_time_s if self.total_time_s > 0 else 0.0

    def summary(self) -> str:
        """Return formatted summary string."""
        lines = [
            f"\n{'=' * 80}",
            f"Benchmark Results: {self.name}",
            f"{'=' * 80}",
            f"Device: {self.device}",
            f"Number of calls: {self.n_calls}",
            f"",
            f"Latency Statistics (ms per call):",
            f"  Mean:     {self.mean_ms:8.4f} ms ± {self.std_ms:.4f}",
            f"  Median:   {self.median_ms:8.4f} ms",
            f"  Min:      {self.min_ms:8.4f} ms",
            f"  Max:      {self.max_ms:8.4f} ms",
            f"  P95:      {self.p95_ms:8.4f} ms",
            f"  P99:      {self.p99_ms:8.4f} ms",
            f"",
            f"Throughput:",
            f"  Calls/sec: {self.calls_per_second:,.1f}",
            f"  Total time: {self.total_time_s:.2f} s",
            f"",
            f"Memory Usage (GB):",
            f"  Before:    {self.memory_before_gb:.4f}",
            f"  After:     {self.memory_after_gb:.4f}",
            f"  Delta:     {self.memory_delta_gb:+.4f}",
            f"  Peak:      {self.peak_memory_gb:.4f}",
            f"{'=' * 80}\n",
        ]
        return '\n'.join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'n_calls': self.n_calls,
            'device': self.device,
            'latency_ms': {
                'mean': self.mean_ms,
                'std': self.std_ms,
                'median': self.median_ms,
                'min': self.min_ms,
                'max': self.max_ms,
                'p95': self.p95_ms,
                'p99': self.p99_ms,
            },
            'throughput': {
                'calls_per_second': self.calls_per_second,
                'total_time_s': self.total_time_s,
            },
            'memory_gb': {
                'before': self.memory_before_gb,
                'after': self.memory_after_gb,
                'delta': self.memory_delta_gb,
                'peak': self.peak_memory_gb,
            },
        }


class CUDATimer:
    """
    High-precision CUDA timer using CUDA events.

    Provides microsecond-accurate timing for GPU operations by using
    CUDA events instead of host-side timers. Critical for MD benchmarking.
    """

    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize CUDA timer.

        Args:
            device: CUDA device (default: current device)
        """
        self.device = device
        self.use_cuda = torch.cuda.is_available() and (
            device is None or device.type == 'cuda'
        )

        if self.use_cuda:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        else:
            self.start_time = None

    def start(self) -> None:
        """Start timer."""
        if self.use_cuda:
            self.start_event.record()
        else:
            self.start_time = time.perf_counter()

    def stop(self) -> float:
        """
        Stop timer and return elapsed time.

        Returns:
            Elapsed time in milliseconds
        """
        if self.use_cuda:
            self.end_event.record()
            torch.cuda.synchronize(self.device)
            return self.start_event.elapsed_time(self.end_event)
        else:
            return (time.perf_counter() - self.start_time) * 1000.0


@contextmanager
def timer_context(name: str = "", verbose: bool = True) -> None:
    """
    Context manager for timing code blocks.

    Args:
        name: Name for the timed section
        verbose: Whether to print results

    Example:
        >>> with timer_context("Model inference"):
        ...     output = model(input)
        Model inference: 2.34 ms
    """
    timer = CUDATimer()
    timer.start()

    try:
        yield timer
    finally:
        elapsed = timer.stop()
        if verbose:
            name_str = f"{name}: " if name else ""
            print(f"{name_str}{elapsed:.4f} ms")


def benchmark_function(
    func: Callable,
    *args,
    n_warmup: int = 10,
    n_calls: int = 100,
    device: Optional[torch.device] = None,
    name: str = "Function",
    **kwargs,
) -> BenchmarkResult:
    """
    Benchmark a function with comprehensive statistics.

    Designed for MD use case: measures per-call latency and memory stability.

    Args:
        func: Function to benchmark
        *args: Positional arguments to func
        n_warmup: Number of warmup calls (default: 10)
        n_calls: Number of benchmark calls (default: 100)
        device: Device for memory tracking
        name: Name for the benchmark
        **kwargs: Keyword arguments to func

    Returns:
        BenchmarkResult with comprehensive statistics

    Example:
        >>> def model_forward(x):
        ...     return model(x)
        >>> result = benchmark_function(model_forward, input_tensor, n_calls=1000)
        >>> print(result.summary())
    """
    if device is None and torch.cuda.is_available():
        device = torch.device('cuda')
    elif device is None:
        device = torch.device('cpu')

    device_str = str(device)

    # Warmup
    for _ in range(n_warmup):
        _ = func(*args, **kwargs)

    if device.type == 'cuda':
        synchronize_device(device)
        empty_cache()

    # Record initial memory
    mem_before = get_gpu_memory_info(device)['allocated']
    reset_peak_memory_stats(device)

    # Benchmark
    timer = CUDATimer(device)
    times_ms = []

    for _ in range(n_calls):
        timer.start()
        _ = func(*args, **kwargs)
        elapsed = timer.stop()
        times_ms.append(elapsed)

    # Record final memory
    if device.type == 'cuda':
        synchronize_device(device)

    mem_after = get_gpu_memory_info(device)['allocated']
    peak_mem = get_peak_memory_allocated(device)

    return BenchmarkResult(
        name=name,
        n_calls=n_calls,
        times_ms=times_ms,
        memory_before_gb=mem_before,
        memory_after_gb=mem_after,
        peak_memory_gb=peak_mem,
        device=device_str,
    )


def benchmark_md_trajectory(
    model_func: Callable,
    inputs: List[Any],
    n_warmup: int = 10,
    device: Optional[torch.device] = None,
    name: str = "MD Trajectory",
    check_memory_leak: bool = True,
    leak_tolerance_mb: float = 10.0,
) -> BenchmarkResult:
    """
    Benchmark model on a trajectory of inputs (simulating MD simulation).

    This is the key benchmark for MD use cases: repeatedly calling the model
    with different inputs (different MD timesteps).

    Args:
        model_func: Function that takes single input and returns output
        inputs: List of inputs (e.g., atomic configurations)
        n_warmup: Number of warmup calls
        device: Device for computation
        name: Name for benchmark
        check_memory_leak: Whether to check for memory leaks
        leak_tolerance_mb: Memory leak tolerance in MB

    Returns:
        BenchmarkResult with trajectory statistics

    Example:
        >>> # Simulate 1000-step MD trajectory
        >>> atoms_list = [generate_atoms() for _ in range(1000)]
        >>> result = benchmark_md_trajectory(
        ...     lambda atoms: model(atoms),
        ...     atoms_list
        ... )
    """
    if device is None and torch.cuda.is_available():
        device = torch.device('cuda')
    elif device is None:
        device = torch.device('cpu')

    n_calls = len(inputs)

    # Warmup with first few inputs
    warmup_inputs = inputs[:min(n_warmup, len(inputs))]
    for inp in warmup_inputs:
        _ = model_func(inp)

    if device.type == 'cuda':
        synchronize_device(device)
        empty_cache()

    # Record initial memory
    mem_before = get_gpu_memory_info(device)['allocated']
    reset_peak_memory_stats(device)

    # Benchmark trajectory
    timer = CUDATimer(device)
    times_ms = []

    for inp in inputs:
        timer.start()
        _ = model_func(inp)
        elapsed = timer.stop()
        times_ms.append(elapsed)

    # Record final memory
    if device.type == 'cuda':
        synchronize_device(device)

    mem_after = get_gpu_memory_info(device)['allocated']
    peak_mem = get_peak_memory_allocated(device)

    # Check for memory leak
    if check_memory_leak:
        mem_delta_mb = (mem_after - mem_before) * 1024
        if mem_delta_mb > leak_tolerance_mb:
            warnings.warn(
                f"Potential memory leak in trajectory: {mem_delta_mb:.2f} MB increase "
                f"over {n_calls} calls. This will cause issues in long MD runs.",
                RuntimeWarning,
            )

    return BenchmarkResult(
        name=name,
        n_calls=n_calls,
        times_ms=times_ms,
        memory_before_gb=mem_before,
        memory_after_gb=mem_after,
        peak_memory_gb=peak_mem,
        device=str(device),
    )


def compare_implementations(
    implementations: Dict[str, Callable],
    *args,
    n_warmup: int = 10,
    n_calls: int = 100,
    device: Optional[torch.device] = None,
    **kwargs,
) -> Dict[str, BenchmarkResult]:
    """
    Compare multiple implementations (e.g., baseline vs optimized).

    Args:
        implementations: Dict mapping names to callables
        *args: Arguments to pass to all implementations
        n_warmup: Number of warmup calls
        n_calls: Number of benchmark calls
        device: Device for benchmarking
        **kwargs: Keyword arguments to pass to all implementations

    Returns:
        Dictionary mapping implementation names to BenchmarkResults

    Example:
        >>> implementations = {
        ...     'PyTorch baseline': model_forward,
        ...     'TensorRT optimized': tensorrt_forward,
        ...     'torch.compile': compiled_forward,
        ... }
        >>> results = compare_implementations(implementations, input_tensor, n_calls=1000)
        >>> for name, result in results.items():
        ...     print(f"{name}: {result.mean_ms:.4f} ms")
    """
    results = {}

    for name, func in implementations.items():
        print(f"\nBenchmarking: {name}")
        result = benchmark_function(
            func,
            *args,
            n_warmup=n_warmup,
            n_calls=n_calls,
            device=device,
            name=name,
            **kwargs,
        )
        results[name] = result
        print(f"  Mean latency: {result.mean_ms:.4f} ms")

    return results


def print_comparison_table(results: Dict[str, BenchmarkResult]) -> None:
    """
    Print formatted comparison table of benchmark results.

    Args:
        results: Dictionary of benchmark results from compare_implementations
    """
    if not results:
        print("No results to compare")
        return

    print("\n" + "=" * 100)
    print("Benchmark Comparison")
    print("=" * 100)

    # Header
    print(f"{'Implementation':<30} {'Mean (ms)':>12} {'Std (ms)':>12} {'P95 (ms)':>12} "
          f"{'Calls/s':>12} {'Memory (GB)':>12}")
    print("-" * 100)

    # Find baseline (first entry) for speedup calculation
    baseline_name = list(results.keys())[0]
    baseline_mean = results[baseline_name].mean_ms

    # Rows
    for name, result in results.items():
        speedup = baseline_mean / result.mean_ms if result.mean_ms > 0 else 0
        speedup_str = f" ({speedup:.2f}x)" if name != baseline_name else ""

        print(f"{name:<30} {result.mean_ms:12.4f} {result.std_ms:12.4f} "
              f"{result.p95_ms:12.4f} {result.calls_per_second:12.1f} "
              f"{result.peak_memory_gb:12.4f}{speedup_str}")

    print("=" * 100 + "\n")


class ProfilerContext:
    """
    Context manager for PyTorch profiler with sensible defaults for MD profiling.

    Integrates with PyTorch's built-in profiler and can export results
    for visualization in TensorBoard or Chrome tracing.
    """

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        profile_memory: bool = True,
        record_shapes: bool = True,
        with_stack: bool = False,
        name: str = "profile",
    ):
        """
        Initialize profiler context.

        Args:
            output_dir: Directory to save profiler results
            profile_memory: Whether to profile memory
            record_shapes: Whether to record tensor shapes
            with_stack: Whether to record stack traces (expensive)
            name: Name for the profiling session
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self.profile_memory = profile_memory
        self.record_shapes = record_shapes
        self.with_stack = with_stack
        self.name = name
        self.profiler = None

    def __enter__(self):
        """Start profiling."""
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        self.profiler = torch.profiler.profile(
            activities=activities,
            profile_memory=self.profile_memory,
            record_shapes=self.record_shapes,
            with_stack=self.with_stack,
        )

        self.profiler.__enter__()
        return self.profiler

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop profiling and save results."""
        self.profiler.__exit__(exc_type, exc_val, exc_tb)

        # Print summary
        print("\n" + "=" * 80)
        print(f"Profiler Results: {self.name}")
        print("=" * 80)
        print(self.profiler.key_averages().table(
            sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
            row_limit=20
        ))
        print("=" * 80 + "\n")

        # Export if output directory specified
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Chrome trace
            trace_file = self.output_dir / f"{self.name}_trace.json"
            self.profiler.export_chrome_trace(str(trace_file))
            print(f"Chrome trace saved to: {trace_file}")

            # TensorBoard (if available)
            try:
                tb_dir = self.output_dir / "tensorboard"
                tb_dir.mkdir(exist_ok=True)
                print(f"TensorBoard logs: {tb_dir}")
            except Exception as e:
                print(f"Could not save TensorBoard logs: {e}")


def profile_with_nsys(
    command: str,
    output_file: str = "profile.nsys-rep",
    cuda_events: bool = True,
) -> str:
    """
    Generate nsys profiling command.

    Returns the command to run with nsys for detailed GPU profiling.
    User must execute this command manually.

    Args:
        command: Python command to profile
        output_file: Output file for nsys report
        cuda_events: Whether to collect CUDA events

    Returns:
        String containing the full nsys command

    Example:
        >>> cmd = profile_with_nsys("python benchmark_model.py", "model_profile.nsys-rep")
        >>> print(f"Run this command:\n{cmd}")
    """
    nsys_cmd = [
        "nsys", "profile",
        "--trace=cuda,nvtx",
        f"--output={output_file}",
        "--force-overwrite=true",
    ]

    if cuda_events:
        nsys_cmd.append("--cuda-memory-usage=true")

    nsys_cmd.append(command)

    full_cmd = " ".join(nsys_cmd)

    print("\n" + "=" * 80)
    print("Nsight Systems Profiling")
    print("=" * 80)
    print("Run this command to profile with nsys:")
    print(f"\n{full_cmd}\n")
    print(f"View results with: nsys-ui {output_file}")
    print("=" * 80 + "\n")

    return full_cmd


if __name__ == "__main__":
    # Demonstration
    print("CUDA Benchmark Utilities Demo")

    if torch.cuda.is_available():
        # Warmup
        print("\nWarming up CUDA...")
        warmup_cuda()

        # Simple benchmark
        print("\nBenchmarking matrix multiplication:")

        def matmul_benchmark(size=1000):
            a = torch.randn(size, size, device='cuda')
            b = torch.randn(size, size, device='cuda')
            return torch.matmul(a, b)

        result = benchmark_function(matmul_benchmark, n_calls=100, name="MatMul 1000x1000")
        print(result.summary())

    else:
        print("\nCUDA not available - skipping GPU benchmarks")
