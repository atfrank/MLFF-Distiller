"""CUDA utilities for MLFF Distiller."""

from .benchmark_utils import (
    BenchmarkResult,
    CUDATimer,
    ProfilerContext,
    benchmark_function,
    benchmark_md_trajectory,
    compare_implementations,
    print_comparison_table,
    profile_with_nsys,
    timer_context,
)
from .device_utils import (
    GPUInfo,
    check_memory_leak,
    cuda_memory_tracker,
    empty_cache,
    get_cuda_stream,
    get_device,
    get_gpu_info,
    get_gpu_memory_info,
    get_peak_memory_allocated,
    list_available_devices,
    print_device_summary,
    print_gpu_memory_summary,
    reset_peak_memory_stats,
    synchronize_device,
    torch_device_context,
    warmup_cuda,
)
from .md_profiler import (
    MDProfileResult,
    MDProfiler,
    identify_hotspots,
    profile_md_trajectory as profile_md_trajectory_detailed,
)

__all__ = [
    # Device utilities
    "GPUInfo",
    "get_device",
    "get_gpu_info",
    "get_gpu_memory_info",
    "print_device_summary",
    "print_gpu_memory_summary",
    "empty_cache",
    "reset_peak_memory_stats",
    "get_peak_memory_allocated",
    "cuda_memory_tracker",
    "torch_device_context",
    "check_memory_leak",
    "warmup_cuda",
    "get_cuda_stream",
    "synchronize_device",
    "list_available_devices",
    # Benchmark utilities
    "BenchmarkResult",
    "CUDATimer",
    "benchmark_function",
    "benchmark_md_trajectory",
    "compare_implementations",
    "print_comparison_table",
    "ProfilerContext",
    "profile_with_nsys",
    "timer_context",
    # MD profiler utilities
    "MDProfiler",
    "MDProfileResult",
    "profile_md_trajectory_detailed",
    "identify_hotspots",
]
