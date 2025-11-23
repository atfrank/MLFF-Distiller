"""
CUDA Device Management Utilities

Provides comprehensive device detection, memory management, and GPU utilities
optimized for molecular dynamics simulations with repeated inference.

Key Features:
- Smart device selection with fallback
- Memory monitoring and leak detection
- Device context managers
- CUDA stream management
- Performance monitoring utilities
"""

import gc
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.cuda


@dataclass
class GPUInfo:
    """GPU device information container."""

    index: int
    name: str
    compute_capability: Tuple[int, int]
    total_memory_gb: float
    multi_processor_count: int
    max_threads_per_multiprocessor: int
    warp_size: int
    regs_per_multiprocessor: int
    l2_cache_size: int

    @property
    def compute_capability_str(self) -> str:
        """Return compute capability as string (e.g., '8.6')."""
        return f"{self.compute_capability[0]}.{self.compute_capability[1]}"

    def __str__(self) -> str:
        return (
            f"GPU {self.index}: {self.name}\n"
            f"  Compute Capability: {self.compute_capability_str}\n"
            f"  Memory: {self.total_memory_gb:.2f} GB\n"
            f"  Multiprocessors: {self.multi_processor_count}\n"
            f"  Max Threads/MP: {self.max_threads_per_multiprocessor}\n"
            f"  Warp Size: {self.warp_size}\n"
            f"  L2 Cache: {self.l2_cache_size / (1024**2):.2f} MB"
        )


def get_device(
    device: Optional[Union[str, torch.device]] = None,
    prefer_cuda: bool = True,
    gpu_index: Optional[int] = None,
) -> torch.device:
    """
    Get PyTorch device with intelligent fallback.

    Args:
        device: Explicit device specification ('cuda', 'cpu', torch.device)
        prefer_cuda: If True, prefer CUDA if available
        gpu_index: Specific GPU index to use (0, 1, etc.)

    Returns:
        torch.device: Selected device

    Examples:
        >>> device = get_device()  # Auto-select best device
        >>> device = get_device(gpu_index=0)  # Force GPU 0
        >>> device = get_device(device='cpu')  # Force CPU
    """
    # Explicit device provided
    if device is not None:
        if isinstance(device, str):
            return torch.device(device)
        return device

    # CUDA available and preferred
    if prefer_cuda and torch.cuda.is_available():
        if gpu_index is not None:
            if gpu_index >= torch.cuda.device_count():
                warnings.warn(
                    f"GPU {gpu_index} not available. Using GPU 0 instead.",
                    RuntimeWarning,
                )
                gpu_index = 0
            return torch.device(f"cuda:{gpu_index}")
        return torch.device("cuda")

    # Fallback to CPU
    if prefer_cuda and not torch.cuda.is_available():
        warnings.warn(
            "CUDA requested but not available. Falling back to CPU.",
            RuntimeWarning,
        )

    return torch.device("cpu")


def get_gpu_info(device_index: int = 0) -> GPUInfo:
    """
    Get comprehensive GPU information.

    Args:
        device_index: GPU index (default: 0)

    Returns:
        GPUInfo: GPU specifications

    Raises:
        RuntimeError: If CUDA not available or invalid device index
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    if device_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"GPU index {device_index} out of range. "
            f"Available GPUs: {torch.cuda.device_count()}"
        )

    props = torch.cuda.get_device_properties(device_index)

    return GPUInfo(
        index=device_index,
        name=props.name,
        compute_capability=torch.cuda.get_device_capability(device_index),
        total_memory_gb=props.total_memory / (1024**3),
        multi_processor_count=props.multi_processor_count,
        max_threads_per_multiprocessor=props.max_threads_per_multi_processor,
        warp_size=props.warp_size,
        regs_per_multiprocessor=props.regs_per_multiprocessor,
        l2_cache_size=props.L2_cache_size,
    )


def get_gpu_memory_info(device: Optional[Union[str, torch.device]] = None) -> Dict[str, float]:
    """
    Get current GPU memory usage information.

    Args:
        device: Device to query (default: current CUDA device)

    Returns:
        Dictionary with memory info in GB:
            - allocated: Memory allocated by PyTorch
            - reserved: Memory reserved by PyTorch caching allocator
            - free: Free memory in cache
            - total: Total GPU memory
            - utilization: Percentage of total memory allocated

    Examples:
        >>> mem_info = get_gpu_memory_info()
        >>> print(f"GPU Memory: {mem_info['allocated']:.2f} GB / {mem_info['total']:.2f} GB")
    """
    if not torch.cuda.is_available():
        return {
            'allocated': 0.0,
            'reserved': 0.0,
            'free': 0.0,
            'total': 0.0,
            'utilization': 0.0,
        }

    device = torch.device(device) if device is not None else torch.cuda.current_device()

    if isinstance(device, torch.device):
        device_idx = device.index if device.index is not None else 0
    else:
        device_idx = 0

    # Memory in bytes
    allocated = torch.cuda.memory_allocated(device_idx)
    reserved = torch.cuda.memory_reserved(device_idx)
    total = torch.cuda.get_device_properties(device_idx).total_memory

    # Convert to GB
    gb = 1024**3
    allocated_gb = allocated / gb
    reserved_gb = reserved / gb
    total_gb = total / gb
    free_gb = reserved_gb - allocated_gb

    return {
        'allocated': allocated_gb,
        'reserved': reserved_gb,
        'free': free_gb,
        'total': total_gb,
        'utilization': (allocated / total * 100) if total > 0 else 0.0,
    }


def print_gpu_memory_summary(device: Optional[Union[str, torch.device]] = None) -> None:
    """
    Print formatted GPU memory summary.

    Args:
        device: Device to query (default: current CUDA device)
    """
    mem_info = get_gpu_memory_info(device)

    print("\n" + "=" * 60)
    print("GPU Memory Summary")
    print("=" * 60)
    print(f"  Allocated:   {mem_info['allocated']:6.2f} GB")
    print(f"  Reserved:    {mem_info['reserved']:6.2f} GB")
    print(f"  Free (cache):{mem_info['free']:6.2f} GB")
    print(f"  Total:       {mem_info['total']:6.2f} GB")
    print(f"  Utilization: {mem_info['utilization']:6.2f} %")
    print("=" * 60 + "\n")


def empty_cache() -> None:
    """
    Empty CUDA cache and run garbage collection.

    This is useful for MD simulations to prevent memory accumulation
    over millions of inference calls.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def reset_peak_memory_stats(device: Optional[Union[str, torch.device]] = None) -> None:
    """
    Reset peak memory statistics for monitoring.

    Useful for benchmarking individual operations or tracking
    memory usage over segments of MD simulations.

    Args:
        device: Device to reset stats for (default: current CUDA device)
    """
    if torch.cuda.is_available():
        device_idx = None
        if device is not None:
            device = torch.device(device)
            device_idx = device.index if device.index is not None else 0

        torch.cuda.reset_peak_memory_stats(device_idx)


def get_peak_memory_allocated(device: Optional[Union[str, torch.device]] = None) -> float:
    """
    Get peak memory allocated since last reset.

    Args:
        device: Device to query (default: current CUDA device)

    Returns:
        Peak memory allocated in GB
    """
    if not torch.cuda.is_available():
        return 0.0

    device_idx = None
    if device is not None:
        device = torch.device(device)
        device_idx = device.index if device.index is not None else 0

    peak_bytes = torch.cuda.max_memory_allocated(device_idx)
    return peak_bytes / (1024**3)


@contextmanager
def cuda_memory_tracker(device: Optional[Union[str, torch.device]] = None, name: str = ""):
    """
    Context manager for tracking CUDA memory usage.

    Args:
        device: Device to track (default: current CUDA device)
        name: Optional name for the tracked section

    Example:
        >>> with cuda_memory_tracker(name="Model Forward"):
        ...     output = model(input)
        Memory used (Model Forward): 2.34 GB
    """
    if not torch.cuda.is_available():
        yield
        return

    device_idx = None
    if device is not None:
        device = torch.device(device)
        device_idx = device.index if device.index is not None else 0

    # Record initial memory
    torch.cuda.synchronize(device_idx)
    initial_allocated = torch.cuda.memory_allocated(device_idx)

    try:
        yield
    finally:
        # Record final memory
        torch.cuda.synchronize(device_idx)
        final_allocated = torch.cuda.memory_allocated(device_idx)
        delta = (final_allocated - initial_allocated) / (1024**3)

        name_str = f" ({name})" if name else ""
        print(f"Memory delta{name_str}: {delta:+.4f} GB")


@contextmanager
def torch_device_context(device: Union[str, torch.device]):
    """
    Context manager for temporarily setting default CUDA device.

    Args:
        device: Device to set as default

    Example:
        >>> with torch_device_context('cuda:1'):
        ...     tensor = torch.randn(100, 100)  # Created on cuda:1
    """
    if not torch.cuda.is_available():
        yield
        return

    device = torch.device(device)

    if device.type != 'cuda':
        yield
        return

    device_idx = device.index if device.index is not None else 0
    old_device = torch.cuda.current_device()

    try:
        torch.cuda.set_device(device_idx)
        yield
    finally:
        torch.cuda.set_device(old_device)


def check_memory_leak(
    initial_allocated: float,
    tolerance_mb: float = 10.0,
    device: Optional[Union[str, torch.device]] = None,
) -> bool:
    """
    Check for memory leaks by comparing current allocation to initial.

    Critical for MD simulations where model is called millions of times.

    Args:
        initial_allocated: Initial memory allocation in GB
        tolerance_mb: Acceptable memory increase in MB (default: 10 MB)
        device: Device to check (default: current CUDA device)

    Returns:
        True if no leak detected, False if leak suspected

    Example:
        >>> mem_info = get_gpu_memory_info()
        >>> initial_mem = mem_info['allocated']
        >>> # ... run many inference calls ...
        >>> no_leak = check_memory_leak(initial_mem)
    """
    if not torch.cuda.is_available():
        return True

    current_mem = get_gpu_memory_info(device)['allocated']
    delta_mb = (current_mem - initial_allocated) * 1024

    if delta_mb > tolerance_mb:
        warnings.warn(
            f"Potential memory leak detected: {delta_mb:.2f} MB increase",
            RuntimeWarning,
        )
        return False

    return True


def warmup_cuda(
    device: Optional[Union[str, torch.device]] = None,
    size: int = 1000,
    n_iterations: int = 10,
) -> None:
    """
    Warm up CUDA for accurate benchmarking.

    Runs dummy operations to initialize CUDA context and kernel compilation.
    Critical for accurate MD performance measurements.

    Args:
        device: Device to warm up (default: current CUDA device)
        size: Size of dummy tensors
        n_iterations: Number of warmup iterations

    Example:
        >>> warmup_cuda()  # Before benchmarking
        >>> # Now run actual benchmarks
    """
    if not torch.cuda.is_available():
        return

    device = get_device(device)

    # Create dummy tensors and run operations
    for _ in range(n_iterations):
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        c = torch.matmul(a, b)
        _ = c.sum()

    # Synchronize to ensure completion
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

    # Clean up
    empty_cache()


def get_cuda_stream(device: Optional[Union[str, torch.device]] = None) -> Optional[torch.cuda.Stream]:
    """
    Create a new CUDA stream for concurrent operations.

    Args:
        device: Device for the stream (default: current CUDA device)

    Returns:
        torch.cuda.Stream or None if CUDA not available

    Example:
        >>> stream = get_cuda_stream()
        >>> with torch.cuda.stream(stream):
        ...     # Operations on this stream
    """
    if not torch.cuda.is_available():
        return None

    device = get_device(device)
    if device.type != 'cuda':
        return None

    return torch.cuda.Stream(device=device)


def synchronize_device(device: Optional[Union[str, torch.device]] = None) -> None:
    """
    Synchronize CUDA device.

    Ensures all operations on the device have completed.
    Critical for accurate timing measurements in MD benchmarks.

    Args:
        device: Device to synchronize (default: current CUDA device)
    """
    if not torch.cuda.is_available():
        return

    device = get_device(device) if device is None else torch.device(device)

    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def list_available_devices() -> List[str]:
    """
    List all available PyTorch devices.

    Returns:
        List of device strings ['cpu', 'cuda:0', 'cuda:1', ...]

    Example:
        >>> devices = list_available_devices()
        >>> print(f"Available devices: {devices}")
    """
    devices = ['cpu']

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(f'cuda:{i}')

    return devices


def print_device_summary() -> None:
    """Print comprehensive summary of available devices."""
    print("\n" + "=" * 80)
    print("PyTorch Device Summary")
    print("=" * 80)

    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print()

        for i in range(torch.cuda.device_count()):
            gpu_info = get_gpu_info(i)
            print(gpu_info)
            print()

            # Memory info
            mem_info = get_gpu_memory_info(f'cuda:{i}')
            print(f"  Current Memory Usage:")
            print(f"    Allocated: {mem_info['allocated']:.2f} GB")
            print(f"    Reserved:  {mem_info['reserved']:.2f} GB")
            print(f"    Total:     {mem_info['total']:.2f} GB")
            print()
    else:
        print("\nNo CUDA devices available. Using CPU only.")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Demonstration
    print_device_summary()

    if torch.cuda.is_available():
        print("\nTesting memory tracking:")
        with cuda_memory_tracker(name="Dummy allocation"):
            dummy = torch.randn(10000, 10000, device='cuda')

        print("\nWarming up CUDA...")
        warmup_cuda()
        print("Warmup complete.")

        print_gpu_memory_summary()
