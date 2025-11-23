# CUDA Utilities for MLFF Distiller

CUDA device management and benchmarking utilities optimized for molecular dynamics (MD) simulations.

## Overview

This package provides GPU utilities specifically designed for MD performance profiling and optimization. Unlike typical ML benchmarking, MD simulations require:

- **Low latency** per inference call (called millions of times)
- **Memory stability** over long trajectories (no leaks)
- **Variance monitoring** (outliers affect MD stability)
- **Sustained performance** (not just peak throughput)

## Modules

### `device_utils.py`

Device detection, memory management, and GPU utilities.

**Key Functions**:
- `get_device()` - Smart device selection with fallback
- `get_gpu_info()` - Comprehensive GPU specifications
- `get_gpu_memory_info()` - Real-time memory monitoring
- `warmup_cuda()` - CUDA initialization for accurate benchmarking
- `check_memory_leak()` - Detect memory leaks over repeated calls

**Example**:
```python
from cuda.device_utils import get_device, warmup_cuda, get_gpu_memory_info

device = get_device()
warmup_cuda()

mem_info = get_gpu_memory_info()
print(f"GPU Memory: {mem_info['allocated']:.2f} GB")
```

### `benchmark_utils.py`

MD-focused benchmarking and profiling tools.

**Key Classes**:
- `BenchmarkResult` - Statistical analysis of timing data
- `CUDATimer` - High-precision GPU timing
- `ProfilerContext` - PyTorch profiler integration

**Key Functions**:
- `benchmark_function()` - General function benchmarking
- `benchmark_md_trajectory()` - MD trajectory profiling
- `compare_implementations()` - Multi-implementation comparison

**Example**:
```python
from cuda.benchmark_utils import benchmark_md_trajectory

result = benchmark_md_trajectory(
    model_func=lambda atoms: model(atoms),
    inputs=trajectory,
    check_memory_leak=True
)

print(f"Mean latency: {result.mean_ms:.4f} ms")
print(f"Memory delta: {result.memory_delta_gb * 1024:.2f} MB")
```

## Quick Start

### Basic Benchmarking

```python
from cuda.device_utils import get_device, warmup_cuda
from cuda.benchmark_utils import benchmark_function

device = get_device()
warmup_cuda()

result = benchmark_function(
    model.forward,
    input_tensor,
    n_calls=100,
    device=device,
    name="Model Forward"
)

print(result.summary())
```

### MD Trajectory Profiling

```python
from cuda.benchmark_utils import benchmark_md_trajectory

# Simulate 1000-step MD trajectory
trajectory = [generate_atoms(i) for i in range(1000)]

result = benchmark_md_trajectory(
    model_func=lambda atoms: model(atoms),
    inputs=trajectory,
    check_memory_leak=True  # Critical for MD!
)

# Check results
if result.memory_delta_gb * 1024 > 10:
    print("WARNING: Memory leak detected!")
else:
    print("PASS: Memory stable over trajectory")
```

### Memory Leak Detection

```python
from cuda.device_utils import get_gpu_memory_info, check_memory_leak

initial_mem = get_gpu_memory_info()['allocated']

# Run many iterations
for i in range(10000):
    output = model(input)

# Check for leak
no_leak = check_memory_leak(initial_mem, tolerance_mb=10.0)
```

### Implementation Comparison

```python
from cuda.benchmark_utils import compare_implementations, print_comparison_table

implementations = {
    "PyTorch Baseline": baseline_forward,
    "torch.compile": compiled_forward,
    "TensorRT": tensorrt_forward,
}

results = compare_implementations(implementations, input_tensor, n_calls=1000)
print_comparison_table(results)
```

## Performance Metrics

### Latency Statistics
- **Mean**: Average time per call
- **Median**: Typical time (robust to outliers)
- **Std**: Variance in timing
- **P95/P99**: Tail latency (critical for MD stability)

### Memory Metrics
- **Allocated**: Currently allocated by PyTorch
- **Reserved**: Reserved by caching allocator
- **Peak**: Maximum allocation during run
- **Delta**: Change over trajectory (should be ~0)

### Throughput
- **Calls/sec**: Average throughput
- Note: For MD, latency is more important than throughput

## Best Practices

### DO:
1. Always warmup CUDA before benchmarking
2. Use `CUDATimer` for GPU timing (not `time.time()`)
3. Check for memory leaks in long runs
4. Monitor P95/P99 latency for MD applications
5. Synchronize device before timing measurements

### DON'T:
1. Use CPU timers for GPU operations
2. Forget to disable gradients (`torch.no_grad()`)
3. Ignore memory stability in MD simulations
4. Benchmark without warmup
5. Rely solely on throughput metrics for MD

## Testing

```bash
# Run unit tests
pytest tests/unit/test_cuda_device_utils.py -v

# Run with coverage
pytest tests/unit/test_cuda_device_utils.py --cov=src/cuda
```

## Documentation

- **Full Guide**: `docs/CUDA_SETUP_GUIDE.md`
- **Quick Reference**: `docs/PROFILING_QUICK_REFERENCE.md`
- **Examples**: `benchmarks/profile_example.py`

## Environment Verification

```bash
# Check CUDA environment
python scripts/check_cuda.py

# Export GPU specs
python scripts/check_cuda.py --export-json docs/gpu_specs.json
```

## Requirements

- Python 3.9+
- PyTorch 2.0+ with CUDA support
- NVIDIA GPU with compute capability 7.0+
- CUDA Toolkit 11.0+ (for custom kernels)

## GPU Support

Tested on:
- NVIDIA GeForce RTX 3080 Ti (Ampere, CC 8.6)
- Compatible with all modern NVIDIA GPUs

## Future Extensions

- Custom CUDA kernels for force field operations
- Triton kernel implementations
- TensorRT integration
- Multi-GPU support for parallel MD
- Kernel fusion utilities

## See Also

- PyTorch CUDA documentation
- Nsight Systems profiling guide
- TensorRT optimization guide
