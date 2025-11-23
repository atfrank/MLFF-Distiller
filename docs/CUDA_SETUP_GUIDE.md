# CUDA Development Environment Setup Guide

**Date**: 2025-11-23
**Status**: Complete
**Issue**: #8 - Set Up CUDA Development Environment

## Executive Summary

This guide documents the CUDA development environment for MLFF Distiller, optimized for molecular dynamics (MD) simulation performance profiling and optimization. The environment is configured to measure per-call latency and memory stability over millions of inference calls, which are critical for MD applications.

## System Specifications

### Hardware
- **GPU**: NVIDIA GeForce RTX 3080 Ti
- **Compute Capability**: 8.6 (Ampere architecture)
- **Memory**: 11.65 GB GDDR6X
- **Multiprocessors**: 80 SMs
- **Max Threads per MP**: 1536
- **Warp Size**: 32
- **L2 Cache**: 6.00 MB

### Software Environment
- **OS**: Linux 6.8.0-52-generic (Ubuntu)
- **Python**: 3.13.9
- **CUDA Toolkit**: 12.6
- **CUDA Runtime (PyTorch)**: 12.1
- **Driver Version**: 565.57.01
- **PyTorch**: 2.5.1+cu121
- **cuDNN**: 9.1.0

### Profiling Tools
- **Nsight Systems (nsys)**: 2024.5.1 ✓ Installed
- **Nsight Compute (ncu)**: Not installed
- **torch.profiler**: ✓ Available

### Optional CUDA Packages
- **CuPy**: Not installed
- **Triton**: Not installed (planned for kernel development)
- **PyCUDA**: ✓ Installed

## Quick Start

### Verify CUDA Environment
```bash
# Run comprehensive CUDA check
python scripts/check_cuda.py --verbose

# Export GPU specifications
python scripts/check_cuda.py --export-json docs/gpu_specs.json
```

### Run Example Profiling
```bash
# Basic profiling example
python benchmarks/profile_example.py --n-calls 100

# MD trajectory simulation (1000 steps)
python benchmarks/profile_example.py --n-trajectory-steps 1000

# Memory scaling analysis
python benchmarks/profile_example.py --profile-memory

# Export PyTorch profiler traces
python benchmarks/profile_example.py --export-traces
```

## Architecture Overview

### Directory Structure
```
MLFF_Distiller/
├── src/cuda/
│   ├── __init__.py
│   ├── device_utils.py       # Device management, memory tracking
│   └── benchmark_utils.py    # MD-focused benchmarking tools
├── scripts/
│   └── check_cuda.py         # Environment verification
├── benchmarks/
│   └── profile_example.py    # Profiling workflow examples
├── tests/unit/
│   └── test_cuda_device_utils.py  # Unit tests
└── docs/
    ├── CUDA_SETUP_GUIDE.md   # This file
    └── gpu_specs.json        # Exported GPU specifications
```

### Key Modules

#### 1. Device Utilities (`src/cuda/device_utils.py`)

**Core Functions**:
- `get_device()`: Intelligent device selection with fallback
- `get_gpu_info()`: Comprehensive GPU specifications
- `get_gpu_memory_info()`: Real-time memory monitoring
- `warmup_cuda()`: CUDA initialization for benchmarking
- `check_memory_leak()`: Critical for MD stability testing

**Context Managers**:
- `cuda_memory_tracker`: Track memory usage in code sections
- `torch_device_context`: Temporarily switch default device

**MD-Specific Features**:
- Memory leak detection (critical for million-call simulations)
- Peak memory tracking
- Cache management

#### 2. Benchmark Utilities (`src/cuda/benchmark_utils.py`)

**Key Classes**:
- `BenchmarkResult`: Statistical analysis of timing data
- `CUDATimer`: High-precision GPU timing using CUDA events
- `ProfilerContext`: PyTorch profiler integration

**Core Functions**:
- `benchmark_function()`: General function benchmarking
- `benchmark_md_trajectory()`: **MD-specific trajectory profiling**
- `compare_implementations()`: Multi-implementation comparison
- `profile_with_nsys()`: Nsight Systems integration

**MD-Focused Metrics**:
- **Per-call latency** (mean, median, std, P95, P99)
- **Memory stability** over trajectories
- **Throughput** (calls/second)
- **Statistical distributions** (for outlier detection)

#### 3. Environment Verification (`scripts/check_cuda.py`)

Comprehensive checks:
- NVIDIA driver and GPU detection
- CUDA toolkit (nvcc, libraries)
- PyTorch CUDA support
- Profiling tools availability
- Optional package detection
- JSON export for documentation

## MD Performance Profiling Philosophy

### Critical Differences from Typical ML Benchmarking

In MD simulations:
1. **Latency >> Throughput**: Model called millions of times sequentially
2. **Memory Stability**: No leaks over millions of calls
3. **Variance Matters**: Outliers cause MD instability
4. **Sustained Performance**: Peak performance must be maintained

### Our Benchmarking Approach

```python
# Example: MD trajectory profiling
from cuda.benchmark_utils import benchmark_md_trajectory

# Simulate 1000-step MD trajectory
results = benchmark_md_trajectory(
    model_func=lambda atoms: model.forward(atoms),
    inputs=trajectory_atoms,  # 1000 atomic configurations
    check_memory_leak=True,   # CRITICAL for MD
    leak_tolerance_mb=10.0
)

# Analyze per-call latency (not throughput)
print(f"Mean latency: {results.mean_ms:.4f} ms")
print(f"P95 latency:  {results.p95_ms:.4f} ms")  # Critical for MD stability
print(f"Memory delta: {results.memory_delta_gb * 1024:.2f} MB")  # Should be ~0
```

### Key Metrics for MD

| Metric | Target | Why It Matters |
|--------|--------|----------------|
| Mean latency | <0.5 ms/call | Per-step simulation time |
| P95 latency | <2x mean | Outliers slow entire simulation |
| Memory leak | <10 MB/1000 calls | Prevents OOM in long runs |
| Memory variance | <1% | Stability over millions of calls |

## Profiling Workflows

### 1. Basic Performance Profiling

```python
from cuda.benchmark_utils import benchmark_function

result = benchmark_function(
    model.forward,
    input_tensor,
    n_warmup=10,
    n_calls=100,
    device='cuda',
    name="Model Forward"
)

print(result.summary())
```

### 2. MD Trajectory Simulation

```python
from cuda.benchmark_utils import benchmark_md_trajectory

# Generate trajectory (simulating MD timesteps)
trajectory = [generate_atoms() for _ in range(1000)]

result = benchmark_md_trajectory(
    model_func=lambda atoms: model(atoms),
    inputs=trajectory,
    check_memory_leak=True
)

# Check for memory leaks
if abs(result.memory_delta_gb * 1024) > 10:
    print("WARNING: Memory leak detected!")
```

### 3. Implementation Comparison

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

### 4. PyTorch Profiler Integration

```python
from cuda.benchmark_utils import ProfilerContext

with ProfilerContext(
    output_dir="profiling_results",
    profile_memory=True,
    name="model_profile"
) as prof:
    for _ in range(20):
        output = model(input)
        prof.step()

# Results saved to profiling_results/model_profile_trace.json
# View in chrome://tracing
```

### 5. Nsight Systems Profiling

```bash
# Generate detailed GPU profiling
nsys profile \
    --trace=cuda,nvtx \
    --output=model_profile.nsys-rep \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    python benchmark_model.py

# View results
nsys-ui model_profile.nsys-rep
```

## GPU Architecture Insights (RTX 3080 Ti)

### Ampere Architecture (Compute Capability 8.6)

**Key Features**:
- **Tensor Cores**: 3rd generation (supports FP16, BF16, TF32, INT8)
- **RT Cores**: 2nd generation (for ray tracing)
- **CUDA Cores**: 10,240
- **Memory Bandwidth**: ~912 GB/s (theoretical)

**Optimization Opportunities**:
1. **FP16/BF16**: 2x throughput vs FP32 on tensor cores
2. **TF32**: Automatic mixed precision for matrix operations
3. **Warp-level Primitives**: Efficient for small atomic systems
4. **Shared Memory**: 164 KB per SM (use for atom pair caching)

### Performance Characteristics

| Operation Type | Expected Performance |
|----------------|---------------------|
| Matrix multiply (FP32) | ~20 TFLOPS |
| Matrix multiply (TF32) | ~40 TFLOPS |
| Matrix multiply (FP16) | ~80 TFLOPS |
| Memory Bandwidth | ~900 GB/s |
| L2 Cache Hit Latency | ~200 cycles |

### Memory Hierarchy

```
Registers (per thread)      : 255 max
Shared Memory (per block)   : 48-164 KB
L1 Cache (per SM)           : Combined with shared memory
L2 Cache (global)           : 6 MB
Global Memory               : 11.65 GB @ ~900 GB/s
```

## Optimization Strategies for MD

### 1. Memory Optimization

```python
# Pre-allocate buffers to avoid repeated allocation
class MDOptimizedModel:
    def __init__(self):
        self._buffer = None

    def forward(self, atoms):
        # Reuse buffer instead of allocating
        if self._buffer is None:
            self._buffer = torch.empty(...)
        # Use in-place operations where possible
```

### 2. Kernel Fusion

Target operations for fusion:
- Element-wise operations (activation, scaling)
- Neighbor list computation + feature extraction
- Force computation + aggregation

### 3. Mixed Precision

```python
# Use automatic mixed precision
with torch.cuda.amp.autocast():
    output = model(input)
```

### 4. CUDA Graphs (for static graphs)

```python
# Capture static computation graph
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model(static_input)

# Replay graph (faster than re-launching kernels)
g.replay()
```

## Profiling Tools Reference

### torch.profiler

**Advantages**:
- Built into PyTorch
- Python-level integration
- Memory profiling
- TensorBoard visualization

**Best For**:
- Python-level bottlenecks
- Memory profiling
- Operation-level analysis

### Nsight Systems (nsys)

**Advantages**:
- System-level view
- CPU + GPU timeline
- Kernel launch overhead
- Multi-GPU profiling

**Best For**:
- End-to-end profiling
- GPU utilization analysis
- Identifying idle time
- Multi-process profiling

### Nsight Compute (ncu)

**Note**: Not currently installed, but recommended for kernel optimization.

**Advantages**:
- Detailed kernel metrics
- Occupancy analysis
- Memory throughput
- Warp efficiency

**Best For**:
- Custom kernel optimization
- Memory access patterns
- Instruction-level analysis

## Testing and Validation

### Unit Tests

```bash
# Run CUDA device utils tests
pytest tests/unit/test_cuda_device_utils.py -v

# All tests should pass
# 23 passed, 4 skipped (expected - testing failure cases)
```

### Validation Checklist

- [x] CUDA toolkit installed and accessible
- [x] PyTorch recognizes GPU
- [x] Device utilities work correctly
- [x] Benchmark utilities produce valid results
- [x] Memory tracking detects leaks
- [x] Profiling tools available
- [x] Example scripts run successfully
- [x] Unit tests pass

## Common Issues and Solutions

### Issue: CUDA Out of Memory

**Solution**:
```python
# Clear cache between runs
from cuda.device_utils import empty_cache
empty_cache()

# Monitor memory
from cuda.device_utils import print_gpu_memory_summary
print_gpu_memory_summary()

# Reduce batch size or model size
```

### Issue: Slow First Iteration

**Solution**:
```python
# Always warmup CUDA before benchmarking
from cuda.device_utils import warmup_cuda
warmup_cuda(n_iterations=10)
```

### Issue: Memory Leak in Long Runs

**Solution**:
```python
# Use memory leak detection
from cuda.device_utils import check_memory_leak, get_gpu_memory_info

initial_mem = get_gpu_memory_info()['allocated']

# ... run simulation ...

no_leak = check_memory_leak(initial_mem, tolerance_mb=10.0)
if not no_leak:
    print("Memory leak detected!")
```

## Next Steps

### Immediate (Issue #9 - MD Profiling Framework)
1. Integrate with teacher model wrappers (blocked by Issue #2)
2. Create MD-specific profiling scripts
3. Profile teacher models on realistic trajectories
4. Establish baseline performance metrics

### Short-term (Milestone 1)
1. Profile Orb-v2 and FeNNol teacher models
2. Identify computational bottlenecks
3. Measure memory requirements for typical MD systems
4. Document performance characteristics

### Medium-term (Milestone 5 - CUDA Optimization)
1. Implement custom CUDA kernels for bottlenecks
2. Apply torch.compile with various backends
3. Explore TensorRT optimization
4. Target 5-10x speedup on MD trajectories

## Resources

### Documentation
- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [PyTorch Profiler Tutorial](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

### Internal Documentation
- `docs/MD_REQUIREMENTS_UPDATE_SUMMARY.md`: MD performance requirements
- `docs/DROP_IN_COMPATIBILITY_GUIDE.md`: Interface requirements
- `docs/gpu_specs.json`: Exported GPU specifications

### Code Examples
- `benchmarks/profile_example.py`: Profiling workflow examples
- `tests/unit/test_cuda_device_utils.py`: Usage examples in tests

## Conclusion

The CUDA development environment is fully configured and validated for MLFF Distiller. All tools are in place for profiling and optimizing MD simulation performance, with a focus on per-call latency and memory stability over millions of inference calls.

**Key Deliverables**:
- ✅ Device management utilities (`src/cuda/device_utils.py`)
- ✅ MD-focused benchmarking tools (`src/cuda/benchmark_utils.py`)
- ✅ Environment verification script (`scripts/check_cuda.py`)
- ✅ Profiling examples (`benchmarks/profile_example.py`)
- ✅ Comprehensive unit tests (23 passing)
- ✅ GPU specifications documented
- ✅ Profiling tools configured (nsys, torch.profiler)

**Status**: Issue #8 COMPLETE - Ready for Issue #9 (MD profiling framework)

---

**Document Version**: 1.0
**Last Updated**: 2025-11-23
**Maintained By**: CUDA Optimization Engineer
**Next Review**: After Issue #9 completion
