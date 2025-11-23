# CUDA Profiling Quick Reference

**Quick reference for profiling MLFF models in MD simulations**

## Setup and Verification

```bash
# Verify CUDA environment
python scripts/check_cuda.py

# Quick device check
python -c "from cuda.device_utils import print_device_summary; print_device_summary()"
```

## Common Profiling Patterns

### 1. Quick Benchmark (Single Function)

```python
from cuda.benchmark_utils import benchmark_function

result = benchmark_function(
    model.forward,
    input_tensor,
    n_warmup=10,
    n_calls=100,
    device='cuda',
    name="Forward Pass"
)
print(result.summary())
```

### 2. MD Trajectory Profiling

```python
from cuda.benchmark_utils import benchmark_md_trajectory

# Simulate MD trajectory
trajectory = [generate_atoms(i) for i in range(1000)]

result = benchmark_md_trajectory(
    model_func=lambda atoms: model(atoms),
    inputs=trajectory,
    check_memory_leak=True,  # IMPORTANT for MD!
    leak_tolerance_mb=10.0
)

print(f"Mean latency: {result.mean_ms:.4f} ms")
print(f"P95 latency: {result.p95_ms:.4f} ms")
print(f"Memory delta: {result.memory_delta_gb * 1024:.2f} MB")
```

### 3. Compare Implementations

```python
from cuda.benchmark_utils import compare_implementations, print_comparison_table

implementations = {
    "Baseline": baseline_model.forward,
    "Optimized": optimized_model.forward,
    "torch.compile": compiled_model.forward,
}

results = compare_implementations(
    implementations,
    input_tensor,
    n_calls=1000,
    device='cuda'
)

print_comparison_table(results)
```

### 4. Memory Tracking

```python
from cuda.device_utils import cuda_memory_tracker, get_gpu_memory_info

# Track specific code section
with cuda_memory_tracker(name="Model Forward"):
    output = model(input)

# Check current memory
mem_info = get_gpu_memory_info()
print(f"GPU Memory: {mem_info['allocated']:.2f} / {mem_info['total']:.2f} GB")
```

### 5. Memory Leak Detection

```python
from cuda.device_utils import get_gpu_memory_info, check_memory_leak

# Before long run
initial_mem = get_gpu_memory_info()['allocated']

# Run many iterations
for i in range(10000):
    output = model(input)
    # ... MD simulation step ...

# Check for leak
no_leak = check_memory_leak(initial_mem, tolerance_mb=10.0)
if not no_leak:
    print("WARNING: Memory leak detected!")
```

## PyTorch Profiler

### Basic Usage

```python
from cuda.benchmark_utils import ProfilerContext

with ProfilerContext(
    output_dir="profiling_results",
    profile_memory=True,
    name="model_inference"
) as prof:
    for _ in range(20):
        output = model(input)
        prof.step()

# View results in chrome://tracing
# File: profiling_results/model_inference_trace.json
```

### Manual Profiler

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True
) as prof:
    output = model(input)

# Print results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# Export trace
prof.export_chrome_trace("trace.json")
```

## Nsight Systems

### Command Line

```bash
# Basic profiling
nsys profile --trace=cuda,nvtx \
    --output=profile.nsys-rep \
    --force-overwrite=true \
    python benchmark_script.py

# With memory tracking
nsys profile --trace=cuda,nvtx \
    --output=profile.nsys-rep \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    python benchmark_script.py

# View results
nsys-ui profile.nsys-rep
```

### Add NVTX Markers (Optional)

```python
import torch.cuda.nvtx as nvtx

nvtx.range_push("Model Forward")
output = model(input)
nvtx.range_pop()

# Or use context manager
with nvtx.range("Model Forward"):
    output = model(input)
```

## Device Management

### Select Device

```python
from cuda.device_utils import get_device

# Auto-select (prefer CUDA)
device = get_device()

# Force specific GPU
device = get_device(gpu_index=0)

# Force CPU
device = get_device(device='cpu')
```

### Warmup CUDA

```python
from cuda.device_utils import warmup_cuda

# Always warmup before benchmarking!
warmup_cuda(n_iterations=10)
```

### Clear Cache

```python
from cuda.device_utils import empty_cache

# Clear CUDA cache
empty_cache()
```

## Performance Tips

### DO: Warmup Before Benchmarking

```python
from cuda.device_utils import warmup_cuda

warmup_cuda()  # Initialize CUDA, compile kernels
# Now run actual benchmarks
```

### DO: Use CUDA Events for Timing

```python
from cuda.benchmark_utils import CUDATimer

timer = CUDATimer()
timer.start()
output = model(input)
elapsed_ms = timer.stop()
```

### DO: Check Memory Leaks

```python
# For MD: Check memory stability over many calls
result = benchmark_md_trajectory(
    model_func=model.forward,
    inputs=trajectory,
    check_memory_leak=True  # ← CRITICAL
)
```

### DON'T: Use time.time() for GPU

```python
# BAD: CPU timer doesn't account for GPU async execution
start = time.time()
output = model(input)
elapsed = time.time() - start  # WRONG!

# GOOD: Use CUDA events or synchronize
from cuda.device_utils import synchronize_device

start = time.time()
output = model(input)
synchronize_device()  # Wait for GPU
elapsed = time.time() - start  # Correct
```

### DON'T: Forget to Synchronize

```python
# BAD: Timing without sync
timer.start()
output = model(input)
elapsed = timer.stop()  # Incomplete results!

# GOOD: CUDATimer handles this automatically
from cuda.benchmark_utils import CUDATimer
timer = CUDATimer()  # Auto-synchronizes
```

## Troubleshooting

### Out of Memory

```python
from cuda.device_utils import empty_cache, print_gpu_memory_summary

empty_cache()
print_gpu_memory_summary()

# Reduce batch size or model size
```

### Inconsistent Timing

```python
# Cause: No warmup
# Solution: Always warmup
from cuda.device_utils import warmup_cuda
warmup_cuda()
```

### Memory Leak Detected

```python
# Check gradients are disabled
with torch.no_grad():
    output = model(input)

# Check for retained references
del output  # Explicitly delete
import gc
gc.collect()
```

## Metrics Interpretation (MD Focus)

### Latency
- **Mean**: Average time per call
- **Median**: Typical time (robust to outliers)
- **P95/P99**: Worst-case latency (important for MD stability)

### Memory
- **Delta**: Should be ~0 for stable MD
- **Peak**: Maximum memory during run
- **Leak**: >10 MB/1000 calls indicates problem

### Throughput
- **Calls/sec**: Not critical for MD (latency matters more)
- Use only for batch processing comparisons

## Example Workflow

```python
#!/usr/bin/env python3
"""Complete profiling workflow for MD model."""

from cuda.device_utils import (
    print_device_summary,
    warmup_cuda,
    get_gpu_memory_info,
    check_memory_leak
)
from cuda.benchmark_utils import (
    benchmark_md_trajectory,
    ProfilerContext
)

# 1. Check environment
print_device_summary()

# 2. Warmup
warmup_cuda()

# 3. Load model
model = load_model()
model.eval()

# 4. Generate trajectory
trajectory = [generate_atoms(i) for i in range(1000)]

# 5. Benchmark
with torch.no_grad():
    result = benchmark_md_trajectory(
        model_func=lambda atoms: model(atoms),
        inputs=trajectory,
        check_memory_leak=True
    )

print(result.summary())

# 6. Detailed profiling (if needed)
with ProfilerContext(output_dir="results", name="md_profile"):
    for atoms in trajectory[:20]:
        output = model(atoms)

print("Profiling complete!")
```

## Quick Command Reference

```bash
# Verify CUDA
python scripts/check_cuda.py

# Run example profiling
python benchmarks/profile_example.py

# Run tests
pytest tests/unit/test_cuda_device_utils.py -v

# Profile with nsys
nsys profile --trace=cuda python script.py

# View nsys results
nsys-ui profile.nsys-rep
```

## Key Files

- `src/cuda/device_utils.py` - Device management
- `src/cuda/benchmark_utils.py` - Benchmarking tools
- `scripts/check_cuda.py` - Environment check
- `benchmarks/profile_example.py` - Examples
- `docs/CUDA_SETUP_GUIDE.md` - Full documentation

---

**For detailed information, see**: `docs/CUDA_SETUP_GUIDE.md`
