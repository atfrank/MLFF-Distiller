# Profiling Reports

This directory contains profiling results from teacher and student model analysis on MD trajectories.

## Directory Structure

```
profiling_reports/
├── README.md                           # This file
├── orb_v2_profile.json                # Orb-v2 baseline profile
├── fennol_ani2x_profile.json          # FeNNol ANI-2x profile
├── orb_v2_scaling_*.json              # System size scaling analysis
└── comparison_*.json                   # Model comparison results
```

## File Format

All profiling results are stored in JSON format for reproducibility and analysis.

### JSON Schema

```json
{
  "name": "Model name and configuration",
  "model_name": "orb-v2",
  "n_steps": 1000,
  "device": "cuda",
  "latencies_ms": [1.23, 1.25, ...],
  "mean_latency_ms": 1.24,
  "median_latency_ms": 1.23,
  "std_latency_ms": 0.15,
  "min_latency_ms": 1.10,
  "max_latency_ms": 1.50,
  "p95_latency_ms": 1.40,
  "p99_latency_ms": 1.45,
  "energy_time_ms": 0.30,
  "forces_time_ms": 0.85,
  "stress_time_ms": null,
  "memory_initial_gb": 1.5,
  "memory_final_gb": 1.52,
  "memory_peak_gb": 1.8,
  "memory_per_step_gb": [1.5, 1.51, ...],
  "n_atoms": 64,
  "system_size": "64 atoms",
  "total_time_s": 1.24,
  "steps_per_second": 806.5,
  "timestamp": "2025-11-23 10:30:00",
  "notes": ""
}
```

## Generating Reports

### Profile a Single Model

```bash
python benchmarks/profile_teachers.py \
    --model orb-v2 \
    --n-steps 1000 \
    --n-atoms 64 \
    --system silicon
```

Results saved to: `orb_v2_profile.json`

### System Size Scaling

```bash
python benchmarks/profile_teachers.py \
    --model orb-v2 \
    --system-sizes 32,64,128,256 \
    --n-steps 100
```

Results saved to: `orb_v2_scaling_*.json`

### Compare Models

```bash
python benchmarks/profile_teachers.py --compare-all --n-steps 1000
```

Results saved to multiple JSON files.

## Analyzing Reports

### Load and Analyze in Python

```python
from mlff_distiller.cuda.md_profiler import MDProfileResult

# Load result
result = MDProfileResult.load_json('profiling_reports/orb_v2_profile.json')

# Print summary
print(result.summary())

# Check performance
print(f"Mean latency: {result.mean_latency_ms:.4f} ms")
print(f"Memory stable: {result.memory_stable}")

# Identify hotspots
from mlff_distiller.cuda.md_profiler import identify_hotspots
hotspots = identify_hotspots(result)
print(f"Recommendations: {hotspots['recommendations']}")
```

### Compare Results

```python
import json

# Load multiple results
with open('profiling_reports/orb_v2_profile.json') as f:
    orb_data = json.load(f)

with open('profiling_reports/fennol_ani2x_profile.json') as f:
    fennol_data = json.load(f)

# Compare
speedup = orb_data['mean_latency_ms'] / fennol_data['mean_latency_ms']
print(f"FeNNol is {speedup:.2f}x faster than Orb-v2")
```

## Profiling Metrics

### Key Metrics

- **Mean Latency**: Average time per MD step (ms)
- **P95/P99 Latency**: Tail latency for stability analysis
- **Memory Delta**: Memory leak detection (<10 MB acceptable)
- **µs/atom**: Scalability metric (latency / n_atoms)
- **Component Times**: Energy, forces, stress breakdown

### Performance Targets

For 64-atom systems:
- **Teacher Models**: 2-10 ms/step baseline
- **Student Models**: 0.5-1.0 ms/step target (5-10x speedup)
- **Memory**: Stable over 1000+ steps (<10 MB increase)
- **Variance**: CV <20% for stable MD simulation

## Profiling Best Practices

1. **Warmup**: Always use 10+ warmup steps
2. **Trajectory Length**: ≥100 steps for statistics, ≥1000 for leak detection
3. **Multiple Runs**: Average across 3+ runs for reliability
4. **System Sizes**: Test 32, 64, 128, 256 atoms for scaling
5. **Save Results**: Always export to JSON for reproducibility

## Notes

- Results are hardware-dependent (GPU model, CUDA version)
- All profiling done on: RTX 3080 Ti, CUDA 12.6, PyTorch 2.5.1
- Trajectory generation uses synthetic thermal perturbations
- For production, use real MD trajectories from equilibrated simulations

## Contact

Issues or questions about profiling:
- See: `docs/PROFILING_RESULTS.md`
- Issue #9 - Performance Profiling Framework for MD Workloads
