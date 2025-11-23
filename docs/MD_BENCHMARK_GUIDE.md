# MD Trajectory Benchmark Framework Guide

**Author**: Testing & Benchmark Engineer
**Date**: 2025-11-23
**Status**: Production Ready

## Overview

The MD Trajectory Benchmark Framework provides comprehensive tools for measuring the performance of molecular dynamics simulations with machine learning force fields. Unlike single-inference benchmarks, this framework focuses on metrics critical for production MD workloads:

1. **Per-call latency** (mean, median, P95, P99) - critical for MD where models are called millions of times
2. **Memory stability** - tracks memory usage over long trajectories to detect leaks
3. **Energy conservation** - validates correctness in NVE simulations
4. **Total trajectory time** - measures real-world MD performance

## Key Features

- **Multiple MD protocols**: NVE (microcanonical), NVT (canonical), NPT (isothermal-isobaric)
- **Comprehensive metrics**: Latency distribution, throughput, memory usage, energy conservation
- **Comparison framework**: Compare multiple calculators on identical trajectories
- **Visualization tools**: Automatic generation of plots and reports
- **JSON export**: Results saved for tracking performance over time
- **CI integration**: Designed for performance regression detection

## Architecture

The framework consists of three main components:

### 1. Core Benchmark Module (`src/mlff_distiller/benchmarks/md_trajectory.py`)

**Key Classes**:
- `MDProtocol`: Enum for MD ensemble types (NVE, NVT, NPT)
- `BenchmarkResults`: Data class storing results with automatic statistical analysis
- `MDTrajectoryBenchmark`: Main benchmark runner class
- `create_benchmark_system`: Utility for creating test systems

**Key Functions**:
- `compare_calculators()`: Compare multiple calculators on same trajectory

### 2. Visualization Module (`src/mlff_distiller/benchmarks/visualization.py`)

**Functions**:
- `plot_latency_distribution()`: Histogram of per-step latencies
- `plot_energy_conservation()`: Energy vs time for NVE validation
- `plot_performance_comparison()`: Multi-panel comparison plots
- `create_benchmark_report()`: Comprehensive HTML/markdown report

### 3. CLI Script (`benchmarks/md_benchmark.py`)

Command-line interface for running benchmarks, comparisons, and analysis.

## Installation

The benchmark framework is part of the `mlff_distiller` package:

```bash
# Install package in development mode
cd /home/aaron/ATX/software/MLFF_Distiller
pip install -e .

# Required dependencies
pip install ase numpy matplotlib torch
```

## Usage

### Quick Start

```python
from mlff_distiller.models.teacher_wrappers import OrbCalculator
from mlff_distiller.benchmarks import MDTrajectoryBenchmark, MDProtocol

# Create calculator
calc = OrbCalculator(model_name="orb-v2", device="cuda")

# Setup benchmark
benchmark = MDTrajectoryBenchmark(
    calculator=calc,
    system_type="silicon",
    n_atoms=64,
    protocol=MDProtocol.NVE
)

# Run 1000-step trajectory
results = benchmark.run(n_steps=1000)

# Print results
print(results.summary())

# Save to file
results.save("results/orb_v2_si64_nve.json")
```

### Command-Line Usage

#### 1. Single Calculator Benchmark

```bash
# Basic benchmark
python benchmarks/md_benchmark.py \
    --calculator orb-v2 \
    --system silicon \
    --atoms 64 \
    --steps 1000 \
    --protocol NVE \
    --device cuda \
    --output results/orb_v2_si64.json

# NVT simulation
python benchmarks/md_benchmark.py \
    --calculator orb-v2 \
    --system silicon \
    --atoms 128 \
    --steps 1000 \
    --protocol NVT \
    --device cuda
```

#### 2. Compare Multiple Calculators

```bash
# Compare Orb-v2 vs Orb-v3
python benchmarks/md_benchmark.py \
    --compare orb-v2 orb-v3 \
    --system silicon \
    --atoms 128 \
    --steps 1000 \
    --output results/comparison_si128

# This creates:
# - results/comparison_si128/orb-v2.json
# - results/comparison_si128/orb-v3.json
# - results/comparison_si128/report.md
# - results/comparison_si128/comparison.png
# - results/comparison_si128/latency_dist.png
# - results/comparison_si128/energy_conservation.png
```

#### 3. Benchmark Suites

```bash
# Quick suite (for CI/testing)
python benchmarks/md_benchmark.py --suite quick --device cuda

# Baseline characterization
python benchmarks/md_benchmark.py --suite baseline --output results/baseline

# Comprehensive benchmark
python benchmarks/md_benchmark.py --suite comprehensive --output results/full_benchmark
```

#### 4. Analyze Existing Results

```bash
# Analyze single result
python benchmarks/md_benchmark.py --analyze results/orb_v2_si64.json

# Analyze directory of results
python benchmarks/md_benchmark.py --analyze results/baseline/
```

### Python API

#### Basic Benchmark

```python
from mlff_distiller.benchmarks import MDTrajectoryBenchmark, MDProtocol
from mlff_distiller.models.teacher_wrappers import OrbCalculator

# Create calculator
calc = OrbCalculator(model_name="orb-v2", device="cuda")

# Run benchmark
benchmark = MDTrajectoryBenchmark(
    calculator=calc,
    system_type="silicon",
    n_atoms=64,
    protocol=MDProtocol.NVE,
    timestep_fs=1.0,
    temperature_K=300.0,
)

results = benchmark.run(
    n_steps=1000,
    warmup_steps=10,
    memory_sample_interval=100,
    energy_sample_interval=1,
)

print(f"Mean latency: {results.mean_step_time_ms:.2f} ms")
print(f"Throughput: {results.steps_per_second:.1f} steps/s")
print(f"Energy drift: {results.energy_drift:.2e}")
```

#### Compare Multiple Calculators

```python
from mlff_distiller.benchmarks import compare_calculators
from mlff_distiller.models.teacher_wrappers import OrbCalculator

# Create calculators
calculators = {
    "Orb-v2": OrbCalculator(model_name="orb-v2", device="cuda"),
    "Orb-v3": OrbCalculator(model_name="orb-v3", device="cuda"),
}

# Run comparison
results = compare_calculators(
    calculators=calculators,
    system_type="silicon",
    n_atoms=128,
    protocol=MDProtocol.NVE,
    n_steps=1000,
)

# Calculate speedup
baseline_time = results["Orb-v2"].mean_step_time_ms
optimized_time = results["Orb-v3"].mean_step_time_ms
speedup = baseline_time / optimized_time
print(f"Speedup: {speedup:.2f}x")
```

#### Create Visualizations

```python
from mlff_distiller.benchmarks import create_benchmark_report

create_benchmark_report(
    results=results,
    output_dir="reports/comparison_2025-11-23",
    title="Orb-v2 vs Orb-v3 Comparison"
)
```

## Benchmark Metrics

### Latency Metrics

| Metric | Description | Importance |
|--------|-------------|------------|
| **Mean** | Average step time | Primary performance metric |
| **Median** | Middle value (50th percentile) | Typical performance |
| **P95** | 95th percentile | Handling moderate outliers |
| **P99** | 99th percentile | Worst-case performance |
| **Std** | Standard deviation | Performance consistency |

For MD simulations, **P95 and P99 matter** because outliers accumulate over millions of steps.

### Memory Metrics

| Metric | Description | Critical For |
|--------|-------------|--------------|
| **Peak Memory** | Maximum GPU memory used | Resource planning |
| **Memory Delta** | Growth over trajectory | Leak detection |
| **Memory Samples** | Memory at intervals | Long-run stability |

**Memory leaks** are detected automatically (>10 MB growth flagged as potential leak).

### Energy Metrics (NVE only)

| Metric | Description | Threshold |
|--------|-------------|-----------|
| **Energy Drift** | Relative energy change | < 1e-4 acceptable |
| **Energy Std** | Energy fluctuation | System-dependent |

Energy conservation validates MD correctness. Large drift indicates numerical instability.

## Supported Systems

### Crystal Systems

```python
# Silicon (diamond structure)
atoms = create_benchmark_system("silicon", n_atoms=64)

# Copper (FCC)
atoms = create_benchmark_system("copper", n_atoms=108)

# Aluminum (FCC)
atoms = create_benchmark_system("aluminum", n_atoms=108)
```

### Molecular Systems

```python
# Water box (periodic)
atoms = create_benchmark_system("water", n_atoms=96)  # ~32 molecules
```

All systems are periodic with appropriate lattice constants and Maxwell-Boltzmann velocity initialization.

## Output Format

### BenchmarkResults JSON

```json
{
  "metadata": {
    "name": "OrbCalculator_silicon_64atoms_NVE",
    "protocol": "NVE",
    "n_steps": 1000,
    "n_atoms": 64,
    "system_type": "silicon",
    "device": "cuda"
  },
  "performance": {
    "total_time_s": 12.34,
    "steps_per_second": 81.0,
    "mean_step_time_ms": 12.34,
    "std_step_time_ms": 0.56,
    "median_step_time_ms": 12.28,
    "p95_step_time_ms": 13.45,
    "p99_step_time_ms": 14.12
  },
  "memory": {
    "before_gb": 2.15,
    "after_gb": 2.16,
    "delta_gb": 0.01,
    "peak_gb": 2.18,
    "leak_detected": false
  },
  "energy": {
    "std": 0.0023,
    "drift": -1.2e-05
  }
}
```

## Best Practices

### 1. Warmup Steps

Always use warmup steps to allow JIT compilation and cache warmup:

```python
results = benchmark.run(
    n_steps=1000,
    warmup_steps=10,  # Recommended: 10-20 steps
)
```

### 2. System Size Selection

| System Size | Use Case | Typical Time |
|-------------|----------|--------------|
| 32-64 atoms | Quick testing, CI | < 1 minute |
| 128-256 atoms | Standard benchmark | 2-5 minutes |
| 512-1024 atoms | Large system test | 10-30 minutes |

### 3. Step Count

| Step Count | Use Case |
|------------|----------|
| 50-100 | Smoke test |
| 1000 | Standard benchmark |
| 10000+ | Production validation |

### 4. Protocol Selection

| Protocol | When to Use | Energy Conservation |
|----------|-------------|---------------------|
| **NVE** | Correctness check | Yes - must be conserved |
| **NVT** | Realistic MD | No - thermostat active |
| **NPT** | Production MD | No - barostat active |

## Performance Targets

Based on the project's 5-10x speedup goal:

### Teacher Model Baseline (Orb-v2, 128 atoms)
- Target: ~15-20 ms/step on A100 GPU
- Throughput: ~50-60 steps/second
- Memory: ~2-4 GB

### Student Model Target
- Target: ~2-3 ms/step (5-10x faster)
- Throughput: ~300-500 steps/second
- Memory: <2 GB (lower than teacher)

## Troubleshooting

### Issue: Memory Leak Detected

**Symptom**: Warning about memory growth

**Solutions**:
1. Check for persistent caching in calculator
2. Verify torch operations don't accumulate graphs
3. Add explicit `torch.cuda.empty_cache()` calls
4. Use `torch.no_grad()` context for inference

### Issue: Large Energy Drift

**Symptom**: NVE energy drift > 1e-4

**Solutions**:
1. Reduce timestep (try 0.5 fs instead of 1.0 fs)
2. Check force calculations are correct
3. Verify numerical precision (use float64 for testing)
4. Increase warmup steps

### Issue: High Latency Variance

**Symptom**: Large std or P99-P50 gap

**Solutions**:
1. Ensure GPU is not throttling (check temperature)
2. Disable other GPU processes
3. Check for dynamic graph construction
4. Use fixed-size batching

## Integration with CI

### GitHub Actions Example

```yaml
name: Performance Benchmark

on:
  push:
    branches: [main]
  pull_request:

jobs:
  benchmark:
    runs-on: ubuntu-latest-gpu
    steps:
      - uses: actions/checkout@v2
      - name: Install dependencies
        run: pip install -e .[benchmark]

      - name: Run quick benchmark
        run: |
          python benchmarks/md_benchmark.py \
            --suite quick \
            --device cuda \
            --output results/ci_benchmark

      - name: Check for regressions
        run: |
          python scripts/check_performance_regression.py \
            --current results/ci_benchmark \
            --baseline results/baseline_benchmark \
            --threshold 1.1  # Allow 10% slowdown
```

## Advanced Usage

### Custom Calculator

```python
from ase.calculators.calculator import Calculator

class MyCustomCalculator(Calculator):
    implemented_properties = ['energy', 'forces']

    def calculate(self, atoms=None, properties=['energy'], system_changes=None):
        # Your implementation
        pass

# Benchmark custom calculator
benchmark = MDTrajectoryBenchmark(
    calculator=MyCustomCalculator(),
    system_type="silicon",
    n_atoms=64,
)
```

### Batch Processing

```python
# Benchmark multiple configurations
configs = [
    ("silicon", 64, MDProtocol.NVE),
    ("silicon", 128, MDProtocol.NVE),
    ("copper", 108, MDProtocol.NVT),
]

all_results = {}
for system, n_atoms, protocol in configs:
    benchmark = MDTrajectoryBenchmark(
        calculator=calc,
        system_type=system,
        n_atoms=n_atoms,
        protocol=protocol,
    )
    all_results[f"{system}_{n_atoms}_{protocol.value}"] = benchmark.run(1000)
```

## References

- **ASE Documentation**: https://wiki.fysik.dtu.dk/ase/
- **MD Protocols**: Allen & Tildesley, "Computer Simulation of Liquids"
- **MLFF Benchmarking**: Best practices from Orb, MACE, NequIP papers

## Support

For questions or issues:
1. Check this guide first
2. Review test examples in `tests/unit/test_md_benchmark.py`
3. Open GitHub issue with benchmark configuration and error message

---

**Last Updated**: 2025-11-23
**Version**: 1.0
**Status**: Production Ready
