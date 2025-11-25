# Student Model Performance Baseline Report

**Date**: 2025-11-24 07:43:48

**Device**: cuda
**Model**: checkpoints/best_model.pt

---

## Single Inference Performance

- **Mean time**: 22.32 ± 0.85 ms
- **Median time**: 22.06 ms
- **95th percentile**: 23.69 ms
- **Throughput**: 44.80 structures/second

### Performance by System Size

| Atoms | Mean (ms) | Std (ms) | Samples |
|-------|-----------|----------|----------|
| 3 | 21.76 | 0.00 | 1 |
| 4 | 21.86 | 0.00 | 1 |
| 5 | 24.83 | 0.00 | 1 |
| 8 | 22.22 | 0.00 | 1 |
| 11 | 21.94 | 0.01 | 2 |
| 12 | 22.16 | 0.14 | 4 |

## Batch Inference Performance

| Batch Size | Total Time (ms) | Time/Structure (ms) | Throughput (struct/s) | Speedup vs Batch=1 |
|------------|------------------|---------------------|------------------------|--------------------|
| 1 | 0.79 | 0.79 | 1258.13 | 1.00x |
| 2 | 43.64 | 21.82 | 45.83 | 0.04x |
| 4 | 87.58 | 21.90 | 45.67 | 0.04x |

## Memory Usage

- **Baseline**: 17.88 MB
- **Peak**: 21.77 MB
- **Inference overhead**: 3.89 MB

## Scaling with System Size

| Atoms | Category | Time (ms) | ms/atom |
|-------|----------|-----------|----------|
| 3 | small | 4.99 ± 8.42 | 1.662 |
| 4 | small | 4.98 ± 8.39 | 1.245 |
| 5 | small | 5.04 ± 8.49 | 1.007 |
| 11 | medium | 5.02 ± 8.46 | 0.456 |
| 12 | medium | 4.95 ± 8.34 | 0.413 |
| 12 | medium | 4.96 ± 8.35 | 0.414 |
| 60 | large | 7.54 ± 13.48 | 0.126 |

## Profiling Analysis

Top 5 most expensive operations:

1. **cudaDeviceSynchronize**
   - CPU time: 0.03 ms
   - Calls: 1

Full profiling results: `/home/aaron/ATX/software/MLFF_Distiller/benchmarks/profiling_results.txt`

---

## Key Findings

- Current inference speed: **22.32 ms/structure**
- Current throughput: **44.80 structures/second**
- Memory footprint: **21.77 MB**

## Next Steps

1. Analyze profiling results to identify bottlenecks
2. Review optimization roadmap (see `OPTIMIZATION_ROADMAP.md`)
3. Implement high-priority optimizations
4. Re-benchmark after optimizations
