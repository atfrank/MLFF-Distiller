# Student Model Performance Baseline Report

**Date**: 2025-11-24 08:22:30

**Device**: cuda
**Model**: checkpoints/best_model.pt

---

## Single Inference Performance

- **Mean time**: 38.22 ± 0.92 ms
- **Median time**: 38.00 ms
- **95th percentile**: 39.75 ms
- **Throughput**: 26.16 structures/second

### Performance by System Size

| Atoms | Mean (ms) | Std (ms) | Samples |
|-------|-----------|----------|----------|
| 3 | 37.07 | 0.00 | 1 |
| 4 | 37.47 | 0.00 | 1 |
| 5 | 38.24 | 0.00 | 1 |
| 8 | 40.58 | 0.00 | 1 |
| 11 | 38.25 | 0.50 | 2 |
| 12 | 38.09 | 0.34 | 4 |

## Batch Inference Performance

| Batch Size | Total Time (ms) | Time/Structure (ms) | Throughput (struct/s) | Speedup vs Batch=1 |
|------------|------------------|---------------------|------------------------|--------------------|
| 1 | 0.79 | 0.79 | 1259.80 | 1.00x |
| 2 | 38.53 | 19.26 | 51.91 | 0.04x |
| 4 | 42.55 | 10.64 | 94.01 | 0.07x |

## Memory Usage

- **Baseline**: 65.63 MB
- **Peak**: 69.52 MB
- **Inference overhead**: 3.89 MB

## Scaling with System Size

| Atoms | Category | Time (ms) | ms/atom |
|-------|----------|-----------|----------|
| 3 | small | 8.22 ± 14.82 | 2.739 |
| 4 | small | 8.15 ± 14.68 | 2.038 |
| 5 | small | 8.13 ± 14.69 | 1.626 |
| 11 | medium | 8.76 ± 15.93 | 0.796 |
| 12 | medium | 8.14 ± 14.69 | 0.678 |
| 12 | medium | 8.29 ± 14.98 | 0.691 |
| 60 | large | 10.06 ± 18.50 | 0.168 |

---

## Key Findings

- Current inference speed: **38.22 ms/structure**
- Current throughput: **26.16 structures/second**
- Memory footprint: **69.52 MB**

## Next Steps

1. Analyze profiling results to identify bottlenecks
2. Review optimization roadmap (see `OPTIMIZATION_ROADMAP.md`)
3. Implement high-priority optimizations
4. Re-benchmark after optimizations
