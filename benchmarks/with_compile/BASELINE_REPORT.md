# Student Model Performance Baseline Report

**Date**: 2025-11-24 08:22:41

**Device**: cuda
**Model**: checkpoints/best_model.pt

---

## Single Inference Performance

- **Mean time**: 37.67 ± 0.53 ms
- **Median time**: 37.94 ms
- **95th percentile**: 38.18 ms
- **Throughput**: 26.55 structures/second

### Performance by System Size

| Atoms | Mean (ms) | Std (ms) | Samples |
|-------|-----------|----------|----------|
| 3 | 36.77 | 0.00 | 1 |
| 4 | 37.36 | 0.00 | 1 |
| 5 | 38.07 | 0.00 | 1 |
| 8 | 38.11 | 0.00 | 1 |
| 11 | 37.73 | 0.33 | 2 |
| 12 | 37.73 | 0.56 | 4 |

## Batch Inference Performance

| Batch Size | Total Time (ms) | Time/Structure (ms) | Throughput (struct/s) | Speedup vs Batch=1 |
|------------|------------------|---------------------|------------------------|--------------------|
| 1 | 0.81 | 0.81 | 1228.87 | 1.00x |
| 2 | 37.95 | 18.97 | 52.70 | 0.04x |
| 4 | 41.82 | 10.46 | 95.64 | 0.08x |

## Memory Usage

- **Baseline**: 65.63 MB
- **Peak**: 69.52 MB
- **Inference overhead**: 3.89 MB

## Scaling with System Size

| Atoms | Category | Time (ms) | ms/atom |
|-------|----------|-----------|----------|
| 3 | small | 9.15 ± 16.40 | 3.049 |
| 4 | small | 8.84 ± 15.80 | 2.211 |
| 5 | small | 9.21 ± 16.57 | 1.841 |
| 11 | medium | 9.30 ± 16.54 | 0.846 |
| 12 | medium | 8.99 ± 16.09 | 0.749 |
| 12 | medium | 8.95 ± 16.05 | 0.746 |
| 60 | large | 11.58 ± 21.23 | 0.193 |

---

## Key Findings

- Current inference speed: **37.67 ms/structure**
- Current throughput: **26.55 structures/second**
- Memory footprint: **69.52 MB**

## Next Steps

1. Analyze profiling results to identify bottlenecks
2. Review optimization roadmap (see `OPTIMIZATION_ROADMAP.md`)
3. Implement high-priority optimizations
4. Re-benchmark after optimizations
