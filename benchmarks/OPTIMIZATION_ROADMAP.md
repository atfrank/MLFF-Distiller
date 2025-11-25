# Optimization Roadmap for Student Force Field Model

**Date**: 2025-11-24
**Baseline Performance**: 22.32 ms/structure (44.80 struct/sec)
**Target Performance**: 10x speedup vs Orb-v2 teacher model
**Device**: NVIDIA GeForce RTX 3080 Ti (CUDA)

---

## Executive Summary

The student model currently achieves **22.32 ms/structure** with **427K parameters** on CUDA. Based on profiling and analysis, we have identified multiple optimization opportunities that can deliver the targeted 10x speedup over the teacher model.

### Current Performance Metrics

| Metric | Value |
|--------|-------|
| Mean inference time | 22.32 ± 0.85 ms |
| Throughput | 44.80 structures/second |
| Memory footprint | 21.77 MB (peak) |
| Model parameters | 427,292 |
| Batch efficiency | Poor (only 0.036x speedup for batch=4) |

### Key Observations

1. **Batch processing is broken**: Batch size 4 should be 4x faster per structure, but it's actually 28x slower than single inference
2. **Model is already small**: 427K parameters is excellent (vs 100M+ for teacher)
3. **Memory usage is low**: Only 3.9 MB inference overhead
4. **Scaling is reasonable**: ~0.126 ms/atom for large systems (60 atoms)

---

## Critical Issue: Batch Processing Bug

**HIGHEST PRIORITY - MUST FIX FIRST**

### Problem
- Batch size 1: 0.79 ms/structure (1258 struct/sec)
- Batch size 2: 21.82 ms/structure (46 struct/sec) - **28x SLOWER**
- Batch size 4: 21.90 ms/structure (46 struct/sec) - **28x SLOWER**

This indicates the `calculate_batch()` method is not properly batching operations and may be processing sequentially or has severe overhead.

### Impact
Fixing batch processing alone could provide **10-20x speedup** for batch workflows.

### Action Items
1. **Investigate `calculate_batch()` implementation** (if it exists)
2. **Implement proper batch inference**:
   - Batch all operations: embedding, message passing, readout
   - Use padded tensors for variable-size molecules
   - Minimize Python loops
   - Ensure all GPU operations are batched
3. **Test and benchmark**: Verify batch=16 achieves ~10-15x throughput improvement

**Estimated Time**: 1-2 days
**Expected Speedup**: 10-20x for batch workflows
**Priority**: CRITICAL

---

## Optimization Categories

We've organized optimizations into 4 tiers based on effort vs. impact:

1. **Quick Wins** (1-3 days, 1.5-3x total speedup)
2. **Medium Effort** (1-2 weeks, 2-3x additional speedup)
3. **Major Work** (2-4 weeks, 2-4x additional speedup)
4. **Research Projects** (1-2 months, 1.5-2x additional speedup)

---

## Tier 1: Quick Wins (1-3 days)

### 1.1 Fix Batch Processing
**Status**: CRITICAL BUG
**Effort**: 1-2 days
**Expected Speedup**: 10-20x for batches
**Priority**: P0

**Tasks**:
- Debug current `calculate_batch()` implementation
- Implement proper tensor batching
- Add padding/masking for variable sizes
- Benchmark batch sizes 1, 4, 8, 16, 32

### 1.2 Enable Torch Compile
**Status**: Not implemented
**Effort**: 2-4 hours
**Expected Speedup**: 1.3-1.8x
**Priority**: P1

PyTorch 2.x `torch.compile()` can optimize the computation graph.

**Tasks**:
```python
# Add to model initialization
self.model = torch.compile(
    self.model,
    mode='max-autotune',  # or 'reduce-overhead'
    fullgraph=True
)
```

- Test compilation modes: `default`, `reduce-overhead`, `max-autotune`
- Benchmark compilation overhead vs. speedup
- Update calculator to support compiled models

### 1.3 Mixed Precision (FP16)
**Status**: Not implemented
**Effort**: 4-8 hours
**Expected Speedup**: 1.5-2x
**Priority**: P1

Use automatic mixed precision for faster CUDA kernels.

**Tasks**:
```python
# Add to calculator
with torch.cuda.amp.autocast():
    energy, forces = model.predict_energy_and_forces(...)
```

- Implement AMP in calculator
- Test numerical stability
- Benchmark FP16 vs FP32
- Validate accuracy is maintained (>95% agreement)

### 1.4 Optimize Neighbor Search
**Status**: Using native PyTorch implementation
**Effort**: 1 day
**Expected Speedup**: 1.2-1.5x
**Priority**: P2

Current `radius_graph_native()` computes full N×N distance matrix.

**Tasks**:
- Profile neighbor search overhead
- Implement cell list algorithm for O(N) complexity
- Or integrate with optimized library (torch_geometric, torch_cluster)
- Benchmark on large systems (100+ atoms)

**Tier 1 Total Expected Speedup**: **1.8-2.5x** (excluding batch fix)

---

## Tier 2: Medium Effort (1-2 weeks)

### 2.1 TensorRT Inference Engine
**Status**: Not implemented
**Effort**: 1-2 weeks
**Expected Speedup**: 2-3x
**Priority**: P1

NVIDIA TensorRT optimizes models for specific GPU architectures.

**Tasks**:
- Export model to ONNX format
- Convert ONNX to TensorRT engine
- Create TensorRT inference wrapper
- Handle dynamic input sizes (molecule sizes vary)
- Benchmark TensorRT vs. native PyTorch
- Test numerical accuracy

**Resources**:
- [TensorRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/)
- [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)

### 2.2 Custom CUDA Kernels for Message Passing
**Status**: Not implemented
**Effort**: 1-2 weeks
**Expected Speedup**: 1.5-2x
**Priority**: P2

Fuse multiple operations into single CUDA kernels.

**Tasks**:
- Identify fusion opportunities:
  - RBF computation + edge features
  - Message aggregation + update
  - Force scatter operations
- Implement custom CUDA kernels using:
  - PyTorch C++ extensions
  - Or CuPy for prototyping
- Profile kernel performance
- Validate numerical correctness

**Example Fusions**:
```
# Instead of:
rbf = gaussian_rbf(distances)
edge_features = mlp(rbf)
messages = scatter_add(edge_features)

# Fused kernel:
messages = fused_message_computation(positions, edge_index)
```

### 2.3 Graph Neural Network Optimizations
**Status**: Using manual implementations
**Effort**: 1 week
**Expected Speedup**: 1.3-1.8x
**Priority**: P2

Optimize graph operations using specialized libraries.

**Tasks**:
- Replace manual message passing with torch_geometric ops
- Use optimized scatter/gather operations
- Implement sparse tensor operations where applicable
- Profile and compare to baseline

### 2.4 Gradient Checkpointing
**Status**: Not needed (inference only)
**Effort**: N/A
**Note**: Only relevant for training, skip for inference

**Tier 2 Total Expected Speedup**: **2-3x** additional

---

## Tier 3: Major Work (2-4 weeks)

### 3.1 Model Architecture Optimizations
**Status**: PaiNN baseline
**Effort**: 2-3 weeks
**Expected Speedup**: 1.5-2.5x
**Priority**: P3

Explore architectural changes that reduce computation.

**Tasks**:
- **Reduce message passing layers**: Test 2 layers vs. 3
  - May sacrifice 1-2% accuracy for 30-40% speedup
- **Efficient attention mechanisms**: If using attention
  - Linear attention variants
  - Sparse attention patterns
- **Parameter sharing**: Share weights across interaction blocks
- **Pruning**: Remove low-importance weights
  - Magnitude pruning
  - Structured pruning (entire channels)

### 3.2 Quantization (INT8)
**Status**: Not implemented
**Effort**: 2-3 weeks
**Expected Speedup**: 1.5-2x
**Priority**: P3

Quantize model weights and activations to INT8.

**Tasks**:
- Post-training quantization (PTQ)
- Quantization-aware training (QAT) if needed
- Profile INT8 vs FP16 performance
- Validate accuracy retention
- Test on multiple hardware (RTX 3080 Ti has INT8 Tensor Cores)

**Resources**:
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
- [TensorRT INT8 Calibration](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#working-with-int8)

### 3.3 Asynchronous Execution
**Status**: Not implemented
**Effort**: 1-2 weeks
**Expected Speedup**: 1.3-1.5x (for batch workloads)
**Priority**: P3

Overlap CPU and GPU operations using CUDA streams.

**Tasks**:
- Create multiple CUDA streams
- Pipeline data transfers and computation
- Async copy of inputs to GPU
- Double-buffering for batch processing
- Profile stream utilization

### 3.4 Persistent Kernel Optimization
**Status**: Not implemented
**Effort**: 2-3 weeks
**Expected Speedup**: 1.2-1.5x
**Priority**: P4

Reduce kernel launch overhead for small molecules.

**Tasks**:
- Implement persistent CUDA kernels
- Minimize host-device synchronization
- Use CUDA graphs for fixed-size molecules
- Profile kernel launch overhead

**Tier 3 Total Expected Speedup**: **2-3x** additional

---

## Tier 4: Research Projects (1-2 months)

### 4.1 Knowledge Distillation Refinement
**Status**: Initial distillation complete
**Effort**: 3-4 weeks
**Expected Speedup**: 1.2-1.5x (via smaller model)
**Priority**: P4

Further compress the model via advanced distillation.

**Tasks**:
- Progressive distillation (multi-stage)
- Feature matching losses
- Attention transfer
- Train even smaller model (200K parameters)

### 4.2 Hardware-Specific Optimization
**Status**: Generic CUDA implementation
**Effort**: 3-4 weeks
**Expected Speedup**: 1.3-1.8x
**Priority**: P4

Optimize for specific GPU architectures.

**Tasks**:
- Profile on target hardware (RTX 3080 Ti has Ampere architecture)
- Utilize Tensor Cores efficiently
- Optimize for memory bandwidth
- Use architecture-specific instructions
- Consider CPU-optimized version for deployment

### 4.3 Graph Coarsening
**Status**: Not implemented
**Effort**: 4-6 weeks
**Expected Speedup**: 1.5-2x (for large systems)
**Priority**: P4

Reduce graph size for large molecules.

**Tasks**:
- Implement graph coarsening algorithms
- Hierarchical graph representations
- Test on large systems (100+ atoms)
- Validate accuracy preservation

### 4.4 Approximate Methods
**Status**: Exact computations
**Effort**: 2-3 weeks
**Expected Speedup**: 1.2-1.5x
**Priority**: P5

Trade minimal accuracy for speed.

**Tasks**:
- Approximate RBF with lookup tables
- Fast approximate exp/sqrt operations
- Reduced precision for intermediate computations
- Validate accuracy impact is acceptable

**Tier 4 Total Expected Speedup**: **1.5-2x** additional

---

## Cumulative Speedup Projection

Assuming conservative estimates and diminishing returns:

| Optimization Stage | Incremental Speedup | Cumulative Speedup |
|-------------------|--------------------|--------------------|
| **Baseline** | 1.0x | **1.0x** |
| Fix batch processing | 1.0x (single) | 1.0x (single inference) |
| Tier 1 (Quick Wins) | 2.0x | **2.0x** |
| Tier 2 (Medium) | 2.5x | **5.0x** |
| Tier 3 (Major) | 2.0x | **10.0x** |
| Tier 4 (Research) | 1.5x | **15.0x** |

**Note**: Batch processing fix gives 10-20x for batch workflows separately.

**Target Achievement**: Tier 1 + Tier 2 should achieve the **10x target** speedup.

---

## Implementation Priority Order

### Phase 1: Critical Fixes (Week 1)
1. **Fix batch processing** - P0
2. **Enable torch.compile()** - P1
3. **Implement FP16** - P1

**Expected**: 2-3x speedup, batch processing fixed

### Phase 2: Infrastructure (Weeks 2-3)
4. **Optimize neighbor search** - P2
5. **TensorRT integration** - P1
6. **Custom CUDA kernels** - P2

**Expected**: 5-8x cumulative speedup

### Phase 3: Advanced (Weeks 4-6)
7. **Architecture optimizations** - P3
8. **Quantization (INT8)** - P3
9. **Async execution** - P3

**Expected**: 10-15x cumulative speedup

### Phase 4: Polish (Weeks 7-10, optional)
10. **Knowledge distillation refinement** - P4
11. **Hardware-specific tuning** - P4

**Expected**: 15-20x cumulative speedup

---

## Benchmarking Protocol

After each optimization, run the benchmark suite:

```bash
# Full benchmark
python scripts/benchmark_inference.py --device cuda --output benchmarks/

# Quick check
python scripts/benchmark_inference.py --device cuda --quick --output benchmarks/

# Compare to baseline
python scripts/compare_benchmarks.py \
    benchmarks/baseline_performance.json \
    benchmarks/optimized_performance.json
```

### Key Metrics to Track
1. Single inference time (ms/structure)
2. Batch throughput (structures/sec)
3. Memory usage (MB)
4. Accuracy retention (MAE vs. teacher)
5. Compilation/loading time

---

## Risk Mitigation

### Accuracy Validation
After each optimization, validate:
- Energy MAE < 10 meV/atom vs. baseline
- Force RMSE < 50 meV/Å vs. baseline
- Stress RMSE < 0.1 GPa vs. baseline (if applicable)

Use validation script:
```bash
python scripts/validate_model_detailed.py \
    --checkpoint checkpoints/optimized_model.pt \
    --compare-to checkpoints/best_model.pt
```

### Performance Regression
- Maintain benchmark history
- Automated performance tests in CI/CD
- Alert on >5% regression

### Reproducibility
- Version all optimization code
- Document hyperparameters
- Save optimized model checkpoints
- Track environment (PyTorch version, CUDA version, etc.)

---

## Success Criteria

### Minimum Viable Product (MVP)
- [ ] Single inference: < 5 ms (current: 22.32 ms) - **4.5x speedup**
- [ ] Batch throughput: > 500 struct/sec (current: ~45) - **11x speedup**
- [ ] Accuracy: > 95% vs. baseline student model
- [ ] Memory: < 50 MB peak

### Stretch Goals
- [ ] Single inference: < 2 ms - **11x speedup**
- [ ] Batch throughput: > 1000 struct/sec - **22x speedup**
- [ ] 10x faster than Orb-v2 teacher model
- [ ] Sub-millisecond inference for small molecules (< 10 atoms)

---

## Resource Requirements

### Hardware
- NVIDIA GPU with Compute Capability 7.0+ (Volta or newer)
- Current: RTX 3080 Ti (Ampere, CC 8.6) ✓
- 8+ GB GPU memory (have plenty) ✓

### Software
- PyTorch 2.0+ with CUDA support ✓
- TensorRT 8.5+ (need to install)
- torch_geometric (optional, for optimized GNN ops)
- CuPy (optional, for custom CUDA kernels)

### Time Estimates
- **Phase 1**: 1 week (40 hours)
- **Phase 2**: 2 weeks (80 hours)
- **Phase 3**: 3 weeks (120 hours)
- **Phase 4**: 4 weeks (160 hours, optional)

**Total**: 6-10 weeks for full roadmap

---

## Next Immediate Actions

1. **Create GitHub Issues** for each optimization task
   - Label with priority (P0-P5)
   - Assign to optimization engineer(s)
   - Add to project board

2. **Fix Batch Processing** (CRITICAL)
   - Debug current implementation
   - Rewrite if necessary
   - Validate with tests

3. **Quick Win Sprint** (Week 1)
   - Implement torch.compile()
   - Implement FP16
   - Optimize neighbor search
   - Re-benchmark

4. **Establish CI/CD**
   - Automated performance tests
   - Accuracy validation
   - Benchmark comparison reports

5. **Documentation**
   - Document each optimization
   - Create before/after comparisons
   - Publish optimization guide

---

## Appendix: Profiling Notes

### Current Bottlenecks (From Profiling)
The profiling results show minimal detail because most computation is happening in compiled CUDA kernels. Key observations:

1. **Minimal CPU overhead**: Only 27μs in CPU operations
2. **GPU-bound**: Almost all time in CUDA operations
3. **Need deeper profiling**: Use NVIDIA Nsight Systems for detailed GPU profiling

### Recommended Profiling Tools
- **PyTorch Profiler**: Basic CPU/GPU breakdown (already used)
- **NVIDIA Nsight Systems**: Detailed timeline of GPU operations
- **NVIDIA Nsight Compute**: Kernel-level profiling
- **torch.profiler with Chrome Trace**: Visual timeline

### Next Profiling Steps
```bash
# Detailed GPU profiling
nsys profile -o baseline_profile python scripts/benchmark_inference.py --quick

# Kernel-level analysis
ncu --set full -o kernel_profile python scripts/benchmark_inference.py --quick
```

---

## Contact & Support

For questions or issues:
- **GitHub**: Open issue in ml-forcefield-distillation repo
- **Coordinator**: ml-distillation-coordinator agent
- **CUDA Specialist**: Agent 4 (CUDA Optimization Engineer)

Last Updated: 2025-11-24
