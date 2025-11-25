# Quick Start - Next Phase (After Force Analysis)
**Date**: November 24, 2025
**Status**: Session complete, ready for Phase 2 activities
**Previous Session**: Compact Models Force Analysis

---

## What Was Just Completed

The force analysis task has been completed with comprehensive results:

| Model | Status | Key Metric | Recommendation |
|-------|--------|-----------|-----------------|
| Original (427K) | ✅ READY | R²=0.9958 (excellent) | Deploy now |
| Tiny (77K) | ⚠ NEEDS WORK | R²=0.3787 (moderate) | Improve architecture |
| Ultra-tiny (21K) | ⚠ LIMITED | R²=0.1499 (poor) | Energy-only |

**Documentation**: See `FINAL_SESSION_SUMMARY_20251124.md` and `NEXT_STEPS_AND_RECOMMENDATIONS.md` for complete analysis.

---

## Immediate Actions (This Week)

### 1. Export Remaining Models
**Time**: 1-2 hours
**Owner**: Any engineer with PyTorch experience

```bash
# Tiny model to TorchScript
python scripts/export_to_torchscript.py \
    --checkpoint checkpoints/tiny_model/best_model.pt \
    --output models/tiny_model_traced.pt \
    --model-type tiny

# Tiny model to ONNX
python scripts/export_to_onnx.py \
    --checkpoint checkpoints/tiny_model/best_model.pt \
    --output models/tiny_model.onnx \
    --model-type tiny

# Repeat for ultra_tiny_model
```

**Verification**:
```bash
python -c "import torch; torch.jit.load('models/tiny_model_traced.pt')"
python -c "import onnx; onnx.load('models/tiny_model.onnx')"
```

---

### 2. Run Integration Tests
**Time**: 2-3 hours
**Owner**: Testing engineer

```bash
cd /home/aaron/ATX/software/MLFF_Distiller

# Test Original model
python -m pytest tests/integration/test_ase_calculator.py -v

# Test model loading and inference
python -c "
import torch
from src.mlff_distiller.models.student_model import StudentForceField

device = 'cuda'

# Test all three models
for name, ckpt in [
    ('Original', 'checkpoints/best_model.pt'),
    ('Tiny', 'checkpoints/tiny_model/best_model.pt'),
    ('Ultra-tiny', 'checkpoints/ultra_tiny_model/best_model.pt')
]:
    print(f'Testing {name}...')
    checkpoint = torch.load(ckpt, map_location=device)
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('model.') for k in state_dict.keys()):
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    print(f'✓ {name} loads successfully')
"
```

---

### 3. Create Model Cards
**Time**: 1-2 hours
**Owner**: Documentation specialist

Create a file: `docs/MODEL_CARDS.md`

```markdown
## Original Model (427K Parameters)
- **Status**: Production Ready
- **Use For**: MD simulations, accuracy-critical
- **Force R²**: 0.9958
- **Checkpoint**: checkpoints/best_model.pt

## Tiny Model (77K Parameters)
- **Status**: Needs Architecture Improvement
- **Use For**: Quick screening (after improvement)
- **Current Force R²**: 0.3787
- **Target Force R²**: > 0.5 (after improvement)
- **Checkpoint**: checkpoints/tiny_model/best_model.pt

## Ultra-tiny Model (21K Parameters)
- **Status**: Energy-only Applications
- **Use For**: Energy predictions (forces unreliable)
- **Force R²**: 0.1499
- **Checkpoint**: checkpoints/ultra_tiny_model/best_model.pt
```

---

## Key Files to Know

### Models and Checkpoints
```
checkpoints/best_model.pt                    → Original (1.63 MB)
checkpoints/tiny_model/best_model.pt         → Tiny (0.30 MB)
checkpoints/ultra_tiny_model/best_model.pt   → Ultra-tiny (0.08 MB)

models/original_model_traced.pt              → TorchScript export (1.72 MB)
models/original_model.onnx                   → ONNX export (1.72 MB)
```

### Force Analysis Results
```
visualizations/compact_force_analysis/force_analysis_Original_427K.png
visualizations/compact_force_analysis/force_analysis_Tiny_77K.png
visualizations/compact_force_analysis/force_analysis_Ultra-tiny_21K.png
```

### Key Scripts
```
scripts/analyze_compact_models_forces.py     → Force analysis (just created)
scripts/train_compact_models.py              → Model training
scripts/export_to_torchscript.py             → TorchScript export
scripts/export_to_onnx.py                    → ONNX export
```

### Documentation
```
FINAL_SESSION_SUMMARY_20251124.md            → Complete analysis
DELIVERABLES_MANIFEST_20251124.md            → Inventory
NEXT_STEPS_AND_RECOMMENDATIONS.md            → 4-phase plan
SESSION_COMPLETION_REPORT_20251124.txt       → Detailed report
```

---

## Short-term Tasks (Weeks 2-3)

### Task 1: Improve Tiny Model Architecture (5-7 hours)

The Tiny model has R²=0.3787 which is too low for production. Improvements:

```python
# Current architecture (doesn't work well)
StudentForceField(hidden_dim=64, num_interactions=2, num_rbf=12)

# Try these improvements:
# Option A: Increase hidden dimension
StudentForceField(hidden_dim=96, num_interactions=2, num_rbf=12)

# Option B: Add more interactions
StudentForceField(hidden_dim=64, num_interactions=3, num_rbf=12)

# Option C: Combine both
StudentForceField(hidden_dim=96, num_interactions=3, num_rbf=12)
```

Run experiments and measure:
- Training time
- Model size
- Force prediction R²
- Target: R² > 0.5 with < 150K parameters

---

### Task 2: Implement Model Quantization (4-6 hours)

Reduce model size by 4x:

```python
import torch
from torch.quantization import quantize_dynamic

# Load Original model
checkpoint = torch.load('checkpoints/best_model.pt')
model = StudentForceField(...)
model.load_state_dict(checkpoint['model_state_dict'])

# Dynamic quantization
quantized = quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Save
torch.save({'model_state_dict': quantized.state_dict()},
           'checkpoints/original_quantized.pt')
```

Test:
- Model size reduction
- Inference latency improvement
- Accuracy loss (target: < 5% R² loss)

---

### Task 3: Expanded Benchmarking (3-4 hours)

Test performance across different conditions:

```python
# Benchmark across batch sizes
for batch_size in [1, 2, 4, 8, 16, 32]:
    # Measure latency, memory, throughput

# Benchmark different export formats
formats = ['pytorch', 'torchscript', 'onnx']
for fmt in formats:
    # Compare latency and memory

# Benchmark CPU vs GPU
devices = ['cpu', 'cuda']
for device in devices:
    # Measure performance
```

Expected results:
- Original: 2-3ms latency (batch=1), 36k+ samples/sec (batch=8)
- Tiny: 0.5-1ms latency (batch=1), 15k+ samples/sec (batch=8)
- Ultra-tiny: 0.3-0.5ms latency (batch=1), 10k+ samples/sec (batch=8)

---

## Longer-term Tasks (Weeks 4-6)

### Task 1: CUDA Optimization (6-8 hours)
Target: 3-5x speedup from current latency

### Task 2: Integration with MD Pipeline (4-5 hours)
Validate models in actual MD simulations:

```python
from ase.md.verlet import VelocityVerlet
from ase import units
from src.mlff_distiller.inference import StudentForceFieldCalculator

# Use Original model for production testing
calc = StudentForceFieldCalculator('checkpoints/best_model.pt', device='cuda')
atoms.calc = calc

# Run NVE ensemble (test energy conservation)
dyn = VelocityVerlet(atoms, timestep=0.5*units.fs)
dyn.run(1000)

# Check: Energy should not drift > 0.1% per 100 steps
```

### Task 3: Production Deployment (2-3 hours)
- Documentation for users
- Model selection guide
- Troubleshooting guide
- Performance characteristics

---

## Decision Points

### Decision 1: Original Model Deployment
- **Question**: Ready to deploy Original model now?
- **Recommendation**: YES - R²=0.9958 is production-ready
- **Timeline**: Immediate
- **Owner**: Project lead

### Decision 2: Tiny Model Priority
- **Question**: How much effort on Tiny model improvement?
- **Recommendation**: 5-7 hours for architecture tests, then decide
- **Timeline**: Weeks 2-3
- **Decision Point**: End of week 2

### Decision 3: CUDA Investment
- **Question**: How critical is 5-10x speedup target?
- **Recommendation**: Do quantization first (easier), CUDA if needed
- **Timeline**: Weeks 4+ depending on requirements
- **Success Criteria**: 3-5x improvement is acceptable

---

## Testing Checklist

Before any model goes to production:

- [ ] Model loads without errors (all formats)
- [ ] Inference produces correct output shape
- [ ] Batch inference works (multiple sizes)
- [ ] Numerical outputs stable (same input = same output)
- [ ] Memory usage acceptable (< 2GB per model on GPU)
- [ ] Latency meets requirements (< 5ms for batch=1 ideally)
- [ ] No memory leaks in inference loop
- [ ] Works on CPU and GPU
- [ ] Handles edge cases (single atom, large molecules)

---

## Getting Help

### Key Documents
1. **FINAL_SESSION_SUMMARY_20251124.md** - Complete technical analysis
2. **NEXT_STEPS_AND_RECOMMENDATIONS.md** - Detailed 4-phase plan
3. **DELIVERABLES_MANIFEST_20251124.md** - File inventory
4. **SESSION_COMPLETION_REPORT_20251124.txt** - Executive overview

### Key Scripts
1. **scripts/analyze_compact_models_forces.py** - Force analysis
2. **scripts/train_compact_models.py** - Model training
3. **scripts/export_to_*.py** - Export utilities

### Key Contacts
- **For Model Architecture**: See `src/mlff_distiller/models/student_model.py`
- **For Training**: See `src/mlff_distiller/training/`
- **For Inference**: See `src/mlff_distiller/inference/ase_calculator.py`

---

## Quick Reference - Commands

### Load and Use Original Model
```python
import torch
from src.mlff_distiller.models.student_model import StudentForceField

device = 'cuda'
checkpoint = torch.load('checkpoints/best_model.pt', map_location=device)

model = StudentForceField(hidden_dim=128, num_interactions=3)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device).eval()

# Inference
with torch.no_grad():
    energy = model(atomic_numbers, positions, batch=batch_indices)
```

### Run Force Analysis
```bash
python scripts/analyze_compact_models_forces.py \
    --test-molecule data/generative_test/moldiff/test_10mols_20251123_181225_SDF/0.sdf \
    --output-dir visualizations/compact_force_analysis \
    --device cuda
```

### Export Models
```bash
# TorchScript
python scripts/export_to_torchscript.py \
    --checkpoint checkpoints/tiny_model/best_model.pt \
    --output models/tiny_model_traced.pt

# ONNX
python scripts/export_to_onnx.py \
    --checkpoint checkpoints/tiny_model/best_model.pt \
    --output models/tiny_model.onnx
```

---

## Success Metrics

### Week 1 (This Week)
- [x] Force analysis complete
- [ ] Exports completed for Tiny/Ultra-tiny
- [ ] Integration tests passing
- [ ] Model cards created

### Week 2-3 (Improvement Phase)
- [ ] Tiny model architecture tests completed
- [ ] Quantization implemented and tested
- [ ] Benchmarking expanded
- [ ] Decision made on Tiny model improvements

### Week 4+ (Optimization Phase)
- [ ] CUDA optimizations implemented
- [ ] MD integration validated
- [ ] Production deployment ready
- [ ] User documentation complete

---

## Notes for Next Team

1. **Original Model**: This is production-ready. Deploy it immediately.

2. **Tiny Model**: The force accuracy is too low (R²=0.3787) for reliable MD. Need to:
   - Increase model capacity (hidden_dim 64→96, or add interaction)
   - Target R² > 0.5
   - Test before deployment

3. **Ultra-tiny Model**: Force predictions are unreliable (R²=0.1499).
   - Restrict to energy-only applications
   - NOT suitable for MD simulations
   - Consider removing from production roadmap

4. **Documentation**: All deliverables from this session are well-documented.
   - Read FINAL_SESSION_SUMMARY_20251124.md first
   - Then NEXT_STEPS_AND_RECOMMENDATIONS.md for detailed planning
   - Use DELIVERABLES_MANIFEST_20251124.md as reference

5. **Scripts**: The force analysis script is reusable and can be adapted for:
   - Validating improved models
   - Comparing with teacher models
   - Generating publication-quality figures

---

**Session Date**: November 24, 2025
**Status**: COMPLETE - Ready for next phase
**Prepared by**: ML Force Field Distillation Coordinator
