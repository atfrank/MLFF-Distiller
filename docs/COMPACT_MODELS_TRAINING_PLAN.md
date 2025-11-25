# Compact Student Models Training Plan

**Date**: 2025-11-24
**Status**: Ready for Execution
**Coordinator**: Lead Project Coordinator

---

## Executive Summary

We are creating two compact student models (Tiny and Ultra-tiny) via **progressive distillation** from the current student model (427K params). This provides better training stability and accuracy than distilling directly from Orb.

**Progressive Distillation Chain**:
```
Orb teacher (187M params)
    ↓ (direct distillation - COMPLETED)
Current student (427K params, 1.63 MB)
    ↓ (progressive distillation - THIS TASK)
Tiny (78K params, 0.30 MB) + Ultra-tiny (22K params, 0.08 MB)
```

---

## Problem Identified

The user attempted to train these models but made a critical mistake:
- **Used randomly generated data** instead of the actual training dataset
- **Result**: NaN losses and unstable training

**Root cause**: The training script (`train_compact_models_quick.py`) created random molecular structures on-the-fly rather than using our prepared HDF5 dataset with teacher labels.

---

## Correct Approach

### 1. Dataset Selection

**USE THIS DATASET**:
- **Path**: `/home/aaron/ATX/software/MLFF_Distiller/data/merged_dataset_4883/merged_dataset.h5`
- **Format**: HDF5 with structures + Orb teacher labels
- **Size**: 4,883 molecular structures
- **Contents**:
  - `/structures/`: atomic_numbers, positions, cells, pbc
  - `/labels/`: energy, forces (from Orb teacher)
  - Split indices for variable-sized molecules

**DO NOT**:
- Generate random data
- Use separate structure/label files (they're already merged)
- Create new datasets

### 2. Progressive Distillation Strategy

**Why progressive distillation?**
- The current student (427K) has already learned to compress Orb knowledge
- Training tiny models from the student is more stable than from Orb directly
- Reduces capacity gap: 187M → 427K → 78K is easier than 187M → 78K

**Implementation**:
```python
# Load current student as teacher
teacher = StudentForceField(hidden_dim=128, num_interactions=3, num_rbf=20)
teacher.load_state_dict(torch.load('checkpoints/best_model.pt'))
teacher.eval()

# Train tiny student from this teacher
student_tiny = StudentForceField(hidden_dim=64, num_interactions=2, num_rbf=12)

# Distillation loss: match teacher outputs on same inputs
energy_loss = MSE(student_energy, teacher_energy)
force_loss = MSE(student_forces, teacher_forces)
total_loss = energy_weight * energy_loss + force_weight * force_loss
```

### 3. Model Architectures

**Tiny Model** (18% of current):
```python
{
    'hidden_dim': 64,
    'num_interactions': 2,
    'num_rbf': 12,
    'cutoff': 5.0,
    'max_z': 100,
}
# Expected: 78K params, 0.30 MB
# Target: 90-94% accuracy of current student, 2x faster
```

**Ultra-tiny Model** (5% of current):
```python
{
    'hidden_dim': 32,
    'num_interactions': 2,
    'num_rbf': 10,
    'cutoff': 5.0,
    'max_z': 100,
}
# Expected: 22K params, 0.08 MB
# Target: 80-88% accuracy of current student, 3x faster
```

---

## Training Configuration

Based on successful training of the current student model:

### Hyperparameters

**For Tiny (78K params)**:
- Epochs: 50
- Batch size: 16
- Learning rate: 5e-4 (lower than current student due to smaller size)
- Optimizer: AdamW (weight_decay=1e-5)
- Scheduler: CosineAnnealingLR
- Energy weight: 1.0
- Force weight: 15.0 (higher than current to emphasize force accuracy)
- Gradient clip: 1.0

**For Ultra-tiny (22K params)**:
- Same as Tiny (architecture difference is enough)
- May benefit from longer training (70-100 epochs) if needed

### Loss Weights Rationale

**Force weight = 15.0** (vs 100.0 for current student):
- Compact models have less capacity
- Too high force weight can cause instability
- 15.0 balances energy and force learning

**Energy weight = 1.0**:
- Standard baseline
- Energy is per-structure, easier to learn

---

## Data Pipeline

### Dataset Structure

The HDF5 file uses flattened arrays with split indices:

```
/structures/
    atomic_numbers: (N_atoms_total,) int64
    atomic_numbers_splits: (N_structures+1,) int64
    positions: (N_atoms_total, 3) float64
    cells: (N_structures, 3, 3) float64
    pbc: (N_structures, 3) bool

/labels/
    energy: (N_structures,) float64
    forces: (N_atoms_total, 3) float32
    forces_splits: (N_structures+1,) int64
```

### DataLoader

Use our existing infrastructure:
```python
from mlff_distiller.data.distillation_dataset import (
    DistillationDataset, 
    distillation_collate_fn,
    create_train_val_dataloaders
)

train_loader, val_loader = create_train_val_dataloaders(
    hdf5_path='data/merged_dataset_4883/merged_dataset.h5',
    batch_size=16,
    train_ratio=0.9,
    num_workers=4,
    pin_memory=True,
    random_seed=42,
)
```

This handles:
- Train/val split (90/10)
- Variable-sized molecules
- Batching with proper collation
- GPU memory pinning

---

## Implementation Files

### New Training Script

**File**: `/home/aaron/ATX/software/MLFF_Distiller/scripts/train_compact_models.py`

**Status**: Created (replacing the faulty `train_compact_models_quick.py`)

**Key features**:
- Uses actual HDF5 dataset (not random data)
- Progressive distillation from current student
- Proper batched inference with force computation
- Saves checkpoints and training history
- Detailed logging

**Usage**:
```bash
# Train both models (default)
python scripts/train_compact_models.py

# Train only Tiny
python scripts/train_compact_models.py --models tiny

# Train only Ultra-tiny
python scripts/train_compact_models.py --models ultra_tiny

# Custom settings
python scripts/train_compact_models.py \
    --epochs 70 \
    --batch-size 32 \
    --lr 1e-3 \
    --force-weight 20.0
```

### Infrastructure Used

**Dataset loader**: 
- `src/mlff_distiller/data/distillation_dataset.py` (ALREADY EXISTS)

**Student model**: 
- `src/mlff_distiller/models/student_model.py` (ALREADY EXISTS)

**Teacher checkpoint**: 
- `checkpoints/best_model.pt` (ALREADY EXISTS, current student)

---

## Execution Plan

### Phase 1: Training Setup (5 min)

1. Verify all files exist:
   - Dataset: `data/merged_dataset_4883/merged_dataset.h5`
   - Teacher: `checkpoints/best_model.pt`
   - Training script: `scripts/train_compact_models.py`

2. Create output directories:
   - `checkpoints/compact_models/tiny/`
   - `checkpoints/compact_models/ultra_tiny/`

3. Test data loading:
   ```python
   from mlff_distiller.data.distillation_dataset import DistillationDataset
   ds = DistillationDataset('data/merged_dataset_4883/merged_dataset.h5')
   print(f"Dataset size: {len(ds)}")
   sample = ds[0]
   print(f"Sample: {sample['n_atoms']} atoms, energy={sample['energy']:.2f}")
   ```

### Phase 2: Training Execution (2-4 hours)

1. **Train Tiny model first** (50 epochs, ~1.5 hours):
   ```bash
   python scripts/train_compact_models.py --models tiny --epochs 50
   ```

2. **Monitor training**:
   - Watch logs in `logs/train_compact_models_*.log`
   - Check for NaN losses (should NOT occur with real data)
   - Validate loss curves (should decrease steadily)

3. **Train Ultra-tiny model** (50 epochs, ~1.5 hours):
   ```bash
   python scripts/train_compact_models.py --models ultra_tiny --epochs 50
   ```

### Phase 3: Validation (30 min)

1. **Load trained models**:
   ```python
   tiny_ckpt = torch.load('checkpoints/compact_models/tiny/best_model.pt')
   print(f"Tiny: {tiny_ckpt['total_params']:,} params, loss={tiny_ckpt['best_val_loss']:.4f}")
   
   ultra_ckpt = torch.load('checkpoints/compact_models/ultra_tiny/best_model.pt')
   print(f"Ultra-tiny: {ultra_ckpt['total_params']:,} params, loss={ultra_ckpt['best_val_loss']:.4f}")
   ```

2. **Compare to current student**:
   - Load current student checkpoint
   - Compare validation losses
   - Expected: Tiny ~1.5-2x current loss, Ultra-tiny ~2-3x current loss

3. **Run inference benchmarks**:
   ```bash
   python scripts/benchmark_compact_models.py
   ```
   - Measure inference time
   - Verify speedup targets (2x for Tiny, 3x for Ultra-tiny)

### Phase 4: Documentation (15 min)

1. Create training report:
   - Training curves
   - Final metrics
   - Comparison table

2. Update project status

---

## Expected Outcomes

### Tiny Model (78K params)

**Architecture**:
- Hidden dim: 64 (50% of current)
- Interactions: 2 (67% of current)
- RBF: 12 (60% of current)

**Expected performance**:
- Parameters: ~78K (18% of current 427K)
- Size: ~0.30 MB (18% of current 1.63 MB)
- Accuracy: 90-94% of current student
- Speedup: 2x faster than current student
- Speedup vs Orb: ~4.5x (2x × 2.34x)

**Use cases**:
- Edge devices
- Mobile applications
- High-throughput screening

### Ultra-tiny Model (22K params)

**Architecture**:
- Hidden dim: 32 (25% of current)
- Interactions: 2 (67% of current)
- RBF: 10 (50% of current)

**Expected performance**:
- Parameters: ~22K (5% of current 427K)
- Size: ~0.08 MB (5% of current 1.63 MB)
- Accuracy: 80-88% of current student
- Speedup: 3x faster than current student
- Speedup vs Orb: ~7x (3x × 2.34x)

**Use cases**:
- Embedded systems (ESP32, STM32)
- Pre-screening pipelines
- IoT devices
- Ultra-low-latency applications

---

## Validation Metrics

### Training Metrics

Monitor during training:
- **Energy loss**: MSE between student and teacher energies
- **Force loss**: MSE between student and teacher forces
- **Total loss**: Weighted sum (1.0 × energy + 15.0 × force)

**Success criteria**:
- No NaN losses (guaranteed with real data)
- Steady decrease over epochs
- Validation loss converges

### Accuracy Metrics

Compare to current student on test set:
- **Energy MAE**: Mean absolute error
- **Force RMSE**: Root mean squared error
- **Accuracy ratio**: (student_error / teacher_error)

**Target accuracy**:
- Tiny: 0.90-0.94 ratio (6-10% worse than current)
- Ultra-tiny: 0.80-0.88 ratio (12-20% worse than current)

### Performance Metrics

Benchmark on RTX 3080 Ti:
- **Inference time** (single molecule, median of 100 runs)
- **Throughput** (molecules/second, batch=16)
- **Memory usage** (GPU memory during inference)

**Target speedup**:
- Tiny: 2x faster than current student
- Ultra-tiny: 3x faster than current student

---

## Risk Mitigation

### Risk 1: NaN Losses

**Probability**: Low (using real data, not random)

**Mitigation**:
- Use proper dataset (HDF5 with teacher labels)
- Gradient clipping (1.0)
- Lower learning rate (5e-4 vs 1e-3)

### Risk 2: Poor Accuracy

**Probability**: Medium (compact models have limited capacity)

**Mitigation**:
- Progressive distillation (easier than direct)
- Higher force weight (15.0) to emphasize important features
- Longer training if needed (up to 100 epochs)
- Could try knowledge distillation techniques (temperature scaling)

### Risk 3: Insufficient Speedup

**Probability**: Low (smaller models are inherently faster)

**Mitigation**:
- Verify CUDA optimizations are enabled
- Benchmark with same infrastructure as current student
- Consider torch.compile for additional speedup

---

## Next Steps After Training

Once models are trained and validated:

1. **Export optimized versions**:
   - TorchScript (for C++ deployment)
   - ONNX (for cross-platform)
   - Quantized FP16 (for mobile)

2. **Integration testing**:
   - ASE calculator interface
   - MD simulation validation
   - Downstream application compatibility

3. **Documentation**:
   - Usage examples
   - Performance comparison
   - Deployment guide

4. **Consider additional variants**:
   - Medium (113K params) if gap between Tiny and Current is too large
   - Quantized versions (INT8) for embedded systems

---

## Key Differences from Failed Attempt

### What went wrong:
```python
class RandomMoleculeDataset(Dataset):
    def __getitem__(self, idx):
        # WRONG: Random data has no relationship to teacher
        atomic_numbers = torch.tensor([random.choice(elements) for ...])
        positions = torch.randn(n_atoms, 3) * 2.5
        # No teacher labels!
```

### Correct approach:
```python
class DistillationDataset(Dataset):
    def __getitem__(self, idx):
        # RIGHT: Load real structures with teacher labels
        with h5py.File(self.hdf5_path, 'r') as f:
            atomic_numbers = f['structures']['atomic_numbers'][...]
            positions = f['structures']['positions'][...]
            energy = f['labels']['energy'][...]  # Teacher label
            forces = f['labels']['forces'][...]  # Teacher label
```

**Why this matters**:
- Real data has chemical structure (bonds, geometries, symmetries)
- Teacher labels are meaningful (from Orb DFT-quality predictions)
- Model learns to compress *actual* force field knowledge, not random noise

---

## Files Reference

### Created for this task:
- `/home/aaron/ATX/software/MLFF_Distiller/scripts/train_compact_models.py` - NEW training script
- `/home/aaron/ATX/software/MLFF_Distiller/docs/COMPACT_MODELS_TRAINING_PLAN.md` - This document

### Existing infrastructure (reused):
- `/home/aaron/ATX/software/MLFF_Distiller/data/merged_dataset_4883/merged_dataset.h5` - Training data
- `/home/aaron/ATX/software/MLFF_Distiller/checkpoints/best_model.pt` - Current student (teacher)
- `/home/aaron/ATX/software/MLFF_Distiller/src/mlff_distiller/data/distillation_dataset.py` - Data loader
- `/home/aaron/ATX/software/MLFF_Distiller/src/mlff_distiller/models/student_model.py` - Model class

### Output (will be created):
- `checkpoints/compact_models/tiny/best_model.pt`
- `checkpoints/compact_models/tiny/history.json`
- `checkpoints/compact_models/ultra_tiny/best_model.pt`
- `checkpoints/compact_models/ultra_tiny/history.json`
- `logs/train_compact_models_*.log`

---

**Status**: Ready to execute
**Next action**: Run training script
**Estimated time**: 2-4 hours for both models
**Success criteria**: No NaN losses, validation loss < 2x current student
