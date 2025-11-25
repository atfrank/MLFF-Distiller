# Training Quick Start Guide

**Quick guide to training the student force field model via knowledge distillation**

Date: 2025-11-24
Author: ML Distillation Project

---

## Prerequisites

✅ **Completed** (if you followed M1-M3):
- HDF5 dataset with teacher labels: `data/merged_dataset_4883/merged_dataset.h5`
- Student model architecture: `src/mlff_distiller/models/student_model.py`
- Training infrastructure: `src/mlff_distiller/training/`

**Requirements**:
- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- 8+ GB GPU memory

---

## Quick Start (3 Commands)

### 1. Verify Dataset

```bash
ls -lh data/merged_dataset_4883/merged_dataset.h5
# Should show: 19.53 MB file with 4,883 structures
```

### 2. Test Training Pipeline (Dry Run)

```bash
python scripts/train_student.py --config configs/train_student.yaml --dry-run
```

**Expected output**:
```
============================================================
ML FORCE FIELD DISTILLATION - STUDENT TRAINING
============================================================
Data file: data/merged_dataset_4883/merged_dataset.h5
...
Creating student model...
Created student model with 427,292 parameters
  Interactions: 3
  Hidden dim: 128
  ...
Dataset split: 4394 train, 489 val
  Train batches: 275
  Val batches: 31
DRY RUN - Setup complete, exiting before training
```

### 3. Start Training!

```bash
python scripts/train_student.py --config configs/train_student.yaml
```

**Training starts immediately** - progress will be logged to:
- Console: Real-time updates
- `logs/training.log`: Full training log
- `runs/`: TensorBoard logs
- `checkpoints/`: Model checkpoints

---

## Training Progress

### What to Expect

**Training Time** (4,883 structures, 100 epochs):
- GPU (NVIDIA A100): ~1-2 hours
- GPU (RTX 3090): ~2-4 hours
- CPU: ~24-48 hours (not recommended)

**Output Files**:
```
checkpoints/
  ├── training_config.json           # Your training settings
  ├── checkpoint_epoch_5.pt          # Periodic checkpoint
  ├── checkpoint_epoch_10.pt
  ├── ...
  └── best_model.pt                  # Best model (lowest val loss)

logs/
  └── training.log                   # Full training log

runs/
  └── run_YYYYMMDD_HHMMSS/          # TensorBoard logs
```

### Monitor Training

**Option 1: Watch Console**
```bash
# Real-time progress bars and metrics
Epoch 1/100: 100%|██████████| 275/275 [00:45<00:00, 6.11it/s, loss=X.XX, force_rmse=X.XX]
Train Loss: X.XXXX
Val Loss: X.XXXX
Force RMSE: X.XXXX eV/Å
```

**Option 2: TensorBoard**
```bash
tensorboard --logdir runs/
# Open browser to http://localhost:6006
```

**Option 3: Tail Logs**
```bash
tail -f logs/training.log
```

---

## Target Metrics (What to Aim For)

### Validation Metrics After Training

| Metric | Target | Good | Excellent |
|--------|--------|------|-----------|
| **Force RMSE** | <0.1 eV/Å | <0.05 eV/Å | <0.03 eV/Å |
| **Energy MAE** | <10 meV/atom | <5 meV/atom | <2 meV/atom |
| **Val Loss** | <10 | <5 | <2 |

**Force RMSE is most critical** - this determines MD stability.

### Convergence Timeline

- **Epoch 1-10**: Rapid improvement, loss drops quickly
- **Epoch 10-30**: Steady improvement
- **Epoch 30-50**: Convergence, improvements slow
- **Epoch 50-100**: Fine-tuning, minor improvements

**Early stopping** typically triggers around epoch 60-80.

---

## Command Line Options

### Quick Reference

```bash
# Custom dataset
python scripts/train_student.py --data path/to/dataset.h5

# More epochs
python scripts/train_student.py --epochs 200

# Larger batch size (if GPU memory allows)
python scripts/train_student.py --batch-size 32

# Custom learning rate
python scripts/train_student.py --lr 5e-4

# Enable mixed precision (2x faster on modern GPUs)
python scripts/train_student.py --mixed-precision

# Resume from checkpoint
python scripts/train_student.py --resume checkpoints/checkpoint_epoch_50.pt

# Change model size
python scripts/train_student.py --hidden-dim 256 --num-interactions 4
```

### Full Options

```bash
python scripts/train_student.py --help
```

**Key arguments**:
- `--config`: Path to YAML config (default: `configs/train_student.yaml`)
- `--epochs`: Number of epochs (default: 100)
- `--batch-size`: Batch size (default: 16)
- `--lr`: Learning rate (default: 1e-3)
- `--force-weight`: Weight for force loss (default: 100.0)
- `--energy-weight`: Weight for energy loss (default: 1.0)
- `--resume`: Resume from checkpoint
- `--device`: cuda/cpu/auto (default: auto)
- `--mixed-precision`: Enable FP16 training
- `--dry-run`: Test setup without training

---

## Troubleshooting

### Problem: Out of Memory (OOM)

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size: `--batch-size 8`
2. Enable gradient accumulation in config: `accumulation_steps: 2`
3. Reduce model size: `--hidden-dim 64`
4. Use CPU (slow): `--device cpu`

### Problem: Training Not Converging

**Symptoms**: Loss not decreasing after 20+ epochs

**Solutions**:
1. Reduce learning rate: `--lr 5e-4` or `--lr 1e-4`
2. Increase warmup: Edit config, set `warmup_steps: 2000`
3. Try Huber loss (more robust): Edit config, set `force_loss_type: "huber"`
4. Check data quality: Ensure teacher labels are valid

### Problem: NaN Loss

**Error**: Loss becomes NaN during training

**Solutions**:
1. Reduce learning rate: `--lr 1e-4`
2. Enable gradient clipping: `--grad-clip 0.5`
3. Check for invalid data: `nan` or `inf` in dataset
4. Try different optimizer: Edit config, use `"adam"` instead of `"adamw"`

### Problem: Training Too Slow

**Solutions**:
1. Enable mixed precision: `--mixed-precision`
2. Increase batch size: `--batch-size 32` (if memory allows)
3. Reduce data workers: `--num-workers 2`
4. Use GPU if on CPU: `--device cuda`

### Problem: Force RMSE Not Improving

**Symptoms**: Energy MAE good, but force RMSE stuck > 0.2 eV/Å

**Solutions**:
1. Increase force weight: `--force-weight 200` or `--force-weight 500`
2. Train longer: `--epochs 200`
3. Increase model capacity: `--hidden-dim 256 --num-interactions 4`
4. Check gradient flow (may need architecture changes)

---

## Next Steps After Training

### 1. Evaluate Best Model

```python
import torch
from mlff_distiller.models.student_model import StudentForceField
from mlff_distiller.models.distillation_wrapper import DistillationWrapper

# Load best checkpoint
checkpoint = torch.load('checkpoints/best_model.pt')

# Create and load model
student = StudentForceField(num_interactions=3, hidden_dim=128)
model = DistillationWrapper(student)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
print(f"Training epochs: {checkpoint['epoch']}")
```

### 2. Validate on Test Set

Create a separate test set and evaluate:
- Force RMSE on diverse systems
- Energy conservation in short MD
- Stability over longer trajectories

### 3. CUDA Optimization (M4)

Once you have a trained model with good accuracy:
- Profile performance bottlenecks
- Implement custom CUDA kernels
- Target: 3-5x additional speedup

### 4. Production Testing (M5)

- Run 100ps MD trajectories
- Test on systems outside training distribution
- Compare to teacher model on benchmarks

---

## Configuration Files

### Default Config: `configs/train_student.yaml`

**Good for**: Initial validation, development, testing

**Settings**:
- `max_epochs: 100`
- `batch_size: 16`
- `learning_rate: 1e-3`
- `force_weight: 100.0`

**When to customize**:
- GPU has 16+ GB memory → increase `batch_size: 32`
- Training not converging → reduce `learning_rate: 5e-4`
- Force errors too high → increase `force_weight: 200`

### Create Custom Config

```yaml
# configs/my_custom_config.yaml
max_epochs: 200
batch_size: 32
mixed_precision: true

optimizer:
  learning_rate: 5.0e-4
  weight_decay: 1.0e-4

loss:
  energy_weight: 1.0
  force_weight: 150.0

# ... (see configs/train_student.yaml for all options)
```

```bash
python scripts/train_student.py --config configs/my_custom_config.yaml
```

---

## FAQ

**Q: How long should I train?**
A: 50-100 epochs is usually sufficient. Early stopping will halt training automatically if no improvement for 20 epochs.

**Q: Can I pause and resume training?**
A: Yes! Press `Ctrl+C` to interrupt. Resume with:
```bash
python scripts/train_student.py --resume checkpoints/checkpoint_interrupted.pt
```

**Q: Should I use CPU or GPU?**
A: GPU strongly recommended. CPU training is 10-20x slower.

**Q: How do I know if training is working?**
A: Watch force RMSE - it should decrease steadily. Target < 0.1 eV/Å.

**Q: What if validation loss stops improving?**
A: This is normal! Early stopping will save the best model and halt training.

**Q: Can I train on multiple GPUs?**
A: Not yet implemented. Single GPU only for now.

**Q: How much GPU memory do I need?**
A: 8 GB minimum. 16+ GB recommended for larger batch sizes.

**Q: What's the best learning rate?**
A: Start with 1e-3. Reduce to 5e-4 or 1e-4 if not converging.

**Q: Should I normalize energies and forces?**
A: Not necessary. Loss function handles scaling via weights.

**Q: How do I visualize training progress?**
A: Use TensorBoard: `tensorboard --logdir runs/`

---

## Example Training Session

```bash
# 1. Verify dataset
$ ls -lh data/merged_dataset_4883/merged_dataset.h5
-rw-r--r-- 1 user group 19.53M Nov 24 00:00 merged_dataset.h5

# 2. Test setup
$ python scripts/train_student.py --dry-run
...
Dataset split: 4394 train, 489 val
DRY RUN - Setup complete

# 3. Start training
$ python scripts/train_student.py
============================================================
ML FORCE FIELD DISTILLATION - STUDENT TRAINING
============================================================
Created student model with 427,292 parameters
Dataset split: 4394 train, 489 val

STARTING TRAINING
============================================================
Epoch 1/100: 100%|███| 275/275 [00:45<00:00]
  Train Loss: 52.3456
  Val Loss: 48.7621
  Force RMSE: 0.2134 eV/Å

Epoch 2/100: 100%|███| 275/275 [00:44<00:00]
  Train Loss: 23.1234
  Val Loss: 21.5432
  Force RMSE: 0.1234 eV/Å

... (continues for 100 epochs or until early stopping)

Epoch 67/100: 100%|███| 275/275 [00:44<00:00]
  Train Loss: 2.1234
  Val Loss: 2.3456
  Force RMSE: 0.0456 eV/Å

Early stopping after 67 epochs
============================================================
TRAINING COMPLETE
============================================================
Final epoch: 67
Best validation loss: 2.1234
Checkpoints saved to: checkpoints/

# 4. Check results
$ ls checkpoints/
best_model.pt
checkpoint_epoch_65.pt
checkpoint_epoch_60.pt
checkpoint_epoch_55.pt
training_config.json
```

---

## Support

**Issues**: https://github.com/your-org/MLFF_Distiller/issues
**Documentation**: `docs/`
**Examples**: `examples/student_model_demo.py`

---

**Author**: ML Distillation Project
**Date**: 2025-11-24
**Version**: M3 - Training Pipeline Complete
