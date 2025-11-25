# Quick Start: Student Model

**Last Updated**: 2025-11-24

---

## TL;DR

```python
from mlff_distiller.models.student_model import StudentForceField
import torch

# Create model
model = StudentForceField(hidden_dim=128, num_interactions=3)

# Predict energy and forces
atomic_numbers = torch.tensor([8, 1, 1])  # Water: O, H, H
positions = torch.tensor([[0., 0., 0.], [0.96, 0., 0.], [-0.24, 0.93, 0.]])

energy, forces = model.predict_energy_and_forces(atomic_numbers, positions)
print(f"Energy: {energy.item():.4f} eV")
print(f"Forces:\n{forces}")
```

---

## Installation

No additional dependencies needed beyond base environment:
```bash
# Already installed:
# - PyTorch 2.5.1+
# - NumPy
# - h5py
# - ASE
```

---

## Basic Usage

### 1. Create Model

```python
from mlff_distiller.models.student_model import StudentForceField

# Default configuration (430K parameters)
model = StudentForceField()

# Custom configuration
model = StudentForceField(
    hidden_dim=256,      # Increase for more capacity
    num_interactions=4,  # More layers for larger systems
    num_rbf=20,         # Radial basis functions
    cutoff=5.0,         # Neighbor cutoff (Angstroms)
    max_z=118           # Max atomic number
)

print(f"Parameters: {model.num_parameters():,}")
```

### 2. Run Inference

```python
import torch

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Create inputs
atomic_numbers = torch.tensor([6, 8, 1, 1], device=device)  # C, O, H, H
positions = torch.randn(4, 3, device=device) * 2.0

# Predict
energy, forces = model.predict_energy_and_forces(atomic_numbers, positions)
```

### 3. Load from HDF5 Dataset

```python
import h5py
import numpy as np

def load_structure(hdf5_path, idx=0):
    with h5py.File(hdf5_path, 'r') as f:
        splits = f['structures']['atomic_numbers_splits'][:]
        start, end = splits[idx], splits[idx+1]

        atomic_numbers = torch.from_numpy(
            f['structures']['atomic_numbers'][start:end]
        ).long()

        positions = torch.from_numpy(
            f['structures']['positions'][start:end]
        ).float()

        return atomic_numbers, positions

# Load and predict
atomic_numbers, positions = load_structure('data/merged_dataset_4883/merged_dataset.h5')
energy, forces = model.predict_energy_and_forces(atomic_numbers, positions)
```

### 4. Batch Processing

```python
# Create batch
atomic_numbers = torch.cat([numbers1, numbers2, numbers3])
positions = torch.cat([pos1, pos2, pos3])
batch = torch.tensor([0]*len(numbers1) + [1]*len(numbers2) + [2]*len(numbers3))

# Predict (returns 3 energies)
energies = model(atomic_numbers, positions, batch=batch)
```

### 5. Save and Load

```python
# Save
model.save('checkpoints/student_model.pt')

# Load
loaded_model = StudentForceField.load('checkpoints/student_model.pt', device='cuda')
```

---

## Examples

### Complete Example Scripts

1. **Basic Demo**: `examples/student_model_demo.py`
   ```bash
   python examples/student_model_demo.py
   ```

2. **Unit Tests**: `tests/unit/test_student_model.py`
   ```bash
   pytest tests/unit/test_student_model.py -v
   ```

---

## Common Operations

### Training Loop (Pseudo-code)

```python
from mlff_distiller.models.student_model import StudentForceField
from torch.utils.data import DataLoader

# Create model
model = StudentForceField().to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(100):
    for batch in dataloader:
        # Forward pass
        atomic_numbers = batch['atomic_numbers'].to('cuda')
        positions = batch['positions'].to('cuda').requires_grad_(True)

        energy_pred, forces_pred = model.predict_energy_and_forces(
            atomic_numbers, positions
        )

        # Loss
        energy_loss = F.mse_loss(energy_pred, batch['energy'].to('cuda'))
        force_loss = F.mse_loss(forces_pred, batch['forces'].to('cuda'))
        loss = energy_loss + 100 * force_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Evaluation

```python
model.eval()
with torch.no_grad():
    for batch in val_loader:
        energy, forces = model.predict_energy_and_forces(
            batch['atomic_numbers'],
            batch['positions']
        )

        # Compute metrics
        energy_mae = (energy - batch['energy']).abs().mean()
        force_mae = (forces - batch['forces']).abs().mean()
```

### Benchmark Inference

```python
import time

model.eval()
with torch.no_grad():
    # Warmup
    for _ in range(10):
        _ = model(atomic_numbers, positions)

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()

    for _ in range(100):
        _ = model(atomic_numbers, positions)

    torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"Time per structure: {elapsed/100*1000:.2f} ms")
```

---

## Troubleshooting

### Out of Memory

```python
# Reduce batch size
# Or reduce model size
model = StudentForceField(hidden_dim=64, num_interactions=2)
```

### Slow Inference

```python
# Use GPU
model = model.to('cuda')

# Use eval mode
model.eval()

# Disable gradients for inference
with torch.no_grad():
    energy = model(atomic_numbers, positions)
```

### NaN/Inf in Outputs

```python
# Check inputs
assert torch.isfinite(positions).all()
assert torch.isfinite(atomic_numbers.float()).all()

# Use gradient clipping during training
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## Performance Tips

1. **Use GPU**: 10-100x speedup over CPU
2. **Batch processing**: Process multiple structures together
3. **Eval mode**: `model.eval()` for inference
4. **No gradients**: `with torch.no_grad():` when not training
5. **Mixed precision**: Use `torch.cuda.amp` for 2x speedup

---

## API Reference

### StudentForceField

```python
class StudentForceField(
    hidden_dim: int = 128,
    num_interactions: int = 3,
    num_rbf: int = 20,
    cutoff: float = 5.0,
    max_z: int = 118,
    learnable_rbf: bool = False
)
```

**Methods**:
- `forward(atomic_numbers, positions, cell=None, pbc=None, batch=None) -> energy`
- `predict_energy_and_forces(atomic_numbers, positions, cell=None, pbc=None) -> (energy, forces)`
- `save(path)` - Save checkpoint
- `load(path, device='cpu')` - Load checkpoint (class method)
- `num_parameters()` - Count parameters

---

## File Locations

- **Model**: `src/mlff_distiller/models/student_model.py`
- **Tests**: `tests/unit/test_student_model.py`
- **Examples**: `examples/student_model_demo.py`
- **Docs**:
  - `docs/STUDENT_ARCHITECTURE_DESIGN.md` (detailed design)
  - `docs/STUDENT_MODEL_IMPLEMENTATION_SUMMARY.md` (implementation details)

---

## Next Steps

1. **Training**: Implement distillation training loop
2. **Validation**: Test on held-out structures
3. **Optimization**: Profile and optimize bottlenecks (M4)
4. **Deployment**: Integrate into production pipeline

---

## Support

- Architecture design: See `docs/STUDENT_ARCHITECTURE_DESIGN.md`
- Implementation details: See `docs/STUDENT_MODEL_IMPLEMENTATION_SUMMARY.md`
- Unit tests: Run `pytest tests/unit/test_student_model.py -v`
- Examples: Run `python examples/student_model_demo.py`

---

**Model Status**: ✅ Complete and tested (430K parameters, 18/19 tests passing)

**Ready for**: Distillation training on 4,883 teacher-labeled structures
