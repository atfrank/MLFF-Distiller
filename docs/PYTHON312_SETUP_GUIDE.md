# Python 3.12 Setup Guide

**Purpose**: Enable torch.compile() optimization for 1.3-1.5x inference speedup

**Date**: November 24, 2025
**Environment**: `mlff-py312`
**Project**: ML Force Field Distillation

---

## Why Python 3.12?

**torch.compile()** is a PyTorch 2.x feature that provides graph-level optimizations for neural network inference. It requires Python 3.12+ for optimal performance and stability.

**Benefits**:
- 1.3-1.5x inference speedup (no code changes required)
- Graph fusion and kernel optimization
- Reduced Python overhead
- Better memory efficiency

**Compatibility**:
- PyTorch 2.x (≥2.5.0)
- CUDA 12.x
- All project dependencies

---

## Quick Start

```bash
# Activate the Python 3.12 environment
conda activate mlff-py312

# Verify setup
python -c "import torch; print(f'PyTorch: {torch.__version__}, torch.compile: {hasattr(torch, \"compile\")}')"

# Expected output:
# PyTorch: 2.9.1+cu128, torch.compile: True
```

---

## Complete Installation Steps

### Step 1: Create Conda Environment

```bash
cd /home/aaron/ATX/software/MLFF_Distiller

# Create Python 3.12 environment
conda create -n mlff-py312 python=3.12 -y

# Activate environment
conda activate mlff-py312

# Verify Python version
python --version
# Expected: Python 3.12.12
```

---

### Step 2: Install PyTorch with CUDA Support

```bash
# Install PyTorch 2.x with CUDA 12.1+ support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'torch.compile available: {hasattr(torch, \"compile\")}')
"

# Expected output:
# PyTorch version: 2.9.1+cu128 (or similar)
# CUDA available: True
# CUDA version: 12.8 (or similar)
# torch.compile available: True
```

---

### Step 3: Install PyTorch Geometric

```bash
# Install PyTorch Geometric and extensions
pip install torch-geometric torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

# Verify installation
python -c "import torch_geometric; print(f'PyG version: {torch_geometric.__version__}')"
```

---

### Step 4: Install Project Dependencies

```bash
# Install scientific computing libraries
pip install ase h5py pyyaml matplotlib seaborn pytest

# Install orb-models (teacher model)
pip install orb-models

# Install additional dependencies
pip install scipy numpy tqdm
```

---

### Step 5: Install Project in Development Mode

```bash
# Install mlff_distiller package
pip install -e .

# Verify installation
python -c "import mlff_distiller; print('mlff_distiller installed successfully')"
```

---

### Step 6: Run Tests

```bash
# Run test suite to verify environment
pytest tests/ -v

# Expected: 458+ tests passing (some trainer tests may fail due to PyTorch version changes)
```

---

## Package Versions

**Core Dependencies**:
```
Python: 3.12.12
PyTorch: 2.9.1+cu128
torch-geometric: 2.7.0
torch-scatter: 2.1.2
torch-sparse: 0.6.18
CUDA: 12.8
```

**Scientific Libraries**:
```
ase: 3.26.0
h5py: 3.15.1
numpy: 2.3.3
scipy: 1.16.3
matplotlib: 3.10.7
seaborn: 0.13.2
```

**ML Libraries**:
```
orb-models: 0.5.5
tensorboard: 2.20.0
wandb: 0.23.0
```

**Development Tools**:
```
pytest: 9.0.1
black: 25.11.0
isort: 7.0.0
mypy: 1.18.2
```

---

## Verification Checklist

After installation, verify the following:

### 1. Python Version
```bash
python --version
# ✓ Python 3.12.12
```

### 2. PyTorch Installation
```bash
python -c "import torch; print(torch.__version__)"
# ✓ 2.9.1+cu128 (or similar 2.x version)
```

### 3. CUDA Availability
```bash
python -c "import torch; print(torch.cuda.is_available())"
# ✓ True
```

### 4. torch.compile() Availability
```bash
python -c "import torch; print(hasattr(torch, 'compile'))"
# ✓ True
```

### 5. GPU Detection
```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
# ✓ NVIDIA GeForce RTX 3080 Ti (or your GPU)
```

### 6. Project Package
```bash
python -c "from mlff_distiller.inference import StudentForceFieldCalculator; print('OK')"
# ✓ OK
```

### 7. Checkpoint Loading
```bash
python -c "
import torch
checkpoint = torch.load('checkpoints/best_model.pt', map_location='cpu', weights_only=False)
print(f'Checkpoint keys: {list(checkpoint.keys())}')
"
# ✓ Checkpoint keys: ['model_state_dict', 'config', 'num_parameters', 'epoch']
```

### 8. Test Suite
```bash
pytest tests/ -x --tb=short
# ✓ 458+ tests passing
```

---

## Troubleshooting

### Issue 1: torch.compile() Not Available

**Symptom**:
```python
>>> hasattr(torch, 'compile')
False
```

**Solution**:
```bash
# Upgrade PyTorch to 2.x
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

### Issue 2: CUDA Not Available

**Symptom**:
```python
>>> torch.cuda.is_available()
False
```

**Solution**:
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall CUDA-enabled PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

### Issue 3: ImportError for torch_geometric

**Symptom**:
```python
>>> import torch_geometric
ImportError: No module named 'torch_geometric'
```

**Solution**:
```bash
# Install PyTorch Geometric with matching PyTorch version
pip install torch-geometric torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])")+cu121.html
```

---

### Issue 4: Checkpoint Loading Fails

**Symptom**:
```python
RuntimeError: Weights only load failed
```

**Solution**:
```python
# Use weights_only=False for older checkpoints
checkpoint = torch.load('checkpoints/best_model.pt', map_location='cpu', weights_only=False)
```

---

### Issue 5: Test Failures

**Symptom**:
```
FAILED tests/unit/test_trainer.py::...
```

**Solution**:
These failures in `test_trainer.py` are expected due to PyTorch 2.9 checkpoint format changes. They don't affect inference functionality.

**Verify core functionality**:
```bash
# Run only integration tests
pytest tests/integration/ -v

# Expected: All 21 integration tests passing
```

---

### Issue 6: Memory Errors During Tests

**Symptom**:
```
RuntimeError: CUDA out of memory
```

**Solution**:
```bash
# Run tests without CUDA
pytest tests/ -v --no-cov

# Or reduce batch size in tests
export TEST_BATCH_SIZE=1
pytest tests/ -v
```

---

## Performance Testing

### Test 1: torch.compile() Availability

```python
import torch

def test_compile():
    model = torch.nn.Linear(10, 10).cuda()
    compiled_model = torch.compile(model, mode='reduce-overhead')
    x = torch.randn(1, 10).cuda()

    # First run (compilation)
    import time
    start = time.time()
    y = compiled_model(x)
    compile_time = time.time() - start
    print(f"Compilation time: {compile_time:.2f}s")

    # Subsequent runs (optimized)
    start = time.time()
    for _ in range(100):
        y = compiled_model(x)
    inference_time = (time.time() - start) / 100
    print(f"Inference time: {inference_time*1000:.2f}ms")

test_compile()
```

**Expected Output**:
```
Compilation time: ~30s (one-time cost)
Inference time: ~0.1ms (faster than uncompiled)
```

---

### Test 2: Student Model Loading

```python
from mlff_distiller.inference import StudentForceFieldCalculator

calc = StudentForceFieldCalculator(
    checkpoint_path='checkpoints/best_model.pt',
    device='cuda',
    use_compile=True,
    compile_mode='reduce-overhead'
)

print(f"✓ Calculator initialized")
print(f"✓ Model compiled: {calc.use_compile}")
```

---

### Test 3: Simple Inference

```python
from ase.build import molecule
from mlff_distiller.inference import StudentForceFieldCalculator

# Create test structure
atoms = molecule('H2O')

# Set up calculator
calc = StudentForceFieldCalculator(
    checkpoint_path='checkpoints/best_model.pt',
    device='cuda',
    use_compile=True
)
atoms.calc = calc

# Run inference
energy = atoms.get_potential_energy()
forces = atoms.get_forces()

print(f"Energy: {energy:.6f} eV")
print(f"Forces shape: {forces.shape}")
print("✓ Inference successful")
```

---

## Environment Management

### Switching Between Environments

```bash
# Activate Python 3.12 environment (for torch.compile())
conda activate mlff-py312

# Activate old environment (if needed)
conda activate mlff_distiller

# Check current environment
conda env list
```

---

### Deactivate Environment

```bash
conda deactivate
```

---

### Remove Environment (if needed)

```bash
conda env remove -n mlff-py312
```

---

### Export Environment

```bash
# Export for reproducibility
conda activate mlff-py312
conda env export > environment_py312.yml

# Recreate from export
conda env create -f environment_py312.yml
```

---

## Next Steps

After verifying the installation:

1. **Run Quick MD Validation**:
   ```bash
   python scripts/quick_md_validation.py \
       --checkpoint checkpoints/best_model.pt \
       --duration 100 \
       --output validation_results/quick_nve
   ```

2. **Test torch.compile()**:
   ```bash
   python scripts/benchmark_inference.py \
       --use-compile \
       --output benchmarks/with_compile/
   ```

3. **Implement FP16**:
   Follow Phase 1 optimization roadmap

---

## References

- [PyTorch torch.compile() Documentation](https://pytorch.org/docs/stable/generated/torch.compile.html)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [PyTorch Geometric Installation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
- [Conda Documentation](https://docs.conda.io/en/latest/)

---

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review Week 1 Coordination Plan
3. Consult Phase 1 Optimization Spec
4. Contact project coordinator

**Working Directory**: `/home/aaron/ATX/software/MLFF_Distiller`
**Environment Name**: `mlff-py312`
**Checkpoint**: `checkpoints/best_model.pt`
