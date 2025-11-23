# Teacher Model Wrapper Guide

**Version**: 1.0
**Last Updated**: 2025-11-23
**Status**: Implementation Complete

## Overview

This guide covers the teacher model wrapper calculators (`OrbCalculator` and `FeNNolCalculator`) that provide ASE Calculator interfaces for teacher models used in the MLFF Distiller project.

These wrappers enable:
- **Drop-in replacement** compatibility with existing ASE MD scripts
- **Baseline benchmarking** of teacher model performance
- **Data generation** from teacher models for distillation training
- **Template interface** for student model calculators

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [OrbCalculator](#orbcalculator)
4. [FeNNolCalculator](#fennolcalculator)
5. [Usage Examples](#usage-examples)
6. [API Reference](#api-reference)
7. [Performance Considerations](#performance-considerations)
8. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

```bash
# Base requirements
pip install torch ase numpy

# For OrbCalculator
pip install orb-models

# For FeNNolCalculator
pip install jax  # or jax[cuda12] for GPU
pip install fennol
```

### MLFF Distiller Installation

```bash
# Install MLFF Distiller
cd /path/to/MLFF_Distiller
pip install -e .
```

## Quick Start

### Basic Usage with OrbCalculator

```python
from ase.build import molecule
from mlff_distiller.models.teacher_wrappers import OrbCalculator

# Create molecule
atoms = molecule('H2O')

# Setup calculator
calc = OrbCalculator(model_name='orb-v2', device='cuda')
atoms.calc = calc

# Compute properties
energy = atoms.get_potential_energy()  # eV
forces = atoms.get_forces()            # eV/Angstrom
stress = atoms.get_stress()            # eV/Angstrom^3
```

### Drop-in Replacement in MD

```python
from ase.md.verlet import VelocityVerlet
from ase import units

# Just replace the calculator - everything else stays the same!
calc = OrbCalculator(model_name='orb-v2', device='cuda')
atoms.calc = calc

# Standard ASE MD workflow (no changes)
dyn = VelocityVerlet(atoms, timestep=1.0*units.fs)
dyn.run(10000)
```

## OrbCalculator

### Description

`OrbCalculator` wraps the Orb-models force field from Orbital Materials, providing state-of-the-art universal force fields trained on large-scale DFT datasets.

### Supported Models

| Model Name | Description | Precision Options |
|------------|-------------|-------------------|
| `orb-v1` | Original Orb model | N/A |
| `orb-v2` | Improved Orb model | N/A |
| `orb-v3` | Latest with confidence | float32-high, float32-highest, float64 |
| `orb-v3-conservative-inf-omat` | Conservative inference | float32-high, float32-highest, float64 |
| `orb-v3-strict-inf-omat` | Strict inference | float32-high, float32-highest, float64 |

### Initialization Parameters

```python
OrbCalculator(
    model_name: str = "orb-v2",          # Model version to use
    device: str = "cuda",                 # 'cpu' or 'cuda'
    precision: str = "float32-high",      # Precision mode (v3 only)
    dtype: torch.dtype = torch.float32,   # PyTorch dtype
    **kwargs                              # Additional ASE Calculator args
)
```

### Implemented Properties

- `energy` - Total potential energy (eV)
- `forces` - Atomic forces (eV/Angstrom)
- `stress` - Stress tensor in Voigt notation (eV/Angstrom^3)
- `confidence` - Per-atom confidence (Orb-v3 only)

### Example: Using Orb-v3 with Confidence

```python
calc = OrbCalculator(model_name='orb-v3', device='cuda', precision='float32-highest')
atoms.calc = calc

energy = atoms.get_potential_energy()
forces = atoms.get_forces()

# Access confidence estimates (Orb-v3 only)
if 'confidence' in calc.results:
    confidence = calc.results['confidence']  # Per-atom confidence
    print(f"Confidence range: {confidence.min():.3f} to {confidence.max():.3f}")
```

### Supported System Types

- **Molecules**: Non-periodic molecular systems (10-1000 atoms)
- **Bulk crystals**: Periodic systems with PBC
- **Surfaces**: 2D periodic systems
- **Mixed systems**: Combinations of above

### Performance Characteristics

- **Inference time**: ~10-100ms per structure (100-500 atoms, GPU)
- **Memory**: ~2-4 GB GPU memory for typical systems
- **Batch processing**: Not currently optimized for batching
- **Device support**: CPU and CUDA (GPU)

## FeNNolCalculator

### Description

`FeNNolCalculator` wraps the FeNNol (Force-field-enhanced Neural Network optimized library) models, which combine neural network embeddings with ML-parameterized physical interactions.

### Supported Models

| Model Name | Description | Notes |
|------------|-------------|-------|
| `ani-2x` | ANI-2x pretrained model | Organic molecules (H, C, N, O) |
| Custom | User-trained models | Via `model_path` parameter |

### Initialization Parameters

```python
FeNNolCalculator(
    model_path: Optional[str] = None,     # Path to checkpoint
    model_name: Optional[str] = None,     # Pretrained model name
    device: str = "cuda",                 # 'cpu' or 'cuda'
    dtype = None,                         # JAX dtype
    **kwargs                              # Additional ASE Calculator args
)
```

**Note**: Either `model_path` or `model_name` must be provided.

### Implemented Properties

- `energy` - Total potential energy (eV)
- `forces` - Atomic forces (eV/Angstrom)

**Note**: Stress calculation may not be supported by all FeNNol models.

### Example: Loading Custom Model

```python
# Load from checkpoint
calc = FeNNolCalculator(
    model_path='/path/to/fennol_checkpoint.pt',
    device='cuda'
)

atoms.calc = calc
energy = atoms.get_potential_energy()
```

### Example: Using ANI-2x

```python
# Use pretrained ANI-2x
calc = FeNNolCalculator(model_name='ani-2x', device='cuda')

# ANI-2x supports H, C, N, O
atoms = molecule('CH3COOH')  # Acetic acid
atoms.calc = calc

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

### Performance Characteristics

- **Inference time**: Near force-field speeds on GPU
- **Memory**: ~1-2 GB GPU memory
- **Device support**: CPU and CUDA via JAX
- **Specialization**: Optimized for organic molecules

## Usage Examples

### Example 1: Running NVE MD

```python
from ase import units
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

# Setup
atoms = molecule('H2O')
calc = OrbCalculator(model_name='orb-v2', device='cuda')
atoms.calc = calc

# Initialize velocities
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# Run MD
dyn = VelocityVerlet(atoms, timestep=0.5*units.fs)
dyn.run(1000)
```

### Example 2: Running NVT MD with Langevin

```python
from ase.md.langevin import Langevin

atoms = molecule('CH4')
calc = OrbCalculator(model_name='orb-v2', device='cuda')
atoms.calc = calc

# Langevin thermostat
dyn = Langevin(
    atoms,
    timestep=1.0*units.fs,
    temperature_K=300,
    friction=0.01
)
dyn.run(5000)
```

### Example 3: Geometry Optimization

```python
from ase.optimize import BFGS

atoms = molecule('H2O')
calc = OrbCalculator(model_name='orb-v2', device='cuda')
atoms.calc = calc

opt = BFGS(atoms)
opt.run(fmax=0.05)  # Optimize until max force < 0.05 eV/Angstrom
```

### Example 4: Working with Bulk Crystals

```python
from ase.build import bulk

# Create bulk crystal
atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
calc = OrbCalculator(model_name='orb-v2', device='cuda')
atoms.calc = calc

# Compute properties
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()

print(f"Energy per atom: {energy/len(atoms):.4f} eV")
print(f"Stress (GPa): {stress * 160.21766208}")  # Convert to GPa
```

### Example 5: Trajectory Analysis

```python
from ase.io.trajectory import Trajectory

# Run MD and save trajectory
atoms = molecule('H2O')
calc = OrbCalculator(model_name='orb-v2', device='cuda')
atoms.calc = calc

traj = Trajectory('md.traj', 'w', atoms)
dyn = VelocityVerlet(atoms, timestep=1.0*units.fs)
dyn.attach(traj.write, interval=10)
dyn.run(1000)
traj.close()

# Analyze trajectory
for snapshot in Trajectory('md.traj'):
    energy = snapshot.get_potential_energy()
    # Perform analysis...
```

## API Reference

### OrbCalculator

#### Methods

##### `__init__(model_name, device, precision, dtype, **kwargs)`

Initialize OrbCalculator with specified model.

**Parameters:**
- `model_name` (str): Orb model version
- `device` (str): 'cpu' or 'cuda'
- `precision` (str): Precision mode for v3 models
- `dtype` (torch.dtype): PyTorch data type
- `**kwargs`: Additional ASE Calculator arguments

##### `calculate(atoms, properties, system_changes)`

Calculate properties for given atoms (called by ASE internally).

**Parameters:**
- `atoms` (ase.Atoms): Atoms object
- `properties` (list): Properties to calculate
- `system_changes` (list): System changes since last call

**Populates:**
- `self.results['energy']`: Total energy (eV)
- `self.results['forces']`: Atomic forces (eV/Angstrom)
- `self.results['stress']`: Stress tensor (eV/Angstrom^3)
- `self.results['confidence']`: Confidence (v3 only)

### FeNNolCalculator

#### Methods

##### `__init__(model_path, model_name, device, dtype, **kwargs)`

Initialize FeNNolCalculator with specified model.

**Parameters:**
- `model_path` (str, optional): Path to checkpoint
- `model_name` (str, optional): Pretrained model name
- `device` (str): 'cpu' or 'cuda'
- `dtype`: JAX data type
- `**kwargs`: Additional ASE Calculator arguments

##### `calculate(atoms, properties, system_changes)`

Calculate properties for given atoms (called by ASE internally).

**Parameters:**
- `atoms` (ase.Atoms): Atoms object
- `properties` (list): Properties to calculate
- `system_changes` (list): System changes since last call

**Populates:**
- `self.results['energy']`: Total energy (eV)
- `self.results['forces']`: Atomic forces (eV/Angstrom)

## Performance Considerations

### Optimizing for MD Simulations

1. **Use GPU acceleration**: Always use `device='cuda'` for production MD
2. **Minimize per-call overhead**: Wrappers are optimized to reuse buffers
3. **Batch trajectories**: For data generation, process multiple trajectories in parallel
4. **Monitor memory**: Track memory usage over long simulations to detect leaks

### Benchmarking

```python
import time

atoms = molecule('H2O')
calc = OrbCalculator(model_name='orb-v2', device='cuda')
atoms.calc = calc

# Warmup
for _ in range(10):
    atoms.get_potential_energy()

# Benchmark
n_calls = 1000
start = time.time()
for _ in range(n_calls):
    atoms.positions += np.random.randn(len(atoms), 3) * 0.01
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
elapsed = time.time() - start

print(f"Average time per call: {elapsed/n_calls*1000:.2f} ms")
```

### Memory Management

```python
# For long MD runs, periodically check memory
import tracemalloc

tracemalloc.start()

# Run MD...
dyn.run(10000)

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory: {current / 1e6:.1f} MB")
print(f"Peak memory: {peak / 1e6:.1f} MB")

tracemalloc.stop()
```

## Troubleshooting

### Common Issues

#### Issue: "orb-models package not found"

**Solution:**
```bash
pip install orb-models
```

#### Issue: "Unknown model name: orb-vX"

**Solution:** Check available models:
```python
# Valid model names:
# - orb-v1
# - orb-v2
# - orb-v3
# - orb-v3-conservative-inf-omat
# - orb-v3-strict-inf-omat
```

#### Issue: "CUDA out of memory"

**Solution:**
- Reduce system size
- Use smaller batch sizes
- Switch to CPU: `device='cpu'`
- Clear GPU cache: `torch.cuda.empty_cache()`

#### Issue: "FeNNol import error"

**Solution:**
```bash
# Install JAX first
pip install jax[cuda12]  # For GPU
# or
pip install jax  # For CPU

# Then install FeNNol
pip install fennol
```

#### Issue: Calculator results don't update

**Cause:** ASE caches results. Force recalculation by modifying atoms:
```python
atoms.positions = atoms.positions  # Force update
energy = atoms.get_potential_energy()
```

### Testing Installation

```python
# Test OrbCalculator
from mlff_distiller.models.teacher_wrappers import OrbCalculator
from ase.build import molecule

atoms = molecule('H2')
calc = OrbCalculator(model_name='orb-v2', device='cpu')
atoms.calc = calc

energy = atoms.get_potential_energy()
print(f"Installation successful! Energy: {energy:.4f} eV")
```

## References

- [ASE Calculator Documentation](https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html)
- [Orb-models GitHub](https://github.com/orbital-materials/orb-models)
- [FeNNol GitHub](https://github.com/thomasple/FeNNol)
- [Drop-in Compatibility Guide](DROP_IN_COMPATIBILITY_GUIDE.md)

## Contributing

For issues or improvements:
1. Create an issue on GitHub
2. Tag `@ml-architecture-designer`
3. Reference this guide

## License

MIT License (see project LICENSE file)

---

**Last Updated**: 2025-11-23
**Maintainer**: ML Architecture Designer
**Status**: Production Ready
