# Quick Start Guide

This guide walks you through installing MLFF Distiller and using the production model for molecular simulations.

## Prerequisites

- **Python 3.10+** (Python 3.11 or 3.12 recommended)
- **PyTorch 2.0+** with CUDA support (optional but recommended)
- **ASE** (Atomic Simulation Environment)

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10 | 3.11+ |
| RAM | 8 GB | 16 GB |
| GPU | None (CPU works) | NVIDIA GPU with 4GB+ VRAM |
| CUDA | 11.8 | 12.1+ |

## Installation

### Option 1: Install from Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/atfrank/MLFF_Distiller.git
cd MLFF_Distiller

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with all dependencies
pip install -e ".[dev]"
```

### Option 2: Install with CUDA Support

```bash
# Install PyTorch with CUDA first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Then install MLFF Distiller
pip install -e ".[cuda]"
```

### Option 3: Minimal Installation

```bash
# Core dependencies only
pip install -e .
```

### Verify Installation

```python
import mlff_distiller
print(f"MLFF Distiller version: {mlff_distiller.__version__}")

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Loading the Production Model

The production model is located at `checkpoints/best_model.pt`. Load it using the `StudentForceFieldCalculator`:

```python
from mlff_distiller import StudentForceFieldCalculator

# Load model (automatically detects GPU/CPU)
calc = StudentForceFieldCalculator(
    checkpoint_path='checkpoints/best_model.pt',
    device='cuda'  # Use 'cpu' if no GPU available
)

print(f"Model loaded on {calc.device}")
print(f"Implemented properties: {calc.implemented_properties}")
```

### Model Specifications

| Property | Value |
|----------|-------|
| Architecture | PaiNN-based |
| Parameters | 427,292 |
| Hidden Dimension | 128 |
| Message Passing Layers | 3 |
| Cutoff Distance | 5.0 Angstrom |
| Supported Elements | H, C, N, O, F, S, Cl (Z=1-17) |

## Basic Usage Examples

### Calculate Energy and Forces

```python
from mlff_distiller import StudentForceFieldCalculator
from ase.build import molecule

# Create calculator
calc = StudentForceFieldCalculator('checkpoints/best_model.pt', device='cuda')

# Create a water molecule
atoms = molecule('H2O')
atoms.calc = calc

# Calculate properties
energy = atoms.get_potential_energy()  # Returns energy in eV
forces = atoms.get_forces()            # Returns forces in eV/Angstrom

print(f"Energy: {energy:.4f} eV")
print(f"Forces shape: {forces.shape}")  # (3, 3) for 3 atoms x 3 coordinates
```

### Batch Calculations

For multiple structures, use batch calculation for better performance:

```python
from ase.build import molecule

# Create multiple molecules
molecules = [molecule('H2O'), molecule('CH4'), molecule('NH3'), molecule('C2H6')]

# Batch calculate (much faster than iterating)
results = calc.calculate_batch(molecules, properties=['energy', 'forces'])

for mol, result in zip(molecules, results):
    print(f"{mol.get_chemical_formula()}: E = {result['energy']:.4f} eV")
```

### Structure Optimization

```python
from ase.build import molecule
from ase.optimize import BFGS

# Create molecule with calculator
atoms = molecule('H2O')
atoms.calc = StudentForceFieldCalculator('checkpoints/best_model.pt')

# Optimize geometry
optimizer = BFGS(atoms, trajectory='optimization.traj')
optimizer.run(fmax=0.01)  # Converge when max force < 0.01 eV/Angstrom

print(f"Optimized energy: {atoms.get_potential_energy():.4f} eV")
```

## Running MD Simulations

### NVE Molecular Dynamics

```python
from mlff_distiller import StudentForceFieldCalculator
from mlff_distiller.testing import NVEMDHarness
from ase.build import molecule

# Setup
calc = StudentForceFieldCalculator('checkpoints/best_model.pt', device='cuda')
atoms = molecule('H2O')

# Create NVE harness
harness = NVEMDHarness(
    atoms=atoms,
    calculator=calc,
    temperature=300.0,      # Initial temperature in K
    timestep=0.5,           # Timestep in fs
    trajectory_file='md_trajectory.traj'
)

# Run simulation (1000 steps = 0.5 ps)
results = harness.run_simulation(steps=1000)

# Check results
print(f"Energy drift: {results['energy_drift_pct']:.4f}%")
print(f"Average temperature: {results['avg_temperature']:.2f} K")
print(f"Simulation time: {results['total_time_ps']:.3f} ps")
```

### NVT (Langevin) Dynamics

```python
from ase.md.langevin import Langevin
from ase import units

atoms = molecule('CH4')
atoms.calc = StudentForceFieldCalculator('checkpoints/best_model.pt')

# Initialize velocities
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# Run Langevin dynamics at 300K
dyn = Langevin(
    atoms,
    timestep=0.5 * units.fs,
    temperature_K=300,
    friction=0.01 / units.fs
)

dyn.run(1000)  # 1000 steps
```

## CLI Usage

MLFF Distiller provides three command-line tools:

### mlff-train

Train a student model from scratch or resume training:

```bash
# Train with default settings
mlff-train --dataset data/training.h5 --epochs 100

# Train with custom parameters
mlff-train --dataset data/training.h5 \
    --hidden-dim 128 \
    --num-interactions 3 \
    --epochs 200 \
    --batch-size 64 \
    --lr 1e-4 \
    --device cuda

# Resume from checkpoint
mlff-train --resume checkpoints/checkpoint_epoch_50.pt
```

### mlff-validate

Validate a trained model with NVE MD simulation:

```bash
# Basic validation with water molecule
mlff-validate --checkpoint checkpoints/best_model.pt

# Custom validation
mlff-validate --checkpoint checkpoints/best_model.pt \
    --molecule CH4 \
    --steps 5000 \
    --temperature 300 \
    --timestep 0.5 \
    --output validation_results.json
```

### mlff-benchmark

Benchmark model inference performance:

```bash
# Basic benchmark
mlff-benchmark --checkpoint checkpoints/best_model.pt

# Detailed benchmark
mlff-benchmark --checkpoint checkpoints/best_model.pt \
    --device cuda \
    --warmup 50 \
    --iterations 200 \
    --output benchmark_results.json
```

## Loading Structures from Files

### From SDF Files

```python
from ase.io import read

# Load molecule from SDF
atoms = read('molecule.sdf')
atoms.calc = calc

energy = atoms.get_potential_energy()
```

### From XYZ Files

```python
# Load from XYZ
atoms = read('structure.xyz')
atoms.calc = calc

forces = atoms.get_forces()
```

### From Trajectory Files

```python
from ase.io import read

# Load all frames from trajectory
trajectory = read('md_trajectory.traj', index=':')

# Calculate properties for each frame
for atoms in trajectory:
    atoms.calc = calc
    print(f"E = {atoms.get_potential_energy():.4f} eV")
```

## Performance Tips

### Use GPU When Available

```python
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
calc = StudentForceFieldCalculator('checkpoints/best_model.pt', device=device)
```

### Enable Mixed Precision (FP16)

For faster inference on compatible GPUs:

```python
calc = StudentForceFieldCalculator(
    'checkpoints/best_model.pt',
    device='cuda',
    use_fp16=True  # Enable FP16 mixed precision
)
```

### Use Batch Calculations

For multiple structures, batch calculation is significantly faster:

```python
# Slower: iterate over structures
for atoms in atoms_list:
    atoms.calc = calc
    energy = atoms.get_potential_energy()

# Faster: batch calculation
results = calc.calculate_batch(atoms_list, properties=['energy', 'forces'])
```

### Enable Timing Statistics

```python
calc = StudentForceFieldCalculator(
    'checkpoints/best_model.pt',
    enable_timing=True
)

# After calculations
stats = calc.get_timing_stats()
print(f"Average inference time: {stats['avg_time']*1000:.2f} ms")
```

## Troubleshooting

### CUDA Out of Memory

If you run out of GPU memory:

1. Use CPU instead: `device='cpu'`
2. Reduce batch size for batch calculations
3. Use a smaller model variant (tiny or ultra-tiny)

### Model Not Found

Ensure the checkpoint path is correct:

```python
from pathlib import Path

checkpoint = Path('checkpoints/best_model.pt')
if not checkpoint.exists():
    print(f"Checkpoint not found at {checkpoint.absolute()}")
```

### Slow Performance on CPU

CPU inference is slower than GPU. For production use:

1. Use a GPU if available
2. Enable torch.compile() optimization (requires PyTorch 2.0+)
3. Use batch calculations instead of individual calls

## Next Steps

- Explore the [example notebooks](../notebooks/examples/) for detailed tutorials
- Read the [API Reference](API.md) for complete documentation
- Check the [CHANGELOG](../CHANGELOG.md) for version history

## Support

- GitHub Issues: [Report bugs](https://github.com/atfrank/MLFF_Distiller/issues)
- Documentation: [Full docs](../docs/)
