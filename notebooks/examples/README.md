# MLFF Distiller Example Notebooks

This directory contains Jupyter notebooks demonstrating key workflows for the MLFF Distiller project.

## Prerequisites

Before running these notebooks, ensure you have:

1. **Trained model checkpoint**: `checkpoints/best_model.pt` (427K parameters, PRODUCTION APPROVED)
2. **Python environment** with required packages:
   - PyTorch (CPU or CUDA)
   - ASE (Atomic Simulation Environment)
   - Matplotlib
   - NumPy

3. **Optional** (for teacher model comparison):
   - orb-models package (`pip install orb-models`)

## Quick Start

```bash
# From the project root
cd notebooks/examples
jupyter notebook
```

## Notebook Index

### 1. Quick Start (`01_quick_start.ipynb`)

**Purpose**: Introduction to loading and using the student model for energy/force calculations.

**Topics Covered**:
- Loading the production model
- Creating an ASE calculator
- Calculating energies and forces on molecules
- Working with different molecular structures
- Batch calculations

**Runtime**: 2-5 minutes (CPU), <1 minute (GPU)

**Key Code**:
```python
from mlff_distiller import StudentForceFieldCalculator
from ase.build import molecule

calc = StudentForceFieldCalculator('checkpoints/best_model.pt', device='cuda')
atoms = molecule('H2O')
atoms.calc = calc

energy = atoms.get_potential_energy()  # eV
forces = atoms.get_forces()            # eV/Angstrom
```

---

### 2. MD Simulation (`02_md_simulation.ipynb`)

**Purpose**: Running NVE molecular dynamics simulations with energy conservation analysis.

**Topics Covered**:
- Setting up NVE MD simulations
- Using the `NVEMDHarness` for convenient simulation runs
- Analyzing energy conservation
- Visualizing energy evolution and temperature
- Manual MD with ASE's VelocityVerlet

**Runtime**: 5-10 minutes (CPU), 2-3 minutes (GPU)

**Key Code**:
```python
from mlff_distiller import StudentForceFieldCalculator, NVEMDHarness
from ase.build import molecule

calc = StudentForceFieldCalculator('checkpoints/best_model.pt', device='cuda')
atoms = molecule('C2H6')

harness = NVEMDHarness(
    atoms=atoms,
    calculator=calc,
    temperature=300.0,  # Kelvin
    timestep=0.5        # femtoseconds
)

results = harness.run_simulation(steps=1000)
print(f"Energy drift: {results['energy_drift_pct']:.4f}%")
```

---

### 3. Model Comparison (`03_model_comparison.ipynb`)

**Purpose**: Comparing student model predictions against teacher model (or synthetic data).

**Topics Covered**:
- Setting up student and teacher calculators
- Computing force predictions across multiple configurations
- Analyzing force correlation and R-squared metrics
- Visualizing parity plots and error distributions
- Per-molecule and per-component analysis

**Runtime**: 5-10 minutes (depends on teacher model availability)

**Key Metrics**:
- **R-squared > 0.95**: Good force correlation
- **Force RMSE < 0.1 eV/A**: High accuracy for MD
- **Slope near 1.0**: No systematic bias

---

### 4. Benchmarking (`04_benchmarking.ipynb`)

**Purpose**: Benchmarking inference performance for optimization decisions.

**Topics Covered**:
- Single-structure inference benchmarks
- GPU vs CPU performance comparison
- System size scaling analysis
- Batch inference optimization
- Memory usage analysis

**Runtime**: 5-15 minutes (varies by hardware)

**Key Findings**:
- GPU provides 5-20x speedup over CPU
- Batch inference is more efficient for multiple structures
- Throughput can reach 100s-1000s of structures/second on GPU

---

## Output Files

Running these notebooks may generate the following files:
- `md_energy_evolution.png`: Energy conservation plot from MD simulation
- `force_correlation.png`: Force parity plot from model comparison
- `benchmark_performance.png`: Performance benchmark visualization

## Tips

1. **Device Selection**: Notebooks automatically detect CUDA availability. Force CPU with `device='cpu'` if needed.

2. **Memory Management**: For large systems, consider clearing GPU cache:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

3. **Reproducibility**: Set random seeds for reproducible results:
   ```python
   import numpy as np
   import torch
   np.random.seed(42)
   torch.manual_seed(42)
   ```

4. **Logging**: Enable logging for debugging:
   ```python
   import logging
   logging.basicConfig(level=logging.INFO)
   ```

## Related Documentation

- [Main README](../../README.md): Project overview
- [Testing Guide](../../TESTING.md): Test suite documentation
- [API Reference](../../docs/): Detailed API documentation

## Issues

If you encounter problems running these notebooks, please check:

1. Model checkpoint exists at `checkpoints/best_model.pt`
2. Required packages are installed
3. CUDA drivers are up to date (for GPU)

For additional support, open an issue on the project repository.
