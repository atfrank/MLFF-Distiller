# MLFF Distiller

[![CI](https://github.com/atfrank/MLFF_Distiller/workflows/CI/badge.svg)](https://github.com/atfrank/MLFF_Distiller/actions)
[![codecov](https://codecov.io/gh/atfrank/MLFF_Distiller/branch/main/graph/badge.svg)](https://codecov.io/gh/atfrank/MLFF_Distiller)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**MLFF Distiller** is a production-ready toolkit for creating fast, compact student models distilled from state-of-the-art machine learning force fields. The distilled models serve as drop-in replacements for teacher models like Orb-v2, achieving 5-10x faster inference while maintaining high accuracy for molecular dynamics (MD) simulations.

The project provides a complete pipeline for training, validating, and deploying distilled force field models, including ASE calculator integration, MD testing frameworks, and CLI tools for seamless workflow automation.

## Installation

### From Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/atfrank/MLFF_Distiller.git
cd MLFF_Distiller

# Install with development dependencies
pip install -e ".[dev]"

# For CUDA support (recommended for faster inference)
pip install -e ".[cuda]"
```

### From PyPI (Coming Soon)

```bash
pip install mlff-distiller
```

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- ASE (Atomic Simulation Environment)
- CUDA 11.8+ (optional, for GPU acceleration)

## Quick Start

```python
from mlff_distiller import StudentForceFieldCalculator
from ase.build import molecule
from ase.md.verlet import VelocityVerlet
from ase import units

# Load the production model
calc = StudentForceFieldCalculator(
    checkpoint_path='checkpoints/best_model.pt',
    device='cuda'  # or 'cpu'
)

# Create a molecule and calculate properties
atoms = molecule('H2O')
atoms.calc = calc

energy = atoms.get_potential_energy()  # eV
forces = atoms.get_forces()            # eV/Angstrom

# Run MD simulation
VelocityVerlet(atoms, timestep=0.5*units.fs).run(1000)
```

See `notebooks/examples/` for comprehensive tutorials:
- [`01_quick_start.ipynb`](notebooks/examples/01_quick_start.ipynb) - Loading models and basic calculations
- [`02_md_simulation.ipynb`](notebooks/examples/02_md_simulation.ipynb) - Running MD simulations
- [`03_model_comparison.ipynb`](notebooks/examples/03_model_comparison.ipynb) - Student vs teacher comparison
- [`04_benchmarking.ipynb`](notebooks/examples/04_benchmarking.ipynb) - Performance benchmarking

## Model Performance

### Production Model (v0.1.0)

| Metric | Value | Status |
|--------|-------|--------|
| Parameters | 427K | Compact |
| Checkpoint Size | 1.7 MB | Lightweight |
| Force R^2 | 0.9958 | Excellent |
| Force RMSE | 0.16 eV/A | Excellent |
| Energy Drift (10ps NVE) | 0.14% | Stable |
| Status | **PRODUCTION APPROVED** | Ready |

### Model Variants

| Model | Parameters | Size | Force R^2 | Use Case |
|-------|-----------|------|----------|----------|
| **Original** | 427K | 1.7 MB | 0.9958 | Production MD |
| Tiny | 77K | 0.3 MB | 0.38 | Quick screening |
| Ultra-tiny | 21K | 0.08 MB | 0.15 | Energy-only |

## CLI Tools

MLFF Distiller provides three CLI commands:

```bash
# Train a student model
mlff-train --dataset data/training.h5 --epochs 100

# Validate with NVE MD simulation
mlff-validate --checkpoint checkpoints/best_model.pt --molecule H2O --steps 1000

# Benchmark inference performance
mlff-benchmark --checkpoint checkpoints/best_model.pt --device cuda
```

## Documentation

- [Quick Start Guide](docs/QUICK_START.md) - Detailed getting started instructions
- [API Reference](docs/API.md) - Complete API documentation
- [Changelog](CHANGELOG.md) - Version history and release notes

## Project Structure

```
MLFF_Distiller/
├── src/mlff_distiller/     # Main package
│   ├── models/             # Student model architectures
│   ├── inference/          # ASE calculator interface
│   ├── testing/            # MD validation framework
│   ├── training/           # Distillation training
│   └── cli/                # CLI entry points
├── checkpoints/            # Trained model checkpoints
├── notebooks/examples/     # Jupyter notebook tutorials
├── tests/                  # Test suite
└── docs/                   # Documentation
```

## Citation

If you use MLFF Distiller in your research, please cite:

```bibtex
@software{mlff_distiller,
  title = {MLFF Distiller: Fast Distilled Force Fields for Molecular Dynamics},
  author = {MLFF Distiller Development Team},
  year = {2025},
  url = {https://github.com/atfrank/MLFF_Distiller},
  version = {0.1.0}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Orb-models team for teacher model implementations
- PyTorch and ASE communities
- PaiNN architecture from Schütt et al. (2021)

## Contact

- GitHub Issues: [Report bugs or request features](https://github.com/atfrank/MLFF_Distiller/issues)
- Documentation: [docs/](docs/)

---

**Version**: 0.1.0 | **Status**: Production Ready | **Last Updated**: 2025-11-25
