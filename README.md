# MLFF Distiller

[![CI](https://github.com/atfrank/MLFF_Distiller/workflows/CI/badge.svg)](https://github.com/atfrank/MLFF_Distiller/actions)
[![codecov](https://codecov.io/gh/atfrank/MLFF_Distiller/branch/main/graph/badge.svg)](https://codecov.io/gh/atfrank/MLFF_Distiller)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Fast, CUDA-optimized distilled force fields from Orb-models and FeNNol-PMC for accelerated molecular dynamics simulations.

## Overview

MLFF Distiller creates high-performance student models that maintain the accuracy of state-of-the-art machine learning force fields (Orb-models, FeNNol-PMC) while achieving 5-10x faster inference through:

- Knowledge distillation from teacher models
- CUDA-optimized inference kernels
- Efficient student architectures
- Production-ready deployment tools

## Project Goals

1. **Performance**: 5-10x faster inference compared to teacher models
2. **Accuracy**: Maintain >95% accuracy on energy, force, and stress predictions
3. **Compatibility**: Accept same inputs as original models (atomic positions, species, cells)
4. **Production-Ready**: Packaged, documented, and benchmarked for real-world use

## Repository Structure

```
MLFF_Distiller/
├── src/
│   ├── data/           # Data generation and preprocessing
│   ├── models/         # Student and teacher model architectures
│   ├── training/       # Distillation training pipelines
│   ├── cuda/           # CUDA optimization kernels
│   ├── inference/      # Inference engines and APIs
│   └── utils/          # Shared utilities
├── tests/
│   ├── unit/           # Unit tests
│   └── integration/    # Integration tests
├── benchmarks/         # Performance benchmarking tools
├── docs/               # Documentation
├── examples/           # Usage examples
└── .github/            # CI/CD and issue templates
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/atfrank/MLFF_Distiller.git
cd MLFF_Distiller

# Install dependencies
pip install -e ".[dev]"

# For CUDA support
pip install -e ".[cuda]"
```

### Basic Usage

```python
from mlff_distiller.models import load_student_model
from mlff_distiller.inference import run_inference

# Load a distilled model
model = load_student_model("orb-v2-distilled")

# Run inference
energy, forces, stress = run_inference(
    model,
    positions=atomic_positions,
    species=atomic_numbers,
    cell=unit_cell
)
```

## Development Workflow

### For Contributors

1. Check the [GitHub Projects board](https://github.com/atfrank/MLFF_Distiller/projects) for available tasks
2. Pick an issue assigned to your agent specialty
3. Create a feature branch: `git checkout -b feature/your-feature-name`
4. Implement changes following our [Contributing Guidelines](CONTRIBUTING.md)
5. Run tests: `pytest tests/`
6. Submit a PR with clear description and linked issues

### For Specialized Agents

This project uses a multi-agent development approach:

- **Data Pipeline Engineer**: Data generation, preprocessing, and dataset management
- **ML Architecture Designer**: Model architecture design and optimization
- **Distillation Training Engineer**: Training pipelines and loss functions
- **CUDA Optimization Engineer**: Performance optimization and CUDA kernels
- **Testing & Benchmark Engineer**: Testing frameworks and performance benchmarking

See [AGENT_PROTOCOLS.md](docs/AGENT_PROTOCOLS.md) for detailed agent workflows.

## Milestones

- **M1: Setup & Baseline** (Weeks 1-2)
  - Repository infrastructure
  - Teacher model integration
  - Initial benchmarks

- **M2: Data Pipeline** (Weeks 3-4)
  - Data generation from teacher models
  - Preprocessing and augmentation
  - Dataset validation

- **M3: Model Architecture** (Weeks 5-6)
  - Student architecture design
  - Teacher-student interface
  - Model validation

- **M4: Distillation Training** (Weeks 7-9)
  - Training pipeline implementation
  - Loss function tuning
  - Convergence validation

- **M5: CUDA Optimization** (Weeks 10-12)
  - Kernel optimization
  - Memory efficiency
  - Performance benchmarking

- **M6: Testing & Deployment** (Weeks 13-14)
  - Comprehensive testing
  - Documentation
  - Release preparation

## Performance Targets

| Metric | Target | Baseline (Teacher) | Current Best |
|--------|--------|-------------------|--------------|
| Inference Speed | 5-10x faster | 1x | TBD |
| Energy MAE | <0.05 eV/atom | 0 (reference) | TBD |
| Force MAE | <0.1 eV/Å | 0 (reference) | TBD |
| Memory Usage | <2GB | ~5GB | TBD |

## Citation

If you use MLFF Distiller in your research, please cite:

```bibtex
@software{mlff_distiller,
  title = {MLFF Distiller: Fast CUDA-Optimized Distilled Force Fields},
  author = {ML Force Field Distillation Team},
  year = {2025},
  url = {https://github.com/atfrank/MLFF_Distiller}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Orb-models team for the original force field implementations
- FeNNol-PMC contributors
- PyTorch and CUDA communities

## Contact

- GitHub Issues: [Report bugs or request features](https://github.com/atfrank/MLFF_Distiller/issues)
- Project Board: [Track development progress](https://github.com/atfrank/MLFF_Distiller/projects)

---

**Status**: Active Development | **Version**: 0.1.0 | **Last Updated**: 2025-11-23
