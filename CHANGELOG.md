# Changelog

All notable changes to MLFF Distiller will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-11-25

### Added

#### Core Model
- Production-validated PaiNN-based student model (427K parameters)
- Rotationally and translationally equivariant architecture
- Permutation invariant for same-species atoms
- Gaussian radial basis functions with cosine cutoff
- 3-layer message passing architecture with hidden dimension 128
- Support for elements H, C, N, O, F, S, Cl (Z=1-17)

#### ASE Calculator Integration
- `StudentForceFieldCalculator` - Full ASE Calculator compliance
- Batch inference support via `calculate_batch()` method
- Optional stress tensor computation
- FP16 mixed precision support for faster GPU inference
- TorchScript JIT compilation support
- Performance timing and statistics tracking
- Memory-efficient tensor buffer management

#### MD Testing Framework
- `NVEMDHarness` - NVE (microcanonical) MD simulation harness
- Comprehensive energy conservation metrics
- Force accuracy metrics (RMSE, MAE, R^2, angular error)
- Trajectory analysis utilities (RMSD, bond lengths, temperature)
- Automatic velocity initialization with Maxwell-Boltzmann distribution

#### CLI Tools
- `mlff-train` - Train student models from HDF5 datasets
- `mlff-validate` - Validate models with NVE MD simulations
- `mlff-benchmark` - Benchmark inference performance

#### Example Notebooks
- `01_quick_start.ipynb` - Loading models and basic calculations
- `02_md_simulation.ipynb` - Running MD simulations
- `03_model_comparison.ipynb` - Student vs teacher comparison
- `04_benchmarking.ipynb` - Performance benchmarking

#### CI/CD Pipeline
- GitHub Actions workflow for testing
- Automated linting with ruff
- Code coverage with codecov
- Multi-Python version testing (3.10, 3.11, 3.12)

### Model Performance

Production model metrics (validated on unseen molecules):

| Metric | Value |
|--------|-------|
| Parameters | 427,292 |
| Checkpoint Size | 1.7 MB |
| Force R^2 | 0.9958 |
| Force RMSE | 0.16 eV/Angstrom |
| Force MAE | 0.11 eV/Angstrom |
| Angular Error | 9.6 degrees |
| Energy Drift (10ps NVE) | 0.14% |
| Status | **PRODUCTION APPROVED** |

### Model Variants

| Model | Parameters | Force R^2 | Status |
|-------|-----------|----------|--------|
| Original (production) | 427K | 0.9958 | Recommended |
| Tiny | 77K | 0.38 | Experimental |
| Ultra-tiny | 21K | 0.15 | Energy-only |

### Dependencies

- Python 3.10+
- PyTorch 2.0+
- ASE (Atomic Simulation Environment)
- NumPy
- h5py (for dataset handling)
- Optional: torch-cluster (for optimized neighbor search)
- Optional: CUDA 11.8+ (for GPU acceleration)

### Known Limitations

- Periodic boundary conditions (PBC) support is basic; works for small cells
- Stress tensor computation is experimental
- Training requires pre-generated HDF5 datasets with teacher labels
- Model trained primarily on organic molecules; may have reduced accuracy for metals

### Contributors

- ML Architecture Specialist - Model architecture and implementation
- Data Pipeline Engineer - Dataset generation and processing
- Testing & Benchmarking Engineer - MD validation framework
- CUDA Optimization Engineer - Performance optimization

---

## [Unreleased]

### Planned for v0.2.0
- Improved PBC support with proper image handling
- LAMMPS pair_style integration
- Expanded element support
- Tiny model architecture improvements
- ONNX export optimization
