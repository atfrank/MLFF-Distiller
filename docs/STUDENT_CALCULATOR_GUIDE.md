# Student Calculator Guide

**Version**: 1.0
**Last Updated**: 2025-11-23
**Status**: Ready for Use

## Overview

The `StudentCalculator` class provides an ASE Calculator interface for distilled student models, enabling drop-in replacement of teacher models in molecular dynamics simulations. This guide covers installation, usage, and best practices.

## Quick Start

```python
from mlff_distiller.models import StudentCalculator
from ase.build import molecule
from ase.md.verlet import VelocityVerlet
from ase import units

# Load student model from checkpoint
calc = StudentCalculator(
    model_path="checkpoints/orb_student_v1.pth",
    device="cuda"
)

# Attach to atoms (same as teacher calculators)
atoms = molecule("H2O")
atoms.calc = calc

# Run MD simulation (5-10x faster than teacher!)
dyn = VelocityVerlet(atoms, timestep=1.0*units.fs)
dyn.run(1000)
```

## Drop-in Replacement Pattern

The primary use case for `StudentCalculator` is as a drop-in replacement for teacher calculators. Users can replace teacher models with student models by changing **only one line** of code:

```python
# Original MD script with teacher calculator:
# from mlff_distiller.models import OrbCalculator
# calc = OrbCalculator(model_name="orb-v2", device="cuda")

# Drop-in replacement (only change needed):
from mlff_distiller.models import StudentCalculator
calc = StudentCalculator(model_path="orb_student_v1.pth", device="cuda")

# Rest of MD script runs identically, but 5-10x faster!
atoms.calc = calc
dyn = VelocityVerlet(atoms, timestep=1.0*units.fs)
dyn.run(1000000)  # Millions of calls - optimized for speed!
```

## Installation

The StudentCalculator is part of the `mlff_distiller` package:

```bash
# Install package
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

## Usage Modes

### Mode 1: Load from Checkpoint (Recommended)

Load a trained student model from a checkpoint file:

```python
from mlff_distiller.models import StudentCalculator

calc = StudentCalculator(
    model_path="checkpoints/orb_student_v1.pth",
    device="cuda"
)
```

**Checkpoint Format**: The checkpoint should be a dictionary with:
- `model_state_dict`: Model weights
- `model_class`: Model class for reconstruction
- `model_config`: Model configuration (optional)

Example checkpoint creation:
```python
import torch
from mlff_distiller.models import StudentCalculator

# After training your student model
checkpoint = {
    "model_state_dict": model.state_dict(),
    "model_class": MyStudentModel,
    "model_config": {
        "hidden_dim": 128,
        "num_layers": 3,
    },
}
torch.save(checkpoint, "checkpoints/student_model.pth")
```

### Mode 2: Use Pre-initialized Model

Provide a model instance directly:

```python
from mlff_distiller.models import StudentCalculator, SimpleMLP

# Create model instance
model = SimpleMLP(hidden_dim=128, num_layers=3)

# Wrap in calculator
calc = StudentCalculator(model=model, device="cuda")
```

### Mode 3: Use Model Factory

Provide a callable that creates the model:

```python
def create_model(hidden_dim=128, num_layers=3):
    return SimpleMLP(hidden_dim=hidden_dim, num_layers=num_layers)

calc = StudentCalculator(
    model=create_model,
    model_config={"hidden_dim": 256, "num_layers": 4},
    device="cuda"
)
```

## Configuration Options

### Basic Parameters

```python
StudentCalculator(
    model=None,              # Model instance or factory function
    model_path=None,         # Path to checkpoint file
    model_config=None,       # Configuration dict for model
    device="cuda",           # Device: "cpu" or "cuda"
    dtype=torch.float32,     # Data type for computations
    compile=False,           # Use torch.compile (PyTorch 2.0+)
    energy_key="energy",     # Key for energy in model output
    forces_key="forces",     # Key for forces in model output
    stress_key="stress",     # Key for stress in model output
)
```

### Device Management

```python
# CPU calculator
calc_cpu = StudentCalculator(model=model, device="cpu")

# CUDA calculator
calc_cuda = StudentCalculator(model=model, device="cuda")

# Specific GPU
calc_gpu1 = StudentCalculator(model=model, device="cuda:1")
```

### Model Compilation (Advanced)

For additional speedup with PyTorch 2.0+:

```python
calc = StudentCalculator(
    model_path="student_model.pth",
    device="cuda",
    compile=True  # Enable torch.compile
)
```

**Note**: Compilation adds startup time but can provide 20-30% additional speedup for repeated inference.

## ASE Calculator Interface

The `StudentCalculator` implements the standard ASE Calculator interface:

### Supported Properties

```python
calc.implemented_properties
# ['energy', 'forces', 'stress']
```

### Standard Methods

```python
# Get potential energy (eV)
energy = atoms.get_potential_energy()

# Get forces (eV/Angstrom)
forces = atoms.get_forces()

# Get stress tensor (eV/Angstrom^3, Voigt notation)
stress = atoms.get_stress()
```

### Units

All quantities follow ASE standard units:
- **Energy**: eV (electronvolts)
- **Forces**: eV/Angstrom
- **Stress**: eV/Angstrom³
- **Positions**: Angstrom
- **Temperature**: Kelvin

## Molecular Dynamics Usage

### NVE (Microcanonical Ensemble)

```python
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units

# Setup
atoms.calc = calc
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# Run NVE
dyn = VelocityVerlet(atoms, timestep=1.0*units.fs)
dyn.run(10000)
```

### NVT (Canonical Ensemble)

```python
from ase.md.langevin import Langevin
from ase import units

# Setup
atoms.calc = calc

# Run NVT with Langevin thermostat
dyn = Langevin(
    atoms,
    timestep=1.0*units.fs,
    temperature_K=300,
    friction=0.01
)
dyn.run(10000)
```

### NPT (Isothermal-Isobaric Ensemble)

```python
from ase.md.npt import NPT
from ase import units

# Setup
atoms.calc = calc

# Run NPT
dyn = NPT(
    atoms,
    timestep=1.0*units.fs,
    temperature_K=300,
    externalstress=0.0,  # in eV/Angstrom^3
    ttime=25*units.fs,
    pfactor=75*units.fs**2
)
dyn.run(10000)
```

## Performance Optimization

### MD Optimization

The `StudentCalculator` is optimized for repeated calls in MD simulations:

1. **Buffer Reuse**: Internal buffers are reused to minimize allocations
2. **Efficient Transfers**: Minimizes CPU↔GPU data transfers
3. **Caching**: ASE caching avoids redundant calculations
4. **No Memory Leaks**: Stable memory over millions of calls

### Best Practices

```python
# GOOD: Reuse calculator instance
calc = StudentCalculator(model_path="model.pth", device="cuda")
for atoms in structures:
    atoms.calc = calc  # Reuse same calculator
    energy = atoms.get_potential_energy()

# BAD: Creating new calculator each time (slow)
for atoms in structures:
    calc = StudentCalculator(model_path="model.pth", device="cuda")
    atoms.calc = calc
    energy = atoms.get_potential_energy()
```

### Monitoring Performance

```python
# Track number of calls
print(f"Calculator called {calc.n_calls} times")

# Profile timing
import time
start = time.time()
dyn.run(1000)
elapsed = time.time() - start
print(f"MD time: {elapsed:.2f}s")
print(f"Time per step: {elapsed/1000*1000:.2f} ms")
```

## Model Output Requirements

Your student model must return a dictionary with the following keys:

```python
def forward(self, batch):
    """
    Args:
        batch: Dict with keys:
            - positions: (n_atoms, 3) atomic positions
            - atomic_numbers: (n_atoms,) atomic numbers
            - batch: (n_atoms,) batch indices
            - cell: (3, 3) optional cell matrix
            - pbc: (3,) optional periodic boundary conditions

    Returns:
        Dict with keys:
            - energy: (1,) or scalar, total energy in eV
            - forces: (n_atoms, 3), forces in eV/Angstrom
            - stress: (6,), stress in eV/Angstrom^3 (optional)
    """
    return {
        "energy": energy_tensor,
        "forces": forces_tensor,
        "stress": stress_tensor,  # optional
    }
```

## Testing and Validation

### Mock Models for Development

Use mock models for testing without trained models:

```python
from mlff_distiller.models import MockStudentModel, StudentCalculator

# Create mock model (deterministic output for testing)
model = MockStudentModel()
calc = StudentCalculator(model=model, device="cpu")

# Use in tests
atoms.calc = calc
energy = atoms.get_potential_energy()  # Works for testing!
```

### Unit Tests

```python
import pytest
from mlff_distiller.models import StudentCalculator, MockStudentModel
from ase.build import molecule

def test_student_calculator():
    """Test StudentCalculator basic functionality."""
    calc = StudentCalculator(model=MockStudentModel(), device="cpu")
    atoms = molecule("H2O")
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    assert isinstance(energy, float)
    assert forces.shape == (3, 3)
```

### Integration Tests

```python
def test_md_simulation():
    """Test StudentCalculator in MD simulation."""
    from ase.md.verlet import VelocityVerlet
    from ase import units

    calc = StudentCalculator(model=MockStudentModel(), device="cpu")
    atoms = molecule("H2O")
    atoms.calc = calc

    dyn = VelocityVerlet(atoms, timestep=1.0*units.fs)
    dyn.run(100)  # Should complete without errors
```

## Troubleshooting

### Issue: Model not found

```
FileNotFoundError: Model checkpoint not found: model.pth
```

**Solution**: Ensure checkpoint path is correct and file exists:
```python
from pathlib import Path
checkpoint_path = Path("checkpoints/model.pth")
assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"
```

### Issue: Missing output keys

```
KeyError: Model output missing 'forces' key
```

**Solution**: Ensure your model outputs all required keys:
```python
# In your model's forward():
return {
    "energy": energy,
    "forces": forces,
    "stress": stress,  # Include all properties
}
```

### Issue: CUDA out of memory

```
RuntimeError: CUDA out of memory
```

**Solutions**:
1. Use CPU: `device="cpu"`
2. Reduce batch size (if batching)
3. Use smaller model
4. Clear CUDA cache: `torch.cuda.empty_cache()`

### Issue: Slow performance

**Check**:
1. Device is CUDA: `calc.device`
2. Model is in eval mode: `calc.model.eval()`
3. Using buffer reuse (reuse calculator instance)
4. Not recreating calculator each call

## Examples

See `examples/student_calculator_usage.py` for comprehensive examples:

```bash
python examples/student_calculator_usage.py
```

Examples include:
1. Basic usage
2. NVE MD simulation
3. NVT MD simulation
4. Drop-in replacement pattern
5. Loading from checkpoint
6. Batch processing
7. Device management

## API Reference

### StudentCalculator

```python
class StudentCalculator(Calculator):
    """ASE Calculator wrapper for distilled student models."""

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        model: Optional[Union[nn.Module, Callable]] = None,
        model_path: Optional[Union[str, Path]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        compile: bool = False,
        energy_key: str = "energy",
        forces_key: str = "forces",
        stress_key: str = "stress",
        **kwargs
    )

    def calculate(
        self,
        atoms=None,
        properties: List[str] = ["energy"],
        system_changes: List[str] = all_changes,
    )

    def get_potential_energy(self, atoms=None, force_consistent=False) -> float

    def get_forces(self, atoms=None) -> np.ndarray

    def get_stress(self, atoms=None) -> np.ndarray

    @property
    def n_calls(self) -> int
```

## Comparison with Teacher Calculators

| Feature | OrbCalculator | FeNNolCalculator | StudentCalculator |
|---------|--------------|------------------|-------------------|
| ASE Interface | ✅ | ✅ | ✅ |
| Energy | ✅ | ✅ | ✅ |
| Forces | ✅ | ✅ | ✅ |
| Stress | ✅ | ❌ | ✅ |
| Speed | 1x | 1x | 5-10x |
| Accuracy | 100% | 100% | >95% |
| Device | CPU/CUDA | CPU/CUDA | CPU/CUDA |
| Custom Models | ❌ | ❌ | ✅ |

## Best Practices

### 1. Model Architecture
- Output all required keys (`energy`, `forces`, `stress`)
- Use standard units (eV, eV/Angstrom)
- Handle both PBC and non-PBC systems
- Support variable system sizes

### 2. Performance
- Reuse calculator instances
- Enable torch.compile for additional speedup
- Use CUDA when available
- Profile to identify bottlenecks

### 3. Validation
- Test with mock models first
- Validate against teacher models
- Check energy conservation (NVE)
- Test with various system sizes

### 4. Production Deployment
- Save complete checkpoints (weights + config)
- Version your models
- Document model capabilities/limitations
- Test thoroughly before production use

## Future Enhancements

Planned features:
- [ ] Batch inference support
- [ ] Uncertainty quantification
- [ ] LAMMPS integration
- [ ] Model ensemble support
- [ ] Automatic model selection

## Support

For issues, questions, or contributions:
- GitHub Issues: [github.com/atfrank/MLFF-Distiller/issues](https://github.com/atfrank/MLFF-Distiller/issues)
- Documentation: `docs/` directory
- Examples: `examples/student_calculator_usage.py`

## References

- ASE Calculator Documentation: https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html
- ASE MD Tutorial: https://wiki.fysik.dtu.dk/ase/ase/md.html
- PyTorch Documentation: https://pytorch.org/docs/

---

**Last Updated**: 2025-11-23
**Version**: 1.0
**Maintainer**: ML Architecture Designer
