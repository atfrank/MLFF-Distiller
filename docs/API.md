# API Reference

This document provides complete API documentation for MLFF Distiller.

## Table of Contents

- [Models](#models)
  - [StudentForceField](#studentforcefield)
- [Inference](#inference)
  - [StudentForceFieldCalculator](#studentforcefieldcalculator)
- [Testing](#testing)
  - [NVEMDHarness](#nvemdhardness)
  - [Energy Metrics](#energy-metrics)
  - [Force Metrics](#force-metrics)
  - [Trajectory Analysis](#trajectory-analysis)

---

## Models

### StudentForceField

```python
from mlff_distiller.models import StudentForceField
```

PaiNN-based student model for ML force field distillation. This model predicts potential energy surfaces and forces for molecular and materials systems.

#### Constructor

```python
StudentForceField(
    hidden_dim: int = 128,
    num_interactions: int = 3,
    num_rbf: int = 20,
    cutoff: float = 5.0,
    max_z: int = 118,
    learnable_rbf: bool = False,
    use_torch_cluster: bool = True
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dim` | int | 128 | Dimension of hidden features |
| `num_interactions` | int | 3 | Number of message passing blocks |
| `num_rbf` | int | 20 | Number of radial basis functions |
| `cutoff` | float | 5.0 | Cutoff distance for neighbors (Angstrom) |
| `max_z` | int | 118 | Maximum atomic number supported |
| `learnable_rbf` | bool | False | Whether RBF parameters are learnable |
| `use_torch_cluster` | bool | True | Use torch-cluster for optimized neighbor search |

#### Methods

##### forward

```python
forward(
    atomic_numbers: torch.Tensor,
    positions: torch.Tensor,
    cell: Optional[torch.Tensor] = None,
    pbc: Optional[torch.Tensor] = None,
    batch: Optional[torch.Tensor] = None
) -> torch.Tensor
```

Forward pass to predict total energy.

**Parameters:**

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `atomic_numbers` | [N] | Atomic numbers (int64) |
| `positions` | [N, 3] | Atomic positions in Angstrom (float32) |
| `cell` | [3, 3] or [B, 3, 3] | Unit cell (optional) |
| `pbc` | [3] or [B, 3] | Periodic boundary conditions (optional) |
| `batch` | [N] | Batch indices for batched structures (optional) |

**Returns:**

- `torch.Tensor`: Total energy in eV (scalar or [batch_size])

##### predict_energy_and_forces

```python
predict_energy_and_forces(
    atomic_numbers: torch.Tensor,
    positions: torch.Tensor,
    cell: Optional[torch.Tensor] = None,
    pbc: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]
```

Predict both energy and forces. Forces computed via autograd: F = -nabla E.

**Parameters:**

Same as `forward()`.

**Returns:**

- `energy` (torch.Tensor): Total energy in eV
- `forces` (torch.Tensor): Atomic forces [N, 3] in eV/Angstrom

##### save

```python
save(path: Union[str, Path])
```

Save model checkpoint to file.

##### load (classmethod)

```python
@classmethod
load(path: Union[str, Path], device: str = 'cpu') -> StudentForceField
```

Load model from checkpoint.

**Example:**

```python
# Create and save model
model = StudentForceField(hidden_dim=128, num_interactions=3)
model.save('my_model.pt')

# Load model
model = StudentForceField.load('my_model.pt', device='cuda')
```

##### num_parameters

```python
num_parameters() -> int
```

Count total number of trainable parameters.

---

## Inference

### StudentForceFieldCalculator

```python
from mlff_distiller.inference import StudentForceFieldCalculator
# or
from mlff_distiller import StudentForceFieldCalculator
```

Production ASE Calculator for StudentForceField model. Provides drop-in replacement for ASE-based workflows.

#### Constructor

```python
StudentForceFieldCalculator(
    checkpoint_path: Union[str, Path],
    device: str = 'cuda',
    dtype: torch.dtype = torch.float32,
    enable_stress: bool = False,
    batch_size: Optional[int] = None,
    enable_timing: bool = False,
    use_compile: bool = False,
    use_fp16: bool = False,
    use_jit: bool = False,
    jit_path: Optional[Union[str, Path]] = None,
    use_torch_cluster: bool = True,
    use_analytical_forces: bool = False
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `checkpoint_path` | str/Path | required | Path to trained model checkpoint |
| `device` | str | 'cuda' | Computation device ('cuda' or 'cpu') |
| `dtype` | torch.dtype | float32 | PyTorch dtype for computations |
| `enable_stress` | bool | False | Compute stress tensor |
| `batch_size` | int | None | Batch size for automatic batching |
| `enable_timing` | bool | False | Track calculation timing |
| `use_compile` | bool | False | Apply torch.compile() optimization |
| `use_fp16` | bool | False | Use FP16 mixed precision |
| `use_jit` | bool | False | Use TorchScript JIT model |
| `jit_path` | str/Path | None | Path to TorchScript model |
| `use_torch_cluster` | bool | True | Use torch-cluster for neighbor search |
| `use_analytical_forces` | bool | False | Use analytical force computation |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `implemented_properties` | List[str] | ['energy', 'forces', 'stress'] |
| `model` | StudentForceField | The loaded model |
| `device` | torch.device | Computation device |
| `n_calls` | int | Number of calculations performed |
| `avg_time` | float | Average calculation time (seconds) |

#### Methods

##### calculate

```python
calculate(
    atoms: Optional[Atoms] = None,
    properties: List[str] = ['energy', 'forces'],
    system_changes: List[str] = all_changes
)
```

Calculate properties for given atoms. Called automatically by ASE.

##### calculate_batch

```python
calculate_batch(
    atoms_list: List[Atoms],
    properties: List[str] = ['energy', 'forces']
) -> List[Dict[str, Any]]
```

Calculate properties for multiple structures efficiently using batched inference.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `atoms_list` | List[Atoms] | List of ASE Atoms objects |
| `properties` | List[str] | Properties to calculate |

**Returns:**

- List of dictionaries with 'energy' and 'forces' keys

**Example:**

```python
calc = StudentForceFieldCalculator('checkpoints/best_model.pt')

# Create multiple structures
molecules = [molecule('H2O'), molecule('CH4'), molecule('NH3')]

# Batch calculate
results = calc.calculate_batch(molecules)

for mol, result in zip(molecules, results):
    print(f"{mol.get_chemical_formula()}: E = {result['energy']:.4f} eV")
```

##### get_timing_stats

```python
get_timing_stats() -> Dict[str, float]
```

Get detailed timing statistics.

**Returns:**

```python
{
    'n_calls': int,       # Number of calculations
    'total_time': float,  # Total time in seconds
    'avg_time': float,    # Average time per call
    'min_time': float,    # Minimum time
    'max_time': float,    # Maximum time
    'median_time': float, # Median time
    'std_time': float     # Standard deviation
}
```

##### reset

```python
reset()
```

Reset calculator state and clear caches.

---

## Testing

### NVEMDHarness

```python
from mlff_distiller.testing import NVEMDHarness
```

NVE (microcanonical) molecular dynamics harness for model validation.

#### Constructor

```python
NVEMDHarness(
    atoms: Union[Atoms, str, Path],
    calculator: Calculator,
    temperature: float = 300.0,
    timestep: float = 0.5,
    trajectory_file: Optional[Union[str, Path]] = None,
    log_interval: int = 10,
    remove_com_motion: bool = True
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `atoms` | Atoms/str/Path | required | ASE Atoms object or path to structure file |
| `calculator` | Calculator | required | ASE Calculator for energy/force evaluation |
| `temperature` | float | 300.0 | Initial temperature in Kelvin |
| `timestep` | float | 0.5 | MD timestep in femtoseconds |
| `trajectory_file` | str/Path | None | Path to save trajectory |
| `log_interval` | int | 10 | Steps between logging |
| `remove_com_motion` | bool | True | Remove center-of-mass motion |

#### Methods

##### run_simulation

```python
run_simulation(
    steps: int,
    initialize_velocities: bool = True,
    save_interval: Optional[int] = None
) -> Dict[str, Any]
```

Run NVE molecular dynamics simulation.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `steps` | int | required | Number of MD steps |
| `initialize_velocities` | bool | True | Initialize velocities before running |
| `save_interval` | int | None | Save trajectory every N steps |

**Returns:**

```python
{
    'trajectory_data': Dict,      # Full trajectory data
    'n_steps': int,               # Steps completed
    'total_time_ps': float,       # Simulation time in ps
    'timestep_fs': float,         # Timestep in fs
    'wall_time_s': float,         # Wall clock time in seconds
    'energy_drift_pct': float,    # Total energy drift percentage
    'energy_drift_abs': float,    # Absolute energy drift in eV
    'energy_std': float,          # Energy standard deviation
    'energy_range': float,        # Energy range (peak-to-peak)
    'avg_temperature': float,     # Average temperature in K
    'std_temperature': float,     # Temperature std in K
    'initial_energy': float,      # Initial total energy
    'final_energy': float,        # Final total energy
    'avg_potential_energy': float,
    'avg_kinetic_energy': float,
}
```

**Example:**

```python
from mlff_distiller.testing import NVEMDHarness
from mlff_distiller import StudentForceFieldCalculator
from ase.build import molecule

calc = StudentForceFieldCalculator('checkpoints/best_model.pt')
atoms = molecule('H2O')

harness = NVEMDHarness(
    atoms=atoms,
    calculator=calc,
    temperature=300.0,
    timestep=0.5
)

results = harness.run_simulation(steps=1000)
print(f"Energy drift: {results['energy_drift_pct']:.4f}%")
```

##### initialize_velocities

```python
initialize_velocities(
    temperature: Optional[float] = None,
    remove_com_translation: bool = True,
    remove_com_rotation: bool = True
)
```

Initialize atomic velocities from Maxwell-Boltzmann distribution.

##### get_trajectory_array

```python
get_trajectory_array(key: str) -> np.ndarray
```

Get trajectory data as numpy array.

**Keys:** 'time', 'positions', 'velocities', 'forces', 'potential_energy', 'kinetic_energy', 'total_energy', 'temperature'

##### save_trajectory

```python
save_trajectory(filename: Union[str, Path], format: str = 'traj')
```

Save trajectory to file in specified format.

##### reset

```python
reset()
```

Reset trajectory data and simulation state.

---

### Energy Metrics

```python
from mlff_distiller.testing import (
    compute_energy_drift,
    compute_energy_conservation_ratio,
    compute_energy_fluctuations,
    compute_kinetic_potential_stability,
    compute_time_resolved_drift,
    assess_energy_conservation,
)
```

#### assess_energy_conservation

```python
assess_energy_conservation(
    trajectory_data: Dict,
    tolerance_pct: float = 1.0,
    verbose: bool = False
) -> Dict[str, Any]
```

Comprehensive energy conservation assessment.

**Returns:**

```python
{
    'passed': bool,              # Whether conservation passed
    'drift_pct': float,          # Energy drift percentage
    'conservation_ratio': float, # E_std / |E_mean|
    'fluctuation_ratio': float,  # Energy fluctuation ratio
    'details': Dict              # Additional metrics
}
```

#### compute_energy_drift

```python
compute_energy_drift(
    total_energies: np.ndarray,
    return_percentage: bool = True
) -> float
```

Compute total energy drift over trajectory.

#### compute_energy_conservation_ratio

```python
compute_energy_conservation_ratio(total_energies: np.ndarray) -> float
```

Compute energy conservation ratio: std(E) / |mean(E)|.

---

### Force Metrics

```python
from mlff_distiller.testing import (
    compute_force_rmse,
    compute_force_mae,
    compute_force_magnitude_error,
    compute_angular_error,
    compute_per_atom_force_errors,
    compute_force_correlation,
    assess_force_accuracy,
)
```

#### assess_force_accuracy

```python
assess_force_accuracy(
    predicted_forces: np.ndarray,
    reference_forces: np.ndarray,
    tolerance_rmse: float = 0.2,
    tolerance_r2: float = 0.95,
    verbose: bool = False
) -> Dict[str, Any]
```

Comprehensive force accuracy assessment.

**Returns:**

```python
{
    'passed': bool,
    'rmse': float,              # Force RMSE in eV/Angstrom
    'mae': float,               # Force MAE
    'r2': float,                # Force R-squared
    'angular_error_deg': float, # Mean angular error in degrees
    'magnitude_error': float,   # Mean magnitude error
}
```

#### compute_force_rmse

```python
compute_force_rmse(
    predicted: np.ndarray,
    reference: np.ndarray
) -> float
```

Compute force RMSE in eV/Angstrom.

#### compute_force_mae

```python
compute_force_mae(
    predicted: np.ndarray,
    reference: np.ndarray
) -> float
```

Compute force MAE in eV/Angstrom.

#### compute_angular_error

```python
compute_angular_error(
    predicted: np.ndarray,
    reference: np.ndarray
) -> float
```

Compute mean angular error between force vectors in degrees.

---

### Trajectory Analysis

```python
from mlff_distiller.testing import (
    compute_rmsd,
    kabsch_align,
    compute_atom_displacements,
    analyze_temperature_evolution,
    compute_bond_lengths,
    analyze_trajectory_stability,
    generate_trajectory_summary,
)
```

#### compute_rmsd

```python
compute_rmsd(
    positions1: np.ndarray,
    positions2: np.ndarray,
    align: bool = True
) -> float
```

Compute RMSD between two structures in Angstrom.

#### analyze_trajectory_stability

```python
analyze_trajectory_stability(
    trajectory_data: Dict,
    reference_positions: Optional[np.ndarray] = None
) -> Dict[str, Any]
```

Analyze structural stability over MD trajectory.

**Returns:**

```python
{
    'max_rmsd': float,
    'mean_rmsd': float,
    'max_displacement': float,
    'stable': bool
}
```

#### generate_trajectory_summary

```python
generate_trajectory_summary(results: Dict) -> str
```

Generate human-readable summary of MD results.

---

## Complete Example

```python
from mlff_distiller import StudentForceFieldCalculator
from mlff_distiller.testing import NVEMDHarness, assess_energy_conservation
from ase.build import molecule

# Load model
calc = StudentForceFieldCalculator(
    'checkpoints/best_model.pt',
    device='cuda',
    enable_timing=True
)

# Create test molecule
atoms = molecule('CH4')

# Run NVE MD validation
harness = NVEMDHarness(
    atoms=atoms,
    calculator=calc,
    temperature=300.0,
    timestep=0.5,
    trajectory_file='ch4_md.traj'
)

results = harness.run_simulation(steps=2000)

# Assess energy conservation
assessment = assess_energy_conservation(
    results['trajectory_data'],
    tolerance_pct=1.0,
    verbose=True
)

# Print summary
print(f"\nValidation Results:")
print(f"  Energy drift: {assessment['drift_pct']:.4f}%")
print(f"  Conservation ratio: {assessment['conservation_ratio']:.6f}")
print(f"  Status: {'PASSED' if assessment['passed'] else 'FAILED'}")

# Get timing stats
stats = calc.get_timing_stats()
print(f"\nPerformance:")
print(f"  Average inference: {stats['avg_time']*1000:.2f} ms")
print(f"  Total calls: {stats['n_calls']}")
```
