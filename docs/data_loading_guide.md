# Data Loading Infrastructure Guide

**Version**: 1.0
**Last Updated**: 2025-11-23
**Issue**: #1 - Set Up Data Loading Infrastructure

## Overview

The MLFF Distiller data loading infrastructure provides flexible, efficient dataset loaders for molecular dynamics training data with seamless ASE (Atomic Simulation Environment) integration. This guide covers all aspects of loading, preprocessing, and batching molecular data.

## Key Features

- **Multiple Format Support**: ASE databases, HDF5, XYZ files
- **ASE Atoms Compatibility**: Drop-in compatibility with MD workflows
- **Variable System Sizes**: Efficiently handle 10-1000+ atoms
- **Flexible Batching**: Padding-based or graph-based batching
- **Data Augmentation**: Physics-preserving transformations
- **Easy Splitting**: Train/validation/test split utilities

## Quick Start

### Basic Usage

```python
from mlff_distiller.data import MolecularDataset, MolecularDataLoader

# Load dataset from ASE database
dataset = MolecularDataset('structures.db', format='ase')

# Create dataloader
loader = MolecularDataLoader(dataset, batch_size=32, shuffle=True)

# Iterate through batches
for batch in loader:
    positions = batch['positions']  # [batch_size, max_atoms, 3]
    species = batch['species']      # [batch_size, max_atoms]
    mask = batch['mask']            # [batch_size, max_atoms]
    energy = batch['energy']        # [batch_size]
    forces = batch['forces']        # [batch_size, max_atoms, 3]
```

### Train/Val/Test Split

```python
from mlff_distiller.data import MolecularDataset, train_test_split, create_dataloaders

# Load dataset
dataset = MolecularDataset('structures.db', format='ase')

# Split into train/val/test
train_dataset, val_dataset, test_dataset = train_test_split(
    dataset,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    shuffle=True,
    random_seed=42
)

# Create dataloaders
loaders = create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size=32,
    num_workers=4
)

train_loader = loaders['train']
val_loader = loaders['val']
test_loader = loaders['test']
```

## Data Formats

### ASE Database Format

ASE databases are the recommended format for molecular data. They provide efficient storage with metadata.

```python
from ase.db import connect
from ase import Atoms

# Create ASE database
db = connect('structures.db')

# Add structures
for atoms in my_structures:
    db.write(atoms, energy=energy_value, data={'forces': forces_array})

# Load with MolecularDataset
dataset = MolecularDataset('structures.db', format='ase')
```

**Advantages**:
- Efficient storage and indexing
- Metadata support
- Standard format in computational chemistry
- Easy querying and filtering

### HDF5 Format

HDF5 provides efficient storage for large datasets with hierarchical organization.

```python
import h5py

# Create HDF5 file
with h5py.File('structures.h5', 'w') as f:
    structures = f.create_group('structures')

    for i, atoms in enumerate(my_structures):
        group = structures.create_group(str(i))
        group.create_dataset('positions', data=atoms.positions)
        group.create_dataset('species', data=atoms.numbers)
        group.create_dataset('cell', data=atoms.cell.array)
        group.create_dataset('pbc', data=atoms.pbc)
        group.create_dataset('energy', data=energy)
        group.create_dataset('forces', data=forces)

# Load with MolecularDataset
dataset = MolecularDataset('structures.h5', format='hdf5')
```

**Advantages**:
- Very fast I/O
- Compression support
- Good for very large datasets
- Parallel I/O capable

### XYZ Format

XYZ is a simple text format widely used in computational chemistry.

```python
from ase.io import write

# Write XYZ file
write('structures.xyz', my_structures)

# Load with MolecularDataset
dataset = MolecularDataset('structures.xyz', format='xyz')
```

**Advantages**:
- Human-readable
- Widely supported
- Simple format
- Good for small datasets

## Dataset Features

### ASE Atoms Compatibility

The dataset provides seamless integration with ASE Atoms objects:

```python
dataset = MolecularDataset('structures.db', format='ase', return_atoms=True)

# Get sample as dictionary
sample = dataset[0]
atoms = sample['atoms']  # ASE Atoms object

# Or use convenience method
atoms = dataset.get_atoms(0)

# ASE Atoms can be used directly in MD simulations
from ase.calculators.calculator import Calculator
atoms.calc = my_calculator
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

### Variable System Sizes

The dataset efficiently handles systems of different sizes:

```python
dataset = MolecularDataset('structures.db', format='ase')

for i in range(len(dataset)):
    sample = dataset[i]
    print(f"Sample {i}: {sample['natoms']} atoms")

# Get statistics
stats = dataset.get_statistics()
print(f"System sizes: {stats['natoms_min']} - {stats['natoms_max']} atoms")
print(f"Mean size: {stats['natoms_mean']:.1f} atoms")
```

### Caching

For small datasets that fit in memory, enable caching for faster access:

```python
# Without caching (loads from disk each time)
dataset = MolecularDataset('structures.db', format='ase', cache=False)

# With caching (loads once, stores in memory)
dataset = MolecularDataset('structures.db', format='ase', cache=True)
```

## Batching Strategies

### Padded Batching (Default)

Pads systems to the largest size in the batch:

```python
from mlff_distiller.data import molecular_collate_fn

loader = MolecularDataLoader(
    dataset,
    batch_size=32,
    use_padding=True  # Default
)

batch = next(iter(loader))
# batch['positions']: [batch_size, max_atoms, 3] - padded
# batch['mask']: [batch_size, max_atoms] - True for real atoms
# batch['max_atoms']: maximum number of atoms in batch
```

**Use when**:
- Working with standard neural networks
- System sizes are relatively uniform
- Using convolutions or attention mechanisms

### Graph-Based Batching

Concatenates all atoms without padding:

```python
from mlff_distiller.data import molecular_collate_fn_no_padding

loader = MolecularDataLoader(
    dataset,
    batch_size=32,
    use_padding=False
)

batch = next(iter(loader))
# batch['positions']: [total_atoms, 3] - concatenated
# batch['batch']: [total_atoms] - batch index for each atom
# batch['total_atoms']: total number of atoms across batch
```

**Use when**:
- Working with graph neural networks
- System sizes vary significantly
- Memory is constrained
- Using message-passing architectures

## Data Augmentation

### Physics-Preserving Transforms

Apply transformations that preserve physical properties:

```python
from mlff_distiller.data import (
    Compose,
    RandomRotation,
    RandomReflection,
    AddNoise,
)

# Create transform pipeline
transform = Compose([
    RandomRotation(p=0.5),      # Random 3D rotation
    RandomReflection(p=0.3),    # Random reflection
    AddNoise(std=0.01, p=0.5),  # Small position noise
])

# Apply to dataset
dataset = MolecularDataset(
    'structures.db',
    format='ase',
    transform=transform
)
```

### Normalization

Normalize energies and forces for better training:

```python
from mlff_distiller.data import NormalizeEnergy, NormalizeForces

# Get statistics from dataset
stats = dataset.get_statistics()

# Create normalization transforms
transform = Compose([
    NormalizeEnergy(
        mean=stats['energy_per_atom_mean'],
        std=stats['energy_per_atom_std'],
        per_atom=True
    ),
    NormalizeForces(
        rms=stats['forces_rms']
    ),
])

# Apply to training dataset
train_dataset = MolecularDataset(
    'structures.db',
    format='ase',
    transform=transform
)
```

### Custom Transforms

Create custom transforms by inheriting from `Transform`:

```python
from mlff_distiller.data import Transform

class MyCustomTransform(Transform):
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

    def __call__(self, sample):
        # Modify sample dictionary
        sample['positions'] = self.transform_positions(sample['positions'])
        return sample

    def transform_positions(self, positions):
        # Your custom transformation
        return positions

# Use in pipeline
transform = Compose([
    MyCustomTransform(param1=1.0, param2=2.0),
    RandomRotation(p=0.5),
])
```

## Complete Training Pipeline Example

```python
from mlff_distiller.data import (
    MolecularDataset,
    train_test_split,
    create_dataloaders,
    Compose,
    RandomRotation,
    AddNoise,
    NormalizeEnergy,
    NormalizeForces,
)

# 1. Load full dataset
full_dataset = MolecularDataset('structures.db', format='ase')

# 2. Compute statistics for normalization
stats = full_dataset.get_statistics()
print(f"Dataset: {stats['num_samples']} samples")
print(f"Energy per atom: {stats['energy_per_atom_mean']:.3f} ± {stats['energy_per_atom_std']:.3f} eV")
print(f"Forces RMS: {stats['forces_rms']:.3f} eV/Å")

# 3. Create augmentation transform (training only)
train_transform = Compose([
    RandomRotation(p=0.5),
    AddNoise(std=0.01, p=0.3),
    NormalizeEnergy(
        mean=stats['energy_per_atom_mean'],
        std=stats['energy_per_atom_std'],
        per_atom=True
    ),
    NormalizeForces(rms=stats['forces_rms']),
])

# 4. Create normalization transform (validation/test)
eval_transform = Compose([
    NormalizeEnergy(
        mean=stats['energy_per_atom_mean'],
        std=stats['energy_per_atom_std'],
        per_atom=True
    ),
    NormalizeForces(rms=stats['forces_rms']),
])

# 5. Split dataset
train_indices, val_indices, test_indices = train_test_split(
    full_dataset,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    shuffle=True,
    random_seed=42
)

# 6. Create datasets with transforms
train_dataset = MolecularDataset('structures.db', format='ase', transform=train_transform)
val_dataset = MolecularDataset('structures.db', format='ase', transform=eval_transform)
test_dataset = MolecularDataset('structures.db', format='ase', transform=eval_transform)

# Apply subset indices
from mlff_distiller.data import MolecularSubset
train_dataset = MolecularSubset(train_dataset, train_indices.indices)
val_dataset = MolecularSubset(val_dataset, val_indices.indices)
test_dataset = MolecularSubset(test_dataset, test_indices.indices)

# 7. Create dataloaders
loaders = create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size=32,
    num_workers=4,
    use_padding=True
)

# 8. Training loop
for epoch in range(num_epochs):
    model.train()
    for batch in loaders['train']:
        positions = batch['positions']
        species = batch['species']
        mask = batch['mask']
        energy = batch['energy']
        forces = batch['forces']

        # Forward pass
        pred_energy, pred_forces = model(positions, species, mask)

        # Compute loss (with mask for forces)
        energy_loss = F.mse_loss(pred_energy, energy)
        forces_loss = masked_mse_loss(pred_forces, forces, mask)
        loss = energy_loss + forces_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        for batch in loaders['val']:
            # Validation code
            pass
```

## Performance Considerations

### Memory Management

```python
# For large datasets, disable caching
dataset = MolecularDataset('large.db', format='ase', cache=False)

# Use graph-based batching for variable sizes
loader = MolecularDataLoader(dataset, batch_size=32, use_padding=False)

# Enable multiprocessing for data loading
loader = MolecularDataLoader(dataset, batch_size=32, num_workers=4)
```

### Batch Size Selection

```python
# Get dataset statistics to choose batch size
stats = dataset.get_statistics()
mean_atoms = stats['natoms_mean']
max_atoms = stats['natoms_max']

# Estimate memory usage (rough)
# memory_per_sample ≈ max_atoms * 3 * 4 bytes (positions) + max_atoms * 8 bytes (species)
# memory_per_batch ≈ memory_per_sample * batch_size

# Adjust batch size based on available GPU memory
if max_atoms < 50:
    batch_size = 64
elif max_atoms < 200:
    batch_size = 32
else:
    batch_size = 16
```

### I/O Optimization

```python
# Use HDF5 for large datasets
dataset = MolecularDataset('large.h5', format='hdf5', cache=False)

# Enable pin_memory for faster GPU transfer
loader = MolecularDataLoader(dataset, batch_size=32, pin_memory=True)

# Use multiple workers for parallel loading
loader = MolecularDataLoader(dataset, batch_size=32, num_workers=4)
```

## Troubleshooting

### Issue: Out of Memory

**Solution**:
```python
# Reduce batch size
loader = MolecularDataLoader(dataset, batch_size=16)

# Use graph-based batching (no padding)
loader = MolecularDataLoader(dataset, batch_size=32, use_padding=False)

# Disable caching
dataset = MolecularDataset('data.db', format='ase', cache=False)
```

### Issue: Slow Data Loading

**Solution**:
```python
# Use multiple workers
loader = MolecularDataLoader(dataset, batch_size=32, num_workers=4)

# Enable caching for small datasets
dataset = MolecularDataset('data.db', format='ase', cache=True)

# Use HDF5 format for faster I/O
dataset = MolecularDataset('data.h5', format='hdf5')
```

### Issue: Forces/Energy Missing

**Solution**:
```python
# Check what properties are available
sample = dataset[0]
print(f"Available keys: {sample.keys()}")

# Verify data in source file
from ase.db import connect
db = connect('structures.db')
row = db.get(1)
print(f"Energy: {row.energy}")
print(f"Forces in data: {'forces' in row.data}")
```

## API Reference

### MolecularDataset

```python
MolecularDataset(
    data_path: Union[str, Path],
    format: str = 'ase',
    cache: bool = False,
    transform: Optional[Callable] = None,
    return_atoms: bool = True,
    dtype: torch.dtype = torch.float32,
    energy_key: str = 'energy',
    forces_key: str = 'forces',
    stress_key: str = 'stress',
)
```

### MolecularDataLoader

```python
MolecularDataLoader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = False,
    use_padding: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
)
```

### train_test_split

```python
train_test_split(
    dataset: MolecularDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    random_seed: Optional[int] = 42,
) -> Tuple[MolecularSubset, MolecularSubset, MolecularSubset]
```

## Best Practices

1. **Always compute statistics** before training for proper normalization
2. **Use ASE database format** for best compatibility and features
3. **Enable caching** only for small datasets (< 10k samples)
4. **Use graph-based batching** for highly variable system sizes
5. **Apply augmentation** only to training set, not validation/test
6. **Set random seeds** for reproducible splits
7. **Use multiple workers** for faster loading (typically 4-8)
8. **Monitor memory usage** when selecting batch sizes

## Next Steps

- See [DROP_IN_COMPATIBILITY_GUIDE.md](DROP_IN_COMPATIBILITY_GUIDE.md) for ASE Calculator integration
- See [CONTRIBUTING.md](../CONTRIBUTING.md) for development guidelines
- See Issue #2 for data generation from teacher models
- See Issue #3 for preprocessing pipeline

## Support

For questions or issues:
- Open an issue on GitHub
- Tag `@coordinator` for architectural questions
- Reference this guide in discussions
