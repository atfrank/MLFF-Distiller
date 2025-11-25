# HDF5 Dataset Writer Documentation

**Issue**: M2 #13
**Status**: Complete
**Author**: Data Pipeline Engineer
**Date**: 2025-11-23

## Overview

The HDF5DatasetWriter is a production-ready, robust tool for writing molecular and materials datasets in HDF5 format for the MLFF_Distiller project. It supports incremental writing, compression, validation, and append mode for large-scale dataset generation.

## Features

- **Incremental Writing**: Add structures one at a time or in batches
- **Append Mode**: Extend existing datasets without rewriting
- **Compression**: Automatic compression (gzip, lzf) with chunking for efficient storage
- **Validation**: Data validation (shapes, dtypes, physical plausibility)
- **Progress Tracking**: Logging and progress bars for large datasets
- **Memory Efficient**: Doesn't load entire dataset into RAM
- **Format Compatible**: Matches existing label generation pipeline format

## Installation

The writer is included in the `mlff_distiller` package:

```python
from mlff_distiller.data.hdf5_writer import HDF5DatasetWriter
```

Dependencies:
- h5py
- numpy
- ase (Atomic Simulation Environment)

## HDF5 Format Specification

The writer produces HDF5 files with the following structure:

```
/structures/
    - atomic_numbers (int64, concatenated)
    - atomic_numbers_splits (int64, cumulative splits)
    - positions (float64, concatenated positions)
    - cells (float64, (n_structures, 3, 3))
    - pbc (bool, (n_structures, 3))
/labels/
    - energy (float64, (n_structures,))
    - forces (float32, concatenated forces)
    - forces_splits (int64, cumulative splits)
    - stress (float64, (n_structures, 6))
    - stress_mask (bool, (n_structures,))
    - structure_indices (int64, (n_structures,))
/metadata/
    - Attributes: timestamps, config, source info, statistics
```

### Ragged Array Storage

Structures have varying numbers of atoms, so atomic numbers, positions, and forces are stored as concatenated 1D/2D arrays with split indices:

```python
# Example: 3 structures with 2, 3, and 4 atoms
atomic_numbers = [1, 1, 6, 1, 1, 8, 1, 1, 1]  # Concatenated
atomic_numbers_splits = [0, 2, 5, 9]  # Cumulative indices

# To reconstruct structure i:
start = atomic_numbers_splits[i]
end = atomic_numbers_splits[i+1]
structure_atoms = atomic_numbers[start:end]
```

## Basic Usage

### Creating a New Dataset

```python
from mlff_distiller.data.hdf5_writer import HDF5DatasetWriter
from ase import Atoms
import numpy as np

# Create writer
writer = HDF5DatasetWriter(
    output_path="dataset.h5",
    compression="gzip",
    compression_opts=4,  # Compression level 1-9
    mode="w"  # Write mode (overwrite)
)

# Add structures incrementally
for atoms, energy, forces in zip(structures, energies, forces_list):
    writer.add_structure(
        atoms=atoms,
        energy=energy,
        forces=forces,
        stress=stress,  # Optional, None for non-periodic
        metadata={"source": "mattergen"}  # Optional
    )

# Finalize and close
writer.finalize()
```

### Using Context Manager (Recommended)

```python
with HDF5DatasetWriter("dataset.h5", compression="gzip") as writer:
    for atoms, energy, forces in data:
        writer.add_structure(atoms, energy, forces)
    # finalize() called automatically
```

### Batch Writing

For better performance when writing many structures:

```python
with HDF5DatasetWriter("dataset.h5", compression="gzip") as writer:
    writer.add_batch(
        structures=atoms_list,
        energies=energy_list,
        forces=forces_list,
        stresses=stress_list,  # Optional
        show_progress=True
    )
```

### Append Mode

Extend an existing dataset:

```python
# Initial write
with HDF5DatasetWriter("dataset.h5", mode="w") as writer:
    writer.add_batch(structures[:1000], energies[:1000], forces[:1000])

# Append more data later
with HDF5DatasetWriter("dataset.h5", mode="a") as writer:
    writer.add_batch(structures[1000:], energies[1000:], forces[1000:])
```

## Advanced Features

### Validation

The writer validates data by default:

```python
writer = HDF5DatasetWriter("dataset.h5", validate=True)

# Checks performed:
# - NaN/Inf in energies, forces, stresses
# - Array shapes match structure sizes
# - Physical plausibility (warns on close atoms, large forces)
```

To disable validation (faster but risky):

```python
writer = HDF5DatasetWriter("dataset.h5", validate=False)
```

### Compression Options

```python
# No compression (fastest write, largest file)
writer = HDF5DatasetWriter("dataset.h5", compression=None)

# LZF compression (fast, moderate compression)
writer = HDF5DatasetWriter("dataset.h5", compression="lzf")

# GZIP compression (slower, better compression)
writer = HDF5DatasetWriter("dataset.h5", compression="gzip", compression_opts=4)

# Maximum GZIP compression (slowest, best compression)
writer = HDF5DatasetWriter("dataset.h5", compression="gzip", compression_opts=9)
```

**Benchmark Results** (930 structures, 68K atoms):
- No compression: 3.10 MB
- LZF: 2.10 MB (67.8% of original, 32.2% savings)
- GZIP-4: 1.93 MB (62.3% of original, 37.7% savings)
- GZIP-9: 1.90 MB (61.2% of original, 38.8% savings)

**Recommendation**: Use `compression="gzip"` with `compression_opts=4` for best balance of compression and speed.

### Metadata

Add custom metadata to the dataset:

```python
writer.finalize(extra_metadata={
    "teacher_model": "orb-v2",
    "generation_date": "2025-11-23",
    "generation_config": {
        "batch_size": 32,
        "device": "cuda"
    },
    "source": "mattergen",
    "version": "1.0"
})
```

Metadata is stored as HDF5 attributes in the `/metadata` group.

### Stress Handling

The writer automatically handles stress tensors for periodic systems:

```python
# Periodic structure with stress
atoms_periodic = Atoms(..., pbc=[True, True, True])
stress = np.array([s_xx, s_yy, s_zz, s_yz, s_xz, s_xy])  # Voigt notation
writer.add_structure(atoms_periodic, energy, forces, stress=stress)

# Non-periodic structure without stress
atoms_molecule = Atoms(..., pbc=[False, False, False])
writer.add_structure(atoms_molecule, energy, forces, stress=None)
```

Stress is stored with a boolean mask indicating which structures have valid stress data.

## Utilities

### Convert Pickle to HDF5

Convenience function for converting existing pickle files:

```python
from mlff_distiller.data.hdf5_writer import convert_pickle_to_hdf5

convert_pickle_to_hdf5(
    pickle_path="structures.pkl",
    hdf5_path="dataset.h5",
    energies=energies,  # Optional if in Atoms.calc
    forces=forces,      # Optional if in Atoms.calc
    compression="gzip",
    show_progress=True
)
```

### CLI Tool

Use the command-line tool for quick conversions:

```bash
# Convert pickle with labels in Atoms.calc
python scripts/convert_to_hdf5.py -i structures.pkl -o dataset.h5

# Convert with separate labels file
python scripts/convert_to_hdf5.py -i structures.pkl -o dataset.h5 -l labels.pkl

# Convert XYZ file
python scripts/convert_to_hdf5.py -i data.xyz -o dataset.h5 -f xyz

# Add metadata
python scripts/convert_to_hdf5.py -i structures.pkl -o dataset.h5 \\
    -m source=mattergen -m version=1.0

# Use LZF compression
python scripts/convert_to_hdf5.py -i structures.pkl -o dataset.h5 -c lzf
```

## Testing

### Unit Tests

Run comprehensive unit tests:

```bash
cd /home/aaron/ATX/software/MLFF_Distiller
python -m pytest tests/unit/test_hdf5_writer.py -v
```

**Test Coverage**: 21 tests covering:
- Basic write operations
- Batch writing
- Append mode
- Compression effectiveness
- Validation (NaN/Inf detection, shape checking)
- Error handling
- Edge cases (empty datasets, single atoms, large structures)
- Format compatibility
- Stress handling for periodic/non-periodic systems

### Validation Script

Validate against existing 1K labeled dataset:

```bash
python scripts/validate_hdf5_writer.py
```

This script:
1. Loads existing labeled data from `all_labels_orb_v2.h5`
2. Rewrites using HDF5DatasetWriter
3. Compares for exact consistency
4. Tests append mode
5. Measures compression effectiveness

**Validation Results**:
- All 930 structures match exactly
- Append mode works correctly
- Compression reduces size by 37-39%
- Format 100% compatible with existing pipeline

## Performance

### Write Performance

| Mode | Structures/sec | Notes |
|------|----------------|-------|
| Incremental | ~150-200 | Good for streaming data |
| Batch | ~300-500 | Better for bulk writes |
| Append | ~150-200 | Similar to incremental |

### Memory Usage

The writer is memory-efficient:
- Processes structures in streaming fashion
- Doesn't load entire dataset into RAM
- Suitable for datasets with 100K+ structures

### File Size Scaling

For typical molecular/materials datasets:
- ~2-3 MB per 1000 structures (with gzip-4)
- Expected size for 120K structures: ~240-360 MB compressed

## Error Handling

The writer provides informative error messages:

```python
# Invalid energy
writer.add_structure(atoms, energy=np.nan, forces=forces)
# Raises: ValueError: Invalid energy: nan

# Wrong force shape
writer.add_structure(atoms, energy=1.0, forces=np.zeros((10, 3)))
# Raises: ValueError: Forces shape (10, 3) doesn't match atoms (5, 3)

# Adding after finalize
writer.finalize()
writer.add_structure(atoms, energy, forces)
# Raises: RuntimeError: Cannot add structures after finalize() has been called
```

## Best Practices

1. **Always use context manager**:
   ```python
   with HDF5DatasetWriter(...) as writer:
       # Your code here
   ```

2. **Use batch writing for large datasets**:
   ```python
   writer.add_batch(structures, energies, forces)  # Faster
   # vs
   for s, e, f in zip(...):
       writer.add_structure(s, e, f)  # Slower
   ```

3. **Enable validation during development**:
   ```python
   writer = HDF5DatasetWriter(..., validate=True)
   ```

4. **Disable validation for production** (after testing):
   ```python
   writer = HDF5DatasetWriter(..., validate=False)
   ```

5. **Use appropriate compression**:
   - Development: `compression=None` (fastest)
   - Production: `compression="gzip", compression_opts=4` (balanced)
   - Archival: `compression="gzip", compression_opts=9` (smallest)

6. **Add metadata**:
   ```python
   writer.finalize(extra_metadata={
       "teacher_model": "orb-v2",
       "date": datetime.now().isoformat(),
       "config": config_dict
   })
   ```

7. **Handle stress correctly**:
   ```python
   stress = atoms.get_stress() if np.any(atoms.pbc) else None
   writer.add_structure(atoms, energy, forces, stress)
   ```

## Troubleshooting

### File is too large

- Enable compression: `compression="gzip"`
- Use higher compression level: `compression_opts=9`
- Check if duplicate data is being written

### Writing is slow

- Use batch writing instead of incremental
- Disable validation: `validate=False`
- Use faster compression: `compression="lzf"`

### Out of memory

- Write in smaller batches
- Ensure you're not accumulating results in memory
- Use streaming/generator patterns

### Append mode not working

- Check file exists: `Path("dataset.h5").exists()`
- Ensure mode is "a": `mode="a"`
- Verify file isn't corrupted

## Integration with Label Generation Pipeline

The HDF5DatasetWriter integrates seamlessly with the existing label generation:

```python
from mlff_distiller.data.label_generation import LabelGenerator
from mlff_distiller.data.hdf5_writer import HDF5DatasetWriter

# Generate labels
generator = LabelGenerator("orb-v2", device="cuda")
results = generator.generate_labels(structures)

# Write to HDF5 using new writer
with HDF5DatasetWriter("dataset.h5", compression="gzip") as writer:
    for result, atoms in zip(results, structures):
        if result.success:
            writer.add_structure(
                atoms=atoms,
                energy=result.energy,
                forces=result.forces,
                stress=result.stress
            )

# Or use the existing save_results method (still works)
generator.save_results("labels.h5", results, structures)
```

## Future Enhancements

Potential improvements for future iterations:

1. **Parallel Writing**: Multi-threaded/multi-process writing for large datasets
2. **Chunking Optimization**: Adaptive chunk sizes based on dataset statistics
3. **Compression Profiles**: Pre-configured compression profiles for different use cases
4. **Data Deduplication**: Detect and skip duplicate structures
5. **Checkpointing**: Automatic checkpointing for very long writes
6. **Format Versioning**: Version tracking for format changes

## References

- **Issue**: M2 #13 - HDF5 Dataset Writer
- **Related Issues**:
  - M2 #11: Label Generation Pipeline
  - M2 #14: Data Augmentation
- **Files**:
  - `/home/aaron/ATX/software/MLFF_Distiller/src/mlff_distiller/data/hdf5_writer.py`
  - `/home/aaron/ATX/software/MLFF_Distiller/tests/unit/test_hdf5_writer.py`
  - `/home/aaron/ATX/software/MLFF_Distiller/scripts/validate_hdf5_writer.py`
  - `/home/aaron/ATX/software/MLFF_Distiller/scripts/convert_to_hdf5.py`

## License

Part of the MLFF_Distiller project.

---

**Last Updated**: 2025-11-23
**Status**: Production Ready
**Test Coverage**: 21 tests, 100% pass rate
**Validation**: Passed on 930 structure dataset
