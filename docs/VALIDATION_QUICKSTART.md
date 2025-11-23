# Dataset Validation - Quick Start Guide

## TL;DR

```bash
# Validate dataset from command line
python scripts/validate_dataset.py your_data.db --format ase

# Or in Python
python -c "
from mlff_distiller.data import MolecularDataset, DatasetValidator
dataset = MolecularDataset('your_data.db')
validator = DatasetValidator()
report = validator.validate_dataset(dataset)
print(f'Valid: {report.passed}, Errors: {report.num_errors}')
"
```

## What Gets Checked

### Structure (Geometry)
- ✓ No overlapping atoms (min 0.5 Å apart)
- ✓ No NaN/Inf in positions
- ✓ Valid periodic boundaries
- ✓ Reasonable system size (10-500 atoms)

### Labels (Energy/Forces)
- ✓ No NaN/Inf in labels
- ✓ Energies in range (-50 to 50 eV/atom)
- ✓ Forces not excessive (<100 eV/Å)
- ✓ Statistical outliers (<1%)

### Diversity
- ✓ Good element coverage
- ✓ Variety of system sizes
- ✓ Different compositions

## Quick Examples

### Example 1: Basic Validation

```bash
python scripts/validate_dataset.py train.db
```

Output:
```
================================================================================
DATASET VALIDATION REPORT
================================================================================
Total samples: 1000
Overall status: PASSED
Errors: 0
Warnings: 2
...
```

### Example 2: Save Report

```bash
python scripts/validate_dataset.py train.db \
    --output report.json \
    --save-stats stats.txt
```

### Example 3: Custom Thresholds

```bash
python scripts/validate_dataset.py train.db \
    --min-distance 0.3 \
    --max-force 200.0 \
    --energy-min -100 \
    --energy-max 100
```

### Example 4: Python API

```python
from mlff_distiller.data import MolecularDataset, DatasetValidator

# Load and validate
dataset = MolecularDataset('train.db')
validator = DatasetValidator()
report = validator.validate_dataset(dataset)

# Check results
if report.passed:
    print(f"✓ Valid dataset with {report.num_samples} samples")
else:
    print(f"✗ Failed: {report.num_errors} errors found")
    for issue in report.issues:
        if issue.severity == 'error':
            print(f"  - {issue.message}")
```

## Common Issues and Fixes

### Issue: "Overlapping atoms detected"
**Fix**: Check data generation. Atoms should be >0.5 Å apart.

### Issue: "Excessive force magnitude"
**Fix**: Either fix data or increase threshold:
```bash
--max-force 200.0
```

### Issue: "Energy per atom too high/low"
**Fix**: Adjust energy range:
```bash
--energy-min -100 --energy-max 100
```

### Issue: "N outliers detected"
**Fix**: If <1%, usually OK. If >1%, investigate those samples:
```bash
--show-all-issues  # See which samples are outliers
```

## Validation Thresholds

### Default Values (Good for Most Cases)
```
Structure:
  min_distance: 0.5 Å
  min_atoms: 10
  max_atoms: 500

Labels:
  max_force: 100.0 eV/Å
  energy_range: -50 to 50 eV/atom
  outlier_threshold: 3.0 σ

Diversity:
  min_element_samples: 10
  min_size_bins: 3
```

### When to Adjust

**Increase min_distance** (e.g., 0.8 Å):
- For coarse-grained models
- If you see many false "overlapping" warnings

**Decrease min_distance** (e.g., 0.3 Å):
- For high-energy configurations
- For reactive systems

**Increase max_force** (e.g., 200 eV/Å):
- For MD trajectories with high kinetic energy
- For reactive/transition state data

**Increase energy_range**:
- For systems with reference energies far from isolated atoms
- For high-energy configurations

**Increase outlier_threshold** (e.g., 4.0 σ):
- For small datasets (<100 samples)
- If getting too many false positives

## Integration with Training

### Pre-Training Check

```python
from mlff_distiller.data import MolecularDataset, DatasetValidator

# Load data
train_data = MolecularDataset('train.db')

# Validate
validator = DatasetValidator()
report = validator.validate_dataset(train_data)

# Fail fast if issues
if not report.passed:
    raise ValueError(f"Dataset validation failed: {report.num_errors} errors")

# Proceed with training
print("Dataset validated. Starting training...")
```

### CI/CD Pipeline

```yaml
# .github/workflows/test.yml
- name: Validate training data
  run: |
    python scripts/validate_dataset.py data/train.db || exit 1
```

## Understanding Reports

### Severity Levels

**ERROR**: Must fix before training
- Overlapping atoms
- NaN/Inf in data
- Excessive forces (>100 eV/Å)
- Invalid PBC

**WARNING**: Should investigate
- System too small/large
- Large forces (>50 eV/Å)
- Energy outside typical range
- Limited diversity

**INFO**: FYI, usually OK
- Minor outliers (<0.1%)
- Composition variety notes
- Distribution characteristics

### Exit Codes

- `0`: Validation passed
- `1`: Validation failed (if errors found)

Use `--no-fail-on-error` to always return 0 (but still report issues).

## Need More Details?

See full documentation: [docs/DATASET_VALIDATION.md](./DATASET_VALIDATION.md)

## Quick Command Reference

```bash
# Basic validation
python scripts/validate_dataset.py <file> --format <ase|hdf5|xyz>

# With output
python scripts/validate_dataset.py <file> --output report.json

# Custom thresholds
python scripts/validate_dataset.py <file> \
    --min-distance 0.3 \
    --max-force 150.0 \
    --energy-min -100 \
    --energy-max 100

# Quick check (first N samples)
python scripts/validate_dataset.py <file> --max-samples 100

# Quiet mode (for scripts)
python scripts/validate_dataset.py <file> --quiet

# Show all issues
python scripts/validate_dataset.py <file> --show-all-issues
```

## Python API Quick Reference

```python
from mlff_distiller.data import (
    MolecularDataset,
    DatasetValidator,
    StructureValidator,
    LabelValidator,
    DiversityValidator,
)

# Basic usage
dataset = MolecularDataset('data.db')
validator = DatasetValidator()
report = validator.validate_dataset(dataset)

# Custom validators
struct_val = StructureValidator(min_distance=0.3)
label_val = LabelValidator(max_force=200.0)
div_val = DiversityValidator(min_element_samples=5)

validator = DatasetValidator(
    structure_validator=struct_val,
    label_validator=label_val,
    diversity_validator=div_val,
)

# Single sample validation
sample = dataset[0]
issues = validator.validate_sample(sample)

# Access results
print(f"Passed: {report.passed}")
print(f"Errors: {report.num_errors}")
print(f"Warnings: {report.num_warnings}")
print(f"Statistics: {report.statistics}")

# Save report
import json
with open('report.json', 'w') as f:
    json.dump(report.to_dict(), f, indent=2)
```

---

**Ready to validate your data!**

For comprehensive documentation, see [docs/DATASET_VALIDATION.md](./DATASET_VALIDATION.md).
