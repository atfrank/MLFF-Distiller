# Dataset Quality Validation Framework

Comprehensive validation tools to ensure dataset quality for machine learning force field training.

## Overview

The validation framework provides four categories of checks:

1. **Structure Validation**: Geometry and physical constraints
2. **Label Validation**: Energy and force reasonableness
3. **Diversity Validation**: Coverage and variety
4. **Statistical Validation**: Outlier detection and distributions

## Quick Start

### Command-Line Usage

```bash
# Basic validation
python scripts/validate_dataset.py structures.db --format ase

# With custom thresholds
python scripts/validate_dataset.py data.h5 --format hdf5 \
    --min-distance 0.3 \
    --max-force 50.0 \
    --energy-min -100 \
    --energy-max 100

# Save detailed report
python scripts/validate_dataset.py data.db \
    --output validation_report.json \
    --save-stats statistics.txt

# Quick check (first 100 samples)
python scripts/validate_dataset.py large_dataset.db --max-samples 100
```

### Python API Usage

```python
from mlff_distiller.data import MolecularDataset, DatasetValidator

# Load dataset
dataset = MolecularDataset('structures.db', format='ase')

# Create validator
validator = DatasetValidator()

# Run validation
report = validator.validate_dataset(dataset)

# Check results
print(f"Validation passed: {report.passed}")
print(f"Errors: {report.num_errors}")
print(f"Warnings: {report.num_warnings}")

# Print full report
print(report.summary())

# Save as JSON
import json
with open('report.json', 'w') as f:
    json.dump(report.to_dict(), f, indent=2)
```

## Validation Categories

### 1. Structure Validation

**StructureValidator** checks atomic geometry and physical constraints:

#### Checks Performed:

- **No NaN/Inf in positions**: Ensures all atomic coordinates are finite
- **No overlapping atoms**: Minimum interatomic distance check (default: 0.5 Å)
- **System size in range**: Number of atoms between min/max (default: 10-500)
- **Valid periodic boundary conditions**: Cell parameters match PBC flags
- **Reasonable cell dimensions**: Not too small (<1 Å) or too large (>1000 Å)
- **Close atoms warning**: Atoms closer than 1.0 Å trigger warnings

#### Custom Parameters:

```python
from mlff_distiller.data.validation import StructureValidator

validator = StructureValidator(
    min_distance=0.5,      # Minimum interatomic distance (Å)
    max_distance=5.0,      # Maximum bond length (Å)
    min_atoms=10,          # Minimum system size
    max_atoms=500,         # Maximum system size
    check_pbc=True,        # Validate PBC/cell consistency
)
```

#### Example Issues:

```
ERROR (structure): [Sample 42] Overlapping atoms detected: distance 0.3 Å < 0.5 Å
WARNING (structure): [Sample 15] System too small: 5 atoms (minimum: 10)
ERROR (structure): [Sample 23] Non-finite values (NaN/Inf) in atomic positions
```

### 2. Label Validation

**LabelValidator** checks energy and force labels:

#### Checks Performed:

- **No NaN/Inf in energies**: All energy values finite
- **No NaN/Inf in forces**: All force components finite
- **Energy in reasonable range**: Per-atom energy between bounds (default: -50 to 50 eV)
- **Forces not excessive**: Force magnitude below threshold (default: 100 eV/Å)
- **Force magnitude warnings**: Large forces (>50 eV/Å) trigger warnings
- **Statistical outlier detection**: Z-score based outlier flagging (>3σ)

#### Custom Parameters:

```python
from mlff_distiller.data.validation import LabelValidator

validator = LabelValidator(
    max_force=100.0,                    # Maximum force magnitude (eV/Å)
    energy_range=(-50.0, 50.0),         # Energy per atom range (eV)
    outlier_threshold=3.0,              # Z-score threshold for outliers
)
```

#### Example Issues:

```
ERROR (label): [Sample 8] Excessive force magnitude: 150.3 eV/Å > 100.0 eV/Å
WARNING (label): [Sample 12] Energy per atom too high: 75.2 eV/atom
ERROR (label): [Sample 31] Non-finite energy value: nan
INFO (statistical): energy_per_atom: 5 outliers (0.05%) detected
```

### 3. Diversity Validation

**DiversityValidator** checks dataset coverage and variety:

#### Checks Performed:

- **Element coverage**: Sufficient samples for each element type
- **Element balance**: No single element dominates (>90%)
- **System size diversity**: Variety of system sizes
- **Size distribution**: Not concentrated in narrow range
- **Composition variety**: Different chemical compositions

#### Custom Parameters:

```python
from mlff_distiller.data.validation import DiversityValidator

validator = DiversityValidator(
    min_element_samples=10,     # Minimum samples per element
    min_size_bins=3,            # Minimum unique system sizes
)
```

#### Example Issues:

```
WARNING (diversity): 2 elements have < 10 samples
WARNING (diversity): Dataset dominated by single element: 92.5% of atoms
INFO (diversity): Limited composition variety: 5 unique out of 100
```

### 4. Statistical Validation

**Statistical Analysis** performed across the dataset:

#### Computed Statistics:

**Energy Statistics:**
- Mean, std, min, max (total and per-atom)
- Distribution analysis
- Outlier detection (>3σ from mean)

**Force Statistics:**
- Mean, std, min, max magnitudes
- RMS force
- Distribution analysis
- Outlier detection

**System Statistics:**
- Size distribution (mean, std, min, max)
- Element distribution
- Composition variety

#### Outlier Severity:

- **ERROR**: >1% outliers (dataset quality issue)
- **WARNING**: 0.1-1% outliers (moderate concern)
- **INFO**: <0.1% outliers (acceptable)

## Validation Reports

### Report Structure

```python
ValidationReport(
    passed=True/False,           # Overall pass/fail
    issues=[...],                # List of ValidationIssue objects
    statistics={...},            # Dataset statistics dict
    num_samples=1000,            # Number of samples validated
    num_errors=0,                # Count of errors
    num_warnings=5,              # Count of warnings
    num_info=2,                  # Count of info messages
)
```

### Report Methods

```python
# Text summary
print(report.summary())

# Convert to dictionary (JSON-serializable)
data = report.to_dict()

# Count issues by severity
errors = [i for i in report.issues if i.severity == 'error']
warnings = [i for i in report.issues if i.severity == 'warning']

# Access statistics
mean_energy = report.statistics['energy_mean']
force_rms = report.statistics['force_magnitude_rms']
```

### Example Report Output

```
================================================================================
DATASET VALIDATION REPORT
================================================================================
Total samples: 1000
Overall status: PASSED
Errors: 0
Warnings: 3
Info: 2

ISSUES:
--------------------------------------------------------------------------------
WARNING (diversity): 2 elements have < 10 samples
WARNING (label): [Sample 42] Large force magnitude: 65.3 eV/Å
INFO (diversity): Limited size diversity: only 4 unique system sizes

STATISTICS:
--------------------------------------------------------------------------------
energy_mean: -125.456789
energy_std: 12.345678
energy_per_atom_mean: -6.273339
force_magnitude_mean: 2.345678
force_magnitude_rms: 3.456789
system_size_mean: 18.500000
num_unique_elements: 3
================================================================================
```

## Custom Validation Workflows

### Validate Individual Samples

```python
validator = DatasetValidator()

sample = dataset[0]
issues = validator.validate_sample(sample, sample_idx=0)

for issue in issues:
    print(f"{issue.severity.upper()}: {issue.message}")
```

### Batch Validation

```python
validator = DatasetValidator(verbose=False)

# Validate in batches
batch_size = 100
for i in range(0, len(dataset), batch_size):
    end = min(i + batch_size, len(dataset))
    report = validator.validate_dataset(dataset, max_samples=end)
    print(f"Batch {i//batch_size}: {report.num_errors} errors")
```

### Custom Validator Combination

```python
from mlff_distiller.data.validation import (
    DatasetValidator,
    StructureValidator,
    LabelValidator,
    DiversityValidator,
)

# Create custom validators with specific thresholds
struct_val = StructureValidator(
    min_distance=0.3,
    min_atoms=5,
    max_atoms=1000,
)

label_val = LabelValidator(
    max_force=50.0,
    energy_range=(-100.0, 100.0),
    outlier_threshold=2.5,  # Stricter outlier detection
)

div_val = DiversityValidator(
    min_element_samples=5,
    min_size_bins=5,
)

# Combine into main validator
validator = DatasetValidator(
    structure_validator=struct_val,
    label_validator=label_val,
    diversity_validator=div_val,
    fail_on_error=True,
    verbose=True,
)

report = validator.validate_dataset(dataset)
```

## Integration with Training Pipeline

### Pre-Training Validation

```python
from mlff_distiller.data import MolecularDataset, DatasetValidator

# Load training data
train_dataset = MolecularDataset('train.db', format='ase')

# Validate before training
validator = DatasetValidator()
report = validator.validate_dataset(train_dataset)

if not report.passed:
    print("Dataset validation failed!")
    print(report.summary())
    raise ValueError(f"Found {report.num_errors} errors in training data")

print("Dataset validation passed. Starting training...")
```

### Validation in CI/CD

```python
# ci_validate.py
import sys
from mlff_distiller.data import MolecularDataset, DatasetValidator

dataset = MolecularDataset(sys.argv[1], format='ase')
validator = DatasetValidator(verbose=False)
report = validator.validate_dataset(dataset)

# Exit with non-zero if validation fails
sys.exit(0 if report.passed else 1)
```

```bash
# In GitHub Actions or CI pipeline
python ci_validate.py data/train.db || exit 1
```

## Quality Standards

### Error Thresholds (FAIL validation):

- Any overlapping atoms (distance < 0.5 Å)
- Any NaN/Inf in positions, energies, or forces
- Any excessive forces (>100 eV/Å)
- Invalid PBC configuration
- >1% statistical outliers

### Warning Thresholds (PASS with warnings):

- System size outside typical range
- Energy per atom outside typical range
- Large forces (>50 eV/Å but <100 eV/Å)
- Limited element diversity
- 0.1-1% statistical outliers

### Info Level:

- Minor diversity issues
- <0.1% statistical outliers
- Distribution characteristics

## CLI Reference

### All Command-Line Options

```bash
python scripts/validate_dataset.py --help
```

**Required:**
- `data_path`: Path to dataset file

**Dataset Options:**
- `--format {ase,hdf5,xyz}`: Dataset format (default: ase)
- `--max-samples N`: Validate only first N samples

**Structure Validation:**
- `--min-distance FLOAT`: Minimum interatomic distance (Å, default: 0.5)
- `--min-atoms INT`: Minimum atoms per structure (default: 10)
- `--max-atoms INT`: Maximum atoms per structure (default: 500)

**Label Validation:**
- `--max-force FLOAT`: Maximum force magnitude (eV/Å, default: 100.0)
- `--energy-min FLOAT`: Minimum energy per atom (eV, default: -50.0)
- `--energy-max FLOAT`: Maximum energy per atom (eV, default: 50.0)
- `--outlier-threshold FLOAT`: Outlier detection σ threshold (default: 3.0)

**Diversity Validation:**
- `--min-element-samples INT`: Minimum samples per element (default: 10)
- `--min-size-bins INT`: Minimum unique system sizes (default: 3)

**Output Options:**
- `--output FILE`: Save JSON report
- `--save-stats FILE`: Save statistics text file
- `--show-all-issues`: Show all issues including info level
- `--quiet`: Suppress progress output
- `--no-fail-on-error`: Don't fail on errors (still report them)

## Examples

### Example 1: Strict Validation

```bash
python scripts/validate_dataset.py train.db \
    --min-distance 0.8 \
    --max-force 50.0 \
    --outlier-threshold 2.5 \
    --output strict_report.json
```

### Example 2: Quick Check

```bash
# Fast validation of first 50 samples
python scripts/validate_dataset.py large_dataset.db \
    --max-samples 50 \
    --quiet
```

### Example 3: Permissive Validation

```bash
# Allow larger systems and forces
python scripts/validate_dataset.py md_data.db \
    --max-atoms 1000 \
    --max-force 200.0 \
    --no-fail-on-error
```

### Example 4: Full Analysis

```python
from mlff_distiller.data import MolecularDataset, DatasetValidator

# Load dataset
dataset = MolecularDataset('structures.db')

# Validate
validator = DatasetValidator(verbose=True)
report = validator.validate_dataset(dataset)

# Detailed analysis
print("\n=== ENERGY STATISTICS ===")
print(f"Mean: {report.statistics['energy_per_atom_mean']:.3f} eV/atom")
print(f"Std:  {report.statistics['energy_per_atom_std']:.3f} eV/atom")
print(f"Range: [{report.statistics['energy_per_atom_min']:.3f}, "
      f"{report.statistics['energy_per_atom_max']:.3f}] eV/atom")

print("\n=== FORCE STATISTICS ===")
print(f"RMS: {report.statistics['force_magnitude_rms']:.3f} eV/Å")
print(f"Max: {report.statistics['force_magnitude_max']:.3f} eV/Å")

print("\n=== DIVERSITY ===")
print(f"Elements: {report.statistics['num_unique_elements']}")
print(f"Size range: [{report.statistics['system_size_min']}, "
      f"{report.statistics['system_size_max']}] atoms")

# Filter issues by category
structure_issues = [i for i in report.issues if i.category == 'structure']
label_issues = [i for i in report.issues if i.category == 'label']
diversity_issues = [i for i in report.issues if i.category == 'diversity']

print(f"\nStructure issues: {len(structure_issues)}")
print(f"Label issues: {len(label_issues)}")
print(f"Diversity issues: {len(diversity_issues)}")
```

## Best Practices

1. **Always validate before training**: Catch data quality issues early
2. **Save validation reports**: Track data quality over time
3. **Use appropriate thresholds**: Adjust based on your application
4. **Review warnings**: Even if validation passes, investigate warnings
5. **Validate all splits**: Check train, validation, and test sets separately
6. **Monitor outliers**: Investigate samples flagged as outliers
7. **Check diversity**: Ensure balanced coverage of chemical space
8. **Regular validation**: Re-validate when adding new data

## Troubleshooting

### High False Positive Rate

If validation flags too many valid structures:

```python
# Relax thresholds
validator = DatasetValidator(
    structure_validator=StructureValidator(min_distance=0.3),
    label_validator=LabelValidator(max_force=200.0),
    fail_on_error=False,
)
```

### Memory Issues with Large Datasets

```python
# Validate in batches
validator = DatasetValidator(verbose=False)

for i in range(0, len(dataset), 1000):
    report = validator.validate_dataset(dataset, max_samples=min(i+1000, len(dataset)))
    if not report.passed:
        print(f"Batch {i//1000} failed validation")
```

### Performance Optimization

```bash
# Quick validation of subset
python scripts/validate_dataset.py huge_dataset.db \
    --max-samples 500 \
    --quiet
```

## See Also

- [MolecularDataset Documentation](./DATASETS.md)
- [Training Pipeline](./TRAINING.md)
- [Testing Guide](../TESTING.md)
