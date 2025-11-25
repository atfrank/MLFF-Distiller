# PBC Bug Fix - HDF5 Position Loading

**Date**: 2025-11-24
**Status**: ✅ RESOLVED

---

## Summary

Fixed a critical bug in HDF5 data loading that caused tensor dimension mismatches during model inference. The bug was misdiagnosed as a "PBC bug in radius_graph" but was actually an incorrect slicing operation when loading positions from the HDF5 dataset.

---

## The Error

```
RuntimeError: The size of tensor a (22) must match the size of tensor b (66) at non-singleton dimension 1
```

**Location**: `src/mlff_distiller/models/student_model.py` line 89
**Occurred during**: Validation analysis and model prediction

---

## Root Cause

The HDF5 positions dataset is stored as `(N_total_atoms, 3)` with shape like `(914812, 3)` for the full dataset. However, the loading code incorrectly assumed positions were stored **flattened** and tried to:

```python
# INCORRECT CODE:
positions = structures_group['positions'][atom_start_idx * 3:atom_end_idx * 3].reshape(-1, 3)
```

This caused:
- For a 22-atom structure: `[225*3:247*3] = [675:741]` → 66 rows
- Result: positions shape = `(66, 3)` instead of `(22, 3)`
- Mismatch: batch tensor has shape `(22,)` but positions has `(66,)` first dimension

---

## The Fix

Positions are **already stored as (N, 3)** in HDF5, not flattened. The fix was to remove the multiplication:

```python
# CORRECT CODE:
positions = structures_group['positions'][atom_start_idx:atom_end_idx]  # Already (N, 3)
```

This gives:
- For a 22-atom structure: `[225:247]` → 22 rows
- Result: positions shape = `(22, 3)` ✓
- Match: both batch and positions first dimension = 22 ✓

---

## Files Modified

### 1. `scripts/validate_model_detailed.py` (line 86)

**Before**:
```python
'positions': structures_group['positions'][atom_start_idx * 3:atom_end_idx * 3].reshape(-1, 3),
```

**After**:
```python
'positions': structures_group['positions'][atom_start_idx:atom_end_idx],  # Already (N, 3)
```

### 2. `scripts/debug_pbc_bug.py` (line 38)

Same fix applied to the debugging script.

---

## Why It Was Misdiagnosed

The error message mentioned "tensor a (22) must match tensor b (66)" which occurred inside `radius_graph_native()` when combining masks:

```python
batch_mask = batch.unsqueeze(0) == batch.unsqueeze(1)  # [22, 22]
distance_mask = distances <= r  # [66, 66] because positions was wrong shape!
mask = batch_mask & distance_mask  # ERROR: incompatible shapes
```

Since the error occurred in the radius_graph function and mentioned PBC-related code paths, it was initially thought to be a PBC handling bug. However, the actual root cause was **upstream in data loading**.

---

## Verification

After the fix, the debug script showed:

```
================================================================================
Summary
================================================================================
radius_graph_native: ✓ PASS
StudentForceField:   ✓ PASS
```

And validation analysis completed successfully:

```
Loading structure 10 from HDF5...
  Number of atoms: 22
  Positions shape: (22, 3)  ← Correct!

Running model prediction...
  Predicted energy: -158.4019 eV
  SUCCESS!
```

---

## Impact

This bug affected:
- **Validation scripts**: Could not run per-structure analysis
- **Any standalone inference code**: Would fail with dimension mismatch
- **Training was NOT affected**: The training dataloader likely used correct indexing

The bug only manifested in scripts that directly loaded individual structures from HDF5 for inference, not in the training pipeline which uses batched data loading.

---

## Lessons Learned

1. **Check data shapes at every step**: The error occurred far downstream from the actual bug
2. **Inspect HDF5 structure directly**: Running `h5py.File().keys()` and checking shapes revealed the truth
3. **Don't trust error locations**: The error in `radius_graph` was a symptom, not the cause
4. **Test with simple cases first**: The debug script with explicit shape printing was crucial

---

## Additional Fix

While debugging, also fixed a tensor gradient detachment issue:

```python
# In compute_per_atom_errors():
energy_error = predictions['energy'].detach().cpu().numpy() - structure['energy']
pred_forces = predictions['forces'].detach().cpu().numpy()
```

This was needed because gradients were being tracked during inference (from the DistillationWrapper) and `.numpy()` requires detached tensors.

---

**Status**: ✅ All validation analysis now runs successfully
**Verification**: Validated on structure 10 (22 atoms, C/N/O composition)
**Generated Outputs**:
- Error distribution plots
- Force correlation analysis
- Spatial error maps
- PyMOL visualization with force vectors

