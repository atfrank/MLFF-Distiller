# Original Student Model (427K) - MD Validation Report

**Issue #33: Production Approval Testing**

**Date**: 2025-11-25
**Model**: Original Student (427K parameters)
**Checkpoint**: `/home/aaron/ATX/software/MLFF_Distiller/checkpoints/best_model.pt`

---

## Executive Summary

**Test Molecules**: 5
**Simulation Duration**: 10.0 picoseconds per molecule
**Total MD Time**: 50.0 picoseconds

### Test Results

- **Stability Tests**: 5/5 PASSED
- **Energy Conservation (<1.0% drift)**: 5/5 PASSED
- **Force Accuracy (<0.2 eV/Å RMSE)**: 0/5 PASSED

### Production Approval Decision

**STATUS: NOT APPROVED FOR PRODUCTION**

The model failed one or more validation criteria. Review detailed 
results below and address issues before deployment.

---

## Detailed Results

### 0

**Formula**: C19N2O
**Atoms**: 22
**Simulation Steps**: 20,000
**Duration**: 10.00 ps

**Stability**: PASSED

**Energy Conservation**: PASSED
- Total energy drift: 0.398%
- Maximum drift: 0.603%
- Conservation ratio: 0.998582
- Kinetic energy std: 0.2516 eV
- Potential energy std: 0.4480 eV

**Force Accuracy vs Teacher**: SKIPPED

**Visualizations**:
- Energy evolution: `plots/0_energy.png`
- Force accuracy: `plots/0_forces.png`
- Temperature: `plots/0_temperature.png`
- Trajectory: `trajectories/0.traj`

---

### 2

**Formula**: C12N2O
**Atoms**: 15
**Simulation Steps**: 20,000
**Duration**: 10.00 ps

**Stability**: PASSED

**Energy Conservation**: PASSED
- Total energy drift: 0.021%
- Maximum drift: 0.046%
- Conservation ratio: 0.999847
- Kinetic energy std: 0.0190 eV
- Potential energy std: 0.0953 eV

**Force Accuracy vs Teacher**: SKIPPED

**Visualizations**:
- Energy evolution: `plots/2_energy.png`
- Force accuracy: `plots/2_forces.png`
- Temperature: `plots/2_temperature.png`
- Trajectory: `trajectories/2.traj`

---

### 4

**Formula**: C22NO2
**Atoms**: 25
**Simulation Steps**: 20,000
**Duration**: 10.00 ps

**Stability**: PASSED

**Energy Conservation**: PASSED
- Total energy drift: -0.196%
- Maximum drift: 0.496%
- Conservation ratio: 0.999168
- Kinetic energy std: 0.1667 eV
- Potential energy std: 0.5173 eV

**Force Accuracy vs Teacher**: SKIPPED

**Visualizations**:
- Energy evolution: `plots/4_energy.png`
- Force accuracy: `plots/4_forces.png`
- Temperature: `plots/4_temperature.png`
- Trajectory: `trajectories/4.traj`

---

### 6

**Formula**: C19FN4O
**Atoms**: 25
**Simulation Steps**: 20,000
**Duration**: 10.00 ps

**Stability**: PASSED

**Energy Conservation**: PASSED
- Total energy drift: 0.283%
- Maximum drift: 0.283%
- Conservation ratio: 0.998864
- Kinetic energy std: 0.2294 eV
- Potential energy std: 0.2920 eV

**Force Accuracy vs Teacher**: SKIPPED

**Visualizations**:
- Energy evolution: `plots/6_energy.png`
- Force accuracy: `plots/6_forces.png`
- Temperature: `plots/6_temperature.png`
- Trajectory: `trajectories/6.traj`

---

### 9

**Formula**: C26N3O3
**Atoms**: 32
**Simulation Steps**: 20,000
**Duration**: 10.00 ps

**Stability**: PASSED

**Energy Conservation**: PASSED
- Total energy drift: -0.029%
- Maximum drift: 0.555%
- Conservation ratio: 0.998707
- Kinetic energy std: 0.3261 eV
- Potential energy std: 0.7155 eV

**Force Accuracy vs Teacher**: SKIPPED

**Visualizations**:
- Energy evolution: `plots/9_energy.png`
- Force accuracy: `plots/9_forces.png`
- Temperature: `plots/9_temperature.png`
- Trajectory: `trajectories/9.traj`

---

## Test Configuration

```
Temperature: 300.0 K
Timestep: 0.5 fs
Simulation steps: 20,000
Duration per molecule: 10.0 ps
Force comparison interval: 200 steps

Success Criteria:
  - Energy drift threshold: <1.0%
  - Force RMSE threshold: <0.2 eV/Å
  - No NaN or stability issues
```
