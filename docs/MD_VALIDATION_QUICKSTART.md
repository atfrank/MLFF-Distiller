# MD Validation Quickstart Guide

**Purpose**: Quick validation that student model is stable for molecular dynamics before heavy optimization investment

**Duration**: ~1 hour (30 min setup + 30 min compute)
**Risk**: Low (read-only validation)
**Decision**: Go/No-Go for Phase 1 optimizations

---

## Why Validate?

Before investing 3 days in optimization work, we need to verify that the student model is **stable for MD simulations**. This quick test will catch critical issues like:
- Energy drift (poor conservation)
- Temperature instability
- Atomic explosions
- Force discontinuities

**Decision Criteria**:
- ✅ PASS: Energy drift <1% → Proceed with optimizations
- ❌ FAIL: Energy drift >1% → Fix model first

---

## Quick Start

```bash
# Activate environment
conda activate mlff-py312

# Run validation
python scripts/quick_md_validation.py \
    --checkpoint checkpoints/best_model.pt \
    --timestep 0.5 \
    --duration 100 \
    --output validation_results/quick_nve

# Check results
cat validation_results/quick_nve/validation_report.txt
```

**Expected Runtime**: 30-45 minutes
**Expected Result**: PASS (energy drift <1%)

---

## Test Configuration

### System Details
- **Molecule**: Benzene (C6H6)
- **Atoms**: 12 (6 carbon + 6 hydrogen)
- **Size**: Small test system for quick validation

### MD Parameters
- **Ensemble**: NVE (constant energy, volume)
- **Duration**: 100 picoseconds (100,000 femtoseconds)
- **Timestep**: 0.5 femtoseconds
- **Steps**: 200,000 steps
- **Initial Temperature**: 300 K
- **Data Recording**: Every 10 steps

### Why These Settings?
- **Small system**: Fast computation (~30 min)
- **NVE ensemble**: Tests energy conservation directly
- **Long duration**: 100ps sufficient to detect drift
- **Small timestep**: 0.5fs ensures stability

---

## Validation Script

Create `scripts/quick_md_validation.py`:

```python
#!/usr/bin/env python3
"""
Quick MD Validation - 100ps NVE trajectory

Verifies that the student model is stable for MD simulations
before proceeding with heavy optimization work.

Usage:
    python scripts/quick_md_validation.py \
        --checkpoint checkpoints/best_model.pt \
        --timestep 0.5 \
        --duration 100 \
        --output validation_results/quick_nve
"""

import sys
from pathlib import Path
import numpy as np
from ase.build import molecule
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
import matplotlib.pyplot as plt

# Add src to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from mlff_distiller.inference import StudentForceFieldCalculator

def run_nve_validation(
    checkpoint_path,
    timestep_fs=0.5,
    duration_ps=100,
    temperature_K=300,
    output_dir='validation_results/quick_nve'
):
    """Run quick NVE validation trajectory."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Quick MD Validation - 100ps NVE")
    print("="*60)

    # Create test system
    print("\n[1/5] Creating test system (benzene)...")
    atoms = molecule('C6H6')
    print(f"  Atoms: {len(atoms)}")
    print(f"  Formula: {atoms.get_chemical_formula()}")

    # Set up calculator
    print("\n[2/5] Loading student model...")
    calc = StudentForceFieldCalculator(
        checkpoint_path=checkpoint_path,
        device='cuda',
        use_compile=False,  # No optimizations for validation
        use_fp16=False
    )
    atoms.calc = calc
    print("  ✓ Calculator ready")

    # Initialize velocities
    print(f"\n[3/5] Initializing velocities (T={temperature_K}K)...")
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
    print(f"  Initial temperature: {atoms.get_temperature():.1f} K")

    # Set up NVE dynamics
    print(f"\n[4/5] Setting up NVE dynamics...")
    print(f"  Timestep: {timestep_fs} fs")
    print(f"  Duration: {duration_ps} ps")
    n_steps = int(duration_ps * 1000 / timestep_fs)
    print(f"  Total steps: {n_steps:,}")

    dyn = VelocityVerlet(atoms, timestep=timestep_fs * units.fs)

    # Storage for analysis
    energies = []
    temperatures = []
    times = []

    def record_state():
        energies.append(atoms.get_potential_energy() + atoms.get_kinetic_energy())
        temperatures.append(atoms.get_temperature())
        times.append(dyn.get_time() / units.fs)

        # Progress update
        if len(times) % 1000 == 0:
            progress = len(times) / (n_steps / 10) * 100
            print(f"  Progress: {progress:.1f}% ({len(times)*10:,}/{n_steps:,} steps)", end='\r')

    # Attach observer (record every 10 steps)
    dyn.attach(record_state, interval=10)

    # Run trajectory
    print("\n[5/5] Running MD trajectory...")
    dyn.run(n_steps)
    print("\n  ✓ Trajectory complete")

    # Convert to arrays
    energies = np.array(energies)
    temperatures = np.array(temperatures)
    times = np.array(times)

    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)

    energy_drift = (energies[-1] - energies[0]) / energies[0] * 100
    energy_std = energies.std() / energies.mean() * 100
    temp_mean = temperatures.mean()
    temp_std = temperatures.std()

    print(f"\nEnergy Conservation:")
    print(f"  Initial energy: {energies[0]:.6f} eV")
    print(f"  Final energy:   {energies[-1]:.6f} eV")
    print(f"  Drift:          {energy_drift:+.3f}%")
    print(f"  Fluctuations:   {energy_std:.3f}%")

    print(f"\nTemperature Stability:")
    print(f"  Target:         {temperature_K:.1f} K")
    print(f"  Mean:           {temp_mean:.1f} K")
    print(f"  Std:            {temp_std:.1f} K")
    print(f"  Deviation:      {abs(temp_mean - temperature_K):.1f} K")

    # Pass/fail criteria
    energy_drift_pass = abs(energy_drift) < 1.0
    energy_std_pass = energy_std < 0.5
    temp_pass = abs(temp_mean - temperature_K) < 20

    passed = energy_drift_pass and energy_std_pass and temp_pass

    print(f"\n" + "="*60)
    print("RESULT: {'PASS ✓' if passed else 'FAIL ✗'}")
    print("="*60)

    print(f"\nCriteria:")
    print(f"  Energy drift <1%:           {'✓ PASS' if energy_drift_pass else '✗ FAIL'}")
    print(f"  Energy fluctuations <0.5%:  {'✓ PASS' if energy_std_pass else '✗ FAIL'}")
    print(f"  Temperature within 20K:     {'✓ PASS' if temp_pass else '✗ FAIL'}")

    # Generate plots
    print(f"\nGenerating plots...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Energy conservation
    ax1.plot(times / 1000, energies - energies[0], 'b-', linewidth=1)
    ax1.set_xlabel('Time (ps)')
    ax1.set_ylabel('Energy Drift (eV)')
    ax1.set_title(f'Energy Conservation (Drift: {energy_drift:+.3f}%)')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='r', linestyle='--', alpha=0.5)

    # Temperature
    ax2.plot(times / 1000, temperatures, 'r-', linewidth=1)
    ax2.axhline(temperature_K, color='k', linestyle='--', alpha=0.5, label='Target')
    ax2.set_xlabel('Time (ps)')
    ax2.set_ylabel('Temperature (K)')
    ax2.set_title(f'Temperature Stability (Mean: {temp_mean:.1f}±{temp_std:.1f}K)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plot_path = output_dir / 'quick_md_validation.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  ✓ Plot saved: {plot_path}")

    # Write report
    report = f"""
Quick MD Validation Results
===========================

System: Benzene (C6H6, 12 atoms)
Duration: {duration_ps} ps
Timestep: {timestep_fs} fs
Temperature: {temperature_K} K

Energy Conservation:
  Initial energy: {energies[0]:.6f} eV
  Final energy:   {energies[-1]:.6f} eV
  Drift:          {energy_drift:+.3f}%
  Fluctuations:   {energy_std:.3f}%

Temperature:
  Target:         {temperature_K:.1f} K
  Mean:           {temp_mean:.1f} K
  Std:            {temp_std:.1f} K
  Deviation:      {abs(temp_mean - temperature_K):.1f} K

Result: {'PASS ✓' if passed else 'FAIL ✗'}

Criteria:
  Energy drift <1%:           {'✓ PASS' if energy_drift_pass else '✗ FAIL'}
  Energy fluctuations <0.5%:  {'✓ PASS' if energy_std_pass else '✗ FAIL'}
  Temperature within 20K:     {'✓ PASS' if temp_pass else '✗ FAIL'}

{'Decision: Proceed with Phase 1 optimizations' if passed else 'Decision: Model requires fixes before optimization'}
"""

    report_path = output_dir / 'validation_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  ✓ Report saved: {report_path}")

    # Save raw data
    np.savez(
        output_dir / 'trajectory_data.npz',
        times=times,
        energies=energies,
        temperatures=temperatures
    )
    print(f"  ✓ Data saved: {output_dir / 'trajectory_data.npz'}")

    return passed

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Quick MD validation for student model')
    parser.add_argument('--checkpoint', type=Path, default='checkpoints/best_model.pt',
                        help='Path to student model checkpoint')
    parser.add_argument('--timestep', type=float, default=0.5,
                        help='MD timestep in femtoseconds (default: 0.5)')
    parser.add_argument('--duration', type=float, default=100,
                        help='Trajectory duration in picoseconds (default: 100)')
    parser.add_argument('--temperature', type=float, default=300,
                        help='Initial temperature in Kelvin (default: 300)')
    parser.add_argument('--output', type=Path, default='validation_results/quick_nve',
                        help='Output directory (default: validation_results/quick_nve)')
    args = parser.parse_args()

    passed = run_nve_validation(
        args.checkpoint,
        args.timestep,
        args.duration,
        args.temperature,
        args.output
    )

    sys.exit(0 if passed else 1)
```

---

## Expected Output

### Terminal Output
```
============================================================
Quick MD Validation - 100ps NVE
============================================================

[1/5] Creating test system (benzene)...
  Atoms: 12
  Formula: C6H6

[2/5] Loading student model...
  ✓ Calculator ready

[3/5] Initializing velocities (T=300K)...
  Initial temperature: 298.3 K

[4/5] Setting up NVE dynamics...
  Timestep: 0.5 fs
  Duration: 100 ps
  Total steps: 200,000

[5/5] Running MD trajectory...
  Progress: 100.0% (200,000/200,000 steps)
  ✓ Trajectory complete

============================================================
ANALYSIS
============================================================

Energy Conservation:
  Initial energy: -123.456789 eV
  Final energy:   -123.456012 eV
  Drift:          +0.063%
  Fluctuations:   0.012%

Temperature Stability:
  Target:         300.0 K
  Mean:           301.2 K
  Std:            8.3 K
  Deviation:      1.2 K

============================================================
RESULT: PASS ✓
============================================================

Criteria:
  Energy drift <1%:           ✓ PASS
  Energy fluctuations <0.5%:  ✓ PASS
  Temperature within 20K:     ✓ PASS

Generating plots...
  ✓ Plot saved: validation_results/quick_nve/quick_md_validation.png
  ✓ Report saved: validation_results/quick_nve/validation_report.txt
  ✓ Data saved: validation_results/quick_nve/trajectory_data.npz
```

---

## Interpreting Results

### PASS Scenario (Expected)

**Energy Drift**: < ±1%
- Indicates good energy conservation
- Model forces are consistent and stable
- Safe to proceed with optimizations

**Example PASS**:
```
Energy drift: +0.063%  ✓
Temperature: 301.2 ± 8.3 K  ✓
Result: PASS ✓
Decision: Proceed with Phase 1 optimizations
```

---

### FAIL Scenario (Requires Action)

**Energy Drift**: > ±1%
- Poor energy conservation
- Model may have force discontinuities
- Need to investigate and fix before optimization

**Example FAIL**:
```
Energy drift: +2.5%  ✗
Temperature: 350.0 ± 45.0 K  ✗
Result: FAIL ✗
Decision: Model requires fixes before optimization
```

**Actions if FAIL**:
1. Check training convergence
2. Verify model architecture
3. Test on simpler systems
4. Review force computation
5. Increase training epochs

---

## Validation Plots

### Energy Conservation Plot

**What to Look For**:
- ✅ Flat line around zero (good conservation)
- ❌ Steady upward/downward trend (drift)
- ❌ Large oscillations (instability)

```
Energy Drift (eV)
     ^
 0.1 |     .  .   .  .   .  .
     |   .          .          .
   0 |---------------------------
     |
-0.1 |
     +--------------------------> Time (ps)
     0                        100
```

---

### Temperature Plot

**What to Look For**:
- ✅ Oscillates around target (300K)
- ✅ Fluctuations < ±20K
- ❌ Steady increase (heating)
- ❌ Large fluctuations (instability)

```
Temperature (K)
     ^
 320 |  .     .     .     .
     |    . .   . .   . .
 300 |------.------.------.------  (target)
     |      . .   . .   . .
 280 |        .     .     .
     +--------------------------> Time (ps)
     0                        100
```

---

## Troubleshooting

### Issue 1: CUDA Out of Memory

**Symptom**:
```
RuntimeError: CUDA out of memory
```

**Solution**:
```python
# Reduce system size
atoms = molecule('H2O')  # Use water instead of benzene

# Or run on CPU
calc = StudentForceFieldCalculator(
    checkpoint_path=checkpoint_path,
    device='cpu'  # Slower but uses less memory
)
```

---

### Issue 2: Slow Performance

**Symptom**: Trajectory takes > 1 hour

**Solution**:
```bash
# Reduce duration
python scripts/quick_md_validation.py \
    --duration 50  # 50ps instead of 100ps

# Or increase timestep (less recommended)
python scripts/quick_md_validation.py \
    --timestep 1.0  # 1fs instead of 0.5fs
```

---

### Issue 3: Atomic Explosion

**Symptom**:
```
RuntimeError: Forces are NaN
```

**Solution**:
1. Check checkpoint is correct
2. Reduce timestep to 0.1fs
3. Test on single point energy first
4. Review model training

---

### Issue 4: High Energy Drift

**Symptom**: Energy drift > 1%

**Diagnosis**:
```bash
# Test shorter trajectory
python scripts/quick_md_validation.py --duration 10

# Test different molecule
python -c "
from ase.build import molecule
atoms = molecule('H2O')
# ... test with water
"

# Check force consistency
python scripts/test_force_consistency.py
```

---

## Next Steps

### If PASS ✓
1. Proceed with torch.compile() implementation (Issue #27)
2. Proceed with FP16 implementation (Issue #28)
3. Continue Phase 1 optimizations

### If FAIL ✗
1. Investigate energy drift cause
2. Review training logs
3. Test on simpler systems
4. Consult with team
5. Do NOT proceed with optimization until fixed

---

## Additional Validation (Optional)

### Extended Duration
```bash
# Run longer trajectory for confidence
python scripts/quick_md_validation.py --duration 500
```

### Multiple Systems
```bash
# Test on different molecules
for mol in H2O NH3 CH4; do
    python scripts/quick_md_validation.py \
        --output validation_results/${mol}_nve
done
```

### Different Ensembles
```bash
# NVT (constant temperature)
python scripts/run_nvt_validation.py --duration 100

# NPT (constant pressure)
python scripts/run_npt_validation.py --duration 100
```

---

## FAQ

**Q: Why benzene?**
A: Good test system - organic, realistic size, not too simple.

**Q: Why 100ps?**
A: Long enough to detect drift, short enough to be practical (~30 min).

**Q: Why NVE instead of NVT?**
A: NVE tests energy conservation directly (harder test).

**Q: What if I get 0.9% drift?**
A: Close to threshold - run longer (500ps) to confirm.

**Q: Can I skip validation?**
A: Not recommended - only takes 1 hour and prevents wasted optimization work.

**Q: What's a realistic energy drift?**
A: Good models: <0.1%, Acceptable: <0.5%, Marginal: <1%, Poor: >1%

---

## References

- ASE MD Documentation: https://wiki.fysik.dtu.dk/ase/ase/md.html
- NVE Ensemble: https://en.wikipedia.org/wiki/Microcanonical_ensemble
- Energy Conservation in MD: https://doi.org/10.1063/1.456153

---

## Contact

**Coordinator**: ml-distillation-coordinator
**Working Directory**: `/home/aaron/ATX/software/MLFF_Distiller`
**Environment**: `mlff-py312`
**Script**: `scripts/quick_md_validation.py`
