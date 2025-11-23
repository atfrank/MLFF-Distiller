# Drop-in Compatibility Architecture Guide

**Version**: 1.0
**Last Updated**: 2025-11-23
**Status**: CRITICAL - Core Project Requirement

## Overview

This document defines the architectural requirements and guidelines for achieving drop-in replacement compatibility between teacher and student models. Drop-in compatibility means users can replace teacher models with distilled student models by changing a single line of code in their MD simulation scripts.

## Why Drop-in Compatibility is Critical

### The Use Case
These distilled models are designed for molecular dynamics simulations where:
- Models are called **millions to billions of times** per simulation
- Users have existing, tested MD workflows
- Changing MD scripts is error-prone and time-consuming
- Users want performance gains without workflow disruption

### The Goal
Enable this use case:
```python
# User's existing MD script (working, tested, production)
# from teacher_package.ase import TeacherCalculator
# calc = TeacherCalculator(model="teacher-v2")

# After installing our package (only change needed):
from mlff_distiller.calculators import DistilledCalculator
calc = DistilledCalculator(model="teacher-v2-distilled")

# Rest of MD script runs identically, but 5-10x faster!
atoms.calc = calc
dyn = VelocityVerlet(atoms, timestep=1.0*units.fs)
dyn.run(1000000)  # Millions of calls - must be fast!
```

## Core Architectural Requirements

### 1. ASE Calculator Interface (CRITICAL)

**Requirement**: All models must implement the ASE Calculator interface.

**Why ASE**: ASE (Atomic Simulation Environment) is the standard Python interface for atomistic simulations. Most Python MD codes use ASE Calculators.

**Implementation**:
```python
from ase.calculators.calculator import Calculator, all_changes

class DistilledCalculator(Calculator):
    # Must implement these
    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, model, checkpoint_path=None, device='cuda', **kwargs):
        """
        Initialize calculator.

        Args:
            model: Model name or identifier
            checkpoint_path: Path to model checkpoint
            device: 'cpu' or 'cuda'
            **kwargs: Additional ASE Calculator arguments
        """
        Calculator.__init__(self, **kwargs)
        # Load model
        self.model = load_model(model, checkpoint_path, device)
        self.device = device

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        """
        Main calculation method called by ASE.

        This method is called repeatedly (millions of times) in MD.
        MUST be optimized for minimal overhead.

        Args:
            atoms: ASE Atoms object
            properties: List of properties to calculate
            system_changes: What changed since last call
        """
        Calculator.calculate(self, atoms, properties, system_changes)

        # Extract data from atoms
        positions = atoms.positions  # (n_atoms, 3)
        numbers = atoms.numbers      # (n_atoms,)
        cell = atoms.cell.array if atoms.pbc.any() else None

        # Run model inference
        results = self._predict(positions, numbers, cell)

        # Populate self.results dict (ASE requirement)
        self.results = {
            'energy': results['energy'],    # float, eV
            'forces': results['forces'],    # (n_atoms, 3), eV/Angstrom
            'stress': results['stress'],    # (6,) or (3,3), eV/Angstrom^3
        }

    def _predict(self, positions, numbers, cell):
        """Internal prediction method - optimize this!"""
        # Convert to tensors
        # Run model
        # Convert back
        # Return dict
        pass
```

**Key Points**:
- Inherit from `ase.calculators.calculator.Calculator`
- Implement `calculate()` method
- Populate `self.results` dict with energy, forces, stress
- Handle ASE Atoms objects as input
- Use ASE units (eV, eV/Angstrom, eV/Angstrom^3)
- Implement `implemented_properties` class attribute

### 2. Interface Parity (CRITICAL)

**Requirement**: Student calculators must have identical interfaces to teacher calculators.

**What This Means**:
1. Same initialization parameters
2. Same method signatures
3. Same return types and shapes
4. Same units
5. Same behavior for edge cases
6. Same error messages

**Example**:
```python
# Teacher calculator API
teacher = TeacherCalculator(
    model="model-v2",
    checkpoint_path="path/to/checkpoint.pth",
    device="cuda",
    dtype=torch.float32
)

# Student calculator API (MUST BE IDENTICAL)
student = DistilledCalculator(
    model="model-v2-distilled",
    checkpoint_path="path/to/student_checkpoint.pth",
    device="cuda",
    dtype=torch.float32  # Same parameters
)

# Both work identically in MD
for calc in [teacher, student]:
    atoms.calc = calc
    energy = atoms.get_potential_energy()  # Same method
    forces = atoms.get_forces()            # Same method
    stress = atoms.get_stress()            # Same method
```

### 3. Input/Output Compatibility (CRITICAL)

**Requirement**: Accept same inputs, produce same outputs (within accuracy tolerance).

**Input Compatibility**:
- Accept ASE Atoms objects (not raw tensors)
- Handle same atomic species as teacher
- Support same system size range (10-1000 atoms)
- Handle periodic boundary conditions identically
- Support both PBC and non-PBC systems

**Output Compatibility**:
- Energy: scalar, eV, within 0.05 eV/atom of teacher
- Forces: (n_atoms, 3), eV/Angstrom, within 0.1 eV/Angstrom MAE
- Stress: (3, 3) or (6,), eV/Angstrom^3, within 0.1 GPa
- Units must match exactly
- Shapes must match exactly

### 4. Performance Optimization for MD (CRITICAL)

**The Challenge**: Models are called millions of times in MD simulations.

**Optimization Priorities**:
1. **Latency over Throughput**: Minimize single-call time, not just batch throughput
2. **Low Per-Call Overhead**: Minimize allocations, transfers, overhead
3. **Memory Stability**: No memory leaks over millions of calls
4. **Device Management**: Efficient GPU usage without unnecessary transfers

**Anti-Patterns to Avoid**:
```python
# BAD: Allocating new tensors every call
def calculate(self, atoms, ...):
    positions = torch.tensor(atoms.positions)  # New allocation each call!
    # ... more allocations ...

# GOOD: Reuse buffers when possible
def __init__(self, ...):
    self._position_buffer = None

def calculate(self, atoms, ...):
    n_atoms = len(atoms)
    if self._position_buffer is None or len(self._position_buffer) != n_atoms:
        self._position_buffer = torch.empty((n_atoms, 3), device=self.device)
    self._position_buffer.copy_(torch.from_numpy(atoms.positions))
```

**Performance Requirements**:
- Single inference: 5-10x faster than teacher
- Memory stable over 1M calls (no leaks)
- Per-call overhead < 10% of inference time
- Works efficiently with common MD timesteps (0.5-2 fs)

## Implementation Checklist

### For ML Architecture Designer

When implementing Calculator interfaces:
- [ ] Inherit from `ase.calculators.calculator.Calculator`
- [ ] Implement `calculate()` method correctly
- [ ] Set `implemented_properties` class attribute
- [ ] Handle ASE Atoms objects as input
- [ ] Populate `self.results` dict with correct keys
- [ ] Use correct ASE units everywhere
- [ ] Match teacher calculator interface exactly
- [ ] Optimize for repeated calls (minimize overhead)
- [ ] Test with ASE MD integrators (VelocityVerlet, Langevin)
- [ ] Verify no memory leaks (test with 10000+ calls)

### For Testing & Benchmark Engineer

When testing drop-in compatibility:
- [ ] Test all Calculator methods work
- [ ] Test interface matches teacher exactly
- [ ] Test in actual MD simulations (1000+ steps)
- [ ] Test with multiple MD integrators (NVE, NVT, NPT)
- [ ] Test energy conservation in NVE
- [ ] Test memory stability over long runs
- [ ] Test performance on MD trajectories (not just single inference)
- [ ] Validate outputs match teacher within tolerance
- [ ] Test drop-in replacement (literally swap calculators in existing scripts)

### For CUDA Optimization Engineer

When optimizing for MD performance:
- [ ] Profile repeated calls, not just single inference
- [ ] Minimize per-call overhead (allocations, transfers)
- [ ] Optimize for typical MD system sizes (100-500 atoms)
- [ ] Ensure memory stability over millions of calls
- [ ] Test on realistic MD trajectories (10000+ steps)
- [ ] Measure latency, not just throughput
- [ ] Benchmark with MD integrators, not just raw inference
- [ ] Profile memory usage over time (detect leaks)

## Validation Requirements

### Unit Tests
```python
def test_calculator_interface():
    """Test Calculator implements required interface"""
    calc = DistilledCalculator(model="test", device="cpu")
    assert isinstance(calc, Calculator)
    assert 'energy' in calc.implemented_properties
    assert 'forces' in calc.implemented_properties
    assert hasattr(calc, 'calculate')

def test_calculator_output_types():
    """Test Calculator returns correct types"""
    calc = DistilledCalculator(model="test", device="cpu")
    atoms = create_test_atoms()
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    assert isinstance(energy, float)

    forces = atoms.get_forces()
    assert forces.shape == (len(atoms), 3)
```

### Integration Tests
```python
def test_md_simulation():
    """Test Calculator works in MD simulation"""
    from ase.md.verlet import VelocityVerlet

    atoms = create_test_atoms()
    calc = DistilledCalculator(model="test", device="cuda")
    atoms.calc = calc

    dyn = VelocityVerlet(atoms, timestep=1.0*units.fs)
    dyn.run(1000)  # Should complete without errors

    # Check energy conservation (NVE)
    energies = [atoms.get_potential_energy() for _ in range(100)]
    energy_drift = abs(energies[-1] - energies[0]) / abs(energies[0])
    assert energy_drift < 0.01  # < 1% drift

def test_drop_in_replacement():
    """Test student can replace teacher in MD script"""
    atoms = create_test_atoms()

    # Run with teacher
    teacher_calc = TeacherCalculator(model="test", device="cuda")
    atoms.calc = teacher_calc
    teacher_trajectory = run_md(atoms, steps=100)

    # Run with student (drop-in replacement)
    student_calc = DistilledCalculator(model="test-distilled", device="cuda")
    atoms.calc = student_calc  # Only change
    student_trajectory = run_md(atoms, steps=100)

    # Trajectories should be similar
    compare_trajectories(teacher_trajectory, student_trajectory, tolerance=0.1)
```

### Performance Tests
```python
def test_md_performance():
    """Test performance on MD trajectory"""
    import time

    atoms = create_test_atoms()

    # Benchmark teacher
    teacher_calc = TeacherCalculator(model="test", device="cuda")
    atoms.calc = teacher_calc
    start = time.time()
    run_md(atoms, steps=1000)
    teacher_time = time.time() - start

    # Benchmark student
    student_calc = DistilledCalculator(model="test-distilled", device="cuda")
    atoms.calc = student_calc
    start = time.time()
    run_md(atoms, steps=1000)
    student_time = time.time() - start

    # Verify speedup
    speedup = teacher_time / student_time
    assert speedup >= 5.0, f"Speedup {speedup:.1f}x < 5x target"
    print(f"Achieved {speedup:.1f}x speedup on MD trajectory")
```

## Common Pitfalls and Solutions

### Pitfall 1: Testing Only Single Inference
**Problem**: Single inference benchmarks don't capture MD performance.
**Solution**: Always benchmark full MD trajectories (1000+ steps).

### Pitfall 2: Wrong Units
**Problem**: Using wrong units causes incorrect results.
**Solution**: ASE uses eV, eV/Angstrom, eV/Angstrom^3. Verify units match teacher.

### Pitfall 3: Memory Leaks
**Problem**: Small leaks accumulate over millions of MD steps.
**Solution**: Test with 10000+ repeated calls, monitor memory with tracemalloc.

### Pitfall 4: Different Interface
**Problem**: Different API breaks drop-in compatibility.
**Solution**: Match teacher calculator interface exactly (same parameters, methods, behavior).

### Pitfall 5: Optimizing Wrong Thing
**Problem**: Optimizing batch throughput doesn't help MD (which is sequential).
**Solution**: Optimize single-call latency and per-call overhead.

## Documentation Requirements

All calculators must include:
1. **Docstrings**: Complete API documentation
2. **Usage Examples**: How to use as drop-in replacement
3. **MD Examples**: Working MD simulation scripts
4. **Performance Characteristics**: Expected speedup, memory usage
5. **Limitations**: System size limits, supported atom types

## LAMMPS Integration (Future)

For production MD at scale, LAMMPS integration is required:
- Implement LAMMPS pair_style interface
- Support LAMMPS data formats
- Enable GPU acceleration in LAMMPS context
- Maintain same performance targets

This will be addressed in M6 (Issue #28).

## Success Criteria

A drop-in replacement implementation is successful when:
1. ✅ User can replace teacher with student by changing 1 line
2. ✅ MD scripts run without modifications
3. ✅ Results are within accuracy tolerance (>95% agreement)
4. ✅ Performance is 5-10x faster on MD trajectories
5. ✅ Memory is stable over millions of calls
6. ✅ Works with all common MD protocols (NVE, NVT, NPT)
7. ✅ Interface is identical to teacher calculator
8. ✅ All tests pass (unit, integration, performance)

## References

- ASE Calculator Tutorial: https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html
- ASE MD: https://wiki.fysik.dtu.dk/ase/ase/md.html
- ASE Atoms: https://wiki.fysik.dtu.dk/ase/ase/atoms.html

## Contact

For questions about drop-in compatibility requirements:
- Tag `@coordinator` in issues
- Reference this guide in discussions
- Create RFC issues for architectural decisions

---

**Remember**: Drop-in compatibility is a CRITICAL project requirement. When in doubt, prioritize compatibility over other concerns.
