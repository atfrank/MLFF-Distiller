# [Architecture] [M1] Implement ASE Calculator interface for student models

## Assigned Agent
ml-architecture-designer

## Milestone
M1: Setup & Baseline

## Priority
CRITICAL - Core drop-in replacement requirement

## Task Description
Create ASE Calculator wrapper for student models that provides identical interface to teacher model calculators. This is the foundation for drop-in replacement capability - users should be able to swap teacher calculators for student calculators with zero code changes to their MD scripts.

**CRITICAL REQUIREMENT**: Student models must work identically to teacher models in ASE MD simulations.

## Context & Background
Once we have student models trained, they must be usable as drop-in replacements in existing MD workflows. The ASE Calculator interface is the standard way Python MD codes interact with force fields. By implementing this interface correctly, users can:

1. Replace teacher with student by changing one line (the calculator import/initialization)
2. Run all their existing MD scripts without modifications
3. Use student models with any ASE-compatible MD code
4. Benefit from 5-10x speedup without workflow changes

**Use Case**:
```python
# User's existing MD script with teacher model:
# from mlff_distiller.calculators import OrbTeacherCalculator
# calc = OrbTeacherCalculator(model="orb-v2")

# After drop-in replacement (only change needed):
from mlff_distiller.calculators import DistilledOrbCalculator
calc = DistilledOrbCalculator(model="orb-v2-distilled")

# Rest of MD script runs identically
atoms.calc = calc
dyn = VelocityVerlet(atoms, timestep=1.0*units.fs)
dyn.run(1000000)  # Now 5-10x faster!
```

## Acceptance Criteria
- [ ] Create `src/calculators/student_calculator.py` with ASE Calculator base class
- [ ] Implement `DistilledOrbCalculator` inheriting from ase.calculators.calculator.Calculator
- [ ] Implement `DistilledFeNNolCalculator` inheriting from ase.calculators.calculator.Calculator
- [ ] Support all ASE Calculator methods:
  - [ ] `get_potential_energy(atoms)` - returns energy in eV
  - [ ] `get_forces(atoms)` - returns forces in eV/Angstrom
  - [ ] `get_stress(atoms)` - returns stress tensor in eV/Angstrom^3
  - [ ] `calculate(atoms, properties, system_changes)` - main calculation method
- [ ] Identical interface to teacher calculators (same initialization parameters, same behavior)
- [ ] Support loading from student model checkpoints
- [ ] Handle ASE Atoms objects as input
- [ ] Handle variable system sizes (10-1000 atoms)
- [ ] Handle periodic boundary conditions correctly
- [ ] Support both CPU and CUDA devices
- [ ] Implement `implemented_properties` attribute
- [ ] Optimize for repeated inference (minimize per-call overhead)
- [ ] Add comprehensive docstrings following ASE conventions
- [ ] Unit tests for each Calculator method
- [ ] Integration test running MD trajectory (1000 steps NVE)
- [ ] Drop-in replacement validation test (use in place of teacher)
- [ ] Example script showing drop-in usage

## Technical Notes

### Interface Design (identical to teacher)
```python
from ase import Atoms
from ase.md.langevin import Langevin
from ase import units
from mlff_distiller.calculators import DistilledOrbCalculator

# Same API as teacher calculator
calc = DistilledOrbCalculator(
    model="orb-v2-distilled",
    checkpoint_path="path/to/student_checkpoint.pth",
    device="cuda"
)

# Works identically in MD
atoms = Atoms(...)
atoms.calc = calc
dyn = Langevin(atoms, timestep=1.0*units.fs, temperature_K=300, friction=0.01)
dyn.run(10000)  # 5-10x faster than teacher!
```

### Implementation Requirements
```python
class DistilledOrbCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, model, checkpoint_path=None, device='cuda', **kwargs):
        super().__init__(**kwargs)
        # Load student model
        # Must be fast to initialize

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        # Optimized for repeated calls
        # Minimal overhead per call
        # Populate self.results dict
        pass
```

### Key Considerations
1. **Interface Parity**: Must match teacher calculator interface exactly
2. **Performance**: Optimize for repeated calls (millions in MD)
   - Minimize per-call overhead
   - Efficient device transfers
   - No unnecessary allocations
3. **Memory**: Stable memory usage over long MD runs
4. **Compatibility**: Work with all ASE MD integrators
5. **Units**: Correct ASE units (eV, eV/Angstrom, eV/Angstrom^3)
6. **PBC**: Handle periodic boundaries correctly
7. **Drop-in**: User should only change calculator initialization line

## Related Issues
- Depends on: #6 (teacher calculator interface - must match it)
- Depends on: #9 (student model architecture)
- Enables: #27 (drop-in validation), #30 (drop-in tests)
- Related to: #29 (ASE interface tests)

## Dependencies
- torch
- ase (critical - must understand ASE Calculator interface)
- Student model implementation from #9

## Required Knowledge
- ASE Calculator interface: https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html
- How ASE MD engines call calculators
- Performance optimization for repeated inference
- Memory management for long-running processes

## Estimated Complexity
Medium (3-5 days)

### Challenges
- Ensuring exact interface parity with teacher calculators
- Optimizing for repeated inference without sacrificing first-call performance
- Testing with various MD integrators and conditions

## Definition of Done
- [ ] Code implemented and follows style guide
- [ ] All acceptance criteria met
- [ ] ASE Calculator interface correctly implemented
- [ ] Interface identical to teacher calculator
- [ ] Tests written and passing (unit + integration + MD trajectory)
- [ ] Documentation with drop-in replacement examples
- [ ] Verified works in ASE MD simulations (1000+ steps)
- [ ] Performance optimized for repeated calls
- [ ] No memory leaks during long MD runs (tested with 10000+ calls)
- [ ] Drop-in replacement test passes (swap for teacher calculator)
- [ ] PR created and reviewed
- [ ] PR merged to main

## Success Validation
```python
# Test: Drop-in replacement works identically
from ase.md.verlet import VelocityVerlet
from mlff_distiller.calculators import DistilledOrbCalculator

atoms = ... # test system
atoms.calc = DistilledOrbCalculator(model="orb-v2-distilled", device="cuda")
dyn = VelocityVerlet(atoms, timestep=1.0*units.fs)

# Should run without errors and be 5-10x faster than teacher
import time
start = time.time()
dyn.run(1000)
elapsed = time.time() - start
print(f"MD time: {elapsed:.2f}s (target: 5-10x faster than teacher)")
```

## Resources
- ASE Calculator documentation
- Teacher calculator implementation (#6)
- Example Calculator implementations in ASE
