# [Architecture] [M1] Create teacher model wrapper with ASE Calculator interface

## Assigned Agent
ml-architecture-designer

## Milestone
M1: Setup & Baseline

## Priority
CRITICAL - Required for drop-in replacement capability

## Task Description
Develop wrapper interfaces for teacher models (Orb-models and FeNNol-PMC) that implement the ASE Calculator interface. These wrappers must be drop-in replacements compatible with existing ASE MD simulation scripts. They will be used for data generation, baseline MD benchmarking, and as reference outputs during distillation training.

**CRITICAL REQUIREMENT**: These wrappers must work seamlessly in existing ASE MD scripts without any modifications to user code.

## Context & Background
These distilled models will be used in molecular dynamics simulations where they are called millions of times. We need wrappers that:
1. Implement the ASE Calculator interface (the standard in Python MD)
2. Work as drop-in replacements in existing MD scripts (users should only need to change the calculator import)
3. Enable baseline MD trajectory benchmarking (not just single inference)
4. Provide a template for student model interfaces
5. Support all ASE Calculator methods: get_potential_energy(), get_forces(), get_stress()

**Use Case**: A user running MD with Orb-models should be able to replace:
```python
from orb_models.ase import OrbCalculator
calc = OrbCalculator(model="orb-v2")
```
with:
```python
from mlff_distiller.calculators import OrbTeacherCalculator
calc = OrbTeacherCalculator(model="orb-v2")
```
And their MD script should run identically.

## Acceptance Criteria
- [ ] Create `src/calculators/teacher_calculator.py` with ASE Calculator base class
- [ ] Implement `OrbTeacherCalculator` inheriting from ase.calculators.calculator.Calculator
- [ ] Implement `FeNNolTeacherCalculator` inheriting from ase.calculators.calculator.Calculator
- [ ] Support all ASE Calculator methods:
  - [ ] `get_potential_energy(atoms)` - returns energy in eV
  - [ ] `get_forces(atoms)` - returns forces in eV/Angstrom
  - [ ] `get_stress(atoms)` - returns stress tensor in eV/Angstrom^3
  - [ ] `calculate(atoms, properties, system_changes)` - main calculation method
- [ ] Support loading from checkpoints/pretrained weights
- [ ] Handle ASE Atoms objects as input (not raw tensors)
- [ ] Handle variable system sizes (10-1000 atoms)
- [ ] Handle periodic boundary conditions correctly
- [ ] Support both CPU and CUDA devices
- [ ] Implement `implemented_properties` attribute
- [ ] Add comprehensive docstrings and type hints following ASE conventions
- [ ] Unit tests for each Calculator method
- [ ] Integration test running short MD trajectory (100 steps NVE)
- [ ] Validation test comparing Calculator outputs to original model outputs
- [ ] Example script showing drop-in replacement usage

## Technical Notes

### Required ASE Calculator API

```python
from ase import Atoms
from ase.md.verlet import VelocityVerlet
from ase import units
from mlff_distiller.calculators import OrbTeacherCalculator

# Create atoms object (standard ASE)
atoms = Atoms('H2O',
              positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
              cell=[10, 10, 10],
              pbc=True)

# Use as ASE Calculator (drop-in replacement)
calc = OrbTeacherCalculator(
    model_name="orb-v2",
    checkpoint_path="path/to/checkpoint.pth",
    device="cuda"
)
atoms.calc = calc

# Standard ASE Calculator interface
energy = atoms.get_potential_energy()  # eV
forces = atoms.get_forces()            # eV/Angstrom
stress = atoms.get_stress()            # eV/Angstrom^3

# Use in MD simulation (standard ASE, no modifications)
dyn = VelocityVerlet(atoms, timestep=1.0*units.fs)
dyn.run(1000)  # Run 1000 MD steps
```

### Implementation Requirements

Must inherit from `ase.calculators.calculator.Calculator` and implement:
```python
class OrbTeacherCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, model_name, checkpoint_path=None, device='cuda', **kwargs):
        super().__init__(**kwargs)
        # Load Orb model

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        # Main calculation method called by ASE
        # Must populate self.results dict with 'energy', 'forces', 'stress'
        pass
```

### Teacher Model Information

**Orb-models**:
- Repository: https://github.com/orbital-materials/orb-models
- Models: orb-v1, orb-v2
- Input: atomic positions, atomic numbers, cell (optional)
- Output: energy, forces, stress

**FeNNol-PMC**:
- Paper: FeNNol force fields (need to verify latest implementation)
- Input format: varies by version
- Output: energy, forces

### Key Considerations
1. **ASE Calculator Interface**: Must follow ASE conventions exactly
   - Use ASE Atoms objects as input
   - Store results in `self.results` dict
   - Implement `calculate()` method correctly
   - Handle `system_changes` parameter
2. **Model Loading**: Handle different checkpoint formats
3. **Device Management**: Support CPU and CUDA, handle device transfers
4. **Units**: ASE uses eV, eV/Angstrom, eV/Angstrom^3 - ensure correct units
5. **Periodic Boundaries**: Handle PBC correctly via ASE Atoms.cell
6. **Performance**: Optimize for repeated calls (millions in MD)
7. **Memory**: Minimize per-call overhead, avoid memory leaks
8. **Error Handling**: Graceful failures for invalid inputs
9. **Drop-in Compatibility**: User should only change import statement

## Related Issues
- Related to: #7 (Orb analysis), #8 (FeNNol analysis)
- Enables: #5 (data generation), #18 (MD trajectory profiling), #23 (baseline MD benchmarks)
- Enables: #26 (ASE Calculator for student models), #29 (ASE interface tests)
- Blocks: #30 (drop-in replacement validation)

## Dependencies
- torch
- ase (critical - must understand ASE Calculator interface)
- orb-models package
- fennol package (if available)

## Required Knowledge
- ASE Calculator interface: https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html
- ASE Atoms objects: https://wiki.fysik.dtu.dk/ase/ase/atoms.html
- How ASE MD engines call calculators
- Orb-models API and output formats

## Estimated Complexity
High (5-7 days)

### Challenges
- Different model APIs may require significant adaptation
- Installing and configuring teacher models correctly
- Ensuring output consistency across different model versions

## Definition of Done
- [ ] Code implemented and follows style guide
- [ ] All acceptance criteria met
- [ ] ASE Calculator interface correctly implemented
- [ ] Tests written and passing (unit + integration + MD trajectory)
- [ ] Documentation with MD usage examples
- [ ] Verified Calculator outputs match original model outputs
- [ ] Verified Calculator works in ASE MD simulations (NVE 100 steps)
- [ ] Example script demonstrating drop-in replacement
- [ ] No memory leaks during repeated calls (tested with 1000+ calls)
- [ ] PR created and reviewed
- [ ] PR merged to main

## Success Validation
Run this test to verify drop-in compatibility:
```python
# Test: Replace original calculator with teacher wrapper in ASE MD
from ase.md.verlet import VelocityVerlet
from mlff_distiller.calculators import OrbTeacherCalculator

atoms = ... # create test system
atoms.calc = OrbTeacherCalculator(model="orb-v2", device="cuda")
dyn = VelocityVerlet(atoms, timestep=1.0*units.fs)
dyn.run(1000)  # Should run without errors
# Verify energy conservation, forces correctness
```

## Resources
- Orb-models: https://github.com/orbital-materials/orb-models
- FeNNol: (need to add specific links)
- Example wrappers: SchNetPack calculators, NequIP wrappers

## Blockers / Questions
- [ ] Verify FeNNol-PMC model availability and API
- [ ] Confirm required versions for both teacher models
- [ ] Check if pretrained weights are publicly available
