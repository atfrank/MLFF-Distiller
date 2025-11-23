# [Testing] [M1] Implement ASE Calculator interface tests

## Assigned Agent
testing-benchmark-engineer

## Milestone
M1: Setup & Baseline

## Priority
CRITICAL - Validates drop-in replacement foundation

## Task Description
Create comprehensive test suite to validate that our Calculator implementations (both teacher and student) correctly implement the ASE Calculator interface. These tests ensure drop-in replacement capability by verifying interface compliance.

## Context & Background
The ASE Calculator interface has specific requirements and conventions. If we don't implement them correctly, our calculators won't work as drop-in replacements in user MD scripts. This test suite validates:

1. All required methods are implemented
2. Methods return correct types and shapes
3. Units are correct (eV, eV/Angstrom, eV/Angstrom^3)
4. Behavior matches ASE conventions
5. Works with various ASE MD integrators

## Acceptance Criteria
- [ ] Test suite in `tests/integration/test_ase_calculator_interface.py`
- [ ] Test all required Calculator methods:
  - [ ] `get_potential_energy()`
  - [ ] `get_forces()`
  - [ ] `get_stress()`
  - [ ] `calculate()`
- [ ] Test `implemented_properties` attribute
- [ ] Test handling of ASE Atoms objects
- [ ] Test periodic boundary conditions
- [ ] Test variable system sizes (10, 50, 100, 500 atoms)
- [ ] Test CPU and CUDA device handling
- [ ] Test units are correct
- [ ] Test output shapes and types
- [ ] Test with multiple ASE MD integrators:
  - [ ] VelocityVerlet
  - [ ] Langevin
  - [ ] VelocityVerlet in NPT
- [ ] Test memory stability (1000+ repeated calls)
- [ ] Test error handling for invalid inputs
- [ ] Comparison tests: calculator outputs vs direct model outputs
- [ ] Documentation of test coverage

## Technical Notes

### Example Test Structure
```python
import pytest
from ase import Atoms
from ase.md.verlet import VelocityVerlet
from ase import units
from mlff_distiller.calculators import OrbTeacherCalculator, DistilledOrbCalculator

class TestASECalculatorInterface:
    @pytest.fixture
    def atoms(self):
        # Create test system
        return Atoms('H2O', positions=[[0,0,0],[1,0,0],[0,1,0]],
                     cell=[10,10,10], pbc=True)

    @pytest.fixture(params=[OrbTeacherCalculator, DistilledOrbCalculator])
    def calculator(self, request):
        # Test both teacher and student calculators
        return request.param(model="test-model", device="cpu")

    def test_get_potential_energy(self, atoms, calculator):
        """Test get_potential_energy returns float in eV"""
        atoms.calc = calculator
        energy = atoms.get_potential_energy()
        assert isinstance(energy, float)
        # Energy should be reasonable magnitude for this system
        assert -100 < energy < 100  # eV

    def test_get_forces(self, atoms, calculator):
        """Test get_forces returns correct shape and units"""
        atoms.calc = calculator
        forces = atoms.get_forces()
        assert forces.shape == (len(atoms), 3)
        # Forces should be in eV/Angstrom
        assert forces.max() < 100  # reasonable magnitude

    def test_md_integration(self, atoms, calculator):
        """Test calculator works in ASE MD simulation"""
        atoms.calc = calculator
        dyn = VelocityVerlet(atoms, timestep=1.0*units.fs)
        dyn.run(100)  # Should complete without errors

    def test_memory_stability(self, atoms, calculator):
        """Test no memory leaks during repeated calls"""
        import tracemalloc
        atoms.calc = calculator
        tracemalloc.start()

        for _ in range(1000):
            atoms.get_potential_energy()
            atoms.get_forces()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        # Memory should be stable (not growing linearly)
        assert peak < current * 2  # Rough check
```

### Required Tests
1. **Interface Compliance**:
   - All methods exist
   - Methods have correct signatures
   - Return types are correct

2. **Correctness**:
   - Energy, forces, stress values are reasonable
   - Units are correct (ASE conventions)
   - PBC handled correctly
   - Variable system sizes work

3. **Integration**:
   - Works with ASE MD integrators
   - Works in NVE, NVT, NPT
   - Energy conservation in NVE

4. **Performance**:
   - No memory leaks
   - Reasonable performance
   - Stable over many calls

5. **Robustness**:
   - Error handling for invalid inputs
   - Device management (CPU/CUDA)
   - Different atom types

## Related Issues
- Depends on: #6 (teacher calculator implementation)
- Related to: #26 (student calculator implementation)
- Enables: #30 (drop-in replacement validation)
- Related to: #23 (baseline MD benchmarks)

## Dependencies
- pytest
- ase
- torch
- numpy
- Teacher and student calculator implementations

## Estimated Complexity
Medium (2-3 days)

## Definition of Done
- [ ] All acceptance criteria met
- [ ] Test suite written and passing
- [ ] Tests cover all Calculator methods
- [ ] Tests work with both teacher and student calculators
- [ ] Tests verify MD integration
- [ ] Tests check memory stability
- [ ] Documentation of tests
- [ ] CI integration
- [ ] PR created and reviewed
- [ ] PR merged to main

## Success Metrics
- All interface tests pass
- Test coverage > 90% of calculator code
- Tests run in < 2 minutes
- No false positives or negatives
- Clear error messages when tests fail
