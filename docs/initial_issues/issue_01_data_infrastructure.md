# [Data Pipeline] [M1] Set up data loading infrastructure

## Assigned Agent
data-pipeline-engineer

## Milestone
M1: Setup & Baseline

## Task Description
Create the foundational data loading infrastructure for the MLFF Distiller project. This includes basic data loading utilities, interfaces for handling atomic structures, and the foundational components that will be used throughout the data pipeline.

## Context & Background
This is a foundational task for the entire project. All subsequent data processing, generation, and training work depends on having a robust data loading infrastructure. We need to support:
- Loading atomic structures from various formats (ASE, XYZ, etc.)
- Efficient batching of variable-sized molecular systems
- Integration with PyTorch DataLoader
- Support for periodic boundary conditions

## Acceptance Criteria
- [ ] Create `src/data/loader.py` with base DataLoader class
- [ ] Implement support for ASE Atoms objects
- [ ] Handle variable-sized systems (10-1000 atoms)
- [ ] Support periodic boundary conditions
- [ ] Implement collate function for batching
- [ ] Add docstrings and type hints
- [ ] Unit tests with >80% coverage
- [ ] Example usage in `examples/data_loading_example.py`

## Technical Notes

### Suggested API Design
```python
from mlff_distiller.data import MolecularDataLoader

# Simple usage
loader = MolecularDataLoader(
    data_path="path/to/structures",
    batch_size=32,
    shuffle=True,
    num_workers=4
)

for batch in loader:
    positions = batch["positions"]  # (batch_size, max_atoms, 3)
    species = batch["species"]      # (batch_size, max_atoms)
    cells = batch["cells"]          # (batch_size, 3, 3)
    mask = batch["mask"]            # (batch_size, max_atoms)
```

### Key Considerations
1. **Variable Sizes**: Different molecules have different numbers of atoms. Implement padding with masks.
2. **Memory Efficiency**: Large systems should not cause OOM errors
3. **GPU Transfer**: Support efficient CPU-to-GPU transfer
4. **Periodic Boundaries**: Store cell parameters and handle wrapping

### Dependencies
- ASE (Atomic Simulation Environment)
- PyTorch
- NumPy

## Related Issues
- Depends on: None (foundational task)
- Related to: #2 (data classes), #3 (validation)
- Enables: #5 (data generation), #24 (data tests)

## Estimated Complexity
Medium (3-5 days)

## Definition of Done
- [ ] Code implemented and follows style guide
- [ ] All acceptance criteria met
- [ ] Tests written and passing (>80% coverage)
- [ ] Documentation complete with examples
- [ ] PR created and reviewed
- [ ] PR merged to main

## Resources
- ASE documentation: https://wiki.fysik.dtu.dk/ase/
- PyTorch DataLoader: https://pytorch.org/docs/stable/data.html
- Similar projects: SchNetPack, NequIP data loaders
