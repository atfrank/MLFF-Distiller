# [Architecture] [M1] Create teacher model wrapper interfaces

## Assigned Agent
ml-architecture-designer

## Milestone
M1: Setup & Baseline

## Task Description
Develop wrapper interfaces for teacher models (Orb-models and FeNNol-PMC) that provide a consistent API for inference. These wrappers will be used for data generation, baseline benchmarking, and as reference outputs during distillation training.

## Context & Background
We need to interact with two different force field models that may have different APIs, input formats, and output structures. Creating unified wrapper interfaces will:
1. Simplify data generation code
2. Enable easy switching between teacher models
3. Provide a template for student model interfaces
4. Facilitate baseline benchmarking

## Acceptance Criteria
- [ ] Create `src/models/teacher_wrapper.py` with abstract base class
- [ ] Implement `OrbModelWrapper` for Orb-models
- [ ] Implement `FeNNolWrapper` for FeNNol-PMC
- [ ] Support loading from checkpoints/pretrained weights
- [ ] Consistent output format: energies, forces, stresses
- [ ] Handle variable system sizes (10-1000 atoms)
- [ ] Support batch inference
- [ ] Add comprehensive docstrings and type hints
- [ ] Unit tests for each wrapper
- [ ] Integration test comparing wrapper outputs to original models

## Technical Notes

### Suggested API Design
```python
from mlff_distiller.models import load_teacher_model

# Load teacher model
teacher = load_teacher_model(
    model_name="orb-v2",
    checkpoint_path="path/to/checkpoint.pth",
    device="cuda"
)

# Inference
results = teacher.predict(
    positions=positions,    # (n_atoms, 3)
    species=species,        # (n_atoms,)
    cell=cell              # (3, 3) or None
)

# Access outputs
energy = results["energy"]          # scalar
forces = results["forces"]          # (n_atoms, 3)
stress = results["stress"]          # (3, 3) or None
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
1. **Model Loading**: Handle different checkpoint formats
2. **Device Management**: Support CPU and CUDA
3. **Error Handling**: Graceful failures for invalid inputs
4. **Performance**: Batch inference where possible
5. **Reproducibility**: Set random seeds if models use dropout

## Related Issues
- Related to: #7 (Orb analysis), #8 (FeNNol analysis)
- Enables: #5 (data generation), #18 (profiling), #23 (baseline benchmarks)

## Dependencies
- torch
- ase
- orb-models package
- fennol package (if available)

## Estimated Complexity
High (5-7 days)

### Challenges
- Different model APIs may require significant adaptation
- Installing and configuring teacher models correctly
- Ensuring output consistency across different model versions

## Definition of Done
- [ ] Code implemented and follows style guide
- [ ] All acceptance criteria met
- [ ] Tests written and passing
- [ ] Documentation with usage examples
- [ ] Verified wrapper outputs match original model outputs
- [ ] PR created and reviewed
- [ ] PR merged to main

## Resources
- Orb-models: https://github.com/orbital-materials/orb-models
- FeNNol: (need to add specific links)
- Example wrappers: SchNetPack calculators, NequIP wrappers

## Blockers / Questions
- [ ] Verify FeNNol-PMC model availability and API
- [ ] Confirm required versions for both teacher models
- [ ] Check if pretrained weights are publicly available
