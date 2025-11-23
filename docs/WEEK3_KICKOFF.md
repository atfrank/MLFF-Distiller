# Week 3 Kickoff: M2 Data Pipeline Launch

**Date**: November 24, 2025 (Week 3, Day 1)
**Milestone**: M2 - Data Pipeline
**Status**: M1 Complete (13 days ahead!), M2 Ready to Launch

---

## M2 Quick Facts

- **Duration**: 4 weeks (Nov 24 - Dec 21, 2025)
- **Target Completion**: Dec 14, 2025 (7-day buffer)
- **Issues**: 8 issues (#10-#17)
- **Goal**: Generate 120K high-quality training samples
- **Lead Agent**: Data Pipeline Engineer

---

## Week 3 Objectives (Nov 24-30)

**Primary Goal**: Establish data generation infrastructure

**Key Deliverables**:
1. Sampling strategy designed and approved (#10)
2. Structure generation working (#11)
3. Teacher models installed and tested (#12)
4. 1000 test samples generated and validated (Checkpoint 1)

---

## Agent Assignments - Week 3

### Data Pipeline Engineer (Primary Lead)
**Issues**: #10 (Critical), #11 (Critical)

**Day 1-3 (Nov 24-26)**:
- [ ] Start #10: Design molecular structure sampling strategy
  - Review M1 infrastructure (teacher_wrappers.py, MolecularDataset)
  - Research chemical diversity requirements
  - Draft sampling strategy document
  - Create configuration files (sampling_config.yaml)
  - Get Architecture + Training review/approval

**Day 3-7 (Nov 26-30)**:
- [ ] Start #11: Implement structure generation pipeline
  - Implement MD trajectory sampling (ASE integrators)
  - Implement normal mode perturbations
  - Add parallelization (multiprocessing)
  - Create CLI script (generate_structures.py)
  - Generate 1000 test structures
  - Write unit tests

**Deliverables by Week 3 End**:
- docs/data_generation_strategy.md
- configs/sampling_config.yaml
- src/mlff_distiller/data/generation/structure_sampler.py
- scripts/generate_structures.py
- 1000 test structures (ASE db format)

---

### ML Architecture Designer
**Issues**: #12 (Critical)

**Day 1-3 (Nov 24-26)** - **CRITICAL PATH**:
- [ ] Install teacher models (parallel to #10)
  - Set up conda environment
  - Install orb-models: `pip install orb-models`
  - Install FeNNol + JAX: `pip install fennol jax[cuda12]`
  - Test basic inference with teacher_wrappers.py
  - Document installation process
  - Benchmark inference speed

**Day 3-7 (Nov 26-30)**:
- [ ] Start #12: Create teacher model inference pipeline
  - Design LabelGenerator class
  - Implement batch inference (GPU)
  - Add error handling and retry logic
  - Quality checks (no NaN, force validation)
  - Test with sample structures
  - Label 1000 test structures from #11

**Deliverables by Week 3 End**:
- Teacher models installed and working
- src/mlff_distiller/data/generation/label_generator.py
- scripts/generate_labels.py
- 1000 labeled samples (E/F/S)
- Inference benchmarks

---

### Testing & Benchmark Engineer
**Issues**: #14 (High)

**Day 1-3 (Nov 24-26)**:
- [ ] Review M2 requirements and acceptance criteria
- [ ] Design validation framework architecture (#14)
  - Identify validation checks (energy, forces, geometry)
  - Research outlier detection methods (z-score, IQR)
  - Plan report generation (HTML)
  - Design validator class structure
- [ ] Set up test fixtures for M2 components

**Day 4-7 (Nov 27-30)**:
- [ ] Start #14: Validation framework implementation
  - Implement energy validators
  - Implement force validators
  - Implement geometry validators (atom overlaps, bonds)
  - Create basic test cases
  - Test on 1000 samples from #11/#12

**Deliverables by Week 3 End**:
- Validation framework design document
- src/mlff_distiller/data/validation/quality_checker.py (partial)
- Basic validators working
- Validation report for 1000 test samples

---

### Distillation Training Engineer
**Issues**: #17 (High priority for Week 4)

**Day 1-7 (Nov 24-30)** - **Support Role**:
- [ ] Review #10: Provide data requirements for training
  - Energy/force normalization needs
  - Preferred data formats
  - Train/val/test split requirements
- [ ] Test MolecularDataset with HDF5 format
  - Create mock HDF5 file
  - Test loading with MolecularDataLoader
  - Benchmark data loading performance
- [ ] Plan training integration (#17)
  - Review training config structure
  - Identify normalization strategy
  - Plan dataloader optimization

**Deliverables by Week 3 End**:
- Data requirements documented (comment on #10)
- HDF5 loading tested
- Training integration plan ready

---

### CUDA Optimization Engineer
**Issues**: None (M2 monitoring role)

**Day 1-7 (Nov 24-30)** - **Monitoring**:
- [ ] Monitor M2 progress (passive)
- [ ] Review data generation code for optimization opportunities
- [ ] Prepare profiling tools for bottleneck identification
- [ ] Note any performance issues for M5

**Deliverables by Week 3 End**:
- Performance observations documented
- Bottleneck candidates identified

---

## Week 3 Checkpoint (Nov 30)

**Integration Milestone**: 1000 test samples generated and validated

### Success Criteria:
- [ ] #10 complete: Sampling strategy approved
- [ ] #11 in progress: Structure generation working
- [ ] #12 in progress: Label generation working
- [ ] Teacher models installed (Orb-v2, FeNNol)
- [ ] 1000 structures generated successfully
- [ ] 1000 structures labeled with E/F/S
- [ ] Basic validation passed (no critical errors)

### Checkpoint Review (Nov 30):
All agents sync on:
- Progress on assigned issues
- Blockers identified (if any)
- Timeline adjustments (if needed)
- Plan for Week 4 (Dec 1-7)

---

## Critical Path for Week 3

```
Day 1-3:
  #10 (Sampling Strategy) ──┐
                            ├──> Complete by Nov 26
  Teacher Installation ─────┘

Day 3-7:
  #10 complete ──> #11 (Structure Gen) ──┐
                                         ├──> 1000 samples by Nov 30
  Teachers ready ──> #12 (Label Gen) ────┘
```

**Most Critical**: Teacher installation (Architecture) must complete by Day 3 for #12 to proceed.

---

## Getting Started (Day 1 - Nov 24)

### All Agents:
1. [ ] Review this kickoff document
2. [ ] Review your assigned issue(s) on GitHub
3. [ ] Review M2 Coordination Plan: `/home/aaron/ATX/software/MLFF_Distiller/docs/M2_COORDINATION_PLAN.md`
4. [ ] Create feature branches: `feature/M2-issue-{number}`
5. [ ] Comment on your issue with Day 1 plan
6. [ ] Begin work!

### Data Pipeline Engineer:
- Start #10 immediately (critical path)
- Review teacher_wrappers.py from M1
- Review MolecularDataset from M1

### Architecture Designer:
- Start teacher installation immediately (critical path)
- Set up conda environment
- Test teacher_wrappers.py with mock calculators first

### Testing Engineer:
- Review #14 requirements
- Design validation framework architecture
- Set up test fixtures

### Training Engineer:
- Review training pipeline from M1
- Identify data format requirements
- Plan integration approach

---

## Communication Protocol

### Daily Updates:
Post comment on your issue with:
- What you completed
- What you're working on
- Any blockers

### Asking Questions:
- Comment on relevant issue
- Use @mentions to notify specific agents
- Tag with "question" label if needed

### Reporting Blockers:
- Comment on issue immediately
- Add "blocked" label
- @mention Lead Coordinator
- Suggest mitigation if possible

### Code Reviews:
- Open PR when ready
- Link to issue: "Closes #10"
- Request review from Lead Coordinator
- Address comments within 24 hours

---

## Key Resources

### M1 Infrastructure (Available):
- `/home/aaron/ATX/software/MLFF_Distiller/src/mlff_distiller/models/teacher_wrappers.py`
- `/home/aaron/ATX/software/MLFF_Distiller/src/mlff_distiller/data/dataset.py`
- `/home/aaron/ATX/software/MLFF_Distiller/src/mlff_distiller/data/transforms.py`
- `/home/aaron/ATX/software/MLFF_Distiller/benchmarks/md_benchmark.py`

### Documentation:
- `/home/aaron/ATX/software/MLFF_Distiller/docs/M2_COORDINATION_PLAN.md` - Full M2 plan
- `/home/aaron/ATX/software/MLFF_Distiller/docs/MILESTONES.md` - All milestones
- `/home/aaron/ATX/software/MLFF_Distiller/README.md` - Project overview

### GitHub:
- Issues: https://github.com/atfrank/MLFF-Distiller/issues
- M2 Milestone: https://github.com/atfrank/MLFF-Distiller/milestone/2
- Project Board: https://github.com/atfrank/MLFF-Distiller/projects

---

## Quick Reference

| Issue | Title | Agent | Priority | Week 3 Status |
|-------|-------|-------|----------|---------------|
| #10 | Design sampling strategy | Data Pipeline | Critical | Day 1-3 |
| #11 | Structure generation | Data Pipeline | Critical | Day 3-7 |
| #12 | Teacher inference | Architecture | Critical | Day 3-7 |
| #14 | Validation framework | Testing | High | Day 1-7 |
| #17 | Training integration | Training | High | Week 4 |

---

## M2 Success Vision

By the end of M2 (Dec 14), we will have:
- **120,000 high-quality training samples** from Orb-v2 and FeNNol
- **Diverse chemical space** (organic, biomolecules, inorganic)
- **Validated dataset** (all quality checks passed)
- **Training-ready** (seamless integration with M1 framework)
- **7 days ahead of schedule** (maintaining M1 momentum)

**Let's build an exceptional dataset for world-class model distillation!**

---

**Status**: Ready for Week 3 Launch
**Next Action**: All agents begin assigned tasks (Nov 24)
**Questions?**: Comment on GitHub issues with @mentions

**Good luck, team! Let's make M2 as successful as M1.**
