# M2 Milestone: Data Pipeline - Coordination Plan

**Document Version**: 1.0
**Date**: November 23, 2025
**Coordinator**: Lead Coordinator
**Status**: Planning Complete, Ready for Week 3 Kickoff

---

## Executive Summary

**Milestone Overview**:
- **M1 Status**: 100% complete (13 days ahead of schedule)
- **M2 Start**: Week 3 (November 24, 2025)
- **M2 Deadline**: December 21, 2025 (4 weeks)
- **Target Completion**: December 14, 2025 (1 week buffer)

**M2 Objective**: Generate high-quality, diverse training datasets from teacher models (Orb-v2, FeNNol) to enable distillation training in M3-M4.

**Key Metrics**:
- 8 issues created (Issues #10-#17)
- 120,000 total samples (100K train, 10K val, 10K test)
- Target dataset size: <50 GB (compressed)
- Expected completion: 7 days ahead of schedule

---

## M2 Issue Summary

| Issue | Title | Agent | Priority | Est. Days | Dependencies |
|-------|-------|-------|----------|-----------|--------------|
| [#10](https://github.com/atfrank/MLFF-Distiller/issues/10) | Design molecular structure sampling strategy | Data Pipeline | Critical | 3 | None |
| [#11](https://github.com/atfrank/MLFF-Distiller/issues/11) | Implement structure generation pipeline | Data Pipeline | Critical | 5 | #10 |
| [#12](https://github.com/atfrank/MLFF-Distiller/issues/12) | Create teacher model inference pipeline | Architecture | Critical | 4 | #11 |
| [#13](https://github.com/atfrank/MLFF-Distiller/issues/13) | Implement HDF5 dataset writer and manager | Data Pipeline | High | 3 | #12 |
| [#14](https://github.com/atfrank/MLFF-Distiller/issues/14) | Create dataset quality validation framework | Testing | High | 4 | #13 |
| [#15](https://github.com/atfrank/MLFF-Distiller/issues/15) | Implement dataset statistics and analysis tools | Data Pipeline | Medium | 3 | #14 |
| [#16](https://github.com/atfrank/MLFF-Distiller/issues/16) | Create production dataset generation workflow | Data Pipeline | Critical | 5 | #11,#12,#13,#14,#15 |
| [#17](https://github.com/atfrank/MLFF-Distiller/issues/17) | Integrate dataset with training pipeline | Training | High | 2 | #16 |

**Total Estimated Effort**: 29 agent-days over 4 weeks (parallelization expected)

---

## M2 Timeline & Critical Path

### Week 3 (Nov 24-30, 2025)
**Focus**: Foundation - Sampling Strategy & Structure Generation

```
Day 1-3 (Nov 24-26):
  #10 (Sampling Strategy) ──┐
                            ├──> Complete by Nov 26
                            │
  Teacher Installation ─────┘   (Architecture, parallel work)

Day 3-7 (Nov 26-30):
  #10 complete ──> #11 (Structure Generation) [Start]
  Teacher ready  ──> #12 (Label Generation) [Start, parallel]
```

**Deliverables by Week 3 End**:
- Sampling strategy documented (#10)
- Structure generation infrastructure in place (#11, partial)
- Teacher models installed and tested (#12, setup)

### Week 3-4 Transition (Dec 1-7, 2025)
**Focus**: Data Generation & Storage

```
#11 (Structure Gen) ──────┐
                          ├──> #13 (HDF5 Writer)
#12 (Label Gen) ──────────┘

#13 complete ──> #14 (Validation) [Start]
                 #15 (Analysis) [Start, parallel]
```

**Deliverables**:
- 1,000 test structures generated (#11)
- 1,000 labeled samples (#12)
- HDF5 storage working (#13)

### Week 4 (Dec 8-14, 2025)
**Focus**: Production Workflow & Integration

```
#13 + #14 + #15 ──> #16 (Production Workflow)
                     ├──> 120K dataset generation
                     │
#16 complete ────────┘
     │
     └──> #17 (Training Integration)
```

**Deliverables**:
- Production dataset (120K samples) (#16)
- Validation report (#14, #16)
- Training integration complete (#17)

### Week 4 Final (Dec 15-21, 2025)
**Focus**: Validation, Documentation & M2 Closeout

- Final validation and testing
- Documentation review
- M2 milestone review
- Prepare for M3 kickoff

---

## Agent Assignment Matrix

### Data Pipeline Engineer (Lead for M2)
**Primary Issues**: #10, #11, #13, #15, #16
**Supporting**: #14, #17
**Estimated Load**: 19 days (primary) + 2 days (support) = 21 days

**Week-by-Week**:
- Week 3: #10 (design), #11 (structure gen)
- Week 3-4: #13 (HDF5), #15 (analysis)
- Week 4: #16 (production workflow)

### ML Architecture Designer
**Primary Issues**: #12
**Supporting**: #10, #16
**Estimated Load**: 4 days (primary) + 2 days (support) = 6 days

**Week-by-Week**:
- Week 3: Teacher installation, #12 (label gen)
- Week 4: Support #16 (production workflow)

### Testing & Benchmark Engineer
**Primary Issues**: #14
**Supporting**: #11, #15, #16
**Estimated Load**: 4 days (primary) + 3 days (support) = 7 days

**Week-by-Week**:
- Week 3-4: #14 (validation framework)
- Week 4: Support #16 (production validation)

### Distillation Training Engineer
**Primary Issues**: #17
**Supporting**: #10, #16
**Estimated Load**: 2 days (primary) + 2 days (support) = 4 days

**Week-by-Week**:
- Week 3: Support #10 (data requirements)
- Week 4: #17 (training integration)

### CUDA Optimization Engineer
**Primary Issues**: None (M2)
**Supporting**: Monitor for optimization opportunities
**Estimated Load**: 1 day (monitoring)

**Week-by-Week**:
- Week 3-4: Monitor data generation performance
- Identify bottlenecks for future M5 work

---

## M2 Success Criteria

### Dataset Quality Metrics
- [ ] 120,000 total samples (100K train, 10K val, 10K test)
- [ ] Diverse chemical space (>10 elements: C, H, O, N, S, P, metals)
- [ ] System sizes: 10-500 atoms (covering target range)
- [ ] Both periodic (>50%) and non-periodic systems
- [ ] Energy/force labels from Orb-v2 and FeNNol teachers
- [ ] <0.1% outliers/invalid samples

### Data Quality Checks
- [ ] All samples pass validation (#14)
- [ ] Energy distribution reasonable (no extreme outliers)
- [ ] Force magnitudes <10 eV/Å (99th percentile)
- [ ] No atom overlaps (>0.5 Å separation)
- [ ] Diversity score >0.8 (structural space coverage)

### Performance Metrics
- [ ] Data generation throughput >1000 samples/hour (with teachers)
- [ ] HDF5 read performance: >90% GPU utilization during training
- [ ] Dataset size <50 GB (compressed)
- [ ] Data loading latency <10ms per batch (batch=32)

### Integration Metrics
- [ ] Training pipeline successfully loads M2 dataset
- [ ] No data errors in first 1000 training iterations
- [ ] Normalization statistics properly integrated
- [ ] Train/val/test splits respected

### Documentation Metrics
- [ ] Complete dataset documentation (generation params, composition)
- [ ] All scripts have usage examples
- [ ] Validation report generated and reviewed
- [ ] Dataset versioning and provenance tracked

---

## Risk Assessment & Mitigation Strategies

### Risk 1: Teacher Model Installation Challenges
**Probability**: Medium
**Impact**: High
**Mitigation**:
- Start structure generation (#11) independently (doesn't need real teachers)
- Use mock calculators for pipeline testing
- Architecture agent dedicates time to teacher installation (parallel)
- Document installation process thoroughly
- Consider alternative: use pre-generated data if teachers fail

**Action Items**:
- Architecture agent: Begin teacher installation Day 1
- Data Pipeline: Design #11 to work with mock calculators initially
- Fallback plan: Use ASE built-in calculators (EMT) for testing

### Risk 2: Data Generation Too Slow
**Probability**: Medium
**Impact**: High
**Mitigation**:
- Implement parallel/distributed generation from start (#11, #12)
- Use GPU batching where possible (teacher-dependent)
- Start with smaller dataset (50K) and scale if needed
- Consider cloud/HPC resources if local compute insufficient
- Profile generation early (Week 3) to identify bottlenecks

**Action Items**:
- Week 3: Benchmark generation throughput (target: >1000 samples/hour)
- Data Pipeline: Implement multiprocessing in #11, #12
- CUDA agent: Monitor and optimize bottlenecks

### Risk 3: Insufficient Chemical Diversity
**Probability**: Low
**Impact**: High
**Mitigation**:
- Define diversity metrics upfront (#10)
- Validate diversity continuously during generation (#14)
- Implement active learning-based sampling (#11)
- Include diversity analysis in #15
- Iterate on sampling strategy if needed

**Action Items**:
- #10: Define clear diversity requirements and metrics
- #14: Implement diversity validation checks
- #15: Generate diversity reports (PCA, composition analysis)

### Risk 4: Dataset Quality Issues
**Probability**: Medium
**Impact**: Medium
**Mitigation**:
- Implement validation early (#14)
- Run validation continuously during generation (#16)
- Define quality criteria clearly (#10)
- Build outlier detection and removal
- Manual review of sample data

**Action Items**:
- #14: Comprehensive validation framework
- #16: Inline quality checks during generation
- Testing agent: Manual inspection of 100 random samples

### Risk 5: Storage/Memory Constraints
**Probability**: Low
**Impact**: Medium
**Mitigation**:
- Use HDF5 compression (#13)
- Implement streaming writes (no full dataset in memory)
- Monitor disk space during generation
- Implement cleanup of intermediate files
- Benchmark compression ratios early

**Action Items**:
- #13: Test compression (gzip vs lzf)
- Data Pipeline: Monitor disk usage in #16
- Target: <50 GB for 120K samples

### Risk 6: Integration Failures with Training
**Probability**: Low
**Impact**: Medium
**Mitigation**:
- Test integration early (Week 3 with mock data)
- Training agent involved throughout M2 (#10, #16)
- Integration tests in #17
- Test with MolecularDataset from M1 (already tested)

**Action Items**:
- Week 3: Training agent tests data format compatibility
- #17: Comprehensive integration tests
- Validate with small dataset before full 120K generation

---

## Week 3 Kickoff Plan (Nov 24-30, 2025)

### Pre-Kickoff Checklist (Nov 23)
- [x] All M2 issues created (#10-#17)
- [x] Milestone M2 configured (due Dec 21)
- [x] Agent assignments clear
- [x] Coordination plan documented
- [ ] Kickoff meeting scheduled (optional, async coordination preferred)
- [ ] All agents notified of M2 start

### Day 1-2 (Nov 24-25): Planning & Infrastructure Setup

**Data Pipeline Engineer**:
- [ ] Start #10: Design sampling strategy
  - Review M1 infrastructure (MolecularDataset, teacher wrappers)
  - Research chemical diversity requirements
  - Draft sampling strategy document
  - Create configuration files
- [ ] Prepare for #11: Review ASE MD documentation
- [ ] Set up development branch: `feature/M2-data-pipeline`

**ML Architecture Designer**:
- [ ] Install teacher models (Orb-v2, FeNNol)
  - Set up conda environment
  - Install orb-models package
  - Install FeNNol/JAX
  - Test basic inference
- [ ] Prepare for #12: Review teacher_wrappers.py
- [ ] Document installation process

**Testing & Benchmark Engineer**:
- [ ] Review M2 requirements and acceptance criteria
- [ ] Design validation framework architecture (#14)
  - Identify validation checks needed
  - Research outlier detection methods
  - Plan report generation
- [ ] Set up test fixtures for M2 components
- [ ] Review dataset quality literature

**Distillation Training Engineer**:
- [ ] Review current training pipeline
- [ ] Identify data format requirements for #17
- [ ] Review normalization strategies
- [ ] Plan training integration approach
- [ ] Test MolecularDataset with HDF5 format

**CUDA Optimization Engineer**:
- [ ] Monitor M2 progress (passive)
- [ ] Review data generation code for optimization opportunities
- [ ] Prepare profiling tools for bottleneck identification

### Day 3-5 (Nov 26-28): Core Implementation Begins

**Data Pipeline Engineer**:
- [ ] Complete #10: Finalize sampling strategy
  - Get review/approval from Architecture and Training agents
- [ ] Start #11: Structure generation implementation
  - Implement MD trajectory sampling
  - Implement normal mode sampling
  - Add parallelization support
  - Create CLI script
  - Write unit tests

**ML Architecture Designer**:
- [ ] Complete teacher installation
  - Verify Orb-v2 working (test on sample structures)
  - Verify FeNNol working
  - Benchmark inference speed
- [ ] Start #12: Label generation implementation
  - Design LabelGenerator class
  - Implement batching logic
  - Add error handling
  - Test with sample structures

**Testing & Benchmark Engineer**:
- [ ] Start #14: Validation framework implementation
  - Implement energy validators
  - Implement force validators
  - Implement geometry validators
  - Create test cases

**Distillation Training Engineer**:
- [ ] Support #10: Review and approve sampling strategy
- [ ] Test data loading with mock HDF5 files
- [ ] Prepare training config updates

### Day 6-7 (Nov 29-30): First Integration Checkpoint

**All Agents**:
- [ ] Integration milestone: Generate and validate 1000 test samples
  - Data Pipeline: Generate 1000 structures (#11)
  - Architecture: Label 1000 structures (#12)
  - Testing: Validate 1000 samples (#14, basic checks)
- [ ] Review progress on all M2 issues
- [ ] Identify blockers and risks
- [ ] Adjust timeline if needed

**Specific Tasks**:
- [ ] Data Pipeline: Generate 1000 test structures successfully
- [ ] Architecture: Label test structures with Orb-v2 and FeNNol
- [ ] Testing: Run basic validation (no critical failures)
- [ ] Training: Verify data format compatible with MolecularDataset
- [ ] Lead Coordinator: Week 3 progress review, update timeline

---

## Communication Protocol

### Daily Standup (Async)
Each agent posts daily update on GitHub issues:
- What did you complete yesterday?
- What are you working on today?
- Any blockers or questions?

**Format**: Comment on relevant issue with `@agent` mentions

### Issue Comments
- Use @mentions to notify specific agents
- Tag blockers immediately with "blocked" label
- Request reviews with "needs-review" label
- Link related issues with `#issue_number`

### Progress Tracking
- Update issue status regularly (in-progress, blocked, review, done)
- Move issues on GitHub Projects board
- Add comments on significant milestones
- Report blockers immediately

### Code Reviews
- All PRs require review from Lead Coordinator
- Domain-specific reviews from relevant agents
- Address review comments within 24 hours
- Use PR templates for consistency

### Milestone Reviews
- Week 3 end: Progress review (Nov 30)
- Week 4 mid: Production workflow check (Dec 7)
- Week 4 end: M2 completion review (Dec 14)

---

## Integration Checkpoints

### Checkpoint 1: Week 3 End (Nov 30)
**Goal**: 1000 test samples generated and validated

**Criteria**:
- [ ] #10 complete (sampling strategy approved)
- [ ] #11 in progress (structure generation working)
- [ ] #12 in progress (label generation working)
- [ ] 1000 test samples generated and labeled
- [ ] Basic validation passed (no critical errors)

**Review**: All agents sync on progress, identify blockers

### Checkpoint 2: Week 4 Start (Dec 7)
**Goal**: HDF5 infrastructure complete, validation framework ready

**Criteria**:
- [ ] #11 complete (structure generation)
- [ ] #12 complete (label generation)
- [ ] #13 complete (HDF5 writer)
- [ ] #14 complete (validation framework)
- [ ] #15 complete (analysis tools)
- [ ] Ready to start production workflow (#16)

**Review**: Integration test with all components, validate workflow

### Checkpoint 3: Week 4 End (Dec 14)
**Goal**: M2 complete, production dataset ready

**Criteria**:
- [ ] #16 complete (120K dataset generated)
- [ ] #17 complete (training integration)
- [ ] All validation checks passed
- [ ] Dataset documentation complete
- [ ] Training successfully loads dataset
- [ ] M2 success criteria met

**Review**: M2 milestone review, prepare M3 kickoff

---

## Documentation Requirements

Each issue must deliver:
- Code with docstrings (Google style)
- Unit tests (>80% coverage)
- Usage examples in docstrings or examples/
- README updates if needed
- Integration tests where applicable

M2-specific documentation:
- `docs/data_generation_strategy.md` (#10)
- `docs/dataset_v1.0.md` (#16)
- `docs/training.md` updates (#17)
- Dataset validation report (HTML/PDF) (#14)
- Dataset statistics report (#15)

---

## M2 Deliverables Checklist

### Code Deliverables
- [ ] `src/mlff_distiller/data/generation/structure_sampler.py` (#11)
- [ ] `src/mlff_distiller/data/generation/label_generator.py` (#12)
- [ ] `src/mlff_distiller/data/hdf5_writer.py` (#13)
- [ ] `src/mlff_distiller/data/hdf5_reader.py` (#13)
- [ ] `src/mlff_distiller/data/validation/quality_checker.py` (#14)
- [ ] `src/mlff_distiller/data/analysis/dataset_analyzer.py` (#15)

### Script Deliverables
- [ ] `scripts/generate_structures.py` (#11)
- [ ] `scripts/generate_labels.py` (#12)
- [ ] `scripts/validate_dataset.py` (#14)
- [ ] `scripts/analyze_dataset.py` (#15)
- [ ] `scripts/generate_full_dataset.py` (#16)
- [ ] `examples/train_with_production_data.py` (#17)

### Configuration Deliverables
- [ ] `configs/sampling_config.yaml` (#10)
- [ ] `configs/production_dataset.yaml` (#16)
- [ ] `configs/training_config.yaml` updates (#17)

### Data Deliverables
- [ ] Test dataset (1000 samples) (#11, #12)
- [ ] Production dataset (120K samples, <50 GB) (#16)
- [ ] Normalization statistics (YAML) (#15)
- [ ] Dataset manifest with checksums (#16)

### Documentation Deliverables
- [ ] `docs/data_generation_strategy.md` (#10)
- [ ] `docs/data_generation.md` (#11)
- [ ] `docs/dataset_v1.0.md` (#16)
- [ ] `docs/training.md` updates (#17)
- [ ] Validation report (HTML/PDF) (#14)
- [ ] Statistics report with plots (#15)

### Test Deliverables
- [ ] Unit tests for all modules (>80% coverage)
- [ ] Integration tests (#16, #17)
- [ ] Performance benchmarks (#12, #13, #17)
- [ ] All 315+ M1 tests still passing

---

## Post-M2 Planning

### M2 Success Handoff to M3
Once M2 complete:
- Production dataset (120K samples) ready
- Training integration validated
- Normalization parameters exported
- Dataset documentation complete

**M3 Can Start Immediately With**:
- Real training data (no mocks)
- Validated data quality
- Optimized data loading
- Baseline training benchmarks

### M2 Retrospective (Dec 15-16)
- What went well?
- What could be improved?
- Lessons learned for M3-M6
- Update risk assessment for future milestones
- Adjust timeline if needed

### M3 Preparation (Dec 17-21)
- Review M3 objectives (Model Architecture)
- Identify M3 issues
- Assign M3 agents
- Schedule M3 kickoff (Week 5)

---

## Appendix: Key File Paths

### M1 Infrastructure (Available)
- `/home/aaron/ATX/software/MLFF_Distiller/src/mlff_distiller/models/teacher_wrappers.py` - Teacher calculators
- `/home/aaron/ATX/software/MLFF_Distiller/src/mlff_distiller/data/dataset.py` - MolecularDataset
- `/home/aaron/ATX/software/MLFF_Distiller/src/mlff_distiller/data/transforms.py` - Data augmentation
- `/home/aaron/ATX/software/MLFF_Distiller/src/mlff_distiller/training/trainer.py` - Training framework
- `/home/aaron/ATX/software/MLFF_Distiller/benchmarks/md_benchmark.py` - MD benchmarks

### M2 Directories (To Create)
- `/home/aaron/ATX/software/MLFF_Distiller/src/mlff_distiller/data/generation/` - Data generation
- `/home/aaron/ATX/software/MLFF_Distiller/src/mlff_distiller/data/validation/` - Validation
- `/home/aaron/ATX/software/MLFF_Distiller/src/mlff_distiller/data/analysis/` - Analysis
- `/home/aaron/ATX/software/MLFF_Distiller/data/` - Dataset storage
- `/home/aaron/ATX/software/MLFF_Distiller/configs/` - Configuration files

### Repository Root
- `/home/aaron/ATX/software/MLFF_Distiller/` - Project root

---

## Quick Reference

**M2 Timeline**: Nov 24 - Dec 21, 2025 (4 weeks)
**M2 Issues**: #10-#17 (8 issues)
**M2 Milestone**: https://github.com/atfrank/MLFF-Distiller/milestone/2
**Lead Agent**: Data Pipeline Engineer
**Critical Path**: #10 → #11 → #13 → #16 → #17 (19 days)
**Target Completion**: Dec 14, 2025 (7 days ahead)

**Success Metric**: 120K high-quality samples ready for M3 training

---

**Document Status**: Ready for M2 Kickoff
**Next Action**: Week 3 Day 1 (Nov 24) - All agents begin assigned issues
**Coordinator Contact**: Available for blockers via GitHub issue @mentions

**Last Updated**: November 23, 2025
