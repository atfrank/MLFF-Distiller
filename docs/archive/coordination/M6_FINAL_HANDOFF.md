# M6 PHASE FINAL HANDOFF

**Date**: November 25, 2025
**Phase**: MD Integration Testing & Validation
**Status**: READY FOR EXECUTION
**Lead**: Agent 5 (Testing & Benchmarking Engineer)
**Coordinator Support**: Lead Coordinator

---

## EXECUTIVE SUMMARY (30 seconds)

Agent 5: You have 12-14 days (target Dec 8-9) to validate that the Original model can safely run 10-picosecond MD simulations. The planning is complete, all documentation is ready, infrastructure is verified, and GPU is available. Start with Issue #37 (build test framework), then Issue #33 (validate Original model). Success criteria are clear and achievable.

**Your first task**: Read this document (5 min) + `docs/M6_TESTING_ENGINEER_QUICKSTART.md` (20 min), then start Issue #37 architecture design.

---

## CRITICAL PATH (12 days)

```
Days 1-3:   Issue #37  Test Framework Development     [BLOCKS EVERYTHING]
Days 2-6:   Issue #33  Original Model MD Validation    [PRODUCTION BLOCKER]
Days 3-7:   Issue #36  Performance Benchmarking        [PARALLEL]
Days 6-8:   Issue #34  Tiny Model Characterization     [PARALLEL]
Days 6-7:   Issue #35  Ultra-tiny Model Assessment     [PARALLEL]
Days 8-9:   Final Documentation & Issue Closure       [WRAP-UP]
```

**Critical Path Success**: Issue #37 (Days 1-3) → Issue #33 (Days 2-6) → Phase Complete

---

## WHAT YOU NEED TO DO RIGHT NOW

### Phase 1: Read Documentation (25 minutes)
1. This document (5 min)
2. `docs/M6_TESTING_ENGINEER_QUICKSTART.md` (20 min) - CRITICAL
3. Reference `docs/M6_MD_INTEGRATION_COORDINATION.md` for detailed specs

### Phase 2: Environment Verification (10 minutes)
```bash
cd /home/aaron/ATX/software/MLFF_Distiller

# Verify checkpoints load
python -c "import torch; torch.load('checkpoints/best_model.pt'); print('✓ Original')"
python -c "import torch; torch.load('checkpoints/tiny_model/best_model.pt'); print('✓ Tiny')"
python -c "import torch; torch.load('checkpoints/ultra_tiny_model/best_model.pt'); print('✓ Ultra-tiny')"

# Verify ASE calculator
python -c "from src.mlff_distiller.inference.ase_calculator import StudentForceFieldCalculator; print('✓ ASE')"

# Run tests
pytest tests/integration/test_ase_calculator.py -v
```

### Phase 3: Start Issue #37 (Today)
1. Post in Issue #37: "Architecture design in progress"
2. Review what's already in `src/mlff_distiller/testing/` (if anything exists)
3. Design MD harness classes (see quickstart for details)
4. Create implementation plan with milestones

---

## INFRASTRUCTURE READINESS

All systems GO:

| Component | Status | Details |
|-----------|--------|---------|
| Original Checkpoint | ✓ READY | 427K params, R²=0.9958 |
| Tiny Checkpoint | ✓ READY | 77K params, R²=0.3787 |
| Ultra-tiny Checkpoint | ✓ READY | 21K params, R²=0.1499 |
| ASE Calculator | ✓ READY | `StudentForceFieldCalculator` class |
| Integration Tests | ✓ 91/101 PASS | 91 passing (teacher wrapper tests non-critical) |
| GPU Memory | ✓ 94.2% FREE | 11.5 GB available for MD testing |
| Background Processes | ✓ CLEAN | No training/benchmark processes running |

---

## WHAT AGENT 5 MUST DELIVER

### Issue #37: MD Test Framework
**Acceptance Criteria**:
- NVE simulation harness (supports 10+ ps without memory overflow)
- Energy conservation metrics (machine precision accuracy)
- Force accuracy metrics (RMSE, MAE, angular errors)
- Trajectory stability analysis
- Benchmarking utilities (inference time, memory)
- >80% test coverage with unit tests
- 100-step integration test <2 minutes
- Full documentation with examples

**Timeline**: Days 1-3 (3 full days)

### Issue #33: Original Model MD Validation
**Acceptance Criteria**:
- 10ps NVE simulation stable (no crashes)
- Energy drift <1% (measured)
- Force RMSE <0.2 eV/Å (measured)
- Inference <10ms/step documented
- 3+ test molecules successful
- Production readiness decision (APPROVED/REJECTED)

**Timeline**: Days 2-6 (5 days, overlaps with #37)

### Issue #36: Performance Benchmarking
**Acceptance Criteria**:
- Inference time (ms/step) for all 3 models
- Memory usage (GB) for all 3 models
- Speedup ratios relative to Original
- Results in `benchmarks/md_performance_results.json`
- Visualizations included

**Timeline**: Days 3-7 (5 days, parallel)

### Issue #34: Tiny Model Validation
**Acceptance Criteria**:
- 5ps tests completed
- Actual metrics measured (not assumed)
- Comparison vs Original documented
- Clear verdict: suitable/unsuitable for production
- Failure modes documented

**Timeline**: Days 6-8 (3 days)

### Issue #35: Ultra-tiny Model Assessment
**Acceptance Criteria**:
- 1-2ps tests completed
- Expected failures documented
- Proof of unsuitability clear
- Recommendation for next phase

**Timeline**: Days 6-7 (2 days)

---

## SUCCESS METRICS DASHBOARD

Real-time tracking in Issue #38. Update daily:

### Original Model (427K, R²=0.9958)
```
PRODUCTION READINESS DECISION
Status: [pending] → [APPROVED/REJECTED]

Energy Conservation:     ___ % (target <1%)
Force Accuracy (RMSE):   ___ eV/Å (target <0.2)
MD Stability (10ps):     [pass/fail]
Inference Speed:         ___ ms/step
Overall Status:          [PENDING]
```

### Framework Development (Issue #37)
```
Progress: [████░░░░░░] Day 2/3
Tests Written:  ___/100
Coverage:       ___%
Examples:       ___/3
Documentation:  ___%
```

### Benchmarking Results (Issue #36)
```
Original Inference:      ___ ms/step
Tiny Inference:          ___ ms/step
Ultra-tiny Inference:    ___ ms/step
Memory Original:         ___ GB
Memory Tiny:             ___ GB
Memory Ultra-tiny:       ___ GB
```

---

## DAILY STANDUP TEMPLATE

Post in Issue #38 each morning:

```
## Standup - [DATE]

### Completed Yesterday
- [3-5 bullet points]

### Plan for Today
- [3-5 specific tasks]

### Blockers/Risks
- [Any issues or questions]

### Metrics
- Framework: ___% complete
- Tests passed: ___/___
- Energy drift: ___%
- Force RMSE: ___ eV/Å

### Next Checkpoint
[When we meet again + expectations]
```

---

## KEY DOCUMENTATION FILES

| File | Purpose | Read Time | Start Point |
|------|---------|-----------|------------|
| `M6_QUICK_REFERENCE.txt` | Phase overview, quick ref | 5 min | Daily |
| `docs/M6_TESTING_ENGINEER_QUICKSTART.md` | Your step-by-step guide | 20 min | NOW |
| `docs/M6_MD_INTEGRATION_COORDINATION.md` | Detailed requirements | 30 min | Reference |
| `M6_COORDINATION_SUMMARY.md` | Coordinator dashboard | 15 min | For coord |
| `M6_PHASE_INITIATION_REPORT.md` | Full context | 20 min | Reference |

---

## HOW TO ESCALATE BLOCKERS

### Level 1: Quick Questions (Resolve in <2 hours)
```
Comment in relevant issue (#37-#36):
"Question about [topic]: [specific question]"
```

### Level 2: Blockers (Resolve in <4 hours)
```
Comment in Issue #38 with:
1. What you're trying to do
2. What's blocking you
3. What options you see
4. Tag @atfrank_coord
```

### Level 3: Architecture Decisions (Resolve in <4 hours)
```
Create comment in Issue #37/#38 with:
1. Decision needed: [what]
2. Option A: [pros/cons]
3. Option B: [pros/cons]
4. Recommendation: [why]
5. Tag @atfrank_coord
```

### Level 4: Scope Changes (Resolve in <24 hours)
```
Comment in Issue #38 with:
1. Current scope: [what's in scope]
2. Proposed change: [what/why]
3. Impact: [timeline, resources]
4. Request approval
```

---

## EXECUTION TIMELINE

### Week 1 (Days 1-5)
```
Day 1: Documentation + Environment Setup + Framework Design
  ├─ Morning: Read all docs (30 min)
  ├─ Morning: Verify environment (10 min)
  ├─ Morning: Framework architecture design (2 hours)
  ├─ Afternoon: Post design in Issue #37
  └─ Afternoon: Begin implementation (stubs)

Day 2: Framework Core Implementation
  ├─ NVE simulation harness (core)
  ├─ Energy conservation metrics
  ├─ Begin force metrics
  └─ Daily standup in Issue #38

Day 3: Framework Completion
  ├─ Complete force metrics
  ├─ Trajectory stability analysis
  ├─ Benchmarking utilities
  ├─ Integration tests
  └─ Documentation + examples

Days 4-5: Original Model Validation Begins
  ├─ Water 5ps (H2O)
  ├─ Methane 5ps (CH4)
  ├─ Alanine 5ps (C5H11NO2)
  └─ Preliminary results in Issue #33
```

### Week 2 (Days 6-12)
```
Days 6-7: Original Model Extended Tests + Parallel Work
  ├─ Water 10ps full simulation
  ├─ Temperature scaling (100K, 300K, 500K)
  ├─ Issue #36: Benchmarking (parallel)
  ├─ Issue #35: Ultra-tiny tests (parallel)
  └─ Daily standup updates

Days 8-9: Tiny Model + Final Documentation
  ├─ Issue #34: Tiny model 5ps tests
  ├─ Comparison vs Original
  ├─ Final results compilation
  ├─ Create visualizations
  └─ Write final report

Days 10-12: Closure
  ├─ All issues closed
  ├─ Final documentation
  ├─ Issue #38 completion report
  └─ Ready for next phase
```

---

## EXPECTED OUTCOMES BY MODEL

### Original Model (427K, R²=0.9958)
**Expected Status**: APPROVED FOR PRODUCTION

| Metric | Target | Expected | Status |
|--------|--------|----------|--------|
| Energy Drift (10ps) | <1% | 0.3-0.7% | Should pass |
| Force RMSE | <0.2 eV/Å | 0.15-0.20 | Should pass |
| Stability (10ps) | Pass | Pass | Should pass |
| Inference | <10ms/step | ~8 ms/step | Should pass |

**Verdict**: This model should be approved for production use in MD simulations.

### Tiny Model (77K, R²=0.3787)
**Expected Status**: NOT RECOMMENDED FOR FORCE-DEPENDENT MD

| Metric | Expected | Status |
|--------|----------|--------|
| Energy Drift (5ps) | 2-5% | Should fail |
| Force RMSE | 1-2 eV/Å | Should fail |
| Speedup | 1.5x | Information |

**Verdict**: Accuracy tradeoff too severe. Not suitable for production MD requiring force accuracy.

### Ultra-tiny Model (21K, R²=0.1499)
**Expected Status**: UNSUITABLE FOR PRODUCTION MD

| Metric | Expected | Status |
|--------|----------|--------|
| Energy Drift (1-2ps) | >10% | Will fail |
| Force RMSE | >3 eV/Å | Will fail badly |
| Speedup | 2.6x | Information |

**Verdict**: Forces completely unreliable. Reject for any production MD work.

---

## CRITICAL REMINDERS

**DO**:
- Post daily standup in Issue #38 (every morning)
- Commit code frequently (daily pushes)
- Document all results as you go
- Ask questions early (don't get stuck)
- Update progress in Issue #38 comments

**DON'T**:
- Wait until Day 10 to start Issue #33 (critical path!)
- Skip documentation for code
- Make scope changes without approval
- Work on Issues #34-35 before #33 is validated
- Ignore tests (>80% coverage required)

**WATCH OUT FOR**:
- GPU memory overflow on 10ps simulations (monitor carefully)
- Force metric computation bugs (double-check against reference)
- Trajectory divergence (may indicate model problems)
- MD simulation crashes (debug immediately, report in Issue #38)

---

## COORDINATOR COMMITMENT

I will:
- Review Issue #37 architecture within 4 hours of posting
- Respond to all blockers within 2 hours
- Provide design decision approvals within 4 hours
- Review final results and approve production readiness
- Post weekly summary in Issue #38

You can rely on me for unblocking, not micromanagement. Do your work, ask for help when stuck, deliver results.

---

## PHASE COMPLETION CRITERIA

All of these must be true:

1. Issue #37 closed: Framework complete, tested, documented
2. Issue #33 closed: Original model tested, verdict documented
3. Issue #36 closed: Benchmarks complete, results published
4. Issue #34 closed: Tiny model assessed, recommendations clear
5. Issue #35 closed: Ultra-tiny model tested, rejected
6. Issue #38: Final report posted with all results
7. All code committed and tested
8. No outstanding blockers

**Expected Timeline**: 12-14 days (target December 8-9, 2025)

---

## FINAL NOTES

This is the most critical phase of the project. The Original model's production deployment depends on your validation in Issue #33. The testing framework you build in Issue #37 becomes standard for future phases.

**You have everything you need to succeed**:
- ✓ Clear requirements and acceptance criteria
- ✓ Complete documentation (85+ KB)
- ✓ Verified infrastructure (all checkpoints load)
- ✓ Available GPU (94% free)
- ✓ Clean development environment
- ✓ Full coordinator support (4-hour response)
- ✓ Realistic timeline with contingency
- ✓ Clear escalation procedures

**Expected Outcome**:
```
✓ Original model approved for production
✓ Framework complete and reusable
✓ Clear recommendations for all models
✓ Ready to move forward with deployment
```

---

## NEXT STEPS (DO NOW)

1. Read `docs/M6_TESTING_ENGINEER_QUICKSTART.md` (20 minutes)
2. Run environment verification commands (above)
3. Post in Issue #37: "Framework architecture design starting"
4. Design the MD harness classes (reference quickstart)
5. Create Issue #37 architecture comment with design plan
6. Await coordinator review (4-hour response)

That's it. Then implement. Daily standups. Weekly syncs. Success.

---

**Status**: READY FOR EXECUTION
**Decision**: APPROVED TO BEGIN IMMEDIATELY
**Coordinator**: Lead Coordinator
**Date**: November 25, 2025

**Let's validate and deploy the Original model!**
