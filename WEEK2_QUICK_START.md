# Week 2 Quick Start Guide
## For Project Coordinator

**Date**: 2025-11-23
**Status**: Week 1 Complete, Week 2 Ready to Launch

---

## What Happened in Week 1

✅ 5 critical issues completed
✅ 4,292 lines of production code
✅ 181 tests passing (100% success rate)
✅ All code pushed to GitHub (commit 4ff20d9)
✅ No blockers, no integration issues

**M1 Status**: 55% complete (5/9 issues)

---

## Week 2 Mission

Complete remaining 4 M1 issues:
- Issue #5: MD Benchmark Framework (Agent 5 - Testing)
- Issue #6: Student ASE Calculator (Agent 2 - Architecture)
- Issue #7: ASE Interface Tests (Agent 5 - Testing)
- Issue #9: MD Profiling Framework (Agent 4 - CUDA)

**Target**: M1 100% complete by Nov 30 (7 days)

---

## Week 2 Documents - READ THESE

1. **WEEK2_EXECUTIVE_SUMMARY.md** ← START HERE (this doc)
   - High-level overview
   - Week 1 recap
   - Week 2 strategy
   
2. **docs/WEEK2_COORDINATION_PLAN.md** (629 lines)
   - Detailed strategy
   - Risk analysis
   - Integration checkpoints
   
3. **docs/WEEK2_AGENT_ACTIVATION.md** (574 lines)
   - Agent-specific instructions
   - Clear deliverables
   - Timeline for each agent
   
4. **docs/INTEGRATION_CHECKLIST.md** (601 lines)
   - Day 4 checkpoint procedure
   - Day 7 M1 validation
   - Troubleshooting guide

---

## Agent Assignments

| Agent | Issue | Title | Priority | Timeline |
|-------|-------|-------|----------|----------|
| Architecture (2) | #6 | Student ASE Calculator | CRITICAL | Day 1-6 |
| Testing (5) | #5 | MD Benchmark Framework | CRITICAL | Day 1-6 |
| Testing (5) | #7 | ASE Interface Tests | CRITICAL | Day 3-7 |
| CUDA (4) | #9 | MD Profiling Framework | HIGH | Day 1-6 |

---

## Immediate Actions (Today)

### For Coordinator

**GitHub Setup**:
```bash
cd /home/aaron/ATX/software/MLFF_Distiller

# Option 1: Use gh CLI
gh issue create --title "[Testing] [M1] Create MD simulation benchmark framework" \
  --body-file docs/initial_issues/issue_22_md_benchmark_framework.md \
  --label "agent:testing,milestone:M1,type:task,priority:critical"

gh issue create --title "[Architecture] [M1] Implement ASE Calculator interface for student models" \
  --body-file docs/initial_issues/issue_26_ase_calculator_student.md \
  --label "agent:architecture,milestone:M1,type:task,priority:critical"

gh issue create --title "[Testing] [M1] Implement ASE Calculator interface tests" \
  --body-file docs/initial_issues/issue_29_ase_interface_tests.md \
  --label "agent:testing,milestone:M1,type:task,priority:critical"

gh issue create --title "[CUDA] [M1] Create MD profiling framework" \
  --body-file docs/initial_issues/issue_17_profiling_framework.md \
  --label "agent:cuda,milestone:M1,type:task,priority:high"

# Option 2: Create manually on GitHub
# Use templates in docs/initial_issues/
```

**Agent Activation**:
Send each agent:
1. Link to their section in `docs/WEEK2_AGENT_ACTIVATION.md`
2. Link to their issue template in `docs/initial_issues/`
3. Expected deliverables and timeline
4. Starting point (code references)

---

## Week 2 Schedule

**Day 1-2 (Mon-Tue)**: Kickoff
- All 3 agents start (Issues #5, #6, #9)
- Skeleton code in place
- Daily progress updates

**Day 3-4 (Wed-Thu)**: Implementation
- Issue #7 starts (Agent 5)
- Issue #6 ready for integration
- **Mid-week checkpoint**

**Day 5-6 (Fri-Sat)**: Integration
- Complete implementations
- Integration tests
- Create PRs

**Day 7 (Sun)**: Completion
- Fix integration issues
- **Mark M1 complete**
- Plan M2 kickoff

---

## Daily Coordinator Checklist

**Every Morning**:
- [ ] Check all open issues for new comments
- [ ] Review progress updates from agents
- [ ] Identify any blockers
- [ ] Respond to @mentions

**Every Evening**:
- [ ] Verify all agents posted progress update
- [ ] Check PR status and CI results
- [ ] Update mental model of integration state
- [ ] Plan next day support

**Mid-Week (Day 4)**:
- [ ] Run integration checkpoint (use INTEGRATION_CHECKLIST.md)
- [ ] Verify components working together
- [ ] Adjust schedule if needed

**End of Week (Day 7)**:
- [ ] Run full integration validation
- [ ] Verify 200+ tests passing
- [ ] Mark M1 complete
- [ ] Generate Week 2 completion report

---

## Communication Protocols

**Agent Progress Updates** (daily):
```
Progress: [what was done today]
Next: [what's planned for tomorrow]
Blockers: [any issues or none]
```

**Blocker Resolution**:
- Agent tags "status:blocked"
- Agent @mentions coordinator
- Coordinator responds <4 hours
- Document resolution

**PR Process**:
- Agent creates PR with issue reference
- CI must pass
- Coordinator reviews <24 hours
- Agent addresses comments
- Coordinator merges

---

## Integration Checkpoints

### Mid-Week (Day 4)

**What to Check**:
```bash
cd /home/aaron/ATX/software/MLFF_Distiller

# Run test suite
pytest tests/ -v

# Check new files exist
ls -lh src/mlff_distiller/models/student_calculator.py
ls -lh benchmarks/md_benchmark.py
ls -lh src/mlff_distiller/cuda/profiler.py

# Test imports
python -c "from mlff_distiller.models.student_calculator import StudentCalculator"
python -c "import sys; sys.path.insert(0, 'benchmarks'); import md_benchmark"
python -c "from mlff_distiller.cuda.profiler import MDProfiler"
```

**Success Criteria**:
- Week 1 tests still passing (181+)
- New components import without error
- Issues #5, #6, #9 at 60%+ complete
- Issue #7 started or starting soon
- No blockers

### End of Week (Day 7)

**What to Check**:
```bash
# Full test suite
pytest tests/ -v --tb=short

# Expected: 200+ tests passing
```

**Success Criteria**:
- All 4 issues complete
- M1 at 100% (9/9 issues)
- 200+ tests passing
- Integration tests pass
- Documentation updated
- No unresolved blockers

---

## Key Files Reference

### Week 2 Planning
- `WEEK2_EXECUTIVE_SUMMARY.md` - High-level overview
- `docs/WEEK2_COORDINATION_PLAN.md` - Detailed strategy
- `docs/WEEK2_AGENT_ACTIVATION.md` - Agent instructions
- `docs/INTEGRATION_CHECKLIST.md` - Validation procedures

### Issue Templates
- `docs/initial_issues/issue_22_md_benchmark_framework.md` (Issue #5)
- `docs/initial_issues/issue_26_ase_calculator_student.md` (Issue #6)
- `docs/initial_issues/issue_29_ase_interface_tests.md` (Issue #7)
- `docs/initial_issues/issue_17_profiling_framework.md` (Issue #9)

### Week 1 Code (Reference)
- `src/mlff_distiller/data/` - Data loading
- `src/mlff_distiller/models/teacher_wrappers.py` - Teacher models
- `src/mlff_distiller/training/` - Training framework
- `src/mlff_distiller/cuda/` - CUDA utilities
- `tests/conftest.py` - Test fixtures

---

## Success Metrics

**By End of Week 2**:
- [ ] Issue #5 complete: MD benchmarks
- [ ] Issue #6 complete: Student calculator
- [ ] Issue #7 complete: Interface tests
- [ ] Issue #9 complete: Profiling framework
- [ ] M1 100% complete (9/9 issues)
- [ ] 200+ tests passing
- [ ] No blockers
- [ ] Documentation updated
- [ ] Ready for M2

---

## Quick Troubleshooting

**Import Errors**:
```bash
# Verify Python path
export PYTHONPATH=/home/aaron/ATX/software/MLFF_Distiller/src:$PYTHONPATH

# Or editable install
pip install -e .
```

**Test Failures**:
```bash
# Run specific test
pytest tests/unit/test_student_calculator.py -v

# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

**Integration Issues**:
- Check INTEGRATION_CHECKLIST.md
- Run Week 1 tests first
- Verify new imports work individually
- Check for interface mismatches

---

## Contact & Escalation

**Questions**: Comment in GitHub issues
**Blockers**: Tag "status:blocked", @mention coordinator
**Urgent**: Direct communication

---

## Week 2 Motto

**"Measure twice, optimize once"**

We're building the tools to measure success. Get the metrics right, and M3-M5 optimization will be guided by data, not guesswork.

---

**Ready to launch Week 2!**

---

*Last Updated: 2025-11-23*
*Coordinator: Lead Project Coordinator*
