# Agent 5 - M6 Phase Execution Startup Checklist

## ✅ PRE-EXECUTION VERIFICATION (Do This First)

### Step 1: Read Essential Documentation (30 minutes)
- [ ] Read `M6_FINAL_HANDOFF.md` (5 min)
- [ ] Read `docs/M6_TESTING_ENGINEER_QUICKSTART.md` (20 min)
- [ ] Skim `M6_QUICK_REFERENCE.txt` for quick lookup (5 min)

### Step 2: Verify Environment (10 minutes)
```bash
# Run this verification script
bash /home/aaron/ATX/software/MLFF_Distiller/scripts/m6_execution_startup.sh
```

**Expected Output**: All 8 checks should PASS ✓

### Step 3: Confirm GitHub Issues (5 minutes)
Check that all 6 issues exist:
- [ ] Issue #33: Original Model MD Testing
- [ ] Issue #34: Tiny Model Validation  
- [ ] Issue #35: Ultra-tiny Model Validation
- [ ] Issue #36: Performance Benchmarking
- [ ] Issue #37: Test Framework Enhancement (START HERE)
- [ ] Issue #38: Master Coordination

---

## 🚀 IMMEDIATE ACTIONS (TODAY)

### Action 1: Post in Issue #37
Post this comment in GitHub Issue #37:

```markdown
## Framework Development - Day 1 Starting

**Status**: Starting work on MD test framework

**Today's Tasks**:
1. Design NVE MD harness class structure
2. Define energy conservation metric interface
3. Define force accuracy metric interface
4. Draft trajectory analysis utilities structure
5. Create initial class skeleton

**Expected Output**: Architecture design document

**Questions/Blockers**: None at this time
```

### Action 2: Post Daily Standup in Issue #38
Post this first standup in GitHub Issue #38:

```markdown
## Standup - November 25, 2025

**Completed**: 
- Read all essential documentation
- Verified environment setup
- All 6 GitHub issues confirmed ready
- Ready to begin Issue #37

**Plan for Today**:
- Design MD test framework architecture
- Create class structure for NVE harness
- Define metric interfaces
- Draft trajectory analysis utilities

**Blockers**: None

**Next Checkpoint**: Post architecture design for review by end of day
```

### Action 3: Start Architecture Design
Begin designing the MD test framework with these components:

**NVE MD Harness Class** (`nve_md_harness.py`):
```python
class NVEMDHarness:
    """Handles 10+ picosecond NVE molecular dynamics simulations"""
    
    def __init__(self, model, atoms, temperature_K=300):
        """Initialize harness with model and atoms"""
        pass
    
    def run_simulation(self, steps=1000, dt=0.5):
        """Run NVE simulation, return trajectory"""
        pass
    
    def get_energy_conservation_metrics(self):
        """Compute energy drift, conservation metrics"""
        pass
```

**Energy Metrics** (`metrics/energy_metrics.py`):
- Total energy drift (%)
- Energy conservation ratio
- Kinetic/potential energy stability

**Force Metrics** (`metrics/force_metrics.py`):
- Force RMSE vs teacher
- Force MAE
- Angular error distribution

**Trajectory Analysis** (`analysis/trajectory_analysis.py`):
- Trajectory stability
- Atom displacement analysis
- Statistical summaries

---

## 📊 DAILY WORKFLOW

### Every Morning (9 AM):
1. Post standup in Issue #38 with:
   - What you completed yesterday
   - What you plan today (3-5 tasks)
   - Any blockers
   - Key metrics update

2. Example format:
```markdown
## Standup - [DATE]

**Completed**: [3-5 items from yesterday]

**Plan Today**: 
- Task 1
- Task 2
- Task 3

**Blockers**: [None/describe if any]

**Metrics**: [Progress on framework development]
```

### Every Evening:
1. Commit code: `git add . && git commit -m "Issue #37: [work description]"`
2. Push to remote: `git push origin main`
3. Update Issue #37 with progress comment

### When Blocked:
1. Post in Issue #37: "BLOCKER: [description]"
2. Tag in Issue #38: "@atfrank_coord BLOCKER - [description]"
3. Wait for response (2 hour SLA)
4. Resume work immediately

---

## 🎯 ISSUE #37 SUCCESS CRITERIA (By Day 3)

**MUST Complete ALL**:
- [ ] NVE MD harness implemented (~200 lines)
- [ ] Energy conservation metrics working (~100 lines)
- [ ] Force accuracy metrics working (~100 lines)
- [ ] Trajectory analysis utilities working (~100 lines)
- [ ] Benchmarking decorators implemented (~50 lines)
- [ ] Unit tests written (>80% coverage)
- [ ] Integration test passes (<2 minutes)
- [ ] Full documentation with examples
- [ ] Code committed and pushed

**Definition of "Done"**:
- All tests pass
- No outstanding TODOs
- Documentation complete
- Coordinator has reviewed and approved

---

## 📁 KEY FILES YOU'LL USE

**Documentation**:
```
M6_FINAL_HANDOFF.md                        # Start here
docs/M6_TESTING_ENGINEER_QUICKSTART.md     # Detailed guide
docs/M6_MD_INTEGRATION_COORDINATION.md     # Full spec
M6_QUICK_REFERENCE.txt                     # Daily lookup
```

**Model Checkpoints**:
```
checkpoints/best_model.pt                  # Original (427K)
checkpoints/tiny_model/best_model.pt       # Tiny (77K)
checkpoints/ultra_tiny_model/best_model.pt # Ultra-tiny (21K)
```

**Test Data**:
```
data/generative_test/moldiff/test_10mols_*/0.sdf # Test molecules
```

**Execution**:
```
scripts/m6_execution_startup.sh             # Verification
```

---

## 🔗 CRITICAL DEPENDENCIES

**Before You Start Issue #33**:
- [ ] Issue #37 MUST be 100% complete
- [ ] All tests MUST pass
- [ ] Coordinator must approve framework design

**Issue #33 Depends On**:
- [ ] Issue #37 framework ready to use
- [ ] Framework tested and validated
- [ ] Documentation complete

**Parallel Work (Can Start Anytime)**:
- [ ] Issue #36 (Benchmarking) can start while #37 is in progress

---

## 💬 COMMUNICATION

**Daily Standup**: Issue #38 at 9 AM

**Technical Questions**: Comment in Issue #37

**Architecture/Design Questions**: Comment in Issue #37 and tag @atfrank_coord

**Blockers**: Post in Issue #38 with @atfrank_coord tag

**Response SLAs**:
- Standup response: 1 hour
- Technical questions: 4 hours
- Blockers: 2 hours
- Architecture decisions: 4 hours

---

## 🏁 NEXT MILESTONES

**Day 3**: Issue #37 COMPLETE (Framework ready for #33)

**Day 6**: Issue #33 COMPLETE (Original model validated)

**Day 9**: Phase COMPLETE (All issues closed, final report)

---

## 🚀 YOU'RE READY TO BEGIN!

Everything is in place. All infrastructure is verified. All documentation is prepared. The path is clear.

**Start with Issue #37 now.**

Good luck! 🎯

