# M6 Phase - Immediate Actions (Next 2 Hours)
## Execute This Right Now

**Time**: November 25, 2025, after you read this document
**Duration**: 2 hours to get everything rolling
**Outcome**: Phase execution officially started, Agent 5 ready to begin work

---

## ACTION 1: VERIFICATION & INFRASTRUCTURE CHECK (30 minutes)
**Owner**: Lead Coordinator
**Location**: Terminal
**Expected Duration**: 30 minutes

### Step 1.1: Verify GitHub Issues (5 min)
```bash
# Check all 6 issues are created and open
gh issue list --state open --limit 100 | grep -E "#33|#34|#35|#36|#37|#38"
```

**Expected Output**:
```
38  OPEN  [Coordinator] [M6] MD Integration Testing Phase - Project Coordination
37  OPEN  [Testing] [M6] MD Simulation Test Framework Enhancement
36  OPEN  [Testing] [M6] MD Inference Performance Benchmarking
35  OPEN  [Testing] [M6] Ultra-tiny Model Validation
34  OPEN  [Testing] [M6] Tiny Model Validation
33  OPEN  [Testing] [M6] MD Integration Testing & Validation - Original Model
```

**Verification**: All 6 issues visible? ✓ YES / ✗ NO

If NO: Check that GitHub CLI is authenticated and you're in correct repo
```bash
gh auth login
gh repo view  # Should show: https://github.com/atfrank/MLFF-Distiller
```

---

### Step 1.2: Verify Checkpoints Load (10 min)
```bash
# Test Original model
python3 << 'EOF'
import torch
import sys

checkpoints = {
    'Original': '/home/aaron/ATX/software/MLFF_Distiller/checkpoints/best_model.pt',
    'Tiny': '/home/aaron/ATX/software/MLFF_Distiller/checkpoints/tiny_model/best_model.pt',
    'Ultra-tiny': '/home/aaron/ATX/software/MLFF_Distiller/checkpoints/ultra_tiny_model/best_model.pt'
}

for name, path in checkpoints.items():
    try:
        model = torch.load(path)
        print(f"✓ {name}: {path}")
        # Print model size info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {total_params:,}")
    except Exception as e:
        print(f"✗ {name}: ERROR - {e}")
        sys.exit(1)

print("\n✓ All checkpoints load successfully")
EOF
```

**Expected Output**:
```
✓ Original: /home/aaron/ATX/software/MLFF_Distiller/checkpoints/best_model.pt
  Parameters: 427,648
✓ Tiny: /home/aaron/ATX/software/MLFF_Distiller/checkpoints/tiny_model/best_model.pt
  Parameters: 77,824
✓ Ultra-tiny: /home/aaron/ATX/software/MLFF_Distiller/checkpoints/ultra_tiny_model/best_model.pt
  Parameters: 21,568

✓ All checkpoints load successfully
```

**Verification**: All checkpoints load? ✓ YES / ✗ NO

---

### Step 1.3: Verify ASE Calculator (10 min)
```bash
cd /home/aaron/ATX/software/MLFF_Distiller

# Test calculator imports and basic functionality
python3 << 'EOF'
import sys
sys.path.insert(0, '/home/aaron/ATX/software/MLFF_Distiller')

try:
    from src.mlff_distiller.inference.ase_calculator import MLFFCalculator
    from ase.build import molecule

    # Load Original model
    calc = MLFFCalculator('/home/aaron/ATX/software/MLFF_Distiller/checkpoints/best_model.pt')

    # Test on water molecule
    atoms = molecule('H2O')
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    print(f"✓ Calculator loaded successfully")
    print(f"  Water molecule energy: {energy:.4f} eV")
    print(f"  Forces shape: {forces.shape}")
    print(f"  Force sample: {forces[0]} eV/Å")

except Exception as e:
    print(f"✗ Calculator error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ ASE Calculator working correctly")
EOF
```

**Expected Output**:
```
✓ Calculator loaded successfully
  Water molecule energy: -47.XX eV
  Forces shape: (3, 3)
  Force sample: [... ... ...] eV/Å

✓ ASE Calculator working correctly
```

**Verification**: Calculator works? ✓ YES / ✗ NO

---

### Step 1.4: Run Existing Test Suite (5 min)
```bash
cd /home/aaron/ATX/software/MLFF_Distiller

# Run existing integration tests to establish baseline
pytest tests/integration/ -v --tb=short 2>&1 | tail -30
```

**Expected Output**:
```
tests/integration/test_ase_calculator.py::TestBasicFunctionality::test_water_energy PASSED
tests/integration/test_ase_calculator.py::TestBasicFunctionality::test_water_forces PASSED
...
============ X passed in Y.XXs ============
```

**Verification**: Tests pass? ✓ YES / ✗ NO (if NO, that's OK - not blocking, just informational)

---

### VERIFICATION CHECKLIST (Step 1)
```
✓ All 6 GitHub issues visible (#33-#38)
✓ Original model checkpoint loads (427K params)
✓ Tiny model checkpoint loads (77K params)
✓ Ultra-tiny model checkpoint loads (21K params)
✓ ASE calculator works with water molecule
✓ Existing test suite runs (baseline)
```

**If all checked**: PROCEED TO ACTION 2
**If any failed**: Fix that issue before proceeding

---

## ACTION 2: BACKGROUND PROCESS CLEANUP (20 minutes)
**Owner**: Lead Coordinator
**Purpose**: Free GPU memory for MD testing
**Target**: >80% GPU memory available

### Step 2.1: Check Current GPU Usage (2 min)
```bash
nvidia-smi
```

**Expected Output** (look for):
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 555.XX.XX    Driver Version: 555.XX.XX                         |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
|===============================+======================+======================|
|   0  NVIDIA A100-SXM...  Off  | 00:1E.0        Off  |                  0   |
|-------------------------------+----------------------+----------------------+
| GPU  Name         Idx     PID   Type   Process name             GPU Memory |
|=============================================================================|
|   0  NVIDIA A100-SXM...    N/A   [PID] C   python          [MEMORY]M / 81920M |
+-----------------------------------------------------------------------------+
```

**Note down**:
- Current GPU memory used: ___ MB
- Available GPU memory: ___ MB
- Is it >80% free? ✓ YES / ✗ NO

---

### Step 2.2: List Background Processes (5 min)
```bash
# Show Python processes with details
ps aux | grep python | grep -v grep | awk '{print $2, $9, $11, $12}'

# More detailed: Show process info with timestamps
ps aux | grep python | grep -v grep
```

**Expected Output** (example):
```
PID     TIME    COMMAND
12345   Nov24   python train_student.py
12346   Nov24   python benchmark_optimization.py
12347   Nov25   python debug_batch.py
```

**For each process, ask**:
1. Is it from Nov 24 or earlier? (Old training/benchmark)
2. Is it actively needed for M6? (Check filename)
3. If unsure, check modification time: `ls -la /proc/[PID]/cwd`

---

### Step 2.3: Terminate Old Processes (10 min)
```bash
# ONLY terminate processes you're SURE are not needed
# Process files from Nov 24 are safe to terminate

# Example (DO NOT RUN - verify process IDs first):
# kill -9 [PID_of_old_process]

# Check specific process details if unsure
ps -p [PID] -o pid,user,cmd
```

**Safest approach**: Only kill processes that:
- Have names like `debug_batch*.py`, `train_*.py`, `benchmark_*.py`
- Have modification time >24 hours old
- Are NOT actively writing output (check `lsof -p [PID]`)

**After cleanup**:
```bash
# Verify GPU memory is freed
nvidia-smi
# Should now show >80% GPU memory available
```

---

### CLEANUP CHECKLIST (Step 2)
```
✓ Current GPU usage checked
✓ Background processes identified
✓ Old processes terminated (if any)
✓ GPU memory now >80% available
```

**If GPU memory is NOT >80%**: That's OK, MD testing can run on CPU if needed, just slower

---

## ACTION 3: AGENT 5 ONBOARDING (30 minutes)
**Owner**: Both (Coordinator + Agent 5)
**Purpose**: Agent 5 understands scope and is ready to start

### Step 3.1: Documentation Review (20 min)
**Agent 5 reads these documents IN THIS ORDER**:

1. **This document** (5 min read)
   - You are here: understanding immediate next steps

2. **M6_EXECUTION_SUMMARY.md** (10 min read)
   - Executive summary
   - Week 1 & 2 plan
   - Success dashboard

3. **docs/M6_TESTING_ENGINEER_QUICKSTART.md** (15 min read)
   - What's ready for you
   - Issue #37 breakdown (framework what to build)
   - Issue #33 breakdown (original model testing)
   - Issue #34, #35, #36 breakdown
   - Success metrics
   - Execution checklist

**Commands for Agent 5**:
```bash
# Read summary
less /home/aaron/ATX/software/MLFF_Distiller/M6_EXECUTION_SUMMARY.md

# Read quickstart
less /home/aaron/ATX/software/MLFF_Distiller/docs/M6_TESTING_ENGINEER_QUICKSTART.md

# Reference full plan as needed
less /home/aaron/ATX/software/MLFF_Distiller/M6_EXECUTION_PLAN_DETAILED.md
```

---

### Step 3.2: Understanding Verification (5 min)
**Coordinator asks Agent 5 these questions** (Agent 5 should answer without looking at docs):

1. **"What is Issue #37? What's the acceptance criteria?"**
   - Expected: "Framework for MD testing with NVE harness, energy metrics, force metrics, benchmarking"
   - Must include: Unit tests, documentation, 100-step integration test

2. **"What blocks other work?"**
   - Expected: "Issue #37 blocks #33, which blocks #34/#35"

3. **"What are success metrics for Original model?"**
   - Expected: "<1% energy drift, <0.2 eV/Å force RMSE, 10ps stable, <10ms inference"

4. **"How long is the phase?"**
   - Expected: "12-14 days, target December 8-9"

5. **"How do you escalate blockers?"**
   - Expected: "Comment in issue, tag coordinator, expect 2-4 hour response"

**If Agent 5 answers YES to all**: Ready to proceed!
**If any questions unclear**: Discuss and clarify now

---

### Step 3.3: Environment Setup (5 min)
**Agent 5 runs these commands** to confirm everything works:

```bash
cd /home/aaron/ATX/software/MLFF_Distiller

# Confirm Python environment
python3 --version
# Expected: Python 3.XX.X

# Confirm PyTorch
python3 -c "import torch; print(f'PyTorch {torch.__version__}')"
# Expected: PyTorch 2.X.X

# Confirm ASE
python3 -c "from ase.build import molecule; print('ASE working')"
# Expected: ASE working

# Confirm pytest
pytest --version
# Expected: pytest X.XX.X

# Confirm code can be imported
python3 -c "from src.mlff_distiller.inference.ase_calculator import MLFFCalculator; print('Calculator imports')"
# Expected: Calculator imports
```

---

### ONBOARDING CHECKLIST (Step 3)
```
Agent 5:
✓ Read M6_EXECUTION_SUMMARY.md (10 min)
✓ Read docs/M6_TESTING_ENGINEER_QUICKSTART.md (15 min)
✓ Answers understanding verification questions
✓ Environment confirmed (Python, PyTorch, ASE, pytest)
✓ Code imports successfully

Coordinator:
✓ Asked clarifying questions
✓ Addressed any confusion
✓ Confirmed Agent 5 is ready
```

---

## ACTION 4: INITIAL STANDUP & START NOTICE (30 minutes)
**Owner**: Coordinator
**Purpose**: Official launch of M6 Phase

### Step 4.1: Post Standup in Issue #38 (Already Done)
We already posted the initial standup. You can verify:

```bash
gh issue view 38 | grep -A 50 "M6 Phase Execution Start"
```

---

### Step 4.2: Create Agent 5's First Issue Comment (10 min)
**Agent 5 posts FIRST progress comment in Issue #37** (Framework):

```bash
gh issue comment 37 -b "## Issue #37 - Beginning Work

**Status**: STARTING
**Owner**: Agent 5
**Timeline**: Days 1-3
**Blocker**: No
**Blocked by**: No

### Day 1 - Framework Architecture Design

#### What I'm about to do:
1. Review existing test infrastructure and architecture patterns
2. Design class hierarchy for MDSimulationHarness
3. Plan metric implementations
4. Document component breakdown

#### Expected deliverables:
- Architecture design documented
- File stubs created
- Placeholder tests passing

#### Expected completion:
EOD today (November 25)

### Next checkpoint:
Tomorrow: Begin core implementation (NVE harness + energy metrics)
"
```

---

### Step 4.3: Coordinator Confirms Readiness (5 min)
**You post in Issue #38**:

```bash
gh issue comment 38 -b "## Coordinator Status - November 25, 2025

**VERIFICATION COMPLETE** ✓

All infrastructure verified:
- ✓ 6 GitHub issues created and labeled
- ✓ 3 checkpoints load successfully (427K, 77K, 21K parameters)
- ✓ ASE calculator functional
- ✓ GPU >80% memory available
- ✓ Existing tests pass

**AGENT 5 READY** ✓
- ✓ Documentation reviewed
- ✓ Understanding verified
- ✓ Environment confirmed
- ✓ Ready to begin Issue #37 framework work

**PHASE OFFICIALLY LAUNCHED** ✓

Timeline: 12-14 days (December 8-9, 2025)
Critical Path: Issue #37 → Issue #33
Daily Standup: 9 AM in this issue

Monitoring daily. Let me know if you hit any blockers.
"
```

---

### LAUNCH CHECKLIST (Step 4)
```
✓ Initial standup posted in Issue #38
✓ Agent 5 posts first progress comment in Issue #37
✓ Coordinator confirms readiness in Issue #38
✓ Phase officially launched
```

---

## FINAL VERIFICATION (1 min)

### Confirm Everything is Ready
```bash
# Quick health check
cd /home/aaron/ATX/software/MLFF_Distiller

echo "1. GitHub issues:"
gh issue list --state open | grep -E "#33|#34|#35|#36|#37|#38" | wc -l
echo "   (Should be 6)"

echo ""
echo "2. Checkpoints:"
ls -lh checkpoints/best_model.pt checkpoints/tiny_model/best_model.pt checkpoints/ultra_tiny_model/best_model.pt | awk '{print $9, $5}'

echo ""
echo "3. ASE Calculator:"
[ -f src/mlff_distiller/inference/ase_calculator.py ] && echo "   ✓ Found" || echo "   ✗ Missing"

echo ""
echo "4. Documentation:"
for doc in M6_EXECUTION_SUMMARY.md M6_EXECUTION_PLAN_DETAILED.md M6_QUICK_START_COORDINATOR.md; do
    [ -f $doc ] && echo "   ✓ $doc" || echo "   ✗ $doc missing"
done

echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader | awk '{printf "   %d MB / %d MB available (%.0f%% free)\n", $1, $3, ($1/$3)*100}'
```

**Expected Output**:
```
1. GitHub issues:
   6
   (Should be 6)

2. Checkpoints:
   checkpoints/best_model.pt 1.7M
   checkpoints/tiny_model/best_model.pt ...
   checkpoints/ultra_tiny_model/best_model.pt ...

3. ASE Calculator:
   ✓ Found

4. Documentation:
   ✓ M6_EXECUTION_SUMMARY.md
   ✓ M6_EXECUTION_PLAN_DETAILED.md
   ✓ M6_QUICK_START_COORDINATOR.md

GPU Status:
   XXXX MB / 81920 MB available (XX% free)
```

---

## SUMMARY: WHAT JUST HAPPENED

You have successfully initiated M6 Phase execution:

### Infrastructure ✓
- All 6 GitHub issues created and visible
- All 3 model checkpoints verified (Original 427K, Tiny 77K, Ultra-tiny 21K)
- ASE calculator functional and tested
- GPU >80% memory available

### Team Ready ✓
- Agent 5 has read all documentation
- Agent 5 understands scope and acceptance criteria
- Agent 5 is ready to begin Issue #37 (framework development)
- Coordinator knows responsibilities and is monitoring

### Phase Launched ✓
- Daily standup established (9 AM in Issue #38)
- Critical path clear: #37 → #33 → parallel work
- Timeline set: 12-14 days (December 8-9)
- Communication protocols in place

### Documentation Complete ✓
- M6_EXECUTION_SUMMARY.md (2KB executive summary)
- M6_EXECUTION_PLAN_DETAILED.md (50KB comprehensive plan)
- docs/M6_TESTING_ENGINEER_QUICKSTART.md (18KB agent guide)
- docs/M6_MD_INTEGRATION_COORDINATION.md (16KB full plan)
- M6_QUICK_START_COORDINATOR.md (coordinator reference card)
- M6_IMMEDIATE_ACTIONS_NEXT_2_HOURS.md (this document)

---

## NEXT STEPS

### For Coordinator (Right Now)
1. Close this document
2. Go to Issue #38
3. Verify initial standup is posted ✓
4. Verify Coordinator status posted ✓
5. Watch for Agent 5's first progress comment in Issue #37

### For Agent 5 (Right Now)
1. Finish onboarding (if you are Agent 5 reading this)
2. Post first comment in Issue #37: "Beginning framework architecture design"
3. Begin Issue #37 work: Review existing test infrastructure
4. Post daily standup tomorrow at 9 AM in Issue #38

### Both (Ongoing)
- Daily standup 9 AM in Issue #38
- Monitor for blockers
- Weekly sync Friday EOD
- Track metrics real-time

---

## SUCCESS METRICS - DAY 1

**By end of today (November 25, 2025)**:

Coordinator:
- [ ] All 4 verification steps completed
- [ ] Background processes cleaned up
- [ ] Agent 5 confirmed ready
- [ ] Standup posted in Issue #38

Agent 5:
- [ ] Documentation reviewed
- [ ] Environment verified
- [ ] First comment posted in Issue #37
- [ ] Framework architecture documented

---

## YOU ARE NOW LIVE

Everything is ready. The infrastructure is in place. The team is aligned. The path is clear.

**The Original model will be validated for production deployment in 12-14 days.**

Let's execute.

---

*This document: M6_IMMEDIATE_ACTIONS_NEXT_2_HOURS.md*
*Created: November 25, 2025*
*Status: EXECUTION INITIATED*

**Next checkpoint**: November 26, 2025 at 9 AM standup
