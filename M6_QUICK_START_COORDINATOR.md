# M6 Phase - Coordinator Quick Reference Card
## One-Page Daily Checklist

**Date**: November 25, 2025
**Phase Duration**: 12-14 days (Target: December 8-9, 2025)
**Status**: EXECUTION INITIATED

---

## YOUR ROLE

You are the Lead Coordinator orchestrating M6 execution. You:
- Enable Agent 5 to do great work
- Monitor critical path (Issues #37 → #33)
- Make decisions when needed
- Resolve blockers quickly
- Track progress and timeline
- Approve production decisions

---

## DAILY DUTIES (9 AM)

### 1. Read Standup (5 min)
Check Issue #38 for Agent 5's standup posted at 9 AM:
- What did they complete yesterday?
- What's planned for today?
- Any blockers or risks?
- On track for timeline?

### 2. Review Progress (10 min)
Update mental model of phase progress:

**Critical Path Check**:
```
Issue #37 (Framework):     Should be ___% complete
Issue #33 (Original):      Should be ___% complete
Issue #36 (Benchmarks):    Should be ___% complete
```

**Ask yourself**:
- Are we on track for 12-14 day timeline?
- Are there any blockers preventing progress?
- Is Agent 5 having issues I should know about?
- Do I need to make any decisions?

### 3. Respond to Issues (5-10 min)
Check for comments in Issues #33-#38:
- Questions asking for your input?
- Blockers tagged for your attention?
- Design decisions needing your approval?
- Result reports to review?

**Response time target**: 4 hours for normal questions, 2 hours for blockers

### 4. Decision Checkpoints
Are there decisions waiting on you?

**Common decisions**:
- Framework architecture options? → Make choice, post in #37
- Original model results? → Review, approve "production ready" or investigate
- Metric threshold disagreements? → Set policy, communicate
- Timeline impact? → Assess, decide on extension or adjustment

---

## WEEKLY SYNC (FRIDAY EOD)

Post summary comment in Issue #38:
```
## Weekly Summary - Week [#]

Completed this week:
- [Issue #X: partial/complete status]
- [Issue #Y: partial/complete status]

Progress metrics:
- Issue #37 (Framework): [status]
- Issue #33 (Original): [status]
- Issue #36 (Benchmarks): [status]

Timeline status:
- On track for Dec 8-9? [YES/ON TRACK/NEEDS ATTENTION]

Next week plan:
- [3-5 focus areas]

Risks/concerns:
- [Any emerging issues?]
```

---

## BLOCKER RESPONSE PROTOCOL

**When Agent 5 Tags You for a Blocker**:

1. **Assess** (5 min)
   - What's the actual problem?
   - What has already been tried?
   - How does it impact timeline?

2. **Gather Context** (5 min)
   - Related issues/PRs?
   - Past similar problems?
   - What are the options?

3. **Decide** (5-10 min)
   - What's the best path forward?
   - Any resources needed?
   - Timeline impact?

4. **Communicate** (2 min)
   - Post decision in issue comment
   - Explain reasoning
   - Set next steps and timeline
   - Check: Is Agent 5 unblocked and can proceed?

**Example response**:
```
@Agent5 - I reviewed the energy conservation metric issue.

The problem is [summary]. I recommend option [X] because [reasoning].

Next steps:
1. [Action by Agent 5]
2. [Action by you if needed]
3. [Check-in point]

Timeline: Should be resolved by [time], unblocks [Issue #X] on schedule.

Let me know if you hit any issues implementing this.
```

---

## CRITICAL SUCCESS METRICS

Track these daily. They determine phase success:

### Original Model (Issue #33)
```
Energy Drift:        [target <1%]      → Actual: ___ %
Force RMSE:          [target <0.2]     → Actual: ___ eV/Å
MD Stability (10ps):  [target PASS]     → Status: ___
Production Ready:    [target YES]      → Decision: ___
```

### Framework (Issue #37)
```
Unit Tests:          [target pass]     → Status: ___
Code Coverage:       [target >80%]     → Actual: ___ %
Documentation:       [target complete] → Status: ___
Integration Test:    [target <2min]    → Actual: ___ s
```

### Benchmarking (Issue #36)
```
All 3 models measured: [YES/NO]
Speedup calculated:   [YES/NO]
Visualizations:       [YES/NO]
```

---

## TIMELINE AT A GLANCE

```
DAY 1-3:    Issue #37 Framework          [████░░░░░] Design & Build
DAY 2-6:    Issue #33 Original Model     [░░░░░░░░░░] Blocked by #37
DAY 3-7:    Issue #36 Benchmarking       [░░░░░░░░░░] Parallel
DAY 6-8:    Issue #34 Tiny               [░░░░░░░░░░] Parallel
DAY 6-7:    Issue #35 Ultra-tiny         [░░░░░░░░░░] Parallel
DAY 8-9:    Final Docs & Closure         [░░░░░░░░░░] Wrap-up

CRITICAL PATH: #37 → #33
ON SCHEDULE? [YES/NO/NEEDS ATTENTION]
```

---

## DECISION TREE

**Agent 5 asks: "Should I do X or Y?"**
```
Is it an implementation detail?          → "Your call, either works"
Is it a framework architecture choice?   → "Let me review, give me 2 hours"
Is it a metric threshold?                → "Let me decide, 4-hour turnaround"
Will it impact timeline >1 day?          → "Let's discuss all options, I decide"
Is it blocking other work?               → "Let's prioritize, 2-hour response"
```

---

## ESCALATION SIGNS (WATCH FOR THESE)

Alert level if you see:

🟢 **GREEN** (On track):
- Daily standups posted on time
- Issues moving through work steadily
- No blockers older than 4 hours
- Timeline on track

🟡 **YELLOW** (Watch closely):
- Standup missing or delayed
- Same blocker appearing twice
- Questions about core framework design
- Concerns about Original model stability

🔴 **RED** (Immediate attention):
- Blockers unresolved >4 hours
- Original model failing validation unexpectedly
- Framework architecture issues
- Timeline impact >2 days

---

## KEY CONTACTS & RESOURCES

**Agent 5**: Testing & Benchmarking Engineer
- Posts standup in Issue #38 at 9 AM
- Tags you for decisions/blockers
- Delivers results in GitHub issues

**GitHub Issues**:
- #37: Framework architecture & status
- #33: Original model validation & results
- #34: Tiny model analysis
- #35: Ultra-tiny model assessment
- #36: Performance benchmarking
- #38: Daily coordination & master status

**Key Documents**:
- `M6_EXECUTION_SUMMARY.md` - Executive summary
- `M6_EXECUTION_PLAN_DETAILED.md` - 50KB comprehensive plan
- `docs/M6_TESTING_ENGINEER_QUICKSTART.md` - Agent 5's implementation guide

---

## PRODUCTION READINESS DECISION

**For Original Model (Issue #33)**:

When you see final results, evaluate:

✓ **APPROVED** if:
- [ ] 10ps simulation completed without crashes
- [ ] Energy drift <1% (measured, not assumed)
- [ ] Force RMSE <0.2 eV/Å (measured)
- [ ] Inference time <10 ms/step
- [ ] 3+ molecule types tested
- [ ] Results stable and reproducible

✗ **INVESTIGATE FURTHER** if:
- [ ] Energy drift 1-2% (borderline)
- [ ] Force RMSE 0.2-0.3 eV/Å (marginal)
- [ ] Only tested 1-2 molecules
- [ ] Results show high variance

✗ **REJECT** if:
- [ ] Energy drift >2%
- [ ] Force RMSE >0.3 eV/Å
- [ ] Crashes or instabilities observed
- [ ] Inference time >10 ms/step

Your approval in Issue #33 comment is the decision point.

---

## PHASE COMPLETION CHECKLIST

At end of timeline (Day 9), verify:

**All Issues Closed**:
- [ ] #37: Framework complete & tested
- [ ] #33: Original model validated & decision made
- [ ] #34: Tiny model characterized
- [ ] #35: Ultra-tiny model assessed
- [ ] #36: Benchmarking complete
- [ ] #38: Final report published

**All Deliverables**:
- [ ] Code in `src/mlff_distiller/testing/`
- [ ] Tests in `tests/integration/test_md_integration.py`
- [ ] Results in `benchmarks/md_performance_results.json`
- [ ] Visualizations in `visualizations/md_validation_*.png`
- [ ] Documentation in `docs/MD_VALIDATION_*.md`

**All Decisions Made**:
- [ ] Original: Production status determined
- [ ] Tiny: Use cases established
- [ ] Ultra-tiny: Limitations documented
- [ ] Framework: Production-ready confirmed
- [ ] Next phase: Optimization targets clear

**Sign-Off**:
- [ ] Agent 5: Work complete, results reviewed
- [ ] You: Deliverables acceptable, phase approved
- [ ] Team: Lessons learned documented

---

## QUICK REFERENCE: WHO DECIDES WHAT

| Decision | Coordinator | Agent 5 | Both? |
|----------|-------------|---------|-------|
| Framework architecture | ✓ | ✓ consults | Decision: Coordinator |
| Implementation details | | ✓ | Agent 5 decides |
| Metric thresholds | ✓ | | Coordinator decides |
| Production readiness | ✓ | ✓ recommends | Decision: Coordinator |
| Blocker workarounds | | ✓ | Agent 5 tries first |
| Timeline adjustments | ✓ | ✓ recommends | Decision: Coordinator |
| Scope changes | ✓ | | Coordinator decides |
| Issue prioritization | | ✓ | Within critical path |
| Test case selection | | ✓ | Agent 5 decides |
| Results interpretation | ✓ | ✓ | Discuss, Coordinator decides |

---

## YOUR RESPONSE TIME COMMITMENTS

| Issue Type | Response Time | Format |
|-----------|---------------|--------|
| Normal questions | 4 hours | Issue comment |
| Blockers | 2 hours | Issue comment + decision |
| Design decisions | 4 hours | Issue comment with reasoning |
| Production approval | 1 hour of review | Issue comment + sign-off |
| Timeline impact | 2 hours | Issue comment with assessment |
| Emergencies | 30 min | Direct message + follow-up |

---

## WHAT SUCCESS LOOKS LIKE

**Day 3**: Framework complete, basic tests working
**Day 6**: Original model passed basic validation, shows promise
**Day 9**: Original model approved for production, framework documented, recommendations clear

**In 14 days**:
- Original model deployed to production ✓
- Framework standard tool for future validation ✓
- Clear understanding of compression tradeoffs ✓
- Next phase (optimization) ready to proceed ✓

---

## QUICK LINKS

**GitHub Project**: [MLFF-Distiller](https://github.com/atfrank/MLFF-Distiller)
**Issues**: #33, #34, #35, #36, #37, #38
**Repository**: `/home/aaron/ATX/software/MLFF_Distiller`

---

**You've got this. Agent 5 is ready. Framework is planned. Let's execute.**

*Update this as phase progresses.*
