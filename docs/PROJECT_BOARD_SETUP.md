# GitHub Project Board Setup Guide

This document provides instructions for setting up a GitHub Project Board to track the ML Force Field Distillation project.

---

## Project Board Overview

We will create a GitHub Projects (v2) board with multiple views to track issues by status, agent, and milestone.

### Board Name
**ML Force Field Distillation - Development Board**

---

## Setup Instructions

### Option 1: Using GitHub Web Interface (Recommended)

1. **Create Project Board**:
   - Go to: https://github.com/atfrank/MLFF-Distiller/projects
   - Click "New project"
   - Select "Board" template
   - Name: "ML Force Field Distillation - Development Board"
   - Click "Create"

2. **Configure Columns**:
   - Rename default columns to:
     - **Backlog**: Issues not yet started
     - **Ready**: Issues ready to be worked on (dependencies met)
     - **In Progress**: Issues currently being worked on
     - **Review**: PRs/issues under review
     - **Done**: Completed work

3. **Add All Issues to Board**:
   - Click "Add item" or "+"
   - Search for and add Issues #1-9
   - All issues start in "Backlog" column

4. **Move Issues to Appropriate Columns**:
   - **Ready** column: Issues #1, #2, #3, #4, #8 (no dependencies, can start immediately)
   - **Backlog** column: Issues #5, #6, #7, #9 (have dependencies)

5. **Create Custom Views**:

   **View 1: By Status (Default)**
   - Already configured with columns above

   **View 2: By Agent**
   - Click "+" next to views
   - Select "Table" layout
   - Name: "By Agent"
   - Group by: Label (filter to show agent:* labels)
   - Fields to show: Title, Status, Milestone, Priority, Assignees

   **View 3: By Milestone**
   - Click "+" next to views
   - Select "Board" layout
   - Name: "By Milestone"
   - Group by: Milestone
   - Shows all M1 issues in one column, M2 in another, etc.

   **View 4: Week 1 Focus**
   - Click "+" next to views
   - Select "Board" layout
   - Name: "Week 1 Focus"
   - Filter: milestone:M1 AND (priority:critical OR priority:high)
   - Shows only critical Week 1 issues

6. **Configure Automation** (optional but recommended):
   - Go to Project Settings → Workflows
   - Enable automation:
     - "Item added to project" → Move to "Backlog"
     - "Item closed" → Move to "Done"
     - "Pull request merged" → Move to "Done"
     - "Pull request opened" → Move to "Review"

---

### Option 2: Using GitHub CLI

```bash
# Note: Projects API v2 is still in beta, may require additional setup

# Create project (requires org-level project creation)
gh project create --owner atfrank --title "ML Force Field Distillation - Development Board"

# Get project ID (replace <project-number> with actual number from above)
PROJECT_ID=$(gh project list --owner atfrank --format json | jq -r '.projects[] | select(.title=="ML Force Field Distillation - Development Board") | .id')

# Add issues to project
gh project item-add $PROJECT_ID --owner atfrank --url https://github.com/atfrank/MLFF-Distiller/issues/1
gh project item-add $PROJECT_ID --owner atfrank --url https://github.com/atfrank/MLFF-Distiller/issues/2
gh project item-add $PROJECT_ID --owner atfrank --url https://github.com/atfrank/MLFF-Distiller/issues/3
gh project item-add $PROJECT_ID --owner atfrank --url https://github.com/atfrank/MLFF-Distiller/issues/4
gh project item-add $PROJECT_ID --owner atfrank --url https://github.com/atfrank/MLFF-Distiller/issues/5
gh project item-add $PROJECT_ID --owner atfrank --url https://github.com/atfrank/MLFF-Distiller/issues/6
gh project item-add $PROJECT_ID --owner atfrank --url https://github.com/atfrank/MLFF-Distiller/issues/7
gh project item-add $PROJECT_ID --owner atfrank --url https://github.com/atfrank/MLFF-Distiller/issues/8
gh project item-add $PROJECT_ID --owner atfrank --url https://github.com/atfrank/MLFF-Distiller/issues/9
```

Note: The CLI approach for Projects v2 is more complex. Web interface is recommended.

---

## Initial Board State

After setup, the board should look like:

### Backlog Column
- Issue #5: [Testing] MD simulation benchmark framework (blocked by #2)
- Issue #6: [Architecture] Student ASE Calculator (blocked by #2)
- Issue #7: [Testing] ASE Calculator interface tests (blocked by #2)
- Issue #9: [CUDA] MD profiling framework (blocked by #2)

### Ready Column
- Issue #1: [Data Pipeline] Data loading infrastructure
- Issue #2: [Architecture] Teacher model wrappers (CRITICAL PATH)
- Issue #3: [Training] Training framework
- Issue #4: [Testing] Pytest setup
- Issue #8: [CUDA] CUDA environment setup

### In Progress Column
- (Empty initially - agents move issues here when they start work)

### Review Column
- (Empty initially - PRs move here when ready for review)

### Done Column
- (Empty initially - completed work moves here)

---

## Using the Project Board

### For Agents (How to Update Board)

**When starting work on an issue**:
1. Move issue from "Ready" to "In Progress"
2. Self-assign the issue
3. Add label: `status:in-progress`
4. Add comment: "Starting work on this issue"

**When creating a PR**:
1. Link PR to issue (use "Closes #X" in PR description)
2. Move issue to "Review" column
3. Add label: `status:needs-review`

**When PR is merged**:
1. Issue automatically moves to "Done" (if automation enabled)
2. Close the issue (if not auto-closed)

**If blocked**:
1. Add label: `status:blocked`
2. Add comment explaining blocker
3. Tag @Lead-Coordinator
4. Issue stays in "In Progress" until unblocked

### For Lead Coordinator (How to Monitor)

**Daily Board Check**:
1. **Backlog**: Any new issues to add?
2. **Ready**: Are blocked issues ready to move here?
3. **In Progress**: Is work progressing? Any stuck issues?
4. **Review**: Are PRs being reviewed promptly?
5. **Done**: Celebrate completed work!

**Red Flags to Watch**:
- Issues stuck in "In Progress" for >3 days without updates
- "Review" column backing up (PRs not reviewed)
- "Ready" column empty (agents need new work)
- Critical path issues not moving (Issue #2 in Week 1)

**Weekly Updates**:
- Count issues moved to "Done" (velocity)
- Identify bottlenecks (which column is backing up?)
- Adjust plans based on progress

---

## Board Views Explained

### View 1: Status Board (Default)
**Purpose**: Day-to-day workflow tracking
**Best for**: Standups, daily coordination, seeing what's in progress
**Columns**: Backlog → Ready → In Progress → Review → Done

### View 2: By Agent (Table)
**Purpose**: Workload balancing and agent assignments
**Best for**: Checking agent capacity, assigning new work
**Groups**: One row per agent showing their issues

### View 3: By Milestone (Board)
**Purpose**: Milestone progress tracking
**Best for**: Planning, seeing M1 vs M2 vs M3 progress
**Columns**: One per milestone (M1, M2, M3, M4, M5, M6)

### View 4: Week 1 Focus (Board)
**Purpose**: Current sprint focus
**Best for**: Week 1 prioritization, avoiding distraction
**Filter**: Only shows Week 1 critical/high priority issues

---

## Workflow States

### Backlog
**Meaning**: Issue exists but not ready to start
**Reasons**:
- Blocked by dependencies
- Lower priority
- Needs refinement

**Example**: Issue #5 is blocked by Issue #2

### Ready
**Meaning**: All dependencies met, ready to start
**Actions**: Agent can claim and start work
**Criteria**:
- All blocking issues resolved
- Clear acceptance criteria
- Resources available

**Example**: Issue #2 has no dependencies, ready to start Monday

### In Progress
**Meaning**: Agent actively working on this
**Actions**: Regular updates, flag blockers
**Expectations**:
- Daily progress or updates
- PR created when work is testable

**Example**: Agent claimed Issue #2, working on implementation

### Review
**Meaning**: PR created, waiting for review
**Actions**: Review code, test, provide feedback
**Expectations**:
- Review within 24 hours
- Address feedback promptly

**Example**: PR for Issue #4 created, waiting for review

### Done
**Meaning**: PR merged, issue closed
**Actions**: None (celebrate!)
**Criteria**:
- PR merged to main
- All tests passing
- Documentation updated

**Example**: Issue #4 PR merged, pytest infrastructure working

---

## Automation Rules (If Enabled)

### Recommended Automations

1. **New Issue → Backlog**
   - Trigger: Issue added to project
   - Action: Move to "Backlog" column

2. **PR Opened → Review**
   - Trigger: Pull request opened
   - Action: Move linked issue to "Review"

3. **PR Merged → Done**
   - Trigger: Pull request merged
   - Action: Move linked issue to "Done"

4. **Issue Closed → Done**
   - Trigger: Issue closed
   - Action: Move to "Done" column

### Manual Transitions (No Automation)

These require manual updates:
- Backlog → Ready (when dependencies complete)
- Ready → In Progress (when agent starts work)
- In Progress → Review (when PR created)
- Review → In Progress (if changes requested)

---

## Best Practices

### Keep Board Current
- Update issue status when state changes
- Move cards promptly (don't wait for daily standup)
- Add comments explaining state changes

### Use Labels Effectively
- Labels supplement columns (e.g., `status:blocked` + comment)
- Labels enable filtering and views
- Consistent labeling helps automation

### Link PRs to Issues
- Always use "Closes #X" in PR description
- Enables automatic transitions
- Maintains traceability

### Regular Cleanup
- Archive completed milestones
- Remove outdated labels
- Update issue descriptions if requirements change

### Communication
- Board shows status, but use comments for context
- Tag people in comments for questions
- Use issue discussions for technical details

---

## Integration with Other Tools

### GitHub Actions (CI/CD)
- CI status shows on cards in "Review" column
- Failed CI = blockers visible on board

### Milestones
- Board views can group/filter by milestone
- Track M1 → M2 → M3 progress

### Labels
- Enable filtering and custom views
- Support automation rules
- Show priority/type/agent at a glance

---

## Metrics to Track

### Velocity
- Issues completed per week
- Useful for planning future sprints

### Cycle Time
- Time from "Ready" to "Done"
- Identifies bottlenecks in process

### Review Time
- Time in "Review" column
- If high, need faster PR reviews

### Block Rate
- Percentage of issues blocked
- If high, improve dependency planning

### Column Limits (Optional)
- Limit "In Progress" to N items per agent
- Prevents overcommitment
- Focus on finishing vs starting

---

## Troubleshooting

### Issue Not Moving to Done Automatically
- Check PR is linked to issue ("Closes #X")
- Verify automation is enabled
- Manually move if needed

### Can't Find Issue on Board
- Check filters on current view
- Switch to "All items" view
- Issue may be in archived milestone

### Board Not Syncing with Issues
- Refresh page
- Check project permissions
- Verify issue is added to project

---

## Week 1 Board Snapshot

### Expected State Friday EOD

**Done**:
- Issue #2: Teacher model wrappers (CRITICAL)
- Issue #4: Pytest infrastructure
- Issue #8: CUDA environment

**Review**:
- Issue #1: Data loading infrastructure
- Issue #3: Training framework

**In Progress**:
- Issue #5: MD benchmark framework
- Issue #7: ASE interface tests
- Issue #9: MD profiling framework

**Ready**:
- Issue #6: Student ASE Calculator (unblocked by #2 completion)

**Backlog**:
- (New issues created during week)

---

## Quick Reference Commands

### View Project Board
```bash
# Web interface (easiest)
https://github.com/atfrank/MLFF-Distiller/projects

# List projects
gh project list --owner atfrank
```

### Add Issue to Board
```bash
# Via web: Click "+" on project board, search issue number
# Via CLI: gh project item-add <project-id> --url <issue-url>
```

### Update Issue Status
```bash
# Recommended: Use web interface or labels
# Labels trigger automations and are easier than CLI for Projects v2
```

---

## Next Steps After Setup

1. **Add Issues to Board**: Add all 9 created issues
2. **Configure Views**: Set up 4 views as described above
3. **Enable Automation**: Configure automatic state transitions
4. **Test Workflow**: Move one issue through full lifecycle to test
5. **Share with Team**: Post board URL in repository README
6. **Daily Updates**: Lead coordinator checks board every morning

---

**Project Board URL** (after creation):
https://github.com/atfrank/MLFF-Distiller/projects/[PROJECT_NUMBER]

(Update this link once board is created)
