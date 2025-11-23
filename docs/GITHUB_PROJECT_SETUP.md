# GitHub Project Board Setup Guide

This guide walks through setting up the GitHub Project board for MLFF Distiller.

## Step 1: Create a New Project

1. Navigate to: https://github.com/atfrank/MLFF_Distiller/projects
2. Click "New project"
3. Choose "Board" template
4. Name it: "MLFF Distiller Development"
5. Description: "Multi-agent development board for ML Force Field Distillation"

## Step 2: Configure Board Columns

Create these columns (in order):

### 1. Backlog
- **Purpose**: New issues that haven't been triaged or prioritized yet
- **Automation**: None

### 2. To Do
- **Purpose**: Ready to be worked on, prioritized and assigned
- **Automation**:
  - Add newly created issues here by default

### 3. In Progress
- **Purpose**: Currently being worked on by agents
- **Automation**:
  - When issue/PR is assigned → move here
  - Limit: Prefer 1 item per agent (avoid WIP explosion)

### 4. Review
- **Purpose**: PRs waiting for review or feedback
- **Automation**:
  - When PR is created → move here
  - When review is requested → move here

### 5. Done
- **Purpose**: Completed and merged work
- **Automation**:
  - When PR is merged → move here
  - When issue is closed → move here

## Step 3: Configure Views

### View 1: By Agent (Default)
- **Group by**: Labels (agent:*)
- **Sort by**: Priority
- **Filter**: None
- Shows all issues grouped by agent assignment

### View 2: By Milestone
- **Group by**: Milestone
- **Sort by**: Priority
- **Filter**: None
- Shows progress across milestones

### View 3: Current Sprint (M1)
- **Group by**: Status (column)
- **Sort by**: Priority
- **Filter**: milestone:M1
- Shows only M1 work

### View 4: Blocked Items
- **Group by**: Agent
- **Sort by**: Updated (oldest first)
- **Filter**: label:status:blocked
- Shows all blocked items needing attention

### View 5: High Priority
- **Group by**: Agent
- **Sort by**: Priority
- **Filter**: label:priority:high OR label:priority:critical
- Shows urgent work

## Step 4: Add Issues to Project

### Manual Method
1. Go to each issue
2. Click "Projects" in right sidebar
3. Select "MLFF Distiller Development"

### Automated Method (Recommended)
Add to `.github/workflows/project-automation.yml`:

```yaml
name: Project Automation

on:
  issues:
    types: [opened, labeled]
  pull_request:
    types: [opened, ready_for_review]

jobs:
  add_to_project:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/add-to-project@v0.5.0
        with:
          project-url: https://github.com/users/atfrank/projects/YOUR_PROJECT_NUMBER
          github-token: ${{ secrets.ADD_TO_PROJECT_PAT }}
```

### Bulk Add Existing Issues
```bash
# Using GitHub CLI
gh project item-add PROJECT_NUMBER --owner atfrank --url ISSUE_URL
```

## Step 5: Configure Project Settings

1. **Access**:
   - Private (team only) or Public (recommended for open source)

2. **Fields** (custom fields to add):
   - **Complexity**: Single select (Low, Medium, High, Extra High)
   - **Agent**: Single select (all 5 agents)
   - **Estimated Days**: Number field
   - **Blocked Reason**: Text field (for blocked items)

3. **Workflows**:
   - Auto-archive items in Done after 14 days
   - Move to "In Progress" when assigned
   - Move to "Review" when PR created
   - Move to "Done" when merged/closed

## Step 6: Project Board Best Practices

### For Agents
1. **Claiming Work**:
   - Move issue from "To Do" to "In Progress"
   - Assign yourself to the issue
   - Add comment stating you're starting work

2. **Updating Status**:
   - Update issue comments with progress
   - Move cards as work progresses
   - Add `status:blocked` label if blocked

3. **Completing Work**:
   - Link PR to issue
   - PR should automatically move to "Review"
   - Issue moves to "Done" when PR merged

### For Coordinator
1. **Daily Board Review**:
   - Check "In Progress" for stalled items
   - Review "Blocked Items" view
   - Ensure "Review" items get timely feedback

2. **Weekly Milestone Check**:
   - Review "By Milestone" view
   - Assess progress toward milestone goals
   - Adjust priorities as needed

3. **Issue Triage**:
   - New issues start in "Backlog"
   - Assign agent and priority labels
   - Move to "To Do" when ready

## Step 7: Dashboard Metrics (Optional)

GitHub Projects supports insights. Create these views:

### Velocity Chart
- Track issues completed per week
- Group by milestone
- Helps predict milestone completion

### Burndown Chart
- Show remaining work for current milestone
- Track daily progress
- Identify if milestone is at risk

### Agent Workload
- Count of in-progress issues per agent
- Helps balance workload
- Identify overloaded agents

## Step 8: Integration with Slack/Discord (Optional)

For real-time notifications:

```yaml
# .github/workflows/notifications.yml
name: Notifications

on:
  issues:
    types: [opened, labeled]
  pull_request_review:
    types: [submitted]

jobs:
  notify:
    runs-on: ubuntu-latest
    steps:
      - name: Send notification
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          text: 'New issue or PR needs attention'
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

## Quick Reference: Project Board URLs

After creation, bookmark these:

- **Main Board**: https://github.com/users/atfrank/projects/YOUR_PROJECT_NUMBER
- **M1 View**: https://github.com/users/atfrank/projects/YOUR_PROJECT_NUMBER/views/3
- **Blocked View**: https://github.com/users/atfrank/projects/YOUR_PROJECT_NUMBER/views/4
- **By Agent View**: https://github.com/users/atfrank/projects/YOUR_PROJECT_NUMBER/views/1

## Troubleshooting

**Issue not showing up?**
- Check if it's added to project
- Verify filters aren't hiding it
- Check if it's in "Backlog" column

**Automation not working?**
- Verify project workflows are enabled
- Check GitHub Actions permissions
- Ensure labels match exactly

**Too many issues in "In Progress"?**
- Review with agents to close or defer
- Move non-active items back to "To Do"
- Add `status:blocked` if waiting on something

---

**Next Steps**: After running `./scripts/initialize_project.sh`, follow this guide to set up the project board.
