#!/bin/bash

# MLFF Distiller Project Initialization Script
# This script sets up GitHub labels, milestones, and creates initial issues

set -e

echo "=========================================="
echo "MLFF Distiller Project Initialization"
echo "=========================================="
echo ""

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "Error: GitHub CLI (gh) is not installed."
    echo "Please install it from: https://cli.github.com/"
    exit 1
fi

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Error: Not in a git repository."
    exit 1
fi

echo "Step 1: Creating GitHub labels..."
echo "-----------------------------------"

# Agent labels
gh label create "agent:data-pipeline" --color "0E8A16" --description "Data Pipeline Engineer tasks" --force
gh label create "agent:architecture" --color "1D76DB" --description "ML Architecture Designer tasks" --force
gh label create "agent:training" --color "5319E7" --description "Distillation Training Engineer tasks" --force
gh label create "agent:cuda" --color "D93F0B" --description "CUDA Optimization Engineer tasks" --force
gh label create "agent:testing" --color "0052CC" --description "Testing & Benchmark Engineer tasks" --force

# Milestone labels
gh label create "milestone:M1" --color "C2E0C6" --description "M1: Setup & Baseline" --force
gh label create "milestone:M2" --color "BFDADC" --description "M2: Data Pipeline" --force
gh label create "milestone:M3" --color "C5DEF5" --description "M3: Model Architecture" --force
gh label create "milestone:M4" --color "D4C5F9" --description "M4: Distillation Training" --force
gh label create "milestone:M5" --color "F9D0C4" --description "M5: CUDA Optimization" --force
gh label create "milestone:M6" --color "FEF2C0" --description "M6: Testing & Deployment" --force

# Type labels
gh label create "type:task" --color "0075CA" --description "Task or feature work" --force
gh label create "type:bug" --color "D73A4A" --description "Bug report" --force
gh label create "type:feature" --color "A2EEEF" --description "New feature" --force
gh label create "type:research" --color "7057FF" --description "Research or investigation" --force
gh label create "type:refactor" --color "FBCA04" --description "Code refactoring" --force
gh label create "type:docs" --color "0E8A16" --description "Documentation" --force

# Priority labels
gh label create "priority:critical" --color "B60205" --description "Critical priority" --force
gh label create "priority:high" --color "D93F0B" --description "High priority" --force
gh label create "priority:medium" --color "FBCA04" --description "Medium priority" --force
gh label create "priority:low" --color "0E8A16" --description "Low priority" --force

# Status labels
gh label create "status:blocked" --color "D73A4A" --description "Blocked on dependencies" --force
gh label create "status:in-progress" --color "0075CA" --description "Currently in progress" --force
gh label create "status:needs-review" --color "FBCA04" --description "Needs review" --force
gh label create "status:needs-decision" --color "7057FF" --description "Needs coordinator decision" --force

echo "✓ Labels created successfully"
echo ""

echo "Step 2: Creating GitHub milestones..."
echo "--------------------------------------"

gh api repos/:owner/:repo/milestones -f title="M1: Setup & Baseline" -f state="open" -f description="Repository setup, teacher model integration, baseline benchmarks" -f due_on="2025-12-07T00:00:00Z" || echo "Milestone M1 may already exist"

gh api repos/:owner/:repo/milestones -f title="M2: Data Pipeline" -f state="open" -f description="Data generation, preprocessing, dataset management" -f due_on="2025-12-21T00:00:00Z" || echo "Milestone M2 may already exist"

gh api repos/:owner/:repo/milestones -f title="M3: Model Architecture" -f state="open" -f description="Student model design and implementation" -f due_on="2026-01-04T00:00:00Z" || echo "Milestone M3 may already exist"

gh api repos/:owner/:repo/milestones -f title="M4: Distillation Training" -f state="open" -f description="Training pipeline and loss functions" -f due_on="2026-01-25T00:00:00Z" || echo "Milestone M4 may already exist"

gh api repos/:owner/:repo/milestones -f title="M5: CUDA Optimization" -f state="open" -f description="Performance optimization and CUDA kernels" -f due_on="2026-02-15T00:00:00Z" || echo "Milestone M5 may already exist"

gh api repos/:owner/:repo/milestones -f title="M6: Testing & Deployment" -f state="open" -f description="Comprehensive testing and release preparation" -f due_on="2026-03-01T00:00:00Z" || echo "Milestone M6 may already exist"

echo "✓ Milestones created successfully"
echo ""

echo "Step 3: Creating initial issues..."
echo "-----------------------------------"

# Get milestone numbers (GitHub CLI should handle this automatically with milestone names)

# Priority 1 Issues - M1 Setup
echo "Creating Priority 1 issues (M1 Setup)..."

gh issue create --title "[Data Pipeline] [M1] Set up data loading infrastructure" \
  --body-file docs/initial_issues/issue_01_data_infrastructure.md \
  --label "agent:data-pipeline,milestone:M1,type:task,priority:high" \
  --milestone "M1: Setup & Baseline" || echo "Issue may already exist"

gh issue create --title "[Architecture] [M1] Create teacher model wrapper interfaces" \
  --body-file docs/initial_issues/issue_06_teacher_wrappers.md \
  --label "agent:architecture,milestone:M1,type:task,priority:high" \
  --milestone "M1: Setup & Baseline" || echo "Issue may already exist"

gh issue create --title "[Training] [M1] Set up baseline training framework" \
  --body-file docs/initial_issues/issue_11_training_framework.md \
  --label "agent:training,milestone:M1,type:task,priority:high" \
  --milestone "M1: Setup & Baseline" || echo "Issue may already exist"

gh issue create --title "[Testing] [M1] Configure pytest and test infrastructure" \
  --body-file docs/initial_issues/issue_21_pytest_setup.md \
  --label "agent:testing,milestone:M1,type:task,priority:high" \
  --milestone "M1: Setup & Baseline" || echo "Issue may already exist"

echo "✓ Priority 1 issues created"
echo ""

echo "=========================================="
echo "Initialization Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Visit https://github.com/$(gh repo view --json nameWithOwner -q .nameWithOwner) to view issues"
echo "2. Create a GitHub Project board at: https://github.com/$(gh repo view --json nameWithOwner -q .nameWithOwner)/projects"
echo "3. Add issues to the project board"
echo "4. Review docs/initial_issues/ISSUES_PLAN.md for additional issues to create"
echo "5. Agents can start claiming issues and working!"
echo ""
echo "Documentation:"
echo "- Project Overview: README.md"
echo "- Contributing Guide: CONTRIBUTING.md"
echo "- Milestones: docs/MILESTONES.md"
echo "- Agent Protocols: docs/AGENT_PROTOCOLS.md"
echo "- Issue Plan: docs/initial_issues/ISSUES_PLAN.md"
echo ""
