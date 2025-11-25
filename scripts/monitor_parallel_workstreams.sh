#!/bin/bash
# Monitor both parallel workstreams for M2->M3 transition
# Workstream A: 10K MolDiff generation
# Workstream B: Student architecture design

set -e

REPO_ROOT="/home/aaron/ATX/software/MLFF_Distiller"
LOG_DIR="$REPO_ROOT/logs"
MOLDIFF_LOG="$LOG_DIR/moldiff_10k_generation.log"
MONITOR_LOG="$LOG_DIR/parallel_workstreams_monitor.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Timestamp function
timestamp() {
    date "+%Y-%m-%d %H:%M:%S UTC"
}

# Log function
log_message() {
    echo "$(timestamp) - $1" | tee -a "$MONITOR_LOG"
}

echo "======================================"
echo "  Parallel Workstreams Monitor"
echo "  $(timestamp)"
echo "======================================"
echo ""

# =======================
# WORKSTREAM A: 10K Generation
# =======================
echo -e "${BLUE}[Workstream A] 10K MolDiff Generation${NC}"
echo "--------------------------------------"

# Check if generation process is running
MOLDIFF_PID=$(ps aux | grep "[m]oldiff_generate_10k.py" | awk '{print $2}' || echo "")

if [ -z "$MOLDIFF_PID" ]; then
    echo -e "${RED}Status: NOT RUNNING${NC}"
    log_message "ERROR: Workstream A - Generation process not found"
else
    echo -e "${GREEN}Status: RUNNING${NC} (PID: $MOLDIFF_PID)"
    log_message "INFO: Workstream A - Generation running (PID: $MOLDIFF_PID)"
fi

# Check progress from log file
if [ -f "$MOLDIFF_LOG" ]; then
    # Extract progress info
    TOTAL_STRUCTURES=$(grep -oP 'Generated \K\d+(?=/10000)' "$MOLDIFF_LOG" | tail -1 || echo "0")
    SUCCESS_RATE=$(grep -oP 'Success rate: \K[\d.]+' "$MOLDIFF_LOG" | tail -1 || echo "0")
    FAILURES=$(grep -c "ERROR" "$MOLDIFF_LOG" || echo "0")

    echo "Progress: $TOTAL_STRUCTURES/10,000 structures ($(echo "scale=1; $TOTAL_STRUCTURES/100" | bc)%)"
    echo "Success Rate: ${SUCCESS_RATE}%"
    echo "Failures: $FAILURES"

    # Estimate time remaining
    if [ "$TOTAL_STRUCTURES" -gt 0 ]; then
        # Get time of first and last log entries
        FIRST_TIME=$(head -20 "$MOLDIFF_LOG" | grep -m1 "Generated" | awk '{print $1, $2}' || echo "")
        LAST_TIME=$(tail -20 "$MOLDIFF_LOG" | grep "Generated" | tail -1 | awk '{print $1, $2}' || echo "")

        if [ -n "$FIRST_TIME" ] && [ -n "$LAST_TIME" ]; then
            # Calculate structures per hour (rough estimate)
            REMAINING=$((10000 - TOTAL_STRUCTURES))
            echo "Remaining: $REMAINING structures"

            # Estimate based on current progress
            if [ "$TOTAL_STRUCTURES" -ge 100 ]; then
                ETA_HOURS=$(echo "scale=1; $REMAINING * 3.3 / 10000" | bc)
                echo -e "ETA: ~${ETA_HOURS} hours"
            fi
        fi
    fi

    # Show last 5 log lines
    echo ""
    echo "Recent Activity:"
    tail -5 "$MOLDIFF_LOG" | sed 's/^/  /'

    # Check for errors
    if [ "$FAILURES" -gt 100 ]; then
        echo -e "${RED}WARNING: High failure count ($FAILURES)${NC}"
        log_message "WARNING: Workstream A - High failure count: $FAILURES"
    fi

    # Check success rate
    if (( $(echo "$SUCCESS_RATE < 95" | bc -l) )); then
        echo -e "${RED}WARNING: Success rate below 95% ($SUCCESS_RATE%)${NC}"
        log_message "WARNING: Workstream A - Success rate below target: $SUCCESS_RATE%"
    fi
else
    echo -e "${RED}Log file not found: $MOLDIFF_LOG${NC}"
    log_message "ERROR: Workstream A - Log file not found"
fi

echo ""

# =======================
# WORKSTREAM B: Architecture Design
# =======================
echo -e "${BLUE}[Workstream B] Student Architecture Design${NC}"
echo "--------------------------------------"

# Check if design spec exists
DESIGN_SPEC="$REPO_ROOT/docs/STUDENT_ARCHITECTURE_DESIGN.md"
STUDENT_MODEL="$REPO_ROOT/src/mlff_distiller/models/student_model.py"
STUDENT_TESTS="$REPO_ROOT/tests/unit/test_student_model.py"

if [ -f "$DESIGN_SPEC" ]; then
    echo -e "${GREEN}Design Spec: EXISTS${NC}"
    SPEC_SIZE=$(wc -l < "$DESIGN_SPEC")
    echo "  Lines: $SPEC_SIZE"
    echo "  Last Modified: $(stat -c %y "$DESIGN_SPEC" | cut -d' ' -f1,2)"
    log_message "INFO: Workstream B - Design spec exists ($SPEC_SIZE lines)"
else
    echo -e "${YELLOW}Design Spec: NOT YET CREATED${NC}"
    log_message "INFO: Workstream B - Design spec not yet created"
fi

if [ -f "$STUDENT_MODEL" ]; then
    echo -e "${GREEN}Student Model: EXISTS${NC}"
    MODEL_SIZE=$(wc -l < "$STUDENT_MODEL")
    echo "  Lines: $MODEL_SIZE"
    echo "  Last Modified: $(stat -c %y "$STUDENT_MODEL" | cut -d' ' -f1,2)"
    log_message "INFO: Workstream B - Student model exists ($MODEL_SIZE lines)"
else
    echo -e "${YELLOW}Student Model: NOT YET CREATED${NC}"
    log_message "INFO: Workstream B - Student model not yet created"
fi

if [ -f "$STUDENT_TESTS" ]; then
    echo -e "${GREEN}Unit Tests: EXISTS${NC}"
    TEST_SIZE=$(wc -l < "$STUDENT_TESTS")
    echo "  Lines: $TEST_SIZE"
    echo "  Last Modified: $(stat -c %y "$STUDENT_TESTS" | cut -d' ' -f1,2)"
    log_message "INFO: Workstream B - Unit tests exist ($TEST_SIZE lines)"
else
    echo -e "${YELLOW}Unit Tests: NOT YET CREATED${NC}"
    log_message "INFO: Workstream B - Unit tests not yet created"
fi

# Check GitHub issue status
echo ""
echo "GitHub Issue #19 Status:"
gh issue view 19 --json state,labels,comments --jq '{state: .state, labels: [.labels[].name], comment_count: (.comments | length)}' 2>/dev/null || echo "Could not fetch issue status"

echo ""

# =======================
# OVERALL STATUS
# =======================
echo -e "${BLUE}[Overall Status]${NC}"
echo "--------------------------------------"

# Determine completion status
WORKSTREAM_A_COMPLETE=false
WORKSTREAM_B_COMPLETE=false

if [ "$TOTAL_STRUCTURES" -ge 10000 ] && (( $(echo "$SUCCESS_RATE >= 95" | bc -l) )); then
    WORKSTREAM_A_COMPLETE=true
fi

if [ -f "$DESIGN_SPEC" ] && [ -f "$STUDENT_MODEL" ] && [ -f "$STUDENT_TESTS" ]; then
    WORKSTREAM_B_COMPLETE=true
fi

if $WORKSTREAM_A_COMPLETE && $WORKSTREAM_B_COMPLETE; then
    echo -e "${GREEN}Status: BOTH WORKSTREAMS COMPLETE ✓${NC}"
    echo "Ready to proceed to M3 training implementation!"
    log_message "SUCCESS: Both workstreams complete - Ready for M3"
elif $WORKSTREAM_A_COMPLETE; then
    echo -e "${YELLOW}Status: Workstream A complete, Workstream B in progress${NC}"
    log_message "INFO: Workstream A complete, waiting for Workstream B"
elif $WORKSTREAM_B_COMPLETE; then
    echo -e "${YELLOW}Status: Workstream B complete, Workstream A in progress${NC}"
    log_message "INFO: Workstream B complete, waiting for Workstream A"
else
    echo -e "${YELLOW}Status: Both workstreams in progress${NC}"
    log_message "INFO: Both workstreams in progress"
fi

echo ""
echo "Monitor log: $MONITOR_LOG"
echo "======================================"

# Return exit code based on any failures
if [ ! -z "$MOLDIFF_PID" ] || $WORKSTREAM_A_COMPLETE || $WORKSTREAM_B_COMPLETE; then
    exit 0
else
    exit 1
fi
