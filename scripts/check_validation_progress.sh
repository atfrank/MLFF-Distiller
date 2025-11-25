#!/bin/bash
# Quick status check for Original Model MD validation

echo "=== ORIGINAL MODEL MD VALIDATION STATUS ==="
echo ""
echo "Date: $(date)"
echo ""

# Check if process is running
if ps aux | grep -q "[v]alidate_original_model_md.py"; then
    echo "Status: RUNNING"
    echo ""

    # GPU usage
    echo "GPU Usage:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | grep python || echo "  No Python process on GPU"
    echo ""

    # Check log file
    LOG_FILE="validation_results/original_model/validation_log.txt"
    if [ -f "$LOG_FILE" ] && [ -s "$LOG_FILE" ]; then
        echo "Latest log output:"
        tail -30 "$LOG_FILE"
    else
        echo "Log file not yet populated or empty"
    fi

else
    echo "Status: NOT RUNNING or COMPLETED"
    echo ""

    # Check for results
    if [ -f "validation_results/original_model/original_model_md_report.md" ]; then
        echo "Validation report found!"
        echo ""
        echo "Summary from report:"
        head -50 "validation_results/original_model/original_model_md_report.md"
    elif [ -f "validation_results/original_model/validation_log.txt" ]; then
        echo "Log file exists. Last 50 lines:"
        tail -50 "validation_results/original_model/validation_log.txt"
    fi
fi

echo ""
echo "=== END STATUS ==="
