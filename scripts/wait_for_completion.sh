#!/bin/bash
# Wait for validation to complete

for i in {1..20}; do
    COUNT=$(ls -1 validation_results/original_model/trajectories/*.traj 2>/dev/null | wc -l)
    echo "Check $i: $COUNT/5 molecules complete at $(date +%H:%M:%S)"

    if [ $COUNT -ge 5 ]; then
        echo "ALL 5 MOLECULES COMPLETE!"
        exit 0
    fi

    # Check if process still running
    if ! ps aux | grep -q "[v]alidate_original_model_md.py"; then
        echo "Process no longer running. Final count: $COUNT/5"
        exit 1
    fi

    sleep 120
done

echo "Timeout after 40 minutes. Final count: $COUNT/5"
exit 2
