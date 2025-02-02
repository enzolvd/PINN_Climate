#!/bin/bash

WORK_DIR="/home/ensta/ensta-louvard/projet_IA/PINN_Climate"
QUEUE_MANAGER="$WORK_DIR/experiment_queue.py"

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 experiments.json"
    exit 1
fi

# Add experiments to queue
python $QUEUE_MANAGER --action add --experiments "$1"

# Check if experiments were added successfully
STATUS=$(python $QUEUE_MANAGER --action status)
QUEUED=$(echo "$STATUS" | python -c "import sys, json; print(json.load(sys.stdin)['queued'])")

if [ "$QUEUED" -gt 0 ]; then
    echo "Successfully added experiments to queue. Setting up crontab..."
    
    # Check if our job is already in crontab
    if ! crontab -l | grep -q "$WORK_DIR/run_queue.sh"; then
        # Add to crontab if not already present
        (crontab -l 2>/dev/null; echo "0 */4 * * * $WORK_DIR/run_queue.sh") | crontab -
        echo "Added job to crontab. Will run every 4 hours."
        
        # Start the first run immediately
        $WORK_DIR/run_queue.sh
    else
        echo "Crontab entry already exists. No changes needed."
    fi
    
    echo "Setup complete! Current queue status:"
    python $QUEUE_MANAGER --action status
else
    echo "No experiments were added to the queue. Please check your experiments file."
fi