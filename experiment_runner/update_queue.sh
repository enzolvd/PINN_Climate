#!/bin/bash

# Define the root directory (assuming the script is run from the project root)
ROOT_DIR="$(pwd)"
QUEUE_DIR="$ROOT_DIR/experiment_queue"
QUEUE_MANAGER="$ROOT_DIR/experiment_runner/experiment_queue.py"
MASTER_FILE="$ROOT_DIR/experiment_runner/experiments.json"

# Get current status
STATUS=$(python $QUEUE_MANAGER --action status --queue_dir "$QUEUE_DIR")

# Get all experiments name from master file
ALL_EXPERIMENTS=$(cat "$MASTER_FILE" | python -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if data:
        latest_experiment = max(data, key=lambda x: int(x['experiment_name'].split('_')[1]))
        print(latest_experiment['experiment_name'])
    else:
        print('No experiments found')
except Exception as e:
    print(f'Error parsing experiments: {e}')
")
echo "Latest experiment: $ALL_EXPERIMENTS"

# Add experiments from master file to queue
python $QUEUE_MANAGER --action add --experiments "$MASTER_FILE" --queue_dir "$QUEUE_DIR"

# Check new status
NEW_STATUS=$(python $QUEUE_MANAGER --action status --queue_dir "$QUEUE_DIR")
NEW_QUEUED=$(echo "$NEW_STATUS" | python -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('queued', 0))
except Exception as e:
    print(0)
")

if [ -n "$NEW_QUEUED" ] && [ "$NEW_QUEUED" -gt 0 ]; then
    if ! crontab -l | grep -q "$ROOT_DIR/experiment_runner/run_queue.sh"; then
        (crontab -l 2>/dev/null; echo "0 */4 * * * $ROOT_DIR/experiment_runner/run_queue.sh") | crontab -
        echo "Added crontab job. Running every 4 hours."
        "$ROOT_DIR/experiment_runner/run_queue.sh"
    fi
    python $QUEUE_MANAGER --action status --queue_dir "$QUEUE_DIR"
else
    echo "No experiments queued."
fi