#!/bin/bash
WORK_DIR="./experiment_runner"
QUEUE_MANAGER="./experiment_runner/experiment_queue.py"
MASTER_FILE="./experiment_runner/experiments.json"

# Get current status
STATUS=$(python $QUEUE_MANAGER --action status)

# Get all experiments name from master file
ALL_EXPERIMENTS=$(cat "$MASTER_FILE" | python -c "
import sys, json
data = json.load(sys.stdin)
latest_experiment = max(data, key=lambda x: int(x['experiment_name'].split('_')[1]))
print(latest_experiment['experiment_name'])
")

echo "Latest experiment: $ALL_EXPERIMENTS"

python $QUEUE_MANAGER --action add --experiments "$MASTER_FILE"

NEW_STATUS=$(python $QUEUE_MANAGER --action status)
NEW_QUEUED=$(echo "$NEW_STATUS" | python -c "import sys, json; print(json.load(sys.stdin)['queued'])")

if [ "$NEW_QUEUED" -gt 0 ]; then
    if ! crontab -l | grep -q "$PWD/run_queue.sh"; then
        (crontab -l 2>/dev/null; echo "0 */4 * * * $PWD/run_queue.sh") | crontab -
        echo "Added crontab job. Running every 4 hours."
        ./run_queue.sh
    fi
    python $QUEUE_MANAGER --action status
else
    echo "No new experiments added."
fi