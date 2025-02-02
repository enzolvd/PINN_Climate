#!/bin/bash

# Directory where the script will be executed
WORK_DIR="/home/ensta/ensta-louvard/projet_IA/PINN_Climate"
LOG_FILE="$WORK_DIR/training.log"
QUEUE_MANAGER="$WORK_DIR/experiment_queue.py"

# Initialize conda
eval "$(/home/ensta/ensta-louvard/miniconda3/bin/conda shell.bash hook)"
conda activate projet_IA

# Function to get next experiment from queue and validate JSON
get_next_experiment() {
    echo "Fetching next experiment from queue..."
    local exp_json
    exp_json=$(python $QUEUE_MANAGER --action get_next)
    local get_next_status=$?
    
    if [ $get_next_status -ne 0 ]; then
        echo "Error: get_next command failed with status $get_next_status"
        echo "Command output: $exp_json"
        return 1
    fi
    
    if [ -z "$exp_json" ]; then
        echo "Queue is empty (no experiments returned)"
        return 0
    fi
    
    # Validate JSON using Python with detailed error reporting
    echo "Validating experiment JSON..."
    validation_result=$(echo "$exp_json" | python3 -c '
import json
import sys
try:
    data = json.load(sys.stdin)
    if not isinstance(data, dict):
        print("Error: Expected JSON object, got " + str(type(data)))
        sys.exit(1)
    print("valid")
except json.JSONDecodeError as e:
    print("JSON parsing error: " + str(e))
    sys.exit(1)
except Exception as e:
    print("Unexpected error: " + str(e))
    sys.exit(1)
' 2>&1)
    
    if [ $? -ne 0 ] || [ "$validation_result" != "valid" ]; then
        echo "JSON validation failed: $validation_result"
        echo "Raw JSON: $exp_json"
        return 1
    fi
    
    echo "$exp_json"
    return 0
}

# Function to check if a partition is available
check_partition() {
    local partition=$1
    echo "Checking partition status for: $partition" >> "$LOG_FILE"
    
    sinfo -p $partition -h -o "%a %T" | grep -q "up idle"
    local status=$?
    
    if [ $status -eq 0 ]; then
        echo "Partition $partition is available and has idle nodes" >> "$LOG_FILE"
        return 0
    else
        echo "Partition $partition is not available or has no idle nodes" >> "$LOG_FILE"
        return 1
    fi
}

# Main execution
{
    echo "=== Starting queue processing at $(date) ==="
    
    # Cancel all existing jobs for the user
    echo "Cancelling any existing jobs for user ensta-louvard"
    scancel -u ensta-louvard
    # Wait a moment for the jobs to be fully cancelled
    sleep 5
    
    # Get queue status before proceeding
    echo "Checking queue status..."
    python $QUEUE_MANAGER --action status
    
    # Get next experiment from queue with validation
    echo "Attempting to get next experiment..."
    EXPERIMENT=$(get_next_experiment)
    get_next_status=$?
    
    if [ $get_next_status -ne 0 ]; then
        echo "Error: Failed to get valid experiment data. Exiting."
        echo "Current queue state:"
        python $QUEUE_MANAGER --action status
        exit 1
    fi
    
    if [ -z "$EXPERIMENT" ]; then
        echo "No experiments in queue. Removing from crontab..."
        echo "Final queue status:"
        python $QUEUE_MANAGER --action status
        crontab -l | grep -v "$WORK_DIR/run_queue.sh" | crontab -
        echo "Removed from crontab. Exiting."
        exit 0
    fi
    
    echo "Starting experiment with parameters:"
    echo "$EXPERIMENT" | python3 -m json.tool
    
    # Change to working directory
    cd $WORK_DIR
    
    # Check which partition to use
    if check_partition "ENSTA-l40s"; then
        PARTITION="ENSTA-l40s"
    else
        if check_partition "ENSTA-h100"; then
            PARTITION="ENSTA-h100"
        else
            echo "No partitions available. Exiting."
            exit 1
        fi
    fi

    # Save experiment to a file that will be read by the job
    echo "$EXPERIMENT" > experiment.json
    
    # Create the training script with proper conda initialization and partition
    cat << EOF > temp_training_script.sh
#!/bin/bash
#SBATCH --job-name=pinn_train
#SBATCH --time=03:55:00
#SBATCH --gpus=1
#SBATCH --partition=$PARTITION

# Change to work directory
cd /home/ensta/ensta-louvard/projet_IA/PINN_Climate

# Initialize conda properly in a way that works with SLURM
__conda_setup="\$('/home/ensta/ensta-louvard/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ \$? -eq 0 ]; then
    eval "\$__conda_setup"
else
    if [ -f "/home/ensta/ensta-louvard/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/ensta/ensta-louvard/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/ensta/ensta-louvard/miniconda3/bin:\$PATH"
    fi
fi
unset __conda_setup

# Activate environment
conda activate projet_IA

# Build command from experiment parameters from the file
CMD=\$(python3 -c '
import json
import sys

try:
    with open("experiment.json", "r") as f:
        params = json.load(f)
    cmd_parts = []
    for k, v in params.items():
        if isinstance(v, bool) and v:
            cmd_parts.append(f"--{k}")
        elif v is not None:
            cmd_parts.append(f"--{k}={v}")
    print(" ".join(cmd_parts))
except Exception as e:
    print(f"Error processing parameters: {e}", file=sys.stderr)
    sys.exit(1)
')

if [ \$? -ne 0 ]; then
    echo "Error processing experiment parameters"
    exit 1
fi

# Add fixed parameters
CMD="\$CMD --checkpoint_dir=checkpoints --visual_interval=1"

echo "Running command: python train.py \$CMD"
python train.py \$CMD

# Capture exit code
training_exit_code=\$?

if [ \$training_exit_code -eq 0 ]; then
    echo "Training completed successfully!"
    python /home/ensta/ensta-louvard/projet_IA/PINN_Climate/experiment_queue.py --action mark_completed
    rm -f experiment.json
fi

exit \$training_exit_code
EOF

    chmod +x temp_training_script.sh
    
    # Submit the job using sbatch
    echo "Submitting job to partition: $PARTITION"
    sbatch temp_training_script.sh
    
    # Cleanup
    rm -f temp_training_script.sh
    
    echo "=== Queue processing completed at $(date) ==="

} >> "$LOG_FILE" 2>&1