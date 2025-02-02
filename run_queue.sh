#!/bin/bash

# Directory where the script will be executed
WORK_DIR="/home/ensta/ensta-louvard/projet_IA/PINN_Climate"
LOG_FILE="$WORK_DIR/training.log"
QUEUE_MANAGER="$WORK_DIR/experiment_queue.py"

# Initialize conda
eval "$(/home/ensta/ensta-louvard/miniconda3/bin/conda shell.bash hook)"
conda activate projet_IA

# Function to get next experiment from queue
get_next_experiment() {
    python $QUEUE_MANAGER --action get_next
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
    
    # Check if there's already a job running
    if [ $(squeue -u ensta-louvard -h | wc -l) -gt 0 ]; then
        echo "Job already running. Exiting."
        exit 0
    fi
    
    # Get next experiment from queue
    EXPERIMENT=$(get_next_experiment)
    
    if [ -z "$EXPERIMENT" ]; then
        echo "No experiments in queue. Removing from crontab..."
        crontab -l | grep -v "$WORK_DIR/run_queue.sh" | crontab -
        echo "Removed from crontab. Exiting."
        exit 0
    fi
    
    echo "Starting experiment: $EXPERIMENT"
    
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
    
    # Create the training script
    cat << EOF > temp_training_script.sh
#!/bin/bash

# Change to work directory
cd $WORK_DIR

# Initialize conda properly
export CONDA_ROOT=/home/ensta/ensta-louvard/miniconda3
source \$CONDA_ROOT/etc/profile.d/conda.sh
conda activate projet_IA

# Build command from experiment parameters
CMD=\$(echo '$EXPERIMENT' | python3 -c '
import json
import sys
params = json.loads(sys.stdin.read())
cmd_parts = []
for k, v in params.items():
    if isinstance(v, bool) and v:
        cmd_parts.append(f"--{k}")
    else:
        cmd_parts.append(f"--{k}={v}")
print(" ".join(cmd_parts))
')

# Add fixed parameters
CMD="\$CMD --checkpoint_dir=checkpoints --visual_interval=1 --use_progress_bar"

echo "Running command: python train.py \$CMD"
PYTHONPATH=\$PYTHONPATH:. python train.py \$CMD

# Capture exit code
training_exit_code=\$?

if [ \$training_exit_code -eq 0 ]; then
    echo "Training completed successfully!"
    $QUEUE_MANAGER --action mark_completed
fi

exit \$training_exit_code
EOF

    chmod +x temp_training_script.sh
    
    # Submit the job
    echo "Submitting job to partition: $PARTITION"
    srun --partition=$PARTITION --time=03:55:00 --gpus=1 ./temp_training_script.sh
    
    # Cleanup
    rm -f temp_training_script.sh
    
    echo "=== Queue processing completed at $(date) ==="

} >> "$LOG_FILE" 2>&1