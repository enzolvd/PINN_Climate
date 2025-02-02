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

# Function to mark current experiment as completed
mark_experiment_completed() {
    python $QUEUE_MANAGER --action mark_completed
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

# Function to handle existing jobs
handle_existing_jobs() {
    echo "Handling existing jobs..." >> "$LOG_FILE"
    local jobs=$(squeue -u ensta-louvard -h -o "%i")
    if [ -n "$jobs" ]; then
        echo "Cancelling existing jobs..." >> "$LOG_FILE"
        scancel -u ensta-louvard
        sleep 10
    fi
}

# Main execution
{
    echo "=== Starting queue processing at $(date) ==="
    
    # Get next experiment from queue
    EXPERIMENT=$(get_next_experiment)
    
    if [ -z "$EXPERIMENT" ]; then
        echo "No experiments in queue. Removing from crontab..."
        crontab -l | grep -v "$WORK_DIR/run_queue.sh" | crontab -
        echo "Removed from crontab. Exiting."
        exit 0
    fi
    
    echo "Starting experiment: $EXPERIMENT"
    
    # Handle any existing jobs
    handle_existing_jobs
    
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
    
    # Create the training script from experiment parameters
    echo "Creating training script..."
    cat << 'EOF' > temp_training_script.sh
#!/bin/bash

eval "$(/home/ensta/ensta-louvard/miniconda3/bin/conda shell.bash hook)"
conda activate projet_IA

# Parse experiment parameters from environment variable
PARAMS=$(echo $EXPERIMENT_PARAMS | python3 -c '
import sys, json, os
params = json.loads(os.environ["EXPERIMENT_PARAMS"])
args = []
for k, v in params.items():
    if isinstance(v, bool) and v:
        args.append(f"--{k}")
    else:
        args.append(f"--{k}={v}")
print(" ".join(args))
')

PYTHONPATH=$PYTHONPATH:. python train.py $PARAMS --checkpoint_dir=checkpoints --visual_interval=1 --use_progress_bar

# Capture exit code
training_exit_code=$?

if [ $training_exit_code -eq 0 ]; then
    echo "Training completed successfully!"
fi

exit $training_exit_code
EOF

    chmod +x temp_training_script.sh
    
    # Submit the job with experiment parameters as environment variable
    echo "Submitting job to partition: $PARTITION"
    JOB_ID=$(EXPERIMENT_PARAMS="$EXPERIMENT" sbatch --parsable --partition=$PARTITION --time=03:55:00 --gpus=1 --export=ALL,EXPERIMENT_PARAMS="$EXPERIMENT" ./temp_training_script.sh)
    
    if [ -n "$JOB_ID" ]; then
        echo "Submitted job $JOB_ID"
        
        # Create a completion handler script
        cat << 'EOH' > completion_handler.sh
#!/bin/bash
if [ "$SLURM_JOB_ID" ]; then
    # Mark the current experiment as completed
    $WORK_DIR/experiment_queue.py --action mark_completed
    
    # Submit the next job from the queue
    $WORK_DIR/run_queue.sh
fi
EOH
        chmod +x completion_handler.sh
        
        # Submit the completion handler as a dependent job
        HANDLER_ID=$(sbatch --parsable --partition=$PARTITION --dependency=afterany:$JOB_ID completion_handler.sh)
        echo "Submitted completion handler job $HANDLER_ID"
    else
        echo "Failed to submit job!"
    fi
    
    # Cleanup
    rm -f temp_training_script.sh completion_handler.sh
    
    echo "=== Queue processing completed at $(date) ==="

} >> "$LOG_FILE" 2>&1