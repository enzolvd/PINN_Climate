#!/bin/bash

# Directory where the script will be executed
WORK_DIR="/home/ensta/ensta-louvard/projet_IA/PINN_Climate"
LOG_FILE="$WORK_DIR/training.log"
QUEUE_MANAGER="$WORK_DIR/experiment_queue.py"

# Main execution
{
    echo "=== Starting queue processing at $(date) ==="
    
    # Change to working directory first thing
    echo "Changing to working directory: $WORK_DIR"
    cd "$WORK_DIR" || {
        echo "Failed to change to working directory"
        exit 1
    }
    pwd

    # Cancel all existing jobs for the user
    echo "Cancelling any existing jobs for user ensta-louvard"
    scancel -u ensta-louvard
    # Wait a moment for the jobs to be fully cancelled
    sleep 5
    
    # Check queue status before proceeding
    echo "Checking queue status..."
    python3 "$QUEUE_MANAGER" --action status
    
    # Get next experiment from queue
    echo "Attempting to get next experiment..."
    EXPERIMENT=$(python3 "$QUEUE_MANAGER" --action get_next)
    if [ $? -ne 0 ]; then
        echo "Error: Failed to get experiment from queue manager. Exiting."
        exit 1
    fi
    
    if [ -z "$EXPERIMENT" ]; then
        echo "No experiments in queue. Removing from crontab..."
        crontab -l | grep -v "$WORK_DIR/run_queue.sh" | crontab -
        echo "Removed from crontab. Exiting."
        exit 0
    fi
    
    # Pretty print the experiment parameters for the log
    echo "Starting experiment with parameters:"
    echo "$EXPERIMENT" | python3 -m json.tool
    
    # Check which partition to use
    if sinfo -p ENSTA-l40s -h -o "%a %T" | grep -q "up idle"; then
        PARTITION="ENSTA-l40s"
    elif sinfo -p ENSTA-h100 -h -o "%a %T" | grep -q "up idle"; then
        PARTITION="ENSTA-h100"
    else
        echo "No partitions available. Exiting."
        exit 1
    fi
    echo "Using partition: $PARTITION"

    # Save experiment to a file that will be read by the job
    echo "$EXPERIMENT" > experiment.json
    
    # Create the training script with proper conda initialization and partition
    cat << EOF > temp_training_script.sh
#!/bin/bash
#SBATCH --job-name=pinn_train
#SBATCH --time=03:59:00
#SBATCH --gpus=1
#SBATCH --partition=$PARTITION

# Change to work directory
cd "$WORK_DIR"

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
    python3 "$QUEUE_MANAGER" --action mark_completed
fi

# Always cleanup experiment.json at the end
rm -f experiment.json

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