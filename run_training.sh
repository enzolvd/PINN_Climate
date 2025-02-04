# Quand vous lancer ce script il lance automatiquement un train avec les paramètres spécifiés. 
# Pour avoir un lancement du script toutes les 4h, il faut ajouter avec crontab -e
# 0 */4 * * * /home/ensta/ensta-louvard/projet_IA/PINN_Climate/run_training.sh en modifiant tous les paths

#!/bin/bash

# Directory where the script will be executed
WORK_DIR="/home/ensta/ensta-louvard/projet_IA/PINN_Climate"
LOG_FILE="$WORK_DIR/training.log"

# Log environment for debugging
env >> "$LOG_FILE"
echo "Script started with PATH: $PATH" >> "$LOG_FILE"

# Initialize conda
eval "$(/home/ensta/ensta-louvard/miniconda3/bin/conda shell.bash hook)"

# Function to check if training is complete
check_training_complete() {
    # Look for the last line in the log that might indicate completion
    if grep -q "Training completed!" "$LOG_FILE"; then
        echo "Training has completed successfully!" >> "$LOG_FILE"
        echo "Stopping cron job execution..." >> "$LOG_FILE"
        # Optionally, remove from crontab
        crontab -l | grep -v "$WORK_DIR/run_training.sh" | crontab -
        echo "Removed job from crontab" >> "$LOG_FILE"
        exit 0
    fi
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

# Improved function to check if a training job is already running
is_job_running() {
    echo "Checking for existing jobs at $(date)..." >> "$LOG_FILE"
    
    # Ensure SLURM environment
    export PATH="/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin:/usr/lib/slurm-wlm"
    
    # Log the actual command we're running
    echo "Running: squeue -u ensta-louvard" >> "$LOG_FILE"
    
    # Check if squeue command exists
    if ! command -v squeue &> /dev/null; then
        echo "Error: squeue command not found in PATH" >> "$LOG_FILE"
        return 1
    fi
    
    # Run squeue with explicit username and capture both output and exit code
    local output=$(squeue -u ensta-louvard -h 2>> "$LOG_FILE")
    local exit_code=$?
    
    # Log the results
    echo "squeue exit code: $exit_code" >> "$LOG_FILE"
    echo "squeue output: '$output'" >> "$LOG_FILE"
    
    if [ -n "$output" ]; then
        echo "Found existing job(s):" >> "$LOG_FILE"
        squeue -u ensta-louvard >> "$LOG_FILE" 2>&1
        return 0
    fi
    
    if [ $exit_code -ne 0 ]; then
        echo "squeue command failed with exit code $exit_code" >> "$LOG_FILE"
        return 1
    fi
    
    echo "No existing jobs found" >> "$LOG_FILE"
    return 1
}

# Function to handle existing jobs
handle_existing_jobs() {
    echo "Handling existing jobs..." >> "$LOG_FILE"
    
    # Get all jobs with explicit username
    local jobs=$(squeue -u ensta-louvard -h -o "%i")
    if [ -n "$jobs" ]; then
        echo "Cancelling existing jobs..." >> "$LOG_FILE"
        # Cancel all jobs for user
        scancel -u ensta-louvard
        echo "Issued cancel command for all jobs" >> "$LOG_FILE"
        
        # Wait for jobs to be fully cancelled (30 seconds max)
        local wait_time=0
        while [ $wait_time -lt 30 ]; do
            if ! is_job_running; then
                echo "All jobs successfully cancelled" >> "$LOG_FILE"
                # Wait additional 10 seconds to ensure cleanup
                sleep 10
                return 0
            fi
            sleep 1
            wait_time=$((wait_time + 1))
        done
        echo "Warning: Timeout waiting for jobs to cancel" >> "$LOG_FILE"
        return 1
    fi
}

# Function to cleanup temporary files
cleanup() {
    if [ -f temp_training_script.sh ]; then
        rm -f temp_training_script.sh
        echo "Cleaned up temporary script" >> "$LOG_FILE"
    fi
}

# Set trap for cleanup
trap cleanup EXIT

# Main execution
{
    echo "=== Starting training job at $(date) ==="
    
    # Check if training has completed
    check_training_complete
    
    # Check and handle existing jobs
    if is_job_running; then
        echo "Found existing jobs, attempting to clean up..."
        if ! handle_existing_jobs; then
            echo "Failed to clean up existing jobs. Exiting."
            exit 1
        fi
    fi

    # Change to working directory
    cd $WORK_DIR
    echo "Current directory: $(pwd)"

    # Check which partition to use
    if check_partition "ENSTA-l40s"; then
        PARTITION="ENSTA-l40s"
        echo "Using ENSTA-l40s partition"
    else
        echo "ENSTA-l40s not available, trying ENSTA-h100..."
        if check_partition "ENSTA-h100"; then
            PARTITION="ENSTA-h100"
            echo "Using ENSTA-h100 partition"
        else
            echo "No partitions available. Exiting."
            exit 1
        fi
    fi

    # Create the training script
    echo "Creating training script..."
    cat << EOF > temp_training_script.sh
#!/bin/bash

# Initialize conda in the submission script
eval "\$(/home/ensta/ensta-louvard/miniconda3/bin/conda shell.bash hook)"

# Activate the environment
conda activate projet_IA

# Run the training with minimal logging
PYTHONPATH=\$PYTHONPATH:. TQDM_DISABLE=1 python train.py \\
    --experiment_name=run_2 \\
    --wandb_project=climate_pinn \\
    --hidden_dim=64 \\
    --initial_re=100.0 \\
    --nb_years=10 \\
    --train_val_split=0.8 \\
    --batch_size=128 \\
    --epochs=100 \\
    --learning_rate=1e-3 \\
    --physics_weight=0.5 \\
    --data_weight=1.0 \\
    --checkpoint_dir=checkpoints \\
    --visual_interval=1 \\
    --data_dir=./data/era_5_data 2>&1 | grep -v "^Epoch"

# Capture the exit code
training_exit_code=\$?

# If training completed successfully, mark it in the log
if [ \$training_exit_code -eq 0 ]; then
    echo "Training completed successfully!"
fi

exit \$training_exit_code
EOF

    chmod +x temp_training_script.sh

    # Submit the job with slightly shorter duration
    echo "Submitting job to partition: $PARTITION"
    srun --time=03:55:00 --partition=$PARTITION --gpus=1 ./temp_training_script.sh
    
    # Check if job failed due to completion
    if grep -q "Training completed!" "$LOG_FILE"; then
        echo "Training has completed successfully! Removing from crontab..."
        crontab -l | grep -v "$WORK_DIR/run_training.sh" | crontab -
        echo "Removed job from crontab"
        exit 0
    fi

    echo "=== Job completed at $(date) ==="

} >> "$LOG_FILE" 2>&1
