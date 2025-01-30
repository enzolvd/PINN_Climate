#!/bin/bash

# Directory where the script will be executed
WORK_DIR="/home/ensta/ensta-louvard/projet_IA/PINN_Climate"
LOG_FILE="$WORK_DIR/training.log"

echo "============================================" >> "$LOG_FILE" 2>&1
echo "Starting script at $(date)" >> "$LOG_FILE" 2>&1
echo "Current user: $USER" >> "$LOG_FILE" 2>&1
echo "Current directory before cd: $(pwd)" >> "$LOG_FILE" 2>&1

# Initialize conda
echo "Initializing conda..." >> "$LOG_FILE" 2>&1
eval "$(/home/ensta/ensta-louvard/miniconda3/bin/conda shell.bash hook)"
if [ $? -eq 0 ]; then
    echo "Conda initialization successful" >> "$LOG_FILE" 2>&1
else
    echo "Error: Conda initialization failed" >> "$LOG_FILE" 2>&1
    exit 1
fi

# Function to check if a partition is available
check_partition() {
    local partition=$1
    echo "Checking partition availability for: $partition" >> "$LOG_FILE" 2>&1
    sinfo -p $partition -h -o "%A" | grep -q "up"
    local result=$?
    if [ $result -eq 0 ]; then
        echo "Partition $partition is available" >> "$LOG_FILE" 2>&1
    else
        echo "Partition $partition is not available" >> "$LOG_FILE" 2>&1
    fi
    return $result
}

# Function to check if a training job is already running
is_job_running() {
    echo "Checking for running jobs..." >> "$LOG_FILE" 2>&1
    squeue -u $USER >> "$LOG_FILE" 2>&1
    squeue -u $USER | grep -q "$USER"
    local result=$?
    if [ $result -eq 0 ]; then
        echo "Found existing job" >> "$LOG_FILE" 2>&1
    else
        echo "No existing job found" >> "$LOG_FILE" 2>&1
    fi
    return $result
}

# Main execution
{
    echo "=== Starting training job at $(date) ==="
    
    # Check if a job is already running
    if is_job_running; then
        echo "A job is already running. Exiting."
        exit 0
    fi

    # Change to working directory
    echo "Changing to working directory: $WORK_DIR"
    cd $WORK_DIR
    echo "Current directory after cd: $(pwd)"

    # Check which partition to use
    if check_partition "ENSTA-l40s"; then
        PARTITION="ENSTA-l40s"
    else
        PARTITION="ENSTA-h100"
    fi
    echo "Selected partition: $PARTITION"

    # Create the training script
    echo "Creating temporary training script..."
    cat << EOF > temp_training_script.sh
#!/bin/bash

echo "=== Starting training script at \$(date) ===" >> "$LOG_FILE" 2>&1
echo "Running as user: \$USER" >> "$LOG_FILE" 2>&1
echo "In directory: \$(pwd)" >> "$LOG_FILE" 2>&1

# Initialize conda in the submission script
echo "Initializing conda in training script..." >> "$LOG_FILE" 2>&1
eval "\$(/home/ensta/ensta-louvard/miniconda3/bin/conda shell.bash hook)"
if [ \$? -eq 0 ]; then
    echo "Conda initialization in training script successful" >> "$LOG_FILE" 2>&1
else
    echo "Error: Conda initialization in training script failed" >> "$LOG_FILE" 2>&1
    exit 1
fi

# Activate the environment
echo "Activating conda environment..." >> "$LOG_FILE" 2>&1
conda activate projet_IA
if [ \$? -eq 0 ]; then
    echo "Environment activation successful" >> "$LOG_FILE" 2>&1
else
    echo "Error: Environment activation failed" >> "$LOG_FILE" 2>&1
    exit 1
fi

echo "Starting Python training script..." >> "$LOG_FILE" 2>&1
python train.py \
    --experiment_name=run_1 \
    --wandb_project=climate_pinn \
    --hidden_dim=32 \
    --initial_re=100.0 \
    --nb_years=10 \
    --train_val_split=0.8 \
    --batch_size=32 \
    --epochs=100 \
    --learning_rate=1e-3 \
    --physics_weight=1.0 \
    --data_weight=1.0 \
    --checkpoint_dir=checkpoints \
    --visual_interval=1 \
    --data_dir=./data/era_5_data >> "$LOG_FILE" 2>&1

echo "Python script completed with status: \$?" >> "$LOG_FILE" 2>&1
EOF

    chmod +x temp_training_script.sh
    echo "Created temporary script with permissions:"
    ls -l temp_training_script.sh

    # Submit the job
    echo "Submitting job to partition: $PARTITION"
    srun --time=04:00:00 --partition=$PARTITION --gpus=1 ./temp_training_script.sh
    SRUN_STATUS=$?
    echo "srun completed with status: $SRUN_STATUS"

    # Clean up
    echo "Cleaning up temporary script..."
    rm temp_training_script.sh
    echo "Cleanup completed"

    echo "=== Job completed at $(date) ==="

} >> "$LOG_FILE" 2>&1