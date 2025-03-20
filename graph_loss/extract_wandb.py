import wandb
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Set your wandb API key
# os.environ["WANDB_API_KEY"] = "YOUR_API_KEY_HERE"

# Initialize wandb and connect to your project
api = wandb.Api()
entity = "enzo-louvard-ensta-paris"  # Your wandb username or organization
project = "climate_pinn"  # Your project name

# Get runs data
runs = api.runs(f"{entity}/{project}")

# Extract the metrics you need for visualization
model_run = {
    "run_1": "model_0",
    "run_3": "model_1",
    "run_4": "model_0_Re",
    "run_8": "model_2",
    "run_9": "model_3"
}

MAX_STEPS = 25000
SAMPLING_RATE = 10
model_data = {}

for run in runs:
    run_name = run.name
    if run_name in model_run.keys():
        print(f"Fetching history for {run_name}...")
        # Get the history metrics for data loss
        try:
            history = run.history(samples=MAX_STEPS//SAMPLING_RATE, keys=["batch/data_loss", "_step"])
            
            # Convert to list of [step, value] pairs
            steps = history["_step"].tolist()
            data_loss = history["batch/data_loss"].tolist()
            
            # Create pairs of [step, value]
            step_loss_pairs = [[steps[i], data_loss[i]] for i in range(len(steps))]
            
            # Store in model_data dictionary
            model_data[model_run[run_name]] = {
                "data_loss": step_loss_pairs
            }
            print(f" Got {len(step_loss_pairs)} data points for {run_name}")
            
        except Exception as e:
            print(f"Error fetching history for {run_name}: {e}")

# Save the data to a JSON file
with open('wandb_data.json', 'w') as f:
    json.dump(model_data, f)

print("Data saved to wandb_data.json")

# Create a simple plot to verify the data
plt.figure(figsize=(10, 6))
for model_name, model_values in model_data.items():
    data_points = np.array(model_values["data_loss"])
    plt.semilogy(data_points[:, 0], data_points[:, 1], label=model_name)

plt.xlabel("Steps")
plt.ylabel("Data Loss")
plt.title("Training Data Loss")
plt.legend()
plt.show()