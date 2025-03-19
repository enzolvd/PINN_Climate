import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tqdm import tqdm
from torch.utils.data import DataLoader
import json
from pathlib import Path
import os
import sys

# Import from the project modules
from dataset import load_dataset, ERADataset
from video_gen import load_checkpoint, get_animation_writer, denormalize_variable

# Enable anti-aliasing and set backend
mpl.rcParams['text.antialiased'] = True
mpl.rcParams['lines.antialiased'] = True
mpl.use('Agg')

def transform_longitude(arr):
    # For longitude 64 points, split at index 32
    second_half = arr[:, :, :, :32]   
    first_half = arr[:, :, :, 32:]     
    return np.concatenate([first_half, second_half], axis=-1)

def generate_mc_dropout_predictions(model, dataloader, device, num_samples=20, duration=10):
    """
    Generate multiple predictions using MC dropout for uncertainty estimation.
    
    Args:
        model: The PINN model
        dataloader: DataLoader for the validation set
        device: Device to run the model on
        num_samples: Number of MC samples to generate
        duration: Number of timesteps to predict
        
    Returns:
        all_preds: List of all predictions [num_samples, timesteps, features, height, width]
        targets: Ground truth values
        norm_params: Normalization parameters
    """
    # Set model to evaluation mode but keep dropout active
    model.train()  # This keeps dropout active
    
    # Store predictions for each sample
    all_predictions = []
    targets = None
    norm_params = None
    
    # Process for the specified duration
    batch_data = []
    for i, batch in enumerate(dataloader):
        if i >= duration:
            break
        batch_data.append(batch)
    
    # Get normalization parameters from first batch
    if batch_data:
        norm_params = batch_data[0]['norm_params']
    
    # For each MC sample
    print(f"Generating {num_samples} MC dropout samples...")
    for sample in tqdm(range(num_samples)):
        sample_predictions = []
        
        # For each timestep
        for batch in batch_data:
            # Move data to device
            inputs = batch['input'].to(device)
            batch_targets = batch['target'].to(device)
            masks = batch['masks'].to(device)
            coords = [coord.to(device) for coord in batch['coords']]

            # Store targets only once
            if targets is None:
                targets = [batch_targets.cpu()]
            
            # Forward pass with dropout active (model.train() keeps dropout active)
            with torch.no_grad():  # Still no grad needed for inference
                outputs = model(inputs, masks, coords, compute_physics=False)['output']
                
            # Add to predictions
            sample_predictions.append(outputs.cpu())
            
            # Clean up GPU memory
            del outputs, inputs, batch_targets, masks, coords
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Combine predictions for this sample
        sample_predictions = torch.cat(sample_predictions, dim=0)
        all_predictions.append(sample_predictions)
    
    # Combine targets
    if targets:
        targets = torch.cat(targets, dim=0).numpy()
    
    # Convert predictions to numpy arrays
    all_predictions = [p.numpy() for p in all_predictions]
    
    return all_predictions, targets, norm_params

def calculate_uncertainty_metrics(predictions):
    """
    Calculate uncertainty metrics from MC dropout predictions.
    
    Args:
        predictions: List of predictions from MC dropout [num_samples, timesteps, features, height, width]
        
    Returns:
        mean: Mean prediction
        std: Standard deviation of predictions (uncertainty)
        lower_ci: Lower confidence interval (mean - 2*std)
        upper_ci: Upper confidence interval (mean + 2*std)
    """
    # Stack predictions along a new axis
    stacked_preds = np.stack(predictions)
    
    # Calculate mean and standard deviation
    mean = np.mean(stacked_preds, axis=0)
    std = np.std(stacked_preds, axis=0)
    
    # Calculate 95% confidence intervals (approximately mean ± 2*std)
    lower_ci = mean - 2 * std
    upper_ci = mean + 2 * std
    
    return mean, std, lower_ci, upper_ci

def plot_uncertainty_map(mean, std, lat, lon, title, save_path, timestep, var_name):
    """
    Create a static plot showing mean prediction and uncertainty.
    
    Args:
        mean: Mean prediction for the given timestep
        std: Standard deviation for the given timestep
        lat, lon: Coordinate arrays
        title: Plot title
        save_path: Path to save the figure
        timestep: The timestep to plot
        var_name: Variable name for the title
    """
    fig = plt.figure(figsize=(16, 7), dpi=300)
    
    # Mean prediction
    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ax1.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
    ax1.add_feature(cfeature.OCEAN, facecolor='lightblue', edgecolor='none')
    ax1.coastlines()
    ax1.gridlines(draw_labels=True)
    
    im1 = ax1.imshow(mean[timestep], origin='lower',
                   extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                   cmap='RdBu_r', transform=ccrs.PlateCarree())
    ax1.set_title(f"Mean Prediction - {var_name}")
    cbar1 = fig.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.05, shrink=0.8)
    
    if "Temperature" in var_name:
        cbar1.set_label("Temperature (°C)")
    else:
        cbar1.set_label(var_name)
    
    # Uncertainty (standard deviation)
    ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
    ax2.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
    ax2.add_feature(cfeature.OCEAN, facecolor='lightblue', edgecolor='none')
    ax2.coastlines()
    ax2.gridlines(draw_labels=True)
    
    im2 = ax2.imshow(std[timestep], origin='lower',
                   extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                   cmap='viridis', transform=ccrs.PlateCarree())
    ax2.set_title(f"Uncertainty (Std Dev) - {var_name}")
    cbar2 = fig.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.05, shrink=0.8)
    
    if "Temperature" in var_name:
        cbar2.set_label("Temperature Uncertainty (°C)")
    else:
        cbar2.set_label(f"{var_name} Uncertainty")
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def create_uncertainty_animation(mean, std, lower_ci, upper_ci, true_data, lat, lon, title, save_path, fps=24):
    """
    Create an animation showing prediction with confidence intervals.
    
    Args:
        mean: Mean predictions [timesteps, height, width]
        std: Standard deviation [timesteps, height, width]
        lower_ci, upper_ci: Confidence intervals [timesteps, height, width]
        true_data: Ground truth [timesteps, height, width]
        lat, lon: Coordinate arrays
        title: Plot title
        save_path: Path to save animation
        fps: Frames per second
    """
    n_timesteps = len(mean)
    
    # Set up figure
    fig = plt.figure(figsize=(20, 10), dpi=180)
    
    # Main prediction with uncertainty
    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    
    # Error/uncertainty map
    ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
    
    # Add map features
    for ax in (ax1, ax2):
        ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', edgecolor='none')
        ax.coastlines()
    
    # Calculate global min/max for consistent colormap
    vmin, vmax = min(np.min(mean), np.min(true_data)), max(np.max(mean), np.max(true_data))
    
    # Max std for consistent colormap
    max_std = np.max(std)
    
    # Initialize colorbar holders
    cbar1 = None
    cbar2 = None
    
    def update_plot(frame):
        nonlocal cbar1, cbar2
        ax1.clear()
        ax2.clear()
        
        # Re-add map features
        for ax in (ax1, ax2):
            ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
            ax.add_feature(cfeature.OCEAN, facecolor='lightblue', edgecolor='none')
            ax.coastlines()
            ax.gridlines(draw_labels=True)
        
        # Plot mean prediction with confidence
        im1 = ax1.imshow(mean[frame], origin='lower',
                       extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                       cmap='RdBu_r', vmin=vmin, vmax=vmax,
                       transform=ccrs.PlateCarree())
        
        # Add contours for confidence intervals
        ax1.contour(lon, lat, upper_ci[frame], colors=['k'], alpha=0.5,
                  linestyles=['--'], transform=ccrs.PlateCarree())
        ax1.contour(lon, lat, lower_ci[frame], colors=['k'], alpha=0.5,
                  linestyles=['--'], transform=ccrs.PlateCarree())
        
        ax1.set_title(f"Prediction with Confidence - Time step {frame}")
        
        # Plot uncertainty (standard deviation)
        im2 = ax2.imshow(std[frame], origin='lower',
                       extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                       cmap='viridis', vmin=0, vmax=max_std,
                       transform=ccrs.PlateCarree())
        
        ax2.set_title(f"Prediction Uncertainty - Time step {frame}")
        
        # Add colorbars (only once)
        if cbar1 is None:
            cbar1 = fig.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.05, shrink=0.8)
            if "Temperature" in title:
                cbar1.set_label("Temperature (°C)")
            else:
                cbar1.set_label(title)
                
        if cbar2 is None:
            cbar2 = fig.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.05, shrink=0.8)
            if "Temperature" in title:
                cbar2.set_label("Temperature Uncertainty (°C)")
            else:
                cbar2.set_label(f"{title} Uncertainty")
        
        return [im1, im2]
    
    # Get writer based on file extension
    writer, save_path = get_animation_writer(save_path, fps)
    
    # Create animation
    ani = FuncAnimation(fig, update_plot, frames=tqdm(range(n_timesteps), leave=False),
                      interval=1000/fps, blit=True)
    
    print(f"Saving animation to {save_path}")
    ani.save(save_path, writer=writer, dpi=180)
    plt.close()

def mc_dropout_visualization(run_name, year, model_name="model_2", num_samples=30, fps=24, duration=10, 
                           data_dir='./data/era_5_data', save_dir='visualizations'):
    """
    Generate and save MC dropout uncertainty visualizations.
    
    Args:
        run_name: Name of the experiment run
        year: Year to analyze
        model_name: Name of the model to use (default: model_2)
        num_samples: Number of MC dropout samples to generate
        fps: Frames per second for animations
        duration: Number of timesteps to predict
        data_dir: Directory with ERA5 data
        save_dir: Directory to save visualizations
    """

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint with configuration
    print(f"Loading checkpoint for {run_name}...")
    model, epoch, config, device, model_name = load_checkpoint(run_name, device=device)
    
    save_dir = os.path.join(save_dir, f'{model_name}_mc_dropout')
    os.makedirs(save_dir, exist_ok=True)
    
    # Load normalization parameters from training set
    train_norm_param = load_dataset(
        nb_file=10,
        train_val_split=config.get('train_val_split'),
        root_dir=data_dir,
        normalize=True
    )['train'].get_norm_params()
    
    # Create validation dataset
    dataset_val = ERADataset(
        root_dir=data_dir,
        years=[year],
        normalize=True,
        norm_params=train_norm_param
    )
    val_loader = DataLoader(dataset_val, batch_size=fps, shuffle=False)
    
    # Generate MC dropout predictions
    print(f"Generating MC dropout predictions with {num_samples} samples...")
    all_predictions, targets, norm_params = generate_mc_dropout_predictions(
        model, val_loader, device, num_samples=num_samples, duration=duration
    )
    
    # Transform longitude (rearrange grid)
    all_predictions = [transform_longitude(pred) for pred in all_predictions]
    targets = transform_longitude(targets)
    
    # Get coordinates
    lat = np.linspace(-90, 90, all_predictions[0].shape[-2])
    lon = np.linspace(-180, 180, all_predictions[0].shape[-1])
    
    # Calculate uncertainty metrics
    print("Calculating uncertainty metrics...")
    mean_preds, std_preds, lower_ci, upper_ci = calculate_uncertainty_metrics(all_predictions)
    
    # Process temperature uncertainty
    temp_mean = mean_preds[:, 0]           # Mean temperature predictions
    temp_std = std_preds[:, 0]             # Temperature uncertainty
    temp_lower_ci = lower_ci[:, 0]         # Lower CI
    temp_upper_ci = upper_ci[:, 0]         # Upper CI
    temp_true = targets[:, 0]              # True temperature
    
    # Denormalize temperature
    temp_mean = denormalize_variable(temp_mean, norm_params['2m_temperature'])
    
    # Convert tensor to numpy if needed
    max_val = norm_params['2m_temperature']['max']
    min_val = norm_params['2m_temperature']['min']
    
    # Handle different types (tensor or float)
    if hasattr(max_val, 'cpu'):
        max_val = max_val[0].cpu().numpy()
    elif isinstance(max_val, list):
        max_val = max_val[0]
        
    if hasattr(min_val, 'cpu'):
        min_val = min_val[0].cpu().numpy()
    elif isinstance(min_val, list):
        min_val = min_val[0]
    
    # Calculate denormalized std dev
    temp_std = std_preds[:, 0] * (max_val - min_val)
    
    temp_lower_ci = denormalize_variable(temp_lower_ci, norm_params['2m_temperature'])
    temp_upper_ci = denormalize_variable(temp_upper_ci, norm_params['2m_temperature'])
    temp_true = denormalize_variable(temp_true, norm_params['2m_temperature'])
    
    # Convert temperature from Kelvin to Celsius
    temp_mean = temp_mean - 273.15
    temp_std = temp_std  # Std dev remains the same in Celsius
    temp_lower_ci = temp_lower_ci - 273.15
    temp_upper_ci = temp_upper_ci - 273.15
    temp_true = temp_true - 273.15
    
    # Create static plots for specific timesteps
    for ts in [0, 5, 10, 15]:
        if ts < len(temp_mean):
            plot_uncertainty_map(
                temp_mean, temp_std, lat, lon,
                f"Temperature Uncertainty Timestep {ts}",
                os.path.join(save_dir, f'temperature_uncertainty_t{ts}_{year}.png'),
                ts, "Temperature (°C)"
            )
    
    # Create temperature uncertainty animation
    print("Creating temperature uncertainty animation...")
    create_uncertainty_animation(
        temp_mean, temp_std, temp_lower_ci, temp_upper_ci, temp_true,
        lat, lon, "Temperature (°C)",
        os.path.join(save_dir, f'temperature_uncertainty_{year}.mp4'),
        fps=fps
    )
    print(f"All visualizations saved to {save_dir}")
    
if __name__ == "__main__":
    # Set default parameters
    run_name = "run_8"  # This should be a model_2 variant
    year = 2000
    num_samples = 30
    fps = 24
    duration = 20
    
    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        run_name = sys.argv[1]
    if len(sys.argv) > 2:
        year = int(sys.argv[2])
    if len(sys.argv) > 3:
        num_samples = int(sys.argv[3])
    
    print(f"Running MC dropout inference for run: {run_name}, year: {year} with {num_samples} samples")
    mc_dropout_visualization(run_name, year, num_samples=num_samples, fps=fps, duration=duration)