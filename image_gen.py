import torch
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
from torch.utils.data import DataLoader
from pathlib import Path

# Import necessary functions from visualisation.py
from video_gen import (
    load_checkpoint, 
    generate_predictions, 
    denormalize_variable, 
    transform_longitude
)
from dataset import ERADataset, load_dataset

def create_static_comparison_images(run_name, year, timestep=0, data_dir='./data/era_5_data', save_dir='.'):
    """
    Generate static PNG images showing side-by-side comparisons of true vs predicted values
    for wind and temperature at a specific timestep.
    
    Args:
        run_name (str): Name of the experiment run
        year (int): Year of data to visualize
        timestep (int): Specific timestep to visualize (default=0)
        data_dir (str): Directory containing ERA5 data
        save_dir (str): Directory to save visualizations
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint with configuration
    model, epoch, config, device, model_name = load_checkpoint(run_name, device=device)
    vis_save_dir = f'{save_dir}/visualizations/{model_name}/static' 
    os.makedirs(vis_save_dir, exist_ok=True)
    
    # Load dataset normalization parameters from training data
    train_norm_param = load_dataset(
        nb_file=10,
        train_val_split=config.get('train_val_split'),
        root_dir=data_dir,
        normalize=True
    )['train'].get_norm_params()
    
    # Create validation dataset for the specified year
    dataset_val = ERADataset(
        root_dir=data_dir,
        years=[year],
        normalize=True,
        norm_params=train_norm_param
    )
    val_loader = DataLoader(dataset_val, batch_size=1, shuffle=False)
    
    # Generate predictions (just enough to get the specified timestep)
    max_batches = timestep + 1
    predictions, targets, norm_params = generate_predictions(model, val_loader, device, duration=max_batches)
    
    # Ensure we have data for the requested timestep
    if timestep >= len(predictions):
        raise ValueError(f"Requested timestep {timestep} exceeds available data length {len(predictions)}")
    
    # Transform longitude to have -180 to 180 range
    predictions = transform_longitude(predictions)
    targets = transform_longitude(targets)
    
    # Get coordinates
    lat = np.linspace(-90, 90, predictions.shape[-2])
    lon = np.linspace(-180, 180, predictions.shape[-1])
    
    # 1. Create temperature comparison image
    temp_pred = predictions[timestep, 0]  # Temperature predictions
    temp_true = targets[timestep, 0]      # True temperature
    
    # Denormalize temperature
    temp_pred = denormalize_variable(temp_pred, norm_params['2m_temperature'])
    temp_true = denormalize_variable(temp_true, norm_params['2m_temperature'])
    
    # Convert from Kelvin to Celsius
    temp_pred = temp_pred - 273.15
    temp_true = temp_true - 273.15
    
    # Create figure for temperature
    fig = plt.figure(figsize=(16, 8), dpi=300)
    
    # Set up overall title
    fig.suptitle(f'Temperature Comparison - {model_name} - Year {year} - Timestep {timestep}', fontsize=16)
    
    # True temperature
    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ax1.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
    ax1.add_feature(cfeature.OCEAN, facecolor='lightblue', edgecolor='none')
    ax1.coastlines()
    ax1.gridlines(draw_labels=True)
    
    # Get global min/max for consistent colormap
    vmin, vmax = min(np.min(temp_true), np.min(temp_pred)), max(np.max(temp_true), np.max(temp_pred))
    
    true_img = ax1.imshow(temp_true, origin='lower',
                         extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                         cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax1.set_title("True Temperature (°C)")
    
    # Predicted temperature
    ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
    ax2.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
    ax2.add_feature(cfeature.OCEAN, facecolor='lightblue', edgecolor='none')
    ax2.coastlines()
    ax2.gridlines(draw_labels=True)
    
    pred_img = ax2.imshow(temp_pred, origin='lower',
                         extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                         cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax2.set_title("Predicted Temperature (°C)")
    
    # Add single colorbar
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # Position for the colorbar
    cbar = fig.colorbar(true_img, cax=cbar_ax)
    cbar.set_label('Temperature (°C)')
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # Adjust layout to make room for colorbar
    plt.savefig(os.path.join(vis_save_dir, f'temperature_comparison_{year}_ts{timestep}.pdf'))
    plt.close()
    
    # 2. Create wind comparison image
    wind_pred = predictions[timestep, 1:3]  # Wind predictions (u and v components)
    wind_true = targets[timestep, 1:3]      # True wind
    
    # Denormalize u component
    wind_pred[0] = denormalize_variable(wind_pred[0], norm_params['10m_u_component_of_wind'])
    wind_true[0] = denormalize_variable(wind_true[0], norm_params['10m_u_component_of_wind'])
    
    # Denormalize v component
    wind_pred[1] = denormalize_variable(wind_pred[1], norm_params['10m_v_component_of_wind'])
    wind_true[1] = denormalize_variable(wind_true[1], norm_params['10m_v_component_of_wind'])
    
    # Calculate wind magnitudes
    magnitude_true = np.sqrt(wind_true[0]**2 + wind_true[1]**2)
    magnitude_pred = np.sqrt(wind_pred[0]**2 + wind_pred[1]**2)
    
    # Create meshgrid for quiver plot (exactly like in the original visualization code)
    lon2d, lat2d = np.meshgrid(lon, lat)
    
    # Create figure for wind
    fig = plt.figure(figsize=(16, 8), dpi=300)
    
    # Set up overall title
    fig.suptitle(f'Wind Comparison - {model_name} - Year {year} - Timestep {timestep}', fontsize=16)
    
    # True wind
    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ax1.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
    ax1.add_feature(cfeature.OCEAN, facecolor='lightblue', edgecolor='none')
    ax1.coastlines()
    ax1.gridlines(draw_labels=True)
    
    # Use exactly the same parameters as in compute_animation_for_vector
    n = 1  # Sample every nth point, same as original
    q_true = ax1.quiver(lon2d[::n, ::n], lat2d[::n, ::n],
                       wind_true[0][::n, ::n], wind_true[1][::n, ::n], magnitude_true[::n, ::n],
                       transform=ccrs.PlateCarree(),
                       scale=2,
                       scale_units='xy',
                       cmap='viridis',
                       width=0.004,
                       headwidth=4,
                       headlength=5,
                       headaxislength=4.5,
                       minshaft=2)
    ax1.set_title("True Wind (m/s)")
    
    # Predicted wind
    ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
    ax2.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
    ax2.add_feature(cfeature.OCEAN, facecolor='lightblue', edgecolor='none')
    ax2.coastlines()
    ax2.gridlines(draw_labels=True)
    
    q_pred = ax2.quiver(lon2d[::n, ::n], lat2d[::n, ::n],
                       wind_pred[0][::n, ::n], wind_pred[1][::n, ::n], magnitude_pred[::n, ::n],
                       transform=ccrs.PlateCarree(),
                       scale=2,
                       scale_units='xy',
                       cmap='viridis',
                       width=0.004,
                       headwidth=4,
                       headlength=5,
                       headaxislength=4.5,
                       minshaft=2)
    ax2.set_title("Predicted Wind (m/s)")
    
    # Add single colorbar
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # Position for the colorbar
    cbar = fig.colorbar(q_pred, cax=cbar_ax)
    cbar.set_label('Wind Speed (m/s)')
    
    # Add reference scale
    ax1.quiverkey(q_true, X=0.85, Y=0.05, U=10, label='10 m/s', labelpos='E')
    ax2.quiverkey(q_pred, X=0.85, Y=0.05, U=10, label='10 m/s', labelpos='E')
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # Adjust layout to make room for colorbar
    plt.savefig(os.path.join(vis_save_dir, f'wind_comparison_{year}_ts{timestep}.pdf'))
    plt.close()
    
    print(f"Comparison images saved to {vis_save_dir}")
    return vis_save_dir

if __name__ == "__main__":
    # Example usage
    run_name = 'run_8'
    year = 2000
    timestep = 5  # Choose a specific timestep to visualize
    
    output_dir = create_static_comparison_images(run_name, year, timestep)
    print(f"Images saved to: {output_dir}")