import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import matplotlib.animation as animation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tqdm import tqdm
from dataset import load_dataset
import os
from torch.utils.data import DataLoader
import json
from pathlib import Path
from dataset import ERADataset

# Enable anti-aliasing and set backend
mpl.rcParams['text.antialiased'] = True
mpl.rcParams['lines.antialiased'] = True
mpl.use('Agg')

def load_checkpoint(run_name, device=torch.device('cpu'), type='best', checkpoints='./checkpoints'):
    checkpoints = Path(checkpoints)
    with open('experiments.json', 'r') as file:
        configs = json.load(file)
    idx = np.flatnonzero([run_name == configs[i]['experiment_name'] for i in range(len(configs))])[0]
    config = configs[idx]
    file_name = run_name + f'_{type}.pt'
    checkpoint_path = checkpoints / config['model'] / file_name
    checkpoint = torch.load(checkpoint_path)

    # Load model
    ClimatePINN = getattr(__import__(f'models.{config.get('model')}', fromlist=['ClimatePINN']), 'ClimatePINN')

    # Get values for the model
    hidden_dim = config.get('hidden_dim')
    initial_re = config.get('initial_re')

    print(f"Loading model with configuration: hidden_dim={hidden_dim}, initial_re={initial_re}")

    # Initialize model with loaded configuration
    model = ClimatePINN(hidden_dim=hidden_dim, initial_re=initial_re, device=device)
    model = model.to(device)

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode

    return model, checkpoint['epoch'], config, device

def denormalize_variable(data, var_params):
    # Convert var_params to numpy if they're tensors
    if hasattr(var_params['max'], 'cpu'):
        var_min = var_params['min'][0].cpu().numpy()
        var_max = var_params['max'][0].cpu().numpy()
    else:
        var_min = var_params['min'][0]
        var_max = var_params['max'][0]

    # Now perform the denormalization with numpy arrays
    return data * (var_max - var_min) + var_min

def get_animation_writer(save_path, fps):
    """Get appropriate animation writer based on file extension and availability."""
    if save_path.endswith('.mp4'):
        # Check if ffmpeg is available
        if animation.writers['ffmpeg'].isAvailable():
            writer = animation.FFMpegWriter(
                fps=fps,
                metadata=dict(artist='Me'),
                bitrate=5000
            )
        else:
            print("FFmpeg not available. Falling back to GIF format.")
            save_path = save_path.replace('.mp4', '.gif')
            writer = animation.PillowWriter(fps=fps)
    else:
        writer = animation.PillowWriter(fps=fps)

    return writer, save_path

def compute_animation_for_scalar(true_data, predicted_data, lat, lon, title, save_path, year, fps=24):
    """Create animation for scalar fields (like temperature) comparing true vs predicted values."""
    n_timesteps = len(true_data)

    # Set up the figure and axes
    fig = plt.figure(figsize=(20, 12), dpi=300)
    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())

    # Add map features
    for ax in (ax1, ax2):
        ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', edgecolor='none')
        ax.coastlines()

    # Initialize colorbar holder
    cbar = None

    # Calculate global min/max for consistent colormap
    vmin, vmax = min(np.min(true_data), np.min(predicted_data)), max(np.max(true_data), np.max(predicted_data))

    def update_scalar(frame):
        nonlocal cbar
        ax1.clear()
        ax2.clear()

        # Re-add map features (required for animation)
        for ax in (ax1, ax2):
            ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
            ax.add_feature(cfeature.OCEAN, facecolor='lightblue', edgecolor='none')
            ax.coastlines()
            ax.gridlines(draw_labels=True)

        # Plot true data
        true_img = ax1.imshow(true_data[frame], origin='lower',
                              extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                              cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax1.set_title(f"True {title} - Time step {frame}")

        # Plot predicted data
        pred_img = ax2.imshow(predicted_data[frame], origin='lower',
                              extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                              cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax2.set_title(f"Predicted {title} - Time step {frame}")

        # Add colorbar (only once)
        if cbar is None:
            cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # Position for the colorbar
            cbar = fig.colorbar(pred_img, cax=cbar_ax)
            if "Temperature" in title:
                cbar.set_label("Temperature (°C)")
            else:
                cbar.set_label(title)

        return [true_img, pred_img]

    # Get appropriate writer and possibly modified save path
    writer, save_path = get_animation_writer(save_path, fps)
    fig.suptitle(f'Temperature during year {year}')

    # Create and save animation
    interval = np.floor(1000 / fps)
    ani = FuncAnimation(fig, update_scalar, frames=tqdm(range(n_timesteps), leave=False),
                        interval=interval, blit=True)

    print(f"Saving animation to {save_path}")
    ani.save(save_path, writer=writer, dpi=300)
    plt.close()

def compute_animation_for_temperature_difference(true_data, predicted_data, lat, lon, title, save_path, year, fps=24):
    """Create animation for the difference between true and predicted temperature fields."""
    n_timesteps = len(true_data)

    # Set up the figure and axis
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()},
                           figsize=(12, 8), dpi=300)

    # Add map features
    ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', edgecolor='none')
    ax.coastlines()

    # Initialize colorbar holder
    cbar = None

    # Calculate global min/max for consistent colormap
    vmin, vmax = min(np.min(true_data - predicted_data), np.min(predicted_data - true_data)), \
                 max(np.max(true_data - predicted_data), np.max(predicted_data - true_data))

    def update_temperature(frame):
        nonlocal cbar
        ax.clear()

        # Re-add map features (required for animation)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', edgecolor='none')
        ax.coastlines()
        ax.gridlines(draw_labels=True)

        # Compute temperature difference
        diff_data = true_data[frame] - predicted_data[frame]

        # Plot data
        img = ax.imshow(diff_data, origin='lower',
                        extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                        cmap='coolwarm',
                        vmin=vmin, vmax=vmax)

        ax.set_title(f"Temperature Difference - Time step {frame}")

        # Add colorbar (only once)
        if cbar is None:
            cbar = fig.colorbar(img, ax=ax, orientation='horizontal', pad=0.05)
            cbar.set_label('Temperature Difference (°C)')

        return [img]

    # Get appropriate writer and possibly modified save path
    writer, save_path = get_animation_writer(save_path, fps)
    fig.suptitle(f'Temperature Difference during year {year}')

    # Create and save animation
    interval = 1000 / fps  # Calculate interval based on fps
    ani = FuncAnimation(fig, update_temperature, frames=tqdm(range(n_timesteps), leave=False),
                        interval=interval, blit=True)

    print(f"Saving animation to {save_path}")
    ani.save(save_path, writer=writer, dpi=300)
    plt.close()


def compute_animation_for_vector_difference(true_vector_data, predicted_vector_data, lat, lon, title, save_path, year, fps=24):
    """Create animation for the difference between true and predicted vector fields (wind)."""
    n_timesteps = len(true_vector_data)

    # Create meshgrid for quiver plot
    lon2d, lat2d = np.meshgrid(lon, lat)

    # Set up figure
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()},
                           figsize=(12, 8), dpi=300)

    # Add map features
    ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', edgecolor='none')
    ax.coastlines()
    ax.gridlines(draw_labels=True)

    # Initialize colorbar holder
    cbar = None

    def update_vector(frame):
        nonlocal cbar
        ax.clear()

        # Re-add map features
        ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', edgecolor='none')
        ax.coastlines()
        ax.gridlines(draw_labels=True)

        # Compute difference vectors
        u_true = true_vector_data[frame, 0]
        v_true = true_vector_data[frame, 1]
        u_pred = predicted_vector_data[frame, 0]
        v_pred = predicted_vector_data[frame, 1]

        u_diff = u_pred - u_true
        v_diff = v_pred - v_true
        magnitude_diff = np.sqrt(u_diff**2 + v_diff**2)

        # Create quiver plot for difference
        q_diff = ax.quiver(lon2d[::2, ::2], lat2d[::2, ::2],
                            u_diff[::2, ::2], v_diff[::2, ::2], magnitude_diff[::2, ::2],
                            transform=ccrs.PlateCarree(),
                            scale=2,
                            scale_units='xy',
                            cmap='coolwarm',
                            width=0.004,
                            headwidth=4,
                            headlength=5,
                            headaxislength=4.5,
                            minshaft=2)

        ax.set_title(f"Difference {title} - Time step {frame}")

        # Add colorbar (only once)
        if cbar is None:
            cbar = fig.colorbar(q_diff, ax=ax, orientation='horizontal', pad=0.05)
            cbar.set_label('Wind Speed Difference (m/s)')

        return [q_diff]

    # Get appropriate writer and possibly modified save path
    writer, save_path = get_animation_writer(save_path, fps)
    fig.suptitle(f'Wind Difference during year {year}')

    # Create and save animation
    interval = 1000 / fps  # Calculate interval based on fps
    ani = FuncAnimation(fig, update_vector, frames=tqdm(range(n_timesteps), leave=False),
                        interval=interval, blit=True)

    print(f"Saving animation to {save_path}")
    ani.save(save_path, writer=writer, dpi=300)
    plt.close()

def compute_animation_for_vector(true_vector_data, predicted_vector_data, lat, lon, title, save_path, year, fps=24):
    """Create animation for vector fields (wind) comparing true vs predicted values."""
    n_timesteps = len(true_vector_data)

    # Create meshgrid for quiver plot
    lon2d, lat2d = np.meshgrid(lon, lat)

    # Set up figure
    fig = plt.figure(figsize=(20, 12), dpi=300)
    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())

    # Add map features
    for ax in (ax1, ax2):
        ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', edgecolor='none')
        ax.coastlines()

    # Initialize colorbar holder
    cbar = None

    def update_vector(frame):
        nonlocal cbar
        ax1.clear()
        ax2.clear()

        # Re-add map features
        for ax in (ax1, ax2):
            ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
            ax.add_feature(cfeature.OCEAN, facecolor='lightblue', edgecolor='none')
            ax.coastlines()
            ax.gridlines(draw_labels=True)

        # Compute wind magnitude for true data
        u_true = true_vector_data[frame, 0]
        v_true = true_vector_data[frame, 1]
        magnitude_true = np.sqrt(u_true**2 + v_true**2)

        # Create quiver plot for true data
        q_true = ax1.quiver(lon2d[::2, ::2], lat2d[::2, ::2],
                            u_true[::2, ::2], v_true[::2, ::2], magnitude_true[::2, ::2],
                            transform=ccrs.PlateCarree(),
                            scale=2,
                            scale_units='xy',
                            cmap='viridis',
                            width=0.004,
                            headwidth=4,
                            headlength=5,
                            headaxislength=4.5,
                            minshaft=2)

        ax1.set_title(f"True {title} - Time step {frame}")

        # Compute wind magnitude for predicted data
        u_pred = predicted_vector_data[frame, 0]
        v_pred = predicted_vector_data[frame, 1]
        magnitude_pred = np.sqrt(u_pred**2 + v_pred**2)

        # Create quiver plot for predicted data
        q_pred = ax2.quiver(lon2d[::2, ::2], lat2d[::2, ::2],
                            u_pred[::2, ::2], v_pred[::2, ::2], magnitude_pred[::2, ::2],
                            transform=ccrs.PlateCarree(),
                            scale=2,
                            scale_units='xy',
                            cmap='viridis',
                            width=0.004,
                            headwidth=4,
                            headlength=5,
                            headaxislength=4.5,
                            minshaft=2)

        ax2.set_title(f"Predicted {title} - Time step {frame}")

        # Add colorbar (only once)
        if cbar is None:
            cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # Position for the colorbar
            cbar = fig.colorbar(q_pred, cax=cbar_ax)
            cbar.set_label('Wind Speed (m/s)')

        return [q_true, q_pred]

    # Get appropriate writer and possibly modified save path
    writer, save_path = get_animation_writer(save_path, fps)
    fig.suptitle(f'Wind during year {year}')

    # Create and save animation
    interval = 1000 / fps  # Calculate interval based on fps
    ani = FuncAnimation(fig, update_vector, frames=tqdm(range(n_timesteps), leave=False),
                        interval=interval, blit=True)

    print(f"Saving animation to {save_path}")
    ani.save(save_path, writer=writer, dpi=300)
    plt.close()


def generate_predictions(model, dataloader, device, duration=10):
    """Generate predictions for multiple timesteps."""
    model.eval()
    predictions = []
    targets = []
    norm_params = None

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= duration:
                break

            
            # Store normalization parameters from first batch
            if norm_params is None:
                norm_params = batch['norm_params']

            # Move data to device
            inputs = batch['input'].to(device)
            batch_targets = batch['target'].to(device)
            masks = batch['masks'].to(device)
            coords = [coord.to(device) for coord in batch['coords']]

            # Get predictions
            outputs = model(inputs, masks, coords, compute_physics=False)['output']

            # Move predictions and targets back to CPU and convert to numpy immediately
            predictions.append(outputs.cpu())
            targets.append(batch_targets.cpu())

            # Explicitly delete tensors
            del outputs, inputs, batch_targets, masks, coords

            # Clear GPU memory if using CUDA
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    predictions = torch.cat(predictions, axis=0).numpy()
    targets = torch.cat(targets, axis=0).numpy()
    return predictions, targets, norm_params

def transform_longitude(arr):
    # For longitude 64 points, split at index 32
    second_half = arr[:, :, :, :32]   
    first_half = arr[:, :, :, 32:]     
    return np.concatenate([first_half, second_half], axis=-1)

def visualize_predictions(run_name, year, fps=24, duration=10, data_dir='./data/era_5_data', save_dir='visualizations'):
    """Generate and save static and animated visualizations."""
    save_dir += f'/{run_name}' 
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load checkpoint with configuration and get device
    model, epoch, config, device = load_checkpoint(run_name, device=device)

    # Load dataset
    train_norm_param = load_dataset(
        nb_file=10,
        train_val_split=config.get('train_val_split'),
        root_dir=data_dir,
        normalize=True
    )['train'].get_norm_params()

    dataset_val = ERADataset(
            root_dir=data_dir,
            years=[year],
            normalize=True,
            norm_params=train_norm_param  # Pass training normalization parameters
        )
    val_loader = DataLoader(dataset_val, batch_size=fps, shuffle=False)

    # Generate predictions
    predictions, targets, norm_params = generate_predictions(model, val_loader, device, duration)
    predictions = transform_longitude(predictions)
    targets = transform_longitude(targets)

    # # Get coordinates
    lat = np.linspace(-90, 90, predictions.shape[-2])
    lon = np.linspace(-180, 180, predictions.shape[-1])

    # Denormalize temperature predictions and targets
    temp_pred = predictions[:, 0]  # Temperature predictions
    temp_true = targets[:, 0]      # True temperature

    temp_pred = denormalize_variable(temp_pred, norm_params['2m_temperature'])
    temp_true = denormalize_variable(temp_true, norm_params['2m_temperature'])

    # Convert temperature from Kelvin to Celsius if needed
    temp_pred = temp_pred - 273.15  # Kelvin to Celsius
    temp_true = temp_true - 273.15  # Kelvin to Celsius

    compute_animation_for_scalar(temp_true, temp_pred, lat, lon, "Temperature (°C)",
        os.path.join(save_dir, f'temperature_prediction_{year}.mp4'), year, fps=fps)

    compute_animation_for_temperature_difference(temp_true, temp_pred, lat, lon, "Temperature (°C)", 
                                                 os.path.join(save_dir, f'temperature_prediction_{year}_comp.mp4'), year, fps=fps)
    # Denormalize wind predictions and targets
    wind_pred = predictions[:, 1:3]  # Wind predictions (u and v components)
    wind_true = targets[:, 1:3]      # True wind

    # Denormalize u component
    wind_pred[:, 0] = denormalize_variable(wind_pred[:, 0], norm_params['10m_u_component_of_wind'])
    wind_true[:, 0] = denormalize_variable(wind_true[:, 0], norm_params['10m_u_component_of_wind'])

    # Denormalize v component
    wind_pred[:, 1] = denormalize_variable(wind_pred[:, 1], norm_params['10m_v_component_of_wind'])
    wind_true[:, 1] = denormalize_variable(wind_true[:, 1], norm_params['10m_v_component_of_wind'])

    compute_animation_for_vector(wind_true, wind_pred, lat, lon, "Predicted Wind (m/s)", 
                                 os.path.join(save_dir, f'wind_prediction_{year}.mp4'), year, fps=fps)

    compute_animation_for_vector_difference(wind_true, wind_pred, lat, lon, "Predicted Wind (m/s)", 
                                 os.path.join(save_dir, f'wind_prediction_{year}_diff.mp4'), year, fps=fps)
if __name__ == "__main__":
    runs = ['run_4', 'run_7', 'run_8', 'run_9']

    fps = 48
    year = 2000
    duration = 20

    for run in tqdm(runs):
        visualize_predictions(run, year, fps=fps, duration=duration)
