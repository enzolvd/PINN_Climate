import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import matplotlib.animation as animation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from model import ClimatePINN
from dataset import load_dataset
import os
from torch.utils.data import DataLoader

# Enable anti-aliasing and set backend
mpl.rcParams['text.antialiased'] = True
mpl.rcParams['lines.antialiased'] = True
mpl.use('Agg')

def load_checkpoint(checkpoint_path):
    """Load model checkpoint and configuration."""
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint to CPU first to avoid potential GPU memory issues
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model configuration from checkpoint
    config = {}
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # If config not in checkpoint, try to extract from wandb_config
        if 'wandb_config' in checkpoint:
            config = checkpoint['wandb_config']
    
    # Set default values if not found in checkpoint
    hidden_dim = config.get('hidden_dim', 64)
    initial_re = config.get('initial_re', 100.0)
    
    print(f"Loading model with configuration: hidden_dim={hidden_dim}, initial_re={initial_re}")
    
    # Initialize model with loaded configuration
    model = ClimatePINN(hidden_dim=hidden_dim, initial_re=initial_re, device=device)
    model = model.to(device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    
    return model, checkpoint['epoch'], config, device

def denormalize_variable(data, var_params):
    """Denormalize data using stored parameters.
    
    Args:
        data: numpy.ndarray or torch.Tensor - The data to denormalize
        var_params: dict - Dictionary containing 'min' and 'max' values as torch.Tensor or numbers
        
    Returns:
        Denormalized data of the same type as input
    """
    # Convert var_params to numpy if they're tensors
    if hasattr(var_params['max'], 'cpu'):
        var_min = var_params['min'].cpu().numpy()
        var_max = var_params['max'].cpu().numpy()
    else:
        var_min = var_params['min']
        var_max = var_params['max']
    
    # Now perform the denormalization with numpy arrays
    return data * (var_max - var_min) + var_min

def get_animation_writer(save_path):
    """Get appropriate animation writer based on file extension and availability."""
    if save_path.endswith('.mp4'):
        # Check if ffmpeg is available
        if animation.writers['ffmpeg'].isAvailable():
            writer = animation.FFMpegWriter(
                fps=10, 
                metadata=dict(artist='Me'),
                bitrate=5000
            )
        else:
            print("FFmpeg not available. Falling back to GIF format.")
            save_path = save_path.replace('.mp4', '.gif')
            writer = animation.PillowWriter(fps=10)
    else:
        writer = animation.PillowWriter(fps=10)
    
    return writer, save_path

def compute_animation_for_scalar(scalar_data, lat, lon, title, save_path):
    """Create animation for scalar fields (like temperature)."""
    n_timesteps = len(scalar_data)
    
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
    vmin, vmax = np.min(scalar_data), np.max(scalar_data)
    
    def update_scalar(frame):
        nonlocal cbar
        ax.clear()
        
        # Re-add map features (required for animation)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', edgecolor='none')
        ax.coastlines()
        ax.gridlines(draw_labels=True)
        
        # Plot data
        data = scalar_data[frame]
        img = ax.imshow(data, origin='lower', 
                       extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                       cmap='RdBu_r',
                       vmin=vmin, vmax=vmax)
        
        ax.set_title(f"{title} - Time step {frame}")
        
        # Add colorbar (only once)
        if cbar is None:
            cbar = fig.colorbar(img, ax=ax)
            if "Temperature" in title:
                cbar.set_label("Temperature (°C)")
            else:
                cbar.set_label(title)
        
        return [img]
    
    # Get appropriate writer and possibly modified save path
    writer, save_path = get_animation_writer(save_path)
    
    # Create and save animation
    ani = FuncAnimation(fig, update_scalar, frames=n_timesteps, 
                       interval=200, blit=True)
    
    print(f"Saving animation to {save_path}")
    ani.save(save_path, writer=writer, dpi=300)
    plt.close()

def compute_animation_for_vector(vector_data, lat, lon, title, save_path):
    """Create animation for vector fields (wind)."""
    n_timesteps = len(vector_data)
    
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
        
        # Compute wind magnitude
        u = vector_data[frame, 0]
        v = vector_data[frame, 1]
        magnitude = np.sqrt(u**2 + v**2)
        
        # Create quiver plot with longer arrows
        q = ax.quiver(lon2d[::2, ::2], lat2d[::2, ::2], 
                     u[::2, ::2], v[::2, ::2], magnitude[::2, ::2],
                     transform=ccrs.PlateCarree(),
                     scale=2,  # Smaller scale value makes arrows longer
                     scale_units='xy',
                     cmap='viridis',
                     width=0.004,  # Increased width for better visibility
                     headwidth=4,  # Wider arrow heads
                     headlength=5,  # Longer arrow heads
                     headaxislength=4.5,  # Longer head axis
                     minshaft=2)  # Minimum shaft length
        
        ax.set_title(f"{title} - Time step {frame}")
        
        # Add colorbar (only once)
        if cbar is None:
            cbar = fig.colorbar(q, ax=ax, orientation='horizontal', pad=0.05)
            cbar.set_label('Wind Speed (m/s)')
        
        return [q]
    
    # Get appropriate writer and possibly modified save path
    writer, save_path = get_animation_writer(save_path)
    
    # Create and save animation
    ani = FuncAnimation(fig, update_vector, frames=n_timesteps, 
                       interval=200, blit=True)
    
    print(f"Saving animation to {save_path}")
    ani.save(save_path, writer=writer, dpi=300)
    plt.close()

def generate_predictions(model, dataloader, device, num_frames=24):
    """Generate predictions for multiple timesteps."""
    model.eval()
    predictions = []
    targets = []
    norm_params = None
    
    print(f"Generating predictions for {num_frames} frames...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_frames:
                break
            
            print(f"Processing frame {i+1}/{num_frames}")
            
            # Store normalization parameters from first batch
            if norm_params is None:
                norm_params = batch['norm_params']
            
            try:
                # Move data to device
                inputs = batch['input'].to(device)
                batch_targets = batch['target'].to(device)
                masks = batch['masks'].to(device)
                coords = [coord.to(device) for coord in batch['coords']]
                
                # Get predictions
                outputs = model(inputs, masks, coords, compute_physics=False)['output']
                
                # Move predictions and targets back to CPU and convert to numpy immediately
                predictions.append(outputs.cpu().numpy())
                targets.append(batch_targets.cpu().numpy())
                
                # Explicitly delete tensors
                del outputs, inputs, batch_targets, masks, coords
                
                # Clear GPU memory if using CUDA
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"WARNING: Out of memory at frame {i}. Trying to recover...")
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    # If we have some predictions, return them
                    if predictions:
                        print(f"Returning {len(predictions)} frames instead of requested {num_frames}")
                        return np.array(predictions), np.array(targets), norm_params
                    else:
                        raise RuntimeError("Not enough memory to process even one frame")
                else:
                    raise e
    
    return np.array(predictions), np.array(targets), norm_params

def process_predictions_in_batches(predictions, targets, norm_params, save_dir, epoch, lat, lon, batch_size=5):
    """Process and save animations in batches to manage memory."""
    total_frames = len(predictions)
    num_batches = (total_frames + batch_size - 1) // batch_size

    for var_name, var_idx, converter, title_prefix in [
        ('2m_temperature', 0, lambda x: x - 273.15, 'Temperature'),
        ('wind', slice(1, 3), lambda x: x, 'Wind')
    ]:
        print(f"\nProcessing {title_prefix} animations...")
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_frames)
            print(f"Processing frames {start_idx} to {end_idx-1}...")

            if var_name == '2m_temperature':
                # Process temperature data
                pred_batch = predictions[start_idx:end_idx, 0, var_idx]
                true_batch = targets[start_idx:end_idx, 0, var_idx]
                
                # Denormalize
                pred_batch = denormalize_variable(pred_batch, norm_params[var_name])
                true_batch = denormalize_variable(true_batch, norm_params[var_name])
                
                # Convert to Celsius
                pred_batch = converter(pred_batch)
                true_batch = converter(true_batch)
                
                # Create animations for this batch
                compute_animation_for_scalar(
                    pred_batch, lat, lon,
                    f"Predicted {title_prefix} (°C)",
                    os.path.join(save_dir, f'temperature_prediction_epoch_{epoch}_batch_{batch_idx}.mp4')
                )
                compute_animation_for_scalar(
                    true_batch, lat, lon,
                    f"True {title_prefix} (°C)",
                    os.path.join(save_dir, f'temperature_true_epoch_{epoch}_batch_{batch_idx}.mp4')
                )
            else:
                # Process wind data
                pred_batch = predictions[start_idx:end_idx, 0, var_idx]
                true_batch = targets[start_idx:end_idx, 0, var_idx]
                
                # Denormalize u component
                pred_batch[:, 0] = denormalize_variable(pred_batch[:, 0], 
                                                      norm_params['10m_u_component_of_wind'])
                true_batch[:, 0] = denormalize_variable(true_batch[:, 0], 
                                                      norm_params['10m_u_component_of_wind'])
                
                # Denormalize v component
                pred_batch[:, 1] = denormalize_variable(pred_batch[:, 1], 
                                                      norm_params['10m_v_component_of_wind'])
                true_batch[:, 1] = denormalize_variable(true_batch[:, 1], 
                                                      norm_params['10m_v_component_of_wind'])
                
                compute_animation_for_vector(
                    pred_batch, lat, lon,
                    f"Predicted {title_prefix} (m/s)",
                    os.path.join(save_dir, f'wind_prediction_epoch_{epoch}_batch_{batch_idx}.mp4')
                )
                compute_animation_for_vector(
                    true_batch, lat, lon,
                    f"True {title_prefix} (m/s)",
                    os.path.join(save_dir, f'wind_true_epoch_{epoch}_batch_{batch_idx}.mp4')
                )
            
            # Clear variables explicitly
            del pred_batch, true_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

def visualize_predictions(checkpoint_path, data_dir, num_frames=24, save_dir='visualizations'):
    """Generate and save static and animated visualizations."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Load checkpoint with configuration and get device
    model, epoch, config, device = load_checkpoint(checkpoint_path)
    
    # Save configuration for reference
    config_path = os.path.join(save_dir, 'model_config.txt')
    with open(config_path, 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    
    # Load dataset
    datasets = load_dataset(
        nb_file=10,
        train_val_split=0.8,
        root_dir=data_dir,
        normalize=True
    )
    
    val_loader = DataLoader(datasets['val'], batch_size=1, shuffle=False)
    
    # Generate predictions
    predictions, targets, norm_params = generate_predictions(model, val_loader, device, num_frames)
    
    # Get coordinates
    lat = np.linspace(-90, 90, predictions.shape[-2])
    lon = np.linspace(-180, 180, predictions.shape[-1])
    
    # Denormalize temperature predictions and targets
    temp_pred = predictions[:, 0, 0]  # Temperature predictions
    temp_true = targets[:, 0, 0]      # True temperature
    
    temp_pred = denormalize_variable(temp_pred, norm_params['2m_temperature'])
    temp_true = denormalize_variable(temp_true, norm_params['2m_temperature'])
    
    # Convert temperature from Kelvin to Celsius if needed
    temp_pred = temp_pred - 273.15  # Kelvin to Celsius
    temp_true = temp_true - 273.15  # Kelvin to Celsius
    
    compute_animation_for_scalar(
        temp_pred, 
        lat, lon, 
        "Predicted Temperature (°C)", 
        os.path.join(save_dir, f'temperature_prediction_epoch_{epoch}.mp4')
    )
    
    compute_animation_for_scalar(
        temp_true, 
        lat, lon, 
        "True Temperature (°C)", 
        os.path.join(save_dir, f'temperature_true_epoch_{epoch}.mp4')
    )
    
    # Denormalize wind predictions and targets
    wind_pred = predictions[:, 0, 1:3]  # Wind predictions (u and v components)
    wind_true = targets[:, 0, 1:3]      # True wind
    
    # Denormalize u component
    wind_pred[:, 0] = denormalize_variable(wind_pred[:, 0], norm_params['10m_u_component_of_wind'])
    wind_true[:, 0] = denormalize_variable(wind_true[:, 0], norm_params['10m_u_component_of_wind'])
    
    # Denormalize v component
    wind_pred[:, 1] = denormalize_variable(wind_pred[:, 1], norm_params['10m_v_component_of_wind'])
    wind_true[:, 1] = denormalize_variable(wind_true[:, 1], norm_params['10m_v_component_of_wind'])
    
    compute_animation_for_vector(
        wind_pred, 
        lat, lon, 
        "Predicted Wind (m/s)", 
        os.path.join(save_dir, f'wind_prediction_epoch_{epoch}.mp4')
    )
    
    compute_animation_for_vector(
        wind_true, 
        lat, lon, 
        "True Wind (m/s)", 
        os.path.join(save_dir, f'wind_true_epoch_{epoch}.mp4')
    )
    
    print(f"Saved animations in {save_dir}")
