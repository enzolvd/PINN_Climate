import argparse
import os
import io
import glob
import sys
import torch
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

from model import ClimatePINN
from dataset import load_dataset

def parse_arguments():
    parser = argparse.ArgumentParser(description="Training PINN for climate modeling with physics constraints.")
    
    # Model parameters
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension for the model")
    parser.add_argument("--initial_re", type=float, default=100.0, help="Initial Reynolds number")
    
    # Training parameters
    parser.add_argument("--nb_years", type=int, default=10, help="Number of years to use for training")
    parser.add_argument("--train_val_split", type=str, default="0.8", help="Train-validation split ratio (float between 0 and 1, or 'None' for train-only)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    
    # Loss weights
    parser.add_argument("--physics_weight", type=float, default=1.0, help="Weight for physics loss")
    parser.add_argument("--data_weight", type=float, default=1.0, help="Weight for data loss")
    
    # Checkpointing and logging
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    parser.add_argument("--experiment_name", type=str, default="climate_pinn", help="Name of the experiment")
    parser.add_argument("--wandb_project", type=str, default="climate_pinn", help="WandB project name")
    parser.add_argument("--visual_interval", type=int, default=1, help="Epoch interval for visualization")
    
    # Data paths
    parser.add_argument("--data_dir", type=str, default="./data/era_5_data", help="Path to ERA5 data")
    
    return parser.parse_args()

def fig_to_wandb_image(fig):
    """Convert matplotlib figure to wandb image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    image = Image.open(buf).convert("RGB")
    return wandb.Image(image)

def plot_comparison(true, pred, title, normalize=True, norm_params=None):
    """Create comparison plots for model predictions vs ground truth."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    if normalize and norm_params is not None:
        vmin = norm_params['min']
        vmax = norm_params['max']
        true = true * (vmax - vmin) + vmin
        pred = pred * (vmax - vmin) + vmin
    else:
        vmin = min(true.min(), pred.min())
        vmax = max(true.max(), pred.max())
    
    im1 = axes[0].imshow(true, vmin=vmin, vmax=vmax)
    axes[0].set_title('True')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(pred, vmin=vmin, vmax=vmax)
    axes[1].set_title('Predicted')
    plt.colorbar(im2, ax=axes[1])
    
    plt.suptitle(title)
    return fig

def train_pinn(args, model, train_loader, val_loader, device):
    """Train the PINN model with checkpointing and logging capabilities."""
    # Initialize step counter
    starting_step = 0  # Track the global step for wandb logging
    # Check if experiment already exists
    api = wandb.Api()
    try:
        runs = api.runs(
            f"{wandb.api.default_entity}/{args.wandb_project}",
            {"displayName": args.experiment_name}
        )
        
        if len(runs) > 0:
            print(f"\nFound existing experiment: {args.experiment_name}")
            existing_run = sorted(runs, key=lambda x: x.created_at)[-1]
            
            # Compare configurations
            existing_config = existing_run.config
            current_config = vars(args)
            
            # Check for differences in key parameters
            different_params = {}
            key_params = ['hidden_dim', 'initial_re', 'nb_years', 'train_val_split', 
                         'batch_size', 'learning_rate', 'physics_weight', 'data_weight']
            
            def normalize_value(value):
                """Convert values to float for comparison if possible."""
                if value is None or value == 'None':
                    return None
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return str(value)
            
            for param in key_params:
                if param in existing_config:
                    old_val = normalize_value(existing_config[param])
                    new_val = normalize_value(current_config[param])
                    
                    # Only consider as different if the normalized values are different
                    if old_val != new_val:
                        different_params[param] = {
                            'old': existing_config[param],
                            'new': current_config[param]
                        }
            
            if different_params:
                print("\nWARNING: Different parameters detected between existing run and current configuration:")
                for param, values in different_params.items():
                    print(f"  {param}: {values['old']} -> {values['new']}")
                
                while True:
                    response = input("\nOptions:\n"
                                  "1. Continue with new parameters (will delete existing run)\n"
                                  "2. Create a new experiment with a different name\n"
                                  "3. Cancel\n"
                                  "Please choose (1/2/3): ").strip()
                    
                    if response == '1':
                        # Delete existing run
                        existing_run.delete()
                        print(f"\nDeleted existing run. Starting new run with updated parameters...")
                        wandb.init(
                            project=args.wandb_project,
                            name=args.experiment_name,
                            config=vars(args)
                        )
                        break
                    elif response == '2':
                        new_name = input("\nPlease enter a new experiment name: ").strip()
                        args.experiment_name = new_name
                        print(f"\nStarting new experiment: {new_name}")
                        wandb.init(
                            project=args.wandb_project,
                            name=new_name,
                            config=vars(args)
                        )
                        break
                    elif response == '3':
                        print("\nExiting...")
                        sys.exit(0)
                    else:
                        print("\nInvalid choice. Please select 1, 2, or 3.")
            else:
                # No parameter differences, resume the run
                print("Resuming existing run with same parameters...")
                wandb.init(
                    project=args.wandb_project,
                    name=args.experiment_name,
                    id=existing_run.id,
                    resume="must"
                )
        else:
            # No existing run found
            wandb.init(
                project=args.wandb_project,
                name=args.experiment_name,
                config=vars(args)
            )
    except Exception as e:
        print(f"Error checking for existing experiment: {e}")
        # Fallback to creating new run
        wandb.init(
            project=args.wandb_project,
            name=args.experiment_name,
            config=vars(args)
        )
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.experiment_name}.pt")
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Load checkpoint either from args.resume or from existing experiment
    start_epoch = 0
    best_val_loss = float('inf')
    
    # First check if there's a direct checkpoint path provided
    if args.resume and os.path.exists(args.resume):
        checkpoint_to_load = args.resume
    else:
        # Look for existing checkpoint in the experiment directory
        checkpoint_pattern = os.path.join(args.checkpoint_dir, f"{args.experiment_name}*.pt")
        checkpoints = sorted(glob.glob(checkpoint_pattern))
        if checkpoints:
            checkpoint_to_load = checkpoints[-1]  # Get the latest checkpoint
        else:
            checkpoint_to_load = None
    
    if checkpoint_to_load:
        print(f"Loading checkpoint from {checkpoint_to_load}")
        checkpoint = torch.load(checkpoint_to_load)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resuming from epoch {start_epoch}")
        
        # If we're resuming a wandb run, also log the checkpoint loading
        if wandb.run is not None:
            wandb.log({
                "checkpoint_loaded": checkpoint_to_load,
                "resumed_from_epoch": start_epoch,
                "best_val_loss": best_val_loss
            })
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        physics_loss_total = 0.0
        data_loss_total = 0.0
        
        pbar = tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}/{args.epochs}", total=len(train_loader))
        for batch_idx, batch in pbar:
            optimizer.zero_grad()
            
            # Move data to device
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            masks = batch['masks'].to(device)
            coords = [coord.to(device) for coord in batch['coords']]
            
            # Forward pass
            predictions = model(inputs, masks, coords)
            outputs = predictions['output']
            physics_losses = predictions['physics_loss']
            
            # Calculate losses
            data_loss = model.MSE(outputs, targets)
            physics_loss = sum(physics_losses.values())
            
            # Combine losses
            total_loss = args.data_weight * data_loss + args.physics_weight * physics_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += total_loss.item()
            physics_loss_total += physics_loss.item()
            data_loss_total += data_loss.item()
            
            # Update progress bar and log to wandb
            global_step = starting_step + (epoch * len(train_loader) + batch_idx)
            
            pbar.set_postfix({
                'total_loss': total_loss.item(),
                'physics_loss': physics_loss.item(),
                'data_loss': data_loss.item()
            })
            
            # Log batch-level metrics
            wandb.log({
                'batch/total_loss': total_loss.item(),
                'batch/physics_loss': physics_loss.item(),
                'batch/data_loss': data_loss.item(),
                'batch/reynolds_number': model.get_reynolds_number().item()
            }, step=global_step)
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_physics_loss = physics_loss_total / len(train_loader)
        avg_data_loss = data_loss_total / len(train_loader)
        
        # Validation and visualization (if validation set exists)
        if val_loader is not None:
            model.eval()
            val_total_loss = 0.0
            val_physics_loss = 0.0
            val_data_loss = 0.0
            
            if (epoch + 1) % args.visual_interval == 0:
                # Get first batch for visualization
                val_batch = next(iter(val_loader))
                val_inputs = val_batch['input'].to(device)
                val_targets = val_batch['target'].to(device)
                val_masks = val_batch['masks'].to(device)
                val_coords = [coord.to(device) for coord in val_batch['coords']]
                
                # Need to enable gradients for physics computation
                with torch.set_grad_enabled(True):
                    # Generate predictions with physics computation
                    val_predictions = model(val_inputs, val_masks, val_coords, compute_physics=True)
                    val_outputs = val_predictions['output']
                    
                    # Create comparison plots
                    norm_params = val_batch['norm_params']
                    variables = ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind']
                    
                    for i, var in enumerate(variables):
                        fig = plot_comparison(
                            val_targets[0, i].cpu().numpy(),
                            val_outputs[0, i].cpu().numpy(),
                            f"Comparison for {var}",
                            normalize=True,
                            norm_params=norm_params[var]
                        )
                        wandb.log({f"visualization/{var}": fig_to_wandb_image(fig)}, step=current_step)
                        plt.close(fig)
                
                # Calculate validation loss for all batches
                for val_batch in val_loader:
                    inputs = val_batch['input'].to(device)
                    targets = val_batch['target'].to(device)
                    masks = val_batch['masks'].to(device)
                    coords = [coord.to(device) for coord in val_batch['coords']]
                    
                    # Need to enable gradients for physics computation
                    with torch.set_grad_enabled(True):
                        predictions = model(inputs, masks, coords, compute_physics=True)
                        outputs = predictions['output']
                        physics_losses = predictions['physics_loss']
                        
                        # Calculate individual losses
                        data_loss = model.MSE(outputs, targets)
                        physics_loss = sum(physics_losses.values())
                        total_loss = args.data_weight * data_loss + args.physics_weight * physics_loss
                        
                        val_total_loss += total_loss.item()
                        val_physics_loss += physics_loss.item()
                        val_data_loss += data_loss.item()
            
            # Calculate averages
            avg_val_total_loss = val_total_loss / len(val_loader)
            avg_val_physics_loss = val_physics_loss / len(val_loader)
            avg_val_data_loss = val_data_loss / len(val_loader)
            
            # Log epoch-level metrics
            global_step = starting_step + ((epoch + 1) * len(train_loader))
            wandb.log({
                'epoch/train_loss': avg_train_loss,
                'epoch/train_physics_loss': avg_physics_loss,
                'epoch/train_data_loss': avg_data_loss,
                'epoch/val_total_loss': avg_val_total_loss,
                'epoch/val_physics_loss': avg_val_physics_loss,
                'epoch/val_data_loss': avg_val_data_loss,
                'epoch/reynolds_number': model.get_reynolds_number().item(),
                'epoch': epoch + 1
            }, step=global_step)
            
            # Save checkpoint if validation loss improved
            if avg_val_total_loss < best_val_loss:
                best_val_loss = avg_val_total_loss
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'global_step': global_step
                }, checkpoint_path)
                print(f"Saved checkpoint at epoch {epoch + 1}")
    
    wandb.finish()
    return model

def main():
    args = parse_arguments()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Process train_val_split parameter
    if args.train_val_split.lower() == 'none':
        train_val_split = None
    else:
        try:
            train_val_split = float(args.train_val_split)
            if not 0 < train_val_split < 1:
                raise ValueError("train_val_split must be between 0 and 1")
        except ValueError as e:
            print(f"Error: {e}. train_val_split must be 'None' or a float between 0 and 1")
            return

    # Load datasets
    print("Loading datasets...")
    datasets = load_dataset(
        nb_file=args.nb_years,
        train_val_split=train_val_split,
        root_dir=args.data_dir,
        normalize=True
    )
    
    train_loader = DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True)
    
    if train_val_split is not None:
        val_loader = DataLoader(datasets['val'], batch_size=args.batch_size, shuffle=False)
        print(f"Loaded {len(datasets['train'])} training samples and {len(datasets['val'])} validation samples")
    else:
        val_loader = None
        print(f"Loaded {len(datasets['train'])} training samples (no validation split)")
    
    # Initialize model
    print("Initializing model...")
    model = ClimatePINN(
        hidden_dim=args.hidden_dim,
        initial_re=args.initial_re,
        device=device
    )
    
    # Train model
    print("Starting training...")
    train_pinn(args, model, train_loader, val_loader, device)
    print("Training completed!")

if __name__ == "__main__":
    main()