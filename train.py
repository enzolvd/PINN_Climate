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
        vmin = norm_params['min'][0].detach().cpu().numpy()
        vmax = norm_params['max'][0].detach().cpu().numpy()
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
    """Train the PINN model with enhanced checkpointing capabilities."""
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Define checkpoint paths
    last_checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.experiment_name}.pt")
    best_checkpoint_path = os.path.join(args.checkpoint_dir, f"best_{args.experiment_name}.pt")
    
    # Initialize variables
    start_epoch = 0
    best_val_loss = float('inf')
    global_step = 0
    global_batch_idx = 0
    
    # Initialize wandb first
    api = wandb.Api()
    try:
        runs = api.runs(
            f"{wandb.api.default_entity}/{args.wandb_project}",
            {"displayName": args.experiment_name}
        )
        
        if len(runs) > 0:
            print(f"\nFound existing experiment: {args.experiment_name}")
            existing_run = sorted(runs, key=lambda x: x.created_at)[-1]
            
            # Check if there's a checkpoint for this run
            checkpoint_exists = False
            if args.resume and os.path.exists(args.resume):
                checkpoint_to_load = args.resume
                checkpoint_exists = True
            else:
                # Load the last checkpoint, not the best one
                if os.path.exists(last_checkpoint_path):
                    checkpoint_to_load = last_checkpoint_path
                    checkpoint_exists = True
            
            if checkpoint_exists:
                print(f"Found checkpoint at {checkpoint_to_load}")
                checkpoint = torch.load(checkpoint_to_load)
                model.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = checkpoint['epoch']
                best_val_loss = checkpoint['best_val_loss']
                
                try:
                    history = existing_run.scan_history()
                    history_list = list(history)
                    
                    if history_list:
                        max_step_row = max(history_list, key=lambda x: x.get('_step', 0))
                        global_step = max_step_row.get('_step', 0) + 1
                        global_batch_idx = max_step_row.get('batch/batch_idx', 0) + 1
                        
                        print(f"Resuming from epoch {start_epoch} with global_step {global_step} and batch_idx {global_batch_idx}")
                except Exception as e:
                    print(f"Error retrieving wandb history: {e}")
                    global_step = checkpoint.get('global_step', start_epoch * len(train_loader))
                    global_batch_idx = checkpoint.get('global_batch_idx', 0)
                
                wandb.init(
                    project=args.wandb_project,
                    name=args.experiment_name,
                    id=existing_run.id,
                    resume="must"
                )
            else:
                print("No checkpoint found. Starting new run with same name...")
                existing_run.delete()
                wandb.init(
                    project=args.wandb_project,
                    name=args.experiment_name,
                    config=vars(args)
                )
        else:
            wandb.init(
                project=args.wandb_project,
                name=args.experiment_name,
                config=vars(args)
            )
    except Exception as e:
        print(f"Error checking for existing experiment: {e}")
        wandb.init(
            project=args.wandb_project,
            name=args.experiment_name,
            config=vars(args)
        )
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if 'checkpoint' in locals() and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        physics_loss_total = 0.0
        data_loss_total = 0.0
        
        # Create batches iterator
        batch_iterator = enumerate(train_loader)
        
        # Skip batches if resuming from middle of epoch
        if epoch == start_epoch and global_batch_idx % len(train_loader) > 0:
            batches_to_skip = global_batch_idx % len(train_loader)
            print(f"Skipping to batch {batches_to_skip}")
            for _ in range(batches_to_skip):
                next(batch_iterator)
        
        pbar = tqdm(batch_iterator, desc=f"Epoch {epoch+1}/{args.epochs}", total=len(train_loader))
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
            
            # Update progress bar
            pbar.set_postfix({
                'total_loss': total_loss.item(),
                'physics_loss': physics_loss.item(),
                'data_loss': data_loss.item()
            })
            
            # Log batch metrics
            batch_metrics = {
                'batch/total_loss': total_loss.item(),
                'batch/physics_loss': physics_loss.item(),
                'batch/data_loss': data_loss.item(),
                'batch/reynolds_number': model.get_reynolds_number().item(),
                'batch/epoch': epoch + 1,
                'batch/batch_idx': global_batch_idx
            }
            
            wandb.log(batch_metrics, step=global_step)
            global_step += 1
            global_batch_idx += 1
            
            # Clear memory
            del predictions, outputs, physics_losses, inputs, targets, masks, coords
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_physics_loss = physics_loss_total / len(train_loader)
        avg_data_loss = data_loss_total / len(train_loader)
        
        # Initialize epoch metrics dictionary
        epoch_metrics = {
            'epoch/train_loss': avg_train_loss,
            'epoch/train_physics_loss': avg_physics_loss,
            'epoch/train_data_loss': avg_data_loss,
            'epoch': epoch + 1
        }
        
        # Save checkpoint after every epoch
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'global_step': global_step,
            'global_batch_idx': global_batch_idx
        }, last_checkpoint_path)
        print(f"Saved regular checkpoint at epoch {epoch + 1}")
        
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
                
                try:
                    with torch.set_grad_enabled(True):
                        # Clear memory and gradients
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        optimizer.zero_grad()
                        
                        val_predictions = model(val_inputs, val_masks, val_coords, compute_physics=True)
                        val_outputs = val_predictions['output']
                        
                        norm_params = val_batch['norm_params']
                        variables = ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind']
                        
                        for i, var in enumerate(variables):
                            fig = plot_comparison(
                                val_targets[0, i].detach().cpu().numpy(),
                                val_outputs[0, i].detach().cpu().numpy(),
                                f"Comparison for {var}",
                                normalize=True,
                                norm_params=norm_params[var]
                            )
                            epoch_metrics[f"visualization/{var}"] = fig_to_wandb_image(fig)
                            plt.close(fig)
                        
                        # Clear memory
                        del val_predictions, val_outputs
                        torch.cuda.empty_cache()
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("WARNING: Out of memory in visualization. Trying without physics computation")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        # Try again without physics computation
                        val_predictions = model(val_inputs, val_masks, val_coords, compute_physics=False)
                        val_outputs = val_predictions['output']
                        
                        for i, var in enumerate(variables):
                            fig = plot_comparison(
                                val_targets[0, i].detach().cpu().numpy(),
                                val_outputs[0, i].detach().cpu().numpy(),
                                f"Comparison for {var}",
                                normalize=True,
                                norm_params=norm_params[var]
                            )
                            epoch_metrics[f"visualization/{var}"] = fig_to_wandb_image(fig)
                            plt.close(fig)
                    else:
                        raise e
                
                # Clear visualization data
                del val_inputs, val_targets, val_masks, val_coords
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Calculate validation loss for all batches
            for val_batch in val_loader:
                # Clear memory before each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                inputs = val_batch['input'].to(device)
                targets = val_batch['target'].to(device)
                masks = val_batch['masks'].to(device)
                coords = [coord.to(device) for coord in val_batch['coords']]
                
                try:
                    with torch.set_grad_enabled(True):
                        # Clear gradients
                        optimizer.zero_grad()
                        
                        # Forward pass with physics computation
                        predictions = model(inputs, masks, coords, compute_physics=True)
                        outputs = predictions['output']
                        physics_losses = predictions['physics_loss']
                        
                        # Calculate losses
                        data_loss = model.MSE(outputs, targets)
                        physics_loss = sum(physics_losses.values())
                        total_loss = args.data_weight * data_loss + args.physics_weight * physics_loss
                        
                        # Accumulate losses
                        val_total_loss += total_loss.item()
                        val_physics_loss += physics_loss.item()
                        val_data_loss += data_loss.item()
                        
                        # Clear intermediate results
                        del predictions, outputs, physics_losses
                        torch.cuda.empty_cache()
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("WARNING: Out of memory in validation. Clearing cache and trying with compute_physics=False")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        # Try again without physics computation
                        predictions = model(inputs, masks, coords, compute_physics=False)
                        outputs = predictions['output']
                        data_loss = model.MSE(outputs, targets)
                        val_total_loss += data_loss.item()
                        val_data_loss += data_loss.item()
                    else:
                        raise e
                
                # Clear batch data
                del inputs, targets, masks, coords
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Calculate averages
            avg_val_total_loss = val_total_loss / len(val_loader)
            avg_val_physics_loss = val_physics_loss / len(val_loader)
            avg_val_data_loss = val_data_loss / len(val_loader)
            
            # Add validation metrics to epoch metrics
            epoch_metrics.update({
                'epoch/val_total_loss': avg_val_total_loss,
                'epoch/val_physics_loss': avg_val_physics_loss,
                'epoch/val_data_loss': avg_val_data_loss
            })
            
            # Save best checkpoint if validation loss improved
            if avg_val_total_loss < best_val_loss:
                best_val_loss = avg_val_total_loss
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'global_step': global_step,
                    'global_batch_idx': global_batch_idx
                }, best_checkpoint_path)
                print(f"Saved best checkpoint at epoch {epoch + 1} with val_loss {best_val_loss:.4f}")
        
        # Log epoch metrics
        wandb.log(epoch_metrics, step=global_step)
        global_step += 1
    
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