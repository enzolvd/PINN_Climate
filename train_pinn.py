import glob
import sys
from tqdm import tqdm
import os
import wandb
import torch

def train_pinn(args, model, train_loader, val_loader, device):
    """Train the PINN model with checkpointing and logging capabilities."""
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.experiment_name}.pt")
    
    # Initialize variables
    start_epoch = 0
    best_val_loss = float('inf')
    global_step = 0
    checkpoint_to_load = None
    
    # Check for existing checkpoint
    if args.resume and os.path.exists(args.resume):
        checkpoint_to_load = args.resume
    else:
        checkpoint_pattern = os.path.join(args.checkpoint_dir, f"{args.experiment_name}*.pt")
        checkpoints = sorted(glob.glob(checkpoint_pattern))
        if checkpoints:
            checkpoint_to_load = checkpoints[-1]
    
    # Initialize wandb first
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
                print("Resuming existing run with same parameters...")
                wandb.init(
                    project=args.wandb_project,
                    name=args.experiment_name,
                    id=existing_run.id,
                    resume="must"
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
    
    # Load checkpoint if it exists
    if checkpoint_to_load:
        print(f"Loading checkpoint from {checkpoint_to_load}")
        checkpoint = torch.load(checkpoint_to_load)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        global_step = checkpoint.get('global_step', start_epoch * len(train_loader))
        print(f"Resuming from epoch {start_epoch} with global_step {global_step}")
        
        # Now we can safely log to wandb since it's initialized
        wandb.log({
            "checkpoint_loaded": checkpoint_to_load,
            "resumed_from_epoch": start_epoch,
            "best_val_loss": best_val_loss
        }, step=global_step)
    
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
            
            global_step += 1
        
        # Rest of the training loop remains the same...
        [Previous validation and checkpoint saving code remains unchanged]
    
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
                            val_targets[0, i].detach().cpu().numpy(),
                            val_outputs[0, i].detach().cpu().numpy(),
                            f"Comparison for {var}",
                            normalize=True,
                            norm_params=norm_params[var]
                        )
                        wandb.log({f"visualization/{var}": fig_to_wandb_image(fig)}, step=global_step)
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
                    'global_step': global_step,
                    'train_loader_len': len(train_loader)  # Save this for proper step restoration
                }, checkpoint_path)
                print(f"Saved checkpoint at epoch {epoch + 1} with global_step {global_step}")
    
    wandb.finish()
    return model