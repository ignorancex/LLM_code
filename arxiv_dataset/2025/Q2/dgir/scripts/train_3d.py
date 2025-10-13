#!/usr/bin/env python3
"""
3D Registration Training Script

This script trains a 3D diffusion-regularized registration network.
Based on your original train3d.py script.
"""

import argparse
import sys
import os
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import itk

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from diffusion_registration.training.config import Config
from diffusion_registration.core.models import setup_diffusion_model, setup_registration_net
from diffusion_registration.core import networks, wrappers, losses


def load_3d_oasis_data(max_subjects=300, data_pattern=None):
    """
    Load 3D OASIS dataset.
    
    Args:
        max_subjects: Maximum number of subjects to load
        data_pattern: Pattern for data files (optional)
        
    Returns:
        Tuple of (dataset_A, dataset_B) as torch tensors
    """
    print(f"Loading 3D OASIS data (max {max_subjects} subjects)...")
    
    dataset_A = []
    dataset_B = []
    count = 0
    
    base_path = '/path-to-OASIS'
    if data_pattern:
        base_path = str(Path(data_pattern).parent.parent)
    
    for i in range(1, 458):
        if count >= max_subjects:
            break
            
        norm_path = f'{base_path}/OASIS_OAS1_{i:04}_MR1/aligned_norm.nii.gz'
        orig_path = f'{base_path}/OASIS_OAS1_{i:04}_MR1/aligned_orig.nii.gz'
        
        if not os.path.exists(norm_path) or not os.path.exists(orig_path):
            continue
            
        try:
            img_A = itk.imread(norm_path)
            img_B = itk.imread(orig_path)
            
            dataset_A.append(torch.from_numpy(np.asarray(img_A))[None])
            dataset_B.append(torch.from_numpy(np.asarray(img_B))[None])
            count += 1
            
            if count % 50 == 0:
                print(f"Loaded {count} subjects...")
                
        except Exception as e:
            print(f"Warning: Failed to load subject {i}: {e}")
            continue
    
    print(f"Successfully loaded {count} subjects")
    
    # Stack into tensors
    dataset_A = torch.stack(dataset_A)
    dataset_B = torch.stack(dataset_B)
    
    return dataset_A, dataset_B


def create_3d_network(config, model, diffusion, input_shape):
    """Create 3D registration network."""
    # Create base 3D network
    inner_net = wrappers.FunctionFromVectorField(networks.tallUNet2(dimension=3))

    # Add multiscale levels
    for _ in range(3):
        inner_net = wrappers.TwoStepRegistration(
            wrappers.DownsampleRegistration(inner_net, dimension=3),
            wrappers.FunctionFromVectorField(networks.tallUNet2(dimension=3))
        )

    # Create loss function
    if config.loss.type == "NewLNCC3D":
        loss_fn = losses.NewLNCC3D(
            diffusion=diffusion, 
            model=model, 
            sigma=config.loss.sigma
        )
    else:
        loss_fn = losses.LNCC(sigma=config.loss.sigma)

    # Create final network with diffusion regularization
    net = wrappers.DiffusionRegularizedNet(
        inner_net, 
        loss_fn, 
        lmbda=config.loss.lambda_regularization
    )
    
    net.assign_identity_map(input_shape)
    return net


def train_3d(config):
    """Main 3D training function."""
    device = torch.device(config.training.device)
    
    # Load data
    max_subjects = getattr(config.data, 'max_subjects', 300)
    dataset_A, dataset_B = load_3d_oasis_data(max_subjects)
    
    # Setup diffusion model
    print("Setting up diffusion model...")
    model, diffusion = setup_diffusion_model(config)
    model = model.eval().to(device)
    
    # Create network
    print("Creating 3D registration network...")
    input_shape = [1, 1, 224, 192, 160]  # Standard 3D shape
    net = create_3d_network(config, model, diffusion, input_shape)
    net = net.to(device)
    net.train()
    
    print(f"Network parameters: {sum(p.numel() for p in net.parameters()):,}")
    
    # Setup optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=config.training.learning_rate)
    
    # Training loop
    all_loss = []
    sim_loss = []
    reg_loss = []
    
    print(f"Starting training for {config.training.epochs} epochs...")
    print(f"Batch size: {config.training.batch_size}")
    
    for epoch in tqdm(range(config.training.epochs), desc="Training"):
        # Sample random batch
        image_A = dataset_A[random.sample(range(len(dataset_A)), config.training.batch_size)].to(device)
        image_B = dataset_B[random.sample(range(len(dataset_B)), config.training.batch_size)].to(device)
        
        optimizer.zero_grad()
        loss_object = net(image_A, image_B)
        loss_object.all_loss.backward()
        
        # Record losses
        all_loss.append(loss_object.all_loss.item())
        sim_loss.append(loss_object.similarity_loss.item())
        reg_loss.append(loss_object.bending_energy_loss.item())
        
        optimizer.step()
        
        # Periodic logging
        if epoch % config.training.print_every == 0:
            print(f"[{epoch}] Total: {loss_object.all_loss.item():.4f}, "
                  f"Sim: {loss_object.similarity_loss.item():.4f}, "
                  f"Reg: {loss_object.bending_energy_loss.item():.4f}")
        
        # Save checkpoint
        if epoch % config.training.save_every == 0 and epoch > 0:
            checkpoint_path = Path(config.output.checkpoint_dir) / f'net_3d_{epoch}.pth'
            torch.save(net.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_path = Path(config.output.checkpoint_dir) / 'net_3d_final.pth'
    torch.save(net.state_dict(), final_path)
    print(f"Final model saved: {final_path}")
    
    return {
        'all_loss': all_loss,
        'sim_loss': sim_loss,
        'reg_loss': reg_loss
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Train 3D diffusion registration network',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config_3d.yaml',
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--data-root',
        type=str,
        help='Override data root directory'
    )
    
    parser.add_argument(
        '--max-subjects',
        type=int,
        default=300,
        help='Maximum number of subjects to load'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=2,
        help='Override batch size (default: 2 for 3D)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        help='Override learning rate'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100000,
        help='Override number of epochs'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'auto'],
        default='auto',
        help='Device to use for training'
    )
    
    parser.add_argument(
        '--lambda-reg',
        type=float,
        default=0.5,
        help='Regularization weight'
    )
    
    parser.add_argument(
        '--loss-type',
        type=str,
        default='NewLNCC3D',
        choices=['NewLNCC3D', 'LNCC'],
        help='Loss function type'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Override config with command line arguments
    config.model.dimension = 3  # Force 3D mode
    config.training.batch_size = args.batch_size
    config.training.epochs = args.epochs
    config.loss.lambda_regularization = args.lambda_reg
    config.loss.type = args.loss_type
    
    if args.data_root:
        config.data.data_root = args.data_root
    
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    
    if args.device != 'auto':
        config.training.device = args.device
    elif args.device == 'auto':
        config.training.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Store max subjects in config
    config.data.max_subjects = args.max_subjects
    
    print("Starting 3D registration training with:")
    print(f"  Config: {args.config}")
    print(f"  Device: {config.training.device}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  Max subjects: {args.max_subjects}")
    print(f"  Loss type: {config.loss.type}")
    print(f"  Lambda regularization: {config.loss.lambda_regularization}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Train model
    try:
        loss_history = train_3d(config)
        print("Training completed successfully!")
        
        # Plot losses if matplotlib available
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].plot(loss_history['all_loss'])
            axes[0].set_title('Total Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].grid(True)
            
            axes[1].plot(loss_history['sim_loss'])
            axes[1].set_title('Similarity Loss')
            axes[1].set_xlabel('Epoch')
            axes[1].grid(True)
            
            axes[2].plot(loss_history['reg_loss'])
            axes[2].set_title('Regularization Loss')
            axes[2].set_xlabel('Epoch')
            axes[2].grid(True)
            
            plt.tight_layout()
            plt.savefig(Path(config.output.results_dir) / 'training_losses_3d.png')
            print(f"Loss plot saved to: {config.output.results_dir}/training_losses_3d.png")
            
        except ImportError:
            print("matplotlib not available, skipping loss plot")
        
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
