#!/usr/bin/env python
"""
Script to visualize the pipeline from latent vectors to selected pixels.
For each sample, visualizes:
1. The latent vector z (512-dim) as a heatmap
2. The generated image x
3. The selected pixels p as a heatmap
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw

# Add the parent directory (project root) to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.default_config import get_default_config
from models.model_utils import load_stylegan2_model
from utils.logging_utils import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize the pipeline from latent vectors to selected pixels")
    
    # Model configuration
    parser.add_argument("--stylegan2_url", type=str,
                        default="https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq70k-paper256-ada.pkl",
                        help="URL for the pretrained StyleGAN2 model")
    parser.add_argument("--stylegan2_local_path", type=str,
                        default="ffhq70k-paper256-ada.pkl",
                        help="Local path to store/load the StyleGAN2 model")
    parser.add_argument("--output_dir", type=str, default="pipeline_visualization",
                        help="Directory to save visualization results")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to generate")
    parser.add_argument("--num_pixels", type=int, default=16,
                        help="Number of pixels to select")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    return args


def generate_pixel_indices(img_size: int, num_pixels: int, seed: int) -> torch.Tensor:
    """
    Generate random pixel indices for visualization.
    
    Args:
        img_size (int): Size of the image (assuming square)
        num_pixels (int): Number of pixels to select
        seed (int): Random seed for reproducibility
        
    Returns:
        torch.Tensor: Selected pixel indices
    """
    torch.manual_seed(seed)
    
    # Calculate total number of pixels
    channels = 3  # RGB image
    total_pixels = channels * img_size * img_size
    
    # Generate random indices (without replacement)
    indices = torch.randperm(total_pixels)[:num_pixels]
    
    return indices


def visualize_latent_vector(z: torch.Tensor, output_path: str):
    """
    Visualize a latent vector as a heatmap.
    
    Args:
        z (torch.Tensor): Latent vector [1, 512]
        output_path (str): Path to save the visualization
    """
    # Convert to numpy and reshape to 2D grid (16x32)
    z_np = z.cpu().numpy().reshape(16, 32)
    
    # Create figure with specific size
    plt.figure(figsize=(10, 5))
    
    # Create heatmap with improved aesthetics
    sns.heatmap(z_np, cmap='RdBu_r', center=0, 
                xticklabels=False, yticklabels=False, 
                cbar=False)
    
    plt.tight_layout()
    
    # Save and close
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a tensor to a PIL Image with proper RGB handling."""
    # Convert from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2
    # Clamp values to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    # Convert to PIL Image
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return Image.fromarray(ndarr)


def visualize_selected_pixels(pixel_values: torch.Tensor, output_path: str):
    """
    Visualize selected pixel values as a heatmap.
    
    Args:
        pixel_values (torch.Tensor): Selected pixel values [num_pixels]
        output_path (str): Path to save the visualization
    """
    # Convert to numpy and reshape to square-like grid
    num_pixels = len(pixel_values)
    grid_size = int(np.ceil(np.sqrt(num_pixels)))
    
    # Pad with zeros to make perfect square
    values_np = pixel_values.cpu().numpy()
    padding = grid_size * grid_size - num_pixels
    if padding > 0:
        values_np = np.pad(values_np, (0, padding), mode='constant', constant_values=np.nan)
    
    values_grid = values_np.reshape(grid_size, grid_size)
    
    # Create figure
    plt.figure(figsize=(5, 5))
    
    # Create heatmap with improved aesthetics
    sns.heatmap(values_grid, cmap='RdBu_r', 
                xticklabels=False, yticklabels=False,
                cbar=False,
                mask=np.isnan(values_grid))
    
    plt.tight_layout()
    
    # Save and close
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


def extract_pixels(image: torch.Tensor, pixel_indices: torch.Tensor) -> torch.Tensor:
    """
    Extract pixel values at selected indices.
    
    Args:
        image (torch.Tensor): Input image [1, C, H, W]
        pixel_indices (torch.Tensor): Selected pixel indices
        
    Returns:
        torch.Tensor: Selected pixel values
    """
    batch_size = image.shape[0]
    flattened = image.view(batch_size, -1)
    pixel_values = flattened.index_select(1, pixel_indices)
    return pixel_values[0]  # Return first batch element


def main():
    """Main entry point for visualization."""
    args = parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(args.output_dir, 0)
    logging.info(f"Configuration:\n{args}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load model
    model = load_stylegan2_model(
        args.stylegan2_url,
        args.stylegan2_local_path,
        device
    ).eval()
    
    # Generate pixel indices
    img_size = 256  # Assuming 256x256 images
    pixel_indices = generate_pixel_indices(img_size, args.num_pixels, args.seed)
    pixel_indices = pixel_indices.to(device)
    
    # Generate and visualize samples
    for sample_idx in range(args.num_samples):
        # Create sample directory
        sample_dir = output_dir / f"sample_{sample_idx:02d}"
        sample_dir.mkdir(exist_ok=True)
        
        # Generate latent vector
        z = torch.randn(1, model.z_dim, device=device)
        
        # Generate image
        with torch.no_grad():
            if hasattr(model, 'module'):
                w = model.module.mapping(z, None)
                x = model.module.synthesis(w, noise_mode="const")
            else:
                w = model.mapping(z, None)
                x = model.synthesis(w, noise_mode="const")
        
        # Extract selected pixels
        pixel_values = extract_pixels(x, pixel_indices)
        
        # Save visualizations
        visualize_latent_vector(z[0], str(sample_dir / "z.png"))
        tensor_to_pil(x[0]).save(str(sample_dir / "x.png"))
        visualize_selected_pixels(pixel_values, str(sample_dir / "p.png"))
        
        logging.info(f"Generated visualizations for sample {sample_idx}")
    
    logging.info("Visualization completed successfully")


if __name__ == "__main__":
    main() 