#!/usr/bin/env python
"""
Script to visualize and compare images generated from different models and transformations.
Takes fixed latent vectors and generates images using original model and all negative cases.
Each case's images are combined into a single grid visualization.
"""
import argparse
import logging
import os
import sys
import math
from pathlib import Path

import torch
import torchvision.utils as vutils
import numpy as np
from PIL import Image, ImageDraw

# Add the parent directory (project root) to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.default_config import get_default_config
from models.model_utils import load_stylegan2_model
from utils.model_loading import load_pretrained_models
from utils.image_transforms import (
    quantize_model_weights,
    downsample_and_upsample
)
from utils.logging_utils import setup_logging


def is_perfect_square(n):
    """Check if a number is a perfect square."""
    root = int(math.sqrt(n))
    return root * root == n


def tensor_to_pil(tensor):
    """Convert a tensor to a PIL Image with proper RGB handling."""
    # Convert from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2
    # Clamp values to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    # Convert to PIL Image
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return Image.fromarray(ndarr)


def create_grid_with_borders(images, nrow):
    """Create a grid of images with black borders."""
    # Convert tensors to PIL images
    pil_images = [tensor_to_pil(img) for img in images]
    
    # Get dimensions
    w, h = pil_images[0].size
    margin = 2  # Border width
    grid_size = int(math.sqrt(len(images)))
    
    # Create new image with space for borders
    grid_w = grid_size * (w + margin) + margin
    grid_h = grid_size * (h + margin) + margin
    grid_img = Image.new('RGB', (grid_w, grid_h), 'black')
    
    # Paste images
    for idx, img in enumerate(pil_images):
        row = idx // grid_size
        col = idx % grid_size
        x = col * (w + margin) + margin
        y = row * (h + margin) + margin
        grid_img.paste(img, (x, y))
    
    return grid_img


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize comparisons between original and negative cases")
    
    # Model configuration
    parser.add_argument("--stylegan2_url", type=str,
                        default="https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq70k-paper256-ada.pkl",
                        help="URL for the pretrained StyleGAN2 model")
    parser.add_argument("--stylegan2_local_path", type=str,
                        default="ffhq70k-paper256-ada.pkl",
                        help="Local path to store/load the StyleGAN2 model")
    parser.add_argument("--img_size", type=int, default=256,
                        help="Image resolution")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="comparison_images",
                        help="Directory to save comparison images")
    parser.add_argument("--num_samples", type=int, default=9,
                        help="Number of different z vectors to generate comparisons for (must be a perfect square)")
    
    args = parser.parse_args()
    
    if not is_perfect_square(args.num_samples):
        raise ValueError(f"num_samples must be a perfect square (e.g., 4, 9, 16). Got {args.num_samples}")
    
    return args


def generate_image(model, z, noise_mode="const"):
    """Generate an image using the given model and latent vector."""
    with torch.no_grad():
        if hasattr(model, 'module'):
            w = model.module.mapping(z, None)
            img = model.module.synthesis(w, noise_mode=noise_mode)
        else:
            w = model.mapping(z, None)
            img = model.synthesis(w, noise_mode=noise_mode)
    return img


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
    
    # Load original model
    original_model = load_stylegan2_model(
        args.stylegan2_url,
        args.stylegan2_local_path,
        device
    ).eval()
    
    # Load pretrained models
    pretrained_models = load_pretrained_models(device, 0)
    
    # Create quantized models
    quantized_models = {}
    try:
        quantized_models['int8'] = quantize_model_weights(original_model, 'int8').eval()
        logging.info("Successfully created int8 quantized model")
    except Exception as e:
        logging.error(f"Failed to create int8 model: {str(e)}")
    
    try:
        quantized_models['int4'] = quantize_model_weights(original_model, 'int4').eval()
        logging.info("Successfully created int4 quantized model")
    except Exception as e:
        logging.error(f"Failed to create int4 model: {str(e)}")
    
    # Generate latent vectors for all samples
    z_vectors = [torch.randn(1, original_model.z_dim, device=device) for _ in range(args.num_samples)]
    grid_size = int(math.sqrt(args.num_samples))
    
    # Dictionary to store images for each case
    case_images = {
        'original': [],
        **{f'pretrained_{name}': [] for name in pretrained_models.keys()},
        **{f'quantized_{name}': [] for name in quantized_models.keys()},
        **{f'downsampled_{size}': [] for size in [128, 224]}
    }
    
    # Generate images for all cases
    for z in z_vectors:
        # Original model
        original_img = generate_image(original_model, z)[0]
        case_images['original'].append(original_img)
        
        # Pretrained models
        for model_name, model in pretrained_models.items():
            img = generate_image(model, z)[0]
            case_images[f'pretrained_{model_name}'].append(img)
        
        # Quantized models
        for quant_name, model in quantized_models.items():
            img = generate_image(model, z)[0]
            case_images[f'quantized_{quant_name}'].append(img)
        
        # Downsampled images
        for size in [128, 224]:
            img = generate_image(original_model, z)[0]
            downsampled_img = downsample_and_upsample(img.unsqueeze(0), downsample_size=size)[0]
            case_images[f'downsampled_{size}'].append(downsampled_img)
    
    # Create and save grid for each case
    for case_name, images in case_images.items():
        if images:  # Only process if we have images for this case
            grid_img = create_grid_with_borders(images, grid_size)
            grid_img.save(output_dir / f"{case_name}_grid.png")
            logging.info(f"Saved grid for {case_name}")
    
    # Save configuration
    with open(output_dir / "visualization_config.txt", "w") as f:
        f.write(f"Number of samples: {args.num_samples}\n")
        f.write(f"Grid size: {grid_size}x{grid_size}\n")
        f.write("\nCases generated:\n")
        for case_name in case_images.keys():
            f.write(f"- {case_name}\n")
    
    logging.info("Visualization completed successfully")

    # Sleep indefinitely
    while True:
        import time
        time.sleep(1)

if __name__ == "__main__":
    main() 