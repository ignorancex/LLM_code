#!/usr/bin/env python
"""
Script to visualize samples from Stable Diffusion model.
Generates a 4x4 grid of images using the same configuration as training.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add the parent directory (project root) to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.stable_diffusion_model import StableDiffusionModel
from utils.logging_utils import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize samples from Stable Diffusion")
    
    # Model configuration
    parser.add_argument("--sd_model_name", type=str,
                        default="stabilityai/stable-diffusion-2-1-base",
                        help="Name of the Stable Diffusion model to use")
    parser.add_argument("--sd_enable_cpu_offload", action="store_true",
                        help="Enable CPU offloading for Stable Diffusion")
    parser.add_argument("--img_size", type=int, default=768,
                        help="Size of generated images")
    parser.add_argument("--output_dir", type=str, default="sd_samples",
                        help="Directory to save visualization results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--prompt", type=str,
                        default="A photo of a cat in a variety of real-world scenes, candid shot, natural lighting, diverse settings, DSLR photo",
                        help="Prompt for the Stable Diffusion model")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Classifier-free guidance scale")
    
    args = parser.parse_args()
    return args


def create_image_grid(images: list, rows: int = 4, cols: int = 4) -> Image.Image:
    """
    Create a grid of images.
    
    Args:
        images (list): List of PIL Images
        rows (int): Number of rows in grid
        cols (int): Number of columns in grid
        
    Returns:
        Image.Image: Grid of images as a single PIL Image
    """
    assert len(images) == rows * cols
    
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    
    for idx, image in enumerate(images):
        i = idx // cols
        j = idx % cols
        grid.paste(image, box=(j * w, i * h))
    
    return grid


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
    
    # Initialize model
    model = StableDiffusionModel(
        model_name=args.sd_model_name,
        device=device,
        img_size=args.img_size,
        enable_cpu_offload=args.sd_enable_cpu_offload
    )
    
    # Generate images
    logging.info("Generating images...")
    images = model.generate_images(
        batch_size=16,  # 4x4 grid
        prompt=args.prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale
    )
    
    # Convert to PIL images
    pil_images = []
    for i in range(images.shape[0]):
        # Convert from [0,1] to [0,255]
        img = images[i].permute(1, 2, 0).cpu().numpy() * 255
        img = img.astype(np.uint8)
        pil_images.append(Image.fromarray(img))
    
    # Create and save grid
    logging.info("Creating image grid...")
    grid = create_image_grid(pil_images)
    
    # Save grid
    output_path = output_dir / f"sd_samples_grid_{args.img_size}px.png"
    grid.save(output_path)
    logging.info(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main() 