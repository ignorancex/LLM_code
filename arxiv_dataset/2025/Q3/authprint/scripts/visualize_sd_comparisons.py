#!/usr/bin/env python
"""
Script to visualize and compare images generated from Stable Diffusion model and its negative cases.
Takes fixed prompts and generates images using original SD model and all negative cases.
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
from models.stable_diffusion_model import StableDiffusionModel
from utils.model_loading import load_pretrained_models
from utils.image_transforms import (
    downsample_and_upsample,
    apply_jpeg_compression
)
from utils.logging_utils import setup_logging


def is_perfect_square(n):
    """Check if a number is a perfect square."""
    root = int(math.sqrt(n))
    return root * root == n


def tensor_to_pil(tensor):
    """Convert a tensor to a PIL Image with proper RGB handling."""
    # Convert from [0, 1] to [0, 255]
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
    parser = argparse.ArgumentParser(description="Visualize comparisons between original SD model and negative cases")
    
    # Model configuration
    parser.add_argument("--sd_model_name", type=str,
                        default="stabilityai/stable-diffusion-2-1",
                        help="Name of the Stable Diffusion model to use")
    parser.add_argument("--sd_enable_cpu_offload", action="store_true",
                        help="Enable CPU offloading for Stable Diffusion")
    parser.add_argument("--img_size", type=int, default=1024,
                        help="Image resolution")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="sd_comparison_images",
                        help="Directory to save comparison images")
    parser.add_argument("--num_samples", type=int, default=9,
                        help="Number of different prompts to generate comparisons for (must be a perfect square)")
    parser.add_argument("--prompt", type=str,
                        default="A photorealistic advertisement poster for a Japanese cafe named 'NOVA CAFE', with the name written clearly in both English and Japanese on a street sign, a storefront banner, and a coffee cup. The scene is set at night with neon lighting, rain-slick streets reflecting the glow, and people walking by in motion blur. Cinematic tone, Leica photo quality, ultra-detailed textures.",
                        help="Base prompt for the Stable Diffusion model")
    parser.add_argument("--num_inference_steps", type=int, default=25,
                        help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Classifier-free guidance scale")
    
    args = parser.parse_args()
    
    if not is_perfect_square(args.num_samples):
        raise ValueError(f"num_samples must be a perfect square (e.g., 4, 9, 16). Got {args.num_samples}")
    
    return args


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
    
    # Initialize original model
    original_model = StableDiffusionModel(
        model_name=args.sd_model_name,
        device=device,
        img_size=args.img_size,
        enable_cpu_offload=args.sd_enable_cpu_offload
    )
    
    # Load pretrained models
    pretrained_models = load_pretrained_models(
        device=device,
        rank=0,
        model_type="stable-diffusion",
        img_size=args.img_size,
        enable_cpu_offload=args.sd_enable_cpu_offload
    )
    
    # Generate prompts for all samples
    prompts = [args.prompt] * args.num_samples
    grid_size = int(math.sqrt(args.num_samples))
    
    # Dictionary to store images for each case
    case_images = {
        'original': [],
        **{f'pretrained_{name}': [] for name in pretrained_models.keys()},
        **{f'downsampled_{size}': [] for size in [16, 224]},
        'jpeg_compressed': []
    }
    
    # First, generate all original images with fixed seeds
    original_images = []
    for i, prompt in enumerate(prompts):
        # Set a specific seed for each image
        torch.manual_seed(args.seed + i)
        np.random.seed(args.seed + i)
        
        original_img = original_model.generate_images(
            batch_size=1,
            prompt=prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale
        )[0]
        original_images.append(original_img)
        case_images['original'].append(original_img)
    
    # Now process each original image through the different cases
    for i, original_img in enumerate(original_images):
        # Set the same seed as used for the original image
        torch.manual_seed(args.seed + i)
        np.random.seed(args.seed + i)
        
        # Pretrained models (using the same seed)
        for model_name, model in pretrained_models.items():
            img = model.generate_images(
                batch_size=1,
                prompt=prompts[i],
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale
            )[0]
            case_images[f'pretrained_{model_name}'].append(img)
        
        # Downsampled images (using the same original image)
        for size in [16, 224]:
            downsampled_img = downsample_and_upsample(original_img.unsqueeze(0), downsample_size=size)[0]
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
        f.write(f"Base prompt: {args.prompt}\n")
        f.write(f"Image size: {args.img_size}\n")
        f.write(f"Number of inference steps: {args.num_inference_steps}\n")
        f.write(f"Guidance scale: {args.guidance_scale}\n")
        f.write(f"Base seed: {args.seed}\n")
        f.write("\nCases generated:\n")
        for case_name in case_images.keys():
            f.write(f"- {case_name}\n")
    
    logging.info("Visualization completed successfully")


if __name__ == "__main__":
    while True:
        try:
            main()
            print("\nCompleted one visualization run. Starting next run...\n")
        except KeyboardInterrupt:
            print("\nStopping visualization loop...")
            break
        except Exception as e:
            print(f"\nError during visualization: {e}")
            print("Restarting visualization loop...\n")
            continue 