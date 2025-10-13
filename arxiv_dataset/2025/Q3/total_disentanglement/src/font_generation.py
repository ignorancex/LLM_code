#!/usr/bin/env python3
"""
Font Generation Script

This script generates new font characters by combining character features from one font 
with font features from another font.
"""

import os
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import cv2
from tqdm import tqdm
import pandas as pd
from pathlib import Path

# Add current directory to path for imports
import sys
sys.path.append('.')
from models import DISENTANGLE_MODEL

# Check if metrics module exists, if not define simple alternatives
try:
    from metrics import MSE, max_hausdorff_dist, chamfer_dist, IoU, L1
    METRICS_AVAILABLE = True
except ImportError:
    print("Metrics module not found, using simple alternatives")
    METRICS_AVAILABLE = False
    
    class SimpleMetric:
        def calc_dist(self, img1, img2):
            img1_np = img1.cpu().numpy() if torch.is_tensor(img1) else img1
            img2_np = img2.cpu().numpy() if torch.is_tensor(img2) else img2
            return np.mean((img1_np - img2_np) ** 2)
    
    MSE = lambda: SimpleMetric()
    L1 = lambda: SimpleMetric()
    max_hausdorff_dist = lambda: SimpleMetric()
    chamfer_dist = lambda: SimpleMetric()
    IoU = lambda: SimpleMetric()



def images_to_tensor(images, device):
    """Convert numpy images to torch tensor."""
    images_tensor = images[:, np.newaxis, :, :]
    return torch.from_numpy(images_tensor.astype(np.float32)).to(device)



def save_generation_plot(content_font, style_font, chars, content_images, style_images, 
                        generated_images, output_dir, show_plot=False):
    """Save generation visualization plot."""
    fig = plt.figure(figsize=(26, 3))
    fig.suptitle(f'Font Generation: {content_font} (character) + {style_font} (style)', fontsize=16)
    plt.gray()
    
    generated_np = generated_images.cpu().numpy()
    
    # Row 1: Content images (character features)
    for j in range(len(chars)):
        ax = fig.add_subplot(3, 26, j + 1)
        ax.imshow(content_images[j], cmap='gray')
        ax.set_title(chars[j], fontsize=8)
        ax.axis('off')
        if j == 0:
            ax.text(-0.1, 0.5, 'Content\n(character)', transform=ax.transAxes, 
                   rotation=90, va='center', ha='right', fontsize=10)
    
    # Row 2: Style images (font style)
    for j in range(len(chars)):
        ax = fig.add_subplot(3, 26, j + 27)
        ax.imshow(style_images[j], cmap='gray')
        ax.axis('off')
        if j == 0:
            ax.text(-0.1, 0.5, 'Style\n(reference)', transform=ax.transAxes, 
                   rotation=90, va='center', ha='right', fontsize=10)
    
    # Row 3: Generated images
    for j in range(len(chars)):
        ax = fig.add_subplot(3, 26, j + 53)
        ax.imshow(generated_np[j, 0, :, :], cmap='gray')
        ax.axis('off')
        if j == 0:
            ax.text(-0.1, 0.5, 'Generated', transform=ax.transAxes, 
                   rotation=90, va='center', ha='right', fontsize=10)
    
    plt.tight_layout()
    
    # Save visualization
    save_path = f"{output_dir}/generation_{content_font}_to_{style_font}.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Saved: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def main(args):
    """Main function to run font generation."""
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print(f"Using device: {args.device}")
    print(f"Model path: {args.model_path}")
    print(f"Content font directory: {args.content_dir}")
    print(f"Style reference font directory: {args.style_ref_dir}")
    print(f"Style reference character: {args.style_ref_char}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = DISENTANGLE_MODEL(
        args.zdim, 
        args.char_num, 
        args.batch_size, 
        args.device, 
        args.img_size
    ).to(args.device)

    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=args.device))
        print(f"✅ Model loaded from: {args.model_path}")
    else:
        print(f"❌ Model not found at: {args.model_path}")
        print("Please check the model path.")
        return
        
    model.eval()

    # Get character list
    chars = [chr(i + ord('A')) for i in range(args.char_num)]
    print(f"Characters: {chars}")

    # Verify directories exist
    if not os.path.exists(args.content_dir):
        print(f"❌ Content font directory not found: {args.content_dir}")
        return
    
    if not os.path.exists(args.style_ref_dir):
        print(f"❌ Style reference font directory not found: {args.style_ref_dir}")
        return
    
    # Verify style reference character file exists
    style_ref_path = os.path.join(args.style_ref_dir, f"{args.style_ref_char}.png")
    if not os.path.exists(style_ref_path):
        print(f"❌ Style reference character file not found: {style_ref_path}")
        return
    
    print(f"✅ Content directory: {args.content_dir}")
    print(f"✅ Style reference: {style_ref_path}")

    # Initialize distance metrics
    if METRICS_AVAILABLE:
        l2_distance = MSE()
        l1_distance = L1()
        haus_distance = max_hausdorff_dist()
        chamfer_distance = chamfer_dist()
        iou_distance = IoU()
        print("✅ All metrics initialized")
    else:
        l2_distance = MSE()
        l1_distance = L1()
        haus_distance = max_hausdorff_dist()
        chamfer_distance = chamfer_dist()
        iou_distance = IoU()
        print("⚠️ Using simplified metrics")

    # Initialize results storage
    results_data = {
        'character': [],
        'l2_dist': [],
        'l1_dist': [],
        'hausdorff_dist': [],
        'chamfer_dist': [],
        'iou_dist': [],
    }

    print("\nProcessing font generation...")
    
    with torch.no_grad():
        # Load content images (all characters from content font)
        print(f"Loading content images from: {args.content_dir}")
        content_images = []
        for ch in chars:
            img_path = os.path.join(args.content_dir, f'{ch}.png')
            if os.path.exists(img_path):
                img = cv2.imread(img_path, 0)
                if img is not None:
                    img = cv2.resize(img, (args.img_size, args.img_size))
                    img = img / 255.0
                    content_images.append(img)
                else:
                    content_images.append(np.zeros((args.img_size, args.img_size)))
            else:
                print(f"⚠️ Missing character: {img_path}")
                content_images.append(np.zeros((args.img_size, args.img_size)))
        
        content_images = np.array(content_images)
        content_tensor = images_to_tensor(content_images, args.device)
        
        # Load style reference image (single character)
        print(f"Loading style reference from: {style_ref_path}")
        style_img = cv2.imread(style_ref_path, 0)
        if style_img is not None:
            style_img = cv2.resize(style_img, (args.img_size, args.img_size))
            style_img = style_img / 255.0
        else:
            print(f"❌ Failed to load style reference image: {style_ref_path}")
            return
        
        # Create style tensor (replicate for all characters to extract font features)
        style_images = np.array([style_img] * len(chars))
        style_tensor = images_to_tensor(style_images, args.device)
        
        # Extract features
        print("Extracting character features from content font...")
        z_c_content, _, _ = model.encode(content_tensor)
        print("Extracting style features from reference character...")
        _, z_f_style, _ = model.encode(style_tensor)
        
        # Combine features: character from content + font style from style reference
        print("Combining features and generating new font...")
        z_combined = torch.cat((z_c_content, z_f_style), axis=1)
        
        # Generate new images
        generated_images = model.decode(z_combined)
        
        # Calculate metrics for each character
        print("Calculating metrics...")
        for char_idx, ch in enumerate(chars):
            style_img_tensor = style_tensor[char_idx, 0, :, :]
            generated_img = generated_images[char_idx, 0, :, :]
            
            # Calculate distances (comparing generated with style reference)
            l2_dist = l2_distance.calc_dist(style_img_tensor, generated_img)
            l1_dist = l1_distance.calc_dist(style_img_tensor, generated_img)
            haus_dist = haus_distance.calc_dist(style_img_tensor, generated_img)
            cham_dist = chamfer_distance.calc_dist(style_img_tensor, generated_img)
            iou_dist = iou_distance.calc_dist(style_img_tensor, generated_img)
            
            # Store results
            results_data['character'].append(ch)
            results_data['l2_dist'].append(float(l2_dist))
            results_data['l1_dist'].append(float(l1_dist))
            results_data['hausdorff_dist'].append(float(haus_dist))
            results_data['chamfer_dist'].append(float(cham_dist))
            results_data['iou_dist'].append(float(iou_dist))
        
        # Always save individual generated character images
        print("Saving individual character images...")
        content_font_name = os.path.basename(args.content_dir)
        style_font_name = os.path.basename(args.style_ref_dir)
        
        # Create subdirectory for individual images
        images_dir = os.path.join(args.output_dir, f"generated_{content_font_name}_to_{style_font_name}_{args.style_ref_char}")
        os.makedirs(images_dir, exist_ok=True)
        
        # Save each generated character as individual PNG
        generated_np = generated_images.cpu().numpy()
        for char_idx, ch in enumerate(chars):
            img = generated_np[char_idx, 0, :, :]  # Remove channel dimension
            img = (img * 255).astype(np.uint8)    # Convert to 0-255 range
            img_path = os.path.join(images_dir, f"{ch}.png")
            cv2.imwrite(img_path, img)
        
        print(f"✅ Saved {len(chars)} character images to: {images_dir}")

        # Create visualization
        # Load all style images (A-Z) for visualization
            style_all_images = []
            for ch in chars:
                img_path = os.path.join(args.style_ref_dir, f'{ch}.png')
                if os.path.exists(img_path):
                    img = cv2.imread(img_path, 0)
                    if img is not None:
                        img = cv2.resize(img, (args.img_size, args.img_size))
                        img = img / 255.0
                        style_all_images.append(img)
                    else:
                        style_all_images.append(np.zeros((args.img_size, args.img_size)))
                else:
                    style_all_images.append(np.zeros((args.img_size, args.img_size)))
            
            style_all_images = np.array(style_all_images)
            save_generation_plot(content_font_name, f"{style_font_name}_{args.style_ref_char}", 
                               chars, content_images, style_all_images, 
                               generated_images, args.output_dir, args.show_plots)

    # Create and save results DataFrame
    print("\nSaving results...")
    results_df = pd.DataFrame(results_data)

    # Save results to CSV with consistent naming
    content_font_name = os.path.basename(args.content_dir)
    style_font_name = os.path.basename(args.style_ref_dir)
    csv_path = f"{args.output_dir}/metrics_{content_font_name}_{style_font_name}_{args.style_ref_char}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"✅ Results saved to: {csv_path}")

    # Display summary statistics
    print("\n" + "="*50)
    print("GENERATION METRICS SUMMARY")
    print("="*50)

    if len(results_df) > 0:
        metric_columns = ['l2_dist', 'l1_dist', 'hausdorff_dist', 'chamfer_dist', 'iou_dist']
        summary_stats = results_df[metric_columns].describe()
        print(summary_stats)

        # Show character-wise metrics
        print("\nCharacter-wise metrics (lower is better):")
        for metric in metric_columns:
            print(f"\n{metric.replace('_', ' ').title()}:")
            char_stats = results_df.groupby('character')[metric].mean().sort_values()
            for char, value in char_stats.items():
                print(f"  {char}: {value:.4f}")
        
        # Overall statistics
        print(f"\nOverall average metrics:")
        for metric in metric_columns:
            avg_val = results_df[metric].mean()
            print(f"  {metric.replace('_', ' ').title()}: {avg_val:.4f}")


    # Print final summary
    print("\n" + "="*50)
    print("FONT GENERATION SUMMARY")
    print("="*50)
    content_font_name = os.path.basename(args.content_dir)
    style_font_name = os.path.basename(args.style_ref_dir)
    print(f"Content font directory: {args.content_dir}")
    print(f"Style reference: {args.style_ref_dir} (character: {args.style_ref_char})")
    print(f"Model: {args.model_path}")
    print(f"Total characters generated: {len(results_df)}")
    print(f"Output directory: {args.output_dir}")
    print(f"\nGenerated files:")
    print(f"  - Individual character images: generated_{content_font_name}_to_{style_font_name}_{args.style_ref_char}/ (26 PNG files)")
    print(f"  - Generation visualization: generation_{content_font_name}_to_{style_font_name}_{args.style_ref_char}.png")
    print(f"  - Metrics CSV: metrics_{content_font_name}_{style_font_name}_{args.style_ref_char}.csv")
    
    if len(results_df) > 0:
        print("\nAverage L2 distance (lower is better):")
        print(f"  Overall: {results_df['l2_dist'].mean():.4f}")
        char_stats = results_df.groupby('character')['l2_dist'].mean()
        print(f"  Best character: {char_stats.idxmin()} ({char_stats.min():.4f})")
        print(f"  Worst character: {char_stats.idxmax()} ({char_stats.max():.4f})")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate new fonts by combining features")
    
    # Model parameters
    parser.add_argument('--model_path', type=str, default='checkpoints/ICDAR2025_finetuning/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--zdim', type=int, default=256, help='Latent space dimension')
    parser.add_argument('--char_num', type=int, default=26, help='Number of characters')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--img_size', type=int, default=64, help='Image size')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    
    # Dataset parameters
    parser.add_argument('--content_dir', type=str, default='sample_data/test/YsabeauInfant-Italic[wght]', 
                       help='Directory of content font (for character features)')
    parser.add_argument('--style_ref_dir', type=str, default='sample_data/test/BreeSerif-Regular', 
                       help='Directory of style reference font (for style features)')
    parser.add_argument('--style_ref_char', type=str, default='A', 
                       help='Character to use as style reference (e.g., A, B, C, ...)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='results/font_generation/ICDAR2025_finetuning',
                       help='Output directory')
    parser.add_argument('--show_plots', action='store_true',
                       help='Show plots interactively')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    
    args = parser.parse_args()
    main(args)