#!/usr/bin/env python3
"""
Feature Visualization Script

This script visualizes font and character features extracted by the disentanglement model using PCA.
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
from sklearn.decomposition import PCA
import matplotlib
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pathlib import Path

# Add current directory to path for imports
import sys
sys.path.append('.')
from models import DISENTANGLE_MODEL
from dataset import load_pretraining_data


def get_colors():
    """Generate a diverse set of colors for visualization."""
    colors = []
    cmap_names = ['tab20b', 'tab20']
    for cmap_name in cmap_names:
        try:
            cm = matplotlib.colormaps[cmap_name]  # New matplotlib API
        except AttributeError:
            cm = plt.cm.get_cmap(cmap_name)  # Fallback for older matplotlib
        for rgb in cm.colors:
            colors.append(rgb)
    return list(set(colors))


def imscatter(x, y, image_list, ax=None, zoom=0.2, color='black'):
    """Scatter plot with images instead of points."""
    if ax is None:
        ax = plt.gca()
    
    artists = []
    for i in range(len(image_list)):
        image = np.array(image_list[i], dtype=np.uint8)
        im = OffsetImage(image, zoom=zoom)
        x0, y0 = x[i], y[i]
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=True, 
                          bboxprops=dict(color=color[i]))
        artists.append(ax.add_artist(ab))
    return artists


def load_sample_images(dataset_path, max_fonts=30):
    """Load sample images for visualization."""
    chars = [chr(i + ord('A')) for i in range(26)]
    
    img_list = []
    ch_list = []
    font_list = []
    font_names = []
    
    # Use the user-provided dataset path directly
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path not found: {dataset_path}")
        return [], [], [], []
        
    font_dirs = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    
    font_dirs = font_dirs[:max_fonts]  # Limit number of fonts
    
    print(f"Loading {len(font_dirs)} fonts from {dataset_path}")
    
    for font_idx, font_name in enumerate(tqdm(font_dirs)):
        for char_idx, ch in enumerate(chars):
            img_path = os.path.join(dataset_path, font_name, f'{ch}.png')
            
            if os.path.exists(img_path):
                img = cv2.imread(img_path, 0)
                if img is not None:
                    img = cv2.resize(img, (64, 64))
                    img_list.append(img)
                    ch_list.append(char_idx)
                    font_list.append(font_idx)
                    font_names.append(font_name)
    
    print(f"Loaded {len(img_list)} images from {len(set(font_names))} fonts")
    return img_list, ch_list, font_list, font_names


def main(args):
    """Main function to run feature visualization."""
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print(f"Using device: {args.device}")
    print(f"Model path: {args.model_path}")
    print(f"Dataset: {args.dataset}")
    
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
    
    # Load images
    print("\nLoading images...")
    img_list, ch_list, font_list, font_names = load_sample_images(args.dataset, args.max_fonts)

    if len(img_list) == 0:
        print("❌ No images loaded. Please check the dataset configuration.")
        return

    # Generate colors
    colors = get_colors()
    font_colors = [colors[i % len(colors)] for i in font_list]
    char_colors = [colors[i % len(colors)] for i in ch_list]

    print(f"Generated {len(colors)} colors for visualization")

    # Extract features
    print("\nExtracting features...")
    char_feat_list = []
    font_feat_list = []

    with torch.no_grad():
        for img in tqdm(img_list):
            # Normalize image
            img_norm = img / 255.0
            img_torch = torch.from_numpy(img_norm.astype(np.float32)).clone()
            x = img_torch.reshape(1, 1, 64, 64).to(args.device, torch.float32)
            
            # Extract features
            z_c, z_f, _ = model.encode(x)
            
            font_feat_list.append(z_f.cpu().numpy().flatten())
            char_feat_list.append(z_c.cpu().numpy().flatten())

    print(f"Extracted {len(char_feat_list)} character features")
    print(f"Extracted {len(font_feat_list)} font features")

    # Apply dimensionality reduction using PCA
    print(f"\nApplying PCA for dimensionality reduction...")

    char_reducer = PCA(n_components=2)
    font_reducer = PCA(n_components=2)

    char_feat_2d = char_reducer.fit_transform(char_feat_list)
    font_feat_2d = font_reducer.fit_transform(font_feat_list)

    print(f"Reduced to 2D: char_feat {char_feat_2d.shape}, font_feat {font_feat_2d.shape}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nGenerating visualizations...")

    # Generate scatter plots with images (if requested)
    if args.show_images:
        # Font-colored version with images
        fig, ax = plt.subplots(figsize=(20, 20))
        plt.gray()
        imscatter(font_feat_2d[:, 0], char_feat_2d[:, 0], img_list, ax=ax, zoom=0.2, color=font_colors)
        ax.plot(font_feat_2d[:, 0], char_feat_2d[:, 0], 'ko', alpha=0.0)
        ax.set_title(f'Feature Visualization with Images (PCA) - Colored by Font', fontsize=16)
        ax.set_xlabel('Font Feature Dimension 1', fontsize=14)
        ax.set_ylabel('Character Feature Dimension 1', fontsize=14)
        save_path = f"{args.output_dir}/font_pca_images.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")
        if args.show_plots:
            plt.show()
        else:
            plt.close()

        # Character-colored version with images
        fig, ax = plt.subplots(figsize=(20, 20))
        plt.gray()
        imscatter(font_feat_2d[:, 0], char_feat_2d[:, 0], img_list, ax=ax, zoom=0.2, color=char_colors)
        ax.plot(font_feat_2d[:, 0], char_feat_2d[:, 0], 'ko', alpha=0.0)
        ax.set_title(f'Feature Visualization with Images (PCA) - Colored by Character', fontsize=16)
        ax.set_xlabel('Font Feature Dimension 1', fontsize=14)
        ax.set_ylabel('Character Feature Dimension 1', fontsize=14)
        save_path = f"{args.output_dir}/char_pca_images.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")
        if args.show_plots:
            plt.show()
        else:
            plt.close()

    # Generate scatter plots with dots (always generated)
    # Font-colored dot plot
    fig, ax = plt.subplots(figsize=(12, 12))
    for i in range(len(img_list)):
        ax.scatter(font_feat_2d[i, 0], char_feat_2d[i, 0], color=font_colors[i], s=50, alpha=0.7)
    ax.set_title(f'Feature Space (PCA) - Colored by Font', fontsize=16)
    ax.set_xlabel('Font Feature Dimension 1', fontsize=14)
    ax.set_ylabel('Character Feature Dimension 1', fontsize=14)
    ax.grid(True, alpha=0.3)
    save_path = f"{args.output_dir}/font_pca_dots.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {save_path}")
    if args.show_plots:
        plt.show()
    else:
        plt.close()

    # Character-colored dot plot
    fig, ax = plt.subplots(figsize=(12, 12))
    for i in range(len(img_list)):
        ax.scatter(font_feat_2d[i, 0], char_feat_2d[i, 0], color=char_colors[i], s=50, alpha=0.7)
    ax.set_title(f'Feature Space (PCA) - Colored by Character', fontsize=16)
    ax.set_xlabel('Font Feature Dimension 1', fontsize=14)
    ax.set_ylabel('Character Feature Dimension 1', fontsize=14)
    ax.grid(True, alpha=0.3)
    save_path = f"{args.output_dir}/char_pca_dots.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {save_path}")
    if args.show_plots:
        plt.show()
    else:
        plt.close()

    # Comparison plot (both font and character coloring side by side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))

    # Font-colored plot
    for i in range(len(img_list)):
        ax1.scatter(font_feat_2d[i, 0], char_feat_2d[i, 0], color=font_colors[i], s=50, alpha=0.7)
    ax1.set_title(f'Feature Space - Colored by Font (PCA)', fontsize=14)
    ax1.set_xlabel('Font Feature Dimension 1')
    ax1.set_ylabel('Character Feature Dimension 1')
    ax1.grid(True, alpha=0.3)

    # Character-colored plot
    for i in range(len(img_list)):
        ax2.scatter(font_feat_2d[i, 0], char_feat_2d[i, 0], color=char_colors[i], s=50, alpha=0.7)
    ax2.set_title(f'Feature Space - Colored by Character (PCA)', fontsize=14)
    ax2.set_xlabel('Font Feature Dimension 1')
    ax2.set_ylabel('Character Feature Dimension 1')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = f"{args.output_dir}/comparison_pca.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {save_path}")
    if args.show_plots:
        plt.show()
    else:
        plt.close()

    # Print summary
    print("\n" + "="*50)
    print("FEATURE VISUALIZATION SUMMARY")
    print("="*50)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model_path}")
    print(f"Reduction method: PCA")
    print(f"Total images processed: {len(img_list)}")
    print(f"Unique fonts: {len(set(font_list))}")
    print(f"Unique characters: {len(set(ch_list))}")
    print(f"Output directory: {args.output_dir}")
    print(f"Generated plots:")
    if args.show_images:
        print(f"  - Font-colored images scatter plot")
        print(f"  - Character-colored images scatter plot")
    print(f"  - Font-colored dots scatter plot")
    print(f"  - Character-colored dots scatter plot")
    print(f"  - Comparison plot (both colorings side by side)")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize font and character features")
    
    # Model parameters
    parser.add_argument('--model_path', type=str, default='checkpoints/ICDAR2025_finetuning/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--zdim', type=int, default=256, help='Latent space dimension')
    parser.add_argument('--char_num', type=int, default=26, help='Number of characters')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--img_size', type=int, default=64, help='Image size')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='sample_data/test', 
                       help='Path to dataset directory containing font subdirectories')
    parser.add_argument('--max_fonts', type=int, default=30, help='Maximum number of fonts to process')
    
    # Visualization parameters
    parser.add_argument('--output_dir', type=str, default='results/feature_visualization/ICDAR2025_finetuning/',
                       help='Output directory')
    parser.add_argument('--show_images', action='store_true',
                       help='Generate scatter plot with actual images')
    parser.add_argument('--show_plots', action='store_true',
                       help='Show plots interactively')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    
    args = parser.parse_args()
    main(args)