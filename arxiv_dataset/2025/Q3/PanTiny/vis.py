#!/usr/bin/env python3
"""
Enhanced visualization script for pan-sharpening results
Creates comparison figures with improved layout and color handling
"""

import os
import sys
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import glob
from pathlib import Path

def load_image(image_path):
    """Load image and convert to RGB numpy array with proper color handling"""
    try:
        if not os.path.exists(image_path):
            return None

        # Try different loading methods for different file types
        img_array = None

        # Try TIFF loading first (common in scientific images)
        if image_path.lower().endswith(('.tif', '.tiff')):
            try:
                import tifffile
                img_array = tifffile.imread(image_path)
            except ImportError:
                # Fall back to PIL if tifffile not available
                pass
            except Exception as e:
                print(f"TIFF loading failed for {image_path}: {e}")

        # If TIFF loading failed or not a TIFF, use PIL
        if img_array is None:
            try:
                img = Image.open(image_path)

                # Handle different color modes
                if img.mode == 'CMYK':
                    img = img.convert('RGB')
                elif img.mode == 'L':
                    img = img.convert('RGB')
                elif img.mode in ['RGBA', 'LA']:
                    img = img.convert('RGB')
                elif img.mode not in ['RGB', 'L']:
                    # Convert any other mode to RGB
                    img = img.convert('RGB')

                img_array = np.array(img)
            except Exception as e:
                print(f"PIL loading failed for {image_path}: {e}")
                return None

        # Validate array
        if img_array is None or not isinstance(img_array, np.ndarray):
            print(f"Failed to load image array for {image_path}")
            return None

        # Validate array shape
        if len(img_array.shape) < 2:
            print(f"Invalid image shape: {img_array.shape} for {image_path}")
            return None

        # Handle different array shapes
        if len(img_array.shape) == 2:
            # Grayscale to RGB
            img_array = np.stack([img_array] * 3, axis=2)
        elif len(img_array.shape) == 3:
            # Handle multi-channel images
            if img_array.shape[2] > 4:
                # Too many channels, take first 3
                img_array = img_array[:, :, :3]
            elif img_array.shape[2] == 4:
                # RGBA, remove alpha
                img_array = img_array[:, :, :3]
            elif img_array.shape[2] == 1:
                # Single channel, convert to RGB
                img_array = np.repeat(img_array, 3, axis=2)
            elif img_array.shape[2] == 2:
                # Two channels, duplicate first channel
                img_array = np.concatenate([img_array, img_array[:, :, :1]], axis=2)
        else:
            print(f"Unexpected image shape: {img_array.shape} for {image_path}")
            return None

        # Ensure we have exactly 3 channels
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            print(f"Could not convert to 3-channel image: {img_array.shape} for {image_path}")
            return None

        # Safely check BGR only if we have valid 3 channels
        try:
            # Check if image might be BGR by comparing channel means
            r_mean = np.mean(img_array[:, :, 0])
            g_mean = np.mean(img_array[:, :, 1])
            b_mean = np.mean(img_array[:, :, 2])

            # If blue channel is much brighter than red, might be BGR
            if b_mean > r_mean * 1.5 and b_mean > g_mean * 1.2:
                img_array = img_array[:, :, [2, 1, 0]]  # BGR to RGB
        except Exception as e:
            print(f"Channel analysis error for {image_path}: {e}")
            # Continue with original array

        # Ensure values are in proper range
        if img_array.dtype in [np.float32, np.float64]:
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        else:
            # Ensure uint8 type
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)

        # Apply AutoPS enhancement for better visualization
        enhanced_array = auto_ps_enhance(img_array)

        # If enhancement fails, return original
        if enhanced_array is not None:
            return enhanced_array
        else:
            return img_array

    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_placeholder_image(size=(128, 128)):
    """Create a placeholder image when actual image is not available"""
    placeholder = np.ones((size[0], size[1], 3), dtype=np.uint8) * 255  # White placeholder
    # Add text or pattern to indicate it's a placeholder
    return placeholder

def histogram(pixels):
    """统计像素直方图，为后续percent裁剪准备"""
    try:
        if pixels.size == 0:
            return [0]
        max_val = int(np.max(pixels))
        if max_val < 0:
            return [0]
        gram = [0] * (max_val + 1)
        for pixel in np.nditer(pixels):
            pixel_val = int(pixel)
            if 0 <= pixel_val <= max_val:
                gram[pixel_val] = gram[pixel_val] + 1
        return gram
    except Exception as e:
        print(f"Error in histogram: {e}")
        return [0]

def percent(x, togram):
    """依percent按比例阴影裁剪"""
    try:
        if len(togram) == 0 or np.sum(togram) == 0:
            return x

        image = x.copy()
        gram_lowsum = 0
        gram_highsum = 0
        total_sum = np.sum(togram)
        threshold = total_sum * 0.001

        # Low threshold
        for i in range(len(togram)):
            gram_lowsum = gram_lowsum + togram[i]
            if gram_lowsum >= threshold:
                image[image < i] = i
                break

        # High threshold
        for i in range(len(togram))[::-1]:
            gram_highsum = gram_highsum + togram[i]
            if gram_highsum >= threshold:
                image[image > i] = i
                break

        return image
    except Exception as e:
        print(f"Error in percent: {e}")
        return x

def auto_ps_enhance(image):
    """AutoPS色彩增强，参考AutoPS.py"""
    try:
        if image is None:
            return None

        # Validate input
        if not isinstance(image, np.ndarray) or image.size == 0:
            return image

        # Ensure image is uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)

        # Handle different shapes
        if len(image.shape) == 2:
            # Grayscale to RGB
            image = np.stack([image] * 3, axis=2)
        elif len(image.shape) == 3:
            if image.shape[2] == 4:
                image = image[:, :, :3]  # Remove alpha channel
            elif image.shape[2] == 1:
                image = np.repeat(image, 3, axis=2)
            elif image.shape[2] != 3:
                # Unexpected channel count, return original
                return image
        else:
            # Unexpected shape
            return image

        # Ensure we have 3 channels
        if len(image.shape) != 3 or image.shape[2] != 3:
            return image

        # Apply histogram equalization and percent clipping per channel
        img_ps = image.copy()

        # Process each channel separately
        for c in range(3):
            try:
                channel = img_ps[:, :, c]
                img_his = histogram(channel)
                img_per = percent(channel, img_his)
                img_ps[:, :, c] = img_per
            except Exception as e:
                print(f"Error processing channel {c}: {e}")
                # Keep original channel if processing fails
                continue

        # Apply final normalization per channel (similar to PS function)
        for c in range(3):
            try:
                channel = img_ps[:, :, c].astype(np.float32)
                x_max = np.max(channel)
                x_min = np.min(channel)
                if x_max > x_min:  # Avoid division by zero
                    img_ps[:, :, c] = ((channel - x_min) / (x_max - x_min) * 255).astype(np.uint8)
            except Exception as e:
                print(f"Error normalizing channel {c}: {e}")
                # Keep original channel if normalization fails
                continue

        return img_ps

    except Exception as e:
        print(f"Error in auto_ps_enhance: {e}")
        return image

def calculate_mse_residue(pred_img, gt_img):
    """Calculate MSE residue map between prediction and ground truth"""
    try:
        if pred_img is None or gt_img is None:
            return create_placeholder_image()

        # Ensure same shape
        if pred_img.shape != gt_img.shape:
            print(f"Shape mismatch: pred {pred_img.shape} vs gt {gt_img.shape}")
            return create_placeholder_image()

        # Convert to float for calculation
        pred_float = pred_img.astype(np.float32)
        gt_float = gt_img.astype(np.float32)

        # Calculate MSE per pixel
        mse = np.mean((pred_float - gt_float) ** 2, axis=2)

        # Clip extreme values for better visualization
        mse_clipped = np.clip(mse, 0, np.percentile(mse, 95))

        # Normalize to 0-1 range for visualization
        mse_norm = (mse_clipped - mse_clipped.min()) / (mse_clipped.max() - mse_clipped.min() + 1e-8)

        # Apply colormap: blue (low MSE) to yellow (high MSE) as per academic convention
        # Use 'coolwarm' colormap which goes from blue (low) to red/yellow (high)
        # Or create custom colormap for blue to yellow transition
        from matplotlib.colors import LinearSegmentedColormap

        # Create custom blue-to-yellow colormap
        colors = ['#0000FF', '#0080FF', '#00FFFF', '#80FF80', '#FFFF00', '#FF8000', '#FF0000']  # Blue to yellow/red
        n_bins = 256
        custom_cmap = LinearSegmentedColormap.from_list('blue_yellow', colors, N=n_bins)

        colormap = custom_cmap(mse_norm)[:, :, :3]  # Remove alpha channel
        colormap = (colormap * 255).astype(np.uint8)

        return colormap
    except Exception as e:
        print(f"Error calculating MSE residue: {e}")
        return create_placeholder_image()

def find_experiment_methods(exp_path):
    """Find all methods in the experiment directory"""
    methods = []

    if not os.path.exists(exp_path):
        print(f"Experiment path does not exist: {exp_path}")
        return methods

    # Look for test_results directories in the nested structure
    for item in os.listdir(exp_path):
        item_path = os.path.join(exp_path, item)
        if os.path.isdir(item_path):
            # Check for direct test_results
            test_results_path = os.path.join(item_path, 'test_results')
            if os.path.exists(test_results_path):
                methods.append((item, test_results_path))
                continue

            # Check for nested structure: method/method/timestamp/results/test_results
            method_subdir = os.path.join(item_path, item)
            if os.path.exists(method_subdir):
                # Look for timestamp directories
                for timestamp_dir in os.listdir(method_subdir):
                    timestamp_path = os.path.join(method_subdir, timestamp_dir)
                    if os.path.isdir(timestamp_path):
                        results_path = os.path.join(timestamp_path, 'results')
                        if os.path.exists(results_path):
                            test_results_path = os.path.join(results_path, 'test_results')
                            if os.path.exists(test_results_path):
                                methods.append((item, test_results_path))
                                break  # Use the first valid timestamp found

    print(f"Found {len(methods)} methods: {[m[0] for m in methods]}")
    return methods

def get_available_samples(methods, dataset, model_type='Latest'):
    """Get list of available sample IDs for a dataset"""
    if not methods:
        return []
    
    # Use first method to get sample list
    method_name, test_results_path = methods[0]
    dataset_path = os.path.join(test_results_path, model_type, dataset)

    if not os.path.exists(dataset_path):
        return []

    # Look for _gt.tif files to get sample IDs
    gt_files = glob.glob(os.path.join(dataset_path, '*_gt.tif'))
    sample_ids = [os.path.basename(f).replace('_gt.tif', '') for f in gt_files]

    return sorted(sample_ids)

def load_method_images(methods, dataset, sample_id, model_type='Latest'):
    """Load images for all methods for a specific sample"""
    method_images = {}

    for method_name, test_results_path in methods:
        try:
            dataset_path = os.path.join(test_results_path, model_type, dataset)

            # Load prediction image
            pred_path = os.path.join(dataset_path, f'{sample_id}_pred.tif')

            # Check if file exists and is not empty
            if os.path.exists(pred_path):
                file_size = os.path.getsize(pred_path)
                if file_size == 0:
                    print(f"✗ Empty file for {method_name}: {pred_path}")
                    continue

                pred_img = load_image(pred_path)

                if pred_img is not None:
                    method_images[method_name] = pred_img
                    print(f"✓ Loaded prediction for {method_name}: {pred_img.shape}")
                else:
                    print(f"✗ Could not load prediction for {method_name}: {pred_path}")
            else:
                print(f"✗ Prediction file not found for {method_name}: {pred_path}")
        except Exception as e:
            print(f"✗ Error loading method {method_name}: {e}")

    print(f"Successfully loaded {len(method_images)} method images")
    return method_images

def load_reference_images(methods, dataset, sample_id, model_type='Latest'):
    """Load PAN, MS (GT), and BIC reference images"""
    # Use the first available method to get reference images
    for method_name, test_results_path in methods:
        dataset_path = os.path.join(test_results_path, model_type, dataset)
        
        gt_path = os.path.join(dataset_path, f'{sample_id}_gt.tif')
        bic_path = os.path.join(dataset_path, f'{sample_id}_bic.tif')
        
        # Check file existence and size first
        if os.path.exists(gt_path) and os.path.getsize(gt_path) > 0:
            gt_img = load_image(gt_path)
            if gt_img is not None:
                print(f"✓ Loaded GT image: {gt_img.shape}")

                # Try to load BIC
                bic_img = None
                if os.path.exists(bic_path) and os.path.getsize(bic_path) > 0:
                    bic_img = load_image(bic_path)
                    if bic_img is not None:
                        print(f"✓ Loaded BIC image: {bic_img.shape}")

                # If BIC doesn't exist, create from GT
                if bic_img is None:
                    bic_img = np.clip(gt_img.astype(np.int16) // 2, 0, 255).astype(np.uint8)
                    print("✓ Created BIC from GT (simulation)")

                # For PAN, we'll use a grayscale version of GT as placeholder
                # In real scenarios, you might have separate PAN images
                pan_img = np.mean(gt_img, axis=2, keepdims=True)
                pan_img = np.repeat(pan_img, 3, axis=2).astype(np.uint8)
                print("✓ Created PAN from GT (simulation)")

                return {
                    'PAN': pan_img,
                    'MS': bic_img,  # Use BIC as MS (low-resolution input)
                    'GT': gt_img
                }
            else:
                print(f"✗ Could not load GT image: {gt_path}")
        else:
            print(f"✗ GT file not found or empty: {gt_path}")
    
    return None

def calculate_paper_layout(n_methods):
    """Calculate layout optimized for paper publication"""
    # Total items = PAN + MS + methods + GT
    total_items = n_methods + 3

    # If total items can be evenly divided by 2, use 2 rows
    if total_items % 2 == 0:
        n_rows = 2
        n_cols = total_items // 2
    else:
        # If odd number, use 2 rows with placeholder in bottom right
        n_rows = 2
        n_cols = (total_items + 1) // 2  # Add 1 for placeholder

    return n_rows, n_cols


def create_comparison_figure(reference_images, method_images, dataset, sample_id, save_path):
    """Create comparison figure with all methods in paper-like format"""
    # Use default font to avoid warnings
    plt.rcParams['font.family'] = 'DejaVu Sans'

    # Sort method names for consistent ordering
    sorted_methods = sorted(method_images.items())

    # Prepare images in order: PAN, MS, methods..., GT
    ordered_images = []
    ordered_names = []

    # Add reference images first
    if reference_images:
        ordered_images.extend([reference_images['PAN'], reference_images['MS']])
        ordered_names.extend(['PAN', 'MS'])

    # Add method images in sorted order
    for method_name, method_img in sorted_methods:
        ordered_images.append(method_img)
        ordered_names.append(method_name)

    # Add GT at the end
    if reference_images:
        ordered_images.append(reference_images['GT'])
        ordered_names.append('GT')

    # Calculate layout
    n_methods = len(method_images)
    total_items = len(ordered_images)
    _, n_cols = calculate_paper_layout(n_methods)

    # Create figure with optimized dimensions for 4-row layout
    fig_width = max(12, n_cols * 1.8)
    fig_height = 8.5  # Height for 4 rows: 2 for main images + 2 for residues

    # Create subplots: 4 rows (2 for main images + 2 for residues)
    fig, axes = plt.subplots(4, n_cols, figsize=(fig_width, fig_height))

    # Ensure axes is 4D array
    if n_cols == 1:
        axes = axes.reshape(4, 1)

    # Plot main images in first two rows
    for i, (img, name) in enumerate(zip(ordered_images, ordered_names)):
        row = i // n_cols
        col = i % n_cols

        if row < 2:  # Only use first 2 rows for main images
            if img is not None:
                axes[row, col].imshow(img)
            else:
                axes[row, col].imshow(create_placeholder_image())

            axes[row, col].set_title(name, fontsize=10, fontweight='bold')
            axes[row, col].axis('off')

    # Add placeholder if needed (for odd number of total items)
    if total_items % 2 == 1:
        # Add placeholder in bottom right of second row
        placeholder_row = 1
        placeholder_col = n_cols - 1
        axes[placeholder_row, placeholder_col].imshow(create_placeholder_image())
        axes[placeholder_row, placeholder_col].set_title('Placeholder', fontsize=10)
        axes[placeholder_row, placeholder_col].axis('off')

    # Hide unused subplots in first two rows
    for row in range(2):
        for col in range(n_cols):
            idx = row * n_cols + col
            if idx >= total_items and not (total_items % 2 == 1 and row == 1 and col == n_cols - 1):
                axes[row, col].axis('off')

    # Add MSE residue rows (rows 3 and 4) - same layout as main images
    if reference_images is not None and len(method_images) > 0:
        gt_img = reference_images['GT']

        # Create residue images for all methods
        residue_images = []
        residue_names = []

        for method_name, method_img in sorted_methods:
            residue = calculate_mse_residue(method_img, gt_img)
            residue_images.append(residue)
            residue_names.append(method_name)

        # Plot residue images in rows 3 and 4 using same layout as main images
        for i, (residue_img, name) in enumerate(zip(residue_images, residue_names)):
            row = (i // n_cols) + 2  # Start from row 2 (0-indexed, so row 3)
            col = i % n_cols

            if row < 4:  # Only use rows 2 and 3 (0-indexed, so rows 3 and 4)
                axes[row, col].imshow(residue_img)
                axes[row, col].set_title(name, fontsize=10, fontweight='bold')
                axes[row, col].axis('off')

        # Add placeholder if needed for residue rows
        total_residues = len(residue_images)
        if total_residues % 2 == 1:
            # Add placeholder in the next available position
            placeholder_row = ((total_residues) // n_cols) + 2
            placeholder_col = total_residues % n_cols
            if placeholder_row < 4:
                axes[placeholder_row, placeholder_col].imshow(create_placeholder_image())
                axes[placeholder_row, placeholder_col].set_title('Placeholder', fontsize=10)
                axes[placeholder_row, placeholder_col].axis('off')

        print(f"Showing residues for all {len(sorted_methods)} methods in 4-row layout")

    # Hide unused subplots in residue rows (rows 3 and 4)
    if reference_images is not None and len(method_images) > 0:
        total_residues = len(sorted_methods)

        # Hide unused subplots in rows 3 and 4
        for row in range(2, 4):  # Rows 3 and 4 (0-indexed: 2 and 3)
            for col in range(n_cols):
                idx = (row - 2) * n_cols + col  # Calculate position in residue sequence
                if idx >= total_residues and not (total_residues % 2 == 1 and idx == total_residues):
                    axes[row, col].axis('off')
    else:
        # Hide all residue rows if no reference images
        for row in range(2, 4):
            for col in range(n_cols):
                axes[row, col].axis('off')

    # Add vertical colorbar on the right
    if len(method_images) > 0 and reference_images is not None:
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        from matplotlib.colors import LinearSegmentedColormap

        # Create the same custom blue-to-yellow colormap for colorbar
        colors = ['#0000FF', '#0080FF', '#00FFFF', '#80FF80', '#FFFF00', '#FF8000', '#FF0000']  # Blue to yellow/red
        n_bins = 256
        custom_cmap = LinearSegmentedColormap.from_list('blue_yellow', colors, N=n_bins)

        norm = mcolors.Normalize(vmin=0, vmax=15)  # Typical MSE range
        sm = cm.ScalarMappable(norm=norm, cmap=custom_cmap)
        sm.set_array([])

        # Position vertical colorbar on the right
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
        cbar.set_label('MSE Residue (Blue=Low, Yellow=High)', fontsize=10)

    # Set main title
    plt.suptitle(f'Visual comparison on {dataset} dataset', fontsize=14, fontweight='bold')

    # Adjust layout with ultra-tight spacing for paper format
    plt.subplots_adjust(left=0.05, right=0.90, bottom=0.05, top=0.92, hspace=0.075, wspace=0.08)

    # Save as SVG
    plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved enhanced visualization to {save_path}")

def main(exp_path, sample_id=None, dataset=None, save_dir='Work_dir', num_samples=1, model_type='Latest'):
    """Main visualization function with enhanced features"""
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Find all methods in experiment
    methods = find_experiment_methods(exp_path)

    if not methods:
        print("No methods found in experiment directory")
        return

    print(f"Found {len(methods)} methods: {[m[0] for m in methods]}")

    # Process each sample (allowing different datasets for each sample)
    for i in range(num_samples):
        print(f"\n--- Processing sample {i+1}/{num_samples} ---")

        # For each sample, potentially select a different dataset
        current_dataset = dataset
        current_sample_id = sample_id

        # If no dataset specified, randomly select from available datasets for each sample
        if current_dataset is None:
            # Try common dataset names (without _data suffix)
            common_datasets = ['WV3', 'GF2', 'WV2']
            available_datasets = []

            for ds in common_datasets:
                samples = get_available_samples(methods, ds, model_type)
                if samples:
                    available_datasets.append(ds)

            if not available_datasets:
                print("No datasets found with available samples")
                return

            # Randomly select a dataset for this sample
            current_dataset = random.choice(available_datasets)
            print(f"Randomly selected dataset: {current_dataset}")

        # Get available samples for current dataset
        available_samples = get_available_samples(methods, current_dataset, model_type)

        if not available_samples:
            print(f"No samples found for dataset {current_dataset}")
            continue

        print(f"Found {len(available_samples)} samples in {current_dataset}")

        # Select sample to visualize
        if current_sample_id is not None:
            if current_sample_id in available_samples:
                selected_sample = current_sample_id
            else:
                print(f"Sample {current_sample_id} not found. Available: {available_samples[:5]}...")
                continue
        else:
            # Randomly select a sample for this iteration
            selected_sample = random.choice(available_samples)

        print(f"Processing sample: {selected_sample} from dataset: {current_dataset}")

        # Load images for all methods
        method_images = load_method_images(methods, current_dataset, selected_sample, model_type)
        reference_images = load_reference_images(methods, current_dataset, selected_sample, model_type)

        if not method_images:
            print(f"No method images found for sample {selected_sample}")
            continue

        if not reference_images:
            print(f"No reference images found for sample {selected_sample}")
            continue

        # Create visualization
        output_filename = f"comparison_{current_dataset}_{selected_sample}.svg"
        output_path = os.path.join(save_dir, output_filename)

        create_comparison_figure(reference_images, method_images, current_dataset, selected_sample, output_path)

    print(f"\nVisualization complete. Files saved in {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced Pan-sharpening Results Visualization')
    parser.add_argument('exp_path', help='Path to experiment directory')
    parser.add_argument('--sample_id', help='Specific sample ID to visualize')
    parser.add_argument('--dataset', help='Dataset name (e.g., WV3_data, GF2_data)')
    parser.add_argument('--save_dir', default='Work_dir', help='Directory to save visualizations')
    parser.add_argument('--num', type=int, default=1, help='Number of random samples to visualize')
    parser.add_argument('--model_type', default='Latest', help='Model type (Latest, Best, etc.)')

    args = parser.parse_args()

    main(args.exp_path, args.sample_id, args.dataset, args.save_dir, args.num, args.model_type)
