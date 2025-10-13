#!/usr/bin/env python
"""
Script to analyze how reconstructor predictions shift when pixels are manipulated.
"""
import argparse
import logging
import os
import sys
from typing import List, Dict, Tuple
import json
import numpy as np
import torch
from tqdm import tqdm

# Add the parent directory (project root) to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.distributed import setup_distributed, cleanup_distributed
from utils.logging_utils import setup_logging
from utils.checkpoint import load_checkpoint
from models.reconstructor import ReconstructorSD_L, ReconstructorSD_M, ReconstructorSD_S, StyleGAN2Reconstructor
from models.stylegan2_model import StyleGAN2Model
from models.stable_diffusion_model import StableDiffusionModel


def validate_args(args):
    """Validate command line arguments."""
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        raise ValueError(f"Checkpoint not found at: {args.checkpoint_path}")
    
    # Validate image_pixel_count
    total_pixels = 3 * args.img_size * args.img_size  # 3 channels
    if args.image_pixel_count > total_pixels:
        raise ValueError(
            f"image_pixel_count ({args.image_pixel_count}) cannot be larger than "
            f"total pixels in image ({total_pixels})"
        )
    
    # Validate max_pixels
    if args.max_pixels > total_pixels:
        raise ValueError(
            f"max_pixels ({args.max_pixels}) cannot be larger than "
            f"total pixels in image ({total_pixels})"
        )
    
    # Validate StyleGAN2 local path if using StyleGAN2
    if args.model_type == "stylegan2" and not os.path.exists(args.stylegan2_local_path):
        if args.rank == 0:
            logging.warning(f"StyleGAN2 model not found at {args.stylegan2_local_path}, "
                          f"will attempt to download from {args.stylegan2_url}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Pixel Manipulation Analysis for Reconstructor Predictions")
    
    # Model type selection
    parser.add_argument("--model_type", type=str, default="stylegan2",
                       choices=["stylegan2", "stable-diffusion"],
                       help="Type of generative model to use")
    
    # Common configuration
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to reconstructor checkpoint to evaluate")
    parser.add_argument("--img_size", type=int, default=256,
                       help="Image resolution")
    
    # Reconstructor configuration
    parser.add_argument("--reconstructor_size", type=str, default="M",
                       choices=["S", "M", "L"],
                       help="Size of reconstructor model (for Stable Diffusion)")
    parser.add_argument("--image_pixel_count", type=int, default=32,
                       help="Number of pixels to predict (output dimension of reconstructor)")
    
    # Experiment configuration
    parser.add_argument("--num_images", type=int, default=10,
                       help="Number of different images to test")
    parser.add_argument("--num_repeats", type=int, default=5,
                       help="Number of random repeats for each manipulation level")
    parser.add_argument("--max_pixels", type=int, default=65536,
                       help="Maximum number of pixels to manipulate")
    parser.add_argument("--base", type=int, default=2,
                       help="Base for exponential increase in pixel count (default: 2 for 1,2,4,8,...)")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="pixel_manipulation_results",
                       help="Directory to save results")
    
    # StyleGAN2 specific args
    parser.add_argument("--stylegan2_url", type=str,
                       default="https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq70k-paper256-ada.pkl",
                       help="URL for the pretrained StyleGAN2 model")
    parser.add_argument("--stylegan2_local_path", type=str,
                       default="ffhq70k-paper256-ada.pkl",
                       help="Local path to store/load the StyleGAN2 model")
    
    # Stable Diffusion specific args
    parser.add_argument("--sd_model_name", type=str,
                       default="stabilityai/stable-diffusion-xl-base-1.0",
                       help="Name of the Stable Diffusion model to use")
    parser.add_argument("--sd_prompt", type=str, default="a photo of a cat",
                       help="Text prompt for Stable Diffusion")
    parser.add_argument("--sd_num_inference_steps", type=int, default=30,
                       help="Number of inference steps for Stable Diffusion")
    parser.add_argument("--sd_guidance_scale", type=float, default=7.5,
                       help="Guidance scale for Stable Diffusion")
    parser.add_argument("--sd_enable_cpu_offload", action="store_true",
                       help="Enable CPU offloading for Stable Diffusion")
    parser.add_argument("--sd_dtype", type=str, default="float16",
                       choices=["float16", "float32"],
                       help="Data type for Stable Diffusion model")
    
    # Add rank argument for validation
    parser.add_argument("--rank", type=int, default=0,
                       help="Process rank for distributed training")
    
    args = parser.parse_args()
    validate_args(args)
    return args


class PixelManipulationExperiment:
    def __init__(self, args, device, rank=0):
        """
        Initialize the experiment.
        
        Args:
            args: Parsed command line arguments
            device: PyTorch device
            rank: Process rank for distributed training
        """
        self.args = args
        self.device = device
        self.rank = rank
        
        # Initialize models
        self._setup_models()
        
        # Calculate manipulation levels
        self.manipulation_levels = self._get_manipulation_levels()
        if self.rank == 0:
            logging.info(f"Testing pixel manipulation levels: {self.manipulation_levels}")
    
    def _setup_models(self):
        """
        Set up the generative model and reconstructor.
        """
        if self.rank == 0:
            logging.info("Setting up models...")
            logging.info(f"Loading reconstructor from checkpoint: {self.args.checkpoint_path}")
        
        try:
            # Initialize generative model based on type
            if self.args.model_type == "stylegan2":
                # First ensure the model file exists or download it
                if not os.path.exists(self.args.stylegan2_local_path):
                    if self.rank == 0:
                        logging.info(f"Downloading StyleGAN2 model from {self.args.stylegan2_url}")
                    import urllib.request
                    urllib.request.urlretrieve(self.args.stylegan2_url, self.args.stylegan2_local_path)
                
                self.generative_model = StyleGAN2Model(
                    model_url=self.args.stylegan2_url,
                    model_path=self.args.stylegan2_local_path,
                    device=self.device,
                    img_size=self.args.img_size
                )
                
                # Initialize StyleGAN2 reconstructor
                self.reconstructor = StyleGAN2Reconstructor(
                    image_size=self.args.img_size,
                    channels=3,
                    output_dim=self.args.image_pixel_count
                ).to(self.device)
                
            else:  # stable-diffusion
                self.generative_model = StableDiffusionModel(
                    model_name=self.args.sd_model_name,
                    img_size=self.args.img_size,
                    device=self.device,
                    enable_cpu_offload=self.args.sd_enable_cpu_offload,
                    dtype=getattr(torch, self.args.sd_dtype)
                )
                
                # Initialize SD reconstructor based on size
                reconstructor_class = {
                    "S": ReconstructorSD_S,
                    "M": ReconstructorSD_M,
                    "L": ReconstructorSD_L
                }[self.args.reconstructor_size]
                
                self.reconstructor = reconstructor_class(
                    image_size=self.args.img_size,
                    channels=3,
                    output_dim=self.args.image_pixel_count
                ).to(self.device)
            
            # Load reconstructor checkpoint
            if self.rank == 0:
                logging.info(f"Loading reconstructor checkpoint from {self.args.checkpoint_path}")
            
            load_checkpoint(
                checkpoint_path=self.args.checkpoint_path,
                reconstructor=self.reconstructor,
                device=self.device
            )
            
            # Set models to eval mode
            self.generative_model.eval()
            self.reconstructor.eval()
            
            if self.rank == 0:
                logging.info("Models setup completed")
                
        except Exception as e:
            if self.rank == 0:
                logging.error(f"Error setting up models: {str(e)}")
            raise
    
    def _get_manipulation_levels(self) -> List[int]:
        """
        Generate list of pixel counts to test.
        """
        levels = []
        current = 1
        total_pixels = 3 * self.args.img_size * self.args.img_size
        
        while current <= min(self.args.max_pixels, total_pixels):
            levels.append(current)
            current *= self.args.base
        
        return levels
    
    def _manipulate_pixels(
        self,
        image: torch.Tensor,
        num_pixels: int,
        seed: int
    ) -> torch.Tensor:
        """
        Manipulate specified number of random pixels in the image.
        
        Args:
            image: Input image tensor [B, C, H, W]
            num_pixels: Number of pixels to manipulate
            seed: Random seed for reproducibility
            
        Returns:
            Manipulated image tensor
        """
        # Set seed on both CPU and GPU
        torch.manual_seed(seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(seed)
        
        # Calculate total number of pixels
        batch_size = image.shape[0]
        channels = image.shape[1]
        total_pixels = channels * self.args.img_size * self.args.img_size
        
        # Generate random indices directly on device
        indices = torch.randperm(total_pixels, device=self.device)[:num_pixels]
        
        # Create manipulated copy
        manipulated = image.clone()
        flattened = manipulated.view(batch_size, -1)
        flattened[:, indices] = -1.0  # Set to -1 for StyleGAN range
        
        return flattened.view_as(image)
    
    def _compute_prediction_shift(
        self,
        pred_original: torch.Tensor,
        pred_manipulated: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute various metrics for prediction shift.
        
        Args:
            pred_original: Original reconstructor prediction
            pred_manipulated: Manipulated reconstructor prediction
            
        Returns:
            Dictionary of metrics
        """
        with torch.no_grad():
            # L2 distance
            l2_dist = torch.norm(pred_manipulated - pred_original, p=2, dim=1).mean().item()
            
            # L1 distance
            l1_dist = torch.norm(pred_manipulated - pred_original, p=1, dim=1).mean().item()
            
            # Cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(
                pred_manipulated,
                pred_original,
                dim=1
            ).mean().item()
        
        return {
            'l2_distance': l2_dist,
            'l1_distance': l1_dist,
            'cosine_similarity': cos_sim
        }
    
    def run_experiment(self) -> Dict[str, Dict[str, List[float]]]:
        """
        Run the pixel manipulation experiment.
        
        Returns:
            Dictionary containing results for each metric and manipulation level
        """
        results = {
            'l2_distance': [],
            'l1_distance': [],
            'cosine_similarity': [],
            'std_l2': [],
            'std_l1': [],
            'std_cos': []
        }
        
        if self.rank == 0:
            logging.info(f"Running experiment with {self.args.num_images} images, "
                        f"{self.args.num_repeats} repeats per level")
        
        try:
            # Generate images and get predictions
            for level in tqdm(self.manipulation_levels, desc="Manipulation levels", disable=self.rank != 0):
                level_metrics = {
                    'l2_distance': [],
                    'l1_distance': [],
                    'cosine_similarity': []
                }
                
                # Test multiple images
                for img_idx in range(self.args.num_images):
                    try:
                        # Generate image
                        with torch.no_grad():
                            if self.args.model_type == "stylegan2":
                                z = torch.randn(1, self.generative_model.z_dim, device=self.device)
                                image = self.generative_model.generate_images(
                                    batch_size=1,
                                    device=self.device,
                                    z=z,
                                    noise_mode="const"
                                )
                            else:  # stable-diffusion
                                image = self.generative_model.generate_images(
                                    batch_size=1,
                                    device=self.device,
                                    prompt=self.args.sd_prompt,
                                    num_inference_steps=self.args.sd_num_inference_steps,
                                    guidance_scale=self.args.sd_guidance_scale
                                )
                        
                        # Get original prediction
                        with torch.no_grad():
                            pred_original = self.reconstructor(image)
                        
                        # Test multiple random manipulations
                        for repeat in range(self.args.num_repeats):
                            # Set different seed for each repeat
                            seed = img_idx * 1000 + repeat
                            
                            # Manipulate pixels
                            manipulated = self._manipulate_pixels(image, level, seed)
                            
                            # Get prediction for manipulated image
                            with torch.no_grad():
                                pred_manipulated = self.reconstructor(manipulated)
                            
                            # Compute metrics
                            metrics = self._compute_prediction_shift(pred_original, pred_manipulated)
                            
                            # Store results
                            for metric_name, value in metrics.items():
                                level_metrics[metric_name].append(value)
                        
                        # Clear CUDA cache after processing each image
                        if self.device.type == 'cuda':
                            torch.cuda.empty_cache()
                            
                    except Exception as e:
                        if self.rank == 0:
                            logging.error(f"Error processing image {img_idx} at level {level}: {str(e)}")
                            logging.error("Skipping this image and continuing...")
                        continue
                
                # Average results for this level
                results['l2_distance'].append(np.mean(level_metrics['l2_distance']))
                results['l1_distance'].append(np.mean(level_metrics['l1_distance']))
                results['cosine_similarity'].append(np.mean(level_metrics['cosine_similarity']))
                
                # Store standard deviations
                results['std_l2'].append(np.std(level_metrics['l2_distance']))
                results['std_l1'].append(np.std(level_metrics['l1_distance']))
                results['std_cos'].append(np.std(level_metrics['cosine_similarity']))
                
                if self.rank == 0:
                    logging.info(f"Level {level} pixels - "
                               f"L2: {results['l2_distance'][-1]:.4f} ± {results['std_l2'][-1]:.4f}, "
                               f"L1: {results['l1_distance'][-1]:.4f} ± {results['std_l1'][-1]:.4f}, "
                               f"Cos: {results['cosine_similarity'][-1]:.4f} ± {results['std_cos'][-1]:.4f}")
            
        except Exception as e:
            if self.rank == 0:
                logging.error(f"Error in experiment: {str(e)}")
            raise
        
        return results

    def print_results(self, results: Dict[str, List[float]]):
        """
        Print the experimental results in a table format.
        
        Args:
            results: Dictionary of results
        """
        if self.rank == 0:
            # Print header
            logging.info("\nPixel Manipulation Results:")
            logging.info("-" * 120)
            logging.info(
                f"{'Pixels':>10} | "
                f"{'L2 Distance':>20} | "
                f"{'L1 Distance':>20} | "
                f"{'Cosine Sim':>20}"
            )
            logging.info("-" * 120)
            
            # Print results for each manipulation level
            for i, level in enumerate(self.manipulation_levels):
                logging.info(
                    f"{level:>10} | "
                    f"{results['l2_distance'][i]:>10.4f} ± {results['std_l2'][i]:>7.4f} | "
                    f"{results['l1_distance'][i]:>10.4f} ± {results['std_l1'][i]:>7.4f} | "
                    f"{results['cosine_similarity'][i]:>10.4f} ± {results['std_cos'][i]:>7.4f}"
                )
            
            logging.info("-" * 120)


def main():
    """Main entry point for pixel manipulation experiment."""
    args = parse_args()
    
    # Setup distributed environment
    local_rank, rank, world_size, device = setup_distributed()
    args.rank = rank  # Add rank to args for validation
    
    # Create output directory and setup logging
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        setup_logging(args.output_dir, rank)
        logging.info(f"Arguments:\n{args}")
    
    try:
        # Initialize and run experiment
        experiment = PixelManipulationExperiment(args, device, rank)
        results = experiment.run_experiment()
        
        if rank == 0:
            # Save results
            results_path = os.path.join(args.output_dir, 'results.json')
            with open(results_path, 'w') as f:
                json.dump({
                    'manipulation_levels': experiment.manipulation_levels,
                    'metrics': results
                }, f, indent=2)
            
            # Print results in table format
            experiment.print_results(results)
            logging.info(f"Experiment completed. Results saved to {args.output_dir}")
        
    except Exception as e:
        if rank == 0:
            logging.error(f"Error in experiment: {str(e)}", exc_info=True)
        raise
    finally:
        # Clean up distributed environment
        cleanup_distributed()


if __name__ == "__main__":
    main()