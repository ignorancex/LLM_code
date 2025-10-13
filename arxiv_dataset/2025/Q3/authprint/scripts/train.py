#!/usr/bin/env python
"""
Main training script for StyleGAN fingerprinting.
"""
import argparse
import logging
import sys
import os

# Add torch dynamo configuration
import torch._dynamo
torch._dynamo.config.suppress_errors = True  # Disable dynamo compilation

# Add the parent directory (project root) to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.default_config import Config, get_default_config
from trainers.fingerprint_trainer import FingerprintTrainer
from utils.distributed import setup_distributed, cleanup_distributed
from utils.logging_utils import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fingerprinting Training Pipeline")
    
    # Model type selection
    parser.add_argument("--model_type", type=str, default="stylegan2",
                       choices=["stylegan2", "stable-diffusion"],
                       help="Type of generative model to use")
    
    # Stable Diffusion configuration
    parser.add_argument("--sd_reconstructor_size", type=str, default="M",
                       choices=["S", "M", "L"],
                       help="Size of the Stable Diffusion reconstructor model to use (S=Small, M=Medium, L=Large)")
    
    # StyleGAN2 configuration
    parser.add_argument("--stylegan2_url", type=str,
                        default="https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq70k-paper256-ada.pkl",
                        help="URL for the pretrained StyleGAN2 model")
    parser.add_argument("--stylegan2_local_path", type=str,
                        default="ffhq70k-paper256-ada.pkl",
                        help="Local path to store/load the StyleGAN2 model")
    
    # Stable Diffusion configuration
    parser.add_argument("--sd_model_name", type=str,
                        default="stabilityai/stable-diffusion-xl-base-1.0",
                        help="Name of the Stable Diffusion model to use")
    parser.add_argument("--sd_enable_cpu_offload", action="store_true",
                        help="Enable CPU offloading for Stable Diffusion")
    parser.add_argument("--sd_dtype", type=str, default="float16",
                        choices=["float16", "float32"],
                        help="Data type for Stable Diffusion model")
    parser.add_argument("--sd_num_inference_steps", type=int, default=50,
                        help="Number of inference steps for Stable Diffusion")
    parser.add_argument("--sd_guidance_scale", type=float, default=7.5,
                        help="Guidance scale for Stable Diffusion")
    parser.add_argument("--sd_prompt", type=str,
                        default="A photo of a cat in a variety of real-world scenes, candid shot, natural lighting, diverse settings, DSLR photo",
                        help="Default prompt for Stable Diffusion generation")
    
    # Multi-prompt training configuration
    parser.add_argument("--enable_multi_prompt", action="store_true",
                        help="Enable multi-prompt training mode")
    parser.add_argument("--prompt_source", type=str, default="local",
                        choices=["local", "diffusiondb", "parti-prompts"],
                        help="Source of prompts: local file, DiffusionDB dataset, or Parti-Prompts dataset")
    parser.add_argument("--prompt_dataset_path", type=str, default="",
                        help="Path to local prompt dataset file (one prompt per line)")
    parser.add_argument("--prompt_dataset_size", type=int, default=10000,
                        help="Number of prompts to load from dataset")
    parser.add_argument("--diffusiondb_subset", type=str, default="2m_random_10k",
                        choices=["2m_random_10k", "large_random_10k", "2m_random_5k"],
                        help="Which DiffusionDB subset to use (2m_random_10k, large_random_10k, 2m_random_5k)")
    parser.add_argument("--parti_prompts_category", type=str, default="",
                        help="Category to use from parti-prompts dataset (e.g., 'People', 'Animals', etc.)")
    parser.add_argument("--train_eval_split_ratio", type=float, default=0.8,
                        help="Ratio of prompts to use for training vs evaluation (default: 0.8)")
    
    # Common configuration
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--img_size", type=int, default=256, help="Image resolution")
    parser.add_argument("--image_pixel_set_seed", type=int, default=42,
                        help="Random seed for selecting pixel indices from the generated image")
    parser.add_argument("--image_pixel_count", type=int, default=32,
                        help="Number of pixels to select from the image (default: 32)")
    
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--total_iterations", type=int, default=100000, help="Total number of training iterations")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--log_interval", type=int, default=1, help="Interval for logging training progress")
    parser.add_argument("--checkpoint_interval", type=int, default=10000, help="Interval for saving checkpoints")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save logs and checkpoints")
    
    # Other configuration
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    
    return parser.parse_args()


def main():
    """Main entry point for training."""
    args = parse_args()
    
    try:
        # Setup distributed training first
        local_rank, rank, world_size, device = setup_distributed()
        
        # Load default configuration and update with args
        config = get_default_config()
        config.update_from_args(args, mode='train')
        
        # Setup logging
        setup_logging(config.output_dir, rank)
        
        # Log configuration
        if rank == 0:
            logging.info(f"Configuration:\n{config}")
            logging.info(f"Distributed setup: local_rank={local_rank}, rank={rank}, world_size={world_size}, device={device}")
        
        # Initialize trainer
        trainer = FingerprintTrainer(config, local_rank, rank, world_size, device)
        
        # Run training
        trainer.train()
        
    except Exception as e:
        logging.error(f"Error in training: {str(e)}", exc_info=True)
        raise
    finally:
        # Clean up distributed environment
        cleanup_distributed()


if __name__ == "__main__":
    main() 