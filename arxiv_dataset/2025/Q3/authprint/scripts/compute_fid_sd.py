#!/usr/bin/env python
"""
Script to compute FID scores between Stable Diffusion 2.1 and other SD models (1.5 to 1.1).
Supports multi-GPU processing for faster image generation and incremental FID computation.
"""
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Generator

import torch
import torch.distributed as dist
import numpy as np
import torchvision
import time
from datetime import datetime
import psutil
import torch.cuda

# Add the parent directory (project root) to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.stable_diffusion_model import StableDiffusionModel
from utils.metrics import calculate_fid, extract_inception_features, compute_fid_from_features
from utils.distributed import setup_distributed, cleanup_distributed
from utils.logging_utils import setup_logging
from utils.model_loading import STABLE_DIFFUSION_MODELS


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compute FID scores between SD 2.1 and other SD models")
    
    # Model configuration
    parser.add_argument("--reference_model", type=str, default="sd-2.1",
                        help="Reference model to compare against (default: sd-2.1)")
    parser.add_argument("--comparison_models", type=str, nargs='+',
                        default=["sd-1.5", "sd-1.4", "sd-1.3", "sd-1.2", "sd-1.1"],
                        help="List of models to compare with the reference model")
    
    # Generation configuration
    parser.add_argument("--prompt", type=str,
                        default="A photorealistic advertisement poster for a Japanese cafe named 'NOVA CAFE', with the name written clearly in both English and Japanese on a street sign, a storefront banner, and a coffee cup. The scene is set at night with neon lighting, rain-slick streets reflecting the glow, and people walking by in motion blur. Cinematic tone, Leica photo quality, ultra-detailed textures.",
                        help="Prompt for image generation")
    parser.add_argument("--img_size", type=int, default=1024,
                        help="Size of generated images")
    parser.add_argument("--num_images", type=int, default=10000,
                        help="Number of images to generate for FID calculation")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size per GPU")
    parser.add_argument("--chunk_size", type=int, default=1000,
                        help="Number of images to process at once for FID computation")
    parser.add_argument("--num_inference_steps", type=int, default=25,
                        help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Classifier-free guidance scale")
    
    # Hardware configuration
    parser.add_argument("--enable_cpu_offload", action="store_true",
                        help="Enable CPU offloading for SD models")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "float32"],
                        help="Model dtype")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="fid_results",
                        help="Directory to save results")
    parser.add_argument("--save_images", action="store_true",
                        help="Save generated images for inspection")
    
    return parser.parse_args()


def get_memory_usage():
    """Get current memory usage of CPU and GPU."""
    # CPU Memory
    cpu_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    # GPU Memory
    gpu_memory = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024 / 1024  # MB
            reserved = torch.cuda.memory_reserved(i) / 1024 / 1024   # MB
            gpu_memory.append({
                'id': i,
                'allocated': allocated,
                'reserved': reserved,
                'total': torch.cuda.get_device_properties(i).total_memory / 1024 / 1024
            })
    else:
        gpu_memory = "CUDA not available"
    
    return cpu_memory, gpu_memory


def log_progress(rank: int, message: str, level: str = "info"):
    """Log progress with timestamp and memory usage."""
    if rank == 0:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cpu_mem, gpu_mem = get_memory_usage()
        
        mem_info = f"CPU Memory: {cpu_mem:.2f}MB"
        if isinstance(gpu_mem, list):
            for gpu in gpu_mem:
                mem_info += f", GPU{gpu['id']}: {gpu['allocated']:.0f}MB allocated/{gpu['reserved']:.0f}MB reserved/{gpu['total']:.0f}MB total"
        else:
            mem_info += f", {gpu_mem}"
        
        log_msg = f"[{timestamp}] {message}\n{mem_info}"
        
        if level == "info":
            logging.info(log_msg)
        elif level == "warning":
            logging.warning(log_msg)
        elif level == "error":
            logging.error(log_msg)
        elif level == "debug":
            logging.debug(log_msg)


def generate_images_distributed(
    model: StableDiffusionModel,
    num_images: int,
    batch_size: int,
    rank: int,
    world_size: int,
    chunk_size: int,
    **kwargs
) -> Generator[torch.Tensor, None, None]:
    """Generate images in a distributed manner across GPUs.
    
    Args:
        model: SD model to generate images with
        num_images: Total number of images to generate
        batch_size: Batch size per GPU
        rank: Current process rank
        world_size: Total number of processes
        chunk_size: Number of images to generate per chunk
        **kwargs: Additional arguments for image generation
        
    Yields:
        torch.Tensor: Generated images [N, C, H, W] for each chunk
    """
    # Calculate number of images per GPU
    images_per_gpu = num_images // world_size
    if rank < num_images % world_size:
        images_per_gpu += 1
    
    total_chunks = (images_per_gpu + chunk_size - 1) // chunk_size
    
    # Process in chunks
    for chunk_idx in range(total_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, images_per_gpu)
        chunk_images = []
        
        log_progress(rank, f"Starting chunk {chunk_idx + 1}/{total_chunks} (images {chunk_start}-{chunk_end})")
        chunk_start_time = time.time()
        
        # Generate images for this chunk
        for i in range(chunk_start, chunk_end, batch_size):
            batch_start_time = time.time()
            current_batch_size = min(batch_size, chunk_end - i)
            
            log_progress(rank, f"Generating batch of {current_batch_size} images in chunk {chunk_idx + 1}")
            
            batch = model.generate_images(
                batch_size=current_batch_size,
                **kwargs
            )
            chunk_images.append(batch)
            
            batch_time = time.time() - batch_start_time
            if rank == 0:
                total_progress = (chunk_start + i + current_batch_size) * world_size
                log_progress(rank, 
                    f"Progress: {total_progress}/{num_images} images "
                    f"(Batch time: {batch_time:.2f}s, "
                    f"Images/sec: {current_batch_size/batch_time:.2f})"
                )
        
        # Concatenate and yield chunk images
        chunk_images = torch.cat(chunk_images, dim=0)
        chunk_time = time.time() - chunk_start_time
        log_progress(rank, 
            f"Completed chunk {chunk_idx + 1}/{total_chunks} "
            f"(Time: {chunk_time:.2f}s, "
            f"Images/sec: {len(chunk_images)/chunk_time:.2f})"
        )
        
        yield chunk_images
        
        # Clear memory
        del chunk_images
        torch.cuda.empty_cache()
        log_progress(rank, f"Cleared memory after chunk {chunk_idx + 1}")


def compute_fid_scores(args, rank: int, world_size: int, device: torch.device) -> Dict[str, float]:
    """Compute FID scores between reference model and comparison models."""
    log_progress(rank, "Starting FID computation")
    start_time = time.time()
    
    # Initialize models
    log_progress(rank, "Initializing models...")
    
    # Initialize reference model
    ref_model = StableDiffusionModel(
        model_name=STABLE_DIFFUSION_MODELS[args.reference_model],
        device=device,
        img_size=args.img_size,
        dtype=getattr(torch, args.dtype),
        enable_cpu_offload=args.enable_cpu_offload
    )
    
    # Compute FID scores for each comparison model
    fid_scores = {}
    for model_idx, model_name in enumerate(args.comparison_models):
        model_start_time = time.time()
        log_progress(rank, f"\nStarting comparison {model_idx + 1}/{len(args.comparison_models)}: {args.reference_model} vs {model_name}")
        
        # Initialize comparison model
        comp_model = StableDiffusionModel(
            model_name=STABLE_DIFFUSION_MODELS[model_name],
            device=device,
            img_size=args.img_size,
            dtype=getattr(torch, args.dtype),
            enable_cpu_offload=args.enable_cpu_offload
        )
        
        # Initialize feature accumulators
        total_chunks = (args.num_images // world_size + args.chunk_size - 1) // args.chunk_size
        ref_features_list = []
        comp_features_list = []
        
        # Process chunks one at a time
        for chunk_idx in range(total_chunks):
            chunk_start_time = time.time()
            log_progress(rank, f"Processing chunk {chunk_idx + 1}/{total_chunks}")
            
            # Generate reference model images for this chunk
            log_progress(rank, f"Generating {args.reference_model} images for chunk {chunk_idx + 1}")
            ref_chunk = next(generate_images_distributed(
                model=ref_model,
                num_images=min(args.chunk_size, args.num_images - chunk_idx * args.chunk_size),
                batch_size=args.batch_size,
                rank=rank,
                world_size=world_size,
                chunk_size=args.chunk_size,
                prompt=args.prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale
            ))
            
            # Save reference images if requested
            if args.save_images and rank == 0:
                save_dir = Path(args.output_dir) / "images" / args.reference_model
                save_dir.mkdir(parents=True, exist_ok=True)
                for i, img in enumerate(ref_chunk):
                    img_path = save_dir / f"chunk{chunk_idx}_img{i:05d}.png"
                    torchvision.utils.save_image(img, img_path)
            
            # Extract features for reference images
            ref_features = extract_inception_features(ref_chunk, batch_size=args.batch_size, device=device)
            ref_features_list.append(ref_features)
            
            # Clear reference images from memory
            del ref_chunk
            torch.cuda.empty_cache()
            
            # Generate comparison model images for this chunk
            log_progress(rank, f"Generating {model_name} images for chunk {chunk_idx + 1}")
            comp_chunk = next(generate_images_distributed(
                model=comp_model,
                num_images=min(args.chunk_size, args.num_images - chunk_idx * args.chunk_size),
                batch_size=args.batch_size,
                rank=rank,
                world_size=world_size,
                chunk_size=args.chunk_size,
                prompt=args.prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale
            ))
            
            # Save comparison images if requested
            if args.save_images and rank == 0:
                save_dir = Path(args.output_dir) / "images" / model_name
                save_dir.mkdir(parents=True, exist_ok=True)
                for i, img in enumerate(comp_chunk):
                    img_path = save_dir / f"chunk{chunk_idx}_img{i:05d}.png"
                    torchvision.utils.save_image(img, img_path)
            
            # Extract features for comparison images
            comp_features = extract_inception_features(comp_chunk, batch_size=args.batch_size, device=device)
            comp_features_list.append(comp_features)
            
            # Clear comparison images from memory
            del comp_chunk
            torch.cuda.empty_cache()
            
            chunk_time = time.time() - chunk_start_time
            log_progress(rank, 
                f"Processed chunk {chunk_idx + 1}:\n"
                f"- Processing Time: {chunk_time:.2f}s\n"
                f"- Accumulated features from {len(ref_features_list)} chunks"
            )
        
        # Compute final FID score using all accumulated features
        log_progress(rank, "Computing final FID score...")
        
        # Concatenate all features
        ref_all_features = np.concatenate(ref_features_list, axis=0)
        comp_all_features = np.concatenate(comp_features_list, axis=0)
        
        # Compute final FID
        final_fid = compute_fid_from_features(ref_all_features, comp_all_features)
        
        model_time = time.time() - model_start_time
        log_progress(rank, 
            f"\nFinal Results for {model_name}:\n"
            f"- FID Score: {final_fid:.4f}\n"
            f"- Total Images: {len(ref_all_features)}\n"
            f"- Total Processing Time: {model_time:.2f}s"
        )
        
        fid_scores[model_name] = final_fid
        
        # Clean up comparison model and features
        del comp_model, ref_features_list, comp_features_list, ref_all_features, comp_all_features
        torch.cuda.empty_cache()
        log_progress(rank, f"Completed comparison with {model_name}")
    
    # Clean up reference model
    del ref_model
    torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    log_progress(rank, f"\nCompleted all FID computations in {total_time:.2f}s")
    
    return fid_scores


def main():
    """Main entry point for FID computation."""
    args = parse_args()
    
    # Setup distributed processing
    local_rank, rank, world_size, device = setup_distributed()
    
    # Create output directory and setup logging
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        setup_logging(args.output_dir, rank)
        log_progress(rank, 
            f"Starting FID computation with configuration:\n"
            f"- Reference Model: {args.reference_model}\n"
            f"- Comparison Models: {args.comparison_models}\n"
            f"- Number of Images: {args.num_images}\n"
            f"- Chunk Size: {args.chunk_size}\n"
            f"- Batch Size: {args.batch_size}\n"
            f"- Image Size: {args.img_size}\n"
            f"- Number of GPUs: {world_size}"
        )
    
    try:
        # Compute FID scores
        fid_scores = compute_fid_scores(args, rank, world_size, device)
        
        # Save results
        if rank == 0:
            results_file = Path(args.output_dir) / "fid_scores.txt"
            with open(results_file, "w") as f:
                f.write(f"FID Scores (Reference: {args.reference_model})\n")
                f.write(f"Computation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("-" * 50 + "\n")
                for model_name, score in fid_scores.items():
                    f.write(f"{model_name}: {score:.4f}\n")
            
            log_progress(rank, f"\nResults saved to {results_file}")
    
    except Exception as e:
        log_progress(rank, f"Error during FID computation: {str(e)}", level="error")
        raise
    finally:
        # Clean up distributed environment
        cleanup_distributed()
        if rank == 0:
            log_progress(rank, "Cleaned up distributed environment")


if __name__ == "__main__":
    main() 