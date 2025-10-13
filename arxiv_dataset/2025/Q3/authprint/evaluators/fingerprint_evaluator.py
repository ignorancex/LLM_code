"""
Evaluator for generative model fingerprinting.
"""
import logging
import os
import re
from typing import Dict, Optional, List, Tuple, Any

import torch
import numpy as np
import random
from datasets import load_dataset

from config.default_config import Config
from models.reconstructor import ReconstructorSD_L, ReconstructorSD_M, ReconstructorSD_S, StyleGAN2Reconstructor
from models.base_model import BaseGenerativeModel
from utils.checkpoint import load_checkpoint
from utils.metrics import save_metrics_text, calculate_fid, extract_inception_features
from utils.distribution_metrics import (
    InceptionScore,
    calculate_kid,
    calculate_precision_recall,
    calculate_wasserstein,
    calculate_mmd
)
from utils.image_transforms import (
    quantize_model_weights, 
    downsample_and_upsample
)
from utils.model_loading import (
    load_pretrained_models,
    STYLEGAN2_MODELS,
    STABLE_DIFFUSION_MODELS
)


def clean_prompt(prompt: str) -> str:
    """
    Clean a prompt by removing excessive punctuation and normalizing whitespace.
    
    Args:
        prompt (str): Input prompt to clean.
        
    Returns:
        str: Cleaned prompt.
    """
    # Remove URLs
    prompt = re.sub(r'http\S+|www\.\S+', '', prompt)
    
    # Remove excessive punctuation (more than 1 of the same character)
    prompt = re.sub(r'([!?.]){2,}', r'\1', prompt)
    
    # Remove excessive whitespace
    prompt = ' '.join(prompt.split())
    
    # Remove <|endoftext|> tokens
    prompt = prompt.replace('<|endoftext|>', '')
    
    # Remove empty parentheses and brackets
    prompt = re.sub(r'\(\s*\)|\[\s*\]|\{\s*\}', '', prompt)
    
    # Normalize commas and colons
    prompt = re.sub(r'\s*,\s*', ', ', prompt)
    prompt = re.sub(r'\s*:\s*', ': ', prompt)
    
    return prompt.strip()


class FingerprintEvaluator:
    """
    Evaluator for generative model fingerprinting.
    """
    def __init__(
        self,
        config: Config,
        local_rank: int,
        rank: int,
        world_size: int,
        device: torch.device,
        selected_pretrained_models: Optional[List[str]] = None,
        custom_pretrained_models: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the evaluator.
        
        Args:
            config (Config): Configuration object.
            local_rank (int): Local process rank.
            rank (int): Global process rank.
            world_size (int): Total number of processes.
            device (torch.device): Device to run evaluation on.
            selected_pretrained_models (Optional[List[str]]): List of pretrained model names to use.
                If None or empty, all default models will be used.
            custom_pretrained_models (Optional[Dict[str, Any]]): Dictionary mapping
                custom model names to their configurations.
        """
        self.config = config
        self.local_rank = local_rank
        self.rank = rank
        self.world_size = world_size
        self.device = device
        
        # Store pretrained model selections
        self.selected_pretrained_models = selected_pretrained_models
        self.custom_pretrained_models = custom_pretrained_models or {}
        
        # Enable timing logs
        self.enable_timing = getattr(self.config.evaluate, 'enable_timing_logs', True)
        if self.rank == 0 and self.enable_timing:
            logging.info("Timing logs are enabled for detailed process monitoring")
        
        # Initialize timing dictionary for tracking durations
        self.timing_stats = {}
        
        # Initialize models
        self.generative_model = None
        self.reconstructor = None
        
        # Initialize pixel selection parameters
        self.image_pixel_indices = None
        self.image_pixel_count = self.config.model.image_pixel_count
        self.image_pixel_set_seed = self.config.model.image_pixel_set_seed
        
        # Initialize prompt dataset if multi-prompt mode is enabled
        self.prompts: Optional[List[str]] = None
        if self.config.model.enable_multi_prompt:
            self._load_prompt_dataset()
        
        if self.rank == 0:
            logging.info(f"Using direct pixel prediction with {self.image_pixel_count} pixels and seed {self.image_pixel_set_seed}")
            if self.config.model.enable_multi_prompt:
                logging.info("Multi-prompt evaluation mode is enabled")
        
        # Initialize quantized models dictionary
        self.quantized_models = {}
        
        # Setup models
        self.setup_models()
        
        # Load pretrained models with custom configuration
        if self.selected_pretrained_models or self.custom_pretrained_models:
            # Create combined model dictionary
            model_dict = {}
            
            # Add selected default models
            if not self.selected_pretrained_models:
                # If no models specified, use all default models
                if self.config.model.model_type == "stylegan2":
                    model_dict.update(STYLEGAN2_MODELS)
                else:
                    model_dict.update(STABLE_DIFFUSION_MODELS)
            else:
                # Add only selected default models
                default_models = STYLEGAN2_MODELS if self.config.model.model_type == "stylegan2" else STABLE_DIFFUSION_MODELS
                for model_name in self.selected_pretrained_models:
                    if model_name in default_models:
                        model_dict[model_name] = default_models[model_name]
                    elif self.rank == 0:
                        logging.warning(f"Requested model '{model_name}' not found in default models")
            
            # Add custom models
            model_dict.update(self.custom_pretrained_models)
            
            # Load the models
            self.pretrained_models = load_pretrained_models(
                device=self.device,
                rank=self.rank,
                model_type=self.config.model.model_type,
                selected_models=model_dict,
                img_size=self.config.model.img_size,
                enable_cpu_offload=self.config.model.sd_enable_cpu_offload if self.config.model.model_type == "stable-diffusion" else False,
                dtype=getattr(torch, self.config.model.sd_dtype) if self.config.model.model_type == "stable-diffusion" else torch.float32
            )
        else:
            # Load all default models
            self.pretrained_models = load_pretrained_models(
                device=self.device,
                rank=self.rank,
                model_type=self.config.model.model_type,
                img_size=self.config.model.img_size,
                enable_cpu_offload=self.config.model.sd_enable_cpu_offload if self.config.model.model_type == "stable-diffusion" else False,
                dtype=getattr(torch, self.config.model.sd_dtype) if self.config.model.model_type == "stable-diffusion" else torch.float32
            )
    
    def _generate_pixel_indices(self) -> None:
        """
        Generate random pixel indices for image-based approach.
        """
        # Set seed for reproducibility
        torch.manual_seed(self.image_pixel_set_seed)
        
        # Calculate total number of pixels
        img_size = self.config.model.img_size
        channels = 3  # RGB image
        total_pixels = channels * img_size * img_size
        
        # Generate random indices (without replacement) on CPU first
        if self.image_pixel_count > total_pixels:
            if self.rank == 0:
                logging.warning(f"Requested {self.image_pixel_count} pixels exceeds total pixels {total_pixels}. Using all pixels.")
            self.image_pixel_count = total_pixels
            self.image_pixel_indices = torch.arange(total_pixels)
        else:
            self.image_pixel_indices = torch.randperm(total_pixels)[:self.image_pixel_count]
        
        # Move to device after generation
        self.image_pixel_indices = self.image_pixel_indices.to(self.device)
        
        if self.rank == 0:
            logging.info(f"Generated {len(self.image_pixel_indices)} pixel indices with seed {self.image_pixel_set_seed}")
            logging.info(f"Selected pixel indices: {self.image_pixel_indices.tolist()}")

    def extract_image_partial(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract partial image using selected pixel indices.
        
        Args:
            images (torch.Tensor): Batch of images [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Batch of flattened pixel values at selected indices
        """
        batch_size = images.shape[0]
        
        # Flatten the spatial dimensions using a view operation
        flattened = images.view(batch_size, -1)
        
        # Get values at selected indices: [batch_size, pixel_count]
        image_partial = flattened.index_select(1, self.image_pixel_indices)
        
        return image_partial
    
    def setup_models(self):
        """
        Initialize and set up all models.
        """
        # Initialize generative model
        if self.rank == 0:
            logging.info(f"Loading {self.config.model.model_type} model...")
            
        model_class = self.config.model.get_model_class()
        model_kwargs = self.config.model.get_model_kwargs(self.device)
        self.generative_model = model_class(**model_kwargs)
        
        # Initialize reconstructor based on model type and size
        reconstructor_output_dim = self.image_pixel_count  # For direct pixel prediction
        
        if self.config.model.model_type == "stylegan2":
            self.reconstructor = StyleGAN2Reconstructor(
                image_size=self.config.model.img_size,
                channels=3,
                output_dim=reconstructor_output_dim
            ).to(self.device)
            if self.rank == 0:
                logging.info(f"Initialized StyleGAN2Reconstructor with output_dim={reconstructor_output_dim}")
        else:  # stable-diffusion
            reconstructor_class = {
                "S": ReconstructorSD_S,
                "M": ReconstructorSD_M,
                "L": ReconstructorSD_L
            }[self.config.model.sd_reconstructor_size]
            
            self.reconstructor = reconstructor_class(
                image_size=self.config.model.img_size,
                channels=3,
                output_dim=reconstructor_output_dim
            ).to(self.device)
            
            if self.rank == 0:
                logging.info(f"Initialized SD-Reconstructor-{self.config.model.sd_reconstructor_size} with output_dim={reconstructor_output_dim}")
        
        self.reconstructor.eval()
        
        # Generate pixel indices
        self._generate_pixel_indices()
        
        # Load checkpoint
        if self.rank == 0:
            logging.info(f"Loading checkpoint from {self.config.checkpoint_path}...")
        
        load_checkpoint(
            checkpoint_path=self.config.checkpoint_path,
            reconstructor=self.reconstructor,
            device=self.device
        )
        
        # Setup quantized and pruned models if enabled in evaluation config
        self.quantized_models = {}
        self.pruned_models = {}
        
        if getattr(self.config.evaluate, 'enable_quantization', True):
            try:
                # Dictionary of quantization levels and their descriptions
                quantization_levels = {
                    'int8': 'int8 (8-bit)',
                    'int7': 'int7 (7-bit)',
                    'int6': 'int6 (6-bit)',
                    'int5': 'int5 (5-bit)',
                    'int4': 'int4 (4-bit)'
                }
                
                for precision, description in quantization_levels.items():
                    if self.rank == 0:
                        logging.info(f"Setting up {description} quantized model...")
                    
                    try:
                        if self.config.model.model_type == "stylegan2":
                            quantized_model = quantize_model_weights(self.generative_model, precision)
                        else:  # stable-diffusion
                            quantized_model = self.generative_model.quantize(precision)
                        quantized_model.eval()
                        self.quantized_models[precision] = quantized_model
                        if self.rank == 0:
                            logging.info(f"Successfully created {description} quantized model")
                    except Exception as e:
                        if self.rank == 0:
                            logging.error(f"Failed to create {description} model: {str(e)}")
                            logging.error(f"{description} quantization error:", exc_info=True)
                
                if self.rank == 0:
                    logging.info(f"Available quantized models: {list(self.quantized_models.keys())}")
            
            except Exception as e:
                if self.rank == 0:
                    logging.error(f"Error in quantization setup: {str(e)}")
                    logging.error("Quantization setup error:", exc_info=True)
                    logging.error("Continuing without quantized models...")
                self.quantized_models = {}
        else:
            if self.rank == 0:
                logging.info("Quantization disabled in evaluation config")
            self.quantized_models = {}
        
        # Setup pruned models
        if getattr(self.config.evaluate, 'enable_pruning', True):
            try:
                sparsity_levels = getattr(self.config.model, 'pruning_sparsity_levels', [0.01, 0.05, 0.1, 0.25, 0.375, 0.5, 0.625, 0.75, 0.9, 0.95, 0.99])
                pruning_methods = getattr(self.config.model, 'pruning_methods', ['magnitude', 'random'])
                
                for sparsity in sparsity_levels:
                    for method in pruning_methods:
                        model_key = f'pruned_{method}_{int(sparsity*100)}'
                        if self.rank == 0:
                            logging.info(f"Setting up pruned model with {int(sparsity*100)}% sparsity using {method} pruning...")
                        
                        try:
                            # Use model-specific pruning method
                            pruned_model = self.generative_model.prune(sparsity=sparsity, method=method)
                            pruned_model.eval()
                            self.pruned_models[model_key] = pruned_model
                            if self.rank == 0:
                                logging.info(f"Successfully created pruned model: {model_key}")
                        except Exception as e:
                            if self.rank == 0:
                                logging.error(f"Failed to create pruned model {model_key}: {str(e)}")
                                logging.error(f"Pruning error for {model_key}:", exc_info=True)
            except Exception as e:
                if self.rank == 0:
                    logging.error(f"Error in pruning setup: {str(e)}")
                    logging.error("Pruning setup error:", exc_info=True)
                    logging.error("Continuing without pruned models...")
                self.pruned_models = {}
        else:
            if self.rank == 0:
                logging.info("Pruning disabled in evaluation config")
            self.pruned_models = {}
    
    def evaluate_batch(self) -> Dict[str, float]:
        """
        Run batch evaluation to compute metrics.
        
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics.
        """
        if self.rank == 0:
            logging.info(f"Running batch evaluation with {self.config.evaluate.num_samples} samples...")
            if self.config.model.enable_multi_prompt:
                logging.info("Using multi-prompt evaluation mode")
        
        try:
            # Set seed for reproducibility
            if hasattr(self.config.evaluate, 'seed') and self.config.evaluate.seed is not None:
                np.random.seed(self.config.evaluate.seed)
                torch.manual_seed(self.config.evaluate.seed)
                random.seed(self.config.evaluate.seed)  # For prompt sampling
                if self.rank == 0:
                    logging.info(f"Using fixed random seed {self.config.evaluate.seed} for evaluation")
            
            # Create empty accumulators
            batch_size = self.config.evaluate.batch_size
            num_samples = self.config.evaluate.num_samples
            num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
            
            # Generate latents or prepare generation based on model type
            if self.config.model.model_type == "stylegan2":
                all_z_original = torch.randn(num_samples, self.generative_model.z_dim, device=self.device)
                all_z_negative = torch.randn(num_samples, self.generative_model.z_dim, device=self.device)
                gen_kwargs = {"noise_mode": "const"}
            else:  # stable-diffusion
                all_z_original = None  # Not used for SD
                all_z_negative = None  # Not used for SD
                gen_kwargs = self.config.model.get_generation_kwargs()
            
            # Process batches for original model
            mse_per_sample = []  # Changed from mse_values to mse_per_sample
            
            with torch.no_grad():
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, num_samples)
                    current_batch_size = end_idx - start_idx
                    
                    # For Stable Diffusion, update prompts if multi-prompt mode is enabled
                    if self.config.model.model_type == "stable-diffusion":
                        prompts = self._sample_prompts(current_batch_size)
                        gen_kwargs["prompt"] = prompts
                        if self.rank == 0 and i == 0:  # Log sample prompts from first batch
                            logging.info(f"Sample prompts for evaluation: {prompts[:3]}")
                    
                    # For Stable Diffusion, update prompts if multi-prompt mode is enabled
                    if self.config.model.model_type == "stable-diffusion":
                        prompts = self._sample_prompts(current_batch_size)
                        gen_kwargs["prompt"] = prompts
                        if self.rank == 0 and i == 0:  # Log sample prompts from first batch
                            logging.info(f"Sample prompts for evaluation: {prompts[:3]}")
                    
                    # Generate images based on model type
                    if self.config.model.model_type == "stylegan2":
                        z = all_z_original[start_idx:end_idx]
                        x = self.generative_model.generate_images(
                            batch_size=current_batch_size,
                            device=self.device,
                            z=z,  # Pass the latent vectors explicitly
                            **gen_kwargs
                        )
                    else:  # stable-diffusion
                        x = self.generative_model.generate_images(
                            batch_size=current_batch_size,
                            device=self.device,
                            **gen_kwargs
                        )
                    
                    # Extract features (real pixel values)
                    features = self.extract_image_partial(x)
                    true_values = features
                    
                    # Predict values
                    pred_values = self.reconstructor(x)
                    
                    # Calculate metrics - now calculating MSE per sample
                    mse = torch.mean(torch.pow(pred_values - true_values, 2), dim=1).cpu().numpy()
                    mse_per_sample.extend(mse.tolist())  # extend list with individual sample MSEs
                    
                    # Progress reporting
                    if self.rank == 0 and num_batches > 10 and (i+1) % max(1, num_batches//10) == 0:
                        logging.info(f"Processed {i+1}/{num_batches} batches")
            
            # Convert to numpy array for calculations
            mse_per_sample = np.array(mse_per_sample)
            
            # Combine results - now taking mean and std of per-sample MSEs
            mse_all = np.mean(mse_per_sample)
            mse_std = np.std(mse_per_sample)
            
            # Calculate threshold for 95% TPR - using per-sample MSEs
            threshold = np.percentile(mse_per_sample, 95)
            if self.rank == 0:
                logging.info(f"Threshold at 95% TPR: {threshold:.6f}")
            
            # Calculate metrics
            metrics = {
                'pixel_mse_mean': mse_all,
                'pixel_mse_std': mse_std,
                'pixel_mse_values': mse_per_sample,  # Now storing all per-sample MSEs
                'threshold_95tpr': threshold
            }
            
            # Evaluate negative samples
            negative_results = self._evaluate_negative_samples(all_z_original, all_z_negative, threshold, gen_kwargs)
            if negative_results:
                metrics['negative_results'] = negative_results
            
            # Save metrics
            if self.rank == 0:
                save_metrics_text(metrics, self.config.output_dir)
            
            return metrics
                
        except Exception as e:
            if self.rank == 0:
                logging.error(f"Error in batch evaluation: {str(e)}")
                logging.error(str(e), exc_info=True)
            return {}
    
    def _evaluate_negative_samples(
        self,
        original_z: Optional[torch.Tensor],
        negative_z: Optional[torch.Tensor],
        threshold: float,
        gen_kwargs: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate negative samples by comparing against pretrained models and transformations.
        Computes FPR at 95% TPR threshold for each negative case.
        
        Args:
            original_z (Optional[torch.Tensor]): Original model latent vectors for FID comparison (StyleGAN2 only)
            negative_z (Optional[torch.Tensor]): Negative model latent vectors for evaluation (StyleGAN2 only)
            threshold (float): MSE threshold at 95% TPR from original model
            gen_kwargs (Dict[str, Any]): Generation kwargs for the model
            
        Returns:
            dict: Dictionary mapping negative sample types to their metrics
        """
        negative_results = {}
        batch_size = self.config.evaluate.batch_size
        num_samples = self.config.evaluate.num_samples
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        # Initialize Inception Score calculator
        inception_score_calc = InceptionScore(device=self.device)
        
        # Define the evaluations to run
        evaluations_to_run = []
        
        # Add pretrained model evaluations - use all available models
        for model_name in self.pretrained_models.keys():
            evaluations_to_run.append((model_name, None))
        
        # Add transformations based on model type
        if self.config.model.model_type == "stylegan2":
            # Add quantization evaluations for StyleGAN2 models
            if self.quantized_models:
                for precision in self.quantized_models.keys():
                    evaluations_to_run.append((None, f'quantization_{precision}'))
                    if self.rank == 0:
                        logging.info(f"Added {precision} quantization evaluation")
        
        # Add pruning evaluations
        if self.pruned_models:
            for model_key in self.pruned_models.keys():
                evaluations_to_run.append((None, model_key))
        
        # Add downsample evaluations for both sizes
        evaluations_to_run.append((None, 'downsample_16'))
        evaluations_to_run.append((None, 'downsample_32'))
        evaluations_to_run.append((None, 'downsample_64'))
        evaluations_to_run.append((None, 'downsample_96'))
        evaluations_to_run.append((None, 'downsample_128'))
        evaluations_to_run.append((None, 'downsample_160'))
        evaluations_to_run.append((None, 'downsample_192'))
        evaluations_to_run.append((None, 'downsample_224'))
        
        # Add pixel manipulation evaluation
        evaluations_to_run.append((None, 'set_pixels_minus_one'))
        
        # Add pixel manipulation evaluations
        evaluations_to_run.append((None, 'set_random_pixels_minus_one'))
        evaluations_to_run.append((None, 'set_mixed_50_50_pixels_minus_one'))
        evaluations_to_run.append((None, 'set_mixed_75_25_pixels_minus_one'))
        evaluations_to_run.append((None, 'set_mixed_25_75_pixels_minus_one'))
        evaluations_to_run.append((None, 'set_mixed_10_90_pixels_minus_one'))
        evaluations_to_run.append((None, 'set_mixed_5_95_pixels_minus_one'))
        evaluations_to_run.append((None, 'set_mixed_1_99_pixels_minus_one'))
        
        total_evals = len(evaluations_to_run)
        if self.rank == 0:
            logging.info(f"Running {total_evals} evaluations with extended distribution metrics...")
        
        # Generate original images for distribution comparison
        original_images = []
        original_features = []
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_samples)
                current_batch_size = end_idx - start_idx
                
                if self.config.model.model_type == "stylegan2":
                    z = original_z[start_idx:end_idx]
                    x = self.generative_model.generate_images(
                        batch_size=current_batch_size,
                        device=self.device,
                        z=z,
                        **gen_kwargs
                    )
                else:  # stable-diffusion
                    x = self.generative_model.generate_images(
                        batch_size=current_batch_size,
                        device=self.device,
                        **gen_kwargs
                    )
                
                original_images.append(x)
                # Extract inception features for distribution metrics
                features = extract_inception_features(x, batch_size=batch_size, device=self.device)
                original_features.append(features)
        
        # Concatenate all original images and features
        original_images = torch.cat(original_images, dim=0)
        original_features = np.concatenate(original_features, axis=0)
        
        # Calculate Inception Score for original distribution
        is_mean, is_std = inception_score_calc.calculate_score(
            (original_images + 1) / 2,  # Convert to [0, 1] range
            batch_size=batch_size
        )
        
        if self.rank == 0:
            logging.info(f"Original distribution Inception Score: {is_mean:.4f} ± {is_std:.4f}")
        
        # Generate negative case images and compute all metrics
        with torch.no_grad():
            for idx, (model_name, transformation) in enumerate(evaluations_to_run):
                key = model_name if model_name else transformation
                if self.rank == 0:
                    logging.info(f"Starting evaluation for: {key}")
                
                mse_per_sample = []
                negative_images = []
                negative_features = []
                
                # Process in batches
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, num_samples)
                    current_batch_size = end_idx - start_idx
                    
                    # Generate negative sample images
                    if model_name is not None:
                        model = self.pretrained_models[model_name]
                        if self.config.model.model_type == "stylegan2":
                            z = negative_z[start_idx:end_idx]
                            x = model.generate_images(
                                batch_size=current_batch_size,
                                device=self.device,
                                z=z,
                                **gen_kwargs
                            )
                        else:
                            x = model.generate_images(
                                batch_size=current_batch_size,
                                device=self.device,
                                **gen_kwargs
                            )
                    else:
                        if self.config.model.model_type == "stylegan2":
                            z = negative_z[start_idx:end_idx]
                            x = self.generative_model.generate_images(
                                batch_size=current_batch_size,
                                device=self.device,
                                z=z,
                                **gen_kwargs
                            )
                        else:
                            x = self.generative_model.generate_images(
                                batch_size=current_batch_size,
                                device=self.device,
                                **gen_kwargs
                            )
                        
                        if transformation and transformation.startswith('quantization'):
                            precision = transformation.split('_')[-1]
                            if precision in self.quantized_models:
                                model = self.quantized_models[precision]
                                if self.config.model.model_type == "stylegan2":
                                    x = model.generate_images(
                                        batch_size=current_batch_size,
                                        device=self.device,
                                        z=z,
                                        **gen_kwargs
                                    )
                                else:
                                    x = model.generate_images(
                                        batch_size=current_batch_size,
                                        device=self.device,
                                        **gen_kwargs
                                    )
                            else:
                                if self.rank == 0 and i == 0:
                                    logging.warning(f"Quantized model for precision {precision} not found")
                        elif transformation and transformation.startswith('pruned_'):
                            if transformation in self.pruned_models:
                                model = self.pruned_models[transformation]
                                if self.config.model.model_type == "stylegan2":
                                    x = model.generate_images(
                                        batch_size=current_batch_size,
                                        device=self.device,
                                        z=z,
                                        **gen_kwargs
                                    )
                                else:
                                    x = model.generate_images(
                                        batch_size=current_batch_size,
                                        device=self.device,
                                        **gen_kwargs
                                    )
                            else:
                                if self.rank == 0 and i == 0:
                                    logging.warning(f"Pruned model {transformation} not found")
                        elif transformation.startswith('downsample'):
                            downsample_size = int(transformation.split('_')[1])
                            x = downsample_and_upsample(x, downsample_size=downsample_size)
                        elif transformation == 'set_pixels_minus_one':
                            x = self._set_pixels_to_value(x, value=-1.0)
                        elif transformation == 'set_random_pixels_minus_one':
                            # Use a different random seed (e.g. original seed + 1000)
                            random_seed = self.image_pixel_set_seed + 1000
                            random_indices = self._generate_random_pixel_indices(random_seed)
                            x = self._set_pixels_to_value(x, value=-1.0, pixel_indices=random_indices)
                        elif transformation == 'set_mixed_50_50_pixels_minus_one':
                            mixed_indices = self._mix_pixel_indices(0.5, self.image_pixel_set_seed + 2000)
                            x = self._set_pixels_to_value(x, value=-1.0, pixel_indices=mixed_indices)
                        elif transformation == 'set_mixed_75_25_pixels_minus_one':
                            mixed_indices = self._mix_pixel_indices(0.75, self.image_pixel_set_seed + 3000)
                            x = self._set_pixels_to_value(x, value=-1.0, pixel_indices=mixed_indices)
                        elif transformation == 'set_mixed_25_75_pixels_minus_one':
                            mixed_indices = self._mix_pixel_indices(0.25, self.image_pixel_set_seed + 4000)
                            x = self._set_pixels_to_value(x, value=-1.0, pixel_indices=mixed_indices)
                        elif transformation == 'set_mixed_10_90_pixels_minus_one':
                            mixed_indices = self._mix_pixel_indices(0.10, self.image_pixel_set_seed + 5000)
                            x = self._set_pixels_to_value(x, value=-1.0, pixel_indices=mixed_indices)
                        elif transformation == 'set_mixed_5_95_pixels_minus_one':
                            mixed_indices = self._mix_pixel_indices(0.05, self.image_pixel_set_seed + 6000)
                            x = self._set_pixels_to_value(x, value=-1.0, pixel_indices=mixed_indices)
                        elif transformation == 'set_mixed_1_99_pixels_minus_one':
                            mixed_indices = self._mix_pixel_indices(0.01, self.image_pixel_set_seed + 7000)
                            x = self._set_pixels_to_value(x, value=-1.0, pixel_indices=mixed_indices)
                    
                    # Store images and extract features
                    negative_images.append(x)
                    features = extract_inception_features(x, batch_size=batch_size, device=self.device)
                    negative_features.append(features)
                    
                    # Calculate MSE (existing code)
                    features = self.extract_image_partial(x)
                    true_values = features
                    pred_values = self.reconstructor(x)
                    mse = torch.mean(torch.pow(pred_values - true_values, 2), dim=1).cpu().numpy()
                    mse_per_sample.extend(mse.tolist())
                
                # Combine all negative images and features
                negative_images = torch.cat(negative_images, dim=0)
                negative_features = np.concatenate(negative_features, axis=0)
                
                # Calculate all distribution metrics
                fid_score = calculate_fid(
                    (original_images + 1) / 2,
                    (negative_images + 1) / 2,
                    batch_size=batch_size,
                    device=self.device
                )
                
                kid_score = calculate_kid(original_features, negative_features)
                
                is_mean, is_std = inception_score_calc.calculate_score(
                    (negative_images + 1) / 2,
                    batch_size=batch_size
                )
                
                precision, recall = calculate_precision_recall(
                    original_features,
                    negative_features
                )
                
                wasserstein_dist = calculate_wasserstein(
                    original_features,
                    negative_features
                )
                
                mmd_score = calculate_mmd(
                    original_features,
                    negative_features
                )
                
                # Calculate standard metrics (existing code)
                mse_per_sample = np.array(mse_per_sample)
                mse_all = np.mean(mse_per_sample)
                mse_std = np.std(mse_per_sample)
                fpr = np.mean(mse_per_sample <= threshold)
                
                # Store all metrics
                negative_results[key] = {
                    'mse_mean': mse_all,
                    'mse_std': mse_std,
                    'mse_values': mse_per_sample,
                    'fpr_at_95tpr': fpr,
                    'fid_score': fid_score,
                    'kid_score': kid_score,
                    'inception_score_mean': is_mean,
                    'inception_score_std': is_std,
                    'precision': precision,
                    'recall': recall,
                    'wasserstein': wasserstein_dist,
                    'mmd': mmd_score
                }
                
                if self.rank == 0:
                    logging.info(
                        f"Results for {key}:\n"
                        f"- FPR at 95% TPR: {fpr:.4f}\n"
                        f"- FID Score: {fid_score:.4f}\n"
                        f"- KID Score: {kid_score:.4f}\n"
                        f"- Inception Score: {is_mean:.4f} ± {is_std:.4f}\n"
                        f"- Precision/Recall: {precision:.4f}/{recall:.4f}\n"
                        f"- Wasserstein: {wasserstein_dist:.4f}\n"
                        f"- MMD: {mmd_score:.4f}"
                    )
                
                # Progress reporting
                if self.rank == 0 and (idx+1) % max(1, total_evals//5) == 0:
                    logging.info(f"Completed {idx+1}/{total_evals} evaluations")
        
        return negative_results
    
    def evaluate(self):
        """
        Run batch evaluation to compute metrics.
        
        Returns:
            dict: Evaluation metrics.
        """
        # Log evaluation configuration
        if self.rank == 0:
            logging.info("Starting evaluation with the following configuration:")
            logging.info(f"  Pixel count: {self.image_pixel_count}")
            logging.info(f"  Pixel seed: {self.image_pixel_set_seed}")
        
        metrics = self.evaluate_batch()
        
        # Print metrics table if we have results and are on rank 0
        if metrics and self.rank == 0:
            logging.info("\nEvaluation Results:")
            logging.info("-" * 200)
            logging.info(
                f"{'Model/Transform':<30}"
                f"{'MSE Mean':>12}{'MSE Std':>12}"
                f"{'FPR@95%':>10}{'FID':>10}{'KID':>10}"
                f"{'IS Mean':>10}{'IS Std':>10}"
                f"{'Prec':>8}{'Rec':>8}"
                f"{'Wass':>10}{'MMD':>10}"
            )
            logging.info("-" * 200)
            
            # Print original model results and threshold
            logging.info(
                f"{'Original Model':<30}"
                f"{metrics['pixel_mse_mean']:>12.4f}{metrics['pixel_mse_std']:>12.4f}"
                f"{'N/A':>10}{'N/A':>10}{'N/A':>10}"
                f"{'N/A':>10}{'N/A':>10}"
                f"{'N/A':>8}{'N/A':>8}"
                f"{'N/A':>10}{'N/A':>10}"
            )
            logging.info(f"Threshold at 95% TPR: {metrics['threshold_95tpr']:.6f}")
            
            # Print negative sample results if available
            if 'negative_results' in metrics:
                logging.info("\nNegative Sample Results:")
                for name, result in metrics['negative_results'].items():
                    logging.info(
                        f"{name:<30}"
                        f"{result['mse_mean']:>12.4f}{result['mse_std']:>12.4f}"
                        f"{result['fpr_at_95tpr']:>10.4f}{result['fid_score']:>10.2f}{result['kid_score']:>10.4f}"
                        f"{result['inception_score_mean']:>10.2f}{result['inception_score_std']:>10.2f}"
                        f"{result['precision']:>8.2f}{result['recall']:>8.2f}"
                        f"{result['wasserstein']:>10.4f}{result['mmd']:>10.4f}"
                    )
            
            logging.info("-" * 200)
        
        return metrics

    def _load_prompt_dataset(self) -> None:
        """
        Load prompts from the dataset file.
        """
        if self.config.model.prompt_source == "local":
            self._load_prompt_dataset_local()
        elif self.config.model.prompt_source == "diffusiondb":
            self._load_prompt_dataset_diffusiondb()
        else:  # parti-prompts
            self._load_prompt_dataset_parti()

    def _load_prompt_dataset_local(self) -> None:
        """
        Load prompts from a local file.
        """
        if self.rank == 0:
            logging.info(f"Loading prompts from local file: {self.config.model.prompt_dataset_path}")
        
        try:
            with open(self.config.model.prompt_dataset_path, 'r', encoding='utf-8') as f:
                all_prompts = [line.strip() for line in f if line.strip()]
            
            # Sample the specified number of prompts
            if len(all_prompts) > self.config.model.prompt_dataset_size:
                self.prompts = random.sample(all_prompts, self.config.model.prompt_dataset_size)
            else:
                self.prompts = all_prompts
                if self.rank == 0:
                    logging.warning(f"Prompt dataset contains fewer prompts ({len(all_prompts)}) "
                                  f"than requested ({self.config.model.prompt_dataset_size})")
            
            if self.rank == 0:
                logging.info(f"Loaded {len(self.prompts)} prompts from local file")
                logging.info(f"Sample prompts: {self.prompts[:3]}")
        
        except Exception as e:
            if self.rank == 0:
                logging.error(f"Error loading prompt dataset: {str(e)}")
            raise

    def _load_prompt_dataset_diffusiondb(self) -> None:
        """
        Load prompts from DiffusionDB dataset.
        """
        if self.rank == 0:
            logging.info("Loading prompts from DiffusionDB dataset")
        
        try:
            # Load the metadata table from DiffusionDB
            subset_mapping = {
                "2m_random_10k": "2m_random_10k",  # Using random 10k subset instead of full dataset
                "large_random_10k": "large_random_10k",
                "2m_random_5k": "2m_random_5k",
            }
            subset = subset_mapping[self.config.model.diffusiondb_subset]
            dataset = load_dataset("poloclub/diffusiondb", subset, split="train", trust_remote_code=True)
            
            # Extract and clean all unique prompts
            all_prompts = []
            raw_prompts = list(set(dataset["prompt"]))
            
            if self.rank == 0:
                logging.info(f"Found {len(raw_prompts)} unique prompts before cleaning")
            
            for prompt in raw_prompts:
                if not prompt:  # Skip empty prompts
                    continue
                    
                cleaned_prompt = clean_prompt(prompt)
                if cleaned_prompt and len(cleaned_prompt.split()) <= 50:  # Only keep reasonably sized prompts
                    all_prompts.append(cleaned_prompt)
            
            if self.rank == 0:
                logging.info(f"Retained {len(all_prompts)} prompts after cleaning")
            
            # Sample the specified number of prompts
            if len(all_prompts) > self.config.model.prompt_dataset_size:
                self.prompts = random.sample(all_prompts, self.config.model.prompt_dataset_size)
            else:
                self.prompts = all_prompts
                if self.rank == 0:
                    logging.warning(f"DiffusionDB contains fewer clean prompts ({len(all_prompts)}) "
                                  f"than requested ({self.config.model.prompt_dataset_size})")
            
            if self.rank == 0:
                logging.info(f"Loaded {len(self.prompts)} prompts from DiffusionDB")
                logging.info("Sample prompts after cleaning:")
                for i, prompt in enumerate(self.prompts[:10]):  # Show first 10 prompts
                    logging.info(f"  {i+1}. {prompt}")
        
        except Exception as e:
            if self.rank == 0:
                logging.error(f"Error loading DiffusionDB dataset: {str(e)}")
            raise

    def _load_prompt_dataset_parti(self) -> None:
        """
        Load prompts from the Parti-Prompts dataset for evaluation.
        Uses the evaluation split (20% by default) of the specified category.
        """
        if self.rank == 0:
            logging.info(f"Loading evaluation prompts from Parti-Prompts dataset for category: {self.config.model.parti_prompts_category}")
        
        try:
            # Load the Parti-Prompts dataset
            dataset = load_dataset("nateraw/parti-prompts", split="train", trust_remote_code=True)
            
            # Filter by category if specified
            if self.config.model.parti_prompts_category:
                dataset = dataset.filter(lambda x: x["Category"] == self.config.model.parti_prompts_category)
                if self.rank == 0:
                    logging.info(f"Found {len(dataset)} prompts in category '{self.config.model.parti_prompts_category}'")
            
            # Extract and clean all prompts
            all_prompts = []
            for item in dataset:
                if not item["Prompt"]:  # Skip empty prompts
                    continue
                
                cleaned_prompt = clean_prompt(item["Prompt"])
                if cleaned_prompt:
                    all_prompts.append(cleaned_prompt)
            
            if self.rank == 0:
                logging.info(f"Retained {len(all_prompts)} prompts after cleaning")
            
            # Split into train and eval sets
            random.shuffle(all_prompts)  # Shuffle before splitting
            split_idx = int(len(all_prompts) * self.config.model.train_eval_split_ratio)
            eval_prompts = all_prompts[split_idx:]  # Use the evaluation split
            
            # Sample the specified number of prompts for evaluation
            if len(eval_prompts) > self.config.model.prompt_dataset_size:
                self.prompts = random.sample(eval_prompts, self.config.model.prompt_dataset_size)
            else:
                self.prompts = eval_prompts
                if self.rank == 0:
                    logging.warning(f"Evaluation set contains fewer prompts ({len(eval_prompts)}) "
                                  f"than requested ({self.config.model.prompt_dataset_size})")
            
            if self.rank == 0:
                logging.info(f"Using {len(self.prompts)} prompts for evaluation")
                logging.info("Sample evaluation prompts:")
                for i, prompt in enumerate(self.prompts[:10]):  # Show first 10 prompts
                    logging.info(f"  {i+1}. {prompt}")
        
        except Exception as e:
            if self.rank == 0:
                logging.error(f"Error loading Parti-Prompts dataset: {str(e)}")
            raise

    def _sample_prompts(self, batch_size: int) -> List[str]:
        """
        Sample prompts for the current batch.
        
        Args:
            batch_size (int): Number of prompts to sample.
            
        Returns:
            List[str]: List of sampled prompts.
        """
        if not self.config.model.enable_multi_prompt or not self.prompts:
            return [self.config.model.sd_prompt] * batch_size
        
        return random.sample(self.prompts, min(batch_size, len(self.prompts)))

    # Add helper method for pixel manipulation
    def _set_pixels_to_value(self, images: torch.Tensor, value: float = -1.0, pixel_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Set the selected pixel indices to a specific value.
        
        Args:
            images (torch.Tensor): Input images tensor [B, C, H, W]
            value (float): Value to set the selected pixels to
            pixel_indices (Optional[torch.Tensor]): Custom pixel indices to set. If None, uses self.image_pixel_indices
            
        Returns:
            torch.Tensor: Modified images with selected pixels set to value
        """
        batch_size = images.shape[0]
        modified_images = images.clone()
        
        # Use provided indices or default to self.image_pixel_indices
        indices_to_use = pixel_indices if pixel_indices is not None else self.image_pixel_indices
        
        # Reshape images to [batch_size, channels*height*width]
        flattened = modified_images.view(batch_size, -1)
        
        # Set selected indices to value
        flattened[:, indices_to_use] = value
        
        # Reshape back to original shape
        return flattened.view_as(images)

    def _generate_random_pixel_indices(self, seed: int) -> torch.Tensor:
        """
        Generate random pixel indices with a specific seed.
        
        Args:
            seed (int): Random seed for pixel selection
            
        Returns:
            torch.Tensor: Random pixel indices
        """
        # Set seed for reproducibility
        torch.manual_seed(seed)
        
        # Calculate total number of pixels
        img_size = self.config.model.img_size
        channels = 3  # RGB image
        total_pixels = channels * img_size * img_size
        
        # Generate random indices (without replacement)
        if self.image_pixel_count > total_pixels:
            indices = torch.arange(total_pixels)
        else:
            indices = torch.randperm(total_pixels)[:self.image_pixel_count]
        
        return indices.to(self.device)

    # Add helper method for mixing pixel indices
    def _mix_pixel_indices(self, original_ratio: float, random_seed: int) -> torch.Tensor:
        """
        Mix original pixel indices with random ones at specified ratio.
        
        Args:
            original_ratio (float): Ratio of original pixels to keep (0.0 to 1.0)
            random_seed (int): Random seed for generating new pixel indices
            
        Returns:
            torch.Tensor: Mixed pixel indices
        """
        total_pixels = len(self.image_pixel_indices)
        num_original = int(total_pixels * original_ratio)
        num_random = total_pixels - num_original
        
        # Get original indices
        original_indices = self.image_pixel_indices[:num_original]
        
        # Generate random indices ensuring no overlap with original indices
        if num_random > 0:
            torch.manual_seed(random_seed)
            
            # Calculate total available pixels
            img_size = self.config.model.img_size
            channels = 3
            total_available = channels * img_size * img_size
            
            # Create mask of available indices (excluding original indices)
            available_mask = torch.ones(total_available, dtype=torch.bool, device=self.device)
            available_mask[self.image_pixel_indices] = False
            available_indices = torch.nonzero(available_mask).squeeze()
            
            # Sample random indices from available ones
            random_indices = available_indices[torch.randperm(len(available_indices))[:num_random]]
            
            # Combine original and random indices
            mixed_indices = torch.cat([original_indices, random_indices])
        else:
            mixed_indices = original_indices
        
        return mixed_indices