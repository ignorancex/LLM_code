#!/usr/bin/env python
"""
Unified attack script implementing three variants of PGD attacks:
1. baseline: Naive ResNet18 classifier (both target and gradient source)
2. yu_2019: Yu2019AttributionClassifier (both target and gradient source)  
3. authprint: AuthPrint reconstructor (target) with ResNet18 classifier (gradient source)
"""
import argparse
import logging
import os
import sys
import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from dataclasses import dataclass

# Add the parent directory (project root) to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.default_config import get_default_config
from models.model_utils import load_stylegan2_model
from models.reconstructor import ReconstructorSD_L, ReconstructorSD_M, ReconstructorSD_S, StyleGAN2Reconstructor
from models.yu_2019_classifier import Yu2019AttributionClassifier
from utils.distributed import setup_distributed, cleanup_distributed
from utils.logging_utils import setup_logging
from utils.image_transforms import quantize_model_weights, downsample_and_upsample
from utils.model_loading import load_pretrained_models
from utils.metrics import calculate_fid
from utils.checkpoint import load_checkpoint


class NaiveClassifier(nn.Module):
    """ResNet-18 based binary classifier for baseline and authprint gradient source."""
    def __init__(self, img_size=256):
        super().__init__()
        # Load pretrained ResNet-18 but modify for our use case
        self.resnet = models.resnet18(pretrained=True)
        # Modify first conv layer to accept 3 channels and maintain size
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        # Modify final layer for binary classification
        self.resnet.fc = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x is expected to be in [-1, 1]
        # Convert to [0, 1] for ResNet
        x = (x + 1) / 2
        x = self.resnet(x)
        return self.sigmoid(x)


class Yu2019Classifier(nn.Module):
    """Yu2019 Attribution Classifier wrapper for binary classification."""
    def __init__(self, img_size=256):
        super().__init__()
        self.classifier = Yu2019AttributionClassifier(
            num_channels=3,            # RGB input
            resolution=img_size,       # Will be 256
            label_size=1,             # Binary classification
            fmap_base=64,             # Base feature maps
            fmap_decay=1.0,           # Standard decay (powers of 2)
            fmap_max=512,             # Higher max channels
            latent_res=4,             # 4x4 final resolution
            mode='postpool',          # Postpool mode
            switching_res=32,         # Switch later to allow more conv layers
            use_wscale=False,         # Disable weight scaling for now
            mbstd_group_size=0,       # No minibatch stddev
            fused_scale=False         # Disable fused convolutions for debugging
        )
        self.sigmoid = nn.Sigmoid()   # For binary output
    
    def forward(self, x):
        # x is expected to be in [-1, 1]
        x = (x + 1) / 2  # Convert to [0, 1]
        x = self.classifier(x)  # Now guaranteed to be [B, 1]
        return self.sigmoid(x)  # Returns [B, 1] to match NaiveClassifier


class ReconstructorWrapper:
    """Wrapper for the AuthPrint reconstructor to provide binary prediction interface."""
    def __init__(self, reconstructor, threshold, image_pixel_indices):
        self.reconstructor = reconstructor
        self.threshold = threshold
        self.image_pixel_indices = image_pixel_indices
    
    def extract_features(self, x):
        """Extract features using the same method as the evaluator."""
        batch_size = x.shape[0]
        flattened = x.view(batch_size, -1)
        return flattened[:, self.image_pixel_indices]
    
    def predict(self, x):
        """Return True if image is detected as original, False if undetected.
        Calculates MSE per sample, matching evaluator's implementation."""
        with torch.no_grad():
            features = self.extract_features(x)
            pred_values = self.reconstructor(x)
            # Calculate MSE per sample (dim=1)
            mse = torch.mean(torch.pow(pred_values - features, 2), dim=1)
            # Compare each sample's MSE with threshold
            return mse <= self.threshold


class ImageQualityMetrics:
    """Wrapper for various image quality metrics."""
    def __init__(self, device):
        self.device = device
        # Initialize LPIPS
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        # Initialize SSIM
        self.ssim = StructuralSimilarityIndexMeasure().to(device)
        # Initialize PSNR
        self.psnr = PeakSignalNoiseRatio().to(device)
    
    def compute_metrics(self, original, perturbed):
        """Compute all image quality metrics."""
        return {
            'lpips': self.lpips_fn(original, perturbed).item(),
            'psnr': self.psnr(original, perturbed).item(),
            'ssim': self.ssim(original, perturbed).item()
        }


class UnifiedAttack:
    """Unified attack implementation supporting three variants."""
    
    def __init__(
        self,
        attack_type,
        original_model,
        device=None,
        rank=0,
        config=None
    ):
        self.attack_type = attack_type
        self.original_model = original_model
        self.device = device
        self.rank = rank
        self.config = config
        
        # Initialize quality metrics for evaluation
        self.quality_metrics = ImageQualityMetrics(self.device)
        
        # Initialize attack-specific components
        self.evade_target = None
        self.gradient_source = None
        
    def setup_attack_components(self, reconstructor_wrapper=None):
        """Setup evade target and gradient source based on attack type."""
        img_size = self.config.model.img_size
        
        if self.attack_type == "baseline":
            # Both target and gradient source are naive classifier
            self.evade_target = NaiveClassifier(img_size).to(self.device)
            self.gradient_source = self.evade_target  # Same model
            
        elif self.attack_type == "yu_2019":
            # Both target and gradient source are Yu2019 classifier
            self.evade_target = Yu2019Classifier(img_size).to(self.device)
            self.gradient_source = self.evade_target  # Same model
            
        elif self.attack_type == "authprint":
            # Target is AuthPrint reconstructor, gradient source is naive classifier
            if reconstructor_wrapper is None:
                raise ValueError("AuthPrint attack requires reconstructor_wrapper")
            self.evade_target = reconstructor_wrapper
            self.gradient_source = NaiveClassifier(img_size).to(self.device)
            
        else:
            raise ValueError(f"Unknown attack type: {self.attack_type}")
    
    def train_gradient_source(self, negative_model, negative_case_type=None):
        """Train the gradient source model (classifier) if needed."""
        # AuthPrint reconstructor doesn't need training, it's already loaded
        if self.attack_type == "authprint" and isinstance(self.evade_target, ReconstructorWrapper):
            pass  # Reconstructor is already trained
        
        # Train classifier for baseline and yu_2019
        if isinstance(self.gradient_source, (NaiveClassifier, Yu2019Classifier)):
            if self.rank == 0:
                logging.info(f"Training gradient source ({self.attack_type}) classifier...")
            
            optimizer = torch.optim.Adam(self.gradient_source.parameters(), lr=self.config.attack.classifier_lr)
            criterion = nn.BCELoss()
            
            self.gradient_source.train()
            for iteration in range(self.config.attack.classifier_iterations):
                # Generate batch of z vectors
                z_batch = torch.randn(self.config.attack.batch_size, self.original_model.z_dim, device=self.device)
                
                # Generate original images
                with torch.no_grad():
                    if hasattr(self.original_model, 'module'):
                        w = self.original_model.module.mapping(z_batch, None)
                        original_images = self.original_model.module.synthesis(w, noise_mode="const")
                    else:
                        w = self.original_model.mapping(z_batch, None)
                        original_images = self.original_model.synthesis(w, noise_mode="const")
                    
                    # Generate negative case images
                    if negative_case_type and negative_case_type.startswith('downsample'):
                        size = int(negative_case_type.split('_')[1])
                        negative_images = downsample_and_upsample(original_images, downsample_size=size)
                    else:
                        if hasattr(negative_model, 'module'):
                            w = negative_model.module.mapping(z_batch, None)
                            negative_images = negative_model.module.synthesis(w, noise_mode="const")
                        else:
                            w = negative_model.mapping(z_batch, None)
                            negative_images = negative_model.synthesis(w, noise_mode="const")
                
                # Combine into training batch
                all_images = torch.cat([original_images, negative_images])
                labels = torch.cat([
                    torch.ones(self.config.attack.batch_size, 1),
                    torch.zeros(self.config.attack.batch_size, 1)
                ]).to(self.device)
                
                # Train step
                optimizer.zero_grad()
                predictions = self.gradient_source(all_images)
                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()
                
                if self.rank == 0 and iteration % self.config.attack.log_interval == 0:
                    logging.info(f"Gradient source training iteration {iteration}/{self.config.attack.classifier_iterations}, Loss: {loss.item():.4f}")
            
            self.gradient_source.eval()
    
    def check_evade_target(self, image):
        """Check if image fools the evade target."""
        if isinstance(self.evade_target, ReconstructorWrapper):
            # For AuthPrint reconstructor
            return self.evade_target.predict(image)
        else:
            # For classifiers
            with torch.no_grad():
                pred = self.evade_target(image)
                return pred.item() > 0.5  # Binary classification threshold
    
    def pgd_attack(self, image, z, initial_check_info=None):
        """Perform PGD attack using trained gradient source."""
        if self.config.attack.enable_step_size_sweep:
            # Try each step size and return the best result
            best_result = None
            best_success = False
            best_step_size = None
            original_step_size = self.config.attack.pgd_step_size
            
            for step_size in self.config.attack.step_size_sweep_values:
                if self.rank == 0:
                    logging.info(f"\nTrying step size: {step_size}")
                self.config.attack.pgd_step_size = step_size
                result = self._single_pgd_attack(image, z, initial_check_info)
                
                # Update best result if this is more successful
                if not best_success and result[1]:  # If we haven't succeeded yet and this one succeeds
                    best_result = result
                    best_success = True
                    best_step_size = step_size
                elif best_success and result[1]:  # If we've succeeded before and this one succeeds
                    # Compare based on image quality metrics
                    current_metrics = self.quality_metrics.compute_metrics(image, result[0])
                    best_metrics = self.quality_metrics.compute_metrics(image, best_result[0])
                    if current_metrics['lpips'] < best_metrics['lpips']:  # Lower LPIPS is better
                        best_result = result
                        best_step_size = step_size
            
            # Restore original step size
            self.config.attack.pgd_step_size = original_step_size
            
            if self.rank == 0 and best_step_size is not None:
                logging.info(f"Best step size was: {best_step_size}")
            
            return best_result if best_result is not None else (image, False, self.config.attack.pgd_steps, initial_check_info)
        else:
            return self._single_pgd_attack(image, z, initial_check_info)
    
    def _single_pgd_attack(self, image, z, initial_check_info=None):
        """Perform a single PGD attack with current step size."""
        perturbed = image.clone()
        criterion = nn.BCELoss()
        
        # Target is 1 (original class)
        target = torch.ones(1, 1).to(self.device)
        
        # Initialize momentum
        momentum = torch.zeros_like(image)
        
        # First check if the image already fools the evade target
        if self.check_evade_target(image):
            if self.rank == 0:
                logging.info("Image already fools evade target, no attack needed")
            return image, True, 0, initial_check_info
        
        # Track best result for AuthPrint attacks
        best_info = initial_check_info
        
        for step in range(self.config.attack.pgd_steps):
            perturbed.requires_grad = True
            
            # Forward pass through gradient source
            pred = self.gradient_source(perturbed)
            loss = - criterion(pred, target)
            
            # Compute gradient
            grad = torch.autograd.grad(loss, perturbed)[0]
            
            # Update momentum using MI-FGSM formula
            momentum = self.config.attack.momentum * momentum + grad / torch.norm(grad, p=1)
            
            # PGD step with momentum
            with torch.no_grad():
                perturbed = perturbed + self.config.attack.pgd_step_size * momentum.sign()
                
                # Project back to epsilon ball
                delta = perturbed - image
                delta = torch.clamp(delta, -self.config.attack.epsilon, self.config.attack.epsilon)
                perturbed = image + delta
                perturbed = torch.clamp(perturbed, -1, 1)
                
                # Track MSE for AuthPrint attacks
                if isinstance(self.evade_target, ReconstructorWrapper):
                    features = self.evade_target.extract_features(perturbed)
                    pred_values = self.evade_target.reconstructor(perturbed)
                    current_mse = torch.mean(torch.pow(pred_values - features, 2), dim=1).item()
                    if best_info is None or current_mse < best_info:
                        best_info = current_mse
            
            # Check if evade target is fooled
            if self.check_evade_target(perturbed):
                if self.rank == 0:
                    logging.info(f"Attack succeeded at step {step+1} with step size {self.config.attack.pgd_step_size:.6f}")
                    if isinstance(self.evade_target, ReconstructorWrapper):
                        logging.info(f"Best MSE achieved: {best_info:.6f}")
                return perturbed, True, step + 1, best_info
        
        if self.rank == 0:
            if isinstance(self.evade_target, ReconstructorWrapper):
                logging.info(f"Best MSE achieved: {best_info:.6f}")
            logging.info(f"Attack failed with step size {self.config.attack.pgd_step_size:.6f}")
        
        return perturbed, False, self.config.attack.pgd_steps, best_info
    
    def attack_negative_case(self, negative_model, num_samples, negative_case_type=None):
        """Attack a specific negative case using PGD."""
        if self.rank == 0:
            logging.info(f"Attacking {negative_case_type or 'base'} case with {self.attack_type} attack...")
            if self.config.attack.enable_step_size_sweep:
                logging.info("Step size sweep enabled. Will try the following step sizes:")
                logging.info(", ".join(f"{s:.6f}" for s in self.config.attack.step_size_sweep_values))
        
        # Train gradient source if needed
        self.train_gradient_source(negative_model, negative_case_type)
        
        # If step size sweep is enabled, we'll store results for each step size
        if self.config.attack.enable_step_size_sweep:
            all_sweep_results = {}
            original_step_size = self.config.attack.pgd_step_size
            
            for step_size in self.config.attack.step_size_sweep_values:
                if self.rank == 0:
                    logging.info(f"\nTrying step size: {step_size}")
                self.config.attack.pgd_step_size = step_size
                
                successful_attacks = 0
                total_queries = 0
                results = []
                
                # Lists to store images for FID calculation
                original_images = []
                perturbed_images = []
                
                # Lists to store initial check info (MSE for AuthPrint, prediction confidence for others)
                initial_check_infos = []
                best_infos = []
                
                for i in range(num_samples):
                    if self.rank == 0:
                        logging.info(f"\nProcessing sample {i+1}/{num_samples} with step size {step_size}")
                    
                    # Generate z vector and image from negative model
                    z = torch.randn(1, negative_model.z_dim, device=self.device)
                    with torch.no_grad():
                        if hasattr(negative_model, 'module'):
                            w = negative_model.module.mapping(z, None)
                            negative_img = negative_model.module.synthesis(w, noise_mode="const")
                        else:
                            w = negative_model.mapping(z, None)
                            negative_img = negative_model.synthesis(w, noise_mode="const")
                    
                    # Apply transformations for special cases
                    if negative_case_type and negative_case_type.startswith('downsample'):
                        size = int(negative_case_type.split('_')[1])
                        negative_img = downsample_and_upsample(negative_img, downsample_size=size)
                    
                    # Get initial check info
                    initial_check_info = None
                    if isinstance(self.evade_target, ReconstructorWrapper):
                        features = self.evade_target.extract_features(negative_img)
                        pred_values = self.evade_target.reconstructor(negative_img)
                        initial_check_info = torch.mean(torch.pow(pred_values - features, 2), dim=1).item()
                        if self.rank == 0:
                            logging.info(f"Initial MSE: {initial_check_info:.6f}")
                    
                    # Perform PGD attack
                    perturbed, success, queries_used, best_info = self.pgd_attack(negative_img, z, initial_check_info)
                    total_queries += queries_used
                    
                    # Store check info
                    initial_check_infos.append(initial_check_info)
                    best_infos.append(best_info)
                    
                    if success:
                        successful_attacks += 1
                        metrics = self.quality_metrics.compute_metrics(negative_img, perturbed)
                        
                        # Store images for FID calculation
                        original_images.append(negative_img)
                        perturbed_images.append(perturbed)
                        
                        results.append({
                            'metrics': metrics,
                            'queries': queries_used
                        })
                        
                        if self.rank == 0:
                            logging.info(f"Attack successful - Queries: {queries_used}, Metrics: LPIPS={metrics['lpips']:.4f}, PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}")
                    else:
                        if self.rank == 0:
                            logging.info(f"Attack failed after {queries_used} steps")
                
                # Calculate FID score for successful attacks
                fid = float('inf')
                if successful_attacks > 0:
                    original_images = torch.cat(original_images, dim=0)
                    perturbed_images = torch.cat(perturbed_images, dim=0)
                    
                    try:
                        fid = calculate_fid(
                            (original_images + 1) / 2,  # Convert to [0, 1] range
                            (perturbed_images + 1) / 2,
                            batch_size=self.config.attack.batch_size,
                            device=self.device
                        )
                    except Exception as e:
                        if self.rank == 0:
                            logging.error(f"Error computing FID score: {str(e)}")
                
                # Aggregate results for this step size
                success_rate = successful_attacks / num_samples
                avg_queries = total_queries / num_samples
                
                # Compute statistics for initial and best check info
                avg_metrics = {
                    'lpips': np.mean([r['metrics']['lpips'] for r in results]) if results else float('inf'),
                    'psnr': np.mean([r['metrics']['psnr'] for r in results]) if results else 0,
                    'ssim': np.mean([r['metrics']['ssim'] for r in results]) if results else 0,
                    'fid': fid
                }
                
                # Add MSE statistics for AuthPrint attacks
                if isinstance(self.evade_target, ReconstructorWrapper):
                    initial_check_infos = [x for x in initial_check_infos if x is not None]
                    best_infos = [x for x in best_infos if x is not None]
                    if initial_check_infos:
                        avg_metrics.update({
                            'initial_mse_mean': np.mean(initial_check_infos),
                            'initial_mse_std': np.std(initial_check_infos),
                            'best_mse_mean': np.mean(best_infos),
                            'best_mse_std': np.std(best_infos)
                        })
                
                all_sweep_results[step_size] = {
                    'success_rate': success_rate,
                    'avg_queries': avg_queries,
                    'avg_metrics': avg_metrics,
                    'results': results
                }
            
            # Restore original step size
            self.config.attack.pgd_step_size = original_step_size
            
            # Find best step size based on success rate
            best_step_size = max(all_sweep_results.keys(), key=lambda k: all_sweep_results[k]['success_rate'])
            if self.rank == 0:
                logging.info(f"\nBest step size: {best_step_size} with success rate: {all_sweep_results[best_step_size]['success_rate']*100:.2f}%")
            
            return all_sweep_results[best_step_size]
        
        else:
            # Original non-sweep implementation
            successful_attacks = 0
            total_queries = 0
            results = []
            
            # Lists to store images for FID calculation
            original_images = []
            perturbed_images = []
            
            # Lists to store initial check info (MSE for AuthPrint, prediction confidence for others)
            initial_check_infos = []
            best_infos = []
            
            for i in range(num_samples):
                if self.rank == 0:
                    logging.info(f"\nProcessing sample {i+1}/{num_samples}")
                
                # Generate z vector and image from negative model
                z = torch.randn(1, negative_model.z_dim, device=self.device)
                with torch.no_grad():
                    if hasattr(negative_model, 'module'):
                        w = negative_model.module.mapping(z, None)
                        negative_img = negative_model.module.synthesis(w, noise_mode="const")
                    else:
                        w = negative_model.mapping(z, None)
                        negative_img = negative_model.synthesis(w, noise_mode="const")
                
                # Apply transformations for special cases
                if negative_case_type and negative_case_type.startswith('downsample'):
                    size = int(negative_case_type.split('_')[1])
                    negative_img = downsample_and_upsample(negative_img, downsample_size=size)
                
                # Get initial check info
                initial_check_info = None
                if isinstance(self.evade_target, ReconstructorWrapper):
                    features = self.evade_target.extract_features(negative_img)
                    pred_values = self.evade_target.reconstructor(negative_img)
                    initial_check_info = torch.mean(torch.pow(pred_values - features, 2), dim=1).item()
                    if self.rank == 0:
                        logging.info(f"Initial MSE: {initial_check_info:.6f}")
                
                # Perform PGD attack
                perturbed, success, queries_used, best_info = self.pgd_attack(negative_img, z, initial_check_info)
                total_queries += queries_used
                
                # Store check info
                initial_check_infos.append(initial_check_info)
                best_infos.append(best_info)
                
                if success:
                    successful_attacks += 1
                    metrics = self.quality_metrics.compute_metrics(negative_img, perturbed)
                    
                    # Store images for FID calculation
                    original_images.append(negative_img)
                    perturbed_images.append(perturbed)
                    
                    results.append({
                        'metrics': metrics,
                        'queries': queries_used
                    })
                    
                    if self.rank == 0:
                        logging.info(f"Attack successful - Queries: {queries_used}, Metrics: LPIPS={metrics['lpips']:.4f}, PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}")
                else:
                    if self.rank == 0:
                        logging.info(f"Attack failed after {queries_used} steps")
            
            # Calculate FID score for successful attacks
            fid = float('inf')
            if successful_attacks > 0:
                original_images = torch.cat(original_images, dim=0)
                perturbed_images = torch.cat(perturbed_images, dim=0)
                
                try:
                    fid = calculate_fid(
                        (original_images + 1) / 2,  # Convert to [0, 1] range
                        (perturbed_images + 1) / 2,
                        batch_size=self.config.attack.batch_size,
                        device=self.device
                    )
                except Exception as e:
                    if self.rank == 0:
                        logging.error(f"Error computing FID score: {str(e)}")
            
            # Aggregate results
            success_rate = successful_attacks / num_samples
            avg_queries = total_queries / num_samples
            
            # Compute statistics for initial and best check info
            avg_metrics = {
                'lpips': np.mean([r['metrics']['lpips'] for r in results]) if results else float('inf'),
                'psnr': np.mean([r['metrics']['psnr'] for r in results]) if results else 0,
                'ssim': np.mean([r['metrics']['ssim'] for r in results]) if results else 0,
                'fid': fid
            }
            
            # Add MSE statistics for AuthPrint attacks
            if isinstance(self.evade_target, ReconstructorWrapper):
                initial_check_infos = [x for x in initial_check_infos if x is not None]
                best_infos = [x for x in best_infos if x is not None]
                if initial_check_infos:
                    avg_metrics.update({
                        'initial_mse_mean': np.mean(initial_check_infos),
                        'initial_mse_std': np.std(initial_check_infos),
                        'best_mse_mean': np.mean(best_infos),
                        'best_mse_std': np.std(best_infos)
                    })
            
            return {
                'success_rate': success_rate,
                'avg_queries': avg_queries,
                'avg_metrics': avg_metrics,
                'results': results
            }
    
    def attack_all_cases(self, pretrained_models, quantized_models, num_samples):
        """Attack all negative cases and return combined results."""
        all_results = {}
        
        # Attack pretrained models
        for model_name, model in pretrained_models.items():
            if self.rank == 0:
                logging.info(f"\nAttacking pretrained model: {model_name}")
            all_results[model_name] = self.attack_negative_case(model, num_samples)
        
        # Attack quantized models if enabled
        if self.config.attack.enable_quantization:
            for precision, model in quantized_models.items():
                case_name = f"quantization_{precision}"
                if self.rank == 0:
                    logging.info(f"\nAttacking quantized model: {precision}")
                all_results[case_name] = self.attack_negative_case(model, num_samples, case_name)
        
        # Attack downsample cases if enabled
        if self.config.attack.enable_downsampling:
            for size in [128, 224]:
                case_name = f"downsample_{size}"
                if self.rank == 0:
                    logging.info(f"\nAttacking downsample case: {size}")
                all_results[case_name] = self.attack_negative_case(self.original_model, num_samples, case_name)
        
        return all_results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Unified Attack Script for Fingerprint Detection")
    
    # Attack type selection
    parser.add_argument("--attack_type", type=str, required=True,
                        choices=["baseline", "yu_2019", "authprint"],
                        help="Type of attack to perform")
    
    # Model configuration
    parser.add_argument("--stylegan2_url", type=str,
                        default="https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq70k-paper256-ada.pkl",
                        help="URL for the pretrained StyleGAN2 model")
    parser.add_argument("--stylegan2_local_path", type=str,
                        default="ffhq70k-paper256-ada.pkl",
                        help="Local path to store/load the StyleGAN2 model")
    parser.add_argument("--checkpoint_path", type=str,
                        help="Path to reconstructor checkpoint (required for authprint attack)")
    parser.add_argument("--img_size", type=int, default=256,
                        help="Image resolution")
    parser.add_argument("--image_pixel_set_seed", type=int, default=42,
                        help="Random seed for selecting pixel indices (authprint only)")
    parser.add_argument("--image_pixel_count", type=int, default=32,
                        help="Number of pixels to select from the image (authprint only)")
    
    # Attack configuration
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of samples to attack")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for classifier training and evaluation")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Maximum perturbation size")
    parser.add_argument("--detection_threshold", type=float, default=0.002883,
                        help="MSE threshold for detection (authprint only)")
    
    # Evaluation cases
    parser.add_argument("--enable_quantization", action="store_true",
                        help="Enable evaluation on quantized models")
    parser.add_argument("--enable_downsampling", action="store_true",
                        help="Enable evaluation on downsampling transformations")
    
    # Classifier training parameters
    parser.add_argument("--classifier_iterations", type=int, default=10000,
                        help="Number of iterations for classifier training")
    parser.add_argument("--classifier_lr", type=float, default=1e-4,
                        help="Learning rate for classifier training")
    
    # PGD parameters
    parser.add_argument("--pgd_step_size", type=float, default=0.01,
                        help="Step size for PGD attack")
    parser.add_argument("--pgd_steps", type=int, default=50,
                        help="Number of PGD steps")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum coefficient for PGD attack")
    
    # Step size sweep parameters
    parser.add_argument("--enable_step_size_sweep", action="store_true",
                        help="Enable step size sweep during attack")
    parser.add_argument("--step_size_sweep_values", type=str, default=None,
                        help="Comma-separated list of step sizes to try (e.g. '0.0001,0.0002,0.0005'). If not provided, uses default values.")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="unified_attack_results",
                        help="Directory to save attack results")
    
    args = parser.parse_args()
    
    # Parse step size sweep values if provided
    if args.step_size_sweep_values:
        try:
            args.step_size_sweep_values = [float(x.strip()) for x in args.step_size_sweep_values.split(',')]
        except ValueError:
            raise ValueError("Invalid step size sweep values. Please provide comma-separated numbers.")
    
    return args


def format_results_table(all_results, attack_type):
    """Format results into a nice table using fixed-width columns."""
    table_str = f"\n{attack_type.upper()} Attack Results Summary:\n"
    table_str += "-" * 200 + "\n"
    
    # Different header for AuthPrint attacks (includes MSE columns)
    if attack_type == "authprint":
        table_str += f"{'Negative Case':<30}{'Step Size':<15}{'Initial MSE':<25}{'Best MSE':<25}{'Success Rate':>15}{'Avg Queries':>15}{'LPIPS':>15}{'PSNR':>15}{'SSIM':>15}{'FID':>15}\n"
    else:
        table_str += f"{'Negative Case':<30}{'Step Size':<15}{'Success Rate':>15}{'Avg Queries':>15}{'LPIPS':>15}{'PSNR':>15}{'SSIM':>15}{'FID':>15}\n"
    
    table_str += "-" * 200 + "\n"
    
    # Add rows
    for case_name, results in all_results.items():
        metrics = results['avg_metrics']
        
        row = f"{case_name:<30}"
        row += f"{results.get('step_size', '-'):<15}"  # Add step size if available
        
        # Add MSE columns for AuthPrint attacks
        if attack_type == "authprint" and 'initial_mse_mean' in metrics:
            initial_mse = f"{metrics['initial_mse_mean']:.6f} ± {metrics['initial_mse_std']:.6f}"
            best_mse = f"{metrics['best_mse_mean']:.6f} ± {metrics['best_mse_std']:.6f}"
            row += f"{initial_mse:<25}"
            row += f"{best_mse:<25}"
        elif attack_type == "authprint":
            row += f"{'N/A':<25}{'N/A':<25}"
        
        row += f"{results['success_rate']*100:>15.2f}%"
        row += f"{results['avg_queries']:>15.1f}"
        row += f"{metrics['lpips']:>15.4f}"
        row += f"{metrics['psnr']:>15.2f}"
        row += f"{metrics['ssim']:>15.4f}"
        row += f"{metrics['fid']:>15.2f}\n"
        table_str += row
    
    table_str += "-" * 200 + "\n"
    
    # Add attack-specific footer information
    if attack_type == "authprint":
        table_str += f"Detection Threshold: {0.002883:.6f}\n"
    
    table_str += f"Attack Type: {attack_type}\n"
    
    # Add step size sweep information if enabled
    if hasattr(results, 'step_size_sweep_enabled') and results['step_size_sweep_enabled']:
        table_str += f"Step Size Sweep: Enabled (tried {len(results['step_size_sweep_values'])} values)\n"
        table_str += f"Step Size Values: {', '.join(f'{s:.6f}' for s in results['step_size_sweep_values'])}\n"
    
    return table_str


def main():
    """Main entry point for unified attack."""
    args = parse_args()
    
    # Validate arguments based on attack type
    if args.attack_type == "authprint" and args.checkpoint_path is None:
        raise ValueError("--checkpoint_path is required for authprint attack")
    
    # Setup distributed environment
    local_rank, rank, world_size, device = setup_distributed()
    
    # Load default configuration and update with args
    config = get_default_config()
    config.update_from_args(args, mode='attack')
    
    # No need to handle step size sweep arguments here as they are already handled in update_from_args()
    
    # Create output directory and setup logging
    if rank == 0:
        os.makedirs(config.output_dir, exist_ok=True)
        setup_logging(config.output_dir, rank)
        
        # Log configuration
        logging.info(f"Configuration:\n{config}")
        logging.info(f"Attack Type: {args.attack_type}")
        logging.info(f"Distributed setup: local_rank={local_rank}, rank={rank}, world_size={world_size}, device={device}")
        
        if config.attack.enable_step_size_sweep:
            logging.info("Step size sweep enabled")
            logging.info(f"Step size values: {config.attack.step_size_sweep_values}")
    
    try:
        # Load models
        if rank == 0:
            logging.info("Loading models...")
        
        # Load original StyleGAN2 model
        original_model = load_stylegan2_model(
            config.model.stylegan2_url,
            config.model.stylegan2_local_path,
            device
        )
        
        # Load pretrained models
        pretrained_models = load_pretrained_models(device, rank)
        
        # Setup quantized models
        quantized_models = {}
        try:
            if rank == 0:
                logging.info("Setting up quantized models...")
            for precision in ['int8', 'int4']:
                try:
                    quantized_model = quantize_model_weights(original_model, precision)
                    quantized_model.eval()
                    quantized_models[precision] = quantized_model
                    if rank == 0:
                        logging.info(f"Successfully created {precision} model")
                except Exception as e:
                    if rank == 0:
                        logging.error(f"Failed to create {precision} model: {str(e)}")
        except Exception as e:
            if rank == 0:
                logging.error(f"Error in quantization setup: {str(e)}")
        
        # Setup reconstructor wrapper for AuthPrint attacks
        reconstructor_wrapper = None
        if args.attack_type == "authprint":
            if rank == 0:
                logging.info("Loading reconstructor for AuthPrint attack...")
            
            # Initialize reconstructor based on model type and size
            reconstructor_output_dim = config.model.image_pixel_count
            
            if config.model.model_type == "stylegan2":
                reconstructor = StyleGAN2Reconstructor(
                    image_size=config.model.img_size,
                    channels=3,
                    output_dim=reconstructor_output_dim
                ).to(device)
                if rank == 0:
                    logging.info(f"Initialized StyleGAN2Reconstructor with output_dim={reconstructor_output_dim}")
            else:  # stable-diffusion
                reconstructor_class = {
                    "S": ReconstructorSD_S,
                    "M": ReconstructorSD_M,
                    "L": ReconstructorSD_L
                }[config.model.sd_reconstructor_size]
                
                reconstructor = reconstructor_class(
                    image_size=config.model.img_size,
                    channels=3,
                    output_dim=reconstructor_output_dim
                ).to(device)
                
                if rank == 0:
                    logging.info(f"Initialized SD-Reconstructor-{config.model.sd_reconstructor_size} with output_dim={reconstructor_output_dim}")
            
            reconstructor.eval()
            
            # Load checkpoint
            if rank == 0:
                logging.info(f"Loading checkpoint from {config.checkpoint_path}...")
            
            load_checkpoint(
                checkpoint_path=config.checkpoint_path,
                reconstructor=reconstructor,
                device=device
            )
            
            # Generate pixel indices (same as in evaluator)
            torch.manual_seed(config.model.image_pixel_set_seed)
            total_pixels = config.model.img_size * config.model.img_size * 3
            image_pixel_indices = torch.randperm(total_pixels)[:config.model.image_pixel_count].to(device)
            
            if rank == 0:
                logging.info(f"Generated {len(image_pixel_indices)} pixel indices with seed {config.model.image_pixel_set_seed}")
            
            # Create reconstructor wrapper
            reconstructor_wrapper = ReconstructorWrapper(
                reconstructor=reconstructor,
                threshold=config.attack.detection_threshold,
                image_pixel_indices=image_pixel_indices
            )
        
        # Create unified attacker
        attacker = UnifiedAttack(
            attack_type=args.attack_type,
            original_model=original_model,
            device=device,
            rank=rank,
            config=config  # Pass full config object
        )
        
        # Setup attack components
        attacker.setup_attack_components(reconstructor_wrapper)
        
        # Run attack on all cases
        if rank == 0:
            logging.info(f"Starting {args.attack_type} attacks on all negative cases...")
        
        all_results = attacker.attack_all_cases(
            pretrained_models=pretrained_models,
            quantized_models=quantized_models,
            num_samples=config.attack.num_samples
        )
        
        # Add step size sweep information to results
        if config.attack.enable_step_size_sweep:
            for case_results in all_results.values():
                case_results['step_size_sweep_enabled'] = True
                case_results['step_size_sweep_values'] = config.attack.step_size_sweep_values
        
        # Print results table
        if rank == 0:
            table = format_results_table(all_results, args.attack_type)
            logging.info(table)
    
    except Exception as e:
        if rank == 0:
            logging.error(f"Error in {args.attack_type} attack: {str(e)}")
            logging.error("Attack error:", exc_info=True)
        raise
    finally:
        # Clean up distributed environment
        cleanup_distributed()


if __name__ == "__main__":
    main() 