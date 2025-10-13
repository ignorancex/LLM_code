#!/usr/bin/env python
"""
Query-based attack script for StyleGAN fingerprinting.
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
from utils.distributed import setup_distributed, cleanup_distributed
from utils.logging_utils import setup_logging
from utils.image_transforms import quantize_model_weights, downsample_and_upsample
from utils.model_loading import load_pretrained_models
from utils.metrics import calculate_fid, InceptionV3
from utils.checkpoint import load_checkpoint


class ClassifierModel(nn.Module):
    """ResNet-18 based binary classifier."""
    def __init__(self):
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


class ReconstructorWrapper:
    """Wrapper for the reconstructor to provide binary prediction interface."""
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



class QueryBasedAttack:
    """Query-based attack implementation for StyleGAN fingerprinting using PGD."""
    def __init__(
        self,
        original_model,
        reconstructor,
        device=None,
        rank=0,
        batch_size=32,
        config=None
    ):
        self.original_model = original_model
        self.reconstructor = reconstructor
        self.device = device or next(reconstructor.parameters()).device
        self.rank = rank
        self.batch_size = batch_size
        self.config = config
        
        # Initialize quality metrics for evaluation only
        self.quality_metrics = ImageQualityMetrics(self.device)
    
    def train_classifier(self, negative_model, negative_case_type=None):
        """Train classifier to distinguish between original and negative images."""
        if self.rank == 0:
            logging.info("Starting classifier training...")
        
        classifier = ClassifierModel().to(self.device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=self.config.classifier_lr)
        criterion = nn.BCELoss()
        
        classifier.train()
        for iteration in range(self.config.classifier_iterations):
            # Generate batch of z vectors
            z_batch = torch.randn(self.config.batch_size, self.original_model.z_dim, device=self.device)
            
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
                torch.ones(self.config.batch_size, 1),
                torch.zeros(self.config.batch_size, 1)
            ]).to(self.device)
            
            # Train step
            optimizer.zero_grad()
            predictions = classifier(all_images)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            if self.rank == 0 and iteration % self.config.log_interval == 0:
                logging.info(f"Classifier training iteration {iteration}/{self.config.classifier_iterations}, Loss: {loss.item():.4f}")
        
        classifier.eval()
        return classifier
    
    def pgd_attack(self, image, classifier, z):
        """Perform PGD attack using trained classifier with momentum."""
        perturbed = image.clone()
        criterion = nn.BCELoss()
        
        # Target is 1 (original class)
        target = torch.ones(1, 1).to(self.device)
        
        # Initialize momentum
        momentum = torch.zeros_like(image)
        
        # First check if the image is already predicted as True by reconstructor
        # This initial check doesn't count as a query since no perturbation was made
        with torch.no_grad():
            features = self.reconstructor.extract_features(image)
            pred_values = self.reconstructor.reconstructor(image)
            initial_mse = torch.mean(torch.pow(pred_values - features, 2), dim=1).item()
            if self.rank == 0:
                logging.info(f"Initial MSE: {initial_mse:.6f}")
        
        if self.reconstructor.predict(image):
            if self.rank == 0:
                logging.info("Image already predicted as True by reconstructor, no attack needed")
            return image, True, 0, initial_mse, initial_mse  # Return initial MSE as best since no attack needed
        
        best_mse = initial_mse
        for step in range(self.config.pgd_steps):
            perturbed.requires_grad = True
            
            # Forward pass through classifier
            pred = classifier(perturbed)
            loss = criterion(pred, target)
            
            # Compute gradient
            grad = torch.autograd.grad(loss, perturbed)[0]
            
            # Update momentum
            momentum = self.config.momentum * momentum + (1 - self.config.momentum) * grad
            
            # PGD step with momentum
            with torch.no_grad():
                perturbed = perturbed + self.config.pgd_step_size * momentum.sign()
                
                # Project back to epsilon ball
                delta = perturbed - image
                delta = torch.clamp(delta, -self.config.epsilon, self.config.epsilon)
                perturbed = image + delta
                perturbed = torch.clamp(perturbed, -1, 1)
                
                # Calculate current MSE
                features = self.reconstructor.extract_features(perturbed)
                pred_values = self.reconstructor.reconstructor(perturbed)
                current_mse = torch.mean(torch.pow(pred_values - features, 2), dim=1).item()
                best_mse = min(best_mse, current_mse)
            
            # Check if reconstructor is fooled - this counts as a query since we made a perturbation
            if self.reconstructor.predict(perturbed):
                if self.rank == 0:
                    logging.info(f"Attack succeeded at step {step+1}")
                    logging.info(f"Best MSE achieved: {best_mse:.6f}")
                return perturbed, True, step + 1, initial_mse, best_mse
        
        if self.rank == 0:
            logging.info(f"Best MSE achieved: {best_mse:.6f}")
        return perturbed, False, self.config.pgd_steps, initial_mse, best_mse
    
    def attack_negative_case(self, negative_model, num_samples, negative_case_type=None):
        """Attack a specific negative case using PGD."""
        if self.rank == 0:
            logging.info(f"Training classifier for {negative_case_type or 'base'} case...")
        
        # Train classifier first
        classifier = self.train_classifier(negative_model, negative_case_type)
        
        successful_attacks = 0
        total_queries = 0
        results = []
        
        # Lists to store images for FID calculation
        original_images = []
        perturbed_images = []
        
        # Lists to store MSE scores
        initial_mses = []
        best_mses = []
        
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
            
            # Perform PGD attack
            perturbed, success, queries_used, initial_mse, best_mse = self.pgd_attack(negative_img, classifier, z)
            total_queries += queries_used
            
            # Store MSE scores
            initial_mses.append(initial_mse)
            best_mses.append(best_mse)
            
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
                    batch_size=self.config.batch_size,
                    device=self.device
                )
            except Exception as e:
                if self.rank == 0:
                    logging.error(f"Error computing FID score: {str(e)}")
        
        # Convert MSE lists to numpy arrays for statistics
        initial_mses = np.array(initial_mses)
        best_mses = np.array(best_mses)
        
        # Aggregate results
        success_rate = successful_attacks / num_samples
        avg_queries = total_queries / num_samples
        
        avg_metrics = {
            'lpips': np.mean([r['metrics']['lpips'] for r in results]) if results else float('inf'),
            'psnr': np.mean([r['metrics']['psnr'] for r in results]) if results else 0,
            'ssim': np.mean([r['metrics']['ssim'] for r in results]) if results else 0,
            'fid': fid,
            'initial_mse_mean': np.mean(initial_mses),
            'initial_mse_std': np.std(initial_mses),
            'best_mse_mean': np.mean(best_mses),
            'best_mse_std': np.std(best_mses)
        }
        
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
        
        # Attack quantized models
        for precision, model in quantized_models.items():
            case_name = f"quantization_{precision}"
            if self.rank == 0:
                logging.info(f"\nAttacking quantized model: {precision}")
            all_results[case_name] = self.attack_negative_case(model, num_samples, case_name)
        
        # Attack downsample cases
        for size in [128, 224]:
            case_name = f"downsample_{size}"
            if self.rank == 0:
                logging.info(f"\nAttacking downsample case: {size}")
            all_results[case_name] = self.attack_negative_case(self.original_model, num_samples, case_name)
        
        return all_results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Query-based attack for StyleGAN2 Fingerprinting")
    
    # Model configuration
    parser.add_argument("--stylegan2_url", type=str,
                        default="https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq70k-paper256-ada.pkl",
                        help="URL for the pretrained StyleGAN2 model")
    parser.add_argument("--stylegan2_local_path", type=str,
                        default="ffhq70k-paper256-ada.pkl",
                        help="Local path to store/load the StyleGAN2 model")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to reconstructor checkpoint to attack")
    parser.add_argument("--img_size", type=int, default=256,
                        help="Image resolution")
    parser.add_argument("--image_pixel_set_seed", type=int, default=42,
                        help="Random seed for selecting pixel indices")
    parser.add_argument("--image_pixel_count", type=int, default=32,
                        help="Number of pixels to select from the image")
    
    # Attack configuration
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of samples to attack")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for classifier training and evaluation")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Maximum perturbation size")
    parser.add_argument("--detection_threshold", type=float, default=0.002883,
                        help="MSE threshold for detection (95% TPR threshold)")
    
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
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="pgd_attack_authprint_results",
                        help="Directory to save attack results")
    
    return parser.parse_args()


def format_results_table(all_results):
    """Format results into a nice table using fixed-width columns."""
    # Print table header
    table_str = "\nAttack Results Summary:\n"
    table_str += "-" * 200 + "\n"
    table_str += f"{'Negative Case':<30}{'Initial MSE':<25}{'Best MSE':<25}{'Success Rate':>15}{'Avg Queries':>15}{'LPIPS':>15}{'PSNR':>15}{'SSIM':>15}{'FID':>15}\n"
    table_str += "-" * 200 + "\n"
    
    # Add rows
    for case_name, results in all_results.items():
        metrics = results['avg_metrics']
        initial_mse = f"{metrics['initial_mse_mean']:.6f} ± {metrics['initial_mse_std']:.6f}"
        best_mse = f"{metrics['best_mse_mean']:.6f} ± {metrics['best_mse_std']:.6f}"
        
        row = f"{case_name:<30}"
        row += f"{initial_mse:<25}"
        row += f"{best_mse:<25}"
        row += f"{results['success_rate']*100:>15.2f}%"
        row += f"{results['avg_queries']:>15.1f}"
        row += f"{metrics['lpips']:>15.4f}"
        row += f"{metrics['psnr']:>15.2f}"
        row += f"{metrics['ssim']:>15.4f}"
        row += f"{metrics['fid']:>15.2f}\n"
        table_str += row
    
    table_str += "-" * 200 + "\n"
    table_str += f"Detection Threshold: {0.002883:.6f}\n"  # Add threshold at the bottom
    return table_str


def main():
    """Main entry point for attack."""
    args = parse_args()
    
    # Setup distributed environment
    local_rank, rank, world_size, device = setup_distributed()
    
    # Load default configuration and update with args
    config = get_default_config()
    config.update_from_args(args, mode='pgd_attack_authprint')
    
    # Create output directory and setup logging
    if rank == 0:
        os.makedirs(config.output_dir, exist_ok=True)
        setup_logging(config.output_dir, rank)
        
        # Log configuration
        logging.info(f"Configuration:\n{config}")
        logging.info(f"Distributed setup: local_rank={local_rank}, rank={rank}, world_size={world_size}, device={device}")
    
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
        
        # Load reconstructor and create wrapper
        if rank == 0:
            logging.info("Loading reconstructor...")
        
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
            logging.info(f"Selected pixel indices: {image_pixel_indices.tolist()}")
        
        # Create reconstructor wrapper with threshold from config
        reconstructor_wrapper = ReconstructorWrapper(
            reconstructor=reconstructor,
            threshold=config.pgd_attack_authprint.detection_threshold,
            image_pixel_indices=image_pixel_indices
        )
        
        # Create attacker
        attacker = QueryBasedAttack(
            original_model=original_model,
            reconstructor=reconstructor_wrapper,
            device=device,
            rank=rank,
            batch_size=config.pgd_attack_authprint.batch_size,
            config=config.pgd_attack_authprint  # Pass the config to the attacker
        )
        
        # Run attack on all cases
        if rank == 0:
            logging.info("Starting attacks on all negative cases...")
        
        all_results = attacker.attack_all_cases(
            pretrained_models=pretrained_models,
            quantized_models=quantized_models,
            num_samples=config.pgd_attack_authprint.num_samples
        )
        
        # Print results table
        if rank == 0:
            table = format_results_table(all_results)
            logging.info(table)
    
    except Exception as e:
        if rank == 0:
            logging.error(f"Error in attack: {str(e)}")
            logging.error("Attack error:", exc_info=True)
        raise
    finally:
        # Clean up distributed environment
        cleanup_distributed()


if __name__ == "__main__":
    main() 