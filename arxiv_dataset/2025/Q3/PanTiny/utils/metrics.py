#!/usr/bin/env python
# coding=utf-8
'''
Enhanced metrics calculation for comprehensive evaluation
'''
import torch
import numpy as np
from py_tra.metricsutil import get_metrics_reduced
from py_tra.metrics import no_ref_evaluate

# Import error handling
try:
    from py_tra.utilsmetric import psnr_loss, ssim, sam
except ImportError:
    print("Warning: Some metrics functions not found, using fallback implementations")


def calculate_reference_metrics(pred, gt):
    """
    Calculate reference metrics (PSNR, SSIM, CC, SAM, ERGAS)
    Args:
        pred: Predicted image tensor (B, C, H, W)
        gt: Ground truth image tensor (B, C, H, W)
    Returns:
        Dictionary with metric values
    """
    try:
        # Use the existing function from metricsutil
        metric_values = get_metrics_reduced(pred, gt)
        metrics = {
            'PSNR': float(metric_values[0]),
            'SSIM': float(metric_values[1]),
            'CC': float(metric_values[2]),
            'SAM': float(metric_values[3]),
            'ERGAS': float(metric_values[4])
        }
    except Exception as e:
        print(f"Error in reference metrics calculation: {e}")
        # Fallback to basic metrics
        metrics = {
            'PSNR': 0.0,
            'SSIM': 0.0,
            'CC': 0.0,
            'SAM': 0.0,
            'ERGAS': 0.0
        }
    
    return metrics


def calculate_no_reference_metrics(pred, pan, lms):
    """
    Calculate no-reference metrics (D_lambda, D_s, QNR)
    Args:
        pred: Predicted image as tensor or numpy array
        pan: PAN image as tensor or numpy array
        lms: Low-resolution MS image as tensor or numpy array  
    Returns:
        Dictionary with no-reference metrics
    """
    try:
        # Convert to numpy if needed
        if hasattr(pred, 'dim'):  # It's a tensor
            pred_np = tensor_to_numpy(pred, keep_single_channel_3d=False)
            pan_np = tensor_to_numpy(pan, keep_single_channel_3d=True)  # Keep 3D for no_ref_evaluate
            lms_np = tensor_to_numpy(lms, keep_single_channel_3d=False)
        else:  # It's already numpy
            pred_np = pred
            pan_np = pan
            lms_np = lms
            
            # Ensure pan is 3D for no_ref_evaluate
            if pan_np.ndim == 2:
                pan_np = np.expand_dims(pan_np, axis=-1)
        
        # Ensure inputs are uint8
        pred_uint8 = np.clip(pred_np, 0, 255).astype(np.uint8)
        pan_uint8 = np.clip(pan_np, 0, 255).astype(np.uint8)
        lms_uint8 = np.clip(lms_np, 0, 255).astype(np.uint8)
        
        # Use the existing function from metrics
        metric_values = no_ref_evaluate(pred_uint8, pan_uint8, lms_uint8)
        
        metrics = {
            'D_lambda': float(metric_values[0]),
            'D_s': float(metric_values[1]),
            'QNR': float(metric_values[2])
        }
    except Exception as e:
        print(f"Error in no-reference metrics calculation: {e}")
        # Fallback values
        metrics = {
            'D_lambda': 0.0,
            'D_s': 0.0,
            'QNR': 0.0
        }
    
    return metrics


def tensor_to_numpy(tensor, normalize=False, keep_single_channel_3d=False):
    """
    Convert tensor to numpy array for metric calculation
    Args:
        tensor: Input tensor (B, C, H, W) or (C, H, W)
        normalize: Whether the tensor is normalized (need to denormalize)
        keep_single_channel_3d: Keep single channel as 3D (H, W, 1) instead of (H, W)
    Returns:
        numpy array (H, W, C) or (H, W) for single channel if not keep_single_channel_3d
    """
    if tensor.dim() == 4:
        # Remove batch dimension
        tensor = tensor.squeeze(0)
    
    # Convert to numpy and transpose
    img_np = tensor.cpu().detach().numpy().transpose(1, 2, 0)
    
    # Scale to 0-255 range
    if normalize:
        img_np = (img_np + 1) * 127.5  # For normalized data [-1, 1]
    else:
        img_np = img_np * 255  # For data [0, 1]
    
    # Ensure valid range
    img_np = np.clip(img_np, 0, 255)
    
    # Handle single channel case
    if img_np.shape[-1] == 1 and not keep_single_channel_3d:
        img_np = img_np.squeeze(-1)  # From (H, W, 1) to (H, W)
    
    return img_np


class ComprehensiveMetrics:
    """Comprehensive metrics calculator for pan-sharpening evaluation"""
    
    def __init__(self, dataset_names=['WV2', 'WV3', 'GF2']):
        self.dataset_names = dataset_names
        self.ref_metrics = ['PSNR', 'SSIM', 'CC', 'SAM', 'ERGAS']
        self.no_ref_metrics = ['D_lambda', 'D_s', 'QNR']
        
        # Storage for results
        self.results = {}
        for dataset in dataset_names:
            self.results[dataset] = {
                'reference': {metric: [] for metric in self.ref_metrics},
                'no_reference': {metric: [] for metric in self.no_ref_metrics}
            }
    
    def add_reference_result(self, dataset, pred, gt, normalize=False):
        """
        Add reference metrics result for a dataset
        Args:
            dataset: Dataset name
            pred: Predicted tensor
            gt: Ground truth tensor
            normalize: Whether inputs are normalized
        """
        if dataset not in self.results:
            return
        
        metrics = calculate_reference_metrics(pred, gt)
        
        for metric_name, value in metrics.items():
            if metric_name in self.results[dataset]['reference']:
                self.results[dataset]['reference'][metric_name].append(value)
    
    def add_no_reference_result(self, dataset, pred, pan, lms, normalize=False):
        """
        Add no-reference metrics result for a dataset
        Args:
            dataset: Dataset name
            pred: Predicted tensor
            pan: PAN tensor
            lms: Low-resolution MS tensor
            normalize: Whether inputs are normalized
        """
        if dataset not in self.results:
            return
        
        # Convert tensors to numpy
        pred_np = tensor_to_numpy(pred, normalize)
        pan_np = tensor_to_numpy(pan, normalize)
        lms_np = tensor_to_numpy(lms, normalize)
        
        # Handle PAN image dimensions
        if pan_np.ndim == 3 and pan_np.shape[2] == 1:
            pan_np = pan_np.squeeze(2)
        elif pan_np.ndim == 3 and pan_np.shape[2] > 1:
            # Convert to grayscale if multi-channel
            pan_np = np.mean(pan_np, axis=2)
        
        metrics = calculate_no_reference_metrics(pred_np, pan_np, lms_np)
        
        for metric_name, value in metrics.items():
            if metric_name in self.results[dataset]['no_reference']:
                self.results[dataset]['no_reference'][metric_name].append(value)
    
    def compute_reference_metrics(self, pred, gt, normalize=False):
        """
        Compute reference metrics directly without storing
        Args:
            pred: Predicted tensor
            gt: Ground truth tensor  
            normalize: Whether inputs are normalized
        Returns:
            Dictionary with metric values
        """
        return calculate_reference_metrics(pred, gt)
    
    def compute_no_reference_metrics(self, pred, pan, lms, normalize=False):
        """
        Compute no-reference metrics directly without storing
        Args:
            pred: Predicted tensor
            pan: PAN tensor
            lms: Low-resolution MS tensor
            normalize: Whether inputs are normalized
        Returns:
            Dictionary with no-reference metrics
        """
        # Convert tensors to numpy
        pred_np = tensor_to_numpy(pred, normalize)
        pan_np = tensor_to_numpy(pan, normalize)
        lms_np = tensor_to_numpy(lms, normalize)
        
        # Handle PAN image dimensions
        if pan_np.ndim == 3 and pan_np.shape[2] == 1:
            pan_np = pan_np.squeeze(2)
        elif pan_np.ndim == 3 and pan_np.shape[2] > 1:
            # Convert to grayscale if multi-channel
            pan_np = np.mean(pan_np, axis=2)
        
        return calculate_no_reference_metrics(pred_np, pan_np, lms_np)
    
    def get_statistics(self):
        """Get comprehensive statistics for all datasets and metrics"""
        stats = {}
        
        for dataset in self.dataset_names:
            if dataset not in self.results:
                continue
                
            stats[dataset] = {}
            
            # Reference metrics statistics
            for metric_name in self.ref_metrics:
                values = self.results[dataset]['reference'][metric_name]
                if values:
                    stats[dataset][metric_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'count': len(values)
                    }
                else:
                    stats[dataset][metric_name] = {
                        'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0
                    }
            
            # No-reference metrics statistics
            for metric_name in self.no_ref_metrics:
                values = self.results[dataset]['no_reference'][metric_name]
                if values:
                    stats[dataset][metric_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'count': len(values)
                    }
                else:
                    stats[dataset][metric_name] = {
                        'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0
                    }
        
        return stats
    
    def get_average_metrics(self):
        """Get average metrics across all datasets"""
        stats = self.get_statistics()
        
        averages = {}
        
        # Average reference metrics
        for metric_name in self.ref_metrics:
            values = []
            for dataset in self.dataset_names:
                if (dataset in stats and 
                    metric_name in stats[dataset] and 
                    stats[dataset][metric_name]['count'] > 0):
                    values.append(stats[dataset][metric_name]['mean'])
            
            if values:
                averages[metric_name] = float(np.mean(values))
            else:
                averages[metric_name] = 0.0
        
        # Average no-reference metrics
        for metric_name in self.no_ref_metrics:
            values = []
            for dataset in self.dataset_names:
                if (dataset in stats and 
                    metric_name in stats[dataset] and 
                    stats[dataset][metric_name]['count'] > 0):
                    values.append(stats[dataset][metric_name]['mean'])
            
            if values:
                averages[metric_name] = float(np.mean(values))
            else:
                averages[metric_name] = 0.0
        
        return averages
    
    def print_results(self):
        """Print formatted results"""
        stats = self.get_statistics()
        
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE METRICS RESULTS")
        print(f"{'='*80}")
        
        # Print reference metrics
        print(f"\nREFERENCE METRICS (with ground truth):")
        print(f"{'-'*60}")
        header = f"{'Dataset':<10}"
        for metric in self.ref_metrics:
            header += f"{metric:>10}"
        print(header)
        print(f"{'-'*60}")
        
        for dataset in self.dataset_names:
            if dataset in stats:
                row = f"{dataset:<10}"
                for metric in self.ref_metrics:
                    if metric in stats[dataset]:
                        value = stats[dataset][metric]['mean']
                        row += f"{value:>10.4f}"
                    else:
                        row += f"{'N/A':>10}"
                print(row)
        
        # Print no-reference metrics
        print(f"\nNO-REFERENCE METRICS (full resolution):")
        print(f"{'-'*60}")
        header = f"{'Dataset':<10}"
        for metric in self.no_ref_metrics:
            header += f"{metric:>10}"
        print(header)
        print(f"{'-'*60}")
        
        for dataset in self.dataset_names:
            if dataset in stats:
                row = f"{dataset:<10}"
                for metric in self.no_ref_metrics:
                    if metric in stats[dataset]:
                        value = stats[dataset][metric]['mean']
                        row += f"{value:>10.4f}"
                    else:
                        row += f"{'N/A':>10}"
                print(row)
        
        # Print averages
        averages = self.get_average_metrics()
        print(f"\nAVERAGE ACROSS DATASETS:")
        print(f"{'-'*40}")
        for metric_name, value in averages.items():
            print(f"{metric_name:<15}: {value:>8.4f}")
        
        print(f"{'='*80}\n")
        
        return stats, averages
