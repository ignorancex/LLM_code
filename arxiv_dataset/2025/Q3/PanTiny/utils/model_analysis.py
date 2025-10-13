#!/usr/bin/env python
# coding=utf-8
'''
Model analysis utilities for FLOPs and parameter counting
'''
import torch
import torch.nn as nn
import numpy as np
from typing import Union, List


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model: nn.Module, input_size: tuple) -> int:
    """
    Count FLOPs for a model given input size
    Args:
        model: PyTorch model
        input_size: Tuple of input sizes, e.g., (channels, height, width)
    """
    model.eval()
    
    # Hook to count FLOPs
    flops_count = [0]
    
    def flop_count_hook(module, input, output):
        if isinstance(module, nn.Conv2d):
            # For Conv2d: FLOPs = output_elements * (kernel_size^2 * input_channels + bias)
            output_elements = output.numel()
            kernel_flops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
            if module.bias is not None:
                kernel_flops += 1
            flops_count[0] += output_elements * kernel_flops
            
        elif isinstance(module, nn.Linear):
            # For Linear: FLOPs = output_elements * (input_features + bias)
            input_features = module.in_features
            output_elements = output.numel()
            if module.bias is not None:
                input_features += 1
            flops_count[0] += output_elements * input_features
            
        elif isinstance(module, nn.BatchNorm2d):
            # For BatchNorm: FLOPs = 2 * output_elements (mean and var)
            flops_count[0] += 2 * output.numel()
            
        elif isinstance(module, nn.ReLU):
            # For ReLU: FLOPs = output_elements
            flops_count[0] += output.numel()
            
        elif isinstance(module, nn.MultiheadAttention):
            # For MultiheadAttention: approximate FLOPs
            if hasattr(output, 'shape') and len(output.shape) >= 3:
                seq_len, batch_size, embed_dim = output.shape
                num_heads = module.num_heads
                # Approximate: Q*K^T + softmax + attention*V
                flops_count[0] += seq_len * seq_len * embed_dim * batch_size
    
    # Register hooks
    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ReLU, nn.MultiheadAttention)):
            hooks.append(module.register_forward_hook(flop_count_hook))
    
    # Forward pass with dummy inputs for pan-sharpening models
    with torch.no_grad():
        try:
            if len(input_size) == 3:  # (C, H, W)
                c, h, w = input_size
                # Create dummy inputs for pan-sharpening
                l_ms = torch.randn(1, c, h//4, w//4)  # Low-res MS
                b_ms = torch.randn(1, c, h, w)        # Bicubic upsampled MS
                x_pan = torch.randn(1, 1, h, w)       # PAN image
                
                # Try different input patterns
                try:
                    # Try pan-sharpening model format
                    model(l_ms, b_ms, x_pan)
                except:
                    try:
                        # Try single input format
                        dummy_input = torch.randn(1, *input_size)
                        model(dummy_input)
                    except:
                        # Try concatenated input format
                        combined_input = torch.randn(1, c+1+2, h, w)  # MS+PAN+indices
                        model(combined_input, combined_input, combined_input)
            else:
                raise ValueError("Unsupported input size format")
        except Exception as e:
            print(f"Warning: Could not compute FLOPs due to: {e}")
            flops_count[0] = 0
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return flops_count[0]


def analyze_model(model: nn.Module, input_size: tuple = (4, 32, 32)) -> dict:
    """
    Comprehensive model analysis
    Args:
        model: PyTorch model
        input_size: Input size (C, H, W)
    Returns:
        Dictionary with model statistics
    """
    model.eval()
    
    # Count parameters
    total_params = count_parameters(model)
    
    # Count FLOPs
    try:
        total_flops = count_flops(model, input_size)
    except:
        total_flops = 0
        print("Warning: Could not compute FLOPs")
    
    # Model size in MB
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / 1024 / 1024
    
    return {
        'total_parameters': total_params,
        'total_flops': total_flops,
        'model_size_mb': size_mb,
        'flops_per_param': total_flops / total_params if total_params > 0 else 0
    }


def print_model_analysis(model: nn.Module, input_size: tuple = (4, 32, 32), model_name: str = "Model"):
    """Print formatted model analysis"""
    stats = analyze_model(model, input_size)
    
    print(f"\n{'='*60}")
    print(f"MODEL ANALYSIS: {model_name}")
    print(f"{'='*60}")
    print(f"Total Parameters:     {stats['total_parameters']:,}")
    print(f"Parameters (K):       {stats['total_parameters']/1000:.2f}K")
    print(f"Parameters (M):       {stats['total_parameters']/1000000:.3f}M")
    print(f"Total FLOPs:          {stats['total_flops']:,}")
    print(f"FLOPs (M):            {stats['total_flops']/1000000:.2f}M")
    print(f"FLOPs (G):            {stats['total_flops']/1000000000:.3f}G")
    print(f"Model Size (MB):      {stats['model_size_mb']:.2f}")
    print(f"FLOPs per Parameter:  {stats['flops_per_param']:.2f}")
    print(f"{'='*60}\n")
    
    return stats


def compare_models(models: dict, input_size: tuple = (4, 32, 32)):
    """
    Compare multiple models
    Args:
        models: Dictionary of {name: model}
        input_size: Input size for FLOP calculation
    """
    results = {}
    
    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'Params':<12} {'Params(K)':<10} {'FLOPs(M)':<10} {'Size(MB)':<10}")
    print(f"{'-'*80}")
    
    for name, model in models.items():
        stats = analyze_model(model, input_size)
        results[name] = stats
        
        print(f"{name:<20} {stats['total_parameters']:<12,} "
              f"{stats['total_parameters']/1000:<10.2f} "
              f"{stats['total_flops']/1000000:<10.2f} "
              f"{stats['model_size_mb']:<10.2f}")
    
    print(f"{'='*80}\n")
    return results


# Lightweight profiler for inference time
class InferenceProfiler:
    """Simple profiler for measuring inference time"""
    
    def __init__(self, model: nn.Module, warmup_runs: int = 10):
        self.model = model
        self.warmup_runs = warmup_runs
        self.model.eval()
    
    def profile(self, input_tensors: Union[torch.Tensor, List[torch.Tensor]], 
                num_runs: int = 100) -> dict:
        """
        Profile inference time
        Args:
            input_tensors: Input tensor(s) for the model
            num_runs: Number of runs for timing
        """
        if isinstance(input_tensors, torch.Tensor):
            input_tensors = [input_tensors, input_tensors, input_tensors]  # For pan-sharpening
        
        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                _ = self.model(*input_tensors)
        
        # Timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        import time
        times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = self.model(*input_tensors)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()
                times.append(end_time - start_time)
        
        times = np.array(times)
        
        return {
            'mean_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times)),
            'fps': float(1.0 / np.mean(times))
        }
