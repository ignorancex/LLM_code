"""
Image transformation utilities for evaluation.
"""
import io
import copy
import logging
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def apply_truncation(model, z, truncation_psi=2.0, return_w=False):
    """
    Apply truncation trick to latent vectors and generate images.
    
    Args:
        model: StyleGAN2 model
        z (torch.Tensor): Batch of latent vectors
        truncation_psi (float): Truncation psi value
        return_w (bool): Whether to also return the W latent vectors
        
    Returns:
        torch.Tensor: Generated images
        or
        (torch.Tensor, torch.Tensor): Generated images and W latent vectors if return_w is True
    """
    try:
        # Generate with truncation
        with torch.no_grad():
            # Apply truncation
            w = model.mapping(z, None, truncation_psi=truncation_psi)
            x = model.synthesis(w, noise_mode="const")
        
        if return_w:
            return x, w
        return x
    except Exception as e:
        logging.error(f"Error in apply_truncation: {str(e)}")
        # Return zeros for both image and w if requested
        if return_w:
            return torch.zeros_like(z.reshape(z.size(0), 1, 1, 1).expand(-1, 3, 256, 256)), z.unsqueeze(1)
        return torch.zeros_like(z.reshape(z.size(0), 1, 1, 1).expand(-1, 3, 256, 256))


def prune_model_weights(model, sparsity=0.5, method='magnitude'):
    """Prune model weights to achieve target sparsity.
    
    Args:
        model: The model to prune (StyleGAN2 or StableDiffusion)
        sparsity (float): Target sparsity ratio (0.0 to 1.0)
        method (str): Pruning method ('magnitude', 'random')
        
    Returns:
        Pruned model copy
    """
    try:
        pruned_model = copy.deepcopy(model)
        
        def prune_tensor(tensor, sparsity_ratio, prune_method):
            # Handle empty or NaN tensors
            if tensor.numel() == 0 or torch.isnan(tensor).any():
                return tensor.clone()
            
            # Flatten tensor for easier processing
            flat_tensor = tensor.view(-1)
            num_params = flat_tensor.numel()
            num_to_prune = int(num_params * sparsity_ratio)
            
            if prune_method == 'magnitude':
                # Get indices of smallest magnitude values
                _, indices = torch.sort(torch.abs(flat_tensor))
                mask = torch.ones_like(flat_tensor, dtype=torch.bool)
                mask[indices[:num_to_prune]] = False
            else:  # random
                # Generate random mask
                mask = torch.rand_like(flat_tensor) > sparsity_ratio
            
            # Apply mask
            pruned = flat_tensor * mask.float()
            return pruned.view_as(tensor)
        
        # Access the actual model instance
        target_model = pruned_model.model if hasattr(pruned_model, 'model') else pruned_model
        
        # Handle Stable Diffusion case
        if hasattr(target_model, 'pipe'):
            # Prune UNet parameters
            for name, param in target_model.pipe.unet.named_parameters():
                if 'weight' in name:  # Only prune weight tensors
                    with torch.no_grad():
                        pruned_tensor = prune_tensor(param.data, sparsity, method)
                        param.data.copy_(pruned_tensor)
            
            # Prune VAE parameters
            for name, param in target_model.pipe.vae.named_parameters():
                if 'weight' in name:  # Only prune weight tensors
                    with torch.no_grad():
                        pruned_tensor = prune_tensor(param.data, sparsity, method)
                        param.data.copy_(pruned_tensor)
        else:  # StyleGAN2
            # Access the actual model parameters
            if hasattr(target_model, 'module'):  # Handle DataParallel case
                target_model = target_model.module
                
            # Prune synthesis network parameters
            for name, param in target_model.synthesis.named_parameters():
                if 'weight' in name:
                    with torch.no_grad():
                        pruned_tensor = prune_tensor(param.data, sparsity, method)
                        param.data.copy_(pruned_tensor)
            
            # Prune mapping network parameters
            for name, param in target_model.mapping.named_parameters():
                if 'weight' in name:
                    with torch.no_grad():
                        pruned_tensor = prune_tensor(param.data, sparsity, method)
                        param.data.copy_(pruned_tensor)
        
        return pruned_model
    except Exception as e:
        logging.error(f"Error pruning model weights with sparsity {sparsity}: {str(e)}")
        logging.error("Pruning error:", exc_info=True)
        # Return original model as fallback
        return model


def quantize_model_weights(model, precision='int8'):
    """Quantize model weights to specified precision.
    
    Args:
        model: The model to quantize
        precision: Quantization precision ('int8', 'int4', or 'int2')
    
    Returns:
        Quantized model copy
    """
    try:
        quantized_model = copy.deepcopy(model)
        
        def quantize_tensor(tensor, bit_precision):
            # Handle empty or NaN tensors
            if tensor.numel() == 0 or torch.isnan(tensor).any():
                return tensor.clone()
                
            # Get max abs value, avoiding division by zero
            max_abs_val = torch.max(torch.abs(tensor))
            if max_abs_val == 0:
                return tensor.clone()
            
            # Set quantization parameters based on precision
            if bit_precision == 'int8':
                max_val = 127  # 2^7 - 1
            elif bit_precision == 'int7':
                max_val = 63   # 2^6 - 1
            elif bit_precision == 'int6':
                max_val = 31   # 2^5 - 1
            elif bit_precision == 'int5':
                max_val = 15   # 2^4 - 1
            elif bit_precision == 'int4':
                max_val = 7    # 2^3 - 1
            elif bit_precision == 'int2':
                max_val = 1    # 2^1 - 1
            else:
                # Default to int8
                max_val = 127
                
            # Scale to appropriate range
            scale = float(max_val) / max_abs_val
            quantized = torch.round(tensor * scale)
            quantized = torch.clamp(quantized, -max_val, max_val)
            
            # Scale back to original range
            dequantized = quantized / scale
            return dequantized
        
        # Quantize all parameters
        with torch.no_grad():
            for param in quantized_model.parameters():
                param.copy_(quantize_tensor(param, precision))
        
        return quantized_model
    except Exception as e:
        logging.error(f"Error quantizing model weights with precision {precision}: {str(e)}")
        # Return original model as fallback
        return model


def downsample_and_upsample(images, downsample_size=128):
    """Downsample images and then upsample back to original size.
    
    Args:
        images: Input images tensor (B, C, H, W)
        downsample_size: Size to downsample to
    
    Returns:
        Downsampled then upsampled images
    """
    try:
        # Verify input dimensions
        if images.dim() != 4:
            logging.warning(f"Unexpected image dimension: {images.dim()}, expected 4. Returning original.")
            return images
            
        original_size = images.shape[-1]
        # Downsample
        downsampled = F.interpolate(images, size=downsample_size, mode='bilinear', align_corners=False)
        # Upsample back to original size
        upsampled = F.interpolate(downsampled, size=original_size, mode='bilinear', align_corners=False)
        return upsampled
    except Exception as e:
        logging.error(f"Error in downsample_and_upsample: {str(e)}")
        # Return original images as fallback
        return images


def apply_jpeg_compression(images, quality=55):
    """Apply JPEG compression to images.
    
    Args:
        images: Input images tensor (B, C, H, W) in range [-1, 1]
        quality: JPEG quality (0-100)
    
    Returns:
        Compressed images tensor
    """
    try:
        device = images.device
        compressed_batch = []
        
        # Convert to PIL, compress, and back to tensor
        for img in images:
            try:
                # Convert to PIL Image (rescale to 0-255 range)
                img_np = ((img.permute(1, 2, 0) + 1) * 127.5).clamp(0, 255).to(torch.uint8).cpu().numpy()
                img_pil = Image.fromarray(img_np)
                
                # Apply JPEG compression
                buffer = io.BytesIO()
                img_pil.save(buffer, format='JPEG', quality=quality)
                buffer.seek(0)
                compressed_img = Image.open(buffer)
                
                # Convert back to tensor and rescale to [-1, 1]
                compressed_tensor = transforms.ToTensor()(compressed_img) * 2 - 1
                compressed_batch.append(compressed_tensor)
            except Exception as e:
                logging.error(f"Error compressing single image: {str(e)}")
                # Append original image instead
                compressed_batch.append(img.cpu())
        
        # Stack back into batch
        return torch.stack(compressed_batch).to(device)
    except Exception as e:
        logging.error(f"Error in apply_jpeg_compression: {str(e)}")
        # Return original images as fallback
        return images 