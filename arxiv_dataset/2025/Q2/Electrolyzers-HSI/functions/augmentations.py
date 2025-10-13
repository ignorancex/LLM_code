"""
Hyperspectral Image (HSI) Augmentation Utilities

This module provides a suite of data augmentation functions tailored for hyperspectral imagery (HSI). These augmentations
help simulate realistic variations such as sensor noise, spectral distortions, geometric transformations, and intensity
scaling, enhancing the robustness and generalization of machine learning models trained on HSI data.

Included Spectral Augmentations:
- spectral_shift: Shifts spectral bands to simulate sensor misalignment.
- spectral_smoothing: Applies Gaussian smoothing along spectral bands.
- spectral_noise: Adds Gaussian noise to mimic sensor imperfections.
- spectral_scaling: Randomly scales spectral intensity to simulate lighting variations.
- spectral_channel_dropping: Randomly drops bands to simulate sensor failure or occlusion.

Spectral Signature Augmentations (1D or 2D signatures):
- spectral_shift_sig
- spectral_smoothing_sig
- spectral_noise_sig
- spectral_scaling_sig
- spectral_channel_dropping_sig

Spatial and Geometric Augmentations:
- random_rotation: Rotates the spatial dimensions of the image.
- random_translation: Shifts the image spatially in X and Y directions.
- random_flipping: Randomly flips the image horizontally and/or vertically.
- random_zoom: Zooms in or out on the image with optional padding or cropping.
- random_brightness: Adjusts image brightness.
- random_elastic_deformation: Applies non-linear elastic deformation.

Combinatorial Augmentation:
- augment_hsi: Applies a randomized combination of augmentations from a provided list.

All functions accept NumPy arrays or PyTorch tensors and return NumPy arrays for compatibility with downstream HSI pipelines.

Example:
    augmentations = [spectral_shift, spectral_noise, random_flipping]
    augmented_image = augment_hsi(hsi_image, augmentations, prob=0.7)

Author: HiF-Explo: Elias Arbash
"""
import numpy as np
import scipy.ndimage
import torch
from scipy.ndimage import gaussian_filter
import random

def spectral_shift(image, shift=1):
    """
    Description: Shift the spectral bands left or right by a few bands.
    Purpose: Mimics the effect of spectral variations and misalignments in the sensor.
    """
    image = image.numpy() if torch.is_tensor(image) else image
    if image.ndim == 3:
        return np.roll(image, shift, axis=2)
    return image
    


def spectral_smoothing(image, sigma=1):
    """
    Description: Apply a smoothing filter along the spectral dimension.
    Purpose: Reduces noise and simulates the effect of different sensor resolutions.
    """
    image = image.numpy() if torch.is_tensor(image) else image
    if image.ndim == 3:
        return gaussian_filter(image, sigma=[0, 0, sigma])
    return image


def spectral_noise(image, noise_level=0.4):
    """
    Description: Add Gaussian noise to the spectral bands.
    Purpose: Simulates sensor noise and improves the model's robustness to noisy data.
    """
    image = image.numpy() if torch.is_tensor(image) else image
    noise = np.random.normal(loc=0, scale=noise_level, size=image.shape)
    return image + noise

def spectral_scaling(image, scale_factor=np.random.uniform(0.2, 1.2)):
    """
    Description: Scale the intensity of each spectral band by a random factor.
    Purpose: Simulates variations in illumination and sensor response.
    """
    image = image.numpy() if torch.is_tensor(image) else image
    return image * scale_factor


def spectral_channel_dropping(image, drop_prob=0.99):
    """
    Description: Randomly drop some spectral channels.
    Purpose: Mimics sensor failure or missing data.
    """
    image = image.numpy() if torch.is_tensor(image) else image
    if image.ndim == 3:
        drop_mask = np.random.rand(image.shape[2]) > drop_prob
        return image * drop_mask
    return image


def spectral_shift_sig(sig, shift=1):
    return np.roll(sig, shift, axis=-1)

def spectral_smoothing_sig(sig, sigma=1):
    """
    Apply a smoothing filter along the spectral dimension.
    This function assumes 'sig' is either a 1D array (for a single signature) or 2D (for multiple signatures).
    """
    # Ensure 'sig' is a NumPy array
    sig = sig.numpy() if torch.is_tensor(sig) else sig
    
    # If sig is 1D, we only need to apply smoothing along the first dimension
    if sig.ndim == 1:
        return gaussian_filter(sig, sigma=sigma)
    elif sig.ndim == 2:
        return gaussian_filter(sig, sigma=[0, sigma])  # For 2D inputs, apply smoothing along the second dimension
    else:
        raise ValueError("Input must be either a 1D or 2D array.")

def spectral_noise_sig(sig, noise_level=0.08):
    noise = np.random.normal(loc=0, scale=noise_level, size=sig.shape)
    return sig + noise

def spectral_scaling_sig(sig, scale_factor=None):
    if scale_factor is None:
        scale_factor = np.random.uniform(0.8, 1.2)
    return sig * scale_factor

def spectral_channel_dropping_sig(sig, drop_prob=0.99):
    drop_mask = np.random.rand(sig.shape[-1]) > drop_prob
    return sig * drop_mask





def random_rotation(image, angle=np.random.uniform(-30, 30)):
    image = image.numpy() if torch.is_tensor(image) else image
    return scipy.ndimage.rotate(image, angle, axes=(0, 1), reshape=False, mode='reflect')


def random_translation(image, shift=np.random.uniform(-5, 5, size=2)):
    image = image.numpy() if torch.is_tensor(image) else image
    if image.ndim == 3:
        return scipy.ndimage.shift(image, shift=(shift[0], shift[1], 0), mode='reflect')
    elif image.ndim == 2:
        return scipy.ndimage.shift(image, shift=(shift[0], shift[1]), mode='reflect')
    return image

def random_flipping(image):
    image = image.numpy() if torch.is_tensor(image) else image
    if np.random.rand() > 0.05:
        image = np.flipud(image.copy())
    if np.random.rand() > 0.05:
        image = np.fliplr(image.copy())
    return image

def random_zoom(image, zoom_factor=None):
    if zoom_factor is None:
        zoom_factor = np.random.uniform(0.8, 1.2)
    
    image = image.numpy() if torch.is_tensor(image) else image
    
    if image.ndim == 3:
        h, w, c = image.shape
        zoomed_image = scipy.ndimage.zoom(image, (zoom_factor, zoom_factor, 1), order=1)
    elif image.ndim == 2:
        h, w = image.shape
        zoomed_image = scipy.ndimage.zoom(image, (zoom_factor, zoom_factor), order=1)
    else:
        return image

    # Handle zooming out (zoom_factor > 1.0)
    if zoom_factor > 1.0:
        # Crop the center of the zoomed image
        start_h = (zoomed_image.shape[0] - h) // 2
        start_w = (zoomed_image.shape[1] - w) // 2
        if image.ndim == 3:
            zoomed_image = zoomed_image[start_h:start_h + h, start_w:start_w + w, :]
        else:
            zoomed_image = zoomed_image[start_h:start_h + h, start_w:start_w + w]
    else:
        # Handle zooming in (zoom_factor < 1.0)
        pad_h = max((h - zoomed_image.shape[0]) // 2, 0)
        pad_w = max((w - zoomed_image.shape[1]) // 2, 0)
        
        if image.ndim == 3:
            # Add padding to the zoomed image
            zoomed_image = np.pad(zoomed_image,
                                  ((pad_h, h - zoomed_image.shape[0] - pad_h),
                                   (pad_w, w - zoomed_image.shape[1] - pad_w),
                                   (0, 0)),
                                  mode='reflect')
        else:
            zoomed_image = np.pad(zoomed_image,
                                  ((pad_h, h - zoomed_image.shape[0] - pad_h),
                                   (pad_w, w - zoomed_image.shape[1] - pad_w)),
                                  mode='reflect')

    return zoomed_image

def random_brightness(image, brightness_factor=np.random.uniform(0.8, 1.2)):
    image = image.numpy() if torch.is_tensor(image) else image
    return np.clip(image * brightness_factor, 0, 1)

def random_elastic_deformation(image, alpha=1, sigma=10):
    random_state = np.random.RandomState(None)
    shape = image.shape[:2]
    dx = scipy.ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = scipy.ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
    return scipy.ndimage.map_coordinates(image, indices, order=1, mode='reflect').reshape(image.shape)

def augment_hsi(image, augmentations, prob=0.5):
    augmented_image = image.copy()
    for aug in augmentations:
        if random.random() < prob:
            augmented_image = aug(augmented_image)
    return augmented_image

# Example usage
# augmentations = [spectral_shift, spectral_smoothing, spectral_noise, spectral_scaling, spectral_channel_dropping]
# augmented_image = augment_hsi(hsi_image, augmentations)