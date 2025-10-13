"""
Data loading utilities for diffusion registration.

This module provides functions to load and preprocess medical imaging data
for registration tasks.
"""

import os
import torch
import numpy as np
import itk
from pathlib import Path
from typing import Tuple, List, Optional, Union
from ..training.config import Config


def load_tensor(path: Union[str, Path]) -> torch.Tensor:
    """
    Load tensor from file with proper error handling.
    
    Args:
        path: Path to tensor file.
        
    Returns:
        Loaded tensor.
        
    Raises:
        FileNotFoundError: If file doesn't exist.
        RuntimeError: If loading fails.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Tensor file not found: {path}")
    
    try:
        return torch.load(path, weights_only=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load tensor from {path}: {e}") from e


def load_preprocessed_data_2d(config: Config) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load preprocessed 2D data for training.
    
    Args:
        config: Configuration object containing data paths.
        
    Returns:
        Tuple of (train_dxa, train_xray, test_dxa, test_xray) tensors.
        
    Raises:
        FileNotFoundError: If data files are not found.
        ValueError: If data shapes are inconsistent.
    """
    root = Path(config.data.data_root)
    
    # Load data files
    dxa_with_seg = load_tensor(root / 'affine_dxa_images_w_seg.pt')
    xray_with_seg = load_tensor(root / 'affine_radio_images_w_seg.pt')
    dxa_wo_seg = load_tensor(root / 'affine_dxa_images_wo_seg.pt')
    xray_wo_seg = load_tensor(root / 'affine_radio_images_wo_seg.pt')

    # Filter out problematic X-rays
    mask = np.ones(xray_wo_seg.shape[0], dtype=bool)
    mask[config.data.weird_xrays] = False

    train_xray = xray_wo_seg[mask, :]
    train_dxa = torch.vstack([dxa_wo_seg, dxa_with_seg[:75]])

    test_xray = xray_with_seg
    test_dxa = dxa_with_seg[75:]

    # Normalize if requested
    if config.data.normalize_images:
        train_xray = normalize_tensor(train_xray)
        test_xray = normalize_tensor(test_xray)

    return train_dxa, train_xray, test_dxa, test_xray


def load_preprocessed_data_3d(config: Config, max_subjects: Optional[int] = None) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Load preprocessed 3D data for training.
    
    Args:
        config: Configuration object containing data paths.
        max_subjects: Maximum number of subjects to load. If None, uses config value.
        
    Returns:
        Tuple of (dataset_A, dataset_B) lists containing tensors.
        
    Raises:
        FileNotFoundError: If data files are not found.
    """
    if max_subjects is None:
        max_subjects = getattr(config.data, 'max_subjects', 300)
    
    dataset_A = []
    dataset_B = []
    count = 0
    
    for i in range(1, 458):
        if count >= max_subjects:
            break
            
        norm_path = f'/playpen-raid2/nurislam/diffreg/OASIS/OASIS_OAS1_{i:04}_MR1/aligned_norm.nii.gz'
        orig_path = f'/playpen-raid2/nurislam/diffreg/OASIS/OASIS_OAS1_{i:04}_MR1/aligned_orig.nii.gz'
        
        if not os.path.exists(norm_path) or not os.path.exists(orig_path):
            continue
            
        try:
            img_A = itk.imread(norm_path)
            img_B = itk.imread(orig_path)
            
            dataset_A.append(torch.from_numpy(np.asarray(img_A))[None])
            dataset_B.append(torch.from_numpy(np.asarray(img_B))[None])
            count += 1
            
        except Exception as e:
            print(f"Warning: Failed to load subject {i}: {e}")
            continue
    
    return dataset_A, dataset_B


def normalize_tensor(tensor: torch.Tensor, method: str = 'max') -> torch.Tensor:
    """
    Normalize tensor values.
    
    Args:
        tensor: Input tensor to normalize.
        method: Normalization method ('max', 'minmax', 'zscore').
        
    Returns:
        Normalized tensor.
    """
    if method == 'max':
        return tensor / tensor.amax(axis=list(range(1, tensor.ndim)), keepdim=True)
    elif method == 'minmax':
        min_val = tensor.amin(axis=list(range(1, tensor.ndim)), keepdim=True)
        max_val = tensor.amax(axis=list(range(1, tensor.ndim)), keepdim=True)
        return (tensor - min_val) / (max_val - min_val + 1e-8)
    elif method == 'zscore':
        mean_val = tensor.mean(axis=list(range(1, tensor.ndim)), keepdim=True)
        std_val = tensor.std(axis=list(range(1, tensor.ndim)), keepdim=True)
        return (tensor - mean_val) / (std_val + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


class RegistrationDataset:
    """Dataset class for registration tasks."""
    
    def __init__(self, images_A: List[torch.Tensor], images_B: List[torch.Tensor], 
                 device: str = 'cpu'):
        """
        Initialize dataset.
        
        Args:
            images_A: List of source images.
            images_B: List of target images.
            device: Device to move tensors to.
        """
        self.images_A = [img.to(device) for img in images_A] if isinstance(images_A, list) else images_A.to(device)
        self.images_B = [img.to(device) for img in images_B] if isinstance(images_B, list) else images_B.to(device)
        self.device = device
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.images_A)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item by index."""
        return self.images_A[idx], self.images_B[idx]
    
    def get_random_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get random batch of images.
        
        Args:
            batch_size: Size of batch to return.
            
        Returns:
            Tuple of (batch_A, batch_B).
        """
        import random
        
        if isinstance(self.images_A, list):
            indices = random.sample(range(len(self.images_A)), min(batch_size, len(self.images_A)))
            batch_A = torch.stack([self.images_A[i] for i in indices])
            batch_B = torch.stack([self.images_B[i] for i in indices])
        else:
            indices = random.sample(range(len(self.images_A)), min(batch_size, len(self.images_A)))
            batch_A = self.images_A[indices]
            batch_B = self.images_B[indices]
        
        return batch_A, batch_B


def create_data_loaders(config: Config) -> Tuple[RegistrationDataset, RegistrationDataset]:
    """
    Create train and test data loaders based on configuration.
    
    Args:
        config: Configuration object.
        
    Returns:
        Tuple of (train_dataset, test_dataset).
    """
    if config.model.dimension == 2:
        train_dxa, train_xray, test_dxa, test_xray = load_preprocessed_data_2d(config)
        
        train_dataset = RegistrationDataset(train_dxa, train_xray, config.device)
        test_dataset = RegistrationDataset(test_dxa, test_xray, config.device)
        
    elif config.model.dimension == 3:
        dataset_A, dataset_B = load_preprocessed_data_3d(config)
        
        # Split into train/test (80/20 split)
        split_idx = int(0.8 * len(dataset_A))
        train_A, test_A = dataset_A[:split_idx], dataset_A[split_idx:]
        train_B, test_B = dataset_B[:split_idx], dataset_B[split_idx:]
        
        train_dataset = RegistrationDataset(train_A, train_B, config.device)
        test_dataset = RegistrationDataset(test_A, test_B, config.device)
        
    else:
        raise ValueError(f"Unsupported dimension: {config.model.dimension}")
    
    return train_dataset, test_dataset

