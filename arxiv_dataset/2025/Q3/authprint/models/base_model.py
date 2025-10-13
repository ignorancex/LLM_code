from abc import ABC, abstractmethod
import torch
from typing import Optional, Union, Dict, Any

class BaseGenerativeModel(ABC):
    """Abstract base class for generative models."""
    
    @abstractmethod
    def generate_images(
        self,
        batch_size: int,
        device: torch.device,
        **kwargs
    ) -> torch.Tensor:
        """Generate a batch of images.
        
        Args:
            batch_size (int): Number of images to generate
            device (torch.device): Device to generate on
            **kwargs: Additional model-specific arguments
            
        Returns:
            torch.Tensor: Generated images [B, C, H, W]
        """
        pass
    
    @abstractmethod
    def get_model_type(self) -> str:
        """Get the type of model (e.g. 'stylegan2', 'stable-diffusion')"""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the specific model name/version"""
        pass
    
    @property
    @abstractmethod
    def image_size(self) -> int:
        """Get the output image size"""
        pass 