import torch
import torch.nn as nn
from typing import Optional
from .base_model import BaseGenerativeModel
from models.model_utils import load_stylegan2_model
from utils.image_transforms import prune_model_weights

class StyleGAN2Model(BaseGenerativeModel, nn.Module):
    """StyleGAN2 model implementation."""
    
    def __init__(
        self,
        model_url: str,
        model_path: str,
        device: torch.device,
        img_size: int = 768
    ):
        """Initialize StyleGAN2 model.
        
        Args:
            model_url (str): URL to download model from
            model_path (str): Local path to save/load model
            device (torch.device): Device to load model on
            img_size (int): Output image size
        """
        super(StyleGAN2Model, self).__init__()
        self._device = device
        self._img_size = img_size
        self.model_url = model_url
        self.model_path = model_path
        self.model = load_stylegan2_model(model_url, model_path, device)
        self.model.eval()
        
    def forward(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass for compatibility with torch.nn.Module.
        
        Args:
            z (torch.Tensor): Input latent vectors
            **kwargs: Additional arguments passed to synthesis
            
        Returns:
            torch.Tensor: Generated images
        """
        return self.generate_images(z.size(0), z=z, **kwargs)
        
    def generate_images(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        z: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Generate images using StyleGAN2.
        
        Args:
            batch_size (int): Number of images to generate
            device (torch.device, optional): Device override
            z (Optional[torch.Tensor]): Optional latent vectors
            **kwargs: Additional arguments passed to synthesis
            
        Returns:
            torch.Tensor: Generated images [B, C, H, W]
        """
        device = device or self._device
        
        # Generate or use provided latent vectors
        if z is None:
            z = torch.randn(batch_size, self.z_dim, device=device)
        
        with torch.no_grad():
            w = self.mapping(z, None)
            images = self.synthesis(w, noise_mode="const")
                
        return images
    
    def mapping(self, z: torch.Tensor, c: Optional[torch.Tensor], **kwargs) -> torch.Tensor:
        """Map latent vectors from Z space to W space.
        
        Args:
            z (torch.Tensor): Latent vectors in Z space
            c (Optional[torch.Tensor]): Class conditioning, if applicable
            **kwargs: Additional arguments passed to mapping network
            
        Returns:
            torch.Tensor: Latent vectors in W space
        """
        if hasattr(self.model, 'module'):
            return self.model.module.mapping(z, c, **kwargs)
        return self.model.mapping(z, c, **kwargs)
        
    def synthesis(self, w: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate images from W space latent vectors.
        
        Args:
            w (torch.Tensor): Latent vectors in W space
            **kwargs: Additional arguments passed to synthesis network
            
        Returns:
            torch.Tensor: Generated images
        """
        if hasattr(self.model, 'module'):
            return self.model.module.synthesis(w, **kwargs)
        return self.model.synthesis(w, **kwargs)
    
    def get_model_type(self) -> str:
        return "stylegan2"
    
    def get_model_name(self) -> str:
        return "stylegan2-ada"
        
    @property
    def image_size(self) -> int:
        return self._img_size

    @property
    def z_dim(self) -> int:
        """Get the latent dimension size."""
        if hasattr(self.model, 'module'):
            return self.model.module.z_dim
        return self.model.z_dim
        
    def parameters(self):
        """Return model parameters for compatibility with torch.nn.Module."""
        return self.model.parameters()
        
    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
        return self
        
    def train(self, mode=True):
        """Set model to training mode."""
        if mode:
            self.model.train()
        else:
            self.model.eval()
        return self

    def prune(self, sparsity=0.5, method='magnitude'):
        """Prune model weights to achieve target sparsity.
        
        Args:
            sparsity (float): Target sparsity ratio (0.0 to 1.0)
            method (str): Pruning method ('magnitude' or 'random')
            
        Returns:
            StyleGAN2Model: New model instance with pruned weights
        """
        # Create a new model instance
        pruned_model = StyleGAN2Model(
            model_url=self.model_url,
            model_path=self.model_path,
            device=self._device,
            img_size=self._img_size
        )
        
        # Use the utility function to prune weights
        pruned_model = prune_model_weights(pruned_model, sparsity=sparsity, method=method)
        
        return pruned_model 