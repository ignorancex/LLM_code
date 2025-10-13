"""
Model setup and factory functions for diffusion registration.

This module provides functions to set up diffusion models and registration networks
with proper configuration and initialization.
"""

import torch
from typing import Tuple, Any
from pathlib import Path

from ..training.config import Config
from . import networks, wrappers, losses


def setup_diffusion_model(config: Config) -> Tuple[Any, Any]:
    """
    Set up the diffusion model and diffusion process.
    
    Args:
        config: Configuration object containing diffusion parameters.
        
    Returns:
        Tuple of (model, diffusion) objects.
        
    Raises:
        FileNotFoundError: If the diffusion model checkpoint is not found.
        RuntimeError: If model loading fails.
    """
    try:
        # Import guided_diffusion here to handle optional dependency
        from guided_diffusion.script_util import create_model_and_diffusion
    except ImportError as e:
        raise ImportError(
            "guided_diffusion package not found. Please install it or "
            "add it as a submodule to use diffusion-based losses."
        ) from e
    
    # Create model and diffusion
    model, diffusion = create_model_and_diffusion(
        image_size=config.diffusion.image_size,
        class_cond=config.diffusion.class_cond,
        learn_sigma=config.diffusion.learn_sigma,
        num_channels=config.diffusion.num_channels,
        num_res_blocks=config.diffusion.num_res_blocks,
        channel_mult=config.diffusion.channel_mult,
        num_heads=config.diffusion.num_heads,
        num_head_channels=config.diffusion.num_head_channels,
        num_heads_upsample=config.diffusion.num_heads_upsample,
        attention_resolutions=config.diffusion.attention_resolutions,
        dropout=config.diffusion.dropout,
        diffusion_steps=config.diffusion.diffusion_steps,
        noise_schedule=config.diffusion.noise_schedule,
        timestep_respacing=config.diffusion.timestep_respacing,
        use_kl=config.diffusion.use_kl,
        predict_xstart=config.diffusion.predict_xstart,
        rescale_timesteps=config.diffusion.rescale_timesteps,
        rescale_learned_sigmas=config.diffusion.rescale_learned_sigmas,
        use_checkpoint=config.diffusion.use_checkpoint,
        use_scale_shift_norm=config.diffusion.use_scale_shift_norm,
        resblock_updown=config.diffusion.resblock_updown,
        use_fp16=config.diffusion.use_fp16,
        use_new_attention_order=config.diffusion.use_new_attention_order,
    )
    
    # Load pretrained weights
    model_path = Path(config.diffusion.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Diffusion model checkpoint not found: {model_path}")
    
    try:
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
    except Exception as e:
        raise RuntimeError(f"Failed to load diffusion model from {model_path}: {e}") from e
    
    return model, diffusion


def setup_registration_net(config: Config, model: Any = None, diffusion: Any = None, 
                          input_shape: Tuple[int, ...] = None) -> wrappers.RegistrationModule:
    """
    Set up the registration network with appropriate architecture.
    
    Args:
        config: Configuration object.
        model: Pretrained diffusion model (optional, needed for diffusion-based losses).
        diffusion: Diffusion process object (optional, needed for diffusion-based losses).
        input_shape: Input tensor shape. If None, inferred from config.
        
    Returns:
        Configured registration network.
        
    Raises:
        ValueError: If configuration is invalid or incompatible.
    """
    if input_shape is None:
        # Create a default shape based on config
        if config.model.dimension == 2:
            input_shape = (config.training.batch_size, 1, 
                          config.model.image_size, config.model.image_size)
        elif config.model.dimension == 3:
            input_shape = (config.training.batch_size, 1, 224, 192, 160)  # Default 3D shape
        else:
            raise ValueError(f"Unsupported dimension: {config.model.dimension}")
    
    # Create base network
    if config.model.dimension == 2:
        base_net = networks.tallUNet2(dimension=2)
    elif config.model.dimension == 3:
        base_net = networks.tallUNet2(dimension=3)
    else:
        raise ValueError(f"Unsupported dimension: {config.model.dimension}")
    
    # Wrap in function from vector field
    inner_net = wrappers.FunctionFromVectorField(base_net)
    
    # Add multiscale levels
    for _ in range(config.model.num_multiscale_levels):
        inner_net = wrappers.TwoStepRegistration(
            wrappers.DownsampleRegistration(inner_net, dimension=config.model.dimension),
            wrappers.FunctionFromVectorField(
                networks.tallUNet2(dimension=config.model.dimension)
            )
        )
    
    # Setup loss function
    loss_fn = create_loss_function(config, model, diffusion)
    
    # Create final network with regularization
    net = wrappers.DiffusionRegularizedNet(
        inner_net,
        loss_fn,
        lmbda=config.loss.lambda_regularization
    )
    
    # Assign identity map
    net.assign_identity_map(input_shape)
    
    return net


def create_loss_function(config: Config, model: Any = None, diffusion: Any = None) -> Any:
    """
    Create loss function based on configuration.
    
    Args:
        config: Configuration object.
        model: Pretrained diffusion model (needed for diffusion-based losses).
        diffusion: Diffusion process object (needed for diffusion-based losses).
        
    Returns:
        Configured loss function.
        
    Raises:
        ValueError: If loss type is not supported or required parameters are missing.
    """
    loss_type = config.loss.type.upper()
    
    if loss_type == "LNCC":
        return losses.LNCC(sigma=config.loss.sigma)
    
    elif loss_type == "NEWLNCC":
        if model is None or diffusion is None:
            raise ValueError("NewLNCC loss requires both model and diffusion objects")
        if config.model.dimension == 2:
            return losses.NewLNCC(
                diffusion=diffusion,
                model=model,
                sigma=config.loss.sigma,
                eps=config.loss.eps
            )
        elif config.model.dimension == 3:
            return losses.NewLNCC3D(
                diffusion=diffusion,
                model=model,
                sigma=config.loss.sigma,
                eps=config.loss.eps
            )
    
    elif loss_type == "NCC":
        return losses.NCC()
    
    elif loss_type == "SSD":
        return losses.SSD()
    
    elif loss_type == "MINDSSC":
        return losses.MINDSSC()
    
    elif loss_type == "NMI":
        return losses.NMI()
    
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")


class DiffusionRegistrationNet:
    """
    High-level interface for diffusion registration networks.
    
    This class provides a simplified interface for setting up and using
    diffusion-regularized registration networks.
    """
    
    def __init__(self, config: Config, input_shape: Tuple[int, ...] = None):
        """
        Initialize the registration network.
        
        Args:
            config: Configuration object.
            input_shape: Input tensor shape. If None, inferred from config.
        """
        self.config = config
        
        # Setup diffusion model if needed
        self.model = None
        self.diffusion = None
        if config.loss.type.upper() in ["NEWLNCC"]:
            self.model, self.diffusion = setup_diffusion_model(config)
            self.model.eval()  # Set to evaluation mode
        
        # Setup registration network
        self.net = setup_registration_net(
            config, self.model, self.diffusion, input_shape
        )
    
    def to(self, device: str):
        """Move networks to specified device."""
        if self.model is not None:
            self.model = self.model.to(device)
        self.net = self.net.to(device)
        return self
    
    def train(self):
        """Set network to training mode."""
        self.net.train()
        return self
    
    def eval(self):
        """Set network to evaluation mode."""
        self.net.eval()
        return self
    
    def parameters(self):
        """Get network parameters for optimization."""
        return self.net.parameters()
    
    def __call__(self, *args, **kwargs):
        """Forward pass through the network."""
        return self.net(*args, **kwargs)
    
    def state_dict(self):
        """Get network state dictionary."""
        return self.net.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load network state dictionary."""
        return self.net.load_state_dict(state_dict)
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else None,
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint.get('config', None)
