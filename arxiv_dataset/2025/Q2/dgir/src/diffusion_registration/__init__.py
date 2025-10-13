"""
Diffusion-Regularized Medical Image Registration

A PyTorch implementation of diffusion-regularized neural networks for 
medical image registration, supporting both 2D and 3D registration tasks.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Core imports
from .core.models import (
    DiffusionRegistrationNet,
    setup_diffusion_model,
    setup_registration_net,
    create_loss_function
)

from .training.config import Config

from .data.loaders import (
    RegistrationDataset,
    create_data_loaders,
    load_preprocessed_data_2d,
    load_preprocessed_data_3d,
    normalize_tensor
)

# Make key classes available at package level
__all__ = [
    # Core functionality
    'DiffusionRegistrationNet',
    'setup_diffusion_model', 
    'setup_registration_net',
    'create_loss_function',
    
    # Configuration
    'Config',
    
    # Data handling
    'RegistrationDataset',
    'create_data_loaders',
    'load_preprocessed_data_2d',
    'load_preprocessed_data_3d',
    'normalize_tensor',
    
    # Version info
    '__version__',
    '__author__',
    '__email__',
]
