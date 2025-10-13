"""Configuration management for diffusion registration."""

import os
import yaml
from dataclasses import dataclass
from typing import List, Optional, Union
from pathlib import Path


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    epochs: int = 50000
    batch_size: int = 4
    learning_rate: float = 1e-4
    print_every: int = 100
    save_every: int = 10000
    device: str = "cuda"


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    dimension: int = 2
    image_size: int = 256
    num_multiscale_levels: int = 3


@dataclass
class NetworkConfig:
    """Network architecture parameters."""
    unet_channels: List[int] = None
    use_batch_norm: bool = True
    dropout_rate: float = 0.0
    
    def __post_init__(self):
        if self.unet_channels is None:
            self.unet_channels = [2, 16, 32, 64, 256, 512]


@dataclass
class DiffusionConfig:
    """Diffusion model configuration."""
    model_path: str = "guided_diffusion/256x256_diffusion_uncond.pt"
    image_size: int = 256
    class_cond: bool = False
    learn_sigma: bool = True
    num_channels: int = 256
    num_res_blocks: int = 2
    channel_mult: str = ""
    num_heads: int = 4
    num_head_channels: int = 64
    num_heads_upsample: int = -1
    attention_resolutions: str = "32,16,8"
    dropout: float = 0.0
    diffusion_steps: int = 1000
    noise_schedule: str = "linear"
    timestep_respacing: str = ""
    use_kl: bool = False
    predict_xstart: bool = False
    rescale_timesteps: bool = False
    rescale_learned_sigmas: bool = False
    use_checkpoint: bool = False
    use_scale_shift_norm: bool = True
    resblock_updown: bool = True
    use_fp16: bool = False
    use_new_attention_order: bool = False


@dataclass
class LossConfig:
    """Loss function configuration."""
    type: str = "NewLNCC"
    sigma: float = 4.0
    lambda_regularization: float = 1.0
    eps: float = 1e-6


@dataclass
class DataConfig:
    """Data loading configuration."""
    data_root: str = ""
    weird_xrays: List[int] = None
    normalize_images: bool = True
    
    def __post_init__(self):
        if self.weird_xrays is None:
            self.weird_xrays = []


@dataclass
class OutputConfig:
    """Output configuration."""
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    results_dir: str = "results"


class Config:
    """Main configuration class that aggregates all config sections."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML configuration file. If None, uses defaults.
        """
        self.training = TrainingConfig()
        self.model = ModelConfig()
        self.network = NetworkConfig()
        self.diffusion = DiffusionConfig()
        self.loss = LossConfig()
        self.data = DataConfig()
        self.output = OutputConfig()
        
        if config_path is not None:
            self.load_from_yaml(config_path)
    
    def load_from_yaml(self, config_path: Union[str, Path]) -> None:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file.
            
        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If config file is malformed.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # Update configurations from loaded data
        for section_name, section_config in config_dict.items():
            if hasattr(self, section_name):
                section_obj = getattr(self, section_name)
                for key, value in section_config.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
    
    def save_to_yaml(self, config_path: Union[str, Path]) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            config_path: Path where to save the configuration.
        """
        config_dict = {
            'training': self.training.__dict__,
            'model': self.model.__dict__,
            'network': self.network.__dict__,
            'diffusion': self.diffusion.__dict__,
            'loss': self.loss.__dict__,
            'data': self.data.__dict__,
            'output': self.output.__dict__,
        }
        
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def create_output_dirs(self) -> None:
        """Create output directories if they don't exist."""
        for dir_path in [self.output.checkpoint_dir, self.output.log_dir, self.output.results_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    @property
    def device(self) -> str:
        """Get the configured device."""
        return self.training.device
    
    @property
    def lr(self) -> float:
        """Get the learning rate."""
        return self.training.learning_rate
    
    @property
    def epochs(self) -> int:
        """Get number of epochs."""
        return self.training.epochs
    
    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return self.training.batch_size
    
    @property
    def print_every(self) -> int:
        """Get print frequency."""
        return self.training.print_every
