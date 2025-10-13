"""
Default configuration for generative model fingerprinting.
"""
import os
import logging
from dataclasses import dataclass, field, fields
from typing import List, Optional, Union, Dict, Tuple, Type, Any, Protocol, runtime_checkable
import ast
import torch


@runtime_checkable
class Validatable(Protocol):
    """Protocol for objects that can validate their state."""
    def validate(self) -> None: ...


@dataclass
class ModelConfig:
    """Model configuration."""
    # Model type
    model_type: str = "stylegan2"  # One of ["stylegan2", "stable-diffusion"]
    
    # Common parameters
    img_size: int = 768
    image_pixel_set_seed: int = 42
    image_pixel_count: int = 32
    
    # StyleGAN2 parameters
    stylegan2_url: str = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq70k-paper256-ada.pkl"
    stylegan2_local_path: str = "ffhq70k-paper256-ada.pkl"
    
    # Stable Diffusion parameters
    sd_model_name: str = "stabilityai/stable-diffusion-xl-base-1.0"
    sd_enable_cpu_offload: bool = False
    sd_dtype: str = "float16"  # One of ["float16", "float32"]
    sd_num_inference_steps: int = 50
    sd_guidance_scale: float = 7.5
    sd_prompt: str = "A photo of a cat in a variety of real-world scenes, candid shot, natural lighting, diverse settings, DSLR photo"  # Default prompt for generation
    sd_reconstructor_size: str = "M"  # One of ["S", "M", "L"]
    
    # Multi-prompt training configuration
    enable_multi_prompt: bool = False  # Whether to use multiple prompts during training
    prompt_source: str = "local"  # One of ["local", "diffusiondb", "parti-prompts"]
    prompt_dataset_path: str = ""  # Path to local prompt dataset file (one prompt per line)
    prompt_dataset_size: int = 10000  # Number of prompts to load from dataset
    diffusiondb_subset: str = "2m"  # One of ["2m", "large"] - which DiffusionDB subset to use
    parti_prompts_category: str = ""  # Category to use from parti-prompts dataset (e.g., "People", "Animals", etc.)
    train_eval_split_ratio: float = 0.8  # Ratio of prompts to use for training vs evaluation
    
    # Pretrained model configuration
    selected_pretrained_models: List[str] = field(default_factory=list)
    custom_pretrained_models: Dict[str, Any] = field(default_factory=dict)

    def validate(self):
        """Validate configuration parameters."""
        assert self.model_type in ["stylegan2", "stable-diffusion"], f"Unknown model type: {self.model_type}"
        assert self.img_size > 0, "Image size must be positive"
        assert self.image_pixel_count > 0, "Pixel count must be positive"
        assert self.sd_dtype in ["float16", "float32"], f"Invalid dtype: {self.sd_dtype}"
        assert self.sd_num_inference_steps > 0, "Number of inference steps must be positive"
        assert self.sd_guidance_scale > 0, "Guidance scale must be positive"
        assert self.sd_reconstructor_size in ["S", "M", "L"], f"Invalid SD reconstructor size: {self.sd_reconstructor_size}"
        
        # Validate multi-prompt configuration
        if self.enable_multi_prompt:
            assert self.prompt_source in ["local", "diffusiondb", "parti-prompts"], f"Invalid prompt source: {self.prompt_source}"
            if self.prompt_source == "local":
                assert os.path.exists(self.prompt_dataset_path), f"Prompt dataset file not found: {self.prompt_dataset_path}"
            elif self.prompt_source == "parti-prompts":
                assert 0 < self.train_eval_split_ratio < 1, "Train-eval split ratio must be between 0 and 1"
            assert self.prompt_dataset_size > 0, "Prompt dataset size must be positive"
    
    def get_model_class(self) -> Type:
        """Get the model class based on model type."""
        if self.model_type == "stylegan2":
            from models.stylegan2_model import StyleGAN2Model
            return StyleGAN2Model
        elif self.model_type == "stable-diffusion":
            from models.stable_diffusion_model import StableDiffusionModel
            return StableDiffusionModel
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def get_model_kwargs(self, device: torch.device) -> Dict[str, Any]:
        """Get model initialization kwargs based on model type."""
        if self.model_type == "stylegan2":
            return {
                "model_url": self.stylegan2_url,
                "model_path": self.stylegan2_local_path,
                "device": device,
                "img_size": self.img_size
            }
        elif self.model_type == "stable-diffusion":
            return {
                "model_name": self.sd_model_name,
                "device": device,
                "img_size": self.img_size,
                "dtype": getattr(torch, self.sd_dtype),
                "enable_cpu_offload": self.sd_enable_cpu_offload
            }
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
    def get_generation_kwargs(self) -> Dict[str, Any]:
        """Get generation kwargs based on model type."""
        if self.model_type == "stylegan2":
            return {}
        elif self.model_type == "stable-diffusion":
            return {
                "prompt": self.sd_prompt,
                "num_inference_steps": self.sd_num_inference_steps,
                "guidance_scale": self.sd_guidance_scale
            }
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")


@dataclass
class ReconstructorConfig:
    """Reconstructor configuration."""
    hidden_dims: List[int] = field(default_factory=lambda: [1024, 2048, 1024, 512, 256])
    activation: str = "gelu"
    dropout_rate: float = 0.3
    num_residual_blocks: int = 3
    use_spectral_norm: bool = True
    use_layer_norm: bool = True
    use_attention: bool = True

    def validate(self):
        """Validate configuration parameters."""
        assert len(self.hidden_dims) > 0, "Must have at least one hidden dimension"
        assert all(dim > 0 for dim in self.hidden_dims), "All hidden dimensions must be positive"
        assert self.activation in ["gelu", "relu", "leaky_relu"], f"Unsupported activation: {self.activation}"
        assert 0 <= self.dropout_rate <= 1, "Dropout rate must be between 0 and 1"
        assert self.num_residual_blocks >= 0, "Number of residual blocks must be non-negative"


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 16
    total_iterations: int = 100000
    lr: float = 1e-4
    log_interval: int = 1
    checkpoint_interval: int = 10000

    def validate(self):
        """Validate configuration parameters."""
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.total_iterations > 0, "Total iterations must be positive"
        assert self.lr > 0, "Learning rate must be positive"
        assert self.log_interval > 0, "Log interval must be positive"
        assert self.checkpoint_interval > 0, "Checkpoint interval must be positive"


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    backend: str = "nccl"
    init_method: str = "env://"

    def validate(self):
        """Validate configuration parameters."""
        assert self.backend in ["nccl", "gloo"], f"Unsupported backend: {self.backend}"
        assert self.init_method.startswith(("env://", "tcp://", "file://")), f"Invalid init method: {self.init_method}"


@dataclass
class EvaluateConfig:
    """Configuration for evaluation."""
    # Basic evaluation settings
    num_samples: int = 1000
    batch_size: int = 16
    output_dir: str = "evaluation_results"
    seed: Optional[int] = None
    
    # Enable timing logs
    enable_timing_logs: bool = True
    
    # Pretrained model settings
    selected_pretrained_models: List[str] = field(default_factory=list)
    custom_pretrained_models: Dict[str, Any] = field(default_factory=dict)
    
    # Model transformation settings
    enable_quantization: bool = True  # Whether to evaluate quantized models
    enable_pruning: bool = True  # Whether to evaluate pruned models
    pruning_sparsity_levels: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])  # List of sparsity ratios to evaluate
    pruning_methods: List[str] = field(default_factory=lambda: ['magnitude', 'random'])  # List of pruning methods to evaluate
    
    # Downsampling settings
    enable_downsampling: bool = True  # Whether to evaluate downsampling transformations
    downsample_sizes: List[int] = field(default_factory=lambda: [16, 224])  # Sizes for downsampling evaluation

    def validate(self):
        """Validate configuration parameters."""
        assert self.num_samples > 0, "Number of samples must be positive"
        assert self.batch_size > 0, "Batch size must be positive"
        assert os.path.exists(self.output_dir) or os.access(os.path.dirname(self.output_dir), os.W_OK), \
            f"Output directory {self.output_dir} does not exist and cannot be created"
        assert all(size > 0 for size in self.downsample_sizes), "All downsample sizes must be positive"
        assert all(0 < sparsity < 1 for sparsity in self.pruning_sparsity_levels), "Pruning sparsity levels must be between 0 and 1"
        assert all(method in ['magnitude', 'random'] for method in self.pruning_methods), "Invalid pruning method"


@dataclass
class AttackConfig:
    """Configuration for unified attack script supporting three variants:
    1. baseline: Naive ResNet18 classifier (both target and gradient source)
    2. yu_2019: Yu2019AttributionClassifier (both target and gradient source)
    3. authprint: AuthPrint reconstructor (target) with ResNet18 classifier (gradient source)
    """
    # Attack type selection
    attack_type: str = "authprint"  # One of ["baseline", "yu_2019", "authprint"]
    
    # Attack parameters
    num_samples: int = 1000  # Number of samples to attack
    batch_size: int = 32  # Batch size for classifier training and evaluation
    epsilon: float = 0.1  # Maximum perturbation size (L∞ norm)
    detection_threshold: float = 0.002883  # MSE threshold for detection (95% TPR threshold)
    
    # Evaluation cases
    enable_quantization: bool = False  # Whether to evaluate quantized models
    enable_downsampling: bool = False  # Whether to evaluate downsampling transformations
    
    # Step size sweep parameters
    enable_step_size_sweep: bool = False  # Whether to perform step size sweep
    step_size_sweep_values: List[float] = field(default_factory=lambda: [
        0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 
        0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0
    ])  # List of step sizes to try when sweep is enabled
    
    # Classifier training parameters
    classifier_iterations: int = 10000  # Number of iterations for classifier training
    classifier_lr: float = 1e-4  # Learning rate for classifier training
    
    # PGD parameters
    pgd_step_size: float = 0.01  # Step size for PGD attack (default: epsilon/10)
    pgd_steps: int = 50  # Number of PGD iteration steps
    momentum: float = 0.9  # Momentum coefficient for PGD attack
    
    # Evaluation parameters
    enable_fid: bool = True  # Whether to compute FID scores
    enable_lpips: bool = True  # Whether to compute LPIPS scores
    enable_psnr_ssim: bool = True  # Whether to compute PSNR and SSIM
    
    # Logging parameters
    log_interval: int = 10  # How often to log progress during classifier training
    save_images: bool = False  # Whether to save example images from successful attacks
    
    def validate(self):
        """Validate configuration parameters."""
        assert self.attack_type in ["baseline", "yu_2019", "authprint"], \
            f"Invalid attack type: {self.attack_type}"
        assert self.num_samples > 0, "Number of samples must be positive"
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.epsilon > 0, "Epsilon must be positive"
        assert self.detection_threshold > 0, "Detection threshold must be positive"
        assert self.classifier_iterations > 0, "Classifier iterations must be positive"
        assert self.classifier_lr > 0, "Classifier learning rate must be positive"
        assert self.pgd_steps > 0, "PGD steps must be positive"
        assert self.pgd_step_size <= self.epsilon, "PGD step size should not exceed epsilon" if not self.enable_step_size_sweep else True
        assert self.log_interval > 0, "Log interval must be positive"
        if self.enable_step_size_sweep:
            assert len(self.step_size_sweep_values) > 0, "Step size sweep values list cannot be empty"


@dataclass
class Config:
    """Configuration class containing all settings."""
    model: ModelConfig = field(default_factory=ModelConfig)
    reconstructor: ReconstructorConfig = field(default_factory=ReconstructorConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluate: EvaluateConfig = field(default_factory=EvaluateConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    output_dir: str = "results"
    checkpoint_path: Optional[str] = None
    seed: Optional[int] = None
    
    def validate(self):
        """Validate configuration parameters."""
        self.model.validate()
        self.training.validate()
        self.evaluate.validate()
        self.attack.validate()
        assert os.path.exists(self.output_dir) or os.access(os.path.dirname(self.output_dir), os.W_OK), \
            f"Output directory {self.output_dir} does not exist and cannot be created"
    
    def update_from_args(self, args, mode='train'):
        """Update config from command line arguments."""
        # Model type and common configuration
        if hasattr(args, 'model_type'):
            self.model.model_type = args.model_type
        if hasattr(args, 'img_size'):
            self.model.img_size = args.img_size
        if hasattr(args, 'image_pixel_set_seed'):
            self.model.image_pixel_set_seed = args.image_pixel_set_seed
        if hasattr(args, 'image_pixel_count'):
            self.model.image_pixel_count = args.image_pixel_count
            
        # StyleGAN2 configuration
        if hasattr(args, 'stylegan2_url'):
            self.model.stylegan2_url = args.stylegan2_url
        if hasattr(args, 'stylegan2_local_path'):
            self.model.stylegan2_local_path = args.stylegan2_local_path
            
        # Stable Diffusion configuration
        if hasattr(args, 'sd_model_name'):
            self.model.sd_model_name = args.sd_model_name
        if hasattr(args, 'sd_enable_cpu_offload'):
            self.model.sd_enable_cpu_offload = args.sd_enable_cpu_offload
        if hasattr(args, 'sd_dtype'):
            self.model.sd_dtype = args.sd_dtype
        if hasattr(args, 'sd_num_inference_steps'):
            self.model.sd_num_inference_steps = args.sd_num_inference_steps
        if hasattr(args, 'sd_guidance_scale'):
            self.model.sd_guidance_scale = args.sd_guidance_scale
        if hasattr(args, 'sd_prompt'):
            self.model.sd_prompt = args.sd_prompt
        if hasattr(args, 'sd_reconstructor_size'):
            self.model.sd_reconstructor_size = args.sd_reconstructor_size
            
        # Multi-prompt configuration
        if hasattr(args, 'enable_multi_prompt'):
            self.model.enable_multi_prompt = args.enable_multi_prompt
        if hasattr(args, 'prompt_source'):
            self.model.prompt_source = args.prompt_source
        if hasattr(args, 'prompt_dataset_path'):
            self.model.prompt_dataset_path = args.prompt_dataset_path
        if hasattr(args, 'prompt_dataset_size'):
            self.model.prompt_dataset_size = args.prompt_dataset_size
        if hasattr(args, 'diffusiondb_subset'):
            self.model.diffusiondb_subset = args.diffusiondb_subset
        if hasattr(args, 'parti_prompts_category'):
            self.model.parti_prompts_category = args.parti_prompts_category
        if hasattr(args, 'train_eval_split_ratio'):
            self.model.train_eval_split_ratio = args.train_eval_split_ratio
        
        # Mode-specific configuration
        if mode == 'train':
            if hasattr(args, 'batch_size'):
                self.training.batch_size = args.batch_size
            if hasattr(args, 'total_iterations'):
                self.training.total_iterations = args.total_iterations
            if hasattr(args, 'lr'):
                self.training.lr = args.lr
            if hasattr(args, 'log_interval'):
                self.training.log_interval = args.log_interval
            if hasattr(args, 'checkpoint_interval'):
                self.training.checkpoint_interval = args.checkpoint_interval
                
        elif mode == 'evaluate':
            if hasattr(args, 'num_samples'):
                self.evaluate.num_samples = args.num_samples
            if hasattr(args, 'batch_size'):
                self.evaluate.batch_size = args.batch_size
            if hasattr(args, 'output_dir'):
                self.evaluate.output_dir = args.output_dir
            if hasattr(args, 'seed'):
                self.evaluate.seed = args.seed
            if hasattr(args, 'enable_timing_logs'):
                self.evaluate.enable_timing_logs = args.enable_timing_logs
            # Model transformation settings
            if hasattr(args, 'enable_quantization'):
                self.evaluate.enable_quantization = args.enable_quantization
            if hasattr(args, 'enable_pruning'):
                self.evaluate.enable_pruning = args.enable_pruning
            if hasattr(args, 'pruning_sparsity_levels'):
                self.evaluate.pruning_sparsity_levels = args.pruning_sparsity_levels
            if hasattr(args, 'pruning_methods'):
                self.evaluate.pruning_methods = args.pruning_methods
                
        elif mode == 'attack':
            # Update attack parameters
            if hasattr(args, 'attack_type'):
                self.attack.attack_type = args.attack_type
            if hasattr(args, 'num_samples'):
                self.attack.num_samples = args.num_samples
            if hasattr(args, 'batch_size'):
                self.attack.batch_size = args.batch_size
            if hasattr(args, 'epsilon'):
                self.attack.epsilon = args.epsilon
            if hasattr(args, 'detection_threshold'):
                self.attack.detection_threshold = args.detection_threshold
            if hasattr(args, 'log_interval'):
                self.attack.log_interval = args.log_interval
            # Add classifier parameters
            if hasattr(args, 'classifier_iterations'):
                self.attack.classifier_iterations = args.classifier_iterations
            if hasattr(args, 'classifier_lr'):
                self.attack.classifier_lr = args.classifier_lr
            # Add PGD parameters
            if hasattr(args, 'pgd_step_size'):
                self.attack.pgd_step_size = args.pgd_step_size
            if hasattr(args, 'pgd_steps'):
                self.attack.pgd_steps = args.pgd_steps
            if hasattr(args, 'momentum'):
                self.attack.momentum = args.momentum
            # Add step size sweep parameters
            if hasattr(args, 'enable_step_size_sweep'):
                self.attack.enable_step_size_sweep = args.enable_step_size_sweep
                # Only override step_size_sweep_values if explicitly provided
                if hasattr(args, 'step_size_sweep_values') and args.step_size_sweep_values is not None:
                    self.attack.step_size_sweep_values = args.step_size_sweep_values
            
            # Add evaluation case parameters
            if hasattr(args, 'enable_quantization'):
                self.attack.enable_quantization = args.enable_quantization
            if hasattr(args, 'enable_downsampling'):
                self.attack.enable_downsampling = args.enable_downsampling
        
        # Common configuration
        if hasattr(args, 'output_dir'):
            self.output_dir = args.output_dir
        if hasattr(args, 'checkpoint_path'):
            self.checkpoint_path = args.checkpoint_path
        if hasattr(args, 'seed'):
            self.seed = args.seed
        
        # Validate the updated configuration
        self.validate()


def get_default_config() -> Config:
    """Get default configuration."""
    return Config() 