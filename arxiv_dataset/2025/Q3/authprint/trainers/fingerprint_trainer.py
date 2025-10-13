"""
Trainer for StyleGAN fingerprinting.
"""
import logging
import time
import random
import re
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import load_dataset

from config.default_config import Config
from models.reconstructor import ReconstructorSD_L, ReconstructorSD_M, ReconstructorSD_S, StyleGAN2Reconstructor
from models.stable_diffusion_model import StableDiffusionModel
from utils.checkpoint import save_checkpoint, load_checkpoint


def clean_prompt(prompt: str) -> str:
    """
    Clean a prompt by removing excessive punctuation and normalizing whitespace.
    
    Args:
        prompt (str): Input prompt to clean.
        
    Returns:
        str: Cleaned prompt.
    """
    # Remove URLs
    prompt = re.sub(r'http\S+|www\.\S+', '', prompt)
    
    # Remove excessive punctuation (more than 1 of the same character)
    prompt = re.sub(r'([!?.]){2,}', r'\1', prompt)
    
    # Remove excessive whitespace
    prompt = ' '.join(prompt.split())
    
    # Remove <|endoftext|> tokens
    prompt = prompt.replace('<|endoftext|>', '')
    
    # Remove empty parentheses and brackets
    prompt = re.sub(r'\(\s*\)|\[\s*\]|\{\s*\}', '', prompt)
    
    # Normalize commas and colons
    prompt = re.sub(r'\s*,\s*', ', ', prompt)
    prompt = re.sub(r'\s*:\s*', ': ', prompt)
    
    return prompt.strip()


class FingerprintTrainer:
    """
    Trainer for StyleGAN fingerprinting.
    """
    def __init__(
        self,
        config: Config,
        local_rank: int,
        rank: int,
        world_size: int,
        device: torch.device
    ):
        """
        Initialize the trainer.
        
        Args:
            config (Config): Training configuration.
            local_rank (int): Local process rank.
            rank (int): Global process rank.
            world_size (int): Total number of processes.
            device (torch.device): Device to run on.
        """
        self.config = config
        self.local_rank = local_rank
        self.rank = rank
        self.world_size = world_size
        self.device = device
        
        # Initialize models
        self.generative_model = None
        self.reconstructor = None
        
        # Initialize optimizer
        self.optimizer = None
        
        # Track training progress
        self.global_step = 0
        self.start_iteration = 1  # Track starting iteration for resuming
        
        # Image pixel selection parameters
        self.image_pixel_indices = None
        self.image_pixel_count = self.config.model.image_pixel_count
        self.image_pixel_set_seed = self.config.model.image_pixel_set_seed
        logging.info(f"Using image-based approach with {self.image_pixel_count} pixels and seed {self.image_pixel_set_seed}")
        
        # Initialize prompt dataset if multi-prompt mode is enabled
        self.prompts: Optional[List[str]] = None
        if self.config.model.enable_multi_prompt:
            if self.config.model.prompt_source == "local":
                self._load_prompt_dataset_local()
            elif self.config.model.prompt_source == "diffusiondb":
                self._load_prompt_dataset_diffusiondb()
            else:  # parti-prompts
                self._load_prompt_dataset_parti()
    
    def _load_prompt_dataset_local(self) -> None:
        """
        Load prompts from a local file.
        """
        if self.rank == 0:
            logging.info(f"Loading prompts from local file: {self.config.model.prompt_dataset_path}")
        
        try:
            with open(self.config.model.prompt_dataset_path, 'r', encoding='utf-8') as f:
                all_prompts = [line.strip() for line in f if line.strip()]
            
            # Sample the specified number of prompts
            if len(all_prompts) > self.config.model.prompt_dataset_size:
                self.prompts = random.sample(all_prompts, self.config.model.prompt_dataset_size)
            else:
                self.prompts = all_prompts
                if self.rank == 0:
                    logging.warning(f"Prompt dataset contains fewer prompts ({len(all_prompts)}) "
                                  f"than requested ({self.config.model.prompt_dataset_size})")
            
            if self.rank == 0:
                logging.info(f"Loaded {len(self.prompts)} prompts from local file")
                logging.info(f"Sample prompts: {self.prompts[:3]}")
        
        except Exception as e:
            if self.rank == 0:
                logging.error(f"Error loading prompt dataset: {str(e)}")
            raise
    
    def _load_prompt_dataset_diffusiondb(self) -> None:
        """
        Load prompts from DiffusionDB dataset.
        """
        if self.rank == 0:
            logging.info("Loading prompts from DiffusionDB dataset")
        
        try:
            # Load the metadata table from DiffusionDB
            subset_mapping = {
                "2m_random_10k": "2m_random_10k",  # Using random 10k subset instead of full dataset
                "large_random_10k": "large_random_10k",
                "2m_random_5k": "2m_random_5k",
            }
            subset = subset_mapping[self.config.model.diffusiondb_subset]
            dataset = load_dataset("poloclub/diffusiondb", subset, split="train", trust_remote_code=True)
            
            # Extract and clean all unique prompts
            all_prompts = []
            raw_prompts = list(set(dataset["prompt"]))
            
            if self.rank == 0:
                logging.info(f"Found {len(raw_prompts)} unique prompts before cleaning")
            
            for prompt in raw_prompts:
                if not prompt:  # Skip empty prompts
                    continue
                    
                cleaned_prompt = clean_prompt(prompt)
                if cleaned_prompt and len(cleaned_prompt.split()) <= 50:  # Only keep reasonably sized prompts
                    all_prompts.append(cleaned_prompt)
            
            if self.rank == 0:
                logging.info(f"Retained {len(all_prompts)} prompts after cleaning")
            
            # Sample the specified number of prompts
            if len(all_prompts) > self.config.model.prompt_dataset_size:
                self.prompts = random.sample(all_prompts, self.config.model.prompt_dataset_size)
            else:
                self.prompts = all_prompts
                if self.rank == 0:
                    logging.warning(f"DiffusionDB contains fewer clean prompts ({len(all_prompts)}) "
                                  f"than requested ({self.config.model.prompt_dataset_size})")
            
            if self.rank == 0:
                logging.info(f"Loaded {len(self.prompts)} prompts from DiffusionDB")
                logging.info("Sample prompts after cleaning:")
                for i, prompt in enumerate(self.prompts[:10]):  # Show first 10 prompts
                    logging.info(f"  {i+1}. {prompt}")
        
        except Exception as e:
            if self.rank == 0:
                logging.error(f"Error loading DiffusionDB dataset: {str(e)}")
            raise
    
    def _load_prompt_dataset_parti(self) -> None:
        """
        Load prompts from the Parti-Prompts dataset for a specific category.
        """
        if self.rank == 0:
            logging.info(f"Loading prompts from Parti-Prompts dataset for category: {self.config.model.parti_prompts_category}")
        
        try:
            # Load the Parti-Prompts dataset
            dataset = load_dataset("nateraw/parti-prompts", split="train", trust_remote_code=True)
            
            # Filter by category if specified
            if self.config.model.parti_prompts_category:
                dataset = dataset.filter(lambda x: x["Category"] == self.config.model.parti_prompts_category)
                if self.rank == 0:
                    logging.info(f"Found {len(dataset)} prompts in category '{self.config.model.parti_prompts_category}'")
            
            # Extract and clean all prompts
            all_prompts = []
            for item in dataset:
                if not item["Prompt"]:  # Skip empty prompts
                    continue
                
                cleaned_prompt = clean_prompt(item["Prompt"])
                if cleaned_prompt:
                    all_prompts.append(cleaned_prompt)
            
            if self.rank == 0:
                logging.info(f"Retained {len(all_prompts)} prompts after cleaning")
            
            # Split into train and eval sets
            random.shuffle(all_prompts)  # Shuffle before splitting
            split_idx = int(len(all_prompts) * self.config.model.train_eval_split_ratio)
            train_prompts = all_prompts[:split_idx]
            
            # Sample the specified number of prompts for training
            if len(train_prompts) > self.config.model.prompt_dataset_size:
                self.prompts = random.sample(train_prompts, self.config.model.prompt_dataset_size)
            else:
                self.prompts = train_prompts
                if self.rank == 0:
                    logging.warning(f"Training set contains fewer prompts ({len(train_prompts)}) "
                                  f"than requested ({self.config.model.prompt_dataset_size})")
            
            if self.rank == 0:
                logging.info(f"Using {len(self.prompts)} prompts for training")
                logging.info("Sample prompts:")
                for i, prompt in enumerate(self.prompts[:10]):  # Show first 10 prompts
                    logging.info(f"  {i+1}. {prompt}")
        
        except Exception as e:
            if self.rank == 0:
                logging.error(f"Error loading Parti-Prompts dataset: {str(e)}")
            raise
    
    def _sample_prompts(self, batch_size: int) -> List[str]:
        """
        Sample prompts for the current batch.
        
        Args:
            batch_size (int): Number of prompts to sample.
            
        Returns:
            List[str]: List of sampled prompts.
        """
        if not self.config.model.enable_multi_prompt or not self.prompts:
            return [self.config.model.sd_prompt] * batch_size
        
        return random.sample(self.prompts, min(batch_size, len(self.prompts)))
    
    def setup_models(self) -> None:
        """
        Set up all models needed for training.
        """
        try:
            # Initialize generative model
            if self.rank == 0:
                logging.info(f"Loading {self.config.model.model_type} model...")
                
            model_class = self.config.model.get_model_class()
            model_kwargs = self.config.model.get_model_kwargs(self.device)
            self.generative_model = model_class(**model_kwargs)
            
            # Initialize reconstructor based on model type and size
            reconstructor_output_dim = self.image_pixel_count  # For direct pixel prediction
            
            if self.config.model.model_type == "stylegan2":
                self.reconstructor = StyleGAN2Reconstructor(
                    image_size=self.config.model.img_size,
                    channels=3,
                    output_dim=reconstructor_output_dim
                ).to(self.device)
                if self.rank == 0:
                    logging.info(f"Initialized StyleGAN2Reconstructor with output_dim={reconstructor_output_dim}")
            else:  # stable-diffusion
                reconstructor_class = {
                    "S": ReconstructorSD_S,
                    "M": ReconstructorSD_M,
                    "L": ReconstructorSD_L
                }[self.config.model.sd_reconstructor_size]
                
                self.reconstructor = reconstructor_class(
                    image_size=self.config.model.img_size,
                    channels=3,
                    output_dim=reconstructor_output_dim
                ).to(self.device)
                
                if self.rank == 0:
                    logging.info(f"Initialized SD-Reconstructor-{self.config.model.sd_reconstructor_size} with output_dim={reconstructor_output_dim}")
            
            # Initialize optimizer
            self.optimizer = optim.Adam(
                self.reconstructor.parameters(),
                lr=self.config.training.lr
            )
            if self.rank == 0:
                logging.info("Optimizer initialized with reconstructor parameters")
            
            # Ensure models are initialized before DDP wrapping
            torch.cuda.synchronize()
            if self.world_size > 1:
                torch.distributed.barrier()
            
            # Wrap models in DDP if using distributed training
            if self.world_size > 1:
                if self.reconstructor is not None:
                    self.reconstructor = DDP(
                        self.reconstructor,
                        device_ids=[self.local_rank],
                        output_device=self.local_rank,
                        find_unused_parameters=True  # Add this to handle unused parameters
                    )
                    if self.rank == 0:
                        logging.info("Models wrapped in DistributedDataParallel")
                
                # Final sync point after DDP wrapping
                torch.distributed.barrier()
        
        except Exception as e:
            logging.error(f"Error in setup_models: {str(e)}")
            raise
    
    def _generate_pixel_indices(self) -> None:
        """
        Generate indices for selecting pixels from the image.
        """
        # Set random seed for reproducibility
        torch.manual_seed(self.image_pixel_set_seed)
        
        # Calculate total number of pixels
        total_pixels = self.config.model.img_size * self.config.model.img_size * 3  # RGB images
        
        # Generate random indices
        self.image_pixel_indices = torch.randperm(total_pixels)[:self.image_pixel_count]
        
        if self.rank == 0:
            logging.info(f"Generated {self.image_pixel_count} pixel indices with seed {self.image_pixel_set_seed}")
            logging.info(f"Selected pixel indices: {self.image_pixel_indices.tolist()}")
    
    def validate_indices(self) -> None:
        """
        Validate that indices have been generated.
        """
        if self.image_pixel_indices is None:
            self._generate_pixel_indices()
    
    def extract_image_partial(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract selected pixels from images.
        
        Args:
            images (torch.Tensor): Input images [batch_size, channels, height, width].
            
        Returns:
            torch.Tensor: Selected pixels [batch_size, num_pixels].
        """
        # Ensure indices are generated
        self.validate_indices()
        
        # Flatten images and extract selected pixels
        batch_size = images.size(0)
        flattened = images.view(batch_size, -1)
        return flattened[:, self.image_pixel_indices]
    
    def train_iteration(self) -> Dict[str, float]:
        """
        Run a single training iteration.
        
        Returns:
            Dict[str, float]: Dictionary containing training metrics.
        """
        # Get generation kwargs based on model type
        gen_kwargs = self.config.model.get_generation_kwargs()
        
        # For Stable Diffusion, update prompts if multi-prompt mode is enabled
        if self.config.model.model_type == "stable-diffusion":
            prompts = self._sample_prompts(self.config.training.batch_size)
            gen_kwargs["prompt"] = prompts
            # Log prompts used in this iteration
            if self.rank == 0:
                logging.info(f"Iteration {self.global_step + 1} prompts:")
                for i, prompt in enumerate(prompts[:5]):  # Show first 5 prompts
                    logging.info(f"  {i+1}. {prompt}")
        
        # Generate images
        x = self.generative_model.generate_images(
            batch_size=self.config.training.batch_size,
            device=self.device,
            **gen_kwargs
        )
            
        # Extract features (real pixel values)
        features = self.extract_image_partial(x)
        true_values = features
        
        # Get reconstructor (handle DDP wrapping)
        reconstructor = self.reconstructor.module if hasattr(self.reconstructor, 'module') else self.reconstructor
        
        # Get predictions
        pred_values = self.reconstructor(x)
        train_loss = torch.mean(torch.pow(pred_values - true_values, 2))
        
        # Optimize
        self.optimizer.zero_grad()
        train_loss.backward()
        self.optimizer.step()
        
        # Now compute metrics in eval mode
        reconstructor.eval()  # Temporarily set to eval mode
        with torch.no_grad():
            pred_values = self.reconstructor(x)
            mse_distance = torch.mean(torch.pow(pred_values - true_values, 2), dim=1)
            mse_distance_mean = mse_distance.mean().item()
            mse_distance_std = mse_distance.std().item()
        reconstructor.train()  # Set back to train mode
        
        return {
            'train_loss': train_loss.item(),
            'mse_distance_mean': mse_distance_mean,
            'mse_distance_std': mse_distance_std
        }
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """
        try:
            if not self.reconstructor:
                if self.rank == 0:
                    logging.error("Reconstructor is not initialized. Make sure setup_models() is called first.")
                raise RuntimeError("Reconstructor not initialized")
            
            if self.rank == 0:
                logging.info(f"Loading checkpoint from {checkpoint_path}")
            
            # Verify checkpoint exists
            if not os.path.exists(checkpoint_path):
                if self.rank == 0:
                    logging.error(f"Checkpoint not found at {checkpoint_path}")
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            # Load checkpoint to CPU first
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Verify checkpoint contents
            required_keys = ['reconstructor_state', 'iteration']
            missing_keys = [key for key in required_keys if key not in checkpoint]
            if missing_keys:
                if self.rank == 0:
                    logging.error(f"Checkpoint is missing required keys: {missing_keys}")
                raise ValueError(f"Invalid checkpoint: missing keys {missing_keys}")
            
            # Get reconstructor (handle DDP wrapping)
            dec = self.reconstructor.module if hasattr(self.reconstructor, 'module') else self.reconstructor
            
            # Load reconstructor state
            try:
                dec.load_state_dict(checkpoint['reconstructor_state'])
                if self.rank == 0:
                    logging.info("Successfully loaded reconstructor state")
            except Exception as e:
                if self.rank == 0:
                    logging.error(f"Failed to load reconstructor state: {str(e)}")
                raise
            
            # Load optimizer state if available
            if self.optimizer is not None and 'optimizer_state' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state'])
                    if self.rank == 0:
                        logging.info("Successfully loaded optimizer state")
                except Exception as e:
                    if self.rank == 0:
                        logging.warning(f"Failed to load optimizer state: {str(e)}")
            
            # Update training progress
            self.start_iteration = checkpoint.get('iteration', 1)
            self.global_step = checkpoint.get('global_step', self.start_iteration - 1)
            
            if self.rank == 0:
                logging.info(f"Resuming from iteration {self.start_iteration}")
                if 'metrics' in checkpoint:
                    logging.info(f"Previous metrics: {checkpoint['metrics']}")
            
            # Synchronize after loading
            if self.world_size > 1:
                torch.distributed.barrier()
        
        except Exception as e:
            if self.rank == 0:
                logging.error(f"Error loading checkpoint: {str(e)}")
            raise
    
    def train(self) -> None:
        """
        Main training loop.
        """
        try:
            # Set up models first
            self.setup_models()
            
            # Synchronize before loading checkpoint
            if self.world_size > 1:
                torch.distributed.barrier()
            
            # Resume from checkpoint if specified
            if self.config.checkpoint_path:
                if self.rank == 0:
                    logging.info(f"Loading checkpoint from {self.config.checkpoint_path}")
                self.load_checkpoint(self.config.checkpoint_path)
            
            # Synchronize after loading checkpoint
            if self.world_size > 1:
                torch.distributed.barrier()
            
            # Training loop
            if self.rank == 0:
                logging.info("Starting training...")
                start_time = time.time()
            
            for iteration in range(self.start_iteration, self.config.training.total_iterations + 1):
                # Run training iteration
                metrics = self.train_iteration()
                
                # Update global step
                self.global_step = iteration
                
                # Log progress
                if self.rank == 0 and iteration % self.config.training.log_interval == 0:
                    elapsed = time.time() - start_time
                    logging.info(
                        f"Iteration {iteration}/{self.config.training.total_iterations} "
                        f"[{elapsed:.2f}s] "
                        f"Train Loss: {metrics['train_loss']:.6f} "
                        f"MSE: {metrics['mse_distance_mean']:.6f} ± {metrics['mse_distance_std']:.6f}"
                    )
                
                # Save checkpoint
                if self.rank == 0 and iteration % self.config.training.checkpoint_interval == 0:
                    save_checkpoint(
                        iteration=iteration,
                        reconstructor=self.reconstructor,
                        output_dir=self.config.output_dir,
                        rank=self.rank,
                        optimizer=self.optimizer,
                        metrics=metrics,
                        global_step=self.global_step
                    )
            
            if self.rank == 0:
                logging.info("Training completed")
        
        except Exception as e:
            logging.error(f"Error in training: {str(e)}")
            raise 