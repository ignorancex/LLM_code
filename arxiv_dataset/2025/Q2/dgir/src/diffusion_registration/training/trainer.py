"""
Training utilities for diffusion registration networks.

This module provides a comprehensive training framework with logging,
checkpointing, and evaluation capabilities.
"""

import torch
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

from .config import Config
from ..core.models import DiffusionRegistrationNet
from ..data.loaders import RegistrationDataset, create_data_loaders


class Trainer:
    """
    Main training class for diffusion registration networks.
    
    This class handles the complete training loop including:
    - Model training and validation
    - Checkpoint saving and loading
    - Loss logging and visualization
    - Learning rate scheduling
    """
    
    def __init__(self, net: DiffusionRegistrationNet, config: Config):
        """
        Initialize trainer.
        
        Args:
            net: Registration network to train.
            config: Training configuration.
        """
        self.net = net
        self.config = config
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), 
            lr=config.training.learning_rate
        )
        
        # Initialize tracking
        self.current_epoch = 0
        self.loss_history = {
            'all_loss': [],
            'similarity_loss': [],
            'regularization_loss': []
        }
        
        # Setup logging
        self._setup_logging()
        
        # Create output directories
        config.create_output_dirs()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = Path(self.config.output.log_dir) / 'training.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, train_dataset: RegistrationDataset) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_dataset: Training dataset.
            
        Returns:
            Dictionary of average losses for the epoch.
        """
        self.net.train()
        epoch_losses = {
            'all_loss': [],
            'similarity_loss': [],
            'regularization_loss': []
        }
        
        # Training loop
        batch_A, batch_B = train_dataset.get_random_batch(self.config.training.batch_size)
        
        self.optimizer.zero_grad()
        loss_object = self.net(batch_A, batch_B)
        
        # Backward pass
        loss_object.all_loss.backward()
        self.optimizer.step()
        
        # Record losses
        epoch_losses['all_loss'].append(loss_object.all_loss.item())
        epoch_losses['similarity_loss'].append(loss_object.similarity_loss.item())
        
        # Handle different loss types
        if hasattr(loss_object, 'bending_energy_loss'):
            epoch_losses['regularization_loss'].append(loss_object.bending_energy_loss.item())
        elif hasattr(loss_object, 'inverse_consistency_loss'):
            epoch_losses['regularization_loss'].append(loss_object.inverse_consistency_loss.item())
        else:
            epoch_losses['regularization_loss'].append(0.0)
        
        # Calculate averages
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        return avg_losses
    
    def validate(self, val_dataset: RegistrationDataset) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_dataset: Validation dataset.
            
        Returns:
            Dictionary of validation losses.
        """
        self.net.eval()
        
        with torch.no_grad():
            batch_A, batch_B = val_dataset.get_random_batch(
                min(self.config.training.batch_size, len(val_dataset))
            )
            loss_object = self.net(batch_A, batch_B)
            
            val_losses = {
                'all_loss': loss_object.all_loss.item(),
                'similarity_loss': loss_object.similarity_loss.item(),
            }
            
            # Handle different loss types
            if hasattr(loss_object, 'bending_energy_loss'):
                val_losses['regularization_loss'] = loss_object.bending_energy_loss.item()
            elif hasattr(loss_object, 'inverse_consistency_loss'):
                val_losses['regularization_loss'] = loss_object.inverse_consistency_loss.item()
            else:
                val_losses['regularization_loss'] = 0.0
        
        return val_losses
    
    def train(self, train_dataset: RegistrationDataset, 
              val_dataset: Optional[RegistrationDataset] = None):
        """
        Main training loop.
        
        Args:
            train_dataset: Training dataset.
            val_dataset: Optional validation dataset.
        """
        self.logger.info(f"Starting training for {self.config.training.epochs} epochs")
        self.logger.info(f"Batch size: {self.config.training.batch_size}")
        self.logger.info(f"Learning rate: {self.config.training.learning_rate}")
        
        for epoch in tqdm(range(self.current_epoch, self.config.training.epochs), 
                         desc="Training"):
            self.current_epoch = epoch
            
            # Train epoch
            train_losses = self.train_epoch(train_dataset)
            
            # Update loss history
            for key, value in train_losses.items():
                self.loss_history[key].append(value)
            
            # Validation
            val_losses = None
            if val_dataset is not None and epoch % (self.config.training.print_every * 5) == 0:
                val_losses = self.validate(val_dataset)
            
            # Logging
            if epoch % self.config.training.print_every == 0:
                log_msg = (f"[{epoch}] Train - Total: {train_losses['all_loss']:.4f}, "
                          f"Sim: {train_losses['similarity_loss']:.4f}, "
                          f"Reg: {train_losses['regularization_loss']:.4f}")
                
                if val_losses is not None:
                    log_msg += (f" | Val - Total: {val_losses['all_loss']:.4f}, "
                               f"Sim: {val_losses['similarity_loss']:.4f}, "
                               f"Reg: {val_losses['regularization_loss']:.4f}")
                
                self.logger.info(log_msg)
            
            # Save checkpoints
            if epoch % self.config.training.save_every == 0 and epoch > 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pth")
        
        self.logger.info("Training completed!")
        self.save_checkpoint("final_model.pth")
    
    def save_checkpoint(self, filename: str):
        """
        Save training checkpoint.
        
        Args:
            filename: Name of checkpoint file.
        """
        checkpoint_path = Path(self.config.output.checkpoint_dir) / filename
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_history': self.loss_history,
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else None,
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
            
        Returns:
            Checkpoint dictionary.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load states
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.loss_history = checkpoint.get('loss_history', self.loss_history)
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint
    
    def plot_losses(self, save_path: Optional[str] = None):
        """
        Plot training losses.
        
        Args:
            save_path: Optional path to save the plot.
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            for i, (loss_name, loss_values) in enumerate(self.loss_history.items()):
                if loss_values:  # Only plot if we have data
                    axes[i].plot(loss_values)
                    axes[i].set_title(f'{loss_name.replace("_", " ").title()}')
                    axes[i].set_xlabel('Epoch')
                    axes[i].set_ylabel('Loss')
                    axes[i].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                self.logger.info(f"Loss plot saved: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            self.logger.warning("matplotlib not available, skipping plot")


def train_model(config_path: str):
    """
    Convenience function to train a model from config file.
    
    Args:
        config_path: Path to configuration YAML file.
    """
    # Load configuration
    config = Config(config_path)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Create data loaders
    train_dataset, test_dataset = create_data_loaders(config)
    
    # Create network
    net = DiffusionRegistrationNet(config)
    net = net.to(config.device)
    
    # Create trainer
    trainer = Trainer(net, config)
    
    # Train
    trainer.train(train_dataset, test_dataset)
    
    # Plot losses
    loss_plot_path = Path(config.output.results_dir) / 'training_losses.png'
    trainer.plot_losses(str(loss_plot_path))
