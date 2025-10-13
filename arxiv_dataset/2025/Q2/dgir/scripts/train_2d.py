#!/usr/bin/env python3
"""
2D Registration Training Script

This script trains a 2D diffusion-regularized registration network.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from diffusion_registration.training.config import Config
from diffusion_registration.training.trainer import train_model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Train 2D diffusion registration network',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--data-root',
        type=str,
        help='Override data root directory'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Override batch size'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        help='Override learning rate'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        help='Override number of epochs'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'auto'],
        default='auto',
        help='Device to use for training'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        help='Resume from checkpoint path'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Override config with command line arguments
    if args.data_root:
        config.data.data_root = args.data_root
    
    if args.batch_size:
        config.training.batch_size = args.batch_size
    
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    
    if args.epochs:
        config.training.epochs = args.epochs
    
    if args.device != 'auto':
        config.training.device = args.device
    elif args.device == 'auto':
        import torch
        config.training.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Force 2D mode
    config.model.dimension = 2
    
    print(f"Starting 2D registration training with:")
    print(f"  Config: {args.config}")
    print(f"  Device: {config.training.device}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  Data root: {config.data.data_root}")
    
    # Train model
    try:
        train_model(args.config)
        print("Training completed successfully!")
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
