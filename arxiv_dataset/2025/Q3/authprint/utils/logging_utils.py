"""
Utilities for logging setup.
"""
import logging
import os
import sys
from typing import Optional


def setup_logging(
    output_dir: str,
    rank: int,
    log_level: int = logging.INFO,
    log_filename: Optional[str] = None
) -> None:
    """
    Setup logging for the training process.
    
    Args:
        output_dir (str): Directory to save log files to.
        rank (int): Process rank in distributed training.
        log_level (int): Logging level for the root logger.
        log_filename (str, optional): Name of the log file. If None, will use default name.
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine log file name
        if log_filename is None:
            log_filename = f"train_rank{rank}.log"
        
        log_file = os.path.join(output_dir, log_filename)
        
        # Clear existing handlers from root logger
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Configure logging
        logging.basicConfig(
            level=log_level if rank == 0 else logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout) if rank == 0 else logging.NullHandler()
            ]
        )
        
        logging.info(f"Logging initialized (rank {rank}). Log file: {log_file}")
        
    except Exception as e:
        # Fallback to console logging if file logging fails
        print(f"Warning: Failed to setup file logging: {str(e)}")
        logging.basicConfig(
            level=log_level if rank == 0 else logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout) if rank == 0 else logging.NullHandler()]
        ) 