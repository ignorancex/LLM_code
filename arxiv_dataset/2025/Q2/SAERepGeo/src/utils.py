import os
import torch
import logging
import sys

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s:%(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    return logger

def flush_print(message):
    print(message, flush=True)

def setup_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = logging.getLogger(__name__)
    logger.info(f"Device: {device}")
    flush_print(f"INFO:__main__:Device: {device}")
    return device

def create_output_dirs(base_output_dir):
    os.makedirs(base_output_dir, exist_ok=True)
    case1_dir = os.path.join(base_output_dir, "case1_stratified_manifold")
    case2_dir = os.path.join(base_output_dir, "case2_representation_structure")
    case3_dir = os.path.join(base_output_dir, "case3_intervention_analysis")
    os.makedirs(case1_dir, exist_ok=True)
    os.makedirs(case2_dir, exist_ok=True)
    os.makedirs(case3_dir, exist_ok=True)
    return base_output_dir, case1_dir, case2_dir, case3_dir