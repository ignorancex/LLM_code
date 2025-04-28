#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Config for running experiments

"""

from pathlib import Path
import torch 
import gpytorch 
from astropy.io import fits

TORCH_VERSION = torch.__version__
GPYTORCH_VERSION = gpytorch.__version__

AVAILABLE_GPU = torch.cuda.device_count()
GPU_ACTIVE = bool(AVAILABLE_GPU)
EPSILON = 1e-5
BASE_SEED = 24

BASE_PATH = Path(__file__).parent.parent
RESULTS_DIR = BASE_PATH / "results"
DATASET_DIR = BASE_PATH / "data"
LOG_DIR = BASE_PATH / "logs"
TRAINED_MODELS = BASE_PATH / "trained_models"

### experimental config

RANDOM_SEEDS = [34,52,61,70,97,42,96,12,7,4]

size = '20k'  ## or '1k'

if size == '20k':
    
    hdu = fits.open('data/data_norm_sdss16_SNR10_all.fits')
    num_inducing = 250
    latent_dim = 10
    inference_mode = 'point' ## or 'map'
    test_size = 7000
    
else:
    
    hdu = fits.open('data/data_norm_sdss16_SNR10_random_1.fits')
    num_inducing = 120
    latent_dim = 10
    inference_mode = 'point' ## or 'map'
    test_size = 200



