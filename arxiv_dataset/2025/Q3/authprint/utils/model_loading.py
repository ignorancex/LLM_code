"""
Utilities for loading generative models.
"""
import logging
import os
import traceback
from typing import Dict, Tuple, Optional, Any

import torch
from models.stylegan2_model import StyleGAN2Model
from models.stable_diffusion_model import StableDiffusionModel
from models.base_model import BaseGenerativeModel

# Dictionary of pretrained StyleGAN2 model URLs and local paths
STYLEGAN2_MODELS = {
    # FFHQ Models
    'ffhq70k-ada': ("https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq70k-paper256-ada.pkl",
               "ffhq70k-paper256-ada.pkl"), # this serves as a control test
    'ffhq1k': ("https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq1k-paper256-ada.pkl",
               "ffhq1k-paper256-ada.pkl"),
    'ffhq30k': ("https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq30k-paper256-ada.pkl",
                "ffhq30k-paper256-ada.pkl"),
    'ffhq70k-bcr': ("https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq70k-paper256-ada-bcr.pkl",
                    "ffhq70k-paper256-ada-bcr.pkl"),
    'ffhq70k-noaug': ("https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq70k-paper256-noaug.pkl",
                     "ffhq70k-paper256-noaug.pkl"),
    
    # # LSUN Cat Models
    # 'lsuncat100k-ada': ("https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/lsuncat100k-paper256-ada.pkl",
    #                     "lsuncat100k-paper256-ada.pkl"),
    # 'lsuncat1k': ("https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/lsuncat1k-paper256-ada.pkl",
    #               "lsuncat1k-paper256-ada.pkl"),
    # 'lsuncat30k': ("https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/lsuncat30k-paper256-ada.pkl",
    #                "lsuncat30k-paper256-ada.pkl"),
    # 'lsuncat100k-bcr': ("https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/lsuncat100k-paper256-ada-bcr.pkl",
    #                     "lsuncat100k-paper256-ada-bcr.pkl"),
    # 'lsuncat100k-noaug': ("https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/lsuncat100k-paper256-noaug.pkl",
    #                       "lsuncat100k-paper256-noaug.pkl")
}

# Dictionary of pretrained Stable Diffusion models
STABLE_DIFFUSION_MODELS = {
    'sd-2.1': "stabilityai/stable-diffusion-2-1",
    'sd-2.0': "stabilityai/stable-diffusion-2",
    'sd-1.5': "sd-legacy/stable-diffusion-v1-5",
    'sd-1.4': "CompVis/stable-diffusion-v1-4",
    'sd-1.3': "CompVis/stable-diffusion-v1-3",
    'sd-1.2': "CompVis/stable-diffusion-v1-2",
    'sd-1.1': "CompVis/stable-diffusion-v1-1",
}

def load_pretrained_models(
    device: torch.device,
    rank: int = 0,
    model_type: str = "stylegan2",
    selected_models: Optional[Dict[str, Any]] = None,
    img_size: int = 768,
    enable_cpu_offload: bool = False,
    dtype: torch.dtype = torch.float16
) -> Dict[str, BaseGenerativeModel]:
    """
    Load pretrained models.
    
    Args:
        device: Device to load models on
        rank: Distributed training rank
        model_type: Type of models to load ("stylegan2" or "stable-diffusion")
        selected_models: Optional dictionary mapping model names to their configs.
                       For StyleGAN2: (url, local_path) tuples
                       For Stable Diffusion: model names
        img_size: Output image size
        enable_cpu_offload: Whether to enable CPU offloading (SD only)
        dtype: Model dtype (SD only)
        
    Returns:
        Dictionary mapping model names to loaded models
    """
    models = {}
    
    # Use provided model dictionary or default to predefined ones
    if selected_models is None:
        if model_type == "stylegan2":
            model_dict = STYLEGAN2_MODELS
        else:
            model_dict = STABLE_DIFFUSION_MODELS
    else:
        model_dict = selected_models
    
    # Load all models from model dictionary
    for model_name, model_config in model_dict.items():
        if rank == 0:
            logging.info(f"Loading pretrained model: {model_name}")
        
        try:
            if model_type == "stylegan2":
                url, local_path = model_config
                models[model_name] = StyleGAN2Model(
                    model_url=url,
                    model_path=local_path,
                    device=device,
                    img_size=img_size
                )
            else:  # stable-diffusion
                model_path = model_config if isinstance(model_config, str) else model_config["model_name"]
                models[model_name] = StableDiffusionModel(
                    model_name=model_path,
                    device=device,
                    img_size=img_size,
                    dtype=dtype,
                    enable_cpu_offload=enable_cpu_offload
                )
            
            if rank == 0:
                logging.info(f"Successfully loaded pretrained model: {model_name}")
                
        except Exception as e:
            if rank == 0:
                logging.error(f"Failed to load pretrained model {model_name}: {str(e)}")
                logging.error(traceback.format_exc())
            # Skip this model and continue with others
            continue
    
    if rank == 0:
        logging.info(f"Loaded {len(models)} pretrained models: {list(models.keys())}")
    
    return models 