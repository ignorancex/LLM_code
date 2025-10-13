"""
Utility functions for model loading and manipulation.
"""
import copy
import logging
import os
import pickle

import torch
import torch.nn as nn

# Import StyleGAN2 dependencies - adjust this based on your environment
import sys
sys.path.append("./stylegan2-ada-pytorch")
import dnnlib
import legacy


def save_finetuned_model(model, path, filename):
    """
    Save a finetuned model to disk.
    
    Args:
        model (nn.Module): Model to save.
        path (str): Directory path to save to.
        filename (str): Filename to save as.
    """
    model_cpu = copy.deepcopy(model).cpu()
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, filename)
    with open(save_path, 'wb') as f:
        pickle.dump(model_cpu, f)
    logging.info(f"Model saved to {save_path}")


def load_finetuned_model(path):
    """
    Load a finetuned model from disk.
    
    Args:
        path (str): Path to the saved model.
        
    Returns:
        nn.Module: Loaded model.
    """
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


def clone_model(model):
    """
    Clones a model and ensures all parameters in the cloned model require gradients.
    
    Args:
        model (nn.Module): Model to clone.
        
    Returns:
        nn.Module: Cloned model with requires_grad=True for all parameters.
    """
    cloned_model = copy.deepcopy(model)
    cloned_model.train()
    for param in cloned_model.parameters():
        param.requires_grad = True
    return cloned_model


def load_stylegan2_model(url: str, local_path: str, device: torch.device) -> nn.Module:
    """
    Load a pre-trained StyleGAN2 model from a URL or local path.
    
    Args:
        url (str): URL to download the model from if not found locally.
        local_path (str): Local path to save the downloaded model or load existing model.
        device (torch.device): Device to load the model onto.
        
    Returns:
        nn.Module: StyleGAN2 generator model.
    """
    if not os.path.exists(local_path):
        logging.info(f"Downloading StyleGAN2 model to {local_path}...")
        torch.hub.download_url_to_file(url, local_path)
        logging.info("Download complete.")
    with dnnlib.util.open_url(local_path) as f:
        # Load the pickle and extract the generator 'G_ema'
        model = legacy.load_network_pkl(f)['G_ema'].to(device)
    return model 