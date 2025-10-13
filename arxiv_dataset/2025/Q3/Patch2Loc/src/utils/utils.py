import logging
import os
import warnings
from typing import List, Sequence
import numpy as np
import lightning as pl
import re
import torch
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.utilities import rank_zero_only
import yaml 
import lightning
from torch.optim import Optimizer



def extract_volume_patches(volume, patch_size, slice_start, total_slices_num, rejection_rate, stride=(1, 1, 1), brain_masks=None):
    num_slices, C, H, W = volume.shape
    device = volume.device
    patch_h, patch_w = patch_size
    stride_h, stride_w, stride_z = stride
    # Calculate padding sizes
    pad_h = patch_h // 2
    pad_w = patch_w // 2

    # Pad the volume
    padded_volume = torch.nn.functional.pad(
        volume,
        (pad_w, pad_w, pad_h, pad_h),  # left, right, top, bottom
        mode='constant',
        value=0
    )

    # Create starting coordinates for each patch with configurable stride
    y_coords = torch.arange(0, H, stride_h, device=device)
    x_coords = torch.arange(0, W, stride_w, device=device)
    z_coords = torch.arange(0, num_slices, stride_z, device=device)

    # Create meshgrid of all patch starting positions
    grid_y, grid_x, grid_z = torch.meshgrid(y_coords, x_coords, z_coords, indexing='ij')

    # Create patch offset grid
    y_offsets = torch.arange(patch_h, device=device)
    x_offsets = torch.arange(patch_w, device=device)
    y_grid, x_grid = torch.meshgrid(y_offsets, x_offsets, indexing='ij')

    # Reshape grids to [num_patches, ...]
    grid_y = grid_y.reshape(-1)
    grid_x = grid_x.reshape(-1)
    grid_z = grid_z.reshape(-1)

    # Get final coordinates for each patch (adding padding offset)
    y_indices = grid_y[:, None, None] + y_grid[None, :, :]  # [num_patches, patch_h, patch_w]
    x_indices = grid_x[:, None, None] + x_grid[None, :, :]  # [num_patches, patch_h, patch_w]
    z_indices = grid_z[:, None, None]  # [num_patches, 1, 1]

    # Extract all patches at once using advanced indexing
    patches = padded_volume[
        z_indices,  # [num_patches, 1, 1]
        torch.zeros_like(z_indices),  # Channel dimension
        y_indices,  # [num_patches, patch_h, patch_w]
        x_indices   # [num_patches, patch_h, patch_w]
    ]
    # Stack locations and normalize
    #this is stupid workaround but don't I have time.. ICML in 5 days
    if type(slice_start) == torch.Tensor: 
        slice_start=torch.repeat_interleave(slice_start, grid_z.shape[0]//num_slices)
    locations = torch.stack([grid_y, grid_x, grid_z + slice_start], dim=1)  # [num_patches, 3]
    if brain_masks is not None:
        masks = torch.nn.functional.pad(
        brain_masks,
        (pad_w, pad_w, pad_h, pad_h),  # left, right, top, bottom
        mode='constant',
        value=0
        )
        patches_masks = masks[
            z_indices,  # [num_patches, 1, 1]
            torch.zeros_like(z_indices),  # Channel dimension
            y_indices,  # [num_patches, patch_h, patch_w]
            x_indices   # [num_patches, patch_h, patch_w]
        ]
        
        
        percentages = (patches_masks==1).float().mean(dim=(1,2))

        # Create mask for valid patches
        valid_mask = percentages >= rejection_rate

        # Filter patches and locations
        patches = patches[valid_mask]
        locations = locations[valid_mask]
    if type(total_slices_num) == int:
        max_locations = torch.tensor([H, W, total_slices_num], device=device)
    else:
        max_locations = torch.stack([torch.tensor(H,device=device).repeat(locations.shape[0]), torch.tensor(W, device=device).repeat(locations.shape[0]), 
                     total_slices_num.repeat_interleave(locations.shape[0]//num_slices)]).permute(1,0)
    normalized_locations = locations.float() / max_locations
    return patches.unsqueeze(1), 100 * normalized_locations


def extract_fold_number(filename):
    """
    Extracts the fold number from a filename.

    Args:
        filename (str): The filename containing the fold number.

    Returns:
        int: The fold number if found, otherwise None.
    """
    match = re.search(r'_fold-(\d+)\.ckpt', filename)
    if match:
        return int(match.group(1))
    return None

def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - easier access to debug mode
    - forcing debug friendly configuration
    Modifies DictConfig in place.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    log = get_logger()

    # enable adding new keys to config
    OmegaConf.set_struct(config, False)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # set <config.trainer.fast_dev_run=True> if <config.debug=True>
    if config.get("debug"):
        log.info("Running in debug mode! <config.debug=True>")
        config.trainer.fast_dev_run = True

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    if config.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        # Debuggers don't like GPUs or multiprocessing
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0

    # disable adding new keys to config
    OmegaConf.set_struct(config, True)


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "trainer",
        "model",
        "datamodule",
        "callbacks",
        "logger",
        "seed",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.txt", "w") as fp:
        rich.print(tree, file=fp)


def empty(*args, **kwargs):
    pass

from lightning.pytorch.loggers.logger import Logger

@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[Logger],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.
    Additionaly saves:
        - number of trainable model parameters
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["model"]
    hparams["datamodule"] = config["datamodule"]

    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # save number of model parameters
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    hparams['run_id'] = trainer.logger.experiment.id
    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # this is just a trick to prevent trainer from logging hparams of model,
    # since we already did that above
    trainer.logger.log_hyperparams = empty


def finish(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[Logger],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, lightning.pytorch.loggers.WandbLogger):
            import wandb

            wandb.finish()

def summarize(eval_dict, prefix): # removes list entries from dictionary for faster logging
    # for set in list(eval_dict) : 
    eval_dict_new = {}
    for key in list(eval_dict) :
        if type(eval_dict[key]) is not list :
            eval_dict_new[prefix + '/' + key] = eval_dict[key]
    return eval_dict_new

def get_yaml(path): # read yaml 
    with open(path, "r") as stream:
        try:
            file = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return file


def clip_to_percentiles(array, lower=1, upper=99):
    """
    Clip array values to the [lower, upper] percentiles using NumPy.
    
    Parameters:
        array (np.ndarray): Input array.
        lower (float): Lower percentile (0–100).
        upper (float): Upper percentile (0–100).
    
    Returns:
        np.ndarray: Clipped array.
    """
    lower_val = np.percentile(array, lower)
    upper_val = np.percentile(array, upper)
    return np.clip(array, lower_val, upper_val)


def get_specific_checkpoint(path, checkpoint_to_load, fold_num):
    checkpoint_path = path
    all_checkpoints = os.listdir(checkpoint_path + '/checkpoints')
    if checkpoint_to_load == 'last':
        matching_checkpoints = [c for c in all_checkpoints if "last" in c]
        matching_checkpoints.sort(key = lambda x: x.split('fold-')[1][0:1])
        return checkpoint_path + '/checkpoints/' + matching_checkpoints[fold_num]
    
    elif 'best' in checkpoint_to_load :
        matching_checkpoints = [c for c in all_checkpoints if "last" not in c]
        matching_checkpoints.sort(key = lambda x: x.split('loss-')[1][0:4])
        for cp in matching_checkpoints:
            if fold_num in cp:
                return checkpoint_path + '/checkpoints/' + cp

def get_checkpoint(path, checkpoint_to_load, num_folds): 
    checkpoint_path = path
    all_checkpoints = os.listdir(checkpoint_path + '/checkpoints')
    try:
        hparams = get_yaml(path+'/csv//hparams.yaml')
        wandbID = hparams['run_id'] if 'run_id' in hparams else None
    except Exception:
        pass
    checkpoints = {}
    for fold in range(num_folds):
        checkpoints[f'fold-{fold+1}'] = [] # dict to store the checkpoints with their path for different folds

    if checkpoint_to_load == 'last':
        matching_checkpoints = [c for c in all_checkpoints if "last" in c]
        matching_checkpoints.sort(key = lambda x: x.split('fold-')[1][0:1])
        available_folds = [int(x.split('fold-')[1][0:1]) for x in matching_checkpoints ]
        for index, fold in enumerate(available_folds):
            checkpoints[f'fold-{fold}'] = checkpoint_path + '/checkpoints/' + matching_checkpoints[index]
    elif 'best' in checkpoint_to_load :
        matching_checkpoints = [c for c in all_checkpoints if "last" not in c]
        matching_checkpoints.sort(key = lambda x: x.split('loss-')[1][0:4]) # sort by loss value -> increasing
        for fold in checkpoints:
            for cp in matching_checkpoints:
                if fold in cp:
                    checkpoints[fold].append(checkpoint_path + '/checkpoints/' + cp)
            if not 'best_k' in checkpoint_to_load: # best_k loads the k best checkpoints 
                if len(checkpoints[fold]) == 0:
                    checkpoints[fold] = None
                else:
                    checkpoints[fold] = checkpoints[fold][0] # get only the best (first) checkpoint of that fold
    return wandbID, checkpoints




def gradient_norm(optimizer: Optimizer):
    total_norm = 0.0
    for group in optimizer.param_groups:

        parameters = group['params']
        if len(parameters) == 0:
            print(f'len(parameters) is zero')
            continue

        # Calculate the gradient norm for the current parameter group
        grads = [p.grad.flatten() for p in parameters if p.grad is not None]
        if len(grads) == 0:
            print(f'No grad is on')
            continue
        grad_norm = torch.norm(torch.cat(grads))
        total_norm += grad_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm



def max_min_normalize(tensor, min_val=0.0, max_val=1.0):
    """
    Perform max-min normalization on a PyTorch tensor.

    :param tensor: Input tensor of any shape.
    :param min_val: Minimum value of the desired output range (default: 0.0).
    :param max_val: Maximum value of the desired output range (default: 1.0).
    :return: Normalized tensor with the same shape as the input.
    """
    # Get the minimum and maximum values of the tensor
    tensor_min = tensor.min()
    tensor_max = tensor.max()

    # Avoid division by zero
    if tensor_max == tensor_min:
        raise ValueError("Normalization is not possible when all elements are equal.")

    # Apply max-min normalization
    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)

    # Scale to the desired range [min_val, max_val]
    scaled_tensor = normalized_tensor * (max_val - min_val) + min_val
    return scaled_tensor


def remove_prefix_from_keys(state_dict, prefix):
    # Create a new state_dict with updated keys
    updated_state_dict = {key[len(prefix):] if key.startswith(prefix) else key: value 
                          for key, value in state_dict.items()}
    return updated_state_dict
