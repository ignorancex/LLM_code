import logging
import re
import warnings
import io

from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Tuple, List
import numpy as np
import torchvision.transforms.functional as F

import torch
from PIL import Image
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.utils.data import TensorDataset, Subset, DataLoader

from src.models.components.ae_gmm import AEGMM
from src.utils import pylogger, rich_utils
from src.utils.images_utils import extract_patch

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


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


def reindex_dataset(dataset: Subset, targets: List[Tuple[int, List[int]]]):
    new_tensors: List[torch.Tensor] = [None] * len(dataset.dataset.tensors)
    for index, target_indexes in targets:
        new_indexes = torch.argsort(dataset.dataset.tensors[index])
        for target_index in target_indexes:
            new_tensors[target_index] = dataset.dataset.tensors[target_index][new_indexes]
    return TensorDataset(*new_tensors)


def mse_dsw(batch1: torch.Tensor, batch2: torch.Tensor):
    """MSE compartible with DSW loss"""
    squared_diff = (batch1 - batch2) ** 2
    sum_squared_diff = torch.sum(squared_diff, tuple(range(1, squared_diff.dim())))
    mse_loss = torch.mean(sum_squared_diff)
    return mse_loss


def pixel_wise_entropy(batch: torch.Tensor):
    return torch.mean(-batch * torch.log(batch + 1e-16), dim=(2, 3)).mean((0, 1))


def call_background_classifier(model, input, patch_bg_size, patch=False, data_type=float) -> torch.Tensor:
    if patch:
        if input.ndim > 3:
            patches = torch.stack([extract_patch(i, [int(tmp * patch_bg_size) for tmp in i.shape[1:]]) for i in input])
        else:
            patches = extract_patch(input, [int(tmp * patch_bg_size) for tmp in input.shape[1:]])
        input = F.resize(patches, input.shape[-2:])
    # TODO This is a workaround solution.
    with torch.inference_mode():
        if isinstance(model, AEGMM):
            return torch.tensor(
                model.gmm.predict(model.ae.encoder(input).cpu().numpy().astype(data_type))).to(
                input).to(torch.uint8)
        else:
            raise RuntimeError(f'background classifire has an uknown type')


def plotly_to_tensor(fig):
    # Convert Plotly figure to image
    img_bytes = fig.to_image(format="png")

    # Convert image bytes to PIL Image
    image = Image.open(io.BytesIO(img_bytes))

    # Convert PIL Image to PyTorch tensor
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image)
    return image_tensor


def load_net_state_from_checkpoint(checkpoint_path, model_state_dict, keyword):
    checkpoint = torch.load(checkpoint_path)
    # Filter keys that start with 'augmenter'
    augmenter_keys = [key for key in checkpoint['state_dict'].keys() if key.startswith(keyword)]
    # Create a new dictionary with filtered keys and their corresponding values
    filtered_state_dict = {key: checkpoint['state_dict'][key] for key in augmenter_keys}

    mapped_state_dict = {}
    for filtered_key in filtered_state_dict.keys():
        # Apply your custom rules to match keys
        # Example: Remove 'augmenter.' prefix and replace with ''
        model_key = filtered_key.replace(keyword + '.', '')
        # Check if the model key exists in the model's state dictionary
        if model_key in model_state_dict:
            # Add the key-value pair to the mapped state dictionary
            mapped_state_dict[model_key] = filtered_state_dict[filtered_key]
    return mapped_state_dict


def get_channels_number_from_dataloader(data_loader: DataLoader) -> int:
    first_data_sample = data_loader.dataset[0][0]
    channels_number = first_data_sample.shape[0]
    return channels_number


def stratified_cat(tensor0: torch.Tensor, tensor1: torch.Tensor, chunk_size0: int, chunk_size1: int) -> torch.Tensor:
    """
    Concatenate two tensors in a stratified way by alternating between chunks from each tensor.
    Stops filling when tensor0 is exhausted.

    Parameters:
        tensor0 (torch.Tensor): The first input tensor.
        tensor1 (torch.Tensor): The second input tensor.
        chunk_size0 (int): The number of elements to select from tensor0 in each iteration.
        chunk_size1 (int): The number of elements to select from tensor1 in each iteration.

    Returns:
        torch.Tensor: A concatenated tensor formed by alternately selecting chunks from each input tensor.
    """
    # Initialize an empty list to hold the stratified chunks
    stratified_chunks = []

    # Determine the length of each tensor
    len0 = len(tensor0)
    len1 = len(tensor1)

    # Calculate the number of chunks for each tensor
    num_chunks0 = (len0 + chunk_size0 - 1) // chunk_size0

    # Iterate through each chunk index
    for i in range(num_chunks0):
        # Calculate the start and end indices for the current chunk of tensor0
        start0 = i * chunk_size0
        end0 = start0 + chunk_size0

        # Append chunk from tensor0 to the stratified_chunks list if there are remaining elements
        if start0 < len0:
            stratified_chunks.append(tensor0[start0:end0])
        else:
            break  # Stop if tensor0 is exhausted

        # Calculate the start and end indices for the current chunk of tensor1
        start1 = i * chunk_size1
        end1 = start1 + chunk_size1

        # Append chunk from tensor1 to the stratified_chunks list if there are remaining elements
        if start1 < len1:
            stratified_chunks.append(tensor1[start0:start0 + chunk_size1])

    # Concatenate the stratified chunks into a single tensor
    stratified_tensor = torch.cat(stratified_chunks)

    return stratified_tensor


def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()


def clamp(x, min_val, max_val):
    return max(min(x, max_val), min_val)


def unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()


def apply_colormap_to_tensor(tensor, colormap='BuPu_r'):
    """
    Map a batch of single-channel images to 3-channel RGB using a specified colormap.

    Args:
        tensor (torch.Tensor): Input tensor with shape (B, H, W) or (B, 1, H, W).
        colormap (str): Name of the matplotlib colormap to use (default: 'BuPu_r').

    Returns:
        torch.Tensor: Tensor with shape (B, 3, H, W) representing the batch of RGB images.
    """
    assert tensor.dim() in [3, 4], "Input tensor must be 3D (B, H, W) or 4D (B, 1, H, W)"

    if tensor.dim() == 4:
        tensor = tensor.squeeze(1)  # Convert (B, 1, H, W) to (B, H, W)

    # Normalize each image independently to [0, 1]
    tensor_min = tensor.view(tensor.size(0), -1).min(dim=1)[0].view(-1, 1, 1)
    tensor_max = tensor.view(tensor.size(0), -1).max(dim=1)[0].view(-1, 1, 1)
    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-5)

    # Use the new colormap API to get the colormap function
    cmap = plt.colormaps.get_cmap(colormap)

    # Apply colormap and convert the list to a single NumPy array for efficiency
    rgba_images = np.array([cmap(img.numpy())[:, :, :3] for img in normalized_tensor])

    # Convert to PyTorch tensor and permute to (B, 3, H, W)
    rgb_batch = torch.from_numpy(rgba_images).permute(0, 3, 1, 2).float()

    return rgb_batch


def get_background_clusters_num(datamodule):
    if hasattr(datamodule, 'background_classifier_name') and datamodule.background_classifier_name:
        background_classifier_name = datamodule.background_classifier_name
        match = re.search(r'/(\d+)(\.pt)?$', background_classifier_name)
        if match:
            return int(match.group(1))
        else:
            raise ValueError(
                "No cluster number found in background classifier name. The background classifier name should have the cluster number as the file name")
    else:
        log.warning("Datamodule does not have attribute 'background_classifier_name'")
        return None


def throw_invalid_exception(tensor):
    contains_nan = torch.isnan(tensor).any()

    contains_inf = torch.isinf(tensor).any()

    if contains_nan or contains_inf:
        print("The tensor contains invalid values.")
        if contains_nan:
            print("It contains NaNs.")
        if contains_inf:
            print("It contains Infs.")
        raise RuntimeError('Invalid values were found')


def normalize_images(images):
    if images.dtype == torch.uint8:
        images = images / 255
    else:
        # pass
        images = images / 16
    return images
