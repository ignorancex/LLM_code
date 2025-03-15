import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Tuple, List
import math

from ExpToolKit.models import BaseAE
from .const import DENOISER_CONST


def kwargs_parser(
    model: BaseAE, kwargs: dict
) -> Tuple[int, torch.device, torch.optim.Optimizer, str, int]:
    """
    Parse the keyword arguments for the Denoiser class.

    Parameters
    ----------
    model : BaseAE
        The dimension reduction model.
    kwargs : dict
        The keyword arguments.

    Returns
    -------
    random_seed : int
        The random seed to use.
    device : torch.device
        The device to use.
    optim : torch.optim.Optimizer
        The optimizer to use.
    noise_type : str
        The type of noise to add to the data.
    epochs : int
        The number of epochs to train the model.
    """
    if (
        DENOISER_CONST.RANDOM_SEED_KEY not in kwargs
        or type(kwargs[DENOISER_CONST.RANDOM_SEED_KEY]) != int
    ):
        random_seed = DENOISER_CONST.DEFAULT_RANDOM_SEED
    else:
        random_seed = kwargs[DENOISER_CONST.RANDOM_SEED_KEY]

    if (
        DENOISER_CONST.DEVICE_KEY not in kwargs
        or type(kwargs[DENOISER_CONST.DEVICE_KEY]) != torch.device
    ):
        device = DENOISER_CONST.DEFAULT_DEVICE
    else:
        device = torch.device(kwargs[DENOISER_CONST.DEVICE_KEY])

    if (
        DENOISER_CONST.LR_KEY not in kwargs
        or type(kwargs[DENOISER_CONST.LR_KEY]) != float
    ):
        lr = DENOISER_CONST.DEFAULT_LR
    else:
        lr = kwargs[DENOISER_CONST.LR_KEY]

    if (
        DENOISER_CONST.WEIGHT_DECAY_KEY not in kwargs
        or type(kwargs[DENOISER_CONST.WEIGHT_DECAY_KEY]) != float
    ):
        weight_decay = DENOISER_CONST.DEFAULT_WEIGHT_DECAY
    else:
        weight_decay = kwargs[DENOISER_CONST.WEIGHT_DECAY_KEY]

    if (
        DENOISER_CONST.OPTIM_KEY not in kwargs
        or type(kwargs[DENOISER_CONST.OPTIM_KEY]) != str
    ):
        optim_type = DENOISER_CONST.DEFAULT_OPTIM
    else:
        optim_type = kwargs[DENOISER_CONST.OPTIM_KEY]

    if optim_type not in DENOISER_CONST.OPTIMS:
        raise ValueError(
            f"optim_type must be one of the following: {DENOISER_CONST.OPTIMS}"
        )
    if optim_type == DENOISER_CONST.SGD_OPTIM:
        optim = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_type == DENOISER_CONST.ADAM_OPTIM:
        optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_type == DENOISER_CONST.ADAMW_OPTIM:
        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if (
        DENOISER_CONST.NOISE_TYPE_KEY not in kwargs
        or kwargs[DENOISER_CONST.NOISE_TYPE_KEY] not in DENOISER_CONST.NOISES
    ):
        noise_type = DENOISER_CONST.DEFAULT_NOISE_TYPE
    else:
        noise_type = kwargs[DENOISER_CONST.NOISE_TYPE_KEY]

    if (
        DENOISER_CONST.EPOCHS_KEY not in kwargs
        or type(kwargs[DENOISER_CONST.EPOCHS_KEY]) != int
        or kwargs[DENOISER_CONST.EPOCHS_KEY] <= 0
    ):
        epochs = DENOISER_CONST.DEFAULT_EPOCHS
    else:
        epochs = kwargs[DENOISER_CONST.EPOCHS_KEY]

    return random_seed, device, optim, noise_type, epochs


def add_noise(
    data: torch.Tensor,
    noise_type: str,
    kwargs: dict,
) -> torch.Tensor:
    """
    Add noise to the given data.

    Parameters
    ----------
    data : torch.Tensor
        The data to add noise to.
    noise_type : str
        The type of noise to add.
        Must be one of the following: {DENOISER_CONST.GAUSSIAN_NOISE, DENOISER_CONST.SALT_AND_PEPPER_NOISE}
    kwargs : dict
        Additional parameters for the specified noise type.

        If noise_type is DENOISER_CONST.GAUSSIAN_NOISE then kwargs must contain the following key-value pairs:
            > {noise_params: (effect, bias)} where effect is the effect of the noise and bias is the bias of the noise

        If noise_type is DENOISER_CONST.SALT_AND_PEPPER_NOISE then kwargs must contain the following key-value pairs:
            > {noise_params: (lower_bound, upper_bound)} which are used to generate the noise

    Returns
    -------
    torch.Tensor
        The noisy data.
    """
    if noise_type == DENOISER_CONST.GAUSSIAN_NOISE:
        if DENOISER_CONST.NOISE_PARAMS_KEY not in kwargs:
            noise_params = DENOISER_CONST.DEFAULT_GAUSSIAN_NOISE_PARAMS
        else:
            noise_params = kwargs[DENOISER_CONST.NOISE_PARAMS_KEY]
        effect = noise_params[0]
        bias = noise_params[1]
        noise = torch.randn_like(data) * effect + bias
        return data + noise
    elif noise_type == DENOISER_CONST.SALT_AND_PEPPER_NOISE:
        if DENOISER_CONST.NOISE_PARAMS_KEY not in kwargs:
            noise_params = DENOISER_CONST.DEFAULT_SALT_AND_PEPPER_NOISE_PARAMS
        else:
            noise_params = kwargs[DENOISER_CONST.NOISE_PARAMS_KEY]
        lower_bound = noise_params[0]
        upper_bound = noise_params[1]
        rand_matrix = torch.rand_like(data)
        data = torch.where(rand_matrix < lower_bound, torch.zeros_like(data), data)
        data = torch.where(rand_matrix > upper_bound, torch.ones_like(data), data)
        return data
    else:
        raise ValueError(
            f"noise_type must be one of the following: {DENOISER_CONST.NOISES}"
        )


def train_epoch(
    model: BaseAE,
    train_loader: DataLoader,
    random_seed: int,
    device: torch.device,
    optim: torch.optim.Optimizer,
    noise_type: str,
    epoch: int,
    is_print: bool = True,
    kwargs: dict = {},
) -> Tuple[float, List[float]]:
    """
    Train the model for one epoch.

    Parameters
    ----------
    model : BaseAE
        The model to be trained.
    train_loader : DataLoader
        The DataLoader for the training data.
    random_seed : int
        The random seed to use for training.
    device : torch.device
        The device to use for training.
    optim : torch.optim.Optimizer
        The optimizer to use for training.
    noise_type : str
        The type of noise to add to the data.
    epoch : int
        The current epoch.
    is_print : bool, optional
        If True, print the training loss at each batch. Default is True.
    kwargs : dict
        Additional arguments to pass to add_noise.

    Returns
    -------
    train_loss : float
        The average loss for the epoch.
    train_loss_list : List[float]
        The loss at each batch.
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    model.to(device)
    model.train()

    dataset_len = len(train_loader.dataset)
    res_dataset_len = len(train_loader.dataset)
    batch_num = len(train_loader)

    train_loss = 0
    train_loss_list = []
    for batch_idx, (data, _) in enumerate(train_loader):
        data = add_noise(data, noise_type, kwargs)
        data = data.to(device)
        optim.zero_grad()
        loss = model.training_loss(data)
        loss.backward()
        optim.step()

        res_dataset_len -= len(data)
        if is_print and (batch_idx % 100 == 0 or batch_idx == batch_num - 1):
            print(
                "Denoiser Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    dataset_len - res_dataset_len,
                    dataset_len,
                    100.0 * batch_idx / batch_num,
                    loss.item(),
                )
            )
        train_loss += loss.item()
        train_loss_list.append(loss.item())

    if is_print:
        print(
            "\n====> Epoch: {} Train Loss: {:.6f}\n".format(
                epoch, train_loss / batch_num
            )
        )

    return train_loss / batch_num, train_loss_list


def test(
    model: BaseAE,
    test_loader: DataLoader,
    device: torch.device,
    noise_type: str,
    kwargs: dict = {},
) -> float:
    """
    Evaluate the model on the test dataset.

    Parameters
    ----------
    model : BaseAE
        The model to be evaluated.
    test_loader : DataLoader
        The data loader for the test dataset.
    device : torch.device
        The device to run the model on.
    noise_type : str
        The type of noise to add to the data.
    kwargs :
        Additional parameters for the specified noise type.

    Returns
    -------
    float
        The average testing loss.
    """
    model.to(device)
    model.eval()
    testing_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = add_noise(data, noise_type, kwargs)
            data = data.to(device)
            loss = model.testing_loss(data)
            testing_loss += loss.item()
    return testing_loss / len(test_loader)


def get_x(dataloader: DataLoader) -> np.ndarray:

    x_list = []
    for x, _ in dataloader:
        x: torch.Tensor
        x = x.reshape(x.shape[0], -1)
        x_list.append(x.detach().numpy())
    x = np.concatenate(x_list, axis=0)
    return x


def get_examples(
    model: BaseAE,
    dataloader: DataLoader,
    indexes: List[int],
    noise_type: str,
    kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get examples from the given dataloader with the specified model and noise type.

    Parameters
    ----------
    model : BaseAE
        The autoencoder model to be used for denoising.
    dataloader : DataLoader
        The DataLoader for the data to be denoised.
    indexes : List[int]
        The indexes of the examples to be returned.
    noise_type : str
        The type of noise to add to the data.
    **kwargs :
        Additional parameters for the specified noise type.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the original example, the noised example, and the denoised example.
    """
    model.eval()
    original_examples_list = []
    noised_examples_list = []
    denoised_examples_list = []
    for examples, _ in dataloader:
        original_examples: torch.Tensor = examples
        noised_examples: torch.Tensor = add_noise(examples, noise_type, kwargs)
        denoised_examples: torch.Tensor = model(noised_examples)

        original_examples_list.append(original_examples.detach().numpy())
        noised_examples_list.append(noised_examples.detach().numpy())
        denoised_examples_list.append(denoised_examples.detach().numpy())

    original_examples_list = np.concatenate(original_examples_list, axis=0)
    noised_examples_list = np.concatenate(noised_examples_list, axis=0)
    denoised_examples_list = np.concatenate(denoised_examples_list, axis=0)

    original_examples_list = original_examples_list[indexes, :, :, :]
    noised_examples_list = noised_examples_list[indexes, :, :, :]
    denoised_examples_list = denoised_examples_list[indexes, :]

    image_size = original_examples_list.shape[2]

    original_examples_list = original_examples_list.reshape(
        original_examples_list.shape[0], image_size, image_size
    )
    noised_examples_list = noised_examples_list.reshape(
        noised_examples_list.shape[0], image_size, image_size
    )
    denoised_examples_list = denoised_examples_list.reshape(
        denoised_examples_list.shape[0], image_size, image_size
    )

    return (
        original_examples_list,
        noised_examples_list,
        denoised_examples_list,
    )
