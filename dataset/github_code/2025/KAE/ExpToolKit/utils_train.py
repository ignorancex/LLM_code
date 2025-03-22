from typing import Tuple, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import random
import time

from .const import *
from .models import BaseAE

from ClassifierPack import Classifier
from RetrieverPack import Retriever
from DenoiserPack import Denoiser


def set_seed(seed: int) -> None:
    """
    Set seed for torch, torch.cuda, numpy and random.

    :param seed: the seed number
    :return: None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def dataset_loader(
    dataset_name: str,
    batch_size: int = 128,
    path: str = DATASET_CONST.DATASET_DIR,
    preprocessing: str = DATASET_CONST.PREPROCESSING_ORIGINAL,
) -> Tuple[DataLoader, DataLoader]:
    """
    Load a dataset and create a data loader for it.

    Args:
        dataset_name (str): The name of the dataset.
        batch_size (int, optional): The batch size. Defaults to 128.
        path (str, optional): The path to the dataset. Defaults to DATASET_CONST.DATASET_DIR.
        preprocessing (str, optional): The preprocessing method. Defaults to DATASET_CONST.PREPROCESSING_ORIGINAL.

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple of two data loaders, the first one is the training data loader,
            and the second one is the test data loader.
    """
    transform_method = []
    transform_method.append(transforms.ToTensor())
    if dataset_name in DATASET_CONST.DATASET_RGB_TYPES:
        transform_method.append(transforms.Grayscale(num_output_channels=1))
    if preprocessing != DATASET_CONST.PREPROCESSING_ORIGINAL:
        if preprocessing == DATASET_CONST.PREPROCESSING_INVERSE:
            transform_method.append(transforms.Lambda(lambda x: 1 - x))
        if preprocessing == DATASET_CONST.PREPROCESSING_STANDARDIZE:
            transform_method.append(transforms.Normalize(mean=[0.5], std=[0.5]))

    transform = transforms.Compose(transform_method)

    if dataset_name == DATASET_CONST.DATASET_MNIST:
        train_dataset = datasets.MNIST(
            path, train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            path, train=False, download=True, transform=transform
        )
    elif dataset_name == DATASET_CONST.DATASET_FASHION_MNIST:
        train_dataset = datasets.FashionMNIST(
            path, train=True, download=True, transform=transform
        )
        test_dataset = datasets.FashionMNIST(
            path, train=False, download=True, transform=transform
        )
    elif dataset_name == DATASET_CONST.DATASET_CIFAR10:
        train_dataset = datasets.CIFAR10(
            path, train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR10(
            path, train=False, download=True, transform=transform
        )
    elif dataset_name == DATASET_CONST.DATASET_CIFAR100:
        train_dataset = datasets.CIFAR100(
            path, train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR100(
            path, train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"NOT IMPLEMENTED DATASET: {dataset_name}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader


def train(
    model: BaseAE,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    epoch: int,
    device: torch.device,
    random_seed: int,
    is_print: bool = True,
) -> float:
    """
    Train the model for one epoch.

    Parameters
    ----------
    model: BaseAE
        The model to be trained.
    train_loader: DataLoader
        The DataLoader for the training data.
    optimizer: optim.Optimizer
        The optimizer to use for training.
    epoch: int
        The current epoch.
    device: torch.device
        The device to use for training.
    random_seed: int
        The random seed to use for training.
    is_print: bool, optional
        If True, print the training loss at each batch. Default is True.

    Returns
    -------
    train_loss: float
        The average loss for the epoch.
    train_loss_list: List[float]
        The loss at each batch.
    """
    set_seed(random_seed)
    model.to(device)
    model.train()

    dataset_len = len(train_loader.dataset)
    res_dataset_len = len(train_loader.dataset)
    batch_num = len(train_loader)

    train_loss = 0
    train_loss_list = []
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        loss = model.training_loss(data)
        loss.backward()
        optimizer.step()

        res_dataset_len -= len(data)
        if is_print and (batch_idx % 100 == 0 or batch_idx == batch_num - 1):
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    dataset_len - res_dataset_len,
                    dataset_len,
                    100.0 * batch_idx / batch_num,
                    loss.item(),
                )
            )
        train_loss += loss.item()
        train_loss_list.append(loss.item())

    return train_loss / batch_num, train_loss_list


def test(model: BaseAE, test_loader: DataLoader, device: torch.device) -> float:
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
            data = data.to(device)
            loss = model.testing_loss(data)
            testing_loss += loss.item()
    return testing_loss / len(test_loader)


def encode(model: BaseAE, test_loader: DataLoader) -> np.ndarray:
    """
    Encode the data in the test_loader using the given model.

    Parameters
    ----------
    model : BaseAE
        The model to be used for encoding.
    test_loader : DataLoader
        The data loader for the test dataset.

    Returns
    -------
    dict
        A dictionary with the encoded data and labels.
    """
    model.eval()

    encoded_data = []
    labels = []

    with torch.no_grad():
        for data, label in test_loader:
            encoded = model.encode(data)
            encoded_data.append(encoded)
            labels.append(label)

    encoded_data = torch.cat(encoded_data, dim=0)
    labels = torch.cat(labels, dim=0)

    data_dict = {"encoded_data": encoded_data, "labels": labels}

    return data_dict


def train_and_test(
    model: BaseAE,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: optim.Optimizer,
    epochs: int,
    device: torch.device,
    random_seed: int,
    is_print: bool = True,
) -> Tuple[BaseAE, List[float], List[float], List[float], List[float]]:
    """
    Train the model and test it on the test dataset.

    Parameters
    ----------
    model : BaseAE
        The model to be trained.
    train_loader : DataLoader
        The data loader for the training dataset.
    test_loader : DataLoader
        The data loader for the test dataset.
    optimizer : optim.Optimizer
        The optimizer to be used for training.
    epochs : int
        The number of epochs for training.
    device : torch.device
        The device to be used for training.
    random_seed : int
        The random seed to be used for reproducibility.
    is_print : bool, optional
        Whether to print the training and testing loss and time. Defaults to True.

    Returns
    -------
    Tuple[BaseAE, List[float], List[float], List[float], List[float]]
        A tuple containing the trained model, the training loss per epoch, the training loss per batch, the training time per epoch, and the testing loss per epoch.
    """
    train_loss_epoch_list = []
    train_loss_batch_list = []
    test_loss_epoch_list = []
    train_time_epoch_list = []
    train_time_epoch = 0
    for epoch in range(1, epochs + 1):
        train_time_begin = time.time()
        train_loss_epoch, train_loss_batch = train(
            model, train_loader, optimizer, epoch, device, random_seed, is_print
        )
        train_time_epoch += time.time() - train_time_begin
        train_time_epoch_list.append(train_time_epoch)
        test_loss_epoch = test(model, test_loader, device)
        if is_print:
            print(
                "\n====> Epoch: {} Train Loss: {:.6f} Test Loss: {:.6f} Train Time: {:.0f} sec\n".format(
                    epoch, train_loss_epoch, test_loss_epoch, train_time_epoch
                )
            )
        train_loss_epoch_list.append(train_loss_epoch)
        train_loss_batch_list += train_loss_batch
        test_loss_epoch_list.append(test_loss_epoch)

    return (
        model,
        train_loss_epoch_list,
        train_loss_batch_list,
        train_time_epoch_list,
        test_loss_epoch_list,
    )


def evaluate_classifier(
    classifier: Classifier,
    train_loader: DataLoader,
    test_loader: DataLoader,
) -> Tuple[float, float]:
    """
    Evaluate the classifier on the training and test dataset.

    Parameters
    ----------
    classifier : Classifier
        The classifier to be evaluated.
    train_loader : DataLoader
        The data loader for the training dataset.
    test_loader : DataLoader
        The data loader for the test dataset.

    Returns
    -------
    Tuple[float, float]
        A tuple containing the accuracy on the training dataset and the accuracy on the test dataset.
    """
    train_accuracy = classifier.fit(train_loader)
    test_accuracy = classifier.predict(test_loader)
    return train_accuracy, test_accuracy


def evaluate_retriever(
    retriever: Retriever,
    train_loader: DataLoader,
    test_loader: DataLoader,
    top_K: int,
    retrieval_N: int,
    label_num: float = 10,
) -> Tuple[float, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Evaluate the retriever on the training and test dataset.

    Parameters
    ----------
    retriever : Retriever
        The retriever to be evaluated.
    train_loader : DataLoader
        The data loader for the training dataset.
    test_loader : DataLoader
        The data loader for the test dataset.
    top_K : int
        The number of nearest neighbors to be retrieved.
    retrieval_N : int
        The number of nearest neighbors in the latent space to be retrieved.
    label_num : int
        The number of labels to be considered.

    Returns
    -------
    Tuple[float, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]
        A tuple containing the average recall of the retriever on the training dataset, the distance matrix of the data points in the original space of the training dataset, the distance matrix of the data points in the latent space of the training dataset, the average recall of the retriever on the test dataset, the distance matrix of the data points in the original space of the test dataset, the distance matrix of the data points in the latent space of the test dataset.
    """
    train_recall, train_distance_matrix_x, train_distance_matrix_x_latent = (
        retriever.evaluate(train_loader, top_K, retrieval_N, label_num)
    )
    test_recall, test_distance_matrix_x, test_distance_matrix_x_latent = (
        retriever.evaluate(test_loader, top_K, retrieval_N, label_num)
    )
    return (
        train_recall,
        train_distance_matrix_x,
        train_distance_matrix_x_latent,
        test_recall,
        test_distance_matrix_x,
        test_distance_matrix_x_latent,
    )


def evaluate_denoiser(
    denoiser: Denoiser,
    train_loader: DataLoader,
    test_loader: DataLoader,
    is_train: bool = True,
    is_print: bool = True,
    **kwargs,
) -> Tuple[float, float]:
    """
    Evaluate the denoiser.

    Parameters
    ----------
    denoiser : Denoiser
        The denoiser to be evaluated.
    train_loader : DataLoader
        The data loader for the training dataset.
    test_loader : DataLoader
        The data loader for the test dataset.
    is_print : bool
        Whether to print the training process.
    **kwargs :
        Additional parameters for the specified denoiser.

    Returns
    -------
    Tuple[float, float]
        A tuple containing the average loss of the denoiser on the training dataset and the average loss of the denoiser on the test dataset.
    """
    if is_train:
        denoiser.fit(train_loader, is_print, **kwargs)
        denoiser.predict(test_loader, **kwargs)
    train_loss = denoiser.evaluate(train_loader, **kwargs)
    test_loss = denoiser.evaluate(test_loader, **kwargs)
    return train_loss, test_loss
