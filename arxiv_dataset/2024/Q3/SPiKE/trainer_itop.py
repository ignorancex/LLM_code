"""
Module for training and evaluating the SPiKE model on the ITOP dataset.
"""

from tqdm import tqdm
import torch
from torch import nn
import numpy as np
from datasets.itop import ITOP
from utils import metrics, scheduler


def train_one_epoch(
    model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, threshold
):

    model.train()
    header = f"Epoch: [{epoch}]"
    total_loss = 0.0
    total_pck = np.zeros(15)
    total_map = 0.0

    for clip, target, _ in tqdm(data_loader, desc=header):
        clip, target = clip.to(device), target.to(device)
        output = model(clip).reshape(target.shape)
        loss = criterion(output, target)

        pck, mean_ap = metrics.joint_accuracy(output, target, threshold)
        total_pck += pck.detach().cpu().numpy()
        total_map += mean_ap.detach().cpu().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        lr_scheduler.step()

    total_loss /= len(data_loader)
    total_map /= len(data_loader)
    total_pck /= len(data_loader)

    return total_loss, total_pck, total_map


def evaluate(model, criterion, data_loader, device, threshold):
    model.eval()
    total_loss = 0.0
    total_pck = np.zeros(15)
    total_map = 0.0

    with torch.no_grad():
        for clip, target, _ in tqdm(
            data_loader, desc="Validation" if data_loader.dataset.train else "Test"
        ):
            clip, target = clip.to(device, non_blocking=True), target.to(
                device, non_blocking=True
            )
            output = model(clip).reshape(target.shape)
            loss = criterion(output, target)

            pck, mean_ap = metrics.joint_accuracy(output, target, threshold)
            total_pck += pck.detach().cpu().numpy()
            total_map += mean_ap.detach().cpu().item()
            total_loss += loss.item()

    total_loss /= len(data_loader)
    total_map /= len(data_loader)
    total_pck /= len(data_loader)

    return total_loss, total_pck, total_map


def load_data(config, mode="train"):
    """
    Load the ITOP dataset.

    Args:
        config (dict): The configuration dictionary.
        mode (str): The mode to load the data in ("train" or "test").

    Returns:
        tuple: A tuple containing the data loader(s) and the number of coordinate joints.
    """
    dataset_params = {
        "root": config["dataset_path"],
        "frames_per_clip": config["frames_per_clip"],
        "num_points": config["num_points"],
        "use_valid_only": config["use_valid_only"],
        "target_frame": config["target_frame"],
    }

    if mode == "train":
        dataset = ITOP(
            train=True, aug_list=config["PREPROCESS_AUGMENT_TRAIN"], **dataset_params
        )
        dataset_test = ITOP(
            train=False, aug_list=config["PREPROCESS_TEST"], **dataset_params
        )
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["workers"],
        )
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=config["batch_size"], num_workers=config["workers"]
        )
        return data_loader, data_loader_test, dataset.num_coord_joints

    dataset_test = ITOP(
        train=False, aug_list=config["PREPROCESS_TEST"], **dataset_params
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=config["batch_size"], num_workers=config["workers"]
    )
    return data_loader_test, dataset_test.num_coord_joints


def create_criterion(config):
    """
    Create the loss criterion based on the configuration.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        torch.nn.Module: The loss function.
    """
    loss_type = config.get("loss_type", "std_cross_entropy")

    if loss_type == "l1":
        return nn.L1Loss()
    if loss_type == "mse":
        return nn.MSELoss()
    raise ValueError("Invalid loss type. Supported types: 'l1', 'mse'.")


def create_optimizer_and_scheduler(config, model, data_loader):
    lr = config["lr"]
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )
    warmup_iters = config["lr_warmup_epochs"] * len(data_loader)
    lr_milestones = [len(data_loader) * m for m in config["lr_milestones"]]
    lr_scheduler = scheduler.WarmupMultiStepLR(
        optimizer,
        milestones=lr_milestones,
        gamma=config["lr_gamma"],
        warmup_iters=warmup_iters,
        warmup_factor=1e-5,
    )
    return optimizer, lr_scheduler
