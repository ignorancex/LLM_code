"""
Module for evaluating the SPiKE model on the ITOP dataset.
"""

from __future__ import print_function
import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from model import model_builder
from trainer_itop import load_data, create_criterion
from utils.config_utils import load_config, set_random_seed
from utils.metrics import joint_accuracy


def evaluate(model, criterion, data_loader, device, threshold):

    model.eval()
    total_loss = 0.0
    total_pck = np.zeros(15)
    total_map = 0.0
    clip_losses = []

    with torch.no_grad():
        for batch_clips, batch_targets, batch_video_ids in tqdm(
            data_loader, desc="Validation" if data_loader.dataset.train else "Test"
        ):
            for clip, target, video_id in zip(
                batch_clips, batch_targets, batch_video_ids
            ):
                clip = clip.unsqueeze(0).to(device, non_blocking=True)
                target = target.unsqueeze(0).to(device, non_blocking=True)

                output = model(clip).reshape(target.shape)
                loss = criterion(output, target)

                pck, mean_ap = joint_accuracy(output, target, threshold)
                total_pck += pck.detach().cpu().numpy()
                total_map += mean_ap.detach().cpu().item()

                total_loss += loss.item()
                clip_losses.append(
                    (
                        video_id.cpu().detach().numpy(),
                        loss.item(),
                        clip.cpu().detach().numpy(),
                        target.cpu().detach().numpy(),
                        output.cpu().detach().numpy(),
                    )
                )

        total_loss /= len(data_loader.dataset)
        total_map /= len(data_loader.dataset)
        total_pck /= len(data_loader.dataset)

    return clip_losses, total_loss, total_map, total_pck


def main(arguments):

    config = load_config(arguments.config)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["device_args"])
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    device = torch.device(0)

    set_random_seed(config["seed"])

    print(f"Loading data from {config['dataset_path']}")
    data_loader_test, num_coord_joints = load_data(config, mode="test")

    model = model_builder.create_model(config, num_coord_joints)
    model.to(device)

    criterion = create_criterion(config)

    print(f"Loading model from {arguments.model}")
    checkpoint = torch.load(arguments.model, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)

    losses, val_clip_loss, val_map, val_pck = evaluate(
        model, criterion, data_loader_test, device=device, threshold=config["threshold"]
    )
    losses.sort(key=lambda x: x[1], reverse=True)

    print(f"Validation Loss: {val_clip_loss:.4f}")
    print(f"Validation mAP: {val_map:.4f}")
    print(f"Validation PCK: {val_pck}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPiKE Testing on ITOP dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="ITOP-SIDE/1",
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="experiments/ITOP-SIDE/1/log/best_model.pth",
        help="Path to the model checkpoint",
    )

    args = parser.parse_args()
    main(args)
