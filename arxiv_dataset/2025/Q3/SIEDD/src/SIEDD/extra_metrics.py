import os
import wandb
import argparse
import socket
from SIEDD.strainer import Strainer
from SIEDD.configs import (
    RunConfig,
    TrainerConfig,
    StrainerConfig,
)
from SIEDD.data_processing import DataProcess
from SIEDD.utils import helpers, metric
from SIEDD.decode import decode_image
from pathlib import Path
from datetime import datetime
from pydantic_yaml import parse_yaml_file_as
from itertools import batched
from tqdm import tqdm
import torch
from vmaf_torch import VMAF
import numpy as np

PROJECT_ROOT = Path(__file__).absolute().parent.parent.parent
WANDB_ENABLED = (
    os.getenv("WANDB_PROJECT") is not None and os.getenv("WANDB_GROUP") is not None
)


def internet(host="8.8.8.8", port=53, timeout=3):
    """
    Host: 8.8.8.8 (google-public-dns-a.google.com)
    OpenPort: 53/tcp
    Service: domain (DNS/TCP)
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        print("No Internet Connection Available")
        return False


def rgb_to_yuv_y_channel(image: torch.Tensor) -> torch.Tensor:
    if image.shape[1] != 3:
        raise ValueError(
            "Input tensor must have 3 channels (RGB) in the second dimension."
        )

    # Conversion matrix for RGB to YUV (ITU-R BT.601 standard)
    rgb_to_yuv_matrix = torch.tensor(
        [
            [0.299, -0.14713, 0.615],
            [0.587, -0.28886, -0.51499],
            [0.114, 0.436, -0.10001],
        ],
        dtype=image.dtype,
        device=image.device,
    )

    # Extract the Y channel
    y_channel = torch.tensordot(
        image.permute(0, 2, 3, 1), rgb_to_yuv_matrix[:, 0], dims=1
    )

    # Reshape to N,1,H,W
    return y_channel.unsqueeze(1)


def run_frames(strainer: Strainer):
    num_frames = strainer.data_pipeline.num_frames
    group_size = strainer.train_cfg.group_size
    it = tqdm(
        list(
            map(
                list,
                batched(range(num_frames), group_size),
            )
        )
    )
    flips = []
    prev_chunk_refs = None
    prev_chunk_preds = None
    last_val = None
    vmafs = []
    vmaf = VMAF().to("cuda")
    vmaf.compile()
    for frames in it:
        # Get model prediction
        C, H, W = strainer.data_pipeline.data_set.input_data_shape
        predictions, _ = decode_image(strainer, frames[0], (W, H))

        # Get true frames
        strainer.data_pipeline.data_set.cache_frames(frames)
        orig = helpers.process_predictions(
            strainer.data_pipeline.data_set.original_images,
            strainer.enc_cfg,
            input_data_shape=strainer.data_pipeline.data_shape,
        )

        # Calculate FLIP
        _, flip = metric.FLIP(orig.permute(0, 2, 3, 1), predictions.permute(0, 2, 3, 1))
        flips.extend(flip)

        # Calculate VMAF in a batched way (VMAF depends on previous and next frame)
        predictions = rgb_to_yuv_y_channel(predictions)
        orig = rgb_to_yuv_y_channel(orig)
        if prev_chunk_refs is not None and prev_chunk_preds is not None:
            orig = torch.cat((prev_chunk_refs, orig), dim=0)
            predictions = torch.cat((prev_chunk_preds, predictions), dim=0)
        vmaf_chunk = vmaf(orig * 255, (predictions * 255).clamp(0, 255))
        if prev_chunk_refs is not None and prev_chunk_preds is not None:
            last_val = vmaf_chunk[-1]
            vmaf_chunk = vmaf_chunk[:-1]
        else:
            vmaf_chunk = vmaf_chunk[:-1]
        vmafs.extend(vmaf_chunk)
        prev_chunk_refs = orig[-1:]
        prev_chunk_preds = predictions[-1:]
        strainer.data_pipeline.data_set.uncache_frames()

        # Report Metrics except VMAF per frame
        name = "_".join(map(str, frames))
        save_infos = {i: {"FLIP": flip_val} for i, flip_val in zip(frames, flips)}
        filepath = strainer.save_path / f"extra_metrics_{name}.json"
        helpers.save_json(save_infos, filepath)
        for idx in frames:
            info_dict = save_infos[idx]
            if WANDB_ENABLED:
                info_dict = {f"frame/metrics/{k}": v for k, v in info_dict.items()}
                info_dict["Frame"] = idx
                wandb.log(info_dict)
    vmafs.append(last_val)
    all_vmafs = torch.concat(vmafs).reshape(-1).tolist()
    # Now report VMAF per frame
    for i, vmaf_val in enumerate(all_vmafs):
        if WANDB_ENABLED:
            info_dict = {
                "frame/metrics/VMAF": vmaf_val,
            }
            info_dict["Frame"] = i
            wandb.log(info_dict)
    mean_flip = np.mean(flips).item()
    mean_vmaf = np.mean(all_vmafs).item()
    helpers.save_json(
        {"VMAF": mean_vmaf, "FLIP": mean_flip},
        strainer.save_path / "cumulative_extra_metrics.json",
    )


def run(strainer: Strainer):
    if WANDB_ENABLED:
        wandb.define_metric("frame/metrics/VMAF", step_metric="Frame", summary="mean")
        wandb.define_metric("frame/metrics/FLIP", step_metric="Frame", summary="mean")
    run_frames(strainer)
    if WANDB_ENABLED:
        wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path)
    parser.add_argument("--save_path", type=Path)
    parser.add_argument("--name", type=str)
    args = parser.parse_args()
    wandb_project: str = str(os.getenv("WANDB_PROJECT"))
    wandb_group: str = str(os.getenv("WANDB_GROUP"))
    save_path = args.save_path
    cfg: RunConfig = parse_yaml_file_as(RunConfig, save_path / "run_cfg.yaml")
    data_path: Path = args.data_path
    name: str = args.name
    fname: str = name + "_" if name else ""
    id = f"{fname}{datetime.strftime(datetime.now(), '%d_%m_%y_%H_%M_%S')}"

    if WANDB_ENABLED:
        os.makedirs(save_path, exist_ok=True)
        mode = "online" if internet() else "offline"
        wandb.init(
            project=wandb_project,
            group=wandb_group,
            dir=str(save_path),
            config=cfg.model_dump(),
            name=id,
            mode=mode,
        )
    if isinstance(cfg.trainer_cfg, TrainerConfig):
        raise ValueError("Not Implemented")
    elif isinstance(cfg.trainer_cfg, StrainerConfig):
        pipeline = DataProcess(cfg, data_path, True)
        strainer = Strainer(
            cfg,
            data_pipeline=pipeline,
            save_path=save_path,
            resume=False,
            setup_wandb=False,
        )
        run(strainer)


if __name__ == "__main__":
    main()
