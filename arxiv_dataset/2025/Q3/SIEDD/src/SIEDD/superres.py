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
from pathlib import Path
from datetime import datetime
from pydantic_yaml import parse_yaml_file_as
from itertools import batched
from tqdm import tqdm

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
    shared_frames = strainer.data_pipeline.frame_idx
    strainer.data_pipeline.data_set.cache_frames(shared_frames)
    model = strainer.create_model(group_size, False)
    shared_state = strainer.load_artefacts(strainer.save_path / "shared_encoder.bin")
    shared_state = {
        k.removeprefix("_orig_mod.").removeprefix("encoderINR."): v
        for k, v in shared_state.items()
        if "encoderINR" in k
    }
    model.encoderINR.load_state_dict(shared_state, strict=False)
    model.encoderINR.requires_grad_(False)
    quality: list[metric.QualityMetrics] = []
    for frames in it:
        strainer.data_pipeline.data_set.cache_frames(frames)
        name = "_".join(map(str, frames))
        filename = f"model_{name}.bin"
        fn = strainer.save_path / filename
        decoder_state = strainer.load_artefacts(fn)
        model.load_state_dict(decoder_state, strict=False)

        val_qualities, predictions, fps_for_group = strainer.validate_frame(
            model, save_preds=True
        )

        name = "_".join(map(str, frames))
        save_infos = {
            i: inf.model_dump(exclude_none=True)
            for i, inf in zip(frames, val_qualities)
        }
        filepath = strainer.save_path / f"frame_superres_info_{name}.json"
        helpers.save_json(save_infos, filepath)
        for info, idx in zip(val_qualities, frames):
            info_dict = info.model_dump(exclude_none=True)
            if WANDB_ENABLED:
                info_dict = {
                    f"frame/metrics/{k}": v
                    for k, v in metric.flatten(info_dict).items()
                }
                info_dict["Frame"] = idx
                wandb.log(info_dict)
        quality += val_qualities
    qual = metric.reduce_quality_metrics(quality)
    return qual


def run(strainer: Strainer):
    result = run_frames(strainer)

    info = result.model_dump(exclude_none=True)
    helpers.save_json(info, strainer.save_path / "cumulative_superres.json")
    if WANDB_ENABLED and wandb.run is not None:
        info = metric.flatten(info)
        for k, v in info.items():
            wandb.run.summary[f"metrics/{k}"] = v
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
