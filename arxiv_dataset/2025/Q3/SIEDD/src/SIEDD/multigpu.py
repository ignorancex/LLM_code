import numpy as np
import os
import wandb
import argparse
import socket
from SIEDD.strainer import (
    Strainer,
    Quantize,
    metric,
    StrainerNet,
    batched,
    schedulefree,
    helpers,
)
from SIEDD.configs import RunConfig, StrainerConfig
from SIEDD.data_processing import DataProcess
from pathlib import Path
from datetime import datetime
from pydantic_yaml import parse_yaml_file_as
import torch.multiprocessing as mp
import torch

torch.cuda.memory._set_allocator_settings("expandable_segments:False")

PROJECT_ROOT = Path(__file__).absolute().parent.parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=Path)
parser.add_argument("--data_path", type=Path)
parser.add_argument("--save_path", type=Path)
parser.add_argument("--name", type=str)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()


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


def shared_training(strainer: Strainer):
    shared_frames = strainer.data_pipeline.frame_idx
    print(
        f"STRAINER: Using frames {shared_frames}",
        "for shared encoder training",
    )

    strainer.data_pipeline.data_set.cache_frames(shared_frames)
    shared_model, shared_optim, shared_scaler, shared_scheduler, shared_loss_fn = (
        strainer.setup_frame(shared_frames, False)
    )
    print(shared_model)
    if strainer.train_cfg.shared_encoder_path is not None:
        print(
            "Loading shared encoder state from", strainer.train_cfg.shared_encoder_path
        )
        shared_training_time = 0.0
        best_shared_state = strainer.load_artefacts(
            strainer.train_cfg.shared_encoder_path
        )
        best_encoder_state = {
            k.removeprefix("_orig_mod.").removeprefix("encoderINR."): v
            for k, v in best_shared_state.items()
            if "encoderINR" in k
        }
        strainer.quantizer = Quantize(strainer.cfg.quant_cfg, strainer.enc_cfg)
        shared_model.encoderINR.load_state_dict(best_encoder_state, strict=False)
        shared_model.encoderINR.requires_grad_(False)
    else:
        best_shared_state, shared_training_time = strainer.train_loop(
            shared_model,
            shared_optim,
            shared_scaler,
            shared_scheduler,
            shared_loss_fn,
            shared_frames,
            iters=strainer.train_cfg.shared_iters,
            shared=True,
        )

        print("Shared Encoder Training Time:", shared_training_time)
        shared_model.load_state_dict(best_shared_state)
        best_shared_state = {
            k: v for k, v in best_shared_state.items() if "encoderINR" in k
        }
        strainer.save_artefacts(best_shared_state, shared_frames, shared=True)
    del shared_optim, shared_scaler, shared_scheduler, shared_loss_fn
    strainer.data_pipeline.data_set.uncache_frames()
    return shared_model, shared_training_time


def custom_strainer_train(gpu_id, num_gpus, strainer: Strainer, shared_model):
    shared_frames = strainer.data_pipeline.frame_idx
    print("Total Frames: ", strainer.data_pipeline.num_frames)
    print("Data shape: ", strainer.data_pipeline.data_shape)
    with torch.cuda.device(f"cuda:{gpu_id}"):
        strainer.coordinates = strainer.coordinates.to(f"cuda:{gpu_id}")  # type: ignore
        strainer.coords_device = torch.device(f"cuda:{gpu_id}")

        metrics_per_frame: list[list[metric.Metrics]] = []
        prev_model: StrainerNet = shared_model
        group_size = strainer.train_cfg.group_size
        it = list(
            map(
                list,
                batched(range(strainer.data_pipeline.num_frames), group_size),
            )
        )
        it = np.array_split(it, num_gpus)[gpu_id].tolist()
        first = True
        prev_frame_idx = shared_frames
        for frame_idx in it:
            print(f"Training frame(s) {frame_idx}")
            strainer.data_pipeline.data_set.cache_frames(frame_idx)
            model, optim, scaler, scheduler, loss_fn = strainer.setup_frame(
                frame_idx, True, prev_model, prev_frame_idx
            )
            if first:
                print(model)
                first = False
            best_model_state, training_time = strainer.train_loop(
                model,
                optim,
                scaler,
                scheduler,
                loss_fn,
                frame_idx,
                iters=strainer.train_cfg.iters,
                shared=False,
            )
            if isinstance(optim, schedulefree.AdamWScheduleFree):
                optim.eval()
            infos = strainer.calc_final_metrics_and_save(
                model, best_model_state, frame_idx, encoding_time=training_time
            )
            metrics_per_frame.append(infos)
            strainer.report_final_metrics(frame_idx, infos)
            strainer.data_pipeline.data_set.save_state(strainer.save_path)
            strainer.data_pipeline.data_set.uncache_frames()
        # Reduce the metrics (mean, sum) then report the cumulative over all frames
        all_metrics = [met for lst in metrics_per_frame for met in lst]
        reduced_metrics = metric.reduce_metrics(all_metrics)
        info = reduced_metrics.model_dump(exclude_none=True)
        helpers.save_json(info, Path(f"multigpu_{gpu_id}_{num_gpus}.json"))


def main() -> None:
    wandb_enabled = (
        os.getenv("WANDB_PROJECT") is not None and os.getenv("WANDB_GROUP") is not None
    )
    wandb_project: str = str(os.getenv("WANDB_PROJECT"))
    wandb_group: str = str(os.getenv("WANDB_GROUP"))
    cfg: RunConfig = parse_yaml_file_as(RunConfig, args.cfg)
    data_path: Path = args.data_path
    name: str = args.name
    fname: str = name + "_" if name else ""
    resume: bool = args.resume
    sp: Path | None = args.save_path
    id = f"{fname}{datetime.strftime(datetime.now(), '%d_%m_%y_%H_%M_%S')}"
    resumeid = None
    if sp is None:
        save_path = PROJECT_ROOT / "outputs" / "runs" / id
    else:
        save_path = sp
        if resume and save_path.exists():
            resumeid = save_path.name  # wandb id of original training run
        elif resume:
            raise ValueError("Save path must contain a training state to resume")

    if wandb_enabled:
        os.makedirs(save_path, exist_ok=True)
        mode = "online" if internet() else "offline"
        wandb.init(
            project=wandb_project,
            group=wandb_group,
            dir=str(save_path),
            config=cfg.model_dump(),
            resume=resume,
            id=resumeid if resume else id,
            name=id,
            mode=mode,
        )
    if isinstance(cfg.trainer_cfg, StrainerConfig):
        pipeline = DataProcess(cfg, data_path, True)
        strainer = Strainer(
            cfg,
            data_pipeline=pipeline,
            save_path=save_path,
            resume=resume,
        )
    else:
        raise ValueError()
    shared_model, shared_training_time = shared_training(strainer)

    mp.set_start_method("forkserver", force=True)
    num_gpus = torch.cuda.device_count()
    processes: list[mp.Process] = []
    for i in range(num_gpus):
        p = mp.Process(
            target=custom_strainer_train, args=(i, num_gpus, strainer, shared_model)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
