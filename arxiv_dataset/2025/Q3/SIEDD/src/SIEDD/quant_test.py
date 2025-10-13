import os
import wandb
import argparse
import socket
from SIEDD.strainer import Strainer
from SIEDD.configs import (
    RunConfig,
    TrainerConfig,
    StrainerConfig,
    QuantizationConfig,
    QuantMethod,
)
from SIEDD.data_processing import DataProcess
from SIEDD.utils import helpers, Quantize, metric
from SIEDD.decode import replace
from pathlib import Path
import torch
import tempfile
from datetime import datetime
import copy
from pydantic_yaml import parse_yaml_file_as
from itertools import batched
import csv  # <-- NEW
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).absolute().parent.parent.parent


sweep: list[tuple[int, float | None, tuple, list[str], QuantMethod, bool]] = [
    (8, None, (-1,), [], "post", False),  # baseline
    (8, None, (-1,), [], "post", True),
    (7, None, (-1,), [], "post", True),
    (6, None, (-1,), [], "post", True),
    (5, None, (-1,), [], "post", True),
    (4, None, (-1,), [], "post", True),
    (8, None, (-1,), ["3", "bias"], "post", True),
    (7, None, (-1,), ["3", "bias"], "post", True),
    (6, None, (-1,), ["3", "bias"], "post", True),
    (5, None, (-1,), ["3", "bias"], "post", True),
    (4, None, (-1,), ["3", "bias"], "post", True),
    # (8, None, (-1,), ["3", "bias"], "quanto", True),
    # (8, None, (-1,), [], "bnb", True),
    # (4, None, (-1,), [], "bnb", True),
    (8, None, (-1,), [], "hqq", True),
    (6, None, (-1,), [], "hqq", True),
    (4, None, (-1,), [], "hqq", True),
    (5, None, (-1,), [], "hqq", True),
]


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


def run_quant(
    strainer: Strainer, quant_cfg: QuantizationConfig, tmp_path: Path
) -> metric.Metrics:
    strainer_pth = strainer.save_path
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
    strainer.quantizer = Quantize(quant_cfg, strainer.enc_cfg)
    model.encoderINR.load_state_dict(shared_state, strict=False)
    model.encoderINR.requires_grad_(False)
    quality: list[metric.QualityMetrics] = []
    fps: list[float] = []
    compression: list[metric.CompressionMetrics] = []
    for frames in it:
        strainer.data_pipeline.data_set.cache_frames(frames)
        name = "_".join(map(str, frames))
        filename = f"model_{name}.bin"
        fn = strainer.save_path / filename
        decoder_state = strainer.load_artefacts(fn)
        model.load_state_dict(decoder_state, strict=False)
        quant_model = copy.deepcopy(model)
        trainable_params = helpers.trainable_state_dict(quant_model)
        num_params = sum([x.numel() for x in trainable_params.values()])
        if quant_cfg.quantize:
            _, compressed_state = strainer.quantizer.quantize_model(
                quant_model, trainable_params, strainer.coordinates
            )
        else:
            compressed_state = decoder_state
        strainer.save_path = Path(tmp_path)
        strainer.save_artefacts(compressed_state, frames)
        # read bytes from compressed_state and update metric.
        quant_val_comps = strainer.get_bpp(frames, num_params)
        replace(quant_model)

        if strainer.quantizer.method != "quanto":
            quant_model = quant_model.to(torch.bfloat16)

        quant_val_qualities, predictions, fps_for_group = strainer.validate_frame(
            quant_model, save_preds=True
        )
        strainer.save_path = strainer_pth
        quality += quant_val_qualities
        fps += [fps_for_group] * len(frames)
        compression += quant_val_comps
    qual = metric.reduce_quality_metrics(quality)
    comp = metric.reduce_compression_metrics(compression)
    fps_num = sum(fps) / len(fps)
    metrics = metric.Metrics(
        metrics=qual, compression_metrics=comp, time=0.0, fps=fps_num
    )
    return metrics


def run_sweep(strainer: Strainer, tmp_path):
    columns = [
        "quantize",
        "method",
        "bits",
        "sparsity",
        "axis",
        "psnr",
        "ssim",
        "bpp",
        "fps",
        "ignore",
    ]
    wandb_enabled = (
        os.getenv("WANDB_PROJECT") is not None and os.getenv("WANDB_GROUP") is not None
    )
    rows = []  # <-- NEW: collect rows for CSV
    for bits, sparsity, axis, ignore, method, quantize in sweep:
        qcfg = QuantizationConfig(
            quant_bit=bits,
            quantize=quantize,
            quant_axis=axis,
            sparsity=sparsity,
            keys_to_ignore=ignore,
            quant_method=method,
        )
        result = run_quant(strainer, qcfg, tmp_path)
        print(
            dict(
                quantize=quantize,
                method=method,
                bits=bits,
                sparsity=sparsity,
                axis=str(axis),
                psnr=result.metrics.psnr,
                ssim=result.metrics.ssim,
                bpp=result.compression_metrics.bpp,
                fps=result.fps,
                ignore=str(ignore),
            )
        )

        rows.append(
            [
                quantize,
                method,
                bits,
                sparsity,
                str(axis),
                result.metrics.psnr,
                result.metrics.ssim,
                result.compression_metrics.bpp,
                result.fps,
                str(ignore),
            ]
        )

    if wandb_enabled:
        table = wandb.Table(columns=columns, data=rows)
        wandb.log({"quant_table": table})
    if not wandb_enabled:
        # dump CSV if wandb is off
        method = strainer.cfg.quant_cfg.quant_method
        csv_path = strainer.save_path / f"quant_results_{method}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(rows)
        print(f"Saved results to {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path)
    parser.add_argument("--save_path", type=Path)
    parser.add_argument("--name", type=str)
    args = parser.parse_args()
    wandb_enabled = (
        os.getenv("WANDB_PROJECT") is not None and os.getenv("WANDB_GROUP") is not None
    )
    wandb_project: str = str(os.getenv("WANDB_PROJECT"))
    wandb_group: str = str(os.getenv("WANDB_GROUP"))
    save_path = args.save_path
    cfg: RunConfig = parse_yaml_file_as(RunConfig, save_path / "run_cfg.yaml")
    if cfg.quant_cfg.quantize is True:
        raise ValueError("Run should not be quantized")
    data_path: Path = args.data_path
    name: str = args.name
    fname: str = name + "_" if name else ""
    id = f"{fname}{datetime.strftime(datetime.now(), '%d_%m_%y_%H_%M_%S')}"

    if wandb_enabled:
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
        with tempfile.TemporaryDirectory() as f:
            run_sweep(strainer, Path(f))


if __name__ == "__main__":
    main()
