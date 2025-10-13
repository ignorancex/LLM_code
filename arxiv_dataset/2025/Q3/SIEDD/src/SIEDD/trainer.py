import torch
import einops
from torch import nn
from torch.amp.grad_scaler import GradScaler
from torch.optim import Optimizer, lr_scheduler
from torch.optim.lr_scheduler import LRScheduler
import numpy as np
import tqdm
import time
from pathlib import Path
import os
import copy
import random
from itertools import batched
import schedulefree
import bitsandbytes as bnb
import wandb
import pydantic_yaml

from .utils import (
    losses,
    metric,
    helpers,
    Quantize,
    get_coord_sampler,
)
from grokadamw import GrokAdamW
from .models import MLPBlock, SirenNerv, CudaMLP
from .configs import (
    RunConfig,
    TrainerConfig,
    MLPConfig,
    SirenNeRVConfig,
    CudaMLPConfig,
)
from .data_processing.data_process import DataProcess

WANDB_ENABLED = (
    os.getenv("WANDB_PROJECT") is not None and os.getenv("WANDB_GROUP") is not None
)


class Trainer:
    def __init__(
        self,
        cfg: RunConfig,
        data_pipeline: DataProcess,
        save_path: Path,
        resume: bool,
    ):
        self.data_pipeline = data_pipeline
        self.resume = resume
        self.compression_params = cfg.quant_cfg
        self.cfg = cfg
        if not isinstance(cfg.trainer_cfg, TrainerConfig):
            raise ValueError("Needs to be a TrainerConfig")
        self.train_cfg = cfg.trainer_cfg
        self.enc_cfg = cfg.encoder_cfg
        self.save_path = save_path
        self.skip_save = self.train_cfg.skip_save
        self.skip_save_model = self.train_cfg.skip_save_model

        helpers.make_dir(self.save_path)
        pydantic_yaml.to_yaml_file(self.save_path / "run_cfg.yaml", cfg)
        self.setup()
        if WANDB_ENABLED:
            self.setup_wandb()

    def setup(self):
        torch.backends.cudnn.benchmark = True  # False??
        torch.backends.cudnn.deterministic = True

        # run faster
        tf32 = True
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.set_float32_matmul_precision("high" if tf32 else "highest")
        torch.set_printoptions(precision=10)

        if isinstance(self.data_pipeline.coordinates, torch.Tensor):
            self.coordinates = self.data_pipeline.coordinates.cuda()
            self.coords_device = self.coordinates.device
        elif isinstance(self.data_pipeline.coordinates, list):
            self.coordinates = [x.cuda() for x in self.data_pipeline.coordinates]
            self.coords_device = self.coordinates[0].device

        if self.compression_params.quantize:
            self.quantizer = Quantize(self.compression_params, self.enc_cfg)
        else:
            self.quantizer = None

    def setup_wandb(self):
        wandb.define_metric("Frame", hidden=True)
        wandb.define_metric("Iterations", hidden=True)
        for i in range(self.data_pipeline.num_frames):
            wandb.define_metric(
                f"training/frame_{i}/mse", step_metric="Iterations", summary="max"
            )
            wandb.define_metric(
                f"training/frame_{i}/psnr", step_metric="Iterations", summary="max"
            )
            wandb.define_metric(
                f"training/frame_{i}/ssim", step_metric="Iterations", summary="max"
            )
        mets = [""]
        if self.cfg.quant_cfg.quantize:
            mets.append("quant_")
        for met in mets:
            wandb.define_metric(f"frame/{met}metrics/psnr", step_metric="Frame")
            wandb.define_metric(f"frame/{met}metrics/ssim", step_metric="Frame")
            wandb.define_metric(
                f"frame/{met}compression_metrics/bpp", step_metric="Frame"
            )
            wandb.define_metric(
                f"frame/{met}compression_metrics/total_bits", step_metric="Frame"
            )
            wandb.define_metric(
                f"frame/{met}compression_metrics/total_KB", step_metric="Frame"
            )
            wandb.define_metric(
                f"frame/{met}compression_metrics/total_pixels",
                step_metric="Frame",
                hidden=True,
            )
            wandb.define_metric("frame/time", step_metric="Frame")
            wandb.define_metric("frame/fps", step_metric="Frame")
            wandb.define_metric("frame/pred", step_metric="Frame")

    def setup_lr_scheduler(self, optimizer: Optimizer):
        scheduler_str = self.train_cfg.lr_scheduler

        if scheduler_str is not None:
            scheduler = getattr(lr_scheduler, scheduler_str)
            params = self.train_cfg.scheduler_params
            scheduler_params = params if params is not None else {}
            lr_sched: LRScheduler | None = scheduler(optimizer, **scheduler_params)
        else:
            lr_sched = None
        return lr_sched

    def setup_optimizer(
        self,
        model: nn.Module,
        loss_fn: losses.CompositeLoss,
        override_lr=None,
    ):
        optimizer = self.train_cfg.optimizer
        lr = self.train_cfg.lr if override_lr is None else override_lr

        if optimizer == "schedule_free":
            optim = schedulefree.AdamWScheduleFree(
                model.parameters(), lr=lr, betas=self.train_cfg.betas
            )
            optim.train()
        elif optimizer == "grokadamw":
            optim = GrokAdamW(
                model.parameters(),
                lr=lr,
                betas=self.train_cfg.betas,
                grokking_signal_fns=[loss_fn],
            )
        elif optimizer == "adam":
            optim = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                betas=self.train_cfg.betas,
                fused=False,
            )

        elif optimizer == "adam8bit":
            optim = bnb.optim.Adam8bit(
                model.parameters(),
                lr=lr,
                betas=self.train_cfg.betas,
                optim_bits=8,
            )
        else:
            raise ValueError("Invalid optimizer")
        return optim

    def get_bpp(self, frame_idx: list[int], num_params: int):
        name = "_".join(map(str, frame_idx))
        total_pixels = int(np.prod(self.data_pipeline.data_shape[1:]))
        total_num_bits = (
            os.path.getsize(self.save_path / f"model_{name}.bin") * 8 // len(frame_idx)
        )
        # simply dividing the grouped frames by the number of frames for the approx.
        # bpp of each frame
        bpp = total_num_bits / total_pixels
        total_KB = total_num_bits / 8 / 1024
        return [
            metric.CompressionMetrics(
                bpp=bpp,
                total_bits=total_num_bits,
                total_KB=total_KB,
                total_pixels=total_pixels,
                parameters=num_params // len(frame_idx),
            )
            for _ in frame_idx
        ]

    def train(self) -> None:
        print("Total Frames: ", self.data_pipeline.num_frames)
        print("Data shape: ", self.data_pipeline.data_shape)

        metrics_per_frame: list[list[metric.Metrics]] = []
        group_size = self.train_cfg.group_size
        it = map(
            list,
            batched(range(self.data_pipeline.num_frames), group_size),
        )
        first = True
        for frame_idx in it:
            print(f"Training frame(s) {frame_idx}")
            model, optim, scaler, scheduler, loss_fn = self.setup_frame(frame_idx)
            if first:
                print(model)
                first = False
            self.data_pipeline.data_set.cache_frames(frame_idx)
            # wandb.watch(model) TODO: Determine if this is useful, and how to do it
            best_model_state, training_time = self.train_loop(
                model,
                optim,
                scaler,
                scheduler,
                loss_fn,
                frame_idx,
                iters=self.train_cfg.num_iters,
            )
            infos = self.calc_final_metrics_and_save(
                model, best_model_state, frame_idx, encoding_time=training_time
            )
            metrics_per_frame.append(infos)
            for info, idx in zip(infos, frame_idx):
                info_dict = info.model_dump(exclude_none=True)
                # Overwrite the frame-wise metrics with the final one
                helpers.save_json(info_dict, self.save_path / f"frame_info_{idx}.json")
                if WANDB_ENABLED:
                    if idx < self.train_cfg.num_save_images:
                        info_dict["pred"] = wandb.Image(
                            str(self.save_path / f"prediction_{idx}.png")
                        )
                    info_dict = {
                        f"frame/{k}": v for k, v in metric.flatten(info_dict).items()
                    }
                    info_dict["Frame"] = idx
                    wandb.log(info_dict)
            self.data_pipeline.data_set.uncache_frames()
        # Reduce the metrics (mean, average) then report the cumulative over all frames
        all_metrics = [met for lst in metrics_per_frame for met in lst]
        reduced_metrics = metric.reduce_metrics(all_metrics)
        info = reduced_metrics.model_dump(exclude_none=True)
        helpers.save_json(info, self.save_path / "cumulative.json")
        if WANDB_ENABLED and wandb.run is not None:
            info = metric.flatten(info)
            for k, v in info.items():
                wandb.run.summary[k] = v

    def setup_frame(
        self, frame_idx: list[int]
    ) -> tuple[
        nn.Module,
        Optimizer,
        torch.amp.grad_scaler.GradScaler,
        LRScheduler | None,
        losses.CompositeLoss,
    ]:
        model = self.create_model_for_frame()
        loss_fn = losses.CompositeLoss(self.train_cfg)
        optimizer = self.setup_optimizer(model, loss_fn)
        use_mp = self.train_cfg.precision != "fp32"
        scaler = GradScaler(enabled=use_mp)
        scheduler = self.setup_lr_scheduler(optimizer)
        if self.resume:
            self.load_training_state(
                self.save_path, frame_idx, model, optimizer, scaler, scheduler
            )
        return model, optimizer, scaler, scheduler, loss_fn

    def create_model_for_frame(self) -> nn.Module:
        torch.manual_seed(123)
        torch.cuda.manual_seed_all(123)
        np.random.seed(123)
        random.seed(123)
        os.environ["PYTHONHASHSEED"] = str(123)

        net_config = self.cfg.encoder_cfg.net
        group_size = self.train_cfg.group_size
        patch_size = self.cfg.encoder_cfg.patch_size
        net: MLPBlock | CudaMLP | SirenNerv
        if patch_size is None:
            dim_out = 3 * group_size
        else:
            dim_out = (patch_size**2) * 3 * group_size
        data_shape = self.data_pipeline.data_set.data_shape
        if isinstance(net_config, MLPConfig):
            net = MLPBlock(dim_out, data_shape, self.cfg.encoder_cfg)
        elif isinstance(net_config, CudaMLPConfig):
            net = CudaMLP(dim_out, data_shape, self.cfg.encoder_cfg)
        elif isinstance(net_config, SirenNeRVConfig):
            net = SirenNerv(data_shape, self.cfg.encoder_cfg, net_config)
        else:
            raise NotImplementedError()

        if self.cfg.encoder_cfg.compile is True:
            print("Compiling the model")
            net = torch.compile(net, mode="max-autotune")  # type: ignore
        return net.cuda()

    def train_loop(
        self,
        model: nn.Module,
        optim: Optimizer,
        scaler: GradScaler,
        scheduler: LRScheduler | None,
        loss_fn: losses.CompositeLoss,
        frame_idx: list[int],
        iters: int,
    ):
        best_psnr = float("-inf")

        precision = self.train_cfg.precision
        use_mixed_precision = precision != "fp32"

        autocast_dtype = helpers.get_auto_cast_dtype(precision)
        start_iter = 0

        model.train()
        best_model_state = copy.deepcopy(model.state_dict())
        gt = self.data_pipeline.data_set[:]

        C, H, W = self.data_pipeline.input_data_shape
        ps = self.enc_cfg.patch_size
        if ps is None:
            images = gt.reshape((-1, H, W, C))
        else:
            images = einops.rearrange(
                gt,
                "n (h w) c ph pw -> n (h ph) (w pw) c",
                h=H // ps,
                w=W // ps,
            )

        C, H, W = self.data_pipeline.data_shape
        coordinate_sampler = get_coord_sampler(
            self.coordinates,
            cfg=self.cfg,
            images=images,
        )

        exclude_time = 0.0

        iteration = tqdm.tqdm(range(start_iter, iters))
        if start_iter >= iters:
            raise ValueError("Cannot train, too few iterations")
        start_time = time.perf_counter()

        for iter, (inputs, feats) in zip(iteration, coordinate_sampler):
            if iter < 5:
                compile_exclude_start = time.perf_counter()
            optim.zero_grad()
            if use_mixed_precision:
                with torch.autocast(
                    enabled=use_mixed_precision,
                    dtype=autocast_dtype,
                    device_type="cuda",
                ):
                    output: torch.Tensor = model(inputs)
                    loss: torch.Tensor = loss_fn(output, feats)

                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()

            else:
                output = model(inputs)
                loss = loss_fn(output, feats)
                loss.backward()
                optim.step()
            if scheduler is not None:
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(loss)
                else:
                    scheduler.step()

            torch.cuda.synchronize()
            exclude_start = time.perf_counter()
            if iter < 5:
                exclude_time += exclude_start - compile_exclude_start  # type: ignore

            # METRICS / SAVE
            best_model_state, best_psnr = self.calc_metrics_and_save(
                model,
                best_model_state,
                optim,
                scaler,
                scheduler,
                best_psnr,
                frame_idx,
                iter,
            )
            desc = ""
            if scheduler is not None:
                desc += f"LR: {scheduler.get_last_lr()[0]:.2e}, "
            desc += f"PSNR: {best_psnr:.4f}"
            iteration.set_description(desc, refresh=True)
            exclude_time += time.perf_counter() - exclude_start
        total_time = time.perf_counter() - start_time
        return best_model_state, total_time - exclude_time

    def calc_metrics_and_save(
        self,
        model: nn.Module,
        best_model_state: dict,
        optim: Optimizer,
        scaler: GradScaler,
        scheduler: LRScheduler | None,
        best_psnr: float,
        frame_idx: list[int],
        iter: int,
    ):
        do_eval = (iter % self.train_cfg.eval_interval == 0) and iter != 0
        do_save = (
            iter % self.train_cfg.save_interval == 0
        ) and iter >= self.train_cfg.eval_interval

        if do_eval or do_save:  # Need to evaluate if we are saving
            val_infos, _ = self.validate_frame(model)
            reduced_val_info = metric.reduce_quality_metrics(val_infos)
            psnr = reduced_val_info.psnr
            if psnr > best_psnr:
                best_psnr = psnr
                best_model_state = copy.deepcopy(model.state_dict())
            for idx, val_info in zip(frame_idx, val_infos):
                info = val_info.model_dump(exclude_none=True)
                info = metric.flatten(info)
                info = {
                    f"training/frame_{idx}/{k}": v
                    for k, v in metric.flatten(info).items()
                }
                info["Iterations"] = iter
                helpers.save_json(info, self.save_path / f"frame_info_{idx}.json")
                if WANDB_ENABLED:
                    wandb.log(info)
            if do_save:
                self.save_training_state(
                    best_model_state=best_model_state,
                    optimizer=optim,
                    scaler=scaler,
                    lr_scheduler=scheduler,
                    save_dir=self.save_path,
                    frame_idx=frame_idx,
                    iter=iter,
                    torch_save=True,
                )
        return best_model_state, best_psnr

    @torch.no_grad()
    def get_fps(self, model: nn.Module) -> float:
        model = model.eval()
        start_time = time.perf_counter()
        output = model(self.coordinates)
        output = helpers.process_predictions(
            output,
            self.enc_cfg,
            input_data_shape=self.data_pipeline.data_shape,
        ).cpu()
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        num_frames = output.size(0)
        seconds_per_frame = (end_time - start_time) / num_frames
        return 1 / seconds_per_frame

    def calc_final_metrics_and_save(
        self,
        model: nn.Module,
        best_model_state: dict,
        frame_idx: list[int],
        encoding_time: float,
    ):
        model.load_state_dict(best_model_state)
        num_params = sum([x.numel() for x in model.parameters()])
        self.save_artefacts(best_model_state, self.save_path, frame_idx)
        val_comps = self.get_bpp(frame_idx, num_params)
        val_qualities, predictions = self.validate_frame(model, True)
        if self.quantizer is not None:
            # Quantize model then evaluate
            model, compressed_state = self.quantizer.quantize_model(
                model, best_model_state
            )
            self.save_artefacts(compressed_state, self.save_path, frame_idx)
            # read bytes from compressed_state and update metric.

            quant_val_qualities, predictions = self.validate_frame(
                model, save_preds=True
            )
            quant_val_comps = self.get_bpp(frame_idx, num_params)
        else:
            quant_val_qualities = [None] * len(frame_idx)
            quant_val_comps = [None] * len(frame_idx)
        fps_for_group = self.get_fps(model)
        val_infos = [
            metric.Metrics(
                metrics=val_quality,
                compression_metrics=val_comp,
                quant_metrics=quant_val_quality,
                quant_compression_metrics=quant_val_comp,
                time=encoding_time / len(frame_idx),
                fps=fps_for_group,
            )
            for val_quality, val_comp, quant_val_quality, quant_val_comp in zip(
                val_qualities,
                val_comps,
                quant_val_qualities,
                quant_val_comps,
            )
        ]

        if not self.skip_save and predictions is not None:
            for i, idx in enumerate(frame_idx):
                if idx < self.train_cfg.num_save_images:
                    self.save_media(predictions[i][None], idx, self.save_path)
        return val_infos

    def validate_frame(
        self,
        model: nn.Module,
        save_preds: bool = False,
    ):
        model.eval()
        ssim = 0.0
        ssim_calculator = metric.SSIM(device=self.coords_device)
        mse_psnr_calc = metric.MSEPSNR(device=self.coords_device)

        preds = None

        with torch.no_grad():
            model.eval()
            features = self.data_pipeline.data_set[:]
            model_output = model(self.coordinates)
            output: torch.Tensor = model_output
            psnr, mse = mse_psnr_calc(features, output)
            psnr = psnr.to("cpu", non_blocking=True)
            mse = mse.to("cpu", non_blocking=True)

            processed_feats = helpers.process_predictions(
                features,
                self.enc_cfg,
                input_data_shape=self.data_pipeline.data_shape,
            )
            processed_out = helpers.process_predictions(
                output,
                self.enc_cfg,
                input_data_shape=self.data_pipeline.data_shape,
            )
            ssim = ssim_calculator(processed_feats, processed_out)
            ssim = ssim.to("cpu", non_blocking=True)

            if save_preds:
                preds = output.to("cpu", non_blocking=True)

        torch.cuda.synchronize()
        infos = [
            metric.QualityMetrics(mse=mse.item(), psnr=psnr.item(), ssim=ssim.item())
            for mse, psnr, ssim in zip(mse, psnr, ssim)
        ]
        torch.cuda.empty_cache()
        return infos, preds

    def save_media(self, predictions: torch.Tensor, frame_idx: int, save_dir: Path):
        preds = helpers.process_predictions(
            predictions,
            self.enc_cfg,
            input_data_shape=self.data_pipeline.data_shape,
        )
        helpers.save_tensor_img(preds, save_dir / f"prediction_{frame_idx}.png")

    def save_artefacts(
        self, best_model_state, save_dir: Path, frame_idx: list[int], torch_save=False
    ):
        if not self.skip_save_model:
            name = "_".join(map(str, frame_idx))
            save_obj = best_model_state
            helpers.save_pickle(
                save_obj,
                filename=save_dir / f"model_{name}.bin",
                compressed=True,
                torch_save=torch_save,
            )

    def load_artefacts(self, load_path: Path):
        try:
            state = torch.load(load_path)
        except Exception:
            state = helpers.load_pickle(load_path, compressed=True)
        return state

    def save_training_state(
        self,
        best_model_state: dict,
        optimizer: Optimizer,
        scaler: GradScaler,
        lr_scheduler: LRScheduler | None,
        save_dir: Path,
        frame_idx: list[int],
        iter: int,
        torch_save=False,
    ):
        save_obj = {
            "model": best_model_state,
            "iter": iter,
            "optimizer": optimizer.state_dict(),
        }
        if lr_scheduler is not None:
            save_obj["lr_scheduler"] = lr_scheduler.state_dict()
        if scaler.is_enabled():
            save_obj["scaler"] = scaler.state_dict()
        helpers.save_pickle(
            save_obj,
            filename=save_dir / f"frame_{frame_idx}.bin",
            compressed=True,
            torch_save=torch_save,
        )

    def load_training_state(
        self,
        save_dir: Path,
        frame_idx: list[int],
        model: nn.Module,
        optimizer: Optimizer,
        scaler: GradScaler,
        scheduler: LRScheduler | None,
    ):
        try:
            state: dict = torch.load(save_dir / f"frame_{frame_idx}.bin")
        except Exception:
            state = helpers.load_pickle(
                save_dir / f"frame_{frame_idx}.bin", compressed=True
            )
        # Load cfg?
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scaler.is_enabled():
            scaler.load_state_dict(state["scaler"])
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
