import torch
import einops
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
import shutil
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
    optim as optimizers,
)
from SIEDD.decode import replace
from grokadamw import GrokAdamW
from .models import Strainer as StrainerNet
from .configs import RunConfig, StrainerConfig, STRAINERNetConfig
from .data_processing.data_process import DataProcess

WANDB_ENABLED = (
    os.getenv("WANDB_PROJECT") is not None and os.getenv("WANDB_GROUP") is not None
)


class Strainer:
    def __init__(
        self,
        cfg: RunConfig,
        data_pipeline: DataProcess,
        save_path: Path,
        resume: bool,
        setup_wandb=True,
    ):
        self.data_pipeline = data_pipeline
        self.resume = resume
        self.compression_params = cfg.quant_cfg
        self.cfg = cfg
        if not isinstance(cfg.trainer_cfg, StrainerConfig):
            raise ValueError("Needs to be a StrainerConfig")
        self.train_cfg = cfg.trainer_cfg
        self.enc_cfg = cfg.encoder_cfg
        self.amortized = self.train_cfg.amortized
        self.ldtw = self.train_cfg.lora_dec_transfer_decoder
        net_config = cfg.encoder_cfg.net
        if isinstance(net_config, STRAINERNetConfig):
            self.net_cfg: STRAINERNetConfig = net_config
        else:
            raise ValueError("Needs to be a STRAINERNetConfig")
        self.save_path = save_path
        self.skip_save = self.train_cfg.skip_save
        self.skip_save_model = self.train_cfg.skip_save_model

        helpers.make_dir(self.save_path)
        pydantic_yaml.to_yaml_file(self.save_path / "run_cfg.yaml", cfg)
        self.setup()
        if WANDB_ENABLED and setup_wandb:
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
        wandb.define_metric("training/shared_encoder/mse", step_metric="Iterations")
        wandb.define_metric("training/shared_encoder/psnr", step_metric="Iterations")
        wandb.define_metric("training/shared_encoder/ssim", step_metric="Iterations")
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
        with_warmup = lr_scheduler.LinearLR(
            optimizer, 0.01, 1, self.train_cfg.lr_warmup
        )
        if lr_sched is not None:
            with_warmup = lr_scheduler.SequentialLR(
                optimizer, [with_warmup, lr_sched], [self.train_cfg.lr_warmup]
            )
        return with_warmup

    def setup_optimizer(
        self,
        model: StrainerNet,
        loss_fn: losses.CompositeLoss,
        shared_encoder_training: bool = False,
    ):
        optimizer = self.train_cfg.optimizer
        lr = self.train_cfg.shared_lr if shared_encoder_training else self.train_cfg.lr

        params = (p for p in model.parameters() if p.requires_grad)

        if optimizer == "schedule_free":
            optim = schedulefree.AdamWScheduleFree(
                params, lr=lr, betas=self.train_cfg.betas
            )
            optim.train()
        elif optimizer == "grokadamw":
            optim = GrokAdamW(
                params,
                lr=lr,
                betas=self.train_cfg.betas,
            )
        elif optimizer == "adam":
            optim = torch.optim.Adam(
                params,
                lr=lr,
                betas=self.train_cfg.betas,
                fused=False,
            )

        elif optimizer == "adam8bit":
            optim = bnb.optim.Adam8bit(
                params,
                lr=lr,
                betas=self.train_cfg.betas,
                optim_bits=8,
            )
        elif optimizer == "muon":
            optim = optimizers.Muon(model, 1e-2, lr)
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

    def report_final_metrics(self, frame_idx: list[int], infos: list[metric.Metrics]):
        name = "_".join(map(str, frame_idx))
        save_infos = {
            i: inf.model_dump(exclude_none=True) for i, inf in zip(frame_idx, infos)
        }
        filepath = self.save_path / f"frame_info_{name}.json"
        helpers.save_json(save_infos, filepath)
        for info, idx in zip(infos, frame_idx):
            info_dict = info.model_dump(exclude_none=True)
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

    def train(self) -> None:
        print("Total Frames: ", self.data_pipeline.num_frames)
        print("Data shape: ", self.data_pipeline.data_shape)
        shared_frames = self.data_pipeline.frame_idx
        print(
            f"STRAINER: Using frames {shared_frames}",
            "for shared encoder training",
        )

        self.data_pipeline.data_set.cache_frames(shared_frames)
        shared_model, shared_optim, shared_scaler, shared_scheduler, shared_loss_fn = (
            self.setup_frame(shared_frames, False)
        )
        print(shared_model)
        if self.train_cfg.shared_encoder_path is not None:
            print(
                "Loading shared encoder state from", self.train_cfg.shared_encoder_path
            )
            shared_training_time = 0.0
            best_shared_state = self.load_artefacts(self.train_cfg.shared_encoder_path)
            best_encoder_state = {
                k.removeprefix("_orig_mod.").removeprefix("encoderINR."): v
                for k, v in best_shared_state.items()
                if "encoderINR" in k
            }
            self.quantizer = Quantize(self.cfg.quant_cfg, self.enc_cfg)
            shared_model.encoderINR.load_state_dict(best_encoder_state, strict=False)
            shared_model.encoderINR.requires_grad_(False)
        else:
            best_shared_state, shared_training_time = self.train_loop(
                shared_model,
                shared_optim,
                shared_scaler,
                shared_scheduler,
                shared_loss_fn,
                shared_frames,
                iters=self.train_cfg.shared_iters,
                shared=True,
            )
            print("Shared Encoder Training Time:", shared_training_time)
            shared_model.load_state_dict(best_shared_state)
            best_shared_state = {
                k: v for k, v in best_shared_state.items() if "encoderINR" in k
            }
            self.save_artefacts(best_shared_state, shared_frames, shared=True)
        del shared_optim, shared_scaler, shared_scheduler, shared_loss_fn
        self.data_pipeline.data_set.uncache_frames()

        metrics_per_frame: list[list[metric.Metrics]] = []
        prev_model: StrainerNet = shared_model
        group_size = self.train_cfg.group_size
        it = map(
            list,
            batched(range(self.data_pipeline.num_frames), group_size),
        )
        first = True
        prev_frame_idx = shared_frames
        for frame_idx in it:
            print(f"Training frame(s) {frame_idx}")
            self.data_pipeline.data_set.cache_frames(frame_idx)
            model, optim, scaler, scheduler, loss_fn = self.setup_frame(
                frame_idx, True, prev_model, prev_frame_idx
            )
            if first:
                print(model)
                first = False
            best_model_state, training_time = self.train_loop(
                model,
                optim,
                scaler,
                scheduler,
                loss_fn,
                frame_idx,
                iters=self.train_cfg.iters,
                shared=False,
            )
            if self.amortized is False or self.amortized == "LoraDec" and self.ldtw:
                prev_frame_idx = frame_idx
                prev_model = model
            if isinstance(optim, schedulefree.AdamWScheduleFree):
                optim.eval()
            infos = self.calc_final_metrics_and_save(
                model, best_model_state, frame_idx, encoding_time=training_time
            )
            metrics_per_frame.append(infos)
            self.report_final_metrics(frame_idx, infos)
            self.data_pipeline.data_set.save_state(self.save_path)
            self.data_pipeline.data_set.uncache_frames()
        # Reduce the metrics (mean, sum) then report the cumulative over all frames
        all_metrics = [met for lst in metrics_per_frame for met in lst]
        reduced_metrics = metric.reduce_metrics(all_metrics)
        reduced_metrics.time += shared_training_time
        info = reduced_metrics.model_dump(exclude_none=True)
        helpers.save_json(info, self.save_path / "cumulative.json")
        if WANDB_ENABLED and wandb.run is not None:
            info = metric.flatten(info)
            for k, v in info.items():
                wandb.run.summary[k] = v
        self.cleanup()
        if WANDB_ENABLED:
            wandb.finish()

    def setup_frame(
        self,
        frame_idx: list[int],
        transfer_decoder_weights: bool,
        previous_model: StrainerNet | None = None,
        previous_frame_idx: list[int] | None = None,
    ) -> tuple[
        StrainerNet,
        Optimizer,
        torch.amp.grad_scaler.GradScaler,
        LRScheduler | None,
        losses.CompositeLoss,
    ]:
        num_decoders = len(frame_idx)
        shared_encoder_training = previous_model is None
        model = self.create_model(num_decoders, shared_encoder_training)
        if previous_model is not None:
            model.load_encoder_weights_from(previous_model)
            if (
                transfer_decoder_weights
                and previous_frame_idx is not None
                and not self.ldtw
            ):
                for i, frame in enumerate(frame_idx):
                    # Load the model from the closest decoder of previous model
                    framediff = abs(np.array(previous_frame_idx) - frame)
                    closest_frame = np.argmin(framediff)
                    model.load_decoder_weights_from(previous_model, i, closest_frame)
                    print(
                        f"Loading frame {previous_frame_idx[closest_frame]} into {frame}"
                    )
            elif (
                self.ldtw
                and self.amortized == "LoraDec"
                and transfer_decoder_weights is True
                and previous_frame_idx is not None
            ):
                # Load the model from the closest decoder of shared encoder
                framediff = abs(np.array(previous_frame_idx) - frame_idx[0])
                closest_frame = np.argmin(framediff)
                model.load_decoder_weights_from(
                    previous_model, slice(None), closest_frame
                )
                print(
                    f"Loading decoder for frame {previous_frame_idx[closest_frame]} from SHARED ENCODER"
                )
        loss_fn = losses.CompositeLoss(self.train_cfg)
        optimizer = self.setup_optimizer(
            model, loss_fn, shared_encoder_training=shared_encoder_training
        )
        use_mp = self.train_cfg.precision != "fp32"
        scaler = GradScaler(enabled=use_mp)
        scheduler = self.setup_lr_scheduler(optimizer)
        if self.resume:
            # TODO: Load shared encoder training state?
            self.load_training_state(
                self.save_path, frame_idx, model, optimizer, scaler, scheduler
            )
        return model, optimizer, scaler, scheduler, loss_fn

    def create_model(self, num_decoders: int, shared: bool) -> StrainerNet:
        torch.manual_seed(123)
        torch.cuda.manual_seed_all(123)
        np.random.seed(123)
        random.seed(123)
        os.environ["PYTHONHASHSEED"] = str(123)

        patch_size = self.cfg.encoder_cfg.patch_size
        if patch_size is None:
            dim_out = 3
        else:
            dim_out = (patch_size**2) * 3
        data_shape = self.data_pipeline.data_set.data_shape
        try:
            gt_images = helpers.process_predictions(
                self.data_pipeline.data_set[:],
                self.enc_cfg,
                input_data_shape=data_shape,
            )
        except Exception:
            gt_images = torch.Tensor([0])
        net: StrainerNet = StrainerNet(
            dim_out,
            data_shape,
            num_decoders,
            self.enc_cfg,
            self.cfg.quant_cfg,
            gt_images,
            shared,
        )

        if isinstance(self.amortized, str) and not shared:
            net.lora(self.amortized == "LoraFull", True)
        if self.amortized in (True, "LoraDec") and not shared:
            net.encoderINR.requires_grad_(False)

        if self.cfg.encoder_cfg.compile is True:
            print("Compiling the model")
            net = torch.compile(net, mode="max-autotune")  # type: ignore
        return net.cuda()

    def train_loop(
        self,
        model: StrainerNet,
        optim: Optimizer,
        scaler: GradScaler,
        scheduler: LRScheduler | None,
        loss_fn: losses.CompositeLoss,
        frame_idx: list[int],
        iters: int,
        shared: bool = False,
    ):
        best_psnr = float("-inf")

        precision = self.train_cfg.precision
        use_mixed_precision = precision != "fp32"

        autocast_dtype = helpers.get_auto_cast_dtype(precision)
        start_iter = 0

        model = model.train()
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
        coordinate_sampler = get_coord_sampler(
            self.coordinates,
            cfg=self.cfg,
            images=images,
        )

        iteration = tqdm.tqdm(range(start_iter, iters))
        if start_iter >= iters:
            raise ValueError("Cannot train, too few iterations")
        start_time = time.perf_counter()
        exclude_time = 0.0
        for iter, (inputs, feats) in zip(iteration, coordinate_sampler):
            model = model.train()
            if iter < 5:
                compile_exclude_start = time.perf_counter()
            if use_mixed_precision:
                with torch.autocast(
                    enabled=use_mixed_precision,
                    dtype=autocast_dtype,
                    device_type="cuda",
                ):
                    optim.zero_grad()
                    output: torch.Tensor = model(inputs)
                    loss: torch.Tensor = loss_fn(output, feats)

                scaler.scale(loss).backward()
                if isinstance(optim, GrokAdamW):
                    scaler.step(optim, lambda: loss.item())
                else:
                    scaler.step(optim)
                scaler.update()
            else:
                optim.zero_grad()
                output: torch.Tensor = model(inputs)
                loss: torch.Tensor = loss_fn(output, feats)
                loss.backward()
                if isinstance(optim, GrokAdamW):
                    optim.step(lambda: loss.item())
                else:
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

            del output
            del feats
            torch.cuda.empty_cache()
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
                shared,
            )
            if isinstance(optim, schedulefree.AdamWScheduleFree):
                optim.train()
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
        model: StrainerNet,
        best_model_state: dict,
        optim: Optimizer,
        scaler: GradScaler,
        scheduler: LRScheduler | None,
        best_psnr: float,
        frame_idx: list[int],
        iter: int,
        shared_training: bool,
    ):
        do_eval = iter % self.train_cfg.eval_interval == 0
        do_save = (
            iter % self.train_cfg.save_interval == 0
        ) and iter >= self.train_cfg.eval_interval

        if do_eval or do_save:  # Need to evaluate if we are saving
            if isinstance(optim, schedulefree.AdamWScheduleFree):
                optim.eval()
            val_infos = self.validate_frame(model)[0]
            reduced_val_info = metric.reduce_quality_metrics(val_infos)
            psnr = reduced_val_info.psnr
            if psnr > best_psnr:
                best_psnr = psnr
                best_model_state = copy.deepcopy(model.state_dict())
            name = "_".join(map(str, frame_idx))
            save_infos = {
                i: inf.model_dump(exclude_none=True)
                for i, inf in zip(frame_idx, val_infos)
            }
            filename = (
                "shared_encoder_info.json"
                if shared_training
                else f"frame_info_{name}.json"
            )
            helpers.save_json(save_infos, self.save_path / filename)
            for idx, val_info in zip(frame_idx, val_infos):
                info = val_info.model_dump(exclude_none=True)
                info = metric.flatten(info)
                if shared_training:
                    info = {f"training/shared_encoder/{k}": v for k, v in info.items()}
                else:
                    info = {
                        f"training/frame_{idx}/{k}": v
                        for k, v in metric.flatten(info).items()
                    }
                info["Iterations"] = iter
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

    @torch.inference_mode()
    @torch.no_grad()
    def get_fps(self, model: StrainerNet, frame_idx=None):
        precision = self.train_cfg.precision
        use_mixed_precision = precision != "fp32"
        autocast_dtype = helpers.get_auto_cast_dtype(precision)
        model = model.eval()
        torch.cuda.empty_cache()
        with torch.autocast(
            enabled=use_mixed_precision,
            dtype=autocast_dtype,
            device_type="cuda",
        ):
            bsz = self.enc_cfg.inference_batch_size
            csz = self.enc_cfg.inference_chunk_size
            if (self.amortized is True or self.amortized == "LoraDec") and csz == 1:
                coords = model.encoderINR(self.coordinates, preprocess_output=False)
            else:
                coords = self.coordinates
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            if (self.amortized is True or self.amortized == "LoraDec") and csz == 1:
                assert isinstance(coords, torch.Tensor)
                output = model.decoderINRs.forward_batched(
                    coords, batch_size=bsz, chunk_size=csz
                )
                output = model.process_output(output)
            else:
                output = model.forward_batched(coords, bsz, chunk_size=csz)
            if frame_idx is None:
                frame_idx = self.data_pipeline.data_set.current_cache_idx
            outs = []
            for i, idx in enumerate(frame_idx):
                outs.append(
                    self.data_pipeline.data_set.transform.inverse(output[i], idx)
                )
            output = torch.stack(outs)
            # now just reshape and send to CPU
            out = helpers.process_predictions(
                output,
                self.enc_cfg,
                input_data_shape=self.data_pipeline.data_shape,
            )
            out = out.cpu()

        torch.cuda.synchronize()
        end_time = time.perf_counter()
        num_frames = output.size(0)
        seconds_per_frame = (end_time - start_time) / num_frames
        return 1 / seconds_per_frame, output

    @torch.inference_mode()
    @torch.no_grad()
    def calc_final_metrics_and_save(
        self,
        model: StrainerNet,
        best_model_state: dict,
        frame_idx: list[int],
        encoding_time: float,
    ):
        model.load_state_dict(best_model_state)
        trainable_params = helpers.trainable_state_dict(model)
        num_params = sum([x.numel() for x in trainable_params.values()])
        self.save_artefacts(trainable_params, frame_idx)
        val_comps = self.get_bpp(frame_idx, num_params)
        fps_model = copy.deepcopy(model)
        sep_patch_pix = self.net_cfg.sep_patch_pix
        if not sep_patch_pix:
            replace(fps_model)
        fps_model = fps_model.to(torch.bfloat16)
        val_qualities, predictions, fps_for_group = self.validate_frame(fps_model, True)
        if self.quantizer is not None:
            # Quantize model then evaluate
            quant_model = copy.deepcopy(model)
            _, compressed_state = self.quantizer.quantize_model(
                quant_model, trainable_params
            )
            self.save_artefacts(compressed_state, frame_idx)
            # read bytes from compressed_state and update metric.

            if not sep_patch_pix:
                replace(quant_model)
            quant_model = quant_model.to(torch.bfloat16)
            quant_val_qualities, predictions, fps_for_group = self.validate_frame(
                quant_model, save_preds=True
            )
            quant_val_comps = self.get_bpp(frame_idx, num_params)
        else:
            quant_val_qualities = [None] * len(frame_idx)
            quant_val_comps = [None] * len(frame_idx)

        if not self.skip_save and predictions is not None:
            for i, idx in enumerate(frame_idx):
                if idx < self.train_cfg.num_save_images:
                    self.save_media(predictions[None, i], [idx], self.save_path)
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

        return val_infos

    @torch.inference_mode()
    @torch.no_grad()
    def validate_frame(
        self,
        model: StrainerNet,
        save_preds: bool = False,
    ):
        model = model.eval()
        ssim = 0.0
        ssim_calculator = metric.SSIM(device=self.coords_device)
        mse_psnr_calc = metric.MSEPSNR(device=self.coords_device)

        preds = None

        torch.cuda.empty_cache()
        fps, model_output = self.get_fps(model)
        features = self.data_pipeline.data_set[:]
        features = self.data_pipeline.data_set.original_images
        psnr, mse = mse_psnr_calc(features, model_output)
        psnr = psnr.to("cpu", non_blocking=True)
        mse = mse.to("cpu", non_blocking=True)

        processed_feats = helpers.process_predictions(
            features,
            self.enc_cfg,
            input_data_shape=self.data_pipeline.data_shape,
        )
        processed_out = helpers.process_predictions(
            model_output,
            self.enc_cfg,
            input_data_shape=self.data_pipeline.data_shape,
        )
        ssim = ssim_calculator(processed_feats, processed_out, batch=True)
        ssim = ssim.to("cpu", non_blocking=True)

        if save_preds:
            preds = model_output.to("cpu", non_blocking=True)

        torch.cuda.synchronize()
        infos = [
            metric.QualityMetrics(mse=mse.item(), psnr=psnr.item(), ssim=ssim.item())
            for mse, psnr, ssim in zip(mse, psnr, ssim)
        ]
        return infos, preds, fps

    def save_media(
        self, predictions: torch.Tensor, frame_idx: list[int], save_dir: Path
    ):
        preds = helpers.process_predictions(
            predictions,
            self.enc_cfg,
            input_data_shape=self.data_pipeline.data_shape,
        )
        for i, idx in enumerate(frame_idx):
            helpers.save_tensor_img(preds[i], save_dir / f"prediction_{idx}.png")

    def save_artefacts(
        self,
        best_model_state,
        frame_idx: list[int],
        torch_save=False,
        shared: bool = False,
    ):
        if not self.skip_save_model:
            name = "_".join(map(str, frame_idx))
            save_obj = best_model_state
            fn = "shared_encoder.bin" if shared else f"model_{name}.bin"
            helpers.save_pickle(
                save_obj,
                filename=self.save_path / fn,
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
        frame_str = "training_state_" + "_".join(map(str, frame_idx))
        helpers.save_pickle(
            save_obj,
            filename=save_dir / f"{frame_str}.bin",
            compressed=True,
            torch_save=torch_save,
        )

    def load_training_state(
        self,
        save_dir: Path,
        frame_idx: list[int],
        model: StrainerNet,
        optimizer: Optimizer,
        scaler: GradScaler,
        scheduler: LRScheduler | None,
    ):
        frame_str = "training_state_" + "_".join(map(str, frame_idx))
        try:
            state: dict = torch.load(save_dir / f"{frame_str}.bin")
        except Exception:
            state = helpers.load_pickle(save_dir / f"{frame_str}.bin", compressed=True)
        # Load cfg?
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scaler.is_enabled():
            scaler.load_state_dict(state["scaler"])
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])

    def cleanup(self):
        for pth in self.save_path.iterdir():
            if "training_state" in str(pth.absolute()):
                print("Deleting", str(pth))
                pth.unlink()
            elif pth.name == "wandb":
                shutil.rmtree(pth, ignore_errors=True)
