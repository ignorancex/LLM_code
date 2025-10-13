import os
from copy import deepcopy
from datetime import timedelta
from pprint import pformat

import torch
import torch.distributed as dist
import wandb
from colossalai.booster import Booster
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device, set_seed
from tqdm import tqdm

from third_party.opensora.acceleration.checkpoint import set_grad_checkpoint
from third_party.opensora.acceleration.parallel_states import get_data_parallel_group
from third_party.opensora.registry import build_module
from third_party.opensora.utils.ckpt_utils import load, model_gathering, model_sharding, record_model_param_shape, save
from third_party.opensora.utils.config_utils import define_experiment_workspace, parse_configs, save_training_config
from third_party.opensora.utils.lr_scheduler import LinearWarmupLR
from third_party.opensora.utils.misc import (
    all_reduce_mean,
    create_logger,
    create_tensorboard_writer,
    format_numel_str,
    get_model_numel,
    requires_grad,
    to_torch_dtype,
)
from third_party.opensora.utils.train_utils import create_colossalai_plugin, update_ema

from third_party.fairseq.data.data_utils import lengths_to_mask

from v2sflow.registry import DATASETS, MODELS, SCHEDULERS

def main():
    # ======================================================
    # 1. configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs(training=True)

    # == device and dtype ==
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    cfg_dtype = cfg.get("dtype", "bf16")
    # assert cfg_dtype in ["fp16", "bf16"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))

    # == colossalai init distributed training ==
    # NOTE: A very large timeout is set to avoid some processes exit early
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    set_seed(cfg.get("seed", 1024))
    coordinator = DistCoordinator()
    device = get_current_device()

    # == init exp_dir ==
    exp_name, exp_dir = define_experiment_workspace(cfg)
    exp_dir = cfg.outputs
    exp_name = os.path.split(exp_dir)[-1]
    coordinator.block_all()
    if coordinator.is_master():
        os.makedirs(exp_dir, exist_ok=True)
        try:
            import json
            with open(f"{exp_dir}/config.txt") as f:
                prev_cfg_dict = json.load(f)
            if prev_cfg_dict != cfg.to_dict():
                print("cfg mismatch with previous one. overwrite cfg.")
                with open(f"{exp_dir}/config_prev.txt", "w") as f:
                    json.dump(prev_cfg_dict, f, indent=4)
        except:
            pass
        save_training_config(cfg.to_dict(), exp_dir)
    coordinator.block_all()

    # == init logger, tensorboard & wandb ==
    logger = create_logger(exp_dir)
    logger.info("Experiment directory created at %s", exp_dir)
    logger.info("Training configuration:\n %s", pformat(cfg.to_dict()))
    if coordinator.is_master():
        tb_writer = create_tensorboard_writer(exp_dir)
        if cfg.get("wandb", False):
            wandb.init(project="Open-Sora", name=exp_name, config=cfg.to_dict(), dir="./outputs/wandb")

    # == init ColossalAI booster ==
    plugin = create_colossalai_plugin(
        plugin=cfg.get("plugin", "zero2"),
        dtype=cfg_dtype,
        grad_clip=cfg.get("grad_clip", 0),
        sp_size=cfg.get("sp_size", 1),
        reduce_bucket_size_in_m=cfg.get("reduce_bucket_size_in_m", 20),
    )
    booster = Booster(plugin=plugin)
    torch.set_num_threads(1)

    # ======================================================
    # 2. build dataset and dataloader
    # ======================================================
    logger.info("Building dataset...")
    # == build dataset ==
    dataset = build_module(cfg.dataset, DATASETS)
    logger.info("Dataset contains %s samples.", len(dataset))

    # == build dataloader ==
    from third_party.fairseq.tasks.fairseq_task import FairseqTask
    task = FairseqTask()
    process_group = get_data_parallel_group()
    epoch_itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=cfg.get("max_tokens", None),
        max_sentences=cfg.get("batch_size", None),
        required_batch_size_multiple=cfg.get("required_batch_size_multiple", 1),
        seed=cfg.get("seed", 1024),
        num_shards=process_group.size(),
        shard_id=process_group.rank(),
        num_workers=cfg.get("num_workers", 4),
        epoch=0,
        data_buffer_size=10,
    )
    num_steps_per_epoch = len(epoch_itr)
    dummy_batch = epoch_itr.first_batch
    def _prepare_sample(sample, dummy_batch):
        if sample is None or len(sample) == 0:
            assert (
                dummy_batch is not None and len(dummy_batch) > 0
            ), "Invalid dummy batch: {}".format(dummy_batch)
            sample, _ = _prepare_sample(dummy_batch, dummy_batch)
            return sample, True
        return sample, False
    _prepare_sample(dummy_batch, dummy_batch)

    # ======================================================
    # 3. build model
    # ======================================================
    logger.info("Building models...")
    # == build diffusion model ==
    model = (
        build_module(
            cfg.model,
            MODELS,
        )
        .to(device, dtype)
        .train()
    )

    model_numel, model_numel_trainable = get_model_numel(model)
    logger.info(
        "[Diffusion] Trainable model params: %s, Total model params: %s",
        format_numel_str(model_numel_trainable),
        format_numel_str(model_numel),
    )

    # == build ema for diffusion model ==
    ema = deepcopy(model).to(torch.float32).to(device)
    requires_grad(ema, False)
    ema_shape_dict = record_model_param_shape(ema)
    ema.eval()
    update_ema(ema, model, decay=0, sharded=False)

    # == setup loss function, build scheduler ==
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # == setup optimizer ==
    optimizer = HybridAdam(
        filter(lambda p: p.requires_grad, model.parameters()),
        adamw_mode=True,
        lr=cfg.get("lr", 1e-4),
        weight_decay=cfg.get("weight_decay", 0),
        eps=cfg.get("adam_eps", 1e-8),
    )

    warmup_steps = cfg.get("warmup_steps", None)

    if warmup_steps is None:
        lr_scheduler = None
    else:
        lr_scheduler = LinearWarmupLR(optimizer, warmup_steps=cfg.get("warmup_steps"))

    # == additional preparation ==
    if cfg.get("grad_checkpoint", False):
        set_grad_checkpoint(model)

    # =======================================================
    # 4. distributed training preparation with colossalai
    # =======================================================
    logger.info("Preparing for distributed training...")
    # == boosting ==
    # NOTE: we set dtype first to make initialization of model consistent with the dtype; then reset it to the fp32 as we make diffusion scheduler in fp32
    torch.set_default_dtype(dtype)
    model, optimizer, _, epoch_itr, lr_scheduler = booster.boost(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataloader=epoch_itr,
    )
    torch.set_default_dtype(torch.float)
    logger.info("Boosting model for distributed training")

    # == global variables ==
    cfg_epochs = cfg.get("epochs", 1000)
    start_epoch = start_step = log_step = acc_step = 0
    running_loss = 0.0
    logger.info("Training for %s epochs with %s steps per epoch", cfg_epochs, num_steps_per_epoch)

    # == resume ==
    if cfg.get("load", None) is not None:
        logger.info("Loading checkpoint")
        ret = load(
            booster,
            cfg.load,
            model=model,
            ema=ema,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
        if not cfg.get("start_from_scratch", False):
            epoch_itr.load_state_dict(torch.load(os.path.join(cfg.load, "epoch_itr")))
            start_epoch, start_step = epoch_itr.epoch, epoch_itr.n
            # start_epoch, start_step = ret
            print(f"{start_epoch:}, {start_step:}, {ret[0]}, {ret[1]}")
        logger.info("Loaded checkpoint %s at epoch %s step %s", cfg.load, start_epoch, start_step)
    else:
        logger.info("Loading checkpoint from last checkpoint")
        import re
        def parse_folder_name(folder_name):
            match = re.match(r'epoch(\d+)-global_step(\d+)', folder_name)
            if match:
                epoch = int(match.group(1))
                step = int(match.group(2))
                return epoch, step
            return None, None
        def get_sorted_folders(base_path):
            folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
            parsed_folders = [(parse_folder_name(f), f) for f in folders]
            parsed_folders = [f for f in parsed_folders if f[0][0] is not None]
            sorted_folders = sorted(parsed_folders, key=lambda x: (x[0][0], x[0][1]), reverse=True)
            return [f[1] for f in sorted_folders]
        sorted_folders = get_sorted_folders(exp_dir)        
        for folder in sorted_folders:
            ckpt_dir = os.path.join(exp_dir, folder)
            try:
                ret = load(
                    booster,
                    ckpt_dir,
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                )
                if not cfg.get("start_from_scratch", False):
                    epoch_itr.load_state_dict(torch.load(os.path.join(ckpt_dir, "epoch_itr")))
                    start_epoch, start_step = epoch_itr.epoch, epoch_itr.n
                    # start_epoch, start_step = ret
                    print(f"{start_epoch:}, {start_step:}, {ret[0]}, {ret[1]}")
                logger.info("Loaded checkpoint %s at epoch %s step %s", ckpt_dir, start_epoch, start_step)
                break
            except Exception as e:
                logger.info(f"Failed to load checkpoint from {ckpt_dir}: {e}")
                continue

    model_sharding(ema)

    # =======================================================
    # 5. training loop
    # =======================================================
    dist.barrier()
    for epoch in range(start_epoch, cfg_epochs):
        # == set dataloader to new epoch ==
        dataloader_iter = epoch_itr.next_epoch_itr()
        logger.info("Beginning epoch %s...", epoch)

        # == training loop in an epoch ==
        with tqdm(
            enumerate(dataloader_iter, start=start_step),
            desc=f"Epoch {epoch}",
            disable=not coordinator.is_master(),
            initial=start_step,
            total=num_steps_per_epoch,
        ) as pbar:
            for step, batch in pbar:
                batch, ignore_grad = _prepare_sample(batch, dummy_batch)

                if ignore_grad:
                    print(step, "ignore grad for this dummy batch")

                global_step = epoch * num_steps_per_epoch + step
                set_seed(cfg.get("seed", 1024) + global_step)

                audio = batch["audios"].to(device, dtype)  # [B, T, C]
                audio_length = batch["audio_lengths"].to(device)  # [B]

                if batch.get("videos", None) is not None:
                    video = batch["videos"].to(device, dtype)  # [B, T, C]
                    video_length = batch["video_lengths"].to(device)  # [B]
                else:
                    video = None

                if batch.get("contents", None) is not None:
                    content = batch["contents"].to(device, torch.int32)  # [B, T, C]
                    content_length = batch["content_lengths"].to(device)  # [B]
                else:
                    content = None
                if batch.get("pitchs", None) is not None:
                    pitch = batch["pitchs"].to(device, torch.int32)  # [B, T, C]
                    pitch_length = batch["pitch_lengths"].to(device)  # [B]
                else:
                    pitch = None
                if batch.get("speaker", None) is not None:
                    speaker = batch["speaker"].to(device, dtype)
                else:
                    speaker = None

                model_args = {}
                # == visual and content encoding ==
                with torch.no_grad():
                    x = audio
                    model_args["x_mask_for_padding"] = lengths_to_mask(audio_length)
                    if video is not None:
                        model_args["video"] = video
                        model_args["video_mask_for_padding"] = lengths_to_mask(video_length)
                    if content is not None:
                        model_args["content"] = content.squeeze(1)
                        model_args["content_mask_for_padding"] = lengths_to_mask(content_length)
                    if pitch is not None:
                        model_args["pitch"] = pitch.squeeze(1)
                        model_args["pitch_mask_for_padding"] = lengths_to_mask(pitch_length)
                    if speaker is not None:
                        model_args["speaker"] = speaker

                # == diffusion loss computation ==
                loss_dict = scheduler.training_losses(model, x, model_args)

                # == backward & update ==
                loss = loss_dict["loss"].mean()
                if ignore_grad:
                    loss *= 0
                booster.backward(loss=loss, optimizer=optimizer)
                optimizer.step()
                optimizer.zero_grad()

                # update learning rate
                if lr_scheduler is not None:
                    lr_scheduler.step()

                # == update EMA ==
                update_ema(ema, model.module, optimizer=optimizer, decay=cfg.get("ema_decay", 0.9999))

                # == update log info ==
                all_reduce_mean(loss)
                if not ignore_grad:
                    running_loss += loss.item()
                    log_step += 1
                    acc_step += 1

                # == logging ==
                if coordinator.is_master() and (global_step + 1) % cfg.get("log_every", 1) == 0:
                    if log_step > 0:
                        avg_loss = running_loss / log_step
                        # progress bar
                        pbar.set_postfix({"loss": avg_loss, "step": step, "global_step": global_step})
                        # tensorboard
                        tb_writer.add_scalar("loss", loss.item(), global_step)
                        tb_writer.add_scalar("acc_step", acc_step, global_step)
                        tb_writer.add_scalar("epoch", epoch, global_step)
                        tb_writer.add_scalar("avg_loss", avg_loss, global_step)
                        tb_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)

                    running_loss = 0.0
                    log_step = 0

                # == checkpoint saving ==
                ckpt_every = cfg.get("ckpt_every", 0)
                if ckpt_every > 0 and (global_step + 1) % ckpt_every == 0:
                    model_gathering(ema, ema_shape_dict)
                    save_dir = save(
                        booster,
                        exp_dir,
                        model=model,
                        ema=ema,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        epoch=epoch,
                        step=step + 1,
                        global_step=global_step + 1,
                        batch_size=cfg.get("batch_size", None),
                    )
                    torch.save(epoch_itr.state_dict(), os.path.join(save_dir, "epoch_itr"))
                    if dist.get_rank() == 0:
                        model_sharding(ema)
                    logger.info(
                        "Saved checkpoint at epoch %s, step %s, global_step %s to %s",
                        epoch,
                        step + 1,
                        global_step + 1,
                        save_dir,
                    )

                if global_step == 0:
                    torch.cuda.empty_cache()

        start_step = 0

if __name__ == "__main__":
    main()
