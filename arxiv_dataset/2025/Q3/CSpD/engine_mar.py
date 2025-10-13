import os
import math
import sys
from pathlib import Path
from typing import Iterable, Optional

import torch
from accelerate import Accelerator
from accelerate.utils import reduce

import util.misc as misc
from models.vae import DiagonalGaussianDistribution
import torch_fidelity
import shutil
import cv2
import numpy as np
import time


@torch.no_grad()
def update_ema(ema_model, model, rate=0.99):
    """
    Update EMA parameters to be closer to those of model parameters using
    an exponential moving average.

    :param ema_model: the EMA model.
    :param model: the model.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(ema_model.parameters(), model.parameters()):
        targ.mul_(rate).add_(src, alpha=1 - rate)


def train_one_epoch(model, vae,
                    ema_model,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler.LRScheduler,
                    epoch: int,
                    accelerator: Accelerator,
                    args=None):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    for data_iter_step, (samples, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        with torch.no_grad():
            if args.use_cached:
                moments = samples
                posterior = DiagonalGaussianDistribution(moments)
            else:
                posterior = vae.encode(samples)

            # normalize the std of latent to be 1. Change it if you use a different tokenizer
            x = posterior.sample().mul_(0.2325)

        # forward
        with accelerator.autocast():
            loss = model(x, labels)

        loss_value = loss.item()
        accelerator.backward(loss)

        if not math.isfinite(loss_value):
            raise ValueError(f"Loss is {loss_value} on process {accelerator.local_process_index}, stopping training")

        # loss_scaler(loss, optimizer, clip_grad=args.grad_clip, parameters=model.parameters(), update_grad=True)
        accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        accelerator.wait_for_everyone()
        update_ema(ema_model, model, rate=args.ema_rate)
        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = reduce(loss).item()
        epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
        accelerator.log({'train_loss': loss_value_reduce,
                         'lr': lr},
                        step=epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    accelerator.print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(model, vae, ema_model, args, epoch, batch_size, accelerator: Optional[Accelerator], cfg=1.0,
             use_ema=True):
    model.eval()
    num_steps = args.num_images // (batch_size * accelerator.num_processes) + 1
    save_folder = str(Path(
        args.output_dir) / f"ariter{args.num_iter}-diffsteps{args.num_sampling_steps}-temp{args.temperature}-{args.cfg_schedule}cfg{cfg}-image{args.num_images}")
    if use_ema:
        save_folder = save_folder + "_ema"
    if args.evaluate:
        save_folder = save_folder + "_evaluate"
    accelerator.print("Save to:", save_folder)
    if accelerator.is_main_process:
        Path(save_folder).mkdir(exist_ok=True, parents=True)

    # switch to ema params
    if use_ema:
        model_state_dict = model.state_dict()
        accelerator.print("Switch to ema")
        model.load_state_dict(ema_model.state_dict())

    class_num = args.class_num
    assert args.num_images % class_num == 0, "number of images per class must be the same"
    class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])
    world_size = accelerator.num_processes
    local_rank = accelerator.local_process_index
    used_time = 0
    gen_img_cnt = 0

    for i in range(num_steps):
        accelerator.print(f"Generation step {i}/{num_steps}")

        labels_gen = class_label_gen_world[world_size * batch_size * i + local_rank * batch_size:
                                           world_size * batch_size * i + (local_rank + 1) * batch_size]
        labels_gen = torch.Tensor(labels_gen).long().cuda()

        accelerator.wait_for_everyone()
        start_time = time.time()

        # generation
        with torch.inference_mode():
            with accelerator.autocast():
                sampled_tokens = model.sample_tokens(bsz=batch_size, num_iter=args.num_iter, cfg=cfg,
                                                     cfg_schedule=args.cfg_schedule, labels=labels_gen,
                                                     temperature=args.temperature)
                sampled_images = vae.decode(sampled_tokens / 0.2325)

        # measure speed after the first generation batch
        if i >= 1:
            accelerator.wait_for_everyone()
            used_time += time.time() - start_time
            gen_img_cnt += batch_size
            accelerator.print(
                f"Generating {gen_img_cnt} images takes {used_time:.5f} seconds, {used_time / gen_img_cnt:.5f} sec per image")

        accelerator.wait_for_everyone()
        sampled_images = sampled_images.detach().cpu()
        sampled_images = (sampled_images + 1) / 2

        # distributed save
        for b_id in range(sampled_images.size(0)):
            img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
            if img_id >= args.num_images:
                break
            gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(str(Path(save_folder) / f'{str(img_id).zfill(5)}.png'), gen_img)

    accelerator.wait_for_everyone()
    time.sleep(10)

    # back to no ema
    if use_ema:
        accelerator.print("Switch back from ema")
        model.load_state_dict(model_state_dict)

    # compute FID and IS
    if accelerator is not None:
        if args.img_size == 256:
            input2 = None
            fid_statistics_file = 'fid_stats/adm_in256_stats.npz'
        else:
            raise NotImplementedError
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=input2,
            fid_statistics_file=fid_statistics_file,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        postfix = ""
        if use_ema:
            postfix = postfix + "_ema"
        if cfg != 1.0:
            postfix = postfix + "_cfg{}".format(cfg)
        accelerator.log(
            {f"fid{postfix}": fid,
             f"is{postfix}": inception_score, },
            step=epoch
        )
        accelerator.print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))

    accelerator.wait_for_everyone()
    # remove temporal saving folder
    if accelerator.is_main_process:
        shutil.rmtree(save_folder)
    time.sleep(10)


def cache_latents(vae,
                  data_loader: Iterable,
                  device: torch.device,
                  args=None):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Caching: '
    print_freq = 20

    for data_iter_step, (samples, _, paths) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        samples = samples.to(device, non_blocking=True)

        with torch.no_grad():
            posterior = vae.encode(samples)
            moments = posterior.parameters
            posterior_flip = vae.encode(samples.flip(dims=[3]))
            moments_flip = posterior_flip.parameters

        for i, path in enumerate(paths):
            save_path = os.path.join(args.cached_path, path + '.npz')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(str(save_path), moments=moments[i].cpu().numpy(), moments_flip=moments_flip[i].cpu().numpy())

        if misc.is_dist_avail_and_initialized():
            torch.cuda.synchronize()

    return
