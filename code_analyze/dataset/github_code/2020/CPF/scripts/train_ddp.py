import os
import random

import cv2
from argparse import Namespace
from time import time

import numpy as np
import torch
from lib.models.model_abc import ModelABC
from lib.utils.config import get_config
from lib.datasets.hdata import ho_data_collate
from lib.opt import parse_exp_args
from lib.utils import builder
from lib.utils.etqdm import etqdm
from lib.utils.logger import logger
from lib.utils.misc import CONST, bar_prefix, format_args_cfg
from lib.utils.net_utils import clip_gradient, setup_seed, worker_init_fn, build_optimizer
from lib.utils.recorder import Recorder
from lib.utils.summary_writer import DDPSummaryWriter
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from yacs.config import CfgNode as CN
from lib.viztools.viz_o3d_utils import VizContext
from lib.datasets.mix_dataset import MixDataset


def setup_ddp(arg, rank, world_size):
    """Setup distributed data parallels

    Args:
        arg (Namespace): arguments
        rank (int): rank of current process
        world_size (int): total number of processes, equal to number of GPUs
    """
    os.environ["MASTER_ADDR"] = arg.dist_master_addr
    os.environ["MASTER_PORT"] = arg.dist_master_port
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    assert rank == torch.distributed.get_rank(), "Something wrong with DDP setup"
    torch.cuda.set_device(rank)
    dist.barrier()


def main_worker(rank: int, cfg: CN, arg: Namespace, world_size, time_f: float):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    cv2.setNumThreads(0)

    setup_ddp(arg, rank, world_size)
    setup_seed(rank + cfg.TRAIN.MANUAL_SEED, cfg.TRAIN.CONV_REPEATABLE)
    recorder = Recorder(arg.exp_id, cfg, rank=rank, time_f=time_f, root_path="exp")
    summary = DDPSummaryWriter(log_dir=recorder.tensorboard_path, rank=rank)

    dist.barrier()  # wait for recoder to finish setup
    if cfg.DATASET.TRAIN.TYPE == "MIX":
        trainset = MixDataset(cfg.DATASET.TRAIN.MIX,
                              length=cfg.DATASET.TRAIN.LENGTH,
                              data_preset=cfg.DATA_PRESET,
                              transform=cfg.DATASET.TRAIN.TRANSFORM)
    else:
        trainset = builder.build_dataset(cfg.DATASET.TRAIN, data_preset=cfg.DATA_PRESET)

    train_sampler = DistributedSampler(trainset, shuffle=True)
    train_loader = DataLoader(trainset,
                              batch_size=arg.batch_size // world_size,
                              shuffle=(train_sampler is None),
                              num_workers=int(arg.workers),
                              pin_memory=True,
                              drop_last=True,
                              sampler=train_sampler,
                              worker_init_fn=worker_init_fn,
                              collate_fn=ho_data_collate,
                              persistent_workers=(int(arg.workers) > 0))

    if rank == 0 and cfg.DATASET.get("VAL") is not None:
        if cfg.DATASET.VAL.TYPE == "MIX":
            valset = MixDataset(cfg.DATASET.VAL.MIX,
                                length=cfg.DATASET.TRAIN.LENGTH,
                                data_preset=cfg.DATA_PRESET,
                                transform=cfg.DATASET.VAL.TRANSFORM)
        else:
            valset = builder.build_dataset(cfg.DATASET.VAL, data_preset=cfg.DATA_PRESET)
        val_loader = DataLoader(valset,
                                batch_size=arg.val_batch_size,
                                shuffle=True,
                                num_workers=int(arg.workers),
                                drop_last=False,
                                worker_init_fn=worker_init_fn,
                                collate_fn=ho_data_collate)
    else:
        val_loader = None

    model: ModelABC = builder.build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET, train=cfg.TRAIN)

    model.setup(summary_writer=summary, log_freq=arg.log_freq)
    model.to(rank)

    model = DDP(model, device_ids=[rank], find_unused_parameters=cfg.TRAIN.FIND_UNUSED_PARAMETERS)
    n_steps = len(train_loader) * cfg.TRAIN.EPOCH

    optimizer = build_optimizer(model.parameters(), cfg=cfg.TRAIN)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-7)
    start_epoch = 0
    if arg.resume:
        start_epoch = recorder.resume_checkpoints(model, optimizer, scheduler, arg.resume, arg.resume_epoch)

    dist.barrier()  # wait for all processes to finish loading model
    logger.warning(f"===== start training from [{start_epoch}, {cfg.TRAIN.EPOCH}), "
                   f"total iters: {len(train_loader) * (cfg.TRAIN.EPOCH - start_epoch)}/{n_steps} >>>>")

    for epoch_idx in range(start_epoch, cfg.TRAIN.EPOCH):
        train_sampler.set_epoch(epoch_idx)
        model.train()
        trainbar = etqdm(train_loader, rank=rank)
        for bidx, batch in enumerate(trainbar):
            optimizer.zero_grad()
            step_idx = epoch_idx * len(train_loader) + bidx
            prd, loss_dict = model(batch, step_idx, "train", epoch_idx=epoch_idx)
            loss = loss_dict["loss"]
            loss.backward()
            if cfg.TRAIN.GRAD_CLIP_ENABLED:
                clip_gradient(optimizer, cfg.TRAIN.GRAD_CLIP.NORM, cfg.TRAIN.GRAD_CLIP.TYPE)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            trainbar.set_description(f"{bar_prefix['train']} Epoch {epoch_idx} | {loss.item():.4f}")

        dist.barrier()  # wait for all processes to finish training
        logger.info(f"{bar_prefix['train']} Epoch {epoch_idx} | loss: {loss.item():.4f}, Done")
        recorder.record_checkpoints(model, optimizer, scheduler, epoch_idx, arg.snapshot)
        model.module.on_train_finished(recorder, epoch_idx)

        if (rank == 0  # only at rank 0,
                and epoch_idx != 0  # not the first epoch
                and epoch_idx != cfg.TRAIN.EPOCH - 1  # not the last epoch
                and epoch_idx % arg.eval_freq == 0  # at eval freq, do validation
                and cfg.DATASET.get("VAL") is not None):
            logger.info("do validation and save results")
            with torch.no_grad():
                model.eval()
                valbar = etqdm(val_loader, rank=rank)
                for bidx, batch in enumerate(valbar):
                    step_idx = epoch_idx * len(val_loader) + bidx
                    prd, eval_dict = model(batch, step_idx, "val", epoch_idx=epoch_idx)

            model.module.on_val_finished(recorder, epoch_idx)

    dist.destroy_process_group()

    # do last evaluation
    if rank == 0 and cfg.DATASET.get("VAL") is not None:
        logger.info("do last validation and save results")
        with torch.no_grad():
            model.eval()
            valbar = etqdm(val_loader, rank=rank)
            for bidx, batch in enumerate(valbar):
                step_idx = epoch_idx * len(val_loader) + bidx
                prd, eval_dict = model(batch, step_idx, "val", epoch_idx=epoch_idx)

        model.module.on_val_finished(recorder, epoch_idx)


if __name__ == "__main__":
    exp_time = time()
    arg, _ = parse_exp_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu_id
    world_size = torch.cuda.device_count()

    if arg.resume:
        logger.warning(f"config will be reloaded from {os.path.join(arg.resume, 'dump_cfg.yaml')}")
        arg.cfg = os.path.join(arg.resume, "dump_cfg.yaml")
        cfg = get_config(config_file=arg.cfg, arg=arg)
    else:
        cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)

    if arg.reload is not None:
        logger.warning(f"cfg MODEL's pretrained {cfg.MODEL.PRETRAINED} reset to arg.reload: {arg.reload}")
        cfg.MODEL.PRETRAINED = arg.reload

    logger.warning(f"final args and cfg: \n{format_args_cfg(arg, cfg)}")
    logger.info("====> Use Distributed Data Parallel <====")
    mp.spawn(main_worker, args=(cfg, arg, world_size, exp_time), nprocs=world_size)
