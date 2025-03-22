import os
import random
import warnings
from argparse import Namespace
from time import time

import numpy as np
import torch
from torch.nn.parallel import DataParallel as DP
from torch.utils.data import DataLoader

from lib.datasets import create_dataset
from lib.opt import parse_exp_args
from lib.datasets.hdata import ho_data_collate
from lib.utils import builder
from lib.utils.config import CN, get_config
from lib.utils.etqdm import etqdm
from lib.utils.logger import logger
from lib.utils.misc import CONST, bar_prefix, format_args_cfg
from lib.utils import callbacks
from lib.utils.net_utils import setup_seed, worker_init_fn
from lib.utils.recorder import Recorder
from lib.utils.summary_writer import DDPSummaryWriter


def main_worker(cfg: CN, arg: Namespace, time_f: float):
    rank = 0  # only one process.
    setup_seed(cfg.FIT.MANUAL_SEED, cfg.FIT.CONV_REPEATABLE)

    if arg.exp_id != 'default':
        warnings.warn("You shouldn't assign exp_id in test mode")
    cfg_name = arg.cfg.split("/")[-1].split(".")[0]
    exp_id = f"eval_{cfg_name}"

    recorder = Recorder(exp_id, cfg, rank=rank, time_f=time_f, eval_only=True)
    summary = DDPSummaryWriter(log_dir=recorder.tensorboard_path, rank=rank)
    test_data = create_dataset(cfg.DATASET.TEST, data_preset=cfg.DATA_PRESET)
    test_loader = DataLoader(test_data,
                             batch_size=arg.batch_size,
                             shuffle=False,
                             num_workers=int(arg.workers),
                             drop_last=False,
                             worker_init_fn=worker_init_fn,
                             collate_fn=ho_data_collate)

    model = builder.build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET)
    model.setup(summary_writer=summary, log_freq=arg.log_freq)
    model = DP(model).to(device=rank)

    if cfg.get("CALLBACKS") is not None:
        if arg.callback == "draw":
            cb_fn = builder.build_callback(cfg.CALLBACKS.DRAW)
        elif arg.callback == "dump":
            cb_fn = builder.build_callback(cfg.CALLBACKS.DUMP)
    else:
        cb_fn = None

    with torch.no_grad():
        model.eval()
        testbar = etqdm(test_loader, rank=rank)
        for bidx, batch in enumerate(testbar):
            step_idx = 0 * len(test_loader) + bidx
            preds = model(batch, step_idx, "inference", callback=cb_fn)
            testbar.set_description(f"{bar_prefix['test']}")

        # model.module.on_test_finished(recorder, 0)
        cb_fn.on_finished()  # deal with the callback results


if __name__ == "__main__":
    exp_time = time()
    arg, _ = parse_exp_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu_id
    world_size = torch.cuda.device_count()

    cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)

    if arg.reload is not None:
        logger.warning(f"cfg MODEL's pretrained {cfg.MODEL.PRETRAINED} reset to arg.reload: {arg.reload}")
        cfg.MODEL.PRETRAINED = arg.reload

    logger.warning(f"final args and cfg: \n{format_args_cfg(arg, cfg)}")
    # input("Confirm (press enter) ?")

    logger.info("====> Evaluation on single GPU (Data Parallel) <====")
    main_worker(cfg, arg, exp_time)
