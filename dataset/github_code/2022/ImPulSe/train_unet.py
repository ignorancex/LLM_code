import os
import random
from argparse import ArgumentParser
from datetime import datetime
from importlib import import_module
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from data.dataset import LungSegmentDataset
from data import transforms as aug
from models.deeplab import DeepLab3d, DeepLabDecoder3d
from models.fcn import FCN3d, FCNDecoder3d
from models.losses import SegLoss
from models.resnet18 import ResNet3d18Backbone
from models.unet import UNet, UNetDecoder
from utils.logger import logger
from utils.metrics import foreground_dice_score


def _parse_cmd_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--gpu", default="0,1,2,3", help="GPU ID.")
    arg_parser.add_argument("--cfg", required=True,
        help="Python config module.")
    arg_parser.add_argument("--data_dir", required=True,
        help="Data directory")
    arg_parser.add_argument("--df_path", required=True,
        help="Data info csv path.")
    arg_parser.add_argument("--log_dir", required=True,
        help="Tensorboard log directory.")
    args = arg_parser.parse_args()

    return args


def _set_rng_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _init_dataloaders(args):
    data_dir = args.data_dir
    df = pd.read_csv(args.df_path)

    transforms_train = [
        aug.Resample(cfg.resample_cfgs),
        aug.SampleGrid(cfg.in_res, "regular"),
        aug.MinMaxNormalize(cfg.win_min, cfg.win_max),
        aug.ConcatInputs(cfg.input_keys),
        aug.ToTensor()
    ]
    transforms_val = [
        aug.Resample(cfg.resample_cfgs),
        aug.SampleGrid(cfg.in_res, "regular"),
        aug.MinMaxNormalize(cfg.win_min, cfg.win_max),
        aug.ConcatInputs(cfg.input_keys),
        aug.ToTensor()
    ]
    ds_train = LungSegmentDataset(df, data_dir, transforms_train, "train")
    dl_train = LungSegmentDataset.get_dataloader(ds_train, cfg.batch_size,
        True, cfg.num_workers, "unet")
    ds_val = LungSegmentDataset(df, data_dir, transforms_val, "val")
    dl_val = LungSegmentDataset.get_dataloader(ds_val, cfg.batch_size, False,
        cfg.num_workers, "unet")

    return dl_train, dl_val


def _init_model(args):
    encoder = ResNet3d18Backbone(**cfg.enc_cfgs)

    if args.cfg in ["unet", "coord"]:
        decoder = UNetDecoder(**cfg.dec_cfgs)
        model = UNet(encoder, decoder)
    elif args.cfg == "fcn":
        decoder = FCNDecoder3d(**cfg.dec_cfgs)
        model = FCN3d(encoder, decoder)
    elif args.cfg == "deeplab":
        decoder = DeepLabDecoder3d(**cfg.dec_cfgs)
        model = DeepLab3d(encoder, decoder)

    devices = [int(x) for x in args.gpu.split(",")]
    if len(devices) > 1:
        model = nn.DataParallel(model.cuda(), devices)
    else:
        model = model.cuda()

    return model


@logger
def _train_epoch(model, dataloader, criterion, optimizer, scheduler):
    model.train()
    loss_train = 0
    dice_train = 0

    for i, sample in enumerate(dataloader):
        optimizer.zero_grad()

        inputs, targets = sample
        inputs = inputs.cuda()
        targets = targets.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            loss_train += loss.detach().cpu().item()
            y_true = targets.detach().cpu().numpy()
            y_pred = outputs.detach().argmax(dim=1).cpu().numpy()
            dice_train += foreground_dice_score(y_true, y_pred,
                cfg.dec_cfgs["num_classes"])

    loss_train /= len(dataloader)
    dice_train /= len(dataloader)

    results = {
        "loss": loss_train,
        "dice": dice_train
    }

    return results


@logger
@torch.no_grad()
def _eval_epoch(model, dataloader, criterion):
    model.eval()
    loss_val = 0
    dice_val = 0

    for i, sample in enumerate(dataloader):
        inputs, targets = sample
        inputs = inputs.cuda()
        targets = targets.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss_val += loss.detach().cpu().item()
        y_true = targets.cpu().numpy()
        y_pred = outputs.argmax(dim=1).cpu().numpy()
        dice_val += foreground_dice_score(y_true, y_pred,
            cfg.dec_cfgs["num_classes"])

    loss_val /= len(dataloader)
    dice_val /= len(dataloader)

    results = {
        "loss": loss_val,
        "dice": dice_val
    }

    return results


def _log_metrics(results_train, results_val):
    metrics = {"train": results_train, "val": results_val}
    metrics = pd.DataFrame(metrics)
    print(metrics)


def _log_tensorboard(tb_writer, epoch, results_train, results_val):
    for k in results_train.keys():
        tb_writer.add_scalars(k, {"train": results_train[k],
            "val": results_val[k]}, epoch)

    tb_writer.flush()


def main():
    _set_rng_seed(42)

    args = _parse_cmd_args()
    torch.cuda.set_device(int(args.gpu.split(",")[0]))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    global cfg
    cfg = import_module(f"configs.{args.cfg}_config")
    print(args.cfg)

    dl_train, dl_val = _init_dataloaders()

    model = _init_model(args)
    criterion = SegLoss(cfg.w_ce, cfg.w_dice, cfg.dec_cfgs["num_classes"])
    optimizer = optim.AdamW(model.parameters(), cfg.max_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
        len(dl_train) * cfg.epochs, cfg.min_lr)

    # set up tensorboard
    log_dir = args.log_dir
    cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(cur_time)
    log_dir = os.path.join(log_dir, cur_time)
    time_train = 0

    for i in range(cfg.epochs):
        print(f"Epoch {i}")
        epoch_start = perf_counter()
        res_train = _train_epoch(model, dl_train, criterion,
            optimizer, scheduler)
        time_train += (perf_counter() - epoch_start)

        if (i + 1) % cfg.eval_freq == 0:
            res_val = _eval_epoch(model, dl_val, criterion)
            _log_metrics(res_train, res_val)

            torch.save(model.state_dict(), os.path.join(log_dir,
                f"model_{i}.pth"))

    print(f"Total training time: {time_train:.4f}")


if __name__ == "__main__":
    main()
