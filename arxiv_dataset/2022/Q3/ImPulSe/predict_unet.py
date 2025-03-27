import os
import random
from argparse import ArgumentParser
from importlib import import_module

import SimpleITK as sitk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from data.dataset import LungSegmentDataset
from data import transforms as aug
from models.deeplab import DeepLab3d, DeepLabDecoder3d
from models.fcn import FCN3d, FCNDecoder3d
from models.resnet18 import ResNet3d18Backbone
from models.unet import UNet, UNetDecoder
from utils.logger import logger


def _parse_cmd_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--gpu", default="0,1,2,3", help="GPU ID.")
    arg_parser.add_argument("--cfg", required=True,
        help="Python config module.")
    arg_parser.add_argument("--data_dir", required=True,
        help="Data directory")
    arg_parser.add_argument("--df_path", required=True,
        help="Data info csv path.")
    arg_parser.add_argument("--weight_path", required=True,
        help="Train model weight path.")
    arg_parser.add_argument("--output_dir", required=True,
        help="Prediction output directory.")
    args = arg_parser.parse_args()

    return args


def _set_rng_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _init_dataloaders(args):
    data_dir = args.data_dir
    df = pd.read_csv(args.df_path)

    transforms = [
        aug.CropLung(),
        aug.Resample(cfg.resample_cfgs),
        aug.SampleGrid(cfg.in_res, "regular"),
        aug.MinMaxNormalize(cfg.win_min, cfg.win_max),
        aug.ConcatInputs(cfg.input_keys),
        aug.ToTensor()
    ]

    ds_val = LungSegmentDataset(df, data_dir, transforms, "val")
    dl_val = LungSegmentDataset.get_dataloader(ds_val, cfg.eval_batch_size,
        False, cfg.num_workers, "unet_infer")
    ds_test = LungSegmentDataset(df, data_dir, transforms, "test")
    dl_test = LungSegmentDataset.get_dataloader(ds_test, cfg.eval_batch_size,
        False, cfg.num_workers, "unet_infer")

    return dl_val, dl_test


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

    devices = [torch.device(f"cuda:{i}") for i in args.gpu.split(",")]
    if len(devices) > 1:
        model = nn.DataParallel(model.cuda(), devices)
    else:
        model = model.cuda()

    return model


@logger
@torch.no_grad()
def _predict(model, dataloader, output_dir, subset, interpolation):
    model.eval()
    progress = tqdm(total=len(dataloader))
    os.makedirs(os.path.join(output_dir, subset), exist_ok=True)

    for i, sample in enumerate(dataloader):
        inputs, _, pids, bboxes, original_shapes = sample
        inputs = inputs.cuda()

        y_prob_lung = model(inputs).cpu().numpy()
        n_classes = y_prob_lung.shape[1]
        y_pred_lung = np.argmax(y_prob_lung, axis=1).astype(np.uint8)
        batch_size = y_pred_lung.shape[0]
        for j in range(batch_size):
            y_pred = np.zeros(original_shapes[j], dtype=np.uint8)
            lung_shape = tuple((np.diff(bboxes[j], axis=-1) + 1).squeeze())
            lung_shape = tuple([int(x) for x in lung_shape])
            d, h, w = y_pred_lung[j].shape
            y_pred_lung_onehot = np.eye(n_classes)[y_pred_lung[j].reshape(-1)]
            y_pred_lung_onehot = y_pred_lung_onehot.reshape((d, h,
                w, n_classes)).transpose((3, 0, 1, 2))
            pred_lung = np.argmax(np.stack([aug._resample_to_shape(
                y_pred_lung_onehot[c], lung_shape, interpolation)
                for c in range(n_classes)]), axis=0)
            y_pred[
                bboxes[j, 0, 0]:bboxes[j, 0, 1] + 1,
                bboxes[j, 1, 0]:bboxes[j, 1, 1] + 1,
                bboxes[j, 2, 0]:bboxes[j, 2, 1] + 1
            ] = pred_lung

            y_pred_img = sitk.GetImageFromArray(y_pred)
            sitk.WriteImage(y_pred_img, os.path.join(output_dir, subset,
                f"{pids[j]}_pred.nii.gz"))

        progress.update()

    progress.close()


def _load_weights(weight_path):
    raw_weights = torch.load(weight_path)
    weights = {}
    for k in raw_weights.keys():
        new_k = k.replace("module.", "")
        weights[new_k] = raw_weights[k]

    return weights


def main():
    _set_rng_seed(42)

    args = _parse_cmd_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    global cfg
    cfg = import_module(f"configs.{args.cfg}_config")

    dl_val, dl_test = _init_dataloaders()

    model = _init_model(args)
    model_weights = _load_weights(args.weight_path)
    model.load_state_dict(model_weights)

    os.makedirs(args.output_dir, exist_ok=True)
    interp = "linear"
    _predict(model, dl_val, args.output_dir, "val", interp)
    _predict(model, dl_test, args.output_dir, "test", interp)


if __name__ == "__main__":
    main()
