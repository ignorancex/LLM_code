# 2025-FEB-12 Emerson P Grabke
# Taken from the MONAI distributed training (unet_training_ddp.py) script
# Goal: Train and evaluate a ResNet based on specific flags

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

from torch import nn
import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel

import monai
from monai.data import ThreadDataLoader, partition_dataset
from monai.transforms import Compose, MapTransform
from monai.utils import first
from monai.losses import FocalLoss
from monai.networks.nets import DenseNet121, EfficientNetBN

from .diff_model_setting import initialize_distributed, load_config, setup_logging
from .utils import define_instance
import numpy as np
import pandas as pd

import io
import pandas as pd
import pickle

from glob import glob

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import monai
from monai.data import DataLoader, Dataset, DistributedSampler, CacheDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityRangePercentilesd,
    RandFlipd,
    RandRotated,
    RandAffined,
    RandZoomd,
    Activations,
    AsDiscrete,
    CropForegroundd,
    SpatialPadd,
    CenterSpatialCropd,
)
from torch.utils.tensorboard import SummaryWriter


def distributed_all_gather(
    tensor_list, valid_batch_size=None, out_numpy=False, world_size=None, no_barrier=False, is_valid=None
):
    if world_size is None:
        world_size = torch.distributed.get_world_size()
    if valid_batch_size is not None:
        valid_batch_size = min(valid_batch_size, world_size)
    elif is_valid is not None:
        is_valid = torch.tensor(bool(is_valid), dtype=torch.bool, device=tensor_list[0].device)
    if not no_barrier:
        torch.distributed.barrier()
    tensor_list_out = []
    with torch.no_grad():
        if is_valid is not None:
            is_valid_list = [torch.zeros_like(is_valid) for _ in range(world_size)]
            torch.distributed.all_gather(is_valid_list, is_valid)
            is_valid = [x.item() for x in is_valid_list]
        for tensor in tensor_list:
            gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(gather_list, tensor)
            if valid_batch_size is not None:
                gather_list = gather_list[:valid_batch_size]
            elif is_valid is not None:
                gather_list = [g for g, v in zip(gather_list, is_valid_list) if v]
            if out_numpy:
                gather_list = [t.cpu().numpy() for t in gather_list]
            tensor_list_out.append(gather_list)
    return tensor_list_out

def save_model(model, epoch, optimizer, scheduler, val_metric, path):
    if isinstance(model, DistributedDataParallel):
        model = model.module
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_metric': val_metric,
    }, path)

def load_model(model, optimizer, scheduler, path):
    checkpoint = torch.load(path)
    if list(checkpoint['model_state_dict'].keys())[0].startswith("module"):
        for k in list(checkpoint['model_state_dict'].keys()):
            checkpoint['model_state_dict'][k[7:]] = checkpoint['model_state_dict'].pop(k)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    val_metric = checkpoint['val_metric']

    return model, optimizer, scheduler, epoch, val_metric

def train(args,synthsheet_path, real_impath, synth_impath):
    # disable logging for processes except 0 on every node

    cache_rate = 0.0
    
    # initialize the distributed training process, every GPU runs in a process
    dist.init_process_group(backend="nccl", init_method="env://")
    
    args.local_rank = dist.get_rank()
    if args.local_rank != 0:
        f = open(os.devnull, "w")
        sys.stdout = sys.stderr = f

    val_interval = 5

    args.exp_name = f"ds_enb0_r{int(args.real_data)}_s{int(args.synth_data)}_l{int(args.generated_label)}"

    if args.add_to_modelname:
        args.exp_name += f"_{args.add_to_modelname}"

    model_path = f"./models/{args.exp_name}.pth"

    if args.local_rank == 0:
        print(f"Experiment name: {args.exp_name}")
        tensorboard_path = os.path.join("./outputs/downstreamtfevent", args.exp_name)
        Path(tensorboard_path).mkdir(parents=True, exist_ok=True)
        tensorboard_writer = SummaryWriter(tensorboard_path)


    df_path = synthsheet_path
    df = pd.read_excel(df_path)

    train_df = df[df["Label"] == "train"]
    test_df = df[df["Label"] == "test"]

    assert args.real_data or args.synth_data, "Must specify at least one of real or synthetic data"
    

    real_folder = real_impath
    synth_folder = synth_impath

    train_files = []

    for idx, row in train_df.iterrows():
        folder = row["folder"]
        if args.real_data:
            img_path = os.path.join(real_folder, folder,f'{args.imkey}.nii.gz')
            train_files.append({"img": img_path, "lbl": row["PIRADS_real"]})

        if args.synth_data:
            img_path = os.path.join(synth_folder, folder,f'{args.imkey}.nii.gz')
            if args.generated_label:
                train_files.append({"img": img_path, "lbl": row["PIRADS_synth"]}) # Synthetic data with synthetic label
            else:
                train_files.append({"img": img_path, "lbl": row["PIRADS_real"]})
        
    val_files = []
    for idx, row in test_df.iterrows():
        folder = row["folder"]
        img_path = os.path.join(real_folder, folder,f'{args.imkey}.nii.gz')
        val_files.append({"img": img_path, "lbl": row["PIRADS_real"]})

    prob=0.5

    print(f"Train set size: {len(train_files)} and Val set size: {len(val_files)}")

    # define transforms for image
    train_transforms = Compose(
        [
            LoadImaged(keys=["img"], ensure_channel_first=True, image_only=True),
            # Crop foreground then pad to 256x256x32
            CropForegroundd(keys=["img"], source_key="img"),
            ScaleIntensityRangePercentilesd(keys=["img"],lower=0, upper=98.0, b_min=0.0, b_max=1.0, clip=True),
            SpatialPadd(keys=["img"], spatial_size=(256,256,32), mode="constant", constant_values=[0]),
            CenterSpatialCropd(keys=["img"], roi_size=(256,256,32)),
            RandFlipd(keys=["img"], prob=prob, spatial_axis=0), # 0:l/r, 1: ant/post, 2: sup/inf
            RandRotated(keys=["img"], range_z=15*3.14/180, prob=prob, mode=('bilinear')),
            RandAffined(keys=["img"], prob=prob, translate_range=(25,25,2), padding_mode='zeros', mode=('bilinear')), # 10% of 256x256x32
            RandZoomd(keys=["img"], prob=prob, min_zoom=0.9, max_zoom=1.1, mode=('bilinear'), padding_mode='constant',constant_values=[0]),
        ]
    )

    val_transforms = Compose([
        LoadImaged(keys=["img"], ensure_channel_first=True, image_only=True),
        CropForegroundd(keys=["img"], source_key="img"),
        ScaleIntensityRangePercentilesd(keys=["img"],lower=0, upper=98.0, b_min=0.0, b_max=1.0, clip=True),
        SpatialPadd(keys=["img"], spatial_size=(256,256,32), mode="constant", constant_values=[0]),
        CenterSpatialCropd(keys=["img"], roi_size=(256,256,32)),
    ])

    # create a training data loader
    train_ds = CacheDataset(data=train_files, transform=train_transforms, num_workers=args.num_workers, cache_rate=cache_rate)
    # create a training data sampler
    train_sampler = DistributedSampler(dataset=train_ds, even_divisible=True, shuffle=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    if args.local_rank == 0:
        val_ds = CacheDataset(data=val_files, transform=val_transforms, num_workers=2, cache_rate=0.0)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            sampler=None,
            shuffle=False
        )

    # create UNet, DiceLoss and Adam optimizer
    device = torch.device(f"cuda:{args.local_rank}")
    torch.cuda.set_device(device)

    model = EfficientNetBN(
        model_name="efficientnet-b0",
        pretrained=False,
        spatial_dims=3,
        in_channels=1,
        num_classes=2
    ).to(device)
    loss_function = FocalLoss(
        to_onehot_y=True,
        use_softmax=True,
        gamma=2.0, 
        weight=None, 
        reduction="mean"
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), args.lr)
    total_steps = (args.num_epochs * len(train_loader.dataset)) / (args.batch_size*dist.get_world_size())
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=2.0)

    if args.resume:
        model, optimizer, scheduler, start_epoch, val_metric = load_model(model, optimizer, scheduler, model_path)
    else:
        start_epoch = 0
        val_metric = 0

    # wrap the model with DistributedDataParallel module
    model = DistributedDataParallel(model, device_ids=[device])

    scaler = GradScaler(device=device)

    # start a typical PyTorch training
    for epoch in range(start_epoch, args.num_epochs):
        # print("-" * 10)
        print(f"epoch {epoch + 1}/{args.num_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        train_sampler.set_epoch(epoch)
        for batch_data in (pbar:=tqdm(train_loader, disable=args.local_rank != 0)):
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["lbl"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=args.autocast):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)

            if args.autocast:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            scheduler.step()
        
            epoch_loss += loss.item()

            pbar.set_postfix({"loss": epoch_loss/step,"lr": scheduler.get_last_lr()[0]})
        epoch_loss /= step

        epoch_loss = torch.tensor(epoch_loss).to(device)
        dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
        epoch_loss = epoch_loss.item() / dist.get_world_size()
        if args.local_rank == 0:
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
            tensorboard_writer.add_scalar("train_loss", epoch_loss, epoch + 1)
            # Log current LR
            lr = scheduler.get_last_lr()[0]
            tensorboard_writer.add_scalar("learning_rate", lr, epoch + 1)

        if (epoch + 1) % val_interval == 0:
            # Only eval on rank 0
            model.eval()
            if args.local_rank == 0:
                with torch.no_grad():
                    epoch_val_loss = 0
                    step = 0
                    val_labels_all = torch.tensor([], dtype=torch.int64, device=device)
                    val_outputs_all = torch.tensor([], dtype=torch.int64, device=device)
                    for val_data in tqdm(val_loader):
                        val_images, val_labels = val_data["img"].to(device), val_data["lbl"].to(device)
                        with autocast("cuda", enabled=args.autocast):
                            val_outputs = model.module(val_images) # Evaluate on local model when using DDP
                            val_step_loss = loss_function(val_outputs, val_labels)
                        epoch_val_loss += val_step_loss.item()
                        step += 1
                        val_outputs = val_outputs.argmax(dim=1)
                        
                        val_outputs_all= torch.cat((val_outputs_all, val_outputs), dim=0)
                        val_labels_all = torch.cat((val_labels_all, val_labels), dim=0)

                    epoch_val_loss /= step

                    all_valoutputs_all = val_outputs_all
                    all_vallabels_all = val_labels_all
                    
                    if args.local_rank == 0:
                        val_metric = torch.sum(all_valoutputs_all == all_vallabels_all).item() / all_vallabels_all.numel()
                        ava_np = all_valoutputs_all.detach().cpu().numpy()
                        ala_np = all_vallabels_all.detach().cpu().numpy()
                        tn, fp, fn, tp = confusion_matrix(ala_np, ava_np).ravel()

                        print(f"epoch {epoch + 1} val loss: {epoch_val_loss:.4f} val accuracy: {val_metric}, tn: {tn}, fp: {fp}, fn: {fn}, tp: {tp}")
                        tensorboard_writer.add_scalar("val_loss", epoch_val_loss, epoch + 1)
                        tensorboard_writer.add_scalar("val_accuracy", val_metric, epoch + 1)
                        tensorboard_writer.add_scalar("val_tn", tn, epoch + 1)
                        tensorboard_writer.add_scalar("val_fp", fp, epoch + 1)
                        tensorboard_writer.add_scalar("val_fn", fn, epoch + 1)
                        tensorboard_writer.add_scalar("val_tp", tp, epoch + 1)
                        metric_dict = {
                            "epoch": epoch + 1,
                            "val_metric": val_metric,
                            "tn": tn,
                            "fp": fp,
                            "fn": fn,
                            "tp": tp
                        }
                        save_model(model, epoch, optimizer, scheduler, metric_dict, model_path)
        dist.barrier() # Wait for all processes to finish validation

    if dist.get_rank() == 0:
        tensorboard_writer.close()

                    
    if dist.get_rank() == 0:
        save_model(model, epoch, optimizer, scheduler, metric_dict, model_path)
    dist.destroy_process_group()

def validate(args, synthsheet_path, real_impath, synth_impath):
    cache_rate = 0.0 # No point caching for validation
    args.batch_size=1
    n_test_imgs = 1
    
    # initialize the distributed training process, every GPU runs in a process
    dist.init_process_group(backend="nccl", init_method="env://")
    
    args.local_rank = dist.get_rank()
    if args.local_rank != 0:
        f = open(os.devnull, "w")
        sys.stdout = sys.stderr = f

    val_interval = 5

    args.exp_name = f"ds_enb0_r{int(args.real_data)}_s{int(args.synth_data)}_l{int(args.generated_label)}"

    if args.add_to_modelname:
        args.exp_name += f"_{args.add_to_modelname}"

    model_path = f"./models/{args.exp_name}.pth"

    df_path = synthsheet_path
    df = pd.read_excel(df_path)

    train_df = df[df["Label"] == "train"]
    test_df = df[df["Label"] == "test"]

    assert args.real_data or args.synth_data, "Must specify at least one of real or synthetic data"

    real_folder = real_impath
    synth_folder = synth_impath

    train_files = []

    for idx, row in train_df.iterrows():
        folder = row["folder"]
        if args.real_data:
            img_path = os.path.join(real_folder, folder,f'{args.imkey}.nii.gz')
            train_files.append({"img": img_path, "lbl": row["PIRADS_real"]})

        if args.synth_data:
            img_path = os.path.join(synth_folder, folder,f'{args.imkey}.nii.gz')
            if args.generated_label:
                train_files.append({"img": img_path, "lbl": row["PIRADS_synth"]}) # Synthetic data with synthetic label
            else:
                train_files.append({"img": img_path, "lbl": row["PIRADS_real"]})
        
    val_files = []
    for idx, row in test_df.iterrows():
        folder = row["folder"]
        img_path = os.path.join(real_folder, folder,f'{args.imkey}.nii.gz')
        val_files.append({"img": img_path, "lbl": row["PIRADS_real"]})

    prob=0.5

    print(f"Train set size: {len(train_files)} and Val set size: {len(val_files)}")

    # define transforms for image
    train_transforms = Compose(
        [
            LoadImaged(keys=["img"], ensure_channel_first=True, image_only=True),
            # Crop foreground then pad to 256x256x32
            CropForegroundd(keys=["img"], source_key="img"),
            ScaleIntensityRangePercentilesd(keys=["img"],lower=0, upper=98.0, b_min=0.0, b_max=1.0, clip=True),
            SpatialPadd(keys=["img"], spatial_size=(256,256,32), mode="constant", constant_values=[0]),
            CenterSpatialCropd(keys=["img"], roi_size=(256,256,32)),
            RandFlipd(keys=["img"], prob=prob, spatial_axis=0), # 0:l/r, 1: ant/post, 2: sup/inf
            RandRotated(keys=["img"], range_z=15*3.14/180, prob=prob, mode=('bilinear')),
            RandAffined(keys=["img"], prob=prob, translate_range=(25,25,2), padding_mode='zeros', mode=('bilinear')), # 10% of 256x256x32
            RandZoomd(keys=["img"], prob=prob, min_zoom=0.9, max_zoom=1.1, mode=('bilinear'), padding_mode='constant',constant_values=[0]),
        ]
    )

    val_transforms = Compose([
        LoadImaged(keys=["img"], ensure_channel_first=True, image_only=True),
        CropForegroundd(keys=["img"], source_key="img"),
        ScaleIntensityRangePercentilesd(keys=["img"],lower=0, upper=98.0, b_min=0.0, b_max=1.0, clip=True),
        SpatialPadd(keys=["img"], spatial_size=(256,256,32), mode="constant", constant_values=[0]),
        CenterSpatialCropd(keys=["img"], roi_size=(256,256,32)),
    ])

    # create a training data loader
    train_ds = CacheDataset(data=train_files, transform=train_transforms, num_workers=args.num_workers, cache_rate=cache_rate)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=None,
        shuffle=False,
    )

    val_ds = CacheDataset(data=val_files, transform=val_transforms, num_workers=2, cache_rate=cache_rate)
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=2,
        pin_memory=True,
        sampler=None,
        shuffle=False
    )

    device = torch.device(f"cuda:{args.local_rank}")
    torch.cuda.set_device(device)
    
    
    model = EfficientNetBN(
        model_name="efficientnet-b0",
        pretrained=False,
        spatial_dims=3,
        in_channels=1,
        num_classes=2
    ).to(device)
    
    
    optimizer = torch.optim.AdamW(model.parameters(), args.lr)
    total_steps = (args.num_epochs * len(train_loader.dataset)) / (args.batch_size*dist.get_world_size())
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=2.0)


    model, optimizer, scheduler, start_epoch, val_metric = load_model(model, optimizer, scheduler, model_path)

    # wrap the model with DistributedDataParallel module
    model = DistributedDataParallel(model, device_ids=[device])
    model.eval()

    with torch.no_grad():
        if args.local_rank==0:
            val_labels_all = torch.tensor([], dtype=torch.int64, device=device)
            val_outputs_all = torch.tensor([], dtype=torch.int64, device=device)
            step = 0
            for val_data in tqdm(val_loader):
                step += 1
                val_images, val_labels = val_data["img"].to(device), val_data["lbl"].to(device)
                with autocast("cuda", enabled=args.autocast):
                    val_outputs = model.module(val_images) # Evaluate on local model when using DDP
                val_outputs = torch.nn.functional.softmax(val_outputs, dim=1)
                
                val_outputs_all= torch.cat((val_outputs_all, val_outputs), dim=0)
                val_labels_all = torch.cat((val_labels_all, val_labels), dim=0)

            all_valoutputs_all = val_outputs_all
            all_vallabels_all = val_labels_all

            if args.local_rank == 0:
                all_valoutputs_argmax = all_valoutputs_all.argmax(dim=1) # Get the argmax
                val_metric = torch.sum(all_valoutputs_argmax == all_vallabels_all).item() / all_vallabels_all.numel()
                ava_np = all_valoutputs_argmax.detach().cpu().numpy()
                ava_raw_np = all_valoutputs_all.detach().cpu().numpy()
                ala_np = all_vallabels_all.detach().cpu().numpy()
                c = confusion_matrix(ala_np, ava_np).ravel()
                print(f"Confusion matrix shape: {c.shape}")
                tn, fp, fn, tp = c

                auc = roc_auc_score(ala_np, ava_raw_np[:,1])
                ap = average_precision_score(ala_np, ava_raw_np[:,1])

                print(f"Validation AUC: {auc}, AP: {ap}, accuracy: {val_metric}, tn: {tn}, fp: {fp}, fn: {fn}, tp: {tp}")

                # Export predictions and labels to pandas dataframe and save
                dataframe_name = f"{args.exp_name}_val_pred.xlsx"
                df = pd.DataFrame({"pred_raw_0": ava_raw_np[:,0], "pred_raw_1": ava_raw_np[:,1], "pred": ava_np, "label": ala_np})
                df.to_excel(dataframe_name)

                with open(f"{args.exp_name}_val_metrics.pkl", "wb") as f:
                    pickle.dump({"tn": tn, "fp": fp, "fn": fn, "tp": tp, "auc": auc, "ap": ap, "accuracy": val_metric}, f)

        dist.barrier()
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--imkey", type=str, default='axt2', help="Image key")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("-b","--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("-n","--num_workers", type=int, default=8, help="Num workers")
    parser.add_argument("-e","--num_epochs", type=int, default=200, help="Num epochs")
    parser.add_argument("-r","--real_data", action="store_true", help="Use real data")
    parser.add_argument("-s","--synth_data", action="store_true", help="Use synthetic data")
    parser.add_argument("-l","--generated_label", action="store_true", help="Use generated instead of true labels for synthetic data")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("-a","--add_to_modelname", type=str, default='', help="Add to model name")
    parser.add_argument("-v","--validate", action="store_true", help="Validate instead of train")
    parser.add_argument("--autocast", action="store_true", help="Use Torch AMP")
    parser.add_argument("--synthetic_path", default="./synthdata", type=str, help="Directory for synthetic images")
    parser.add_argument("--real_path", default="./realdata", type=str, help="Directory for real images")
    parser.add_argument("--spreadsheet_path", default="./synthdata/synthsheet.xlsx", type=str, help="Path to the spreadsheet for additional data")

    args = parser.parse_args()

    if args.validate:
        validate(args=args, synthsheet_path=args.spreadsheet_path, real_impath=args.real_path, synth_impath=args.synthetic_path)
    else:
        train(args=args, synthsheet_path=args.spreadsheet_path, real_impath=args.real_path, synth_impath=args.synthetic_path)

if __name__ == "__main__":
    main()