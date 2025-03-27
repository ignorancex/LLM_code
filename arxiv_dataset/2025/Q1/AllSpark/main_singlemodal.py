import argparse
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from timm.models.vision_transformer import Block
from torch.backends import cudnn
from torch.utils.data import DataLoader

from models import builder
from models.modal_tokenizer import HSItokenizer
from models.model_wollm import ModelwoLLM
from models.task_head import (HSIClsHead, LinearClsHead,
                              MultiLabelLinearClsHead, SegHead)
from mst_datasets import build_dataset
from utils import Avg_values, MST_Logger, adjust_learning_rate


def get_args_parser():
    parser = argparse.ArgumentParser('Modal expert Model for comparison', add_help=False)
    
    # Common parameters
    parser.add_argument('--cfg', default="configs/sm_common_aiamax.yaml", type=str, help='config file')
    parser.add_argument('--exp-name', default="debug", type=str, help='current experiment name')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num-workers', default=16, type=int)
    parser.add_argument('--accum-iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--print-freq', default=20, type=int)

    # Dataset parameters
    # Support datasets: NWPU_RESISC45 LoveDA OpenSARUrban BigEarthNet_MSI IndianPine
    parser.add_argument('--dataset', default='Pavia', type=str, help='dataset name')
    parser.add_argument('--batch-size', default=32, type=int)

    return parser.parse_args()

def main(args, cfg):
    mstLogger = MST_Logger(args.output_dir)
    mstLogger.logger.info(args)
    mstLogger.logger.info(cfg)
    
    if args.seed is not None:
        mstLogger.logger.info(f"set seed: {args.seed}")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        cudnn.benchmark = True
    
    dataset_name = args.dataset
    
    # build dataset
    train_dataset, test_dataset = build_dataset(dataset_name, cfg[dataset_name]['dataset'])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # build model
    if dataset_name == 'NWPU_RESISC45':
        # NWPU_RESISC45_cls_EVA
        backbone = nn.Sequential(
            builder.build_visual2DTokenizer(img_size=224, embed_dim=1408, params_path=cfg['Visual2DTokenizer_checkpoint'],
                                            freeze_params=True),
            builder.build_evaBackbone(cfg['EVAbackbone_checkpoint'])
        )
        task_head = LinearClsHead(45, 1408)
        token_mode = "cls"
    elif dataset_name == 'LoveDA':
        # LoveDA_seg_EVA
        backbone = nn.Sequential(
            builder.build_visual2DTokenizer(img_size=448, embed_dim=1408, params_path=cfg['Visual2DTokenizer_checkpoint'],
                                            freeze_params=True),
            builder.build_evaBackbone(cfg['EVAbackbone_checkpoint'], vision_feats_return_layer=[9, 19, 29, 39])
        )
        task_head = SegHead(img_size=448, out_layers=4, embed_dim=1408, num_classes=7, token_num=1024)
        token_mode = "no_cls"
    elif dataset_name == 'OpenSARUrban':
        # OpenSARUrban_cls_EVA
        backbone = nn.Sequential(
            builder.build_visual2DTokenizer(img_size=98, embed_dim=1408, params_path=None,
                                    freeze_params=False, in_channels=2),
            builder.build_evaBackbone(cfg['EVAbackbone_checkpoint'])
        )
        task_head = LinearClsHead(12, 1408)
        token_mode = "cls"
    elif dataset_name == 'BigEarthNet_MSI':
        # BigEarthNet_MSI_multilabelCls_EVA
        backbone = nn.Sequential(
            builder.build_visual2DTokenizer(img_size=140, embed_dim=1408, params_path=None,
                                    freeze_params=False, in_channels=12),
            builder.build_evaBackbone(cfg['EVAbackbone_checkpoint'])
        )
        task_head = MultiLabelLinearClsHead(19, 1408)
        token_mode = "cls"
    elif dataset_name == 'Pavia':
        # Pavia_seg
        backbone = nn.Sequential(
            HSItokenizer(image_size=1, near_band=1, num_patches=103, dim=768),
            nn.Sequential(*[
                Block(
                    dim=768,
                    num_heads=12,
                    mlp_ratio=4.,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                    act_layer=nn.GELU
                )
                for i in range(12)])
        )
        task_head = HSIClsHead(768, 9)
        token_mode = "cls"
    
    model = ModelwoLLM(backbone=backbone, task_head=task_head, token_mode=token_mode)

    model.cuda()
    
    # optimizer
    optimizer_cfg = cfg[dataset_name]['optimizer']
    if optimizer_cfg['name'] == 'Adamw':
        optimizer = torch.optim.AdamW(model.parameters(), optimizer_cfg['lr'], weight_decay=optimizer_cfg['weight_decay'])
    else:
        raise NotImplementedError
    
    # summary params
    freeze_param, trainable_param = 0, 0
    for p in model.parameters():
        if p.requires_grad:
            trainable_param += p.numel()
        else:
            freeze_param += p.numel()
            
    mstLogger.logger.info("-" * 40)        
    mstLogger.logger.info(f"### Total Params: {(freeze_param + trainable_param) / 1e6:.2f}M")
    mstLogger.logger.info(f"### Freeze Params: {freeze_param / 1e6:.2f}M")
    mstLogger.logger.info(f"### Trainable Params: {trainable_param / 1e6:.2f}M")
    mstLogger.logger.info("-" * 40) 

    # run
    cur_iters = 0
    iters_per_epoch = len(train_loader)
    max_epoch = int(optimizer_cfg['max_epochs'])
    max_iters = max_epoch * iters_per_epoch
    warmup_iters = int(optimizer_cfg['warmup_epochs']) * iters_per_epoch
    print_loss = 0.
    run_time = Avg_values()
    scaler = torch.cuda.amp.GradScaler()
    for cur_epoch in range(max_epoch):
        for i, (image, label) in enumerate(train_loader):
            
            start_time = datetime.now()
            
            image = image.cuda()
            label = label.cuda()
            
            cur_iters += 1
            lr = adjust_learning_rate(optimizer, cur_iters, warmup_iters, max_iters, optimizer_cfg)
            
            model.train()
            with torch.cuda.amp.autocast(True):
                pred = model(image)
                loss = task_head.loss(pred, label)
            loss /= args.accum_iter
            
            scaler.scale(loss).backward()
            if cur_iters % args.accum_iter == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad() 
        
            print_loss += loss.item()
            
            end_time = datetime.now()
            run_time.update(end_time-start_time, 1)
            eta_str = str(run_time.avg * (max_iters - run_time.count))
            
            if cur_iters % args.print_freq == 0:
                mstLogger.logger.info(f"[accum_iter:{args.accum_iter}] epoch:{cur_epoch+1} cur_iter:{i+1}/{iters_per_epoch} "
                                f"total_iters:{cur_iters} lr:{lr:.6e} loss:{print_loss / args.print_freq:.6f} eta:{eta_str}")
                print_loss = 0.
              
        if (cur_epoch+1) % optimizer_cfg['eval_epochs'] == 0 or cur_epoch == max_epoch-1:
            evaluation(model, test_loader, test_dataset, cur_epoch, args, mstLogger=mstLogger)
            
    print("Done!")
    
@torch.no_grad()
def evaluation(model, test_loader, test_dataset, step, args, mstLogger):
    model.eval()
    
    run_time = Avg_values()
    test_iters = len(test_loader)
    for i, (image, label) in enumerate(test_loader):
        start_time = datetime.now()
        
        image = image.cuda()
        label = label.cuda()
        
        pred = model(image)
        test_dataset.eval(pred, label)
        
        end_time = datetime.now()
        run_time.update(end_time-start_time, 1)
        eta_str = str(run_time.avg * (len(test_loader) - run_time.count))
        
        if i % args.print_freq == 0:
            mstLogger.logger.info(f"[val] cur_iters:{i}/{test_iters} eta:{eta_str}")
        
    metrics = test_dataset.get_eval_res()    
    mstLogger.logger.info("-" * 40)
    for key in metrics.keys():
        mstLogger.logger.info(f"### {key}: {metrics[key]}")
    mstLogger.logger.info("-" * 40)

if __name__ == "__main__":
    args = get_args_parser()
    if args.exp_name is None:
        print("It is recommended to specific your experiment name")
        args.exp_name = ""
    
    with open(args.cfg, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f.read())
        
    args.output_dir = os.path.join("./workdirs", args.exp_name, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, cfg)