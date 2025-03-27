import argparse
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import yaml
from torch.backends import cudnn
from torch.utils.data import DataLoader

from models.modal_tokenizer import TabTokenizer
from models.task_head import RegressHead
from mst_datasets.twoD.Table.PRSA import PRSADataset
from utils import Avg_values, MST_Logger, adjust_learning_rate


def get_args_parser():
    parser = argparse.ArgumentParser('Modal expert Model for comparison', add_help=False)
    
    # Common parameters
    parser.add_argument('--cfg', default="configs/comp_config.yaml", type=str, help='config file')
    parser.add_argument('--exp-name', default="debug", type=str, help='current experiment name')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num-workers', default=16, type=int)
    parser.add_argument('--accum-iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--print-freq', default=1, type=int)
    
    # Model parameters
    parser.add_argument('--model', default="eva_giant", type=str)
    parser.add_argument('--freeze', action='store_false')

    # Dataset parameters
    # Support datasets: PRSA
    parser.add_argument('--dataset', default='PRSA', type=str, help='dataset name')
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
    # train_dataset, test_dataset = build_dataset(dataset_name, cfg[dataset_name]['dataset'])
    dataset_cfg = cfg[dataset_name]['dataset']
    dataset = PRSADataset(root_path=dataset_cfg['root_path'], seq_len=dataset_cfg['seq_len'], stride=dataset_cfg['stride'])
    totalN = len(dataset)
    trainN = int(totalN * 0.6)
    testN = totalN - trainN
    train_dataset, test_dataset = torch.utils.data.dataset.random_split(dataset, [trainN, testN])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # build model
    # vit_large
    if args.model == "vit_large":
        
        backbone = timm.create_model("vit_large_patch16_224", pretrained=True)
        for name, param in backbone.named_parameters():
            param.requires_grad = args.freeze
        
        task_head = RegressHead(token_num=dataset_cfg['seq_len'], embed_dim=1024)
    
        model = nn.Sequential(
            TabTokenizer(vocab=dataset.vocab, ncols=dataset.ncols, embed_dim=1024),
            *backbone.blocks,
            task_head
        )
    
    elif args.model == "deit3_large":
        
        backbone = timm.create_model("deit3_large_patch16_224", pretrained=True)
        for name, param in backbone.named_parameters():
            param.requires_grad = args.freeze
        
        task_head = RegressHead(token_num=dataset_cfg['seq_len'], embed_dim=1024)
    
        model = nn.Sequential(
            TabTokenizer(vocab=dataset.vocab, ncols=dataset.ncols, embed_dim=1024),
            *backbone.blocks,
            task_head
        )
        
    elif args.model == "eva_giant":

        backbone = timm.create_model("eva_giant_patch14_224", pretrained=True)
        for name, param in backbone.named_parameters():
            param.requires_grad = args.freeze
        
        task_head = RegressHead(token_num=dataset_cfg['seq_len'], embed_dim=1408)
    
        model = nn.Sequential(
            TabTokenizer(vocab=dataset.vocab, ncols=dataset.ncols, embed_dim=1408),
            *backbone.blocks,
            task_head
        )

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
    print_loss = dict()
    run_time = Avg_values()
    for cur_epoch in range(max_epoch):
        for i, (image, label) in enumerate(train_loader):
            start_time = datetime.now()
            
            image = image.cuda()
            label = label.cuda()
            
            cur_iters += 1
            lr = adjust_learning_rate(optimizer, cur_iters, warmup_iters, max_iters, optimizer_cfg)
            
            model.train()
            pred = model(image)
            loss = task_head.loss(pred, label)
            
            update_loss = None
            for key in loss.keys():
                if key not in print_loss.keys():
                    print_loss[key] = loss[key].item()
                else:
                    print_loss[key] += loss[key].item()
                    
                if update_loss is None:
                    update_loss = loss[key]
                else:
                    update_loss += loss[key]
            loss = update_loss
            
            loss /= args.accum_iter
            
            loss.backward()
            if cur_iters % args.accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad() 
            
            end_time = datetime.now()
            run_time.update(end_time-start_time, 1)
            eta_str = str(run_time.avg * (max_iters - run_time.count))
            
            if cur_iters % args.print_freq == 0:
                loss_str = ""
                total_loss = 0.
                for key in print_loss.keys():
                    loss_str += f"{key}: {print_loss[key] / args.print_freq:.6} "
                    total_loss += print_loss[key] / args.print_freq
                loss_str += f"total_loss: {total_loss:.6} update_loss: {loss.item()}"
                
                mstLogger.logger.info(f"[accum_iter:{args.accum_iter}] epoch:{cur_epoch+1} cur_iter:{i+1}/{iters_per_epoch} "
                                f"total_iters:{cur_iters} lr:{lr:.6e} loss:[{loss_str}] eta:{eta_str}")
                
                print_loss = dict()
              
        if (cur_epoch+1) % optimizer_cfg['eval_epochs'] == 0 or cur_epoch == max_epoch-1:
            evaluation(model, test_loader, dataset, cur_epoch, args, mstLogger=mstLogger)
            
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