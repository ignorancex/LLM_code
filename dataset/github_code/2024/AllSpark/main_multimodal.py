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
from torch.distributed.pipeline.sync import Pipe
from torch.backends import cudnn
from torch.utils.data import DataLoader

from models import builder
from models.model_wllm import build_ModelwLLM_Pipe
from models.utils import get_text_embed
from mst_datasets import build_dataset
from utils import Avg_values, MST_Logger, adjust_learning_rate


def get_args_parser():
    parser = argparse.ArgumentParser('MultiModal with LLM for model parallel in N GPU', add_help=False)
    
    # Common parameters
    parser.add_argument('--cfg', default="configs/mm_aimax.yaml", type=str, help='config file')
    parser.add_argument('--exp-name', default="debug", type=str, help='current experiment name')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num-workers', default=8, type=int)
    parser.add_argument('--accum-iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--print-freq', default=1, type=int)
    parser.add_argument('--parallel-split-layer', default=[14, 32], type=int, nargs="+")
    parser.add_argument('--out-idxs', default=[8, 16, 24, 32], type=int, nargs="+")
    parser.add_argument('--port', default=29520, type=int)
    
    # Model parameters
    parser.add_argument('--bridge-freeze', action='store_true')

    # Dataset parameters
    # Support datasets: NWPU_RESISC45 BigEarthNet_MSI IndianPine
    parser.add_argument('--dataset', default='Pavia', type=str, help='dataset name')
    parser.add_argument('--fix-prompt', action='store_true', help='whether fix prompt to id 0, default False (eg, random)')
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--pipe-chunks', default=1, type=int)

    return parser.parse_args()

def main(args, cfg):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(args.port)
    torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)

    mstLogger = MST_Logger(args.output_dir)
    mstLogger.logger.info(args)
    mstLogger.logger.info(cfg)
    
    if args.seed is not None:
        mstLogger.logger.info(f"set seed: {args.seed}")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        cudnn.benchmark = True
        
    parallel_list = args.parallel_split_layer
    assert len(parallel_list) <= torch.cuda.device_count()
    mstLogger.logger.info(f"use {len(parallel_list)} GPUs")
    
    dataset_name = args.dataset
    
    # build dataset
    train_dataset, test_dataset = build_dataset(dataset_name, cfg[dataset_name]['dataset'])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # build model
    modal_tokenizer = builder.build_modalTokenizer(cfg[dataset_name]['modalTokenizer'])
    modal_backbone = builder.build_modalBackbone(cfg[dataset_name]['modalBackbone'])
    cfg[dataset_name]['bridge']['freeze_params'] = args.bridge_freeze
    bridge = builder.build_bridge(cfg[dataset_name]['bridge'])
    llm_tokenizer, llm_backbone = builder.build_LLM(cfg)
    
    # freeze_param, trainable_param = 0, 0
    # for n, p in llm_backbone.named_parameters():
    #     if 'adapter' in n:
    #         if p.requires_grad:
    #             trainable_param += p.numel()
    #         else:
    #             freeze_param += p.numel()
            
    # mstLogger.logger.info("-" * 40)        
    # mstLogger.logger.info(f"### Total Params: {(freeze_param + trainable_param) / 1e6:.2f}M")
    # mstLogger.logger.info(f"### Freeze Params: {freeze_param / 1e6:.2f}M")
    # mstLogger.logger.info(f"### Trainable Params: {trainable_param / 1e6:.2f}M")
    # mstLogger.logger.info("-" * 40) 
            
    task_head = builder.build_taskHead(cfg[dataset_name]['taskHead'])
    
    model = build_ModelwLLM_Pipe(modal_tokenizer, modal_backbone, bridge, llm_backbone, task_head, 
                                 args.out_idxs, parallel_list)
    model = Pipe(model, chunks=args.pipe_chunks)
    
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
    text_embeds, input_atts_pads = [], []
    with torch.no_grad():
        for prompt in train_dataset.prompts:
            text_embed, input_atts_pad = get_text_embed(prompt, args.batch_size, llm_tokenizer, llm_backbone)
            text_embeds.append(text_embed)
            input_atts_pads.append(input_atts_pad)
    
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
            
            # random choose prompt
            if args.fix_prompt:
                choose_idx = 0
            else:
                choose_idx = random.randint(0, len(text_embeds)-1)
            cur_text_embed = text_embeds[choose_idx][0:image.size(0)].to('cuda:0')
            cur_input_atts_pad = input_atts_pads[choose_idx][0:image.size(0)].to('cuda:0')
            assert cur_text_embed.size(0) == image.size(0)
            
            cur_iters += 1
            lr = adjust_learning_rate(optimizer, cur_iters, warmup_iters, max_iters, optimizer_cfg)
            
            model.train()
            with torch.cuda.amp.autocast(True):
                pred = model(image.to('cuda:0'), cur_text_embed, cur_input_atts_pad).to_here()
                loss = task_head.loss(pred, label.to(pred.device))
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
            evaluation(model, test_loader, test_dataset, cur_epoch, args, 
                       eva_text_embed=text_embeds[0], eva_input_atts_pad=input_atts_pads[0], mstLogger=mstLogger)
            
    print("Done!")
    
@torch.no_grad()
def evaluation(model, test_loader, test_dataset, step, args, eva_text_embed, eva_input_atts_pad, mstLogger):
    model.eval()
    
    run_time = Avg_values()
    test_iters = len(test_loader)
    for i, (image, label) in enumerate(test_loader):
        start_time = datetime.now()
        
        pred = model(image.to("cuda:0"), eva_text_embed[0:image.size(0)].to("cuda:0"), 
                        eva_input_atts_pad[0:image.size(0)].to("cuda:0")).to_here()
        test_dataset.eval(pred, label.to(pred.device))
        
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