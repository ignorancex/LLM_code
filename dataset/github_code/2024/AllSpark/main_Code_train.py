## MSTAGI for Code
## Based on CodeBERT(https://github.com/microsoft/CodeBERT)

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
from torch.backends import cudnn
from torch.distributed.pipeline.sync import Pipe
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from models import builder
from models.llm.modeling_llama import _expand_mask, _make_causal_mask
from mst_datasets import build_CodeSearchNetDataset, compute_metrics
from utils import Avg_values, MST_Logger, adjust_learning_rate


def get_args_parser():
    parser = argparse.ArgumentParser('MultiModal with LLM for model parallel in N GPU', add_help=False)
    
    # Common parameters
    parser.add_argument('--cfg', default="configs/Code_CodeSearchNet.yaml", type=str, help='config file')
    parser.add_argument('--exp-name', default="debug", type=str, help='current experiment name')
    parser.add_argument('--seed', default=2333, type=int)
    parser.add_argument('--num-workers', default=16, type=int)
    parser.add_argument('--accum-iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--print-freq', default=1, type=int)
    parser.add_argument('--parallel-split-layer', default=[16, 32], type=int, nargs="+")
    parser.add_argument('--port', default=29520, type=int)

    # Dataset parameters
    # Support datasets: codesearch
    parser.add_argument('--dataset', default='codesearch', type=str, help='dataset name')
    parser.add_argument('--root-dir', default='/opt/data/private/dataset/MFM/Code', type=str)
    parser.add_argument('--lang', default='ruby', type=str)
    parser.add_argument('--batch-size', default=40, type=int)
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
    
    # build model
    llm_tokenizer, llm_backbone = builder.build_LLMForSegCls(cfg)
    
    dataset_name = args.dataset
    data_dir = os.path.join(args.root_dir, 'train_valid', args.lang)
    train_dataset = build_CodeSearchNetDataset(cfg, data_dir, llm_tokenizer, dataset_name, ttype='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    model = build_ModelwLLM_Pipe(llm_backbone, parallel_list)
    model = Pipe(model, chunks=args.pipe_chunks)
    loss_fct = CrossEntropyLoss()
    
    # optimizer
    optimizer_cfg = cfg['optimizer']
    optimizer = torch.optim.AdamW(model.parameters(), optimizer_cfg['lr'], weight_decay=optimizer_cfg['weight_decay'],
                                  eps=optimizer_cfg['adam_epsilon'])
    
    # summary params
    freeze_param, trainable_param = 0, 0
    freeze_name, trainable_name = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            trainable_name.append(n)
            trainable_param += p.numel()
        else:
            freeze_name.append(n)
            freeze_param += p.numel()
            
    mstLogger.logger.info("-" * 40)    
    mstLogger.logger.info(f"### Trainable Params")
    for n in trainable_name:
        mstLogger.logger.info(f"### {n}")
    mstLogger.logger.info("-" * 40)       
    mstLogger.logger.info(f"### Freeze Params")
    for n in freeze_name:
        mstLogger.logger.info(f"### {n}")
    mstLogger.logger.info("-" * 40)  
    mstLogger.logger.info(f"### Total Params: {(freeze_param + trainable_param) / 1e6:.2f}M")
    mstLogger.logger.info(f"### Freeze Params: {freeze_param / 1e6:.2f}M")
    mstLogger.logger.info(f"### Trainable Params: {trainable_param / 1e6:.2f}M")
    mstLogger.logger.info("-" * 40) 
    
    cur_iters = 0
    iters_per_epoch = len(train_loader)
    max_epoch = int(optimizer_cfg['max_epochs'])
    max_iters = max_epoch * iters_per_epoch
    warmup_iters = int(optimizer_cfg['warmup_epochs']) * iters_per_epoch
    print_loss = 0.
    run_time = Avg_values()

    for cur_epoch in range(max_epoch):
        for i, batch in enumerate(train_loader):
            
            start_time = datetime.now()
            
            cur_iters += 1
            lr = adjust_learning_rate(optimizer, cur_iters, warmup_iters, max_iters, optimizer_cfg)
            
            batch = tuple(t.cuda() for t in batch)
            labels = batch[3].to('cuda:1')
            
            ouputs = model(batch[0], batch[1]).to_here()
            loss = loss_fct(ouputs.view(-1, llm_backbone.num_labels), labels.view(-1))
            
            loss.backward()

            if cur_iters % args.accum_iter == 0:
                optimizer.step()
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
            evaluation(args, cfg, model, llm_tokenizer, "dev", data_dir, dataset_name, mstLogger)
    
    print("Save model...")        
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pth"))
        
    print("Done!")
    
@torch.no_grad()
def evaluation(args, cfg, model, tokenizer, mode, data_dir, dataset_name, mstLogger):
    model.eval()
    
    if (mode == 'dev'):
        eval_dataset = build_CodeSearchNetDataset(cfg, data_dir, tokenizer, dataset_name, ttype='dev')
    elif (mode == 'test'):
        eval_dataset, instances = build_CodeSearchNetDataset(cfg, data_dir, tokenizer, dataset_name, ttype='test')
    
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    run_time = Avg_values()
    test_iters = len(eval_loader)
    preds = None
    for i, batch in enumerate(eval_loader):
        start_time = datetime.now()
        
        batch = tuple(t.cuda() for t in batch)
        labels = batch[3].to('cuda:1')
        
        ouputs = model(batch[0], batch[1]).to_here()
        
        if preds is None:
            preds = ouputs.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:

            preds = np.append(preds, ouputs.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
        
        end_time = datetime.now()
        run_time.update(end_time-start_time, 1)
        eta_str = str(run_time.avg * (len(eval_loader) - run_time.count))
        
        if i % args.print_freq == 0:
            mstLogger.logger.info(f"[val] cur_iters:{i}/{test_iters} eta:{eta_str}")
    
    preds_label = np.argmax(preds, axis=1)
    result = compute_metrics(args.dataset, preds_label, out_label_ids)
    
    mstLogger.logger.info("-" * 40)
    for key in result.keys():
        mstLogger.logger.info(f"### {key}: {result[key]}")
    mstLogger.logger.info("-" * 40)

class BeginModule(nn.Module):
    def __init__(self, embed_tokens, llm_layers):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.llm_layers = llm_layers
        
    def forward(self, input_ids, attention_mask):

        batch_size, seq_length = input_ids.shape

        seq_length_with_past = seq_length
        past_key_values_length = 0
        
        device = input_ids.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        inputs_embeds = self.embed_tokens(input_ids)

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        # decoder layers
        hidden_states = inputs_embeds
        for idx, decoder_layer in enumerate(self.llm_layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=None,
                use_cache=None,
            )
            hidden_states = layer_outputs[0]

        return input_ids, hidden_states, attention_mask, position_ids
    
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask
    
    
class EndModule(nn.Module):
    def __init__(self, llm_layers, norm, score, config, num_labels):
        super().__init__()
        self.llm_layers = llm_layers
        self.norm = norm
        self.score = score
        self.config = config
        self.num_labels = num_labels
        
    def forward(self, input_ids, hidden_states, attention_mask, position_ids):
        
        for idx, decoder_layer in enumerate(self.llm_layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=None,
                use_cache=None,
            )
            hidden_states = layer_outputs[0]
        
        hidden_states = self.norm(hidden_states)
        logits = self.score(hidden_states)
        
        batch_size = input_ids.shape[0]

        sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
    
        return pooled_logits



def build_ModelwLLM_Pipe(llm_backbone, parallel_list):
    beginBlock = BeginModule(llm_backbone.model.embed_tokens, 
                             llm_backbone.model.layers[0:parallel_list[0]]).to("cuda:0")
    endBlock = EndModule(llm_backbone.model.layers[parallel_list[-2]:], llm_backbone.model.norm,
                        llm_backbone.score, llm_backbone.config, llm_backbone.num_labels).to(f"cuda:1")
    return nn.Sequential(
        beginBlock,
        endBlock
    )


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