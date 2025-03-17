import argparse
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.backends import cudnn
from torch.distributed.pipeline.sync import Pipe
from torch.utils.data import DataLoader

from models import builder
from models.llm.modeling_llama import _expand_mask, _make_causal_mask
from models.task_head import LinearClsHead
from models.utils import get_text_embed
from mst_datasets import NWPURESIS45Dataset
from utils import Avg_values, MST_Logger, adjust_learning_rate


def get_args_parser():
    parser = argparse.ArgumentParser('AllSpark for RGB', add_help=False)
    
    parser.add_argument('--cfg', default="configs/RGB_NWPU_train.yaml", type=str, help='config file')
    parser.add_argument('--exp-name', default="debug", type=str, help='current experiment name')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num-workers', default=8, type=int)
    parser.add_argument('--print-freq', default=1, type=int)
    parser.add_argument('--parallel-split-layer', default=[11], type=int, nargs="+")
    parser.add_argument('--port', default=29520, type=int)
    
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

    args.parallel_split_layer.append(32)
    parallel_list = args.parallel_split_layer
    assert len(parallel_list) <= torch.cuda.device_count()
    mstLogger.logger.info(f"use {len(parallel_list)} GPUs")
    
    # build dataset
    train_dataset = NWPURESIS45Dataset(metainfo_file=cfg['metainfo_train'], root_path=cfg['root_path'], is_train=True)
    test_dataset = NWPURESIS45Dataset(metainfo_file=cfg['metainfo_test'], root_path=cfg['root_path'], is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # build model
    modal_tokenizer = builder.build_visual2DTokenizer(img_size=224, embed_dim=1408, params_path=cfg['modal_tokenizer_checkpoint'], 
                                                      freeze_params=False, in_channels=3)
    modal_backbone = builder.build_evaBackbone(params_path=cfg['modal_backbone_checkpoint'], vision_feats_return_layer=-1)
    bridge = builder.build_bridge(cfg['bridge'])
    llm_tokenizer, llm_backbone = builder.build_LLM(cfg)
    task_head = LinearClsHead(45, 4096)
    if cfg['task_head_checkpoint'] or args.eval:
        print(f"### Load task_head params")
        state_dict = torch.load(cfg['task_head_checkpoint'], map_location='cpu')
        task_head.load_state_dict(state_dict, strict=True)
    model = build_ModelwLLM_Pipe(modal_tokenizer, modal_backbone, bridge, llm_backbone, task_head, parallel_list)
    model = Pipe(model, chunks=args.pipe_chunks)

    # prompts
    text_embeds, input_atts_pads = [], []
    with torch.no_grad():
        for prompt in train_dataset.prompts:
            text_embed, input_atts_pad = get_text_embed(prompt, args.batch_size, llm_tokenizer, llm_backbone)
            text_embeds.append(text_embed)
            input_atts_pads.append(input_atts_pad)

    if args.eval:
        evaluation(model, test_loader, test_dataset, args, eva_text_embed=text_embeds[0], 
                       eva_input_atts_pad=input_atts_pads[0], mstLogger=mstLogger)
    else:
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
            model.train()
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
                
                with torch.cuda.amp.autocast(True):
                    pred = model(image.to('cuda:0'), cur_text_embed, cur_input_atts_pad).to_here()
                    loss = task_head.loss(pred, label.to(pred.device))
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad() 
                
                print_loss += loss.item()
                
                end_time = datetime.now()
                run_time.update(end_time-start_time, 1)
                eta_str = str(run_time.avg * (max_iters - run_time.count))
                
                if cur_iters % args.print_freq == 0:
                    mstLogger.logger.info(f"epoch:{cur_epoch+1} cur_iter:{i+1}/{iters_per_epoch} "
                                    f"total_iters:{cur_iters} lr:{lr:.6e} loss:{print_loss / args.print_freq:.6f} eta:{eta_str}")
                    print_loss = 0.
            
            if cur_epoch == max_epoch-1:
                print(f"### Save params")
                torch.save(modal_tokenizer.state_dict(), os.path.join(args.output_dir, "modal_tokenizer.pth"))
                torch.save(modal_backbone.state_dict(), os.path.join(args.output_dir, "modal_backbone.pth"))
                torch.save(bridge.state_dict(), os.path.join(args.output_dir, "bridge.pth"))
                torch.save(task_head.state_dict(), os.path.join(args.output_dir, "task_head.pth"))
                llm_backbone.save_pretrained(os.path.join(args.output_dir, "llm"))
                llm_tokenizer.save_pretrained(os.path.join(args.output_dir, "llm"))
                
            if (cur_epoch+1) % optimizer_cfg['eval_epochs'] == 0 or cur_epoch == max_epoch-1:
                evaluation(model, test_loader, test_dataset, args, eva_text_embed=text_embeds[0], 
                        eva_input_atts_pad=input_atts_pads[0], mstLogger=mstLogger)
            
    print("Done!")
    
@torch.no_grad()
def evaluation(model, test_loader, test_dataset, args, eva_text_embed, eva_input_atts_pad, mstLogger):
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


class BeginBlock(nn.Module):
    def __init__(self, modal_tokenizer, eva_backbone, bridge, llm_layers, tokenMode):
        super().__init__()
        self.modal_tokenizer = modal_tokenizer
        self.eva_backbone = eva_backbone
        self.bridge = bridge
        self.llm_layers = llm_layers
        self.tokenMode = tokenMode
        
    def forward(self, src_input, text_embed, input_atts_pad):
        
        src_embed = self.eva_backbone(self.modal_tokenizer(src_input))
        if isinstance(src_embed, List):
            src_embed = src_embed[-1]
            
        if self.tokenMode == "no_cls":
            src_embed = src_embed[:, 1:]
        src2t_embed, src2t_atts = self.bridge(src_embed)
        
        inputs_embeds = torch.cat([src2t_embed, text_embed], dim=1)
        attention_mask = torch.cat([src2t_atts, input_atts_pad], dim=1)
        
        batch_size, seq_length, _ = inputs_embeds.shape
        
        past_key_values_length = 0
        
        device = inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )
            
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
        
        return hidden_states, attention_mask, position_ids
        
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
    

class CoreBlock(nn.Module):
    def __init__(self, llm_layers):
        super().__init__()
        self.llm_layers = llm_layers
        
    def forward(self, hidden_states, attention_mask, position_ids):
        
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
        
        return hidden_states, attention_mask, position_ids
    
    
class EndBlock(nn.Module):
    def __init__(self, llm_layers, norm, task_head):
        super().__init__()
        self.llm_layers = llm_layers
        self.norm = norm
        self.task_head = task_head
        
    def forward(self, hidden_states, attention_mask, position_ids):
        
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
        
        pred = self.task_head(hidden_states)
        return pred
    

def build_ModelwLLM_Pipe(modal_tokenizer, eva_backbone, bridge, llm_backbone, 
                         task_head, parallel_list, EVA_tokenMode='no_cls'):
    beginBlock = BeginBlock(modal_tokenizer, eva_backbone, bridge, 
                            llm_backbone.model.layers[0:parallel_list[0]], EVA_tokenMode).to("cuda:0")
    coreBlocks = []
    core_num = len(parallel_list) - 2
    for i in range(core_num):
        coreBlocks.append(CoreBlock(
            llm_backbone.model.layers[parallel_list[i]:parallel_list[i+1]]
            ).to(f"cuda:{i+1}"))
    endBlock = EndBlock(llm_backbone.model.layers[parallel_list[-2]:], llm_backbone.model.norm,
                        task_head).to(f"cuda:{core_num+1}")
    return nn.Sequential(
        beginBlock,
        *coreBlocks,
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