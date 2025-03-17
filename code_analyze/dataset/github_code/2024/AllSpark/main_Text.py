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

from models import builder
from models.llm.modeling_llama import _expand_mask, _make_causal_mask
from models.utils import get_text_embed
from mst_datasets import build_IMDB_dataloader
from utils import Avg_values, MST_Logger, adjust_learning_rate, cls_accuracy


def get_args_parser():
    parser = argparse.ArgumentParser('AllSpark for Text', add_help=False)
    
    # Common parameters
    parser.add_argument('--cfg', default="configs/Text_IMDB_train.yaml", type=str, help='config file')
    parser.add_argument('--exp-name', default="debug", type=str, help='current experiment name')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--seed', default=2333, type=int)
    parser.add_argument('--num-workers', default=8, type=int)
    parser.add_argument('--print-freq', default=1, type=int)
    parser.add_argument('--parallel-split-layer', default=[15], type=int, nargs="+")
    parser.add_argument('--port', default=22520, type=int)

    # Dataset parameters
    parser.add_argument('--batch-size', default=4, type=int)
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
    
    # build model
    llm_tokenizer, llm_backbone = builder.build_LLMForSegCls(cfg)
    model = build_ModelwLLM_Pipe(llm_backbone, parallel_list, cfg['max_seq_length'])
    # model = Pipe(model, chunks=args.pipe_chunks)
    
    # build dataset
    prompt = "Please determine if this movie reviews are positive or negative?"
    with torch.no_grad():
        text_embed, input_atts_pad = get_text_embed(prompt, args.batch_size, llm_tokenizer, llm_backbone)
    train_loader, test_loader = build_IMDB_dataloader(cfg['root_path'], cfg['max_seq_length'],
                                                              llm_tokenizer, args.batch_size)
    
    if args.eval:
        acc = evaluation(args, model, text_embed, input_atts_pad, test_loader, mstLogger)
    else:
        # optimizer
        optimizer_cfg = cfg['optimizer']
        optimizer = torch.optim.AdamW(model.parameters(), optimizer_cfg['lr'], weight_decay=optimizer_cfg['weight_decay'],
                                    eps=optimizer_cfg['adam_epsilon'])
        loss_fct = CrossEntropyLoss()
        
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
        max_acc = 0.
        for cur_epoch in range(max_epoch):
            model.train()
            for i, batch in enumerate(train_loader):
                
                start_time = datetime.now()
                
                cur_iters += 1
                lr = adjust_learning_rate(optimizer, cur_iters, warmup_iters, max_iters, optimizer_cfg)
                
                cur_text_embed = text_embed[0:batch[0].size(0)].to('cuda:0')
                cur_input_atts_pad = input_atts_pad[0:batch[0].size(0)].to('cuda:0')
                assert cur_text_embed.size(0) == batch[0].size(0)
                
                batch = tuple(t.cuda() for t in batch)
                labels = batch[3].to('cuda:1')
                
                ouputs = model(cur_text_embed, cur_input_atts_pad, batch[0], batch[1]).to_here()
                loss = loss_fct(ouputs.view(-1, llm_backbone.num_labels), labels.view(-1))
                
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
                
            if (cur_epoch+1) % optimizer_cfg['eval_epochs'] == 0 or cur_epoch == max_epoch-1:
                acc = evaluation(args, model, text_embed, input_atts_pad, test_loader, mstLogger)
                if acc > max_acc:
                    max_acc = acc
                    print(f"max_acc: {max_acc}")    
                    print(f"### Save params")
                    llm_backbone.save_pretrained(os.path.join(args.output_dir, "llm"))
                    llm_tokenizer.save_pretrained(os.path.join(args.output_dir, "llm"))
        
    print("Done!")
    
@torch.no_grad()
def evaluation(args, model, text_embed, input_atts_pad, test_loader, mstLogger):
    model.eval()
    
    run_time = Avg_values()
    test_iters = len(test_loader)
    preds = None
    for i, batch in enumerate(test_loader):
        start_time = datetime.now()
        
        cur_text_embed = text_embed[0:batch[0].size(0)].to('cuda:0')
        cur_input_atts_pad = input_atts_pad[0:batch[0].size(0)].to('cuda:0')
        assert cur_text_embed.size(0) == batch[0].size(0)
        
        batch = tuple(t.cuda() for t in batch)
        labels = batch[3].to('cuda:0')

        from torchprofile import profile_macs
        
        macs = profile_macs(model, (cur_text_embed, cur_input_atts_pad, batch[0], batch[1]))
        print(f"{macs/1e9:.2f}G")
        exit(0)
        
        ouputs = model(cur_text_embed, cur_input_atts_pad, batch[0], batch[1]).to_here()
        
        if preds is None:
            preds = ouputs.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, ouputs.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
        
        end_time = datetime.now()
        run_time.update(end_time-start_time, 1)
        eta_str = str(run_time.avg * (len(test_loader) - run_time.count))
        
        if i % args.print_freq == 0:
            mstLogger.logger.info(f"[val] cur_iters:{i}/{test_iters} eta:{eta_str}")

    res = cls_accuracy(preds, out_label_ids)
    mstLogger.logger.info("-" * 40)
    for key in res.keys():
        mstLogger.logger.info(f"### {key}: {res[key].item()}")
    mstLogger.logger.info("-" * 40)

    return res['top1']


class BeginModule(nn.Module):
    def __init__(self, embed_tokens, llm_layers):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.llm_layers = llm_layers
        
    def forward(self, prompt_text_embed, prompt_input_atts_pad, input_ids, attention_mask):
        
        inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = torch.cat([inputs_embeds, prompt_text_embed], dim=1)
        attention_mask = torch.cat([attention_mask, prompt_input_atts_pad], dim=1)
        
        batch_size, seq_length, _ = inputs_embeds.shape

        seq_length_with_past = seq_length
        past_key_values_length = 0
        
        device = inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

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
    def __init__(self, llm_layers, norm, score, config, num_labels, num_latents):
        super().__init__()
        self.llm_layers = llm_layers
        self.norm = norm
        self.score = score
        self.config = config
        self.num_labels = num_labels
        self.num_latents = num_latents
        
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
        
        hidden_states = hidden_states[:, 0:self.num_latents]
        
        logits = self.score(hidden_states)
        
        batch_size = input_ids.shape[0]

        sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
    
        return pooled_logits



def build_ModelwLLM_Pipe(llm_backbone, parallel_list, num_latents):
    beginBlock = BeginModule(llm_backbone.model.embed_tokens, 
                             llm_backbone.model.layers[0:parallel_list[0]]).to("cuda:0")
    endBlock = EndModule(llm_backbone.model.layers[parallel_list[-2]:], llm_backbone.model.norm,
                        llm_backbone.score, llm_backbone.config, llm_backbone.num_labels, num_latents).to(f"cuda:0")
    return TempModel(beginBlock, endBlock)

class TempModel(nn.Module):
    def __init__(self, begin_module, end_module):
        super().__init__()
        self.begin_module = begin_module
        self.end_module = end_module

    def forward(self, prompt_text_embed, prompt_input_atts_pad, input_ids, attention_mask):
        input_ids, hidden_states, attention_mask, position_ids = self.begin_module(prompt_text_embed, prompt_input_atts_pad, input_ids, attention_mask)
        return self.end_module(input_ids, hidden_states, attention_mask, position_ids)


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