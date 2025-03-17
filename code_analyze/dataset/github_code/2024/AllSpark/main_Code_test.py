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
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import builder
from models.llm.modeling_llama import _expand_mask, _make_causal_mask
from mst_datasets import build_CodeSearchNetDataset, compute_metrics
from utils import Avg_values, MST_Logger


def get_args_parser():
    parser = argparse.ArgumentParser('MultiModal with LLM for Code Test', add_help=False)
    
    # Common parameters
    parser.add_argument('--cfg', default="configs/Code_CodeSearchNet.yaml", type=str, help='config file')
    parser.add_argument('--exp-name', default="debug", type=str, help='current experiment name')
    parser.add_argument('--seed', default=2333, type=int)
    parser.add_argument('--num-workers', default=16, type=int)
    parser.add_argument('--print-freq', default=1, type=int)
    parser.add_argument('--batch_idx', default=1, type=int)
    parser.add_argument('--checkpoint', default="workdirs/MST_Code/javascript/2023-11-12 22:49:07/model.pth", type=str)
    parser.add_argument('--parallel-split-layer', default=[16, 32], type=int, nargs="+")
    
    # Dataset parameters
    # Support datasets: codesearch
    parser.add_argument('--dataset', default='codesearch', type=str, help='dataset name')
    parser.add_argument('--root-dir', default='/opt/data/private/dataset/MFM/Code/test', type=str)
    parser.add_argument('--lang', default='javascript', type=str)
    parser.add_argument('--batch-size', default=500, type=int)

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
    
    # build model
    llm_tokenizer, llm_backbone = builder.build_LLMForSegCls(cfg)
    
    model = build_ModelwLLM_Pipe(llm_backbone, args.parallel_split_layer)
    print("Load Weights")
    weights = torch.load(args.checkpoint, map_location='cpu')
    old_keys = list(weights.keys())
    for key in old_keys:
        if "partitions.0.0." in key:
            newKey = key.replace("partitions.0.0.", "0.")
        elif "partitions.1.0." in key:
            newKey = key.replace("partitions.1.0.", "1.")
        weights[newKey] = weights.pop(key)
    model.load_state_dict(weights, strict=True)
    model.cuda()
    
    dataset_name = args.dataset
    data_dir = os.path.join(args.root_dir, args.lang)
    batch_file = f"batch_{args.batch_idx}.txt"
    cfg['test_file'] = batch_file
    evaluation(args, cfg, model, llm_tokenizer, data_dir, dataset_name, mstLogger)
        
    print("Done!")
    
@torch.no_grad()
def evaluation(args, cfg, model, tokenizer, data_dir, dataset_name, mstLogger):
    model.eval()
    
    eval_dataset, instances = build_CodeSearchNetDataset(cfg, data_dir, tokenizer, dataset_name, ttype='test')
    
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    run_time = Avg_values()
    test_iters = len(eval_loader)
    preds = None
    for i, batch in enumerate(eval_loader):
        start_time = datetime.now()
        
        batch = tuple(t.cuda() for t in batch)
        labels = batch[3]

        from torchprofile import profile_macs
        
        macs = profile_macs(model, ((batch[0], batch[1])))
        print(f"{macs/1e9:.2f}G")
        exit(0)
        
        ouputs = model((batch[0], batch[1]))
        
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
    
    mstLogger.logger.info("Save Result...")
    output_test_file = os.path.join(args.output_dir, "res", cfg['test_file'])
    output_dir = os.path.dirname(output_test_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_test_file, "w") as writer:
        all_logits = preds.tolist()
        for i, logit in tqdm(enumerate(all_logits), desc='Testing'):
            instance_rep = '<CODESPLIT>'.join(
                [item.encode('ascii', 'ignore').decode('ascii') for item in instances[i]])

            writer.write(instance_rep + '<CODESPLIT>' + '<CODESPLIT>'.join([str(l) for l in logit]) + '\n')
        for key in sorted(result.keys()):
            print("%s = %s" % (key, str(result[key])))


class BeginModule(nn.Module):
    def __init__(self, embed_tokens, llm_layers):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.llm_layers = llm_layers
        
    def forward(self, inputs):
        input_ids, attention_mask = inputs

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
        
    def forward(self, inputs):
        
        input_ids, hidden_states, attention_mask, position_ids = inputs
        
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
                             llm_backbone.model.layers[0:parallel_list[0]])
    endBlock = EndModule(llm_backbone.model.layers[parallel_list[-2]:], llm_backbone.model.norm,
                        llm_backbone.score, llm_backbone.config, llm_backbone.num_labels)
    return TempModel(beginBlock, endBlock)

class TempModel(nn.Module):
    def __init__(self, begin_module, end_module):
        super().__init__()
        self.begin_module = begin_module
        self.end_module = end_module

    def forward(self, inputs):
        input_ids, hidden_states, attention_mask, position_ids = self.begin_module(inputs)
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