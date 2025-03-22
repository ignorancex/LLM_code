import sys
sys.path.append("/opt/data/private/AllSpark/Code/")
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
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models import builder
from models.llm.modeling_llama import _expand_mask, _make_causal_mask
from models.utils import get_text_embed
from mst_datasets import UCMercedFewShotDataset, UCMercedFewShotBatchSampler, RS19FewShotDataset, RS19FewShotBatchSampler
from utils import Avg_values, MST_Logger


def get_args_parser():
    parser = argparse.ArgumentParser('AllSpark for RGB', add_help=False)
    
    parser.add_argument('--cfg', default="eval_configs/RGB_UCMerced_fewshot_test.yaml", type=str, help='config file')
    parser.add_argument('--exp-name', default="debug", type=str, help='current experiment name')
    parser.add_argument('--dataset', default='UCMerced', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--print-freq', default=1, type=int)
    parser.add_argument('--episodes', default=600, type=int)
    parser.add_argument('--shot', default=5, type=int)
    parser.add_argument('--way', default=5, type=int)
    parser.add_argument('--query', default=15, type=int)
    parser.add_argument('--device', default='cuda', type=str)

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

    # build dataset
    if args.dataset == "UCMerced":
        test_dataset = UCMercedFewShotDataset(root_path=cfg['root_path'], is_train=False, n_support=int(args.shot*args.way))
        sampler = UCMercedFewShotBatchSampler(class_to_indices=test_dataset.class_to_indices, num_episodes=args.episodes, n_way=args.way, k_shot=args.shot, q_query=args.query)
        test_loader = DataLoader(test_dataset, batch_sampler=sampler)
    elif args.dataset == "RS19":
        test_dataset = RS19FewShotDataset(root_path=cfg['root_path'], is_train=False, n_support=int(args.shot*args.way))
        sampler = RS19FewShotBatchSampler(class_to_indices=test_dataset.class_to_indices, num_episodes=args.episodes, n_way=args.way, k_shot=args.shot, q_query=args.query)
        test_loader = DataLoader(test_dataset, batch_sampler=sampler)
    else:
        raise NotImplementedError
    
    # build model
    modal_tokenizer = builder.build_visual2DTokenizer(img_size=224, embed_dim=1408, params_path=cfg['modal_tokenizer_checkpoint'], 
                                                      freeze_params=False, in_channels=3)
    modal_backbone = builder.build_evaBackbone(params_path=cfg['modal_backbone_checkpoint'], vision_feats_return_layer=-1)
    bridge = builder.build_bridge(cfg['bridge'])
    llm_tokenizer, llm_backbone = builder.build_LLM(cfg)
    model = ModelwLLMFewShot(modal_tokenizer, modal_backbone, bridge, llm_backbone).eval().to(args.device)

    for name, param in model.named_parameters():
        param.requires_grad = False
    
    # prompts    
    text_embed, input_atts_pad = get_text_embed(test_dataset.prompts[0], int(args.shot*args.way+args.query*args.way), llm_tokenizer, llm_backbone)

    evaluation(model, test_loader, test_dataset, args, eva_text_embed=text_embed, 
                    eva_input_atts_pad=input_atts_pad, mstLogger=mstLogger)
            
    print("Done!")
    
@torch.no_grad()
def evaluation(model, test_loader, test_dataset, args, eva_text_embed, eva_input_atts_pad, mstLogger):
    model.eval()
    
    run_time = Avg_values()
    test_iters = len(test_loader)
    accs = []
    for i, (support_images, support_labels, query_images, query_labels) in enumerate(test_loader):
        start_time = datetime.now()
        
        pred, cur_acc = model(support_images.squeeze().to(args.device), support_labels.squeeze().to(args.device), 
                        query_images.squeeze().to(args.device), query_labels.squeeze().to(args.device), 
                        eva_text_embed.to(args.device), eva_input_atts_pad.to(args.device))
        accs.append(cur_acc)
        
        end_time = datetime.now()
        run_time.update(end_time-start_time, 1)
        eta_str = str(run_time.avg * (len(test_loader) - run_time.count))
        
        if i % args.print_freq == 0:
            mstLogger.logger.info(f"[val] cur_iters:{i}/{test_iters} eta:{eta_str} cur_acc:{cur_acc}")
    
    mstLogger.logger.info("-" * 40)
    mstLogger.logger.info(f"acc: {np.mean(accs)}")
    mstLogger.logger.info("-" * 40)


class ModelwLLMFewShot(nn.Module):
    def __init__(self, modal_tokenizer, modal_backbone, bridge, llm):
        super().__init__()
        self.modal_tokenizer = modal_tokenizer
        self.modal_backbone = modal_backbone
        self.bridge = bridge
        self.llm = llm

    def forward(self, support_images, support_labels, query_images, query_labels, text_embed, input_atts_pad):
        n_support = support_images.shape[0]
        n_query = query_images.shape[0]
        n_class = len(torch.unique(support_labels))

        # Ensure that support and query have the correct sizes
        assert n_support % n_class == 0 and n_query % n_class == 0

        n_support_per_class = n_support // n_class
        n_query_per_class = n_query // n_class

        x = torch.cat([support_images, query_images], 0)

        # Encode the concatenated images
        z = self.forward_features(x, text_embed, input_atts_pad)
        z_dim = z.size(-1)

        # Compute class prototypes by averaging support images
        z_proto = z[:n_support].view(n_class, n_support_per_class, -1, z_dim).mean(1).mean(1) # all mean
        # z_proto = z[:n_support, -1].view(n_class, n_support_per_class, z_dim).mean(1) # last token

        # Query embeddings
        zq = z[n_support:].mean(1)
        # zq = z[n_support:, -1]

        # Compute distances between query embeddings and prototypes
        dists = euclidean_dist(zq, z_proto)
        dists = F.normalize(dists, p=2, dim=1)

        # Apply log softmax to the negative distances
        log_p_y = F.log_softmax(-dists, dim=1)

        _, y_hat_proto = log_p_y.max(1)
        true_cls_id = []
        for num in support_labels:
            if num not in true_cls_id:
                true_cls_id.append(num.item())
        predicted_labels = torch.tensor(true_cls_id).cuda().gather(0, y_hat_proto)

        acc_val = torch.eq(predicted_labels, query_labels).float().mean()

        return predicted_labels, acc_val.detach().cpu().item()
        
    def forward_features(self, src_input, text_embed, input_atts_pad):
        
        src_embed = self.modal_backbone(self.modal_tokenizer(src_input))
        if isinstance(src_embed, List):
            src_embed = src_embed[-1]
            
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
        for idx, decoder_layer in enumerate(self.llm.model.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=None,
                use_cache=None,
            )
            hidden_states = layer_outputs[0]
        
        hidden_states = self.llm.model.norm(hidden_states)
        
        return hidden_states
        
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


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


if __name__ == "__main__":
    args = get_args_parser()
    if args.exp_name is None:
        print("It is recommended to specific your experiment name")
        args.exp_name = ""
    
    with open(args.cfg, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f.read())
        
    args.output_dir = os.path.join("./eval_workdirs", args.exp_name, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, cfg)