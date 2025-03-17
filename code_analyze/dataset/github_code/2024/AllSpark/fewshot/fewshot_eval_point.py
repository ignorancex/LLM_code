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
from models.modal_tokenizer import PointTokenizer
from models.utils import fps, get_text_embed
from mst_datasets import FSLPointDataset, FSLPointBatchSampler
from mst_datasets.threeD.PointCloud import data_transforms
from timm.models.vision_transformer import Block
from utils import Avg_values, MST_Logger


def get_args_parser():
    parser = argparse.ArgumentParser('AllSpark for PointCloud', add_help=False)
    
    # Common parameters
    parser.add_argument('--cfg', default="eval_configs/PointCloud_ShapeNet_fewshot_test.yaml", type=str, help='config file')
    parser.add_argument('--exp-name', default="debug", type=str, help='current experiment name')
    # 'ShapeNet70_FS', 'ScanObjectNN_FS'
    parser.add_argument('--dataset', default='ShapeNet70_FS', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--print-freq', default=1, type=int)
    parser.add_argument('--episodes', default=700, type=int)
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
    
    # build model
    modal_tokenizer = PointTokenizer(cfg)
    print(f"### Load modal_tokenizer params")
    state_dict = torch.load(cfg['modal_tokenizer_checkpoint'], map_location='cpu')
    modal_tokenizer.load_state_dict(state_dict, strict=True)
    modal_backbone = nn.Sequential(*[
            Block(
                dim=1408,
                num_heads=8,
                mlp_ratio=4.,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU
            )
            for i in range(6)])
    print(f"### Load modal_backbone params")
    state_dict = torch.load(cfg['modal_backbone_checkpoint'], map_location='cpu')
    modal_backbone.load_state_dict(state_dict, strict=True)
    bridge = builder.build_bridge(cfg['bridge'])
    llm_tokenizer, llm_backbone = builder.build_LLM(cfg)
    model = ModelwLLMFewShot(args, modal_tokenizer, modal_backbone, bridge, llm_backbone).eval().to(args.device)
    
    # build dataset
    test_dataset = FSLPointDataset(data_path=cfg['root_path'], dataset=args.dataset, split=None, mode='test')
    test_sampler = FSLPointBatchSampler(way=args.way, shot=args.shot, query=args.query, labels=test_dataset.label, iterations=args.episodes)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler)

    prompts = [
        "Classify the provided point cloud sample into the correct category.",
        "Look at the point cloud data characteristics and classify the object.",
        "Please analyze the given point cloud dataset and determine which category "+\
            "it belongs to.Focus on the shape and structure evident in the point cloud."
    ]
    
    # run   
    text_embed, input_atts_pad = get_text_embed(prompts[0], int(args.shot*args.way+args.query*args.way), llm_tokenizer, llm_backbone)
    
    evaluation(model, test_loader, test_dataset, args, eva_text_embed=text_embed, 
            eva_input_atts_pad=input_atts_pad, mstLogger=mstLogger)
        
    print("Done!")
    
@torch.no_grad()
def evaluation(model, test_loader, test_dataset, args, eva_text_embed, eva_input_atts_pad, mstLogger):
    model.eval()
    
    run_time = Avg_values()
    test_iters = len(test_loader)

    npoint = test_dataset.npoint

    accs = []
    
    for i, data in enumerate(test_loader):
        start_time = datetime.now()
        
        cur_text_embed = eva_text_embed[0:data[0].size(0)].to(args.device)
        cur_input_atts_pad = eva_input_atts_pad[0:data[0].size(0)].to(args.device)
        assert cur_text_embed.size(0) == data[0].size(0)
        
        points = data[0].to(args.device)
        label = data[1].to(args.device)

        points = fps(points, npoint)
        
        _, cur_acc = model(points, label, cur_text_embed, cur_input_atts_pad)
        
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
    def __init__(self, args, modal_tokenizer, modal_backbone, bridge, llm):
        super().__init__()
        self.args = args
        self.modal_tokenizer = modal_tokenizer
        self.modal_backbone = modal_backbone
        self.bridge = bridge
        self.llm = llm
    
    def split_supp_and_query(self, y, xyz, p_feat, g_feat):
        """
        split supp set and query set
        input:  y: labels [B,N]
                g_feat:    global features [B,C]
                p_feat:    point-wised features [B,N,C]
        return: xyz:        raw coordinate [B,N,3]
                z_support:  support set features [nway * nshot, C]
                z_query:    query set features [nway * nquery, C]
                p_feat:     point-wised features [B,N,C]
        """
        class_unique = torch.unique(y)
        s_idx = torch.stack(list(map(lambda c: y.eq(c).nonzero()[:self.args.shot], class_unique))).view(-1)  # support id
        q_idx = torch.stack(list(map(lambda c: y.eq(c).nonzero()[self.args.shot:], class_unique))).view(-1)  # query id
        z_support, z_query = g_feat[s_idx], g_feat[q_idx]
        p_feat = torch.cat((p_feat[s_idx], p_feat[q_idx]), dim=0)
        y_support, y_query, xyz = y[s_idx], y[q_idx], torch.cat((xyz[s_idx], xyz[q_idx]), dim=0)
        return y_support, y_query, xyz, p_feat, z_support, z_query

    def forward(self, points, label, text_embed, input_atts_pad):
        # points: B N C
        # z: B n D
        # label: B
        z = self.forward_features(points, text_embed, input_atts_pad)
        # y: B  z_support: ns D  z_query: nq D
        y_support, y_query, xyz, p_feat, z_support, z_query = self.split_supp_and_query(label, points, z, z.mean(1))

        z_proto = z_support.contiguous().view(args.way, args.shot, -1).mean(1)

        # Compute distances between query embeddings and prototypes
        dists = euclidean_dist(z_query, z_proto)
        dists = F.normalize(dists, p=2, dim=1)

        # Apply log softmax to the negative distances
        log_p_y = F.log_softmax(-dists, dim=1)

        _, y_hat_proto = log_p_y.max(1)
        true_cls_id = []
        for num in y_support:
            if num not in true_cls_id:
                true_cls_id.append(num.item())
        predicted_labels = torch.tensor(true_cls_id).cuda().gather(0, y_hat_proto)

        acc_val = torch.eq(predicted_labels, y_query).float().mean()

        return predicted_labels, acc_val.detach().cpu().item()
        
    def forward_features(self, src_input, text_embed, input_atts_pad):
        
        src_embed = self.modal_backbone(self.modal_tokenizer(src_input))
        if isinstance(src_embed, List):
            src_embed = src_embed[-1]
            
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