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

import timm
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
    vit = timm.create_model('vit_base_patch16_224_in21k', pretrained=True)
    model = ModelwoLLMFewShot(vit).eval().to(args.device)

    for name, param in model.named_parameters():
        param.requires_grad = False
    

    evaluation(model, test_loader, test_dataset, args, mstLogger=mstLogger)
            
    print("Done!")
    
@torch.no_grad()
def evaluation(model, test_loader, test_dataset, args, mstLogger):
    model.eval()
    
    run_time = Avg_values()
    test_iters = len(test_loader)
    accs = []
    for i, (support_images, support_labels, query_images, query_labels) in enumerate(test_loader):
        start_time = datetime.now()
        
        pred, cur_acc = model(support_images.squeeze().to(args.device), support_labels.squeeze().to(args.device), 
                        query_images.squeeze().to(args.device), query_labels.squeeze().to(args.device))
        accs.append(cur_acc)
        
        end_time = datetime.now()
        run_time.update(end_time-start_time, 1)
        eta_str = str(run_time.avg * (len(test_loader) - run_time.count))
        
        if i % args.print_freq == 0:
            mstLogger.logger.info(f"[val] cur_iters:{i}/{test_iters} eta:{eta_str} cur_acc:{cur_acc}")
    
    mstLogger.logger.info("-" * 40)
    mstLogger.logger.info(f"acc: {np.mean(accs)}")
    mstLogger.logger.info("-" * 40)


class ModelwoLLMFewShot(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, support_images, support_labels, query_images, query_labels):
        n_support = support_images.shape[0]
        n_query = query_images.shape[0]
        n_class = len(torch.unique(support_labels))

        # Ensure that support and query have the correct sizes
        assert n_support % n_class == 0 and n_query % n_class == 0

        n_support_per_class = n_support // n_class
        n_query_per_class = n_query // n_class

        x = torch.cat([support_images, query_images], 0)

        # Encode the concatenated images
        z = self.model.forward_features(x)
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