# Copyright (c) Meta Platforms, Inc. and affiliates.

import os

from data.vaw_dataset import VAWValSubset
from data.coco_dataset import COCOValSubset

from eval_functions import validate

import clip
import torch

import argparse
import torch.backends.cudnn as cudnn
from utils.dist_utils import fix_random_seeds, init_distributed_mode, get_rank
from utils.gen_utils import bool_flag, strip_state_dict, none_flag
from functools import partial

from utils.model_utils import FeatureComb
from utils.dist_utils import CLIPDistDataParallel

# from config import genecis_root
genecis_root = 'data/genecis'
from functools import partial

from utils.metrics import AverageMeter
import torch
from tqdm import tqdm
import json
import multiprocessing
import os
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from typing import List, Tuple, Dict

from torch import nn
from torchvision.transforms import transforms

import clip
import numpy as np
import torch
import torch.nn.functional as F
from clip.model import CLIP
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.coco_dataset import COCODataset, COCOValSubset
from data.vaw_dataset import VAWDataset, VAWValSubset
from lavis.models import load_model_and_preprocess
from options import get_experiment_config
from set_up import setup_experiment
from models import create_models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_Defualt_Genecis_Root = 'data'

def get_args_parser():

    parser = argparse.ArgumentParser('Eval', add_help=False)

    parser.add_argument('--model', default='ViT-L/14', type=str, help='Which CLIP model we are using as backbone')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--batch_size_per_gpu', default=16, type=int)

    # Dist params
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    # Model params
    parser.add_argument('--combiner_mode', default='image_plus_text', type=str)
    parser.add_argument('--feature_comb_average', default=0.5, type=float)

    # Pretrain paths
    parser.add_argument('--clip_pretrain_path', default=None, type=none_flag)
    parser.add_argument('--combiner_pretrain_path', default=None, type=none_flag)

    # Dataset params
    parser.add_argument('--dataset', default='change_object', type=str, help='Eval dataset')
    parser.add_argument('--use_complete_text_query', default=False, type=bool_flag, help='Only relevant for MIT States')

    # Save params
    parser.add_argument('--pred_save_path', default=None, type=none_flag, help='Where to save predictions, dont save by default')

    return parser


def load_model(model_path, models: dict):
    """
    Load a test model
    :param model_path: path to the model
    :param models: dict of models
    :return: the model with checkpoint loaded
    """
    print(f"Load model")
    # 原先以字典的方式保存为 .pth
    log_data = torch.load(model_path)['model_state_dict']
    for model_name in models.keys():
        # models[model_name].load_state_dict(log_data[model_name])
        if model_name in log_data.keys():
            if isinstance(models[model_name], nn.DataParallel):
                models[model_name].module.load_state_dict(log_data[model_name])
            else:
                models[model_name].load_state_dict(log_data[model_name])
    return models


def main(args):

    transforms_configs = {
        "text_encoders": "clip",
        "image_encoders": "clip",
        "clip_image_model": "ViT-L/14",
        "clip_text_model": "ViT-L/14",
    }
    from data import transform_factory
    image_transforms, text_transforms = transform_factory(transforms_configs)
    image_transforms, text_transforms = image_transforms['val'], text_transforms['val']

    print('Loading models...')
    # configs = get_experiment_config()
    # export_root, configs = setup_experiment(configs)
    # models = create_models(configs)  # pretrain_dataset != 'None' can get model dict

    # define clip model and preprocess pipeline, get input_dim and feature_dim
    clip_model, preprocess = clip.load(args.model)
    # clip_model, preprocess = clip.load(configs["clip_text_model"])
    clip_model.float().eval()
    input_dim = clip_model.visual.input_resolution
    feature_dim = clip_model.visual.output_dim

    # --------------
    # GET COMBINER
    # --------------
    combiner = FeatureComb("image_plus_text", feature_comb_average=0.5)


    # --------------
    # To cuda
    # --------------
    clip_model, combiner = clip_model.cuda(), combiner.cuda()
    # if any([p.requires_grad for p in clip_model.parameters()]):
    #     clip_model = CLIPDistDataParallel(clip_model, device_ids=[args.gpu])
    # if any([p.requires_grad for p in combiner.parameters()]):
    #     combiner = torch.nn.parallel.DistributedDataParallel(combiner, device_ids=[args.gpu])

    # --------------
    # GET DATASET
    # --------------
    print('Loading datasets...')
    genecis_split_path = os.path.join(_Defualt_Genecis_Root, f'genecis/{configs["dataset"]}.json')
    # if configs["checkpoint_path"] != "None":
    #     print(f'Loading model from {configs["checkpoint_path"]}')
    #     models = load_model(configs["checkpoint_path"], models)
    if 'attribute' in configs["dataset"]:

        print(f'Evaluating on GeneCIS {configs["dataset"]} from {genecis_split_path}')

        val_dataset_subset = VAWValSubset(val_split_path=genecis_split_path, tokenizer=text_transforms,
                                          transform=image_transforms)
        print(f'Evaluating on {len(val_dataset_subset)} templates...')

    elif 'object' in configs["dataset"]:

        print(f'Evaluating on GeneCIS {configs["dataset"]} from {genecis_split_path}')

        val_dataset_subset = COCOValSubset(val_split_path=genecis_split_path, tokenizer=text_transforms,
                                           transform=image_transforms)
        print(f'Evaluating on {len(val_dataset_subset)} templates...')
    else:
        raise ValueError
    get_dataloader = partial(torch.utils.data.DataLoader, sampler=None,
                             batch_size=configs["batch_size"],
                             num_workers=configs["num_workers"],
                             pin_memory=True,
                             shuffle=False)

    valloader_subset = get_dataloader(dataset=val_dataset_subset)

    # --------------
    # EVALUTE
    # --------------
    # TODO: Adjust eval code for multiple GPUs
    if get_rank() == 0:
        print("get_rank() == 0")
        validate(clip_model, combiner, valloader_subset, topk=(1, 2, 3), save_path=None, models=models, img_weight=configs["img_weight"])

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Eval', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
    