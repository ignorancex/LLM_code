from functools import partial
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

def get_recall(indices, targets):  # recall --> wether next item in session is within top K recommended items or not
    """
    Code adapted from: https://github.com/hungthanhpham94/GRU4REC-pytorch/blob/master/lib/metric.py
    Calculates the recall score for the given predictions and targets
    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B) or (BxN): torch.LongTensor. actual target indices.
    Returns:
        recall (float): the recall score
    """

    # One hot label branch
    if len(targets.size()) == 1:

        targets = targets.view(-1, 1).expand_as(indices)
        hits = (targets == indices).nonzero()
        if len(hits) == 0:
            return 0
        n_hits = (targets == indices).nonzero()[:, :-1].size(0)
        recall = float(n_hits) / targets.size(0)
        return recall

    # Multi hot label branch
    else:

        recall = []

        for preds, gt in zip(indices, targets):

            max_val = torch.max(torch.cat([preds, gt])).int().item()
            preds_binary = torch.zeros((max_val + 1,), device=preds.device, dtype=torch.float32).scatter_(0, preds, 1)
            gt_binary = torch.zeros((max_val + 1,), device=gt.device, dtype=torch.float32).scatter_(0, gt.long(), 1)

            success = (preds_binary * gt_binary).sum() > 0

            if success:
                recall.append(1)
            else:
                recall.append(0)

        return torch.Tensor(recall).float().mean()
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
@torch.no_grad()
def validate(valloader, topk=(1, 2, 3), save_path=None, models=None, img_weight=0):
    import torch.nn.functional as F
    print('Computing eval with combiner {}'.format(models['compositor']))

    # clip_model.eval()
    # combiner.eval()

    meters = {k: AverageMeter() for k in topk}
    sims_to_save = []

    print(f'Computing eval WITHOUT trained combiner... img_weight: {img_weight}')

    for key in models.keys():
        models[key].float().eval()
    text_encoder = models['text_encoder']
    image_encoder = models['image_encoder']

    def combiner_func(ref_feats, caption_feats):
        if img_weight > 1:# Image only
            return F.normalize(ref_feats, dim=-1)
        # composed_embeds = img_weight * F.normalize(ref_feats, dim=-1) + F.normalize(caption_feats, dim=-1)
        composed_embeds = 0.5 * ref_feats + (1 - 0.5) * caption_feats

        return F.normalize(composed_embeds, dim=-1)
    combiner = combiner_func if configs['compositor'] == "None" else models['compositor']

    with torch.no_grad():
        for batch in tqdm(valloader):

            ref_img, caption, gallery_set, target_rank = [x.cuda(non_blocking=True) for x in batch[:4]]
            bsz, n_gallery, _, h, w = gallery_set.size()
            caption = caption.squeeze()

            # Forward pass in CLIP
            # imgs_ = torch.cat([ref_img, gallery_set.view(-1, 3, h, w)], dim=0)
            # all_img_feats = clip_model.encode_image(imgs_).float()
            # caption_feats = clip_model.encode_text(caption).float()
            imgs_ = torch.cat([ref_img, gallery_set.view(-1, 3, h, w)], dim=0)
            all_img_feats = image_encoder(imgs_).float()
            caption_feats = text_encoder(caption).float()

            # L2 normalize and view into correct shapes
            ref_feats, gallery_feats = all_img_feats.split((bsz, bsz * n_gallery), dim=0)
            gallery_feats = gallery_feats.view(bsz, n_gallery, -1)
            gallery_feats = torch.nn.functional.normalize(gallery_feats, dim=-1)

            # Forward pass in combiner
            if configs['compositor'] == "Combiner":
                combined_feats = F.normalize(combiner.combine_features(ref_feats, caption_feats), dim=-1)
            else:
                combined_feats = combiner(ref_feats, caption_feats)


            # Compute similarity
            similarities = combined_feats[:, None, :] * gallery_feats       # B x N x D
            similarities = similarities.sum(dim=-1)                         # B x N

            # Sort the similarities in ascending order (closest example is the predicted sample)
            _, sort_idxs = similarities.sort(dim=-1, descending=True)                   # B x N

            # Compute recall at K
            for k in topk:

                recall_k = get_recall(sort_idxs[:, :k], target_rank)
                meters[k].update(recall_k, bsz)

            sims_to_save.append(similarities.cpu())

        if save_path is not None:
            sims_to_save = torch.cat(sims_to_save)
            print(f'Saving text only preds to: {save_path}')
            torch.save(sims_to_save, save_path)

        # Print results
        print_str = '\n'.join([f'Recall @ {k} = {v.avg:.4f}' for k, v in meters.items()])
        print(print_str)
        # 两位小数
        latex_str = ' & '.join([f'{v.avg * 100:.2f}' for k, v in meters.items()])
        print(latex_str)

        return meters

from data import transform_factory

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
def main(configs: dict, models: dict):
    # --------------
    # To cuda
    # --------------
    # if any([p.requires_grad for p in clip_model.parameters()]):
    #     clip_model = CLIPDistDataParallel(clip_model, device_ids=[args.gpu])
    # if any([p.requires_grad for p in combiner.parameters()]):
    #     combiner = torch.nn.parallel.DistributedDataParallel(combiner, device_ids=[args.gpu])

    # --------------
    # GET DATASET
    # --------------
    print('Loading datasets...')
    genecis_split_path = os.path.join(_Defualt_Genecis_Root, f'genecis/{configs["dataset"]}.json')
    if configs["checkpoint_path"] != "None":
        print(f'Loading model from {configs["checkpoint_path"]}')
        models = load_model(configs["checkpoint_path"], models)
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
    validate(valloader_subset, topk=(1, 2, 3), save_path=None, models=models, img_weight=configs["img_weight"])

if __name__ == '__main__':
    print('Loading models...')
    configs = get_experiment_config()
    export_root, configs = setup_experiment(configs)

    models = create_models(configs)  # pretrain_dataset != 'None' can get model dict
    print('Loading models ', models.keys())
    for key in models.keys():
        models[key].to(device)
    from data import transform_factory

    image_transforms, text_transforms = transform_factory(configs)
    image_transforms, text_transforms = image_transforms['val'], text_transforms['val']
    if configs["dataset"] == "all":
        dataset = ['change_object', 'change_attribute', 'focus_object', 'focus_attribute']
        for data in dataset:
            configs["dataset"] = data
            main(configs, models)
    else:
        main(configs, models)