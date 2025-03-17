from torchvision import datasets, transforms
import os
import torch
from data_list import  Imagelist
from synthetic_dataset import SyntheticDataset
import logging
import numpy as np
from PIL import ImageFilter
from pathlib import Path


class ResizeImage():
    def __init__(self, size):
      if isinstance(size, int):
        self.size = (int(size), int(size))
      else:
        self.size = size
    def __call__(self, img):
      th, tw = self.size
      return img.resize((th, tw))



def return_dataset_selfsup(
    args, batch_size=36, train_split='train', 
    val_split='val', train_root=None, val_root=None,
    train_split2=None, preprocess_tr=None, preprocess_val=None):
    # NOTE : train_split2 will be used for target domain if provided
            
    train_root = train_root or 'data'
    train_root = Path(train_root)
    val_root = val_root or 'data' # Set it to default if None
    val_root = Path(val_root)
    
    crop_size = 224
    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    if preprocess_tr is None:
        preprocess_tr = data_transforms['train']
    if preprocess_val is None:
        preprocess_val = data_transforms['test']
        

    source_dataset = Imagelist(train_root/args.dataset/f'{args.source}_{train_split}.txt', transform=preprocess_tr)
    if train_split2 is None:
        train_split2 = train_split
        label_offset = 0
    else:
        label_offset = len(np.unique(source_dataset.labels))
    target_dataset = Imagelist(train_root/args.dataset/f'{args.target}_{train_split2}.txt', transform=preprocess_tr, label_offset=label_offset)
    
    class_list = np.unique(list(source_dataset.labels) + list(target_dataset.labels))
    source_dataset_val = Imagelist(val_root/args.dataset/f'{args.source}_{val_split}.txt', transform=preprocess_val)
    target_dataset_val = Imagelist(val_root/args.dataset/f'{args.target}_{val_split}.txt', transform=preprocess_val)
    
    if args.syn_root:
        # NOTE that source_syn_dataset is the target domain counterpart of src dataset
        source_syn_dataset = SyntheticDataset(
            source_dataset, Path(args.real_root) / args.dataset / args.source, Path(args.syn_root) / args.dataset / args.s2t, transform=preprocess_tr)
        target_syn_dataset = SyntheticDataset(
            target_dataset, Path(args.real_root) / args.dataset / args.target, Path(args.syn_root) / args.dataset / args.t2s, transform=preprocess_tr)
        
        # Concatenating the synthetic data with the real data
        source_dataset = torch.utils.data.ConcatDataset([source_dataset, target_syn_dataset])
        source_dataset.labels = np.concatenate([source_dataset.datasets[0].labels, source_dataset.datasets[1].labels])
        source_dataset.transform = preprocess_tr
        target_dataset = torch.utils.data.ConcatDataset([target_dataset, source_syn_dataset])
        target_dataset.labels = np.concatenate([target_dataset.datasets[0].labels, target_dataset.datasets[1].labels])
        target_dataset.transform = preprocess_tr
        

    # some batch size balancing across source and target. Code from CDS
    bs = batch_size
    bs_fact1 = 2
    bs_fact2 = 2
    if len(source_dataset) > len(target_dataset):
        bs_fact1 = int(bs_fact1 * len(source_dataset) / len(target_dataset))
    elif len(target_dataset) > len(source_dataset):
        bs_fact2 = int(bs_fact2 * len(target_dataset) / len(source_dataset))
    
    from torch.utils.data import RandomSampler
    
    src_sampler_seed = int(torch.empty((), dtype=torch.int64).random_().item())
    tgt_sampler_seed = int(torch.empty((), dtype=torch.int64).random_().item())
    sampler_src = RandomSampler(source_dataset, generator=torch.Generator().manual_seed(src_sampler_seed))
    sampler_tgt = RandomSampler(target_dataset, generator=torch.Generator().manual_seed(tgt_sampler_seed))
    source_loader = torch.utils.data.DataLoader(
        source_dataset, batch_size=bs*bs_fact1, num_workers=args.num_workers, sampler=sampler_src, drop_last=True)
    target_loader = torch.utils.data.DataLoader(
        target_dataset, batch_size=bs*bs_fact2, num_workers=args.num_workers, sampler=sampler_tgt, drop_last=True)
    target_loader_val = torch.utils.data.DataLoader(
        target_dataset_val, batch_size=bs, num_workers=args.num_workers, shuffle=False, drop_last=False)
    source_loader_val = torch.utils.data.DataLoader(
        source_dataset_val, batch_size=bs, num_workers=args.num_workers, shuffle=False, drop_last=False)
    
    if args.syn_root:
        source_syn_loader = None
        target_syn_loader = None
        
        # NOTE that target_dataset.datasets[0] is the real data in the target domain
        # source_dataset contains [real_data, syn_data] in source_domain,
        # source_syn_dataset contains [syn_data, real_data] in target_domain
        # real - synthetic counterparts match up with the same indexes in source_dataset and source_syn_dataset 
        source_syn_dataset = torch.utils.data.ConcatDataset([source_syn_dataset, target_dataset.datasets[0]])
        source_syn_dataset.labels = np.concatenate([source_syn_dataset.datasets[0].labels, source_syn_dataset.datasets[1].labels])
        source_syn_dataset.transform = preprocess_tr
        target_syn_dataset = torch.utils.data.ConcatDataset([target_syn_dataset, source_dataset.datasets[0]])
        target_syn_dataset.labels = np.concatenate([target_syn_dataset.datasets[0].labels, target_syn_dataset.datasets[1].labels])
        target_syn_dataset.transform = preprocess_tr
            
        sampler_src_syn = RandomSampler(source_syn_dataset, generator=torch.Generator().manual_seed(src_sampler_seed))
        source_syn_loader = torch.utils.data.DataLoader(
            source_syn_dataset, batch_size=bs*bs_fact1, num_workers=args.num_workers, sampler=sampler_src_syn, drop_last=True)
            
        sampler_tgt_syn = RandomSampler(target_syn_dataset, generator=torch.Generator().manual_seed(tgt_sampler_seed))
        target_syn_loader = torch.utils.data.DataLoader(
            target_syn_dataset, batch_size=bs*bs_fact2, num_workers=args.num_workers, sampler=sampler_tgt_syn, drop_last=True)
        
        return source_loader, target_loader, target_loader_val, source_loader_val, class_list, source_syn_loader, target_syn_loader
    else:
        return source_loader, target_loader, target_loader_val, source_loader_val, class_list


