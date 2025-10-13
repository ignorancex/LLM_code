import os
import pickle
import json
import torch
from torchvision.transforms import Compose
from PIL import Image
import numpy as np

from config import cfg
from log import logger

from task_specific_baselines.cholec80 import Cholec80
from task_specific_baselines.endoscapes import Endoscapes
from task_specific_baselines.cholect50 import CholecT50
    
# class CholecDataset(torch.utils.data.Dataset):

#     def __init__(self,label_list,transform=None,class_num: int = -1):

#         self.transform = transform
#         self.class_num = class_num
#         self.data = label_list

#     def __getitem__(self,index):
#         impath, label, vid = self.data[index]
#         if 'cholec80' in cfg.checkpoint:
#             label = label[:7]
#         elif 'endo' in cfg.checkpoint:
#             label = label[7:10]
#         else:
#             label = label[10:]
#         image = Image.open(impath).convert("RGB")
#         label = torch.Tensor(label)

#         if self.transform:
#             image = self.transform(image)

#         return image , label, vid
    
#     def __len__(self):
#         return len(self.data)

#     def labels(self):
#         if 'cholec80' in cfg.checkpoint:
#             file = "cholec/phase_labels.txt"
#         elif 'endo' in cfg.checkpoint:
#             file = "cholec/endo_labels.txt"
#         else:
#             file = "cholec/triplet_labels.txt"
#         with open(file, 'r')as f:
#             text = f.read()
#         return text.split('\n')

def build_cholec80_dataset(train_preprocess: Compose,
                       val_preprocess: Compose,
                       pin_memory=True):

    val_dataset = Cholec80('val',val_preprocess)
    train_dataset = Cholec80('train',train_preprocess)
    test_dataset = Cholec80('test',val_preprocess)

    print(f"Length of train : {len(train_dataset)}")
    print(f"Length of val : {len(val_dataset)}")
    print(f"Length of test : {len(test_dataset)}")

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(  # type: ignore
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=pin_memory)

    val_loader = torch.utils.data.DataLoader(  # type: ignore
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=False)
        
    test_loader = torch.utils.data.DataLoader(  # type: ignore
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=False)

    logger.info("Build dataset done.")
    return [train_loader, val_loader, test_loader]
    
def build_cholect50_dataset(train_preprocess: Compose,
                       val_preprocess: Compose,
                       pin_memory=True):
    
    val_dataset = CholecT50('val',val_preprocess)
    train_dataset = CholecT50('train',train_preprocess)
    test_dataset = CholecT50('test',val_preprocess)

    print(f"Length of train : {len(train_dataset)}")
    print(f"Length of val : {len(val_dataset)}")
    print(f"Length of test : {len(test_dataset)}")

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(  # type: ignore
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=pin_memory)

    val_loader = torch.utils.data.DataLoader(  # type: ignore
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=False)
        
    test_loader = torch.utils.data.DataLoader(  # type: ignore
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=False)

    logger.info("Build dataset done.")
    return [train_loader, val_loader, test_loader]

def build_endo_dataset(train_preprocess: Compose,
                       val_preprocess: Compose,
                       pin_memory=True):
    
    val_dataset = Endoscapes('val',val_preprocess)
    train_dataset = Endoscapes('train',train_preprocess)
    test_dataset = Endoscapes('test',val_preprocess)

    print(f"Length of train : {len(train_dataset)}")
    print(f"Length of val : {len(val_dataset)}")
    print(f"Length of test : {len(test_dataset)}")

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(  # type: ignore
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=pin_memory)

    val_loader = torch.utils.data.DataLoader(  # type: ignore
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=False)
        
    test_loader = torch.utils.data.DataLoader(  # type: ignore
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=False)

    logger.info("Build dataset done.")
    return [train_loader, val_loader, test_loader]