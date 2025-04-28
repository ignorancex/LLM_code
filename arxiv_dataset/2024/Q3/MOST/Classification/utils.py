""" Utilities """
import os
import logging
import shutil
import torch
import torchvision.datasets as dset
import numpy as np
import torch.nn as nn

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) ==nn.ConvTranspose2d or type(m) ==nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def init_weights_var(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d or type(m) ==nn.Linear:
        nn.init.zeros_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def save_checkpoint(state, ckpt_dir, is_best=False, is_pert=False):
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best.pth.tar')
        shutil.copyfile(filename, best_filename)
    if is_pert:
        best_filename = os.path.join(ckpt_dir, 'pert.pth.tar')
        shutil.copyfile(filename, best_filename)
