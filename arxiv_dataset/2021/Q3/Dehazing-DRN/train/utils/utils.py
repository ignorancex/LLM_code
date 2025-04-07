import os
import math
import torch
import numpy as np
from datasets.pix2pix import pix2pix as commonDataset

def getLoader(dataroot, batchSize=64, shuffle=True):
    dataset = commonDataset(root=dataroot)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=shuffle)
    return dataloader

def compare_psnr(img1, img2):
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse < 1e-10:
        return 100
    psnr1 = 20 * math.log10(math.sqrt(mse))
    return -psnr1

def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    m.weight.data.normal_(0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
    m.weight.data.normal_(1.0, 0.02)
    m.bias.data.fill_(0)

def adjust_learning_rate(optimizer, init_lr, every):
  lrd = init_lr / every
  old_lr = optimizer.param_groups[0]['lr']
  lr = old_lr - lrd
  if lr < 0: lr = 0
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr

def create(path):
    if not os.path.exists(path):
        os.makedirs(path)

# def gradient(y):
#     gradient_h = torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])
#     gradient_y = torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])
#     return gradient_h, gradient_y