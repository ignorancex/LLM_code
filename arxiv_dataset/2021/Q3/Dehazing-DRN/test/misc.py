import torch
import os 
import sys
import math
import numpy as np
import torchvision.transforms as transforms
from datasets.pix2pix_val import pix2pix_val as commonDataset

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
