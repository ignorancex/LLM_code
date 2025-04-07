import os
import sys
import cv2
import time
import math
import h5py
import random
import argparse
import numpy as np
import scipy.misc
from skimage import measure
from skimage.measure import compare_ssim

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.utils as vutils
import torchvision.models as models

from misc import *

torch.cuda.set_device(1)

parser = argparse.ArgumentParser()
parser.add_argument('--valDataroot', required=False, default='', help='path to val dataset')
parser.add_argument('--valBatchSize', type=int, default=1, help='input batch size')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
opt = parser.parse_args()

valDataloader = getLoader(opt.valDataroot, opt.valBatchSize, shuffle=False)


if opt.netG != '':
    netG = torch.load(opt.netG)

netG.cuda()

total_ssim = 0
total_psnr = 0
for val_batch_id, val_data in enumerate(valDataloader, 0):
    varIn, varTar, _, _ = val_data

    varIn, varTar = Variable(varIn), Variable(varTar)
    varIn, varTar = varIn.float().cuda(), varTar.float().cuda()

    prediction, val_tran_hat, val_atp_hat, val_dehaze = netG(varIn)

    prediction = prediction.data.cpu().numpy().squeeze()
    prediction = prediction.transpose((1, 2, 0))

    varTar = varTar.data.cpu().numpy().squeeze()
    varTar = varTar.transpose((1, 2, 0))

    per_ssim = compare_ssim(prediction, varTar, multichannel=True)
    total_ssim = total_ssim + per_ssim

    per_psnr = compare_psnr(prediction, varTar)
    total_psnr = total_psnr + per_psnr

avg_ssim = total_ssim / len(valDataloader)
print('===>avg_ssim: %.6f' % avg_ssim)
avg_psnr = total_psnr / len(valDataloader)
print('===>avg_psnr: %.6f' % avg_psnr)




