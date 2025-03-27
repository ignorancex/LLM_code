import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import numpy as np

### RVSM utils

def thres(w,gamma):
        return w.where(torch.abs(w)>gamma, torch.zeros_like(w))
    
def newgrad(weight):
    w = weight.data
    u = thres(w)
    weight.grad += beta*(w-u)
    
def sparse_layer(layer):
    newgrad(layer.bn1.weight)
    newgrad(layer.conv1.weight)
    newgrad(layer.bn2.weight)
    newgrad(layer.conv2.weight)
    
def replace_sparse_weight(layer):
    layer.bn1.weight.data = thres(layer.bn1.weight.data)
    layer.conv1.weight.data = thres(layer.conv1.weight.data)
    layer.bn2.weight.data = thres(layer.bn2.weight.data)
    layer.conv2.weight.data = thres(layer.conv2.weight.data)
    
def replace_weight(weight,gamma):
    weight.data = thres(weight.data,gamma)

### ADMM utils
    
def projection(weight,percent,sparsity):
    if sparsity == 'elem':
        wt = weight.detach().cpu().clone()
        pcen = np.percentile(wt,percent)
        under_threshold = abs(wt) < pcen
        wt[under_threshold] = 0
        
        
    if sparsity == 'channel':
        wt = weight.detach().cpu().clone()
        ch_wt = wt.pow(2).sum(dim=[0,2,3]).add(1e-8).pow(1/2.)
        for i in range(len(ch_wt)):
            pcen = np.percentile(ch_wt, percent)
            ch_wt[i] = ( ch_wt[i]>pcen )
            wt[:,i,:,:] *= ch_wt[i]
    return wt.cuda()
