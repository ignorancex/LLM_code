import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable



def gpls(weights, lamb):
# compute the norm in channels
  rs = 0
  for wt in weights:
    rs += wt.pow(2).sum(dim=[0,2,3]).add(1e-8).pow(1/2.).sum() 
  rs *= lamb
  return rs

# def gpl0(weights, lamb, eps=1e-8):
#   rs = 0
#   for wt in weights:
#     rs += (wt.pow(2).sum(dim=[0,2,3]).pow(1/2.)>eps).sum()
#   rs = rs.float()
#   rs *= lamb
#   return rs

def gpT1(weights, lamb, a=1.):
  rs = 0
  for wt in weights:
    tmp = wt.pow(2).sum(dim=[0,2,3]).add(1e-8).pow(1/2.)
    tmp = torch.div( tmp*(1.+a), a+tmp  )
    rs += tmp.sum()  #.add(1e-8)
  rs *= lamb
  return rs

def rgsm_ls(weights, lamb, beta, eps=1e-8):
  tmp = []
  if torch.cuda.is_available():
    for wt in weights:
      tmp_wt = wt.detach().clone().cuda()
      tmp.append(tmp_wt)
  else:
    for wt in weights:
      tmp_wt = wt.detach().clone()
      tmp.append(tmp_wt)
  zero_t = torch.FloatTensor([0]) 
  if torch.cuda.is_available():
    zero_t = zero_t.cuda()
  for wt in tmp:
    # sz = torch.FloatTensor([wt.size(0)*wt.size(2)*wt.size(3)])
    ch_wt = wt.pow(2).sum(dim=[0,2,3]).add(1e-8).pow(1/2.)
    for i in range(len(ch_wt)):
      if ch_wt[i] > eps :
        ch_wt[i] = torch.max( ch_wt[i]-(lamb/beta), zero_t )/(ch_wt[i])
        # print(ch_wt[i])
        wt[:,i,:,:] *= ch_wt[i]
      else:
        wt[:,i,:,:] *= 0
  return tmp

def rgsm_l0(weights, lamb, beta):
  tmp = []
  for wt in weights:
    tmp_wt = wt.detach().clone()
    tmp.append(tmp_wt)
  for wt in tmp:
    # sz = torch.FloatTensor([wt.size(0)*wt.size(2)*wt.size(3)])
    ch_wt = wt.pow(2).sum(dim=[0,2,3]).add(1e-8).pow(1/2.)
    for i in range(len(ch_wt)):
      ch_wt[i] = ( ch_wt[i]>np.sqrt(2*(lamb/beta)) )
      wt[:,i,:,:] *= ch_wt[i]
  return tmp