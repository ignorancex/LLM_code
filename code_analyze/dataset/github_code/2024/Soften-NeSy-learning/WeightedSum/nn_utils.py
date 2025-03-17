from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR

from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, DataLoader
from torch.distributions.categorical import Categorical
import os
from PIL import Image
import json
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import time
import sys 
sys.path.append("..") 
from params import arity, model_path 

np.set_printoptions(precision=2, suppress=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def equal_res(preds, gts):
    return (np.abs(preds - gts)) < 1e-2

#res_precision = 5

def eval_expr(preds, seq_len):
    res_preds = []
    expr_preds = []
    for i_pred, i_len in zip(preds, seq_len):
        i_pred = i_pred[:i_len]
        i_pred_1 = [i_pred[i] if i_pred[i] <= 9 else '/' for i in range(arity)]
        i_pred_2 = [i_pred[i]-9 if i_pred[i] >= 10 and i_pred[i] < 15 else '/' for i in range(arity,2*arity)]
        i_pred = i_pred_1 + i_pred_2
        i_expr = '+'.join([str(i_pred[i]) + '*' + str(i_pred[i+arity]) for i in range(arity)])
        try:
            i_res_pred = float(eval(i_expr))
        except:
            i_res_pred = np.inf
        res_preds.append(i_res_pred)
        expr_preds.append(i_expr)
    return expr_preds, res_preds

def eval_pred(i_pred, i_len):
    i_pred = i_pred[:i_len]
    i_pred_1 = [i_pred[i] if i_pred[i] <= 9 else '/' for i in range(arity)]
    i_pred_2 = [i_pred[i]-9 if i_pred[i] >= 10 and i_pred[i] < 15 else '/' for i in range(arity,2*arity)]
    i_pred = i_pred_1 + i_pred_2

    i_expr = '+'.join([str(i_pred[i]) + '*' + str(i_pred[i+arity]) for i in range(arity)])
    try:
        i_res_pred = float(eval(i_expr))
    except:
        i_res_pred = np.inf
    return i_expr, i_res_pred

def compute_rewards(preds, res, seq_len):
    expr_preds, res_preds = eval_expr(preds, seq_len)
    rewards = equal_res(res_preds, res)
    rewards = [1.0 if x else 0. for x in rewards]
    return np.array(rewards)

def save(net, file_name, epoch=0):
    state = {
            'net': net,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    save_point = './checkpoint/' + file_name + '_' + str(epoch) + '.t7'
    torch.save(state, save_point)
    return net

    
def evaluate(model, dataloader):
    model.eval() 
    res_all = []
    res_pred_all = []
    
    expr_all = []
    expr_pred_all = []

    for sample in dataloader:
        img_seq = sample['img_seq']
        label_seq = sample['label_seq']
        res = sample['res']
        seq_len = sample['len']
        expr = sample['expr']
        img_seq = img_seq.to(device)
        label_seq = label_seq.to(device)

        N, M, C, H, W = img_seq.shape
        x = img_seq.reshape(N*M, C, H, W).cuda()            
        batch_logits = model(x).reshape(N, M, -1)
        masked_probs = batch_logits.reshape(N, M, -1)
        selected_probs, preds = torch.max(masked_probs, -1)
        expr_preds, res_preds = eval_expr(preds.data.cpu().numpy(), seq_len)
        
        res_pred_all.append(res_preds)
        res_all.append(res)
        expr_pred_all.extend(expr_preds)
        expr_all.extend(expr)

    res_pred_all = np.concatenate(res_pred_all, axis=0)
    res_all = np.concatenate(res_all, axis=0)
    acc = equal_res(res_pred_all, res_all).mean()

    expr_pred_all = ''.join(expr_pred_all)
    expr_all = ''.join(expr_all)
    sym_acc = np.mean([x == y for x,y in zip(expr_pred_all, expr_all)])
    
    return acc, sym_acc

def Semantic_loss(x, y, num_classes=15):
    n = len(x)
    y_ = F.one_hot(y, num_classes)
    x_ = F.log_softmax(x, dim=-1)
    loss = -(x_*y_ + torch.clamp(1-x_, min=-4.6)*(1-y_)).sum() / n # log(0.01)=-4.6
    return loss

