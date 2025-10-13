# -*- coding: utf-8 -*-
"""
Created on Mon May 12 13:11:07 2025

@author: Molly
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score




#one epoch of training
def train(model,optimizer,x,y,train_loader):
    model.train()
    total_loss = 0
    for batch, sample_ids_adjs, full_ids_adjs in train_loader:
        model.zero_grad()
        out = model(x, sample_ids_adjs, full_ids_adjs)
        
        loss = F.nll_loss(out, y[batch])
        loss.backward()
        optimizer.step()
        total_loss += float(loss) 

    loss = total_loss / len(train_loader)
    return loss





@torch.no_grad()
#testing one epoch
def test(model,x,y,data,test_loader): #forward pass with no neighbor sampling and symmetrically normalized 
    model.eval()
    out, _ = model.inference_batch(x, test_loader)
    
    y_true = y.cpu()
    y_pred = out.cpu()
   
    
    y_pred=y_pred.log_softmax(dim=-1)
    y_pred=torch.argmax(y_pred,dim=1)

    train_acc = (y_pred[data["train_mask"]]==y_true[data["train_mask"]]).sum().item()/torch.where(data["train_mask"])[0].size(0)
    val_acc = (y_pred[data["val_mask"]]==y_true[data["val_mask"]]).sum().item()/torch.where(data["val_mask"])[0].size(0)
    test_acc = (y_pred[data["test_mask"]]==y_true[data["test_mask"]]).sum().item()/torch.where(data["test_mask"])[0].size(0)

    return train_acc, val_acc, test_acc




