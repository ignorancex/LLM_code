#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Aug 14 09:54:50 2024

@author: ishan
"""

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


##Data loader
def imgdata_loader(path_ip, path_op, path_load, batch_size, data_samprate, stats, shuffle):
    mean = stats['arr_0'][0]['mean_ip']
    std = stats['arr_0'][0]['std_ip']
    x = np.load(path_ip, allow_pickle = True)
    x=x['arr_0']
    x = (x-mean)/std
    y = np.load(path_op, allow_pickle=True)
    y=y['arr_0']
    y = (y-mean)/std
    n_fullsim = int(x.shape[0])
    ae = False
    tensor_x = torch.Tensor(x[:int(n_fullsim/data_samprate)])
    tensor_y = torch.Tensor(y[:int(n_fullsim/data_samprate)])
    loading = 'separatefmt'
    if loading == 'lumpedfmt':
        l = np.unique(x[:,:,-1])
        l =l.reshape(-1,1)
        tensor_load = torch.tensor(l)
        grid_dim = np.sqrt(y.shape[-1]).astype('int')
        tensor_x = tensor_x.reshape(x.shape[0], grid_dim, grid_dim, -1)
        tensor_x = torch.swapaxes(tensor_x, 1,-1)
        custom_dataset = TensorDataset(tensor_x, tensor_load, tensor_y) 
    else:
        l = np.load(path_load, allow_pickle=True)
        l = l['arr_0']
        l =l[:int(n_fullsim/data_samprate)].reshape(-1,l.shape[-1])
        tensor_load = torch.tensor(l)
        if ae:
            custom_dataset = TensorDataset(tensor_x) 
        else:
            custom_dataset = TensorDataset(tensor_x,tensor_load, tensor_y) 
    
    dataloader = DataLoader(custom_dataset,batch_size, shuffle)
    return dataloader  

##Forward pass for UNet
def conv_forward_pass_unet(inputs, loading, outputs, model, device):
    br_ip= (loading).to(torch.float32).contiguous().to(device)
    tr_ip= (inputs).to(torch.float32).contiguous().to(device)
    targets = (outputs).to(torch.float32).to(device)
    preds = model(tr_ip, br_ip).to(device)
    return targets, preds

##Forward pass for DeepONet
def conv_forward_pass(inputs, loading, outputs, model, device):
    br_ip= (loading).to(torch.float32).contiguous().to(device)
    tr_ip= (inputs).to(torch.float32).contiguous().to(device)
    targets = (outputs).to(torch.float32).to(device)
    preds, br_op, tr_op = model(br_ip, tr_ip)
    preds = (preds).to(device)
    return targets, preds, br_op, tr_op


##Forward pass for ensemble
def addition_moe_forward_pass(inputs, loading, outputs, model_list, device):
    br_ip= (loading).to(torch.float32).contiguous().to(device)
    tr_ip= (inputs).to(torch.float32).contiguous().to(device)
    targets = (outputs).to(torch.float32).to(device)
    dims = [len(model_list)] + list(targets.size())
    preds = torch.zeros(dims)
    for i in range(len(model_list)):
        preds[i] =  model_list[i](tr_ip, br_ip).to(device)
    
    ## if training multiple models in a single run, modify here
    if len(model_list) == 1:
        preds_add = preds
    else:
        preds_add = torch.mean(preds, dim=0)
  
    return targets, preds_add


def save_checkpoint(model, optimizer, full_path, epoch):
    save_path = full_path + '_chkpt_iter' + str(epoch)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, save_path)
    
def load_checkpoint(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    
    return model, optimizer, epoch




##Recursive prediction with UNet
def recursive_pred_unet(inputs, loading, outputs, model, device, timesteps):
    br_ip= (loading).to(torch.float32).contiguous().to(device)
    n_loads = br_ip.shape[-1]
    targets = (outputs).to(torch.float32).to(device)
    img_dim1, img_dim2 = targets.shape[-2], targets.shape[-1]
    preds_recursive = torch.zeros((timesteps,img_dim1,img_dim2))
    prompt = (inputs[0]).to(torch.float32).contiguous().to(device)
    prompt = prompt.reshape(1,-1, img_dim1, img_dim2)
    for i in range(timesteps):
        preds = model(prompt.reshape(1,-1, img_dim1, img_dim2), 
                                    br_ip[i].reshape(-1,n_loads))
        prompt = torch.cat((prompt[:,1:,:,:], preds), dim=1)
        preds_recursive[i] = preds
    preds_recursive = preds_recursive.to(device)
    return outputs, preds_recursive


##Recursive prediction with ensemble approach
def recursive_pred_moe(inputs, loading, outputs, model_list, device, timesteps):
    br_ip= (loading).to(torch.float32).contiguous().to(device)
    n_loads = br_ip.shape[-1]
    targets = (outputs).to(torch.float32).to(device)
    img_dim1, img_dim2 = targets.shape[-2], targets.shape[-1]
    preds_recursive = torch.zeros((timesteps,img_dim1,img_dim2))
    prompt = (inputs[0]).to(torch.float32).contiguous().to(device)
    prompt = prompt.reshape(1,-1, img_dim1,img_dim2)
    for i in range(timesteps):
        _, preds = addition_moe_forward_pass(prompt.reshape(1,-1, img_dim1,img_dim2), br_ip[i].reshape(-1,n_loads), 
                                 outputs[i].reshape(1,-1,img_dim1,img_dim2), model_list, device)
        prompt = torch.cat((prompt[:,1:,:,:], preds), dim=1)
        preds_recursive[i] = preds[0]
    preds_recursive = preds_recursive.to(device)
    return outputs, preds_recursive





###############Extra def##############################



def recursive_pred_moe_lf(inputs, loading, outputs, model_list, device, timesteps):
    samples = outputs.shape[0]
    br_ip= (loading).to(torch.float32).contiguous().to(device)
    n_loads = br_ip.shape[-1]
    targets = (outputs).to(torch.float32).to(device)
    preds = torch.zeros((samples,360,72,144))
   # for i in range(samples):
    prompt = (inputs[0]).to(torch.float32).contiguous().to(device)
    prompt = prompt.reshape(1,-1, 72,144)
    _, preds = addition_moe_forward_pass(prompt, br_ip.reshape(-1,n_loads), 
                                     outputs, model_list, device)
    
    preds = preds.to(device)
    return outputs, preds



def recursive_pred_lblf(inputs, loading, outputs, model, device, timesteps):
    br_ip= (loading).to(torch.float32).contiguous().to(device)
    n_loads = br_ip.shape[-1]
    tr_ip= (inputs).to(torch.float32).contiguous().to(device)
    #time_hist =tr_ip.shape[-1]-1
    time_hist =tr_ip.shape[1]
    tr_ip_recursive = (inputs).to(torch.float32).contiguous().to(device)
    targets = (outputs).to(torch.float32).to(device)
    preds_recursive = torch.zeros((50-time_hist, 128,128))
    model_calls = int((50-time_hist)/time_hist)
    for i in range(model_calls):
        if i == 0:
            temp =  tr_ip_recursive[0].reshape(1,time_hist, 128,128)
            preds, br_op, tr_op = model(br_ip[i].reshape(-1,n_loads), temp)
        else:
            temp = preds
            preds, br_op, tr_op = model(br_ip[i].reshape(-1,n_loads), temp)
        preds_recursive[i*time_hist:(i+1)*time_hist] = preds[0]
    preds_recursive = (preds_recursive).to(device)
    return targets, preds_recursive, br_op, tr_op



def recursive_pred_unet_lf(inputs, loading, outputs, model, device, timesteps, lb, lf):
    br_ip= (loading).to(torch.float32).contiguous().to(device)
    n_loads = br_ip.shape[-1]
    model_calls = int(np.ceil((50-lb-lf+1)/lf))
    #model_calls = 13
    targets = (outputs).to(torch.float32).to(device)
    preds_recursive = torch.zeros((int(lf*model_calls),128,128))
    prompt = (inputs[0]).to(torch.float32).contiguous().to(device)
    prompt = prompt.reshape(1,-1, 128,128)
    load_idx = np.arange(0,44,lf)
    #load_idx = np.arange(model_calls)
    for i in range(len(load_idx)):
        preds = model(prompt.reshape(1,-1, 128,128), 
                                    br_ip[load_idx[i]].reshape(-1,n_loads))
        #temp = preds
        prompt = preds.reshape(1,lf,128,128)
        #prompt = (inputs[i,0]).to(torch.float32).contiguous().to(device)
        #prompt = torch.cat((prompt.reshape(-1,128,128), temp.reshape(-1,128,128)), dim=0)
        preds_recursive[i*lf:(i+1)*lf] = preds
    preds_recursive = preds_recursive.to(device)
    return targets, preds_recursive

    
