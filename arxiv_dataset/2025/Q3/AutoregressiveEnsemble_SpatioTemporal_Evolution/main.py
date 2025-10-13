#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Apr 26 09:08:39 2024

@author: ishan
"""

## Import modules
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
#from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
#from torch.utils.data import TensorDataset, DataLoader
from utils import (imgdata_loader, recursive_pred_unet, conv_forward_pass_unet,
                   addition_moe_forward_pass)
from unet_2d import Unet2D


## Set device
torch.cuda.empty_cache()    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

## Set file-paths for all datasets

data_dir = '/Users/ishan/JHU/Data/Loadpath_var/fiber_data_loadonly_microvar10/'
path_ip_train = data_dir + 'inputs_train_20fiber_elastoplastic_time3lb1lfloadval4+timeid_loadonlyvar_sig11_1to4micro11to14load.npz'
path_op_train = data_dir + 'outputs_train_20fiber_elastoplastic_time3lb1lfloadval4+timeid_loadonlyvar_sig11_1to4micro11to14load.npz'
path_load_train = data_dir + 'loading_train_20fiber_elastoplastic_time3lb1lfloadval4+timeid_loadonlyvar_sig11_1to4micro11to14load.npz'

path_ip_test = data_dir + 'inputs_test_20fiber_elastoplastic_time3lb1lfloadval4+timeid_loadonlyvar_sig11_1to4micro11to14load.npz'
path_op_test = data_dir + 'outputs_test_20fiber_elastoplastic_time3lb1lfloadval4+timeid_loadonlyvar_sig11_1to4micro11to14load.npz'
path_load_test = data_dir + 'loading_test_20fiber_elastoplastic_time3lb1lfloadval4+timeid_loadonlyvar_sig11_1to4micro11to14load.npz'
stats = np.load(data_dir+'stats_commonnorm_20fiber_elastoplastic_3lb1lfloadval4+timeid_loadonlyvar_sig11_70micro1to10load.npz', allow_pickle=True)

## If loading pretrained model    
model_trained_path = "/Users/ishan/JHU/Data/Outputs/20fiber_loadonlypath_micro70loadpath10var_3lb1lf+timeid_sig11_unet_timemlpmembed_attn_commonnorm_onlymodel"


save_path = '/Users/ishan/JHU/Data/Outputs/'  
expt_path = '20fiber_loadpathvar_unet'
full_path = save_path + expt_path

## Define params
timesteps = 30
bs_train = 10
bs_test = 12
epochs = 100
lr_init = 0.001
lr_decay = 0.1
model = 'unet'
n_basis = 200
grid_dim = 128
learners = 4
model_list = []
pretrained = True
batch_size_viz = timesteps
data_samp = 1



######### Create train and test data loaders   ##########
## Load stats 
mean_ip = stats['arr_0'][0]['mean_ip']
std_ip = stats['arr_0'][0]['std_ip']

ds_train = imgdata_loader(path_ip_train, path_op_train, path_load_train,
                          bs_train, 1, stats, shuffle = False)
ds_test = imgdata_loader(path_ip_test, path_op_test, path_load_test, bs_test, 
                        1, stats, shuffle = False)


#########  Model + training definitions   ########

#available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
if pretrained ==False:
    ## Training
    deep_net = Unet2D(dim=16, out_dim = 1, dim_mults=(1, 2, 4, 8), channels = 5, ip_time_dim = 8, attn = True)
    #deep_net = torch.nn.DataParallel(deep_net, device_ids=[available_gpus])
else:
    ##Testing pre trained modesls
    deep_net = torch.load(model_trained_path, map_location = device)
    #deep_net = torch.nn.DataParallel(deep_net, device_ids=[available_gpus])
  
    

params = []
params = list(deep_net.parameters())

## Define optimizer and lr scheduler (optional)
optimizer = torch.optim.Adam(params, lr=lr_init)
#scheduler = CosineAnnealingLR(optimizer, epochs)


start_time = time.time()
train_loss, test_loss = [], []

## All models set to training mode
for model in model_list:
    model.to(device)
    model.train()


##########    Training + testing loop  #########
for i in range(epochs):
        print('\nEpoch:',i)
        t_curr = time.time()
        loss_test_copy, loss_train_copy = 0, 0
        deep_net.train(True)
        for j, data in enumerate(ds_train): 
            print('Data iteration: ', j)
            targets, preds = conv_forward_pass_unet(data[0], data[1], 
                                           data[2], deep_net, device)
            ## if training multiple models in a single run, modify here and in utils
           # targets, preds = addition_moe_forward_pass(data[0], data[1], data[2],
           #                                       model_list, device)
           
            targets = torch.squeeze(targets, dim=1)
            preds = torch.squeeze(preds, dim=1)
            loss = F.mse_loss(preds, targets)
            loss = loss.to(device)
           
            loss_train_copy += loss.item()
            
            # Compute the gradient of loss and updated weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            ##Epoch time
            if True:
                print("\nImage processed: ", (j+1)*bs_train)
                t_curr1 = time.time()
                print('Time elapsed:', t_curr1 - t_curr)
                
        train_loss.append(loss_train_copy/(j+1))
        
        
        ##Test mode
        deep_net.eval()

        with torch.no_grad():
            for k, data_test in enumerate(ds_test):
                targets_test, preds_test = conv_forward_pass_unet(data_test[0], 
                                     data_test[1], data_test[2], deep_net, device)
                ## if training multiple models in a single run, modify here and in utils
                #targets_test, preds_test = addition_moe_forward_pass(data_test[0], data_test[1], 
                #                    data_test[2],model_list, device)
                
                targets_test = torch.squeeze(targets_test, dim=1)
                preds_test = torch.squeeze(preds_test, dim=1)
                loss_test =  F.mse_loss(preds_test, targets_test)
                loss_test = loss_test.to(device)
                loss_test_copy += loss_test.item()  
  
        test_loss.append(loss_test_copy/(k+1))
        print(f"\nTraining step: {i+1}/{epochs}. Train loss: {train_loss[-1]}. Test loss: {test_loss[-1]}\n") 
       # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
       #scheduler.step(loss_test)

## Save models
torch.save(deep_net, save_path + expt_path + 'model0')       


###### Metric viz ##########
plt.figure()
plt.semilogy(np.asarray(train_loss), label = 'Train')
plt.semilogy(np.asarray(test_loss), label = 'Test')
plt.xlabel('# Training epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.savefig(save_path + expt_path)  



##########  Autoregressive test predictions   ##########
stress_preds, stress_targets, stress_br, stress_tr, stress_loads = [], [], [], [], []    
ds_viz = imgdata_loader(path_ip_test, path_op_test, path_load_test, batch_size_viz, 
                        1, stats, shuffle = False)


deep_net.eval()
start_time = time.time()    
with torch.no_grad():
    for j, data_viz in enumerate(ds_viz):
            print('Processing test sample: ', j)
            targets_viz, preds_viz = recursive_pred_unet(data_viz[0], data_viz[1], 
                                        data_viz[2], deep_net, device, timesteps)
            stress_preds.append(preds_viz.cpu().detach().numpy())
            stress_targets.append(targets_viz.cpu().numpy())

end_time = time.time()
print('Time elapsed:', end_time-start_time)
        
########  Save predictions + ground truth     #################
expt_data = []
outputs = {}
outputs['gt'] = stress_targets
outputs['preds'] = stress_preds
expt_data.append(outputs)
np.savez(save_path + '20fiber_loadonlypathvar_3lb1lf_conv2dpenultimatemoe_recursive', expt_data)








