#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Apr 26 09:08:39 2024

@author: ishan
"""

## Import modules
import time
import numpy as np
import torch
from utils import (imgdata_loader, recursive_pred_moe)


##Set device
torch.cuda.empty_cache()    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



##Set file-paths for all datasets + stats + trained models
dataset = 'micro'

if dataset == 'planetswe':
    data_dir = '/Users/ishan/JHU/Data/HighDim_datasets/datasets/planetswe/data_proc/'
    path_ip_test = data_dir + 'inputs_test_planetswe_96traj_50steps_udim1_timelb3.npz'
    path_op_test = data_dir + 'outputs_test_planetswe_96traj_50steps_udim1_timelb3.npz'
    path_load_test = data_dir + 'params_test_planetswe_96traj_50steps_udim1_timelb3.npz'
    
    stats = np.load(data_dir+'stats_commonnorm_planetswe_96traj_50steps_udim1.npz', allow_pickle=True)

    model_trained_path  = "/Users/ishan/JHU/Data/Results/planetswe_96traj50stepsudim1_3lb1lf+timeid_unet_timemlpmembed_attn_commonnorm_onlymodel"
    
elif dataset == 'grayscott':
    data_dir = '/Users/ishan/JHU/Data/HighDim_datasets/datasets/gray_scott_reaction_diffusion/data_proc/'
    path_ip_test = data_dir + 'inputs_test_grayscott_960traj_1to50stepsinc1_speciesA_timelb5.npz'
    path_op_test = data_dir + 'outputs_test_grayscott_960traj_1to50stepsinc1_speciesA_timelb5.npz'
    path_load_test = data_dir + 'params_test_grayscott_960traj_1to50stepsinc1_speciesA_timelb5.npz'
    
    stats = np.load(data_dir+'stats_commonnorm_grayscott_960traj_50steps_speciesA.npz', allow_pickle=True)
    
    model_trained_path = "/Users/ishan/JHU/Data/Results/grayscott_960traj1to50stepsinc1_speciesA5lb1lf+timeid_unet_timemlpmembed_attn_commonnorm_80iter_onlymodel"

else:
    data_dir = '/Users/ishan/JHU/Data/Loadpath_var/fiber_data_loadonly_microvar10/'
    path_ip_test = data_dir + 'inputs_test_20fiber_elastoplastic_time3lb1lfloadval4+timeid_loadonlyvar_sig11_1to4micro11to14load.npz'
    path_op_test = data_dir + 'outputs_test_20fiber_elastoplastic_time3lb1lfloadval4+timeid_loadonlyvar_sig11_1to4micro11to14load.npz'
    path_load_test = data_dir + 'loading_test_20fiber_elastoplastic_time3lb1lfloadval4+timeid_loadonlyvar_sig11_1to4micro11to14load.npz'
    
    stats = np.load(data_dir+'stats_commonnorm_20fiber_elastoplastic_3lb1lfloadval4+timeid_loadonlyvar_sig11_70micro1to10load.npz', allow_pickle=True)
    
    model_trained_path = "/Users/ishan/JHU/Data/Outputs/20fiber_loadonlypath_micro70loadpath10var_3lb1lf+timeid_sig11_unet_timemlpmembed_attn_commonnorm_onlymodel"

save_path = '/Users/ishan/JHU/Data/Outputs/'  
expt_path = '20fiber_loadpathvar_unet'
full_path = save_path + expt_path

##Define params
timesteps = 30
bs_test = 12
model = 'unet'
grid_dim = 128
learners = 4
model_list = []
pretrained = True
batch_size_viz = timesteps
data_samp = 1

##Load stats 
mean_ip = stats['arr_0'][0]['mean_ip']
std_ip = stats['arr_0'][0]['std_ip']

## Load pre-trained models
#available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
for i in range(learners):
    model = torch.load(model_trained_path +str(i), map_location = device)
    #model = torch.nn.DataParallel(model, device_ids=[available_gpus])
    model_list.append(model)
   
## Instantiate test data loader
stress_preds, stress_targets, stress_br, stress_tr, stress_loads = [], [], [], [], []    
ds_viz = imgdata_loader(path_ip_test, path_op_test, path_load_test, batch_size_viz, 
                        1, stats, shuffle = False)

## Obtain test predictions
for model in model_list:
    model.eval()
start_time = time.time()    
with torch.no_grad():
    for j, data_viz in enumerate(ds_viz):
            print('Processing test sample: ', j)
            targets_viz, preds_viz = recursive_pred_moe(data_viz[0], data_viz[1], 
                                                        data_viz[2], model_list, device, timesteps)
            stress_preds.append(preds_viz.cpu().detach().numpy())
            stress_targets.append(targets_viz.cpu().numpy())

end_time = time.time()
print('Time elapsed:', end_time-start_time)
        
## Save predictions + ground truth
expt_data = []
outputs = {}
outputs['gt'] = stress_targets
outputs['preds'] = stress_preds
expt_data.append(outputs)
np.savez(save_path + '20fiber_loadonlypathvar_3lb1lf_conv2dpenultimatemoe_recursive', expt_data)
   
    
    
    
