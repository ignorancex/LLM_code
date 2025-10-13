#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Sep 17 09:37:31 2024

@author: ishan
"""

import matplotlib.pyplot as plt
import numpy as np


########################## Load stats ##################
## Micro
#data_dir = '/Users/ishan/JHU/Data/Loadpath_var/fiber_data_loadonly_microvar10/'

## Gray-Scott reaction diffusion
#data_dir = '/Users/ishan/JHU/Data/HighDim_datasets/datasets/gray_scott_reaction_diffusion/data_proc/'

## Planet-SWE
data_dir = '/Users/ishan/JHU/Data/HighDim_datasets/datasets/planetswe/data_proc/'


stats = np.load(data_dir+'stats_commonnorm_planetswe_96traj_50steps_udim1.npz', allow_pickle=True)
mean_ip = stats['arr_0'][0]['mean_ip']
std_ip = stats['arr_0'][0]['std_ip']

## Load stored predictions
data = np.load('/Users/ishan/JHU/Data/Outputs/planetswe_moepreds_udim1.npz', allow_pickle=True)
targets_moe = data['arr_0'][0]['gt']
preds_moe = data['arr_0'][0]['preds']

    
###################### Metric computation #################
stress_preds, stress_targets = preds_moe, targets_moe
img_dim1 = stress_preds[0].shape[1]
img_dim2 = stress_preds[0].shape[2]
eps=1e-7
samples = len(stress_preds)
pred_shape = len(stress_preds[0])
diff_org = np.zeros((samples, pred_shape, img_dim1, img_dim2))
mean_error = np.zeros((samples, pred_shape))
mean_error_rel = np.zeros((samples, pred_shape))
mean_fib = np.zeros((samples, pred_shape))
mean_mat = np.zeros((samples, pred_shape))
max_error = np.zeros((samples, pred_shape))
mean_stress_target = np.zeros((samples, pred_shape))
mean_stress_pred = np.zeros((samples, pred_shape))
diff_norm = np.zeros((samples, pred_shape))
diff_l2norm = np.zeros((samples, pred_shape))
diff_l2norm_avg = np.zeros((samples, pred_shape))
l2metric = np.zeros((samples,1))
max_val = np.zeros((samples,pred_shape))
diff_std = np.zeros((samples, pred_shape))
mean_relmax = np.zeros((samples, pred_shape))
vrmse = np.zeros((samples, pred_shape))

for samp in range(samples):
    for i in range(pred_shape):
        
        ind = i
        preds = stress_preds[samp][ind]*std_ip + mean_ip
        targets = stress_targets[samp][ind,0]*std_ip + mean_ip
        mean_stress_target[samp, i] = np.mean(targets)
        mean_stress_pred[samp, i] = np.mean(preds)
        diff_org[samp,i] = preds-targets
        diff = np.abs(preds - targets)
        diff_std[samp,i] = np.std(diff)
        mean_relmax[samp,i] = np.mean((preds-targets)**2)  
        diff_l2norm[samp, i] = np.sqrt(np.sum((preds.reshape(img_dim1,-1) - targets.reshape(img_dim1,-1))**2))/np.sqrt(np.sum(targets**2))
        diff_l2norm_avg[samp, i] = np.sqrt(np.sum((mean_stress_pred - mean_stress_target)**2))/np.sqrt(np.sum(mean_stress_target)**2)
        mean_error[samp, i] = np.mean(diff)
        mean_error_rel[samp, i] = np.mean(diff/np.max(targets))
        max_val[samp,i] = np.max(targets)
        vrmse[samp,i] = np.sqrt(np.mean(np.square(preds-targets))/(np.mean(np.square(targets-np.mean(targets)))+eps))
        max_error[samp, i] = np.max(np.abs(diff))
    l2metric[samp] = np.max(diff_l2norm)
    
mae_avgpred = np.mean(np.abs(mean_stress_pred - mean_stress_target))
mae_mm_3 = np.mean(mean_error,axis=0)
mae_ss_3 = np.std(mean_error,axis=0)

dd_moe=np.mean(diff_l2norm, axis=0)
mm0= np.mean(mean_error, axis=0)
mae_ss2 = np.std(mean_error, axis=0)





####Individual predictions viz###########
plt.rcParams['font.size'] = 18
plt.rcParams['lines.linewidth'] = 3


## Fig 2
ll=loadpaths*0.4/8
plt.plot(loadpaths[0]*0.4/8, linewidth=4, marker='o', markevery=[0,49,99], markerfacecolor = 'red', markersize=15)
#plt.plot(loadpaths[1]*0.4/8, linewidth=3, linestyle = '-.')
plt.plot(loadpaths[2]*0.4/8, linewidth=3, linestyle = ':')
plt.plot(loadpaths[3]*0.4/8, linewidth=3, linestyle = ':')
plt.plot(loadpaths[4]*0.4/8, linewidth=3, linestyle = ':')
plt.plot(loadpaths[5]*0.4/8, linewidth=3, linestyle = ':')
plt.fill_between(np.arange(100), loadpaths[0]*0.4/8 - np.std(ll,axis=0), loadpaths[0]*0.4/8 + np.std(ll,axis=0), alpha = 0.3)
plt.xlabel('Time step')
plt.ylabel('% strain')


## Fig 3
plt.imshow(tt[20,44,0,:,:], origin='lower')
plt.colorbar()

## Fig 4
plt.imshow(tt[0,12,0,:,:], origin='lower')
plt.colorbar()

## Fig 5-7
##micro viz ind 0 ; GS rxn viz ind 20 ;  Planet SWE viz ind 0 
pp = grayscott_moepreds_sconcA*std_ip + mean_ip
tt = grayscott_gt_sconcA*std_ip + mean_ip

ax1 = plt.subplot(2,1,1) ## 2 rows and n column, position 1,1 =1
#plt.subplots_adjust(wspace=0.1) 
im = ax1.imshow(pp[0,29,:,:], origin='lower')
plt.xticks([])
plt.yticks([])
ax2 = plt.subplot(2,1,2)
im = ax2.imshow(tt[0,29,0,:,:], origin='lower')
plt.xticks([])
plt.yticks([])
plt.colorbar(im, ax=[ax1, ax2]) ##Common colobar for ax1 and ax2; aspect used to set colorbar thickness/width


## Fig 8
plt.figure()
plt.plot(np.arange(3,33,1), dd_moe[:30], 'o', label = 'MoE')
plt.plot(np.arange(3,33,1), dd0[:30], label = 'Model-1')
plt.plot(np.arange(3,33,1), dd1[:30], label = 'Model-2')
plt.plot(np.arange(3,33,1), dd2[:30], label = 'Model-3')
plt.plot(np.arange(3,33,1), dd3[:30], label = 'Model-4')
plt.xlabel('Time-step')
plt.ylabel('Rel L2 error')
plt.legend(loc='upper left')

## Fig 9
plt.figure()
plt.plot(np.arange(3,33,1), mm_moe[:30], 'o', label = 'MoE')
plt.plot(np.arange(3,33,1), mm0[:30], label = 'Model-1')
plt.plot(np.arange(3,33,1), mm1[:30], label = 'Model-2')
plt.plot(np.arange(3,33,1), mm2[:30], label = 'Model-3')
plt.plot(np.arange(3,33,1), mm3[:30], label = 'Model-4')
plt.xlabel('Time-step')
plt.ylabel('Mean abs. error')
plt.legend(loc='upper left')


## Fig 10
plt.figure()
plt.plot(np.arange(5,50,1), dd2_sum, label = '2 - MoE')
plt.plot(np.arange(5,50,1), dd4_sum, label = '4 - MoE')
plt.plot(np.arange(5,50,1), dd8_sum, label = '8 - MoE')
#plt.plot(np.arange(5,50,1), mm_moe, label = 'MeoE - pre train (x4)')
plt.xlabel('Time-step')
plt.ylabel('Rel L2 error')
plt.legend(loc='upper left')

plt.figure()
plt.plot(np.arange(5,50,1), mm2_sum, label = '2 - MoE')
plt.plot(np.arange(5,50,1), mm4_sum, label = '4 - MoE')
plt.plot(np.arange(5,50,1), mm8_sum, label = '8 - MoE')
#plt.plot(np.arange(5,50,1), mm_moe, label = 'MeoE - pre train (x4)')
plt.xlabel('Time-step')
plt.ylabel('Mean abs. error')
plt.legend(loc='upper left')








