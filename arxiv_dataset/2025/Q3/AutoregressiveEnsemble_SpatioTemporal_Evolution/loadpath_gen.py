#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 14:14:58 2024

@author: ishan
"""

##Import modules 
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, PchipInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor



########## Load path only simulation ############

n_steps_discrete = 5
X= np.linspace(0,1,n_steps_discrete)
n_steps_approx = 100 ##Load steps to be computed for stress simulation
n_comp = 1 ##e11, e22, e12
n_paths = 200
strain_floor = 0
strain_max = 140 + 80*np.random.rand(n_paths)   ## Variable max strain value

load_path_discrete = np.zeros((n_paths, n_steps_discrete))
interp_loc_discrete = np.zeros((n_paths, n_steps_discrete))
load_path_discrete[:,-1] = strain_max
interp_loc_discrete[:,-1] = 1

## Simulate discrete load paths
for i in range(n_paths):
    strain_floor = 0
    strain_ceil = strain_max[i]
    interp_floor = 0
    interp_ceil = 1
    for j in range(1,n_steps_discrete-1):
        strain_curr = strain_floor + (strain_ceil - strain_floor)*np.random.rand()
        strain_floor = strain_curr
        load_path_discrete[i,j] = strain_curr
        
        interp_curr = interp_floor + (interp_ceil - interp_floor)*np.random.rand()
        interp_floor = interp_curr
        interp_loc_discrete[i,j] = interp_curr
        
## Interpolate with discrete values to obtain full load path
#load_path_linear = np.zeros((n_paths, n_steps_approx))
load_path_ps = np.zeros((n_paths, n_steps_approx))
for i in range(n_paths):
    ps = PchipInterpolator(X, load_path_discrete[i,:])
    load_path_ps[i] = ps(np.linspace(0,1,n_steps_approx))
   
## Convert load paths to % strain values
ll_ps = 100*0.004*load_path_ps[:,:]/8 
#np.save('/Users/ishan/JHU/Data/Loadpath_var/Loadpath_specs/loadonly_simpar140220_pchipinterp5.npy', ll_ps)

mm = np.mean(ll_ps,axis=0)
ss=np.std(ll_ps,axis=0)
min_val = np.min(ll_ps,axis=0)
max_val = np.max(ll_ps, axis=0)

## Load path(s) Visualziation
plt.plot(np.arange(100), mm, label = 'Average value at step t')
plt.fill_between(np.arange(100), min_val, max_val, alpha =0.2, label = 'Range of values')
plt.xlabel('Time-step')
plt.ylabel('% strain')
plt.legend()

plt.figure()
plt.errorbar(np.arange(100), mm, ss)

plt.figure()
for i in range(130,135):
    plt.plot(ll_ps[i])




