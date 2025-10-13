#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Sep 12 13:31:53 2024

@author: ishan
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


#### Data parameter initialization   ############
data_dir = '/Users/ishan/JHU/Data/Loadpath_var/fiber_data_loadonly_microvar10/'
case = 'new_sim'
n_samples = 100
n_load_steps = 100
bndx = [0,8]
bndy = [0,8]
x_grid, y_grid = 128,128
xx, yy = np.mgrid[0:bndx[1]+0.0001:bndx[1]/x_grid-1, 0:bndy[1]+0.0001:bndy[1]/y_grid-1]
grid =np.vstack((xx.flatten(),yy.flatten())).T

train_size = 80
train_ind = np.random.choice(np.arange(n_samples),train_size, replace=False)
indx_rem = ~np.isin(np.arange(n_samples), train_ind)
test_ind = [i for i, x in enumerate(indx_rem) if x]

######### Helper functions  #################

## Assignning phase for mesh integration points
def assign_microphase(Lines):
    bin_start = Lines.index('*Elset, elset=fiber-element-set\n')
    bin_end = Lines.index('** Section: fiber_sec\n')
    fiber_set = Lines[bin_start+1:bin_end]
    node_fiber = []
    for j in range(len(fiber_set)):
        fiber_set[j] = fiber_set[j][:-1]
        if j == len(fiber_set) -1:
            if fiber_set[j][-1] == ',':
                fiber_set[j] = fiber_set[j][:-1]
        temp = [int(x.strip())-1 for x in fiber_set[j].split(',')]
        node_fiber.extend(temp)
    return node_fiber


## Create Microstructure input images
def create_micro():
    
    micro = np.zeros((n_samples,x_grid, y_grid))
    
    for i in range(n_samples):
        print('Computing microstructure: ', i)
        file_path = data_dir + 'data_Nf20_Vf40_case_1_loadpath_' + str(i+1) + '.npz'
        temp = np.load(file_path, allow_pickle = True)['array1']
        coords = np.zeros((temp.shape[1]-2,4))
        for j in range(temp.shape[1]-2):
            coords[j] = temp[0][j][:4]
            
        file_mesh = (data_dir + '/20Fiber_Vf40_SF128_Model_case1'
             + '_path' + str(i+114) + '.inp')
        f = open(file_mesh, "r")
        Lines = f.readlines()
        f.close()
        node_fib = assign_microphase(Lines)
        node_phase = np.zeros((temp.shape[1]-2,1))
        node_phase[[x-1 for x in node_fib]] = 1
        inds=np.argsort(coords[:,0])
        micro[i] = griddata(coords[inds][:,[3,2]], node_phase, (xx,yy), 'nearest').reshape(128,128)
    return micro


##generates dataset with 4 time-step history of stress maps stacked as input and next time step stress as output
def create_dataset_multitimestep(inds, stress_maps, load_paths):
    samples = len(inds)
    time_hist = 10
    prediction_steps = load_paths.shape[1] - time_hist
    y = stress_maps[inds,time_hist:,:,:].reshape(-1,x_grid, y_grid)
    x = np.zeros((len(inds)*prediction_steps, x_grid, y_grid, time_hist))
    for i in range(len(inds)):
        for j in range(prediction_steps):
            x[i*prediction_steps+j,:,:,:] = np.moveaxis(stress_maps[inds[i],j:j+time_hist,:,:],0,-1)
    
    loading_full = []
    for i in range(samples):
        for j in range(time_hist, load_paths.shape[1]):
            loading_full.append([j, load_paths[inds[i],j]])
    loading_full = np.asarray(loading_full)
    
    return x,loading_full,y


##generates dataset with 4 time-step history of stress maps stacked as input and next time step stress as output
def create_dataset_multitimestep_lbvar(inds, micro, stress_maps, load_paths):
   # samples = len(inds)
    time_lb = 3
    time_lf = 1
    prediction_steps = load_paths.shape[1] - time_lf - time_lb + 1
    y = np.zeros((len(inds)*(prediction_steps), time_lf, x_grid, y_grid))
    x = np.zeros((len(inds)*(prediction_steps), time_lb, x_grid, y_grid))
    loading_full = np.zeros((len(inds)*(prediction_steps),2*( time_lb + time_lf)))
    for i in range(len(inds)):
        for j in range(prediction_steps):
            x[i*prediction_steps+(j)] = stress_maps[inds[i],j:j+time_lb,:,:]
            y[i*prediction_steps+(j),:,:,:] = stress_maps[inds[i],j+time_lb:j+time_lb+time_lf,:,:] 
            temp = load_paths[inds[i],j:(j+time_lb+time_lf)]
            loading_full[i*prediction_steps+(j),:(time_lb + time_lf)] = temp
            loading_full[i*prediction_steps+(j),time_lb + time_lf:] = np.arange(j,j+time_lb + time_lf)
   
    return x,loading_full,y



##Generates dataset with per load-step standardization
def create_tempnorm_data(stress_maps, micro, load_paths, inds, stats, train_flag):
    load_steps = load_paths.shape[1]
    if train_flag:
        data_train = stress_maps[inds,:,:,:]
        stress_rearr = np.moveaxis(data_train,1,0)
        mean_time = np.mean(np.mean(stress_rearr.reshape(load_steps,len(inds),-1), axis=-1),axis=-1)
        std_time = np.sqrt(np.sum(np.var(stress_rearr.reshape(load_steps,len(inds), -1), axis=-1), axis=-1)/load_steps)
        stress_norm = (stress_maps - mean_time.reshape(1,-1,1,1))/std_time.reshape(1,-1,1,1)
        x, loading, y = create_dataset_multitimestep_lbvar(inds, [], stress_norm, load_paths)
        summary = []
        stats['mean_ip'] = mean_time
        stats['std_ip'] = std_time
        summary.append(stats)
        ##Save data
        np.savez('/Users/ishan/JHU/Data/Loadpath_var/fiber_data_loadonly_microvar10/stats_temporalnorm_20fiber_elastoplastic_3lb1lfloadval4+timeid_loadonlyvar_sig11.npz', summary)
        
    else:
        mean_time = stats['mean_ip']
        std_time = stats['std_ip']
        stress_norm = (stress_maps - mean_time.reshape(1,-1,1,1))/std_time.reshape(1,-1,1,1)
        x, loading, y = create_dataset_multitimestep_lbvar(inds, [], stress_norm, load_paths)

    return x, loading, y, stats


##Reads data and returns sigma_VonMises compoenent on 2d grid. Dims: [N-samples x N-loadsteps x 128 x 128]
def create_strain_maps():
    ##Create all stress components dataset
    strain = np.zeros((n_samples, n_load_steps, x_grid, y_grid))
   # t1 = time.time()
    for i in range(n_samples):
        print('Simulating sample: ', i)
        file_path = data_dir + 'data_Nf20_Vf40_case_1_loadpath_' + str(i+19) + '.npz'
        temp = np.load(file_path, allow_pickle = True)['array1']
        data= np.zeros((n_load_steps,temp.shape[1]-2))
        coords = np.zeros((temp.shape[1]-2,2))
        for k in range(n_load_steps):
            for j in range(temp.shape[1]-2):
                data[k,j] = temp[k][j][-1]
                coords[j] =temp[k][j][2:4]
    
        for j in range(n_load_steps):
            sigma_all = data[j]
            strain[i,j] = griddata(coords[:,[1,0]], sigma_all, (xx,yy), 'nearest')
        del temp, data
   # t2 = time.time()
    #print('Time elapsed: ', t2-t1)
    return strain


##Reads data and returns sigma_VonMises compoenent on 2d grid. Dims: [N-samples x N-loadsteps x 128 x 128]
def create_stress_maps(data_dir, inds):
    
    ###File check
    #skipped_indices = []
    #for i in inds:
    #    file_path = data_dir + 'data_Nf20_Vf40_case_' + str(i+1) + '_loadpath_' + str(valid_inds[i]+1) + '.npz'
    #    my_file = Path(file_path)
    #    if my_file.is_file():
    #        pass
    #    else:
    #        skipped_indices.append(i)
    #valid_inds = [i for i in range(112) if i not  in skipped_indices]
    
    ##Create all stress components dataset
    n_samples = len(inds)
    sigma_allcomp_grid = np.zeros((n_samples, n_load_steps,x_grid, y_grid,5))
    lpaths, micros = 10, 10
    cnt = 0
    for i in range(micros):
        for j in range(lpaths):
            print('Simulating sample: ', cnt)
            cnt+=1
            file_path = data_dir + 'data_Nf20_Vf40_case_' + str(i+1) + '_loadpath_' + str(j+1) + '.npz'
            temp = np.load(file_path, allow_pickle = True)['array1']
                
            data= np.zeros((n_load_steps,temp.shape[1]-2,5))
                #data= np.zeros((n_load_steps,temp.shape[1]-2,1))
            coords = np.zeros((temp.shape[1]-2,2))
            for k in range(n_load_steps):
                for m in range(temp.shape[1]-2):
                    #data[k,j] = temp[k][j][-4]
                    data[k,m] = temp[k][m][-5:]
                    coords[m] =temp[k][m][2:4]
       
            for l in range(n_load_steps):
                sigma_all = data[l]
                sigma_allcomp_grid[i*lpaths+j,l] = griddata(coords[:,[1,0]], sigma_all, (xx,yy), 'nearest')
            del temp, data
    stress_maps = sigma_allcomp_grid
    np.savez(data_dir + 'stress_loadpathonly_1to10micro_1to10loadsamples.npz', stress_maps)
    return stress_maps


#####  Generate dataset with variable load only paths #####
# Import load paths
load_paths_simpar = np.load('/Users/ishan/JHU/Data/Loadpath_var/Loadpath_specs/loadonly_simpar140220_pchipinterp5.npy')
load_paths = 100*0.004*load_paths_simpar/8
loadpath_samprate = 3
lp = load_paths[10:14]
ll=np.tile(lp.T,4)
load_paths=ll.T
load_paths = load_paths[:,1::loadpath_samprate]


if case == 'new_sim':
    stress_maps = create_stress_maps(data_dir, np.arange(100))
    np.savez(data_dir + 'stress_loadpathonly_1to4micro_11to14loadsamples.npz', stress_maps)
else:
    stress_maps = np.load(data_dir + 'stress_loadpathonly_1to4micro_11to14loadsamples.npz')['arr_0']

stress_11 = stress_maps[:,1::loadpath_samprate,:,:,1]
x=stress_11[:,:,:,:].reshape(-1,x_grid*y_grid)


## Compute and store stats
mean_ip = np.mean(x)
std_ip = np.sqrt(np.sum(np.var(x, axis=1), axis=0)/x.shape[0])
stats = {}
summary = []
stats['mean_ip'] = mean_ip
stats['std_ip'] = std_ip
summary.append(stats)
np.savez('/Users/ishan/JHU/Data/Loadpath_var/fiber_data_loadonly_microvar10/stats_commonnorm_20fiber_elastoplastic_3lb1lfloadval4+timeid_loadonlyvar_sig11_70micro1to10load.npz', summary)

## Generate dataset
x_train, loading_train, y_train = create_dataset_multitimestep_lbvar(train_ind, [], stress_11, load_paths)
x_test, loading_test, y_test = create_dataset_multitimestep_lbvar(test_ind, [], stress_11, load_paths)


## Store dataset ####
np.savez('/Users/ishan/JHU/Data/Loadpath_var/fiber_data_loadonly_microvar10/inputs_train_20fiber_elastoplastic_time3lb1lfloadval4+timeid_loadonlyvar_sig11_70micro1to10load_aug2x.npz', x_train)
np.savez('/Users/ishan/JHU/Data/Loadpath_var/fiber_data_loadonly_microvar10/outputs_train_20fiber_elastoplastic_time3lb1lfloadval4+timeid_loadonlyvar_sig11_70micro1to10load_aug2x.npz', y_train)
np.savez('/Users/ishan/JHU/Data/Loadpath_var/fiber_data_loadonly_microvar10/loading_train_20fiber_elastoplastic_time3lb1lfloadval4+timeid_loadonlyvar_sig11_70micro1to10load_aug2x.npz', loading_train)

np.savez('/Users/ishan/JHU/Data/Loadpath_var/fiber_data_loadonly_microvar10/inputs_test_20fiber_elastoplastic_time3lb1lfloadval4+timeid_loadonlyvar_sig11_37to40micro11to14load.npz', x_test)
np.savez('/Users/ishan/JHU/Data/Loadpath_var/fiber_data_loadonly_microvar10/outputs_test_20fiber_elastoplastic_time3lb1lfloadval4+timeid_loadonlyvar_sig11_37to40micro11to14load.npz', y_test)
np.savez('/Users/ishan/JHU/Data/Loadpath_var/fiber_data_loadonly_microvar10/loading_test_20fiber_elastoplastic_time3lb1lfloadval4+timeid_loadonlyvar_sig11_37to40micro11to14load.npz', loading_test)






