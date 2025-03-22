#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 00:41:52 2020

@author: kflouris
"""
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)

def plot_momentum(gfe, gfe_momentum):
    data_m = pd.read_csv (r'results/csvs_momentum/' + gfe_momentum +'.csv') 
    data = pd.read_csv (r'results/csvs_momentum/' + gfe +'.csv') 

    df = pd.DataFrame(data_m, columns= ['Step','Value'])
    m = df['Value'].to_numpy()/20
    step_m = df['Step'].to_numpy()

    df2 = pd.DataFrame(data, columns= ['Step','Value'])
    gfe = df2['Value'].to_numpy()/20
    step = df2['Step'].to_numpy()
    
    
    plt.figure()
    plt.plot(step,gfe, 'mediumpurple', label='1st order ODE')
    plt.plot(step_m,m,  color='black', linestyle='--', label='2nd order ODE')
    plt.xlabel('iterations')
    plt.ylabel('cross entropy loss')
    plt.xlim([0,1000])
    plt.ylim([0.075,0.25])
    plt.legend()
    plt.show()
    plt.close()

def plot_approx_adj():
    data_app = pd.read_csv (r'results/csvs_approx_adj/mnist_approx_adjoint.csv') 
    data = pd.read_csv (r'results/csvs_approx_adj/mnist_full_adjoint.csv') 

    df = pd.DataFrame(data_app, columns= ['Step','Value'])
    app = df['Value'].to_numpy()/20
    step_app = df['Step'].to_numpy()

    df2 = pd.DataFrame(data, columns= ['Step','Value'])
    adj = df2['Value'].to_numpy()/20
    step_adj = df2['Step'].to_numpy()

    
    plt.figure()
    plt.plot(step_app,app, 'mediumpurple', label='Approximate')
    plt.plot(step_adj,adj,  color='black', linestyle='--', label='Full adjoint')
    plt.xlabel('iterations')
    plt.ylabel('cross entropy loss')
    plt.xlim([0,2000])
    plt.ylim([0.05,0.6])
    plt.legend()
    plt.show()
    plt.close()
    

def plot_ae_amd_val_comp():
    data_gfe = pd.read_csv (r'results/csvs_AE_AMD/mnist_AMD_val.csv') 
    data = pd.read_csv (r'results/csvs_AE_AMD/mnist_AE_val.csv') 

    df = pd.DataFrame(data_gfe)
    gfe = df['Value'].to_numpy()
    step_gfe = df['Step'].to_numpy()
    wall_gfe = df['Wall time'].to_numpy()
    wall_gfe = wall_gfe-wall_gfe[0]

    df2 = pd.DataFrame(data)
    AE = df2['Value'].to_numpy()
    step_AE = df2['Step'].to_numpy()
    wall_AE = df2['Wall time'].to_numpy()
    wall_AE = wall_AE-wall_AE[0]
    
    print(step_AE.shape, AE.shape)
    # plt.figure()
    # plt.plot(wall_gfe,gfe, 'mediumpurple', label='GFE')
    # plt.plot(wall_AE,AE,  color='black', linestyle='--', label='AE')
    # plt.xlabel('time')
    # plt.ylabel('cross entropy loss')
    # plt.xlim([0,2000])
    # plt.ylim([0.05,0.6])
    # plt.legend()
    # plt.show()
    # plt.close()
    
    
    fig, (ax0, ax1, ) = plt.subplots(nrows=1, ncols=2, sharex=False,
                                        figsize=(18, 6))
    
    ax0.plot(step_gfe*16,gfe,  color='black', label='GFE-amd') # linestyle='--    
    ax0.plot(step_AE*16,AE, 'mediumpurple', label='AE')
    ax0.legend()
    ax0.set_xlim([0,6500*16])
    ax0.set_ylabel('cross entropy loss')
    # ax0.set_ylim([0.05,0.6])    
    ax0.set_xlabel('training images')
    
    ax1.plot(wall_gfe,gfe, 'x' ,color='black', label='GFE-amd') #linestyle='--'
    ax1.plot(wall_AE,AE, 'o',color='mediumpurple' , label='AE')
    ax1.legend()
    ax1.set_xlim([0,1000])
    # ax1.set_ylim([0.03,0.12])
    ax1.set_ylabel('cross entropy loss')
    ax1.set_xlabel('time')

    # fig.suptitle('Errorbar subsampling')
    # plt.show()
    plt.savefig('results/figures/mnist_AE_AMD_val.pdf')

def plot_all_comp():
    data_gfe = pd.read_csv (r'results/csvs_GFE_AMD_momentum/mnist_GFE.csv') 
    data_amd = pd.read_csv (r'results/csvs_GFE_AMD_momentum/mnist_AMD.csv')
    data_momentum = pd.read_csv (r'results/csvs_GFE_AMD_momentum/mnist_GFE_momentum.csv') 
    data_approx = pd.read_csv (r'results/csvs_GFE_AMD_momentum/mnist_GFE_approx_adj.csv') 
    data_adj = pd.read_csv (r'results/csvs_GFE_AMD_momentum/mnist_GFE_adj.csv') 

    df = pd.DataFrame(data_gfe)
    gfe = df['Value'].to_numpy()
    step_gfe = df['Step'].to_numpy()
    wall_gfe = df['Wall time'].to_numpy()
    wall_gfe = wall_gfe-wall_gfe[0]
    
    df = pd.DataFrame(data_amd)
    amd = df['Value'].to_numpy()
    step_amd = df['Step'].to_numpy()
    wall_amd = df['Wall time'].to_numpy()
    wall_amd = wall_amd-wall_amd[0]
    
    df = pd.DataFrame(data_momentum)
    mom = df['Value'].to_numpy()
    step_mom = df['Step'].to_numpy()
    wall_mom = df['Wall time'].to_numpy()
    wall_mom = wall_mom-wall_mom[0]
    
    df = pd.DataFrame(data_approx)
    approx = df['Value'].to_numpy()
    step_approx = df['Step'].to_numpy()
    wall_approx = df['Wall time'].to_numpy()
    wall_approx = wall_approx-wall_approx[0]

    df = pd.DataFrame(data_adj)
    adj = df['Value'].to_numpy()
    step_adj = df['Step'].to_numpy()
    wall_adj = df['Wall time'].to_numpy()
    wall_adj = wall_adj-wall_adj[0]
        
    fig, (ax0, ax1, ) = plt.subplots(nrows=1, ncols=2, sharex=False,
                                        figsize=(18, 6))
    
    ax0.plot(step_approx,approx, 'black', label='GFE-approximate')
    ax0.plot(step_adj,adj,  color='mediumpurple', label='GFE-full adjoint') # linestyle='--
    ax0.legend()
    ax0.set_xlim([0,1200])
    ax0.set_ylabel('cross entropy loss')
    # ax0.set_ylim([0.05,0.6])    
    ax0.set_xlabel('iterations')

    ax1.plot(step_amd,amd, 'black', label='GFE-amd')
    ax1.plot(step_gfe, savgol_filter(gfe, 21,2),  color='mediumpurple', label='GFE') #linestyle='--'
    ax1.plot(step_mom,savgol_filter(mom, 21,2),  color='blue', label='GFE-2nd order') #linestyle='--'
    ax1.legend()
    ax1.set_xlim([0,2700])
    # ax1.set_ylim([0.03,0.12])
    ax1.set_ylabel('cross entropy loss')
    ax1.set_xlabel('iterations')

    # fig.suptitle('Errorbar subsampling')
    # plt.show()
    plt.savefig('results/figures/mnist_all_methods.pdf')
    
def table_ae_amd_val_comp():
    data_gfe = pd.read_csv (r'results/csvs_AE_AMD/mnist_AMD_val.csv') 
    data = pd.read_csv (r'results/csvs_AE_AMD/mnist_AE_val.csv') 

    df = pd.DataFrame(data_gfe)
    gfe = df['Value'].to_numpy()
    step_gfe = df['Step'].to_numpy()
    wall_gfe = df['Wall time'].to_numpy()
    wall_gfe = wall_gfe-wall_gfe[0]

    df2 = pd.DataFrame(data)
    AE = df2['Value'].to_numpy()
    step_AE = df2['Step'].to_numpy()
    wall_AE = df2['Wall time'].to_numpy()
    wall_AE = wall_AE-wall_AE[0]
    for i, step in enumerate(step_gfe):
        if(step <1024):
            print(step, step_AE[i])
    for i, step in enumerate(step_gfe):
        if(step == 103 or step == 168 or step == 230 or step == 309 or step == 440 ):
            print('AMD=', step/6250, gfe[i])    
    for i, step in enumerate(step_AE):
        if(step == 103 or step == 168 or step == 230 or step == 309 or step == 440 ):
            print('AE=', step/6250, AE[i] )

def plot_GFE_AMD_conv():
    import glob 
    
    # data_gfe = pd.read_csv (r'results/csvs_GFE_AMD_momentum/mnist_GFE.csv') 
    # data_amd = pd.read_csv (r'results/csvs_GFE_AMD_momentum/mnist_AMD.csv')
    # data_momentum = pd.read_csv (r'results/csvs_GFE_AMD_momentum/mnist_GFE_momentum.csv') 
    # data_approx = pd.read_csv (r'results/csvs_GFE_AMD_momentum/mnist_GFE_approx_adj.csv') 
    # data_adj = pd.read_csv (r'results/csvs_GFE_AMD_momentum/mnist_GFE_adj.csv') 


    path = r'results/convergence_csvs_AMD_GFE/GFE' # use your path
    all_files = glob.glob(path + "/*")
    gfe_dfs = []    
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None)
        gfe_dfs.append(df.to_numpy())
        # steps.append(df.to_numpy())
 
    path = r'results/convergence_csvs_AMD_GFE/momentum' # use your path
    all_files = glob.glob(path + "/*")
    mom_dfs = []    
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None)
        mom_dfs.append(df.to_numpy())
        # steps.append(df.to_numpy())
        
    fig, (ax0 ) = plt.subplots(nrows=1, ncols=1, sharex=False,
                                        figsize=(9, 6))
    
    ax0.plot(gfe_dfs[0], 'black', label='GFE')
    ax0.plot(gfe_dfs[1], 'grey') # linestyle='--
    ax0.plot(gfe_dfs[3],  'lightgrey') # linestyle='--  
    ax0.plot(mom_dfs[0], 'black',  label='GFE 2nd order',linestyle='--')
    ax0.plot(mom_dfs[1],  'grey',  linestyle='--') # linestyle='--
    ax0.plot(mom_dfs[3], 'lightgrey',  linestyle='--') # 
    ax0.legend()
    ax0.set_xlim([0,98])
    ax0.set_ylabel('cross entropy loss')
    # ax0.set_ylim([0.05,0.6])    
    ax0.set_xlabel('integration time')

    plt.savefig('results/figures/mnist_GFE_conv.pdf')


# plot_momentum('GFE_momentum_standard', 'GFE_momentum')
plot_ae_amd_val_comp()
# table_ae_amd_val_comp()    
# plot_loss_comparison()
# plot_all_comp()
# plot_GFE_AMD_conv()

