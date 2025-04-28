#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

label_font = 20
plt.rcParams.update({'font.size':12}) 
 
###################################################
epochs = 300
SIZE = 21
batchsize = 100
beta = 0.9
 
dir = './results/'
alpha = 0.0005
max_rows=77

name11 = dir+'results_cpu_mnist_mpi_Bilinear_r_5_lam0.001_inertial_sync_epochs'+str(epochs)+'_SIZE_'+str(SIZE)+'_bacthsize'+str(batchsize)+'_param_type_const_alpha'+str(alpha)+'_beta';
name2 = '_bacthsize'+str(batchsize)+'_param_type_const'+'_alpha'+str(alpha)+'_beta'+str(beta)+'.txt'
savename = 'SparseBilinearLogisticRegression_mnist' 
 
name12 = '.txt'  
file_name = name11+str(0.9)+name12
function_gaps_9 = np.loadtxt(file_name, skiprows=1, usecols=(0,2,3,5), max_rows=max_rows)
file_name = name11+str(0.8)+name12;  
function_gaps_8 = np.loadtxt(file_name, skiprows=1, usecols=(0,2,3,5), max_rows=max_rows)
file_name = name11+str(0.5)+name12; 
function_gaps_5 = np.loadtxt(file_name, skiprows=1, usecols=(0,2,3,5), max_rows=max_rows)
file_name = name11+str(0.2)+name12; 
function_gaps_2 = np.loadtxt(file_name, skiprows=1, usecols=(0,2,3,5), max_rows=max_rows)
file_name = name11+str(0.0)+name12; 
function_gaps_0 = np.loadtxt(file_name, skiprows=1, usecols=(0,2,3,5), max_rows=max_rows)

plt.figure(figsize=(25,12))
plt.subplot(231)  
plt.plot(function_gaps_9[:,0], function_gaps_9[:,1], marker='o', markevery=4, label=r'$\beta_k$=0.9')
plt.plot(function_gaps_8[:,0], function_gaps_8[:,1], marker='s', markevery=5, label=r'$\beta_k$=0.8')
plt.plot(function_gaps_5[:,0], function_gaps_5[:,1], marker='p', markevery=7, label=r'$\beta_k$=0.5')
plt.plot(function_gaps_2[:,0], function_gaps_2[:,1], marker='d', markevery=11, label=r'$\beta_k$=0.2')
plt.plot(function_gaps_0[:,0], function_gaps_0[:,1], marker='^', markevery=13, label=r'$\beta_k$=0.0')
plt.xlim([1,epochs])
# plt.ylim([60,92])
plt.ylim([85,92])
plt.ylabel('test acc', fontsize=label_font)
plt.xlabel('epoch', fontsize=label_font)
plt.legend(fontsize=label_font)

plt.subplot(234)  
plt.plot(function_gaps_9[:,0], function_gaps_9[:,2], marker='o', markevery=4, label=r'$\beta_k$=0.9')
plt.plot(function_gaps_8[:,0], function_gaps_8[:,2], marker='s', markevery=5, label=r'$\beta_k$=0.8')
plt.plot(function_gaps_5[:,0], function_gaps_5[:,2], marker='p', markevery=7, label=r'$\beta_k$=0.5')
plt.plot(function_gaps_2[:,0], function_gaps_2[:,2], marker='d', markevery=11, label=r'$\beta_k$=0.2')
plt.plot(function_gaps_0[:,0], function_gaps_0[:,2], marker='^', markevery=13, label=r'$\beta_k$=0.0')
plt.xlim([1,epochs])
# plt.ylim([0.4,1.5])
plt.ylim([0.43,1.])
plt.ylabel('train loss', fontsize=label_font)
plt.xlabel('epoch', fontsize=label_font)
plt.legend(fontsize=label_font)
   
name1 = dir+'results_cpu_mnist_mpi_Bilinear_r_5_lam0.001_inertial_sync_epochs'+str(epochs)+'_SIZE_' 

file_name = name1+str(2)+name2
function_gaps_2 = np.loadtxt(file_name, skiprows=1, usecols=(0,2,3,5), max_rows=max_rows)
file_name = name1+str(3)+name2
function_gaps_3 = np.loadtxt(file_name, skiprows=1, usecols=(0,2,3,5), max_rows=max_rows)
file_name = name1+str(6)+name2
function_gaps_6 = np.loadtxt(file_name, skiprows=1, usecols=(0,2,3,5), max_rows=max_rows)
file_name = name1+str(11)+name2
function_gaps_11 = np.loadtxt(file_name, skiprows=1, usecols=(0,2,3,5), max_rows=max_rows)
file_name = name1+str(21)+name2
function_gaps_21 = np.loadtxt(file_name, skiprows=1, usecols=(0,2,3,5), max_rows=max_rows)
 
sync_time=[function_gaps_2[-1,3], function_gaps_3[-1,3], function_gaps_6[-1,3], function_gaps_11[-1,3], function_gaps_21[-1,3]]

name1 = dir+'results_cpu_mnist_mpi_Bilinear_r_5_lam0.001_inertial_async_epochs'+str(epochs)+'_SIZE_' 

file_name = name1+str(2)+name2
function_gaps_2 = np.loadtxt(file_name, skiprows=1, usecols=(0,2,3,5), max_rows=max_rows)
file_name = name1+str(3)+name2
function_gaps_3 = np.loadtxt(file_name, skiprows=1, usecols=(0,2,3,5), max_rows=max_rows)
file_name = name1+str(6)+name2
function_gaps_6 = np.loadtxt(file_name, skiprows=1, usecols=(0,2,3,5), max_rows=max_rows)
file_name = name1+str(11)+name2
function_gaps_11 = np.loadtxt(file_name, skiprows=1, usecols=(0,2,3,5), max_rows=max_rows)
file_name = name1+str(21)+name2
function_gaps_21 = np.loadtxt(file_name, skiprows=1, usecols=(0,2,3,5), max_rows=max_rows)

async_time=[function_gaps_2[-1,3], function_gaps_3[-1,3], function_gaps_6[-1,3], function_gaps_11[-1,3], function_gaps_21[-1,3]]
   
plt.subplot(232)  
plt.plot(function_gaps_2[:,0], function_gaps_2[:,1], marker='o', markevery=4, label=r'#worker=1')
plt.plot(function_gaps_3[:,0], function_gaps_3[:,1], marker='s', markevery=5, label=r'#worker=2')
plt.plot(function_gaps_6[:,0], function_gaps_6[:,1], marker='p', markevery=7, label=r'#worker=5')
plt.plot(function_gaps_11[:,0], function_gaps_11[:,1], marker='d', markevery=11, label=r'#worker=10')
plt.plot(function_gaps_21[:,0], function_gaps_21[:,1], marker='^', markevery=13, label=r'#worker=20')
plt.xlim([1,epochs])
plt.ylim([85,92])
# plt.ylim([60,92])
plt.ylabel('test acc', fontsize=label_font)
plt.xlabel('epoch', fontsize=label_font)
plt.legend(fontsize=label_font,loc='lower right')

plt.subplot(235)  
plt.plot(function_gaps_2[:,0], function_gaps_2[:,2], marker='o', markevery=4, label=r'#worker=1')
plt.plot(function_gaps_3[:,0], function_gaps_3[:,2], marker='s', markevery=5, label=r'#worker=2')
plt.plot(function_gaps_6[:,0], function_gaps_6[:,2], marker='p', markevery=7, label=r'#worker=5')
plt.plot(function_gaps_11[:,0], function_gaps_11[:,2], marker='d', markevery=11, label=r'#worker=10')
plt.plot(function_gaps_21[:,0], function_gaps_21[:,2], marker='^', markevery=13, label=r'#worker=20')
plt.xlim([1,epochs])
plt.ylim([0.43,1.])
# plt.ylim([0.4,1.2])
plt.ylabel('train loss', fontsize=label_font)
plt.xlabel('epoch', fontsize=label_font)
plt.legend(fontsize=label_font)

plt.subplot(2,3,(3,6))   
name_list = ['1','2','5','10','20'] 
x = np.arange(len(name_list)) 
total_width, n = 0.8, 2
width = total_width / n

plt.bar(x[0], sync_time[0], width=width,fc='b')
plt.bar(x[1:]-width/2, sync_time[1:], width=width, label='sync',fc='b')
plt.bar(x,[0,0,0,0,0], tick_label=name_list, width=0.0)
plt.bar(x[1:]+width/2, async_time[1:], width=width, label='async',fc='g')
plt.legend(fontsize=label_font)
plt.ylabel('times (sec)', fontsize=label_font)
plt.xlabel('#workers', fontsize=label_font)
# plt.title('compare')
   
plt.savefig(savename+'.png', bbox_inches='tight', format='png') 
