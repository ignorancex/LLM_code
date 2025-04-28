#!/usr/bin/env python
# coding: utf-8
 
import numpy as np
import matplotlib.pyplot as plt

label_font = 20
plt.rcParams.update({'font.size':12})
dir = './results/'
epochs = 100
[m,d] = [50000,20000]
alpha = 5e-5
beta = 0.9
batchsize = 100

name1 = dir+'results_phaseretrieval_distributed_data_mpi_cpu_m'+str(m)+'_d'+str(d)+'_inertial_sync_epochs100_SIZE_6_bacthsize'+str(batchsize)+'_param_type_const_alpha'+str(alpha)+'_beta'
name2 = '.txt'
file_name = name1+str(0.9)+name2
function_gaps_9 = np.loadtxt(file_name, skiprows=1, max_rows=100)
file_name = name1+str(0.8)+name2
function_gaps_8 = np.loadtxt(file_name, skiprows=1, max_rows=100)
file_name = name1+str(0.5)+name2
function_gaps_5 = np.loadtxt(file_name, skiprows=1, max_rows=100)
file_name = name1+str(0.2)+name2
function_gaps_2 = np.loadtxt(file_name, skiprows=1, max_rows=100)
file_name = name1+str(0.0)+name2
function_gaps_0 = np.loadtxt(file_name, skiprows=1, max_rows=100)
 

file_name = dir+'results_phaseretrieval_distributed_data_mpi_cpu_m'+str(m)+'_d'+str(d)+'_inertial_sync_epochs100_SIZE_21_bacthsize'+str(batchsize)+'_param_type_reduce_period_type_epoch_alpha'+str(alpha)+'_beta'+str(beta)+'.txt'
function_gaps_reduce = np.loadtxt(file_name, skiprows=1, max_rows=100)

file_name = dir+'results_phaseretrieval_distributed_data_mpi_cpu_m'+str(m)+'_d'+str(d)+'_inertial_sync_epochs100_SIZE_6_bacthsize'+str(batchsize)+'_param_type_alpha_reduce_period_type_epoch_alpha'+str(alpha)+'_beta0.0.txt'
function_gaps_alphareduce = np.loadtxt(file_name, skiprows=1, max_rows=100)

plt.figure(figsize=(25,12))

plt.subplot(231)  
plt.semilogy(function_gaps_9[:,0], function_gaps_9[:,1], marker='o', markevery=4, label=r'$\beta_k$=0.9')
plt.semilogy(function_gaps_8[:,0], function_gaps_8[:,1], marker='s', markevery=5, label=r'$\beta_k$=0.8')
plt.semilogy(function_gaps_5[:,0], function_gaps_5[:,1], marker='p', markevery=7, label=r'$\beta_k$=0.5')
plt.semilogy(function_gaps_2[:,0], function_gaps_2[:,1], marker='d', markevery=11, label=r'$\beta_k$=0.2')
plt.semilogy(function_gaps_0[:,0], function_gaps_0[:,1], marker='^', markevery=13, label=r'$\beta_k$=0.0')
plt.semilogy(function_gaps_alphareduce[:,0], function_gaps_alphareduce[:,1], marker='X', markevery=13, label=r'$\alpha_k$ diminish, $\beta_k$=0.0')
plt.semilogy(function_gaps_reduce[:,0], function_gaps_reduce[:,1], marker='*', markevery=17, label=r'$\alpha_k,\beta_k$ diminish')
plt.ylabel(r'$\|\mathbf{x}-\mathbf{x}^*\|$',fontsize=label_font,usetex=True)
plt.xlabel('epoch',fontsize=label_font) 
plt.legend(fontsize=label_font,loc='upper right')
plt.xlim([0,epochs])

plt.subplot(234)  
plt.semilogy(function_gaps_9[:,0], function_gaps_9[:,2], marker='o', markevery=4, label=r'$\beta_k$=0.9')
plt.semilogy(function_gaps_8[:,0], function_gaps_8[:,2], marker='s', markevery=5, label=r'$\beta_k$=0.8')
plt.semilogy(function_gaps_5[:,0], function_gaps_5[:,2], marker='p', markevery=7, label=r'$\beta_k$=0.5')
plt.semilogy(function_gaps_2[:,0], function_gaps_2[:,2], marker='d', markevery=11, label=r'$\beta_k$=0.2')
plt.semilogy(function_gaps_0[:,0], function_gaps_0[:,2], marker='^', markevery=13, label=r'$\beta_k$=0.0')
plt.semilogy(function_gaps_alphareduce[:,0], function_gaps_alphareduce[:,2], marker='X', markevery=13, label=r'$\alpha_k$ diminish, $\beta_k$=0.0')
plt.semilogy(function_gaps_reduce[:,0], function_gaps_reduce[:,2], marker='*', markevery=17, label=r'$\alpha_k,\beta_k$ diminish')
plt.ylabel('objective value',fontsize=label_font)
plt.xlabel('epoch',fontsize=label_font) 
plt.legend(fontsize=label_font,loc='upper right')
plt.xlim([0,epochs])

#####

name1=dir+'results_phaseretrieval_distributed_data_mpi_cpu_m'+str(m)+'_d'+str(d)+'_inertial_async_epochs100_SIZE_'
name2='_bacthsize'+str(batchsize)+'_param_type_reduce_period_type_epoch_alpha'+str(alpha)+'_beta'+str(beta)+'.txt'
file_name = name1+str(2)+name2
function_gaps_async_2 = np.loadtxt(file_name, skiprows=1, max_rows=100)
file_name = name1+str(3)+name2
function_gaps_async_3 = np.loadtxt(file_name, skiprows=1, max_rows=100)
file_name = name1+str(6)+name2
function_gaps_async_6 = np.loadtxt(file_name, skiprows=1, max_rows=100)
file_name = name1+str(11)+name2
function_gaps_async_11 = np.loadtxt(file_name, skiprows=1, max_rows=100)
file_name = name1+str(21)+name2
function_gaps_async_21 = np.loadtxt(file_name, skiprows=1, max_rows=100)

async_time=[function_gaps_async_2[-1,3], function_gaps_async_3[-1,3], function_gaps_async_6[-1,3], function_gaps_async_11[-1,3], function_gaps_async_21[-1,3]]

plt.subplot(232)  
plt.semilogy(function_gaps_async_2[:,0], function_gaps_async_2[:,1], marker='o', markevery=4, label=r'#workers=1')
plt.semilogy(function_gaps_async_3[:,0], function_gaps_async_3[:,1], marker='s', markevery=5, label=r'#workers=2')
plt.semilogy(function_gaps_async_6[:,0], function_gaps_async_6[:,1], marker='p', markevery=7, label=r'#workers=5')
plt.semilogy(function_gaps_async_11[:,0], function_gaps_async_11[:,1], marker='d', markevery=11, label=r'#workers=10')
plt.semilogy(function_gaps_async_21[:,0], function_gaps_async_21[:,1], marker='*', markevery=13, label=r'#workers=20')
plt.ylabel(r'$\|\mathbf{x}-\mathbf{x}^*\|$',fontsize=label_font,usetex=True)
plt.xlabel('epoch',fontsize=label_font)
plt.legend(fontsize=label_font,loc='upper right')
plt.xlim([0,epochs])

plt.subplot(235)  
plt.semilogy(function_gaps_async_2[:,0], function_gaps_async_2[:,2], marker='o', markevery=4, label=r'#workers=1')
plt.semilogy(function_gaps_async_3[:,0], function_gaps_async_3[:,2], marker='s', markevery=5, label=r'#workers=2')
plt.semilogy(function_gaps_async_6[:,0], function_gaps_async_6[:,2], marker='p', markevery=7, label=r'#workers=5')
plt.semilogy(function_gaps_async_11[:,0], function_gaps_async_11[:,2], marker='d', markevery=11, label=r'#workers=10')
plt.semilogy(function_gaps_async_21[:,0], function_gaps_async_21[:,2], marker='*', markevery=13, label=r'#workers=20')
plt.ylabel('objective value',fontsize=label_font)
plt.xlabel('epoch',fontsize=label_font)
plt.legend(fontsize=label_font,loc='upper right')
plt.xlim([0,epochs])
  
name1=dir+'results_phaseretrieval_distributed_data_mpi_cpu_m'+str(m)+'_d'+str(d)+'_inertial_sync_epochs100_SIZE_'
name2='_bacthsize'+str(batchsize)+'_param_type_reduce_period_type_epoch_alpha'+str(alpha)+'_beta'+str(beta)+'.txt'
file_name = name1+str(2)+name2
function_gaps_sync_2 = np.loadtxt(file_name, skiprows=1, max_rows=100)
file_name = name1+str(3)+name2
function_gaps_sync_3 = np.loadtxt(file_name, skiprows=1, max_rows=100)
file_name = name1+str(6)+name2
function_gaps_sync_6 = np.loadtxt(file_name, skiprows=1, max_rows=100)
file_name = name1+str(11)+name2
function_gaps_sync_11 = np.loadtxt(file_name, skiprows=1, max_rows=100)
file_name = name1+str(21)+name2
function_gaps_sync_21 = np.loadtxt(file_name, skiprows=1, max_rows=100)

sync_time = [function_gaps_sync_2[-1,3], function_gaps_sync_3[-1,3], function_gaps_sync_6[-1,3], function_gaps_sync_11[-1,3], function_gaps_sync_21[-1,3]]
 
plt.subplot(2,3,(3,6))   
name_list = ['1','2','5','10','20'] 
x = np.arange(len(name_list))
# x = np.array([1,2,5,10,21]
total_width, n = 0.8, 2
width = total_width / n

plt.bar(x[0], sync_time[0], width=width,fc='b')
plt.bar(x[1:]-width/2, sync_time[1:], width=width, label='sync',fc='b')
plt.bar(x,[0,0,0,0,0], tick_label=name_list, width=0.0)
plt.bar(x[1:]+width/2, async_time[1:], width=width, label='async',fc='g')
plt.legend(fontsize=label_font)
plt.ylabel('times (sec)',fontsize=label_font)
plt.xlabel('#workers',fontsize=label_font)
 
savename = 'PhaseRetrieval'+'_m'+str(m)+'_d'+str(d)
plt.savefig('./'+savename+'.png',bbox_inches='tight',format='png') 
