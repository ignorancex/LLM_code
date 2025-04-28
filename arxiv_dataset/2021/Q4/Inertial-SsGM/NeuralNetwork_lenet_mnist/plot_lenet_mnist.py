#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

label_font = 18
plt.rcParams.update({'font.size':12})

######################################
dir = 'results/'     
SIZE = 6
batchsize = 40
alpha = 0.001
epochs = 300

name1 = dir+'results_cpu_mnist_mpi_lenet_inertial_sync_epochs'+str(epochs)+'_SIZE_'+str(SIZE)+'_bacthsize'+str(batchsize)+'_param_type_const'+'_alpha'+str(alpha)+'_beta'
name2 = '.txt'
 
file_name = name1+str(0.9)+name2
function_gaps_9 = np.loadtxt(file_name, skiprows=1, usecols=(0,2,3,5), max_rows=int(epochs/5))
file_name = name1+str(0.8)+name2;  
function_gaps_8 = np.loadtxt(file_name, skiprows=1, usecols=(0,2,3,5), max_rows=int(epochs/5))
file_name = name1+str(0.5)+name2; 
function_gaps_5 = np.loadtxt(file_name, skiprows=1, usecols=(0,2,3,5), max_rows=int(epochs/5))
file_name = name1+str(0.2)+name2; 
function_gaps_2 = np.loadtxt(file_name, skiprows=1, usecols=(0,2,3,5), max_rows=int(epochs/5))
file_name = name1+str(0.0)+name2; 
function_gaps_0 = np.loadtxt(file_name, skiprows=1, usecols=(0,2,3,5), max_rows=int(epochs/5))
 
file_name = dir + 'results_cpu_mnist_mpi_lenet_inertial_sync_epochs'+str(epochs)+'_SIZE_'+str(SIZE)+'_bacthsize'+str(batchsize)+'_param_type_beta_reduce'+'_alpha'+str(alpha)+'_beta'+str(0.9)+'.txt'
function_gaps_reduce_beta = np.loadtxt(file_name, skiprows=1, usecols=(0,2,3,5), max_rows=int(epochs/5))

plt.figure(figsize=(30,12))

plt.subplot(241) 
plt.plot(function_gaps_9[:,0], function_gaps_9[:,1], marker='o', markevery=4, label=r'$\beta_k$=0.9')
plt.plot(function_gaps_8[:,0], function_gaps_8[:,1], marker='s', markevery=5, label=r'$\beta_k$=0.8')
plt.plot(function_gaps_5[:,0], function_gaps_5[:,1], marker='p', markevery=7, label=r'$\beta_k$=0.5')
plt.plot(function_gaps_2[:,0], function_gaps_2[:,1], marker='d', markevery=11, label=r'$\beta_k$=0.2')
plt.plot(function_gaps_0[:,0], function_gaps_0[:,1], marker='^', markevery=7, label=r'$\beta_k$=0.0')
plt.plot(function_gaps_reduce_beta[:,0], function_gaps_reduce_beta[:,1], marker='*', markevery=5, label=r'$\beta_k$ diminish')
# plt.plot(function_gaps_reduce[:,0], function_gaps_reduce[:,1], marker='X', markevery=4, label=r'$\alpha_k,\beta_k$ diminish')
plt.xlim([5, epochs])
plt.ylim([97.2, 99.2])
plt.ylabel('test acc', fontsize=label_font)
plt.xlabel('epoch', fontsize=label_font)
plt.legend(fontsize=label_font)

plt.subplot(245) 
plt.semilogy(function_gaps_9[:,0], function_gaps_9[:,2], marker='o', markevery=4, label=r'$\beta_k$=0.9')
plt.semilogy(function_gaps_8[:,0], function_gaps_8[:,2], marker='s', markevery=5, label=r'$\beta_k$=0.8')
plt.semilogy(function_gaps_5[:,0], function_gaps_5[:,2], marker='p', markevery=7, label=r'$\beta_k$=0.5')
plt.semilogy(function_gaps_2[:,0], function_gaps_2[:,2], marker='d', markevery=11, label=r'$\beta_k$=0.2')
plt.semilogy(function_gaps_0[:,0], function_gaps_0[:,2], marker='^', markevery=7, label=r'$\beta_k$=0.0')
plt.semilogy(function_gaps_reduce_beta[:,0], function_gaps_reduce_beta[:,2], marker='*', markevery=5, label=r'$\beta_k$ diminish')
# plt.semilogy(function_gaps_reduce[:,0], function_gaps_reduce[:,2], marker='X', markevery=4, label=r'$\alpha_k,\beta_k$ diminish')
plt.xlim([5,epochs])
plt.ylim([0.00002,0.2])
plt.ylabel('train loss', fontsize=label_font)
plt.xlabel('epoch', fontsize=label_font)
plt.legend(fontsize=label_font)

  
epochs=300; batchsize=40;alpha=0.001;  

beta=0.9; 

name2 = '_bacthsize'+str(batchsize)+'_param_type_const'+'_alpha'+str(alpha)+'_beta'+str(beta)+'.txt'; #savename = 'mnist_with_const_bate'+str(beta)
name1 = dir+'results_cpu_mnist_mpi_lenet_inertial_sync_epochs'+str(epochs)+'_SIZE_' 

file_name = name1+str(2)+name2
function_gaps_2 = np.loadtxt(file_name, skiprows=1, usecols=(0,2,3,5), max_rows=int(epochs/5))
file_name = name1+str(3)+name2;  
function_gaps_3 = np.loadtxt(file_name, skiprows=1, usecols=(0,2,3,5), max_rows=int(epochs/5))
file_name = name1+str(6)+name2; 
function_gaps_6 = np.loadtxt(file_name, skiprows=1, usecols=(0,2,3,5), max_rows=int(epochs/5))
file_name = name1+str(11)+name2; 
function_gaps_11 = np.loadtxt(file_name, skiprows=1, usecols=(0,2,3,5), max_rows=int(epochs/5))
file_name = name1+str(21)+name2; 
function_gaps_21 = np.loadtxt(file_name, skiprows=1, usecols=(0,2,3,5), max_rows=int(epochs/5))
 
sync_time=[function_gaps_2[-1,3], function_gaps_3[-1,3], function_gaps_6[-1,3], function_gaps_11[-1,3], function_gaps_21[-1,3]]
 
name1 = dir+'results_cpu_mnist_mpi_lenet_inertial_async_epochs'+str(epochs)+'_SIZE_' 
 
file_name = name1+str(2)+name2
function_gaps_2 = np.loadtxt(file_name, skiprows=1, usecols=(0,2,3,5), max_rows=int(epochs/5))
file_name = name1+str(3)+name2;  
function_gaps_3 = np.loadtxt(file_name, skiprows=1, usecols=(0,2,3,5), max_rows=int(epochs/5))
file_name = name1+str(6)+name2; 
function_gaps_6 = np.loadtxt(file_name, skiprows=1, usecols=(0,2,3,5), max_rows=int(epochs/5))
file_name = name1+str(11)+name2; 
function_gaps_11 = np.loadtxt(file_name, skiprows=1, usecols=(0,2,3,5), max_rows=int(epochs/5))
file_name = name1+str(21)+name2; 
function_gaps_21 = np.loadtxt(file_name, skiprows=1, usecols=(0,2,3,5), max_rows=int(epochs/5))

async_time=[function_gaps_2[-1,3], function_gaps_3[-1,3], function_gaps_6[-1,3], function_gaps_11[-1,3], function_gaps_21[-1,3]]
 
    
    
plt.subplot(242)  
plt.plot(function_gaps_2[:,0], function_gaps_2[:,1], marker='o', markevery=4, label=r'#worker=1')
plt.plot(function_gaps_3[:,0], function_gaps_3[:,1], marker='s', markevery=5, label=r'#worker=2')
plt.plot(function_gaps_6[:,0], function_gaps_6[:,1], marker='p', markevery=7, label=r'#worker=5')
plt.plot(function_gaps_11[:,0], function_gaps_11[:,1], marker='d', markevery=5, label=r'#worker=10')
plt.plot(function_gaps_21[:,0], function_gaps_21[:,1], marker='^', markevery=4, label=r'#worker=20')
plt.xlim([5,epochs])
plt.ylim([97.2,99.2])
plt.ylabel('test acc', fontsize=label_font)
plt.xlabel('epoch', fontsize=label_font)
plt.legend(fontsize=label_font)

plt.subplot(246)  
plt.semilogy(function_gaps_2[:,0], function_gaps_2[:,2], marker='o', markevery=4, label=r'#worker=1')
plt.semilogy(function_gaps_3[:,0], function_gaps_3[:,2], marker='s', markevery=5, label=r'#worker=2')
plt.semilogy(function_gaps_6[:,0], function_gaps_6[:,2], marker='p', markevery=7, label=r'#worker=5')
plt.semilogy(function_gaps_11[:,0], function_gaps_11[:,2], marker='d', markevery=5, label=r'#worker=10')
plt.semilogy(function_gaps_21[:,0], function_gaps_21[:,2], marker='^', markevery=4, label=r'#worker=20')
plt.xlim([5,epochs])
plt.ylim([0.00002,0.2])
plt.ylabel('train loss', fontsize=label_font)
plt.xlabel('epoch', fontsize=label_font)
plt.legend(fontsize=label_font)
 

epochs=300; batchsize=40;alpha=0.001;

beta=0.9; name2 = '_bacthsize'+str(batchsize)+'_param_type_beta_reduce'+'_alpha'+str(alpha)+'_beta'+str(beta)+'.txt'; 
 
name1 = dir+'results_cpu_mnist_mpi_lenet_inertial_sync_epochs'+str(epochs)+'_SIZE_' 

file_name = name1+str(2)+name2
function_gaps_2 = np.loadtxt(file_name, skiprows=1, usecols=(0,2,3,5), max_rows=int(epochs/5))
file_name = name1+str(3)+name2;  
function_gaps_3 = np.loadtxt(file_name, skiprows=1, usecols=(0,2,3,5), max_rows=int(epochs/5))
file_name = name1+str(6)+name2; 
function_gaps_6 = np.loadtxt(file_name, skiprows=1, usecols=(0,2,3,5), max_rows=int(epochs/5))
file_name = name1+str(11)+name2; 
function_gaps_11 = np.loadtxt(file_name, skiprows=1, usecols=(0,2,3,5), max_rows=int(epochs/5))
file_name = name1+str(21)+name2; 
function_gaps_21 = np.loadtxt(file_name, skiprows=1, usecols=(0,2,3,5), max_rows=int(epochs/5))
 
sync_time=[function_gaps_2[-1,3], function_gaps_3[-1,3], function_gaps_6[-1,3], function_gaps_11[-1,3], function_gaps_21[-1,3]]
 
name1 = dir+'results_cpu_mnist_mpi_lenet_inertial_async_epochs'+str(epochs)+'_SIZE_' 
 
file_name = name1+str(2)+name2
function_gaps_2 = np.loadtxt(file_name, skiprows=1, usecols=(0,2,3,5), max_rows=int(epochs/5))
file_name = name1+str(3)+name2;  
function_gaps_3 = np.loadtxt(file_name, skiprows=1, usecols=(0,2,3,5), max_rows=int(epochs/5))
file_name = name1+str(6)+name2; 
function_gaps_6 = np.loadtxt(file_name, skiprows=1, usecols=(0,2,3,5), max_rows=int(epochs/5))
file_name = name1+str(11)+name2; 
function_gaps_11 = np.loadtxt(file_name, skiprows=1, usecols=(0,2,3,5), max_rows=int(epochs/5))
file_name = name1+str(21)+name2; 
function_gaps_21 = np.loadtxt(file_name, skiprows=1, usecols=(0,2,3,5), max_rows=int(epochs/5))

async_time=[function_gaps_2[-1,3], function_gaps_3[-1,3], function_gaps_6[-1,3], function_gaps_11[-1,3], function_gaps_21[-1,3]]
 
plt.subplot(243)  
plt.plot(function_gaps_2[:,0], function_gaps_2[:,1], marker='o', markevery=4, label=r'#worker=1')
plt.plot(function_gaps_3[:,0], function_gaps_3[:,1], marker='s', markevery=5, label=r'#worker=2')
plt.plot(function_gaps_6[:,0], function_gaps_6[:,1], marker='p', markevery=7, label=r'#worker=5')
plt.plot(function_gaps_11[:,0], function_gaps_11[:,1], marker='d', markevery=5, label=r'#worker=10')
plt.plot(function_gaps_21[:,0], function_gaps_21[:,1], marker='^', markevery=4, label=r'#worker=20')
plt.xlim([5,epochs])
plt.ylim([97.2,99.2])
plt.ylabel('test acc', fontsize=label_font)
plt.xlabel('epoch', fontsize=label_font)
plt.legend(fontsize=label_font)

plt.subplot(247)  
plt.semilogy(function_gaps_2[:,0], function_gaps_2[:,2], marker='o', markevery=4, label=r'#worker=1')
plt.semilogy(function_gaps_3[:,0], function_gaps_3[:,2], marker='s', markevery=5, label=r'#worker=2')
plt.semilogy(function_gaps_6[:,0], function_gaps_6[:,2], marker='p', markevery=7, label=r'#worker=5')
plt.semilogy(function_gaps_11[:,0], function_gaps_11[:,2], marker='d', markevery=5, label=r'#worker=10')
plt.semilogy(function_gaps_21[:,0], function_gaps_21[:,2], marker='^', markevery=4, label=r'#worker=20')
plt.xlim([5,epochs])
plt.ylim([0.00002,0.2])
plt.ylabel('train loss', fontsize=label_font)
plt.xlabel('epoch', fontsize=label_font)
plt.legend(fontsize=label_font)

plt.subplot(2,4,(4,8))  
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

savename = 'lenet_mnist'
plt.savefig(savename+'.png', bbox_inches='tight', format='png') 
#plt.show()


# In[ ]:




