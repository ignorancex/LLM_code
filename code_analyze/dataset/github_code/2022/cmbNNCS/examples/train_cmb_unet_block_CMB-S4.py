import os
import sys
sys.path.append('../')
sys.path.append('../..')
import cmbnncs.utils as utils
import cmbnncs.unet as unet
import cmbnncs.optimize as optimize
import torch
import numpy as np
from torch.autograd import Variable
import loader
import time
start_time = time.time()


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


nside = 512

sides = (nside ,nside)


num_train = 1000

iteration_n = 10000


#%% save logs
randn_num = round(abs(np.random.randn()), 5)
print ('randn_num: %s'%randn_num)


logName = 'log_%s'%(randn_num)
print (logName)
utils.logger(path='cnn_lens', fileName=logName)


#%%
comp_sample_in = 'Tot_oneModel'; comp_sample_out = 'CMB'


map_type = 'Q' #'Q', 'U'


input_channels=[85, 95, 145, 155, 220, 270]
input_c = len(input_channels)


components_in='TotFwhmNoiseA0Beta0.1OneT0.05MultiBlock0';components_out='CMBfwhm';out_freq=[220];fwhm_cmb=10.7


print(comp_sample_in, map_type, num_train, input_channels, out_freq)
print(components_in, components_out, fwhm_cmb)


norm_input = False
finalBN = False
if map_type=='I':
    norm_target = True
elif map_type=='Q' or map_type=='U':
    norm_target = False
print ('norm_input=%s'%norm_input, 'norm_target=%s'%norm_target, 'finalBN=%s'%finalBN)
mainActive = 'prelu'; finalActive = 'prelu'


#%% for block maps
kernels_size = [4,4,4,4,4, 4,4,4,4,2]


strides=[2,2,2,2,2,2,2,2,2,2]
channels=(32,64,128,256,512);batch_size = 128


print ('UNet5')
net = unet.UNet5(channels,channel_in=input_c,channel_out=1,kernels_size=kernels_size,strides=strides,extra_pads=None,mainActive=mainActive,
                  finalActive=finalActive,finalBN=finalBN, sides=sides)


#%% 
print ('batch_size:%s'%batch_size, kernels_size, 'mainActive:%s'%mainActive, 'finalActive:%s'%finalActive)
use_multiGPU = False
net = torch.nn.DataParallel(net, device_ids=None); use_multiGPU = True
# print (net)
if torch.cuda.is_available():
    net = net.cuda()


#%%
train_name = 'train%s__batchSize%s__%s'%(num_train,batch_size,randn_num)


lr = 1e-1
lr_min = 1e-6
loss_func = torch.nn.L1Loss(reduction='mean')
optimizer = torch.optim.Adam(net.parameters(), lr = lr)


repeat_n = 3
np.random.seed(1000)
loss_all = []
for iteration in range(1, iteration_n+1):
    map_nums = np.random.choice(num_train, batch_size, replace=False)
    xx = loader.random_arrange_fullMap_CMBS4(map_nums, comp_sample=comp_sample_in,component=components_in,map_type=map_type,
                                             channels=input_channels,out_type='cuda',normed=norm_input,learn_component='CMB', 
                                             nside=nside, map_shape=sides, block_map=False)
    
    yy = loader.random_arrange_fullMap_CMBS4(map_nums, comp_sample=comp_sample_out,component=components_out,map_type=map_type,
                                             channels=out_freq,fwhm=fwhm_cmb,out_type='cuda',normed=norm_target,learn_component='CMB', 
                                             nside=nside, map_shape=sides, block_map=True)
        
    if iteration%(500+1)==0:
        print ('X_mean:%.4f'%(xx.mean()),'X_min:%.4f'%(xx.min()),'X_max:%.4f'%(xx.max()))
        print ('y_mean:%.4f'%(yy.mean()),'y_min:%.4f'%(yy.min()),'y_max:%.4f'%(yy.max()))
    xx = Variable(xx); yy = Variable(yy, requires_grad=False)
    
    repeat_n = repeat_n
    for t in range(repeat_n):
        predicted = net(xx)
        if t+1==repeat_n and iteration%(500+1)==0:
            print ('p_mean:%.4f'%(predicted.data.mean()),'p_min:%.4f'%(predicted.data.min()),'p_max:%.4f \n'%(predicted.data.max()))
        loss = loss_func(predicted, yy)
        loss_all.append(loss.item())
        if t+1==repeat_n and iteration%50==0:
            print ('(iteration:%s/%s; loss:%.5f; lr:%.8f)'%(iteration, iteration_n, loss.item(), optimizer.param_groups[0]['lr']))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # save the model
    if iteration%1000==0:
        print('The network and loss are saved;', 'randn_num: %s'%randn_num)
        utils.savenpy('cnn_lens', 'loss-%s'%train_name, np.array(loss_all))
        if use_multiGPU:
            torch.save(net.module.cpu(), 'cnn_lens/net-%s.pt'%(train_name))
        else:
            torch.save(net.cpu(), 'cnn_lens/net-%s.pt'%(train_name))
        net.cuda()
    
    # reduce the learning rate
    lrdc = optimize.LearningRateDecay(iteration,iteration_max=iteration_n,lr=lr,lr_min=lr_min)
    optimizer.param_groups[0]['lr'] = lrdc.exp()


#%%
print ("Time elapsed: %.3f" %((time.time()-start_time)/60), "mins")


