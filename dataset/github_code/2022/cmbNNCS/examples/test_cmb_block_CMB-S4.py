import sys
sys.path.append('..')
sys.path.append('../..')
import cmbnncs.loadFile as loadFile
import cmbnncs.data_process as dp
import cmbnncs.spherical as spherical
import numpy as np
import loader
import torch
import healpy as hp
from torch.autograd import Variable
import plotter
import matplotlib.pyplot as plt


nside = 512

input_channels=[100,143,217,353]


comp_sample_in = 'Tot_oneModel'; comp_sample_out = 'CMB'

map_type='Q'

block_n = 0

#%% for CMB-S4
#Q
component_in='TotFwhmNoiseA0Beta0.1OneT0.05MultiBlock0';component_out='CMBfwhm';randn_num='';input_channels=[85, 95, 145, 155, 220, 270];out_freq=[220];fwhm=10.7;norm_input=False;norm_target=False

#U
# component_in='TotFwhmNoiseA0Beta0.1OneT0.05MultiBlock0';component_out='CMBfwhm';randn_num='';input_channels=[85, 95, 145, 155, 220, 270];out_freq=[220];fwhm=10.7;norm_input=False;map_type='U';norm_target=False#train1k_epoch1w,batchSize:128,finalBN:False,use 2GPUs !!!!, use this


#%%
sides = [nside, nside]

outbeam_freq = out_freq[0]
input_c = len(input_channels)
print('randn_num:%s'%randn_num)
indir = 'cnn_unet'


#Simulated data
data_source = 'sim'; figure_type = 'test'


net_file = loadFile.FilePath(filedir=indir, randn_num=randn_num, suffix='.pt').filePath()
net = torch.load(net_file)
net = net.cpu()
net.eval()
# print (net)

#%% load data
map_n = 0 #map_n should be reset to be a index of map in the test set
print('map_n=%s'%map_n)


if nside==512:
    lmax = 1500
elif nside==256:
    lmax = 760

tot = loader.random_arrange_fullMap_CMBS4([map_n], comp_sample=comp_sample_in,component=component_in,map_type=map_type,
                                          channels=input_channels,out_type='torch',normed=norm_input,learn_component='CMB', 
                                          nside=nside, map_shape=sides, data_source=data_source)

cmb = loader.random_arrange_fullMap_CMBS4([map_n], comp_sample=comp_sample_out,component=component_out,map_type=map_type,
                                          channels=out_freq,fwhm=fwhm,out_type='torch',normed=False,learn_component='CMB', 
                                          nside=nside, map_shape=sides, data_source=data_source, block_map=True)


#%% get map
# ML
cmb_ML = net(Variable(tot))

cmb_ML = cmb_ML.data.numpy()[0,0,:,:]
if norm_target:
    cmb_ML = loader.inverse_transform(cmb_ML,component='CMB',map_type=map_type,block=4)

print('y_mean:%.3f'%(cmb.mean()),'y_min:%.3f'%(cmb.min()),'y_max:%.3f'%(cmb.max()))
print('p_mean:%.3f'%(cmb_ML.mean()),'p_min:%.3f'%(cmb_ML.min()),'p_max:%.3f'%(cmb_ML.max()))

tot = dp.torch2numpy(tot)[0,:,:,:]
cmb = dp.torch2numpy(cmb)[0,0,:,:]
if norm_input:
    tot = loader.inverse_transform(tot,component='CMB',map_type=map_type,block=4)


#%%    
cmb_plt = plotter.PlotCMBBlock(cmb, cmb_ML, randn_num=randn_num, map_type=map_type,
                               fig_type=figure_type, map_n=map_n, input_freqs=input_channels,
                               out_freq=outbeam_freq, block_n=block_n)

sf = True
sf = False


cmb_plt.mask_manual()

# cmb_plt.plot_cmb(savefig=sf, root='figures_blocks')
# cmb_plt.plot_cmb_ML(savefig=sf, root='figures_blocks')
# cmb_plt.plot_residual(savefig=sf, root='figures_blocks')
# cmb_plt.plot_dl(savefig=sf, root='figures_blocks', show_title=False, show_mse=False,
#                 fwhm=fwhm, aposize=1, nlb=10)

cmb_plt.plot_all(savefig=sf, root='figures_blocks', fwhm=fwhm, aposize=1, nlb=10)


plt.show()


