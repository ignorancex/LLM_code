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
component_out = 'CMBbeam'
comp_sample_in = 'Tot_oneModel'; comp_sample_out = 'CMB'

map_type = 'I'

fwhm = None

#%%
component_in='TotBeamNoiseA0Beta0.1OneT0.05Multi';randn_num='';input_channels=[100,143,217,353];out_freq=[143];norm_input=False;norm_target=True


#%%
sides = [nside*4, nside*3]

outbeam_freq = out_freq[0]
input_c = len(input_channels)
print('randn_num:%s'%randn_num)
indir = 'cnn_unet'


#Simulated data
data_source = 'sim'; figure_type = 'test'


# Planck map
# data_source='obs'; comp_sample_in='single_frequency'; comp_sample_out='cmb'; figure_type='obs'
# component_in='Totfull'; component_out='CMBcommanderfull'; out_freq=[90]


net_file = loadFile.FilePath(filedir=indir, randn_num=randn_num, suffix='.pt').filePath()
net = torch.load(net_file)
net = net.cpu()
net.eval()

#%% load data
map_n = 0 #map_n should be reset to be a index of map in the test set
print('map_n=%s'%map_n)


tot = loader.random_arrange_fullMap([map_n], comp_sample=comp_sample_in,component=component_in,map_type=map_type,
                                    channels=input_channels,out_type='torch',normed=norm_input,learn_component='CMB', 
                                    nside=nside, map_shape=sides, data_source=data_source)

if fwhm==None:
    cmb = loader.random_arrange_fullMap([map_n], comp_sample=comp_sample_out,component=component_out,map_type=map_type,
                                        channels=out_freq,out_type='torch',normed=False,learn_component='CMB', 
                                        nside=nside, map_shape=sides, data_source=data_source)
else:
    cmb = loader.random_arrange_fullMap_CMBS4([map_n], comp_sample=comp_sample_out,component=component_out,map_type=map_type,
                                              channels=out_freq,fwhm=fwhm,out_type='torch',normed=False,learn_component='CMB', 
                                              nside=nside, map_shape=sides, data_source=data_source)


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

#plane map to sphere map
cmb = spherical.piecePlanes2spheres(cmb, nside=nside)
cmb_ML = spherical.piecePlanes2spheres(cmb_ML, nside=nside)

tot_sphere = []
for i in range(input_c):
    tot_sphere.append(spherical.piecePlanes2spheres(tot[i], nside=nside))
tot = tot_sphere

#%%
cmb_plt = plotter.PlotCMBFull(cmb, cmb_ML, randn_num=randn_num, map_type=map_type,
                              fig_type=figure_type, map_n=map_n, input_freqs=input_channels,
                              out_freq=outbeam_freq)

sf = True
# sf = False

if figure_type=='obs':
    cmb_plt.mask_plk()
else:
    cmb_plt.mask_manual() #no mask
    
cmb_plt.plot_cmb(savefig=sf, root='figures')
cmb_plt.plot_cmb_ML(savefig=sf, root='figures')
cmb_plt.plot_residual(savefig=sf, root='figures')
cmb_plt.plot_dl(savefig=sf, root='figures', show_title=False,
                fwhm=fwhm, aposize=None, nlb=1, bin_residual=True)

cmb_plt.plot_all(savefig=sf, root='figures', fwhm=fwhm, 
                  aposize=None, nlb=1, bin_residual=True)

cmb_plt.plot_miniPatch(savefig=sf, root='figures')


plt.show()

