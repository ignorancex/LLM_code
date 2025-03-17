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

block_n = 0

#%% for CMB-S4
#Q
component_in='TotFwhmNoiseA0Beta0.1OneT0.05MultiBlock0';component_out='CMBfwhm';randn_num_q='';input_channels=[85, 95, 145, 155, 220, 270];out_freq=[220];fwhm=10.7;norm_input=False;norm_target=False


#U
component_in='TotFwhmNoiseA0Beta0.1OneT0.05MultiBlock0';component_out='CMBfwhm';randn_num_u='';input_channels=[85, 95, 145, 155, 220, 270];out_freq=[220];fwhm=10.7;norm_input=False;norm_target=False

#%%
if nside==512:
    lmax = 1500
elif nside==256:
    lmax = 760

randn_marker_q = plotter.change_randn_num(randn_num_q)
randn_marker_u = plotter.change_randn_num(randn_num_u)
randn_marker = randn_marker_q + '_' + randn_marker_u

#%%
#Simulated data
data_source = 'sim'; figure_type = 'test'

#% load data
map_n = 0 #map_n should be reset to be a index of map in the test set
print('map_n=%s'%map_n)


#%%
def get_recoveredMap(randn_num, map_type=''):
    sides = [nside, nside]
    
    indir = 'cnn_unet'

    net_file = loadFile.FilePath(filedir=indir, randn_num=randn_num, suffix='.pt').filePath()
    net = torch.load(net_file)
    net = net.cpu()
    net.eval()
    # print (net)

    tot = loader.random_arrange_fullMap_CMBS4([map_n], comp_sample=comp_sample_in,component=component_in,map_type=map_type,
                                              channels=input_channels,out_type='torch',normed=norm_input,learn_component='CMB', 
                                              nside=nside, map_shape=sides, data_source=data_source)
    
    cmb = loader.random_arrange_fullMap_CMBS4([map_n], comp_sample=comp_sample_out,component=component_out,map_type=map_type,
                                              channels=out_freq,fwhm=fwhm,out_type='torch',normed=False,learn_component='CMB', 
                                              nside=nside, map_shape=sides, data_source=data_source, block_map=True)

    #% get map
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

    #plane map to sphere full map
    cmb = spherical.Block2Full(cmb, block_n).full()
    cmb_ML = spherical.Block2Full(cmb_ML, block_n).full()
    return cmb, cmb_ML


#%% get and save recovered maps 
cmb_q, cmb_ML_q = get_recoveredMap(randn_num_q, map_type='Q')
cmb_u, cmb_ML_u = get_recoveredMap(randn_num_u, map_type='U')

cmb_qu = np.array([cmb_q, cmb_u])
cmb_ML_qu = np.array([cmb_ML_q, cmb_ML_u])


#%% read & plot cls
savefig = True
savefig = False


cmb_plt = plotter.PlotCMB_EEBB(cmb_qu, cmb_ML_qu, map_n=map_n, nside=nside, block_n=block_n, 
                               randn_marker=randn_marker)

cmb_plt.mask_manual() #1/12 of full map


cmb_plt.plot_all(savefig=savefig, root='figures_blocks', fwhm=fwhm, 
                 aposize=5, nlb=5, bin_residual=True)

plt.show()

