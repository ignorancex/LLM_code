#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 11:14:33 2023
@author: mik
"""
import numpy as np

import os
import shutil

import matplotlib.pyplot as plt
###############################################################################
###############################################################################

# OV5647 sensor parameters
width = 2592  # Number of pixels in width
height = 1944 # Number of pixels in height
pixel_size = 1.4  # Pixel size in micrometers

# Constants
pair_eh = 3.6  # Energy to create an electron-hole pair in silicon (eV)
e_h_pair_energy = 3.6e-3  # Energy to create an electron-hole pair in silicon (keV)
emax = 4300  # Maximum well capacity
emin = 5     # Minimum energy threshold
k = 0.0062  # Micrometers/keV

# Simulation parameters
nframes = 1000  # Number of frames
alpha = 1.75  # Exponent in energy-range relationship
#zff = 0.25  # Field-free region thickness in micrometers
#nsigm = 0.25  # Number of sigma for Gaussian spread
z_count = 1  # Count of Z levels

rgb_data_s = np.zeros((height, width))
rgb_data_s2 = np.zeros((height, width))

ag = '8'
tim ='_0.5s'

source = 'Sr90'
source = 'Cs137'

source_bg = 'backgnd'
fecha_bg = '2023_Oct_22_00h'
#fecha_bg ='2023_Oct_21_09h'
#fecha_bg ='2023_Oct_20_19h'

nfrm = nframes

if(source == 'Sr90'):
    winback='254'
    fecha = '2023_Oct_24_23h'
    ev = '1514'
    evt = winback+'_ev'+ev
    nfrm =1000


if(source == 'Cs137'):
    winback='635'
    fecha = '2023_Oct_23_23h'
    ev = '3811'
    evt = winback+'_ev'+ev
    nfrm =1000

'''
source_bg = 'backgnd'
fecha_bg = '2022_Nov_10'
nfrm =100

if(source == 'Sr90'):
    winback='254'
    fecha = '2021_Nov_09'
    ev = '1587'
    evt = winback+'_ev'+ev
    nfrm =100

if(source == 'Cs137'):
    winback='635'
    fecha = '2021_Nov_23'
    ev = '3991'
    evt = winback+'_ev'+ev
    nfrm =100
'''

satu =''

# Conditions for energy deposition
condit_edep = ''  # '', '_dif0', '_zero' or other conditions can be added here
#condit_edep = '_dif0'

level_z = list(range(z_count))
#level_z = list(range(0,6,2))
#level_z = [0,2,6,12]
#level_z = [0]

strE= 'eh_'+str(pair_eh)+'eV'
print(strE)

nsigm_self = 0
adc_cut = 0
size_thresh =  0
porc_max = 0

log_y = True

bg_filt = ''

ssd_obs = '/home/mbonnett/mik/data_2023/'
ssd = '/home/mbonnett/mik/'
path_main = 'dat_sim_2024/'
path_main = 'dat_sim_2024/g4_'+source+'_2024/'


gauss_dist = 'Dg2'

sim_n = ''
for sim_n in range(10):
    sim_n ='_s'+str(sim_n)
    #sim_n ='_0'
    # Number of sigma for Gaussian spread
    for nsigm in [0, ]:
        for k in [0]:
            k=np.round(k,3)
            k_str = str(int(1000*np.round(k,3)))
            #i+=1; print(i, np.round(k,3))
            #for alpha in np.arange(1.73, 1.775, 0.005):
            for alpha in [0]:
                alpha=np.round(alpha,3)
                a_str = str(int(1000*np.round(alpha,3)))
                #print(np.round(alpha,3))
                # Field-free region thickness in micrometers
                #for zff in np.arange(0.2, 1.1, 0.2):
                for zff in [0]:
                # Field-free region thickness in micrometers
                    zff = np.round(zff,3)
                   
                    if nsigm > 0 :
                        difu_gauss = f"_{gauss_dist}_{k}k_{alpha}a_{nsigm}sig_{zff}zff"
                    if nsigm == 0 :
                        difu_gauss = f"_{gauss_dist}_{k}k_{alpha}a_{zff}zff"

                    for type_dat in ['sim', 'sim_bg' ] :

                        if(type_dat == 'obs'or type_dat == 'bg'):
                            n_mean = 1
                            navrg = 0
                            nsigm_bg = 5
                        #
                        if(type_dat == 'sim'):
                            n_mean = 1
                            navrg = -1
                            nsigm_bg =0

                        if(type_dat == 'sim_bg' ):
                            n_mean = 1
                            navrg = 0
                            nsigm_bg = 5


                        path_dir_bg = ssd_obs+'data_obs_2023/'
                        path_dir_bg = path_dir_bg + 'data_'+fecha_bg+'_'+ag+'ag'+'_'+source_bg+'_'+str(1)+'level'+'/'
                        dirname_bg = 'data_'+source_bg+'_f'+str(nfrm)+'_iso0_br50_ag'+ag+'_ad1_'+fecha_bg
                        MeanFrame_bg  = np.load(path_dir_bg + dirname_bg+'/' +'RGB_mean_'+'f'+str(nframes)+'_'+'z%03d' % 0+'.npz')
                        MeanFrame_bg  = np.float64(MeanFrame_bg.f.arr_0)
                        SigmaFrame_bg = np.load(path_dir_bg + dirname_bg+'/' +'RGB_sigm_'+'f'+str(nframes)+'_'+'z%03d' % 0+'.npz')
                        SigmaFrame_bg = np.float64(SigmaFrame_bg.f.arr_0)


                        if(adc_cut>0):
                            cut_adc = '_cutg_'+str(adc_cut)+'ADC'
                        if(adc_cut == 0 ):
                            cut_adc = ''

                        if(navrg>=0):
                            avrg_less =  n_mean*MeanFrame_bg + navrg*SigmaFrame_bg #filter average less
                            avrg_cut = '_less_'+str(n_mean)+'avrg_bg_'+str(navrg)+'sbg'
                        if(navrg < 0 ):
                            avrg_less = 0 #filter threshold
                            avrg_cut = ''

                        if(nsigm_bg>0):
                            tshld_sig_bg = nsigm_bg*SigmaFrame_bg #filter threshold
                            sig_bg_thresh = '_tshld_'+str(nsigm_bg)+'sig_bg'
                        if(nsigm_bg == 0 ):
                            tshld_sig_bg = 0 #filter threshold
                            sig_bg_thresh = ''


                        if(size_thresh>0):
                            cut_clst_size = '_clst'+ str(size_thresh)
                            print(cut_clst_size)
                        if(size_thresh == 0 ):
                            cut_clst_size = ''

                        if(porc_max>0):
                            cut_max_clst_porc = '_max'+ str(porc_max)
                            print(cut_max_clst_porc)
                        if(porc_max == 0 ):
                            cut_max_clst_porc = ''


                        if(nsigm_self>0 and nsigm_bg == 0 and navrg < 0):
                            sigthresh = '_tshld_'+str(nsigm_self)+'sig_self'
                        if(nsigm_self == 0 ):
                            sigthresh = ''


                        if(type_dat == 'bg' or type_dat == 'all'):
                            #path_dir_bg = 'C:/data_2023/data_obs_2023/'
                            path_dir_bg = ssd_obs+'data_obs_2023/'
                            path_dir_bg = path_dir_bg + 'data_'+fecha_bg+'_'+ag+'ag'+'_'+source_bg+'_'+str(z_count)+'level'+'/'
                            dirname_bg = 'data_'+source_bg+'_f'+str(nfrm)+'_iso0_br50_ag'+ag+'_ad1_'+fecha_bg

                            MeanFrame_bg  = np.load(path_dir_bg + dirname_bg+'/' +'RGB_mean_'+'f'+str(nframes)+'_'+'z%03d' % 0+'.npz')
                            MeanFrame_bg  = np.float64(MeanFrame_bg.f.arr_0)
                            SigmaFrame_bg = np.load(path_dir_bg + dirname_bg+'/' +'RGB_sigm_'+'f'+str(nframes)+'_'+'z%03d' % 0+'.npz')
                            SigmaFrame_bg = np.float64(SigmaFrame_bg.f.arr_0)

                            nameframe_bg = 'RGB_z' +str(0).zfill(3)+ '_f' #filename format
                            dir_save_dat_bg = path_dir_bg + source_bg +'/'+ 'data_'+source_bg+'_f'+str(nframes)+'_iso0_br50_ag'+ag+'_ad1_'+fecha_bg

                            try:
                                os.makedirs(dir_save_dat_bg + cut_adc + avrg_cut + sig_bg_thresh + sigthresh )
                            except FileExistsError:
                                pass

                        if(type_dat == 'obs' or type_dat == 'all'):
                            #path_dir_obs = 'C:/data_2023/data_obs_2023/'
                            path_dir_obs = ssd+'data_obs_2023/'
                            path_dir_obs = path_dir_obs + 'data_'+fecha+'_'+ag+'ag'+'_'+source+'_'+str(z_count)+'level'+'/'
                            dirname_obs = 'data_'+source+'_f'+str(nfrm)+'_iso0_br50_ag'+ag+'_ad1_'+fecha
                            dir_save_dat_obs = path_dir_obs + source +'/'+ 'data_'+source+'_f'+str(nframes)+'_iso0_br50_ag'+ag+'_ad1_'+fecha
                            try:
                                os.makedirs(dir_save_dat_obs + cut_adc + avrg_cut + sig_bg_thresh + sigthresh)
                            except FileExistsError:
                                pass

                        if(type_dat == 'sim' or type_dat == 'all'):
                            #path_dir_sim = 'C:/data_2023/data_sim_2024/'
                            path_dir_sim = ssd+path_main
                            dirname_sim = 'data_'+source+'_pixZ_2um_'+str(z_count )+'l_'+evt+sim_n+'_'+satu+strE  \
                                            +condit_edep+difu_gauss+cut_clst_size+cut_max_clst_porc

                            dir_save_dat_sim = path_dir_sim + source +'/'+ 'sim_'+source+'_f'+str(nframes)+'_'+str(z_count) \
                                                +'l_edpxl_w'+evt+sim_n+condit_edep \
                                                +difu_gauss+cut_clst_size+cut_max_clst_porc
                            try:
                                os.makedirs(dir_save_dat_sim + cut_adc + avrg_cut + sig_bg_thresh + sigthresh)
                            except FileExistsError:
                                pass

                        if(type_dat == 'sim_bg' or type_dat == 'all'):
                            #path_dir_sim_bg = '/media/hep_group01/761C6DEA1C6DA5B91/mik_2023/data_sim_2024/'
                            #path_dir_sim_bg = 'C:/data_2023/data_sim_2024/'
                            path_dir_sim_bg = ssd+path_main
                            dirname_sim_bg = 'data_'+source+'_pixZ_2um_'+str(z_count )+'l_'+evt+sim_n+'_'+satu+strE  \
                                                +condit_edep+difu_gauss +'_ag'+ag+'_bg'+cut_clst_size+cut_max_clst_porc

                            dir_save_dat_sim_bg = path_dir_sim_bg + source +'/'+ 'sim_'+source+'_f'+str(nframes)+'_'+str(z_count) \
                                                    +'l_edpxl_w'+evt+sim_n+condit_edep \
                                                    +difu_gauss+'_ag'+ag+'_bg'+cut_clst_size+cut_max_clst_porc
                            try:
                                os.makedirs(dir_save_dat_sim_bg + cut_adc + avrg_cut + sig_bg_thresh + sigthresh)
                            except FileExistsError:
                                pass


                        #dirfile = '/media/hep_group01/761C6DEA1C6DA5B91/mik_2023/data_sim_2024/'+'plot_sig_cut/'+source
                        if(type_dat == 'obs'):dirfile = ssd+'plot_avrg_sig_cut/'+fecha+'_'+source+'_'+str(z_count)+'l'
                        if(type_dat == 'bg'):dirfile = ssd+'plot_avrg_sig_cut/'+fecha_bg+'_'+source_bg+'_'+str(z_count)+'l'
                        if(type_dat == 'sim'):dirfile = ssd+path_main+'plot_avrg_sig_cut/'+source+ '_w'+evt+'_'+str(z_count)+'l'
                        if(type_dat == 'sim_bg'):dirfile = ssd+path_main+'plot_avrg_sig_cut/'+source+ '_w'+evt+'_'+str(z_count)+'l_bg'
                        if(type_dat == 'all'):dirfile = ssd+'plot_avrg_sig_cut/'+fecha+'_'+source+'_all_'+str(z_count)+'l'


                        try:
                            os.makedirs(dirfile)
                        except FileExistsError:
                            pass

                        font_siz = 28

                        #for z in range(z_count-1, z_count):
                        for z in level_z:

                            if(type_dat == 'obs' or type_dat == 'all'):
                                nameframe_obs = 'RGB_z' +str(2*z).zfill(3)+ '_f' #filename format
                                MeanFrame_obs  = np.load(path_dir_obs + dirname_obs+'/' +'RGB_mean_'+'f'+str(nframes)+'_'+'z%03d' % (2*z)+'.npz')
                                MeanFrame_obs  = np.float64(MeanFrame_obs.f.arr_0)
                                SigmaFrame_obs = np.load(path_dir_obs + dirname_obs+'/' +'RGB_sigm_'+'f'+str(nframes)+'_'+'z%03d' % (2*z)+'.npz')
                                SigmaFrame_obs = np.float64(SigmaFrame_obs.f.arr_0)

                                data_cnt_obs = np.zeros((2, 1024),dtype=np.int64)

                                if(nsigm_self>0):
                                    threshold_obs = abs(MeanFrame_obs) + nsigm_self*SigmaFrame_obs #filter threshold
                                    #sigthresh = '_tshld_'+str(nsigm_self)+'sig'
                                if(nsigm_self == 0 ):
                                    threshold_obs = 0 #filter threshold
                                    #sigthresh = '_tshld_0'

                            if(type_dat == 'sim' or type_dat == 'all'):
                                nameframe_sim = 'RGB_eh_3.6eV'+bg_filt+'_z' +str(2*z).zfill(3)+ '_f' #filename format
                                if(size_thresh == 0 ):
                                    MeanFrame_sim  = np.load(path_dir_sim + dirname_sim+'/' +'RGB_eh_3.6eV'+bg_filt+'_mean_'+'f'+str(nframes)+'_'+'z%03d' % (2*z)+'.npz')
                                    MeanFrame_sim  = np.float64(MeanFrame_sim.f.arr_0)
                                    SigmaFrame_sim = np.load(path_dir_sim + dirname_sim+'/' +'RGB_eh_3.6eV'+bg_filt+'_sigm_'+'f'+str(nframes)+'_'+'z%03d' % (2*z)+'.npz')
                                    SigmaFrame_sim = np.float64(SigmaFrame_sim.f.arr_0)

                                data_cnt_sim = np.zeros((2, 1024),dtype=np.int64)

                                if(nsigm_self>0):
                                    threshold_sim = abs(MeanFrame_sim) + nsigm_self*SigmaFrame_sim #filter threshold
                                    #sigthresh = '_tshld_'+str(nsigm_self)+'sig'
                                if(nsigm_self == 0 ):
                                    threshold_sim = 0 #filter threshold
                                    #sigthresh = '_tshld_0'

                            if(type_dat == 'sim_bg' or type_dat == 'all'):
                                nameframe_sim_bg = 'RGB_noise_eh_3.6eV'+bg_filt+'_z' +str(2*z).zfill(3)+ '_f' #filename format
                                if(size_thresh == 0 ):
                                    MeanFrame_sim_bg  = np.load(path_dir_sim_bg + dirname_sim_bg+'/' +'RGB_noise_eh_3.6eV'+bg_filt+'_mean_'+'f'+str(nframes)+'_'+'z%03d' % (2*z)+'.npz')
                                    MeanFrame_sim_bg  = np.float64(MeanFrame_sim_bg.f.arr_0)
                                    SigmaFrame_sim_bg = np.load(path_dir_sim_bg + dirname_sim_bg+'/' +'RGB_noise_eh_3.6eV'+bg_filt+'_sigm_'+'f'+str(nframes)+'_'+'z%03d' % (2*z)+'.npz')
                                    SigmaFrame_sim_bg = np.float64(SigmaFrame_sim_bg.f.arr_0)

                                data_cnt_sim_bg = np.zeros((2, 1024),dtype=np.int64)

                                if(nsigm_self>0):
                                    threshold_sim_bg = abs(MeanFrame_sim_bg) + nsigm_self*SigmaFrame_sim_bg #filter threshold
                                    #sigthresh = '_tshld_'+str(nsigm_self)+'sig'
                                if(nsigm_self == 0 ):
                                    threshold_sim_bg = 0 #filter threshold
                                    #sigthresh = '_tshld_0'

                               # plt.show()

                            if(z == 0 and (type_dat == 'bg' or type_dat == 'all')):
                                data_cnt_bg = np.zeros((2, 1024),dtype=np.int64)

                            for n in range(nframes):

                                if(type_dat == 'obs' or type_dat == 'all'):
                                    tmpname_obs = path_dir_obs + dirname_obs + '/' + nameframe_obs +str(n).zfill(3) + '.npz'
                                    rgb_data_obs = np.load(tmpname_obs)
                                    rgb_data_obs = np.float64(rgb_data_obs.f.arr_0)

                                if(type_dat == 'sim' or type_dat == 'all'):
                                    tmpname_sim = path_dir_sim + dirname_sim + '/' + nameframe_sim +str(n).zfill(3) + '.npz'
                                    rgb_data_sim = np.load(tmpname_sim)
                                    rgb_data_sim = np.float64(rgb_data_sim.f.arr_0)

                                if(type_dat == 'sim_bg' or type_dat == 'all'):
                                    tmpname_sim_bg = path_dir_sim_bg + dirname_sim_bg + '/' + nameframe_sim_bg +str(n).zfill(3) + '.npz'
                                    rgb_data_sim_bg = np.load(tmpname_sim_bg)
                                    rgb_data_sim_bg = np.float64(rgb_data_sim_bg.f.arr_0)

                                if(z == 0 and (type_dat == 'bg' or type_dat == 'all')):
                                    tmpname_bg = path_dir_bg + dirname_bg + '/' + nameframe_bg +str(n).zfill(3) + '.npz'
                                    rgb_data_bg = np.load(tmpname_bg)
                                    rgb_data_bg = np.float64(rgb_data_bg.f.arr_0)

                                if(adc_cut>0):
                                    if(type_dat == 'obs' or type_dat == 'all'):
                                        rgb_data_obs[rgb_data_obs<=adc_cut]=0
                                    if(type_dat == 'sim' or type_dat == 'all'):
                                        rgb_data_sim[rgb_data_sim<=adc_cut]=0
                                    if(type_dat == 'sim_bg' or type_dat == 'all'):
                                        rgb_data_sim_bg[rgb_data_sim_bg<=adc_cut]=0
                                    if(type_dat == 'bg' or type_dat == 'all'):
                                        rgb_data_bg[rgb_data_bg<=adc_cut]=0


                                if(navrg < 0 ):
                                    if(type_dat == 'bg' or type_dat == 'all'):
                                        rgb_cut_avg_bg = rgb_data_bg
                                    if(type_dat == 'obs' or type_dat == 'all'):
                                        rgb_cut_avg_obs = rgb_data_obs
                                    if(type_dat == 'sim' or type_dat == 'all'):
                                        rgb_cut_avg_sim = rgb_data_sim
                                    if(type_dat == 'sim_bg' or type_dat == 'all'):
                                        rgb_cut_avg_sim_bg = rgb_data_sim_bg


                                if(navrg>=0):
                                    if(type_dat == 'bg' or type_dat == 'all'):
                                        rgb_cut_avg_bg = np.round(rgb_data_bg - avrg_less) #filtering
                                        rgb_cut_avg_bg[rgb_cut_avg_bg<0]=0

                                    if(type_dat == 'obs' or type_dat == 'all'):
                                        rgb_cut_avg_obs = np.round(rgb_data_obs - avrg_less) #filtering
                                        rgb_cut_avg_obs[rgb_cut_avg_obs<0]=0

                                    if(type_dat == 'sim' or type_dat == 'all'):
                                        rgb_cut_avg_sim = np.round(rgb_data_sim - avrg_less) #filtering
                                        rgb_cut_avg_sim[rgb_cut_avg_sim<0]=0

                                    if(type_dat == 'sim_bg' or type_dat == 'all'):
                                        rgb_cut_avg_sim_bg = np.round(rgb_data_sim_bg - avrg_less) #filtering
                                        rgb_cut_avg_sim_bg[rgb_cut_avg_sim_bg<0]=0


                                if(nsigm_bg == 0 ):
                                    if(type_dat == 'bg' or type_dat == 'all'):
                                        rgb_sig_tshl_bg = rgb_cut_avg_bg
                                    if(type_dat == 'obs' or type_dat == 'all'):
                                        rgb_sig_tshl_obs = rgb_cut_avg_obs
                                    if(type_dat == 'sim' or type_dat == 'all'):
                                        rgb_sig_tshl_sim = rgb_cut_avg_sim
                                    if(type_dat == 'sim_bg' or type_dat == 'all'):
                                        rgb_sig_tshl_sim_bg = rgb_cut_avg_sim_bg


                                if(nsigm_bg>0):
                                    if(type_dat == 'bg' or type_dat == 'all'):
                                        ev_cut_bg = (rgb_cut_avg_bg) > tshld_sig_bg #filtering
                                        #if(nsigm_bg>0):print(threshold.max())
                                        rgb_sig_tshl_bg = np.round(rgb_cut_avg_bg)*ev_cut_bg
                                    if(type_dat == 'obs' or type_dat == 'all'):
                                        ev_cut_obs = (rgb_cut_avg_obs) > tshld_sig_bg #filtering
                                        #if(nsigm_bg>0):print('threshold = ', threshold.max())
                                        rgb_sig_tshl_obs = np.round(rgb_cut_avg_obs)*ev_cut_obs
                                    if(type_dat == 'sim' or type_dat == 'all'):
                                        ev_cut_sim = (rgb_cut_avg_sim) > tshld_sig_bg #filtering
                                        rgb_sig_tshl_sim = np.round(rgb_cut_avg_sim)*ev_cut_sim
                                    if(type_dat == 'sim_bg' or type_dat == 'all'):
                                        ev_cut_sim_bg = (rgb_cut_avg_sim_bg) > tshld_sig_bg #filtering
                                        rgb_sig_tshl_sim_bg = np.round(rgb_cut_avg_sim_bg)*ev_cut_sim_bg


                                if(nsigm_self>0 and nsigm_bg == 0 and navrg < 0):
                                    if(type_dat == 'obs' or type_dat == 'all'):
                                        ev_matx_obs = abs(rgb_data_obs) > threshold_obs #filtering
                                        #if(nsigm_self>0):print('threshold = ', threshold.max())
                                        rgb_sig_tshl_obs = abs(rgb_data_obs)*ev_matx_obs
                                    if(type_dat == 'sim' or type_dat == 'all'):
                                        ev_matx_sim = abs(rgb_data_sim) > threshold_sim #filtering
                                        #if(nsigm_self>0):print(threshold.max())
                                        rgb_sig_tshl_sim = abs(rgb_data_sim)*ev_matx_sim
                                    if(type_dat == 'sim_bg' or type_dat == 'all'):
                                        ev_matx_sim_bg = abs(rgb_data_sim_bg) > threshold_sim_bg #filtering
                                        #if(nsigm_self>0):print(threshold.max())
                                        rgb_sig_tshl_sim_bg = abs(rgb_data_sim_bg)*ev_matx_sim_bg
                                    if(type_dat == 'bg' or type_dat == 'all'):
                                        ev_matx_bg = abs(rgb_data_bg) > tshld_sig_bg #filtering
                                        #if(nsigm_self>0):print(threshold.max())
                                        rgb_sig_tshl_bg = abs(rgb_data_bg)*ev_matx_bg

                                if(type_dat == 'obs' or type_dat == 'all'):
                                    dat_sig_cut_tshl_obs = rgb_sig_tshl_obs
                                if(type_dat == 'sim' or type_dat == 'all'):
                                    dat_sig_cut_tshl_sim = rgb_sig_tshl_sim
                                if(type_dat == 'sim_bg' or type_dat == 'all'):
                                    dat_sig_cut_tshl_sim_bg = rgb_sig_tshl_sim_bg
                                if(type_dat == 'bg' or type_dat == 'all'):
                                    dat_sig_cut_tshl_bg = rgb_sig_tshl_bg

                                if(type_dat == 'obs' or type_dat == 'all'):
                                    np.savez_compressed(dir_save_dat_obs + cut_adc + avrg_cut + sig_bg_thresh + sigthresh+ '/' +'RGB'+'_z%03d' % (2*z) +'_f%03d' % n, np.int16(dat_sig_cut_tshl_obs))
                                    print('RGB'+ cut_adc + avrg_cut + sig_bg_thresh + sigthresh, n, dat_sig_cut_tshl_obs.min(), dat_sig_cut_tshl_obs.max() )

                                if(type_dat == 'sim' or type_dat == 'all'):
                                    np.savez_compressed(dir_save_dat_sim + cut_adc + avrg_cut + sig_bg_thresh + sigthresh+ '/' +'RGB_eh_3.6eV'+bg_filt+'_z%03d' % (2*z) +'_f%03d' % n, np.int16(dat_sig_cut_tshl_sim))
                                    print('RGB_eh_3.6eV'+bg_filt+ cut_adc + avrg_cut + sig_bg_thresh + sigthresh, n, dat_sig_cut_tshl_sim.min(), dat_sig_cut_tshl_sim.max() )

                                if(type_dat == 'sim_bg' or type_dat == 'all'):
                                    np.savez_compressed(dir_save_dat_sim_bg + cut_adc + avrg_cut + sig_bg_thresh + sigthresh+ '/' +'RGB_noise_eh_3.6eV'+bg_filt+'_z%03d' % (2*z) +'_f%03d' % n, np.int16(dat_sig_cut_tshl_sim_bg))
                                    print('RGB_noise_eh_3.6eV'+bg_filt+ cut_adc + avrg_cut + sig_bg_thresh + sigthresh, n, dat_sig_cut_tshl_sim_bg.min(), dat_sig_cut_tshl_sim_bg.max() )

                                if(z == 0 and (type_dat == 'bg' or type_dat == 'all')):
                                    np.savez_compressed(dir_save_dat_bg+cut_adc + avrg_cut + sig_bg_thresh + sigthresh+ '/' +'RGB'+'_z%03d' % (2*z) +'_f%03d' % n, np.int16(dat_sig_cut_tshl_bg))
                                    print('RGB'+ cut_adc + avrg_cut + sig_bg_thresh + sigthresh, n, dat_sig_cut_tshl_bg.min(), dat_sig_cut_tshl_bg.max() )

                                ########################################################################
                                if(type_dat == 'obs' or type_dat == 'all'):
                                    #adc_n_obs, n_count_obs = np.unique(dat_sig_cut_tshl_obs, return_counts=True)
                                    adc_n_obs = np.unique(dat_sig_cut_tshl_obs, return_counts=True)

                                    for j in range(adc_n_obs[0].size):
                                        data_cnt_obs[0,int(adc_n_obs[0][j])] = (adc_n_obs[0][j])
                                        data_cnt_obs[1,int(adc_n_obs[0][j])] = data_cnt_obs[1,int(adc_n_obs[0][j])]+(adc_n_obs[1][j])

                                    data_cnt_obs[:1].max()
                                    data_cnt_obs[1:].max()
                                    print(n, data_cnt_obs[:1].max(),data_cnt_obs[1:].max())

                                ########################################################################
                                if(type_dat == 'sim' or type_dat == 'all'):
                                    #adc_n_sim, n_count_sim = np.unique(dat_sig_cut_tshl_sim, return_counts=True)
                                    adc_n_sim = np.unique(dat_sig_cut_tshl_sim, return_counts=True)

                                    for j in range(adc_n_sim[0].size):
                                        data_cnt_sim[0,int(adc_n_sim[0][j])] = (adc_n_sim[0][j])
                                        data_cnt_sim[1,int(adc_n_sim[0][j])] = data_cnt_sim[1,int(adc_n_sim[0][j])]+(adc_n_sim[1][j])

                                    data_cnt_sim[:1].max()
                                    data_cnt_sim[1:].max()
                                    print(n, data_cnt_sim[:1].max(),data_cnt_sim[1:].max())

                                ########################################################################
                                if(type_dat == 'sim_bg' or type_dat == 'all'):
                                    #adc_n_sim_bg, n_count_sim_bg = np.unique(dat_sig_cut_tshl_sim_bg, return_counts=True)
                                    adc_n_sim_bg = np.unique(dat_sig_cut_tshl_sim_bg, return_counts=True)

                                    for j in range(adc_n_sim_bg[0].size):
                                        data_cnt_sim_bg[0,int(adc_n_sim_bg[0][j])] = (adc_n_sim_bg[0][j])
                                        data_cnt_sim_bg[1,int(adc_n_sim_bg[0][j])] = data_cnt_sim_bg[1,int(adc_n_sim_bg[0][j])]+(adc_n_sim_bg[1][j])

                                    data_cnt_sim_bg[:1].max()
                                    data_cnt_sim_bg[1:].max()
                                    print(n, data_cnt_sim_bg[:1].max(),data_cnt_sim_bg[1:].max())

                                ########################################################################
                                if(z == 0 and (type_dat == 'bg' or type_dat == 'all')):
                                    #adc_n_bg, n_count_bg = np.unique(dat_sig_cut_tshl_bg, return_counts=True)
                                    adc_n_bg = np.unique(dat_sig_cut_tshl_bg, return_counts=True)

                                    for j in range(adc_n_bg[0].size):
                                        data_cnt_bg[0,int(adc_n_bg[0][j])] = (adc_n_bg[0][j])
                                        data_cnt_bg[1,int(adc_n_bg[0][j])] = data_cnt_bg[1,int(adc_n_bg[0][j])]+(adc_n_bg[1][j])

                                    data_cnt_bg[:1].max()
                                    data_cnt_bg[1:].max()
                                    print(n, data_cnt_bg[:1].max(),data_cnt_bg[1:].max())

                                ########################################################################

                            #del dat_sig_cut_tshl_obs, dat_sig_cut_tshl_sim, dat_sig_cut_tshl_bg

                            if(type_dat == 'obs' or type_dat == 'all'):
                                del dat_sig_cut_tshl_obs
                                dirsave_plt_obs ='adc_count_'+ source+'_f'+str(nframes)+'_iso0_br50_ag'+ag+'_ad1'+ cut_adc + avrg_cut + sig_bg_thresh + sigthresh
                                #source+'_frame100_iso0_br50_ag'+ag+'_ad1'
                                try:
                                    os.makedirs(path_dir_obs+source +'/'+ dirsave_plt_obs)
                                except FileExistsError:
                                    pass
                                #np.savez_compressed(path_dir_obs+dirsave_plt_obs+'/'+ source+'_frame100_iso0_br50_ag'+ag+'_ad1'+'_z'+str(2*z).zfill(3)+'.npz', data_cnt_obs)
                                np.savez_compressed(path_dir_obs+source +'/'+ dirsave_plt_obs+'/'+ source+ cut_adc + avrg_cut + sig_bg_thresh + sigthresh+cut_clst_size+'_z'+str(2*z).zfill(3)+'.npz', data_cnt_obs)
                                print(path_dir_obs+dirsave_plt_obs+'/'+ source+ cut_adc + avrg_cut + sig_bg_thresh + sigthresh+cut_clst_size+'_z'+str(2*z).zfill(3)+'.npz')

                            if(type_dat == 'sim' or type_dat == 'all'):
                                del dat_sig_cut_tshl_sim
                                dirsave_plt_sim ='adc_count_'+ 'sim_'+source+'_'+str(nframes)+'f_'+evt+sim_n +'_'+satu+strE  \
                                                    +condit_edep+difu_gauss + cut_adc + avrg_cut + sig_bg_thresh + sigthresh
                                #'sim_'+source+'_'+str(nframes)+'f_'+evt

                                try:
                                    os.makedirs(path_dir_sim+source +'/'+ dirsave_plt_sim)
                                except FileExistsError:
                                    pass
                                #np.savez_compressed(path_dir_sim+dirsave_plt_sim+'/'+ 'sim_'+source+'_'+str(nframes)+'f_'+evt+'_z'+str(2*z).zfill(3)+'.npz', data_cnt_sim)
                                np.savez_compressed(path_dir_sim+source +'/'+ dirsave_plt_sim+'/'+ source+'_ev'+ev +difu_gauss+ cut_adc + avrg_cut + sig_bg_thresh + sigthresh+cut_clst_size+'_z'+str(2*z).zfill(3)+'.npz', data_cnt_sim)
                                print(path_dir_sim+dirsave_plt_sim+'/'+ source+'_ev'+ev+ cut_adc + avrg_cut + sig_bg_thresh + sigthresh+cut_clst_size+'_z'+str(2*z).zfill(3)+'.npz')


                            if(type_dat == 'sim_bg' or type_dat == 'all'):
                                del dat_sig_cut_tshl_sim_bg
                                dirsave_plt_sim_bg ='adc_count_'+ 'sim_bg_'+source+'_'+str(nframes)+'f_'+evt+sim_n+'_'+satu+strE  \
                                                    +condit_edep+difu_gauss+ cut_adc + avrg_cut + sig_bg_thresh + sigthresh
                                #'sim_bg_'+source+'_'+str(nframes)+'f_'+evt

                                try:
                                    os.makedirs(path_dir_sim_bg+source +'/'+ dirsave_plt_sim_bg)
                                except FileExistsError:
                                    pass
                                #np.savez_compressed(path_dir_sim_bg+dirsave_plt_sim_bg+'/'+ 'sim_bg_'+source+'_'+str(nframes)+'f_'+evt+'_z'+str(2*z).zfill(3)+'.npz', data_cnt_sim_bg)
                                np.savez_compressed(path_dir_sim_bg+source +'/'+ dirsave_plt_sim_bg+'/'+ source+'_ev'+ev +difu_gauss+ cut_adc + avrg_cut + sig_bg_thresh + sigthresh+cut_clst_size+'_z'+str(2*z).zfill(3)+'.npz', data_cnt_sim_bg)
                                print(path_dir_sim_bg+dirsave_plt_sim_bg+'/'+ source+'_ev'+ev+ cut_adc + avrg_cut + sig_bg_thresh + sigthresh+cut_clst_size+'_z'+str(2*z).zfill(3)+'.npz')


                            if(z == 0 and (type_dat == 'bg' or type_dat == 'all')):
                                del dat_sig_cut_tshl_bg
                                dirsave_plt_bg ='adc_count_'+ source_bg+'_f'+str(nframes)+'_iso0_br50_ag'+ag+'_ad1'+ cut_adc + avrg_cut + sig_bg_thresh + sigthresh
                                #source_bg+'_frame100_iso0_br50_ag'+ag+'_ad1'
                                try:
                                    os.makedirs(path_dir_bg+source_bg +'/'+ dirsave_plt_bg)
                                except FileExistsError:
                                    pass
                                np.savez_compressed(path_dir_bg+source_bg +'/'+ dirsave_plt_bg+'/'+ source_bg+ cut_adc + avrg_cut + sig_bg_thresh + sigthresh+cut_clst_size+'_z'+str(2*z).zfill(3)+'.npz', data_cnt_bg)
                                print(path_dir_bg+dirsave_plt_bg+'/'+ source_bg+ cut_adc + avrg_cut + sig_bg_thresh + sigthresh+cut_clst_size+'_z'+str(2*z).zfill(3)+'.npz')


                            ############################################################################
                            if(type_dat == 'obs'):
                                a=1*data_cnt_obs[0]
                                a[a>=a.max()]=0
                                if(data_cnt_obs[0].max()-a.max()<=5):lim_xx = data_cnt_obs[0].max()
                                else:lim_xx = a.max()

                            if(type_dat == 'sim'):
                                a=1*data_cnt_sim[0]
                                a[a>=a.max()]=0
                                if(data_cnt_sim[0].max()-a.max()<=5):lim_xx = data_cnt_sim[0].max()
                                else:lim_xx = a.max()

                            if(type_dat == 'sim_bg'):
                                a=1*data_cnt_sim_bg[0]
                                a[a>=a.max()]=0
                                if(data_cnt_sim_bg[0].max()-a.max()<=5):lim_xx = data_cnt_sim_bg[0].max()
                                else:lim_xx = a.max()

                            if(type_dat == 'bg'):
                                a=1*data_cnt_bg[0]
                                a[a>=a.max()]=0
                                if(data_cnt_bg[0].max()-a.max()<=5):lim_xx = data_cnt_bg[0].max()
                                else:lim_xx = a.max()

                            if(type_dat == 'all'):
                                a=1*data_cnt_obs[0]
                                a[a>=a.max()]=0
                                if(data_cnt_obs[0].max()-a.max()<=5):
                                    lim_xa = data_cnt_obs[0].max()
                                else:lim_xa = a.max()

                                b=1*data_cnt_sim_bg[0]
                                b[b>=b.max()]=0
                                if(data_cnt_sim_bg[0].max()-b.max()<=5):
                                    lim_xb = data_cnt_sim_bg[0].max()
                                else:lim_xb = b.max()
                                lim_xx = np.max(np.array((lim_xa, lim_xb )))
                                lim_xx = lim_xx + int(0.05*lim_xx +1)
                                print(lim_xa, lim_xb, lim_xx )

                            bin_hist = 0

                            max_adc = lim_xx+1
                            min_adc = 0

                            if(type_dat == 'obs' or type_dat == 'all'):
                                d_cnt_obs = data_cnt_obs[:,min_adc:max_adc]
                                nbins_adc = round(d_cnt_obs[0].max() - min_adc + 1)

                            if(type_dat == 'sim' or type_dat == 'all'):
                                d_cnt_sim = data_cnt_sim[:,min_adc:max_adc]
                                nbins_adc = round(d_cnt_sim[0].max() - min_adc + 1)

                            if(type_dat == 'sim_bg' or type_dat == 'all'):
                                d_cnt_sim_bg = data_cnt_sim_bg[:,min_adc:max_adc]
                                nbins_adc = round(d_cnt_sim_bg[0].max() - min_adc + 1)

                            if(z == 0 and (type_dat == 'bg' or type_dat == 'all')):
                                d_cnt_bg = data_cnt_bg[:,min_adc:max_adc]
                                nbins_adc = round(d_cnt_bg[0].max() - min_adc + 1)

                            if(z > 0 and (type_dat == 'bg' or type_dat == 'all')):
                                d_cnt_bg = np.load(path_dir_bg+source_bg +'/'+ dirsave_plt_bg+'/'+ source_bg+ cut_adc + avrg_cut + sig_bg_thresh + sigthresh+'_z'+str(0).zfill(3)+'.npz')
                                d_cnt_bg = np.float64(d_cnt_bg.f.arr_0)
                                nbins_adc = round(d_cnt_bg[0].max() - min_adc + 1)

                            if(type_dat == 'all'):
                                #nbins_adc = round(np.min(np.array((d_cnt_obs[0].max(), d_cnt_sim[0].max()  )))-min_adc+1)
                                nbins_adc = round(np.min(np.array((d_cnt_obs[0].max(), d_cnt_sim[0].max()  )))-min_adc+1)

                            if(bin_hist>1 and bin_hist<nbins_adc):
                                nbins_adc = np.arange(min_adc, max_adc+1, (max_adc-min_adc)/bin_hist )

                            else:
                                nbins_adc = np.arange(min_adc, max_adc+1 )

                            #hist_adc, bins_adc = np.histogram(d_cnt_obs[0], bins=nbins_adc)


                            plt.figure(figsize=(15,9.5))
                            if(type_dat == 'obs' or type_dat == 'all'):
                                plt.hist(d_cnt_obs[0], weights=d_cnt_obs[1], bins=nbins_adc, histtype='step', log=log_y, color='C3', linewidth=2.5,
                                    label= 'Data '+source+ cut_adc + avrg_cut + sig_bg_thresh + sigthresh+ '\n all_evt = '+ str(d_cnt_obs[1].sum()) )
                            if(type_dat == 'sim' or type_dat == 'all'):
                                plt.hist(d_cnt_sim[0], weights=d_cnt_sim[1], bins=nbins_adc, histtype='step', log=log_y, color='C0', linewidth=2.5,
                                    label= 'Simulation '+source+ cut_adc + avrg_cut + sig_bg_thresh + sigthresh+ '\n all_evt = '+ str(d_cnt_sim[1].sum()) )
                            if(type_dat == 'sim_bg' or type_dat == 'all'):
                                plt.hist(d_cnt_sim_bg[0], weights=d_cnt_sim_bg[1], bins=nbins_adc, histtype='step', log=log_y, color='C0', linewidth=2.5,
                                    label= 'Simulation_bg '+source+ cut_adc + avrg_cut + sig_bg_thresh + sigthresh+ '\n all_evt = '+ str(d_cnt_sim_bg[1].sum()) )
                            if(type_dat == 'bg' or type_dat == 'all'):
                                if(d_cnt_bg[1].sum()>0):
                                    plt.hist(d_cnt_bg[0], weights=d_cnt_bg[1], bins=nbins_adc, histtype='step', log=log_y, color='k', linewidth=2.5,
                                    label= 'Data '+source_bg+ cut_adc + avrg_cut + sig_bg_thresh + sigthresh + '\n all_evt = '+ str(d_cnt_bg[1].sum()) )
                            #plt.xlim(120,max_adc_obs)
                            #if(log_y==False):plt.yscale('log')
                            #plt.xlim(0,lim_xx)
                            plt.grid(True)

                            plt.xticks(size=font_siz)
                            plt.yticks(size=font_siz)
                            plt.xlabel('ADC', fontsize=font_siz)
                            plt.ylabel('ADC count in all frames', fontsize=font_siz)
                            plt.legend(fontsize=font_siz-4)
                            if(log_y==False):titulo = 'adc_count_with'+ cut_adc + avrg_cut + sig_bg_thresh + sigthresh+'_z' +str(2*z).zfill(3) +'_f'+str(nframes)
                            if(log_y==True):titulo = 'adc_count_with'+ cut_adc + avrg_cut + sig_bg_thresh + sigthresh+'_z' +str(2*z).zfill(3) + '_log_y'+'_f'+str(nframes)
                            plt.title(titulo, color='k', fontsize=font_siz)
                            plt.tight_layout()
                            plt.savefig(dirfile+ '/'+'plot_'+source+'_'+titulo+cut_clst_size+'_'+satu+strE  \
                                                +condit_edep+difu_gauss+'.png', dpi=150)

                            #plt.show()
                            #plt.clf()
                            #plt.close()

                            if(type_dat == 'obs'):
                                del data_cnt_obs
                            if(type_dat == 'sim'):
                                del data_cnt_sim
                            if(type_dat == 'sim_bg'):
                                del data_cnt_sim_bg
                                '''
                                if os.path.exists(path_dir_sim_bg + dirname_sim_bg):
                                    shutil.rmtree(path_dir_sim_bg + dirname_sim_bg)
                                    print(f"La carpeta {path_dir_sim_bg}{dirname_sim_bg} ha sido eliminada")
                                else:
                                    print("La carpeta no existe")    
                                '''
                            if(type_dat == 'bg'):
                                del data_cnt_bg
                            if(type_dat == 'all'):
                                del data_cnt_obs, data_cnt_sim, data_cnt_sim_bg, data_cnt_bg

                        ############################################################################
