# -*- coding: utf-8 -*-
"""
Created on Wed May  3 10:36:44 2023

@author: mikb
"""

import numpy as np
import cv2
###############################################################################
import sys
import os

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
from matplotlib import pyplot as plt, cm
from matplotlib import colors
###############################################################################
###############################################################################
import scipy


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
alpha = 1.75  # Exponent in energy-range relationship
#zff = 0.25  # Field-free region thickness in micrometers
#nsigm = 0.25  # Number of sigma for Gaussian spread

# Simulation parameters
nframes = 1000  # Number of frames
z_count = 1  # Count of Z levels


rgb_data_s = np.zeros((height, width))
rgb_data_s2 = np.zeros((height, width))

ag = '8'
tim ='_0.5s'

source = 'Sr90'
#source = 'Cs137'


source_bg = 'backgnd'
fecha_bg = '2023_Oct_22_00h'
#fecha_bg = '2023_Oct_21_09h'
#fecha_bg = '2023_Oct_20_19h'

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

nfrm =100
#z_count = 1  # Count of Z levels
source_bg = 'backgnd'
fecha_bg = '2022_Nov_10'

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
#level_z = list(range(6))
#level_z = [0,2,6,12]
#level_z = [0]

log_y = False
densi=False
l_width = 2
font_siz=24
nbins =50

bg_filt = ''

sim_cut = True

sig_cut = 0
adc_cut = 0 #116 120 #65 88

nsigm_self = 0

#Cluster_big = 0
size_thresh = 0
max_thresh = 0

plt_clst = True
siz_clst_plot = 0
rang_frame = {38}
#rang_frame = range(0, 100)

nsigma =0

#type_dat = 'sim'
#type_dat = 'sim_bg'

nframes = nfrm #
nsigm_bg_sim = 0


ssd_obs = '/home/mbonnett/mik/data_2023/'
ssd = '/home/mbonnett/mik/'
path_main = 'dat_sim_2024/'
path_main = 'dat_sim_2024/g4_'+source+'_2024/'


path_sim = ssd+path_main
path_obs = ssd_obs + 'data_obs_2023/'
#path_sim ='/home/milton/g4_works/data_2023/data_sim_2024/'
#path_obs = '/home/milton/g4_works/data_2023/data_obs_2023/'

gauss_dist = 'Dg2'

sim_n = ''
for sim_n in range(1):
    sim_n ='_s'+str(sim_n)
    sim_n ='_0'
    print('mik')
    # Number of sigma for Gaussian spread
    for nsigm in [0, ]:
        for k in [0.002]:
            k=np.round(k,3)
            k_str = str(int(1000*np.round(k,3)))
            #i+=1; print(i, np.round(k,3))
            #for alpha in np.arange(1.73, 1.775, 0.005):
            for alpha in [1.75]:
                alpha=np.round(alpha,3)
                a_str = str(int(1000*np.round(alpha,3)))
                #print(np.round(alpha,3))
                # Field-free region thickness in micrometers
                #for zff in np.arange(0.2, 1.1, 0.2):
                for zff in [1.0]:
                # Field-free region thickness in micrometers
                    zff = np.round(zff,3)

                    if nsigm > 0 :
                        difu_gauss = f"_{gauss_dist}_{k}k_{alpha}a_{nsigm}sig_{zff}zff"
                    if nsigm == 0 :
                        difu_gauss = f"_{gauss_dist}_{k}k_{alpha}a_{zff}zff"

                    for type_dat in ['obs', 'bg',
                                     #'sim',
                                     ] :

                        if(type_dat == 'obs' or type_dat == 'bg'):
                            navrg = 0
                            n_mean = 1
                            nsigm_bg = 5

                        if(type_dat == 'sim' ):
                            navrg = -1
                            n_mean = 1
                            nsigm_bg =0


                        if(adc_cut>0):
                            cut_adc = '_cutsg_'+str(adc_cut)+'ADC'
                        if(adc_cut == 0 ):
                            cut_adc = ''

                        if(navrg>=0):
                            avrg_cut = '_less_avrg_bg_'+str(navrg)+'sbg'
                            avrg_cut = '_less_'+str(n_mean)+'avrg_bg_'+str(navrg)+'sbg'
                        if(navrg < 0 ):
                            avrg_cut = ''

                        if(nsigm_bg>0):
                            sig_bg_thresh = '_tshld_'+str(nsigm_bg)+'sig_bg'
                        if(nsigm_bg == 0 ):
                            sig_bg_thresh = ''

                        if(nsigm_bg_sim>0):
                            sig_bg_thresh_sim = '_tshld_'+str(nsigm_bg_sim)+'sig_bg'
                            print(sig_bg_thresh_sim)
                        if(nsigm_bg_sim == 0 ):
                            sig_bg_thresh_sim = ''

                        if(nsigm_self>0 and nsigm_bg == 0 and navrg < 0):
                            sigthresh = '_tshld_'+str(nsigm_self)+'sig_self'
                        if(nsigm_self == 0 ):
                            sigthresh = ''

                        if(size_thresh>0):
                            cut_clst_size = '_clst'+ str(size_thresh)
                        if(size_thresh == 0 ):
                            cut_clst_size = ''

                        if(max_thresh>0):
                            cut_max = '_max'+ str(max_thresh)
                        if(max_thresh == 0 ):
                            cut_max = ''

                        #path_obs = '/home/milton/g4_works/data_2023/data_obs_2023/'

                        if(type_dat == 'bg' or type_dat == 'all'):
                            path_dir_bg = path_obs + 'data_'+fecha_bg+'_'+ag+'ag'+'_'+source_bg+'_'+str(1)+'level'+'/'+source_bg+'/'
                            dirname_bg = 'data_'+source_bg+'_f'+str(nfrm)+'_iso0_br50_ag'+ag+'_ad1_'+fecha_bg + avrg_cut + sig_bg_thresh + sigthresh

                            nameframe_bg = 'RGB_z' +str(0).zfill(3)+ '_f' #filename format

                            try:
                                if(plt_clst==False and (size_thresh>0 or max_thresh>0) ):os.makedirs(path_dir_bg + dirname_bg + cut_clst_size+cut_max+cut_adc)
                                #os.makedirs(path_dir_bg + dirname_bg + cut_clst_size+cut_max+'_mf')
                            except FileExistsError:
                                pass

                        if(type_dat == 'obs' or type_dat == 'all'):
                            path_dir_obs = path_obs + 'data_'+fecha+'_'+ag+'ag'+'_'+source+'_'+str(z_count)+'level'+'/'+source+'/'
                            dirname_obs = 'data_'+source+'_f'+str(nfrm)+'_iso0_br50_ag'+ag+'_ad1_'+fecha + avrg_cut + sig_bg_thresh + sigthresh

                            try:
                                if(plt_clst==False and (size_thresh>0 or max_thresh>0) ):os.makedirs(path_dir_obs + dirname_obs + cut_clst_size+cut_max+cut_adc)
                                #os.makedirs(path_dir_obs + dirname_obs + cut_clst_size+cut_max+'_mf')
                            except FileExistsError:
                                pass

                        if(type_dat == 'sim' or type_dat == 'all'):
                            path_dir_sim = path_sim+source+'/'
                            dirname_sim = 'sim_'+source+'_f'+str(nframes)+'_'+str(z_count) \
                                                +'l_edpxl_w'+evt+sim_n+condit_edep \
                                                +difu_gauss
                            try:
                                if(plt_clst==False and (size_thresh>0 or max_thresh>0) ):os.makedirs(path_dir_sim + dirname_sim + cut_clst_size+cut_max+cut_adc)
                                #os.makedirs(path_dir_sim + dirname_sim + cut_clst_size+cut_max+'_mf')
                            except FileExistsError:
                                pass

                        #for z in range(z_count-1, z_count):
                        for z in level_z:

                            #path_cluster = '/home/milton/g4_works/data_2023/dat'+sim_n+'_'+fecha+difu_gauss+'_clstr_filt'+cut_clst_size+cut_max +'/'#+sigthresh + cut_adc + '_z' +str(2*z)+'/'
                            path_cluster = path_sim+source+'/'+'dat'+sim_n+'_'+fecha+difu_gauss+'_clstr_filt'+cut_clst_size+cut_max +'/'#+sigthresh + cut_adc + '_z' +str(2*z)+'/'

                            if(type_dat == 'obs' or type_dat == 'all'):
                                nameframe_obs = 'RGB_z' +str(z).zfill(3)+ '_f' #filename format

                            if(type_dat == 'sim' or type_dat == 'all'):
                                nameframe_sim = 'RGB_eh_3.6eV'+bg_filt+'_z' +str(2*z).zfill(3)+ '_f' #filename format

                            long = 0

                            plt.rcParams.update({'font.size': 12})

                            clst_frame_size_mean_bg =np.zeros((4,nframes*(z+1)), dtype=np.float64)
                            clst_frame_size_mean_obs =np.zeros((4,nframes*(z+1)), dtype=np.float64)
                            clst_frame_size_mean_sim =np.zeros((4,nframes*(z+1)), dtype=np.float64)
                            
                            #path_cluster + 'dat_clstr_frm_siz_mean' +difu_gauss + \
                            file_cluster_size_mean = path_cluster + 'data_clstr_frm_siz_mean' + difu_gauss + \
                                                     cut_adc + avrg_cut + sig_bg_thresh +'_sim' + sig_bg_thresh_sim + sigthresh + '_f' +str(nframes) + '_z' +str(2*z)+'/'
                            try:
                                os.makedirs(file_cluster_size_mean)
                            except FileExistsError:
                                pass

                            if(type_dat == 'bg' or type_dat == 'all'):
                                data_cluste_frame_bg = file_cluster_size_mean+'/'+'dat_'+source_bg+'_f'+str(nframes)+'_clstr_frm_z'+str(z).zfill(3)+\
                                                        '_'+fecha_bg+cut_adc + avrg_cut + sig_bg_thresh + sigthresh + cut_clst_size+cut_max +'.csv'
                                file_clust_frame_bg = open(data_cluste_frame_bg,'w')
                                file_clust_frame_bg.write("Level\t Frame\t x_left\t top\t w_width\t h_height\t" +
                                                        " Nro_clustes\t ClusterSize\t Mean\t ETotal\t Emax\n")
                                file_clust_frame_bg.close()

                                cluste_frame_bg = file_cluster_size_mean + '/' +'clstr_frm_siz_mean_'+source_bg+'_f'+str(nframes)+'_z'+str(z).zfill(3)+\
                                                  '_'+fecha_bg+cut_adc + avrg_cut + sig_bg_thresh + sigthresh + cut_clst_size+cut_max +'.csv'
                                file_clust_bg=open(cluste_frame_bg,'w')
                                file_clust_bg.write("Frame\t numb_clust\t mean_fram\t pixl_act\t pixl_act_mean\n")
                                file_clust_bg.close()

                            if(type_dat == 'obs' or type_dat == 'all'):
                                data_cluste_frame_obs = file_cluster_size_mean+'/'+'dat_obs_'+source+'_f'+str(nframes)+'_clstr_frm_z'+str(z).zfill(3)+\
                                                         '_'+fecha+cut_adc + avrg_cut + sig_bg_thresh + sigthresh+ cut_clst_size+cut_max +'.csv'
                                file_clust_frame_obs = open(data_cluste_frame_obs,'w')
                                file_clust_frame_obs.write("Level\t Frame\t x_left\t top\t w_width\t h_height\t" +
                                                        " Nro_clustes\t ClusterSize\t Mean\t ETotal\t Emax\n")
                                file_clust_frame_obs.close()

                                cluste_frame_obs = file_cluster_size_mean + '/' +'clstr_frm_siz_mean_obs_'+source+'_f'+str(nframes)+'_z'+str(z).zfill(3)+\
                                                   '_'+fecha+cut_adc + avrg_cut + sig_bg_thresh + sigthresh+ cut_clst_size+cut_max +'.csv'
                                file_clust_obs=open(cluste_frame_obs,'w')
                                file_clust_obs.write("Frame\t numb_clust\t mean_fram\t pixl_act\t pixl_act_mean\n")
                                file_clust_obs.close()

                            if(type_dat == 'sim' or type_dat == 'all'):
                                data_cluste_frame_sim = file_cluster_size_mean+'/'+'dat_sim_'+source+'_f'+str(nframes)+'_clstr_frm_z'+str(2*z).zfill(3)+\
                                                        '_w'+evt+difu_gauss + cut_adc + sig_bg_thresh_sim + sigthresh + cut_clst_size+cut_max +'.csv'
                                file_clust_frame_sim = open(data_cluste_frame_sim,'w')
                                file_clust_frame_sim.write("Level\t Frame\t x_left\t top\t w_width\t h_height\t" +
                                                        " Nro_clustes\t ClusterSize\t Mean\t ETotal\t Emax\n")
                                file_clust_frame_sim.close()

                                cluste_frame_sim = file_cluster_size_mean + '/' +'clstr_frm_siz_mean_sim_'+source+'_f'+str(nframes)+'_z'+str(2*z).zfill(3)+\
                                                    '_w'+evt+difu_gauss + cut_adc + sig_bg_thresh_sim + sigthresh + cut_clst_size+cut_max  +'.csv'
                                file_clust_sim=open(cluste_frame_sim,'w')
                                file_clust_sim.write("Frame\t numb_clust\t mean_fram\t pixl_act\t pixl_act_mean\n")
                                file_clust_sim.close()


                            MaxEz_bg   = np.empty(0,dtype=np.float64)
                            TotalEz_bg = np.empty(0,dtype=np.float64)
                            MeanEz_bg  = np.empty(0,dtype=np.float64)
                            ClSizez_bg = np.empty(0,dtype=np.float64)

                            MaxEz_obs   = np.empty(0,dtype=np.float64)
                            TotalEz_obs = np.empty(0,dtype=np.float64)
                            MeanEz_obs  = np.empty(0,dtype=np.float64)
                            ClSizez_obs = np.empty(0,dtype=np.float64)

                            MaxEz_sim   = np.empty(0,dtype=np.float64)
                            TotalEz_sim = np.empty(0,dtype=np.float64)
                            MeanEz_sim  = np.empty(0,dtype=np.float64)
                            ClSizez_sim = np.empty(0,dtype=np.float64)

                            #for n in range(nframes-1, nframes):
                            j=0

                            #for n in frams_adc:
                            for n in range(nframes):
                                #if(z == 0 and (type_dat == 'bg' or type_dat == 'all')):
                                if((type_dat == 'bg' or type_dat == 'all')):
                                    tmpname_bg = path_dir_bg + dirname_bg + '/' + nameframe_bg +str(n).zfill(3) + '.npz'
                                    rgb_data_bg = np.load(tmpname_bg)
                                    rgb_data_bg = np.float64(rgb_data_bg.f.arr_0)
                                    rgb_data_bg[rgb_data_bg<=adc_cut]=0

                                    adc_n = np.unique(rgb_data_bg,return_counts=True)
                                    print (adc_n[1][1:].sum(), rgb_data_bg.max())
                                    tmp_npx= np.zeros((height, width))
                                    tmp_npx[rgb_data_bg>0]=1

                                    data_f = np.copy(rgb_data_bg)
                                    #data_f = scipy.ndimage.median_filter(rgb_data_bg, size=4)

                                    data_f8b = ((data_f.astype(float) - data_f.min())*255/(data_f.max() - data_f.min())).astype(np.uint8)
                                    ret, thresh = cv2.threshold(data_f8b,0,255,cv2.THRESH_BINARY)
                                    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
                                    ClusterSize=np.zeros(n_labels -1, dtype=np.float64) #Size of a Cluster

                                    MeanE  = np.zeros(n_labels -1, dtype=np.float64) #average energy of a given cluster in ADC units
                                    TotalE = np.zeros(n_labels -1, dtype=np.float64) #Total energy of a given cluster in ADC units
                                    MaxE   = np.zeros(n_labels -1, dtype=np.float64) #Max  energy of a given cluster in ADC units

                                    print('Nro clusters bg : ', n_labels-1)
                                    fram_clst_cut_bg = np.zeros((height, width))
                                    data_mf_bg = np.zeros((height, width))
                                    nro_clst = 0

                                    for i in range(1, n_labels):
                                        if stats[i, cv2.CC_STAT_AREA] > size_thresh:
                                            nro_clst+=1
                                            ClusterSize[i-1] = stats[i, cv2.CC_STAT_AREA]
                                            x = stats[i, cv2.CC_STAT_LEFT]
                                            y = stats[i, cv2.CC_STAT_TOP]
                                            w = stats[i, cv2.CC_STAT_WIDTH]
                                            h = stats[i, cv2.CC_STAT_HEIGHT]

                                            m_clust = data_f[y:y+h, x:x+w]
                                            if(np.max(m_clust)>max_thresh):
                                                if(size_thresh>0): fram_clst_cut_bg[y:y+h, x:x+w] = m_clust
                                                if(size_thresh==0 and max_thresh>0): fram_clst_cut_bg[y:y+h, x:x+w] = m_clust
                                                data_mf_bg[y:y+h, x:x+w] = m_clust/np.max(m_clust)

                                                TotalE[i-1] = np.sum(m_clust)
                                                MaxE[i-1]   = np.max(m_clust)
                                                MeanE[i-1]  = np.sum(m_clust)/ClusterSize[i-1]

                                                if(plt_clst==False):
                                                    file_clust_frame_bg = open(data_cluste_frame_bg,'a')
                                                    file_clust_frame_bg.write(str(2*z)+'\t'+str(n)+'\t'+str(x)+'\t'+str(y)+'\t'+str(w)+'\t'+str(h)+'\t'
                                                                +str(nro_clst)+'\t'+str(ClusterSize[i-1])+'\t'+str(np.sum(m_clust)/ClusterSize[i-1])
                                                                +'\t'+str(np.sum(m_clust))+'\t'+str(np.max(m_clust))+'\n')
                                                    file_clust_frame_bg.close()

                                            if plt_clst and ClusterSize[i-1] > siz_clst_plot and n in rang_frame:
                                                plt.figure(figsize=(10,6))
                                                plt.imshow(data_f, cmap=plt.cm.rainbow, norm=colors.LogNorm())
                                                #plt.imshow(data_f,  cmap=plt.cm.binary)
                                                plt.colorbar()
                                                x1, x2 = x - 1,  x + w
                                                y1, y2 = y - 1,  y + h
                                                plt.xlim(x1, x2)
                                                plt.ylim(y1, y2)

                                                for xx in range(x, x+w):
                                                    for yy in range(y, y+h):
                                                        plt.text(xx, yy, str(int(data_f[yy, xx])), color='k', ha='center', va='center')

                                                #plt.grid(True)
                                                plt.xticks(size=font_siz*(1-0.5))
                                                plt.yticks(size=font_siz*(1-0.5))
                                                plt.xlabel('pixels', fontsize=font_siz*(1-0.5))
                                                plt.ylabel('pixels', fontsize=font_siz*(1-0.5))
                                                titulo = 'backgnd ' + '\nZoom_LogNorm_'+ 'ClstrSiz_' + str(int(ClusterSize[i-1]))
                                                plt.title(titulo , fontsize=font_siz*(1-0.5))
                                                plt.tight_layout()
                                                ##plt.savefig(path_cluster+titulo+".png", dpi=150)
                                                plt.savefig(file_cluster_size_mean +'bg_cluster_'+str(i) +'_siz_'+ str(int(ClusterSize[i-1]))+ '_'+source + sigthresh + cut_adc + '_frm_' +str(n)+'.png', dpi=150)

                                                plt.show()


                                    if(plt_clst==False and (size_thresh>0 or max_thresh>0 )):np.savez_compressed(path_dir_bg + dirname_bg + cut_clst_size+cut_max + cut_adc +'/' + nameframe_bg +str(n).zfill(3) + '.npz', np.float64(fram_clst_cut_bg))
                                    #np.savez_compressed(path_dir_bg + dirname_bg + cut_clst_size+cut_max+'_mf/' + nameframe_bg +str(n).zfill(3) + '.npz', np.float64(data_mf_bg))

                                    MaxEz_bg   = np.append(MaxEz_bg, MaxE)
                                    TotalEz_bg = np.append(TotalEz_bg,TotalE)
                                    MeanEz_bg  = np.append(MeanEz_bg, MeanE)
                                    ClSizez_bg = np.append(ClSizez_bg, ClusterSize)

                                    print(n,'Nro clusters_bg > '+str(size_thresh)+' : ', nro_clst)
                                    print(n,'Clustersize_mean_frame_bg : ', ClusterSize.sum()/nro_clst)
                                    print(n,'pixel activ frame_bg      : ', tmp_npx.sum())
                                    print(n,'pixel activ frame_mean_bg : ', rgb_data_bg.sum()/tmp_npx.sum())
                                    clst_frame_size_mean_bg[0][j] = nro_clst
                                    clst_frame_size_mean_bg[1][j] = ClusterSize.sum()/nro_clst
                                    clst_frame_size_mean_bg[2][j] = tmp_npx.sum()
                                    clst_frame_size_mean_bg[3][j] = rgb_data_bg.sum()/tmp_npx.sum()

                                    file_clust_bg=open(cluste_frame_bg,'a')
                                    file_clust_bg.write(str(n)+'\t'+str(nro_clst)+'\t'+str(ClusterSize.sum()/nro_clst)+'\t'+str(tmp_npx.sum())+'\t'+str(rgb_data_bg.sum()/tmp_npx.sum())+'\n')
                                    file_clust_bg.close()

                                if(type_dat == 'obs' or type_dat == 'all'):
                                    tmpname_obs = path_dir_obs + dirname_obs + '/' + nameframe_obs +str(n).zfill(3) + '.npz'
                                    rgb_data_obs = np.load(tmpname_obs)
                                    rgb_data_obs = np.float64(rgb_data_obs.f.arr_0)
                                    rgb_data_obs[rgb_data_obs<=adc_cut]=0

                                    adc_n0 = np.unique(rgb_data_obs,return_counts=True)
                                    print (adc_n0[1][1:].sum(), rgb_data_obs.max())
                                    tmp_npx= np.zeros((height, width))
                                    tmp_npx[rgb_data_obs>0]=1

                                    data_f = np.copy(rgb_data_obs)
                                    #data_f = scipy.ndimage.median_filter(rgb_data_obs, size=3)
                                    #data_f = scipy.ndimage.median_filter(data_f, size=2)

                                    data_f8b = ((data_f.astype(float) - data_f.min())*255/(data_f.max() - data_f.min())).astype(np.uint8)
                                    ret, thresh = cv2.threshold(data_f8b,0,255,cv2.THRESH_BINARY)
                                    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
                                    ClusterSize=np.zeros(n_labels -1, dtype=np.float64) #Size of a Cluster

                                    MeanE  = np.zeros(n_labels -1, dtype=np.float64) #average energy of a given cluster in ADC units
                                    TotalE = np.zeros(n_labels -1, dtype=np.float64) #Total energy of a given cluster in ADC units
                                    MaxE   = np.zeros(n_labels -1, dtype=np.float64) #Max  energy of a given cluster in ADC units

                                    print('Nro clusters obs: ', n_labels-1)
                                    fram_clst_cut_obs = np.zeros((height, width))
                                    data_mf_obs = np.zeros((height, width))
                                    nro_clst = 0

                                    for i in range(1, n_labels):

                                        if stats[i, cv2.CC_STAT_AREA] > size_thresh:
                                            nro_clst+=1
                                            ClusterSize[i-1]=stats[i, cv2.CC_STAT_AREA]
                                            x = stats[i, cv2.CC_STAT_LEFT]
                                            y = stats[i, cv2.CC_STAT_TOP]
                                            w = stats[i, cv2.CC_STAT_WIDTH]
                                            h = stats[i, cv2.CC_STAT_HEIGHT]

                                            #data_mf_obs = np.copy(rgb_data_obs)
                                            #print(data_f.max())
                                            m_clust = data_f[y:y+h, x:x+w]
                                            if(np.max(m_clust)>max_thresh):
                                                if(size_thresh>0): fram_clst_cut_obs[y:y+h, x:x+w] = m_clust
                                                if(size_thresh==0 and max_thresh>0): fram_clst_cut_obs[y:y+h, x:x+w] = m_clust
                                                data_mf_obs[y:y+h, x:x+w] = m_clust/np.max(m_clust)
                                                #print(data_f[y:y+h, x:x+w], data_mf_obs[y:y+h, x:x+w])

                                                TotalE[i-1] = np.sum(m_clust)
                                                MaxE[i-1]   = np.max(m_clust)
                                                MeanE[i-1]  = np.sum(m_clust)/ClusterSize[i-1]
                                                #print(m_clust/np.max(m_clust))
                                               # print(np.uint32(m_clust/np.max(m_clust)))
                                               # print(np.round(m_clust/np.max(m_clust)))

                                                if(plt_clst==False):
                                                    file_clust_frame_obs = open(data_cluste_frame_obs,'a')
                                                    file_clust_frame_obs.write(str(2*z)+'\t'+str(n)+'\t'+str(x)+'\t'+str(y)+'\t'+str(w)+'\t'+str(h)+'\t'
                                                                +str(nro_clst)+'\t'+str(ClusterSize[i-1])+'\t'+str(np.sum(m_clust)/ClusterSize[i-1])
                                                                +'\t'+str(np.sum(m_clust))+'\t'+str(np.max(m_clust))+'\n')
                                                    file_clust_frame_obs.close()

                                            if plt_clst and ClusterSize[i-1] > siz_clst_plot and n in rang_frame:
                                                plt.figure(figsize=(10,6))
                                                plt.imshow(data_f, cmap=plt.cm.rainbow, norm=colors.LogNorm())
                                                #plt.imshow(data_f,  cmap=plt.cm.binary)
                                                plt.colorbar()
                                                x1, x2 = x - 1,  x + w
                                                y1, y2 = y - 1,  y + h
                                                plt.xlim(x1, x2)
                                                plt.ylim(y1, y2)

                                                for xx in range(x, x+w):
                                                    for yy in range(y, y+h):
                                                        plt.text(xx, yy, str(int(data_f[yy, xx])), color='k', ha='center', va='center')

                                                #plt.grid(True)
                                                plt.xticks(size=font_siz*(1-0.5))
                                                plt.yticks(size=font_siz*(1-0.5))
                                                plt.xlabel('pixels', fontsize=font_siz*(1-0.5))
                                                plt.ylabel('pixels', fontsize=font_siz*(1-0.5))
                                                titulo = 'obs ' + '\nZoom_LogNorm_'+ 'ClstrSiz_' + str(int(ClusterSize[i-1]))
                                                plt.title(titulo , fontsize=font_siz*(1-0.4))
                                                plt.tight_layout()
                                                ##plt.savefig(path_cluster+titulo+".png", dpi=150)
                                                plt.savefig(file_cluster_size_mean +'obs_cluster_'+str(i) +'_siz_'+ str(int(ClusterSize[i-1]))+ '_'+source + sigthresh + cut_adc + '_frm_' +str(n)+'.png', dpi=150)

                                                plt.show()


                                    if(plt_clst==False and (size_thresh>0 or max_thresh>0) ):np.savez_compressed(path_dir_obs + dirname_obs + cut_clst_size+cut_max+ cut_adc +'/' + nameframe_obs +str(n).zfill(3) + '.npz', np.float64(fram_clst_cut_obs))
                                    #np.savez_compressed(path_dir_obs + dirname_obs + cut_clst_size+cut_max+'_mf/' + nameframe_obs +str(n).zfill(3) + '.npz', np.float64(data_mf_obs))

                                    ClSizez_obs = np.append(ClSizez_obs, ClusterSize)
                                    MaxEz_obs   = np.append(MaxEz_obs, MaxE)
                                    TotalEz_obs = np.append(TotalEz_obs,TotalE)
                                    MeanEz_obs  = np.append(MeanEz_obs, MeanE)

                                    print(n,'Nro clusters_obs > '+str(size_thresh)+' : ', nro_clst)
                                    print(n,'Clustersize_mean_frame_obs : ', ClusterSize.sum()/nro_clst)
                                    print(n,'pixel activ frame_obs      : ', tmp_npx.sum())
                                    print(n,'pixel activ frame_mean_obs : ', rgb_data_obs.sum()/tmp_npx.sum())
                                    clst_frame_size_mean_obs[0][j] = nro_clst
                                    clst_frame_size_mean_obs[1][j] = ClusterSize.sum()/nro_clst
                                    clst_frame_size_mean_obs[2][j] = tmp_npx.sum()
                                    clst_frame_size_mean_obs[3][j] = rgb_data_obs.sum()/tmp_npx.sum()

                                    file_clust_obs=open(cluste_frame_obs,'a')
                                    file_clust_obs.write(str(n)+'\t'+str(nro_clst)+'\t'+str(ClusterSize.sum()/nro_clst)+'\t'+str(tmp_npx.sum())+'\t'+str(rgb_data_obs.sum()/tmp_npx.sum())+'\n')
                                    file_clust_obs.close()

                                if(type_dat == 'sim' or type_dat == 'all'):
                                    tmpname_sim = path_dir_sim + dirname_sim + '/' + nameframe_sim +str(n).zfill(3) + '.npz'
                                    rgb_data_sim = np.load(tmpname_sim)
                                    rgb_data_sim = np.float64(rgb_data_sim.f.arr_0)
                                    rgb_data_sim[rgb_data_sim<=adc_cut]=0

                                    adc_n1 = np.unique(rgb_data_sim,return_counts=True)
                                    print (adc_n1[1][1:].sum(), rgb_data_sim.max())
                                    tmp_npx= np.zeros((height, width))
                                    tmp_npx[rgb_data_sim>0]=1

                                    data_f = np.copy(rgb_data_sim)
                                    #data_f = scipy.ndimage.median_filter(rgb_data_sim, size=2)

                                    data_f8b = ((data_f.astype(float) - data_f.min())*255/(data_f.max() - data_f.min())).astype(np.uint8)
                                    ret, thresh = cv2.threshold(data_f8b,0,255,cv2.THRESH_BINARY)
                                    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
                                    ClusterSize=np.zeros(n_labels -1, dtype=np.float64) #Size of a Cluster

                                    MeanE  = np.zeros(n_labels -1, dtype=np.float64) #average energy of a given cluster in ADC units
                                    TotalE = np.zeros(n_labels -1, dtype=np.float64) #Total energy of a given cluster in ADC units
                                    MaxE   = np.zeros(n_labels -1, dtype=np.float64) #Max  energy of a given cluster in ADC units

                                    print('Nro clusters sim: ', n_labels-1)
                                    fram_clst_cut_sim = np.zeros((height, width))
                                    data_mf_sim = np.zeros((height, width))
                                    nro_clst = 0

                                    for i in range(1, n_labels):

                                        if stats[i, cv2.CC_STAT_AREA] > size_thresh:
                                            nro_clst+=1
                                            ClusterSize[i-1]=stats[i, cv2.CC_STAT_AREA]
                                            x = stats[i, cv2.CC_STAT_LEFT]
                                            y = stats[i, cv2.CC_STAT_TOP]
                                            w = stats[i, cv2.CC_STAT_WIDTH]
                                            h = stats[i, cv2.CC_STAT_HEIGHT]

                                            m_clust = data_f[y:y+h, x:x+w]
                                            if(np.max(m_clust)>max_thresh):
                                                if(size_thresh>0): fram_clst_cut_sim[y:y+h, x:x+w] = m_clust
                                                if(size_thresh==0 and max_thresh>0): fram_clst_cut_sim[y:y+h, x:x+w] = m_clust
                                                data_mf_sim[y:y+h, x:x+w] = m_clust/np.max(m_clust)

                                                TotalE[i-1] = np.sum(m_clust)
                                                MaxE[i-1]   = np.max(m_clust)
                                                MeanE[i-1]  = np.sum(m_clust)/ClusterSize[i-1]

                                                if(plt_clst==False):
                                                    file_clust_frame_sim = open(data_cluste_frame_sim,'a')
                                                    file_clust_frame_sim.write(str(2*z)+'\t'+str(n)+'\t'+str(x)+'\t'+str(y)+'\t'+str(w)+'\t'+str(h)+'\t'
                                                                +str(nro_clst)+'\t'+str(ClusterSize[i-1])+'\t'+str(np.sum(m_clust)/ClusterSize[i-1])
                                                                +'\t'+str(np.sum(m_clust))+'\t'+str(np.max(m_clust))+'\n')
                                                    file_clust_frame_sim.close()

                                            if plt_clst and ClusterSize[i-1] > siz_clst_plot and n in rang_frame:
                                                plt.figure(figsize=(14,8))
                                                plt.imshow(data_f, cmap=plt.cm.rainbow, norm=colors.LogNorm())
                                                #plt.imshow(data_f,  cmap=plt.cm.binary)
                                                plt.colorbar()
                                                x1, x2 = x - 1,  x + w
                                                y1, y2 = y - 1,  y + h
                                                plt.xlim(x1, x2)
                                                plt.ylim(y1, y2)

                                                for xx in range(x, x+w):
                                                    for yy in range(y, y+h):
                                                        plt.text(xx, yy, str(int(data_f[yy, xx])), color='k', ha='center', va='center')

                                                #plt.grid(True)
                                                plt.xticks(size=font_siz*(1-0.5))
                                                plt.yticks(size=font_siz*(1-0.5))
                                                plt.xlabel('pixels', fontsize=font_siz*(1-0.5))
                                                plt.ylabel('pixels', fontsize=font_siz*(1-0.5))
                                                titulo = 'Sim'+difu_gauss + '_frame'+str(int(n)) + '\nZoom_LogNorm_'+ 'ClstrSiz_' + str(int(ClusterSize[i-1]))
                                                plt.title(titulo , fontsize=font_siz*(1-0.4))
                                                plt.tight_layout()
                                                ##plt.savefig(path_cluster+titulo+".png", dpi=150)
                                                plt.savefig(file_cluster_size_mean+'sim_cluster_'+str(i) +'_siz_'+ str(int(ClusterSize[i-1]))+ '_'+source + sigthresh + cut_adc + '_frm_' +str(n)+'.png', dpi=150)

                                                plt.show()

                                    if(plt_clst==False and (size_thresh>0 or max_thresh>0) ):np.savez_compressed(path_dir_sim + dirname_sim + cut_clst_size+cut_max+cut_adc+'/' + nameframe_sim +str(n).zfill(3) + '.npz', np.float64(fram_clst_cut_sim))
                                    #np.savez_compressed(path_dir_sim + dirname_sim + cut_clst_size+cut_max+'_mf/' + nameframe_sim +str(n).zfill(3) + '.npz', np.float64(data_mf_sim))

                                    MaxEz_sim   = np.append(MaxEz_sim, MaxE)
                                    TotalEz_sim = np.append(TotalEz_sim,TotalE)
                                    MeanEz_sim  = np.append(MeanEz_sim, MeanE)
                                    ClSizez_sim = np.append(ClSizez_sim, ClusterSize)

                                    print(n,'Nro clusters-sim > '+str(size_thresh)+' : ', nro_clst)
                                    print(n,'Clustersize_mean_frame_sim : ', ClusterSize.sum()/nro_clst)
                                    print(n,'pixel activ frame_sim      : ', tmp_npx.sum())
                                    print(n,'pixel activ frame_mean_sim : ', rgb_data_sim.sum()/tmp_npx.sum())
                                    clst_frame_size_mean_sim[0][j] = nro_clst
                                    clst_frame_size_mean_sim[1][j] = ClusterSize.sum()/nro_clst
                                    clst_frame_size_mean_sim[2][j] = tmp_npx.sum()
                                    clst_frame_size_mean_sim[3][j] = rgb_data_sim.sum()/tmp_npx.sum()

                                    file_clust_sim=open(cluste_frame_sim,'a')
                                    file_clust_sim.write(str(n)+'\t'+str(nro_clst)+'\t'+str(ClusterSize.sum()/nro_clst)+'\t'+str(tmp_npx.sum())+'\t'+str(rgb_data_sim.sum()/tmp_npx.sum())+'\n')
                                    file_clust_sim.close()

                                j+=1

                            if(type_dat == 'bg' or type_dat == 'all'):
                                np.savez_compressed(file_cluster_size_mean +'clstr_frm_siz_mean_'+source_bg+'_f'+str(nframes)+'_'+fecha_bg+sigthresh+cut_adc+'_clstmin_'+str(size_thresh), np.float64(clst_frame_size_mean_bg))
                            if(type_dat == 'obs' or type_dat == 'all'):
                                np.savez_compressed(file_cluster_size_mean +'clstr_frm_siz_mean_obs_'+source+'_f'+str(nframes)+'_'+fecha+sigthresh+cut_adc+'_clstmin_'+str(size_thresh), np.float64(clst_frame_size_mean_obs))
                            if(type_dat == 'sim' or type_dat == 'all'):
                                np.savez_compressed(file_cluster_size_mean +'clstr_frm_siz_mean_sim_'+source+'_f'+str(nframes)+'_w'+evt+difu_gauss+sigthresh+cut_adc+'_clstmin_'+str(size_thresh), np.float64(clst_frame_size_mean_sim))
                                print('clstr_frm_siz_mean_sim_'+source+'_f'+str(nframes)+'_w'+evt+difu_gauss+sigthresh+cut_adc+'_clstmin_'+str(size_thresh))
                            #plt.figure(figsize=(15,9.5))
                            #titule = source+ sigthresh + cut_adc + '_frm_' +str(sim_cut) + '\n pixel_activ_'+ str(tmp_npx.sum()) + '_dist_' +str(2*z)+'mm'

                            if(type_dat != 'sim_bg'):
                                titule = source+ '_f'+str(nframes)+'_'+fecha+'_w'+evt+difu_gauss + cut_adc + avrg_cut + sig_bg_thresh + '_sim' +sig_bg_thresh_sim + sigthresh + '_dist_' +str(2*z)+'mm'

                            plt1.rcParams.update({'font.size': 14})
                            fig, ([ax0, ax1], [ax2, ax3]) = plt1.subplots(2,2, figsize=(14,8.5) )#, sharex=True )

                            #number of clusters per frame

                            if(type_dat == 'bg' or type_dat == 'all'):
                                #ax0.figure(figsize=(15,9.5))
                                c0=clst_frame_size_mean_bg[0]
                                c1= c0[c0>0]
                                ax0.hist(c1, bins = nbins, histtype='step'
                                         , density=densi, log=log_y, color='k', linewidth=l_width+0.5*l_width,
                                         label = source_bg+sigthresh+cut_adc)
                                #ax0.set_xticks(size=font_siz)
                                #ax0.set_yticks(size=font_siz)
                                ax0.legend()
                                ax0.set_xlabel('Number of clusters per frame')
                                ax0.set_ylabel('count in all frames')
                                ##ax0.title(titule, fontsize=font_siz)
                                #ax0.show()

                            if(type_dat == 'obs' or type_dat == 'all'):
                                #plt.figure(figsize=(15,9.5))
                                c0 = clst_frame_size_mean_obs[0]
                                c2 = c0[c0>0]
                                ax1.hist(c2, bins = nbins, histtype='step'
                                         , density=densi, log=log_y, color='C3', linewidth=l_width+0.5*l_width,
                                         label = source+sigthresh+cut_adc)
                                #ax1.set_xticks(size=font_siz)
                                #ax1.set_yticks(size=font_siz)
                                ax1.legend()
                                ax1.set_xlabel('Number of clusters per frame')
                                ax1.set_ylabel('count in all frames')
                                ##ax1.title(titule, fontsize=font_siz)
                                #ax1.show()

                            if(type_dat == 'sim' or type_dat == 'all'):
                                #ax2.figure(figsize=(15,9.5))
                                if(sim_cut == True):
                                    c0 = clst_frame_size_mean_sim[0]
                                    c3 = c0[c0>0]
                                    ax2.hist(c3, bins = nbins, histtype='step'
                                         , density=densi, log=log_y, color='C2', linewidth=l_width+0.5*l_width,
                                         label = 'Sim_'+source+sigthresh+cut_adc)
                                if(sim_cut == False):
                                    c0 = clst_frame_size_mean_sim[0]
                                    c3 = c0[c0>0]
                                    ax2.hist(c3, bins = nbins, histtype='step'
                                         , density=densi, log=log_y, color='C2', linewidth=l_width+0.5*l_width,
                                         label = 'Sim_'+source)
                                #ax2.set_xticks(size=font_siz)
                                #ax2.set_yticks(size=font_siz)
                                ax2.legend()
                                ax2.set_xlabel('Number of clusters per frame')
                                ax2.set_ylabel('count in all frames')
                                ##ax2.title(titule, fontsize=font_siz)
                                #ax2.show()

                            fig.suptitle(titule + '\n Cluster > '+str(size_thresh)+' Number of clusters per frame')
                            fig.tight_layout()
                            plt1.savefig(file_cluster_size_mean + '_num_clst_pf'+str(size_thresh)+difu_gauss+'.png', dpi=150)
                            # plt1.show()

                            if(type_dat == 'all'):
                                plt.figure(figsize=(15,9.5))
                                plt.hist(c1, bins = nbins, histtype='step'
                                         , density=densi, log=log_y, color='k', linewidth=l_width+0.5*l_width,
                                         label = source_bg+'\n'+sigthresh+cut_adc)
                                plt.hist(c2, bins = nbins, histtype='step'
                                         , density=densi, log=log_y, color='C3', linewidth=l_width+0.5*l_width,
                                         label = source+'\n'+sigthresh+cut_adc)
                                if(sim_cut == True):
                                    plt.hist(c3, bins = nbins, histtype='step'
                                             , density=densi, log=log_y, color='C2', linewidth=l_width+0.5*l_width,
                                             label = 'Sim_'+source+'\n'+sigthresh+cut_adc)
                                if(sim_cut == False):
                                    plt.hist(c3, bins = nbins, histtype='step'
                                             , density=densi, log=log_y, color='C2', linewidth=l_width+0.5*l_width,
                                             label = 'Sim_'+source)

                                plt.xticks(size=font_siz)
                                plt.yticks(size=font_siz)
                                plt.legend(fontsize=font_siz-4)
                                plt.xlabel('Number of clusters per frame', fontsize=font_siz+4)
                                plt.ylabel('count in all frames', fontsize=font_siz+2)
                                plt.title(titule + '\n Cluster > '+str(size_thresh)+' Number of clusters per frame', fontsize=font_siz)
                                plt.savefig(file_cluster_size_mean + '_num_clst_pf_all'+str(size_thresh)+ difu_gauss+'.png', dpi=150)
                                # plt.show()
                                #plt.close()


                            #the average cluster size per frame
                            fig, ([ax0, ax1], [ax2, ax3]) = plt1.subplots(2,2, figsize=(14,8.5) )#, sharex=True )

                            if(type_dat == 'bg' or type_dat == 'all'):
                                #plt.figure(figsize=(15,9.5))
                                cm0=clst_frame_size_mean_bg[1]
                                cm1= cm0[cm0>0]
                                ax0.hist(cm1, bins = nbins, histtype='step'
                                         , density=densi, log=log_y, color='k', linewidth=l_width+0.5*l_width,
                                         label = source_bg+sigthresh+cut_adc)
                                #ax0.set_xticks(size=font_siz)
                                #ax0.set_yticks(size=font_siz)
                                ax0.legend()
                                ax0.set_xlabel('Average cluster size per frame')
                                ax0.set_ylabel('count in all frames')
                                #plt.title(titule, fontsize=font_siz)
                                ## plt.show()
                                #plt.close()

                            if(type_dat == 'obs' or type_dat == 'all'):
                                #plt.figure(figsize=(15,9.5))
                                cm0=clst_frame_size_mean_obs[1]
                                cm2= cm0[cm0>0]
                                ax1.hist(cm2, bins = nbins, histtype='step'
                                         , density=densi, log=log_y, color='C3', linewidth=l_width+0.5*l_width,
                                         label = source+sigthresh+cut_adc)
                                #ax1.set_xticks(size=font_siz)
                                #ax1.set_yticks(size=font_siz)
                                ax1.legend()
                                ax1.set_xlabel('Average cluster size per frame')
                                ax1.set_ylabel('count in all frames')
                                ##ax1.title(titule, fontsize=font_siz)
                                ## plt.show()
                                #plt.close()

                            if(type_dat == 'sim' or type_dat == 'all'):
                                #plt.figure(figsize=(15,9.5))
                                if(sim_cut == True):
                                    cm0 = clst_frame_size_mean_sim[1]
                                    cm3 = cm0[cm0>0]
                                    ax2.hist(cm3, bins = nbins, histtype='step'
                                         , density=densi, log=log_y, color='C2', linewidth=l_width+0.5*l_width,
                                         label = 'Sim_'+source+sigthresh+cut_adc)
                                if(sim_cut == False):
                                    cm0 = clst_frame_size_mean_sim[1]
                                    cm3 = cm0[cm0>0]
                                    ax2.hist(cm3, bins = nbins, histtype='step'
                                         , density=densi, log=log_y, color='C2', linewidth=l_width+0.5*l_width,
                                         label = 'Sim_'+source)
                                #ax2.set_xticks(size=font_siz)
                                #ax2.set_yticks(size=font_siz)
                                ax2.legend()
                                ax2.set_xlabel('Average cluster size per frame')
                                ax2.set_ylabel('count in all frames')
                                ##ax2.title(titule, fontsize=font_siz)
                                #ax2.show()

                            fig.suptitle(titule + '\n Cluster > '+str(size_thresh)+' Average cluster size per frame')
                            fig.tight_layout()
                            plt1.savefig(file_cluster_size_mean + '_avrg_clst_pf'+str(size_thresh)+ difu_gauss+'.png', dpi=150)
                            # # # plt1.show()

                            if(type_dat == 'all'):
                                plt.figure(figsize=(15,9.5))
                                plt.hist(cm1, bins = nbins, histtype='step'
                                         , density=densi, log=log_y, color='k', linewidth=l_width+0.5*l_width,
                                         label = source_bg+'\n'+sigthresh+cut_adc)
                                plt.hist(cm2, bins = nbins, histtype='step'
                                         , density=densi, log=log_y, color='C3', linewidth=l_width+0.5*l_width,
                                         label = source+'\n'+sigthresh+cut_adc)
                                if(sim_cut == True):
                                    plt.hist(cm3, bins = nbins, histtype='step'
                                             , density=densi, log=log_y, color='C2', linewidth=l_width+0.5*l_width,
                                             label = 'Sim_'+source+'\n'+sigthresh+cut_adc)
                                if(sim_cut == False):
                                    plt.hist(cm3, bins = nbins, histtype='step'
                                             , density=densi, log=log_y, color='C2', linewidth=l_width+0.5*l_width,
                                             label = 'Sim_'+source)

                                plt.xticks(size=font_siz)
                                plt.yticks(size=font_siz)
                                plt.legend(fontsize=font_siz-4)
                                plt.xlabel('Average cluster size per frame', fontsize=font_siz+4)
                                plt.ylabel('count in all frames', fontsize=font_siz+2)
                                plt.title(titule + '\n Cluster > '+str(size_thresh)+' Average cluster size per frame', fontsize=font_siz)
                                plt.savefig(file_cluster_size_mean + '_avrg_clst_pf_all'+str(size_thresh)+ difu_gauss+'.png', dpi=150)
                                # plt.show()
                                #plt.close()

                            #number of active pixels per frame
                            fig, ([ax0, ax1], [ax2, ax3]) = plt1.subplots(2,2, figsize=(14,8.5) )#, sharex=True )

                            if(type_dat == 'bg' or type_dat == 'all'):
                                #plt.figure(figsize=(15,9.5))
                                p0=clst_frame_size_mean_bg[2]
                                p1= p0[p0>0]
                                ax0.hist(p1, bins = nbins, histtype='step'
                                         , density=densi, log=log_y, color='k', linewidth=l_width+0.5*l_width,
                                         label = source_bg+sigthresh+cut_adc)
                                #ax0.set_xticks(size=font_siz)
                                #ax0.set_yticks(size=font_siz)
                                ax0.legend()
                                ax0.set_xlabel('Number of active pixels per frame')
                                ax0.set_ylabel('count in all frames')
                                #ax0.title(titule, fontsize=font_siz)
                                #ax0.show()


                            if(type_dat == 'obs' or type_dat == 'all'):
                                #ax1.figure(figsize=(15,9.5))
                                p0=clst_frame_size_mean_obs[2]
                                p2= p0[p0>0]
                                ax1.hist(p2, bins = nbins, histtype='step'
                                         , density=densi, log=log_y, color='C3', linewidth=l_width+0.5*l_width,
                                         label = source+sigthresh+cut_adc)
                                #ax1.set_xticks(size=font_siz)
                                #ax1.set_yticks(size=font_siz)
                                ax1.legend()
                                ax1.set_xlabel('Number of active pixels per frame')
                                ax1.set_ylabel('count in all frames')
                                #ax1.title(titule, fontsize=font_siz)
                                #ax1.show()

                            if(type_dat == 'sim' or type_dat == 'all'):
                                #ax2.figure(figsize=(15,9.5))
                                if(sim_cut == True):
                                    p0=clst_frame_size_mean_sim[2]
                                    p3= p0[p0>0]
                                    ax2.hist(p3, bins = nbins, histtype='step'
                                         , density=densi, log=log_y, color='C2', linewidth=l_width+0.5*l_width,
                                         label = 'Sim_'+source+sigthresh+cut_adc)
                                if(sim_cut == False):
                                    p0=clst_frame_size_mean_sim[2]
                                    p3= p0[p0>0]
                                    ax2.hist(p3, bins = nbins, histtype='step'
                                         , density=densi, log=log_y, color='C2', linewidth=l_width+0.5*l_width,
                                         label = 'Sim_'+source)
                                #ax2.set_xticks(size=font_siz)
                                #ax2.set_yticks(size=font_siz)
                                ax2.legend()
                                ax2.set_xlabel('Number of active pixels per frame')
                                ax2.set_ylabel('count in all frames')
                                #ax2.title(titule, fontsize=font_siz)
                                #ax2.show()

                            fig.suptitle(titule + '\n Number of active pixels per frame')
                            fig.tight_layout()
                            plt1.savefig(file_cluster_size_mean +'_num_pxl_pf'+ difu_gauss+'.png', dpi=150)
                            # plt1.show()
                            #plt.close()

                            if(type_dat == 'all'):
                                plt.figure(figsize=(15,9.5))
                                plt.hist(p1, bins = nbins, histtype='step'
                                         , density=densi, log=log_y, color='k', linewidth=l_width+0.5*l_width,
                                         label = source_bg+'\n'+sigthresh+cut_adc)
                                plt.hist(p2, bins = nbins, histtype='step'
                                         , density=densi, log=log_y, color='C3', linewidth=l_width+0.5*l_width,
                                         label = source+'\n'+sigthresh+cut_adc)
                                if(sim_cut == True):
                                    plt.hist(p3, bins = nbins, histtype='step'
                                             , density=densi, log=log_y, color='C2', linewidth=l_width+0.5*l_width,
                                             label = 'Sim_'+source+'\n'+sigthresh+cut_adc)
                                if(sim_cut == False):
                                    plt.hist(p3, bins = nbins, histtype='step'
                                             , density=densi, log=log_y, color='C2', linewidth=l_width+0.5*l_width,
                                             label = 'Sim_'+source)

                                plt.xticks(size=font_siz)
                                plt.yticks(size=font_siz)
                                plt.legend(fontsize=font_siz-4)
                                plt.xlabel('Number of active pixels per frame', fontsize=font_siz+4)
                                plt.ylabel('count in all frames', fontsize=font_siz+2)
                                plt.title(titule + '\n Number of active pixels per frame', fontsize=font_siz)
                                plt.savefig(file_cluster_size_mean + '_num_pxl_pf_all'+ difu_gauss+'.png', dpi=150)
                                # plt.show()
                                #plt.close()

                            #the average active pixels per frame
                            fig, ([ax0, ax1], [ax2, ax3]) = plt1.subplots(2,2, figsize=(14,8.5) )#, sharex=True )

                            if(type_dat == 'bg' or type_dat == 'all'):
                                #ax0.figure(figsize=(15,9.5))
                                pm0=clst_frame_size_mean_bg[3]
                                pm1= pm0[pm0>0]
                                ax0.hist(pm1, bins = nbins, histtype='step'
                                         , density=densi, log=log_y, color='k', linewidth=l_width+0.5*l_width,
                                         label = source_bg+sigthresh+cut_adc)
                                #ax0.set_xticks(size=font_siz)
                                #ax0.set_yticks(size=font_siz)
                                ax0.legend()
                                ax0.set_xlabel('Average active pixels per frame')
                                ax0.set_ylabel('count in all frames')
                                #ax0.title(titule, fontsize=font_siz)
                                #ax0.show()

                            if(type_dat == 'obs' or type_dat == 'all'):
                                #ax1.figure(figsize=(15,9.5))
                                pm0=clst_frame_size_mean_obs[3]
                                pm2= pm0[pm0>0]
                                ax1.hist(pm2, bins = nbins, histtype='step'
                                         , density=densi, log=log_y, color='C3', linewidth=l_width+0.5*l_width,
                                         label = source+sigthresh+cut_adc)
                                #ax1.set_xticks(size=font_siz)
                                #ax1.set_yticks(size=font_siz)
                                ax1.legend()
                                ax1.set_xlabel('Average active pixels per frame')
                                ax1.set_ylabel('count in all frames')
                                #ax1.title(titule, fontsize=font_siz)
                                #ax1.show()

                            if(type_dat == 'sim' or type_dat == 'all'):
                                #ax2.figure(figsize=(15,9.5))
                                if(sim_cut == True):
                                    pm0=clst_frame_size_mean_sim[3]
                                    pm3= pm0[pm0>0]
                                    ax2.hist(pm3, bins = nbins, histtype='step'
                                         , density=densi, log=log_y, color='C2', linewidth=l_width+0.5*l_width,
                                         label = 'Sim_'+source+sigthresh+cut_adc)
                                if(sim_cut == False):
                                    pm0=clst_frame_size_mean_sim[3]
                                    pm3= pm0[pm0>0]
                                    ax2.hist(pm3, bins = nbins, histtype='step'
                                         , density=densi, log=log_y, color='C2', linewidth=l_width+0.5*l_width,
                                         label = 'Sim_'+source)
                                #ax2.set_xticks(size=font_siz)
                                #ax2.set_yticks(size=font_siz)
                                ax2.legend()
                                ax2.set_xlabel('Average active pixels per frame')
                                ax2.set_ylabel('count in all frames')
                                #ax2.title(titule, fontsize=font_siz)
                                #ax2.show()

                            fig.suptitle(titule + '\n Average active pixels per frame')
                            fig.tight_layout()
                            plt1.savefig(file_cluster_size_mean + '_avrg_pxl_pf'+ difu_gauss+'.png', dpi=150)
                            # plt1.show()

                            if(type_dat == 'all'):
                                plt.figure(figsize=(15,9.5))
                                plt.hist(pm1, bins = nbins, histtype='step'
                                         , density=densi, log=log_y, color='k', linewidth=l_width+0.5*l_width,
                                         label = source_bg+'\n'+sigthresh+cut_adc)
                                plt.hist(pm2, bins = nbins, histtype='step'
                                         , density=densi, log=log_y, color='C3', linewidth=l_width+0.5*l_width,
                                         label = source+'\n'+sigthresh+cut_adc)
                                if(sim_cut == True):
                                    plt.hist(pm3, bins = nbins, histtype='step'
                                             , density=densi, log=log_y, color='C2', linewidth=l_width+0.5*l_width,
                                             label = 'Sim_'+source+'\n'+sigthresh+cut_adc)
                                if(sim_cut == False):
                                    plt.hist(pm3, bins = nbins, histtype='step'
                                             , density=densi, log=log_y, color='C2', linewidth=l_width+0.5*l_width,
                                             label = 'Sim_'+source)

                                plt.xticks(size=font_siz)
                                plt.yticks(size=font_siz)
                                plt.legend(fontsize=font_siz-4)
                                plt.xlabel('Average active pixels per frame', fontsize=font_siz+4)
                                plt.ylabel('count in all frames', fontsize=font_siz+2)
                                plt.title(titule + '\n Average active pixels per frame', fontsize=font_siz)
                                plt1.savefig(file_cluster_size_mean + '_avrg_pxl_pf_all'+ difu_gauss+'.png', dpi=150)
                                # plt.show()
                                #plt.close()
                                plt.clf()

                            ############################################################################
