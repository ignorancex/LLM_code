#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Mon Jul 12 21:22:15 2021

@author: mik
'''
import numpy as np

from io import StringIO

import os

import random
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1

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
alpha = 1.75  # Exponent in energy-range relationship
#zff = 0.25  # Field-free region thickness in micrometers#
#nsigm = 0.25  # Number of sigma for Gaussian spread

# Simulation parameters
nframes = 1000  # Number of frames
z_count = 1  # Count of Z levels

rgb_data_s = np.zeros((height, width))
rgb_data_s2 = np.zeros((height, width))
rgb_data_d = np.zeros((height, width))
rgb_data_a = np.zeros((height, width))

strE= 'eh_'+str(pair_eh)+'eV'
print(strE)
# Conditions for energy deposition
condit_edep = ''  # '', '_dif0', '_zero' or other conditions can be added here
#condit_edep = '_dif0'


source = 'Sr90'
source = 'Cs137'

#source = 'background'
ag ='8'
#for source in [  'Sr90',
#            #'Cs137'
#            ]:


ssd_obs = '/home/mbonnett/mik/data_2023/'
ssd = '/home/mbonnett/mik/'
path_main = 'dat_sim_2024/'
path_main = 'dat_sim_2024/g4_'+source+'_2024/'



if(source == 'Sr90'):
    winback ='254'
    fecha = '2023_Oct_24_23h'
    ev = '1514'
    evt = winback+'_ev'+ev
    nfrm =1000
    z_count = 1


if(source == 'Cs137'):
    winback='635'
    fecha = '2023_Oct_23_23h'
    ev = '3811'
    evt = winback+'_ev'+ev
    nfrm =1000
    z_count = 1


if(source == 'backgnd'):
    fecha='2023_Oct_22_00h'
    #fecha='2023_Oct_21_09h'
    #fecha='2023_Oct_20_19h'
    nfrm =1000
    z_count = 1
'''  

if(source == 'Sr90'):
    winback = '254'
    fecha = '2021_Nov_09'
    ev = '1587'
    evt = winback+'_ev'+ev
    nfrm =100
    z_count = 11
    
if(source == 'Cs137'):
    winback = '635'
    fecha = '2021_Nov_23'
    ev = '3991'
    evt = winback+'_ev'+ev
    nfrm =100
    z_count = 11


if(source == 'backgnd'):
    fecha = '2022_Nov_10'
    nfrm =100
    z_count = 1
'''

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

                    #difu_gauss = '_DgausZ02_'+str(nsigm)+'sig_'+str(zff)+'zff'

                    nframes = nfrm #

                    satu = ''
                    #satu = '_nosat'
                    #satu = '_nosatn'

                    tim ='_0.5s'

                    for rgb_op in [
                                    'RGB_eh_3.6eV',
                                    'RGB_noise_eh_3.6eV'
                                ]:

                        if(rgb_op == 'RGB_eh_3.6eV'  and source != 'backgnd'):
                            path_dir = ssd+path_main
                            dirname = 'data_'+source+'_pixZ_2um_'+str(z_count )+'l_'+evt+sim_n+'_'+satu+strE  \
                                                +condit_edep+difu_gauss
                        if(rgb_op == 'RGB_noise_eh_3.6eV' and source != 'backgnd'):
                            path_dir = ssd+path_main
                            dirname = 'data_'+source+'_pixZ_2um_'+str(z_count )+'l_'+evt+sim_n+'_'+satu+strE  \
                                                +condit_edep+difu_gauss +'_ag'+ag+'_bg'

                        dirfile=path_dir+dirname

                        s = np.zeros((height, width))
                        s2 = np.zeros((height, width))

                        #ii=0

                        #for z in range(z_count-1, z_count):
                        for z in range(z_count):

                            nameframe = rgb_op+'_z' +str(2*z).zfill(3)+ '_f' #filename format

                            dt_s = np.zeros((height, width))
                            dt_s2 = np.zeros((height, width))

                            rgb_data_s = np.zeros((height, width))
                            rgb_data_s2 = np.zeros((height, width))
                            rgb_data_d = np.zeros((height, width))
                            rgb_data_a = np.zeros((height, width))

                            data_noise_s = np.zeros((height, width))
                            data_noise_s2 = np.zeros((height, width))

                            rgb_data_tmp = 0

                            for n in range(nframes):
                                tmpname = path_dir + dirname + '/' + nameframe +str(n).zfill(3) + '.npz'
                                tmp_data = np.load(tmpname)
                                #rgb_data = np.float64(tmp_data.f.arr_0)
                                rgb_data = np.int16(tmp_data.f.arr_0)

                                if(rgb_data_tmp <= rgb_data.max()):rgb_data_tmp = rgb_data.max()
                                rgb_data_max = rgb_data_tmp

                                print(n, rgb_data.max(), 'max: ', rgb_data_max)

                                rgb_data_s += np.float64(rgb_data)
                                rgb_data_s2 += np.float64(rgb_data)*np.float64(rgb_data)


                                #if (n == nframes-1):
                            N=1.*(nframes)
                            prom=np.float64(rgb_data_s/nframes)
                            #sigm = sqrt((rgb_data_s2-N*prom**2)/(N-1))
                            sigm = np.float64(((rgb_data_s2 - nframes*(prom**2))/(nframes-1))**0.5)

                            #np.save(dirfile+ '/' +rgb_op+'_sum'+'_z%03d' % k, rgb_data_s)
                            #np.save(dirfile+ '/' +rgb_op+'_C_sum2'+'_z%03d' % k, rgb_data_s2)
                            np.savez_compressed(dirfile+ '/' +rgb_op+'_mean'+'_f'+str(nframes)+'_z%03d' % (2*z), np.float64(prom))
                            np.savez_compressed(dirfile+ '/' +rgb_op+'_sigm'+'_f'+str(nframes)+'_z%03d' % (2*z), np.float64(sigm))
                            print('ok_',z, n, N)

                            #path_dir+dirname
                            #ii=0
                            #print(ii)

                            s += rgb_data_s
                            s2 += rgb_data_s2
                            print(rgb_data_s.mean(),rgb_data_s2.mean(),s.mean(),s2.mean())


                        N = (nframes)*z_count
                        prom_all=np.float64(s/N)
                                   #sigm = sqrt((rgb_data_s2-N*prom_all**2)/(N-1))
                        desv_all = np.float64(((s2 - N*(prom_all**2))/(N-1))**0.5)

                        #np.savez_compressed(dirfile+ '/' +rgb_op+'_mean_all', np.float64(prom_all))
                        #np.savez_compressed(dirfile+ '/' +rgb_op+'_sigm_all', np.float64(desv_all))
                        print('All', z, N, n*z)
                        print(rgb_data_s.mean(),rgb_data_s2.mean(),s.mean(),s2.mean())

                       # plt.imshow(prom)
                        #plt.imshow(sigm)


                        ###########################################################################
                        nfiles = int( nframes/1) #number of files

                        filenamef = rgb_op+'_z' #filename format
                        #filenamef = 'E_sim_z'

                        for sigma in[ #0.001,
                                     0,
                                  ]:
                            if(sigma == 0.001):filtr = ''
                            else:filtr = '_'+str(sigma)+'sigm'

                            if(sigma>0):
                                MeanFrame = np.load(path_dir+dirname+'/' +rgb_op+'_mean'+'_f'+str(nframes)+'_z%03d' % (2*z)+'.npz')
                                MeanFrame = np.float64(MeanFrame.f.arr_0)
                                SigmaFrame = np.load(path_dir+dirname+'/' +rgb_op+'_mean'+'_f'+str(nframes)+'_z%03d' % (2*z)+'.npz')
                                SigmaFrame = np.float64(SigmaFrame.f.arr_0)
                                threshold = abs(MeanFrame) + sigma*SigmaFrame #filter threshold

                            if(sigma == 0 or filtr == '' ):threshold = 0 #filter threshold

                            if(rgb_op == 'RGB_eh_3.6eV'  and source != 'backgnd'):
                                dirsave ='adc_count_sim_'+source+'_'+str(nframes)+'f'+'_'+evt+sim_n+satu+difu_gauss
                            if(rgb_op == 'RGB_noise_eh_3.6eV'  and source != 'backgnd'):
                                dirsave ='adc_count_sim_'+source+'_'+str(nframes)+'f_'+evt+sim_n+satu+difu_gauss+'_ag'+ag+'_bg'

                            try:
                                os.makedirs(path_dir+dirsave)
                            except FileExistsError:
                                # directory already exists
                                pass

                            #c=np.zeros((2, rgb_data_max+1),dtype=np.int64)

                           # for z in range(z_count-1, z_count):
                            for z in range(z_count):
                                #data_cnt=np.zeros((2, rgb_data_max+1),dtype=np.int64)
                                #data_cnt=np.zeros((2, 32660),dtype=np.int64)
                                data_cnt=np.zeros((2, 1024),dtype=np.int64)

                                for n in range(nfiles):
                                    #filenamef == rgb_op+'_z'
                                    #pixel activated
                                    tmpname = dirfile +'/' + filenamef +str(2*z).zfill(3)+ '_f'+ str(n).zfill(3) + '.npz'
                                    tmp_data = np.load(tmpname)
                                    data = np.float64(tmp_data.f.arr_0)
                                    print(tmpname)

                                    ev_matrix = (data) > threshold #filtering


                                    if(satu == '_nosatn'):data_f = data
                                    else:data_f = (data)*ev_matrix

                                    #n_adc, n_count=np.unique(data_f,return_counts=True)

                                    adc_n = np.unique(data_f,return_counts=True)

                                    for j in range(adc_n[0].size):
                                        data_cnt[0,int(adc_n[0][j])]=(adc_n[0][j])
                                        data_cnt[1,int(adc_n[0][j])]=data_cnt[1,int(adc_n[0][j])]+(adc_n[1][j])

                                    #if(satu == '_nosatn'):
                                     #   data_cnt[0] = np.roll(data_cnt[0], 2)
                                      #  data_cnt[1] = np.roll(data_cnt[1], 2)

                                    data_cnt[:1].max()
                                    data_cnt[1:].max()
                                    print(n, data_cnt[:1].max(),data_cnt[1:].max())

                                #if(rgb_op!='RGB'):np.savez_compressed(path_dir+dirsave+'/'+ dirname+filtr+'_z'+str(2*z).zfill(3)+'_'+rgb_op+'_f'+str(nframes)+'.npz', data_cnt)

                                if(rgb_op == 'RGB_eh_3.6eV'  and source != 'backgnd'):
                                    np.savez_compressed(path_dir+dirsave+'/'+ 'sim_'+source+'_'+str(nfrm)+'f_'+evt+filtr+satu+condit_edep+difu_gauss+'_z'+str(2*z).zfill(3)+'_'+rgb_op+'_f'+str(nframes)+'.npz', data_cnt)
                                if(rgb_op == 'RGB_noise_eh_3.6eV'  and source != 'backgnd'):
                                    np.savez_compressed(path_dir+dirsave+'/'+ 'sim_'+source+'_'+str(nfrm)+'f_'+evt+'_ag8_bg'+filtr+satu+condit_edep+difu_gauss+'_z'+str(2*z).zfill(3)+'_'+rgb_op+'_f'+str(nframes)+'.npz', data_cnt)


                                print(path_dir+dirsave+'/'+ dirname+filtr+'_z'+str(2*z).zfill(3)+'.npz')
                                ###########################################################################################
                                font_siz=22
                                x=data_cnt;

                                bins_x=max(x[0])+2

                                max_adc=(max(x[0])+1)
                                min_adc=data_cnt[0].min()

                                plt.figure(figsize=(14,8))
                                #plt.plot(x[0],x[1],'r',label= 'data '+ filtr)
                                #plt.plot(y[0],y[1],label= 'sim '+ filtr)
                                if(rgb_op=='RGB'): colr = 'C5'
                                if(rgb_op == 'RGB_eh_3.6eV' ): colr = 'C6'
                                if(rgb_op == 'RGB_noise_eh_3.6eV' ): colr = 'C6'

                                if(rgb_op=='RGB_prm'): colr = 'C4'
                                if(rgb_op=='RGB_std'): colr = 'C1'

                                if(rgb_op=='R'): colr = 'C3'
                                if(rgb_op=='G'): colr = 'C2'
                                if(rgb_op=='B'): colr = 'C0'
                                plt.hist(x[0], weights=x[1], bins=np.arange(min_adc, bins_x), histtype='step', log=True, color=colr, linewidth=2.5, label= 'data_ag_'+ag+tim+ filtr)
                                plt.yscale('log')
                                plt.grid(True)
                                plt.xticks(size=font_siz)
                                plt.yticks(size=font_siz)
                                plt.xlabel('ADC', fontsize=font_siz)
                                plt.ylabel('ADC count in all frames', fontsize=font_siz)
                                plt.legend(fontsize=font_siz)
                                plt.title('plot_hist_'+source+'_ag_'+ag+'_'+rgb_op+filtr+'_z'+str(z*2).zfill(3)+'mm'+'_f'+str(nframes), color='blue', fontsize=font_siz)

                                plt.tight_layout()

                                #plt.savefig(path_dir+dirsave+'/'+'plot_hist_ag_'+ag+tim+filtr+'_z'+str(2*z).zfill(3)+'.png', dpi=150)
                                if(rgb_op=='RGB'):plt.savefig(path_dir+dirsave+'/'+'plot_hist_'+source+'_ag_'+ag+tim+filtr+'_z'+str(2*z).zfill(3)+'_f'+str(nframes)+'.png', dpi=150)
                                if(rgb_op!='RGB'):plt.savefig(path_dir+dirsave+'/'+'plot_hist_'+source+'_ag_'+ag+tim+filtr+'_z'+str(2*z).zfill(3)+'_'+rgb_op+'_f'+str(nframes)+'.png', dpi=150)
                                #plt.show()
                                #plt.clf()
                                #plt.close()
