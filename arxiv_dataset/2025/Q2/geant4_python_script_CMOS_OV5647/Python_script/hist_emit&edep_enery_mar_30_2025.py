# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 18:02:07 2024

@author: mikb

"""
import numpy as np

###############################################################################

from io import StringIO
#from subprocess import call
import os

import random
import matplotlib.pyplot as plt

from scipy.stats import chi2
#from numba import njit
# @njit
###############################################################################
###############################################################################
def chi_2_sigm(obs, cal, sigma):
    chi_2= 1* np.sum([(((a - b) / c)**2)
            for (a, b, c) in zip(obs[sigma > 0], cal[sigma > 0], sigma[sigma > 0])])
    #return chi_2, obs[sigma > 0].size
    return chi_2, obs.size

def chi_2_sigm_test(obs, cal, sigma):
    chi_2, le = chi_2_sigm(obs, cal, sigma)
    df = le -1
    critical_val= chi2.ppf(q = 0.95, # Encuentre el valor crítico para el 95% de confianza*
                      df = df) # df= grado de libertad
    p_chi2 = 1- chi2.cdf(chi_2, df = df )
    return chi_2, p_chi2, critical_val, df, chi_2/df

width = 2592   # 32 * x
height = 1944

rgb_data_s = np.zeros((height, width))
rgb_data_s2 = np.zeros((height, width))

pair_eh = 3.6  # eV
emax = 4300  # well capacity  4.1
emin = 5
strE = 'eh_'+str(pair_eh)+'eV'
print(strE)

source1 = 'Sr90'
source2 = 'Cs137'

nfrm = 1

if(source1 == 'Sr90'):
    winback1 = '254'
    fecha1 = '2023_Oct_24_23h'
    ev1 = 1514000//nfrm
   # ev1 = 1514
    tim1 = ev1/3028
#    ev1 = 3028000
    evt1 = winback1+'_ev'+str(ev1)
   # nfrm = 1
    z_count = 1

if(source2 == 'Cs137'):
    winback2 = '635'
    fecha2 = '2023_Oct_23_23h'
    ev2 = 3811000//nfrm
   # ev2 = 3811
    tim2 = ev2/7622
    #ev2 = 7622000
    #ev2 = 6859800
    evt2 = winback2+'_ev'+str(ev2)
    #nfrm = 1
    z_count = 1

level_z = list(range(z_count))
#level_z = [0]


path_main = 'C:/dat_2025/emit/'
path_name1 = 'data_'+source1+'_pixl_thickZ_2um_' + \
    str(z_count)+'level_'+str(winback1)+'_100um_ev'+str(ev1)+'_emit/'
path_name2 = 'data_'+source2+'_pixl_thickZ_2um_' + \
    str(z_count)+'level_'+str(winback2)+'_100um_ev'+str(ev2)+'_emit/'

dirfile = path_main+'count_sim_ALL0_'+source1+'_'+source2+'_'+str(nfrm)+'f_'+evt1+'_'+evt2

U_eV = 1000000

unidad = 'keV'

if(unidad == 'keV'):U_eV = 1000
if(unidad == 'MeV'):U_eV = 1000000

try:
    os.makedirs(dirfile)
except FileExistsError:
    pass

condit_edep = '_dif0' #'_zero','_dif0'

if(condit_edep=='_dif0'): strCond_dep = r'$E > 0$' #r'$E \neq 0$'
if(condit_edep=='_zero'): strCond_dep = r'$E = 0$'
if(condit_edep==''): strCond_dep = r'$E \geq 0$'

l_width = 4
font_siz = 28

lista_all = [

    'electron',
    'anti_nu_e',
    'gamma',

    'emit_electron',
    'emit_anti_nu_elec',
    'emit_gamma',
    'particle',
    'emit_all'
    ]

if(condit_edep == '_zero'):
    lista_e = [ 'electron', 'emit_electron', # 'particle', 'emit_all', # 'pixel_edep_emit'
           ]
else:
    lista_e = [  'electron', 'emit_electron', 'edep_electron'  # 'particle', 'emit_all', # 'pixel_edep_emit'
               ]
lista_ae = [ 'anti_nu_e', 'emit_anti_nu_elec', 'particle', 'emit_all',
            ]
lista_g = [  'edep_gamma',
           ]
lista_eg = [ 'particle', 'emit_e_g' ,'edep_particle'  ]

lista = lista_eg
plt_edep = 1
plt_emitdep = 0

bin_hist = 40

min_bin =((2-1)*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
#min_bin = 0.018


plt_err = 1
errors_sqrt = 1

log_y = 1
log_x = 1
if log_x: binlog = 1
else: binlog = 0
densi = 1


plt_sim = 0
plt_simgauss = True
plt_sim0 = 1
plt_all = 0

chi_label = 0
plt_data = 'between'
plt_data = 'step'

grid_ = False

lim_min_adc=101
lim_max_adc=1024
min_cam =((lim_min_adc-1)*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
sat_cam =(1023*(emax-emin)/1023.0 + emin)*pair_eh/U_eV


for partic_name in lista:
    for z in level_z:
        #partic_name = 'particle'

        path_1 = 'data_'+source1+'_pixl_thickZ_2um_'+'distSD_' + \
            str(z)+'mm_'+str(z_count)+'level'+'_csv_tar_bz'+'/'
        path_2 = 'data_'+source2+'_pixl_thickZ_2um_'+'distSD_' + \
            str(z)+'mm_'+str(z_count)+'level'+'_csv_tar_bz'+'/'

        if(partic_name == 'emit_electron' or partic_name == 'emit_anti_nu_elec'or partic_name == 'emit_gamma'):
            type_partic = '_'+partic_name
            colum = [0]

        if(partic_name == 'pixel_edep_zero_emit'):
            type_partic = '_'+partic_name
            colum = [0,1]

        if(partic_name == 'pixel_edep_emit'):
            type_partic = '_'+partic_name
            colum = [0,1]

        if(partic_name == 'particle' or partic_name == 'electron' or partic_name == 'anti_nu_e'or partic_name == 'gamma'):
            type_partic = '_edep'+condit_edep+'_all_'+partic_name+'_arrived_detector'
            #colum = [4]
            #colum = [2,4]
            if(partic_name == 'particle'):colum = [4,7,10]
            else:colum = [4,7,9]

        if(partic_name == 'edep_particle'):
            partic_name1 = 'particle'
            type_partic = '_edep'+condit_edep+'_all_'+partic_name1+'_arrived_detector'
            colum = [2,7,10,4]

        if(partic_name == 'edep_electron'):
            partic_name1 = 'electron'
            type_partic = '_edep'+condit_edep+'_all_'+partic_name1+'_arrived_detector'
            colum = [2,7,9,4]

        if(partic_name == 'edep_anti_nu_e'):
            partic_name1 = 'anti_nu_e'
            type_partic = '_edep'+condit_edep+'_all_'+partic_name1+'_arrived_detector'
            colum = [2,7,9,4]

        if(partic_name == 'edep_gamma'):
            partic_name1 = 'gamma'
            type_partic = '_edep'+condit_edep+'_all_'+partic_name1+'_arrived_detector'
            colum = [2,7,9,4]

        dat_cnt_adc1 = np.zeros((2, 1024),dtype=np.int64)
        dat_cnt_adc_f1 = np.zeros((2, 1000000),dtype=np.int64)
        dat_cnt_adc2 = np.zeros((2, 1024),dtype=np.int64)
        dat_cnt_adc_f2 = np.zeros((2, 1000000),dtype=np.int64)
        if(partic_name == 'emit_all' or partic_name == 'emit_e_g' or partic_name == 'emit_electron' or partic_name == 'emit_anti_nu_elec'or partic_name == 'emit_gamma'):
            dat1 = np.array([])
            dat2 = np.array([])
        else:
            dat1 = np.empty((0, len(colum)))
            dat2 = np.empty((0, len(colum)))
        data_cnt_adc_sim1 = np.zeros((2, 1000000),dtype=np.int64)
        data_cnt_adc_sim2 = np.zeros((2, 1000000),dtype=np.int64)
        for n in range(nfrm):
            print(partic_name)
            # for n in range(955,956):

            if(partic_name != 'emit_all' and partic_name != 'emit_e_g' ):
                print("mik ",partic_name)
                tmpfile1 = path_main+path_name1+path_1+'pixel' + \
                    str(n)+'_SD'+str(z)+'mm_'+source1+type_partic+'.csv'
                dat01 = np.loadtxt(tmpfile1, delimiter='\t', skiprows=1, usecols=colum)
                dat01 = np.atleast_1d(dat01)

                tmpfile2 = path_main+path_name2+path_2+'pixel' + \
                    str(n)+'_SD'+str(z)+'mm_'+source2+type_partic+'.csv'
                dat02 = np.loadtxt(tmpfile2, delimiter='\t', skiprows=1, usecols=colum)
                dat02 = np.atleast_1d(dat02)

                dat1 = np.concatenate((dat1, dat01), axis=0)
                dat2 = np.concatenate((dat2, dat02), axis=0)

                data1 = np.copy(dat1)
                data2 = np.copy(dat2)
                print(n, data1.size, data2.size)
                print(dat1.shape)


            if(partic_name == 'emit_e_g'):
                for emit in ['emit_electron','emit_gamma' ]:
                    type_partic = '_'+emit
                    colum = [0]
                    tmpfile1 = path_main+path_name1+path_1+'pixel' + \
                        str(n)+'_SD'+str(z)+'mm_'+source1+type_partic+'.csv'
                    dat01 = np.loadtxt(tmpfile1, delimiter='\t', skiprows=1, usecols=colum)
                    dat01 = np.atleast_1d(dat01)
                    print(dat01.size)
                    tmpfile2 = path_main+path_name2+path_2+'pixel' + \
                        str(n)+'_SD'+str(z)+'mm_'+source2+type_partic+'.csv'
                    dat02 = np.loadtxt(tmpfile2, delimiter='\t', skiprows=1, usecols=colum)
                    dat02 = np.atleast_1d(dat02)
                    print(dat02.size)
                    dat1 = np.concatenate((dat1, dat01), axis=0)
                    dat2 = np.concatenate((dat2, dat02), axis=0)
                    if(emit=='emit_electron'):
                        data1 = np.copy(dat1)
                        data2 = np.copy(dat2)
                    else:
                        data1 = np.concatenate((data1, dat1), axis=0)
                        data2 = np.concatenate((data2, dat2), axis=0)
                print(n, data1.size, data2.size)

            if(partic_name == 'emit_all'):
                for emit in ['emit_electron','emit_anti_nu_elec','emit_gamma' ]:
                    type_partic = '_'+emit
                    colum = [0]
                    tmpfile1 = path_main+path_name1+path_1+'pixel' + \
                        str(n)+'_SD'+str(z)+'mm_'+source1+type_partic+'.csv'
                    dat01 = np.loadtxt(tmpfile1, delimiter='\t', skiprows=1, usecols=colum)
                    dat01 = np.atleast_1d(dat01)
                    print(dat01.size)
                    tmpfile2 = path_main+path_name2+path_2+'pixel' + \
                        str(n)+'_SD'+str(z)+'mm_'+source2+type_partic+'.csv'
                    dat02 = np.loadtxt(tmpfile2, delimiter='\t', skiprows=1, usecols=colum)
                    dat02 = np.atleast_1d(dat02)
                    print(dat02.size)
                    dat1 = np.concatenate((dat1, dat01), axis=0)
                    dat2 = np.concatenate((dat2, dat02), axis=0)
                    if(emit=='emit_electron'):
                        data1 = np.copy(dat1)
                        data2 = np.copy(dat2)
                    else:
                        data1 = np.concatenate((data1, dat1), axis=0)
                        data2 = np.concatenate((data2, dat2), axis=0)
                print(n, data1.size, data2.size)
            print(data1.size)
            print(data2.size)


            '''
            if(partic_name == 'particle' or partic_name == 'electron' or partic_name == 'anti_nu_e'or partic_name == '_gamma'):
                if(condit_edep == '_dif0'):
                    data1=dat1[:,1]*(dat1[:,0]>0)
                    data2=dat2[:,1]*(dat2[:,0]>0)
                if(condit_edep == '_zero'):
                    data1=dat1[:,1]*(dat1[:,0]==0)
                    data2=dat2[:,1]*(dat2[:,0]==0)
                if(condit_edep == ''):
                    data1=dat1[:,1]
                    data2=dat2[:,1]
            '''
            if(partic_name == 'pixel_edep_zero_emit'):
                data1=dat1[:,1]
                data2=dat2[:,1]

            if(partic_name == 'pixel_edep_emit'):
                if(condit_edep == '_dif0'):
                    dat_t1=np.copy(dat1[:,1])
                    dat_t1[dat1[:,0]==0]=-1
                    data1=dat_t1[dat_t1>-1]

                    dat_t2=np.copy(dat2[:,1])
                    dat_t2[dat2[:,0]==0]=-1
                    data2=dat_t2[dat_t2>-1]

                if(condit_edep == '_zero'):
                    dat_t1=np.copy(dat1[:,1])
                    dat_t1[dat1[:,0]>0]=-1
                    data1=dat_t1[dat_t1>-1]

                    dat_t2=np.copy(dat2[:,1])
                    dat_t2[dat2[:,0]>0]=-1
                    data2=dat_t2[dat_t2>-1]

                if(condit_edep == ''):
                    data1=dat1[:,1]
                    data2=dat2[:,1]

            if(partic_name == 'particle' or partic_name == 'electron' or partic_name == 'anti_nu_e'or partic_name == 'gamma'):
                a1 = np.roll(dat1[:,0], 1)  # rotate right
                a2 = np.roll(dat2[:,0], 1)  # rotate right
                #print(a1)
                #print(a2)
                b1 = np.roll(dat1[:,1], 1)  # rotate right
                b2 = np.roll(dat2[:,1], 1)  # rotate right
                #print(b1)
                #print(b2)
                c1 = np.roll(dat1[:,2], 1)  # rotate right
                c2 = np.roll(dat2[:,2], 1)  # rotate right
                #print(c1)
                #print(c2)
                a1[(dat1[:,0]==a1) & (dat1[:,1]==b1) & (dat1[:,2]==c1)]=-1
                a1=np.roll(a1, -1) # rotate left
                a2[(dat2[:,0]==a2) & (dat2[:,1]==b2) & (dat2[:,2]==c2)]=-1
                a2=np.roll(a2, -1) # rotate left
                #print(a1)
                #print(a1.size, b1.size, c1.size)
                #print(a2)
                #print(a2.size, b2.size, c2.size)
                d1=a1[a1>-1]
                d2=a2[a2>-1]
                #print(d1)
                #print(d2)
                data1 = d1 ; data2 = d2
                data1=np.copy(dat1[:,0]) ; data2=np.copy(dat2[:,0])

            if(partic_name == 'edep_particle' or partic_name == 'edep_electron' or partic_name == 'edep_anti_nu_e'or partic_name == 'edep_gamma'):
                e1 = np.roll(dat1[:,0], 1)  # rotate right
                e2 = np.roll(dat2[:,0], 1)  # rotate right

                aa1 = np.roll(dat1[:,3], 1)  # rotate right
                aa2 = np.roll(dat2[:,3], 1)  # rotate right

                b1 = np.roll(dat1[:,1], 1)  # rotate right
                b2 = np.roll(dat2[:,1], 1)  # rotate right

                c1 = np.roll(dat1[:,2], 1)  # rotate right
                c2 = np.roll(dat2[:,2], 1)  # rotate right

                e1[(dat1[:,3]==aa1) & (dat1[:,1]==b1) & (dat1[:,2]==c1)]=-1
                e1=np.roll(e1, -1) # rotate left
                e2[(dat2[:,3]==aa2) & (dat2[:,1]==b2) & (dat2[:,2]==c2)]=-1
                e2=np.roll(e2, -1) # rotate left
                ede1=e1[e1>-1]
                ede2=e2[e2>-1]

                data1 = ede1 ; data2 = ede2
                data1=np.copy(dat1[:,0]) ; data2=np.copy(dat2[:,0])

            if(partic_name == 'pixel_edep_zero_emit' or partic_name == 'pixel_edep_emit'
               or partic_name == 'edep_particle' or partic_name == 'edep_electron'
               or partic_name == 'edep_anti_nu_e'or partic_name == 'edep_gamma'):
                if(unidad == 'keV'):data1 = data1; data2 = data2
                if(unidad == 'MeV'):data1 = data1/1000; data2 = data2/1000
            else:
                if(unidad == 'keV'):data1 = data1*1000; data2 = data2*1000
                if(unidad == 'MeV'):data1 = data1; data2 = data2

            if(data1.size == 0):
                dat1 = 0.0
                data_adc1 = (U_eV*dat1 / pair_eh)
                dat_adc1 =  np.uint64(np.clip((data_adc1-emin)*1023.0/(emax-emin), 0, 1023))
                dat_adc_f1 = np.uint64(np.clip((data_adc1-emin)*1023.0/(emax-emin), 0, 10000000))
                print('adc_max1: ', 0)
            else:
                print('max = ', data1.max(), unidad )
                #if(partic_name == 'pixel'):print('max = ',data.max(), 'keV')
                data_adc1 = (data1 / pair_eh)*U_eV
                dat_adc1 =  np.uint64(np.clip((data_adc1-emin)*1023.0/(emax-emin), 0, 1023))
                dat_adc_f1 = np.uint64(np.clip((data_adc1-emin)*1023.0/(emax-emin), 0, 2*int(data_adc1.max())))
                #dat_adc_f1 = np.array([0, dat_adc_f1])
                print('adc_max1: ', dat_adc_f1.max())

            if(data2.size == 0):
                dat2 = 0.0
                data_adc2 = (U_eV*dat2 / pair_eh)
                dat_adc2 =  np.uint64(np.clip((data_adc2-emin)*1023.0/(emax-emin), 0, 1023))
                dat_adc_f2 = np.uint64(
                    np.clip((data_adc2-emin)*1023.0/(emax-emin), 0, 10000000))
                print('adc_max2: ', 0)
            else:
                print('max = ', data2.max(), unidad )
                #if(partic_name == 'pixel'):print('max = ',data.max(), 'keV')
                data_adc2 = (data2 / pair_eh)*U_eV
                dat_adc2 =  np.uint64(np.clip((data_adc2-emin)*1023.0/(emax-emin), 0, 1023))
                dat_adc_f2 = np.uint64(np.clip((data_adc2-emin)*1023.0/(emax-emin), 0, 2*int(data_adc2.max())))
                #dat_adc_f2 = np.array([0, dat_adc_f2])
                print('adc_max2: ', dat_adc_f2.max())


            adc_n1 = np.unique(dat_adc1, return_counts=True)
            adc_n_f1 = np.unique(dat_adc_f1, return_counts=True)
            adc_n2 = np.unique(dat_adc2, return_counts=True)
            adc_n_f2 = np.unique(dat_adc_f2, return_counts=True)
            """
            dat_cnt_adc1 = np.zeros((2, 1024),dtype=np.int64)
            dat_cnt_adc2 = np.zeros((2, 1024),dtype=np.int64)
            dat_cnt_adc_f1 = np.zeros((2, dat1.size),dtype=np.int64)
            dat_cnt_adc_f2 = np.zeros((2, dat2.size),dtype=np.int64)
            max_index1 = int(np.max(adc_n_f1[0])) + 1  # +1 porque los índices empiezan en 0
            max_index2 = int(np.max(adc_n_f2[0])) + 1  # +1 porque los índices empiezan en 0
            dat_cnt_adc_f1 = np.zeros((2, max_index1),dtype=np.int64)
            dat_cnt_adc_f2 = np.zeros((2, max_index2),dtype=np.int64)

            """
            for j in range(adc_n1[0].size):
                dat_cnt_adc1[0,int(adc_n1[0][j])] = (adc_n1[0][j])
                dat_cnt_adc1[1,int(adc_n1[0][j])] = dat_cnt_adc1[1,int(adc_n1[0][j])]+(adc_n1[1][j])

            for j in range(adc_n_f1[0].size):
                dat_cnt_adc_f1[0,int(adc_n_f1[0][j])] = (adc_n_f1[0][j])
                dat_cnt_adc_f1[1,int(adc_n_f1[0][j])] = dat_cnt_adc_f1[1,int(adc_n_f1[0][j])]+(adc_n_f1[1][j])

            dat_cnt_adc1[:1].max()
            dat_cnt_adc1[1:].max()
            print(n, dat_cnt_adc1[:1].max(),dat_cnt_adc1[1:].max())
            dat_cnt_adc_f1[:1].max()
            dat_cnt_adc_f1[1:].max()
            print(n, dat_cnt_adc_f1[:1].max(),dat_cnt_adc_f1[1:].max())

            for j in range(adc_n2[0].size):
                dat_cnt_adc2[0,int(adc_n2[0][j])] = (adc_n2[0][j])
                dat_cnt_adc2[1,int(adc_n2[0][j])] = dat_cnt_adc2[1,int(adc_n2[0][j])]+(adc_n2[1][j])

            for j in range(adc_n_f2[0].size):
                dat_cnt_adc_f2[0,int(adc_n_f2[0][j])] = (adc_n_f2[0][j])
                dat_cnt_adc_f2[1,int(adc_n_f2[0][j])] = dat_cnt_adc_f2[1,int(adc_n_f2[0][j])]+(adc_n_f2[1][j])

            dat_cnt_adc2[:1].max()
            dat_cnt_adc2[1:].max()
            print(n, dat_cnt_adc2[:1].max(),dat_cnt_adc2[1:].max())
            dat_cnt_adc_f2[:1].max()
            dat_cnt_adc_f2[1:].max()
            print(n, dat_cnt_adc_f2[:1].max(),dat_cnt_adc_f2[1:].max())

        if(data1.size>0):
            event_dect1 = dat_cnt_adc1[1].sum()
            #event_dect_f1 = dat_cnt_adc_f1[1].sum()
            event_dect_f1 = dat_adc1.size
            print('#envent1 : ', dat_cnt_adc_f1[1].sum(), dat_adc1.size)
        else:
            event_dect1 = 0
            event_dect_f1 = 0
        if(data2.size>0):
            event_dect2 = dat_cnt_adc2[1].sum()
            #event_dect_f2 = dat_cnt_adc_f2[1].sum()
            event_dect_f2 = dat_adc2.size
            print('#envent2 : ', dat_cnt_adc_f2[1].sum(), dat_adc2.size)
        else:
            event_dect2 = 0
            event_dect_f2 = 0

        dirsave_plt_sim1 ='adc_count_'+ 'sim_'+source1+'_'+str(nfrm)+'f' + type_partic
        try:
            os.makedirs(dirfile +'/'+ dirsave_plt_sim1)
        except FileExistsError:
            pass
        np.savez_compressed(dirfile +'/'+ dirsave_plt_sim1 +'/'+ source1+'_'+partic_name+'_ev'+str(ev1)+'_z'+str(z).zfill(3)+'.npz', dat_cnt_adc1)
        np.savez_compressed(dirfile +'/'+ dirsave_plt_sim1 +'/'+ source1+'_full_'+partic_name+'_ev'+str(ev1)+'_z'+str(z).zfill(3)+'.npz', dat_cnt_adc_f1)
        print(dirfile +'/'+ dirsave_plt_sim1 +'/'+ source1+'_'+partic_name+'_ev'+str(ev1)+'_z'+str(z).zfill(3)+'.npz')

        dirsave_plt_sim2 ='adc_count_'+ 'sim_'+source2+'_'+str(nfrm)+'f' + type_partic
        try:
            os.makedirs(dirfile +'/'+ dirsave_plt_sim2)
        except FileExistsError:
            pass
        np.savez_compressed(dirfile +'/'+ dirsave_plt_sim2 +'/'+ source2+'_'+partic_name+'_ev'+str(ev2)+'_z'+str(z).zfill(3)+'.npz', dat_cnt_adc2)
        np.savez_compressed(dirfile +'/'+ dirsave_plt_sim2 +'/'+ source2+'_full_'+partic_name+'_ev'+str(ev2)+'_z'+str(z).zfill(3)+'.npz', dat_cnt_adc_f2)
        print(dirfile +'/'+ dirsave_plt_sim2 +'/'+ source2+'_'+partic_name+'_ev'+str(ev2)+'_z'+str(z).zfill(3)+'.npz')

    dat_cnt1 = np.copy(dat_cnt_adc_f1.astype(float))
    dat_cnt2 = np.copy(dat_cnt_adc_f2.astype(float))
    dat_cnt1[0] = (dat_cnt_adc_f1.astype(float)[0]*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
    dat_cnt2[0] = (dat_cnt_adc_f2.astype(float)[0]*(emax-emin)/1023.0 + emin)*pair_eh/U_eV

    dirsave_pltE_sim01 ='E0_count_'+ 'sim_'+source1+'_'+str(nfrm)+'f'
    try:
        os.makedirs(dirfile +'/'+ dirsave_pltE_sim01)
    except FileExistsError:
        pass
    np.savez_compressed(dirfile +'/'+ dirsave_pltE_sim01 +'/E0'+ source1+'_'+partic_name+'_ev'+str(ev1)+'_z'+str(z).zfill(3)+'.npz', dat_cnt1)
    print(dirfile +'/'+ dirsave_pltE_sim01 +'/E0'+ source1+'_'+partic_name+'_ev'+str(ev1)+'_z'+str(z).zfill(3)+'.npz')

    dirsave_pltE_sim02 ='E0_count_'+ 'sim_'+source2+'_'+str(nfrm)+'f'
    try:
        os.makedirs(dirfile +'/'+ dirsave_pltE_sim02)
    except FileExistsError:
        pass
    np.savez_compressed(dirfile +'/'+ dirsave_pltE_sim02 +'/E0'+ source2+'_'+partic_name+'_ev'+str(ev2)+'_z'+str(z).zfill(3)+'.npz', dat_cnt2)
    print(dirfile +'/'+ dirsave_pltE_sim02 +'/E0'+ source2+'_'+partic_name+'_ev'+str(ev2)+'_z'+str(z).zfill(3)+'.npz')


    if(partic_name == 'particle'):
        dat_cnt_adc_pt1 = np.copy(dat_cnt_adc_f1.astype(float))
        dat_cnt_adc_pt2 = np.copy(dat_cnt_adc_f2.astype(float))
        dat_cnt_adc_pt1[0] = (dat_cnt_adc_f1.astype(float)[0]*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
        dat_cnt_adc_pt2[0] = (dat_cnt_adc_f2.astype(float)[0]*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
        dat_all_emit_pt1 = dat_cnt1[0]
        weights_all_emit_pt1 = dat_cnt1[1]
        event_dect_all_emit_pt1 = event_dect_f1
        dat_all_emit_pt2 = dat_cnt2[0]
        weights_all_emit_pt2 = dat_cnt2[1]
        event_dect_all_emit_pt2 = event_dect_f2
        label_plt = 'all_particles_emit_on_detect' + condit_edep
        colors1 = 'C0'
        colors2 = 'C9'

    if(partic_name == 'electron'):
        dat_cnt_adc_e1 = np.copy(dat_cnt_adc_f1.astype(float))
        dat_cnt_adc_e2 = np.copy(dat_cnt_adc_f2.astype(float))
        dat_cnt_adc_e1[0] = (dat_cnt_adc_f1.astype(float)[0]*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
        dat_cnt_adc_e2[0] = (dat_cnt_adc_f2.astype(float)[0]*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
        dat_all_emit_e1 = dat_cnt1[0]
        weights_all_emit_e2 = dat_cnt1[1]
        event_dect_all_emit_e1 = event_dect_f1
        dat_all_emit_e2 = dat_cnt2[0]
        weights_all_emit_e2 = dat_cnt2[1]
        event_dect_all_emit_e2 = event_dect_f2
        label_plt = ' all electrons emit on detect' + condit_edep
        colors1 = 'C0'
        colors2 = 'C6'

    if(partic_name == 'anti_nu_e'):
        dat_cnt_adc_ae1 = np.copy(dat_cnt_adc_f1.astype(float))
        dat_cnt_adc_ae2 = np.copy(dat_cnt_adc_f2.astype(float))
        dat_cnt_adc_ae1[0] = (dat_cnt_adc_f1.astype(float)[0]*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
        dat_cnt_adc_ae2[0] = (dat_cnt_adc_f2.astype(float)[0]*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
        dat_all_emit_ae1 = dat_cnt1[0]
        weights_all_emit_ae1 = dat_cnt1[1]
        event_dect_all_emit_ae1 = event_dect_f1
        dat_all_emit_ae2 = dat_cnt2[0]
        weights_all_emit_ae2 = dat_cnt2[1]
        event_dect_all_emit_ae2 = event_dect_f2
        label_plt = ' all anti_nu_elec emit on detect' + condit_edep
        colors1 = 'C7'
        colors2 = 'C7'

    if(partic_name == 'gamma'):
        dat_cnt_adc_g1 = np.copy(dat_cnt_adc_f1.astype(float))
        dat_cnt_adc_g2 = np.copy(dat_cnt_adc_f2.astype(float))
        dat_cnt_adc_g1[0] = (dat_cnt_adc_f1.astype(float)[0]*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
        dat_cnt_adc_g2[0] = (dat_cnt_adc_f2.astype(float)[0]*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
        dat_all_emit_g1 = dat_cnt1[0]
        weights_all_emit_g1 = dat_cnt1[1]
        event_dect_all_emit_g1 = event_dect_f1
        dat_all_emit_g2 = dat_cnt2[0]
        weights_all_emit_g2 = dat_cnt2[1]
        event_dect_all_emit_g2 = event_dect_f2
        label_plt = ' all gammas emit on detect' + condit_edep
        colors1 = 'C2'
        colors2 = 'C8'

    if(partic_name == 'emit_electron'):
        emit_cnt_adc_e1 = np.copy(dat_cnt_adc_f1.astype(float))
        emit_cnt_adc_e2 = np.copy(dat_cnt_adc_f2.astype(float))
        emit_cnt_adc_e1[0] = (dat_cnt_adc_f1.astype(float)[0]*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
        emit_cnt_adc_e2[0] = (dat_cnt_adc_f2.astype(float)[0]*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
        dat_emit_e1 = dat_cnt1[0]
        weights_emit_e1 = dat_cnt1[1]
        event_emit_e1 = event_dect_f1
        dat_emit_e2 = dat_cnt2[0]
        weights_emit_e2 = dat_cnt2[1]
        event_emit_e2 = event_dect_f2
        label_plt = ' all electrons emit'
        colors1 = 'C1'
        colors2 = 'C8'

    if(partic_name == 'emit_anti_nu_elec'):
        emit_cnt_adc_ae1 = np.copy(dat_cnt_adc_f1.astype(float))
        emit_cnt_adc_ae2 = np.copy(dat_cnt_adc_f2.astype(float))
        emit_cnt_adc_ae1[0] = (dat_cnt_adc_f1.astype(float)[0]*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
        emit_cnt_adc_ae2[0] = (dat_cnt_adc_f2.astype(float)[0]*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
        dat_emit_ae1 = dat_cnt1[0]
        weights_emit_ae1 = dat_cnt1[1]
        event_emit_ae1 = event_dect_f1
        dat_emit_ae2 = dat_cnt2[0]
        weights_emit_ae2 = dat_cnt2[1]
        event_emit_ae2 = event_dect_f2
        label_plt = ' all anti_nu_elec emit'
        colors1 = 'C4'
        colors2 = 'C8'

    if(partic_name == 'emit_gamma'):
        emit_cnt_adc_g1 = np.copy(dat_cnt_adc_f1.astype(float))
        emit_cnt_adc_g2 = np.copy(dat_cnt_adc_f2.astype(float))
        emit_cnt_adc_g1[0] = (dat_cnt_adc_f1.astype(float)[0]*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
        emit_cnt_adc_g2[0] = (dat_cnt_adc_f2.astype(float)[0]*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
        dat_emit_g1 = dat_cnt1[0]
        weights_emit_g1 = dat_cnt1[1]
        event_emit_g1 = event_dect_f1
        dat_emit_g2 = dat_cnt2[0]
        weights_emit_g2 = dat_cnt2[1]
        event_emit_g2 = event_dect_f2
        label_plt = ' all gammas emit'
        colors1 = 'C8'
        colors2 = 'g'

    if(partic_name == 'emit_all'):
        emit_cnt_adc_all1 = np.copy(dat_cnt_adc_f1.astype(float))
        emit_cnt_adc_all2 = np.copy(dat_cnt_adc_f2.astype(float))
        emit_cnt_adc_all1[0] = (dat_cnt_adc_f1.astype(float)[0]*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
        emit_cnt_adc_all2[0] = (dat_cnt_adc_f2.astype(float)[0]*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
        dat_emit_all1 = dat_cnt1[0]
        weights_emit_all1 = dat_cnt1[1]
        event_emit_all1 = event_dect_f1
        dat_emit_all2 = dat_cnt2[0]
        weights_emit_all2 = dat_cnt2[1]
        event_emit_all2 = event_dect_f2
        label_plt = ' all ' + r'$e^-$, $\overline{\nu}_e$ , $\gamma$ '+ 'emit'
        colors1 = 'k'
        colors2 = 'b'

    if(partic_name == 'emit_e_g'):
        emit_cnt_adc_e_g1 = np.copy(dat_cnt_adc_f1.astype(float))
        emit_cnt_adc_e_g2 = np.copy(dat_cnt_adc_f2.astype(float))
        emit_cnt_adc_e_g1[0] = (dat_cnt_adc_f1.astype(float)[0]*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
        emit_cnt_adc_e_g2[0] = (dat_cnt_adc_f2.astype(float)[0]*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
        dat_emit_e_g1 = dat_cnt1[0]
        weights_emit_e_g1 = dat_cnt1[1]
        event_emit_e_g1 = event_dect_f1
        dat_emit_e_g2 = dat_cnt2[0]
        weights_emit_e_g2 = dat_cnt2[1]
        event_emit_e_g2 = event_dect_f2
        label_plt = 'all_e_g_emit'
        colors1 = 'C0'
        colors2 = 'C7'

    if(partic_name == 'edep_particle'):
        edep_cnt_adc_pt1 = np.copy(dat_cnt_adc_f1.astype(float))
        edep_cnt_adc_pt2 = np.copy(dat_cnt_adc_f2.astype(float))
        edep_cnt_adc_pt1[0] = (dat_cnt_adc_f1.astype(float)[0]*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
        edep_cnt_adc_pt2[0] = (dat_cnt_adc_f2.astype(float)[0]*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
        dat_edep_pt1 = dat_cnt1[0]
        weights_edep_pt1 = dat_cnt1[1]
        event_edep_pt1 = event_dect_f1
        dat_edep_pt2 = dat_cnt2[0]
        weights_edep_pt2 = dat_cnt2[1]
        event_edep_pt2 = event_dect_f2
        label_plt = 'all_particles_edep_on_detect' + condit_edep
        colors1 = 'C0'
        colors2 = 'C9'

    if(partic_name == 'edep_electron'):
        edep_cnt_adc_e1 = np.copy(dat_cnt_adc_f1.astype(float))
        edep_cnt_adc_e2 = np.copy(dat_cnt_adc_f2.astype(float))
        edep_cnt_adc_e1[0] = (dat_cnt_adc_f1.astype(float)[0]*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
        edep_cnt_adc_e2[0] = (dat_cnt_adc_f2.astype(float)[0]*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
        dat_edep_e1 = dat_cnt1[0]
        weights_edep_e1 = dat_cnt1[1]
        event_edep_e1 = event_dect_f1
        dat_edep_e2 = dat_cnt2[0]
        weights_edep_e2 = dat_cnt2[1]
        event_edep_e2 = event_dect_f2
        label_plt = ' all electrons edep on detect' + condit_edep
        colors1 = 'C0'
        colors2 = 'C6'

    if(partic_name == 'edep_anti_nu_e'):
        edep_cnt_adc_ae1 = np.copy(dat_cnt_adc_f1.astype(float))
        edep_cnt_adc_ae2 = np.copy(dat_cnt_adc_f2.astype(float))
        edep_cnt_adc_ae1[0] = (dat_cnt_adc_f1.astype(float)[0]*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
        edep_cnt_adc_ae2[0] = (dat_cnt_adc_f2.astype(float)[0]*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
        dat_edep_ae1 = dat_cnt1[0]
        weights_edep_ae1 = dat_cnt1[1]
        event_edep_ae1 = event_dect_f1
        dat_edep_ae2 = dat_cnt2[0]
        weights_edep_ae2 = dat_cnt2[1]
        event_edep_ae2 = event_dect_f2
        label_plt = ' all anti_nu_elec edep on detect' + condit_edep
        colors1 = 'C7'
        colors2 = 'C7'

    if(partic_name == 'edep_gamma'):
        edep_cnt_adc_g1 = np.copy(dat_cnt_adc_f1.astype(float))
        edep_cnt_adc_g2 = np.copy(dat_cnt_adc_f2.astype(float))
        edep_cnt_adc_g1[0] = (dat_cnt_adc_f1.astype(float)[0]*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
        edep_cnt_adc_g2[0] = (dat_cnt_adc_f2.astype(float)[0]*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
        dat_edep_g1 = dat_cnt1[0]
        weights_edep_g1 = dat_cnt1[1]
        event_edep_g1 = event_dect_f1
        dat_edep_g2 = dat_cnt2[0]
        weights_edep_g2 = dat_cnt2[1]
        event_edep_g2 = event_dect_f2
        label_plt = ' all gammas edep on detect' + condit_edep
        colors1 = 'C2'
        colors2 = 'C8'

    if(partic_name == 'pixel_edep_emit'):
        emit_cnt_adc_1 = np.copy(dat_cnt_adc_f1.astype(float))
        emit_cnt_adc_2 = np.copy(dat_cnt_adc_f2.astype(float))
        emit_cnt_adc_1[0] = (dat_cnt_adc_f1.astype(float)[0]*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
        emit_cnt_adc_2[0] = (dat_cnt_adc_f2.astype(float)[0]*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
        dat_emit_1 = dat_cnt1[0]
        weights_emit_1 = dat_cnt1[1]
        event_emit_1 = event_dect_f1
        dat_emit_2 = dat_cnt2[0]
        weights_emit_2 = dat_cnt2[1]
        event_emit_2 = event_dect_f2
        label_plt = ' all particls emit edep'
        colors1 = 'C7'
        colors2 = 'y'

    if(partic_name == 'pixel_edep_zero_emit'):
        emit_cnt_adc_01 = np.copy(dat_cnt_adc_f1.astype(float))
        emit_cnt_adc_02 = np.copy(dat_cnt_adc_f2.astype(float))
        emit_cnt_adc_01[0] = (dat_cnt_adc_f1.astype(float)[0]*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
        emit_cnt_adc_02[0] = (dat_cnt_adc_f2.astype(float)[0]*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
        dat_emit_01 = dat_cnt1[0]
        weights_emit_01 = dat_cnt1[1]
        event_dect_emit_01 = event_dect_f1
        dat_emit_02 = dat_cnt2[0]
        weights_emit_02 = dat_cnt2[1]
        event_dect_emit_02 = event_dect_f2
        label_plt = ' all particls emit edep_zero'
        colors1 = 'C6'
        colors2 = 'c'

    if(bin_hist == 0): strbn = ''
    if(bin_hist > 0): strbn = '_bin'+str(bin_hist)

    max_adc = np.max(np.array(( dat_cnt_adc_f1[0].max(), dat_cnt_adc_f2[0].max())))+1
    max_adc = np.uint64(max_adc)
    #min_adc = 200
    min_adc = 1#1, 3
    #min_adc = np.int64((-1*(emax-emin)/1023.0 + emin)*pair_eh)

    nbins_adc = round(np.min(np.array((dat_cnt_adc_f1[0].max(), dat_cnt_adc_f2[0].max()  )))-min_adc+1)

    if(bin_hist>1 and bin_hist<nbins_adc):
        nbins_adc = (np.arange(min_adc, max_adc+1, (max_adc-min_adc)/bin_hist )*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
    else:
        nbins_adc = (np.arange(min_adc, max_adc+1 )*(emax-emin)/1023.0 + emin)*pair_eh/U_eV

    ###############################################################################
    ###############################################################################

    #plt.figure(figsize=(15, 9.5))
    fig, ax0 = plt.subplots(figsize=(14.5, 9.5))  # , sharex=True )

    max_bin = np.max(np.array(( data1.max(), data2.max())))
    max_bin = (1.10)*max_bin
    print(max_bin)


    if(binlog ==False): nbins = (np.arange(min_bin, max_bin, (max_bin-min_bin)/(bin_hist + 1) ))
    if(binlog ==True): nbins = np.logspace(np.log10(min_bin), np.log10(max_bin), bin_hist + 1)
    print(nbins)

    if(plt_sim0 == True or plt_all == True):
        hist01 = np.histogram(dat_cnt1[0], weights=dat_cnt1[1], bins=nbins)
        count01, bin01 = hist01[0], hist01[1]
        if binlog:
            bincenters01 = np.sqrt(bin01[:-1] * bin01[1:])
        else:
            bincenters01 = 0.5*(bin01[1:]+bin01[:-1])
        norm_01 = np.sum(count01 * np.diff(bin01))

        if(densi==False):
            sim_w01 = count01
            if(errors_sqrt == True): err_sim_w01 = np.sqrt(0.04*count01*count01 + count01)
            else: err_sim_w01 = (np.sqrt(count01))
        if(densi==True):
            sim_w01 = count01 / norm_01
            if(errors_sqrt == True): err_sim_w01 = np.sqrt(0.04*count01*count01 + count01) / norm_01
            else: err_sim_w01 = (np.sqrt(count01)) / norm_01


        hist02 = np.histogram(dat_cnt2[0], weights=dat_cnt2[1], bins=nbins)
        count02, bin02 = hist02[0], hist02[1]
        if binlog:
            bincenters02 = np.sqrt(bin02[:-1] * bin02[1:])
        else:
            bincenters02 = 0.5*(bin02[1:]+bin02[:-1])
        norm_02 = np.sum(count02 * np.diff(bin02))

        if(densi==False):
            sim_w02 = count02
            if(errors_sqrt == True): err_sim_w02 = np.sqrt(0.04*count02*count02 + count02)
            else: err_sim_w02 = (np.sqrt(count02))
        if(densi==True):
            sim_w02 = count02 / norm_02
            if(errors_sqrt == True): err_sim_w02 = np.sqrt(0.04*count02*count02 + count02) / norm_02
            else: err_sim_w02 = (np.sqrt(count02)) / norm_02

        err_sim0 = np.sqrt(err_sim_w01*err_sim_w01 + err_sim_w02*err_sim_w02 )
        ch2_sim0 = chi_2_sigm_test(sim_w01, sim_w02, err_sim0)
        tmp_str_sim0  = 'Chi2 test: ' + r'%.2f'%ch2_sim0[0] \
                    +'       '+'ndf: ' + r'%.2f'%ch2_sim0[3] \
                    +'\n'+'Chi2/ndf : ' + r'%.2f'%ch2_sim0[4]
        print(tmp_str_sim0)
        if chi_label == True :
            chi_str_sim0 = r'$\chi^2_\nu$ = ' + r'%.2f'%ch2_sim0[4]
        else: chi_str_sim0 = ''

        if plt_data == 'step' :
            ax0.step( np.insert(bin01, 0, 0), np.concatenate(([0], sim_w01, [0])), where='post', color='C0',
                markersize=6, alpha=0.6, linewidth=2.5,  label = 'Sim '+source1 )

            ax0.step( np.insert(bin02, 0, 0), np.concatenate(([0], sim_w02, [0])), where='post', color='C7',
                markersize=6, alpha=0.6, linewidth=2.5,  label = 'Sim '+source2 )
        if plt_data == 'between' :
            ax0.fill_between(bincenters01, sim_w01 - err_sim_w01, sim_w01 + err_sim_w01,
                    color='C0', linewidth=l_width*2/3, alpha=0.6,  label = 'Sim '+source1 )
            ax0.fill_between(bincenters02, sim_w02 - err_sim_w02, sim_w02 + err_sim_w02,
                    color='C7', linewidth=l_width*2/3, alpha=0.6,  label = 'Sim '+source2 )
        if plt_err == True :
            ax0.errorbar(bincenters01, sim_w01, yerr=err_sim_w01, fmt='C0'+'o',
                                    ecolor='C0', markersize=6, alpha=0.4, linewidth=2.5)
            ax0.errorbar(bincenters02, sim_w02, yerr=err_sim_w02, fmt='C7'+'o',
                                    ecolor='C7', markersize=6, alpha=0.4, linewidth=2.5,
                                    label = chi_str_sim0
                                    )

    if(plt_sim0 != True or plt_all == True):
        ax0.hist(dat_cnt1[0], weights=dat_cnt1[1], bins=nbins, density=densi, histtype='step', log=log_y, color='C0', linewidth=2.5,
                              alpha=0.4, label=' Sim '+source1 + label_plt
                              #+ '\n time = ' + r'%.2f' % (tim1) + ' s'
                              + '\n evt = ' + str(event_dect_f1)
                              )
        ax0.hist(dat_cnt2[0], weights=dat_cnt2[1], bins=nbins, density=densi, histtype='step', log=log_y, color='C7', linewidth=2.5,
                              alpha=0.4, label=' Sim '+source2 + label_plt
                              + '\n time = ' + r'%.2f' % (tim2) + ' s'
                              + '\n evt = ' + str(event_dect_f2)
                          )
    ax0.axvline(x = sat_cam, color = 'k', linestyle="--", linewidth=2.5)
    ax0.axvline(x = min_cam, color = 'k', linestyle="--", linewidth=2.5)

    # Agregar etiquetas en la parte superior
    if log_x:
        ax0.text(min_cam, 2*ax0.get_ylim()[1], '100 ADC', ha='center', va='bottom', fontsize=font_siz-4)
        ax0.text(sat_cam, 2*ax0.get_ylim()[1], '1023 ADC', ha='center', va='bottom', fontsize=font_siz-4)

    else:
        ax0.text(min_cam, 2*ax0.get_ylim()[1], '100 ADC', ha='right', va='bottom', fontsize=font_siz-4)
        ax0.text(sat_cam, 2*ax0.get_ylim()[1], '1023 ADC', ha='left', va='bottom', fontsize=font_siz-4)

    ax0.grid(True)
    if(log_y == True):
        ax0.set_yscale('log')
    if(log_x == True):
        ax0.set_xscale('log')
    ax0.tick_params(axis='x', labelsize=font_siz)
    ax0.tick_params(axis='y', labelsize=font_siz)
    #if(condit_edep == '_zero' and partic_name == 'electron'):ax0.set_ylim(0,10e5)
    #if(condit_edep != '_zero' and partic_name == 'electron'):ax0.set_ylim(0,10e5)
    #if(partic_name != 'electron'):ax0.set_ylim(0,20e3)
    #if(source2 == 'Cs137' and partic_name == 'electron'):ax0.set_ylim(0,10e2)
    #if(source2 == 'Cs137' and partic_name != 'electron'):ax0.set_ylim(0,10e7)
    #ax0.set_ylim(0,10e4)
    ax0.set_xlabel('Energy ('+unidad+')', fontsize=font_siz-2)
    if(densi==False):ax0.set_ylabel('counts in all frame', fontsize=font_siz)
    if(densi==True):ax0.set_ylabel('normalized counts in all frames', fontsize=font_siz)
    ax0.legend(loc='best' , fontsize=font_siz-6)
    if(log_y == False):
        titulo = 'energy' + '_z' + \
          str(z).zfill(3) + '_t'+r'%.0f' % (tim1) + 's' + strbn
    if(log_y == True):
        titulo = 'energy'+'_z' + \
          str(z).zfill(3) + '_log_y'+'_t'+r'%.0f' % (tim1) + 's' + strbn
    if(log_x == True and log_y == True):
        titulo = 'energy'+'_z' + \
            str(z).zfill(3) + '_log_xy'+'_t'+r'%.0f' % (tim2) + 's' + strbn
    #fig.suptitle(titulo , fontsize=font_siz)

    fig.tight_layout()
    #plt.savefig(dirfile + '/'+'plot_'+source+'_'+titulo+'_' +
    #             str(min_adc)+'ADC'+type_partic+'.png', dpi=150)
    if(plt_sim0 == True and lista == lista_eg):
        plt.savefig(dirfile + '/'+'plot_'+plt_data+'_'+titulo+'_'+label_plt +'_sim_simple'+'.png', dpi=150)
        plt.savefig(dirfile + '/'+'plot_'+plt_data+'_'+titulo+'_'+label_plt +'_sim_simple'+'.pdf', dpi=150)

    plt.show()
    #plt.clf()
    #plt.close()


###############################################################################
###############################################################################

#plt.figure(figsize=(15, 9.5))
fig, ax = plt.subplots(figsize=(14.5, 9.5))  # , sharex=True )


if(bin_hist == 0): strbn = ''
if(bin_hist > 0): strbn = '_bin'+str(bin_hist)

if(lista == lista_all):
    max_bin = np.max(np.array(( dat_all_emit_pt1.max(), dat_all_emit_pt2.max() ) ))
    print(max_bin)
if(lista == lista_e):
    max_bin = np.max(np.array(( dat_all_emit_e1.max(), dat_all_emit_e2.max() ) ))
    print(max_bin)
if(lista == lista_ae):
    max_bin = np.max(np.array(( dat_all_emit_ae1.max(), dat_all_emit_ae2.max() ) ))
    print(max_bin)
if(lista == lista_g):
    max_bin = np.max(np.array(( dat_all_emit_g1.max(), dat_all_emit_g2.max() ) ))
    print(max_bin)
if(lista == lista_eg):
    max_bin = np.max(np.array(( dat_emit_e_g1.max(), dat_emit_e_g2.max(),
    dat_all_emit_pt1.max(), dat_all_emit_pt2.max(), dat_edep_pt1.max(), dat_edep_pt2.max()  ) ))
    print(max_bin)


max_bin = (1.10)*max_bin
print(max_bin)
#min_bin = 0.
#nbins = (np.arange(min_bin, max_bin, (max_bin-min_bin)/bin_hist ))
#print(nbins)

if(binlog ==False): nbins = (np.arange(min_bin, max_bin, (max_bin-min_bin)/(bin_hist + 1)  ))
if(binlog ==True): nbins = np.logspace(np.log10(min_bin), np.log10(max_bin), bin_hist + 1 )
print(nbins)


if(lista == lista_e):
    ax.hist(dat_emit_e1, weights=weights_emit_e1, bins=nbins, density=densi, histtype='step', log=log_y, color='b', linewidth=2.5,
                            label= source1 + r' $E_{emit}$ '+r'$e^-$'#+' $E > 0$'

                           # + '\n time = ' + r'%.2f' % (tim1) + ' s'
                           # + '\n evt = ' + str(event_emit_e1)
                            )
    if(plt_emitdep == True):
        ax.hist(dat_all_emit_e1, weights=weights_all_emit_e1, bins=nbins, density=densi, histtype='step', log=log_y, color='b', linestyle='dotted', linewidth=2.5,
                            label= source1 + r' $E_{emit}$ '+r'$e^-$ dep '# + strCond_dep#+' on detect'
                           # + '\n time = ' + r'%.2f' % (tim1) + ' s'
                           # + '\n evt = ' + str(event_dect_all_emit_e1)
                            )
    if(plt_edep == True):
        ax.hist(dat_edep_e1, weights=weights_edep_e1, bins=nbins, density=densi, histtype='step', log=log_y, color='C0', linestyle='dashed', linewidth=2.5,
                                label= source1 + r' $E_{dep}$ ' + r'$e^-$ '# + strCond_dep#+' in detect' #+  condit_edep
                               # + '\n time = ' + r'%.2f' % (tim1) + ' s'
                               # + '\n evt = ' + str(event_edep_e1)
                               )

    ax.hist(dat_emit_e2, weights=weights_emit_e2, bins=nbins, density=densi, histtype='step', log=log_y, color='r', linewidth=2.5,
                            label= source2 + r' $E_{emit}$ '+r'$e^-$'#+' $E > 0$' + ''
                           # + '\n time = ' + r'%.2f' % (tim2) + ' s'
                           # + '\n evt = ' + str(event_emit_e2)
                            )
    if(plt_emitdep == True):
        ax.hist(dat_all_emit_e2, weights=weights_all_emit_e2, bins=nbins, density=densi, histtype='step', log=log_y, color='r', linestyle='dotted', linewidth=2.5,
                            label= source2 + r' $E_{emit}$ '+r'$e^-$ dep '# + strCond_dep#+' on detect' #+  condit_edep
                           # + '\n time = ' + r'%.2f' % (tim2) + ' s'
                           # + '\n evt = ' + str(event_dect_all_emit_e2)
                            )
    if(plt_edep == True):
        ax.hist(dat_edep_e2, weights=weights_edep_e2, bins=nbins, density=densi, histtype='step', log=log_y, color='C0', linestyle='dashed', linewidth=2.5,
                                label= source2 + r' $E_{dep}$ ' + r'$e^-$ '# + strCond_dep#+' in detect' #+  condit_edep
                               # + '\n time = ' + r'%.2f' % (tim2) + ' s'
                               # + '\n evt = ' + str(event_edep_e2)
                                )

if(lista == lista_g):
    ax.hist(dat_emit_g1, bins=nbins, density=densi, histtype='step', log=log_y, color='C8', linewidth=2.5,
                            label=' Sim '+source1 + ' all '+'gammas  emit ' + ''
                           # + '\n time = ' + r'%.2f' % (tim1) + ' s'
                           # + '\n evt = ' + str(event_emit_g1)
                            )
    if(plt_emitdep == True):
        ax.hist(dat_all_emit_g1, bins=nbins, density=densi, histtype='step', log=log_y, color='C2', linewidth=2.5,
                            label=' Sim '+source1 + ' all '+'gammas emit on detect ' + condit_edep
                           # + '\n time = ' + r'%.2f' % (tim1) + ' s'
                           # + '\n evt = ' + str(event_dect_all_emit_g1)
                            )
    if(plt_edep == True):
        ax.hist(dat_edep_g1, bins=nbins, density=densi, histtype='step', log=log_y, color='C2', linewidth=2.5,
                                label=' Sim '+source1 + ' all ' + 'gammas edep on detect' +  condit_edep
                               # + '\n time = ' + r'%.2f' % (tim1) + ' s'
                               # + '\n evt = ' + str(event_edep_g1)
                                )
    ax.hist(dat_emit_g2, bins=nbins, density=densi, histtype='step', log=log_y, color='g', linewidth=2.5,
                            label=' Sim '+source2 + ' all '+'gammas  emit ' + ''
                           # + '\n time = ' + r'%.2f' % (tim2) + ' s'
                           # + '\n evt = ' + str(event_emit_g2)
                            )
    if(plt_emitdep == True):
        ax.hist(dat_all_emit_g2, bins=nbins, density=densi, histtype='step', log=log_y, color='C8', linewidth=2.5,
                            label=' Sim '+source2 + ' all '+'gammas emit on detect ' + condit_edep
                           # + '\n time = ' + r'%.2f' % (tim2) + ' s'
                           # + '\n evt = ' + str(event_dect_all_emit_g2)
                            )
    if(plt_edep == True):
        ax.hist(dat_edep_g2, bins=nbins, density=densi, histtype='step', log=log_y, color='C8', linewidth=2.5,
                             label=' Sim '+source2 + ' all ' + 'gammas edep on detect' +  condit_edep
                               # + '\n time = ' + r'%.2f' % (tim2) + ' s'
                               # + '\n evt = ' + str(event_edep_g2)
                             )


if(lista == lista_eg):

    if(plt_sim0 != True or plt_all == True):
        ax.hist(dat_emit_e_g1, weights=weights_emit_e_g1, bins=nbins, density=densi, histtype='step', log=log_y, color='C0', linewidth=2.5,
                            label= source1 + r' $E_{emit}$ '+r'$e^-$, $\gamma$'#+' $E > 0$'
                           # + '\n time = ' + r'%.2f' % (tim1) + ' s'
                            + '\n evt = ' + str(event_emit_e_g1)
                            )

        ax.hist(dat_emit_e_g2, weights=weights_emit_e_g2, bins=nbins, density=densi, histtype='step', log=log_y, color='C7', linewidth=2.5,
                            label= source2 + r' $E_{emit}$ '+r'$e^-$, $\gamma$'#+' $E > 0$' + ''
                           # + '\n time = ' + r'%.2f' % (tim2) + ' s'
                            + '\n evt = ' + str(event_emit_e_g2)
                            )
        if(plt_emitdep == True):
            ax.hist(dat_all_emit_pt1, weights=weights_all_emit_pt1, bins=nbins, density=densi, histtype='step', log=log_y, color='C0', linestyle='dotted', linewidth=2.5,
                                label= source1 + r' $E_{emit}$ '+r'$e^-$, $\gamma$ dep' #+ strCond_dep#+' on detect'
                               # + '\n time = ' + r'%.2f' % (tim1) + ' s'
                                + '\n evt = ' + str(event_dect_all_emit_pt1)
                                )
            ax.hist(dat_all_emit_pt2, weights=weights_all_emit_pt2, bins=nbins, density=densi, histtype='step', log=log_y, color='C7', linestyle='dotted', linewidth=2.5,
                                label= source2 + r' $E_{emit}$ '+r'$e^-$, $\gamma$ dep' #+ strCond_dep#+' on detect' #+  condit_edep
                               # + '\n time = ' + r'%.2f' % (tim2) + ' s'
                                + '\n evt = ' + str(event_dect_all_emit_pt2)
                                )
        if(plt_edep == True):
            ax.hist(dat_edep_pt1, weights=weights_edep_pt1, bins=nbins, density=densi, histtype='step', log=log_y, color='C0', linestyle='dashed', linewidth=2.5,
                                    label= source1 + r' $E_{dep}$ ' + r'$e^-$, $\gamma$' #+ strCond_dep#+' in detect' #+  condit_edep
                                   # + '\n time = ' + r'%.2f' % (tim1) + ' s'
                                    + '\n evt = ' + str(event_edep_pt1)
                                   )
            ax.hist(dat_edep_pt2, weights=weights_edep_pt2, bins=nbins, density=densi, histtype='step', log=log_y, color='C7', linestyle='dashed', linewidth=2.5,
                                    label= source2 + r' $E_{dep}$ ' + r'$e^-$, $\gamma$' #+ strCond_dep#+' in detect' #+  condit_edep
                                   # + '\n time = ' + r'%.2f' % (tim2) + ' s'
                                    + '\n evt = ' + str(event_edep_pt2)
                                    )


    if(plt_sim0 == True or plt_all == True):
        hist_emit_e_g1 = np.histogram(dat_emit_e_g1, weights=weights_emit_e_g1, bins=nbins)
        count_emit_e_g1, bin_emit_e_g1 = hist_emit_e_g1[0], hist_emit_e_g1[1]
        if binlog:
            bincenters_emit_e_g1 = np.sqrt(bin_emit_e_g1[:-1] * bin_emit_e_g1[1:])
        else:
            bincenters_emit_e_g1 = 0.5*(bin_emit_e_g1[1:]+bin_emit_e_g1[:-1])
        norm_emit_e_g1 = np.sum(count_emit_e_g1 * np.diff(bin_emit_e_g1))

        if(densi==False):
            sim_w_emit_e_g1 = count_emit_e_g1
            if(errors_sqrt == True): err_sim_w_emit_e_g1 = np.sqrt(0.04*count_emit_e_g1*count_emit_e_g1 + count_emit_e_g1)
            else: err_sim_w_emit_e_g1 = (np.sqrt(count_emit_e_g1))
        if(densi==True):
            sim_w_emit_e_g1 = count_emit_e_g1 / norm_emit_e_g1
            if(errors_sqrt == True): err_sim_w_emit_e_g1 = np.sqrt(0.04*count_emit_e_g1*count_emit_e_g1 + count_emit_e_g1) / norm_emit_e_g1
            else: err_sim_w_emit_e_g1 = (np.sqrt(count_emit_e_g1)) / norm_emit_e_g1


        hist_emit_e_g2 = np.histogram(dat_emit_e_g2, weights=weights_emit_e_g2, bins=nbins)
        count_emit_e_g2, bin_emit_e_g2 = hist_emit_e_g2[0], hist_emit_e_g2[1]
        if binlog:
            bincenters_emit_e_g2 = np.sqrt(bin_emit_e_g2[:-1] * bin_emit_e_g2[1:])
        else:
            bincenters_emit_e_g2 = 0.5*(bin_emit_e_g2[1:]+bin_emit_e_g2[:-1])
        norm_emit_e_g2 = np.sum(count_emit_e_g2 * np.diff(bin_emit_e_g2))

        if(densi==False):
            sim_w_emit_e_g2 = count_emit_e_g2
            if(errors_sqrt == True): err_sim_w_emit_e_g2 = np.sqrt(0.04*count_emit_e_g2*count_emit_e_g2 + count_emit_e_g2)
            else: err_sim_w_emit_e_g2 = (np.sqrt(count_emit_e_g2))
        if(densi==True):
            sim_w_emit_e_g2 = count_emit_e_g2 / norm_emit_e_g2
            if(errors_sqrt == True): err_sim_w_emit_e_g2 = np.sqrt(0.04*count_emit_e_g2*count_emit_e_g2 + count_emit_e_g2) / norm_emit_e_g2
            else: err_sim_w_emit_e_g2 = (np.sqrt(count_emit_e_g2)) / norm_emit_e_g2

        err_emit_e_g12 = np.sqrt(err_sim_w_emit_e_g1*err_sim_w_emit_e_g1 + err_sim_w_emit_e_g2*err_sim_w_emit_e_g2 )
        ch2_emit_e_g12 = chi_2_sigm_test(sim_w_emit_e_g1, sim_w_emit_e_g2, err_emit_e_g12)
        tmp_str_emit_e_g12  = 'Chi2 test: ' + r'%.2f'%ch2_emit_e_g12[0] \
                    +'       '+'ndf: ' + r'%.2f'%ch2_emit_e_g12[3] \
                    +'\n'+'Chi2/ndf : ' + r'%.2f'%ch2_emit_e_g12[4]
        print(tmp_str_emit_e_g12)
        if chi_label == True :
            chi_str_emit_e_g12 = r'$\chi^2_\nu$ = ' + r'%.2f'%ch2_emit_e_g12[4]
        else:
            chi_str_emit_e_g12 = ''

        if plt_data == 'step' :
            ax.step( np.insert(bin_emit_e_g1, 0, 0), np.concatenate(([0], sim_w_emit_e_g1, [0])), where='post', color='C0',
                markersize=6, alpha=0.6, linewidth=2.5,  label = source1 + r' $E_{emit}$ '+r'$e^-$, $\gamma$'
                #+ '\n evt = ' + str(event_emit_e_g1)
                )
            ax.step( np.insert(bin_emit_e_g2, 0, 0), np.concatenate(([0], sim_w_emit_e_g2, [0])), where='post', color='C7',
                markersize=6, alpha=0.6, linewidth=2.5,  label = source2 + r' $E_{emit}$ '+r'$e^-$, $\gamma$'
                #+ '\n evt = ' + str(event_emit_e_g2)
                )
        if plt_data == 'between' :
            ax.fill_between(bincenters_emit_e_g1, sim_w_emit_e_g1 - err_sim_w_emit_e_g1, sim_w_emit_e_g1 + err_sim_w_emit_e_g1,
                    color='C0', linewidth=l_width*2/3, alpha=0.6,  label = source1 + r' $E_{emit}$ '+r'$e^-$, $\gamma$'
                    #+ '\n evt = ' + str(event_emit_e_g1)
                    )
            ax.fill_between(bincenters_emit_e_g2, sim_w_emit_e_g2 - err_sim_w_emit_e_g2, sim_w_emit_e_g2 + err_sim_w_emit_e_g2,
                    color='C7', linewidth=l_width*2/3, alpha=0.6,  label = source2 + r' $E_{emit}$ '+r'$e^-$, $\gamma$'
                    #+ '\n evt = ' + str(event_emit_e_g2)
                    )
        ax.plot(0,0, color='w', label = chi_str_emit_e_g12)
        if plt_err == True :
            ax.errorbar(bincenters_emit_e_g1, sim_w_emit_e_g1, yerr=err_sim_w_emit_e_g1, fmt='C0'+'o',
                                    ecolor='C0', markersize=6, alpha=0.4, linewidth=2.5)
            ax.errorbar(bincenters_emit_e_g2, sim_w_emit_e_g2, yerr=err_sim_w_emit_e_g2, fmt='C7'+'o',
                                    ecolor='C7', markersize=6, alpha=0.4, linewidth=2.5,
                                    #label = chi_str_emit_e_g12
                                    )


        if(plt_emitdep == True):
            hist_all_emit_pt1 = np.histogram(dat_all_emit_pt1, weights=weights_all_emit_pt1, bins=nbins)
            count_all_emit_pt1, bin_all_emit_pt1 = hist_all_emit_pt1[0], hist_all_emit_pt1[1]
            if binlog:
                bincenters_all_emit_pt1 = np.sqrt(bin_all_emit_pt1[:-1] * bin_all_emit_pt1[1:])
            else:
                bincenters_all_emit_pt1 = 0.5*(bin_all_emit_pt1[1:]+bin_all_emit_pt1[:-1])
            norm_all_emit_pt1 = np.sum(count_all_emit_pt1 * np.diff(bin_all_emit_pt1))

            if(densi==False):
                sim_w_all_emit_pt1 = count_all_emit_pt1
                if(errors_sqrt == True): err_sim_w_all_emit_pt1 = np.sqrt(0.04*count_all_emit_pt1*count_all_emit_pt1 + count_all_emit_pt1)
                else: err_sim_w_all_emit_pt1 = (np.sqrt(count_all_emit_pt1))
            if(densi==True):
                sim_w_all_emit_pt1 = count_all_emit_pt1 / norm_all_emit_pt1
                if(errors_sqrt == True): err_sim_w_all_emit_pt1 = np.sqrt(0.04*count_all_emit_pt1*count_all_emit_pt1 + count_all_emit_pt1) / norm_all_emit_pt1
                else: err_sim_w_all_emit_pt1 = (np.sqrt(count_all_emit_pt1)) / norm_all_emit_pt1


            hist_all_emit_pt2 = np.histogram(dat_all_emit_pt2, weights=weights_all_emit_pt2, bins=nbins)
            count_all_emit_pt2, bin_all_emit_pt2 = hist_all_emit_pt2[0], hist_all_emit_pt2[1]
            if binlog:
                bincenters_all_emit_pt2 = np.sqrt(bin_all_emit_pt2[:-1] * bin_all_emit_pt2[1:])
            else:
                bincenters_all_emit_pt2 = 0.5*(bin_all_emit_pt2[1:]+bin_all_emit_pt2[:-1])
            norm_all_emit_pt2 = np.sum(count_all_emit_pt2 * np.diff(bin_all_emit_pt2))

            if(densi==False):
                sim_w_all_emit_pt2 = count_all_emit_pt2
                if(errors_sqrt == True): err_sim_w_all_emit_pt2 = np.sqrt(0.04*count_all_emit_pt2*count_all_emit_pt2 + count_all_emit_pt2)
                else: err_sim_w_all_emit_pt2 = (np.sqrt(count_all_emit_pt2))
            if(densi==True):
                sim_w_all_emit_pt2 = count_all_emit_pt2 / norm_all_emit_pt2
                if(errors_sqrt == True): err_sim_w_all_emit_pt2 = np.sqrt(0.04*count_all_emit_pt2*count_all_emit_pt2 + count_all_emit_pt2) / norm_all_emit_pt2
                else: err_sim_w_all_emit_pt2 = (np.sqrt(count_all_emit_pt2)) / norm_all_emit_pt2

            err_all_emit_pt12 = np.sqrt(err_sim_w_all_emit_pt1*err_sim_w_all_emit_pt1 + err_sim_w_all_emit_pt2*err_sim_w_all_emit_pt2 )
            ch2_all_emit_pt12 = chi_2_sigm_test(sim_w_all_emit_pt1, sim_w_all_emit_pt2, err_all_emit_pt12)
            tmp_str_all_emit_pt12  = 'Chi2 test: ' + r'%.2f'%ch2_all_emit_pt12[0] \
                        +'       '+'ndf: ' + r'%.2f'%ch2_all_emit_pt12[3] \
                        +'\n'+'Chi2/ndf : ' + r'%.2f'%ch2_all_emit_pt12[4]
            print(tmp_str_all_emit_pt12)
            if chi_label == True :
                chi_str_all_emit_pt12 = r'$\chi^2_\nu$ = ' + r'%.2f'%ch2_all_emit_pt12[4]
            else:
                chi_str_all_emit_pt12 = ''

            if plt_data == 'step' :
                ax.step( np.insert(bin_all_emit_pt1, 0, 0), np.concatenate(([0], sim_w_all_emit_pt1, [0])), where='post', color='C0',
                    linestyle='dotted', markersize=6, alpha=0.6, linewidth=2.5,  label = source1 + r' $E_{emit}$ '+r'$e^-$, $\gamma$ dep'
                    #+ '\n evt = ' + str(event_dect_all_emit_pt1)
                    )
                ax.step( np.insert(bin_all_emit_pt2, 0, 0), np.concatenate(([0], sim_w_all_emit_pt2, [0])), where='post', color='C7',
                    linestyle='dotted', markersize=6, alpha=0.6, linewidth=2.5,  label = source2 + r' $E_{emit}$ '+r'$e^-$, $\gamma$ dep'
                    #+ '\n evt = ' + str(event_dect_all_emit_pt2)
                    )
            if plt_data == 'between' :
                ax.fill_between(bincenters_all_emit_pt1, sim_w_all_emit_pt1 - err_sim_w_all_emit_pt1, sim_w_all_emit_pt1 + err_sim_w_all_emit_pt1,
                        color='C0', linestyle='dotted', linewidth=l_width*2/3, alpha=0.6,  label = source1 + r' $E_{emit}$ '+r'$e^-$, $\gamma$ dep'
                        #+ '\n evt = ' + str(event_dect_all_emit_pt1)
                        )
                ax.fill_between(bincenters_all_emit_pt2, sim_w_all_emit_pt2 - err_sim_w_all_emit_pt2, sim_w_all_emit_pt2 + err_sim_w_all_emit_pt2,
                        color='C7', linestyle='dotted', linewidth=l_width*2/3, alpha=0.6,  label = source2 + r' $E_{emit}$ '+r'$e^-$, $\gamma$ dep'
                        #+ '\n evt = ' + str(event_dect_all_emit_pt2)
                        )

            ax.plot(0,0, color='w', label = chi_str_all_emit_pt12)

            if plt_err == True :
                ax.errorbar(bincenters_all_emit_pt1, sim_w_all_emit_pt1, yerr=err_sim_w_all_emit_pt1, fmt='C0'+'o',
                                        ecolor='C0', markersize=6, alpha=0.4, linewidth=2.5)
                ax.errorbar(bincenters_all_emit_pt2, sim_w_all_emit_pt2, yerr=err_sim_w_all_emit_pt2, fmt='C7'+'o',
                                        ecolor='C7', markersize=6, alpha=0.4, linewidth=2.5,
                                        #label = chi_str_all_emit_pt12
                                        )

        if(plt_edep == True):
            hist_edep_pt1 = np.histogram(dat_edep_pt1, weights=weights_edep_pt1, bins=nbins)
            count_edep_pt1, bin_edep_pt1 = hist_edep_pt1[0], hist_edep_pt1[1]
            if binlog:
                bincenters_edep_pt1 = np.sqrt(bin_edep_pt1[:-1] * bin_edep_pt1[1:])
            else:
                bincenters_edep_pt1 = 0.5*(bin_edep_pt1[1:]+bin_edep_pt1[:-1])
            norm_edep_pt1 = np.sum(count_edep_pt1 * np.diff(bin_edep_pt1))

            if(densi==False):
                sim_w_edep_pt1 = count_edep_pt1
                if(errors_sqrt == True): err_sim_w_edep_pt1 = np.sqrt(0.04*count_edep_pt1*count_edep_pt1 + count_edep_pt1)
                else: err_sim_w_edep_pt1 = (np.sqrt(count_edep_pt1))
            if(densi==True):
                sim_w_edep_pt1 = count_edep_pt1 / norm_edep_pt1
                if(errors_sqrt == True): err_sim_w_edep_pt1 = np.sqrt(0.04*count_edep_pt1*count_edep_pt1 + count_edep_pt1) / norm_edep_pt1
                else: err_sim_w_edep_pt1 = (np.sqrt(count_edep_pt1)) / norm_edep_pt1


            hist_edep_pt2 = np.histogram(dat_edep_pt2, weights=weights_edep_pt2, bins=nbins)
            count_edep_pt2, bin_edep_pt2 = hist_edep_pt2[0], hist_edep_pt2[1]
            if binlog:
                bincenters_edep_pt2 = np.sqrt(bin_edep_pt2[:-1] * bin_edep_pt2[1:])
            else:
                bincenters_edep_pt2 = 0.5*(bin_edep_pt2[1:]+bin_edep_pt2[:-1])
            norm_edep_pt2 = np.sum(count_edep_pt2 * np.diff(bin_edep_pt2))

            if(densi==False):
                sim_w_edep_pt2 = count_edep_pt2
                if(errors_sqrt == True): err_sim_w_edep_pt2 = np.sqrt(0.04*count_edep_pt2*count_edep_pt2 + count_edep_pt2)
                else: err_sim_w_edep_pt2 = (np.sqrt(count_edep_pt2))
            if(densi==True):
                sim_w_edep_pt2 = count_edep_pt2 / norm_edep_pt2
                if(errors_sqrt == True): err_sim_w_edep_pt2 = np.sqrt(0.04*count_edep_pt2*count_edep_pt2 + count_edep_pt2) / norm_edep_pt2
                else: err_sim_w_edep_pt2 = (np.sqrt(count_edep_pt2)) / norm_edep_pt2

            err_edep_pt12 = np.sqrt(err_sim_w_edep_pt1*err_sim_w_edep_pt1 + err_sim_w_edep_pt2*err_sim_w_edep_pt2 )
            ch2_edep_pt12 = chi_2_sigm_test(sim_w_edep_pt1, sim_w_edep_pt2, err_edep_pt12)
            tmp_str_edep_pt12  = 'Chi2 test: ' + r'%.2f'%ch2_edep_pt12[0] \
                        +'       '+'ndf: ' + r'%.2f'%ch2_edep_pt12[3] \
                        +'\n'+'Chi2/ndf : ' + r'%.2f'%ch2_edep_pt12[4]
            print(tmp_str_edep_pt12)
            if chi_label == True :
                chi_str_edep_pt12 = r'$\chi^2_\nu$ = ' + r'%.2f'%ch2_edep_pt12[4]
            else:
                chi_str_edep_pt12 = ''

            if plt_data == 'step' :
                ax.step( np.insert(bin_edep_pt1, 0, 0), np.concatenate(([0], sim_w_edep_pt1, [0])), where='post', color='C0',
                     linestyle='dashed', markersize=6, alpha=0.6, linewidth=2.5,  label = source1 + r' $E_{dep}$ ' + r'$e^-$, $\gamma$'
                     #+ '\n evt = ' + str(event_edep_pt1)
                     )
                ax.step( np.insert(bin_edep_pt2, 0, 0), np.concatenate(([0], sim_w_edep_pt2, [0])), where='post', color='C7',
                     linestyle='dashed', markersize=6, alpha=0.6, linewidth=2.5,  label = source2 + r' $E_{dep}$ ' + r'$e^-$, $\gamma$'
                     #+ '\n evt = ' + str(event_edep_pt2)
                     )
            if plt_data == 'between' :
                ax.fill_between(bincenters_edep_pt1, sim_w_edep_pt1 - err_sim_w_edep_pt1, sim_w_edep_pt1 + err_sim_w_edep_pt1,
                        color='C0', linestyle='dashed', linewidth=l_width*2/3, alpha=0.6,  label = source1 + r' $E_{dep}$ ' + r'$e^-$, $\gamma$'
                        #+ '\n evt = ' + str(event_edep_pt1)
                        )
                ax.fill_between(bincenters_edep_pt2, sim_w_edep_pt2 - err_sim_w_edep_pt2, sim_w_edep_pt2 + err_sim_w_edep_pt2,
                        color='C7', linestyle='dashed', linewidth=l_width*2/3, alpha=0.6,  label = source2 + r' $E_{dep}$ ' + r'$e^-$, $\gamma$'
                        #+ '\n evt = ' + str(event_edep_pt2)
                        )
            ax.plot(0,0, color='w', label = chi_str_edep_pt12)
            if plt_err == True :
                ax.errorbar(bincenters_edep_pt1, sim_w_edep_pt1, yerr=err_sim_w_edep_pt1, fmt='C0'+'o',
                                        ecolor='C0', markersize=6, alpha=0.4, linewidth=2.5)
                ax.errorbar(bincenters_edep_pt2, sim_w_edep_pt2, yerr=err_sim_w_edep_pt2, fmt='C7'+'o',
                                        ecolor='C7', markersize=6, alpha=0.4, linewidth=2.5,
                                        #label = chi_str_edep_pt12
                                        )



if(lista == lista_all):

    ax.hist(dat_all_emit_pt1, bins=nbins, density=densi, histtype='step', log=log_y, color='C0', linewidth=2.5,
                            label=' Sim '+source1 + ' all particles emit on detect ' + condit_edep
                           # + '\n time = ' + r'%.2f' % (tim1) + ' s'
                           # + '\n evt = ' + str(event_dect_all_emit_pt1)
                            )
    if(plt_emitdep == True):
        ax.hist(dat_all_emit_e1, bins=nbins, density=densi, histtype='step', log=log_y, color='b', linewidth=2.5,
                            label=' Sim '+source1 + ' all '+'electrons emit on detect ' + condit_edep
                           # + '\n time = ' + r'%.2f' % (tim1) + ' s'
                           # + '\n evt = ' + str(event_dect_all_emit_e1)
                            )
        ax.hist(dat_all_emit_ae1, bins=nbins, density=densi, histtype='step', log=log_y, color='C7', linewidth=2.5,
                            label=' Sim '+source1 + ' all '+'anti_nu_elec emit on detect ' + condit_edep
                           # + '\n time = ' + r'%.2f' % (tim1) + ' s'
                           # + '\n evt = ' + str(event_dect_all_emit_ae1)
                            )
        ax.hist(dat_all_emit_g1, bins=nbins, density=densi, histtype='step', log=log_y, color='C2', linewidth=2.5,
                            label=' Sim '+source1 + ' all '+'gammas emit on detect ' + condit_edep
                           # + '\n time = ' + r'%.2f' % (tim1) + ' s'
                           # + '\n evt = ' + str(event_dect_all_emit_g1)
                            )
    ax.hist(dat_emit_e1, bins=nbins, density=densi, histtype='step', log=log_y, color='b', linewidth=1,
                            label=' Sim '+source1 + ' all '+'electrons emit ' + ''
                           # + '\n time = ' + r'%.2f' % (tim1) + ' s'
                           # + '\n evt = ' + str(event_emit_e1)
                            )
    ax.hist(dat_emit_ae1, bins=nbins, density=densi, histtype='step', log=log_y, color='C4', linewidth=2.5,
                            label=' Sim '+source1 + ' all '+'anti_nu_elec emit ' + ''
                           # + '\n time = ' + r'%.2f' % (tim1) + ' s'
                           # + '\n evt = ' + str(event_emit_ae1)
                            )
    ax.hist(dat_emit_g1, bins=nbins, density=densi, histtype='step', log=log_y, color='C8', linewidth=2.5,
                            label=' Sim '+source1 + ' all '+'gammas  emit ' + ''
                           # + '\n time = ' + r'%.2f' % (tim1) + ' s'
                           # + '\n evt = ' + str(event_emit_g1)
                            )
    ax.hist(dat_emit_all1, bins=nbins, density=densi, histtype='step', log=log_y, color='k', linewidth=2.5,
                            label=' Sim '+source1 + ' all ' + r'$e^-$, $\overline{\nu}_e$ , $\gamma$ '+ 'emit ' + ''
                           # + '\n time = ' + r'%.2f' % (tim1) + ' s'
                           # + '\n evt = ' + str(event_emit_all1)
                            )
    if(plt_edep == True):
        ax.hist(dat_edep_pt1, bins=nbins, density=densi, histtype='step', log=log_y, color='C0', linewidth=2.5,
                                label=' Sim '+source1 + ' all ' + 'particles edep on detect' +  condit_edep
                               # + '\n time = ' + r'%.2f' % (tim1) + ' s'
                               # + '\n evt = ' + str(event_edep_pt1)
                                )
        ax.hist(dat_edep_e1, bins=nbins, density=densi, histtype='step', log=log_y, color='C0', linewidth=2.5,
                                label=' Sim '+source1 + ' all ' + 'electrons edep on detect' +  condit_edep
                               # + '\n time = ' + r'%.2f' % (tim1) + ' s'
                               # + '\n evt = ' + str(event_edep_e1)
                                )
        ax.hist(dat_edep_ae1, bins=nbins, density=densi, histtype='step', log=log_y, color='C7', linewidth=2.5,
                                label=' Sim '+source1 + ' all ' + 'anti_nu_elec edep on detect' +  condit_edep
                               # + '\n time = ' + r'%.2f' % (tim1) + ' s'
                               # + '\n evt = ' + str(event_edep_ae1)
                                )

        ax.hist(dat_edep_g1, bins=nbins, density=densi, histtype='step', log=log_y, color='C2', linewidth=2.5,
                                label=' Sim '+source1 + ' all ' + 'gammas edep on detect' +  condit_edep
                               # + '\n time = ' + r'%.2f' % (tim1) + ' s'
                               # + '\n evt = ' + str(event_edep_g1)
                                )

        ax.hist(dat_all_emit_pt2, bins=nbins, density=densi, histtype='step', log=log_y, color='C9', linewidth=2.5,
                            label=' Sim '+source2 + ' all particles emit on detect ' + condit_edep
                           # + '\n time = ' + r'%.2f' % (tim2) + ' s'
                           # + '\n evt = ' + str(event_dect_all_emit_pt2)
                            )
        ax.hist(dat_all_emit_e2, bins=nbins, density=densi, histtype='step', log=log_y, color='r', linewidth=2.5,
                            label=' Sim '+source2 + ' all '+'electrons emit on detect ' + condit_edep
                           # + '\n time = ' + r'%.2f' % (tim2) + ' s'
                           # + '\n evt = ' + str(event_dect_all_emit_e2)
                            )
        ax.hist(dat_all_emit_ae2, bins=nbins, density=densi, histtype='step', log=log_y, color='C7', linewidth=2.5,
                            label=' Sim '+source2 + ' all '+'anti_nu_elec emit on detect ' + condit_edep
                           # + '\n time = ' + r'%.2f' % (tim2) + ' s'
                           # + '\n evt = ' + str(event_dect_all_emit_ae2)
                            )
        ax.hist(dat_all_emit_g2, bins=nbins, density=densi, histtype='step', log=log_y, color='C8', linewidth=2.5,
                            label=' Sim '+source2 + ' all '+'gammas emit on detect ' + condit_edep
                           # + '\n time = ' + r'%.2f' % (tim2) + ' s'
                           # + '\n evt = ' + str(event_dect_all_emit_g2)
                            )
    ax.hist(dat_emit_e2, bins=nbins, density=densi, histtype='step', log=log_y, color='r', linewidth=1,
                            label=' Sim '+source2 + ' all '+'electrons emit ' + ''
                           # + '\n time = ' + r'%.2f' % (tim2) + ' s'
                           # + '\n evt = ' + str(event_emit_e2)
                            )
    ax.hist(dat_emit_ae2, bins=nbins, density=densi, histtype='step', log=log_y, color='C8', linewidth=2.5,
                            label=' Sim '+source2 + ' all '+'anti_nu_elec emit ' + ''
                           # + '\n time = ' + r'%.2f' % (tim2) + ' s'
                           # + '\n evt = ' + str(event_emit_ae2)
                            )
    ax.hist(dat_emit_g2, bins=nbins, density=densi, histtype='step', log=log_y, color='g', linewidth=2.5,
                            label=' Sim '+source2 + ' all '+'gammas  emit ' + ''
                           # + '\n time = ' + r'%.2f' % (tim2) + ' s'
                           # + '\n evt = ' + str(event_emit_g2)
                            )
    ax.hist(dat_emit_all2, bins=nbins, density=densi, histtype='step', log=log_y, color='b', linewidth=2.5,
                            label=' Sim '+source2 + ' all ' + r'$e^-$, $\overline{\nu}_e$ , $\gamma$ '+ 'emit ' + ''
                           # + '\n time = ' + r'%.2f' % (tim2) + ' s'
                           # + '\n evt = ' + str(event_emit_all2)
                            )
    if(plt_edep == True):
        ax.hist(dat_edep_pt2, bins=nbins, density=densi, histtype='step', log=log_y, color='C9', linewidth=2.5,
                                label=' Sim '+source2 + ' all ' + 'particles edep on detect' +  condit_edep
                               # + '\n time = ' + r'%.2f' % (tim2) + ' s'
                               # + '\n evt = ' + str(event_edep_pt2)
                                )
        ax.hist(dat_edep_e2, bins=nbins, density=densi, histtype='step', log=log_y, color='C6', linewidth=2.5,
                                label=' Sim '+source2 + ' all ' + 'electrons edep on detect' +  condit_edep
                               # + '\n time = ' + r'%.2f' % (tim2) + ' s'
                               # + '\n evt = ' + str(event_edep_e2)
                                )
        ax.hist(dat_edep_ae2, bins=nbins, density=densi, histtype='step', log=log_y, color='C7', linewidth=2.5,
                                label=' Sim '+source2 + ' all ' + 'anti_nu_elec edep on detect' +  condit_edep
                               # + '\n time = ' + r'%.2f' % (tim2) + ' s'
                               # + '\n evt = ' + str(event_edep_ae2)
                                )
        ax.hist(dat_edep_g2, bins=nbins, density=densi, histtype='step', log=log_y, color='C8', linewidth=2.5,
                                label=' Sim '+source2 + ' all ' + 'gammas edep on detect' +  condit_edep
                               # + '\n time = ' + r'%.2f' % (tim2) + ' s'
                               # + '\n evt = ' + str(event_edep_g2)
                                )

if(partic_name == 'pixel_edep_emit'):
    ax.hist(dat_emit_all1, bins=nbins, density=densi, histtype='step', log=log_y, color='C7', linewidth=2.5,
                                label=' Sim '+source1 + ' all '+'particle emit edep' + ' '
                                # + '\n time = ' + r'%.2f' % (tim1) + ' s'
                                # + '\n evt = ' + str(event_emit_all1)
                                )
    ax.hist(dat_emit_all2, bins=nbins, density=densi, histtype='step', log=log_y, color='y', linewidth=2.5,
                                label=' Sim '+sourc2e + ' all '+'particle emit edep' + ' '
                                # + '\n time = ' + r'%.2f' % (tim2) + ' s'
                                # + '\n evt = ' + str(event_emit_all2)
                                )
if(partic_name == 'pixel_edep_zero_emit'):
    ax.hist(dat_emit_01, bins=nbins, density=densi, histtype='step', log=log_y, color='C6', linewidth=2.5,
                                label=' Sim '+source1 + ' all '+'particle emit edep_zero' + ' '
                                # + '\n time = ' + r'%.2f' % (tim1) + ' s'
                                # + '\n evt = ' + str(event_emit_01)
                                )
    ax.hist(dat_emit_02, bins=nbins, density=densi, histtype='step', log=log_y, color='c', linewidth=2.5,
                                label=' Sim '+source2 + ' all '+'particle emit edep_zero' + ' '
                                # + '\n time = ' + r'%.2f' % (tim2) + ' s'
                                # + '\n evt = ' + str(event_emit_02)
                                )

################################################################################
################################################################################
ax.axvline(x = sat_cam, color = 'k', linestyle="--", linewidth=2.5)
ax.axvline(x = min_cam, color = 'k', linestyle="--", linewidth=2.5)

# Agregar etiquetas en la parte superior

if(chi_label == True and densi==True):escalog=40
if(chi_label == True and densi==False): escalog=5.5

if(chi_label == False and densi==True):escalog=2
if(chi_label == False and densi==False): escalog=0.2

if log_x:
    if densi:
        ax.text(min_cam, escalog*ax.get_ylim()[1], '100 ADC', ha='center', va='bottom', fontsize=font_siz-4)
        ax.text(sat_cam, escalog*ax.get_ylim()[1], '1023 ADC', ha='center', va='bottom', fontsize=font_siz-4)
    else:
        ax.text(min_cam, 2.5e4*escalog*ax.get_ylim()[1], '100 ADC', ha='center', va='bottom', fontsize=font_siz-4)
        ax.text(sat_cam, 2.5e4*escalog*ax.get_ylim()[1], '1023 ADC', ha='center', va='bottom', fontsize=font_siz-4)

else:
    if densi:
        ax.text(min_cam, escalog*ax.get_ylim()[1], '100 ADC', ha='right', va='bottom', fontsize=font_siz-4)
        ax.text(sat_cam, escalog*ax.get_ylim()[1], '1023 ADC', ha='left', va='bottom', fontsize=font_siz-4)
    else:
        ax.text(min_cam, 18*escalog*ax.get_ylim()[1], '100 ADC', ha='right', va='bottom', fontsize=font_siz-4)
        ax.text(sat_cam, 18*escalog*ax.get_ylim()[1], '1023 ADC', ha='left', va='bottom', fontsize=font_siz-4)

#ax.grid(True)
if(log_y == True):
    ax.set_yscale('log')
if(log_x == True):
    ax.set_xscale('log')
#ax.set_xscale('log')
ax.tick_params(axis='x', labelsize=font_siz)
ax.tick_params(axis='y', labelsize=font_siz)
if densi :
    if log_x: ax.set_ylim(0, escalog)
    else: ax.set_ylim(0, escalog/50)
else:
    if log_x: ax.set_ylim(0, escalog*0.9e11)
    else:
        if chi_label == True : ax.set_ylim(0, 0.8e9)
        else: ax.set_ylim(0, 0.3e8)

#if(source2 == 'Cs137'):ax.set_ylim(0,10e7)
ax.set_xlabel('Energy ('+unidad+')', fontsize=font_siz-2)
if(densi==False):ax.set_ylabel('counts in all frame', fontsize=font_siz)
if(densi==True):ax.set_ylabel('normalized counts in all frames', fontsize=font_siz)
#ax.legend(loc='lower center', fontsize=font_siz-4)
ax.legend(loc='best' , fontsize=font_siz-6)
if(log_y == False):
    titulo = 'energy' + '_z' + \
        str(z).zfill(3) + '_t'+r'%.0f' % (tim2) + 's' + strbn
if(log_y == True):
    titulo = 'energy'+'_z' + \
        str(z).zfill(3) + '_log_y'+'_t'+r'%.0f' % (tim2) + 's' + strbn
if(log_x == True and log_y == True):
    titulo = 'energy'+'_z' + \
        str(z).zfill(3) + '_log_xy'+'_t'+r'%.0f' % (tim2) + 's' + strbn
#fig.suptitle(titulo , fontsize=font_siz)

fig.tight_layout()
if(plt_sim0 == True and lista == lista_eg):
    plt.savefig(dirfile + '/'+'plot_'+plt_data+'_'+titulo+'_emit_dep_e_g' +'sim_simple'+'.png', dpi=150)
    plt.savefig(dirfile + '/'+'plot_'+plt_data+'_'+titulo+'_emit_dep_e_g' +'sim_simple'+'.pdf', dpi=150)

#  plt.savefig(dirfile + '/'+'plot_'+source+'_'+titulo+'_' +
#             str(min_adc)+'ADC'+type_partic+'.png', dpi=150)

plt.show()
# plt.clf()
# plt.close()
