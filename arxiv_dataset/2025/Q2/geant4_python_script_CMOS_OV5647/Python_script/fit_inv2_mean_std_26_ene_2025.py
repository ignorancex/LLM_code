'Clusters_geq5'#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 15:55:29 2021

@author: mik
"""

import numpy as np

from collections import Counter

import matplotlib.pyplot as plt
from scipy import *
from scipy.optimize import curve_fit
import math

from scipy.stats import chi2
from scipy import stats
import os

#def fit_func(x, a, b):
#def inver_r2_0(r,a,b,c):
#    return a/(b*r*r +c)

def inver_r2(r,a, b):
    return a/((r+b)*(r+b))
    #return a/((r*r+b))

# Derived Chi Squared Value For This Model
def chi_square(obs,cal,sigma):
   chi2= np.sum(((obs-cal)/sigma)**2)
   ss_res = np.sum((cal - obs)**2)
   ss_tot = np.sum((cal - np.mean(cal) )**2)
   R2 = 1 - (ss_res / ss_tot)
   df=obs.size-1
   return chi2, R2

def chi_2_sigm(obs, cal, sigma, nparm):
    chi_2= 1* np.sum([(((a - b) / c)**2)
            for (a, b, c) in zip(obs[sigma > 0], cal[sigma > 0], sigma[sigma > 0])])
    #return chi_2, obs[sigma > 0].size
    return chi_2, chi_2/(obs.size-nparm)

# Derived Chi Squared Value For This Model
def chi_square1(f,y):
   chi2= np.sum(((f-y)*(f-y)/y))
   ss_res = np.sum((y - f)**2)
   ss_tot = np.sum((y - np.mean(y) )**2)
   R2 = 1 - (ss_res / ss_tot)
   return chi2,R2

def chi_sq2(obs, cal):
    chi_2 = 1* np.sum([((a - b) ** 2) / (b)
                            for (a, b) in zip(obs[cal > 0], cal[cal > 0])])
    #return chi_2, obs[cal > 0].size
    return chi_2, obs.size

def chi_sq(obs, cal):
   chi_2= np.sum(((obs-cal)*(obs-cal)/cal))
   return chi_2

def chi_sq2_test(obs, cal):
    chi_2, le = chi_sq2(obs, cal)
    df = le -1
    critical_val= chi2.ppf(q = 0.95, # Encuentre el valor crítico para el 95% de confianza*
                      df = df) # df= grado de libertad
    p_chi2 = 1- chi2.cdf(chi_2, df = df)
    return chi_2, p_chi2, critical_val, df, chi_2/df

def chi_square_sigm(obs, cal, sigma):
    chi_2= np.sum(((obs-cal)/sigma)**2)
    return chi_2

# Derived Chi Squared Value For This Model
def chi_sq(obs, cal):
   chi_2= np.sum(((obs-cal)*(obs-cal)/cal))
   return chi_2


def ks_test(data0, data1,  signif_level = 0.05):
    ks, pvalue = stats.ks_2samp(data0, data1)
    if signif_level < pvalue :
        response = "Accepted"
    else:
        response = "Rejected"
    return ks, pvalue, signif_level, response


emax =  4300### well capacity  4.1
pair_eh = 3.6 ## eV
emin = 5

siz_matz = 3
prc_r2 = 0.075
lower = 0.075
upper=lower

#strE= 'elect_hole_pairs_'+str(pair_eh)+'eV_'+str(emin)+'_'+str(emax)
strE= 'eh_'+str(pair_eh)+'eV'
print(strE)

eta_gaussdif = '_eta_'+ str(lower) +'_gauss'+str(siz_matz)+'_'+str(prc_r2)
eta_nogaussdif ='_eta_0' +'_gauss'+str(siz_matz)+'_0'

#eta_gaussdif = '_eta_'+ str(lower) +'_gauss'+str(siz_matz)
#eta_nogaussdif ='_eta_0' +'_gauss'+str(siz_matz)


z_count = 11

setpixel_z = 2

#dat = 'both'
ag='8'

sim_bg = '_ag8_bg'
sim_bg = ''

source = 'Sr90'
#source = 'Cs137'

chi2_lable = 1
title_lable = 0

cut_cl_size=0


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

ssd = 'C:/dat_2025/dat_2024_11/'
ssd = 'C:/dat_2025/Dg2_Sr90_40bin/'

log_y=0
zmax = 18
zmin = 0

x_dist =np.arange(zmin,zmax+1,2)
x_smooth = np.linspace(zmin, zmax, 100)

font_siz=24
grid_ = False
l_width = 5


plt_NOgaussdif = 0
pltgaussdif =1
plt_obs = 1
plt_comp = 0

bin_hist = 40
if(bin_hist>0): strbin = str(bin_hist)+'bin_z_'
else:strbin = 'z_'

simu_no_cut ='_sim_nocut'
simu_no_cut =''

max_adc=1024
#max_adc=150
min_adc=100

size_thresh = 5
max_thresh = 100

cut0 = 0; cut_dat00= 4; cut_dat01=None; cut_type0='_clsiz'
#cut0 = 1; cut_dat00= 10; cut_dat01=cut_dat00; cut_type0='_cl_size'
cut1 = 0; cut_dat10=50; cut_dat11=650; cut_type1='_cl_mean'
#cut1 = 1; cut_dat10=55; cut_dat11=55.00001; cut_type1='_cl_mean'
cut2 = 0; cut_dat20=101; cut_dat21=None; cut_type2='_clmax_adc'
cut3 = 0; cut_dat30=90; cut_dat31=None; cut_type3='_cl_totalE'


if(min_adc==0): cut_adc_str = ''
if(min_adc>0):cut_adc_str = '_adcG'+str(min_adc)

if(cut0==0):cut_type0 =''; cut_clst=''
if(cut0==True):
    if(cut_dat00== 0 and cut_dat01 < 100): cut_clst='_clstL'+str(cut_dat01)
    if(cut_dat00 > 0 and cut_dat01== None): cut_clst='_clstG'+str(cut_dat00)
    if(cut_dat00 > 0 and cut_dat01 != None): cut_clst='_clstB'+str(cut_dat00)+'_'+str(cut_dat01)
    if(cut_dat00==cut_dat01): cut_clst='_clst_'+str(cut_dat01)

if(cut1==0):cut_type1 =''

if(cut2==0):cut_type2 =''; cut_clmaxadc=''
if(cut2==True):
    if(cut_dat20== 0 and cut_dat21 < 1024): cut_clmaxadc='_cmaxL'+str(cut_dat21)
    if(cut_dat20 > 0 and cut_dat21== None): cut_clmaxadc='_cmaxG'+str(cut_dat20)
    if(cut_dat20 > 0 and cut_dat21 != None): cut_clmaxadc='_cmaxB'+str(cut_dat20)+'_'+str(cut_dat21)
    if(cut_dat20==cut_dat21): cut_clmaxadc='_cmaxL'+str(cut_dat21)


if(cut3==0):cut_type3 =''

str_cuts= cut_adc_str+cut_clst+cut_clmaxadc



if(plt_NOgaussdif==False and pltgaussdif==True): row = 2 ; str_hist_comp = '_gauss'
if(plt_NOgaussdif==True and pltgaussdif==False): row = 2; str_hist_comp = '_Nogauss'
if(plt_NOgaussdif==True and pltgaussdif==True): row = 3; str_hist_comp = '_ALL'
if(plt_NOgaussdif==False and pltgaussdif==False): row = 1 ; str_hist_comp = '_Obs'


adc_cut = 0

nsigm_bg = 5
navrg = 0
n_mean = 1

nframes = nfrm

nsigm_bg_sim = 0

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

if(size_thresh>0):
    cut_clst_size = '_clst'+ str(size_thresh)
    print(cut_clst_size)
if(size_thresh == 0 ):
    cut_clst_size = ''

if(max_thresh>0):
    cut_max = '_max'+ str(max_thresh)
if(max_thresh == 0 ):
    cut_max = ''

gauss_dist = 'Dg2'
nsigm = 0
k = 0.002
alpha = 1.75
zff =  1.0

total_sim = 10
n_sampl = total_sim - 0

sum_adc = 0
sum_clstr_siz = 0
sum2_adc = 0
sum2_clstr_siz = 0

sum_adc_nocut = 0
sum_clstr_siz_nocut = 0
sum2_adc_nocut = 0
sum2_clstr_siz_nocut = 0

for sim_n in range(total_sim):
    sim_n = '_s' + str(sim_n)

    if nsigm > 0 :
        difu_gauss = f"_{gauss_dist}_{k}k_{alpha}a_{nsigm}sig_{zff}zff"
        d_gauss = gauss_dist+'_z_'+r'$\sigma$ = '+str(nsigm)+r', $\k$ = '+str(k)+r', $\alpha$ = '+str(alpha)+ r', $z_{ff}$ = '+str(zff)
    if nsigm == 0 :
        difu_gauss = f"_{gauss_dist}_{k}k_{alpha}a_{zff}zff"
        d_gauss =gauss_dist+'_z_'+r'$\k$ = '+str(k)+r', $\alpha$ = '+str(alpha)+ r', $z_{ff}$ = '+str(zff)

    filt_cut = avrg_cut + sig_bg_thresh + sig_bg_thresh_sim + cut_clst_size+cut_max + cut_adc
    name_cut = '_'+str(z_count)+'l_'+fecha+'_'+evt+eta_gaussdif + filt_cut
    dirsave_plt = ssd +'Dg2_'+source+'_rslhist_'+strbin +'00mm'+cut_clst_size+cut_max+str_cuts+ \
                '/plots2'+sim_n+'_'+source+'_'+str(z_count)+'l'+'_'+str(nframes)+'f'+simu_no_cut+difu_gauss+ \
                '/plot_hist_mult_'+source + '_'+str(z_count)+'l_'+fecha+'_'+evt+sim_bg


    file_clst_inv_level = dirsave_plt+'/'+source+'_'+str(z_count)+'l_'+fecha+'_'+evt +'_sim_s2_num_clst'+cut_type2+cut_clmaxadc+'_dist'

    dat_clst_inv_level = np.load(file_clst_inv_level+'.npz')
    dat_clst_inv_level = np.float64(dat_clst_inv_level.f.arr_0)
    sum_clstr_siz += dat_clst_inv_level
    sum2_clstr_siz  += dat_clst_inv_level ** 2

    #dat_inv_level = np.load(file_total_adc+'.npy')
    #dat_inv_level = np.loadtxt(file_total_adc+'.csv', delimiter='\t', skiprows=1)

    file_adc_inv_level = dirsave_plt+'/'+source+'_'+str(z_count)+'l_'+fecha+'_'+evt  +'_sim_s2_total_ADC'+'_'+str(min_adc)+'_'+str(max_adc)+'ADC'+'_dist'

    dat_adc_inv_level = np.load(file_adc_inv_level+'.npz')
    dat_adc_inv_level = np.float64(dat_adc_inv_level.f.arr_0)
    sum_adc += dat_adc_inv_level
    sum2_adc += dat_adc_inv_level ** 2
    
    if (simu_no_cut == '_sim_nocut'):
        dirsave_plt_nocut = ssd +'Dg2_'+source+'_rslhist_'+strbin +'00mm'+'_adcG1'+ \
                    '/plots2'+sim_n+'_'+source+'_'+str(z_count)+'l'+'_'+str(nframes)+'f'+difu_gauss+ \
                    '/plot_hist_mult_'+source + '_'+str(z_count)+'l_'+fecha+'_'+evt+sim_bg
    
    
        file_clst_inv_level_nocut = dirsave_plt_nocut+'/'+source+'_'+str(z_count)+'l_'+fecha+'_'+evt +'_sim_s2_num_clst'+cut_type2+cut_clmaxadc+'_dist'
    
        dat_clst_inv_level_nocut = np.load(file_clst_inv_level_nocut+'.npz')
        dat_clst_inv_level_nocut = np.float64(dat_clst_inv_level_nocut.f.arr_0)
        sum_clstr_siz_nocut += dat_clst_inv_level_nocut
        sum2_clstr_siz_nocut  += dat_clst_inv_level_nocut ** 2
    
        #dat_inv_level = np.load(file_total_adc+'.npy')
        #dat_inv_level = np.loadtxt(file_total_adc+'.csv', delimiter='\t', skiprows=1)
    
        file_adc_inv_level_nocut = dirsave_plt_nocut+'/'+source+'_'+str(z_count)+'l_'+fecha+'_'+evt  +'_sim_s2_total_ADC'+'_'+str(1)+'_'+str(max_adc)+'ADC'+'_dist'
    
        dat_adc_inv_level_nocut = np.load(file_adc_inv_level_nocut+'.npz')
        dat_adc_inv_level_nocut = np.float64(dat_adc_inv_level_nocut.f.arr_0)
        sum_adc_nocut += dat_adc_inv_level_nocut
        sum2_adc_nocut += dat_adc_inv_level_nocut ** 2   
        
# Calcular promedios y desviaciones estándar
mean_clst = sum_clstr_siz / total_sim
std_clst = np.sqrt((sum2_clstr_siz - total_sim * mean_clst ** 2) / n_sampl )
err_clst_obs = np.sqrt((sum_clstr_siz[4] - 1000 * mean_clst[2] ** 2) / 999 )
err_clst_sim0 = np.sqrt((sum_clstr_siz[9] - 1000 * mean_clst[7] ** 2) / 999 )
err_clst_sim1 = np.sqrt((sum_clstr_siz[14] - 1000 * mean_clst[12] ** 2) / 999 )

err_clst_sim0_sqN = np.sqrt((sum_clstr_siz[9] - 1000 * mean_clst[7] ** 2) / 999 ) / np.sqrt(100)
err_clst_sim1_sqN = np.sqrt((sum_clstr_siz[14] - 1000 * mean_clst[12] ** 2) / 999 ) / np.sqrt(100)

mean_adc = sum_adc / total_sim
std_adc = np.sqrt((sum_adc - total_sim * mean_adc ** 2) / n_sampl )
err_adc_obs = np.sqrt((sum_adc[4] - 10000 * mean_adc[2] ** 2) / 9999 )
err_adc_sim0 = np.sqrt((sum_adc[9] - 10000 * mean_adc[7] ** 2) / 9999 )
err_adc_sim1 = np.sqrt((sum_adc[14] - 10000 * mean_adc[12] ** 2) / 9999 ) 

err_adc_sim0_sqN = np.sqrt((sum_adc[9] - 10000 * mean_adc[7] ** 2) / 9999 ) / np.sqrt(1000)
err_adc_sim1_sqN = np.sqrt((sum_adc[14] - 10000 * mean_adc[12] ** 2) / 9999 ) / np.sqrt(1000)

if (simu_no_cut == '_sim_nocut'):
    # Calcular promedios y desviaciones estándar
    mean_clst_nocut = sum_clstr_siz_nocut / total_sim
    std_clst_nocut = np.sqrt((sum2_clstr_siz_nocut - total_sim * mean_clst ** 2) / n_sampl )
    err_clst_sim0_nocut = np.sqrt((sum_clstr_siz_nocut[9] - 1000 * mean_clst_nocut[7] ** 2) / 999 )
    err_clst_sim1_nocut = np.sqrt((sum_clstr_siz_nocut[14] - 1000 * mean_clst_nocut[12] ** 2) / 999 )
    
    err_clst_sim0_sqN_nocut = np.sqrt((sum_clstr_siz_nocut[9] - 1000 * mean_clst_nocut[7] ** 2) / 999 ) / np.sqrt(100)
    err_clst_sim1_sqN_nocut = np.sqrt((sum_clstr_siz_nocut[14] - 1000 * mean_clst_nocut[12] ** 2) / 999 ) / np.sqrt(100)
    
    mean_adc_nocut = sum_adc_nocut / total_sim
    std_adc_nocut = np.sqrt((sum_adc_nocut - total_sim * mean_adc_nocut ** 2) / n_sampl )
    err_adc_sim0_nocut = np.sqrt((sum_adc_nocut[9] - 10000 * mean_adc_nocut[7] ** 2) / 9999 )
    err_adc_sim1_nocut = np.sqrt((sum_adc_nocut[14] - 10000 * mean_adc_nocut[12] ** 2) / 9999 ) 
    
    err_adc_sim0_sqN_nocut = np.sqrt((sum_adc_nocut[9] - 10000 * mean_adc_nocut[7] ** 2) / 9999 ) / np.sqrt(1000)
    err_adc_sim1_sqN_nocut = np.sqrt((sum_adc_nocut[14] - 10000 * mean_adc_nocut[12] ** 2) / 9999 ) / np.sqrt(1000)


for file_level in [#'file_total_adc_csv',
                       'file_clst_sum_csv'
                       ]:

    for err in ['std',
                #'std_sqN'
                ]:
        
        if( file_level == 'file_clst_sum_csv' ) :
            dat_inv_level = mean_clst;
            err_obs = err_clst_obs
            err_sim0 = err_clst_sim0
            err_sim1 = err_clst_sim1
            err_sim0_sqN = err_clst_sim0_sqN
            err_sim1_sqN = err_clst_sim1_sqN
            if (simu_no_cut == '_sim_nocut'):
                dat_inv_level[7] = mean_clst_nocut[7]
                dat_inv_level[10] = mean_clst_nocut[10]
                dat_inv_level[12] = mean_clst_nocut[12]
                dat_inv_level[15] = mean_clst_nocut[15]
                err_sim0t = err_clst_sim0_nocut
                err_sim1 = err_clst_sim1_nocut
                err_sim0_sqN = err_clst_sim0_sqN_nocut
                err_sim1_sqN = err_clst_sim1_sqN_nocut
            err='std'
        if( file_level == 'file_total_adc_csv' ):
            dat_inv_level = mean_adc ;
            err_obs = err_adc_obs
            err_sim0 = err_adc_sim0
            err_sim1 = err_adc_sim1
            err_sim0_sqN = err_adc_sim0_sqN
            err_sim1_sqN = err_adc_sim1_sqN
            if (simu_no_cut == '_sim_nocut'):
                dat_inv_level[7] =  mean_adc_nocut[7]
                dat_inv_level[10] =  mean_adc_nocut[10]
                dat_inv_level[12] =  mean_adc_nocut[12]
                dat_inv_level[15] =  mean_adc_nocut[15]
                err_sim0 = err_adc_sim0_nocut
                err_sim1 = err_adc_sim1_nocut
                err_sim0_sqN = err_adc_sim0_sqN_nocut 
                err_sim1_sqN = err_adc_sim1_sqN_nocut
            err='std'

        err_gr0 = np.where(dat_inv_level[13][zmin//2:zmax//2+1]>0) or np.where(dat_inv_level[8][zmin//2:zmax//2+1]>0) or np.where(dat_inv_level[3][zmin//2:zmax//2+1]>0)
        x_dist=x_dist[err_gr0]

        for norm in [True,
                     False
                     ]:
            if(norm == True):dens = 'norm_'
            if(norm == False):dens = ''

            if(norm == True):
                y_dist_obs = dat_inv_level[2][err_gr0]/dat_inv_level[2][err_gr0].sum()
                err_sum_obs = np.sqrt(np.sum(dat_inv_level[3][err_gr0]*dat_inv_level[3][err_gr0]) )
                #if(err=='std'):err_y_dist_obs = y_dist_obs*np.sqrt((dat_inv_level[3][err_gr0]/dat_inv_level[2][err_gr0])**2+(err_sum_obs/dat_inv_level[2][err_gr0].sum())**2)
                if(err=='std'):err_y_dist_obs = dat_inv_level[3][err_gr0]/dat_inv_level[2][err_gr0].sum()
                #if(err=='std'):err_y_dist_obs = np.sqrt((dat_inv_level[3][err_gr0]/dat_inv_level[2][err_gr0].sum())*(dat_inv_level[3][err_gr0]/dat_inv_level[2][err_gr0].sum()) + 0.04*y_dist_obs*y_dist_obs)
                if(err=='std_sqN'):err_y_dist_obs = dat_inv_level[5][err_gr0]/dat_inv_level[5][err_gr0].sum()

                y_dist_sim = dat_inv_level[7][err_gr0]/dat_inv_level[7][err_gr0].sum()
                err_sum_sim = np.sqrt(np.sum(err_sim0[err_gr0]*err_sim0[err_gr0]) )
                #if(err=='std'):err_y_dist_sim = y_dist_sim*np.sqrt((err_sim0[err_gr0]/dat_inv_level[7][err_gr0])**2+(err_sum_sim/dat_inv_level[7][err_gr0].sum())**2)
                if(err=='std'):err_y_dist_sim = err_sim0[err_gr0]/dat_inv_level[7][err_gr0].sum()
                #if(err=='std'):err_y_dist_sim =  np.sqrt((err_sim0[err_gr0]/dat_inv_level[7][err_gr0].sum())*(err_sim0[err_gr0]/dat_inv_level[7][err_gr0].sum()) + 0.04*y_dist_sim*y_dist_sim)
                if(err=='std_sqN'):err_y_dist_sim = err_sim0_sqN[err_gr0]/dat_inv_level[10][err_gr0].sum()

                y_dist_sim_gauss = dat_inv_level[12][err_gr0]/dat_inv_level[12][err_gr0].sum()
                if(err=='std'):err_y_dist_sim_gauss = err_sim1[err_gr0]/dat_inv_level[12][err_gr0].sum()
                #if(err=='std'):err_y_dist_sim_gauss = np.sqrt((err_sim1[err_gr0]/err_sim1[err_gr0].sum())*(err_sim1[err_gr0]/err_sim1[err_gr0].sum()) + 0.04*y_dist_sim_gauss*y_dist_sim_gauss)
                if(err=='std_sqN'):err_y_dist_sim_gauss = err_sim1_sqN[err_gr0]/dat_inv_level[15][err_gr0].sum()

            if(norm == False):
                y_dist_obs = dat_inv_level[2][err_gr0]
                if(err=='std'):err_y_dist_obs = dat_inv_level[3][err_gr0]#+dat_inv_level[2][err_gr0]/5
                #if(err=='std'):err_y_dist_obs = np.sqrt(dat_inv_level[3][err_gr0]*dat_inv_level[3][err_gr0] + 0.04*y_dist_obs*y_dist_obs)
                if(err=='std_sqN'):err_y_dist_obs = dat_inv_level[5][err_gr0]

                y_dist_sim = dat_inv_level[7][err_gr0]
                if(err=='std'):err_y_dist_sim = err_sim0[err_gr0]##+dat_inv_level[7][err_gr0]/5
                #if(err=='std'):err_y_dist_sim = np.sqrt(err_sim0[err_gr0]*err_sim0[err_gr0] + 0.04*y_dist_sim*y_dist_sim)
                if(err=='std_sqN'):err_y_dist_sim = err_sim0_sqN[err_gr0]

                y_dist_sim_gauss = dat_inv_level[12][err_gr0]
                if(err=='std'):err_y_dist_sim_gauss = err_sim1[err_gr0]#+dat_inv_level[14][err_gr0]/5
                #if(err=='std'):err_y_dist_sim_gauss = np.sqrt(err_sim1[err_gr0]*err_sim1[err_gr0] + 0.04*y_dist_sim_gauss*y_dist_sim_gauss)
                if(err=='std_sqN'):err_y_dist_sim_gauss = err_sim1_sqN[err_gr0]

            pfit_dist_obs = np.polyfit(x_dist,y_dist_obs,len(x_dist)-1)
            ysmooth_dist_obs = np.polyval(pfit_dist_obs,x_smooth)

            pfit_dist_sim = np.polyfit(x_dist,y_dist_sim,len(x_dist)-1)
            ysmooth_dist_sim = np.polyval(pfit_dist_sim,x_smooth)

            pfit_dist_sim_gauss = np.polyfit(x_dist,y_dist_sim_gauss,len(x_dist)-1)
            ysmooth_dist_sim_gauss = np.polyval(pfit_dist_sim_gauss,x_smooth)

            ###################################################################################################################
            ###################################################################################################################


            try:

                popt_dist_obs, pcov_dist_obs = curve_fit(inver_r2, x_dist, y_dist_obs,
                                   sigma = err_y_dist_obs, p0 = [1.,0.0000001],  check_finite = False)
                sig_dist_obs = np.sqrt(np.diag(pcov_dist_obs))
                chiq_dist_obs, R_dist_obs = chi_square(inver_r2(x_dist, *popt_dist_obs), y_dist_obs, err_y_dist_obs)
                print('chi2_obs',chiq_dist_obs)
                print('obs: a = '+str(popt_dist_obs[0])+' \+- '+str(sig_dist_obs[0]))
                print('obs: b = '+str(popt_dist_obs[1])+' \+- '+str(sig_dist_obs[1]))

                ks_dist_obs = ks_test(inver_r2(x_dist, *popt_dist_obs), y_dist_obs)
                print(ks_dist_obs)
            except RuntimeError:
                if( file_level == 'file_total_adc_csv' ):
                    print('no fit total_adc_level_OBS')
                if( file_level == 'file_clst_sum_csv' ):
                    print('no fit Total_Number_Culsters_per_level_OBS')

                chiq_dist_obs=100000000
                pass

            ###################################################################################################################

            try:

                popt_dist_sim, pcov_dist_sim = curve_fit(inver_r2, x_dist, y_dist_sim,
                                    sigma = err_y_dist_sim, p0 =  [1.,0.0000001],  check_finite = False)
                sig_dist_sim = np.sqrt(np.diag(pcov_dist_sim))
                #chiq_dist_sim, chiq_dist_ndf_sim = chi_2_sigm(inver_r2(x_dist, *popt_dist_sim), y_dist_sim, err_y_dist_sim, len(popt_dist_sim))
                chiq_dist_sim, R_dist_sim = chi_square(inver_r2(x_dist, *popt_dist_sim), y_dist_sim, err_y_dist_sim)
                print('chi2_sim',chiq_dist_sim)
                print('sim: a = '+str(popt_dist_sim[0])+' \+- '+str(sig_dist_sim[0]))
                print('sim: b = '+str(popt_dist_sim[1])+' \+- '+str(sig_dist_sim[1]))

                ks_dist_sim = ks_test(inver_r2(x_dist, *popt_dist_sim), y_dist_sim)
                print(ks_dist_sim)
            except RuntimeError:
                if( file_level == 'file_total_adc_csv' ):
                    print('no fit total_adc_level_sim')
                if( file_level == 'file_clst_sum_csv' ):
                    print('no fit Total_Number_Culsters_per_level_sim')
                chiq_dist_sim=100000000

                pass

            ###################################################################################################################

            try:

                popt_dist_sim_gauss, pcov_dist_sim_gauss = curve_fit(inver_r2, x_dist, y_dist_sim_gauss,
                                   sigma = err_y_dist_sim_gauss, p0 = [1.,0.0000001],  check_finite = False)
                sig_dist_sim_gauss = np.sqrt(np.diag(pcov_dist_sim_gauss))
                chiq_dist_sim_gauss, R_dist_sim_gauss = chi_square(inver_r2(x_dist, *popt_dist_sim_gauss), y_dist_sim_gauss, err_y_dist_sim_gauss)
                print('chi2-sim_gauss',chiq_dist_sim_gauss)
                print('sim_gauss: a = '+str(popt_dist_sim_gauss[0])+' \+- '+str(sig_dist_sim_gauss[0]))
                print('sim_gauss: b = '+str(popt_dist_sim_gauss[1])+' \+- '+str(sig_dist_sim_gauss[1]))

                ks_dist_sim_gauss = ks_test(inver_r2(x_dist, *popt_dist_sim_gauss), y_dist_sim_gauss)
                print(ks_dist_sim_gauss)
            except RuntimeError:
                if( file_level == 'file_total_adc_csv' ):
                    print('no fit total_adc_level_sim_gauss')
                if( file_level == 'file_clst_sum_csv' ):
                    print('no fit Total_Number_Culsters_per_level_sim_gauss')
                chiq_dist_sim_gauss=100000000
                pass
            ###################################################################################################################
            ###################################################################################################################
            err_obs_sim = np.sqrt(err_y_dist_obs*err_y_dist_obs + err_y_dist_sim*err_y_dist_sim)
            chi2_dist_obs_sim = chi_2_sigm(y_dist_obs, y_dist_sim, err_obs_sim, 1)
            print('chi2_obs_sim:', chi2_dist_obs_sim[0], ' ch2/df: ', chi2_dist_obs_sim[1] )
            ks_dist_obs_sim = ks_test(y_dist_obs, y_dist_sim)
            print(ks_dist_obs_sim)
            ###################################################################################################################
            err_obs_sim_gauss = np.sqrt(err_y_dist_obs*err_y_dist_obs + err_y_dist_sim_gauss*err_y_dist_sim_gauss)
            chi2_dist_obs_sim_gauss = chi_2_sigm(y_dist_obs, y_dist_sim_gauss, err_obs_sim_gauss, 1)
            print('chi2_obs_sim_gauss:', chi2_dist_obs_sim_gauss[0], ' ch2/df: ', chi2_dist_obs_sim_gauss[1] )
            ks_dist_obs_sim_gauss = ks_test(y_dist_obs, y_dist_sim_gauss)
            print(ks_dist_obs_sim_gauss)
            ###################################################################################################################


            tex_ch2_dist_obs = "$\chi^{2}$  = " +"%.3f" %(chiq_dist_obs)
            tex_ch2_df_dist_obs = r"$\chi^2_\nu$" +" = " +"%.3f" %(chiq_dist_obs/(len(x_dist)-len(popt_dist_obs))) \
                + '\n a = '+"%.3f" %(popt_dist_sim[0])+' $\pm$ '+"%.3f" %(sig_dist_sim[0]) \
                + ' b = '+"%.3f" %(popt_dist_sim[1])+' $\pm$ '+"%.3f" %(sig_dist_sim[1])
            lbl_dist_obs = source+' data'
            ###################################################################################################################
            tex_ch2_dist_sim = "$\chi^{2}$  = " +"%.3f" %(chiq_dist_sim)
            tex_ch2_df_dist_sim =   r"$\chi^2_\nu$" +" = " +"%.3f" %(chiq_dist_sim/(len(x_dist)-len(popt_dist_sim))) \
                +'\n a = '+"%.3f" %(popt_dist_sim[0])+' $\pm$ '+"%.3f" %(sig_dist_sim[0]) \
                +' b = '+"%.3f" %(popt_dist_sim[1])+' $\pm$ '+"%.3f" %(sig_dist_sim[1])
            if(err=='std'):tex_ch2_df_dist_obs_sim =  r"$\chi^2_\nu$"+" = " +"%.3f" %(chi2_dist_obs_sim[1]) +'(obs & sim )'
            if(err=='std_sqN'):tex_ch2_df_dist_obs_sim =  r"$\chi^2_\nu$"+" = " +"%.3f" %(chi2_dist_obs_sim[1]) +'(obs & sim $Err=sdt/\sqrt{N}$)'
            lbl_dist_sim = source+' simulation NO gauss' + simu_no_cut
            ###################################################################################################################
            tex_ch2_dist_sim_gauss = "$\chi^{2}$  = " +"%.3f" %(chiq_dist_sim_gauss) 
            tex_ch2_df_dist_sim_gauss =  r"$\chi^2_\nu$" +" = " +"%.3f" %(chiq_dist_sim_gauss/(len(x_dist)-len(popt_dist_sim_gauss))) \
                +'\n a = '+"%.3f" %(popt_dist_sim_gauss[0])+' $\pm$ '+"%.3f" %(sig_dist_sim_gauss[0]) \
                +' b = '+"%.3f" %(popt_dist_sim_gauss[1])+' $\pm$ '+"%.3f" %(sig_dist_sim_gauss[1])
            if(err=='std'):tex_ch2_df_dist_obs_sim_gauss =  r"$\chi^2_\nu$"+" = " +"%.3f" %(chi2_dist_obs_sim_gauss[1]) +'(obs & sim )'
            if(err=='std_sqN'):tex_ch2_df_dist_obs_sim_gauss =  r"$\chi^2_\nu$"+" = " +"%.3f" %(chi2_dist_obs_sim_gauss[1]) +'(obs & sim $Err=sdt/\sqrt{N}$)'
            lbl_dist_sim_gauss = source+' simulation' + simu_no_cut
            ###################################################################################################################

            colr='k'

            if(file_level == 'file_total_adc_csv'):etiq = 'avg_ADC_dist_'+err
            if(file_level == 'file_clst_sum_csv'):etiq = 'avg_clstr_dist_'+err

            path_dir_plot= 'C:/dat_2025/plots_ivn2_mean_'+source+'_'+fecha+'_'+str(zmin)+'_'+str(zmax)+'mm'
            
            try:
                os.makedirs(path_dir_plot)
            except FileExistsError:
                #directory already exists
                pass

            labl_obs=lbl_dist_obs
            labl_sim=lbl_dist_sim
            labl_sim_gauss=lbl_dist_sim_gauss

            if(chi2_lable==True):
                tex_ch2_obs = '\n'+tex_ch2_dist_obs
                tex_ch2df_obs = '\t'+tex_ch2_df_dist_obs
                tex_ch2_sim = '\n'+tex_ch2_dist_sim
                tex_ch2df_sim = '\t'+tex_ch2_df_dist_sim
                tex_ch2df_obs_sim = '\n'+tex_ch2_df_dist_obs_sim
                tex_ch2_sim_gauss = '\n'+tex_ch2_dist_sim_gauss
                tex_ch2df_sim_gauss = '\t'+tex_ch2_df_dist_sim_gauss
                tex_ch2df_obs_sim_gauss = '\n'+tex_ch2_df_dist_obs_sim_gauss


            data_y_dist_obs=y_dist_obs
            er_data_y_dist_obs=err_y_dist_obs
            data_y_dist_sim=y_dist_sim
            er_data_y_dist_sim=err_y_dist_sim
            data_y_dist_sim_gauss=y_dist_sim_gauss
            er_data_y_dist_sim_gauss=err_y_dist_sim_gauss

            try:
                ch2_obs=chiq_dist_obs
                popt_obs=popt_dist_obs
                y_smooth_obs =ysmooth_dist_obs
                ch2_sim=chiq_dist_sim
                popt_sim=popt_dist_sim
                y_smooth_sim =ysmooth_dist_sim
                ch2_sim_gauss=chiq_dist_sim_gauss
                popt_sim_gauss=popt_dist_sim_gauss
                y_smooth_sim_gauss =ysmooth_dist_sim_gauss
            except NameError:
                if( file_level == 'file_total_adc_csv' ):
                    print('no chi2 Tolal ADC per level')
                if( file_level == 'file_clst_sum_csv' ):
                    print('no chi2 Total Number Culsters per level')

                pass

            colr_obs='C3'
            colr_sim='C2'
            colr_sim_gauss='C0'

            plt.rcParams.update({'font.size': font_siz})
            if(plt_comp==True):fig, (ax0, ax1) = plt.subplots(2,1,figsize=(14.5,12),gridspec_kw={'height_ratios': [5, 1]} )#, sharex=True )
            if(plt_obs==True and plt_NOgaussdif!=True and pltgaussdif!=True and plt_comp==True):
                fig, (ax0) = plt.subplots(figsize=(14.5,9.5) )
            if(plt_comp==False):
                fig, (ax0) = plt.subplots(figsize=(14.5,9.5) )

            if(plt_obs==True):
                ax0.plot(x_dist, data_y_dist_obs, colr_obs+'o', markersize=l_width*2.5, linewidth=l_width)
                ax0.errorbar(x_dist, data_y_dist_obs,  yerr= er_data_y_dist_obs, ecolor = colr_obs, fmt = 'none', capsize=l_width*2, capthick=l_width, linewidth=l_width)

                try:
                    if(chi2_lable==True):ax0.plot(x_smooth,inver_r2(x_smooth, *popt_obs), colr_obs+'-', label=labl_obs +tex_ch2_obs +tex_ch2df_obs, linewidth=l_width)
                    if(chi2_lable==False):ax0.plot(x_smooth,inver_r2(x_smooth, *popt_obs), colr_obs+'-', label=labl_obs, linewidth=l_width)

                except NameError:
                    ax0.plot(x_dist, data_y_dist_obs, colr_obs+'-', label=labl_obs, linewidth=l_width)
                    print('no plot ', labl_obs)
                    pass

            if(plt_NOgaussdif==True ):
                ax0.plot(x_dist, data_y_dist_sim, colr_sim+'o', markersize=l_width*2.5, linewidth=l_width)
                ax0.errorbar(x_dist, data_y_dist_sim,  yerr= er_data_y_dist_sim, ecolor = colr_sim, fmt = 'none', capsize=l_width*2, capthick=l_width, linewidth=l_width-1)
                #ax0.plot(x_smooth,inver_r2(x_smooth, *popt_sim), colr_sim+'--', label=labl_sim+tex_ch2_sim+tex_ch2df_sim+tex_ch2df_obs_sim, linewidth=l_width)
                try:
                    #if(chiq_dist_sim!=inf):
                    if(chi2_lable==True):ax0.plot(x_smooth,inver_r2(x_smooth, *popt_sim), colr_sim+'--', label=labl_sim +tex_ch2_sim +tex_ch2df_sim+tex_ch2df_obs_sim, linewidth=l_width)
                    if(chi2_lable==False):ax0.plot(x_smooth,inver_r2(x_smooth, *popt_sim), colr_sim+'--', label=labl_sim, linewidth=l_width)
                    #ax0.plot(x_smooth,ysmooth_dist_sim, colr_sim+'--', label=labl_sim+tex_ch2_sim+tex_ch2df_sim+tex_ch2df_obs_sim, linewidth=l_width)

                except NameError:
                    ax0.plot(x_dist, data_y_dist_sim, colr_sim+'-', label=labl_sim, linewidth=l_width)
                    print('no plot ', labl_sim)
                    pass

            if(pltgaussdif==True ):
                ax0.plot(x_dist, data_y_dist_sim_gauss, colr_sim_gauss+'o', markersize=l_width*2.5, linewidth=l_width)
                ax0.errorbar(x_dist, data_y_dist_sim_gauss,  yerr= er_data_y_dist_sim_gauss, ecolor = colr_sim_gauss, fmt = 'none', capsize=l_width*2, capthick=l_width, linewidth=l_width-1)
                try:
                    if(chi2_lable==True):ax0.plot(x_smooth,inver_r2(x_smooth, *popt_sim_gauss), colr_sim_gauss+'--', label=labl_sim_gauss+tex_ch2_sim_gauss+tex_ch2df_sim_gauss+tex_ch2df_obs_sim_gauss, linewidth=l_width)
                    if(chi2_lable==False):ax0.plot(x_smooth,inver_r2(x_smooth, *popt_sim_gauss), colr_sim_gauss+'--', label=labl_sim_gauss, linewidth=l_width)

                except NameError:
                    ax0.plot(x_dist, data_y_dist_sim_gauss, colr_sim_gauss+'-', label=labl_sim_gauss, linewidth=l_width)
                    #ax0.plot(x_smooth,inver_r2(x_smooth, *popt_sim_gauss), colr_sim_gauss+'--', label=labl_sim_gauss, linewidth=l_width)

                    print('no plot ', labl_sim_gauss)
                    pass

            if(plt_obs==True):
                if(log_y==True):ax0.set_yscale('log')
                ax0.grid(grid_)
                ax0.set_xticks(x_dist)#, size=font_siz+2)
                #minor_ticks_x= np.arange(min_adc, max_adc, 200 )
                ax0.tick_params(labelbottom=False, direction='inout', width=3, length=14 )
                ax0.minorticks_on()
                ax0.tick_params(axis='both', direction='inout', which='minor',length=8 ,width=2)
                #ax0.yticks(size=font_siz+2)
                if(plt_obs==True and plt_NOgaussdif!=True and pltgaussdif!=True or plt_comp==False):
                    ax0.set_xlabel('Source-detector distance (mm)', fontsize=font_siz+2)
                    ax0.tick_params(labelbottom=True, direction='inout', width=3, length=14 )
                #fig.suptitle('fit inv r2 '+etiq+cut_clst_size, fontsize=font_siz )
                if(title_lable==True):fig.suptitle('fit inv r2 '+'cut'+cut_clst_size, fontsize=font_siz )


            ax0.legend(loc='upper right', fontsize=font_siz-2)
            if( file_level == 'file_total_adc_csv' ): y_label_dat = 'Average Tolal ADC per level'
            if( file_level == 'file_clst_sum_csv' ): y_label_dat = 'Average Number of Culsters'

            ax0.set_ylabel(y_label_dat, fontsize=font_siz)

            if(plt_comp==True):
                if(plt_NOgaussdif==True):
                    comp_sim = (y_dist_obs - y_dist_sim)/err_obs_sim

                    ax1.axhline(y = 0., color = 'k', linestyle = '--')
                    ax1.errorbar(x_dist, comp_sim, yerr=0.*comp_sim, fmt='C2'+'>',  markersize=12 )
                    ax1.set_xticks(x_dist, size=font_siz+2)

                if(pltgaussdif==True):
                    comp_sim_gauss = (y_dist_obs - y_dist_sim_gauss)/err_obs_sim_gauss

                    ax1.axhline(y = 0., color = 'k', linestyle = '--')
                    ax1.errorbar(x_dist, comp_sim_gauss, yerr=0.*comp_sim_gauss, fmt='C0'+'o',  markersize=10 )
                    ax1.set_xticks(x_dist, size=font_siz+2)

                if(pltgaussdif==True or plt_NOgaussdif==True):
                    ylabl_comp1 = '$( data - sim)/\sigma$'
                    #ylabl_comp1= r'$\frac{\Delta ( data - sim)}{\sigma}$'
                    ylabl_comp1= r'$\frac{ ( data - sim)}{\sigma}$'

                    #ax[2].xticks(size=font_siz+2)
                    #ax[2].yticks(size=font_siz+2)

                    ax1.tick_params(labelbottom=True, width=3,  direction='inout', length=14 )
                    ax1.minorticks_on()
                    ax1.tick_params(axis='x', direction='inout', which='minor',length=8 ,width=2)
                    ax1.set_xlabel('Source-detector distance (mm)', fontsize=font_siz+2)
                    ax1.set_ylabel(ylabl_comp1, fontsize=font_siz+2)
                    #ax[2].yaxis.tick_right()
                    #ax[2].yaxis.set_label_position("right")
                    if(grid_==True):ax1.grid(axis = 'y',  linestyle = '--',)

            fig.tight_layout()
            if(chi2_lable==False):
                plt.savefig(path_dir_plot+'/'+'plot_fit_inv2_'+dens+source+'_'+fecha+'_'+etiq+filt_cut+simu_no_cut+str_cuts+'.png', dpi=150)
                plt.savefig(path_dir_plot+'/'+'plot_fit_inv2_'+dens+source+'_'+fecha+'_'+etiq+filt_cut+simu_no_cut+str_cuts+'.pdf', dpi=150)
            if(chi2_lable==True):
                plt.savefig(path_dir_plot+'/'+'plot_fit_inv2_'+dens+source+'_'+fecha+'_'+etiq+filt_cut+simu_no_cut+str_cuts+'_chi2.png', dpi=150)
                plt.savefig(path_dir_plot+'/'+'plot_fit_inv2_'+dens+source+'_'+fecha+'_'+etiq+filt_cut+simu_no_cut+str_cuts+'_chi2.pdf', dpi=150)

            plt.show()
