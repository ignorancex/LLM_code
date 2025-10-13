#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Tue Jun 28 11:43:38 2022

@author: mik
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib

import scipy
from scipy.stats import norm

from scipy.stats import ks_2samp
from scipy.stats import kstest

from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.image as mpimg

import os

from scipy import stats
#from mlxtend.evaluate import permutation_test

#from netneurotools import stats as nnstats

from statsmodels.stats.weightstats import DescrStatsW

from scipy.stats import chi2

from scipy import signal

from scipy.optimize import curve_fit

###############################################################################
###############################################################################


def chi_square(obs,cal,sigma):
   chi2= np.sum(((obs-cal)/sigma)**2)
   ss_res = np.sum((cal - obs)**2)
   ss_tot = np.sum((cal - np.mean(cal) )**2)
   R2 = 1 - (ss_res / ss_tot)
   df=obs.size-1
   return chi2, R2

def chi_square_sigm(obs, cal, sigma):
    chi_2= np.sum(((obs-cal)/sigma)**2)
    return chi_2, obs.size

# Derived Chi Squared Value For This Model
def chi_sq(obs, cal):
   chi_2= np.sum(((obs-cal)*(obs-cal)/cal))
   return chi_2, obs.size


def chi2_dist(histA,histB, eps = 1.e-10):
    dist = np.sum((histA-histB)**2/(histA+histB+eps))
    return 0.5*dist


def chi_sq2(obs, cal):
    chi_2 = 1* np.sum([((a - b) ** 2) / (b)
                            for (a, b) in zip(obs[cal > 0], cal[cal > 0])])
    #return chi_2, obs[cal > 0].size
    return chi_2, obs.size

def chi_2_sigm(obs, cal, sigma):
    chi_2= 1* np.sum([(((a - b) / c)**2)
            for (a, b, c) in zip(obs[sigma > 0], cal[sigma > 0], sigma[sigma > 0])])
    #return chi_2, obs[sigma > 0].size
    return chi_2, obs.size

# Function to calculate Chi-distace
def chi2_distance(histA, histB,  eps = 1.e-10):
	# compute the chi-squared distance using above formula
	chi = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
					for (a, b) in zip(histA, histB)])
	return chi

def chi2_dist_norm(obs, sim , eps = 1.e-10):
	# compute the chi-squared distance using above formula
    obst=np.sum([a for a in zip(obs)])
    simt=np.sum([b for b in zip(sim)])
    chi_2 = 0.5 *np.sum([((a/obst - b/simt) ** 2) / (a/obst + b/simt + eps)
					for (a, b) in zip(obs, sim)])
    return chi_2


def chi2_distance0( obs, cal ):

    #chi0 = 0.5 * np.sum([((a - b) ** 2) / (a +  b + 1)
	#				for (a, b) in zip(obs[obs+cal == 0], cal[obs+cal == 0] )])
    #print(obs[obs+cal == 0].size)

    chi1 = 0.5 * np.sum([((a - b) ** 2) / (a + b)
					for (a, b) in zip(obs[obs+cal > 0], cal[obs+cal > 0] )])
    #print(obs[obs+cal > 0].size)

    chi_2 = chi1 #+ chi0
    #return chi_2, obs[obs+cal > 0].size
    return chi_2, obs[obs].size

def chi2_dist_norm0(obs, sim ):

    obst=np.sum([a for a in zip(obs)])
    simt=np.sum([b for b in zip(sim)])

   # chi0 = 0.5 *np.sum([((a/obst - b/simt) ** 2) / (a/obst + b/simt + 1 )
	#				for (a, b) in zip(obs[obs+sim == 0], sim[obs+sim == 0] )])

    chi1 = 0.5 *np.sum([((a/obst - b/simt) ** 2) / (a/obst + b/simt )
					for (a, b) in zip(obs[obs+sim > 0], sim[obs+sim > 0] )])

    chi_2 = chi1 #+ chi0
    #return chi_2, obs[obs+sim > 0].size
    return chi_2, obs[obs].size


def chi_sq2_test(obs, cal):
    chi_2, le = chi_sq2(obs, cal)
    df = le -1
    critical_val= chi2.ppf(q = 0.95, # Encuentre el valor crítico para el 95% de confianza*
                      df = df) # df= grado de libertad
    p_chi2 = 1- chi2.cdf(chi_2, df = df)
    return chi_2, p_chi2, critical_val, df, chi_2/df

def chi_2_sigm_test(obs, cal, sigma):
    chi_2, le = chi_2_sigm(obs, cal, sigma)
    df = le -1
    critical_val= chi2.ppf(q = 0.95, # Encuentre el valor crítico para el 95% de confianza*
                      df = df) # df= grado de libertad
    p_chi2 = 1- chi2.cdf(chi_2, df = df )
    return chi_2, p_chi2, critical_val, df, chi_2/df

def chi2_distance_test(obs, cal,  eps = 1.e-10):
    chi_2 = chi2_distance(obs, cal, eps)
    df = obs.size -1
    critical_val= chi2.ppf(q = 0.95, # Encuentre el valor crítico para el 95% de confianza*
                      df = df) # df= grado de libertad
    p_chi2 = 1- chi2.cdf(chi_2, df = df)
    return chi_2, p_chi2,  critical_val, df, chi_2/df

def chi2_dist_norm_test(obs, cal,  eps = 1.e-10):
    chi_2 = chi2_dist_norm(obs, cal, eps)
    df = obs.size -1
    critical_val= chi2.ppf(q = 0.95, # Encuentre el valor crítico para el 95% de confianza*
                      df = df) # df= grado de libertad
    p_chi2 = 1- chi2.cdf(chi_2, df = df)
    return chi_2, p_chi2,  critical_val, df, chi_2/df


def chi2_distance_test0(obs, cal ):
    chi_2, le = chi2_distance0(obs, cal)
    df = le -1
    critical_val= chi2.ppf(q = 0.95, df = df) # df= grado de libertad
    p_chi2 = 1- chi2.cdf(chi_2, df = df)
    return chi_2, p_chi2,  critical_val, df, chi_2/df

def chi2_dist_norm_test0(obs, cal ):
    chi_2, le = chi2_dist_norm0(obs, cal)
    df = le -1
    critical_val= chi2.ppf(q = 0.95,  df = df) # df= grado de libertad
    p_chi2 = 1- chi2.cdf(chi_2, df = df)
    return chi_2, p_chi2,  critical_val, df, chi_2/df


def comp_chi2(A, B):
    chi = 1 * np.sum([((a - b) ** 2) / (a * b)
                                       for (a, b) in zip(A>0, B>0)])
    return chi

def comp_interset(A, B):
	#dist = np.sum([min(a , b) for (a, b) in zip(A, B)])
	return np.sum(np.minimum(A,B))

def comp_inter_nor(A, B):
	#dist = np.sum([min(a , b) for (a, b) in zip(A, B)])
	return np.sum(np.minimum(A,B))/np.sum(A)

def dist_Bhat(A, B):
    #BC = np.sum([np.sqrt(a*b) for (a, b) in zip(A, B)])
    #dist=-np.log(BC)
    dist=-np.log(np.sum(np.sqrt(A*B)) )
    return dist

def dist_Bhat1(A, B):
    sAB = np.sum(A)*np.sum(B)
    #BC = np.sum([np.sqrt(a*b/sAB) for (a, b) in zip(A, B)])
    #dist=np.sqrt(1-BC)

    dist=np.sqrt(1-np.sum(np.sqrt(A*B/sAB)))
    return dist

def dist_Hellinger(A, B):
    #BC = np.sum([np.sqrt(a*b) for (a, b) in zip(A, B)])
    #dist=np.sqrt(1-BC)
    dist=np.sqrt(1-np.sum(np.sqrt(A*B)))
    return dist


#only for continuos variable
def ks_test(data0, data1,  signif_level = 0.01):
    ks, pvalue = stats.ks_2samp(data0, data1)
    if signif_level < pvalue :
        response = 'Accepted'
    else:
        response = 'Rejected'
    return ks, pvalue, signif_level, response

def ad_test(data0, data1):
    ad, cv, val = stats.anderson_ksamp([data0, data1])
    if ad < cv[2] :
        response = 'Accepted'
    else:
        response = 'Rejected'
    return ad, cv[2], response
###############################################################################

def prmut_test(data0, data1, signif_level = 0.01):
    #signif_level = 0.01 as our significance level.
    pvalue = permutation_test(data0, data1, method='approximate', num_rounds=2000, seed=0)  #Two-sided permutation test
    if signif_level < pvalue :
        response = 'Accepted'
    else:
        response = 'Rejected'
    return pvalue, signif_level, response
###############################################################################


#def cut_data(cut, cut_dat0, cut_dat1, cut_type, data1, data2, data_fondo, cut_str):
def cut_data(cut, cut_dat0, cut_dat1, cut_type, data1, cut_str):
    if(cut==True):

        if(cut_type=='_clsiz'):
            colm=7
            #r0 = np.max(np.array((  np.max(data1[:,colm]), np.max(data2[:,colm]),  np.max(data_fondo[:,colm])  )))
            r0 = np.max(data1[:,colm])
        if(cut_type=='_cl_mean'):
            colm=8
            #r0 = np.max(np.array((  np.max(data1[:,colm]), np.max(data2[:,colm]),  np.max(data_fondo[:,colm])  )))
            r0 = np.max(data1[:,colm])
        if(cut_type=='_cl_totalE'):
            colm=9
            #r0 = np.max(np.array((  np.max(data1[:,colm]), np.max(data2[:,colm]),  np.max(data_fondo[:,colm])  )))
            r0 = np.max(data1[:,colm])
        if(cut_type=='_clmax_adc'):
            colm=10
            #r0 = np.max(np.array((  np.max(data1[:,colm]), np.max(data2[:,colm]),  np.max(data_fondo[:,colm])  )))
            r0 = np.max(data1[:,colm])

        #if(cut_type=='_ADC_cl_size'):
        #    colm=0
        #    r0 = np.max(data1[:,colm])

        if(cut_dat1==None):
            cut_dat1 = r0

        '''
        cut0 = 0; cut_dat00= 5 ; cut_dat01=None; cut_type0='_cl_size'
        #cut0 = 1; cut_dat00= 10; cut_dat01=cut_dat00; cut_type0='_cl_size'

        cut1 = 0; cut_dat10=78; cut_dat11=None; cut_type1='_cl_mean'
        #cut1 = 1; cut_dat10=55; cut_dat11=55.00001; cut_type1='_cl_mean'
        cut2 = 0; cut_dat20=79; cut_dat21=None; cut_type2='cl_max_adc'
        cut3 = 0; cut_dat30=199; cut_dat31=None; cut_type3='_cl_totalE'

        data_cut_obs0, cut_str0 = cut_data(cut0, cut_dat00, cut_dat01, cut_type0, dat_obs1, '')
        data_cut_obs1, cut_str1 = cut_data(cut1, cut_dat10, cut_dat11, cut_type1, data_cut_obs0, cut_str0)
        data_cut_obs2, cut_str2 = cut_data(cut2, cut_dat20, cut_dat21, cut_type2, data_cut_obs1, cut_str1)
        data_cut_obs,  cut_str  = cut_data(cut3, cut_dat30, cut_dat31, cut_type3, data_cut_obs2, cut_str2)

        '''

        if(cut_dat0== 0 and cut_dat1 < r0): cut_str=cut_str+'_less_'+str(cut_dat1)+cut_type
        if(cut_dat0 > 0 and cut_dat1== r0): cut_str=cut_str+'_greater_'+str(cut_dat0)+cut_type
        if(cut_dat0 > 0 and cut_dat1 < r0): cut_str=cut_str+'_between_'+str(cut_dat0)+'_'+str(cut_dat1)+cut_type
        if(cut_dat0==cut_dat1): cut_str=cut_str+'_equal_'+str(cut_dat1)+cut_type
        if(cut_dat0== 0 and cut_dat1== r0): cut_str
        #if(cut_dat0 < 0 and cut_dat1 < 0 ): cut_str=cut_str+'_adc_less50_and_cls_less_4'+cut_type



        if(colm<11):data_cut1 = np.delete(data1, np.where( (data1[:,colm] < cut_dat0) | (data1[:,colm] > cut_dat1) ), axis=0)
        #if(colm==8):data_cut1= np.delete(data1, np.where( (data1[:,7] < 4 ) & (data1[:,colm] > 0 )  ), axis=0)
        if(cut_dat0==cut_dat1):
            cut_str=cut_str+'_equal_'+str(cut_dat1)+cut_type
            data_cut1= np.delete(data1, np.where( (data1[:,colm] != cut_dat1 ) ), axis=0)

        #data_cut2 = np.delete(data2, np.where( (data2[:,colm] < cut_dat0) | (data2[:,colm] > cut_dat1) ), axis=0)
        #data_bg_cut = np.delete(data_fondo, np.where( (data_fondo[:,colm] < cut_dat0) | (data_fondo[:,colm] > cut_dat1) ), axis=0)

    if(cut==False):
        data_cut1 = data1
        #data_cut2 = data2
        cut_str=cut_str+''
        #data_bg_cut = data_fondo

    #return data_cut1, data_cut2, data_bg_cut,  cut_str
    return data_cut1,  cut_str


def comp_obs_sim(obs, sim):
    dif = np.abs(obs - sim)
    err = np.sqrt(obs + 0.04*sim*sim )
    comp = np.divide(dif, err , where=(err != 0))
    err_comp = comp*np.sqrt( (err*err) / (dif*dif)
            + (obs + 0.000256*sim*sim*sim*sim) / (4*err*err*err*err) )
    return comp, np.abs(err_comp)


def comp_2obs(obs1, obs2):
    dif = obs1 - obs2
    err = np.sqrt(obs1 + obs2)
    comp = np.divide(dif, err , where=(err != 0))
    err_comp = comp*np.sqrt( (err*err) / (dif*dif)
            + (1) / (4*err*err) )
    return comp, np.abs(err_comp)


def comp_2sim(sim1, sim2):
    dif = sim1 - sim2
    err = 0.2*np.sqrt(sim1*sim1 + sim2*sim2 )
    comp = np.divide(dif, err , where=(err != 0))
    err_comp = 0.2*comp*np.sqrt( (err*err) / (dif*dif)
            + (sim1*sim1*sim1*sim1 + sim2*sim2*sim2*sim2) / (4*err*err*err*err) )
    return comp, np.abs(err_comp)

def ratio_obs_sim(dat1, dat2):
    # Set ratio where dat2 is not zero
    ratio = np.divide(dat1, dat2 , where=(dat2 != 0))
    # Compute error on ratio (null if cannot be computed)
    er_ratio = ratio*np.sqrt( (1)/(dat1) + 0.04 )

    return ratio, er_ratio

def ratio_2obs(obs1, obs2):
    # Set ratio where obs2 is not zero
    ratio = np.divide(obs1, obs2 , where=(obs2 != 0.00000000e+00))
    # Compute error on ratio (null if cannot be computed)
    er_ratio = ratio*np.sqrt( (1)/(obs1) + (1)/(obs2) )
    er_ratio[np.isinf(er_ratio)] = 0

    return ratio, er_ratio

def ratio_2sim(sim1, sim2):
    # Set ratio where obs2 is not zero
    ratio = np.divide(sim1, sim2 , where=(sim2 != 0.00000000e+00))
    # Compute error on ratio (null if cannot be computed)
    er_ratio = ratio*np.sqrt( 0.08)
    er_ratio[np.isinf(er_ratio)] = 0

    return ratio, er_ratio

#############################################################################################################################################################

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

strE= 'eh_'+str(pair_eh)+'eV'
print(strE)

ag ='8'
satu =''

condit_edep = ''

setpixel_z ='2'

efic = 1

siz_matz = 3


###############################################################

plot_nsig = 1
zoom_delta=0

plot_ratio = 0
zoom_ratio = 0

plt_obs = 1
plt_sim = 1

plt_bg = 1

plt_err=1

labl_opt = 'full' #'full' , 'simple', 'off'

log_y = 1

label_xy = True

grid_ = False

file_csv = 1
file_clst_sum_csv = True
file_total_adc_csv = True

if(plt_obs==False and plt_sim ==True): str_hist_comp = '_sim'
if(plt_obs==True and plt_sim ==False): str_hist_comp = '_obs'
if(plt_obs==True and plt_sim ==True): str_hist_comp = '_both'

###############################################################

z_count = 11

level_z = list(range(z_count))
#level_z = list(range(0,6,2))
#level_z = [0,2,6,12]
#level_z = [0]
#level_z = list(range(z_count))

#level_z = [1]

tiemp = '_500ms'

source1 = 'Sr90'
source2 = 'Cs137'

yscal_adc = 2e-2
yscal_cls = 10e-1
yscal_max = 10e-2

yscal_cls = 5e-1
yscal_max = 15e-3

yscal_adc = 0
yscal_cls = 0
yscal_max = 0


size_thresh = 0
max_thresh = 0

nsigm_self = 0

sim_bg = '_ag8_bg'
sim_bg = ''

simu_no_cut ='_sim_nocut'
simu_no_cut =''
'''
source_bg = 'backgnd'
fecha_bg = '2023_Oct_22_00h'
nfrm =1000

if(source1 == 'Sr90'):
    winback1='254'
    fecha1 = '2023_Oct_24_23h'
    ev1 = '1514'
    evt1 = winback1+'_ev'+ev1
    nfrm =1000

if(source2 == 'Cs137'):
    winback2='635'
    fecha2 = '2023_Oct_23_23h'
    ev2 = '3811'
    evt2 = winback2+'_ev'+ev2
    nfrm =1000
'''
source_bg = 'backgnd'
fecha_bg = '2022_Nov_10'
nfrm =100

if(source1 == 'Sr90'):
    winback1='254'
    fecha1 = '2021_Nov_09'
    ev1 = '1587'
    evt1 = winback1+'_ev'+ev1
    nfrm =100

if(source2 == 'Cs137'):
    winback2='635'
    fecha2 = '2021_Nov_23'
    ev2 = '3991'
    evt2 = winback2+'_ev'+ev2
    nfrm =100

########################################################################################

ssd_obs = '/home/mbonnett/mik/data_2023/'
ssd_sim = '/home/mbonnett/mik/dat_sim_2024_11/'

pname_obs = ssd_obs + 'data_obs_2023/'

drive_file_obs = ssd_obs+'data_obs_2023/'
drive_file_sim1 = ssd_sim+'g4_'+source1+'_2024/'
drive_file_sim2 = ssd_sim+'g4_'+source2+'_2024/'

maxim = np.zeros((2,2),dtype=int)

l_width = 4

font_siz=24

########################################################################################
plots_2d=1
#log_y = 1

bin_hist = 0
if(bin_hist>0): strbin = str(bin_hist)+'bin_z_'
else:strbin = 'z_'

adc_cut = 0
max_adc=1024
max_adc=150
min_adc=adc_cut
min_adc=1
#if(adc_cut == 0):min_adc=1
if(max_adc<1024):cut_adcmax_str = "_mx"+str(max_adc)
else: cut_adcmax_str = ''

cut0 = 0; cut_dat00= 6; cut_dat01=None; cut_type0='_clsiz'
#cut0 = 1; cut_dat00= 10; cut_dat01=cut_dat00; cut_type0='_cl_size'

cut1 = 0; cut_dat10=50; cut_dat11=650; cut_type1='_cl_mean'
#cut1 = 1; cut_dat10=55; cut_dat11=55.00001; cut_type1='_cl_mean'
cut2 = 0; cut_dat20=101; cut_dat21=None; cut_type2='_clmax_adc'
cut3 = 0; cut_dat30=90; cut_dat31=None; cut_type3='_cl_totalE'


if(min_adc==0): cut_adc_str = ''
if(min_adc>0):cut_adc_str = '_adcG'+str(min_adc)+cut_adcmax_str

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

#max_adc=1024
#min_adc=150
#min_adc=82
min_adc_clust=False

### Zoom histo1D culster
mx_cs = -1 # maxim[0,0]=-1
#mx_cs = 35
mxs = 1
if(mxs==1): maxs=''
else: maxs='_zoom'+str(mxs)
##############################
#max_adc = max_adc+1
#min_adc = min_adc+1

#if(min_adc-1==0):cut_adc0=''
#else: cut_adc0 = '_'+str(min_adc-1)+'adc_cut'

#if(min_adc-1==0 or min_adc_clust==False):cut_adc = ''
#else: cut_adc =  '_'+str(min_adc-1)+'adc_cut'

########################################################################################


porc_max = 0 # 5% del max
porc_max_sim = 0 # 5% del max

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

if(nsigm_self>0 and nsigm_bg == 0 and navrg < 0):
    sigthresh = '_tshld_'+str(nsigm_self)+'sig_self'
if(nsigm_self == 0 ):
    sigthresh = ''

if(size_thresh>0):
    cut_clst_size = '_clst'+ str(size_thresh)
    print(cut_clst_size)
if(size_thresh == 0 ):
    cut_clst_size = ''

if(max_thresh>0):
    cut_max = '_max'+ str(max_thresh)
if(max_thresh == 0 ):
    cut_max = ''

if(porc_max>0):
    cut_max_clst_porc = '_cutmf'+ str(porc_max)
    print(cut_max_clst_porc)
if(porc_max == 0 ):
    cut_max_clst_porc = ''

if(porc_max_sim>0):
    cut_max_clst_porc_sim = '_cutmf'+ str(porc_max_sim)
    print(cut_max_clst_porc_sim)
if(porc_max_sim == 0 ):
    cut_max_clst_porc_sim = ''



for densi in [ False,
              True,
          ]:

    if(densi==True): dens='_norm'
    if(densi==False): dens=''

    gauss_dist = 'Dg2'

    sim_n = ''
    for sim_n in range(0,10):
        sim_n ='_s'+str(sim_n)
        #sim_n ='_0'

        i_cont = 0
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
                         zff = np.round(zff,3)

                         if nsigm > 0 :
                             difu_gauss = f"_{gauss_dist}_{k}k_{alpha}a_{nsigm}sig_{zff}zff"
                             d_gauss = gauss_dist+'_z_'+r'$\sigma$ = '+str(nsigm)+r', $\k$ = '+str(k)+r', $\alpha$ = '+str(alpha)+ r', $z_{ff}$ = '+str(zff)
                         if nsigm == 0 :
                             difu_gauss = f"_{gauss_dist}_{k}k_{alpha}a_{zff}zff"
                             d_gauss =gauss_dist+'_z_'+r'$\k$ = '+str(k)+r', $\alpha$ = '+str(alpha)+ r', $z_{ff}$ = '+str(zff)

                         filt_cut = avrg_cut + sig_bg_thresh + sig_bg_thresh_sim + cut_clst_size+cut_max + cut_adc+ cut_max_clst_porc

                         path_cluster_obs1 = ssd_obs + 'dat_'+fecha1+'_clstr_filt'+cut_clst_size+cut_max +'/'#+cut_max_clst_porc+'/'
                         path_cluster_obs2 = ssd_obs + 'dat_'+fecha2+'_clstr_filt'+cut_clst_size+cut_max +'/'#+cut_max_clst_porc+'/'

                         path_cluster_sim1 = drive_file_sim1 + source1 +'/dat'+sim_n+'_'+fecha1+difu_gauss+'_clstr_filt'+cut_clst_size+cut_max +condit_edep+'/'#+cut_max_clst_porc_sim+'/'
                         path_cluster_sim2 = drive_file_sim2 + source2 +'/dat'+sim_n+'_'+fecha2+difu_gauss+'_clstr_filt'+cut_clst_size+cut_max +condit_edep+'/'#+cut_max_clst_porc_sim+'/'

                         sigma_sim = nsigm_bg_sim
                         sigma_obs = nsigm_bg

                         pname_bg = pname_obs + 'data_'+fecha_bg+'_'+ag+'ag'+'_'+source_bg+'_'+str(1)+'level'+'/'
                         name_filter_bg = avrg_cut + sig_bg_thresh +  cut_clst_size+cut_max + cut_adc + cut_max_clst_porc
                         dirname_bg ='adc_count_'+ source_bg+'_f'+str(nframes)+'_iso0_br50_ag'+ag+'_ad1' + name_filter_bg

                         backg = np.load(pname_bg+source_bg +'/'+dirname_bg+'/' + source_bg + name_filter_bg + '_z'+str(0).zfill(3)+'.npz')
                         backg = np.int64(backg.f.arr_0)
                         backg_mean = DescrStatsW(backg[0,0:backg[0].max()+2], weights=backg[1,0:backg[0].max()+2], ddof=0).mean
                         backg_std  = DescrStatsW(backg[0,0:backg[0].max()+2], weights=backg[1,0:backg[0].max()+2], ddof=0).std
                         print('backg_mean = ', backg_mean, 'backg_std = ', backg_std)
                         #############################################################################################################################################################
                         #############################################################################################################################################################

                         file_cluster_size_mean_bg = path_cluster_obs1 + 'dat_clstr_frm_siz_mean' + \
                                        cut_adc+avrg_cut + sig_bg_thresh +'_sim' +sig_bg_thresh_sim + \
                                        '_f' +str(nframes) + '_z' +str(2*0)+ cut_clst_size+cut_max + cut_max_clst_porc#+'/'

                         data_cluste_frame_bg = file_cluster_size_mean_bg+'/'+'dat_'+source_bg+'_f'+str(nframes)+'_clstr_frm_z'+str(0).zfill(3)+\
                                        '_'+fecha_bg+cut_adc+avrg_cut + sig_bg_thresh + cut_clst_size+cut_max +'.csv'

                         data_fond = np.loadtxt(data_cluste_frame_bg, delimiter='\t', skiprows=1)
                         #data_fond = np.genfromtxt(data_cluste_frame_bg, delimiter='\t', filling_values=None, skip_header=1)

                         if(data_fond.size > 0):
                             c_dat_real=1; c_dat_sim=1; c_bg_real=1;
                             data_fondo = data_fond

                         if(data_fond.size == 0):
                             c_dat_real=1; c_dat_sim=1; c_bg_real=0;
                             data_fondo = np.zeros((1, 11))

                         #############################################################################################################################################################
                         numb_clst_level = np.zeros((26, z_count))
                         total_adc_level = np.zeros((26, z_count))

                         #############################################################################################################################################################
                         ##########################################################################################

                         for z in level_z:

                            dirsavepatn_plt = '/home/mbonnett/mik/dat_sim_2025_11/' +gauss_dist+'_comp'+dens+'_dghist_'+strbin +str(z*2).zfill(2)+'mm'+str_cuts+'/'
                            dirsave_plt = dirsavepatn_plt + 'plots'+sim_n+'_comp_'+str(z_count)+'l'+'_'+str(nframes)+'f'+simu_no_cut+condit_edep+difu_gauss+'/plot_hist_mult_comp_'+str(z_count)+'l_'+sim_bg  #+ cuts_dat +simu_no_cut

                            try:
                                os.makedirs(dirsave_plt)
                            except FileExistsError:
                                pass

                            if( file_csv==True and i_cont == 0):

                                chi2_reduc_distr_test = dirsavepatn_plt+'All_comp_' + sim_n +dens+'_'+strbin+str(z_count)+'l_'+str(z*2).zfill(2)+'mm'+'sim_clst'+ str_cuts+'_'+str(max_adc)+'ADC'+'_chi2.csv'
                            i_cont+=1

                            file_cluster_size_mean_obs1 = path_cluster_obs1 + 'dat_clstr_frm_siz_mean' + \
                                        cut_adc + avrg_cut + sig_bg_thresh +'_sim' +sig_bg_thresh_sim + \
                                        '_f' +str(nframes) + '_z' +str(2*z) + cut_clst_size+cut_max + cut_max_clst_porc+'/'

                            file_cluster_size_mean_obs2 = path_cluster_obs2 + 'dat_clstr_frm_siz_mean' + \
                                        cut_adc + avrg_cut + sig_bg_thresh +'_sim' +sig_bg_thresh_sim + \
                                        '_f' +str(nframes) + '_z' +str(2*z) + cut_clst_size+cut_max + cut_max_clst_porc+'/'


                            file_cluster_size_mean_sim1 = path_cluster_sim1 + 'dat_clstr_frm_siz_mean' +difu_gauss + \
                                        '_sim' + sig_bg_thresh_sim + sigthresh + \
                                        '_f' +str(nframes) + '_z' +str(2*z)  + cut_clst_size+cut_max + cut_max_clst_porc_sim+'/'

                            file_cluster_size_mean_sim2 = path_cluster_sim2 + 'dat_clstr_frm_siz_mean' +difu_gauss + \
                                        '_sim' + sig_bg_thresh_sim + sigthresh + \
                                        '_f' +str(nframes) + '_z' +str(2*z)  + cut_clst_size+cut_max + cut_max_clst_porc_sim+'/'

                            path_dir_obs1 = pname_obs + 'data_'+fecha1+'_'+ag+'ag'+'_'+source1+'_'+str(z_count)+'level'+'/'
                            path_dir_obs2 = pname_obs + 'data_'+fecha2+'_'+ag+'ag'+'_'+source2+'_'+str(z_count)+'level'+'/'
                            name_filter_obs = avrg_cut + sig_bg_thresh +cut_clst_size+cut_max + cut_adc + cut_max_clst_porc
                            dirname1 = 'adc_count_'+ source1+'_f'+str(nframes)+'_iso0_br50_ag'+ag+'_ad1' + name_filter_obs
                            dirname2 = 'adc_count_'+ source2+'_f'+str(nframes)+'_iso0_br50_ag'+ag+'_ad1' + name_filter_obs

                            data_obs1 = np.load(path_dir_obs1 + source1 +'/'+ dirname1 +'/'+ source1 + name_filter_obs + '_z'+str(z).zfill(3)+'.npz')
                            data_obs1 = np.int64(data_obs1.f.arr_0)

                            data_obs2 = np.load(path_dir_obs2 + source2 +'/'+ dirname2 +'/'+ source2 + name_filter_obs + '_z'+str(z).zfill(3)+'.npz')
                            data_obs2= np.int64(data_obs2.f.arr_0)

                            background = backg

                            ################################################################################################################
                            pname_sim1 = drive_file_sim1+source1
                            pname_sim2 = drive_file_sim2+source2
                            name_filter_sim = sig_bg_thresh_sim +condit_edep+difu_gauss+ cut_clst_size+cut_max + cut_adc + cut_max_clst_porc_sim
                            #####################################################################################################################################################
                            name_sim1 = 'sim_'+source1+'_'+str(nframes)+'f_'+evt1+sim_n +'_'+satu+strE + name_filter_sim + sim_bg
                            dir_sim1 = pname_sim1+'/'+'adc_count_'+ name_sim1
                            ##################################################################################################
                            name_sim2 = 'sim_'+source2+'_'+str(nframes)+'f_'+evt2+sim_n +'_'+satu+strE + name_filter_sim + sim_bg
                            dir_sim2 = pname_sim2+'/'+'adc_count_'+ name_sim2
                            #####################################################################################################################################################
                            data_cnt_sim1 = np.load(dir_sim1+'/'+source1+'_ev'+ev1+ name_filter_sim +'_z'+str(2*z).zfill(3)+'.npz' )
                            data_cnt_sim2 = np.load(dir_sim2+'/'+source2+'_ev'+ev2+ name_filter_sim +'_z'+str(2*z).zfill(3)+'.npz' )
                            ################################################################################################################################################
                            data_cnt_sim1 = np.int64(data_cnt_sim1.f.arr_0)
                            data_cnt_sim2 = np.int64(data_cnt_sim2.f.arr_0)
                            ################################################################################################################################################
                            if(densi==True): histo='norm_'
                            if(densi!=True): histo=''
                            ################################################################################################################################################
                            if( file_total_adc_csv==True and densi==False and z==0):
                                file_total_adc1 = dirsave_plt+'/'+source1+'_'+source2+'_'+str(z_count)+'l_'+evt1+'_'+evt2 +'_s2_total_ADC'+'_'+str(min_adc)+'_'+str(max_adc)+'ADC'+'_dist'
                                print('\n')
                                fl_total_adc1 = open(file_total_adc1+'.csv','w')
                                fl_total_adc1.write("dist(mm)\t ADC_obs1_total\t ADC_obs1_mean\t ADC_obs1_std_1\t s2_obs\t ADC_obs1_err_2\t" +
                                                "ADC_obs2_total\t ADC_obs2_mean\t ADC_obs2_std_1\t s2_sim\t ADC_obs2_err_2\t" +
                                                "ADC_sim1_total\t ADC_sim1_mean\t ADC_sim1_std_1\t s2_sim1\t ADC_sim1_err_2\t" +
                                                "ADC_sim2_total\t ADC_sim2_mean\t ADC_sim2_std_1\t s2_sim_bg\t ADC_sim2_err_2\n")
                                fl_total_adc1.close()

                            if( file_total_adc_csv==True and densi==False ):
                                adc_dist_obs1 = data_obs1[:1]*data_obs1[1:]
                                total_adc_level[0][z]=z*2
                                total_adc_level[1][z]=np.sum(adc_dist_obs1)
                                total_adc_level[2][z]=np.mean(adc_dist_obs1)
                                total_adc_level[3][z]=np.std(adc_dist_obs1, ddof=1)
                                total_adc_level[4][z]=np.sum(adc_dist_obs1*adc_dist_obs1)  #np.size(adc_dist_obs1)
                                total_adc_level[5][z]=np.std(adc_dist_obs1, ddof=1)/np.sqrt(np.size(adc_dist_obs1))

                                adc_dist_obs2 = data_obs2[:1]*data_obs2[1:]
                                total_adc_level[6][z]=np.sum(adc_dist_obs2)
                                total_adc_level[7][z]=np.mean(adc_dist_obs2)
                                total_adc_level[8][z]=np.std(adc_dist_obs2, ddof=1)
                                total_adc_level[9][z]=np.sum(adc_dist_obs2*adc_dist_obs2)
                                total_adc_level[10][z]=np.std(adc_dist_obs2, ddof=1)/np.sqrt(np.size(adc_dist_obs2))

                                adc_dist_sim1 = data_cnt_sim1[:1]*data_cnt_sim1[1:]
                                total_adc_level[11][z]=np.sum(adc_dist_sim1)
                                total_adc_level[12][z]=np.mean(adc_dist_sim1)
                                total_adc_level[13][z]=np.std(adc_dist_sim1, ddof=1)
                                total_adc_level[14][z]=np.sum(adc_dist_sim1*adc_dist_sim1)
                                total_adc_level[15][z]=np.std(adc_dist_sim1, ddof=1)/np.sqrt(np.size(adc_dist_sim1))

                                adc_dist_sim2 = data_cnt_sim2[:1]*data_cnt_sim2[1:]
                                total_adc_level[16][z]=np.sum(adc_dist_sim2)
                                total_adc_level[17][z]=np.mean(adc_dist_sim2)
                                total_adc_level[18][z]=np.std(adc_dist_sim2, ddof=1)
                                total_adc_level[19][z]=np.sum(adc_dist_sim2*adc_dist_sim2)
                                total_adc_level[20][z]=np.std(adc_dist_sim2, ddof=1)/np.sqrt(np.size(adc_dist_sim2))


                                fl_total_adc1 = open(file_total_adc1+'.csv','a')
                                fl_total_adc1.write(str(z*2)+'\t'+str(total_adc_level[1][z])+'\t'+str(total_adc_level[2][z])+'\t'+str(total_adc_level[3][z])+'\t'+str(total_adc_level[4][z])+'\t'+str(total_adc_level[5][z])+'\t'
                                                 +str(total_adc_level[6][z])+'\t'+str(total_adc_level[7][z])+'\t'+str(total_adc_level[8][z])+'\t'+str(total_adc_level[9][z])+'\t'+str(total_adc_level[10][z])+'\t'
                                                 +str(total_adc_level[11][z])+'\t'+str(total_adc_level[12][z])+'\t'+str(total_adc_level[13][z])+'\t'+str(total_adc_level[14][z])+'\t'+str(total_adc_level[15][z])+'\t'
                                                 +str(total_adc_level[16][z])+'\t'+str(total_adc_level[17][z])+'\t'+str(total_adc_level[18][z])+'\t'+str(total_adc_level[19][z])+'\t'+str(total_adc_level[20][z])+'\n'
                                                 )
                                fl_total_adc1.close()


                            x_1=data_obs1;
                            x_2=data_obs2;
                            y_sim1=data_cnt_sim1;
                            y_sim2=data_cnt_sim2;
                            bg0=background

                            nbins_adc = round(np.max(x_1[0]))
                            #nbins_adc = round(np.max(np.array((x_1[0].max(), x_2[0].max(), y_sim1[0].max(), y_sim2[0].max()  ))) )
                            hist_adc, bins_adc = np.histogram(x_1[0], bins=nbins_adc)
                            adclinbins = np.linspace( bins_adc[0], bins_adc[-1] ,len(bins_adc))

                            wxadc_all1, xadc_all1 = np.histogram(x_1[0], weights=x_1[1], bins = adclinbins, density = densi)
                            wxadc_all2, xadc_all2 = np.histogram(x_2[0], weights=x_2[1], bins = adclinbins, density = densi)

                            wyadc_all1, yadc_all1 = np.histogram(y_sim1[0], weights=y_sim1[1], bins = adclinbins, density = densi)
                            wyadc_all2, yadc_all2 = np.histogram(y_sim2[0], weights=y_sim2[1], bins = adclinbins, density = densi)

                            x1 = x_1[:,min_adc:max_adc]
                            ysim1 = y_sim1[:,min_adc:max_adc]
                            #######################################################
                            x2 = x_2[:,min_adc:max_adc]
                            ysim2 = y_sim2[:,min_adc:max_adc]
                            #######################################################
                            bg= bg0[:,min_adc:max_adc]

                            nbins_adc = round(np.min(np.array((x1[0].max(), x2[0].max(), ysim1[0].max(), ysim1[0].max()  )))-min_adc+1)

                            hist_adc, bins_adc = np.histogram(x1[0], bins=nbins_adc)
                            adclinbins = np.linspace( bins_adc[0], bins_adc[-1], len(bins_adc) )
                            print('all',adclinbins.size)

                            wxadc1, xadc1 = np.histogram(x1[0], weights=x1[1], bins = adclinbins, density = densi)
                            wyadc1, yadc1 = np.histogram(ysim1[0], weights=ysim1[1], bins = adclinbins, density = densi)

                            wxadc2, xadc2 = np.histogram(x2[0], weights=x2[1], bins = adclinbins, density = densi)
                            wyadc2, yadc2 = np.histogram(ysim2[0], weights=ysim2[1], bins = adclinbins, density = densi)


                            if(bin_hist>1 and bin_hist<=nbins_adc):
                                min_adc_despl = int(np.ceil(min_adc/(np.max(x1[0])/bin_hist)))
                                nbins_adc = bin_hist + min_adc_despl
                                #if(bin_hist > round(np.max(x[0]) ) ):
                                 #   min_adc_despl = int( np.ceil(min_adc/(np.max(x[0])/ round(np.max(x[0])) )) )
                                  #  nbins_adc = round(np.max(x1[0]) ) + min_adc_despl
                                #else: nbins_adc = bin_hist
                            else:
                                min_adc_despl = int( np.ceil(min_adc/(np.max(x1[0])/ round(np.max(x1[0])) )) )
                                nbins_adc = round(np.max(x1[0])) + min_adc_despl

                            hist_adc, bins_adc = np.histogram(x1[0], bins=nbins_adc)
                            bins_adc = bins_adc[ min_adc_despl : ]
                            #adclinbins = np.linspace( bins_adc[0], bins_adc[-1], len(bins_adc) )
                            adclinbins = np.linspace( bins_adc[0], 1024, len(bins_adc) )
                            print('bin',adclinbins.size)

                            wxbh1, xbh1 = np.histogram(x1[0], weights=x1[1], bins = adclinbins, density = densi)
                            wybh1, ybh1 = np.histogram(ysim1[0], weights=ysim1[1], bins = adclinbins, density = densi)
                            wxbh2, xbh2 = np.histogram(x2[0], weights=x2[1], bins = adclinbins, density = densi)
                            wybh2, ybh2 = np.histogram(ysim2[0], weights=ysim2[1], bins = adclinbins, density = densi)


                            titulo = 'Hist:_'+ histo +source1+'_'+source2+'_Z'+str(z*2).zfill(1)+'mm'+sim_bg +filt_cut \
                                   # +'\n ev_' +ev+''+ '_IPC_crosstalk_'+ 'eta_'+ str(lower)

                            #plt.figure(figsize=(14,8))

                            x_weight1, x_adc1 = np.histogram(x1[0], weights=x1[1], bins = adclinbins )
                            his_adc_obs1 = {'bins': x_adc1, 'counts': x_weight1}
                            x_weight2, x_adc2 = np.histogram(x2[0], weights=x2[1], bins = adclinbins )
                            his_adc_obs2 = {'bins': x_adc2, 'counts': x_weight2}
                            #################################################################################
                            y_weight_sim1, y_adc_sim1 = np.histogram(ysim1[0], weights=ysim1[1], bins = adclinbins )
                            his_adc_sim1 = {'bins': y_adc_sim1, 'counts': y_weight_sim1}
                            y_weight_sim2, y_adc_sim2 = np.histogram(ysim2[0], weights=ysim2[1], bins = adclinbins )
                            his_adc_sim2 = {'bins': y_adc_sim2, 'counts': y_weight_sim2}
                            #################################################################################
                            bg_weight, bg_adc = np.histogram(bg[0], weights=bg[1], bins = adclinbins )
                            his_adc_bg = {'bins': bg_adc, 'counts': bg_weight}
                            #################################################################################

                            bincenters_x1 = 0.5*(x_adc1[1:]+x_adc1[:-1])
                            norm_x1 = (np.sum(x_weight1) * np.diff(x_adc1))
                            bincenters_x2 = 0.5*(x_adc2[1:]+x_adc2[:-1])
                            norm_x2 = (np.sum(x_weight2) * np.diff(x_adc2))
                            #################################################################################
                            bincenters_ysim1 = 0.5*(y_adc_sim1[1:]+y_adc_sim1[:-1])
                            norm_ysim1 = (np.sum(y_weight_sim1) * np.diff(y_adc_sim1))
                            bincenters_ysim2 = 0.5*(y_adc_sim2[1:]+y_adc_sim2[:-1])
                            norm_ysim2 = (np.sum(y_weight_sim2) * np.diff(y_adc_sim2))
                            #################################################################################
                            bincenters_bg = 0.5*(bg_adc[1:]+bg_adc[:-1])
                            norm_bg = (np.sum(bg_weight) * np.diff(bg_adc))
                            #################################################################################
                            #################################################################################
                            # Guardar en un archivo de texto,  .npy
                            np.save(dirsave_plt+'/'+'hist_adc_obs_'+source1+\
                                        sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', his_adc_obs1)
                            np.save(dirsave_plt+'/'+'hist_adc_obs_'+source2+\
                                        sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', his_adc_obs2)
                            np.save(dirsave_plt+'/'+'hist_adc_sim_'+source1+\
                                        sim_n+difu_gauss+ sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', his_adc_sim1)
                            np.save(dirsave_plt+'/'+'hist_adc_sim_'+source2+\
                                        sim_n+difu_gauss+ sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', his_adc_sim2)
                            np.save(dirsave_plt+'/'+'hist_adc_bg_'+source_bg+\
                                        sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', his_adc_bg )
                            #################################################################################

                            if(densi==True):
                                x_w1 = x_weight1/ norm_x1
                                err_x1 = np.sqrt(x_weight1)/ norm_x1

                                x_w2 = x_weight2/ norm_x2
                                err_x2 = np.sqrt(x_weight2)/ norm_x2
                                #################################################################################
                                y_w1 = y_weight_sim1/ norm_ysim1
                                err_ysim1 = np.sqrt(0.04*y_weight_sim1*y_weight_sim1+y_weight_sim1)/ norm_ysim1
                                err_ysim1 = (0.2*y_weight_sim1)/ norm_ysim1

                                y_w2 = y_weight_sim2/ norm_ysim2
                                err_ysim2 = np.sqrt(0.04*y_weight_sim2*y_weight_sim2+y_weight_sim2)/ norm_ysim2
                                err_ysim2 = (0.2*y_weight_sim2)/ norm_ysim2
                                #################################################################################
                                bg_w = bg_weight/ norm_bg
                                err_bg = np.sqrt(bg_weight)/ norm_bg

                            if(densi==False):
                                x_w1 = x_weight1
                                err_x1 = np.sqrt(x_weight1)

                                x_w2 = x_weight2
                                err_x2 = np.sqrt(x_weight2)
                                #################################################################################
                                y_w1 = y_weight_sim1
                                err_ysim1 = np.sqrt(0.04*y_weight_sim1*y_weight_sim1+y_weight_sim1)
                                err_ysim1 = (0.2*y_weight_sim1)

                                y_w2 = y_weight_sim2
                                err_ysim2 = np.sqrt(0.04*y_weight_sim2*y_weight_sim2+y_weight_sim2)
                                err_ysim2 = (0.2*y_weight_sim2)
                                #################################################################################
                                bg_w = bg_weight
                                err_bg = np.sqrt(bg_weight)

                            ######################################################################################
                            err_obs_adc = np.sqrt(err_x1*err_x1 + err_x2*err_x2)
                            ch2_obs_adc = chi_2_sigm_test(x_w1, x_w2, err_obs_adc)
                            tmp_obs_str = 'Chi2 test: ' + r'%.2f'%ch2_obs_adc[0] \
                                        +'       '+'ndf: ' + r'%.2f'%ch2_obs_adc[3] \
                                        +'\n'+'Chi2/ndf : ' + r'%.2f'%ch2_obs_adc[4]

                            err_sim_adc = np.sqrt(err_ysim1*err_ysim1 + err_ysim2*err_ysim2)
                            ch2_sim_adc = chi_2_sigm_test(y_w1, y_w2, err_sim_adc)
                            tmp_sim_str = 'Chi2 test: ' + r'%.2f'%ch2_sim_adc[0] \
                                        + '       '+'ndf: ' + r'%.2f'%ch2_sim_adc[3] \
                                        + '\n'+'Chi2/ndf : ' + r'%.2f'%ch2_sim_adc[4]

                            if(labl_opt=='full'):
                                lbl_obs1 = source1 + ' data'+ sig_bg_thresh +cut_clst_size+cut_max + cut_adc
                                lbl_obs2 = source2 + ' data'+ sig_bg_thresh +cut_clst_size+cut_max + cut_adc \
                                            +'\n'+'Chi2/ndf : ' + r'%.2f'%ch2_obs_adc[4]
                                lbl_sim1 = source1 + ' sim '+d_gauss+ sig_bg_thresh_sim +cut_clst_size+cut_max + cut_adc
                                lbl_sim2 = source2 + ' sim '+d_gauss+ sig_bg_thresh_sim +cut_clst_size+cut_max + cut_adc \
                                            +'\n'+'Chi2/ndf : ' + r'%.2f'%ch2_sim_adc[4]
                                lbl_bg = 'Background' + ' data'+ sig_bg_thresh +cut_clst_size+cut_max + cut_adc
                            if(labl_opt=='simple'):
                                lbl_obs1 = source1 + ' data'
                                lbl_obs2 = source2 + ' data'
                                lbl_sim1 = source1 + ' simulation '
                                lbl_sim2 = source2 + ' simulation '
                                lbl_bg = 'Background' + ' data'
                            if(labl_opt=='off'):
                                lbl_obs1 = None
                                lbl_obs2 = None
                                lbl_sim1 = None
                                lbl_sim2 = None
                                lbl_bg = None


                            plt.rcParams.update({'font.size': font_siz})

                            if( plot_nsig==True and plot_ratio==True ):
                                fig, (ax0, ax1, ax2) = plt.subplots(3,1,figsize=(14.5,14),gridspec_kw={'height_ratios': [5, 1, 1]} )#, sharex=True )

                            if( plot_nsig==True and plot_ratio==False ):
                                fig, (ax0, ax1) = plt.subplots(2,1,figsize=(14.5,12),gridspec_kw={'height_ratios': [5, 1]} )#, sharex=True )

                            if( plot_nsig==False and plot_ratio==True ):
                                fig, (ax0, ax2) = plt.subplots(2,1,figsize=(14.5,12),gridspec_kw={'height_ratios': [5, 1]} )#, sharex=True )


                            if( plot_nsig==False and plot_ratio==False ):
                                fig, (ax0) = plt.subplots(figsize=(14.5,9.5) )#, sharex=True )


                            if(plt_obs==True):
                                ax0.hist(x1[0], weights=x1[1], bins = adclinbins, histtype='step',
                                            density=densi, log=log_y, color='C3', linewidth=l_width*1,
                                            label = lbl_obs1 + cut_max_clst_porc
                                            )#  +'_'+ fecha )
                                if(plt_err==True):
                                    ax0.errorbar(bincenters_x1, x_w1, yerr=err_x1, fmt='C3'+'.', ecolor='C3', linewidth=l_width )

                                ax0.hist(x2[0], weights=x2[1], bins = adclinbins, histtype='step',
                                            density=densi, log=log_y, color='C1', linewidth=l_width*1,
                                            label = lbl_obs2 + cut_max_clst_porc
                                            )#  +'_'+ fecha )
                                if(plt_err==True):
                                    ax0.errorbar(bincenters_x2, x_w2, yerr=err_x2, fmt='C1'+'.', ecolor='C1', linewidth=l_width )

                            ################################################################################################################################

                            if(plt_sim==True):
                                ax0.hist(ysim1[0], weights=ysim1[1], bins = adclinbins, histtype='step'
                                            , density=densi, log=log_y, color='C0', linewidth=l_width*1,
                                            label = lbl_sim1 + cut_max_clst_porc_sim)
                                if(plt_err==True):
                                    ax0.errorbar(bincenters_ysim1, y_w1, yerr=err_ysim1, fmt='C0'+'.',
                                                    ecolor='C0', linewidth=l_width*1 )

                                ax0.hist(ysim2[0], weights=ysim2[1], bins = adclinbins, histtype='step'
                                            , density=densi, log=log_y, color='C7', linewidth=l_width*1,
                                            label = lbl_sim2 + cut_max_clst_porc_sim)

                                if(plt_err==True):
                                    ax0.errorbar(bincenters_ysim2, y_w2, yerr=err_ysim2, fmt='C7'+'.',
                                                ecolor='C7', linewidth=l_width*1 )

                            ################################################################################################################################
                            ################################################################################################################################
                            if(plt_bg==True):
                                ax0.hist(bg[0], weights=bg[1], bins = adclinbins, histtype='step'
                                            , density=densi, log=log_y, color='k', linewidth=l_width*1,
                                            #label='Back_ground'  +'_'+ag+'ag'+'_' +str(nsigm_bg_sim)+'sig' +'_t'+tiemp )
                                            label = lbl_bg )# + '_' + fecha_bg )
                                if(plt_err==True):
                                    ax0.errorbar(bincenters_bg, bg_w, yerr=err_bg, fmt='k'+'.',
                                                    ecolor='k', linewidth=l_width*1 )#, label = 'distance: '+str(2*z)+'mm' )

                            U_eV = 1000
                            min_cam =((min_adc-1)*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
                            sat_cam =(1023*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
                            #ax0.axvline(x = min_adc-1, color = 'k', linewidth=l_width*1, label = 'min_cam: '+str(min_adc-1)+' ADC = ' + str(min_cam)+' keV')
                            if(max_adc>=1023):ax0.axvline(x = 1023, color = 'k', linestyle="--", linewidth=l_width*1)#, label = 'sat_cam: 1023 ADC = ' + str(sat_cam)+' keV')


                            if( file_csv==True):
                                file_activ_pixel = dirsave_plt+'/comp_'+source1+'_'+source2+dens+'_'+str(z_count)+'l_'+strbin+str(z*2).zfill(2)+'mm'+'_active_pixel'+str_hist_comp +'_'+str(min_adc)+'_'+str(max_adc)+'ADC'+'.csv'
                                file_activ = open(file_activ_pixel,'w')

                                txt_activ = 'data'+'\t' + '0_ADC\t' + 'Active_Pixel\t' + 'all_pixel\t' + '%_0_ADC\t' + '%_Active_Pixel'
                                file_activ.write(txt_activ + '\n')
                                txt_activ = 'OBS_'+source1+'_'+ name_filter_obs +':\t'+ str(data_obs1[1][0:min_adc].sum())+'\t' \
                                            + str(data_obs1[1][min_adc:1024].sum()) +'\t'+ str(data_obs1[1].sum())+'\t' \
                                            + str(100*data_obs1[1][0:min_adc].sum()/data_obs1[1].sum())+'\t' \
                                            + str(100*data_obs1[1][min_adc:1024].sum()/data_obs1[1].sum())
                                file_activ.write(txt_activ + '\n')
                                txt_activ = 'SIM_'+source1+'_'+sim_bg+ name_filter_sim +':\t'+ str(data_cnt_sim1[1][0:min_adc].sum())+'\t' \
                                            + str(data_cnt_sim1[1][min_adc:1024].sum()) +'\t'+ str(data_cnt_sim1[1].sum())+'\t' \
                                            + str(100*data_cnt_sim1[1][0:min_adc].sum()/data_cnt_sim1[1].sum())+'\t' \
                                            + str(100*data_cnt_sim1[1][min_adc:1024].sum()/data_cnt_sim1[1].sum())
                                file_activ.write(txt_activ + '\n')

                                txt_activ = 'OBS_'+source2+'_'+ name_filter_obs +':\t'+ str(data_obs2[1][0:min_adc].sum())+'\t' \
                                            + str(data_obs2[1][min_adc:1024].sum()) +'\t'+ str(data_obs2[1].sum())+'\t' \
                                            + str(100*data_obs2[1][0:min_adc].sum()/data_obs2[1].sum())+'\t' \
                                            + str(100*data_obs2[1][min_adc:1024].sum()/data_obs2[1].sum())
                                file_activ.write(txt_activ + '\n')
                                txt_activ = 'SIM_'+source2+'_'+sim_bg+ name_filter_sim +':\t'+ str(data_cnt_sim2[1][0:min_adc].sum())+'\t' \
                                            + str(data_cnt_sim2[1][min_adc:1024].sum()) +'\t'+ str(data_cnt_sim2[1].sum())+'\t' \
                                            + str(100*data_cnt_sim2[1][0:min_adc].sum()/data_cnt_sim2[1].sum())+'\t' \
                                            + str(100*data_cnt_sim2[1][min_adc:1024].sum()/data_cnt_sim2[1].sum())
                                file_activ.write(txt_activ + '\n')
                                file_activ.close()

                            print('\n')
                            print('OBS_'+source1+'_'+str(nsigm_bg)+'sig'+':\t', data_obs1[1][0:min_adc].sum(),
                                    '\t', data_obs1[1][min_adc:1024].sum(),'\t',data_obs1[1].sum(),
                                    '\t'+ str(100*data_obs1[1][0:min_adc].sum()/data_obs1[1].sum()),
                                    '\t'+ str(100*data_obs1[1][min_adc:1024].sum()/data_obs1[1].sum()))
                            print('SIM_'+source1+'_'+str(nsigm_bg_sim)+'sig'+':\t', data_cnt_sim1[1][0:min_adc].sum(),
                                    '\t', data_cnt_sim1[1][min_adc:1024].sum(),'\t', data_cnt_sim1[1].sum(),
                                    '\t'+ str(100*data_cnt_sim1[1][0:min_adc].sum()/data_cnt_sim1[1].sum()),
                                    '\t'+ str(100*data_cnt_sim1[1][min_adc:1024].sum()/data_cnt_sim1[1].sum()))
                            print('OBS_'+source2+'_'+str(nsigm_bg)+'sig'+':\t', data_obs2[1][0:min_adc].sum(),
                                    '\t', data_obs2[1][min_adc:1024].sum(),'\t',data_obs2[1].sum(),
                                    '\t'+ str(100*data_obs2[1][0:min_adc].sum()/data_obs2[1].sum()),
                                    '\t'+ str(100*data_obs2[1][min_adc:1024].sum()/data_obs2[1].sum()))
                            print('SIM_'+source2+'_'+str(nsigm_bg_sim)+'sig'+':\t', data_cnt_sim2[1][0:min_adc].sum(),
                                    '\t', data_cnt_sim2[1][min_adc:1024].sum(),'\t',data_cnt_sim2[1].sum(),
                                    '\t'+ str(100*data_cnt_sim2[1][0:min_adc].sum()/data_cnt_sim2[1].sum()),
                                    '\t'+ str(100*data_cnt_sim2[1][min_adc:1024].sum()/data_cnt_sim2[1].sum()))
                            print('\n')


                            if(plt_obs==True or plt_sim==True):
                                if(log_y==True):ax0.set_yscale('log')
                                ax0.set_ylim(0, yscal_adc)
                                ax0.grid(grid_)
                                #ax0.set_xticks(size=font_siz+2)
                                #minor_ticks_x= np.arange(min_adc, max_adc, 200 )
                                ax0.tick_params(labelbottom=False, direction='inout', width=3, length=14 )
                                ax0.minorticks_on()
                                ax0.tick_params(axis='both', direction='inout', which='minor',length=8 ,width=2)
                                #ax0.yticks(size=font_siz+2)
                                if( plot_nsig==False and plot_ratio==False ):
                                    ax0.set_xlabel('ADC', fontsize=font_siz+4)
                                    ax0.tick_params(labelbottom=True, direction='inout', width=3, length=14 )
                                ax0.set_ylabel('Pixels Number', fontsize=font_siz+4)
                                ax0.legend(fontsize=font_siz+0)
                                #fig.suptitle(titulo , fontsize=font_siz)

                            ylabl_comp = r'$\frac{ ('+source1+' - '+source2+')}{\sigma}$'
                            comp_obs, err_comp_obs = comp_2obs(x_w1, x_w2)
                            comp_sim, err_comp_sim = comp_2sim(y_w1, y_w2)
                            if np.isnan(comp_obs).all() and np.isnan(comp_sim).all():
                                # Ambos arreglos son NaN
                                delta_min = 0
                                delta_max = 0
                            elif np.isnan(comp_sim).all():
                                # Solo comp_sim es NaN
                                delta_min = np.nanmin(comp_obs)
                                delta_max = np.nanmax(comp_obs[np.isfinite(comp_obs)])
                            elif np.isnan(comp_obs).all():
                                # Solo comp_obs es NaN
                                delta_min = np.nanmin(comp_sim)
                                delta_max = np.nanmax(comp_sim[np.isfinite(comp_sim)])
                            else:
                                # Ambos tienen valores válidos
                                delta_min = np.nanmin([np.nanmin(comp_obs), np.nanmin(comp_sim)])
                                delta_max = np.nanmax([
                                    np.nanmax(comp_obs[np.isfinite(comp_obs)]),
                                    np.nanmax(comp_sim[np.isfinite(comp_sim)])
                                ])

                            if(plt_obs==True and plot_nsig==True):
                                #ax1.axhline(y = 0., xmin=min_adc, xmax=max_adc, color = 'k', linestyle = '--')
                                ax1.hist(x1[0], weights=0*x1[1], bins = adclinbins, histtype='step', ls='--', density=densi, color='k', linewidth=l_width*1 )
                                ax1.errorbar(bincenters_x1, comp_obs, yerr=plt_err*err_comp_obs, fmt='C1'+'>', lw=2, markersize=12 )
                                if(zoom_delta==True):ax1.set_ylim(delta_min,delta_max)

                            if(plt_sim==True and plot_nsig==True):
                                #ax1.axhline(y = 0., xmin=min_adc, xmax=max_adc, color = 'k', linestyle = '--')
                                ax1.hist(x1[0], weights=0*x1[1], bins = adclinbins, histtype='step', ls='--', density=densi, color='k', linewidth=l_width*1 )
                                ax1.errorbar(bincenters_ysim1, comp_sim, yerr=plt_err*err_comp_sim, fmt='C0'+'o', lw=2, markersize=10 )
                                if(zoom_delta==True):ax1.set_ylim(delta_min,delta_max)

                            if( (plt_obs==True or plt_sim==True) and plot_nsig==True ):
                                ax1.tick_params(labelbottom=False, direction='inout', width=3, length=14 )
                                ax1.minorticks_on()
                                ax1.set_ylabel(ylabl_comp, fontsize=font_siz+2)
                                ax1.tick_params(axis='x', direction='inout', which='minor',length=8 ,width=2)
                                if(plot_ratio==False):
                                    ax1.set_xlabel('ADC', fontsize=font_siz+4)
                                    ax1.tick_params(labelbottom=True, width=3,  direction='inout', length=14 )

                                #ax[2].yaxis.tick_right()
                                #ax[2].yaxis.set_label_position("right")
                                if(grid_==True):
                                    #ax1.grid(axis = 'x',  linestyle = '--',)
                                    ax1.grid(grid_)

                            ylabl_ratio = r'$ratio = \frac{'+source1+'}{'+source2+'}$'
                            ratio_obs, err_ratio_obs = ratio_obs_sim(x_w1, x_w2)
                            ratio_sim, err_ratio_sim = ratio_obs_sim(y_w1, y_w2)
                            if np.isnan(ratio_obs).all() and np.isnan(ratio_sim).all():
                                # Ambos arreglos son NaN
                                ratio_min = 0
                                ratio_max = 0
                            elif np.isnan(ratio_sim).all():
                                # Solo ratio_sim es NaN
                                ratio_min = np.nanmin(ratio_obs)
                                ratio_max = np.nanmax(ratio_obs[np.isfinite(ratio_obs)])
                            elif np.isnan(ratio_obs).all():
                                # Solo ratio_obs es NaN
                                ratio_min = np.nanmin(ratio_sim)
                                ratio_max = np.nanmax(ratio_sim[np.isfinite(ratio_sim)])
                            else:
                                # Ambos tienen valores válidos
                                ratio_min = np.nanmin([np.nanmin(ratio_obs), np.nanmin(ratio_sim)])
                                ratio_max = np.nanmax([
                                    np.nanmax(ratio_obs[np.isfinite(ratio_obs)]),
                                    np.nanmax(ratio_sim[np.isfinite(ratio_sim)])
                                ])

                            if(plt_obs==True and plot_ratio==True):
                                #ax2.axhline(y = 0., xmin=min_adc, xmax=max_adc, color = 'k', linestyle = '--')
                                ax2.hist(x1[0], weights=0*x1[1], bins = adclinbins, histtype='step', ls='--', density=densi, color='k', linewidth=l_width*1 )
                                ax2.errorbar(bincenters_x1, ratio_obs, yerr=plt_err*err_ratio_obs, fmt='C1'+'>', lw=2, markersize=12 )
                                if(zoom_ratio==True):ax2.set_ylim(-0.5+ratio_min,ratio_max)

                            if(plt_sim==True and plot_ratio==True):
                                #ax2.axhline(y = 0., xmin=min_adc, xmax=max_adc, color = 'k', linestyle = '--')
                                ax2.hist(x1[0], weights=0*x1[1], bins = adclinbins, histtype='step', ls='--', density=densi, color='k', linewidth=l_width*1 )
                                ax2.errorbar(bincenters_ysim1, ratio_sim, yerr=plt_err*err_ratio_sim, fmt='C0'+'o', lw=2, markersize=10 )
                                if(zoom_ratio==True):ax2.set_ylim(-0.5+ratio_min,ratio_max)

                            if( (plt_obs==True or plt_sim==True) and plot_ratio==True ):
                                #ax[2].xticks(size=font_siz+2)
                                #ax[2].yticks(size=font_siz+2)
                                #minor_ticks_x= np.arange(min_adc, max_adc,100 )
                                ax2.tick_params(labelbottom=True, width=3,  direction='inout', length=14 )
                                ax2.minorticks_on()
                                ax2.tick_params(axis='x', direction='inout', which='minor',length=8 ,width=2)
                                ax2.set_xlabel('ADC', fontsize=font_siz+4)
                                ax2.set_ylabel(ylabl_ratio, fontsize=font_siz+2)
                                #ax[2].yaxis.tick_right()
                                #ax[2].yaxis.set_label_position("right")
                                if(grid_==True):
                                    #ax2.grid(axis = 'x',  linestyle = '--',)
                                    ax2.grid(grid_)

                            ax3 = ax0.twiny()
                            #min_cam =((45.7)*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
                            #min_cam =((-25)*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
                            #max_cam =(1070.6*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
                            #ax3.set_xlim(min_cam,max_cam )

                            mm=(ysim1[0]*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
                            ax3.plot(mm, ysim1[1]*0., color='w', linewidth=0.001 )
                            ax3.set_xlabel('Energy (keV)', fontsize=font_siz+4)

                            ax0.legend(loc='lower right')
                            ax0.legend(loc='lower left')

                            fig.tight_layout()
                            name_cuts = '_'+str(nframes)+'f'+sim_bg + filt_cut + str_hist_comp
                            namesave = 'comp_plot_hist_'+histo+'ADC_'+strbin+str(z*2).zfill(2)+'mm'+ '_'+source1+'_'+source2+ name_cuts
                            plt.savefig(dirsave_plt+'/'+namesave +'_'+str(min_adc)+'_'+str(max_adc)+'ADC' +'.png', dpi=150)
                            plt.savefig(dirsave_plt+'/'+namesave +'_'+str(min_adc)+'_'+str(max_adc)+'ADC' +'.pdf', dpi=150)

                            #plt.show()
                            #plt.clf()
                            #plt.close()

                            #######################################################################################################
                            #######################################################################################################
                            data_cluste_frame_obs1 = file_cluster_size_mean_obs1+'dat_obs_'+source1+'_f'+str(nframes)+'_clstr_frm_z'+str(z).zfill(3)+\
                                              '_'+fecha1+cut_adc+avrg_cut + sig_bg_thresh +cut_clst_size+cut_max+'.csv'
                            dat_obs1 = np.loadtxt(data_cluste_frame_obs1, delimiter='\t', skiprows=1)

                            n_clst_level_obs1 = file_cluster_size_mean_obs1 + 'clstr_frm_siz_mean_obs_'+source1+'_f'+str(nframes)+'_z'+str(z).zfill(3)+\
                                                '_'+fecha1+cut_adc+avrg_cut + sig_bg_thresh + cut_clst_size+cut_max +'.csv'
                            n_clst_level_obs1 = np.loadtxt(n_clst_level_obs1, delimiter='\t', skiprows=1)

                            data_cluste_frame_obs2 = file_cluster_size_mean_obs2+'dat_obs_'+source2+'_f'+str(nframes)+'_clstr_frm_z'+str(z).zfill(3)+\
                                              '_'+fecha2+cut_adc+avrg_cut + sig_bg_thresh +cut_clst_size+cut_max+'.csv'
                            dat_obs2 = np.loadtxt(data_cluste_frame_obs2, delimiter='\t', skiprows=1)

                            n_clst_level_obs2 = file_cluster_size_mean_obs2 + 'clstr_frm_siz_mean_obs_'+source2+'_f'+str(nframes)+'_z'+str(z).zfill(3)+\
                                                '_'+fecha2+cut_adc+avrg_cut + sig_bg_thresh + cut_clst_size+cut_max +'.csv'
                            n_clst_level_obs2 = np.loadtxt(n_clst_level_obs2, delimiter='\t', skiprows=1)
                            #######################################################################################################
                            #######################################################################################################
                            data_cluste_frame_sim1 = file_cluster_size_mean_sim1+'dat_sim_'+source1+'_f'+str(nframes)+'_clstr_frm_z'+str(2*z).zfill(3)+\
                                                    '_w'+evt1 +difu_gauss + cut_adc + sig_bg_thresh_sim + sigthresh +cut_clst_size+cut_max +'.csv'

                            n_clst_level_sim1 = file_cluster_size_mean_sim1 +'clstr_frm_siz_mean_sim_'+source1+'_f'+str(nframes)+'_z'+str(2*z).zfill(3)+\
                                                    '_w'+evt1 +difu_gauss + cut_adc + sig_bg_thresh_sim + sigthresh + cut_clst_size+cut_max +'.csv'

                            data_cluste_frame_sim2 = file_cluster_size_mean_sim2+'dat_sim_'+source2+'_f'+str(nframes)+'_clstr_frm_z'+str(2*z).zfill(3)+\
                                                    '_w'+evt2 +difu_gauss + cut_adc + sig_bg_thresh_sim + sigthresh +cut_clst_size+cut_max +'.csv'

                            n_clst_level_sim2 = file_cluster_size_mean_sim2 +'clstr_frm_siz_mean_sim_'+source2+'_f'+str(nframes)+'_z'+str(2*z).zfill(3)+\
                                                    '_w'+evt2 +difu_gauss + cut_adc + sig_bg_thresh_sim + sigthresh + cut_clst_size+cut_max +'.csv'
                            #######################################################################################################
                            n_clst_level_sim1 = np.loadtxt(n_clst_level_sim1, delimiter='\t', skiprows=1)
                            n_clst_level_sim2 = np.loadtxt(n_clst_level_sim2, delimiter='\t', skiprows=1)
                            #######################################################################################################
                            #######################################################################################################

                            if os.path.isdir(file_cluster_size_mean_sim1 and file_cluster_size_mean_sim2):
                                print('Folder Found:', file_cluster_size_mean_sim1, file_cluster_size_mean_sim2)
                                #######################################################################################################
                                dat_sim1 = np.loadtxt(data_cluste_frame_sim1, delimiter='\t', skiprows=1)
                                #######################################################################################################
                                #######################################################################################################
                                dat_sim2 = np.loadtxt(data_cluste_frame_sim2, delimiter='\t', skiprows=1)
                                #######################################################################################################

                                data_cut_obs10, cut_str0 = cut_data(cut0, cut_dat00, cut_dat01, cut_type0, dat_obs1, "")
                                data_cut_obs11, cut_str1 = cut_data(cut1, cut_dat10, cut_dat11, cut_type1, data_cut_obs10, cut_str0)
                                data_cut_obs12, cut_str2 = cut_data(cut2, cut_dat20, cut_dat21, cut_type2, data_cut_obs11, cut_str1)
                                data_cut_obs1, cut_str  = cut_data(cut3, cut_dat30, cut_dat31, cut_type3, data_cut_obs12, cut_str2)
                                #######################################################################################################
                                data_cut_obs20, cut_str0 = cut_data(cut0, cut_dat00, cut_dat01, cut_type0, dat_obs2, "")
                                data_cut_obs21, cut_str1 = cut_data(cut1, cut_dat10, cut_dat11, cut_type1, data_cut_obs20, cut_str0)
                                data_cut_obs22, cut_str2 = cut_data(cut2, cut_dat20, cut_dat21, cut_type2, data_cut_obs21, cut_str1)
                                data_cut_obs2, cut_str  = cut_data(cut3, cut_dat30, cut_dat31, cut_type3, data_cut_obs22, cut_str2)


                                data_bg_cut0, cut_str0 = cut_data(cut0, cut_dat00, cut_dat01, cut_type0, data_fondo, '')
                                if(data_bg_cut0.size == 0): data_bg_cut1 = np.zeros((1, 11)); c_bg_real=0;
                                else:data_bg_cut1, cut_str1 = cut_data(cut1, cut_dat10, cut_dat11, cut_type1, data_bg_cut0, cut_str0)
                                if(data_bg_cut1.size == 0): data_bg_cut2 = np.zeros((1, 11)); c_bg_real=0;
                                else:data_bg_cut2, cut_str2 = cut_data(cut2, cut_dat20, cut_dat21, cut_type2, data_bg_cut1, cut_str1)
                                if(data_bg_cut2.size == 0): data_bg_cut = np.zeros((1, 11));  c_bg_real=0;
                                else:data_bg_cut,  cut_str  = cut_data(cut3, cut_dat30, cut_dat31, cut_type3, data_bg_cut2, cut_str2)

                                #######################################################################################################
                                data_cut_sim10, cut_str0 = cut_data(cut0, cut_dat00, cut_dat01, cut_type0, dat_sim1, '')
                                data_cut_sim11, cut_str1 = cut_data(cut1, cut_dat10, cut_dat11, cut_type1, data_cut_sim10, cut_str0)
                                data_cut_sim12, cut_str2 = cut_data(cut2, cut_dat20, cut_dat21, cut_type2, data_cut_sim11, cut_str1)
                                data_cut_sim1,  cut_str  = cut_data(cut3, cut_dat30, cut_dat31, cut_type3, data_cut_sim12, cut_str2)
                                #######################################################################################################
                                data_cut_sim20, cut_str0 = cut_data(cut0, cut_dat00, cut_dat01, cut_type0, dat_sim2, '')
                                data_cut_sim21, cut_str1 = cut_data(cut1, cut_dat10, cut_dat11, cut_type1, data_cut_sim20, cut_str0)
                                data_cut_sim22, cut_str2 = cut_data(cut2, cut_dat20, cut_dat21, cut_type2, data_cut_sim21, cut_str1)
                                data_cut_sim2,  cut_str  = cut_data(cut3, cut_dat30, cut_dat31, cut_type3, data_cut_sim22, cut_str2)

                                #######################################################################################################

                                if(data_bg_cut.size == 0): c_bg_real=0; data_bg_cut = np.zeros((1, 8))

                                #############################################################################################################################################################
                                #############################################################################################################################################################
                                if( file_clst_sum_csv==True and densi==False and z==0 ):
                                    file_clst_sum = dirsave_plt+'/'+source1+'_'+source2+'_'+str(z_count)+'l_'+evt1+'_'+evt2+'_s2_num_clst'+cut_type2+cut_clmaxadc+'_dist'
                                    print('\n')
                                    fl_clst_sum = open(file_clst_sum+'.csv','w')
                                    fl_clst_sum.write("dist(mm)\t num_clst_obs1_total\t num_clst_obs1_mean\t num_clst_obs1_std_1\t s2_obs\t num_clst_obs1_err_2\t" +
                                                      "num_clst_obs2_total\t num_clst_obs2_mean\t num_clst_obs2_std_1\t s2_sim\t num_clst_obs2_err_2\t" +
                                                      "num_clst_sim1_total\t num_clst_sim1_mean\t num_clst_sim1_std_1\t s2_sim1\t num_clst_sim1_err_2\t" +
                                                      "num_clst_sim2_total\t num_clst_sim2_mean\t num_clst_sim2_std_1\t s2_sim_bg\t num_clst_sim2_err_2\n")
                                    fl_clst_sum.close()

                                sum_nclst_obs1=0
                                sum2_nclst_obs1=0
                                obs1_nc_apnd = np.append(data_cut_obs1[:,1], 0)
                                for i in range (data_cut_obs1[:,6].size):
                                    if(data_cut_obs1[:,1][i]!= obs1_nc_apnd[i+1]):
                                        sum_nclst_obs1 = sum_nclst_obs1+data_cut_obs1[:,6][i]
                                        sum2_nclst_obs1 = sum2_nclst_obs1+data_cut_obs1[:,6][i]*data_cut_obs1[:,6][i]
                                        #print(i,data_cut_obs1[:,6][i],sum_nclst_obs1, sum2_nclst_obs1)
                                avg_nclst_obs1 = sum_nclst_obs1/ nfrm
                                std_nclst_obs1 = np.sqrt((sum2_nclst_obs1-sum_nclst_obs1*sum_nclst_obs1/nfrm)/(nfrm))
                                std_n1clst_obs1 = np.sqrt((sum2_nclst_obs1-sum_nclst_obs1*sum_nclst_obs1/nfrm)/(nfrm-1))
                                print('\n')
                                print(sum_nclst_obs1, sum2_nclst_obs1, avg_nclst_obs1, std_nclst_obs1, std_n1clst_obs1)
                                print(np.sum(n_clst_level_obs1[:,1]), np.mean(n_clst_level_obs1[:,1]), np.std(n_clst_level_obs1[:,1]), np.std(n_clst_level_obs1[:,1], ddof=1))

                                sum_nclst_obs2=0
                                sum2_nclst_obs2=0
                                obs2_nc_apnd = np.append(data_cut_obs2[:,1], 0)
                                for i in range (data_cut_obs2[:,6].size):
                                    if(data_cut_obs2[:,1][i]!= obs2_nc_apnd[i+1]):
                                        sum_nclst_obs2 = sum_nclst_obs2+data_cut_obs2[:,6][i]
                                        sum2_nclst_obs2 = sum2_nclst_obs2+data_cut_obs2[:,6][i]*data_cut_obs2[:,6][i]
                                        #print(i,data_cut_obs2[:,6][i],sum_nclst_obs2, sum2_nclst_obs2)
                                avg_nclst_obs2 = sum_nclst_obs2/ nfrm
                                std_nclst_obs2 = np.sqrt((sum2_nclst_obs2-sum_nclst_obs2*sum_nclst_obs2/nfrm)/(nfrm))
                                std_n1clst_obs2 = np.sqrt((sum2_nclst_obs2-sum_nclst_obs2*sum_nclst_obs2/nfrm)/(nfrm-1))
                                print('\n')
                                print(sum_nclst_obs2, sum2_nclst_obs2, avg_nclst_obs2, std_nclst_obs2, std_n1clst_obs2)
                                print(np.sum(n_clst_level_obs2[:,1]), np.mean(n_clst_level_obs2[:,1]), np.std(n_clst_level_obs2[:,1]), np.std(n_clst_level_obs2[:,1], ddof=1))

                                sum_nclst_sim1=0
                                sum2_nclst_sim1=0
                                sim1_nc_apnd = np.append(data_cut_sim1[:,1], 0)
                                for i in range (data_cut_sim1[:,6].size):
                                    if(data_cut_sim1[:,1][i]!= sim1_nc_apnd[i+1]):
                                        sum_nclst_sim1 = sum_nclst_sim1+data_cut_sim1[:,6][i]
                                        sum2_nclst_sim1 = sum2_nclst_sim1+data_cut_sim1[:,6][i]*data_cut_sim1[:,6][i]
                                        #print(i,data_cut_sim1[:,6][i],sum_nclst_sim1, sum2_nclst_sim1)
                                avg_nclst_sim1 = sum_nclst_sim1/ nfrm
                                std_nclst_sim1 = np.sqrt((sum2_nclst_sim1-sum_nclst_sim1*sum_nclst_sim1/nfrm)/(nfrm))
                                std_n1clst_sim1 = np.sqrt((sum2_nclst_sim1-sum_nclst_sim1*sum_nclst_sim1/nfrm)/(nfrm-1))
                                print('\n')
                                print(sum_nclst_sim1, sum2_nclst_sim1, avg_nclst_sim1, std_nclst_sim1, std_n1clst_sim1)
                                print(np.sum(n_clst_level_sim1[:,1]), np.mean(n_clst_level_sim1[:,1]), np.std(n_clst_level_sim1[:,1]), np.std(n_clst_level_sim1[:,1], ddof=1))

                                sum_nclst_sim2=0
                                sum2_nclst_sim2=0
                                sim2_nc_apnd = np.append(data_cut_sim2[:,1], 0)
                                for i in range (data_cut_sim2[:,6].size):
                                    if(data_cut_sim2[:,1][i]!= sim2_nc_apnd[i+1]):
                                        sum_nclst_sim2 = sum_nclst_sim2+data_cut_sim2[:,6][i]
                                        sum2_nclst_sim2 = sum2_nclst_sim2+data_cut_sim2[:,6][i]*data_cut_sim2[:,6][i]
                                        #print(i,data_cut_sim2[:,6][i],sum_nclst_sim2, sum2_nclst_sim2)
                                avg_nclst_sim2 = sum_nclst_sim2/ nfrm
                                std_nclst_sim2 = np.sqrt((sum2_nclst_sim2-sum_nclst_sim2*sum_nclst_sim2/nfrm)/(nfrm))
                                std_n1clst_sim2 = np.sqrt((sum2_nclst_sim2-sum_nclst_sim2*sum_nclst_sim2/nfrm)/(nfrm-1))
                                print('\n')
                                print(sum_nclst_sim2, sum2_nclst_sim2, avg_nclst_sim2, std_nclst_sim2, std_n1clst_sim2)
                                print(np.sum(n_clst_level_sim2[:,1]), np.mean(n_clst_level_sim2[:,1]), np.std(n_clst_level_sim2[:,1]), np.std(n_clst_level_sim2[:,1], ddof=1))
                                print('\n')


                                if(file_clst_sum_csv==True and densi==False):
                                    numb_clst_level[0][z] = z*2

                                    numb_clst_level[1][z] = sum_nclst_obs1
                                    numb_clst_level[2][z] = avg_nclst_obs1
                                    numb_clst_level[3][z] = std_n1clst_obs1
                                    numb_clst_level[4][z] = sum2_nclst_obs1
                                    numb_clst_level[5][z] = std_n1clst_obs1/np.sqrt(nfrm)

                                    numb_clst_level[6][z] = sum_nclst_obs2
                                    numb_clst_level[7][z] = avg_nclst_obs2
                                    numb_clst_level[8][z] = std_n1clst_obs2
                                    numb_clst_level[9][z] = sum2_nclst_obs2
                                    numb_clst_level[10][z] = std_n1clst_obs2/np.sqrt(nfrm)

                                    numb_clst_level[11][z] = sum_nclst_sim1
                                    numb_clst_level[12][z] = avg_nclst_sim1
                                    numb_clst_level[13][z] = std_n1clst_sim1
                                    numb_clst_level[14][z] = sum2_nclst_sim1
                                    numb_clst_level[15][z] = std_n1clst_sim1/np.sqrt(nfrm)

                                    numb_clst_level[16][z] = sum_nclst_sim2
                                    numb_clst_level[17][z] = avg_nclst_sim2
                                    numb_clst_level[18][z] = std_n1clst_sim2
                                    numb_clst_level[19][z] = sum2_nclst_sim2
                                    numb_clst_level[20][z] = std_n1clst_sim2/np.sqrt(nfrm)

                                    fl_clst_sum = open(file_clst_sum+'.csv','a')
                                    fl_clst_sum.write(str(z*2)+'\t'+str(numb_clst_level[1][z])+'\t'+str(numb_clst_level[2][z])+'\t'+str(numb_clst_level[3][z])+'\t'+str(numb_clst_level[4][z])+'\t'+str(numb_clst_level[5][z])+'\t'
                                                     +str(numb_clst_level[6][z])+'\t'+str(numb_clst_level[7][z])+'\t'+str(numb_clst_level[8][z])+'\t'+str(numb_clst_level[9][z])+'\t'+str(numb_clst_level[10][z])+'\t'
                                                     +str(numb_clst_level[11][z])+'\t'+str(numb_clst_level[12][z])+'\t'+str(numb_clst_level[13][z])+'\t'+str(numb_clst_level[14][z])+'\t'+str(numb_clst_level[15][z])+'\t'
                                                     +str(numb_clst_level[16][z])+'\t'+str(numb_clst_level[17][z])+'\t'+str(numb_clst_level[18][z])+'\t'+str(numb_clst_level[19][z])+'\t'+str(numb_clst_level[20][z])+'\n')
                                    fl_clst_sum.close()

                                ####################################################################################################################
                                ####################################################################################################################
                                xlab = list((('Cluster Size', 'Mean ADC'),('Maximum ADC', 'Total ADC')))

                                if(data_bg_cut.ndim!=1 and data_bg_cut.shape[1]<11):
                                    r= np.max(np.array((np.max(data_cut_obs1[:,7]), np.max(data_cut_obs2[:,7]), np.max(data_cut_sim1[:,7]), np.max(data_cut_sim2[:,7])  )))
                                    maxim[0,0]=r+1
                                    r= np.max(np.array((np.max(data_cut_obs1[:,8]), np.max(data_cut_obs2[:,8]), np.max(data_cut_sim1[:,8]), np.max(data_cut_sim2[:,8])  )))
                                    maxim[0,1]= 1.1*r
                                    r= np.max(np.array((np.max(data_cut_obs1[:,9]), np.max(data_cut_obs2[:,9]), np.max(data_cut_sim1[:,9]), np.max(data_cut_sim2[:,9])  )))
                                    maxim[1,1]= np.ceil(1.1*r)
                                    r= np.max(np.array((np.max(data_cut_obs1[:,10]), np.max(data_cut_obs2[:,10]), np.max(data_cut_sim1[:,10]), np.max(data_cut_sim2[:,10])  )))
                                    maxim[1,0]= 1.1*r

                                if(data_bg_cut.ndim==1):
                                    r= np.max(np.array((np.max(data_cut_obs1[:,7]), np.max(data_cut_obs2[:,7]), np.max(data_cut_sim1[:,7]), np.max(data_cut_sim2[:,7]), np.max(data_bg_cut[7])  )))
                                    maxim[0,0]=r+1
                                    r= np.max(np.array((np.max(data_cut_obs1[:,8]), np.max(data_cut_obs2[:,8]), np.max(data_cut_sim1[:,8]), np.max(data_cut_sim2[:,8]), np.max(data_bg_cut[8])  )))
                                    maxim[0,1]= 1.1*r
                                    r= np.max(np.array((np.max(data_cut_obs1[:,9]), np.max(data_cut_obs2[:,9]), np.max(data_cut_sim1[:,9]), np.max(data_cut_sim2[:,9]), np.max(data_bg_cut[9])  )))
                                    maxim[1,1]= np.ceil(1.1*r)
                                    r= np.max(np.array((np.max(data_cut_obs1[:,10]), np.max(data_cut_obs2[:,10]), np.max(data_cut_sim1[:,10]), np.max(data_cut_sim2[:,10]), np.max(data_bg_cut[10])  )))
                                    maxim[1,0]= 1.1*r

                                if(data_bg_cut.ndim>1 and data_bg_cut.shape[1]==11):
                                    #r= np.max(np.array((np.max(data_cut_obs1[:,7]), np.max(data_cut_obs2[:,7]), np.max(data_cut_sim1[:,7]), np.max(data_cut_sim2[:,7]), np.max(data_bg_cut[:,7])  )))
                                    # Lista para almacenar los máximos válidos
                                    max_values = []
                                    # Agregar el máximo de cada arreglo si no está vacío
                                    for array in [data_cut_obs1[:, 7], data_cut_obs2[:, 7], data_cut_sim1[:, 7], data_cut_sim2[:, 7], data_bg_cut[:, 7]]:
                                        if array.size > 0:  # Verifica si el arreglo no está vacío
                                            max_values.append(np.max(array))

                                    # Verifica si hay valores válidos en max_values
                                    if max_values:
                                        r = np.max(max_values)  # Calcula el máximo de los máximos válidos
                                    else:
                                        r = 0  # Manejo de caso en el que todos los arreglos estén vacíos
                                    maxim[0,0]=r+1
                                    #r= np.max(np.array((np.max(data_cut_obs1[:,8]), np.max(data_cut_obs2[:,8]), np.max(data_cut_sim1[:,8]), np.max(data_cut_sim2[:,8]), np.max(data_bg_cut[:,8])  )))
                                    max_values = []
                                    # Agregar el máximo de cada arreglo si no está vacío
                                    for array in [data_cut_obs1[:, 8], data_cut_obs2[:, 8], data_cut_sim1[:, 8], data_cut_sim2[:, 8], data_bg_cut[:, 8]]:
                                        if array.size > 0:  # Verifica si el arreglo no está vacío
                                            max_values.append(np.max(array))

                                    # Verifica si hay valores válidos en max_values
                                    if max_values:
                                        r = np.max(max_values)  # Calcula el máximo de los máximos válidos
                                    else:
                                        r = 0  # Manejo de caso en el que todos los arreglos estén vacíos
                                    maxim[0,1]= 1.1*r
                                    #r= np.max(np.array((np.max(data_cut_obs1[:,9]), np.max(data_cut_obs2[:,9]), np.max(data_cut_sim1[:,9]), np.max(data_cut_sim2[:,9]), np.max(data_bg_cut[:,9])  )))
                                    max_values = []
                                    # Agregar el máximo de cada arreglo si no está vacío
                                    for array in [data_cut_obs1[:, 9], data_cut_obs2[:, 9], data_cut_sim1[:, 9], data_cut_sim2[:, 9], data_bg_cut[:, 9]]:
                                        if array.size > 0:  # Verifica si el arreglo no está vacío
                                            max_values.append(np.max(array))

                                    # Verifica si hay valores válidos en max_values
                                    if max_values:
                                        r = np.max(max_values)  # Calcula el máximo de los máximos válidos
                                    else:
                                        r = 0  # Manejo de caso en el que todos los arreglos estén vacíos
                                    maxim[1,1]= np.ceil(1.1*r)
                                    #r= np.max(np.array((np.max(data_cut_obs1[:,10]), np.max(data_cut_obs2[:,10]), np.max(data_cut_sim1[:,10]), np.max(data_cut_sim2[:,10]), np.max(data_bg_cut[:,10])  )))
                                    max_values = []
                                    # Agregar el máximo de cada arreglo si no está vacío
                                    for array in [data_cut_obs1[:, 10], data_cut_obs2[:, 10], data_cut_sim1[:, 10], data_cut_sim2[:, 10], data_bg_cut[:, 10]]:
                                        if array.size > 0:  # Verifica si el arreglo no está vacío
                                            max_values.append(np.max(array))

                                    # Verifica si hay valores válidos en max_values
                                    if max_values:
                                        r = np.max(max_values)  # Calcula el máximo de los máximos válidos
                                    else:
                                        r = 0  # Manejo de caso en el que todos los arreglos estén vacíos
                                    maxim[1,0]= 1.1*r

                                if(mx_cs == -1): mx_cs = maxim[0,0]

                            #for densi in[ False, #True ]:
                                for plt_one in[ True,
                                               False
                                               ]:
                                    if( plot_nsig==True and plot_ratio==True ):
                                        fig, ([ax0, ax1], [ax2, ax3], [ax4, ax5], [ax6,ax7], [ax8,ax9], [ax10,ax11] ) = plt.subplots(6,2,figsize=(25,24),  gridspec_kw={'height_ratios': [5, 1, 1, 5, 1, 1]} )

                                    if( plot_nsig==True and plot_ratio==False ):
                                        fig, ([ax0, ax1], [ax2, ax3], [ax6,ax7], [ax8,ax9] ) = plt.subplots(4,2,figsize=(25,20),  gridspec_kw={'height_ratios': [5, 1, 5, 1]} )

                                    if( plot_nsig==False and plot_ratio==True ):
                                        fig, ([ax0, ax1], [ax4, ax5], [ax6,ax7], [ax10,ax11] ) = plt.subplots(4,2,figsize=(25,20),  gridspec_kw={'height_ratios': [5, 1, 5, 1]} )
                                    #fig, ax = plt.subplots(6,2,figsize=(25,24),  gridspec_kw={'height_ratios': [5, 1, 1, 5, 1, 1], 'hspace': 0.25} )

                                    if( plot_nsig==False and plot_ratio==False ):
                                        fig, ([ax0, ax1], [ax6,ax7]) = plt.subplots(2,2, figsize=(25,15) )#, sharex=True )


                                    for type_plt in ['cl_size', 'cl_mean', 'cl_etotal', 'cl_max_adc' ]:

                                        #hist_etotal, bins_datc1 = np.histogram(data_cut_obs1[:,7], bins=nbins_datc1)

                                        #### Cluster Sizemaxim[0,0]
                                        if(type_plt=='cl_size'):
                                            if(plt_one==True):
                                                if( plot_nsig==True and plot_ratio==True ):
                                                    fig, (ax0, ax2, ax4) = plt.subplots(3,1,figsize=(14.5,14),gridspec_kw={'height_ratios': [5, 1, 1] } )

                                                if( plot_nsig==True and plot_ratio==False ):
                                                    fig, (ax0, ax2)  = plt.subplots(2,1,figsize=(14.5,12),  gridspec_kw={'height_ratios': [5, 1]} )

                                                if( plot_nsig==False and plot_ratio==True ):
                                                    fig, (ax0, ax4) = plt.subplots(2,1,figsize=(14.5,12),  gridspec_kw={'height_ratios': [5, 1]} )

                                                if( plot_nsig==False and plot_ratio==False ):
                                                    fig, (ax0) = plt.subplots(figsize=(14.5,9.5) )#, sharex=True )


                                            cltsbin=range(maxim[0,0])
                                            #cltsbin=50

                                            xlabs = 'Cluster Size'
                                            #maxic = np.max(np.array((np.max(data_cut_obs1[:,7]), np.max(data_cut_obs2[:,7]),
                                            #        np.max(data_cut_sim1[:,7]), np.max(data_cut_sim2[:,7])  )))+1
                                            # Lista de arreglos que deseas evaluar
                                            arrays_c = [ data_cut_obs1[:, 7], data_cut_obs2[:, 7], data_cut_sim1[:, 7], data_cut_sim2[:, 7]]
                                            # Filtrar los arreglos no vacíos
                                            valid_max_values = [np.max(arr) for arr in arrays_c if arr.size > 0]
                                            # Verificar si hay valores válidos y calcular el máximo
                                            if valid_max_values:
                                                maxic = np.max(valid_max_values) + 1
                                            else:
                                                maxic = 0

                                            #minic = np.min(np.array((np.min(data_cut_obs1[:,7]), np.min(data_cut_obs2[:,7]),
                                            #        np.min(data_cut_sim1[:,7]), np.min(data_cut_sim2[:,7])  )))
                                            # Filtrar los arreglos no vacíos
                                            valid_min_values = [np.min(arr) for arr in arrays_c if arr.size > 0]
                                            # Verificar si hay valores válidos y calcular el máximo
                                            if valid_min_values:
                                                minic = np.min(valid_min_values)
                                            else:
                                                minic = 0

                                            #max_c = np.max(np.array((np.max(data_cut_obs1[:,7]), np.max(data_cut_obs2[:,7]),
                                            #        np.max(data_cut_sim1[:,7]), np.max(data_cut_sim2[:,7])  )))
                                            # Filtrar los arreglos no vacíos
                                            valid_maxc_values = [np.max(arr) for arr in arrays_c if arr.size > 0]
                                            # Verificar si hay valores válidos y calcular el máximo
                                            if valid_maxc_values:
                                                max_c = np.max(valid_maxc_values)
                                            else:
                                                max_c = 0

                                            if data_cut_obs1[:, 7].size > 0:  # Verificar si el arreglo no está vacío
                                                if np.max(data_cut_obs1[:, 7]) == max_c:
                                                    data_cut = data_cut_obs1
                                            if data_cut_obs2[:, 7].size > 0:  # Verificar si el arreglo no está vacío
                                                if np.max(data_cut_obs2[:, 7]) == max_c:
                                                    data_cut = data_cut_obs2
                                            if data_cut_sim1[:, 7].size > 0:  # Verificar si el arreglo no está vacío
                                                if np.max(data_cut_sim1[:, 7]) == max_c:
                                                    data_cut = data_cut_sim1
                                            if data_cut_sim2[:, 7].size > 0:  # Verificar si el arreglo no está vacío
                                                if np.max(data_cut_sim2[:, 7]) == max_c:
                                                    data_cut = data_cut_sim2

                                            #nbins_clts=int(round(np.max(data_cut_obs1[:,7] ))) #nbins_clts = maxim[0,0]
                                            nbins_clts=int(round(maxic-minic )) #nbins_clts = maxim[0,0]
                                            hist_clts, bins_clts = np.histogram(data_cut_obs1[:,7], bins=nbins_clts)
                                            cltslinbins_all = np.linspace( bins_clts[0], bins_clts[-1] ,len(bins_clts))

                                            wxc1, xc1 = np.histogram(data_cut_obs1[:,7], bins = cltslinbins_all, density = densi)
                                            wxc2, xc2 = np.histogram(data_cut_obs2[:,7], bins = cltslinbins_all, density = densi)
                                            wyc1, yc1 = np.histogram(data_cut_sim1[:,7], bins = cltslinbins_all, density = densi)
                                            wyc2, yc2 = np.histogram(data_cut_sim2[:,7], bins = cltslinbins_all, density = densi)

                                            ncb=1

                                            if(bin_hist>1):
                                                nbins_clts = bin_hist
                                                if(bin_hist > round(maxic-minic) ): nbins_clts = round(maxic-minic)
                                                #if(bin_hist > round(np.max(data_cut_obs1[:,7] )) ): nbins_clts = round(np.max(data_cut_obs1[:,7] ))

                                                nbins_clts_sim = bin_hist
                                                if(bin_hist > round(maxic-minic) ): nbins_clts_sim = round(maxic-minic)
                                                #if(bin_hist > round(np.max(data_cut_sim2[:,7] )) ): nbins_clts_sim = round(np.max(data_cut_sim2[:,7] ))

                                                #else: nbins_clts = bin_hist
                                            else:
                                                nbins_clts = round(maxic-minic)
                                                nbins_clts_sim = round(maxic-minic)
                                                #nbins_clts = round(np.max(data_cut_obs1[:,7]/ncb)) #nbins_clts = maxim[0,0]
                                                #nbins_clts_sim = round(np.max(data_cut_sim2[:,7]/ncb)) #nbins_clts = maxim[0,0]

                                            hist_clts, bins_clts = np.histogram(data_cut_obs1[:,7], bins=nbins_clts)
                                            cltslinbins = np.linspace( bins_clts[0], bins_clts[-1] ,len(bins_clts))

                                            hist_clts_sim, bins_clts_sim = np.histogram(data_cut_sim1[:,7], bins=nbins_clts_sim)
                                            cltslinbins_sim = np.linspace( bins_clts_sim[0], bins_clts_sim[-1] ,len(bins_clts_sim))
                                            ########################################################################################################

                                            cltsbin_obs  = cltslinbins
                                            cltsbin_sim = cltslinbins
                                            cltsbin_bg = cltslinbins
                                            #######################################

                                            wxc_bh1, xc_bh1 = np.histogram(data_cut_obs1[:,7], bins = cltslinbins, density = densi)
                                            wxc_bh2, xc_bh2 = np.histogram(data_cut_obs2[:,7], bins = cltslinbins, density = densi)
                                            wyc_bh1, yc_bh1 = np.histogram(data_cut_sim1[:,7], bins = cltslinbins, density = densi)
                                            wyc_bh2, yc_bh2 = np.histogram(data_cut_sim2[:,7], bins = cltslinbins, density = densi)

                                            #########################################################################################
                                            count_obs1, bin_obs1 = np.histogram(data_cut_obs1[:,7], bins=cltsbin_obs)
                                            hist_clstr_siz_obs1 = {'bins': bin_obs1, 'counts': count_obs1}
                                            bincenters_obs1 = 0.5*(bin_obs1[1:]+bin_obs1[:-1])
                                            norm_obs1 = (np.sum(count_obs1) * np.diff(bin_obs1))

                                            count_obs2, bin_obs2 = np.histogram(data_cut_obs2[:,7], bins=cltsbin_obs)
                                            hist_clstr_siz_obs2 = {'bins': bin_obs2, 'counts': count_obs2}
                                            bincenters_obs2 = 0.5*(bin_obs2[1:]+bin_obs2[:-1])
                                            norm_obs2 = (np.sum(count_obs2) * np.diff(bin_obs2))
                                            #########################################################################################
                                            count_sim1, bin_sim1 = np.histogram(data_cut_sim1[:,7], bins=cltsbin_sim)
                                            hist_clstr_siz_sim1 = {'bins': bin_sim1, 'counts': count_sim1}
                                            bincenters_sim1 = 0.5*(bin_sim1[1:]+bin_sim1[:-1])
                                            norm_sim1 = (np.sum(count_sim1) * np.diff(bin_sim1))

                                            count_sim2, bin_sim2 = np.histogram(data_cut_sim2[:,7], bins=cltsbin_sim)
                                            hist_clstr_siz_sim2 = {'bins': bin_sim2, 'counts': count_sim2}
                                            bincenters_sim2 = 0.5*(bin_sim2[1:]+bin_sim2[:-1])
                                            norm_sim2 = (np.sum(count_sim2) * np.diff(bin_sim2))
                                            #########################################################################################
                                            if(data_bg_cut.ndim>1):
                                                count_bg, bin_bg = np.histogram(data_bg_cut[:,7], bins=cltsbin_bg)
                                            if(data_bg_cut.ndim==1):
                                                count_bg, bin_bg = np.histogram(data_bg_cut[0], bins=cltsbin_bg)
                                            hist_clstr_siz_bg = {'bins': bin_bg, 'counts': count_bg}
                                            #########################################################################################
                                            #########################################################################################
                                            # Guardar en un archivo de texto,  .npy
                                            np.save(dirsave_plt+'/'+'hist_clstr_siz_obs_'+source1+\
                                                    sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist_clstr_siz_obs1 )
                                            np.save(dirsave_plt+'/'+'hist_clstr_siz_obs_'+source2+\
                                                    sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist_clstr_siz_obs2 )

                                            np.save(dirsave_plt+'/'+'hist_clstr_siz_sim_'+source1+\
                                                    sim_n+'_'+difu_gauss+ sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist_clstr_siz_sim1 )
                                            np.save(dirsave_plt+'/'+'hist_clstr_siz_sim_'+source2+\
                                                    sim_n+'_'+difu_gauss+ sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist_clstr_siz_sim2 )

                                            np.save(dirsave_plt+'/'+'hist_clstr_siz_bg_'+source_bg+\
                                                    sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist_clstr_siz_bg )
                                            #########################################################################################

                                            if(densi==True):
                                                obs1_c = count_obs1/norm_obs1
                                                obs2_c = count_obs2/norm_obs2
                                                err_std_obs1_c = np.sqrt(count_obs1)/norm_obs1
                                                err_std_obs2_c = np.sqrt(count_obs2)/norm_obs2
                                                sim1_c = count_sim1/norm_sim1
                                                sim2_c = count_sim2/norm_sim2
                                                err_std_sim1_c = np.sqrt(0.04*count_sim1*count_sim1+count_sim1)/norm_sim1
                                                err_std_sim2_c = np.sqrt(0.04*count_sim2*count_sim2+count_sim2)/norm_sim2
                                                err_std_sim1_c = (0.2*count_sim1)/norm_sim1
                                                err_std_sim2_c = (0.2*count_sim2)/norm_sim2

                                            if(densi==False):
                                                obs1_c = count_obs1
                                                obs2_c = count_obs2
                                                err_std_obs1_c = np.sqrt(count_obs1)
                                                err_std_obs2_c = np.sqrt(count_obs2)
                                                sim1_c = count_sim1
                                                sim2_c = count_sim2
                                                err_std_sim1_c = np.sqrt(0.04*count_sim1*count_sim1+count_sim1)
                                                err_std_sim2_c = np.sqrt(0.04*count_sim2*count_sim1+count_sim2)
                                                err_std_sim1_c = (0.2*count_sim1)
                                                err_std_sim2_c = (0.2*count_sim2)
                                            #############################################################################################

                                            #err_c = np.sqrt(wxc_bh + 0.04*wyc_bh*wyc_bh )
                                            err_obs_clst = np.sqrt(err_std_obs1_c*err_std_obs1_c + err_std_obs2_c*err_std_obs2_c )
                                            ch2_obs_clst = chi_2_sigm_test(obs1_c, obs2_c, err_obs_clst)
                                            tmp_str = 'Chi2 test: ' + r'%.2f'%ch2_obs_clst[0] \
                                                    + '       '+'ndf: ' + r'%.2f'%ch2_obs_clst[3] \
                                                    + '\n'+'Chi2/ndf : ' + r'%.2f'%ch2_obs_clst[4]

                                            err_sim_clst = np.sqrt(err_std_sim1_c*err_std_sim1_c + err_std_sim2_c*err_std_sim2_c )
                                            ch2_sim_clst = chi_2_sigm_test(sim1_c, sim2_c, err_sim_clst)
                                            tmp_sim_str = 'Chi2 test: ' + r'%.2f'%ch2_sim_clst[0] \
                                                    + '       '+'ndf: ' + r'%.2f'%ch2_sim_clst[3] \
                                                    + '\n'+'Chi2/ndf : ' + r'%.2f'%ch2_sim_clst[4]

                                            if(labl_opt=='full'):
                                                lbl_obs1 = source1 + ' data'+ sig_bg_thresh +cut_clst_size+cut_max + cut_adc
                                                lbl_obs2 = source2 + ' data'+ sig_bg_thresh +cut_clst_size+cut_max + cut_adc \
                                                            +'\n'+'Chi2/ndf : ' + r'%.2f'%ch2_obs_clst[4]
                                                lbl_sim1 = source1 + ' sim '+d_gauss+ sig_bg_thresh_sim +cut_clst_size+cut_max + cut_adc
                                                lbl_sim2 = source2 + ' sim '+d_gauss+ sig_bg_thresh_sim +cut_clst_size+cut_max + cut_adc \
                                                            +'\n'+'Chi2/ndf : ' + r'%.2f'%ch2_sim_clst[4]
                                                lbl_bg = 'Background' + ' data'+ sig_bg_thresh +cut_clst_size+cut_max + cut_adc
                                            if(labl_opt=='simple'):
                                                lbl_obs1 = source1 + ' data'
                                                lbl_obs2 = source2 + ' data'
                                                lbl_sim1 = source1 + ' simulation '
                                                lbl_sim2 = source2 + ' simulation '
                                                lbl_bg = 'Background' + ' data'
                                            if(labl_opt=='off'):
                                                lbl_obs1 = None
                                                lbl_obs2 = None
                                                lbl_sim1 = None
                                                lbl_sim2 = None
                                                lbl_bg = None


                                            if(c_dat_real==True and plt_obs==True ):
                                                ax0.hist(bincenters_obs1, weights=count_obs1, bins=cltsbin_obs, density=densi, histtype='step'
                                                        , label =  lbl_obs1 + cut_max_clst_porc
                                                        , color='C3', log=log_y, linewidth=l_width*1)

                                                ax0.hist(bincenters_obs2, weights=count_obs2, bins=cltsbin_obs, density=densi, histtype='step'
                                                        , label =  lbl_obs2 + cut_max_clst_porc
                                                        , color='C1', log=log_y, linewidth=l_width*1)

                                                if(densi==True):
                                                    ax0.errorbar(bincenters_obs1, obs1_c, yerr=err_std_obs1_c
                                                                                , fmt='C3'+'.', ecolor='C3', linewidth=l_width)
                                                    ax0.errorbar(bincenters_obs2, obs2_c, yerr=err_std_obs2_c
                                                                                , fmt='C1'+'.', ecolor='C1', linewidth=l_width)
                                                if(densi==False):
                                                    ax0.errorbar(bincenters_obs1, count_obs1, yerr=err_std_obs1_c
                                                                                , fmt='C3'+'.', ecolor='C3', linewidth=l_width)
                                                    ax0.errorbar(bincenters_obs2, count_obs2, yerr=err_std_obs2_c
                                                                                , fmt='C1'+'.', ecolor='C1', linewidth=l_width)

                                            if(c_dat_sim==True and plt_sim==True ):
                                                #######################################################################################################################################################################################################################
                                                ax0.hist(bincenters_sim1, weights=count_sim1, bins=cltsbin_sim, density=densi, histtype='step'
                                                            , label = lbl_sim1 + cut_max_clst_porc_sim
                                                            , color='C0', log=log_y, linewidth=l_width*1)

                                                ax0.hist(bincenters_sim2, weights=count_sim2, bins=cltsbin_sim, density=densi, histtype='step'
                                                            , label = lbl_sim2 + cut_max_clst_porc_sim
                                                            , color='C7', log=log_y, linewidth=l_width*1)

                                                ##########################################################################
                                                if(densi==True):
                                                    ax0.errorbar(bincenters_sim1, sim1_c, yerr=err_std_sim1_c
                                                                                , fmt='C0'+'.', ecolor='C0', linewidth=l_width*1)
                                                    ax0.errorbar(bincenters_sim2, sim2_c, yerr=err_std_sim2_c
                                                                                , fmt='C7'+'.', ecolor='C7', linewidth=l_width*1)

                                                if(densi==False):
                                                    ax0.errorbar(bincenters_sim1, count_sim1, yerr= err_std_sim1_c
                                                                                , fmt='C0'+'.', ecolor='C0', linewidth=l_width*1)#,label = tmp_str)
                                                    ax0.errorbar(bincenters_sim2, count_sim2, yerr= err_std_sim2_c
                                                                                , fmt='C7'+'.', ecolor='C7', linewidth=l_width*1)#,label = tmp_str)
                                            #######################################################################################################################################################################################################################
                                            ##############################################################################################################################################################################################################################

                                            if(c_bg_real==True and plt_bg==True ):

                                                if(data_bg_cut.ndim>1):
                                                    count_bg, bin_bg = np.histogram(data_bg_cut[:,7], bins=cltsbin_bg)
                                                    bincenters_bg = 0.5*(bin_bg[1:]+bin_bg[:-1])
                                                    ax0.hist(bincenters_bg, weights=count_bg, bins=cltsbin_bg, density=densi, histtype='step'
                                                               #, label='backgnd'+'_'+str(sigma_obs)+'sig'+' '
                                                                , label=lbl_bg
                                                                , color='k', log=log_y, linewidth=l_width*1)

                                                if(data_bg_cut.ndim==1):
                                                    count_bg, bin_bg = np.histogram(data_bg_cut[0], bins=cltsbin_bg)
                                                    bincenters_bg = 0.5*(bin_bg[1:]+bin_bg[:-1])
                                                    ax0.hist(bincenters_bg, weights=count_bg, bins=cltsbin_bg, density=densi, histtype='step'
                                                                , label=lbl_bg
                                                                , color='k', log=log_y, linewidth=l_width*1)

                                                #bincenters_bg = 0.5*(bin_bg[1:]+bin_bg[:-1])
                                                norm_bg = (np.sum(count_bg) * np.diff(bin_bg))
                                                err_std_bg = np.sqrt(count_bg)/ norm_bg
                                                if(densi==True):
                                                    ax0.errorbar(bincenters_bg, count_bg/norm_bg, yerr=err_std_bg, fmt='k'+'.', ecolor='k')
                                                if(densi==False):
                                                    ax0.errorbar(bincenters_bg, count_bg, yerr=np.sqrt(count_bg), fmt='k'+'.', ecolor='k')

                                            if(plt_obs==True or plt_sim==True):
                                                #ax0.set_xlim(0,mx_cs*mxs )
                                                ax0.set_ylim(0, yscal_cls)
                                                ax0.set_xlim(-2.25,mx_cs*mxs+3 )
                                                #
                                                ax0.set_ylabel('Clusters number', fontsize=font_siz+2)
                                                if(plt_one==True):ax0.legend()
                                                if(grid_==True):ax0.grid(grid_)
                                                ax0.tick_params(labelbottom=False, direction='inout', width=3, length=14 )
                                                ax0.minorticks_on()
                                                ax0.tick_params(axis='both', direction='inout', which='minor',length=8 ,width=2)
                                                if( plot_nsig==False and plot_ratio==False ):
                                                    ax0.set_xlabel(xlab[0][0], fontsize=font_siz+2)
                                                    ax0.tick_params(labelbottom=True, direction='inout', width=3, length=14 )

                                            ylabl_comp = r'$\frac{ ('+source1+' - '+source2+')}{\sigma}$'
                                            comp_obs, err_comp_obs = comp_2obs(wxc_bh1, wxc_bh2 )
                                            comp_sim, err_comp_sim = comp_2sim(wyc_bh1, wyc_bh2 )
                                            if np.isnan(comp_obs).all() and np.isnan(comp_sim).all():
                                                # Ambos arreglos son NaN
                                                delta_min = 0
                                                delta_max = 0
                                            elif np.isnan(comp_sim).all():
                                                # Solo comp_sim es NaN
                                                delta_min = np.nanmin(comp_obs)
                                                delta_max = np.nanmax(comp_obs[np.isfinite(comp_obs)])
                                            elif np.isnan(comp_obs).all():
                                                # Solo comp_obs es NaN
                                                delta_min = np.nanmin(comp_sim)
                                                delta_max = np.nanmax(comp_sim[np.isfinite(comp_sim)])
                                            else:
                                                # Ambos tienen valores válidos
                                                delta_min = np.nanmin([np.nanmin(comp_obs), np.nanmin(comp_sim)])
                                                delta_max = np.nanmax([
                                                    np.nanmax(comp_obs[np.isfinite(comp_obs)]),
                                                    np.nanmax(comp_sim[np.isfinite(comp_sim)])
                                                ])

                                            if(plt_obs==True and plot_nsig==True):
                                                #ax2.axhline(y = 0., color = 'k', linestyle = '--')
                                                ax2.hist(cltslinbins, weights=0*cltslinbins, bins = cltslinbins, histtype='step', ls='--', density=densi, color='k', linewidth=l_width*1 )
                                                ax2.errorbar(bincenters_obs1, comp_obs, yerr=plt_err*err_comp_obs, fmt='C1'+'>', lw=2, markersize=12 )
                                                if(zoom_delta==True):ax2.set_ylim(delta_min,delta_max)

                                            if(plt_sim==True and plot_nsig==True):
                                                #ax2.axhline(y = 0., color = 'k', linestyle = '--')
                                                ax2.hist(cltslinbins, weights=0*cltslinbins, bins = cltslinbins, histtype='step', ls='--', density=densi, color='k', linewidth=l_width*1 )
                                                ax2.errorbar(bincenters_obs1, comp_sim, yerr=plt_err*err_comp_sim, fmt='C0'+'o', lw=2, markersize=10 )
                                                if(zoom_delta==True):ax2.set_ylim(delta_min,delta_max)

                                            if( (plt_obs==True or plt_sim==True) and plot_nsig==True):
                                                ax2.tick_params(labelbottom=False, width=3,  direction='inout', length=14 )
                                                ax2.minorticks_on()
                                                ax2.tick_params(axis='x', direction='inout', which='minor',length=8 ,width=2)
                                                ax2.set_ylabel(ylabl_comp, fontsize=font_siz+2)
                                                if(plot_ratio==False):
                                                    ax2.set_xlabel(xlab[0][0], fontsize=font_siz+4)
                                                    ax2.tick_params(labelbottom=True, width=3,  direction='inout', length=14 )
                                                #ax2.yaxis.tick_right()
                                                #ax2.yaxis.set_label_position("right")
                                                if(grid_==True):
                                                    #ax2.grid( axis = 'x',  linestyle = '--',)
                                                    ax2.grid(grid_)

                                            ylabl_ratio= r'$ratio = \frac{'+source1+'}{'+source2+'}$'
                                            ratio_obs, err_ratio_obs = ratio_2obs(wxc_bh1, wxc_bh2 )
                                            ratio_sim, err_ratio_sim = ratio_2sim(wyc_bh1, wyc_bh2 )
                                            if np.isnan(ratio_obs).all() and np.isnan(ratio_sim).all():
                                                # Ambos arreglos son NaN
                                                ratio_min = 0
                                                ratio_max = 0
                                            elif np.isnan(ratio_sim).all():
                                                # Solo ratio_sim es NaN
                                                ratio_min = np.nanmin(ratio_obs)
                                                ratio_max = np.nanmax(ratio_obs[np.isfinite(ratio_obs)])
                                            elif np.isnan(ratio_obs).all():
                                                # Solo ratio_obs es NaN
                                                ratio_min = np.nanmin(ratio_sim)
                                                ratio_max = np.nanmax(ratio_sim[np.isfinite(ratio_sim)])
                                            else:
                                                # Ambos tienen valores válidos
                                                ratio_min = np.nanmin([np.nanmin(ratio_obs), np.nanmin(ratio_sim)])
                                                ratio_max = np.nanmax([
                                                    np.nanmax(ratio_obs[np.isfinite(ratio_obs)]),
                                                    np.nanmax(ratio_sim[np.isfinite(ratio_sim)])
                                                ])

                                            if(plt_obs==True and plot_ratio==True):
                                                #ax4.axhline(y = 0., color = 'k', linestyle = '--')
                                                ax4.hist(cltslinbins, weights=0*cltslinbins, bins = cltslinbins, histtype='step', ls='--', density=densi, color='k', linewidth=l_width*1 )
                                                ax4.errorbar(bincenters_obs1, ratio_obs, yerr=plt_err*err_ratio_obs, fmt='C1'+'>', lw=2, markersize=12 )
                                                if(zoom_ratio==True):ax4.set_ylim(-0.5+ratio_min,ratio_max)

                                            if(plt_sim==True and plot_ratio==True):
                                                #ax4.axhline(y = 0., color = 'k', linestyle = '--')
                                                ax4.hist(cltslinbins, weights=0*cltslinbins, bins = cltslinbins, histtype='step', ls='--', density=densi, color='k', linewidth=l_width*1 )
                                                ax4.errorbar(bincenters_obs1, ratio_sim, yerr=plt_err*err_ratio_sim, fmt='C0'+'o', lw=2, markersize=10 )
                                                if(zoom_ratio==True):ax4.set_ylim(-0.5+ratio_min,ratio_max)

                                            if( (plt_obs==True or plt_sim==True) and plot_ratio==True):
                                                ax4.tick_params(labelbottom=True, width=3,  direction='inout', length=14 )
                                                ax4.minorticks_on()
                                                ax4.tick_params(axis='x', direction='inout', which='minor',length=8 ,width=2)
                                                ax4.set_xlabel(xlab[0][0], fontsize=font_siz+4)
                                                ax4.set_ylabel(ylabl_ratio, fontsize=font_siz+2)
                                                #ax4.yaxis.tick_right()
                                                #ax4.yaxis.set_label_position("right")
                                                if(grid_==True):
                                                    #ax4.grid( axis = 'x',  linestyle = '--',)
                                                    ax4.grid(grid_)


                                            fig.tight_layout()
                                            #save_subplt = 'comp_plot_hist'+dens+'_cluster_'+strbin +str(z*2).zfill(2)+'mm_sim'+sim_filt \
                                            #                +'_'+source1+'_'+source2+'_'+str(sigma)+'sig'+str_hist_comp+cut_adc+cut_str
                                            save_subplt = 'comp_plot_hist'+dens+'_cluster_'+strbin +str(z*2).zfill(2)+'mm' + name_cuts
                                            print(save_subplt)

                                            # Save just the portion _inside_ the second axis's boundaries
                                            #extent = ax0.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                                            if(plt_one==True):
                                                plt.savefig(dirsave_plt+'/'+ save_subplt +'.pdf', dpi=150)#,  bbox_inches=extent.expanded(1.2, 1.3), dpi=150)
                                                plt.savefig(dirsave_plt+'/'+ save_subplt +'.png', dpi=150)#,  bbox_inches=extent.expanded(1.2, 1.3), dpi=150)
                                                #plt.show()

                                                #plt.clf()
                                                #plt.close()


                                        ####MEAN
                                        if(type_plt=='cl_mean'):
                                            if(plt_one==True):
                                                if( plot_nsig==True and plot_ratio==True ):
                                                    fig, (ax1, ax3, ax5) = plt.subplots(3,1,figsize=(14.5,14),gridspec_kw={'height_ratios': [5, 1, 1] } )

                                                if( plot_nsig==True and plot_ratio==False ):
                                                    fig, (ax1, ax3)  = plt.subplots(2,1,figsize=(14.5,12),  gridspec_kw={'height_ratios': [5, 1]} )

                                                if( plot_nsig==False and plot_ratio==True ):
                                                    fig, (ax1, ax5) = plt.subplots(2,1,figsize=(14.5,12),  gridspec_kw={'height_ratios': [5, 1]} )

                                                if( plot_nsig==False and plot_ratio==False ):
                                                    fig, (ax1) = plt.subplots(figsize=(14.5,9.5) )#, sharex=True )

                                            xlabs = 'Mean ADC'

                                            #maxime = 1.1*np.min(np.array((np.max(data_cut_obs1[:,8]), np.max(data_cut_obs2[:,8]),
                                            #        np.max(data_cut_sim1[:,8]), np.max(data_cut_sim2[:,8])  )))
                                            # Lista de arreglos que deseas evaluar
                                            arrays_me = [ data_cut_obs1[:, 8], data_cut_obs2[:, 8], data_cut_sim1[:, 8], data_cut_sim2[:, 8]]
                                            # Filtrar los arreglos no vacíos
                                            valid_max_values = [np.max(arr) for arr in arrays_me if arr.size > 0]
                                            # Verificar si hay valores válidos y calcular el máximo
                                            if valid_max_values:
                                                maxime = 1.1*np.min(valid_max_values)
                                            else:
                                                maxime = 0

                                            #minime = np.min(np.array((np.min(data_cut_obs1[:,8]), np.min(data_cut_obs2[:,8]),
                                            #        np.min(data_cut_sim1[:,8]), np.min(data_cut_sim2[:,8]) )))
                                            # Filtrar los arreglos no vacíos
                                            valid_min_values = [np.min(arr) for arr in arrays_me if arr.size > 0]
                                            # Verificar si hay valores válidos y calcular el máximo
                                            if valid_min_values:
                                                minime = np.min(valid_min_values)
                                            else:
                                                minime = 0

                                            #max_me = np.max(np.array((np.max(data_cut_obs1[:,8]), np.max(data_cut_obs2[:,8]),
                                            #        np.max(data_cut_sim1[:,8]), np.max(data_cut_sim2[:,8])  )))
                                            # Filtrar los arreglos no vacíos
                                            valid_maxme_values = [np.max(arr) for arr in arrays_me if arr.size > 0]
                                            # Verificar si hay valores válidos y calcular el máximo
                                            if valid_maxme_values:
                                                max_me = np.max(valid_maxme_values)
                                            else:
                                                max_me = 0

                                            if( min_adc==adc_cut):data_cut = data_cut_obs1
                                            else:
                                                if data_cut_obs1[:, 8].size > 0:  # Verificar si el arreglo no está vacío
                                                    if np.max(data_cut_obs1[:, 8]) == max_me:
                                                        data_cut = data_cut_obs1
                                                if data_cut_obs2[:, 8].size > 0:  # Verificar si el arreglo no está vacío
                                                    if np.max(data_cut_obs2[:, 8]) == max_me:
                                                        data_cut = data_cut_obs2
                                                if data_cut_sim1[:, 8].size > 0:  # Verificar si el arreglo no está vacío
                                                    if np.max(data_cut_sim1[:, 8]) == max_me:
                                                        data_cut = data_cut_sim1
                                                if data_cut_sim2[:, 8].size > 0:  # Verificar si el arreglo no está vacío
                                                    if np.max(data_cut_sim2[:, 8]) == max_me:
                                                        data_cut = data_cut_sim2



                                            #nbins_mean=int(round(np.max(data_cut_obs1[:,8] ))) #nbins_mean = maxim[0,1]
                                            nbins_mean=int(round(maxime-minime )) #nbins_mean = maxim[0,1]
                                            hist_mean, bins_mean = np.histogram(data_cut_obs1[:,8], bins=nbins_mean)
                                            meanlinbins_all = np.linspace( bins_mean[0], bins_mean[-1] ,len(bins_mean))

                                            wxm1,xm1 = np.histogram(data_cut_obs1[:,8], bins = meanlinbins_all, density = densi)
                                            wxm2,xm2 = np.histogram(data_cut_obs2[:,8], bins = meanlinbins_all, density = densi)
                                            wym1,ym1 = np.histogram(data_cut_sim1[:,8], bins = meanlinbins_all, density = densi)
                                            wym2,ym2 = np.histogram(data_cut_sim2[:,8], bins = meanlinbins_all, density = densi)

                                            nmb = 1 #16

                                            if(bin_hist>1):
                                                nbins_mean = bin_hist
                                                if(bin_hist > round(maxime-minime) ): nbins_mean = round(maxime-minime)
                                                #if(bin_hist >round(np.max(data_cut_obs1[:,8] )) ): nbins_mean = round(np.max(data_cut_obs1[:,8] ))

                                                nbins_mean_sim = bin_hist
                                                if(bin_hist > round(maxime-minime) ): nbins_mean_sim = round(maxime-minime)
                                                #if(bin_hist >round(np.max(data_cut_sim2[:,8] )) ): nbins_mean_sim = round(np.max(data_cut_sim2[:,8] ))
                                                #else: nbins_mean = bin_hist
                                            else:
                                                nbins_mean = round(maxime-minime)
                                                nbins_mean_sim = round(maxime-minime)
                                                #nbins_mean = round(np.max(data_cut_obs1[:,8]/nmb)) #nbins_mean = maxim[0,0]
                                                #nbins_mean_sim = round(np.max(data_cut_sim2[:,8]/nmb)) #nbins_mean = maxim[0,0]

                                            hist_mean, bins_mean = np.histogram(data_cut_obs1[:,8], bins=nbins_mean)
                                            meanlinbins = np.linspace( bins_mean[0], bins_mean[-1] ,len(bins_mean))

                                            hist_mean_sim, bins_mean_sim = np.histogram(data_cut_sim1[:,8], bins=nbins_mean_sim)
                                            meanlinbins_sim = np.linspace( bins_mean_sim[0], bins_mean_sim[-1] ,len(bins_mean_sim))
                                            #########################################################################################

                                            meanbin_obs  = meanlinbins
                                            meanbin_sim  = meanlinbins
                                            meanbin_bg = meanlinbins
                                            #########################################################################################
                                            wxm_bh1,xm_bh1 = np.histogram(data_cut_obs1[:,8], bins = meanlinbins, density = densi)
                                            wxm_bh2,xm_bh2 = np.histogram(data_cut_obs2[:,8], bins = meanlinbins, density = densi)
                                            wym_bh1,ym_bh1 = np.histogram(data_cut_sim1[:,8], bins = meanlinbins, density = densi)
                                            wym_bh2,ym_bh2 = np.histogram(data_cut_sim2[:,8], bins = meanlinbins, density = densi)

                                            #########################################################################################
                                            count_obs1, bin_obs1 = np.histogram(data_cut_obs1[:,8], bins=meanbin_obs)
                                            hist_clstr_mean_obs1 = {'bins': bin_obs1, 'counts': count_obs1}
                                            bincenters_obs1 = 0.5*(bin_obs1[1:]+bin_obs1[:-1])
                                            norm_obs1 = (np.sum(count_obs1) * np.diff(bin_obs1))

                                            count_obs2, bin_obs2 = np.histogram(data_cut_obs2[:,8], bins=meanbin_obs)
                                            hist_clstr_mean_obs2 = {'bins': bin_obs2, 'counts': count_obs2}
                                            bincenters_obs2 = 0.5*(bin_obs2[1:]+bin_obs2[:-1])
                                            norm_obs2 = (np.sum(count_obs2) * np.diff(bin_obs2))
                                            #########################################################################################
                                            count_sim1, bin_sim1 = np.histogram(data_cut_sim1[:,8], bins=meanbin_sim)
                                            hist_clstr_mean_sim1 = {'bins': bin_sim1, 'counts': count_sim1}
                                            bincenters_sim1 = 0.5*(bin_sim1[1:]+bin_sim1[:-1])
                                            norm_sim1 = (np.sum(count_sim1) * np.diff(bin_sim1))

                                            count_sim2, bin_sim2 = np.histogram(data_cut_sim2[:,8], bins=meanbin_sim)
                                            hist_clstr_mean_sim2 = {'bins': bin_sim2, 'counts': count_sim2}
                                            bincenters_sim2 = 0.5*(bin_sim2[1:]+bin_sim2[:-1])
                                            norm_sim2 = (np.sum(count_sim2) * np.diff(bin_sim2))
                                            #########################################################################################
                                            if(data_bg_cut.ndim>1):
                                                count_bg, bin_bg = np.histogram(data_bg_cut[:,8], bins=meanbin_bg)
                                            if(data_bg_cut.ndim==1):
                                                count_bg, bin_bg = np.histogram(data_bg_cut[1], bins=meanbin_bg)
                                            hist_clstr_mean_bg = {'bins': bin_bg, 'counts': count_bg}
                                            #########################################################################################
                                            #########################################################################################
                                            # Guardar en un archivo de texto,  .npy
                                            np.save(dirsave_plt+'/'+'hist_clstr_mean_obs_'+source1+\
                                                    sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist_clstr_mean_obs1 )
                                            np.save(dirsave_plt+'/'+'hist_clstr_mean_obs_'+source2+\
                                                    sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist_clstr_mean_obs2 )

                                            np.save(dirsave_plt+'/'+'hist_clstr_mean_sim_'+source1+\
                                                    sim_n+'_'+difu_gauss+ sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist_clstr_mean_sim1 )
                                            np.save(dirsave_plt+'/'+'hist_clstr_mean_sim_'+source2+\
                                                    sim_n+'_'+difu_gauss+ sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist_clstr_mean_sim2 )

                                            np.save(dirsave_plt+'/'+'hist_clstr_mean_bg_'+source_bg+\
                                                    sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist_clstr_mean_bg )
                                            #########################################################################################

                                            if(densi==True):
                                                obs1_m = count_obs1/norm_obs1
                                                obs2_m = count_obs2/norm_obs2
                                                err_std_obs1_m = np.sqrt(count_obs1)/norm_obs1
                                                err_std_obs2_m = np.sqrt(count_obs2)/norm_obs2
                                                sim1_m = count_sim1/norm_sim1
                                                sim2_m = count_sim2/norm_sim2
                                                err_std_sim1_m = 0.2*count_sim1/norm_sim1#+ np.sqrt(count_sim1)
                                                err_std_sim2_m = 0.2*count_sim2/norm_sim2

                                            if(densi==False):
                                                obs1_m = count_obs1
                                                obs2_m = count_obs2
                                                err_std_obs1_m = np.sqrt(count_obs1)
                                                err_std_obs2_m = np.sqrt(count_obs2)
                                                sim1_m = count_sim1/norm_sim1
                                                sim2_m = count_sim2/norm_sim2
                                                err_std_sim1_m = 0.2*count_sim1 #+ np.sqrt(count_sim1)
                                                err_std_sim2_m = 0.2*count_sim2

                                            #############################################################################################

                                            err_obs_mean = np.sqrt(err_std_obs1_m*err_std_obs1_m + err_std_obs2_m*err_std_obs2_m )
                                            ch2_obs_mean = chi_2_sigm_test(obs1_m, obs2_m, err_obs_mean)
                                            tmp_obs_str = 'Chi2 test: ' + r'%.2f'%ch2_obs_mean[0] \
                                                    + '       '+'ndf: ' + r'%.2f'%ch2_obs_mean[3] \
                                                    + '\n'+'Chi2/ndf : ' + r'%.2f'%ch2_obs_mean[4]

                                            err_sim_mean = np.sqrt(err_std_sim1_m*err_std_sim1_m + err_std_sim2_m*err_std_sim2_m )
                                            ch2_sim_mean = chi_2_sigm_test(sim1_m, sim2_m, err_sim_mean)
                                            tmp_sim_str = 'Chi2 test: ' + r'%.2f'%ch2_sim_mean[0] \
                                                    + '       '+'ndf: ' + r'%.2f'%ch2_sim_mean[3] \
                                                    + '\n'+'Chi2/ndf : ' + r'%.2f'%ch2_sim_mean[4]

                                            if(labl_opt=='full'):
                                                lbl_obs1 = source1 + ' data'+ sig_bg_thresh +cut_clst_size+cut_max + cut_adc
                                                lbl_obs2 = source2 + ' data'+ sig_bg_thresh +cut_clst_size+cut_max + cut_adc \
                                                            +'\n'+'Chi2/ndf : ' + r'%.2f'%ch2_obs_mean[4]
                                                lbl_sim1 = source1 + ' sim '+d_gauss+ sig_bg_thresh_sim +cut_clst_size+cut_max + cut_adc
                                                lbl_sim2 = source2 + ' sim '+d_gauss+ sig_bg_thresh_sim +cut_clst_size+cut_max + cut_adc \
                                                            +'\n'+'Chi2/ndf : ' + r'%.2f'%ch2_sim_mean[4]
                                                lbl_bg = 'Background' + ' data'+ sig_bg_thresh +cut_clst_size+cut_max + cut_adc
                                            if(labl_opt=='simple'):
                                                lbl_obs1 = source1 + ' data'
                                                lbl_obs2 = source2 + ' data'
                                                lbl_sim1 = source1 + ' simulation '
                                                lbl_sim2 = source2 + ' simulation '
                                                lbl_bg = 'Background' + ' data'
                                            if(labl_opt=='off'):
                                                lbl_obs1 = None
                                                lbl_obs2 = None
                                                lbl_sim1 = None
                                                lbl_sim2 = None
                                                lbl_bg = None


                                            if(c_dat_real==True and plt_obs==True ):
                                                ax1.hist(bincenters_obs1, weights=count_obs1, bins=meanbin_obs, density=densi, histtype='step'
                                                            #, label=source1+'_'+ag+'ag'+'_'+str(sigma_obs)+'sig'+'_z_'+str(z*2)+'mm'+'_t'+tiemp  # + '_Clst_'+str(data_cut_obs1[:,8].size)
                                                            , label =  lbl_obs1 + cut_max_clst_porc
                                                            , color='C3', log=log_y, linewidth=l_width*1)

                                                ax1.hist(bincenters_obs2, weights=count_obs2, bins=meanbin_obs, density=densi, histtype='step'
                                                            #, label=source2+'_'+ag+'ag'+'_'+str(sigma_obs)+'sig'+'_z_'+str(z*2)+'mm'+'_t'+tiemp  # + '_Clst_'+str(data_cut_obs1[:,8].size)
                                                            , label =  lbl_obs2 + cut_max_clst_porc
                                                            , color='C1', log=log_y, linewidth=l_width*1)

                                                if(densi==True):
                                                    ax1.errorbar(bincenters_obs1, obs1_m, yerr=err_std_obs1_m
                                                                                , fmt='C3'+'.', ecolor='C3', linewidth=l_width)
                                                    ax1.errorbar(bincenters_obs2, obs2_m, yerr=err_std_obs2_m
                                                                                , fmt='C1'+'.', ecolor='C1', linewidth=l_width)
                                                if(densi==False):
                                                    ax1.errorbar(bincenters_obs1, count_obs1, yerr=err_std_obs1_m
                                                                                , fmt='C3'+'.', ecolor='C3', linewidth=l_width)
                                                    ax1.errorbar(bincenters_obs2, count_obs2, yerr=err_std_obs2_m
                                                                                , fmt='C1'+'.', ecolor='C1', linewidth=l_width)

                                            if(c_dat_sim==True and plt_sim==True ):
                                                ##############################################################################################################################################################################################################################
                                                ax1.hist(bincenters_sim1, weights=count_sim1, bins=meanbin_sim, density=densi, histtype='step'
                                                            , label = lbl_sim1 + cut_max_clst_porc_sim
                                                            , color='C0', log=log_y, linewidth=l_width*1)

                                                ax1.hist(bincenters_sim2, weights=count_sim2, bins=meanbin_sim, density=densi, histtype='step'
                                                            , label = lbl_sim2 + cut_max_clst_porc_sim
                                                            , color='C7', log=log_y, linewidth=l_width*1)

                                                if(densi==True):
                                                    ax1.errorbar(bincenters_sim1, sim1_m, yerr=err_std_sim1_m
                                                                                , fmt='C0'+'.', ecolor='C0', linewidth=l_width*1)
                                                    ax1.errorbar(bincenters_sim2, sim2_m, yerr=err_std_sim2_m
                                                                                , fmt='C7'+'.', ecolor='C7', linewidth=l_width*1)
                                                if(densi==False):
                                                    ax1.errorbar(bincenters_sim1, count_sim1, yerr= err_std_sim1_m
                                                                                , fmt='C0'+'.', ecolor='C0', linewidth=l_width*1) #,label = tmp_str)
                                                    ax1.errorbar(bincenters_sim2, count_sim2, yerr= err_std_sim2_m
                                                                                , fmt='C7'+'.', ecolor='C7', linewidth=l_width*1) #,label = tmp_str)
                                                ##############################################################################################################################################################################################################################
                                                ##############################################################################################################################################################################################################################

                                            if(c_bg_real==True and plt_bg==True ):

                                                if(data_bg_cut.ndim>1):
                                                    count_bg, bin_bg = np.histogram(data_bg_cut[:,8], bins=meanbin_bg)
                                                    bincenters_bg = 0.5*(bin_bg[1:]+bin_bg[:-1])
                                                    ax1.hist(bincenters_bg, weights=count_bg, bins=meanbin_bg, density=densi, histtype='step'
                                                               , label=lbl_bg
                                                               , color='k', log=log_y, linewidth=l_width*1)

                                                if(data_bg_cut.ndim==1):
                                                    count_bg, bin_bg = np.histogram(data_bg_cut[1], bins=meanbin_bg)
                                                    bincenters_bg = 0.5*(bin_bg[1:]+bin_bg[:-1])
                                                    ax1.hist(bincenters_bg, weights=count_bg, bins=meanbin_bg, density=densi, histtype='step'
                                                               , label=lbl_bg
                                                               , color='k', log=log_y, linewidth=l_width*1)

                                                bincenters_bg = 0.5*(bin_bg[1:]+bin_bg[:-1])
                                                norm_bg = (np.sum(count_bg) * np.diff(bin_bg))
                                                err_std_bg = np.sqrt(count_bg)/ norm_bg
                                                if(densi==True):
                                                    ax1.errorbar(bincenters_bg, count_bg/norm_bg, yerr=err_std_bg, fmt='k'+'.', ecolor='k')
                                                if(densi==False):
                                                    ax1.errorbar(bincenters_bg, count_bg, yerr=np.sqrt(count_bg), fmt='k'+'.', ecolor='k')



                                            if(plt_obs==True or plt_sim==True):
                                                #ax1.set_xlim(0,maxim[0,1]*mxs )
                                                ax1.set_ylabel('Clusters number', fontsize=font_siz+2)
                                                ax1.legend()
                                                if(grid_==True):ax1.grid(grid_)
                                                ax1.tick_params(labelbottom=False, direction='inout', width=3, length=14 )
                                                ax1.minorticks_on()
                                                ax1.tick_params(axis='both', direction='inout', which='minor',length=8 ,width=2)
                                                if( plot_nsig==False and plot_ratio==False ):
                                                    ax1.set_xlabel(xlab[0][1], fontsize=font_siz+2)
                                                    ax1.tick_params(labelbottom=True, direction='inout', width=3, length=14 )

                                            ylabl_comp = r'$\frac{ ('+source1+' - '+source2+')}{\sigma}$'
                                            comp_obs, err_comp_obs = comp_2obs(wxm_bh1, wxm_bh2)
                                            comp_sim, err_comp_sim = comp_2sim(wym_bh1, wym_bh2)

                                            if np.isnan(comp_obs).all() and np.isnan(comp_sim).all():
                                                # Ambos arreglos son NaN
                                                delta_min = 0
                                                delta_max = 0
                                            elif np.isnan(comp_sim).all():
                                                # Solo comp_sim es NaN
                                                delta_min = np.nanmin(comp_obs)
                                                delta_max = np.nanmax(comp_obs[np.isfinite(comp_obs)])
                                            elif np.isnan(comp_obs).all():
                                                # Solo comp_obs es NaN
                                                delta_min = np.nanmin(comp_sim)
                                                delta_max = np.nanmax(comp_sim[np.isfinite(comp_sim)])
                                            else:
                                                # Ambos tienen valores válidos
                                                delta_min = np.nanmin([np.nanmin(comp_obs), np.nanmin(comp_sim)])
                                                delta_max = np.nanmax([
                                                    np.nanmax(comp_obs[np.isfinite(comp_obs)]),
                                                    np.nanmax(comp_sim[np.isfinite(comp_sim)])
                                                ])

                                            if(plt_obs==True and plot_nsig==True):
                                                #ax3.axhline(y = 0., color = 'k', linestyle = '--')
                                                ax3.hist(meanlinbins, weights=0*meanlinbins, bins = meanlinbins, histtype='step', ls='--', density=densi, color='k', linewidth=l_width*1 )
                                                ax3.errorbar(bincenters_obs1, comp_obs, yerr=plt_err*err_comp_obs, fmt='C1'+'>', lw=2, markersize=12 )
                                                if(zoom_delta==True):ax3.set_ylim(delta_min,delta_max)

                                            if(plt_sim==True and plot_nsig==True):
                                                #ax3.axhline(y = 0., color = 'k', linestyle = '--')
                                                ax3.hist(meanlinbins, weights=0*meanlinbins, bins = meanlinbins, histtype='step', ls='--', density=densi, color='k', linewidth=l_width*1 )
                                                ax3.errorbar(bincenters_obs1, comp_sim, yerr=plt_err*err_comp_sim, fmt='C0'+'o', lw=2, markersize=10 )
                                                if(zoom_delta==True):ax3.set_ylim(delta_min,delta_max)

                                            if( (plt_obs==True or plt_sim==True) and plot_nsig==True):
                                                ax3.tick_params(labelbottom=False, width=3,  direction='inout', length=14 )
                                                ax3.minorticks_on()
                                                ax3.tick_params(axis='x', direction='inout', which='minor',length=8 ,width=2)
                                                ax3.set_ylabel(ylabl_comp, fontsize=font_siz+2)
                                                if(plot_ratio==False):
                                                    ax3.set_xlabel(xlab[0][1], fontsize=font_siz+4)
                                                    ax3.tick_params(labelbottom=True, width=3,  direction='inout', length=14 )
                                                #ax3.yaxis.tick_right()
                                                #ax3.yaxis.set_label_position("right")
                                                if(grid_==True):
                                                    #ax3.grid( axis = 'x',  linestyle = '--',)
                                                    ax3.grid(grid_)

                                            ylabl_ratio= r'$ratio = \frac{'+source1+'}{'+source2+'}$'
                                            ratio_obs, err_ratio_obs = ratio_2obs(wxm_bh1, wxm_bh2)
                                            ratio_sim, err_ratio_sim = ratio_2sim(wym_bh1, wym_bh2)
                                            if np.isnan(ratio_obs).all() and np.isnan(ratio_sim).all():
                                                # Ambos arreglos son NaN
                                                ratio_min = 0
                                                ratio_max = 0
                                            elif np.isnan(ratio_sim).all():
                                                # Solo ratio_sim es NaN
                                                ratio_min = np.nanmin(ratio_obs)
                                                ratio_max = np.nanmax(ratio_obs[np.isfinite(ratio_obs)])
                                            elif np.isnan(ratio_obs).all():
                                                # Solo ratio_obs es NaN
                                                ratio_min = np.nanmin(ratio_sim)
                                                ratio_max = np.nanmax(ratio_sim[np.isfinite(ratio_sim)])
                                            else:
                                                # Ambos tienen valores válidos
                                                ratio_min = np.nanmin([np.nanmin(ratio_obs), np.nanmin(ratio_sim)])
                                                ratio_max = np.nanmax([
                                                    np.nanmax(ratio_obs[np.isfinite(ratio_obs)]),
                                                    np.nanmax(ratio_sim[np.isfinite(ratio_sim)])
                                                ])

                                            if(plt_obs==True and plot_ratio==True):
                                                #ax5.axhline(y = 0., color = 'k', linestyle = '--')
                                                ax5.hist(meanlinbins, weights=0*meanlinbins, bins = meanlinbins, histtype='step', ls='--', density=densi, color='k', linewidth=l_width*1 )
                                                ax5.errorbar(bincenters_obs1, ratio_obs, yerr=plt_err*err_ratio_obs, fmt='C1'+'>', lw=2, markersize=12 )
                                                if(zoom_ratio==True):ax5.set_ylim(-0.5+ratio_min,ratio_max)

                                            if(plt_sim==True and plot_ratio==True):
                                                #ax5.axhline(y = 0., color = 'k', linestyle = '--')
                                                ax5.hist(meanlinbins, weights=0*meanlinbins, bins = meanlinbins, histtype='step', ls='--', density=densi, color='k', linewidth=l_width*1 )
                                                ax5.errorbar(bincenters_obs1, ratio_sim, yerr=plt_err*err_ratio_sim, fmt='C0'+'o', lw=2, markersize=10 )
                                                if(zoom_ratio==True):ax5.set_ylim(-0.5+ratio_min,ratio_max)

                                            if( (plt_obs==True or plt_sim==True) and plot_ratio==True):
                                                ax5.tick_params(labelbottom=True, width=3,  direction='inout', length=14 )
                                                ax5.minorticks_on()
                                                ax5.tick_params(axis='x', direction='inout', which='minor',length=8 ,width=2)
                                                ax5.set_xlabel(xlab[0][1], fontsize=font_siz+4)
                                                ax5.set_ylabel(ylabl_ratio, fontsize=font_siz+2)
                                                #ax5.yaxis.tick_right()
                                                #ax5.yaxis.set_label_position("right")
                                                if(grid_==True):
                                                    #ax5.grid( axis = 'x',  linestyle = '--',)
                                                    ax5.grid(grid_)

                                            fig.tight_layout()
                                            save_subplt = 'comp_plot_hist'+dens+'_mean_'+strbin +str(z*2).zfill(2)+'mm'+ name_cuts
                                            print(save_subplt)
                                            # Save just the portion _inside_ the second axis's boundaries
                                            #extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                                            if(plt_one==True):
                                                plt.savefig(dirsave_plt+'/'+ save_subplt+'.pdf', dpi=150)#,  bbox_inches=extent.expanded(1.2, 1.3))
                                                plt.savefig(dirsave_plt+'/'+ save_subplt+'.png', dpi=150)#,  bbox_inches=extent.expanded(1.2, 1.3))
                                                #plt.show()

                                                #plt.clf()
                                                #plt.close()


                                        #### Maximum
                                        if(type_plt=='cl_max_adc'):
                                            if(plt_one==True):
                                                if( plot_nsig==True and plot_ratio==True ):
                                                    fig, (ax6, ax8, ax10) = plt.subplots(3,1,figsize=(14.5,14),gridspec_kw={'height_ratios': [5, 1, 1] } )

                                                if( plot_nsig==True and plot_ratio==False ):
                                                    fig, (ax6, ax8)  = plt.subplots(2,1,figsize=(14.5,12),  gridspec_kw={'height_ratios': [5, 1]} )

                                                if( plot_nsig==False and plot_ratio==True ):
                                                    fig, (ax6, ax10) = plt.subplots(2,1,figsize=(14.5,12),  gridspec_kw={'height_ratios': [5, 1]} )

                                                if( plot_nsig==False and plot_ratio==False ):
                                                    fig, (ax6) = plt.subplots(figsize=(14.5,9.5) )#, sharex=True )


                                            xlabs = 'Maximum ADC'
                                            #r= np.max(np.array((np.max(data_cut_obs1[:,10]), np.max(data_cut_sim2[:,10]) )))
                                            #maxims = 1.1*r
                                            #maxis = np.min(np.array((np.max(data_cut_obs1[:,10]), np.max(data_cut_obs2[:,10]),
                                            #                        np.max(data_cut_sim1[:,10]), np.max(data_cut_sim2[:,10])  )))+1
                                            # Lista de arreglos que deseas evaluar
                                            arrays_mx = [ data_cut_obs1[:, 10], data_cut_obs2[:, 10], data_cut_sim1[:, 10], data_cut_sim2[:, 10]]
                                            # Filtrar los arreglos no vacíos
                                            valid_max_values = [np.max(arr) for arr in arrays_mx if arr.size > 0]
                                            # Verificar si hay valores válidos y calcular el máximo
                                            if valid_max_values:
                                                maxis = np.min(valid_max_values)+1
                                            else:
                                                maxis = 0

                                            #minis = np.min(np.array((np.min(data_cut_obs1[:,10]), np.min(data_cut_obs2[:,10]),
                                            #                        np.min(data_cut_sim1[:,10]), np.min(data_cut_sim2[:,10]) )))
                                            # Filtrar los arreglos no vacíos
                                            valid_min_values = [np.min(arr) for arr in arrays_mx if arr.size > 0]
                                            # Verificar si hay valores válidos y calcular el máximo
                                            if valid_min_values:
                                                minis = np.min(valid_min_values)
                                            else:
                                                minis = 0

                                            #max_s = np.max(np.array((np.max(data_cut_obs1[:,10]), np.max(data_cut_obs2[:,10]),
                                            #                        np.max(data_cut_sim1[:,10]), np.max(data_cut_sim2[:,10])  )))
                                            # Filtrar los arreglos no vacíos
                                            valid_maxmx_values = [np.max(arr) for arr in arrays_mx if arr.size > 0]
                                            # Verificar si hay valores válidos y calcular el máximo
                                            if valid_maxme_values:
                                                max_s = np.max(valid_maxmx_values)
                                            else:
                                                max_s = 0

                                            if( min_adc==adc_cut):data_cut = data_cut_obs1

                                            else:
                                                if data_cut_obs1[:, 10].size > 0:  # Verificar si el arreglo no está vacío
                                                    if np.max(data_cut_obs1[:, 10]) == max_s:
                                                        data_cut = data_cut_obs1
                                                if data_cut_obs2[:, 10].size > 0:  # Verificar si el arreglo no está vacío
                                                    if np.max(data_cut_obs2[:, 10]) == max_s:
                                                        data_cut = data_cut_obs2
                                                if data_cut_sim1[:, 10].size > 0:  # Verificar si el arreglo no está vacío
                                                    if np.max(data_cut_sim1[:, 10]) == max_s:
                                                        data_cut = data_cut_sim1
                                                if data_cut_sim2[:, 10].size > 0:  # Verificar si el arreglo no está vacío
                                                    if np.max(data_cut_sim2[:, 10]) == max_s:
                                                        data_cut = data_cut_sim2



                                            nbins_seed=int(round(maxis-minis ))
                                            #nbins_seed=int(round(np.max(data_cut_obs1[:,10] ))) #nbins_seed = maxim[1,0]
                                            hist_Max, bins_seed = np.histogram(data_cut_obs1[:,10], bins=nbins_seed)
                                            seedlinbins_all = np.linspace( bins_seed[0], bins_seed[-1] ,len(bins_seed))

                                            wxs1, xs1 = np.histogram(data_cut_obs1[:,10], bins = seedlinbins_all, density = densi)
                                            wxs2, xs2 = np.histogram(data_cut_obs2[:,10], bins = seedlinbins_all, density = densi)
                                            wys1, ys1 = np.histogram(data_cut_sim1[:,10], bins = seedlinbins_all, density = densi)
                                            wys2, ys2 = np.histogram(data_cut_sim2[:,10], bins = seedlinbins_all, density = densi)

                                            nsb=1#16

                                            if(bin_hist>1):
                                                nbins_seed = bin_hist
                                                if(bin_hist > round(maxis-minis ) ): nbins_seed = round(maxis-minis )
                                                #if(bin_hist > round(np.max(data_cut_obs1[:,10] )) ): nbins_seed = round(np.max(data_cut_obs1[:,10] ))
                                                #else: nbins_seed = bin_hist
                                                nbins_seed_sim = bin_hist
                                                if(bin_hist > round(maxis-minis ) ): nbins_seed_sim = round(maxis-minis )
                                                #if(bin_hist > round(np.max(data_cut_sim2[:,10] )) ): nbins_seed_sim = round(np.max(data_cut_sim2[:,10] ))
                                            else:
                                                nbins_seed = round(maxis-minis ) #nbins_seed = maxim[1,0]
                                                nbins_seed_sim = round(maxis-minis ) #nbins_seed = maxim[1,0]
                                                #nbins_seed = round(np.max(data_cut_obs1[:,10]/nsb)) #nbins_seed = maxim[1,0]
                                                #nbins_seed_sim = round(np.max(data_cut_sim2[:,10]/nsb)) #nbins_seed = maxim[1,0]

                                            hist_seed, bins_seed = np.histogram(data_cut_obs1[:,10], bins=nbins_seed)
                                            seedlinbins = np.linspace( bins_seed[0], bins_seed[-1] ,len(bins_seed))

                                            hist_seed_sim, bins_seed_sim = np.histogram(data_cut_sim2[:,10], bins=nbins_seed_sim)
                                            seedlinbins_sim = np.linspace( bins_seed_sim[0], bins_seed_sim[-1] ,len(bins_seed_sim))
                                             ################################################

                                            seedbin_obs  = seedlinbins
                                            seedbin_sim  = seedlinbins
                                            seedbinbg = seedlinbins
                                            #########################################################################################
                                            wxs_bh1,xs_bh1 = np.histogram(data_cut_obs1[:,10], bins = seedlinbins, density = densi)
                                            wxs_bh2,xs_bh2 = np.histogram(data_cut_obs2[:,10], bins = seedlinbins, density = densi)
                                            wys_bh1,ys_bh1 = np.histogram(data_cut_sim1[:,10], bins = seedlinbins, density = densi)
                                            wys_bh2,ys_bh2 = np.histogram(data_cut_sim2[:,10], bins = seedlinbins, density = densi)

                                            #########################################################################################
                                            count_obs1, bin_obs1 = np.histogram(data_cut_obs1[:,10], bins=seedbin_obs)
                                            hist_clstr_max_obs1 = {'bins': bin_obs1, 'counts': count_obs1}
                                            bincenters_obs1 = 0.5*(bin_obs1[1:]+bin_obs1[:-1])
                                            norm_obs1 = (np.sum(count_obs1) * np.diff(bin_obs1))

                                            count_obs2, bin_obs2 = np.histogram(data_cut_obs2[:,10], bins=seedbin_obs)
                                            hist_clstr_max_obs2 = {'bins': bin_obs2, 'counts': count_obs2}
                                            bincenters_obs2 = 0.5*(bin_obs2[1:]+bin_obs2[:-1])
                                            norm_obs2 = (np.sum(count_obs2) * np.diff(bin_obs2))
                                            #########################################################################################
                                            count_sim1, bin_sim1 = np.histogram(data_cut_sim1[:,10], bins=seedbin_sim)
                                            hist_clstr_max_sim1 = {'bins': bin_sim1, 'counts': count_sim1}
                                            bincenters_sim1 = 0.5*(bin_sim1[1:]+bin_sim1[:-1])
                                            norm_sim1 = (np.sum(count_sim1) * np.diff(bin_sim1))

                                            count_sim2, bin_sim2 = np.histogram(data_cut_sim2[:,10], bins=seedbin_sim)
                                            hist_clstr_max_sim2 = {'bins': bin_sim2, 'counts': count_sim2}
                                            bincenters_sim2 = 0.5*(bin_sim2[1:]+bin_sim2[:-1])
                                            norm_sim2 = (np.sum(count_sim2) * np.diff(bin_sim2))
                                            #########################################################################################
                                            if(data_bg_cut.ndim>1):
                                                count_bg, bin_bg = np.histogram(data_bg_cut[:,10], bins=seedbinbg)
                                            if(data_bg_cut.ndim==1):
                                                count_bg, bin_bg = np.histogram(data_bg_cut[3], bins=seedbinbg)
                                            hist_clstr_max_bg = {'bins': bin_bg, 'counts': count_bg}
                                            #########################################################################################
                                            #########################################################################################
                                            # Guardar en un archivo de texto,  .npy
                                            np.save(dirsave_plt+'/'+'hist_clstr_max_obs_'+source1+\
                                                    sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist_clstr_max_obs1 )
                                            np.save(dirsave_plt+'/'+'hist_clstr_max_obs_'+source2+\
                                                    sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist_clstr_max_obs2 )

                                            np.save(dirsave_plt+'/'+'hist_clstr_max_sim_'+source1+\
                                                    sim_n+'_'+difu_gauss+ sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist_clstr_max_sim1 )
                                            np.save(dirsave_plt+'/'+'hist_clstr_max_sim_'+source2+\
                                                    sim_n+'_'+difu_gauss+ sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist_clstr_max_sim2 )

                                            np.save(dirsave_plt+'/'+'hist_clstr_max_bg_'+source_bg+\
                                                    sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist_clstr_max_bg )
                                            #########################################################################################

                                            if(densi==True):
                                                obs1_s = count_obs1/norm_obs1
                                                obs2_s = count_obs2/norm_obs2
                                                err_std_obs1_s = np.sqrt(count_obs1)/norm_obs1
                                                err_std_obs2_s = np.sqrt(count_obs2)/norm_obs2
                                                sim1_s = count_sim1/norm_sim1
                                                sim2_s = count_sim2/norm_sim2
                                                err_std_sim1_s = 0.2*count_sim1/norm_sim1#+ np.sqrt(count_sim1)
                                                err_std_sim2_s = 0.2*count_sim2/norm_sim2

                                            if(densi==False):
                                                obs1_s = count_obs1
                                                obs2_s = count_obs2
                                                err_std_obs1_s = np.sqrt(count_obs1)
                                                err_std_obs2_s = np.sqrt(count_obs2)
                                                sim1_s = count_sim1
                                                sim2_s = count_sim2
                                                err_std_sim1_s = 0.2*count_sim1
                                                err_std_sim2_s = 0.2*count_sim2 #+ np.sqrt(count_sim2)

                                            ##############################################################################################

                                            err_obs_max = np.sqrt(err_std_obs1_s*err_std_obs1_s + err_std_obs2_s*err_std_obs2_s )
                                            ch2_obs_max = chi_2_sigm_test(obs1_s, obs2_s, err_obs_max)
                                            tmp_obs_str = 'Chi2 test: ' + r'%.2f'%ch2_obs_max[0] \
                                                    + '       '+'ndf: ' + r'%.2f'%ch2_obs_max[3] \
                                                    + '\n'+'Chi2/ndf : ' + r'%.2f'%ch2_obs_max[4]

                                            err_sim_max = np.sqrt(err_std_sim1_s*err_std_sim1_s + err_std_sim2_s*err_std_sim2_s )
                                            ch2_sim_max = chi_2_sigm_test(sim1_s, sim2_s, err_sim_max)
                                            tmp_sim_str = 'Chi2 test: ' + r'%.2f'%ch2_sim_max[0] \
                                                    + '       '+'ndf: ' + r'%.2f'%ch2_sim_max[3] \
                                                    + '\n'+'Chi2/ndf : ' + r'%.2f'%ch2_sim_max[4]

                                            if(labl_opt=='full'):
                                                lbl_obs1 = source1 + ' data'+ sig_bg_thresh +cut_clst_size+cut_max + cut_adc
                                                lbl_obs2 = source2 + ' data'+ sig_bg_thresh +cut_clst_size+cut_max + cut_adc \
                                                            +'\n'+'Chi2/ndf : ' + r'%.2f'%ch2_obs_max[4]
                                                lbl_sim1 = source1 + ' sim '+d_gauss+ sig_bg_thresh_sim +cut_clst_size+cut_max + cut_adc
                                                lbl_sim2 = source2 + ' sim '+d_gauss+ sig_bg_thresh_sim +cut_clst_size+cut_max + cut_adc \
                                                            +'\n'+'Chi2/ndf : ' + r'%.2f'%ch2_sim_max[4]
                                                lbl_bg = 'Background' + ' data'+ sig_bg_thresh +cut_clst_size+cut_max + cut_adc
                                            if(labl_opt=='simple'):
                                                lbl_obs1 = source1 + ' data'
                                                lbl_obs2 = source2 + ' data'
                                                lbl_sim1 = source1 + ' simulation '
                                                lbl_sim2 = source2 + ' simulation '
                                                lbl_bg = 'Background' + ' data'
                                            if(labl_opt=='off'):
                                                lbl_obs1 = None
                                                lbl_obs2 = None
                                                lbl_sim1 = None
                                                lbl_sim2 = None
                                                lbl_bg = None


                                            if(c_dat_real==True and plt_obs==True ):
                                                ax6.hist(bincenters_obs1, weights=count_obs1, bins=seedbin_obs, density=densi, histtype='step'
                                                           , label =  lbl_obs1 + cut_max_clst_porc
                                                            , color='C3', log=log_y, linewidth=l_width*1)

                                                ax6.hist(bincenters_obs2, weights=count_obs2, bins=seedbin_obs, density=densi, histtype='step'
                                                           , label =  lbl_obs2 + cut_max_clst_porc
                                                            , color='C1', log=log_y, linewidth=l_width*1)

                                                if(densi==True):
                                                    ax6.errorbar(bincenters_obs1, obs1_s, yerr=err_std_obs1_s
                                                                                , fmt='C3'+'.', ecolor='C3', linewidth=l_width)
                                                    ax6.errorbar(bincenters_obs2, obs2_s, yerr=err_std_obs2_s
                                                                                , fmt='C1'+'.', ecolor='C1', linewidth=l_width)
                                                if(densi==False):
                                                    ax6.errorbar(bincenters_obs1, count_obs1, yerr=err_std_obs1_s
                                                                                , fmt='C3'+'.', ecolor='C3', linewidth=l_width)
                                                    ax6.errorbar(bincenters_obs2, count_obs2, yerr=err_std_obs2_s
                                                                                , fmt='C1'+'.', ecolor='C1', linewidth=l_width)

                                            if(c_dat_sim==True and plt_sim==True ):
                                                ##############################################################################################################################################################################################################################
                                                ax6.hist(bincenters_sim1, weights=count_sim1, bins=seedbin_sim, density=densi, histtype='step'
                                                            , label = lbl_sim1 + cut_max_clst_porc_sim
                                                            , color='C0', log=log_y, linewidth=l_width*1)

                                                ax6.hist(bincenters_sim2, weights=count_sim2, bins=seedbin_sim, density=densi, histtype='step'
                                                            , label = lbl_sim2 + cut_max_clst_porc_sim
                                                            , color='C7', log=log_y, linewidth=l_width*1)

                                                if(densi==True):
                                                    ax6.errorbar(bincenters_sim1, sim1_s, yerr=err_std_sim1_s
                                                                                , fmt='C0'+'.', ecolor='C0', linewidth=l_width*1)
                                                    ax6.errorbar(bincenters_sim2, sim2_s, yerr=err_std_sim2_s
                                                                                , fmt='C7'+'.', ecolor='C7', linewidth=l_width*1)

                                                if(densi==False):
                                                    ax6.errorbar(bincenters_sim1, count_sim1, yerr= err_std_sim1_s
                                                                                , fmt='C0'+'.', ecolor='C0', linewidth=l_width*1) #,label = tmp_str)
                                                    ax6.errorbar(bincenters_sim2, count_sim2, yerr= err_std_sim2_s
                                                                                , fmt='C7'+'.', ecolor='C7', linewidth=l_width*1) #,label = tmp_str)
                                            ##############################################################################################################################################################################################################################
                                            ##############################################################################################################################################################################################################################

                                            if(c_bg_real==True and plt_bg==True ):

                                                if(data_bg_cut.ndim>1):
                                                    count_bg, bin_bg = np.histogram(data_bg_cut[:,10], bins=seedbinbg)
                                                    bincenters_bg = 0.5*(bin_bg[1:]+bin_bg[:-1])
                                                    ax6.hist(bincenters_bg, weights=count_bg, bins=seedbinbg, density=densi, histtype='step'
                                                               , label=lbl_bg
                                                               , color='k', log=log_y, linewidth=l_width*1)

                                                if(data_bg_cut.ndim==1):
                                                    count_bg, bin_bg = np.histogram(data_bg_cut[3], bins=seedbinbg)
                                                    bincenters_bg = 0.5*(bin_bg[1:]+bin_bg[:-1])
                                                    ax6.hist(bincenters_bg, weights=count_bg, bins=seedbinbg, density=densi, histtype='step'
                                                               , label=lbl_bg
                                                               , color='k', log=log_y, linewidth=l_width*1)

                                                bincenters_bg = 0.5*(bin_bg[1:]+bin_bg[:-1])
                                                norm_bg = (np.sum(count_bg) * np.diff(bin_bg))
                                                err_std_bg = np.sqrt(count_bg)/ norm_bg
                                                if(densi==True):
                                                    ax6.errorbar(bincenters_bg, count_bg/norm_bg, yerr=err_std_bg, fmt='k'+'.', ecolor='k')
                                                if(densi==False):
                                                    ax6.errorbar(bincenters_bg, count_bg, yerr=np.sqrt(count_bg), fmt='k'+'.', ecolor='k')


                                            if(plt_obs==True or plt_sim==True):
                                                #ax6.set_xlim(0,maxim[1,0]*mxs )
                                                ax6.set_ylim(0, yscal_max)
                                                #ax6.set_xlabel(xlab[1][0], fontsize=font_siz+2)
                                                ax6.set_ylabel('Clusters number', fontsize=font_siz+2)
                                                if(plt_one==True):ax6.legend()
                                                if(grid_==True):ax6.grid(grid_)
                                                ax6.tick_params(labelbottom=False, direction='inout', width=3, length=14 )
                                                ax6.minorticks_on()
                                                ax6.tick_params(axis='both', direction='inout', which='minor',length=8 ,width=2)
                                                if( plot_nsig==False and plot_ratio==False ):
                                                    ax6.set_xlabel(xlab[1][0], fontsize=font_siz+2)
                                                    ax6.tick_params(labelbottom=True, direction='inout', width=3, length=14 )

                                            ylabl_comp = r'$\frac{ ('+source1+' - '+source2+')}{\sigma}$'
                                            comp_obs, err_comp_obs = comp_2obs(wxs_bh1, wxs_bh2)
                                            comp_sim, err_comp_sim = comp_2sim(wys_bh1, wys_bh2)
                                            if np.isnan(comp_obs).all() and np.isnan(comp_sim).all():
                                                # Ambos arreglos son NaN
                                                delta_min = 0
                                                delta_max = 0
                                            elif np.isnan(comp_sim).all():
                                                # Solo comp_sim es NaN
                                                delta_min = np.nanmin(comp_obs)
                                                delta_max = np.nanmax(comp_obs[np.isfinite(comp_obs)])
                                            elif np.isnan(comp_obs).all():
                                                # Solo comp_obs es NaN
                                                delta_min = np.nanmin(comp_sim)
                                                delta_max = np.nanmax(comp_sim[np.isfinite(comp_sim)])
                                            else:
                                                # Ambos tienen valores válidos
                                                delta_min = np.nanmin([np.nanmin(comp_obs), np.nanmin(comp_sim)])
                                                delta_max = np.nanmax([
                                                    np.nanmax(comp_obs[np.isfinite(comp_obs)]),
                                                    np.nanmax(comp_sim[np.isfinite(comp_sim)])
                                                ])

                                            if(plt_obs==True and plot_nsig==True):
                                                #ax8.axhline(y = 0., color = 'k', linestyle = '--')
                                                ax8.hist(seedlinbins, weights=0*seedlinbins, bins = seedlinbins, histtype='step', ls='--', density=densi, color='k', linewidth=l_width*1 )
                                                ax8.errorbar(bincenters_obs1, comp_obs, yerr=plt_err*err_comp_obs, fmt='C1'+'>', lw=2, markersize=12 )
                                                if(zoom_delta==True):ax8.set_ylim(delta_min,delta_max)

                                            if(plt_sim==True and plot_nsig==True):
                                                #ax8.axhline(y = 0., color = 'k', linestyle = '--')
                                                ax8.hist(seedlinbins, weights=0*seedlinbins, bins = seedlinbins, histtype='step', ls='--', density=densi, color='k', linewidth=l_width*1 )
                                                ax8.errorbar(bincenters_obs1, comp_sim, yerr=plt_err*err_comp_sim, fmt='C0'+'o', lw=2, markersize=10 )
                                                if(zoom_delta==True):ax8.set_ylim(delta_min,delta_max)

                                            if( (plt_obs==True or plt_sim==True) and plot_nsig==True):
                                                #ax8.xticks(size=font_siz+2)
                                                #ax8.yticks(size=font_siz+2)
                                                ax8.tick_params(labelbottom=False, width=3,  direction='inout', length=14 )
                                                ax8.minorticks_on()
                                                ax8.tick_params(axis='x', direction='inout', which='minor',length=8 ,width=2)
                                                ax8.set_ylabel(ylabl_comp, fontsize=font_siz+2)
                                                if(plot_ratio==False):
                                                    ax8.set_xlabel(xlab[1][0], fontsize=font_siz+4)
                                                    ax8.tick_params(labelbottom=True, width=3,  direction='inout', length=14 )
                                                #ax8.yaxis.tick_right()
                                                #ax8.yaxis.set_label_position("right")
                                                if(grid_==True):
                                                    #ax8.grid( axis = 'x',  linestyle = '--',)
                                                    ax8.grid(grid_)

                                            ylabl_ratio= r'$ratio = \frac{'+source1+'}{'+source2+'}$'
                                            ratio_obs, err_ratio_obs = ratio_2obs(wxs_bh1, wxs_bh2 )
                                            ratio_sim, err_ratio_sim = ratio_2sim(wys_bh1, wys_bh2 )
                                            if np.isnan(ratio_obs).all() and np.isnan(ratio_sim).all():
                                                # Ambos arreglos son NaN
                                                ratio_min = 0
                                                ratio_max = 0
                                            elif np.isnan(ratio_sim).all():
                                                # Solo ratio_sim es NaN
                                                ratio_min = np.nanmin(ratio_obs)
                                                ratio_max = np.nanmax(ratio_obs[np.isfinite(ratio_obs)])
                                            elif np.isnan(ratio_obs).all():
                                                # Solo ratio_obs es NaN
                                                ratio_min = np.nanmin(ratio_sim)
                                                ratio_max = np.nanmax(ratio_sim[np.isfinite(ratio_sim)])
                                            else:
                                                # Ambos tienen valores válidos
                                                ratio_min = np.nanmin([np.nanmin(ratio_obs), np.nanmin(ratio_sim)])
                                                ratio_max = np.nanmax([
                                                    np.nanmax(ratio_obs[np.isfinite(ratio_obs)]),
                                                    np.nanmax(ratio_sim[np.isfinite(ratio_sim)])
                                                ])

                                            if(plt_obs==True and plot_ratio==True):
                                                #ax10.axhline(y = 0., color = 'k', linestyle = '--')
                                                ax10.hist(seedlinbins, weights=0*seedlinbins, bins = seedlinbins, histtype='step', ls='--', density=densi, color='k', linewidth=l_width*1 )
                                                ax10.errorbar(bincenters_obs1, ratio_obs, yerr=plt_err*err_ratio_obs, fmt='C1'+'>', lw=2, markersize=12 )
                                                if(zoom_ratio==True):ax10.set_ylim(-0.5+ratio_min,ratio_max)

                                            if(plt_sim==True and plot_ratio==True):
                                                #ax10.axhline(y = 0., color = 'k', linestyle = '--')
                                                ax10.hist(seedlinbins, weights=0*seedlinbins, bins = seedlinbins, histtype='step', ls='--', density=densi, color='k', linewidth=l_width*1 )
                                                ax10.errorbar(bincenters_obs1, ratio_sim, yerr=plt_err*err_ratio_sim, fmt='C0'+'o', lw=2, markersize=10 )
                                                if(zoom_ratio==True):ax10.set_ylim(-0.5+ratio_min,ratio_max)

                                            if( (plt_obs==True or plt_sim==True) and plot_ratio==True):
                                                ax10.tick_params(labelbottom=True, width=3,  direction='inout', length=14 )
                                                ax10.minorticks_on()
                                                ax10.tick_params(axis='x', direction='inout', which='minor',length=8 ,width=2)
                                                ax10.set_xlabel(xlab[1][0], fontsize=font_siz+4)
                                                ax10.set_ylabel(ylabl_ratio, fontsize=font_siz+2)
                                                #ax10.yaxis.tick_right()
                                                #ax10.yaxis.set_label_position("right")
                                                if(grid_==True):
                                                    #ax10.grid( axis = 'x',  linestyle = '--',)
                                                    ax10.grid(grid_)


                                            fig.tight_layout()
                                            save_subplt = 'comp_plot_hist'+dens+'_maxadc_'+strbin +str(z*2).zfill(2)+'mm'+ name_cuts
                                            print(save_subplt)
                                            # Save just the portion _inside_ the second axis's boundaries
                                            #extent =ax8.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                                            if(plt_one==True):
                                                plt.savefig(dirsave_plt+'/'+ save_subplt+'.pdf', dpi=150)#,  bbox_inches=extent.expanded(1.2, 1.3))
                                                plt.savefig(dirsave_plt+'/'+ save_subplt+'.png', dpi=150)#,  bbox_inches=extent.expanded(1.2, 1.3))
                                                #plt.show()

                                                #plt.clf()
                                                #plt.close()


                                        ####### E_TOTAL
                                        if(type_plt=='cl_etotal'):
                                            if(plt_one==True):
                                                if( plot_nsig==True and plot_ratio==True ):
                                                    fig, (ax7, ax9, ax11) = plt.subplots(3,1,figsize=(14.5,14),gridspec_kw={'height_ratios': [5, 1, 1] } )

                                                if( plot_nsig==True and plot_ratio==False ):
                                                    fig, (ax7, ax9)  = plt.subplots(2,1,figsize=(14.5,12),  gridspec_kw={'height_ratios': [5, 1]} )

                                                if( plot_nsig==False and plot_ratio==True ):
                                                    fig, (ax7, ax11) = plt.subplots(2,1,figsize=(14.5,12),  gridspec_kw={'height_ratios': [5, 1]} )

                                                if( plot_nsig==False and plot_ratio==False ):
                                                    fig, (ax7) = plt.subplots(figsize=(14.5,9.5) )#, sharex=True )


                                            xlabs = 'Tolal ADC'
                                            #r= np.max(np.array((np.max(data_cut_obs1[:,9]), np.max(data_cut_sim2[:,9]) )))
                                            #maxims = np.ceil(1.1*r)
                                            #maxie = np.min(np.array((np.max(data_cut_obs1[:,9]), np.max(data_cut_obs2[:,9]),
                                            #                        np.max(data_cut_sim1[:,9]), np.max(data_cut_sim2[:,9])  )))+1
                                            # Lista de arreglos que deseas evaluar
                                            arrays_e = [ data_cut_obs1[:, 9], data_cut_obs2[:, 9], data_cut_sim1[:, 9], data_cut_sim2[:, 9]]
                                            # Filtrar los arreglos no vacíos
                                            valid_max_values = [np.max(arr) for arr in arrays_e if arr.size > 0]
                                            # Verificar si hay valores válidos y calcular el máximo
                                            if valid_max_values:
                                                maxie = np.min(valid_max_values)+1
                                            else:
                                                maxie = 0

                                            #minie = np.min(np.array((np.min(data_cut_obs1[:,9]), np.min(data_cut_obs2[:,9]),
                                            #                        np.min(data_cut_sim1[:,9]), np.min(data_cut_sim12[:,9]) )))
                                            # Filtrar los arreglos no vacíos
                                            valid_min_values = [np.min(arr) for arr in arrays_e if arr.size > 0]
                                            # Verificar si hay valores válidos y calcular el máximo
                                            if valid_min_values:
                                                minie = np.min(valid_min_values)
                                            else:
                                                minie = 0

                                            #max_e = np.max(np.array((np.max(data_cut_obs1[:,9]), np.min(data_cut_obs2[:,9]),
                                            #                        np.max(data_cut_sim1[:,9]), np.max(data_cut_sim2[:,9])  )))
                                            # Filtrar los arreglos no vacíos
                                            valid_max_e_values = [np.max(arr) for arr in arrays_e if arr.size > 0]
                                            # Verificar si hay valores válidos y calcular el máximo
                                            if valid_max_e_values:
                                                max_e = np.max(valid_max_e_values)
                                            else:
                                                max_e = 0


                                            if( min_adc==adc_cut):data_cut = data_cut_obs1
                                            else:
                                                if data_cut_obs1[:, 9].size > 0:  # Verificar si el arreglo no está vacío
                                                    if np.max(data_cut_obs1[:, 9]) == max_e:
                                                        data_cut = data_cut_obs1
                                                if data_cut_obs2[:, 9].size > 0:  # Verificar si el arreglo no está vacío
                                                    if np.max(data_cut_obs2[:, 9]) == max_e:
                                                        data_cut = data_cut_obs2
                                                if data_cut_sim1[:, 9].size > 0:  # Verificar si el arreglo no está vacío
                                                    if np.max(data_cut_sim1[:, 9]) == max_e:
                                                        data_cut = data_cut_sim1
                                                if data_cut_sim2[:, 9].size > 0:  # Verificar si el arreglo no está vacío
                                                    if np.max(data_cut_sim2[:, 9]) == max_e:
                                                        data_cut = data_cut_sim2


                                            #sig_etotal = np.std(data_cut_obs1[:,9], ddof=1)
                                            #h_etotal = 1*sig_etotal/np.cbrt(data_cut_obs1[:,9].size)
                                            #nbins_etotal = np.int(np.ceil(np.max(data_cut_obs1[:,9])/h_etotal))

                                            nbins_etotal=int(round(maxie-minie ))
                                            #nbins_etotal = round(np.max(data_cut_obs1[:,9] ))
                                            if( min_adc==adc_cut):
                                                hist_etotal, bins_etotal = np.histogram(data_cut_obs1[:,9], bins=nbins_etotal)
                                            else:
                                                hist_etotal, bins_etotal = np.histogram(np.append(data_cut_obs1[:,9],minie), bins=nbins_etotal)
                                            etlogbins_all = np.logspace(np.log10(bins_etotal[0]),np.log10(bins_etotal[-1]),len(bins_etotal))

                                            wxe1,xe1 = np.histogram(data_cut_obs1[:,9], bins = etlogbins_all, density = densi)
                                            wxe2,xe2 = np.histogram(data_cut_obs2[:,9], bins = etlogbins_all, density = densi)
                                            wye1,ye1 = np.histogram(data_cut_sim1[:,9], bins = etlogbins_all, density = densi)
                                            wye2,ye2 = np.histogram(data_cut_sim2[:,9], bins = etlogbins_all, density = densi)

                                            #sbi=160 #3sigma

                                            #if(sigma_obs==3):sbi = 1# 20 # 40, 20 , 9
                                            #if(sigma_obs==3):sbi = 2 # adc <50 & cluster <4
                                            #if(sigma_obs==4):sbi = 2
                                            #if(sigma_obs==5):sbi = 1

                                            sbi = 1

                                           # sbi=40 #5sigma

                                            if(bin_hist>1):
                                                nbins_etotal = bin_hist
                                                if(bin_hist > round(maxie-minie ) ): nbins_etotal = round(maxie-minie )
                                                #if(bin_hist > round(np.max(data_cut_obs1[:,9] )) ): nbins_etotal = round(np.max(data_cut_obs1[:,9] ))
                                                #else: nbins_etotal = bin_hist
                                                nbins_etotal_sim = bin_hist
                                                if(bin_hist > round(maxie-minie ) ): nbins_etotal_sim = round(maxie-minie )
                                                #if(bin_hist > round(np.max(data_cut_sim2[:,9] )) ): nbins_etotal_sim = round(np.max(data_cut_sim2[:,9] ))

                                            else:
                                                nbins_etotal = round(maxie-minie )
                                                nbins_etotal_sim = round(maxie-minie )
                                                #nbins_etotal = round(np.max(data_cut_obs1[:,9]/sbi ))
                                                #nbins_etotal_sim = round(np.max(data_cut_sim2[:,9]/sbi ))

                                            if( min_adc==adc_cut):
                                                hist_etotal, bins_etotal = np.histogram(data_cut_obs1[:,9], bins=nbins_etotal)
                                            else:
                                                hist_etotal, bins_etotal = np.histogram(np.append(data_cut_obs1[:,9],minie), bins=nbins_etotal)
                                            etlogbins = np.logspace(np.log10(bins_etotal[0]),np.log10(bins_etotal[-1]),len(bins_etotal))

                                            hist_etotal_sim, bins_etotal_sim = np.histogram(data_cut_sim2[:,9], bins=nbins_etotal_sim)
                                            etlogbins_sim = np.logspace(np.log10(bins_etotal_sim[0]),np.log10(bins_etotal_sim[-1]),len(bins_etotal_sim))
                                            ###############################################################################################################################################

                                            etlogbin_obs = etlogbins
                                            etlogbin_sim = etlogbins
                                            etlogbins_bg = etlogbins
                                            ######################################
                                            wxe_bh1, xe_bh1 = np.histogram(data_cut_obs1[:,9], bins = etlogbins, density = densi)
                                            wxe_bh2, xe_bh2 = np.histogram(data_cut_obs2[:,9], bins = etlogbins, density = densi)
                                            wye_bh1, ye_bh1 = np.histogram(data_cut_sim1[:,9], bins = etlogbins, density = densi)
                                            wye_bh2, ye_bh2 = np.histogram(data_cut_sim2[:,9], bins = etlogbins, density = densi)

                                            #########################################################################################
                                            count_obs1, bin_obs1 = np.histogram(data_cut_obs1[:,9], bins=etlogbin_obs)
                                            hist_clstr_total_obs1 = {'bins': bin_obs1, 'counts': count_obs1}
                                            bincenters_obs1 = 0.5*(bin_obs1[1:]+bin_obs1[:-1])
                                            norm_obs1 = (np.sum(count_obs1) * np.diff(bin_obs1))

                                            count_obs2, bin_obs2 = np.histogram(data_cut_obs2[:,9], bins=etlogbin_obs)
                                            hist_clstr_total_obs2 = {'bins': bin_obs2, 'counts': count_obs2}
                                            bincenters_obs2 = 0.5*(bin_obs2[1:]+bin_obs2[:-1])
                                            norm_obs2 = (np.sum(count_obs2) * np.diff(bin_obs2))
                                            #########################################################################################
                                            count_sim1, bin_sim1 = np.histogram(data_cut_sim1[:,9], bins=etlogbin_sim)
                                            hist_clstr_total_sim1 = {'bins': bin_sim1, 'counts': count_sim1}
                                            bincenters_sim1 = 0.5*(bin_sim1[1:]+bin_sim1[:-1])
                                            norm_sim1 = (np.sum(count_sim1) * np.diff(bin_sim1))

                                            count_sim2, bin_sim2 = np.histogram(data_cut_sim2[:,9], bins=etlogbin_sim)
                                            hist_clstr_total_sim2 = {'bins': bin_sim2, 'counts': count_sim2}
                                            bincenters_sim2 = 0.5*(bin_sim2[1:]+bin_sim2[:-1])
                                            norm_sim2 = (np.sum(count_sim2) * np.diff(bin_sim2))
                                            #########################################################################################
                                            if(data_bg_cut.ndim>1):
                                                count_bg, bin_bg = np.histogram(data_bg_cut[:,9], bins=etlogbins_bg)
                                            if(data_bg_cut.ndim==1):
                                                count_bg, bin_bg = np.histogram(data_bg_cut[2], bins=etlogbins_bg)
                                            hist_clstr_total_bg = {'bins': bin_bg, 'counts': count_bg}
                                            #########################################################################################
                                            #########################################################################################
                                            # Guardar en un archivo de texto,  .npy
                                            np.save(dirsave_plt+'/'+'hist_clstr_total_obs_'+source1+\
                                                    sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist_clstr_total_obs1 )
                                            np.save(dirsave_plt+'/'+'hist_clstr_total_obs_'+source2+\
                                                    sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist_clstr_total_obs2 )

                                            np.save(dirsave_plt+'/'+'hist_clstr_total_sim_'+source1+\
                                                    sim_n+'_'+difu_gauss+ sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist_clstr_total_sim1 )
                                            np.save(dirsave_plt+'/'+'hist_clstr_total_sim_'+source2+\
                                                    sim_n+'_'+difu_gauss+ sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist_clstr_total_sim2 )

                                            np.save(dirsave_plt+'/'+'hist_clstr_total_bg_'+source_bg+\
                                                    sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist_clstr_total_bg )
                                            #########################################################################################

                                            if(densi==True):
                                                obs1_e = count_obs1/norm_obs1
                                                obs2_e = count_obs2/norm_obs2
                                                err_std_obs1_e = np.sqrt(count_obs1)/norm_obs1
                                                err_std_obs2_e = np.sqrt(count_obs2)/norm_obs2
                                                sim1_e = count_sim1/norm_sim1
                                                sim2_e = count_sim2/norm_sim2
                                                err_std_sim1_e = 0.2*count_sim1/norm_sim1#+ np.sqrt(count_sim1)
                                                err_std_sim2_e = 0.2*count_sim2/norm_sim2

                                            if(densi==False):
                                                obs1_e = count_obs1/norm_obs1
                                                obs2_e = count_obs2/norm_obs2
                                                err_std_obs1_e = np.sqrt(count_obs1)
                                                err_std_obs2_e = np.sqrt(count_obs2)
                                                sim1_e = count_sim1/norm_sim1
                                                sim2_e = count_sim2/norm_sim2
                                                err_std_sim1_e = 0.2*count_sim1
                                                err_std_sim2_e = 0.2*count_sim2 #+ np.sqrt(count_sim2)

                                            ###############################################################################################

                                            err_obs_total = np.sqrt(err_std_obs1_e*err_std_obs1_e + err_std_obs2_e*err_std_obs2_e )
                                            ch2_obs_total = chi_2_sigm_test(obs1_e, obs2_e, err_obs_total)
                                            tmp_obs_str = 'Chi2 test: ' + r'%.2f'%ch2_obs_total[0] \
                                                    + '       '+'ndf: ' + r'%.2f'%ch2_obs_total[3] \
                                                    + '\n'+'Chi2/ndf : ' + r'%.2f'%ch2_obs_total[4]

                                            err_sim_total = np.sqrt(err_std_sim1_e*err_std_sim1_e + err_std_sim2_e*err_std_sim2_e )
                                            ch2_sim_total = chi_2_sigm_test(sim1_e, sim2_e, err_sim_total)
                                            tmp_sim_str = 'Chi2 test: ' + r'%.2f'%ch2_sim_total[0] \
                                                    + '       '+'ndf: ' + r'%.2f'%ch2_sim_total[3] \
                                                    + '\n'+'Chi2/ndf : ' + r'%.2f'%ch2_sim_total[4]

                                            if(labl_opt=='full'):
                                                lbl_obs1 = source1 + ' data'+ sig_bg_thresh +cut_clst_size+cut_max + cut_adc
                                                lbl_obs2 = source2 + ' data'+ sig_bg_thresh +cut_clst_size+cut_max + cut_adc \
                                                            +'\n'+'Chi2/ndf : ' + r'%.2f'%ch2_obs_total[4]
                                                lbl_sim1 = source1 + ' sim '+d_gauss+ sig_bg_thresh_sim +cut_clst_size+cut_max + cut_adc
                                                lbl_sim2 = source2 + ' sim '+d_gauss+ sig_bg_thresh_sim +cut_clst_size+cut_max + cut_adc \
                                                            +'\n'+'Chi2/ndf : ' + r'%.2f'%ch2_sim_total[4]
                                                lbl_bg = 'Background' + ' data'+ sig_bg_thresh +cut_clst_size+cut_max + cut_adc
                                            if(labl_opt=='simple'):
                                                lbl_obs1 = source1 + ' data'
                                                lbl_obs2 = source2 + ' data'
                                                lbl_sim1 = source1 + ' simulation '
                                                lbl_sim2 = source2 + ' simulation '
                                                lbl_bg = 'Background' + ' data'
                                            if(labl_opt=='off'):
                                                lbl_obs1 = None
                                                lbl_obs2 = None
                                                lbl_sim1 = None
                                                lbl_sim2 = None
                                                lbl_bg = None

                                            ax7.set_xscale('log')

                                            if(c_dat_real==True and plt_obs==True ):
                                                ax7.hist(bincenters_obs1, weights=count_obs1, bins=etlogbin_obs, density=densi, histtype='step'
                                                            , label =  lbl_obs1 + cut_max_clst_porc
                                                            , color='C3', log=log_y, linewidth=l_width*1)

                                                ax7.hist(bincenters_obs2, weights=count_obs2, bins=etlogbin_obs, density=densi, histtype='step'
                                                            , label =  lbl_obs2 + cut_max_clst_porc
                                                            , color='C1', log=log_y, linewidth=l_width*1)

                                                if(densi==True):
                                                    ax7.errorbar(bincenters_obs1, obs1_e, yerr=err_std_obs1_e
                                                                                , fmt='C3'+'.', ecolor='C3', linewidth=l_width)
                                                    ax7.errorbar(bincenters_obs2, obs2_e, yerr=err_std_obs2_e
                                                                                , fmt='C1'+'.', ecolor='C1', linewidth=l_width)
                                                if(densi==False):
                                                    ax7.errorbar(bincenters_obs1, count_obs1, yerr=err_std_obs1_e
                                                                                , fmt='C3'+'.', ecolor='C3', linewidth=l_width)
                                                    ax7.errorbar(bincenters_obs2, count_obs2, yerr=err_std_obs2_e
                                                                                , fmt='C1'+'.', ecolor='C1', linewidth=l_width)

                                            if(c_dat_sim==True and plt_sim==True ):
                                                ##############################################################################################################################################################################################################################
                                                ax7.hist(bincenters_sim1, weights=count_sim1, bins=etlogbin_sim, density=densi, histtype='step'
                                                            , label = lbl_sim1 + cut_max_clst_porc_sim
                                                            , color='C0', log=log_y, linewidth=l_width*1)

                                                ax7.hist(bincenters_sim2, weights=count_sim2, bins=etlogbin_sim, density=densi, histtype='step'
                                                            , label = lbl_sim2 + cut_max_clst_porc_sim
                                                            , color='C7', log=log_y, linewidth=l_width*1)

                                                ##########################################################################
                                                if(densi==True):
                                                   ax7.errorbar(bincenters_sim1, sim1_e, yerr=err_std_sim1_e
                                                                                , fmt='C0'+'.', ecolor='C0', linewidth=l_width*1)
                                                   ax7.errorbar(bincenters_sim2, sim2_e, yerr=err_std_sim2_e
                                                                                , fmt='C7'+'.', ecolor='C7', linewidth=l_width*1)

                                                if(densi==False):
                                                   ax7.errorbar(bincenters_sim1, count_sim1, yerr= err_std_sim1_e
                                                                                , fmt='C0'+'.', ecolor='C0', linewidth=l_width*1) #,label = tmp_str)
                                                   ax7.errorbar(bincenters_sim2, count_sim2, yerr= err_std_sim2_e
                                                                                , fmt='C7'+'.', ecolor='C7', linewidth=l_width*1) #,label = tmp_str)
                                            ##############################################################################################################################################################################################################################                           ##############################################################################################################################################################################################################################

                                            if(c_bg_real==True and plt_bg==True ):

                                                if(data_bg_cut.ndim>1):
                                                    count_bg, bin_bg = np.histogram(data_bg_cut[:,9], bins=etlogbins_bg)
                                                    bincenters_bg = 0.5*(bin_bg[1:]+bin_bg[:-1])
                                                    ax7.hist(bincenters_bg, weights=count_bg, bins=etlogbins_bg, density=densi, histtype='step'
                                                               , label=lbl_bg
                                                               , color='k', log=log_y, linewidth=l_width*1)

                                                if(data_bg_cut.ndim==1):
                                                    count_bg, bin_bg = np.histogram(data_bg_cut[2], bins=etlogbins_bg)
                                                    bincenters_bg = 0.5*(bin_bg[1:]+bin_bg[:-1])
                                                    ax7.hist(bincenters_bg, weights=count_bg, bins=etlogbins_bg, density=densi, histtype='step'
                                                               , label=lbl_bg
                                                               , color='k', log=log_y, linewidth=l_width*1)

                                                bincenters_bg = 0.5*(bin_bg[1:]+bin_bg[:-1])
                                                norm_bg = (np.sum(count_bg) * np.diff(bin_bg))
                                                err_std_bg = np.sqrt(count_bg)/ norm_bg
                                                if(densi==True):
                                                    ax7.errorbar(bincenters_bg, count_bg/norm_bg, yerr=err_std_bg, fmt='k'+'.', ecolor='k')
                                                if(densi==False):
                                                    ax7.errorbar(bincenters_bg, count_bg, yerr=np.sqrt(count_bg), fmt='k'+'.', ecolor='k')


                                            if(plt_obs==True or plt_sim==True):
                                                #ax7.set_xlim(0,maxim[1,1]*mxs )
                                                #ax7.set_xlabel(xlab[1][1], fontsize=font_siz+2)
                                                ax7.set_ylabel('Clusters number', fontsize=font_siz+2)
                                                if(plt_one==True):ax7.legend()
                                                if(grid_==True):ax7.grid(grid_)
                                                ax7.tick_params(labelbottom=False, direction='inout', width=3, length=14 )
                                                ax7.minorticks_on()
                                                ax7.tick_params(axis='both', direction='inout', which='minor',length=8 ,width=2)
                                                if( plot_nsig==False and plot_ratio==False ):
                                                    ax7.set_xlabel(xlab[1][1], fontsize=font_siz+2)
                                                    ax7.tick_params(labelbottom=True, direction='inout', width=3, length=14 )

                                            ylabl_comp = r'$\frac{ ('+source1+' - '+source2+')}{\sigma}$'
                                            comp_obs, err_comp_obs = comp_2obs(wxe_bh1, wxe_bh2)
                                            comp_sim, err_comp_sim = comp_2sim(wye_bh1, wye_bh2)
                                            if np.isnan(comp_obs).all() and np.isnan(comp_sim).all():
                                                # Ambos arreglos son NaN
                                                delta_min = 0
                                                delta_max = 0
                                            elif np.isnan(comp_sim).all():
                                                # Solo comp_sim es NaN
                                                delta_min = np.nanmin(comp_obs)
                                                delta_max = np.nanmax(comp_obs[np.isfinite(comp_obs)])
                                            elif np.isnan(comp_obs).all():
                                                # Solo comp_obs es NaN
                                                delta_min = np.nanmin(comp_sim)
                                                delta_max = np.nanmax(comp_sim[np.isfinite(comp_sim)])
                                            else:
                                                # Ambos tienen valores válidos
                                                delta_min = np.nanmin([np.nanmin(comp_obs), np.nanmin(comp_sim)])
                                                delta_max = np.nanmax([
                                                    np.nanmax(comp_obs[np.isfinite(comp_obs)]),
                                                    np.nanmax(comp_sim[np.isfinite(comp_sim)])
                                                ])

                                            if(plt_obs==True and plot_nsig==True):
                                                ax9.set_xscale('log')
                                                #ax9.axhline(y = 0., color = 'k', linestyle = '--')
                                                ax9.hist(etlogbins, weights=0*etlogbins, bins = etlogbins, histtype='step', ls='--', density=densi, color='k', linewidth=l_width*1 )
                                                ax9.errorbar(bincenters_obs1, comp_obs, yerr=plt_err*err_comp_obs, fmt='C1'+'>', lw=2, markersize=12 )
                                                if(zoom_delta==True):ax9.set_ylim(delta_min,delta_max)

                                            if(plt_sim==True and plot_nsig==True):
                                                ax9.set_xscale('log')
                                                #ax9.axhline(y = 0., color = 'k', linestyle = '--')
                                                ax9.hist(etlogbins, weights=0*etlogbins, bins = etlogbins, histtype='step', ls='--', density=densi, color='k', linewidth=l_width*1 )
                                                ax9.errorbar(bincenters_obs1, comp_sim, yerr=plt_err*err_comp_sim, fmt='C0'+'o', lw=2, markersize=10 )
                                                if(zoom_delta==True):ax9.set_ylim(delta_min,delta_max)

                                            if( (plt_obs==True or plt_sim==True) and plot_nsig==True):
                                                ax9.tick_params(labelbottom=False, width=3,  direction='inout', length=14 )
                                                ax9.minorticks_on()
                                                ax9.tick_params(axis='x', direction='inout', which='minor',length=8 ,width=2)
                                                ax9.set_ylabel(ylabl_comp, fontsize=font_siz+2)
                                                if(plot_ratio==False):
                                                    ax9.set_xlabel(xlab[1][1], fontsize=font_siz+4)
                                                    ax9.tick_params(labelbottom=True, width=3,  direction='inout', length=14 )
                                                #ax9.yaxis.tick_right()
                                                #ax9.yaxis.set_label_position("right")
                                                if(grid_==True):
                                                    #ax9.grid( axis = 'x',  linestyle = '--',)
                                                    ax9.grid(grid_)

                                            ylabl_ratio= r'$ratio = \frac{'+source1+'}{'+source2+'}$'
                                            ratio_obs, err_ratio_obs = ratio_2obs(wxc_bh1, wxc_bh2 )
                                            ratio_sim, err_ratio_sim = ratio_2sim(wyc_bh1, wyc_bh2 )
                                            if np.isnan(ratio_obs).all() and np.isnan(ratio_sim).all():
                                                # Ambos arreglos son NaN
                                                ratio_min = 0
                                                ratio_max = 0
                                            elif np.isnan(ratio_sim).all():
                                                # Solo ratio_sim es NaN
                                                ratio_min = np.nanmin(ratio_obs)
                                                ratio_max = np.nanmax(ratio_obs[np.isfinite(ratio_obs)])
                                            elif np.isnan(ratio_obs).all():
                                                # Solo ratio_obs es NaN
                                                ratio_min = np.nanmin(ratio_sim)
                                                ratio_max = np.nanmax(ratio_sim[np.isfinite(ratio_sim)])
                                            else:
                                                # Ambos tienen valores válidos
                                                ratio_min = np.nanmin([np.nanmin(ratio_obs), np.nanmin(ratio_sim)])
                                                ratio_max = np.nanmax([
                                                    np.nanmax(ratio_obs[np.isfinite(ratio_obs)]),
                                                    np.nanmax(ratio_sim[np.isfinite(ratio_sim)])
                                                ])

                                            if(plt_obs==True and plot_ratio==True):
                                                ax11.set_xscale('log')
                                                ax11.axhline(y = 0., color = 'k', linestyle = '--')
                                                ax11.hist(etlogbins, weights=0*etlogbins, bins = etlogbins, histtype='step', ls='--', density=densi, color='k', linewidth=l_width*1 )
                                                ax11.errorbar(bincenters_obs1, ratio_obs, yerr=plt_err*err_ratio_obs, fmt='C1'+'>', lw=2, markersize=12 )
                                                if(zoom_ratio==True):ax11.set_ylim(-0.5+ratio_min,ratio_max)

                                            if(plt_sim==True and plot_ratio==True):
                                                ax11.set_xscale('log')
                                                #ax11.axhline(y = 0., color = 'k', linestyle = '--')
                                                ax11.hist(etlogbins, weights=0*etlogbins, bins = etlogbins, histtype='step', ls='--', density=densi, color='k', linewidth=l_width*1 )
                                                ax11.errorbar(bincenters_obs1, ratio_sim, yerr=plt_err*err_ratio_sim, fmt='C0'+'o', lw=2, markersize=10 )
                                                if(zoom_ratio==True):ax11.set_ylim(-0.5+ratio_min,ratio_max)

                                            if( (plt_obs==True or plt_sim==True) and plot_ratio==True):
                                                ax11.tick_params(labelbottom=True, width=3,  direction='inout', length=14 )
                                                ax11.minorticks_on()
                                                ax11.tick_params(axis='x', direction='inout', which='minor',length=8 ,width=2)
                                                ax11.set_xlabel(xlab[1][1], fontsize=font_siz+4)
                                                ax11.set_ylabel(ylabl_ratio, fontsize=font_siz+2)
                                                #ax11.yaxis.tick_right()
                                                #ax11.yaxis.set_label_position("right")
                                                if(grid_==True):
                                                    #ax11.grid( axis = 'x',  linestyle = '--',)
                                                    ax11.grid(grid_)



                                            #fig.tight_layout()
                                            save_subplt = 'comp_plot_hist'+dens+'_total_'+strbin +str(z*2).zfill(2)+'mm'+ name_cuts
                                            print(save_subplt)
                                            # Save just the portion _inside_ the second axis's boundaries
                                            #extent =ax9.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                                            if(plt_one==True):
                                                plt.savefig(dirsave_plt+'/'+ save_subplt+'.png', dpi=150)#,  bbox_inches=extent.expanded(1.2, 1.3))
                                                #plt.show()

                                                #plt.clf()
                                                #plt.close()

                                    save_subplt = 'plot_hists'+dens+'_all_clstrs_'+strbin +str(z*2).zfill(2)+'mm'+ name_cuts

                                    if(plt_one==False):
                                        #fig.suptitle(titulo + '_cut_clst'+cut_clst, fontsize=font_siz)
                                        fig.tight_layout()
                                        plt.savefig(dirsave_plt+'/'+ save_subplt+'.png', dpi=150)
                                        plt.savefig(dirsave_plt+'/'+ save_subplt+'.pdf', dpi=150)
                                        #plt.show()
                                        #plt.clf()
                                        #plt.close()

                            if(cut_dat00==cut_dat01):
                                fig1, (ht0, ht1, ht2) = plt.subplots(1,3,figsize=(43.5,14) )
                                for type_plt in ['cl_mean', 'cl_etotal', 'cl_max_adc' ]:
                                    if(type_plt=='cl_mean'):
                                        save_subplt = 'comp_plot_hist'+dens+'_mean_'+strbin +str(z*2).zfill(2)+'mm'+ name_cuts
                                        img = mpimg.imread(dirsave_plt+'/'+ save_subplt +'.png')
                                        ht0.imshow(img)
                                        ht0.axis('off')

                                    if(type_plt=='cl_max_adc'):
                                        save_subplt = 'comp_plot_hist'+dens+'_maxadc_'+strbin +str(z*2).zfill(2)+'mm'+ name_cuts
                                        img = mpimg.imread(dirsave_plt+'/'+ save_subplt +'.png')
                                        ht1.imshow(img)
                                        ht1.axis('off')

                                    if(type_plt=='cl_etotal'):
                                        save_subplt = 'comp_plot_hist'+dens+'_total_'+strbin +str(z*2).zfill(2)+'mm'+ name_cuts
                                        img = mpimg.imread(dirsave_plt+'/'+ save_subplt +'.png')
                                        ht2.imshow(img)
                                        ht2.axis('off')

                                fig1.tight_layout()
                                save_subplt = 'comp_plot_hist'+dens+'_all_clstrs'+cut_clst+strbin +str(z*2).zfill(2)+'mm'+ name_cuts
                                plt.savefig(dirsave_plt+'/'+ save_subplt+'MIKI.png', dpi=150)
                                #plt.show()
                            #####################################################################################################################################################################################################

                            lab2d = list(('Cluster Size', 'Maximum ADC'))
                            if(plots_2d==True):
                                wxy_sc_obs1, x_sc_obs1, y_sc_obs1 = np.histogram2d(data_cut_obs1[:,10], data_cut_obs1[:,7], bins=(seedlinbins, cltslinbins) )
                                hist2_max_clstr_obs1 = {'binx': x_sc_obs1, 'biny': y_sc_obs1, 'counts': wxy_sc_obs1}
                                np.save(dirsave_plt+'/'+'hist2_max_vs_clstr_obs_'+source1+\
                                sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist2_max_clstr_obs1 )
                                ##################################################################################
                                wxy_sc_obs2, x_sc_obs2, y_sc_obs2 = np.histogram2d(data_cut_obs2[:,10], data_cut_obs2[:,7], bins=(seedlinbins, cltslinbins) )
                                hist2_max_clstr_obs2 = {'binx': x_sc_obs2, 'biny': y_sc_obs2, 'counts': wxy_sc_obs2}
                                np.save(dirsave_plt+'/'+'hist2_max_vs_clstr_obs_'+source2+\
                                sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist2_max_clstr_obs2 )
                                ##################################################################################
                                ##################################################################################
                                wxy_sc_sim1, x_sc_sim1, y_sc_sim1 = np.histogram2d(data_cut_sim1[:,10], data_cut_sim1[:,7], bins=(seedlinbins, cltslinbins) )
                                hist2_max_clstr_sim1 = {'binx': x_sc_sim1, 'biny': y_sc_sim1, 'counts': wxy_sc_sim1}
                                np.save(dirsave_plt+'/'+'hist2_max_vs_clstr_sim_dg_'+source1+\
                                sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist2_max_clstr_sim1 )
                                ##################################################################################
                                wxy_sc_sim2, x_sc_sim2, y_sc_sim2 = np.histogram2d(data_cut_sim2[:,10], data_cut_sim2[:,7], bins=(seedlinbins, cltslinbins) )
                                hist2_max_clstr_sim2 = {'binx': x_sc_sim2, 'biny': y_sc_sim2, 'counts': wxy_sc_sim2}
                                np.save(dirsave_plt+'/'+'hist2_max_vs_clstr_sim_dg_'+source2+\
                                sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist2_max_clstr_sim2 )
                                ##################################################################################
                                ##################################################################################
                            ##########################
                            ##########################
                            print('\n\n\n\n')
                            #print('ADC =', backg[0].max(), bg[0].max(), ' cluster = ', data_bg_cut[:,7].max(), ' mean = ' , data_bg_cut[:,8].max(), ' max = ' , data_bg_cut[:,10].max(), ' total = ' , data_bg_cut[:,9].max())

                            if( file_csv==True):
                                file_name_test = dirsave_plt+'/comp_'+source1+'_'+source2+dens+'_'+str(z_count)+'l_'+strbin+str(z*2).zfill(2)+'mm'+'_clst'+cut_str+'_chi2'+str_hist_comp+'_'+str(min_adc)+'_'+str(max_adc)+'ADC'+'_'+str(min_adc)+'_'+str(max_adc)+'ADC'+'.csv'
                                # np.savetxt(file_test, np.vstack(('Chi_sq2_test', 'hist_ADC', prc_r2, lower, bin_hist, test_corr[0], test_corr[1], test_corr[2], test_corr[3], test_corr[4] )).T, delimiter='\t', newline='\n',fmt='%s' ,  header='test bins, histo, proc, eta, bins, chi_2, pvalue, criti_val, ndf, chi2/nf')
                                print('\n\n')

                                file_test = open(file_name_test,'w')


                                ######################################################################################################################################################################################################

                                print('\n')
                                txt_test = 'test bins'+sig_bg_thresh + cut_clst_size+cut_max + cut_adc + cut_max_clst_porc+'\n'+sig_bg_thresh_sim+cut_max_clst_porc_sim+'\t' + \
                                            'histo\t' + 'proc\t' + 'eta\t' + 'bins\t' + 'chi_2\t' + 'pvalue\t' + 'criti_val\t' + 'ndf\t' + 'chi2/nf'
                                file_test.write('\n\n'+ txt_test + '\n')
                                print(txt_test)

                                #err_ch = np.sqrt(x_w1 + x_w2 )
                                err_ch = np.sqrt(err_x1*err_x1 + err_x2*err_x2 )
                                test_corr = chi_2_sigm_test(x_w1, x_w2, err_ch)
                                txt_test = 'Chi2_sig_test_Obs'+'\t'+'hist_ADC'+'\t'+ str(bin_hist) +'\t'+ str(test_corr[0]) +'\t'+ str(test_corr[1]) +'\t'+ str(test_corr[2]) +'\t'+ str(test_corr[3]) +'\t'+ str(test_corr[4])
                                file_test.write(txt_test + '\n')
                                print(txt_test)

                                #err_ch = np.sqrt(wxc_bh1 + wxc_bh2 )
                                err_ch = np.sqrt(err_std_obs1_c*err_std_obs1_c + err_std_obs2_c*err_std_obs2_c )
                                #test_corr = chi_2_sigm_test(wxc_bh1, wxc_bh2, err_ch)
                                test_corr = chi_2_sigm_test(obs1_c, obs2_c, err_ch)
                                txt_test = 'Chi2_sig_test_Obs'+'\t'+'hist_Cluster_size'+'\t'+ str(bin_hist) +'\t'+ str(test_corr[0]) +'\t'+ str(test_corr[1]) +'\t'+ str(test_corr[2]) +'\t'+ str(test_corr[3]) +'\t'+ str(test_corr[4])
                                file_test.write(txt_test + '\n')
                                print(txt_test)

                                #err_ch = np.sqrt(wxm_bh1 + wxm_bh2 )
                                err_ch = np.sqrt(err_std_obs1_m*err_std_obs1_m + err_std_obs2_m*err_std_obs2_m )
                                #test_corr = chi_2_sigm_test(wxm_bh1, wxm_bh2, err_ch)
                                test_corr = chi_2_sigm_test(obs1_m, obs2_m, err_ch)
                                txt_test = 'Chi2_sig_test_Obs'+'\t'+'hist_Mean_clst'+'\t'+ str(bin_hist) +'\t'+ str(test_corr[0]) +'\t'+ str(test_corr[1]) +'\t'+ str(test_corr[2]) +'\t'+ str(test_corr[3]) +'\t'+ str(test_corr[4])
                                file_test.write(txt_test + '\n')
                                print(txt_test)

                                #err_ch = np.sqrt(wxe_bh1 + wxe_bh2 )
                                err_ch = np.sqrt(err_std_obs1_e*err_std_obs1_e + err_std_obs2_e*err_std_obs2_e )
                                #test_corr = chi_2_sigm_test(wxe_bh1, wxe_bh2, err_ch)
                                test_corr = chi_2_sigm_test(obs1_e, obs2_e, err_ch)
                                txt_test = 'Chi2_sig_test_Obs'+'\t'+'hist_Total_clst'+'\t'+ str(bin_hist) +'\t'+ str(test_corr[0]) +'\t'+ str(test_corr[1]) +'\t'+ str(test_corr[2]) +'\t'+ str(test_corr[3]) +'\t'+ str(test_corr[4])
                                file_test.write(txt_test + '\n')
                                print(txt_test)

                                #err_ch = np.sqrt(wxs_bh1 + wxs_bh2 )
                                err_ch = np.sqrt(err_std_obs1_s*err_std_obs1_s + err_std_obs2_s*err_std_obs2_s )
                                test_corr = chi_2_sigm_test(obs1_s, obs2_s, err_ch)
                                txt_test = 'Chi2_sig_test_Obs'+'\t'+'hist_Max_clst'+'\t'+ str(bin_hist) +'\t'+ str(test_corr[0]) +'\t'+ str(test_corr[1]) +'\t'+ str(test_corr[2]) +'\t'+ str(test_corr[3]) +'\t'+ str(test_corr[4])
                                file_test.write(txt_test + '\n')
                                print(txt_test)
                                ######################################################################################################################################################################################################
                                print('\n')
                                txt_test = 'test bins\t' + 'histo\t' + 'proc\t' + 'eta\t' + 'bins\t' + 'chi_2\t' + 'pvalue\t' + 'criti_val\t' + 'ndf\t' + 'chi2/nf'
                                file_test.write('\n\n'+ txt_test + '\n')
                                print(txt_test)

                                #err_ch = 0.2*np.sqrt(y_w1*y_w1 + y_w2*y_w2)
                                err_ch =  np.sqrt(err_ysim1*err_ysim1 + err_ysim2*err_ysim2 )
                                test_corr = chi_2_sigm_test(y_w1, y_w2, err_ch)
                                txt_test = 'Chi2_sig_test_Sim_'+d_gauss +'\t'+'hist_ADC'+'\t'+ str(0) +'\t'+ str(0) +'\t'+ str(bin_hist) +'\t'+ str(test_corr[0]) +'\t'+ str(test_corr[1]) +'\t'+ str(test_corr[2]) +'\t'+ str(test_corr[3]) +'\t'+ str(test_corr[4])
                                file_test.write(txt_test + '\n')
                                print(txt_test)

                                #err_ch = 0.2*np.sqrt(wyc_bh1*wyc_bh1 + wyc_bh2*wyc_bh2 )
                                err_ch = np.sqrt(err_std_sim1_c*err_std_sim1_c + err_std_sim2_c*err_std_sim2_c )
                                #test_corr = chi_2_sigm_test(wyc_bh1, wyc_bh2, err_ch)
                                test_corr = chi_2_sigm_test(sim1_c, sim2_c, err_ch)
                                txt_test = 'Chi2_sig_test_Sim_'+d_gauss +'\t'+'hist_Cluster_size'+'\t'+ str(0) +'\t'+ str(0) +'\t'+ str(bin_hist) +'\t'+ str(test_corr[0]) +'\t'+ str(test_corr[1]) +'\t'+ str(test_corr[2]) +'\t'+ str(test_corr[3]) +'\t'+ str(test_corr[4])
                                file_test.write(txt_test + '\n')
                                print(txt_test)

                                #err_ch = 0.2*np.sqrt(wym_bh1*wym_bh1 + wym_bh2*wym_bh2 )
                                err_ch = np.sqrt(err_std_sim1_m*err_std_sim1_m + err_std_sim2_m*err_std_sim2_m )
                                #test_corr = chi_2_sigm_test(wym_bh1, wym_bh2, err_ch)
                                test_corr = chi_2_sigm_test(sim1_m, sim2_m, err_ch)
                                txt_test = 'Chi2_sig_test_Sim_'+d_gauss +'\t'+'hist_Mean_clst'+'\t'+ str(0) +'\t'+ str(0) +'\t'+ str(bin_hist) +'\t'+ str(test_corr[0]) +'\t'+ str(test_corr[1]) +'\t'+ str(test_corr[2]) +'\t'+ str(test_corr[3]) +'\t'+ str(test_corr[4])
                                file_test.write(txt_test + '\n')
                                print(txt_test)

                                #err_ch = 0.2*np.sqrt(wye_bh1*wye_bh1 + wye_bh2*wye_bh2 )
                                err_ch = np.sqrt(err_std_sim1_e*err_std_sim1_e + err_std_sim2_e*err_std_sim2_e )
                                test_corr = chi_2_sigm_test(sim1_e, sim2_e, err_ch)
                                txt_test = 'Chi2_sig_test_Sim_'+d_gauss +'\t'+'hist_Total_clst'+'\t'+ str(0) +'\t'+ str(0) +'\t'+ str(bin_hist) +'\t'+ str(test_corr[0]) +'\t'+ str(test_corr[1]) +'\t'+ str(test_corr[2]) +'\t'+ str(test_corr[3]) +'\t'+ str(test_corr[4])
                                file_test.write(txt_test + '\n')
                                print(txt_test)

                                #err_ch = 0.2*np.sqrt(wys_bh1*wys_bh1 + wys_bh2*wys_bh2 )
                                err_ch = np.sqrt(err_std_sim1_s*err_std_sim1_s + err_std_sim2_s*err_std_sim2_s )
                                test_corr = chi_2_sigm_test(sim1_s, sim2_s, err_ch)
                                txt_test = 'Chi2_sig_test_Sim_'+d_gauss +'\t'+'hist_Max_clst'+'\t'+ str(0) +'\t'+ str(0) +'\t'+ str(bin_hist) +'\t'+ str(test_corr[0]) +'\t'+ str(test_corr[1]) +'\t'+ str(test_corr[2]) +'\t'+ str(test_corr[3]) +'\t'+ str(test_corr[4])
                                file_test.write(txt_test + '\n')
                                print(txt_test)

                                #####################################################################################################################################################################################################
                                print('\n')
                                txt_test = 'test_bins'+dens+'\t' + 'histo\t' + 'bins\t' + 'KStes\t' + 'pvalue\t'+ 'signif_level\t' + 'response'
                                file_test.write('\n\n'+ txt_test + '\n')
                                print(txt_test)

                                test_corr = ks_test(x_w1, x_w2)
                                txt_test = 'Kolmogorov_Obs'+'\t'+'hist_ADC'+'\t'+ str(bin_hist) +'\t'+ str(test_corr[0]) +'\t'+ str(test_corr[1]) +'\t'+ str(test_corr[2]) +'\t'+ str(test_corr[3])
                                file_test.write(txt_test + '\n')
                                print(txt_test)

                                test_corr = ks_test(obs1_c, obs2_c)
                                txt_test = 'Kolmogorov_Obs'+'\t'+'hist_Cluster_size'+'\t'+ str(bin_hist) +'\t'+ str(test_corr[0]) +'\t'+ str(test_corr[1]) +'\t'+ str(test_corr[2]) +'\t'+ str(test_corr[3])
                                file_test.write(txt_test + '\n')
                                print(txt_test)

                                test_corr = ks_test(obs1_m, obs2_m)
                                txt_test = 'Kolmogorov_Obs'+'\t'+'hist_Mean_clst'+'\t'+ str(bin_hist) +'\t'+ str(test_corr[0]) +'\t'+ str(test_corr[1]) +'\t'+ str(test_corr[2]) +'\t'+ str(test_corr[3])
                                file_test.write(txt_test + '\n')
                                print(txt_test)

                                test_corr = ks_test(obs1_e, obs2_e)
                                txt_test = 'Kolmogorov_Obs'+'\t'+'hist_Total_clst'+'\t'+ str(bin_hist) +'\t'+ str(test_corr[0]) +'\t'+ str(test_corr[1]) +'\t'+ str(test_corr[2]) +'\t'+ str(test_corr[3])
                                file_test.write(txt_test + '\n')
                                print(txt_test)

                                test_corr = ks_test(obs1_s, obs2_s)
                                txt_test = 'Kolmogorov_Obs'+'\t'+'hist_Max_clst'+'\t'+ str(bin_hist) +'\t'+ str(test_corr[0]) +'\t'+ str(test_corr[1]) +'\t'+ str(test_corr[2]) +'\t'+ str(test_corr[3])
                                file_test.write(txt_test + '\n')
                                print(txt_test)

                                ######################################################################
                                print('\n')
                                txt_test = 'test_bins'+dens+'\t' + 'histo\t' + 'bins\t' + 'KStes\t' + 'pvalue\t'+ 'signif_level\t' + 'response'
                                file_test.write('\n\n'+ txt_test + '\n')
                                print(txt_test)

                                test_corr = ks_test(y_w1, y_w2)
                                txt_test = 'Kolmogorov_Sim_'+d_gauss +'\t'+'hist_ADC'+'\t'+ str(bin_hist) +'\t'+ str(test_corr[0]) +'\t'+ str(test_corr[1]) +'\t'+ str(test_corr[2]) +'\t'+ str(test_corr[3])
                                file_test.write(txt_test + '\n')
                                print(txt_test)

                                test_corr = ks_test(sim1_c, sim2_c)
                                txt_test = 'Kolmogorov_Sim_'+d_gauss +'\t'+'hist_Cluster_size'+'\t'+ str(bin_hist) +'\t'+ str(test_corr[0]) +'\t'+ str(test_corr[1]) +'\t'+ str(test_corr[2]) +'\t'+ str(test_corr[3])
                                file_test.write(txt_test + '\n')
                                print(txt_test)

                                test_corr = ks_test(sim1_m, sim2_m)
                                txt_test = 'Kolmogorov_Sim_'+d_gauss +'\t'+'hist_Mean_clst'+'\t'+ str(bin_hist) +'\t'+ str(test_corr[0]) +'\t'+ str(test_corr[1]) +'\t'+ str(test_corr[2]) +'\t'+ str(test_corr[3])
                                file_test.write(txt_test + '\n')
                                print(txt_test)

                                test_corr = ks_test(sim1_e, sim2_e)
                                txt_test = 'Kolmogorov_Sim_'+d_gauss +'\t'+'hist_Total_clst'+'\t'+ str(bin_hist) +'\t'+ str(test_corr[0]) +'\t'+ str(test_corr[1]) +'\t'+ str(test_corr[2]) +'\t'+ str(test_corr[3])
                                file_test.write(txt_test + '\n')
                                print(txt_test)

                                test_corr = ks_test(sim1_s, sim2_s)
                                txt_test = 'Kolmogorov_Sim_'+d_gauss +'\t'+'hist_Max_clst'+'\t'+ str(bin_hist) +'\t'+ str(test_corr[0]) +'\t'+ str(test_corr[1]) +'\t'+ str(test_corr[2]) +'\t'+ str(test_corr[3])
                                file_test.write(txt_test + '\n')
                                print(txt_test)

                                ######################################################################################################################################################################################################
                                ######################################################################################################################################################################################################

                                file_test.close()

                                print('\n\n\n\n')
                                #print('cluster = ', data_bg_cut[:,7].max(), ' mean = ' , data_bg_cut[:,8].max(), ' max = ' , data_bg_cut[:,10].max(), ' total = ' , data_bg_cut[:,9].max())

                            else:
                                print('Folder Not Found:', file_cluster_size_mean_sim2)
                    #%reset -f
