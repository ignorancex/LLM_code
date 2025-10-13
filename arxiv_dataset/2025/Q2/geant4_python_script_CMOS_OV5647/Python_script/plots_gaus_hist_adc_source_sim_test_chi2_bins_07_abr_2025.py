# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 22:16:33 2023

@author: mikb
"""

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

        data_cut_obs0, cut_str0 = cut_data(cut0, cut_dat00, cut_dat01, cut_type0, dat_obs, '')
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
    return comp, err_comp


def comp_2obs(obs1, obs2):
    dif = obs1 - obs2
    err = np.sqrt(obs1 + obs2)
    comp = np.divide(dif, err , where=(err != 0))
    err_comp = comp*np.sqrt( (err*err) / (dif*dif)
            + (1) / (4*err*err) )
    return comp, err_comp


def comp_2sim(sim1, sim2):
    dif = sim1 - sim2
    err = 0.2*np.sqrt(sim1*sim1 + sim2*sim2 )
    comp = np.divide(dif, err , where=(err != 0))
    err_comp = 0.2*comp*np.sqrt( (err*err) / (dif*dif)
            + (sim1*sim1*sim1*sim1 + sim2*sim2*sim2*sim2) / (4*err*err*err*err) )
    return comp, err_comp

def ratio_obs_sim(dat1, dat2):
    # Set ratio where dat2 is not zero
    ratio = np.divide(dat1, dat2 , where=(dat2 != 0))
    # Compute error on ratio (null if cannot be computed)
    er_ratio = ratio*np.sqrt( (1)/(dat1) + 0.04 )

    return ratio, er_ratio

def ratio_2obs(obs1, obs2):
    # Set ratio where obs2 is not zero
    ratio = np.divide(obs1, obs2 , where=(obs2 != 0))
    # Compute error on ratio (null if cannot be computed)
    er_ratio = ratio*np.sqrt( (1)/(obs1) + (1)/(obs2) )

    return ratio, er_ratio

def ratio_2sim(sim1, sim2):
    # Set ratio where obs2 is not zero
    ratio = np.divide(sim1, sim2 , where=(sim2 != 0))
    # Compute error on ratio (null if cannot be computed)
    er_ratio = ratio*np.sqrt( 0.08)

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

strE= 'eh_'+str(pair_eh)+'eV'
print(strE)

ag ='8'

satu =''

# Conditions for energy deposition
condit_edep = ''  # '', '_dif0', '_zero' or other conditions can be added here
#condit_edep = '_dif0'

###############################################################

plot_nsig = 0
zoom_delta=0

plot_ratio = 0
zoom_ratio = 0

plt_sim = 1
plt_sim1 = 0

plt_obs = 1
plt_bg = 1
plt_err=1

labl_opt = 'simple' #'full' , 'simple', 'off'

log_y = 1
label_xy = True
grid_ = False

file_csv = 1
file_clst_sum_csv = 0
file_total_adc_csv = 1

if(plt_sim==False and plt_sim1==True): row = 2 ; str_hist_comp = '_Sim_bg'
if(plt_sim==True and plt_sim1==False): row = 2; str_hist_comp = '_Sim'
if(plt_sim==True and plt_sim1==True): row = 3; str_hist_comp = '_All'
if(plt_sim==False and plt_sim1==False): row = 1 ; str_hist_comp = '_Obs'; label_xy=False
########################################################################################
maxim = np.zeros((2,2),dtype=int)

l_width = 4
font_siz=24
###############################################################

tiemp = '_500ms'

source = 'Sr90'
source = 'Cs137'


ssd_obs = '/home/mbonnett/mik/data_2023/'
ssd_sim = '/home/mbonnett/mik/dat_sim_2024/'

pname_obs = ssd_obs + 'data_obs_2023/'

drive_file_obs = ssd_obs+'data_obs_2023/'
drive_file_sim = ssd_sim+'g4_'+source+'_2024/'


yscal_adc = 10e5
yscal_cls = 5e5
yscal_max = 10e5

yscal_adc = 0
yscal_cls = 0
yscal_max = 0


adc_cut = 0

nsigm_self = 0

sim_bg = '_ag8_bg'
sim_bg = ''



source_bg = 'backgnd'
fecha_bg = '2023_Oct_22_00h'
nfrm =1000
nframes = 1000  # Number of frames
z_count = 1  # Count of Z levels

level_z = list(range(z_count))
#level_z = list(range(9))
#level_z = [0,2,6,12]
#level_z = list(range(z_count))

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
nframes = 100  # Number of frames
z_count = 11  # Count of Z levels

level_z = list(range(z_count))
#level_z = list(range(9))
#level_z = [0,2,6,12]
#level_z = list(range(z_count))

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

########################################################################################
bin_hist = 923
if(bin_hist>0): strbin = str(bin_hist)+'bin_z_'
else:strbin = 'z_'

max_adc=1024
#max_adc=150
min_adc=100

size_thresh = 0
max_thresh = 0
#if(adc_cut == 0):min_adc=1

if(max_adc<1024):cut_adcmax_str = "_mx"+str(max_adc)
else: cut_adcmax_str = ''

plots_2d=0
#log_y = 1

simu_no_cut ='_sim_nocut'
simu_no_cut =''

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
        # True
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
                         name_cut = '_'+str(z_count)+'l_'+fecha+'_'+evt+sim_bg + filt_cut
                         cuts_dat = name_cut+ cut_clst+cut_type0+cut_type1+cut_type2+cut_clmaxadc+cut_type3


                         sigma_sim = nsigm_bg_sim
                         sigma_obs = nsigm_bg

                         pname_bg = pname_obs + 'data_'+fecha_bg+'_'+ag+'ag'+'_'+source_bg+'_'+str(1)+'level'+'/'
                         name_filter_bg = avrg_cut + sig_bg_thresh + cut_adc + cut_max_clst_porc
                         dirname_bg ='adc_count_'+ source_bg+'_f'+str(nframes)+'_iso0_br50_ag'+ag+'_ad1' + name_filter_bg

                         backg = np.load(pname_bg+source_bg +'/'+dirname_bg+'/' + source_bg + name_filter_bg + '_z'+str(0).zfill(3)+'.npz')
                         backg = np.int64(backg.f.arr_0)
                         backg_mean = DescrStatsW(backg[0,0:backg[0].max()+2], weights=backg[1,0:backg[0].max()+2], ddof=0).mean
                         backg_std  = DescrStatsW(backg[0,0:backg[0].max()+2], weights=backg[1,0:backg[0].max()+2], ddof=0).std
                         print('backg_mean = ', backg_mean, 'backg_std = ', backg_std)
                         #############################################################################################################################################################
                         #############################################################################################################################################################
                         total_adc_level = np.zeros((26, z_count))
                         
                         for z in level_z:

                            dirsavepatn_plt = ssd_sim +gauss_dist+'_'+source +'_rslhist_ADC_'+strbin +str(z*2).zfill(2)+'mm'+ cut_clst_size+cut_max+str_cuts+'/'
                            dirsave_plt = dirsavepatn_plt + 'plots2'+sim_n+'_'+source+'_'+str(z_count)+'l'+'_'+str(nframes)+'f'+simu_no_cut+condit_edep+difu_gauss+'/plot_hist_mult_'+source + '_'+str(z_count)+'l_'+fecha+'_'+evt+sim_bg  #+ cuts_dat +simu_no_cut

                            try:
                                os.makedirs(dirsave_plt)
                            except FileExistsError:
                                pass


                            if( file_csv==True and i_cont == 0):

                                chi2_reduc_distr_test = dirsavepatn_plt+'All_'+source + sim_n +dens+'_'+strbin+str(z_count)+'l_'+fecha+'_'+evt+'_'+str(z*2).zfill(2)+'mm'+'sim_clst'+ simu_no_cut+ str_cuts+'_'+str(max_adc)+'ADC'+'_chi2.csv'
                            i_cont+=1


                            path_dir_obs = pname_obs + 'data_'+fecha+'_'+ag+'ag'+'_'+source+'_'+str(z_count)+'level'+'/'
                            name_filter_obs = avrg_cut + sig_bg_thresh + cut_adc + cut_max_clst_porc
                            dirname0 = 'adc_count_'+ source+'_f'+str(nframes)+'_iso0_br50_ag'+ag+'_ad1' + name_filter_obs
                            #data_obs = np.load(pname_obs+source +'/'+dirname0+'/data_' +source+'_frame'+str(nframe)+'_iso0_br50_ag'+ag+'_ad1_'+fecha+'_z'+str(2*z).zfill(3)+'.npz')
                            data_obs = np.load(path_dir_obs+source +'/'+dirname0+'/'+ source + name_filter_obs + '_z'+str(z).zfill(3)+'.npz')
                            data_obs = np.int64(data_obs.f.arr_0)

                            #threshold = 0 #filter threshold
                            data_obsr = data_obs
                            background = backg
                           ################################################################################################################

                            pname_sim = drive_file_sim + source
                            pname_sim0 = drive_file_sim + source
                            name_filter_sim0 = sig_bg_thresh_sim + '_'+gauss_dist+'_0k_0a_0zff'  + cut_adc + cut_max_clst_porc_sim
                            name_filter_sim1 = sig_bg_thresh_sim +condit_edep+difu_gauss + cut_adc + cut_max_clst_porc_sim
                            #####################################################################################################################################################
                            name_sim_0 = 'sim_'+source+'_'+str(nframes)+'f_'+evt+'_0' +'_'+satu+strE + name_filter_sim0 + sim_bg
                            dir_sim_0 =  pname_sim0+'/'+'adc_count_'+ name_sim_0

                            name_sim_1 = 'sim_'+source+'_'+str(nframes)+'f_'+evt+sim_n +'_'+satu+strE + name_filter_sim1 + sim_bg
                            dir_sim_1 = pname_sim+'/'+'adc_count_'+ name_sim_1
                            #####################################################################################################################################################
                            #####################################################################################################################################################
                            data_cnt_sim_0 = np.load(dir_sim_0+'/'+source+'_ev'+ev + name_filter_sim0 +'_z'+str(2*z).zfill(3)+'.npz' )

                            data_cnt_sim_1 = np.load(dir_sim_1+'/'+source+'_ev'+ev + name_filter_sim1 +'_z'+str(2*z).zfill(3)+'.npz' )
                            ################################################################################################################################################

                            ################################################################################################################################################
                            data_cnt_sim_0 = np.int64(data_cnt_sim_0.f.arr_0)
                            data_cnt_sim_1 = np.int64(data_cnt_sim_1.f.arr_0)
                            ################################################################################################################################################
                            ################################################################################################################################################

                            if(densi==True): histo='norm_'
                            if(densi!=True): histo=''

                            #############################################################################################################################################################

                            x0=data_obsr;
                            y0_sim=data_cnt_sim_0;
                            y1_sim=data_cnt_sim_1;
                            bg0=background

                            nbins_adc = round(np.max(x0[0]))
                            hist_adc, bins_adc = np.histogram(x0[0], bins=nbins_adc)
                            adclinbins = np.linspace( bins_adc[0], bins_adc[-1] ,len(bins_adc))

                            wxadc_all, xadc_all = np.histogram(x0[0], weights=x0[1], bins = adclinbins, density = densi)
                            wyadc_all0, yadc_all0 = np.histogram(y0_sim[0], weights=y0_sim[1], bins = adclinbins, density = densi)
                            wyadc_all1, yadc_all1 = np.histogram(y1_sim[0], weights=y1_sim[1], bins = adclinbins, density = densi)

                            x= x0[:,min_adc:max_adc]
                            ysim0= y0_sim[:,min_adc:max_adc]
                            ysim1= y1_sim[:,min_adc:max_adc]
                            #######################################################
                            bg= bg0[:,min_adc:max_adc]

                            #############################################################################################################################################################
                            if( file_total_adc_csv==True and densi==False and z==0):
                                file_total_adc = dirsave_plt+'/'+source+'_'+str(z_count)+'l_'+fecha+'_'+evt +'_sim_s2_total_ADC'+'_'+str(min_adc)+'_'+str(max_adc)+'ADC'+'_dist'
                                print('\n')
                                fl_total_adc = open(file_total_adc+'.csv','w')
                                fl_total_adc.write("dist(mm)\t ADC_obs_total\t ADC_obs_mean\t ADC_obs_std_1\t s2_obs\t ADC_obs_err_2\t" +
                                                "ADC_sim_total\t ADC_sim_mean\t ADC_sim_std_1\t s2_sim\t ADC_sim_err_2\t" +
                                                "ADC_sim1_total\t ADC_sim1_mean\t ADC_sim1_std_1\t s2_sim1\t ADC_sim1_err_2\n")
                                fl_total_adc.close()

                            if( file_total_adc_csv==True and densi==False ):
                                adc_dist_obs = x[:1]*x[1:]
                                total_adc_level[0][z]=z*2
                                total_adc_level[1][z]=np.sum(adc_dist_obs)
                                total_adc_level[2][z]=np.mean(adc_dist_obs)
                                total_adc_level[3][z]=np.std(adc_dist_obs, ddof=1)
                                total_adc_level[4][z]=np.sum(adc_dist_obs*adc_dist_obs)  #np.size(adc_dist_obs)
                                total_adc_level[5][z]=np.std(adc_dist_obs, ddof=1)/np.sqrt(np.size(adc_dist_obs))

                                adc_dist_sim_0 = ysim0[:1]*ysim0[1:]
                                total_adc_level[6][z]=np.sum(adc_dist_sim_0)
                                total_adc_level[7][z]=np.mean(adc_dist_sim_0)
                                total_adc_level[8][z]=np.std(adc_dist_sim_0, ddof=1)
                                total_adc_level[9][z]=np.sum(adc_dist_sim_0*adc_dist_sim_0)
                                total_adc_level[10][z]=np.std(adc_dist_sim_0, ddof=1)/np.sqrt(np.size(adc_dist_sim_0))

                                adc_dist_sim = ysim1[:1]*ysim1[1:]
                                total_adc_level[11][z]=np.sum(adc_dist_sim)
                                total_adc_level[12][z]=np.mean(adc_dist_sim)
                                total_adc_level[13][z]=np.std(adc_dist_sim, ddof=1)
                                total_adc_level[14][z]=np.sum(adc_dist_sim*adc_dist_sim)
                                total_adc_level[15][z]=np.std(adc_dist_sim, ddof=1)/np.sqrt(np.size(adc_dist_sim))

                                fl_total_adc = open(file_total_adc+'.csv','a')
                                fl_total_adc.write(str(z*2)+'\t'+str(total_adc_level[1][z])+'\t'+str(total_adc_level[2][z])+'\t'+str(total_adc_level[3][z])+'\t'+str(total_adc_level[4][z])+'\t'+str(total_adc_level[5][z])+'\t'
                                                 +str(total_adc_level[6][z])+'\t'+str(total_adc_level[7][z])+'\t'+str(total_adc_level[8][z])+'\t'+str(total_adc_level[9][z])+'\t'+str(total_adc_level[10][z])+'\t'
                                                 +str(total_adc_level[11][z])+'\t'+str(total_adc_level[12][z])+'\t'+str(total_adc_level[13][z])+'\t'+str(total_adc_level[14][z])+'\t'+str(total_adc_level[15][z])+'\n'
                                                 )
                                fl_total_adc.close()

                            ####################################################################################################################

                            nbins_adc = round(np.min(np.array((x[0].max(), ysim0[0].max(), ysim1[0].max()  )))-min_adc+1)
                            #nbins_adc = np.min(np.array((x[0].size, ysim0[0].size)))
                            #nbins_adc = round(np.max(x[0]))
                            hist_adc, bins_adc = np.histogram(x[0], bins=nbins_adc)
                            adclinbins = np.linspace( bins_adc[0], bins_adc[-1], len(bins_adc) )
                            print('all',adclinbins.size)
                            wxadc, xadc = np.histogram(x[0], weights=x[1], bins = adclinbins, density = densi)
                            wyadc0, yadc0 = np.histogram(ysim0[0], weights=ysim0[1], bins = adclinbins, density = densi)
                            wyadc1, yadc1 = np.histogram(ysim1[0], weights=ysim1[1], bins = adclinbins, density = densi)

                            if(bin_hist>1 and bin_hist<=nbins_adc):
                                min_adc_despl = int(np.ceil(min_adc/(np.max(x[0])/bin_hist)))
                                nbins_adc = bin_hist + min_adc_despl
                                #if(bin_hist > round(np.max(x[0]) ) ):
                                 #   min_adc_despl = int( np.ceil(min_adc/(np.max(x[0])/ round(np.max(x[0])) )) )
                                  #  nbins_adc = round(np.max(x0[0]) ) + min_adc_despl
                                #else: nbins_adc = bin_hist
                            else:
                                min_adc_despl = int( np.ceil(min_adc/(np.max(x[0])/ round(np.max(x[0])) )) )
                                nbins_adc = round(np.max(x[0])) + min_adc_despl

                            hist_adc, bins_adc = np.histogram(x[0], bins=nbins_adc)

                            bins_adc = bins_adc[ min_adc_despl : ]

                            #adclinbins = np.linspace( bins_adc[0], bins_adc[-1], len(bins_adc) )
                            adclinbins = np.linspace( bins_adc[0], max_adc, len(bins_adc) )
                            print('bin',adclinbins.size)
                            wxbh, xbh = np.histogram(x[0], weights=x[1], bins = adclinbins, density = densi)

                            wybh0, ybh0 = np.histogram(ysim0[0], weights=ysim0[1], bins = adclinbins, density = densi)
                            wybh1, ybh1 = np.histogram(ysim1[0], weights=ysim1[1], bins = adclinbins, density = densi)

                            bgadcw, bgadc = np.histogram(bg[0], weights=bg[1], bins = adclinbins, density = densi )

                            titulo = 'Hist:_'+ histo +source+'_Z'+str(z*2).zfill(1)+'mm'+sim_bg +filt_cut+ difu_gauss \
                                   # +'\n ev_' +ev+ '_sim_bg_'+ 'eta_'+ str(lower)

                            #plt.figure(figsize=(14,8))

                            x_weight,  x_adc  = np.histogram(x[0],  weights=x[1],  bins = adclinbins )
                            his_adc_obs = {'bins': x_adc, 'counts': x_weight}
                            #################################################################################
                            y_weight_sim0, y_adc_sim0 = np.histogram(ysim0[0], weights=ysim0[1], bins = adclinbins )
                            his_adc_sim0 = {'bins': y_adc_sim0, 'counts': y_weight_sim0}
                            y_weight_sim1, y_adc_sim1 = np.histogram(ysim1[0], weights=ysim1[1], bins = adclinbins )
                            his_adc_sim1 = {'bins': y_adc_sim1, 'counts': y_weight_sim1}
                            #################################################################################
                            bg_weight, bg_adc = np.histogram(bg[0], weights=bg[1], bins = adclinbins )
                            his_adc_bg = {'bins': bg_adc, 'counts': bg_weight}
                            #################################################################################

                            bincenters_x = 0.5*(x_adc[1:]+x_adc[:-1])
                            norm_x = (np.sum(x_weight) * np.diff(x_adc))
                            #################################################################################
                            bincenters_ysim0 = 0.5*(y_adc_sim0[1:]+y_adc_sim0[:-1])
                            norm_ysim0 = (np.sum(y_weight_sim0) * np.diff(y_adc_sim0))
                            bincenters_ysim1 = 0.5*(y_adc_sim1[1:]+y_adc_sim1[:-1])
                            norm_ysim1 = (np.sum(y_weight_sim1) * np.diff(y_adc_sim1))
                            #################################################################################
                            bincenters_bg = 0.5*(bg_adc[1:]+bg_adc[:-1])
                            norm_bg = (np.sum(bg_weight) * np.diff(bg_adc))
                            #################################################################################
                            # Guardar en un archivo de texto,  .npy
                            #hist_dat = np.load(dirsave_plt+'/'+'hist_adc_obs'+source+\
                             #           sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', allow_pickle=True).item()
                            np.save(dirsave_plt+'/'+'hist_adc_obs'+source+\
                                        sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', his_adc_obs)
                            np.save(dirsave_plt+'/'+'hist_adc_sim_'+source+\
                                        sim_n+'_'+gauss_dist+'_0k_0a_0zff'+ sig_bg_thresh +simu_no_cut+ str_cuts + cut_clst_size+cut_max+'.npy', his_adc_sim0)
                            np.save(dirsave_plt+'/'+'hist_adc_sim_'+source+\
                                        sim_n+difu_gauss+ sig_bg_thresh +simu_no_cut+ str_cuts + cut_clst_size+cut_max+'.npy', his_adc_sim1)

                            np.save(dirsave_plt+'/'+'hist_adc_bg_'+source_bg+\
                                        sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', his_adc_bg )
                            #################################################################################

                            if(densi==True):
                                x_w = x_weight/ norm_x
                                err_x = np.sqrt(x_weight)/ norm_x
                                #################################################################################
                                y_w_sim0 = y_weight_sim0/ norm_ysim0
                                err_ysim0 = (0.2*y_weight_sim0 + np.sqrt(y_weight_sim0) )/ norm_ysim0

                                y_w_sim1 = y_weight_sim1/ norm_ysim0
                                err_ysim1 = (0.2*y_weight_sim1 + np.sqrt(y_weight_sim1) )/ norm_ysim0
                                #################################################################################
                                bg_w = bg_weight/ norm_bg
                                err_bg = np.sqrt(bg_weight)/ norm_bg


                            if(densi==False):
                                x_w = x_weight
                                err_x = np.sqrt(x_weight)
                                #################################################################################
                                y_w_sim0 = y_weight_sim0
                                #err_ysim0 = np.sqrt(y_weight_sim0)  + err_porc_dif_sim0  #+ 0.2*y_weight_sim0
                                err_ysim0 = 0.2*y_weight_sim0  + np.sqrt(y_weight_sim0)

                                y_w_sim1 = y_weight_sim1
                                #err_ysim1 = np.sqrt(y_weight_sim1)  + err_porc_dif_sim1  #+ 0.2*y_weight_sim1
                                err_ysim1 = 0.2*y_weight_sim1 + np.sqrt(y_weight_sim1)

                                #################################################################################
                                bg_w = bg_weight
                                err_bg = np.sqrt(bg_weight)

                            #sta_ks, p_ks = stats.ks_2samp(x_w, y_w_sim_bg0)
                            #sta_chi, p_chi = stats.chisquare(x_ww, y_ww) #  chisquare(f_obs, f_exp=None, ddof=0, axis=0)
                            #p_permut = permutation_test(x_w, y_w_sim_bg0, method='approximate', num_rounds=10000, seed=0)  #Two-sided permutation test
                            #chi2_d =chi2_dist(x_w, y_w_sim_bg0)

                            err_adc0 = np.sqrt(err_x*err_x + err_ysim0*err_ysim0 )
                            ch2_adc0 = chi_2_sigm_test(x_w, y_w_sim0, err_adc0)
                            tmp_str0 = 'Chi2 test: ' + r'%.2f'%ch2_adc0[0] \
                                        +'       '+'ndf: ' + r'%.2f'%ch2_adc0[3] \
                                        +'\n'+'Chi2/ndf : ' + r'%.2f'%ch2_adc0[4]

                            err_adc1 = np.sqrt(err_x*err_x + err_ysim1*err_ysim1 )
                            ch2_adc1 = chi_2_sigm_test(x_w, y_w_sim1, err_adc1)
                            tmp_str1 = 'Chi2 test: ' + r'%.2f'%ch2_adc1[0] \
                                        +'       '+'ndf: ' + r'%.2f'%ch2_adc1[3] \
                                        +'\n'+'Chi2/ndf : ' + r'%.2f'%ch2_adc1[4]

                            if(labl_opt=='full'):
                                lbl_obs = source + ' data'+ sig_bg_thresh + str_cuts + cut_clst_size+cut_max + cut_adc
                                lbl_sim0 = source + ' simulation '+ sig_bg_thresh_sim + cut_clst_size+cut_max + cut_adc\
                                            +'\n'+'Chi2/ndf : ' + r'%.2f'%ch2_adc0[4]
                                lbl_sim1 = source + ' simulation '+ sig_bg_thresh_sim + cut_clst_size+cut_max + cut_adc \
                                            + d_gauss + '\n'+'Chi2/ndf : ' + r'%.2f'%ch2_adc1[4]
                                lbl_bg = 'Background' + ' data'+ si_bg_thresh + cut_clst_size+cut_max + cut_adc
                            if(labl_opt=='simple'):
                                lbl_obs = source + ' data'
                                lbl_sim0 = source + ' sim ' +r'$\chi^2_\nu$ : ' + r'%.2f'%ch2_adc0[4] #+'+ bg'
                                lbl_sim1 = source + ' sim ' + d_gauss + ' '+ r'$\chi^2_\nu$ : '  + r'%.2f'%ch2_adc1[4]#+'+ bg'
                                lbl_bg = 'Background' + ' data'
                            if(labl_opt=='off'):
                                lbl_obs = None
                                lbl_sim0 = None
                                lbl_sim1 = None
                                lbl_bg = None


                            plt.rcParams.update({'font.size': font_siz})

                            if( plot_nsig==True and plot_ratio==True ):
                                fig, (ax0, ax1, ax2) = plt.subplots(3,1,figsize=(14.5,14),gridspec_kw={'height_ratios': [5, 1, 1]} )#, sharex=True )

                            if( plot_nsig==True and plot_ratio==False ):
                                fig, (ax0, ax1) = plt.subplots(2,1,figsize=(14.5,12),gridspec_kw={'height_ratios': [5, 1]} )#, sharex=True )

                            if( plot_nsig==False and plot_ratio==True ):
                                fig, (ax0, ax2) = plt.subplots(2,1,figsize=(14.5,12),gridspec_kw={'height_ratios': [5, 1]} )#, sharex=True )

                            #if(plt_obs==True and plt_sim==True and plt_sim_bg!=True):
                            #    fig, (ax0,ax[1]) = plt.subplots(2,1,figsize=(14.5,12),gridspec_kw={'height_ratios': [5, 1]} )#, sharex=True )

                            #if(plt_obs==True and plt_sim!=True and plt_sim_bg==True):
                            #    fig, (ax0,ax[2]) = plt.subplots(2,1,figsize=(14.5,12),gridspec_kw={'height_ratios': [5, 1]} )#, sharex=True )

                            if( plot_nsig==False and plot_ratio==False ):
                                fig, (ax0) = plt.subplots(figsize=(14.5,9.5) )#, sharex=True )


                            if(plt_obs==True):
                                ax0.hist(x[0], weights=x[1], bins = adclinbins, histtype='step',
                                            density=densi, log=log_y, color='C3', linewidth=l_width+0.*l_width,
                                            #  label= source +'_'+ag+'ag'+'_' +str(nsigm_bg)+'sig'+'_z_'+str(z*2)+'mm'+'_t'+tiemp  )
                                            label = lbl_obs + cut_max_clst_porc
                                            )#  +'_'+ fecha )
                                if(plt_err==True):
                                    ax0.errorbar(bincenters_x, x_w, yerr=err_x, fmt='C3'+'.', ecolor='C3', linewidth=l_width )

                            ################################################################################################################################

                            if(plt_sim==True):
                                ax0.hist(ysim0[0], weights=ysim0[1], bins = adclinbins, histtype='step'
                                            , density=densi, log=log_y, color='C2', linewidth=l_width,
                                            label = lbl_sim0 + cut_max_clst_porc_sim)

                                ax0.hist(ysim1[0], weights=ysim1[1], bins = adclinbins, histtype='step'
                                            , density=densi, log=log_y, color='C9', linewidth=l_width,
                                            label = lbl_sim1 + cut_max_clst_porc_sim)

                            if(plt_sim==True and plt_err==True):
                                ax0.errorbar(bincenters_ysim0, y_w_sim0, yerr=err_ysim0, fmt='C2'+'.',
                                                ecolor='C2', linewidth=l_width*2/3 )
                                        # ,label= '\n P_ks = %.3f' % p_ks )# + '\n P_prmut = %.3f' % p_permut )
                                       # + '\n Chi2_Dist= %.3f' %chi2_d + '\n P_ch2 = %.3f' % p_chi  )
                                ax0.errorbar(bincenters_ysim1, y_w_sim1, yerr=err_ysim1, fmt='C9'+'.',
                                                ecolor='C9', linewidth=l_width*2/3 )
                                        # ,label= '\n P_ks = %.3f' % p_ks )# + '\n P_prmut = %.3f' % p_permut )
                                       # + '\n Chi2_Dist= %.3f' %chi2_d + '\n P_ch2 = %.3f' % p_chi  )
                            ################################################################################################################################
                            ################################################################################################################################

                            if(plt_bg==True):
                                ax0.hist(bg[0], weights=bg[1], bins = adclinbins, histtype='step'
                                            , density=densi, log=log_y, color='k', linewidth=l_width*2/3,
                                            #label='Back_ground'  +'_'+ag+'ag'+'_' +str(nsigm_bg_sim)+'sig' +'_t'+tiemp )
                                            label = lbl_bg )# + '_' + fecha_bg )
                                if(plt_err==True):
                                    ax0.errorbar(bincenters_bg, bg_w, yerr=err_bg, fmt='k'+'.',
                                                    ecolor='k', linewidth=l_width*2/3 )#, label = 'distance: '+str(2*z)+'mm' )

                            #i+=1
                            U_eV = 1000
                            min_cam =((min_adc-1)*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
                            sat_cam =(1023*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
                            #ax0.axvline(x = min_adc-1, color = 'k', linewidth=l_width+0.5*l_width, label = 'min_cam: '+str(min_adc-1)+' ADC = ' + str(min_cam)+' keV')
                            if(max_adc>=1023):ax0.axvline(x = 1023, color = 'k', linestyle="--", linewidth=l_width*2/3)#, label = 'sat_cam: 1023 ADC = ' + str(sat_cam)+' keV')


                            if( file_csv==True):
                                file_activ_pixel = dirsave_plt+'/'+source+dens+'_'+strbin+str(z_count)+'l_'+fecha+'_'+evt+'_'+str(z*2).zfill(2)+'mm'+'sim_bg_actv_pixel'+'_'+str(min_adc)
                                file_activ = open(file_activ_pixel,'w')

                                txt_activ = 'data_'+source+'\t' + '0_ADC\t' + 'Active_Pixel\t' + 'all_pixel\t' + '%_0_ADC\t' + '%_Active_Pixel'
                                file_activ.write(txt_activ + '\n')
                                txt_activ = 'OBS'+ name_filter_obs +':\t'+ str(data_obsr[1][0:min_adc].sum())+'\t' \
                                            + str(data_obsr[1][min_adc:1024].sum()) +'\t'+ str(data_obsr[1].sum())+'\t' \
                                            + str(100*data_obsr[1][0:min_adc].sum()/data_obsr[1].sum())+'\t' \
                                            + str(100*data_obsr[1][min_adc:1024].sum()/data_obsr[1].sum())
                                file_activ.write(txt_activ + '\n')
                                txt_activ = 'SIM0'+sim_bg +':\t'+ str(data_cnt_sim_0[1][0:min_adc].sum())+'\t' \
                                            + str(data_cnt_sim_0[1][min_adc:1024].sum()) +'\t'+ str(data_cnt_sim_0[1].sum())+'\t' \
                                            + str(100*data_cnt_sim_0[1][0:min_adc].sum()/data_cnt_sim_0[1].sum())+'\t' \
                                            + str(100*data_cnt_sim_0[1][min_adc:1024].sum()/data_cnt_sim_0[1].sum())
                                file_activ.write(txt_activ + '\n')
                                txt_activ = 'SIM1'+sim_bg+ d_gauss + ':\t'+ str(data_cnt_sim_1[1][0:min_adc].sum())+'\t' \
                                            + str(data_cnt_sim_1[1][min_adc:1024].sum()) +'\t'+ str(data_cnt_sim_1[1].sum())+'\t' \
                                            + str(100*data_cnt_sim_1[1][0:min_adc].sum()/data_cnt_sim_1[1].sum())+'\t' \
                                            + str(100*data_cnt_sim_1[1][min_adc:1024].sum()/data_cnt_sim_1[1].sum())
                                file_activ.write(txt_activ + '\n')
                                file_activ.close()

                            print('\n')
                            print('OBS'+'_'+str(nsigm_bg)+'sig'+':\t', data_obsr[1][0:min_adc].sum(),
                                    '\t', data_obsr[1][min_adc:1024].sum(),'\t',data_obsr[1].sum(),
                                    '\t'+ str(100*data_obsr[1][0:min_adc].sum()/data_obsr[1].sum()),
                                    '\t'+ str(100*data_obsr[1][min_adc:1024].sum()/data_obsr[1].sum()))

                            print('SIM0'+'_'+str(nsigm_bg_sim)+'sig'+':\t', data_cnt_sim_0[1][0:min_adc].sum(),
                                    '\t', data_cnt_sim_0[1][min_adc:1024].sum(),'\t', data_cnt_sim_0[1].sum(),
                                    '\t'+ str(100*data_cnt_sim_0[1][0:min_adc].sum()/data_cnt_sim_0[1].sum()),
                                    '\t'+ str(100*data_cnt_sim_0[1][min_adc:1024].sum()/data_cnt_sim_0[1].sum()))

                            print('SIM1'+'_'+ d_gauss + ':\t', data_cnt_sim_1[1][0:min_adc].sum(),
                                    '\t', data_cnt_sim_1[1][min_adc:1024].sum(),'\t', data_cnt_sim_1[1].sum(),
                                    '\t'+ str(100*data_cnt_sim_1[1][0:min_adc].sum()/data_cnt_sim_1[1].sum()),
                                    '\t'+ str(100*data_cnt_sim_1[1][min_adc:1024].sum()/data_cnt_sim_1[1].sum()))

                            print('\n')


                            if(plt_obs==True):
                                if(log_y==True):ax0.set_yscale('log')
                                if(yscal_adc>0):ax0.set_ylim(0, yscal_adc)
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

                            ylabl_comp= r'$\frac{ (data -sim)}{\sigma}$'
                            comp_sim0, err_comp_sim0 = comp_obs_sim(x_w, y_w_sim0)
                            comp_sim1, err_comp_sim1 = comp_obs_sim(x_w, y_w_sim1)
                            delta_min = np.min(np.array((np.nanmin(comp_sim0), np.nanmin(comp_sim1) )))
                            delta_max = np.max(np.array((comp_sim0[comp_sim0 < np.Inf].max(), comp_sim1[comp_sim1 < np.Inf].max() )))

                            if(plt_sim==True and plot_nsig==True):
                                #ax1.axhline(y = 0., xmin=min_adc, xmax=max_adc, color = 'k', linestyle = '--')
                                ax1.hist(x[0], weights=0*x[1], bins = adclinbins, histtype='step', ls='--', density=densi, color='k', linewidth=l_width*2/3 )
                                ax1.errorbar(bincenters_x, comp_sim0, yerr=plt_err*err_comp_sim0, fmt='C2'+'>', lw=2, markersize=12 )
                                if(zoom_delta==True):ax1.set_ylim(delta_min,delta_max)

                            if(plt_sim1==True and plot_nsig==True):
                                #ax1.axhline(y = 0., xmin=min_adc, xmax=max_adc, color = 'k', linestyle = '--')
                                ax1.hist(x[0], weights=0*x[1], bins = adclinbins, histtype='step', ls='--', density=densi, color='k', linewidth=l_width*2/3 )
                                ax1.errorbar(bincenters_x, comp_sim1, yerr=plt_err*err_comp_sim1, fmt='C0'+'o', lw=2, markersize=10 )
                                if(zoom_delta==True):ax1.set_ylim(delta_min,delta_max)

                            if( (plt_sim1==True or plt_sim==True) and plot_nsig==True ):
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

                            ylabl_ratio= r'$ratio = \frac{data}{sim}$'
                            ratio_sim0, err_ratio_sim0 = ratio_obs_sim(x_w, y_w_sim0)
                            ratio_sim1, err_ratio_sim1 = ratio_obs_sim(x_w, y_w_sim1)
                            ratio_min = np.min(np.array((np.nanmin(ratio_sim0), np.nanmin(ratio_sim1) )))
                            ratio_max = np.max(np.array((ratio_sim0[ratio_sim0 < np.Inf].max(), ratio_sim1[ratio_sim1 < np.Inf].max() )))

                            if(plt_sim==True and plot_ratio==True):
                                #ax2.axhline(y = 0., xmin=min_adc, xmax=max_adc, color = 'k', linestyle = '--')
                                ax2.hist(x[0], weights=0*x[1], bins = adclinbins, histtype='step', ls='--', density=densi, color='k', linewidth=l_width*2/3 )
                                ax2.errorbar(bincenters_x, ratio_sim0, yerr=plt_err*err_ratio_sim0, fmt='C2'+'>', lw=2, markersize=12 )
                                if(zoom_ratio==True):ax2.set_ylim(-0.5+ratio_min,ratio_max)

                            if(plt_sim1==True and plot_ratio==True):
                                #ax2.axhline(y = 0., xmin=min_adc, xmax=max_adc, color = 'k', linestyle = '--')
                                ax2.hist(x[0], weights=0*x[1], bins = adclinbins, histtype='step', ls='--', density=densi, color='k', linewidth=l_width*2/3 )
                                ax2.errorbar(bincenters_x, ratio_sim1, yerr=plt_err*err_ratio_sim1, fmt='C0'+'o', lw=2, markersize=10 )
                                if(zoom_ratio==True):ax2.set_ylim(-0.5+ratio_min,ratio_max)

                            if( (plt_sim1==True or plt_sim==True) and plot_ratio==True ):
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

                            mm=(ysim0[0]*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
                            ax3.plot(mm, ysim0[1]*0., color='w', linewidth=0.001 )
                            ax3.set_xlabel('Energy (keV)', fontsize=font_siz+4)

                            ax0.legend(loc='lower right')
                            ax0.legend(loc='lower left')
                            #ax0.legend(loc='upper center')

                            fig.tight_layout()
                            #name_cuts = '_'+str(nframes)+'f'+'_'+fecha+eta_cross+sim_bg0 + filt_cut + str_hist_comp
                            name_cuts = '_'+str(nframes)+'f'+'_'+source+sim_bg + filt_cut + str_hist_comp
                            #namesave = 'plot_hist_'+histo+'ADC_'+strbin+str(z*2).zfill(2)+'mm_sim'+'_'+source+'_'+ \
                            #            str(nsigm_bg)+'sig'+'_'+ev+'ev_'+fecha+str_hist_comp+cut_adc0
                            namesave = 'plot_hist_'+histo+'ADC_'+strbin+str(z*2).zfill(2)+'mm'+ name_cuts
                            plt.savefig(dirsave_plt+'/'+namesave +'_'+str(min_adc)+'_'+str(max_adc)+'ADC'+difu_gauss+'.png', dpi=150)
                            plt.savefig(dirsave_plt+'/'+namesave +'_'+str(min_adc)+'_'+str(max_adc)+'ADC'+difu_gauss+'.pdf', dpi=150)

                            # plt.show()
                            #plt.clf()
                            #plt.close()

                            #######################################################################################################
                            #######################################################################################################
                            #######################################################################################################

                            ##########################

                            print('\n\n\n\n')
                            #print('ADC =', backg[0].max(), bg[0].max(), ' cluster = ', data_bg_cut[:,7].max(), ' mean = ' , data_bg_cut[:,8].max(), ' max = ' , data_bg_cut[:,10].max(), ' total = ' , data_bg_cut[:,9].max())

                            if( file_csv==True):
                                file_name_test = dirsave_plt+'/'+source+dens+'_'+strbin+str(z_count)+'l_'+fecha+'_'+evt+'_'+str(z*2).zfill(2)+'mm'+'sim_bg_clst'+ str_cuts+'_'+str(max_adc)+'ADC'+'_chi2.csv'
                                # np.savetxt(file_test, np.vstack(('Chi_sq2_test', 'hist_ADC', prc_r2, lower, bin_hist, test_corr[0], test_corr[1], test_corr[2], test_corr[3], test_corr[4] )).T, delimiter='\t', newline='\n',fmt='%s' ,  header='test bins, histo, proc, eta, bins, chi_2, pvalue, criti_val, ndf, chi2/nf')
                                print('\n\n')
                                file_test = open(file_name_test,'w')

                                ######################################################################################################################################################################################################
                                ######################################################################################################################################################################################################

                                if(plt_sim==True):
                                    print('\n')
                                    txt_test = 'test bins'+sig_bg_thresh+ cut_clst_size+cut_max + cut_adc + cut_max_clst_porc+sig_bg_thresh_sim+cut_max_clst_porc_sim+dens+'\t' + 'histo\t' + 'bins\t' + 'chi_2\t' + 'pvalue\t' + 'criti_val\t' + 'ndf\t' + 'chi2/nf'
                                    file_test.write('\n\n'+ txt_test + '\n')
                                    print(txt_test)

                                    err_ch = np.sqrt(err_x*err_x + err_ysim0*err_ysim0 )
                                    test_corr = chi_2_sigm_test(x_w, y_w_sim0, err_ch)
                                    txt_test = 'Chi2_sig_test_sim_'+d_gauss +'\t'+'hist_ADC'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%test_corr[0]) +'\t'+ str(r'%.5f'%test_corr[1])  +'\t'+ str(r'%.5f'%test_corr[2])  +'\t'+ str(test_corr[3]) +'\t'+ str(r'%.5f'%test_corr[4])
                                    file_test.write(txt_test + '\n')
                                    print(txt_test)


                                ######################################################################################################################################################################################################
                                ######################################################################################################################################################################################################

                                file_test.close()

                                print('\n\n\n\n')
                                #print('cluster = ', data_bg_cut[:,7].max(), ' mean = ' , data_bg_cut[:,8].max(), ' max = ' , data_bg_cut[:,10].max(), ' total = ' , data_bg_cut[:,9].max())

                            else:
                                print('Folder Not Found:', file_cluster_size_mean_sim0)
                                print('Folder Not Found:', file_cluster_size_mean_sim1)

                            print('\n')
                            if(i_cont==1):
                                chi2_dict = {'DG_z_sig_'+str(0)+'_k_'+str(0)+'_alpha_'+str(0)+'_zff_'+str(0):{'chi2/nf_ADC'  :  ch2_adc0[4] }   }
                                file_test_chi2_nf = open(chi2_reduc_distr_test,'w')
                                txt_head_test_chi2 = 'test bins'+ str(bin_hist)+ cut_adc + dens+'\t' + 'chi2/nf_ADC\t'
                                file_test_chi2_nf.write( txt_head_test_chi2 + '\n')

                                txt_test_chi2 = 'DG_z_sig_'+str(0)+'_k_'+str(0)+'_alpha_'+str(0)+'_zff_'+str(0) +'\t' + str(ch2_adc0[4])
                                file_test_chi2_nf.write(txt_test_chi2 + '\n')
                                #file_test_chi2_nf.close()
                                print(txt_head_test_chi2)

                            chi2_dict['DG_z_sig_'+str(nsigm)+'_k_'+str(k)+'_alpha_'+str(alpha)+'_zff_'+str(zff)]={'chi2/nf_ADC'   :   ch2_adc1[4] }


                            txt_test_chi2 = 'DG_z_sig_'+str(nsigm)+'_k_'+str(k)+'_alpha_'+str(alpha)+'_zff_'+str(zff) +'\t' + str(ch2_adc1[4])
                            file_test_chi2_nf.write(txt_test_chi2 + '\n')
                            #file_test_chi2_nf.close()
                            print(i_cont, txt_test_chi2)



                            if( file_total_adc_csv==True and densi==False ):
                                np.save(file_total_adc +'.npy', np.float64(total_adc_level))
                                np.savez_compressed(file_total_adc +'.npz', np.float64(total_adc_level))

                
        file_test_chi2_nf.close()
        chi2_dict_save_path = dirsavepatn_plt+'All_'+source + sim_n + dens+'_'+strbin+str(z_count)+'l_'+fecha+'_'+evt+'_'+str(z*2).zfill(2)+'mm'+'sim_clst'+ simu_no_cut+str_cuts+'_'+str(max_adc)+'ADC'+'_chi2'
        np.savez(chi2_dict_save_path+'.npz', **chi2_dict)

    #loaded_data = np.load('data_dict.npz', allow_pickle=True)
    #data_dict_loaded = {key: loaded_data[key].item() for key in loaded_data}
    #print(data_dict_loaded)
