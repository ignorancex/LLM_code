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
file_clst_sum_csv = 1
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
#source = 'Cs137'


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
bin_hist = 20
if(bin_hist>0): strbin = str(bin_hist)+'bin_z_'
else:strbin = 'z_'

max_adc=1024
max_adc=150
min_adc=1

size_thresh = 0
max_thresh = 0
#if(adc_cut == 0):min_adc=1

if(max_adc<1024):cut_adcmax_str = "_mx"+str(max_adc)
else: cut_adcmax_str = ''

plots_2d=1
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


                         path_cluster_obs = ssd_obs + 'dat_'+fecha+'_clstr_filt'+cut_clst_size+cut_max +'/'#+cut_max_clst_porc+'/'
                         ####path_cluster = '/home/milton/g4_works/data_2023/dat_'+fecha+'_clstr_filt'+cut_clst_size+cut_max +'/'#+cut_max_clst_porc+'/'
                         #path_cluster_sim0 = '/home/milton/g4_works/data_2023/dat_'+fecha+cut_clst_size+cut_max +condit_edep+'_Dgaus_0sig_0zff'+'_clstr_filt'+'/'#+cut_max_clst_porc_sim+'/'
                         #path_cluster_sim1 = '/home/milton/g4_works/data_2023/dat'+sim_n+'_'+fecha+difu_gauss+'_clstr_filt'+cut_clst_size+cut_max +condit_edep+'/'#+cut_max_clst_porc_sim+'/'
                         if(simu_no_cut == ''):
                             path_cluster_sim0 = drive_file_sim + source +'/dat'+'_0'+'_'+fecha+'_'+gauss_dist+'_0k_0a_0zff'+'_clstr_filt'+cut_clst_size+cut_max +condit_edep+'/'#+cut_max_clst_porc_sim+'/
                             path_cluster_sim1 = drive_file_sim + source +'/dat'+sim_n+'_'+fecha+difu_gauss+'_clstr_filt'+cut_clst_size+cut_max +condit_edep+'/'#+cut_max_clst_porc_sim+'/'

                         if(simu_no_cut == '_sim_nocut'):
                             path_cluster_sim0 = drive_file_sim + source +'/dat'+'_0'+'_'+fecha+'_'+gauss_dist+'_0k_0a_0zff'+'_clstr_filt'+condit_edep+'/'#+cut_max_clst_porc_sim+'/
                             path_cluster_sim1 = drive_file_sim + source +'/dat'+sim_n+'_'+fecha+difu_gauss+'_clstr_filt'+condit_edep+'/'#+cut_max_clst_porc_sim+'/'


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

                         file_cluster_size_mean_bg = path_cluster_obs + 'dat_clstr_frm_siz_mean' + \
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
                         #############################################################################################################################################################
                         numb_clst_level = np.zeros((26, z_count))
                         total_adc_level = np.zeros((26, z_count))


                         for z in level_z:

                            dirsavepatn_plt = ssd_sim +gauss_dist+'_'+source +'_rslhist_'+strbin +str(z*2).zfill(2)+'mm'+ cut_clst_size+cut_max+str_cuts+'/'
                            dirsave_plt = dirsavepatn_plt + 'plots2'+sim_n+'_'+source+'_'+str(z_count)+'l'+'_'+str(nframes)+'f'+simu_no_cut+condit_edep+difu_gauss+'/plot_hist_mult_'+source + '_'+str(z_count)+'l_'+fecha+'_'+evt+sim_bg  #+ cuts_dat +simu_no_cut

                            try:
                                os.makedirs(dirsave_plt)
                            except FileExistsError:
                                pass


                            if( file_csv==True and i_cont == 0):

                                chi2_reduc_distr_test = dirsavepatn_plt+'All_'+source + sim_n +dens+'_'+strbin+str(z_count)+'l_'+fecha+'_'+evt+'_'+str(z*2).zfill(2)+'mm'+'sim_clst'+ simu_no_cut+ str_cuts+'_'+str(max_adc)+'ADC'+'_chi2.csv'
                            i_cont+=1

                            file_cluster_size_mean = path_cluster_obs + 'dat_clstr_frm_siz_mean' + \
                                        cut_adc + avrg_cut + sig_bg_thresh +'_sim' +sig_bg_thresh_sim + \
                                        '_f' +str(nframes) + '_z' +str(2*z) + cut_clst_size+cut_max + cut_max_clst_porc+'/'
                            if(simu_no_cut == ''):
                                file_cluster_size_mean_sim0 = path_cluster_sim0 + 'dat_clstr_frm_siz_mean' + '_'+gauss_dist+'_0k_0a_0zff'+\
                                    '_sim' +sig_bg_thresh_sim + '_f' +str(nframes) + '_z' +str(2*z) + cut_max_clst_porc_sim+'/'

                                file_cluster_size_mean_sim1 = path_cluster_sim1 + 'dat_clstr_frm_siz_mean' +difu_gauss + \
                                    '_sim' + sig_bg_thresh_sim + sigthresh + \
                                    '_f' +str(nframes) + '_z' +str(2*z) + cut_max_clst_porc_sim+'/'

                            if(simu_no_cut == '_sim_nocut'):
                                file_cluster_size_mean_sim0 = path_cluster_sim0 + 'dat_clstr_frm_siz_mean' + '_'+gauss_dist+'_0k_0a_0zff'+\
                                        '_sim' +sig_bg_thresh_sim + '_f' +str(nframes) + '_z' +str(2*z) +'/'

                                file_cluster_size_mean_sim1 = path_cluster_sim1 + 'dat_clstr_frm_siz_mean' +difu_gauss + \
                                        '_sim' +sig_bg_thresh_sim + '_f' +str(nframes) + '_z' +str(2*z) +'/'


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

                            data_cluste_frame_obs = file_cluster_size_mean+'dat_obs_'+source+'_f'+str(nframes)+'_clstr_frm_z'+str(z).zfill(3)+\
                                                  '_'+fecha+cut_adc+avrg_cut + sig_bg_thresh + cut_clst_size+cut_max +'.csv'

                            dat_obs = np.loadtxt(data_cluste_frame_obs, delimiter='\t', skiprows=1)

                            n_clst_level_obs = file_cluster_size_mean +  'clstr_frm_siz_mean_obs_'+source+'_f'+str(nframes)+'_z'+str(z).zfill(3)+\
                                                '_'+fecha+cut_adc+avrg_cut + sig_bg_thresh + cut_clst_size+cut_max +'.csv'

                            n_clst_level_obs = np.loadtxt(n_clst_level_obs, delimiter='\t', skiprows=1)
                            #######################################################################################################
                            #######################################################################################################
                            if(simu_no_cut == '' ):
                                data_cluste_frame_sim0 =file_cluster_size_mean_sim0+'dat_sim_'+source+'_f'+str(nframes)+'_clstr_frm_z'+str(2*z).zfill(3)+\
                                                    '_w'+evt+ '_'+gauss_dist+'_0k_0a_0zff' + sig_bg_thresh_sim + cut_adc + cut_clst_size+cut_max + '.csv'
                                n_clst_level_sim0 = file_cluster_size_mean_sim0 +  'clstr_frm_siz_mean_sim_'+source+'_f'+str(nframes)+'_z'+str(2*z).zfill(3)+\
                                                '_w'+evt+'_'+gauss_dist+'_0k_0a_0zff' + sig_bg_thresh_sim + cut_adc + cut_clst_size+cut_max + '.csv'

                                data_cluste_frame_sim1 = file_cluster_size_mean_sim1+'dat_sim_'+source+'_f'+str(nframes)+'_clstr_frm_z'+str(2*z).zfill(3)+\
                                                    '_w'+evt+difu_gauss + cut_adc + sig_bg_thresh_sim + sigthresh + cut_clst_size+cut_max +'.csv'

                                n_clst_level_sim1 = file_cluster_size_mean_sim1 +'clstr_frm_siz_mean_sim_'+source+'_f'+str(nframes)+'_z'+str(2*z).zfill(3)+\
                                            '_w'+evt+difu_gauss + cut_adc + sig_bg_thresh_sim + sigthresh + cut_clst_size+cut_max  +'.csv'

                            if(simu_no_cut == '_sim_nocut'):
                                data_cluste_frame_sim0 =file_cluster_size_mean_sim0+'dat_sim_'+source+'_f'+str(nframes)+'_clstr_frm_z'+str(2*z).zfill(3)+\
                                                    '_w'+evt+ '_'+gauss_dist+'_0k_0a_0zff' + sig_bg_thresh_sim + '.csv'
                                n_clst_level_sim0 = file_cluster_size_mean_sim0 +  'clstr_frm_siz_mean_sim_'+source+'_f'+str(nframes)+'_z'+str(2*z).zfill(3)+\
                                                '_w'+evt+'_'+gauss_dist+'_0k_0a_0zff' + sig_bg_thresh_sim + '.csv'

                                data_cluste_frame_sim1 = file_cluster_size_mean_sim1+'dat_sim_'+source+'_f'+str(nframes)+'_clstr_frm_z'+str(2*z).zfill(3)+\
                                                    '_w'+evt+difu_gauss + cut_adc + sig_bg_thresh_sim + sigthresh +'.csv'

                                n_clst_level_sim1 = file_cluster_size_mean_sim1 +'clstr_frm_siz_mean_sim_'+source+'_f'+str(nframes)+'_z'+str(2*z).zfill(3)+\
                                            '_w'+evt+difu_gauss + cut_adc + sig_bg_thresh_sim + sigthresh +'.csv'
                            #######################################################################################################
                            #######################################################################################################
                            n_clst_level_sim0 = np.loadtxt(n_clst_level_sim0, delimiter='\t', skiprows=1)
                            n_clst_level_sim1 = np.loadtxt(n_clst_level_sim1, delimiter='\t', skiprows=1)
                            #######################################################################################################

                            if os.path.isdir(file_cluster_size_mean_sim0 or file_cluster_size_mean_sim1 ):
                                print('Folder Found:', file_cluster_size_mean_sim0)
                                print('Folder Found:', file_cluster_size_mean_sim1)
                                #######################################################################################################
                                dat_sim_0 = np.loadtxt(data_cluste_frame_sim0, delimiter='\t', skiprows=1)
                                dat_sim_1 = np.loadtxt(data_cluste_frame_sim1, delimiter='\t', skiprows=1)
                                #######################################################################################################
                                #######################################################################################################

                                data_cut_obs0, cut_str0 = cut_data(cut0, cut_dat00, cut_dat01, cut_type0, dat_obs, '')
                                data_cut_obs1, cut_str1 = cut_data(cut1, cut_dat10, cut_dat11, cut_type1, data_cut_obs0, cut_str0)
                                data_cut_obs2, cut_str2 = cut_data(cut2, cut_dat20, cut_dat21, cut_type2, data_cut_obs1, cut_str1)
                                data_cut_obs,  cut_str  = cut_data(cut3, cut_dat30, cut_dat31, cut_type3, data_cut_obs2, cut_str2)

                                data_bg_cut0, cut_str0 = cut_data(cut0, cut_dat00, cut_dat01, cut_type0, data_fondo, '')
                                if(data_bg_cut0.size == 0): data_bg_cut1 = np.zeros((1, 11)); c_bg_real=0;
                                else:data_bg_cut1, cut_str1 = cut_data(cut1, cut_dat10, cut_dat11, cut_type1, data_bg_cut0, cut_str0)
                                if(data_bg_cut1.size == 0): data_bg_cut2 = np.zeros((1, 11)); c_bg_real=0;
                                else:data_bg_cut2, cut_str2 = cut_data(cut2, cut_dat20, cut_dat21, cut_type2, data_bg_cut1, cut_str1)
                                if(data_bg_cut2.size == 0): data_bg_cut = np.zeros((1, 11));  c_bg_real=0;
                                else:data_bg_cut,  cut_str  = cut_data(cut3, cut_dat30, cut_dat31, cut_type3, data_bg_cut2, cut_str2)

                                #######################################################################################################
                                if(simu_no_cut == ''):
                                    data_cut_sim_00, cut_str0 = cut_data(cut0, cut_dat00, cut_dat01, cut_type0, dat_sim_0, '')
                                    data_cut_sim_01, cut_str1 = cut_data(cut1, cut_dat10, cut_dat11, cut_type1, data_cut_sim_00, cut_str0)
                                    data_cut_sim_02, cut_str2 = cut_data(cut2, cut_dat20, cut_dat21, cut_type2, data_cut_sim_01, cut_str1)
                                    data_cut_sim_0,  cut_str  = cut_data(cut3, cut_dat30, cut_dat31, cut_type3, data_cut_sim_02, cut_str2)

                                    data_cut_sim_10, cut_str0 = cut_data(cut0, cut_dat00, cut_dat01, cut_type0, dat_sim_1, '')
                                    data_cut_sim_11, cut_str1 = cut_data(cut1, cut_dat10, cut_dat11, cut_type1, data_cut_sim_10, cut_str0)
                                    data_cut_sim_12, cut_str2 = cut_data(cut2, cut_dat20, cut_dat21, cut_type2, data_cut_sim_11, cut_str1)
                                    data_cut_sim_1,  cut_str  = cut_data(cut3, cut_dat30, cut_dat31, cut_type3, data_cut_sim_12, cut_str2)

                                if(simu_no_cut == '_sim_nocut'):
                                    data_cut_sim_0 = dat_sim_0
                                    data_cut_sim_1 = dat_sim_1
                                #######################################################################################################
                                #######################################################################################################

                                if(data_bg_cut.size == 0): c_bg_real=0; data_bg_cut = np.zeros((1, 8))

                                #############################################################################################################################################################
                                #############################################################################################################################################################
                                if( file_clst_sum_csv==True and densi==False and z==0 ):
                                    file_clst_sum = dirsave_plt+'/'+source+'_'+str(z_count)+'l_'+fecha+'_'+evt +'_sim_s2_num_clst'+cut_type2+cut_clmaxadc+'_dist'
                                    print('\n')
                                    fl_clst_sum = open(file_clst_sum+'.csv','w')
                                    fl_clst_sum.write("dist(mm)\t num_clst_obs_total\t num_clst_obs_mean\t num_clst_obs_std_1\t s2_obs\t num_clst_obs_err_2\t" +
                                                      "num_clst_sim_total\t num_clst_sim_mean\t num_clst_sim_std_1\t s2_sim\t num_clst_sim_err_2\t" +
                                                      "num_clst_sim1_total\t num_clst_sim1_mean\t num_clst_sim1_std_1\t s2_sim1\t num_clst_sim1_err_2\n")
                                    fl_clst_sum.close()

                                sum_nclst_obs=0
                                sum2_nclst_obs=0
                                obs_nc_apnd = np.append(data_cut_obs[:,1], 0)
                                for i in range (data_cut_obs[:,6].size):
                                    if(data_cut_obs[:,1][i]!= obs_nc_apnd[i+1]):
                                        sum_nclst_obs = sum_nclst_obs+data_cut_obs[:,6][i]
                                        sum2_nclst_obs = sum2_nclst_obs+data_cut_obs[:,6][i]*data_cut_obs[:,6][i]
                                        #print(i,data_cut_obs[:,6][i],sum_nclst_obs, sum2_nclst_obs)
                                avg_nclst_obs = sum_nclst_obs/ nfrm
                                std_nclst_obs = np.sqrt((sum2_nclst_obs-sum_nclst_obs*sum_nclst_obs/nfrm)/(nfrm))
                                std_n1clst_obs = np.sqrt((sum2_nclst_obs-sum_nclst_obs*sum_nclst_obs/nfrm)/(nfrm-1))
                                print('mikiiii\n')
                                print(sum_nclst_obs, sum2_nclst_obs, avg_nclst_obs, std_nclst_obs, std_n1clst_obs)
                                print(np.sum(n_clst_level_obs[:,1]), np.mean(n_clst_level_obs[:,1]), np.std(n_clst_level_obs[:,1]), np.std(n_clst_level_obs[:,1], ddof=1))

                                sum_nclst_sim0=0
                                sum2_nclst_sim0=0
                                sim0_nc_apnd = np.append(data_cut_sim_0[:,1], 0)
                                for i in range (data_cut_sim_0[:,6].size):
                                    if(data_cut_sim_0[:,1][i]!= sim0_nc_apnd[i+1]):
                                        sum_nclst_sim0 = sum_nclst_sim0+data_cut_sim_0[:,6][i]
                                        sum2_nclst_sim0 = sum2_nclst_sim0+data_cut_sim_0[:,6][i]*data_cut_sim_0[:,6][i]
                                        #print(i,data_cut_sim_0[:,6][i],sum_nclst_sim0, sum2_nclst_sim0)
                                avg_nclst_sim0 = sum_nclst_sim0/ nfrm
                                std_nclst_sim0 = np.sqrt((sum2_nclst_sim0-sum_nclst_sim0*sum_nclst_sim0/nfrm)/(nfrm))
                                std_n1clst_sim0 = np.sqrt((sum2_nclst_sim0-sum_nclst_sim0*sum_nclst_sim0/nfrm)/(nfrm-1))
                                print('\n')
                                print(sum_nclst_sim0, sum2_nclst_sim0, avg_nclst_sim0, std_nclst_sim0, std_n1clst_sim0)
                                print(np.sum(n_clst_level_sim0[:,1]), np.mean(n_clst_level_sim0[:,1]), np.std(n_clst_level_sim0[:,1]), np.std(n_clst_level_sim0[:,1], ddof=1))

                                sum_nclst_sim1=0
                                sum2_nclst_sim1=0
                                sim1_nc_apnd = np.append(data_cut_sim_1[:,1], 0)
                                for i in range (data_cut_sim_1[:,6].size):
                                    if(data_cut_sim_1[:,1][i]!= sim1_nc_apnd[i+1]):
                                        sum_nclst_sim1 = sum_nclst_sim1+data_cut_sim_1[:,6][i]
                                        sum2_nclst_sim1 = sum2_nclst_sim1+data_cut_sim_1[:,6][i]*data_cut_sim_1[:,6][i]
                                        #print(i,data_cut_sim_1[:,6][i],sum_nclst_sim1, sum2_nclst_sim1)
                                avg_nclst_sim1 = sum_nclst_sim1/ nfrm
                                std_nclst_sim1 = np.sqrt((sum2_nclst_sim1-sum_nclst_sim1*sum_nclst_sim1/nfrm)/(nfrm))
                                std_n1clst_sim1 = np.sqrt((sum2_nclst_sim1-sum_nclst_sim1*sum_nclst_sim1/nfrm)/(nfrm-1))
                                print('\n')
                                print(sum_nclst_sim1, sum2_nclst_sim1, avg_nclst_sim1, std_nclst_sim1, std_n1clst_sim1)
                                print(np.sum(n_clst_level_sim1[:,1]), np.mean(n_clst_level_sim1[:,1]), np.std(n_clst_level_sim1[:,1]), np.std(n_clst_level_sim1[:,1], ddof=1))

                                if(file_clst_sum_csv==True and densi==False):
                                    numb_clst_level[0][z]=z*2
                                    '''
                                    numb_clst_level[1][z]=np.sum(n_clst_level_obs[:,1])
                                    numb_clst_level[2][z]=np.mean(n_clst_level_obs[:,1])
                                    numb_clst_level[3][z]=np.std(n_clst_level_obs[:,1], ddof=1)
                                    numb_clst_level[4][z]=np.size(n_clst_level_obs[:,1])
                                    numb_clst_level[5][z]=np.std(n_clst_level_obs[:,1], ddof=1)/np.sqrt(np.size(n_clst_level_obs[:,1]))

                                    numb_clst_level[6][z]=np.sum(n_clst_level_sim0[:,1])
                                    numb_clst_level[7][z]=np.mean(n_clst_level_sim0[:,1])
                                    numb_clst_level[8][z]=np.std(n_clst_level_sim0[:,1], ddof=1)
                                    numb_clst_level[9][z]=np.size(n_clst_level_sim0[:,1])
                                    numb_clst_level[10][z]=np.std(n_clst_level_sim0[:,1], ddof=1)/np.sqrt(np.size(n_clst_level_sim0[:,1]))

                                    numb_clst_level[11][z]=np.sum(n_clst_level_sim1[:,1])
                                    numb_clst_level[12][z]=np.mean(n_clst_level_sim1[:,1])
                                    numb_clst_level[13][z]=np.std(n_clst_level_sim1[:,1], ddof=1)
                                    numb_clst_level[14][z]=np.size(n_clst_level_sim1[:,1])
                                    numb_clst_level[15][z]=np.std(n_clst_level_sim1[:,1], ddof=1)/np.sqrt(np.size(n_clst_level_sim1[:,1]))
                                    '''

                                    numb_clst_level[1][z] = sum_nclst_obs
                                    numb_clst_level[2][z] = avg_nclst_obs
                                    numb_clst_level[3][z] = std_n1clst_obs
                                    numb_clst_level[4][z] = sum2_nclst_obs
                                    numb_clst_level[5][z] = std_n1clst_obs/np.sqrt(nfrm)

                                    numb_clst_level[6][z] = sum_nclst_sim0
                                    numb_clst_level[7][z] = avg_nclst_sim0
                                    numb_clst_level[8][z] = std_n1clst_sim0
                                    numb_clst_level[9][z] = sum2_nclst_sim0
                                    numb_clst_level[10][z] = std_n1clst_sim0/np.sqrt(nfrm)

                                    numb_clst_level[11][z] = sum_nclst_sim1
                                    numb_clst_level[12][z] = avg_nclst_sim1
                                    numb_clst_level[13][z] = std_n1clst_sim1
                                    numb_clst_level[14][z] = sum2_nclst_sim1
                                    numb_clst_level[15][z] = std_n1clst_sim1/np.sqrt(nfrm)

                                    fl_clst_sum = open(file_clst_sum+'.csv','a')
                                    fl_clst_sum.write(str(z*2)+'\t'+str(numb_clst_level[1][z])+'\t'+str(numb_clst_level[2][z])+'\t'+str(numb_clst_level[3][z])+'\t'+str(numb_clst_level[4][z])+'\t'+str(numb_clst_level[5][z])+'\t'
                                                     +str(numb_clst_level[6][z])+'\t'+str(numb_clst_level[7][z])+'\t'+str(numb_clst_level[8][z])+'\t'+str(numb_clst_level[9][z])+'\t'+str(numb_clst_level[10][z])+'\t'
                                                     +str(numb_clst_level[11][z])+'\t'+str(numb_clst_level[12][z])+'\t'+str(numb_clst_level[13][z])+'\t'+str(numb_clst_level[14][z])+'\t'+str(numb_clst_level[15][z])+'\n')
                                    fl_clst_sum.close()

                                ####################################################################################################################
                                xlab = list((('Cluster Size', 'Mean ADC'),('Maximum ADC', 'Total ADC')))

                                if(data_bg_cut.ndim!=1 and data_bg_cut.shape[1]<11):
                                    r= np.max(np.array((np.max(data_cut_obs[:,7]), np.max(data_cut_sim_0[:,7]), np.max(data_cut_sim_1[:,7])  )))
                                    maxim[0,0]=r+1
                                    r= np.max(np.array((np.max(data_cut_obs[:,8]), np.max(data_cut_sim_0[:,8]), np.max(data_cut_sim_1[:,8])  )))
                                    maxim[0,1]= 1.1*r
                                    r= np.max(np.array((np.max(data_cut_obs[:,9]), np.max(data_cut_sim_0[:,9]), np.max(data_cut_sim_1[:,9])  )))
                                    maxim[1,1]= np.ceil(1.1*r)
                                    r= np.max(np.array((np.max(data_cut_obs[:,10]), np.max(data_cut_sim_0[:,10]), np.max(data_cut_sim_1[:,10])  )))
                                    maxim[1,0]= 1.1*r

                                if(data_bg_cut.ndim==1):
                                    r= np.max(np.array((np.max(data_cut_obs[:,7]), np.max(data_cut_sim_0[:,7]), np.max(data_cut_sim_1[:,7]), np.max(data_bg_cut[7])  )))
                                    maxim[0,0]=r+1
                                    r= np.max(np.array((np.max(data_cut_obs[:,8]), np.max(data_cut_sim_0[:,8]), np.max(data_cut_sim_1[:,8]), np.max(data_bg_cut[8])  )))
                                    maxim[0,1]= 1.1*r
                                    r= np.max(np.array((np.max(data_cut_obs[:,9]), np.max(data_cut_sim_0[:,9]), np.max(data_cut_sim_1[:,9]), np.max(data_bg_cut[9])  )))
                                    maxim[1,1]= np.ceil(1.1*r)
                                    r= np.max(np.array((np.max(data_cut_obs[:,10]), np.max(data_cut_sim_0[:,10]), np.max(data_cut_sim_1[:,10]), np.max(data_bg_cut[10])  )))
                                    maxim[1,0]= 1.1*r

                                if(data_bg_cut.ndim>1 and data_bg_cut.shape[1]==11):
                                    r= np.max(np.array((np.max(data_cut_obs[:,7]), np.max(data_cut_sim_0[:,7]), np.max(data_cut_sim_1[:,7]), np.max(data_bg_cut[:,7])  )))
                                    maxim[0,0]=r+1
                                    r= np.max(np.array((np.max(data_cut_obs[:,8]), np.max(data_cut_sim_0[:,8]), np.max(data_cut_sim_1[:,8]), np.max(data_bg_cut[:,8])  )))
                                    maxim[0,1]= 1.1*r
                                    r= np.max(np.array((np.max(data_cut_obs[:,9]), np.max(data_cut_sim_0[:,9]), np.max(data_cut_sim_1[:,9]), np.max(data_bg_cut[:,9])  )))
                                    maxim[1,1]= np.ceil(1.1*r)
                                    r= np.max(np.array((np.max(data_cut_obs[:,10]), np.max(data_cut_sim_0[:,10]), np.max(data_cut_sim_1[:,10]), np.max(data_bg_cut[:,10])  )))
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


                                    #if(densi==True): dens='_norm'
                                    #if(densi==False): dens=''

                                    for type_plt in ['cl_size', 'cl_mean', 'cl_etotal', 'cl_max_adc' ]:

                                        #hist_etotal, bins_datc1 = np.histogram(data_cut_obs[:,7], bins=nbins_datc1)

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

                                            maxic = np.max(np.array((np.max(data_cut_obs[:,7]),
                                                    np.max(data_cut_sim_0[:,7]), np.max(data_cut_sim_1[:,7])  )))+1
                                            minic = np.min(np.array((np.min(data_cut_obs[:,7]),
                                                    np.min(data_cut_sim_0[:,7]), np.min(data_cut_sim_1[:,7])  )))

                                            max_c = np.max(np.array((np.max(data_cut_obs[:,7]), np.max(data_cut_sim_0[:,7]), np.max(data_cut_sim_1[:,7])  )))
                                            if(np.max(data_cut_obs[:,7]) == max_c ):data_cut_c=np.append(data_cut_obs[:,7], data_cut_sim_0[:,7] )

                                            if(np.max(data_cut_sim_0[:,7]) == max_c ):data_cut_c=np.append(data_cut_obs[:,7], data_cut_sim_0[:,7] )
                                            if(np.max(data_cut_sim_1[:,7]) == max_c ):data_cut_c=np.append(data_cut_obs[:,7], data_cut_sim_1[:,7] )

                                            #nbins_clts=int(round(np.max(data_cut_obs[:,7] ))) #nbins_clts = maxim[0,0]
                                            nbins_clts=int(round(maxic-minic )) #nbins_clts = maxim[0,0]
                                            hist_clts, bins_clts = np.histogram(data_cut_c, bins=nbins_clts)
                                            cltslinbins_all = np.linspace( bins_clts[0], bins_clts[-1] ,len(bins_clts))

                                            wxc,xc = np.histogram(data_cut_obs[:,7], bins = cltslinbins_all, density = densi)
                                            wyc0,yc0 = np.histogram(data_cut_sim_0[:,7], bins = cltslinbins_all, density = densi)
                                            wyc1,yc1 = np.histogram(data_cut_sim_1[:,7], bins = cltslinbins_all, density = densi)

                                            ncb=1

                                            if(bin_hist>1):
                                                nbins_clts = bin_hist
                                                if(bin_hist > round(maxic-minic) ): nbins_clts = round(maxic-minic)
                                                #if(bin_hist > round(np.max(data_cut_obs[:,7] )) ): nbins_clts = round(np.max(data_cut_obs[:,7] ))

                                                #else: nbins_clts = bin_hist
                                            else:
                                                nbins_clts = round(maxic-minic)

                                            hist_clts, bins_clts = np.histogram(data_cut_c, bins=nbins_clts)
                                            cltslinbins = np.linspace( bins_clts[0], bins_clts[-1] ,len(bins_clts))

                                            hist_clts_obs, bins_clts_obs = np.histogram(data_cut_obs[:,7], bins=nbins_clts)
                                            cltslinbins_obs = np.linspace( bins_clts_obs[0], bins_clts_obs[-1] ,len(bins_clts))
                                            ########################################################################################################

                                            cltsbin_obs  = cltslinbins
                                            cltsbin_bg = cltslinbins
                                            #######################################
                                            cltsbin_sim_0= cltslinbins
                                            cltsbin_sim_1= cltslinbins
                                            #######################################

                                            wxc_bh, xc_bh  = np.histogram(data_cut_obs[:,7], bins = cltslinbins, density = densi)
                                            wyc_bh0,yc_bh0 = np.histogram(data_cut_sim_0[:,7], bins = cltslinbins, density = densi)
                                            wyc_bh1,yc_bh1 = np.histogram(data_cut_sim_1[:,7], bins = cltslinbins, density = densi)
                                            #########################################################################################
                                            count_obs, bin_obs = np.histogram(data_cut_obs[:,7], bins=cltsbin_obs)
                                            hist_clstr_siz_obs = {'bins': bin_obs, 'counts': count_obs}
                                            bincenters_obs = 0.5*(bin_obs[1:]+bin_obs[:-1])
                                            norm_obs = (np.sum(count_obs) * np.diff(bin_obs))
                                            #########################################################################################
                                            count_sim_0, bin_sim_0 = np.histogram(data_cut_sim_0[:,7], bins=cltsbin_sim_0)
                                            hist_clstr_siz_sim_0 = {'bins': bin_sim_0, 'counts': count_sim_0}
                                            bincenters_sim_0 = 0.5*(bin_sim_0[1:]+bin_sim_0[:-1])
                                            norm_sim_0 = (np.sum(count_sim_0) * np.diff(bin_sim_0))
                                            #########################################################################################
                                            count_sim_1, bin_sim_1 = np.histogram(data_cut_sim_1[:,7], bins=cltsbin_sim_1)
                                            hist_clstr_siz_sim_1 = {'bins': bin_sim_1, 'counts': count_sim_1}
                                            bincenters_sim_1 = 0.5*(bin_sim_1[1:]+bin_sim_1[:-1])
                                            norm_sim_1 = (np.sum(count_sim_1) * np.diff(bin_sim_1))
                                            #########################################################################################
                                            if(data_bg_cut.ndim>1):
                                                count_bg, bin_bg = np.histogram(data_bg_cut[:,7], bins=cltsbin_bg)
                                            if(data_bg_cut.ndim==1):
                                                count_bg, bin_bg = np.histogram(data_bg_cut[0], bins=cltsbin_bg)
                                            hist_clstr_siz_bg = {'bins': bin_bg, 'counts': count_bg}
                                            #########################################################################################
                                            #########################################################################################
                                            # Guardar en un archivo de texto,  .npy
                                            #hist_dat = np.load(dirsave_plt+'/'+'hist_adc_obs'+source+\
                                             #           sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', allow_pickle=True).item()
                                            np.save(dirsave_plt+'/'+'hist_clstr_siz_obs_'+source+\
                                                    sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist_clstr_siz_obs )
                                            np.save(dirsave_plt+'/'+'hist_clstr_siz_sim_'+source+\
                                                    sim_n+'_'+gauss_dist+'_0k_0a_0zff'+ sig_bg_thresh +simu_no_cut+ str_cuts + cut_clst_size+cut_max+'.npy', hist_clstr_siz_sim_0 )
                                            np.save(dirsave_plt+'/'+'hist_clstr_siz_sim_'+source+\
                                                    sim_n+'_'+difu_gauss+ sig_bg_thresh +simu_no_cut+ str_cuts + cut_clst_size+cut_max+'.npy', hist_clstr_siz_sim_1 )

                                            np.save(dirsave_plt+'/'+'hist_clstr_siz_bg_'+source+\
                                                    sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist_clstr_siz_bg )
                                            #########################################################################################

                                            if(densi==True):
                                                obs_c = count_obs/norm_obs
                                                sim_0_c = count_sim_0/norm_sim_0
                                                sim_1_c = count_sim_1/norm_sim_1
                                                err_std_obs_c = np.sqrt(count_obs) /norm_obs
                                                err_std_sim_0_c = (0.2*count_sim_0 + np.sqrt(count_sim_0) )/norm_sim_0
                                                err_std_sim_1_c = (0.2*count_sim_1+ np.sqrt(count_sim_1) )/norm_sim_1

                                            if(densi==False):
                                                obs_c = count_obs
                                                sim_0_c = count_sim_0
                                                sim_1_c = count_sim_1
                                                err_std_obs_c = np.sqrt(count_obs)
                                                err_std_sim_0_c = 0.2*count_sim_0 + np.sqrt(count_sim_0)
                                                err_std_sim_1_c = 0.2*count_sim_1 + np.sqrt(count_sim_1)

                                            #############################################################################################

                                            err_c0 = np.sqrt(err_std_obs_c*err_std_obs_c + err_std_sim_0_c*err_std_sim_0_c )
                                            ch2_c0 = chi_2_sigm_test(obs_c, sim_0_c, err_c0)
                                            tmp_str0 = 'Chi2 test: ' + r'%.2f'%ch2_c0[0] \
                                                    + '       '+'ndf: ' + r'%.2f'%ch2_c0[3] \
                                                    + '\n'+'Chi2/ndf : ' + r'%.2f'%ch2_c0[4] \
                                                    + '           '+'Clst_Sim/Obs: '+str(round(data_cut_sim_0[:,7].size/data_cut_obs[:,7].size,3))

                                            err_c1 = np.sqrt(err_std_obs_c*err_std_obs_c + err_std_sim_1_c*err_std_sim_1_c )
                                            ch2_c1 = chi_2_sigm_test(obs_c, sim_1_c, err_c1)
                                            tmp_str1 = 'Chi2 test: ' + r'%.2f'%ch2_c1[0] \
                                                    + '       '+'ndf: ' + r'%.2f'%ch2_c1[3] \
                                                    + '\n'+'Chi2/ndf : ' + r'%.2f'%ch2_c1[4] \
                                                    + '           '+'Clst_Sim/Obs: '+str(round(data_cut_sim_1[:,7].size/data_cut_obs[:,7].size,3))

                                            if(labl_opt=='full'):
                                                lbl_obs = source + ' data'+ sig_bg_thresh + str_cuts + cut_clst_size+cut_max + cut_adc
                                                lbl_sim0 = source + ' simulation '+ sig_bg_thresh_sim + cut_clst_size+cut_max + cut_adc\
                                                            +'\n'+'Chi2/ndf : ' + r'%.2f'%ch2_c0[4]
                                                lbl_sim1 = source + ' simulation '+ sig_bg_thresh_sim + cut_clst_size+cut_max + cut_adc \
                                                            + d_gauss + '\n'+'Chi2/ndf : ' + r'%.2f'%ch2_c1[4]
                                                lbl_bg = 'Background' + ' data'+ si_bg_thresh + cut_clst_size+cut_max + cut_adc
                                            if(labl_opt=='simple'):
                                                lbl_obs = source + ' data'
                                                lbl_sim0 = source + ' sim ' +r'$\chi^2_\nu$ : ' + r'%.2f'%ch2_c0[4] #+'+ bg'
                                                lbl_sim1 = source + ' sim ' + d_gauss + ' '+ r'$\chi^2_\nu$ : '  + r'%.2f'%ch2_c1[4]#+'+ bg'
                                                lbl_bg = 'Background' + ' data'
                                            if(labl_opt=='off'):
                                                lbl_obs = None
                                                lbl_sim0 = None
                                                lbl_sim1 = None
                                                lbl_bg = None

                                            if(c_dat_real==True ):
                                                #ax0.hist(data_cut_obs[:,7], bins=cltsbin, density=densi, histtype='step', label=source, color='C3', log=log_y, linewidth=l_width)
                                                #ax0.hist(data_cut_obs[:,7], bins=cltsbin_obs, density=densi, histtype='step'
                                                ax0.hist(bincenters_obs, weights=count_obs, bins=cltsbin_obs, density=densi, histtype='step'
                                                        #, label=source+'_'+ag+'ag'+'_'+str(sigma_obs)+'sig'+'_z_'+str(z*2)+'mm'+'_t'+tiemp  # + '_Clst_'+str(data_cut_obs[:,8].size)
                                                        , label =  lbl_obs + cut_max_clst_porc
                                                        , color='C3', log=log_y, linewidth=l_width+0.*l_width)

                                                if(densi==True):
                                                    ax0.errorbar(bincenters_obs, obs_c, yerr=err_std_obs_c
                                                                                , fmt='C3'+'.', ecolor='C3', linewidth=l_width)
                                                if(densi==False):
                                                    ax0.errorbar(bincenters_obs, count_obs, yerr=err_std_obs_c
                                                                                , fmt='C3'+'.', ecolor='C3', linewidth=l_width)

                                            if(c_dat_sim==True and plt_sim==True ):
                                                #######################################################################################################################################################################################################################
                                                #ax0.hist(data_cut_sim_0[:,7], bins=cltsbin_sim_0, density=densi, histtype='step'
                                                ax0.hist(bincenters_sim_0, weights=count_sim_0, bins=cltsbin_sim_0, density=densi, histtype='step'
                                                           # , label='Sim_'+source+'_'+str(sigma_sim)+'sig'+'_z_'+str(z*2)+'mm' +'_ev_' +ev+ '_sim_ipc' # + '_Clst_'+str(data_cut_sim_0[:,8].size)
                                                            , label = lbl_sim0 + cut_max_clst_porc_sim
                                                            , color='C2', log=log_y, linewidth=l_width*2/3)

                                                ##########################################################################
                                                if(densi==True):
                                                    ax0.errorbar(bincenters_sim_0, sim_0_c, yerr=err_std_sim_0_c
                                                                                , fmt='C2'+'.', ecolor='C2', linewidth=l_width*2/3)
                                                                                # , label = '\n P_ks = %.3f' % p_ks_c + '\n P_prmut = %.3f' % p_permut_c,)
                                                if(densi==False):
                                                    ax0.errorbar(bincenters_sim_0, count_sim_0, yerr= err_std_sim_0_c
                                                                                , fmt='C2'+'.', ecolor='C2', linewidth=l_width*2/3)#,label = tmp_str)
                                                ##############################################################################################################################################################################################################################

                                                #ax0.hist(data_cut_sim_1[:,7], bins=cltsbin_sim_1, density=densi, histtype='step'
                                                ax0.hist(bincenters_sim_1, weights=count_sim_1, bins=cltsbin_sim_1, density=densi, histtype='step'
                                                           # , label='Sim_'+source+'_'+str(sigma_sim)+'sig'+'_z_'+str(z*2)+'mm' +'_ev_' +ev+ '_sim_ipc' # + '_Clst_'+str(data_cut_sim_1[:,8].size)
                                                            , label = lbl_sim1 + cut_max_clst_porc_sim
                                                            , color='C9', log=log_y, linewidth=l_width*2/3)

                                                ##########################################################################
                                                if(densi==True):
                                                    ax0.errorbar(bincenters_sim_1, sim_1_c, yerr=err_std_sim_1_c
                                                                                , fmt='C9'+'.', ecolor='C9', linewidth=l_width*2/3)
                                                                                # , label = '\n P_ks = %.3f' % p_ks_c + '\n P_prmut = %.3f' % p_permut_c,)
                                                if(densi==False):
                                                    ax0.errorbar(bincenters_sim_1, count_sim_1, yerr= err_std_sim_1_c
                                                                                , fmt='C9'+'.', ecolor='C9', linewidth=l_width*2/3)#,label = tmp_str)
                                            ##############################################################################################################################################################################################################################
                                            ##############################################################################################################################################################################################################################

                                            if(c_bg_real==True ):

                                                #ax0.hist(data_bg_cut[:,7], bins=cltsbin, density=densi, histtype='step', label='backgnd', color='k', log=log_y, linewidth=l_width*2/3)
                                                if(data_bg_cut.ndim>1):
                                                    count_bg, bin_bg = np.histogram(data_bg_cut[:,7], bins=cltsbin_bg)
                                                    bincenters_bg = 0.5*(bin_bg[1:]+bin_bg[:-1])
                                                    ax0.hist(bincenters_bg, weights=count_bg, bins=cltsbin_bg, density=densi, histtype='step'
                                                               #, label='backgnd'+'_'+str(sigma_obs)+'sig'+' '
                                                                , label=lbl_bg
                                                                , color='k', log=log_y, linewidth=l_width*2/3)

                                                if(data_bg_cut.ndim==1):
                                                    count_bg, bin_bg = np.histogram(data_bg_cut[0], bins=cltsbin_bg)
                                                    bincenters_bg = 0.5*(bin_bg[1:]+bin_bg[:-1])
                                                    ax0.hist(bincenters_bg, weights=count_bg, bins=cltsbin_bg, density=densi, histtype='step'
                                                                , label=lbl_bg
                                                                , color='k', log=log_y, linewidth=l_width*2/3)

                                                #bincenters_bg = 0.5*(bin_bg[1:]+bin_bg[:-1])
                                                norm_bg = (np.sum(count_bg) * np.diff(bin_bg))
                                                err_std_bg = np.sqrt(count_bg)/ norm_bg
                                                if(densi==True):
                                                    ax0.errorbar(bincenters_bg, count_bg/norm_bg, yerr=err_std_bg, fmt='k'+'.', ecolor='k')
                                                if(densi==False):
                                                    ax0.errorbar(bincenters_bg, count_bg, yerr=np.sqrt(count_bg), fmt='k'+'.', ecolor='k')

                                            if(plt_obs==True):
                                                #fig.suptitle('Cluster Multiplicity Z='+str(z*2).zfill(2)+'mm '+ dens)
                                                #ax0.set_xlim(0,maxim[0,0]*mxs )
                                                if(yscal_cls>0):ax0.set_ylim(0, yscal_cls)
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

                                            ylabl_comp = r'$\frac{ (data -sim)}{\sigma}$'
                                            comp_sim, err_comp_sim = comp_obs_sim(wxc_bh, wyc_bh0)
                                            comp_sim1, err_comp_sim1 = comp_obs_sim(wxc_bh, wyc_bh1)
                                            delta_min = np.min(np.array((np.nanmin(comp_sim), np.nanmin(comp_sim1) )))
                                            delta_max = np.max(np.array((comp_sim[comp_sim < np.Inf].max(), comp_sim1[comp_sim1 < np.Inf].max() )))

                                            if(plt_sim==True and plot_nsig==True):
                                                #ax2.axhline(y = 0., color = 'k', linestyle = '--')
                                                ax2.hist(cltslinbins, weights=0*cltslinbins, bins = cltslinbins, histtype='step', ls='--', density=densi, color='k', linewidth=l_width*2/3 )
                                                ax2.errorbar(bincenters_obs, comp_sim, yerr=plt_err*err_comp_sim, fmt='C2'+'>', lw=2, markersize=12 )
                                                if(zoom_delta==True):ax2.set_ylim(delta_min,delta_max)

                                            if(plt_sim1==True and plot_nsig==True):
                                                #ax2.axhline(y = 0., color = 'k', linestyle = '--')
                                                ax2.hist(cltslinbins, weights=0*cltslinbins, bins = cltslinbins, histtype='step', ls='--', density=densi, color='k', linewidth=l_width*2/3 )
                                                ax2.errorbar(bincenters_obs, comp_sim1, yerr=plt_err*err_comp_sim1, fmt='C0'+'o', lw=2, markersize=10 )
                                                if(zoom_delta==True):ax2.set_ylim(delta_min,delta_max)

                                            if( (plt_sim1==True or plt_sim==True) and plot_nsig==True):
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

                                            ylabl_ratio= r'$ratio = \frac{data}{sim}$'
                                            ratio_sim0, err_ratio_sim0 = ratio_obs_sim(wxc_bh, wyc_bh0)
                                            ratio_sim1, err_ratio_sim1 = ratio_obs_sim(wxc_bh, wyc_bh1)
                                            ratio_sim1, err_ratio_sim1 = ratio_obs_sim(wxc_bh, wyc_bh1)
                                            ratio_min = np.min(np.array((np.nanmin(ratio_sim0), np.nanmin(ratio_sim1) )))
                                            ratio_max = np.max(np.array((ratio_sim0[ratio_sim0 < np.Inf].max(), ratio_sim1[ratio_sim1 < np.Inf].max() )))

                                            if(plt_sim==True and plot_ratio==True):
                                                #ax4.axhline(y = 0., color = 'k', linestyle = '--')
                                                ax4.hist(cltslinbins, weights=0*cltslinbins, bins = cltslinbins, histtype='step', ls='--', density=densi, color='k', linewidth=l_width*2/3 )
                                                ax4.errorbar(bincenters_obs, ratio_sim0, yerr=plt_err*err_ratio_sim0, fmt='C2'+'>', lw=2, markersize=12 )
                                                if(zoom_ratio==True):ax4.set_ylim(-0.5+ratio_min,ratio_max)

                                            if(plt_sim1==True and plot_ratio==True):
                                                #ax4.axhline(y = 0., color = 'k', linestyle = '--')
                                                ax4.hist(cltslinbins, weights=0*cltslinbins, bins = cltslinbins, histtype='step', ls='--', density=densi, color='k', linewidth=l_width*2/3 )
                                                ax4.errorbar(bincenters_obs, ratio_sim1, yerr=plt_err*err_ratio_sim1, fmt='C0'+'o', lw=2, markersize=10 )
                                                if(zoom_ratio==True):ax4.set_ylim(-0.5+ratio_min,ratio_max)

                                            if( (plt_sim1==True or plt_sim==True) and plot_ratio==True):
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
                                            #save_subplt = 'plot_hist'+dens+'_cluster_'+strbin +str(z*2).zfill(2)+'mm_sim' \
                                            #                +'_' +source+'_'+str(nsigm_bg)+'sig'+'_'+ev+'ev_'+fecha+str_hist_comp+cut_adc+cut_str
                                            save_subplt = 'plot_hist'+dens+'_cluster_'+strbin +str(z*2).zfill(2)+'mm' + name_cuts
                                            print(save_subplt)

                                            # Save just the portion _inside_ the second axis's boundaries
                                            #extent = ax0.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                                            if(plt_one==True):
                                                plt.savefig(dirsave_plt+'/'+ save_subplt +difu_gauss+'.pdf', dpi=150)#,  bbox_inches=extent.expanded(1.2, 1.3), dpi=150)
                                                plt.savefig(dirsave_plt+'/'+ save_subplt +difu_gauss+'.png', dpi=150)#,  bbox_inches=extent.expanded(1.2, 1.3), dpi=150)
                                                # plt.show()

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
                                            maxime = 1.1*np.min(np.array((np.max(data_cut_obs[:,8]),
                                                    np.max(data_cut_sim_0[:,8]), np.max(data_cut_sim_1[:,8])  )))
                                            minime = np.min(np.array((np.min(data_cut_obs[:,8]),
                                                    np.min(data_cut_sim_0[:,8]), np.min(data_cut_sim_1[:,8])  )))

                                            max_me = np.max(np.array((np.max(data_cut_obs[:,8]),
                                                    np.max(data_cut_sim_0[:,8]), np.max(data_cut_sim_1[:,8])  )))
                                            if( min_adc==adc_cut):data_cut = data_cut_obs
                                            else:
                                                if(np.max(data_cut_obs[:,8]) == max_me ):data_cut = data_cut_obs
                                                if(np.max(data_cut_sim_0[:,8]) == max_me ):data_cut = data_cut_sim_0
                                                if(np.max(data_cut_sim_1[:,8]) == max_me ):data_cut = data_cut_sim_1

                                            #nbins_mean=int(round(np.max(data_cut_obs[:,8] ))) #nbins_mean = maxim[0,1]
                                            nbins_mean=int(round(maxime-minime )) #nbins_mean = maxim[0,1]
                                            hist_mean, bins_mean = np.histogram(data_cut[:,8], bins=nbins_mean)
                                            meanlinbins_all = np.linspace( bins_mean[0], bins_mean[-1] ,len(bins_mean))

                                            wxm,xm = np.histogram(data_cut_obs[:,8], bins = meanlinbins_all, density = densi)
                                            wym0,ym0 = np.histogram(data_cut_sim_0[:,8], bins = meanlinbins_all, density = densi)
                                            wym1,ym1 = np.histogram(data_cut_sim_1[:,8], bins = meanlinbins_all, density = densi)

                                            nmb = 1 #16

                                            if(bin_hist>1):
                                                nbins_mean = bin_hist
                                                if(bin_hist > round(maxime-minime) ): nbins_mean = round(maxime-minime)
                                                #if(bin_hist >round(np.max(data_cut_obs[:,8] )) ): nbins_mean = round(np.max(data_cut_obs[:,8] ))

                                            else:
                                                nbins_mean = round(maxime-minime)

                                            hist_mean, bins_mean = np.histogram(data_cut[:,8], bins=nbins_mean)
                                            meanlinbins = np.linspace( bins_mean[0], bins_mean[-1] ,len(bins_mean))

                                            hist_mean_obs, bins_mean_obs = np.histogram(data_cut_obs[:,8], bins=nbins_mean)
                                            meanlinbins_obs = np.linspace( bins_mean_obs[0], bins_mean_obs[-1] ,len(bins_mean))
                                            ################################################

                                            meanbin_obs  = meanlinbins
                                            meanbin_bg = meanlinbins
                                            #######################################
                                            meanbin_sim_0 = meanlinbins
                                            meanbin_sim_1= meanlinbins
                                            ################################################

                                            wxm_bh,xm_bh = np.histogram(data_cut_obs[:,8], bins = meanlinbins, density = densi)
                                            wym_bh0,ym_bh0 = np.histogram(data_cut_sim_0[:,8], bins = meanlinbins, density = densi)
                                            wym_bh1,ym_bh1 = np.histogram(data_cut_sim_1[:,8], bins = meanlinbins, density = densi)

                                            #########################################################################################
                                            count_obs, bin_obs = np.histogram(data_cut_obs[:,8], bins=meanbin_obs)
                                            hist_clstr_mean_obs = {'bins': bin_obs, 'counts': count_obs}
                                            bincenters_obs = 0.5*(bin_obs[1:]+bin_obs[:-1])
                                            norm_obs = (np.sum(count_obs) * np.diff(bin_obs))
                                            #########################################################################################
                                            count_sim_0, bin_sim_0 = np.histogram(data_cut_sim_0[:,8], bins=meanbin_sim_0)
                                            hist_clstr_mean_sim_0 = {'bins': bin_sim_0, 'counts': count_sim_0}
                                            bincenters_sim_0 = 0.5*(bin_sim_0[1:]+bin_sim_0[:-1])
                                            norm_sim_0 = (np.sum(count_sim_0) * np.diff(bin_sim_0))
                                            #########################################################################################
                                            count_sim_1, bin_sim_1 = np.histogram(data_cut_sim_1[:,8], bins=meanbin_sim_1)
                                            hist_clstr_mean_sim_1 = {'bins': bin_sim_1, 'counts': count_sim_1}
                                            bincenters_sim_1 = 0.5*(bin_sim_1[1:]+bin_sim_1[:-1])
                                            norm_sim_1 = (np.sum(count_sim_1) * np.diff(bin_sim_1))
                                            #########################################################################################
                                            if(data_bg_cut.ndim>1):
                                                count_bg, bin_bg = np.histogram(data_bg_cut[:,8], bins=meanbin_bg)
                                            if(data_bg_cut.ndim==1):
                                                count_bg, bin_bg = np.histogram(data_bg_cut[1], bins=meanbin_bg)
                                            hist_clstr_mean_bg = {'bins': bin_bg, 'counts': count_bg}
                                            #########################################################################################
                                            #########################################################################################
                                            # Guardar en un archivo de texto,  .npy
                                            #hist_dat = np.load(dirsave_plt+'/'+'hist_adc_obs'+source+\
                                             #           sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', allow_pickle=True).item()
                                            np.save(dirsave_plt+'/'+'hist_clstr_mean_obs_'+source+\
                                                    sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist_clstr_mean_obs )
                                            np.save(dirsave_plt+'/'+'hist_clstr_mean_sim_'+source+\
                                                    sim_n+'_'+gauss_dist+'_0k_0a_0zff'+ sig_bg_thresh +simu_no_cut+ str_cuts + cut_clst_size+cut_max+'.npy', hist_clstr_mean_sim_0 )
                                            np.save(dirsave_plt+'/'+'hist_clstr_mean_sim_'+source+\
                                                    sim_n+'_'+difu_gauss+ sig_bg_thresh +simu_no_cut+ str_cuts + cut_clst_size+cut_max+'.npy', hist_clstr_mean_sim_1 )

                                            np.save(dirsave_plt+'/'+'hist_clstr_mean_bg_'+source+\
                                                    sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist_clstr_mean_bg )
                                            #########################################################################################

                                            if(densi==True):
                                                obs_m = count_obs/norm_obs
                                                sim_0_m = count_sim_0/norm_sim_0
                                                sim_1_m = count_sim_1/norm_sim_1
                                                err_std_obs_m = np.sqrt(count_obs)/ norm_obs
                                                err_std_sim_0_m = (0.2*count_sim_0 + np.sqrt(count_sim_0) )/norm_sim_0#+ np.sqrt(count_sim_0)
                                                err_std_sim_1_m = (0.2*count_sim_1 + np.sqrt(count_sim_1) )/norm_sim_1

                                            if(densi==False):
                                                obs_m = count_obs
                                                sim_0_m = count_sim_0
                                                sim_1_m = count_sim_1
                                                err_std_obs_m = np.sqrt(count_obs)
                                                err_std_sim_0_m = 0.2*count_sim_0 + np.sqrt(count_sim_0)
                                                err_std_sim_1_m = 0.2*count_sim_1 + np.sqrt(count_sim_1)

                                            #############################################################################################

                                            err_m0 = np.sqrt(err_std_obs_m*err_std_obs_m + err_std_sim_0_m*err_std_sim_0_m )
                                            ch2_m0 = chi_2_sigm_test(obs_m, sim_0_m, err_m0)
                                            tmp_str0 = 'Chi2 test: ' + r'%.2f'%ch2_m0[0] \
                                                    + '       '+'ndf: ' + r'%.2f'%ch2_m0[3] \
                                                    + '\n'+'Chi2/ndf : ' + r'%.2f'%ch2_m0[4] \
                                                    + '           '+'Clst_Sim/Obs: '+str(round(data_cut_sim_0[:,8].size/data_cut_obs[:,8].size,3))

                                            err_m1 = np.sqrt(err_std_obs_m*err_std_obs_m + err_std_sim_1_m*err_std_sim_1_m )
                                            ch2_m1 = chi_2_sigm_test(obs_m, sim_1_m, err_m1)
                                            tmp_str1 = 'Chi2 test: ' + r'%.2f'%ch2_m1[0] \
                                                    + '       '+'ndf: ' + r'%.2f'%ch2_m1[3] \
                                                    + '\n'+'Chi2/ndf : ' + r'%.2f'%ch2_m1[4] \
                                                    + '           '+'Clst_Sim/Obs: '+str(round(data_cut_sim_1[:,8].size/data_cut_obs[:,8].size,3))

                                            if(labl_opt=='full'):
                                                lbl_obs = source + ' data'+ sig_bg_thresh + str_cuts + cut_clst_size+cut_max + cut_adc
                                                lbl_sim0 = source + ' simulation '+ sig_bg_thresh_sim + cut_clst_size+cut_max + cut_adc\
                                                            +'\n'+'Chi2/ndf : ' + r'%.2f'%ch2_m0[4]
                                                lbl_sim1 = source + ' simulation '+ sig_bg_thresh_sim + cut_clst_size+cut_max + cut_adc \
                                                            + d_gauss + '\n'+'Chi2/ndf : ' + r'%.2f'%ch2_m1[4]
                                                lbl_bg = 'Background' + ' data'+ si_bg_thresh + cut_clst_size+cut_max + cut_adc
                                            if(labl_opt=='simple'):
                                                lbl_obs = source + ' data'
                                                lbl_sim0 = source + ' sim ' +r'$\chi^2_\nu$ : ' + r'%.2f'%ch2_m0[4] #+'+ bg'
                                                lbl_sim1 = source + ' sim ' + d_gauss + ' '+ r'$\chi^2_\nu$ : '  + r'%.2f'%ch2_m1[4]#+'+ bg'
                                                lbl_bg = 'Background' + ' data'
                                            if(labl_opt=='off'):
                                                lbl_obs = None
                                                lbl_sim0 = None
                                                lbl_sim1 = None
                                                lbl_bg = None

                                            if(c_dat_real==True):
                                                #ax1.hist(data_cut_obs[:,8], bins=meanbin_obs, density=densi, histtype='step'
                                                ax1.hist(bincenters_obs, weights=count_obs, bins=meanbin_obs, density=densi, histtype='step'
                                                            #, label=source+'_'+ag+'ag'+'_'+str(sigma_obs)+'sig'+'_z_'+str(z*2)+'mm'+'_t'+tiemp  # + '_Clst_'+str(data_cut_obs[:,8].size)
                                                            , label =  lbl_obs + cut_max_clst_porc
                                                            , color='C3', log=log_y, linewidth=l_width+0.*l_width)

                                                if(densi==True):
                                                    ax1.errorbar(bincenters_obs, obs_m, yerr=err_std_obs_m
                                                                                , fmt='C3'+'.', ecolor='C3', linewidth=l_width)
                                                if(densi==False):
                                                    ax1.errorbar(bincenters_obs, count_obs, yerr=err_std_obs_m
                                                                                , fmt='C3'+'.', ecolor='C3', linewidth=l_width)

                                            if(c_dat_sim==True and plt_sim==True ):
                                                ##############################################################################################################################################################################################################################
                                                #ax1.hist(data_cut_sim_0[:,8], bins=meanbin_sim_0, density=densi, histtype='step'
                                                ax1.hist(bincenters_sim_0, weights=count_sim_0, bins=meanbin_sim_0, density=densi, histtype='step'
                                                            #, label='Sim_'+source+'_'+str(sigma_sim)+'sig'+'_z_'+str(z*2)+'mm' +'_ev_' +ev+ '_sim_ipc' # + '_Clst_'+str(data_cut_sim_0[:,8].size)
                                                            , label = lbl_sim0 + cut_max_clst_porc_sim
                                                            , color='C2', log=log_y, linewidth=l_width*2/3)

                                                if(densi==True):
                                                    ax1.errorbar(bincenters_sim_0, sim_0_m, yerr=err_std_sim_0_m
                                                                                , fmt='C2'+'.', ecolor='C2', linewidth=l_width*2/3)
                                                                                  #, label = '\n P_ks = %.3f' % p_ks_m + '\n P_prmut = %.3f' % p_permut_m,)
                                                if(densi==False):
                                                    ax1.errorbar(bincenters_sim_0, count_sim_0, yerr= err_std_sim_0_m
                                                                                , fmt='C2'+'.', ecolor='C2', linewidth=l_width*2/3) #,label = tmp_str)
                                                ##############################################################################################################################################################################################################################

                                                #ax0.hist(data_cut_sim_1[:,8], bins=meanbin_sim_1, density=densi, histtype='step'
                                                ax1.hist(bincenters_sim_1, weights=count_sim_1, bins=meanbin_sim_1, density=densi, histtype='step'
                                                           # , label='Sim_'+source+'_'+str(sigma_sim)+'sig'+'_z_'+str(z*2)+'mm' +'_ev_' +ev+ '_sim_ipc' # + '_Clst_'+str(data_cut_sim_1[:,8].size)
                                                            , label = lbl_sim1 + cut_max_clst_porc_sim
                                                            , color='C9', log=log_y, linewidth=l_width*2/3)

                                                ##########################################################################
                                                if(densi==True):
                                                    ax1.errorbar(bincenters_sim_1, sim_1_m, yerr=err_std_sim_1_m
                                                                                , fmt='C9'+'.', ecolor='C9', linewidth=l_width*2/3)
                                                                                # , label = '\n P_ks = %.3f' % p_ks_m + '\n P_prmut = %.3f' % p_permut_m,)
                                                if(densi==False):
                                                    ax1.errorbar(bincenters_sim_1, count_sim_1, yerr= err_std_sim_1_m
                                                                                , fmt='C9'+'.', ecolor='C9', linewidth=l_width*2/3)#,label = tmp_str)
                                            ##############################################################################################################################################################################################################################
                                            ##############################################################################################################################################################################################################################

                                            if(c_bg_real==True ):

                                                if(data_bg_cut.ndim>1):
                                                    count_bg, bin_bg = np.histogram(data_bg_cut[:,8], bins=meanbin_bg)
                                                    bincenters_bg = 0.5*(bin_bg[1:]+bin_bg[:-1])
                                                    ax1.hist(bincenters_bg, weights=count_bg, bins=meanbin_bg, density=densi, histtype='step'
                                                               , label=lbl_bg
                                                               , color='k', log=log_y, linewidth=l_width*2/3)

                                                if(data_bg_cut.ndim==1):
                                                    count_bg, bin_bg = np.histogram(data_bg_cut[1], bins=meanbin_bg)
                                                    bincenters_bg = 0.5*(bin_bg[1:]+bin_bg[:-1])
                                                    ax1.hist(bincenters_bg, weights=count_bg, bins=meanbin_bg, density=densi, histtype='step'
                                                               , label=lbl_bg
                                                               , color='k', log=log_y, linewidth=l_width*2/3)

                                                bincenters_bg = 0.5*(bin_bg[1:]+bin_bg[:-1])
                                                norm_bg = (np.sum(count_bg) * np.diff(bin_bg))
                                                err_std_bg = np.sqrt(count_bg)/ norm_bg
                                                if(densi==True):
                                                    ax1.errorbar(bincenters_bg, count_bg/norm_bg, yerr=err_std_bg, fmt='k'+'.', ecolor='k')
                                                if(densi==False):
                                                    ax1.errorbar(bincenters_bg, count_bg, yerr=np.sqrt(count_bg), fmt='k'+'.', ecolor='k')


                                            #ax1.axvline(x = 1023, color = 'k', linewidth=l_width+0.5*l_width, label = 'sat_cam: 1023 ADC = ' + str(sat_cam)+' keV')

                                            if(plt_obs==True):
                                                #fig.suptitle('Cluster Mean for '+source+' Z='+str(z*2).zfill(2)+'mm '+ dens)
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

                                            ylabl_comp = r'$\frac{ (data -sim)}{\sigma}$'
                                            comp_sim, err_comp_sim = comp_obs_sim(wxm_bh, wym_bh0)
                                            comp_sim1, err_comp_sim1 = comp_obs_sim(wxm_bh, wym_bh1)
                                            delta_min = np.min(np.array((np.nanmin(comp_sim), np.nanmin(comp_sim1) )))
                                            delta_max = np.max(np.array((comp_sim[comp_sim < np.Inf].max(), comp_sim1[comp_sim1 < np.Inf].max() )))

                                            if(plt_sim==True and plot_nsig==True):
                                                #ax3.axhline(y = 0., color = 'k', linestyle = '--')
                                                ax3.hist(meanlinbins, weights=0*meanlinbins, bins = meanlinbins, histtype='step', ls='--', density=densi, color='k', linewidth=l_width*2/3 )
                                                ax3.errorbar(bincenters_obs, comp_sim, yerr=plt_err*err_comp_sim, fmt='C2'+'>', lw=2, markersize=12 )
                                                if(zoom_delta==True):ax3.set_ylim(delta_min,delta_max)

                                            if(plt_sim1==True and plot_nsig==True):
                                                #ax3.axhline(y = 0., color = 'k', linestyle = '--')
                                                ax3.hist(meanlinbins, weights=0*meanlinbins, bins = meanlinbins, histtype='step', ls='--', density=densi, color='k', linewidth=l_width*2/3 )
                                                ax3.errorbar(bincenters_obs, comp_sim1, yerr=plt_err*err_comp_sim1, fmt='C0'+'o', lw=2, markersize=10 )
                                                if(zoom_delta==True):ax3.set_ylim(delta_min,delta_max)

                                            if( (plt_sim1==True or plt_sim==True) and plot_nsig==True):
                                                #ax3.xticks(size=font_siz+2)
                                                #ax3.yticks(size=font_siz+2)
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

                                            ylabl_ratio= r'$ratio = \frac{data}{sim}$'
                                            ratio_sim0, err_ratio_sim0 = ratio_obs_sim(wxm_bh, wym_bh0)
                                            ratio_sim1, err_ratio_sim1 = ratio_obs_sim(wxm_bh, wym_bh1)
                                            ratio_sim1, err_ratio_sim1 = ratio_obs_sim(wxm_bh, wym_bh1)
                                            ratio_min = np.min(np.array((np.nanmin(ratio_sim0), np.nanmin(ratio_sim1) )))
                                            ratio_max = np.max(np.array((ratio_sim0[ratio_sim0 < np.Inf].max(), ratio_sim1[ratio_sim1 < np.Inf].max() )))

                                            if(plt_sim==True and plot_ratio==True):
                                                #ax5.axhline(y = 0., color = 'k', linestyle = '--')
                                                ax5.hist(meanlinbins, weights=0*meanlinbins, bins = meanlinbins, histtype='step', ls='--', density=densi, color='k', linewidth=l_width*2/3 )
                                                ax5.errorbar(bincenters_obs, ratio_sim0, yerr=plt_err*err_ratio_sim0, fmt='C2'+'>', lw=2, markersize=12 )
                                                if(zoom_ratio==True):ax5.set_ylim(-0.5+ratio_min,ratio_max)

                                            if(plt_sim1==True and plot_ratio==True):
                                                #ax5.axhline(y = 0., color = 'k', linestyle = '--')
                                                ax5.hist(meanlinbins, weights=0*meanlinbins, bins = meanlinbins, histtype='step', ls='--', density=densi, color='k', linewidth=l_width*2/3 )
                                                ax5.errorbar(bincenters_obs, ratio_sim1, yerr=plt_err*err_ratio_sim1, fmt='C0'+'o', lw=2, markersize=10 )
                                                if(zoom_ratio==True):ax5.set_ylim(-0.5+ratio_min,ratio_max)

                                            if( (plt_sim1==True or plt_sim==True) and plot_ratio==True):
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
                                            #save_subplt = 'plot_hist'+dens+'_mean_'+strbin +str(z*2).zfill(2)+'mm_sim'+'_'+source+'_'+str(nsigm_bg)+'sig'+'_'+ev+'ev_'+fecha+str_hist_comp+cut_adc+cut_str
                                            save_subplt = 'plot_hist'+dens+'_mean_'+strbin +str(z*2).zfill(2)+'mm'+ name_cuts
                                            print(save_subplt)
                                            # Save just the portion _inside_ the second axis's boundaries
                                            #extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                                            if(plt_one==True):
                                                plt.savefig(dirsave_plt+'/'+ save_subplt+difu_gauss+'.pdf', dpi=150)#,  bbox_inches=extent.expanded(1.2, 1.3))
                                                plt.savefig(dirsave_plt+'/'+ save_subplt+difu_gauss+'.png', dpi=150)#,  bbox_inches=extent.expanded(1.2, 1.3))
                                                # plt.show()

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
                                            maxis = np.min(np.array((np.max(data_cut_obs[:,10]),
                                                    np.max(data_cut_sim_0[:,10]), np.max(data_cut_sim_1[:,10])  )))+1
                                            minis = np.min(np.array((np.min(data_cut_obs[:,10]),
                                                    np.min(data_cut_sim_0[:,10]), np.min(data_cut_sim_1[:,10]) )))

                                            max_s = np.max(np.array((np.max(data_cut_obs[:,10]),
                                                    np.max(data_cut_sim_0[:,10]), np.max(data_cut_sim_1[:,10])  )))
                                            if( min_adc==adc_cut):data_cut = data_cut_obs
                                            else:
                                                if(np.max(data_cut_obs[:,10]) == max_s ):data_cut = data_cut_obs
                                                if(np.max(data_cut_sim_0[:,10]) == max_s ):data_cut = data_cut_sim_0
                                                if(np.max(data_cut_sim_1[:,10]) == max_s ):data_cut = data_cut_sim_1

                                            nbins_seed=int(round(maxis-minis ))
                                            #nbins_seed=int(round(np.max(data_cut_obs[:,10] ))) #nbins_seed = maxim[1,0]
                                            hist_seed, bins_seed = np.histogram(data_cut[:,10], bins=nbins_seed)
                                            seedlinbins_all = np.linspace( bins_seed[0], bins_seed[-1] ,len(bins_seed))
                                            wxs,xs = np.histogram(data_cut_obs[:,10], bins = seedlinbins_all, density = densi)
                                            wys0,ys0 = np.histogram(data_cut_sim_0[:,10], bins = seedlinbins_all, density = densi)
                                            wys1,ys1 = np.histogram(data_cut_sim_1[:,10], bins = seedlinbins_all, density = densi)

                                            nsb=1#16

                                            if(bin_hist>1):
                                                nbins_seed = bin_hist
                                                if(bin_hist > round(maxis-minis ) ): nbins_seed = round(maxis-minis )

                                            else:
                                                nbins_seed = round(maxis-minis ) #nbins_seed = maxim[1,0]

                                            hist_seed, bins_seed = np.histogram(data_cut[:,10], bins=nbins_seed)
                                            seedlinbins = np.linspace( bins_seed[0], bins_seed[-1] ,len(bins_seed))

                                            hist_seed_obs, bins_seed_obs = np.histogram(data_cut_obs[:,10], bins=nbins_seed)
                                            seedlinbin_obss = np.linspace( bins_seed_obs[0], bins_seed_obs[-1] ,len(bins_seed))

                                             ################################################

                                            seedbin_obs  = seedlinbins
                                            seedbin_bg = seedlinbins
                                            #######################################
                                            seedbin_sim_0 = seedlinbins
                                            seedbin_sim_1 = seedlinbins
                                            #######################################

                                            wxs_bh,xs_bh = np.histogram(data_cut_obs[:,10], bins = seedlinbins, density = densi)
                                            wys_bh0,ys_bh0 = np.histogram(data_cut_sim_0[:,10], bins = seedlinbins, density = densi)
                                            wys_bh1,ys_bh1 = np.histogram(data_cut_sim_1[:,10], bins = seedlinbins, density = densi)

                                            #########################################################################################
                                            count_obs, bin_obs = np.histogram(data_cut_obs[:,10], bins=seedbin_obs)
                                            hist_clstr_max_obs = {'bins': bin_obs, 'counts': count_obs}
                                            bincenters_obs = 0.5*(bin_obs[1:]+bin_obs[:-1])
                                            norm_obs = (np.sum(count_obs) * np.diff(bin_obs))
                                            #########################################################################################
                                            count_sim_0, bin_sim_0 = np.histogram(data_cut_sim_0[:,10], bins=seedbin_sim_0)
                                            hist_clstr_max_sim_0 = {'bins': bin_sim_0, 'counts': count_sim_0}
                                            bincenters_sim_0 = 0.5*(bin_sim_0[1:]+bin_sim_0[:-1])
                                            norm_sim_0 = (np.sum(count_sim_0) * np.diff(bin_sim_0))
                                            #########################################################################################
                                            count_sim_1, bin_sim_1 = np.histogram(data_cut_sim_1[:,10], bins=seedbin_sim_1)
                                            hist_clstr_max_sim_1 = {'bins': bin_sim_1, 'counts': count_sim_1}
                                            bincenters_sim_1 = 0.5*(bin_sim_1[1:]+bin_sim_1[:-1])
                                            norm_sim_1 = (np.sum(count_sim_1) * np.diff(bin_sim_1))
                                            #########################################################################################
                                            if(data_bg_cut.ndim>1):
                                                count_bg, bin_bg = np.histogram(data_bg_cut[:,10], bins=seedbin_bg)
                                            if(data_bg_cut.ndim==1):
                                                count_bg, bin_bg = np.histogram(data_bg_cut[3], bins=seedbin_bg)
                                            hist_clstr_max_bg = {'bins': bin_bg, 'counts': count_bg}
                                            #########################################################################################
                                            #########################################################################################
                                            # Guardar en un archivo de texto,  .npy
                                            #hist_dat = np.load(dirsave_plt+'/'+'hist_adc_obs'+source+\
                                             #           sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', allow_pickle=True).item()
                                            np.save(dirsave_plt+'/'+'hist_clstr_max_obs_'+source+\
                                                    sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist_clstr_max_obs )
                                            np.save(dirsave_plt+'/'+'hist_clstr_max_sim_'+source+\
                                                    sim_n+'_'+gauss_dist+'_0k_0a_0zff'+ sig_bg_thresh +simu_no_cut+ str_cuts + cut_clst_size+cut_max+'.npy', hist_clstr_max_sim_0 )
                                            np.save(dirsave_plt+'/'+'hist_clstr_max_sim_'+source+\
                                                    sim_n+'_'+difu_gauss+ sig_bg_thresh +simu_no_cut+ str_cuts + cut_clst_size+cut_max+'.npy', hist_clstr_max_sim_1 )

                                            np.save(dirsave_plt+'/'+'hist_clstr_max_bg_'+source+\
                                                    sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist_clstr_max_bg )
                                            #########################################################################################

                                            if(densi==True):
                                                obs_s = count_obs/norm_obs
                                                sim_0_s = count_sim_0/norm_sim_0
                                                sim_1_s = count_sim_1/norm_sim_1

                                                err_std_obs_s = np.sqrt(count_obs)/ norm_obs
                                                err_std_sim_0_s = 0.2*count_sim_0/norm_sim_0 + np.sqrt(count_sim_0)/norm_sim_0
                                                err_std_sim_1_s = 0.2*count_sim_1/norm_sim_1 + np.sqrt(count_sim_1)/norm_sim_1

                                            if(densi==False):
                                                obs_s = count_obs
                                                sim_0_s = count_sim_0
                                                sim_1_s = count_sim_1

                                                err_std_obs_s = np.sqrt(count_obs)
                                                err_std_sim_0_s = 0.2*count_sim_0 + np.sqrt(count_sim_0)
                                                err_std_sim_1_s = 0.2*count_sim_1 + np.sqrt(count_sim_1)

                                            ##############################################################################################

                                            err_s0 = np.sqrt(err_std_obs_s*err_std_obs_s + err_std_sim_0_s*err_std_sim_0_s )
                                            ch2_s0 = chi_2_sigm_test(obs_s, sim_0_s, err_s0)
                                            tmp_str0 = 'Chi2 test: ' + r'%.2f'%ch2_s0[0] \
                                                    + '       '+'ndf: ' + r'%.2f'%ch2_s0[3] \
                                                    + '\n'+'Chi2/ndf : ' + r'%.2f'%ch2_s0[4] \
                                                    + '           '+'Clst_Sim/Obs: '+str(round(data_cut_sim_0[:,10].size/data_cut_obs[:,10].size,3))

                                            err_s1 = np.sqrt(err_std_obs_s*err_std_obs_s + err_std_sim_1_s*err_std_sim_1_s )
                                            ch2_s1 = chi_2_sigm_test(obs_s, sim_1_s, err_s1)
                                            tmp_str1 = 'Chi2 test: ' + r'%.2f'%ch2_s1[0] \
                                                    + '       '+'ndf: ' + r'%.2f'%ch2_s1[3] \
                                                    + '\n'+'Chi2/ndf : ' + r'%.2f'%ch2_s1[4] \
                                                    + '           '+'Clst_Sim/Obs: '+str(round(data_cut_sim_1[:,10].size/data_cut_obs[:,10].size,3))

                                            if(labl_opt=='full'):
                                                lbl_obs = source + ' data'+ sig_bg_thresh + str_cuts + cut_clst_size+cut_max + cut_adc
                                                lbl_sim0 = source + ' simulation '+ sig_bg_thresh_sim + cut_clst_size+cut_max + cut_adc\
                                                            +'\n'+'Chi2/ndf : ' + r'%.2f'%ch2_s0[4]
                                                lbl_sim1 = source + ' simulation '+ sig_bg_thresh_sim + cut_clst_size+cut_max + cut_adc \
                                                            + d_gauss + '\n'+'Chi2/ndf : ' + r'%.2f'%ch2_s1[4]

                                                lbl_bg = 'Background' + ' data'+ si_bg_thresh + cut_clst_size+cut_max + cut_adc
                                            if(labl_opt=='simple'):
                                                lbl_obs = source + ' data'
                                                lbl_sim0 = source + ' sim ' +r'$\chi^2_\nu$ : ' + r'%.2f'%ch2_s0[4] #+'+ bg'
                                                lbl_sim1 = source + ' sim ' + d_gauss + ' '+ r'$\chi^2_\nu$ : '  + r'%.2f'%ch2_s1[4]#+'+ bg'

                                                lbl_bg = 'Background' + ' data'
                                            if(labl_opt=='off'):
                                                lbl_obs = None
                                                lbl_sim0 = None
                                                lbl_sim1 = None
                                                lbl_bg = None

                                            if(c_dat_real==True):
                                                ax6.hist(bincenters_obs, weights=count_obs, bins=seedbin_obs, density=densi, histtype='step'
                                                           # , label=source+'_'+ag+'ag'+'_'+str(sigma_obs)+'sig'+'_z_'+str(z*2)+'mm'+'_t'+tiemp  # + '_Clst_'+str(data_cut_obs[:,8].size)
                                                           , label =  lbl_obs + cut_max_clst_porc
                                                            , color='C3', log=log_y, linewidth=l_width+0.*l_width)

                                                if(densi==True):
                                                    ax6.errorbar(bincenters_obs, obs_s, yerr=err_std_obs_s
                                                                                , fmt='C3'+'.', ecolor='C3', linewidth=l_width)
                                                if(densi==False):
                                                    ax6.errorbar(bincenters_obs, count_obs, yerr=err_std_obs_s
                                                                                , fmt='C3'+'.', ecolor='C3', linewidth=l_width)

                                            if(c_dat_sim==True and plt_sim==True ):
                                                ##############################################################################################################################################################################################################################
                                                ax6.hist(bincenters_sim_0, weights=count_sim_0, bins=seedbin_sim_0, density=densi, histtype='step'
                                                            #, label='Sim_'+source+'_'+str(sigma_sim)+'sig'+'_z_'+str(z*2)+'mm' +'_ev_' +ev+ '_sim_ipc' # + '_Clst_'+str(data_cut_sim_0[:,8].size)
                                                            , label = lbl_sim0 + cut_max_clst_porc_sim
                                                            , color='C2', log=log_y, linewidth=l_width*2/3)

                                                if(densi==True):
                                                    ax6.errorbar(bincenters_sim_0, sim_0_s, yerr=err_std_sim_0_s
                                                                                , fmt='C2'+'.', ecolor='C2', linewidth=l_width*2/3)
                                                                                # , label = '\n P_ks = %.3f' % p_ks_s + '\n P_prmut = %.3f' % p_permut_s,)
                                                if(densi==False):
                                                    ax6.errorbar(bincenters_sim_0, count_sim_0, yerr= err_std_sim_0_s
                                                                                , fmt='C2'+'.', ecolor='C2', linewidth=l_width*2/3) #,label = tmp_str)
                                                ##############################################################################################################################################################################################################################
                                                ax6.hist(bincenters_sim_1, weights=count_sim_1, bins=seedbin_sim_1, density=densi, histtype='step'
                                                           # , label='Sim_'+source+'_'+str(sigma_sim)+'sig'+'_z_'+str(z*2)+'mm' +'_ev_' +ev+ '_sim_ipc' # + '_Clst_'+str(data_cut_sim_1[:,10].size)
                                                            , label = lbl_sim1 + cut_max_clst_porc_sim
                                                            , color='C9', log=log_y, linewidth=l_width*2/3)

                                                ##########################################################################
                                                if(densi==True):
                                                    ax6.errorbar(bincenters_sim_1, sim_1_s, yerr=err_std_sim_1_s
                                                                                , fmt='C9'+'.', ecolor='C9', linewidth=l_width*2/3)
                                                                                # , label = '\n P_ks = %.3f' % p_ks_s + '\n P_prmut = %.3f' % p_permut_s,)
                                                if(densi==False):
                                                    ax6.errorbar(bincenters_sim_1, count_sim_1, yerr= err_std_sim_1_s
                                                                                , fmt='C9'+'.', ecolor='C9', linewidth=l_width*2/3)#,label = tmp_str)
                                                ##############################################################################################################################################################################################################################                      ##############################################################################################################################################################################################################################

                                            if(c_bg_real==True ):

                                                if(data_bg_cut.ndim>1):
                                                    count_bg, bin_bg = np.histogram(data_bg_cut[:,10], bins=seedbin_bg)
                                                    bincenters_bg = 0.5*(bin_bg[1:]+bin_bg[:-1])
                                                    ax6.hist(bincenters_bg, weights=count_bg, bins=seedbin_bg, density=densi, histtype='step'
                                                               , label=lbl_bg
                                                               , color='k', log=log_y, linewidth=l_width*2/3)

                                                if(data_bg_cut.ndim==1):
                                                    count_bg, bin_bg = np.histogram(data_bg_cut[3], bins=seedbin_bg)
                                                    bincenters_bg = 0.5*(bin_bg[1:]+bin_bg[:-1])
                                                    ax6.hist(bincenters_bg, weights=count_bg, bins=seedbin_bg, density=densi, histtype='step'
                                                               , label=lbl_bg
                                                               , color='k', log=log_y, linewidth=l_width*2/3)

                                                bincenters_bg = 0.5*(bin_bg[1:]+bin_bg[:-1])
                                                norm_bg = (np.sum(count_bg) * np.diff(bin_bg))
                                                err_std_bg = np.sqrt(count_bg)/ norm_bg
                                                if(densi==True):
                                                    ax6.errorbar(bincenters_bg, count_bg/norm_bg, yerr=err_std_bg, fmt='k'+'.', ecolor='k')
                                                if(densi==False):
                                                    ax6.errorbar(bincenters_bg, count_bg, yerr=np.sqrt(count_bg), fmt='k'+'.', ecolor='k')

                                            #ax6.axvline(x = 1023, color = 'k', linewidth=l_width+0.5*l_width, label = 'sat_cam: 1023 ADC = ' + str(sat_cam)+' keV')

                                            if(plt_obs==True):
                                                #fig.suptitle('Seed for '+source+' Z='+str(z*2).zfill(2)+'mm '+ dens)
                                                #ax6.set_xlim(0,maxim[1,0]*mxs )
                                                if(yscal_adc>0):ax6.set_ylim(0, yscal_max)
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

                                            if plot_nsig==True :
                                                ylabl_comp = r'$\frac{ (data -sim)}{\sigma}$'
                                                comp_sim, err_comp_sim = comp_obs_sim(wxs_bh, wys_bh0)
                                                comp_sim1, err_comp_sim1 = comp_obs_sim(wxs_bh, wys_bh1)
                                                delta_min = np.min(np.array((np.nanmin(comp_sim), np.nanmin(comp_sim1) )))
                                                delta_max = np.max(np.array((comp_sim[comp_sim < np.Inf].max(), comp_sim1[comp_sim1 < np.Inf].max() )))

                                            if(plt_sim==True and plot_nsig==True):
                                                #ax8.axhline(y = 0., color = 'k', linestyle = '--')
                                                ax8.hist(seedlinbins, weights=0*seedlinbins, bins = seedlinbins, histtype='step', ls='--', density=densi, color='k', linewidth=l_width*2/3 )
                                                ax8.errorbar(bincenters_obs, comp_sim, yerr=plt_err*err_comp_sim, fmt='C2'+'>', lw=2, markersize=12 )
                                                if(zoom_delta==True):ax8.set_ylim(delta_min,delta_max)

                                            if(plt_sim1==True and plot_nsig==True):
                                                #ax8.axhline(y = 0., color = 'k', linestyle = '--')
                                                ax8.hist(seedlinbins, weights=0*seedlinbins, bins = seedlinbins, histtype='step', ls='--', density=densi, color='k', linewidth=l_width*2/3 )
                                                ax8.errorbar(bincenters_obs, comp_sim1, yerr=plt_err*err_comp_sim1, fmt='C0'+'o', lw=2, markersize=10 )
                                                if(zoom_delta==True):ax8.set_ylim(delta_min,delta_max)

                                            if( (plt_sim1==True or plt_sim==True) and plot_nsig==True):
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


                                            if plot_ratio==True:
                                                ylabl_ratio= r'$ratio = \frac{data}{sim}$'
                                                ratio_sim0, err_ratio_sim0 = ratio_obs_sim(wxm_bh, wys_bh0)
                                                ratio_sim1, err_ratio_sim1 = ratio_obs_sim(wxm_bh, wys_bh1)
                                                ratio_sim1, err_ratio_sim1 = ratio_obs_sim(wxs_bh, wys_bh1)
                                                ratio_min = np.min(np.array((np.nanmin(ratio_sim0), np.nanmin(ratio_sim1) )))
                                                ratio_max = np.max(np.array((ratio_sim0[ratio_sim0 < np.Inf].max(), ratio_sim1[ratio_sim1 < np.Inf].max() )))

                                            if(plt_sim==True and plot_ratio==True):
                                                #ax10.axhline(y = 0., color = 'k', linestyle = '--')
                                                ax10.hist(seedlinbins, weights=0*seedlinbins, bins = seedlinbins, histtype='step', ls='--', density=densi, color='k', linewidth=l_width*2/3 )
                                                ax10.errorbar(bincenters_obs, ratio_sim0, yerr=plt_err*err_ratio_sim0, fmt='C2'+'>', lw=2, markersize=12 )
                                                if(zoom_ratio==True):ax10.set_ylim(-0.5+ratio_min,ratio_max)

                                            if(plt_sim1==True and plot_ratio==True):
                                                #ax10.axhline(y = 0., color = 'k', linestyle = '--')
                                                ax10.hist(seedlinbins, weights=0*seedlinbins, bins = seedlinbins, histtype='step', ls='--', density=densi, color='k', linewidth=l_width*2/3 )
                                                ax10.errorbar(bincenters_obs, ratio_sim1, yerr=plt_err*err_ratio_sim1, fmt='C0'+'o', lw=2, markersize=10 )
                                                if(zoom_ratio==True):ax10.set_ylim(-0.5+ratio_min,ratio_max)

                                            if( (plt_sim1==True or plt_sim==True) and plot_ratio==True):
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
                                            #save_subplt = 'plot_hist'+dens+'_maxadc_'+strbin +str(z*2).zfill(2)+'mm_sim' \
                                            #            +'_'+source+'_'+str(nsigm_bg)+'sig'+'_'+ev+'ev_'+fecha+str_hist_comp+cut_adc+cut_str
                                            save_subplt = 'plot_hist'+dens+'_maxadc_'+strbin +str(z*2).zfill(2)+'mm'+ name_cuts
                                            print(save_subplt)
                                            # Save just the portion _inside_ the second axis's boundaries
                                            #extent =ax8.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                                            if(plt_one==True):
                                                plt.savefig(dirsave_plt+'/'+ save_subplt+difu_gauss+'.pdf', dpi=150)#,  bbox_inches=extent.expanded(1.2, 1.3))
                                                plt.savefig(dirsave_plt+'/'+ save_subplt+difu_gauss+'.png', dpi=150)#,  bbox_inches=extent.expanded(1.2, 1.3))
                                                # plt.show()

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
                                            maxie = np.min(np.array((np.max(data_cut_obs[:,9]),
                                                    np.max(data_cut_sim_0[:,9]), np.max(data_cut_sim_1[:,9])  )))+1
                                            minie = np.min(np.array((np.min(data_cut_obs[:,9]),
                                                    np.min(data_cut_sim_0[:,9]), np.min(data_cut_sim_1[:,9])  )))

                                            max_e = np.max(np.array((np.max(data_cut_obs[:,9]),
                                                    np.max(data_cut_sim_0[:,9]), np.max(data_cut_sim_1[:,9])  )))

                                            if( min_adc==adc_cut):data_cut = data_cut_obs
                                            else:
                                                if(np.max(data_cut_obs[:,9]) == max_e ):data_cut = data_cut_obs
                                                if(np.max(data_cut_sim_0[:,9]) == max_e ):data_cut = data_cut_sim_0
                                                if(np.max(data_cut_sim_1[:,9]) == max_e ):data_cut = data_cut_sim_1

                                            #sig_etotal = np.std(data_cut_obs[:,9], ddof=1)
                                            #h_etotal = 1*sig_etotal/np.cbrt(data_cut_obs[:,9].size)
                                            #nbins_etotal = np.int(np.ceil(np.max(data_cut_obs[:,9])/h_etotal))

                                            nbins_etotal=int(round(maxie-minie ))
                                            #nbins_etotal = round(np.max(data_cut_obs[:,9] ))
                                            if( min_adc==adc_cut):
                                                hist_etotal, bins_etotal = np.histogram(data_cut[:,9], bins=nbins_etotal)
                                            else:
                                                hist_etotal, bins_etotal = np.histogram(np.append(data_cut[:,9],minie), bins=nbins_etotal)
                                            etlogbins_all = np.logspace(np.log10(bins_etotal[0]),np.log10(bins_etotal[-1]),len(bins_etotal))

                                            wxe,xe = np.histogram(data_cut_obs[:,9], bins = etlogbins_all, density = densi)
                                            wys0,ys0 = np.histogram(data_cut_sim_0[:,9], bins = etlogbins_all, density = densi)
                                            wys1,ys1 = np.histogram(data_cut_sim_1[:,9], bins = etlogbins_all, density = densi)

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
                                                #if(bin_hist > round(np.max(data_cut_obs[:,9] )) ): nbins_etotal = round(np.max(data_cut_obs[:,9] ))
                                                #else: nbins_etotal = bin_hist

                                            else:
                                                nbins_etotal = round(maxie-minie )
                                                #nbins_etotal = round(np.max(data_cut_obs[:,9]/sbi ))

                                            if( min_adc==adc_cut):
                                                hist_etotal, bins_etotal = np.histogram(data_cut[:,9], bins=nbins_etotal)
                                            else:
                                                hist_etotal, bins_etotal = np.histogram(np.append(data_cut[:,9],minie), bins=nbins_etotal)
                                            etlogbins = np.logspace(np.log10(bins_etotal[0]),np.log10(bins_etotal[-1]),len(bins_etotal))

                                            hist_etotal_obs, bins_etotal_obs = np.histogram(data_cut_obs[:,9], bins=nbins_etotal)
                                            etlogbins_obs = np.logspace(np.log10(bins_etotal_obs[0]),np.log10(bins_etotal_obs[-1]),len(bins_etotal))

                                            ########################################################################################################

                                            etlogbin_obs = etlogbins
                                            etlogbin_bg = etlogbins
                                            ######################################
                                            etlogbin_sim_0 = etlogbins
                                            etlogbin_sim_1 = etlogbins
                                            ######################################

                                            wxe_bh,xe_bh = np.histogram(data_cut_obs[:,9], bins = etlogbins, density = densi)
                                            wye_bh0,ye_bh0 = np.histogram(data_cut_sim_0[:,9], bins = etlogbins, density = densi)
                                            wye_bh1,ye_bh1 = np.histogram(data_cut_sim_1[:,9], bins = etlogbins, density = densi)

                                            #########################################################################################
                                            count_obs, bin_obs = np.histogram(data_cut_obs[:,9], bins=etlogbin_obs)
                                            count_obs, bin_obs = np.histogram(data_cut_obs[:,9], bins=etlogbin_obs)
                                            hist_clstr_total_obs = {'bins': bin_obs, 'counts': count_obs}
                                            bincenters_obs = 0.5*(bin_obs[1:]+bin_obs[:-1])
                                            norm_obs = (np.sum(count_obs) * np.diff(bin_obs))
                                            #########################################################################################
                                            count_sim_0, bin_sim_0 = np.histogram(data_cut_sim_0[:,9], bins=etlogbin_sim_0)
                                            hist_clstr_total_sim_0 = {'bins': bin_sim_0, 'counts': count_sim_0}
                                            bincenters_sim_0 = 0.5*(bin_sim_0[1:]+bin_sim_0[:-1])
                                            norm_sim_0 = (np.sum(count_sim_0) * np.diff(bin_sim_0))
                                            #########################################################################################
                                            count_sim_1, bin_sim_1 = np.histogram(data_cut_sim_1[:,9], bins=etlogbin_sim_1)
                                            hist_clstr_total_sim_1 = {'bins': bin_sim_1, 'counts': count_sim_1}
                                            bincenters_sim_1 = 0.5*(bin_sim_1[1:]+bin_sim_1[:-1])
                                            norm_sim_1 = (np.sum(count_sim_1) * np.diff(bin_sim_1))
                                            #########################################################################################
                                            #########################################################################################
                                            if(data_bg_cut.ndim>1):
                                                count_bg, bin_bg = np.histogram(data_bg_cut[:,9], bins=etlogbin_bg)
                                            if(data_bg_cut.ndim==1):
                                                count_bg, bin_bg = np.histogram(data_bg_cut[2], bins=etlogbin_bg)
                                            hist_clstr_total_bg = {'bins': bin_bg, 'counts': count_bg}
                                            #########################################################################################
                                            #########################################################################################
                                            # Guardar en un archivo de texto,  .npy
                                            #hist_dat = np.load(dirsave_plt+'/'+'hist_adc_obs'+source+\
                                             #           sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', allow_pickle=True).item()
                                            np.save(dirsave_plt+'/'+'hist_clstr_total_obs_'+source+\
                                                    sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist_clstr_total_obs )
                                            np.save(dirsave_plt+'/'+'hist_clstr_total_sim_'+source+\
                                                    sim_n+'_'+gauss_dist+'_0k_0a_0zff'+ sig_bg_thresh +simu_no_cut+ str_cuts + cut_clst_size+cut_max+'.npy', hist_clstr_total_sim_0 )
                                            np.save(dirsave_plt+'/'+'hist_clstr_total_sim_'+source+\
                                                    sim_n+'_'+difu_gauss+ sig_bg_thresh +simu_no_cut+ str_cuts + cut_clst_size+cut_max+'.npy', hist_clstr_total_sim_1 )

                                            np.save(dirsave_plt+'/'+'hist_clstr_total_bg_'+source+\
                                                    sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist_clstr_total_bg )
                                            #########################################################################################

                                            if(densi==True):
                                                obs_e = count_obs/norm_obs
                                                sim_0_e = count_sim_0/norm_sim_0
                                                sim_1_e = count_sim_1/norm_sim_1
                                                err_std_obs_e = np.sqrt(count_obs)/ norm_obs
                                                err_std_sim_0_e = 0.2*count_sim_0/norm_sim_0 + np.sqrt(count_sim_0)/norm_sim_0
                                                err_std_sim_1_e = 0.2*count_sim_1/norm_sim_1 + np.sqrt(count_sim_1)/norm_sim_1

                                            if(densi==False):
                                                obs_e = count_obs
                                                sim_0_e = count_sim_0
                                                sim_1_e = count_sim_1
                                                err_std_obs_e = np.sqrt(count_obs)
                                                err_std_sim_0_e = 0.2*count_sim_0 + np.sqrt(count_sim_1)
                                                err_std_sim_1_e = 0.2*count_sim_1 + np.sqrt(count_sim_0)

                                            ###############################################################################################

                                            err_e0 = np.sqrt(err_std_obs_e*err_std_obs_e + err_std_sim_0_e*err_std_sim_0_e )
                                            ch2_e0 = chi_2_sigm_test(obs_e, sim_0_e, err_e0)
                                            tmp_str0 = 'Chi2 test: ' + r'%.2f'%ch2_e0[0] \
                                                    + '       '+'ndf: ' + r'%.2f'%ch2_e0[3] \
                                                    + '\n'+'Chi2/ndf : ' + r'%.2f'%ch2_e0[4] \
                                                    + '           '+'Clst_Sim/Obs: '+str(round(data_cut_sim_0[:,9].size/data_cut_obs[:,9].size,3))

                                            err_e1 = np.sqrt(err_std_obs_e*err_std_obs_e + err_std_sim_1_e*err_std_sim_1_e )
                                            ch2_e1 = chi_2_sigm_test(obs_e, sim_1_e, err_e1)
                                            tmp_str1 = 'Chi2 test: ' + r'%.2f'%ch2_e1[0] \
                                                    + '       '+'ndf: ' + r'%.2f'%ch2_e1[3] \
                                                    + '\n'+'Chi2/ndf : ' + r'%.2f'%ch2_e1[4] \
                                                    + '           '+'Clst_Sim/Obs: '+str(round(data_cut_sim_1[:,9].size/data_cut_obs[:,9].size,3))

                                            if(labl_opt=='full'):
                                                lbl_obs = source + ' data'+ sig_bg_thresh + str_cuts + cut_clst_size+cut_max + cut_adc
                                                lbl_sim0 = source + ' simulation '+ sig_bg_thresh_sim + cut_clst_size+cut_max + cut_adc\
                                                            +'\n'+'Chi2/ndf : ' + r'%.2f'%ch2_e0[4]
                                                lbl_sim1 = source + ' simulation '+ sig_bg_thresh_sim + cut_clst_size+cut_max + cut_adc \
                                                            + d_gauss + '\n'+'Chi2/ndf : ' + r'%.2f'%ch2_e1[4]

                                                lbl_bg = 'Background' + ' data'+ si_bg_thresh + cut_clst_size+cut_max + cut_adc
                                            if(labl_opt=='simple'):
                                                lbl_obs = source + ' data'
                                                lbl_sim0 = source + ' sim ' +r'$\chi^2_\nu$ : ' + r'%.2f'%ch2_e0[4] #+'+ bg'
                                                lbl_sim1 = source + ' sim ' + d_gauss + ' '+ r'$\chi^2_\nu$ : '  + r'%.2f'%ch2_e1[4]#+'+ bg'
                                                lbl_bg = 'Background' + ' data'
                                            if(labl_opt=='off'):
                                                lbl_obs = None
                                                lbl_sim0 = None
                                                lbl_sim1 = None
                                                lbl_bg = None

                                            ax7.set_xscale('log')

                                            if(c_dat_real==True):
                                                ax7.hist(bincenters_obs, weights=count_obs, bins=etlogbin_obs, density=densi, histtype='step'
                                                            #, label=source+'_'+ag+'ag'+'_'+str(sigma_obs)+'sig'+'_z_'+str(z*2)+'mm'+'_t'+tiemp  # + '_Clst_'+str(data_cut_obs[:,8].size)
                                                            , label =  lbl_obs + cut_max_clst_porc
                                                            , color='C3', log=log_y, linewidth=l_width+0.*l_width)

                                                if(densi==True):
                                                    ax7.errorbar(bincenters_obs, obs_e, yerr=err_std_obs_e
                                                                                , fmt='C3'+'.', ecolor='C3', linewidth=l_width)
                                                if(densi==False):
                                                    ax7.errorbar(bincenters_obs, count_obs, yerr=err_std_obs_e
                                                                                , fmt='C3'+'.', ecolor='C3', linewidth=l_width)

                                            if(c_dat_sim==True and plt_sim==True ):
                                                ##############################################################################################################################################################################################################################
                                                ax7.hist(bincenters_sim_0, weights=count_sim_0, bins=etlogbin_sim_0, density=densi, histtype='step'
                                                            #, label='Sim_'+source+'_'+str(sigma_sim)+'sig'+'_z_'+str(z*2)+'mm' +'_ev_' +ev+ '_sim_ipc' # + '_Clst_'+str(data_cut_sim_0[:,8].size)
                                                            , label = lbl_sim0 + cut_max_clst_porc_sim
                                                            , color='C2', log=log_y, linewidth=l_width*2/3)

                                                if(densi==True):
                                                   ax7.errorbar(bincenters_sim_0, sim_0_e, yerr=err_std_sim_0_e
                                                                                , fmt='C2'+'.', ecolor='C2', linewidth=l_width*2/3)
                                                                                # , label = '\n P_ks = %.3f' % p_ks_e + '\n P_prmut = %.3f' % p_permut_e,)
                                                if(densi==False):
                                                   ax7.errorbar(bincenters_sim_0, count_sim_0, yerr= err_std_sim_0_e
                                                                                , fmt='C2'+'.', ecolor='C2', linewidth=l_width*2/3) #,label = tmp_str)
                                                ##############################################################################################################################################################################################################################
                                                ax7.hist(bincenters_sim_1, weights=count_sim_1, bins=etlogbin_sim_1, density=densi, histtype='step'
                                                           # , label='Sim_'+source+'_'+str(sigma_sim)+'sig'+'_z_'+str(z*2)+'mm' +'_ev_' +ev+ '_sim_ipc' # + '_Clst_'+str(data_cut_sim_1[:,9].size)
                                                            , label = lbl_sim1 + cut_max_clst_porc_sim
                                                            , color='C9', log=log_y, linewidth=l_width*2/3)

                                                if(densi==True):
                                                    ax7.errorbar(bincenters_sim_1, sim_1_e, yerr=err_std_sim_1_e
                                                                                , fmt='C9'+'.', ecolor='C9', linewidth=l_width*2/3)
                                                                                # , label = '\n P_ks = %.3f' % p_ks_e + '\n P_prmut = %.3f' % p_permut_e,)
                                                if(densi==False):
                                                    ax7.errorbar(bincenters_sim_1, count_sim_1, yerr= err_std_sim_1_e
                                                                                , fmt='C9'+'.', ecolor='C9', linewidth=l_width*2/3)#,label = tmp_str)
                                                ##############################################################################################################################################################################################################################
                                                ##############################################################################################################################################################################################################################

                                            if(c_bg_real==True ):

                                                if(data_bg_cut.ndim>1):
                                                    count_bg, bin_bg = np.histogram(data_bg_cut[:,9], bins=etlogbin_bg)
                                                    bincenters_bg = 0.5*(bin_bg[1:]+bin_bg[:-1])
                                                    ax7.hist(bincenters_bg, weights=count_bg, bins=etlogbin_bg, density=densi, histtype='step'
                                                               , label=lbl_bg
                                                               , color='k', log=log_y, linewidth=l_width*2/3)

                                                if(data_bg_cut.ndim==1):
                                                    count_bg, bin_bg = np.histogram(data_bg_cut[2], bins=etlogbin_bg)
                                                    bincenters_bg = 0.5*(bin_bg[1:]+bin_bg[:-1])
                                                    ax7.hist(bincenters_bg, weights=count_bg, bins=etlogbin_bg, density=densi, histtype='step'
                                                               , label=lbl_bg
                                                               , color='k', log=log_y, linewidth=l_width*2/3)

                                                bincenters_bg = 0.5*(bin_bg[1:]+bin_bg[:-1])
                                                norm_bg = (np.sum(count_bg) * np.diff(bin_bg))
                                                err_std_bg = np.sqrt(count_bg)/ norm_bg
                                                if(densi==True):
                                                    ax7.errorbar(bincenters_bg, count_bg/norm_bg, yerr=err_std_bg, fmt='k'+'.', ecolor='k')
                                                if(densi==False):
                                                    ax7.errorbar(bincenters_bg, count_bg, yerr=np.sqrt(count_bg), fmt='k'+'.', ecolor='k')

                                            #ax7.axvline(x = 1023, color = 'k', linewidth=l_width+0.5*l_width, label = 'sat_cam: 1023 ADC = ' + str(sat_cam)+' keV')

                                            if(plt_obs==True):
                                                #fig.suptitle('Cluster Total Energy for '+source+' Z='+str(z*2).zfill(2)+'mm '+ dens)
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

                                            ylabl_comp = r'$\frac{ (data -sim)}{\sigma}$'
                                            comp_sim, err_comp_sim = comp_obs_sim(wxe_bh, wye_bh0)
                                            comp_sim1, err_comp_sim1 = comp_obs_sim(wxe_bh, wye_bh1)
                                            delta_min = np.min(np.array((np.nanmin(comp_sim), np.nanmin(comp_sim1) )))
                                            delta_max = np.max(np.array((comp_sim[comp_sim < np.Inf].max(), comp_sim1[comp_sim1 < np.Inf].max() )))

                                            if(plt_sim==True and plot_nsig==True):
                                                ax9.set_xscale('log')
                                                #ax9.axhline(y = 0., color = 'k', linestyle = '--')
                                                ax9.hist(etlogbins, weights=0*etlogbins, bins = etlogbins, histtype='step', ls='--', density=densi, color='k', linewidth=l_width*2/3 )
                                                ax9.errorbar(bincenters_obs, comp_sim, yerr=plt_err*err_comp_sim, fmt='C2'+'>', lw=2, markersize=12 )
                                                if(zoom_delta==True):ax9.set_ylim(delta_min,delta_max)

                                            if(plt_sim1==True and plot_nsig==True):
                                                ax9.set_xscale('log')
                                                #ax9.axhline(y = 0., color = 'k', linestyle = '--')
                                                ax9.hist(etlogbins, weights=0*etlogbins, bins = etlogbins, histtype='step', ls='--', density=densi, color='k', linewidth=l_width*2/3 )
                                                ax9.errorbar(bincenters_obs, comp_sim1, yerr=plt_err*err_comp_sim1, fmt='C0'+'o', lw=2, markersize=10 )
                                                if(zoom_delta==True):ax9.set_ylim(delta_min,delta_max)

                                            if( (plt_sim1==True or plt_sim==True) and plot_nsig==True):
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


                                            ylabl_ratio= r'$ratio = \frac{data}{sim}$'
                                            ratio_sim0, err_ratio_sim0 = ratio_obs_sim(wxe_bh, wye_bh0)
                                            ratio_sim1, err_ratio_sim1 = ratio_obs_sim(wxe_bh, wye_bh1)
                                            ratio_sim1, err_ratio_sim1 = ratio_obs_sim(wxe_bh, wye_bh1)
                                            ratio_min = np.min(np.array((np.nanmin(ratio_sim0), np.nanmin(ratio_sim1) )))
                                            ratio_max = np.max(np.array((ratio_sim0[ratio_sim0 < np.Inf].max(), ratio_sim1[ratio_sim1 < np.Inf].max() )))

                                            if(plt_sim==True and plot_ratio==True):
                                                ax11.set_xscale('log')
                                                ax11.axhline(y = 0., color = 'k', linestyle = '--')
                                                ax11.hist(etlogbins, weights=0*etlogbins, bins = etlogbins, histtype='step', ls='--', density=densi, color='k', linewidth=l_width*2/3 )
                                                ax11.errorbar(bincenters_obs, ratio_sim0, yerr=plt_err*err_ratio_sim0, fmt='C2'+'>', lw=2, markersize=12 )
                                                if(zoom_ratio==True):ax11.set_ylim(-0.5+ratio_min,ratio_max)

                                            if(plt_sim1==True and plot_ratio==True):
                                                ax11.set_xscale('log')
                                                #ax11.axhline(y = 0., color = 'k', linestyle = '--')
                                                ax11.hist(etlogbins, weights=0*etlogbins, bins = etlogbins, histtype='step', ls='--', density=densi, color='k', linewidth=l_width*2/3 )
                                                ax11.errorbar(bincenters_obs, ratio_sim1, yerr=plt_err*err_ratio_sim1, fmt='C0'+'o', lw=2, markersize=10 )
                                                if(zoom_ratio==True):ax11.set_ylim(-0.5+ratio_min,ratio_max)

                                            if( (plt_sim1==True or plt_sim==True) and plot_ratio==True):
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



                                            fig.tight_layout()
                                            #save_subplt = 'plot_hist'+dens+'_total_'+strbin +str(z*2).zfill(2)+'mm_sim' \
                                            #            +'_'+source+'_'+str(nsigm_bg)+'sig'+'_'+ev+'ev_'+fecha+str_hist_comp+cut_adc+cut_str
                                            save_subplt = 'plot_hist'+dens+'_total_'+strbin +str(z*2).zfill(2)+'mm'+ name_cuts
                                            print(save_subplt)
                                            # Save just the portion _inside_ the second axis's boundaries
                                            #extent =ax9.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                                            if(plt_one==True):
                                                plt.savefig(dirsave_plt+'/'+ save_subplt+difu_gauss+'.pdf', dpi=150)#,  bbox_inches=extent.expanded(1.2, 1.3))
                                                plt.savefig(dirsave_plt+'/'+ save_subplt+difu_gauss+'.png', dpi=150)#,  bbox_inches=extent.expanded(1.2, 1.3))
                                                # plt.show()

                                                #plt.clf()
                                                #plt.close()

                                    #save_subplt = 'plot_hists'+dens+'_all_clstrs_'+strbin +str(z*2).zfill(2)+'mm_sim' \
                                    #            +'_'+source+'_'+str(nsigm_bg)+'sig'+'_'+ev+'ev_'+fecha+str_hist_comp+cut_adc0+cut_adc+'clst'+cut_str
                                    save_subplt = 'plot_hists'+dens+'_all_clstrs_'+strbin +str(z*2).zfill(2)+'mm'+ name_cuts

                                    if(plt_one==False):
                                        fig.suptitle(titulo + '_cut_clst'+cut_clst, fontsize=font_siz)
                                        fig.tight_layout()
                                        #fig.suptitle('Hists'+dens+'_'+strbin +str(z*2).zfill(2)+'mm_sim'+'_'+source+'_'+str(nsigm_bg)+'sig'\
                                       #              +'_'+ev+'ev_sim_bg_'+ 'eta_'+ '_Dat_'+fecha+cut_str+cut_adc, fontsize=36)
                                        plt.savefig(dirsave_plt+'/'+ save_subplt+difu_gauss+'.png', dpi=150)
                                        plt.savefig(dirsave_plt+'/'+ save_subplt+difu_gauss+'.pdf', dpi=150)
                                        # plt.show()
                                        #plt.clf()
                                        #plt.close()

                            if(cut_dat00==cut_dat01):
                                fig1, (ht0, ht1, ht2) = plt.subplots(1,3,figsize=(43.5,14) )
                                for type_plt in ['cl_mean', 'cl_etotal', 'cl_max_adc' ]:
                                    if(type_plt=='cl_mean'):
                                        #save_subplt = 'plot_hist'+dens+'_mean_'+strbin +str(z*2).zfill(2)+'mm_sim' \
                                        #            +'_'+source+'_'+str(nsigm_bg)+'sig'+'_'+ev+'ev_'+fecha+str_hist_comp+cut_adc+cut_str
                                        save_subplt = 'plot_hist'+dens+'_mean_'+strbin +str(z*2).zfill(2)+'mm'+ name_cuts
                                        img = mpimg.imread(dirsave_plt+'/'+ save_subplt +'.png')
                                        ht0.imshow(img)
                                        ht0.axis('off')

                                    if(type_plt=='cl_max_adc'):
                                        #save_subplt = 'plot_hist'+dens+'_maxadc_'+strbin +str(z*2).zfill(2)+'mm_sim' \
                                        #            +'_'+source+'_'+str(nsigm_bg)+'sig'+'_'+ev+'ev_'+fecha+str_hist_comp+cut_adc+cut_str
                                        save_subplt = 'plot_hist'+dens+'_maxadc_'+strbin +str(z*2).zfill(2)+'mm'+ name_cuts
                                        img = mpimg.imread(dirsave_plt+'/'+ save_subplt +'.png')
                                        ht1.imshow(img)
                                        ht1.axis('off')

                                    if(type_plt=='cl_etotal'):
                                        #save_subplt = 'plot_hist'+dens+'_total_'+strbin +str(z*2).zfill(2)+'mm_sim' \
                                        #            +'_'+source+'_'+str(nsigm_bg)+'sig'+'_'+ev+'ev_'+fecha+str_hist_comp+cut_adc+cut_str
                                        save_subplt = 'plot_hist'+dens+'_total_'+strbin +str(z*2).zfill(2)+'mm'+ name_cuts
                                        img = mpimg.imread(dirsave_plt+'/'+ save_subplt +'.png')
                                        ht2.imshow(img)
                                        ht2.axis('off')

                                fig1.tight_layout()
                                #fig1.suptitle(titulo + '_cut_clst'+cut_clst, fontsize=font_siz)
                                #save_subplt = 'plot_hists'+dens+'_all_clstrs'+cut_clst+strbin +str(z*2).zfill(2)+'mm_sim' \
                                #            +'_'+source+'_'+str(nsigm_bg)+'sig'+'_'+ev+'ev_'+fecha+str_hist_comp+cut_adc0+cut_adc+'clst'+cut_str
                                save_subplt = 'plot_hists'+dens+'_all_clstrs'+cut_clst+strbin +str(z*2).zfill(2)+'mm'+ name_cuts
                                plt.savefig(dirsave_plt+'/'+ save_subplt+difu_gauss+'MIKI.png', dpi=150)
                                # plt.show()


                            fig1, ax1 = plt.subplots(2,2,figsize=(29,24) )
                            for type_plt in ['cl_size', 'cl_mean', 'cl_etotal', 'cl_max_adc' ]:
                                if(type_plt=='cl_size'):
                                    #save_subplt = 'plot_hist'+dens+'_cluster_'+strbin +str(z*2).zfill(2)+'mm_sim' \
                                    #            +'_'+source+'_'+str(nsigm_bg)+'sig'+'_'+ev+'ev_'+fecha+str_hist_comp+cut_adc+cut_str
                                    save_subplt = 'plot_hist'+dens+'_cluster_'+strbin +str(z*2).zfill(2)+'mm'+ name_cuts
                                    img = mpimg.imread(dirsave_plt+'/'+ save_subplt +difu_gauss +'.png')
                                    ax1[0,0].imshow(img)
                                    ax1[0,0].axis('off')

                                if(type_plt=='cl_mean'):
                                    #save_subplt = 'plot_hist'+dens+'_mean_'+strbin +str(z*2).zfill(2)+'mm_sim' \
                                    #            +'_'+source+'_'+str(nsigm_bg)+'sig'+'_'+ev+'ev_'+fecha+str_hist_comp+cut_adc+cut_str
                                    save_subplt = 'plot_hist'+dens+'_mean_'+strbin +str(z*2).zfill(2)+'mm'+ name_cuts
                                    img = mpimg.imread(dirsave_plt+'/'+ save_subplt +difu_gauss+'.png')
                                    ax1[0,1].imshow(img)
                                    ax1[0,1].axis('off')

                                if(type_plt=='cl_max_adc'):
                                    #save_subplt = 'plot_hist'+dens+'_maxadc_'+strbin +str(z*2).zfill(2)+'mm_sim' \
                                    #            +'_'+source+'_'+str(nsigm_bg)+'sig'+'_'+ev+'ev_'+fecha+str_hist_comp+cut_adc+cut_str
                                    save_subplt = 'plot_hist'+dens+'_maxadc_'+strbin +str(z*2).zfill(2)+'mm'+ name_cuts
                                    img = mpimg.imread(dirsave_plt+'/'+ save_subplt +difu_gauss+'.png')
                                    ax1[1,0].imshow(img)
                                    ax1[1,0].axis('off')

                                if(type_plt=='cl_etotal'):
                                    #save_subplt = 'plot_hist'+dens+'_total_'+strbin +str(z*2).zfill(2)+'mm_sim' \
                                    #            +'_'+source+'_'+str(nsigm_bg)+'sig'+'_'+ev+'ev_'+fecha+str_hist_comp+cut_adc+cut_str
                                    save_subplt = 'plot_hist'+dens+'_total_'+strbin +str(z*2).zfill(2)+'mm'+ name_cuts
                                    img = mpimg.imread(dirsave_plt+'/'+ save_subplt +difu_gauss+'.png')
                                    ax1[1,1].imshow(img)
                                    ax1[1,1].axis('off')


                            fig1.tight_layout()
                            #fig1.suptitle(titulo + '_cut_clst'+cut_clst, fontsize=font_siz*(1+0.25))
                            save_subplt = 'plot_hists'+dens+'_all_clstrs_'+strbin +str(z*2).zfill(2)+'mm'+ name_cuts
                            plt.savefig(dirsave_plt+'/'+ save_subplt+difu_gauss+difu_gauss+'MIKI.png', dpi=150)
                            # plt.show()

                            ##########################

                            if(plots_2d==True):
                                #read data
                                #hist2_dat = np.load(dirsave_plt+'/'+'hist2_mean_vs_clstr_obs_'+source+\
                                #sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', allow_pickle=True).item()
                                wxy_mc_obs, x_mc_obs, y_mc_obs = np.histogram2d(data_cut_obs[:,8], data_cut_obs[:,7], bins=(meanlinbins, cltslinbins) )
                                hist2_mean_clstr_obs = {'binx': x_mc_obs, 'biny': y_mc_obs, 'counts': wxy_mc_obs}
                                np.save(dirsave_plt+'/'+'hist2_mean_vs_clstr_obs_'+source+\
                                sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist2_mean_clstr_obs )

                                wxy_ec_obs, x_ec_obs, y_ec_obs = np.histogram2d(data_cut_obs[:,9], data_cut_obs[:,7], bins=(etlogbins, cltslinbins) )
                                hist2_total_clstr_obs = {'binx': x_ec_obs, 'biny': y_ec_obs, 'counts': wxy_ec_obs}
                                np.save(dirsave_plt+'/'+'hist2_total_vs_clstr_obs_'+source+\
                                sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist2_total_clstr_obs )

                                wxy_sc_obs, x_sc_obs, y_sc_obs = np.histogram2d(data_cut_obs[:,10], data_cut_obs[:,7], bins=(seedlinbins, cltslinbins) )
                                hist2_max_clstr_obs = {'binx': x_sc_obs, 'biny': y_sc_obs, 'counts': wxy_sc_obs}
                                np.save(dirsave_plt+'/'+'hist2_max_vs_clstr_obs_'+source+\
                                sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist2_max_clstr_obs )

                                wxy_me_obs, x_me_obs, y_me_obs = np.histogram2d(data_cut_obs[:,8], data_cut_obs[:,9], bins=(meanlinbins, etlogbins) )
                                hist2_mean_total_obs = {'binx': x_me_obs, 'biny': y_me_obs, 'counts': wxy_me_obs}
                                np.save(dirsave_plt+'/'+'hist2_mean_vs_total_obs_'+source+\
                                sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist2_mean_total_obs )

                                wxy_es_obs, x_es_obs, y_es_obs = np.histogram2d(data_cut_obs[:,9], data_cut_obs[:,10], bins=(etlogbins, seedlinbins) )
                                hist2_total_max_obs = {'binx': x_es_obs, 'biny': y_es_obs, 'counts': wxy_es_obs}
                                np.save(dirsave_plt+'/'+'hist2_total_vs_max_obs_'+source+\
                                sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist2_total_max_obs )

                                wxy_sm_obs, x_sm_obs, y_sm_obs = np.histogram2d(data_cut_obs[:,10], data_cut_obs[:,8], bins=(seedlinbins, meanlinbins) )
                                hist2_max_mean_obs = {'binx': x_sm_obs, 'biny': y_sm_obs, 'counts': wxy_sm_obs}
                                np.save(dirsave_plt+'/'+'hist2_max_vs_mean_obs_'+source+\
                                sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist2_max_mean_obs )
                                ##################################################################################
                                wxy_mc_sim0, x_mc_sim0, y_mc_sim0 = np.histogram2d(data_cut_sim_0[:,8], data_cut_sim_0[:,7], bins=(meanlinbins, cltslinbins) )
                                hist2_mean_clstr_sim0 = {'binx': x_mc_sim0, 'biny': y_mc_sim0, 'counts': wxy_mc_sim0}
                                np.save(dirsave_plt+'/'+'hist2_mean_vs_clstr_sim0_'+source+\
                                sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist2_mean_clstr_sim0 )

                                wxy_ec_sim0, x_ec_sim0, y_ec_sim0 = np.histogram2d(data_cut_sim_0[:,9], data_cut_sim_0[:,7], bins=(etlogbins, cltslinbins) )
                                hist2_total_clstr_sim0 = {'binx': x_ec_sim0, 'biny': y_ec_sim0, 'counts': wxy_ec_sim0}
                                np.save(dirsave_plt+'/'+'hist2_total_vs_clstr_sim0_'+source+\
                                sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist2_total_clstr_sim0 )

                                wxy_sc_sim0, x_sc_sim0, y_sc_sim0 = np.histogram2d(data_cut_sim_0[:,10], data_cut_sim_0[:,7], bins=(seedlinbins, cltslinbins) )
                                hist2_max_clstr_sim0 = {'binx': x_sc_sim0, 'biny': y_sc_sim0, 'counts': wxy_sc_sim0}
                                np.save(dirsave_plt+'/'+'hist2_max_vs_clstr_sim0_'+source+\
                                sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist2_max_clstr_sim0 )

                                wxy_me_sim0, x_me_sim0, y_me_sim0 = np.histogram2d(data_cut_sim_0[:,8], data_cut_sim_0[:,9], bins=(meanlinbins, etlogbins) )
                                hist2_mean_total_sim0 = {'binx': x_me_sim0, 'biny': y_me_sim0, 'counts': wxy_me_sim0}
                                np.save(dirsave_plt+'/'+'hist2_mean_vs_total_sim0_'+source+\
                                sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist2_mean_total_sim0 )

                                wxy_es_sim0, x_es_sim0, y_es_sim0 = np.histogram2d(data_cut_sim_0[:,9], data_cut_sim_0[:,10], bins=(etlogbins, seedlinbins) )
                                hist2_total_max_sim0 = {'binx': x_es_sim0, 'biny': y_es_sim0, 'counts': wxy_es_sim0}
                                np.save(dirsave_plt+'/'+'hist2_total_vs_max_sim0_'+source+\
                                sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist2_total_max_sim0 )

                                wxy_sm_sim0, x_sm_sim0, y_sm_sim0 = np.histogram2d(data_cut_sim_0[:,10], data_cut_sim_0[:,8], bins=(seedlinbins, meanlinbins) )
                                hist2_max_mean_sim0 = {'binx': x_sm_sim0, 'biny': y_sm_sim0, 'counts': wxy_sm_sim0}
                                np.save(dirsave_plt+'/'+'hist2_max_vs_mean_sim0_'+source+\
                                sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist2_max_mean_sim0 )
                                ##################################################################################
                                wxy_mc_sim1, x_mc_sim1, y_mc_sim1 = np.histogram2d(data_cut_sim_1[:,8], data_cut_sim_1[:,7], bins=(meanlinbins, cltslinbins) )
                                hist2_mean_clstr_sim1 = {'binx': x_mc_sim1, 'biny': y_mc_sim1, 'counts': wxy_mc_sim1}
                                np.save(dirsave_plt+'/'+'hist2_mean_vs_clstr_sim1_'+source+\
                                sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist2_mean_clstr_sim1 )

                                wxy_ec_sim1, x_ec_sim1, y_ec_sim1 = np.histogram2d(data_cut_sim_1[:,9], data_cut_sim_1[:,7], bins=(etlogbins, cltslinbins) )
                                hist2_total_clstr_sim1 = {'binx': x_ec_sim1, 'biny': y_ec_sim1, 'counts': wxy_ec_sim1}
                                np.save(dirsave_plt+'/'+'hist2_total_vs_clstr_sim1_'+source+\
                                sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist2_total_clstr_sim1 )

                                wxy_sc_sim1, x_sc_sim1, y_sc_sim1 = np.histogram2d(data_cut_sim_1[:,10], data_cut_sim_1[:,7], bins=(seedlinbins, cltslinbins) )
                                hist2_max_clstr_sim1 = {'binx': x_sc_sim1, 'biny': y_sc_sim1, 'counts': wxy_sc_sim1}
                                np.save(dirsave_plt+'/'+'hist2_max_vs_clstr_sim1_'+source+\
                                sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist2_max_clstr_sim1 )

                                wxy_me_sim1, x_me_sim1, y_me_sim1 = np.histogram2d(data_cut_sim_1[:,8], data_cut_sim_1[:,9], bins=(meanlinbins, etlogbins) )
                                hist2_mean_total_sim1 = {'binx': x_me_sim1, 'biny': y_me_sim1, 'counts': wxy_me_sim1}
                                np.save(dirsave_plt+'/'+'hist2_mean_vs_total_sim1_'+source+\
                                sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist2_mean_total_sim1 )

                                wxy_es_sim1, x_es_sim1, y_es_sim1 = np.histogram2d(data_cut_sim_1[:,9], data_cut_sim_1[:,10], bins=(etlogbins, seedlinbins) )
                                hist2_total_max_sim1 = {'binx': x_es_sim1, 'biny': y_es_sim1, 'counts': wxy_es_sim1}
                                np.save(dirsave_plt+'/'+'hist2_total_vs_max_sim1_'+source+\
                                sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist2_total_max_sim1 )

                                wxy_sm_sim1, x_sm_sim1, y_sm_sim1 = np.histogram2d(data_cut_sim_1[:,10], data_cut_sim_1[:,8], bins=(seedlinbins, meanlinbins) )
                                hist2_max_mean_sim1 = {'binx': x_sm_sim1, 'biny': y_sm_sim1, 'counts': wxy_sm_sim1}
                                np.save(dirsave_plt+'/'+'hist2_max_vs_mean_sim1_'+source+\
                                sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', hist2_max_mean_sim1 )
                                ##################################################################################
                                ##################################################################################


                                if(label_xy == True):  sharxy = True;  lblxy='_lbl'
                                if(label_xy == False): sharxy = False; lblxy=''

                                for zom in [True,
                                            #False
                                            ]:
                                    if(zom==True):lab_zom = '_zm'
                                    if(zom==False):lab_zom = ''


                                    for plt_one in[ True,
                                                   #False
                                                   ]:
                                        if(plt_one == False):

                                            plt.rcParams.update({'font.size': font_siz})
                                            fig, ax = plt.subplots(row, 3,figsize=(28.5,row*7),sharex=False, sharey=sharxy)

                                        for q in range(3):
                                            if(plt_one == True):
                                                plt.rcParams.update({'font.size': font_siz})
                                                #fig, ax = plt.subplots(row, 1,figsize=(14,row*7.2),sharey=False)
                                                fig, ax = plt.subplots(row, 1,figsize=(11.5,row*7.),sharex=sharxy, sharey=sharxy)

                                            for p in range(row):

                                                if(p==0 and q==0):
                                                    x = data_cut_obs[:,8]; y = data_cut_obs[:,7];
                                                    xbin_all = meanlinbins_all; ybin_all = cltslinbins_all;
                                                    wxy_mc, x_mc, y_mc = np.histogram2d(x,y, bins=(xbin_all, ybin_all) )

                                                    xbin = meanlinbins; ybin = cltslinbins;
                                                    wxy_mc_bh, x_mc_bh, y_mc_bh = np.histogram2d(x,y, bins=(xbin, ybin) )

                                                    labl = lbl_obs;
                                                    label_x=xlab[0][1]; label_y=xlab[0][0]
                                                if(p==0 and q==1):
                                                    x = data_cut_obs[:,9]; y = data_cut_obs[:,7];
                                                    xbin_all = etlogbins_all; ybin_all = cltslinbins_all;
                                                    wxy_ec, x_ec, y_ec = np.histogram2d(x,y, bins=(xbin_all, ybin_all) )

                                                    xbin = etlogbins;     ybin = cltslinbins;
                                                    wxy_ec_bh, x_ec_bh, y_ec_bh = np.histogram2d(x,y, bins=(xbin, ybin) )

                                                    labl = lbl_obs;
                                                    label_x=xlab[1][1]; label_y=xlab[0][0]
                                                if(p==0 and q==2):
                                                    x = data_cut_obs[:,10]; y = data_cut_obs[:,7];
                                                    xbin_all = seedlinbins_all; ybin_all = cltslinbins_all;
                                                    wxy_sc, x_sc, y_sc = np.histogram2d(x,y, bins=(xbin_all, ybin_all) )

                                                    xbin = seedlinbins; ybin = cltslinbins;
                                                    wxy_sc_bh, x_sc_bh, y_sc_bh = np.histogram2d(x,y, bins=(xbin, ybin) )

                                                    labl = lbl_obs;
                                                    label_x=xlab[1][0]; label_y=xlab[0][0]

                                                #####################################################################################
                                                if(plt_sim==True):
                                                    if(p==1 and q==0 ):
                                                        x = data_cut_sim_0[:,8]; y = data_cut_sim_0[:,7];
                                                        xbin_all = meanlinbins_all; ybin_all = cltslinbins_all;
                                                        wxy_mc_0sim, x_mc_0sim, y_mc_0sim = np.histogram2d(x,y, bins=(xbin_all, ybin_all) )

                                                        xbin = meanlinbins; ybin = cltslinbins;
                                                        wxy_mc_0sim_bh, x_mc_0sim_bh, y_mc_0sim_bh = np.histogram2d(x,y, bins=(xbin, ybin) )

                                                        labl = lbl_sim0;
                                                        label_x=xlab[0][1]; label_y=xlab[0][0]
                                                    if(p==1 and q==1):
                                                        x = data_cut_sim_0[:,9]; y = data_cut_sim_0[:,7];
                                                        xbin_all = etlogbins_all; ybin_all = cltslinbins_all;
                                                        wxy_ec_0sim, x_ec_0sim, y_ec_0sim = np.histogram2d(x,y, bins=(xbin_all, ybin_all) )

                                                        xbin = etlogbins;     ybin = cltslinbins;
                                                        wxy_ec_0sim_bh, x_ec_0sim_bh, y_ec_0sim_bh = np.histogram2d(x,y, bins=(xbin, ybin) )

                                                        labl = lbl_sim0;
                                                        label_x=xlab[1][1]; label_y=xlab[0][0]
                                                    if(p==1 and q==2):
                                                        x = data_cut_sim_0[:,10]; y = data_cut_sim_0[:,7];
                                                        xbin_all = seedlinbins_all; ybin_all = cltslinbins_all;
                                                        wxy_sc_0sim, x_sc_0sim, y_sc_0sim = np.histogram2d(x,y, bins=(xbin_all, ybin_all) )

                                                        xbin = seedlinbins; ybin = cltslinbins;
                                                        wxy_sc_0sim_bh, x_sc_0sim_bh, y_sc_0sim_bh = np.histogram2d(x,y, bins=(xbin, ybin) )

                                                        labl = lbl_sim0;
                                                        label_x=xlab[1][0]; label_y=xlab[0][0]

                                                #####################################################################################

                                                if(zom==False):
                                                    if(q==0):x_zoom =  data_cut_obs[:,8].max(); y_zoom = data_cut_obs[:,7].max()
                                                    if(q==1):x_zoom =  data_cut_obs[:,9].max(); y_zoom = data_cut_obs[:,7].max()
                                                    if(q==2):x_zoom =  data_cut_obs[:,10].max(); y_zoom = data_cut_obs[:,7].max()

                                                if(zom==True):
                                                    if(q==0):x_zoom = 300;   y_zoom = 80
                                                    if(q==1):x_zoom = 2000;  y_zoom = 80
                                                    if(q==2):x_zoom = 1000;   y_zoom = 80


                                                if(plt_one == False):
                                                    if(plt_sim==False and plt_sim1==False):
                                                        h=ax[q].hist2d(x,y, bins=(xbin, ybin), range=([0, x.max()],[0, y.max()]), cmin = 0 , cmax = 1e7 , norm=matplotlib.colors.LogNorm(),cmap=plt.cm.jet)
                                                        #hf=ax[q].hist2d(xf,yf, bins=(int(xf.max()),int(yf.max())), range=([0, xf.max()],[0, yf.max()]), norm=matplotlib.colors.LogNorm(),cmap=plt.cm.spring)

                                                        ax[q].set_xlim(0,x_zoom)
                                                        ax[q].set_ylim(0,y_zoom)

                                                        ax[q].set_xlabel(label_x, fontsize=font_siz+2)
                                                        ax[q].set_ylabel(label_y, fontsize=font_siz+2)
                                                        ax[q].set_title(labl, y=1.0, pad=25, fontweight='bold')#+lab_zom)#, fontsize=22 )#+' and '+lab_bg+lab_zom )
                                                        #ax[q].legend()
                                                        ax[q].grid(grid_)

                                                        divider = make_axes_locatable(ax[q])
                                                        #cax = divider.append_axes('right', size='3%', pad=0.07)
                                                        #cax1 = divider.append_axes('right', size='3%', pad=1.35)

                                                        #plt.colorbar(h[3], label=labl, ax=ax[q], cax=cax)

                                                        #fig.colorbar(h[3], ax=ax[q], cax=cax)#, label='Clusters number')


                                                    else:
                                                        #hf=ax[p,q].hist2d(xf,yf, bins=(int(xf.max()),int(yf.max())), range=([0, xf.max()],[0, yf.max()]), norm=matplotlib.colors.LogNorm(),cmap=plt.cm.spring)
                                                        #h=ax[p,q].hist2d(x,y, bins=(int(x.max()),int(y.max())), range=([0, x.max()],[0, y.max()]),  norm=matplotlib.colors.LogNorm(),cmap=plt.cm.jet)

                                                        h=ax[p,q].hist2d(x,y, bins=(xbin, ybin), range=([0, x.max()],[0, y.max()]), cmin = 0 , cmax = 1e7 , norm=matplotlib.colors.LogNorm(),cmap=plt.cm.jet)
                                                        #hf=ax[p,q].hist2d(xf,yf, bins=(int(xf.max()),int(yf.max())), range=([0, xf.max()],[0, yf.max()]), norm=matplotlib.colors.LogNorm(),cmap=plt.cm.spring)

                                                        ax[p,q].set_xlim(0,x_zoom)
                                                        ax[p,q].set_ylim(0,y_zoom)

                                                        if(label_xy==True):
                                                            ax[0,q].xaxis.set_tick_params(labelbottom=False)
                                                            if(row==2):
                                                                ax[1,q].set_xlabel(label_x, fontsize=font_siz+2)
                                                            if(row==3):
                                                                ax[1,q].xaxis.set_tick_params(labelbottom=False)
                                                                ax[2,q].set_xlabel(label_x, fontsize=font_siz+2)

                                                            ax[p,1].set_title(labl, y=1.0, pad=25, fontweight='bold')#+lab_zom)#, fontsize=22 )#+' and '+lab_bg+lab_zom )
                                                            fig.supylabel(label_y, fontsize=font_siz+2)
                                                        if(label_xy==False):
                                                            ax[p,q].set_xlabel(label_x, fontsize=font_siz+2)
                                                            ax[p,q].set_ylabel(label_y, fontsize=font_siz+2)
                                                            ax[p,1].set_title(labl, y=1.0, pad=25, fontweight='bold')#+lab_zom)#, fontsize=22 )#+' and '+lab_bg+lab_zom )
                                                        #ax[p,q].legend()
                                                        ax[p,q].grid(grid_)

                                                        divider = make_axes_locatable(ax[p,q])
                                                        #cax = divider.append_axes('right', size='3%', pad=0.07)
                                                        #cax1 = divider.append_axes('right', size='3%', pad=1.35)


                                                        #plt.colorbar(h[3], label=labl, ax=ax[p,q], cax=cax)
                                                        #fig.colorbar(h[3], ax=ax[p,q], cax=cax)#, label='Clusters number')
                                                        #plt.colorbar(hf[3],label=lab_bg, ax=ax[p,q], cax=cax1)
                                                        ############################################################################################################
                                                        ############################################################################################################

                                                if(plt_one == True):
                                                    if(plt_sim==False and plt_sim1==False):
                                                        h=ax.hist2d(x,y, bins=(xbin, ybin), range=([0, x.max()],[0, y.max()]), cmin = 0 , cmax = 1e7 , norm=matplotlib.colors.LogNorm(),cmap=plt.cm.jet)

                                                        ax.set_xlim(0,x_zoom)
                                                        ax.set_ylim(0,y_zoom)

                                                        ax.set_xlabel(label_x, fontsize=font_siz+2)
                                                        ax.set_ylabel(label_y, fontsize=font_siz+2)
                                                        ax.set_title(labl, y=1.0, pad=25, fontsize=font_siz-1, fontweight='bold')#+lab_zom)#, fontsize=22 )#+' and '+lab_bg+lab_zom )
                                                        #ax.legend()
                                                        ax.grid(grid_)

                                                        divider = make_axes_locatable(ax)
                                                        #cax = divider.append_axes('right', size='3%', pad=0.07)

                                                        #fig.colorbar(h[3], ax=ax, cax=cax, label='Clusters number')#, ticks=v)

                                                    else:
                                                        h=ax[p].hist2d(x,y, bins=(xbin, ybin), range=([0, x.max()],[0, y.max()]), cmin = 0 , cmax = 1e7 , norm=matplotlib.colors.LogNorm(),cmap=plt.cm.jet)

                                                        ax[p].set_xlim(0,x_zoom)
                                                        ax[p].set_ylim(0,y_zoom)
                                                        if(label_xy==True):
                                                            if(row==2):ax[1].set_xlabel(label_x, fontsize=font_siz+2)
                                                            if(row==3):ax[2].set_xlabel(label_x, fontsize=font_siz+2)
                                                            #ax[1].set_ylabel(label_y, fontsize=font_siz+2)
                                                            ax[p].set_title(labl, y=1.0, pad=25, fontsize=font_siz-1, fontweight='bold')#+lab_zom)#, fontsize=22 )#+' and '+lab_bg+lab_zom )
                                                            fig.supylabel(label_y, fontsize=font_siz+2)
                                                        if(label_xy==False):
                                                            ax[p].set_xlabel(label_x, fontsize=font_siz+2)
                                                            ax[p].set_ylabel(label_y, fontsize=font_siz+2)
                                                            ax[p].set_title(labl, y=1.0, pad=25, fontsize=font_siz-1, fontweight='bold')#+lab_zom)#, fontsize=22 )#+' and '+lab_bg+lab_zom )
                                                        #ax[p].legend()
                                                        ax[p].grid(grid_)

                                                        divider = make_axes_locatable(ax[p])
                                                        #cax = divider.append_axes('right', size='3%', pad=0.07)
                                                        cbar= fig.colorbar(h[3], ax=ax[p])
                                                        cbar.mappable.set_clim(1, 3e3)
                                                        #cbar.ax.set_visible(False)
                                                        cbar.remove()

                                                        #fig.colorbar(h[3], ax=ax[p], cax=cax, label='Clusters number')

                                            if(plt_one == True):
                                                if(q==0):str_hist2 = 'Cluste_vs_Mean_ADC'
                                                if(q==1):str_hist2 = 'Cluste_vs_Total_ADC'
                                                if(q==2):str_hist2 = 'Cluste_vs_Max_ADC'
                                                #fig.suptitle(str_hist2+lab_zom +dens+'_'+strbin +str(z*2).zfill(2)+'mm_sim'+'_'+source+'_'+str(nsigm_bg)+'sig', fontsize=36)

                                                # Colorbar
                                                #fig.subplots_adjust(right=0.85)
                                                if(row==2 or row==3):
                                                    p0 = ax[0].get_position().get_points().flatten()
                                                    p1 = ax[1].get_position().get_points().flatten()
                                                    ax_cbar = fig.add_axes([0.8, (row*p1[3]-(row-1)*p0[3])*0.75 , 0.05, p0[3]*1.025 ])
                                                    cbar= fig.colorbar(h[3], ax=ax, location = 'right', pad = -0.4 , aspect=100, shrink=0.0001)
                                                    cbar.mappable.set_clim(1, 3e3)
                                                if(row==1):
                                                    p0 = ax.get_position().get_points().flatten()
                                                    ax_cbar = fig.add_axes([0.8, (p0[1])*0.75 , 0.05, p0[3]*1.025 ])
                                                    cbar= fig.colorbar(h[3], ax=ax, location = 'right', pad = -0.7 , aspect=100, shrink=0.0001)
                                                    cbar.mappable.set_clim(1, 3e3)
                                                #print(p0,p1,p2)
                                                #ax_cbar = fig.add_axes([1, (p2[3]-p0[3]+p1[3])*0.75 , 0.03, p0[3]*1.025 ])
                                                #if(row==3):ax_cbar = fig.add_axes([0.8, (p2[3]-p0[3]+p1[3])*0.75 , 0.06, p0[3]*1.025 ])
                                                #if(row==2):ax_cbar = fig.add_axes([0.8, (2*p1[3]-p0[3])*0.75 , 0.06, p0[3]*1.025 ])
                                                #ax_cbar = fig.add_axes([0.8, (row*p1[3]-(row-1)*p0[3])*0.75 , 0.06, p0[3]*1.025 ])

                                                #cbar_ax = fig.add_axes([1., 0.044, 0.025, 0.927])
                                                #cbar= fig.colorbar(h[3], ax=ax, location = 'right', pad = -0.4 , aspect=100, shrink=0.0001)
                                                cbar.set_ticks([-0.01, 0])
                                                cbar.set_ticklabels(['', ''])
                                                cbar.outline.set_visible(False)


                                                cb = fig.colorbar(h[3], ax=ax, cax=ax_cbar)
                                                cb.set_label(label='Clusters number',size=font_siz+4)
                                                #cb.mappable.set_clim(1, 3e3)
                                                cb.ax.tick_params(labelsize=font_siz)

                                                #fig.colorbar(h[3], ax=ax[:], cax=ax_cbar, label='Clusters number')
                                               # fig.colorbar(h[3], ax=ax, location = 'right',  pad = -0.42, shrink=1.4, #label='Clusters number' )

                                                fig.tight_layout()
                                                #namesave = 'plot_hists2D'+lblxy+lab_zom +dens+'_'+str_hist2+'_'+strbin +str(z*2).zfill(2)+'mm_sim'+'_'+source+'_'+str(nsigm_bg)+'sig'+'_'+ev+'ev_'+fecha+str_hist_comp+cut_adc+cut_str
                                                namesave = 'plot_hists2D'+lblxy+lab_zom +dens+'_'+str_hist2+'_'+strbin +str(z*2).zfill(2)+'mm'+ name_cuts
                                                plt.savefig(dirsave_plt+'/'+ namesave+difu_gauss+'.pdf', dpi=150)
                                                plt.savefig(dirsave_plt+'/'+ namesave+difu_gauss+'.png', dpi=150)
                                                # plt.show()
                                                #plt.clf()
                                                #plt.close()

                                        if(plt_one == False):
                                            #fig.suptitle('Hists_2D'+lab_zom +dens+'_'+strbin +str(z*2).zfill(2)+'mm_sim'+'_'+source+'_'+str(nsigm_bg)+'sig'\
                                               #          +'_'+ev+'ev_sim_bg_'+ 'eta_'+ '_Dat_'+fecha+cut_str+cut_adc, fontsize=36)
                                                        # +'_'+ev+'ev_sim_bg_'+ 'eta_'+ '_Dat_'+fecha+cut_str+cut_adc, fontsize=36)
                                            if(row==2 or row==3):
                                                p0 = ax[0,2].get_position().get_points().flatten()
                                                p1 = ax[1,2].get_position().get_points().flatten()
                                                ax_cbar = fig.add_axes([0.925, (row*p1[3]-(row-1)*p0[3])*0.75 , 0.02, p0[3]*1.025 ])
                                                #cbar= fig.colorbar(h[3], ax=ax, location = 'right', pad = -0.25 , aspect=100, shrink=0.0001)

                                            if(row==1):
                                                p0 = ax[2].get_position().get_points().flatten()
                                                ax_cbar = fig.add_axes([0.925, p0[1]*0.75 , 0.02, p0[3]*1.025 ])
                                                #cbar= fig.colorbar(h[3], ax=ax, location = 'right', pad = -0.25 , aspect=100, shrink=0.0001)

                                            #if(row==3):ax_cbar = fig.add_axes([0.925, (p2[3]-p0[3]+p1[3])*0.75 , 0.02, p0[3]*1.025 ])
                                            #if(row==2):ax_cbar = fig.add_axes([0.925, (2*p1[3]-p0[3])*0.75 , 0.02, p0[3]*1.025 ])
                                            #ax_cbar = fig.add_axes([0.925, (row*p1[3]-(row-1)*p0[3])*0.75 , 0.02, p0[3]*1.025 ])
                                            #cbar_ax = fig.add_axes([1., 0.044, 0.025, 0.927])
                                            cbar= fig.colorbar(h[3], ax=ax, location = 'right', pad = -0.25 , aspect=100, shrink=0.0001)
                                            cbar.set_ticks([-0.01, 0])
                                            cbar.set_ticklabels(['', ''])
                                            cbar.outline.set_visible(False)

                                            cb = fig.colorbar(h[3], ax=ax, cax=ax_cbar)
                                            #ax.set_aspect('auto')
                                            cb.set_label(label='Clusters number',size=font_siz+4)
                                            cb.ax.tick_params(labelsize=font_siz)

                                            fig.tight_layout()

                                            #namesave = 'plot_hists2D'+lblxy+lab_zom +dens+'_All_1_'+strbin +str(z*2).zfill(2)+'mm_sim'+'_'+source+'_'+str(nsigm_bg)+'sig'+'_'+ev+'ev_'+fecha+str_hist_comp+cut_adc+cut_str
                                            namesave = 'plot_hists2D'+lblxy+lab_zom +dens+'_All_1_'+strbin +str(z*2).zfill(2)+'mm'+ name_cuts

                                            plt.savefig(dirsave_plt+'/'+ namesave+difu_gauss+'.png', dpi=150)
                                            plt.savefig(dirsave_plt+'/'+ namesave+difu_gauss+'.pdf', dpi=150)

                                            print(namesave)
                                            # plt.show()
                                            #plt.clf()
                                            #plt.close()

                                ###############################################################
                                ###############################################################
                                ###############################################################

                                #for zom in [True, False ]:
                                 #   if(zom==True):lab_zom= '_zoom'
                                  #  if(zom==False):lab_zom=''

                                   # for plt_one in[ True, False ]:

                                        if(plt_one == False):
                                            plt.rcParams.update({'font.size': font_siz})
                                            fig, ax = plt.subplots(row, 3,figsize=(28.5,row*7),sharex=False, sharey=False)

                                        for q in range(3,6):
                                            if(plt_one == True):
                                                plt.rcParams.update({'font.size': font_siz})
                                                #fig, ax = plt.subplots(row, 1,figsize=(14,row*7.2),sharey=False)
                                                fig, ax = plt.subplots(row, 1,figsize=(11.5,row*7.),sharex=sharxy, sharey=sharxy)

                                            for p in range(row):

                                                if(p==0 and q==3):
                                                    x = data_cut_obs[:,8]; y = data_cut_obs[:,9];
                                                    xbin_all = meanlinbins_all; ybin_all = etlogbins_all;
                                                    wxy_me, x_me, y_me = np.histogram2d(x,y, bins=(xbin_all, ybin_all) )

                                                    xbin = meanlinbins; ybin = etlogbins;
                                                    wxy_me_bh, x_me_bh, y_me_bh = np.histogram2d(x,y, bins=(xbin, ybin) )

                                                    labl = lbl_obs;
                                                    label_x=xlab[0][1]; label_y=xlab[1][1]
                                                if(p==0 and q==4):
                                                    x = data_cut_obs[:,9]; y = data_cut_obs[:,10];
                                                    xbin_all = etlogbins_all;     ybin_all = seedlinbins_all;
                                                    wxy_es, x_es, y_es = np.histogram2d(x,y, bins=(xbin_all, ybin_all) )

                                                    xbin = etlogbins;     ybin = seedlinbins;
                                                    wxy_es_bh, x_es_bh, y_ems_bh = np.histogram2d(x,y, bins=(xbin, ybin) )

                                                    labl = lbl_obs;
                                                    label_x=xlab[1][1]; label_y=xlab[1][0]
                                                if(p==0 and q==5):
                                                    x = data_cut_obs[:,10]; y = data_cut_obs[:,8];
                                                    xbin_all = seedlinbins_all; ybin_all = meanlinbins_all;
                                                    wxy_sm, x_sm, y_sm = np.histogram2d(x,y, bins=(xbin_all, ybin_all) )

                                                    xbin = seedlinbins; ybin = meanlinbins;
                                                    wxy_sm_bh, x_sm_bh, y_sm_bh = np.histogram2d(x,y, bins=(xbin, ybin) )

                                                    labl = lbl_obs;
                                                    label_x=xlab[1][0]; label_y=xlab[0][1]

                                                #####################################################################################
                                                if(plt_sim==True):

                                                    if(p==1 and q==3):
                                                        x = data_cut_sim_0[:,8]; y = data_cut_sim_0[:,9];
                                                        xbin_all = meanlinbins_all; ybin_all = etlogbins_all;
                                                        wxy_me_0sim, x_me_0sim, y_me_0sim = np.histogram2d(x,y, bins=(xbin_all, ybin_all) )

                                                        xbin = meanlinbins; ybin = etlogbins;
                                                        wxy_me_0sim_bh, x_me_0sim_bh, y_me_0sim_bh = np.histogram2d(x,y, bins=(xbin, ybin) )

                                                        labl = lbl_sim0;
                                                        label_x=xlab[0][1]; label_y=xlab[1][1]
                                                    if(p==1 and q==4):
                                                        x = data_cut_sim_0[:,9]; y = data_cut_sim_0[:,10];
                                                        xbin_all = etlogbins_all; ybin_all = seedlinbins_all;
                                                        wxy_es_0sim, x_es_0sim, y_es_0sim = np.histogram2d(x,y, bins=(xbin_all, ybin_all) )

                                                        xbin = etlogbins;     ybin = seedlinbins;
                                                        wxy_es_0sim_bh, x_es_0sim_bh, y_es_0sim_bh = np.histogram2d(x,y, bins=(xbin, ybin) )

                                                        labl = lbl_sim0;
                                                        label_x=xlab[1][1]; label_y=xlab[1][0]
                                                    if(p==1 and q==5):
                                                        x = data_cut_sim_0[:,10]; y = data_cut_sim_0[:,8];
                                                        xbin_all = seedlinbins_all; ybin_all = meanlinbins_all;
                                                        wxy_sm_0sim, x_sm_0sim, y_sm_0sim = np.histogram2d(x,y, bins=(xbin_all, ybin_all) )

                                                        xbin = seedlinbins; ybin = meanlinbins;
                                                        wxy_sm_0sim_bh, x_sm_0sim_bh, y_sm_0sim_bh = np.histogram2d(x,y, bins=(xbin, ybin) )

                                                        labl = lbl_sim0;
                                                        label_x=xlab[1][0]; label_y=xlab[0][1]

                                                #####################################################################################

                                                if(zom==False):
                                                    if(q==3):x_zoom =  data_cut_obs[:,8].max(); y_zoom = data_cut_obs[:,9].max()
                                                    if(q==4):x_zoom =  data_cut_obs[:,9].max(); y_zoom = data_cut_obs[:,10].max()
                                                    if(q==5):x_zoom =  data_cut_obs[:,10].max(); y_zoom = data_cut_obs[:,8].max()

                                                if(zom==True):
                                                    if(q==3):x_zoom = 300;   y_zoom = 2000
                                                    if(q==4):x_zoom = 2000;  y_zoom = 500
                                                    if(q==5):x_zoom = 500;   y_zoom = 300


                                                if(plt_one == False):
                                                    if(plt_sim==False and plt_sim1==False):
                                                        h=ax[q-3].hist2d(x,y, bins=(xbin, ybin), range=([0, x.max()],[0, y.max()]), cmin = 0 , cmax = 1e7 , norm=matplotlib.colors.LogNorm(),cmap=plt.cm.jet)
                                                        #hf=ax[q-3].hist2d(xf,yf, bins=(int(xf.max()),int(yf.max())), range=([0, xf.max()],[0, yf.max()]), norm=matplotlib.colors.LogNorm(),cmap=plt.cm.spring)

                                                        ax[q-3].set_xlim(0,x_zoom)
                                                        ax[q-3].set_ylim(0,y_zoom)

                                                        ax[q-3].set_xlabel(label_x, fontsize=font_siz+2)
                                                        ax[q-3].set_ylabel(label_y, fontsize=font_siz+2)
                                                        ax[q-3].set_title(labl, y=1.0, pad=25, fontweight='bold')#+lab_zom)#, fontsize=22 )#+' and '+lab_bg+lab_zom )
                                                        #ax[q-3].legend()
                                                        ax[q-3].grid(grid_)

                                                        divider = make_axes_locatable(ax[q-3])
                                                        #cax = divider.append_axes('right', size='3%', pad=0.07)
                                                        #cax1 = divider.append_axes('right', size='3%', pad=1.35)

                                                        #plt.colorbar(h[3], label=labl, ax=ax[q-3], cax=cax)

                                                        #fig.colorbar(h[3], ax=ax[q-3], cax=cax)#, label='Clusters number')

                                                    else:
                                                        #hf=ax[p,q-3].hist2d(xf,yf, bins=(int(xf.max()),int(yf.max())), range=([0, xf.max()],[0, yf.max()]), norm=matplotlib.colors.LogNorm(),cmap=plt.cm.spring)
                                                        #h=ax[p,q-3].hist2d(x,y, bins=(int(x.max()),int(y.max())), range=([0, x.max()],[0, y.max()]),  norm=matplotlib.colors.LogNorm(),cmap=plt.cm.jet)

                                                        h=ax[p,q-3].hist2d(x,y, bins=(xbin, ybin), range=([0, x.max()],[0, y.max()]), cmin = 0 , cmax = 1e7 , norm=matplotlib.colors.LogNorm(),cmap=plt.cm.jet)
                                                        #hf=ax[p,q-3].hist2d(xf,yf, bins=(int(xf.max()),int(yf.max())), range=([0, xf.max()],[0, yf.max()]), norm=matplotlib.colors.LogNorm(),cmap=plt.cm.spring)

                                                        ax[p,q-3].set_xlim(0,x_zoom)
                                                        ax[p,q-3].set_ylim(0,y_zoom)

                                                        if(label_xy==True):
                                                            ax[0,q-3].xaxis.set_tick_params(labelbottom=False)
                                                            if(row==2):
                                                                ax[1,q-3].set_xlabel(label_x, fontsize=font_siz+2)
                                                                ax[1,q-3].set_ylabel(label_y, x=0.02, y=1, fontsize=font_siz+2)
                                                            if(row==3):
                                                                ax[1,q-3].xaxis.set_tick_params(labelbottom=False)
                                                                ax[2,q-3].set_xlabel(label_x, fontsize=font_siz+2)
                                                                ax[1,q-3].set_ylabel(label_y, fontsize=font_siz+2)
                                                            #ax[1,q-3].supylabel(label_y, fontsize=font_siz+2)
                                                            ax[p,1].set_title(labl, y=1.0, pad=25, fontweight='bold')#+lab_zom)#, fontsize=22 )#+' and '+lab_bg+lab_zom )
                                                        if(label_xy==False):
                                                            ax[p,q-3].set_xlabel(label_x, fontsize=font_siz+2)
                                                            ax[p,q-3].set_ylabel(label_y, fontsize=font_siz+2)
                                                            ax[p,1].set_title(labl, y=1.0, pad=25, fontweight='bold')#+lab_zom)#, fontsize=22 )#+' and '+lab_bg+lab_zom )
                                                        #ax[p,q-3].legend()
                                                        ax[p,q-3].grid(grid_)

                                                        divider = make_axes_locatable(ax[p,q-3])
                                                        #cax = divider.append_axes('right', size='3%', pad=0.07)
                                                        #cax1 = divider.append_axes('right', size='3%', pad=1.35)

                                                        #plt.colorbar(h[3], label=labl, ax=ax[p,q-3], cax=cax)
                                                        #fig.colorbar(h[3], ax=ax[p,q-3], cax=cax)#, label='Clusters number')
                                                        #plt.colorbar(hf[3],label=lab_bg, ax=ax[p,q-3], cax=cax1)
                                                        ############################################################################################################
                                                        ############################################################################################################

                                                if(plt_one == True):
                                                    if(plt_sim==False and plt_sim1==False):
                                                        h=ax.hist2d(x,y, bins=(xbin, ybin), range=([0, x.max()],[0, y.max()]), cmin = 0 , cmax = 1e7 , norm=matplotlib.colors.LogNorm(),cmap=plt.cm.jet)

                                                        ax.set_xlim(0,x_zoom)
                                                        ax.set_ylim(0,y_zoom)

                                                        ax.set_xlabel(label_x, fontsize=font_siz+2)
                                                        ax.set_ylabel(label_y, fontsize=font_siz+2)
                                                        ax.set_title(labl, y=1.0, pad=25, fontsize=font_siz-1, fontweight='bold')#+lab_zom)#, fontsize=22 )#+' and '+lab_bg+lab_zom )
                                                        #ax.legend()
                                                        ax.grid(grid_)

                                                        divider = make_axes_locatable(ax)
                                                        #cax = divider.append_axes('right', size='3%', pad=0.07)

                                                        #fig.colorbar(h[3], ax=ax, cax=cax, label='Clusters number')#, ticks=v)

                                                    else:
                                                        h=ax[p].hist2d(x,y, bins=(xbin, ybin), range=([0, x.max()],[0, y.max()]), cmin = 0 , cmax = 1e7 , norm=matplotlib.colors.LogNorm(),cmap=plt.cm.jet)

                                                        ax[p].set_xlim(0,x_zoom)
                                                        ax[p].set_ylim(0,y_zoom)
                                                        if(label_xy==True):
                                                            if(row==2):ax[1].set_xlabel(label_x, fontsize=font_siz+2)
                                                            if(row==3):ax[2].set_xlabel(label_x, fontsize=font_siz+2)
                                                            #ax[1].set_ylabel(label_y, fontsize=font_siz+2)
                                                            fig.supylabel(label_y, fontsize=font_siz+2)
                                                            ax[p].set_title(labl, y=1.0, pad=25, fontsize=font_siz-1, fontweight='bold')#+lab_zom)#, fontsize=22 )#+' and '+lab_bg+lab_zom )
                                                        if(label_xy==False):
                                                            ax[p].set_xlabel(label_x, fontsize=font_siz+2)
                                                            ax[p].set_ylabel(label_y, fontsize=font_siz+2)
                                                            ax[p].set_title(labl, y=1.0, pad=25, fontsize=font_siz-1, fontweight='bold')#+lab_zom)#, fontsize=22 )#+' and '+lab_bg+lab_zom )
                                                        #ax[p].legend()
                                                        ax[p].grid(grid_)

                                                        divider = make_axes_locatable(ax[p])
                                                        #cax = divider.append_axes('right', size='3%', pad=0.07)

                                                        #fig.colorbar(h[3], ax=ax[p], cax=cax, label='Clusters number')

                                            if(plt_one == True):
                                                if(q==3):str_hist2 = 'Total_vs_Mean_ADC'
                                                if(q==4):str_hist2 = 'Maximum_vs_Total_ADC'
                                                if(q==5):str_hist2 = 'Mean_vs_Max_ADC'
                                                #fig.suptitle(str_hist2+lab_zom +dens+'_'+strbin +str(z*2).zfill(2)+'mm_sim'+'_'+source+'_'+str(nsigm_bg)+'sig', fontsize=36)

                                                # Colorbar
                                                #fig.subplots_adjust(right=0.85)
                                                if(row==2 or row==3):
                                                    p0 = ax[0].get_position().get_points().flatten()
                                                    p1 = ax[1].get_position().get_points().flatten()
                                                    ax_cbar = fig.add_axes([0.8, (row*p1[3]-(row-1)*p0[3])*0.75 , 0.05, p0[3]*1.025 ])
                                                    cbar= fig.colorbar(h[3], ax=ax, location = 'right', pad = -0.4 , aspect=100, shrink=0.0001)
                                                if(row==1):
                                                    p0 = ax.get_position().get_points().flatten()
                                                    ax_cbar = fig.add_axes([0.8, p0[1]*0.75 , 0.05, p0[3]*1.025 ])
                                                    cbar= fig.colorbar(h[3], ax=ax, location = 'right', pad = -0.7 , aspect=100, shrink=0.0001)
                                                #print(p0,p1,p2)
                                                #ax_cbar = fig.add_axes([1, (p2[3]-p0[3]+p1[3])*0.75 , 0.03, p0[3]*1.025 ])
                                                #if(row==3):ax_cbar = fig.add_axes([0.8, (p2[3]-p0[3]+p1[3])*0.75 , 0.06, p0[3]*1.025 ])
                                                #if(row==2):ax_cbar = fig.add_axes([0.8, (2*p1[3]-p0[3])*0.75 , 0.06, p0[3]*1.025 ])
                                                #ax_cbar = fig.add_axes([0.8, (row*p1[3]-(row-1)*p0[3])*0.75 , 0.06, p0[3]*1.025 ])

                                                #cbar_ax = fig.add_axes([1., 0.044, 0.025, 0.927])
                                                #cbar= fig.colorbar(h[3], ax=ax, location = 'right', pad = -0.4 , aspect=100, shrink=0.0001)
                                                cbar.set_ticks([-0.01, 0])
                                                cbar.set_ticklabels(['', ''])
                                                cbar.outline.set_visible(False)

                                                cb = fig.colorbar(h[3], ax=ax, cax=ax_cbar)
                                                #ax.set_aspect('auto')
                                                cb.set_label(label='Clusters number',size=font_siz+4)
                                                cb.ax.tick_params(labelsize=font_siz)

                                                fig.tight_layout()
                                                #namesave = 'plot_hists2D'+lblxy+lab_zom +dens+'_'+str_hist2+'_'+strbin +str(z*2).zfill(2)+'mm_sim'+'_'+source+'_'+str(nsigm_bg)+'sig'+'_'+ev+'ev_'+fecha+str_hist_comp+cut_adc+cut_str
                                                namesave = 'plot_hists2D'+lblxy+lab_zom +dens+'_'+str_hist2+'_'+strbin +str(z*2).zfill(2)+'mm'+ name_cuts
                                                plt.savefig(dirsave_plt+'/'+ namesave+difu_gauss+'.png', dpi=150)
                                                # plt.show()
                                                #plt.clf()
                                                #plt.close()
                                        if(plt_one == False):
                                            #fig.suptitle('Hists_2D'+lab_zom +dens+'_'+strbin +str(z*2).zfill(2)+'mm_sim'+'_'+source+'_'+str(nsigm_bg)+'sig'\
                                               #          +'_'+ev+'ev_sim_bg_'+ 'eta_'+ '_Dat_'+fecha+cut_str+cut_adc, fontsize=36)
                                                        # +'_'+ev+'ev_sim_bg_'+ 'eta_'+ '_Dat_'+fecha+cut_str+cut_adc, fontsize=36)
                                            if(row==2 or row==3):
                                                p0 = ax[0,2].get_position().get_points().flatten()
                                                p1 = ax[1,2].get_position().get_points().flatten()
                                                ax_cbar = fig.add_axes([0.925, (row*p1[3]-(row-1)*p0[3])*0.75 , 0.02, p0[3]*1.025 ])
                                                #cbar= fig.colorbar(h[3], ax=ax, location = 'right', pad = -0.25 , aspect=100, shrink=0.0001)

                                            if(row==1):
                                                p0 = ax[2].get_position().get_points().flatten()
                                                ax_cbar = fig.add_axes([0.925, p0[1]*0.75 , 0.02, p0[3]*1.025 ])
                                                #cbar= fig.colorbar(h[3], ax=ax, location = 'right', pad = -0.25 , aspect=100, shrink=0.0001)

                                            #if(row==3):ax_cbar = fig.add_axes([0.925, (p2[3]-p0[3]+p1[3])*0.75 , 0.02, p0[3]*1.025 ])
                                            #if(row==2):ax_cbar = fig.add_axes([0.925, (2*p1[3]-p0[3])*0.75 , 0.02, p0[3]*1.025 ])
                                            #ax_cbar = fig.add_axes([0.925, (row*p1[3]-(row-1)*p0[3])*0.75 , 0.02, p0[3]*1.025 ])
                                            #cbar_ax = fig.add_axes([1., 0.044, 0.025, 0.927])
                                            cbar= fig.colorbar(h[3], ax=ax, location = 'right', pad = -0.25 , aspect=100, shrink=0.0001)
                                            cbar.set_ticks([-0.01, 0])
                                            cbar.set_ticklabels(['', ''])
                                            cbar.outline.set_visible(False)

                                            cb = fig.colorbar(h[3], ax=ax[:], cax=ax_cbar)
                                            #ax.set_aspect('auto')
                                            cb.set_label(label='Clusters number',size=font_siz+4)
                                            cb.ax.tick_params(labelsize=font_siz)

                                            fig.tight_layout()

                                            #namesave = 'plot_hists2D'+lblxy+lab_zom +dens+'_All_2_'+strbin +str(z*2).zfill(2)+'mm_sim'+'_'+source+'_'+str(nsigm_bg)+'sig'+'_'+ev+'ev_'+fecha+str_hist_comp+cut_adc+cut_str
                                            namesave = 'plot_hists2D'+lblxy+lab_zom +dens+'_All_2_'+strbin +str(z*2).zfill(2)+'mm'+ name_cuts

                                            plt.savefig(dirsave_plt+'/'+ namesave+difu_gauss+'.png', dpi=150)
                                            plt.savefig(dirsave_plt+'/'+ namesave+difu_gauss+'.pdf', dpi=150)

                                            print(namesave)
                                            # plt.show()
                                            #plt.clf()
                                            #plt.close()

                            ##########################
                            print('\n\n\n\n')
                            #print('ADC =', backg[0].max(), bg[0].max(), ' cluster = ', data_bg_cut[:,7].max(), ' mean = ' , data_bg_cut[:,8].max(), ' max = ' , data_bg_cut[:,10].max(), ' total = ' , data_bg_cut[:,9].max())

                            if( file_csv==True):
                                file_name_test = dirsave_plt+'/'+source+dens+'_'+strbin+str(z_count)+'l_'+fecha+'_'+evt+'_'+str(z*2).zfill(2)+'mm'+'sim_bg_clst'+ cut_str+ str_cuts+'_'+str(max_adc)+'ADC'+'_chi2.csv'
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

                                    err_ch = np.sqrt(err_std_obs_c*err_std_obs_c + err_std_sim_0_c*err_std_sim_0_c )
                                    test_corr = chi_2_sigm_test(obs_c, sim_0_c, err_ch)
                                    txt_test = 'Chi2_sig_test_sim_'+d_gauss +'\t'+'hist_Cluster_size'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%test_corr[0]) +'\t'+ str(r'%.5f'%test_corr[1])  +'\t'+ str(r'%.5f'%test_corr[2])  +'\t'+ str(test_corr[3]) +'\t'+ str(r'%.5f'%test_corr[4])
                                    file_test.write(txt_test + '\n')
                                    print(txt_test)

                                    err_ch = np.sqrt(err_std_obs_m*err_std_obs_m + err_std_sim_0_m*err_std_sim_0_m )
                                    test_corr = chi_2_sigm_test(obs_m, sim_0_m, err_ch)
                                    txt_test = 'Chi2_sig_test_sim_'+d_gauss +'\t'+'hist_Mean_clst'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%test_corr[0]) +'\t'+ str(r'%.5f'%test_corr[1])  +'\t'+ str(r'%.5f'%test_corr[2])  +'\t'+ str(test_corr[3]) +'\t'+ str(r'%.5f'%test_corr[4])
                                    file_test.write(txt_test + '\n')
                                    print(txt_test)

                                    err_ch = np.sqrt(err_std_obs_e*err_std_obs_e + err_std_sim_0_e*err_std_sim_0_e )
                                    test_corr = chi_2_sigm_test(obs_e, sim_0_e, err_ch)
                                    txt_test = 'Chi2_sig_test_sim_'+d_gauss +'\t'+'hist_Total_clst'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%test_corr[0]) +'\t'+ str(r'%.5f'%test_corr[1])  +'\t'+ str(r'%.5f'%test_corr[2])  +'\t'+ str(test_corr[3]) +'\t'+ str(r'%.5f'%test_corr[4])
                                    file_test.write(txt_test + '\n')
                                    print(txt_test)

                                    err_ch = np.sqrt(err_std_obs_s*err_std_obs_s + err_std_sim_0_s*err_std_sim_0_s )
                                    test_corr = chi_2_sigm_test(obs_s, sim_0_s, err_ch)
                                    txt_test = 'Chi2_sig_test_sim_'+d_gauss +'\t'+'hist_Max_clst'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%test_corr[0]) +'\t'+ str(r'%.5f'%test_corr[1])  +'\t'+ str(r'%.5f'%test_corr[2])  +'\t'+ str(test_corr[3]) +'\t'+ str(r'%.5f'%test_corr[4])
                                    file_test.write(txt_test + '\n')
                                    print(txt_test)

                                if(plots_2d==True):
                                    print('\n')
                                    if(plt_sim1==True and plt_sim==False):
                                        txt_test = 'test bins'+sig_bg_thresh+ cut_clst_size+cut_max + cut_adc + cut_max_clst_porc+sig_bg_thresh_sim+cut_max_clst_porc_sim+dens+'\t' + 'hist2D\t' + 'bins\t' + 'chi_2\t' + 'pvalue\t' + 'criti_val\t' + 'ndf\t' + 'chi2/nf'
                                        file_test.write('\n\n'+ txt_test + '\n')
                                        print(txt_test)

                                        err_ch = np.sqrt(wxy_mc_bh+0.04*wxy_mc_sim_bh*wxy_mc_sim_bh)
                                        test_corr = chi_2_sigm_test(wxy_mc_bh, wxy_mc_sim_bh, err_ch)
                                        txt_test = 'Chi2_sig_test_sim_bg_'+d_gauss +'\t'+'hist_Cluste_vs_Mean_clst'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%test_corr[0]) +'\t'+ str(r'%.5f'%test_corr[1])  +'\t'+ str(r'%.5f'%test_corr[2])  +'\t'+ str(test_corr[3]) +'\t'+ str(r'%.5f'%test_corr[4])
                                        file_test.write(txt_test + '\n')
                                        print(txt_test)

                                        err_ch = np.sqrt(wxy_ec_bh+0.04*wxy_ec_sim_bh*wxy_ec_sim_bh)
                                        test_corr = chi_2_sigm_test(wxy_ec_bh, wxy_ec_sim_bh, err_ch)
                                        txt_test = 'Chi2_sig_test_sim_bg_'+d_gauss +'\t'+'hist_Cluste_vs_Total_clst'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%test_corr[0]) +'\t'+ str(r'%.5f'%test_corr[1])  +'\t'+ str(r'%.5f'%test_corr[2])  +'\t'+ str(test_corr[3]) +'\t'+ str(r'%.5f'%test_corr[4])
                                        file_test.write(txt_test + '\n')
                                        print(txt_test)

                                        err_ch = np.sqrt(wxy_sc_bh+0.04*wxy_sc_sim_bh*wxy_sc_sim_bh)
                                        test_corr = chi_2_sigm_test(wxy_sc_bh, wxy_sc_sim_bh, err_ch)
                                        txt_test = 'Chi2_sig_test_sim_bg_'+d_gauss +'\t'+'hist_Cluste_vs_Max_clst'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%test_corr[0]) +'\t'+ str(r'%.5f'%test_corr[1])  +'\t'+ str(r'%.5f'%test_corr[2])  +'\t'+ str(test_corr[3]) +'\t'+ str(r'%.5f'%test_corr[4])
                                        file_test.write(txt_test + '\n')
                                        print(txt_test)

                                        err_ch = np.sqrt(wxy_me_bh+0.04*wxy_me_sim_bh*wxy_me_sim_bh)
                                        test_corr = chi_2_sigm_test(wxy_me_bh, wxy_me_sim_bh, err_ch)
                                        txt_test = 'Chi2_sig_test_sim_bg_'+d_gauss +'\t'+'hist_Total_vs_Mean_clst'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%test_corr[0]) +'\t'+ str(r'%.5f'%test_corr[1])  +'\t'+ str(r'%.5f'%test_corr[2])  +'\t'+ str(test_corr[3]) +'\t'+ str(r'%.5f'%test_corr[4])
                                        file_test.write(txt_test + '\n')
                                        print(txt_test)

                                        err_ch = np.sqrt(wxy_es_bh+0.04*wxy_es_sim_bh*wxy_es_sim_bh)
                                        test_corr = chi_2_sigm_test(wxy_es_bh, wxy_es_sim_bh, err_ch)
                                        txt_test = 'Chi2_sig_test_sim_bg_'+d_gauss +'\t'+'hist_Max_vs_Total_clst'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%test_corr[0]) +'\t'+ str(r'%.5f'%test_corr[1])  +'\t'+ str(r'%.5f'%test_corr[2])  +'\t'+ str(test_corr[3]) +'\t'+ str(r'%.5f'%test_corr[4])
                                        file_test.write(txt_test + '\n')
                                        print(txt_test)

                                        err_ch = np.sqrt(wxy_sm_bh+0.04*wxy_sm_sim_bh*wxy_sm_sim_bh)
                                        test_corr = chi_2_sigm_test(wxy_sm_bh, wxy_sm_sim_bh, err_ch)
                                        txt_test = 'Chi2_sig_test_sim_bg_'+d_gauss +'\t'+'hist_Mean_vs_Max_clst'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%test_corr[0]) +'\t'+ str(r'%.5f'%test_corr[1])  +'\t'+ str(r'%.5f'%test_corr[2])  +'\t'+ str(test_corr[3]) +'\t'+ str(r'%.5f'%test_corr[4])
                                        file_test.write(txt_test + '\n')
                                        print(txt_test)
                                    #####################################################################################################################################################################################################

                                    if(plt_sim1==False and plt_sim==True):

                                        print('\n')
                                        txt_test = 'test bins'+sig_bg_thresh+ cut_clst_size+cut_max + cut_adc + cut_max_clst_porc+sig_bg_thresh_sim+cut_max_clst_porc_sim+dens+'\t' + 'hist2D\t' + 'bins\t' + 'chi_2\t' + 'pvalue\t' + 'criti_val\t' + 'ndf\t' + 'chi2/nf'
                                        file_test.write('\n\n'+ txt_test + '\n')
                                        print(txt_test)

                                        err_ch = np.sqrt(wxy_mc_bh+0.04*wxy_mc_0sim_bh*wxy_mc_0sim_bh)
                                        test_corr = chi_2_sigm_test(wxy_mc_bh, wxy_mc_0sim_bh, err_ch)
                                        txt_test = 'Chi2_sig_test_sim_'+d_gauss +'\t'+'hist_Cluste_vs_Mean_clst'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%test_corr[0]) +'\t'+ str(r'%.5f'%test_corr[1])  +'\t'+ str(r'%.5f'%test_corr[2])  +'\t'+ str(test_corr[3]) +'\t'+ str(r'%.5f'%test_corr[4])
                                        file_test.write(txt_test + '\n')
                                        print(txt_test)

                                        err_ch = np.sqrt(wxy_ec_bh+0.04*wxy_ec_0sim_bh*wxy_ec_0sim_bh)
                                        test_corr = chi_2_sigm_test(wxy_ec_bh, wxy_ec_0sim_bh, err_ch)
                                        txt_test = 'Chi2_sig_test_sim_'+d_gauss +'\t'+'hist_Cluste_vs_Total_clst'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%test_corr[0]) +'\t'+ str(r'%.5f'%test_corr[1])  +'\t'+ str(r'%.5f'%test_corr[2])  +'\t'+ str(test_corr[3]) +'\t'+ str(r'%.5f'%test_corr[4])
                                        file_test.write(txt_test + '\n')
                                        print(txt_test)

                                        err_ch = np.sqrt(wxy_sc_bh+0.04*wxy_sc_0sim_bh*wxy_sc_0sim_bh)
                                        test_corr = chi_2_sigm_test(wxy_sc_bh, wxy_sc_0sim_bh, err_ch)
                                        txt_test = 'Chi2_sig_test_sim_'+d_gauss +'\t'+'hist_Cluste_vs_Max_clst'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%test_corr[0]) +'\t'+ str(r'%.5f'%test_corr[1])  +'\t'+ str(r'%.5f'%test_corr[2])  +'\t'+ str(test_corr[3]) +'\t'+ str(r'%.5f'%test_corr[4])
                                        file_test.write(txt_test + '\n')
                                        print(txt_test)

                                        err_ch = np.sqrt(wxy_me_bh+0.04*wxy_me_0sim_bh*wxy_me_0sim_bh)
                                        test_corr = chi_2_sigm_test(wxy_me_bh, wxy_me_0sim_bh, err_ch)
                                        txt_test = 'Chi2_sig_test_sim_'+d_gauss +'\t'+'hist_Total_vs_Mean_ADC'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%test_corr[0]) +'\t'+ str(r'%.5f'%test_corr[1])  +'\t'+ str(r'%.5f'%test_corr[2])  +'\t'+ str(test_corr[3]) +'\t'+ str(r'%.5f'%test_corr[4])
                                        file_test.write(txt_test + '\n')
                                        print(txt_test)

                                        err_ch = np.sqrt(wxy_es_bh+0.04*wxy_es_0sim_bh*wxy_es_0sim_bh)
                                        test_corr = chi_2_sigm_test(wxy_es_bh, wxy_es_0sim_bh, err_ch)
                                        txt_test = 'Chi2_sig_test_sim_'+d_gauss +'\t'+'hist_Max_vs_Total_clst'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%test_corr[0]) +'\t'+ str(r'%.5f'%test_corr[1])  +'\t'+ str(r'%.5f'%test_corr[2])  +'\t'+ str(test_corr[3]) +'\t'+ str(r'%.5f'%test_corr[4])
                                        file_test.write(txt_test + '\n')
                                        print(txt_test)

                                        err_ch = np.sqrt(wxy_sm_bh+0.04*wxy_sm_0sim_bh*wxy_sm_0sim_bh)
                                        test_corr = chi_2_sigm_test(wxy_sm_bh, wxy_sm_0sim_bh, err_ch)
                                        txt_test = 'Chi2_sig_test_sim_'+d_gauss +'\t'+'hist_Mean_vs_Max_clst'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%test_corr[0]) +'\t'+ str(r'%.5f'%test_corr[1])  +'\t'+ str(r'%.5f'%test_corr[2])  +'\t'+ str(test_corr[3]) +'\t'+ str(r'%.5f'%test_corr[4])
                                        file_test.write(txt_test + '\n')
                                        print(txt_test)

                                    #####################################################################################################################################################################################################

                                    if(plt_sim1==True and plt_sim==True):
                                        txt_test = 'test bins'+sig_bg_thresh+ cut_clst_size+cut_max + cut_adc + cut_max_clst_porc+sig_bg_thresh_sim+cut_max_clst_porc_sim+dens+'\t' + 'hist2D\t' + 'bins\t' + 'chi_2\t' + 'pvalue\t' + 'criti_val\t' + 'ndf\t' + 'chi2/nf'
                                        file_test.write('\n\n'+ txt_test + '\n')
                                        print(txt_test)

                                        err_ch = np.sqrt(wxy_mc_bh+0.04*wxy_mc_sim_bh*wxy_mc_sim_bh)
                                        test_corr = chi_2_sigm_test(wxy_mc_bh, wxy_mc_sim_bh, err_ch)
                                        txt_test = 'Chi2_sig_test_sim_bg_'+d_gauss +'\t'+'hist_Cluste_vs_Mean_clst'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%test_corr[0]) +'\t'+ str(r'%.5f'%test_corr[1])  +'\t'+ str(r'%.5f'%test_corr[2])  +'\t'+ str(test_corr[3]) +'\t'+ str(r'%.5f'%test_corr[4])
                                        file_test.write(txt_test + '\n')
                                        print(txt_test)

                                        err_ch = np.sqrt(wxy_ec_bh+0.04*wxy_ec_sim_bh*wxy_ec_sim_bh)
                                        test_corr = chi_2_sigm_test(wxy_ec_bh, wxy_ec_sim_bh, err_ch)
                                        txt_test = 'Chi2_sig_test_sim_bg_'+d_gauss +'\t'+'hist_Cluste_vs_Total_clst'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%test_corr[0]) +'\t'+ str(r'%.5f'%test_corr[1])  +'\t'+ str(r'%.5f'%test_corr[2])  +'\t'+ str(test_corr[3]) +'\t'+ str(r'%.5f'%test_corr[4])
                                        file_test.write(txt_test + '\n')
                                        print(txt_test)

                                        err_ch = np.sqrt(wxy_sc_bh+0.04*wxy_sc_sim_bh*wxy_sc_sim_bh)
                                        test_corr = chi_2_sigm_test(wxy_sc_bh, wxy_sc_sim_bh, err_ch)
                                        txt_test = 'Chi2_sig_test_sim_bg_'+d_gauss +'\t'+'hist_Cluste_vs_Max_clst'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%test_corr[0]) +'\t'+ str(r'%.5f'%test_corr[1])  +'\t'+ str(r'%.5f'%test_corr[2])  +'\t'+ str(test_corr[3]) +'\t'+ str(r'%.5f'%test_corr[4])
                                        file_test.write(txt_test + '\n')
                                        print(txt_test)

                                        err_ch = np.sqrt(wxy_me_bh+0.04*wxy_me_sim_bh*wxy_me_sim_bh)
                                        test_corr = chi_2_sigm_test(wxy_me_bh, wxy_me_sim_bh, err_ch)
                                        txt_test = 'Chi2_sig_test_sim_bg_'+d_gauss +'\t'+'hist_Total_vs_Mean_clst'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%test_corr[0]) +'\t'+ str(r'%.5f'%test_corr[1])  +'\t'+ str(r'%.5f'%test_corr[2])  +'\t'+ str(test_corr[3]) +'\t'+ str(r'%.5f'%test_corr[4])
                                        file_test.write(txt_test + '\n')
                                        print(txt_test)

                                        err_ch = np.sqrt(wxy_es_bh+0.04*wxy_es_sim_bh*wxy_es_sim_bh)
                                        test_corr = chi_2_sigm_test(wxy_es_bh, wxy_es_sim_bh, err_ch)
                                        txt_test = 'Chi2_sig_test_sim_bg_'+d_gauss +'\t'+'hist_Max_vs_Total_clst'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%test_corr[0]) +'\t'+ str(r'%.5f'%test_corr[1])  +'\t'+ str(r'%.5f'%test_corr[2])  +'\t'+ str(test_corr[3]) +'\t'+ str(r'%.5f'%test_corr[4])
                                        file_test.write(txt_test + '\n')
                                        print(txt_test)

                                        err_ch = np.sqrt(wxy_sm_bh+0.04*wxy_sm_sim_bh*wxy_sm_sim_bh)
                                        test_corr = chi_2_sigm_test(wxy_sm_bh, wxy_sm_sim_bh, err_ch)
                                        txt_test = 'Chi2_sig_test_sim_bg_'+d_gauss +'\t'+'hist_Mean_vs_Max_clst'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%test_corr[0]) +'\t'+ str(r'%.5f'%test_corr[1])  +'\t'+ str(r'%.5f'%test_corr[2])  +'\t'+ str(test_corr[3]) +'\t'+ str(r'%.5f'%test_corr[4])
                                        file_test.write(txt_test + '\n')
                                        print(txt_test)
                                        #####################################################################################################################################################################################################
                                        print('\n')
                                        txt_test = 'test bins'+sig_bg_thresh+ cut_clst_size+cut_max + cut_adc + cut_max_clst_porc+sig_bg_thresh_sim+cut_max_clst_porc_sim+dens+'\t' + 'hist2D\t' + 'bins\t' + 'chi_2\t' + 'pvalue\t' + 'criti_val\t' + 'ndf\t' + 'chi2/nf'
                                        file_test.write('\n\n'+ txt_test + '\n')
                                        print(txt_test)

                                        err_ch = np.sqrt(wxy_mc_bh+0.04*wxy_mc_0sim_bh*wxy_mc_0sim_bh)
                                        test_corr = chi_2_sigm_test(wxy_mc_bh, wxy_mc_0sim_bh, err_ch)
                                        txt_test = 'Chi2_sig_test_sim_'+d_gauss +'\t'+'hist_Cluste_vs_Mean_clst'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%test_corr[0]) +'\t'+ str(r'%.5f'%test_corr[1])  +'\t'+ str(r'%.5f'%test_corr[2])  +'\t'+ str(test_corr[3]) +'\t'+ str(r'%.5f'%test_corr[4])
                                        file_test.write(txt_test + '\n')
                                        print(txt_test)

                                        err_ch = np.sqrt(wxy_ec_bh+0.04*wxy_ec_0sim_bh*wxy_ec_0sim_bh)
                                        test_corr = chi_2_sigm_test(wxy_ec_bh, wxy_ec_0sim_bh, err_ch)
                                        txt_test = 'Chi2_sig_test_sim_'+d_gauss +'\t'+'hist_Cluste_vs_Total_clst'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%test_corr[0]) +'\t'+ str(r'%.5f'%test_corr[1])  +'\t'+ str(r'%.5f'%test_corr[2])  +'\t'+ str(test_corr[3]) +'\t'+ str(r'%.5f'%test_corr[4])
                                        file_test.write(txt_test + '\n')
                                        print(txt_test)

                                        err_ch = np.sqrt(wxy_sc_bh+0.04*wxy_sc_0sim_bh*wxy_sc_0sim_bh)
                                        test_corr = chi_2_sigm_test(wxy_sc_bh, wxy_sc_0sim_bh, err_ch)
                                        txt_test = 'Chi2_sig_test_sim_'+d_gauss +'\t'+'hist_Cluste_vs_Max_clst'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%test_corr[0]) +'\t'+ str(r'%.5f'%test_corr[1])  +'\t'+ str(r'%.5f'%test_corr[2])  +'\t'+ str(test_corr[3]) +'\t'+ str(r'%.5f'%test_corr[4])
                                        file_test.write(txt_test + '\n')
                                        print(txt_test)

                                        err_ch = np.sqrt(wxy_me_bh+0.04*wxy_me_0sim_bh*wxy_me_0sim_bh)
                                        test_corr = chi_2_sigm_test(wxy_me_bh, wxy_me_0sim_bh, err_ch)
                                        txt_test = 'Chi2_sig_test_sim_'+d_gauss +'\t'+'hist_Total_vs_Mean_ADC'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%test_corr[0]) +'\t'+ str(r'%.5f'%test_corr[1])  +'\t'+ str(r'%.5f'%test_corr[2])  +'\t'+ str(test_corr[3]) +'\t'+ str(r'%.5f'%test_corr[4])
                                        file_test.write(txt_test + '\n')
                                        print(txt_test)

                                        err_ch = np.sqrt(wxy_es_bh+0.04*wxy_es_0sim_bh*wxy_es_0sim_bh)
                                        test_corr = chi_2_sigm_test(wxy_es_bh, wxy_es_0sim_bh, err_ch)
                                        txt_test = 'Chi2_sig_test_sim_'+d_gauss +'\t'+'hist_Max_vs_Total_clst'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%test_corr[0]) +'\t'+ str(r'%.5f'%test_corr[1])  +'\t'+ str(r'%.5f'%test_corr[2])  +'\t'+ str(test_corr[3]) +'\t'+ str(r'%.5f'%test_corr[4])
                                        file_test.write(txt_test + '\n')
                                        print(txt_test)

                                        err_ch = np.sqrt(wxy_sm_bh+0.04*wxy_sm_0sim_bh*wxy_sm_0sim_bh)
                                        test_corr = chi_2_sigm_test(wxy_sm_bh, wxy_sm_0sim_bh, err_ch)
                                        txt_test = 'Chi2_sig_test_sim_'+d_gauss +'\t'+'hist_Mean_vs_Max_clst'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%test_corr[0]) +'\t'+ str(r'%.5f'%test_corr[1])  +'\t'+ str(r'%.5f'%test_corr[2])  +'\t'+ str(test_corr[3]) +'\t'+ str(r'%.5f'%test_corr[4])
                                        file_test.write(txt_test + '\n')
                                        print(txt_test)


                                #####################################################################################################################################################################################################

                                if(plt_sim==True):
                                    print('\n')
                                    txt_test = 'test_bins'+dens+'\t' + 'histo\t' + 'bins\t' + 'KStes\t' + 'pvalue\t'+ 'signif_level\t' + 'response'
                                    file_test.write('\n\n'+ txt_test + '\n')
                                    print(txt_test)

                                    test_corr = ks_test(x_w, y_w_sim0)
                                    txt_test = 'Kolmogorov_sim'+'\t'+'hist_ADC'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%test_corr[0]) +'\t'+ str(r'%.5f'%test_corr[1])  +'\t'+ str(r'%.5f'%test_corr[2])  +'\t'+ str(test_corr[3])
                                    file_test.write(txt_test + '\n')
                                    print(txt_test)

                                    test_corr = ks_test(obs_c, sim_0_c)
                                    txt_test = 'Kolmogorov_sim'+'\t'+'hist_Cluster_size'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%test_corr[0]) +'\t'+ str(r'%.5f'%test_corr[1])  +'\t'+ str(r'%.5f'%test_corr[2])  +'\t'+ str(test_corr[3])
                                    file_test.write(txt_test + '\n')
                                    print(txt_test)

                                    test_corr = ks_test(obs_m, sim_0_m)
                                    txt_test = 'Kolmogorov_sim'+'\t'+'hist_Mean_clst'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%test_corr[0]) +'\t'+ str(r'%.5f'%test_corr[1])  +'\t'+ str(r'%.5f'%test_corr[2])  +'\t'+ str(test_corr[3])
                                    file_test.write(txt_test + '\n')
                                    print(txt_test)

                                    test_corr = ks_test(obs_e, sim_0_e)
                                    txt_test = 'Kolmogorov_sim'+'\t'+'hist_Total_clst'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%test_corr[0]) +'\t'+ str(r'%.5f'%test_corr[1])  +'\t'+ str(r'%.5f'%test_corr[2])  +'\t'+ str(test_corr[3])
                                    file_test.write(txt_test + '\n')
                                    print(txt_test)

                                    test_corr = ks_test(obs_s, sim_0_s)
                                    txt_test = 'Kolmogorov_sim'+'\t'+'hist_Max_clst'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%test_corr[0]) +'\t'+ str(r'%.5f'%test_corr[1])  +'\t'+ str(r'%.5f'%test_corr[2])  +'\t'+ str(test_corr[3])
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
                                chi2_dict = {'DG_z_sig_'+str(0)+'_k_'+str(0)+'_alpha_'+str(0)+'_zff_'+str(0):{'chi2/nf_ADC'   :   ch2_adc0[4],
                                                                                                              'chi2/nf_clstr' :   ch2_c0[4],
                                                                                                              'chi2/nf_max'   :   ch2_s0[4],
                                                                                                              'chi2/nf_mean'  :   ch2_m0[4],
                                                                                                              'chi2/nf_total' :   ch2_e0[4],
                                                                                                            }

                                             }
                                file_test_chi2_nf = open(chi2_reduc_distr_test,'w')
                                txt_head_test_chi2 = 'test bins'+ str(bin_hist)+ cut_clst_size+cut_max + cut_adc + \
                                                    cut_max_clst_porc+sig_bg_thresh_sim+cut_max_clst_porc_sim+dens+'\t' + \
                                                    'chi2/nf_ADC\t' + 'chi2/nf_clstr\t' + 'chi2/nf_max\t' + 'chi2/nf_mean\t' + 'chi2/nf_total'
                                file_test_chi2_nf.write( txt_head_test_chi2 + '\n')

                                txt_test_chi2 = 'DG_z_sig_'+str(0)+'_k_'+str(0)+'_alpha_'+str(0)+'_zff_'+str(0) +'\t' + \
                                                str(ch2_adc0[4]) +'\t' + str(ch2_c0[4]) +'\t'+ str(ch2_s0[4]) + \
                                                '\t'+ str(ch2_m0[4]) +'\t'+ str(ch2_e0[4])
                                file_test_chi2_nf.write(txt_test_chi2 + '\n')
                                #file_test_chi2_nf.close()
                                print(txt_head_test_chi2)

                            chi2_dict['DG_z_sig_'+str(nsigm)+'_k_'+str(k)+'_alpha_'+str(alpha)+'_zff_'+str(zff)]={'chi2/nf_ADC'   :   ch2_adc1[4],
                                                                                                                  'chi2/nf_clstr' :   ch2_c1[4],
                                                                                                                  'chi2/nf_max'   :   ch2_s1[4],
                                                                                                                  'chi2/nf_mean'  :   ch2_m1[4],
                                                                                                                  'chi2/nf_total' :   ch2_e1[4],
                                                                                                                  }



                            txt_test_chi2 = 'DG_z_sig_'+str(nsigm)+'_k_'+str(k)+'_alpha_'+str(alpha)+'_zff_'+str(zff) +'\t' + \
                                            str(ch2_adc1[4]) +'\t' + str(ch2_c1[4]) +'\t'+ str(ch2_s1[4]) + \
                                            '\t'+ str(ch2_m1[4]) +'\t'+ str(ch2_e1[4])
                            file_test_chi2_nf.write(txt_test_chi2 + '\n')
                            #file_test_chi2_nf.close()
                            print(i_cont, txt_test_chi2)



                            if( file_total_adc_csv==True and densi==False ):
                                np.save(file_total_adc +'.npy', np.float64(total_adc_level))
                                np.savez_compressed(file_total_adc +'.npz', np.float64(total_adc_level))

                            if( file_clst_sum_csv==True and densi==False ):
                                np.save(file_clst_sum +'.npy', np.float64(numb_clst_level))
                                np.savez_compressed(file_clst_sum +'.npz', np.float64(numb_clst_level))

        file_test_chi2_nf.close()
        chi2_dict_save_path = dirsavepatn_plt+'All_'+source + sim_n + dens+'_'+strbin+str(z_count)+'l_'+fecha+'_'+evt+'_'+str(z*2).zfill(2)+'mm'+'sim_clst'+ simu_no_cut+str_cuts+'_'+str(max_adc)+'ADC'+'_chi2'
        np.savez(chi2_dict_save_path+'.npz', **chi2_dict)

    #loaded_data = np.load('data_dict.npz', allow_pickle=True)
    #data_dict_loaded = {key: loaded_data[key].item() for key in loaded_data}
    #print(data_dict_loaded)
