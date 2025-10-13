# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 22:16:33 2023

@author: mikb
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
from matplotlib.colors import LogNorm
from scipy.interpolate import make_interp_spline

import scipy
from scipy.stats import norm

from scipy.stats import ks_2samp
from scipy.stats import kstest

from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.image as mpimg

import os

from scipy import stats

from scipy.stats import chi2
from scipy.optimize import curve_fit

###############################################################################
###############################################################################

def chi_2_sigm(obs, cal, sigma):
    chi_2= 1* np.sum([((np.abs(a - b) / c)**2)
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


def comp_2sim(sim_g0, sim2):
    dif = sim_g0 - sim2
    err = 0.2*np.sqrt(sim_g0*sim_g0 + sim2*sim2 )
    comp = np.divide(dif, err , where=(err != 0))
    err_comp = 0.2*comp*np.sqrt( (err*err) / (dif*dif)
            + (sim_g0*sim_g0*sim_g0*sim_g0 + sim2*sim2*sim2*sim2) / (4*err*err*err*err) )
    return comp, np.abs(err_comp)

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

def ratio_2sim(sim_g0, sim2):
    # Set ratio where obs2 is not zero
    ratio = np.divide(sim_g0, sim2 , where=(sim2 != 0))
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
nframes = 1000  # Number of frames
z_count = 1  # Count of Z levels

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

plt_sim = 0
plt_obs = 1

plt_sim_gauss = 1
plt_sim_nogauss = 0


splt_obs = ''
splt_simg = ''
splt_sim = ''

if(plt_obs==True): splt_obs = '_obs'
else:splt_obs = ''
if(plt_sim==True):
    if(plt_sim_gauss==True): splt_simg = '_simg'
    else: splt_simg = ''
    if(plt_sim_nogauss==True): splt_sim = '_sim'
    else: splt_sim = ''

plt_bg = 0
plt_err=1


log_y = 1

label_xy = True

grid_ = False

file_csv = 1
file_clst_sum_csv = 1
file_total_adc_csv = 1

if(plt_sim_nogauss==False and plt_sim_gauss==True): row = 2 ; str_hist_comp = '_Sim_gauss'
if(plt_sim_nogauss==True and plt_sim_gauss==False): row = 2; str_hist_comp = '_Sim'
if(plt_sim_nogauss==True and plt_sim_gauss==True): row = 3; str_hist_comp = '_All'
if(plt_sim_nogauss==False and plt_sim_gauss==False): row = 1 ; str_hist_comp = '_Obs'; label_xy=False
########################################################################################
maxim = np.zeros((2,2),dtype=int)

l_width = 4
font_siz=24

###############################################################

z_count = 1

level_z = list(range(z_count))
#level_z = [1]

tiemp = '_500ms'

source0 = 'Sr90'
source1 = 'Cs137'


ssd_sim = 'C:/dat_2025/sim2025/'

size_thresh = 0
max_thresh = 0

nsigm_self = 0

sim_bg = '_ag8_bg'
sim_bg = ''

adc_cut = 0

#if(adc_cut == 0):min_adc=1

simu_no_cut ='_sim_nocut'
simu_no_cut =''

source0_bg = 'backgnd'
fecha_bg = '2023_Oct_22_00h'
nfrm =1000

if(source0 == 'Sr90'):
    winback0='254'
    fecha0 = '2023_Oct_24_23h'
    ev0 = '1514'
    evt0 = winback0+'_ev'+ev0
    nfrm =1000

if(source1 == 'Cs137'):
    winback1='635'
    fecha1 = '2023_Oct_23_23h'
    ev1 = '3811'
    evt1 = winback1+'_ev'+ev1
    nfrm =1000
'''
source0_bg = 'backgnd'
fecha_bg = '2022_Nov_10'
nfrm =100

if(source0 == 'Sr90'):
    winback0='254'
    fecha0 = '2021_Nov_09'
    ev0 = '1587'
    evt0 = winback0+'_ev'+ev0
    nfrm =100

if(source1 == 'Cs137'):
    winback1='635'
    fecha1 = '2021_Nov_23'
    ev1 = '3991'
    evt1 = winback1+'_ev'+ev1
    nfrm =100
'''

########################################################################################
bin_hist = 40
if(bin_hist>0): strbin = str(bin_hist)+'bin_z_'
else:strbin = 'z_'
if(bin_hist==0):plots_2d = 0
else: plots_2d = 1
#log_y = 1

max_adc=1024
#max_adc=150
min_adc=100

cut0 = 1; cut_dat00= 6; cut_dat01=None; cut_type0='_clsiz'
#cut0 = 1; cut_dat00= 10; cut_dat01=cut_dat00; cut_type0='_cl_size'

cut1 = 0; cut_dat10=50; cut_dat11=650; cut_type1='_cl_mean'
#cut1 = 1; cut_dat10=55; cut_dat11=55.00001; cut_type1='_cl_mean'
cut2 = 1; cut_dat20=101; cut_dat21=None; cut_type2='_clmax_adc'
cut3 = 0; cut_dat30=90; cut_dat31=None; cut_type3='_cl_totalE'

errors_sqrt = 1

labl_opt = 'simple' #'full' , 'simple', 'off'
#labl_opt ='chi2_nu'

yscal_adc = 0
yscal_cls = 0
yscal_max = 0

yscal_adc = 2e-2
yscal_cls = 5e-1
yscal_max = 15e-3


if(max_adc<1024):cut_adcmax_str = "_mx"+str(max_adc)
else: cut_adcmax_str = ''

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

nsigm_bg_sim = 0

nframes = nfrm


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


gauss_dist = 'Dg2'

total_sim = 10

n_sampl = total_sim - 0

all_pixel_nframe=width*height*nfrm
for nsigm in [0]:
    for k in [0.002]:
        k = np.round(k, 3)
        k_str = str(int(1000 * np.round(k, 3)))

        for alpha in [1.75]:
            alpha = np.round(alpha, 3)
            a_str = str(int(1000 * np.round(alpha, 3)))

            for zff in [1.0]:
                zff = np.round(zff, 3)

                for densi in [#False,
                              True
                              ]:
                    if densi:   dens = '_norm'; histo='norm_'
                    else:   dens = '' ; histo=''

                    for z in level_z:

                        sum_bins_adc_sim_g0 = 0
                        sum_counts_adc_sim_g0 = 0
                        sum_bins_adc_sim_g1 = 0
                        sum_counts_adc_sim_g1 = 0

                        sum_2_bins_adc_sim_g0 = 0
                        sum_2_counts_adc_sim_g0 = 0
                        sum_2_bins_adc_sim_g1 = 0
                        sum_2_counts_adc_sim_g1 = 0

                        sum_bins_clstr_siz_sim_g0 = 0
                        sum_counts_clstr_siz_sim_g0 = 0
                        sum_bins_clstr_siz_sim_g1 = 0
                        sum_counts_clstr_siz_sim_g1 = 0

                        sum_2_bins_clstr_siz_sim_g0 = 0
                        sum_2_counts_clstr_siz_sim_g0 = 0
                        sum_2_bins_clstr_siz_sim_g1 = 0
                        sum_2_counts_clstr_siz_sim_g1 = 0

                        sum_bins_clstr_max_sim_g0 = 0
                        sum_counts_clstr_max_sim_g0 = 0
                        sum_bins_clstr_max_sim_g1 = 0
                        sum_counts_clstr_max_sim_g1 = 0

                        sum_2_bins_clstr_max_sim_g0 = 0
                        sum_2_counts_clstr_max_sim_g0 = 0
                        sum_2_bins_clstr_max_sim_g1 = 0
                        sum_2_counts_clstr_max_sim_g1 = 0

                        ############################################################
                        ############################################################


                        for sim_n in range(total_sim):
                            sim_n = '_s' + str(sim_n)

                            if nsigm > 0 :
                                difu_gauss = f"_{gauss_dist}_{k}k_{alpha}a_{nsigm}sig_{zff}zff"
                                d_gauss = gauss_dist+'_z_'+r'$\sigma$ = '+str(nsigm)+r', $\k$ = '+str(k)+r', $\alpha$ = '+str(alpha)+ r', $z_{ff}$ = '+str(zff)
                            if nsigm == 0 :
                                difu_gauss = f"_{gauss_dist}_{k}k_{alpha}a_{zff}zff"
                                d_gauss =gauss_dist+'_z_'+r'$\k$ = '+str(k)+r', $\alpha$ = '+str(alpha)+ r', $z_{ff}$ = '+str(zff)

                            dirsavepatn_plt0 = ssd_sim +gauss_dist+'_'+source0 +'_rslhist_'+strbin +str(z*2).zfill(2)+'mm'+str_cuts
                            dirsavepatn_plt1 = ssd_sim +gauss_dist+'_'+source1 +'_rslhist_'+strbin +str(z*2).zfill(2)+'mm'+str_cuts
                            dirsave_plt0 = dirsavepatn_plt0 + '/plots'+sim_n+'_'+source0+'_'+str(z_count)+'l'+'_'+str(nframes)+'f'+simu_no_cut+condit_edep+difu_gauss+'/plot_hist_mult_'+source0 + '_'+str(z_count)+'l_'+fecha0+'_'+evt0+sim_bg  #+ cuts_dat +simu_no_cut
                            dirsave_plt1 = dirsavepatn_plt1 + '/plots'+sim_n+'_'+source1+'_'+str(z_count)+'l'+'_'+str(nframes)+'f'+simu_no_cut+condit_edep+difu_gauss+'/plot_hist_mult_'+source1 + '_'+str(z_count)+'l_'+fecha1+'_'+evt1+sim_bg  #+ cuts_dat +simu_no_cut

                            dirsavepatn_plt = ssd_sim +gauss_dist+'_comp'+dens+'_rslhist_'+strbin +str(z*2).zfill(2)+'mm'+str_cuts+'/'
                            dirsave_plt = dirsavepatn_plt + 'plots'+sim_n+'_comp_'+str(z_count)+'l'+'_'+str(nframes)+'f'+simu_no_cut+condit_edep+difu_gauss+'/plot_hist_mult_comp_'+str(z_count)+'l_'+sim_bg  #+ cuts_dat +simu_no_cut

                            dirsave_plt_mean = 'C:/dat_2025/' +gauss_dist +dens+'_dghist_'+strbin +str(z*2).zfill(2)+'mm'+str_cuts+'_'+labl_opt +'_comp_mean/'+'plots'+'_'+source0+'_'+source1+'_'+str(z_count)+'l'+'_'+str(nframes)+'f'+simu_no_cut+condit_edep+difu_gauss+sim_bg  #+ cuts_dat +simu_no_cut
                            try:
                                os.makedirs(dirsave_plt_mean)
                            except FileExistsError:
                                pass
                            ################################################################################################################
                            #################################################################################
                            if sim_n == '_s0' :
                                hist_adc_obs0 = np.load(dirsave_plt+'/'+'hist_adc_obs'+source0+\
                                            sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', allow_pickle=True).item()
                                hist_adc_obs1 = np.load(dirsave_plt+'/'+'hist_adc_obs'+source1+\
                                            sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', allow_pickle=True).item()

                                bins_adc_obs0, counts_adc_obs0 = hist_adc_obs0['bins'], hist_adc_obs0['counts']
                                bins_adc_obs1, counts_adc_obs1 = hist_adc_obs1['bins'], hist_adc_obs1['counts']

                                hist_clstr_siz_obs0 = np.load(dirsave_plt+'/'+'hist_clstr_siz_obs_'+source0+\
                                                    sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', allow_pickle=True).item()
                                hist_clstr_siz_obs1 = np.load(dirsave_plt+'/'+'hist_clstr_siz_obs_'+source1+\
                                                    sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', allow_pickle=True).item()

                                bins_clstr_siz_obs0, counts_clstr_siz_obs0 = hist_clstr_siz_obs0['bins'], hist_clstr_siz_obs0['counts']
                                bins_clstr_siz_obs1, counts_clstr_siz_obs1 = hist_clstr_siz_obs1['bins'], hist_clstr_siz_obs1['counts']

                                hist_clstr_max_obs0 = np.load(dirsave_plt+'/'+'hist_clstr_max_obs_'+source0+\
                                                    sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', allow_pickle=True).item()
                                hist_clstr_max_obs1 = np.load(dirsave_plt+'/'+'hist_clstr_max_obs_'+source1+\
                                                    sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', allow_pickle=True).item()

                                bins_clstr_max_obs0, counts_clstr_max_obs0 = hist_clstr_max_obs0['bins'], hist_clstr_max_obs0['counts']
                                bins_clstr_max_obs1, counts_clstr_max_obs1 = hist_clstr_max_obs1['bins'], hist_clstr_max_obs1['counts']

                            print(f"Procesando simulación: _s{sim_n}")
                            try:

                                hist_adc_sim_g0 = np.load(dirsave_plt+'/'+'hist_adc_sim_'+source0+\
                                            sim_n+difu_gauss+ sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', allow_pickle=True).item()
                                hist_adc_sim_g1 = np.load(dirsave_plt+'/'+'hist_adc_sim_'+source1+\
                                            sim_n+difu_gauss+ sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', allow_pickle=True).item()

                                bins_adc_sim_g0, counts_adc_sim_g0 = hist_adc_sim_g0['bins'], hist_adc_sim_g0['counts']
                                bins_adc_sim_g1, counts_adc_sim_g1 = hist_adc_sim_g1['bins'], hist_adc_sim_g1['counts']
                                #################################################################################
                                # Acumular datos
                                sum_bins_adc_sim_g0 += bins_adc_sim_g0
                                sum_counts_adc_sim_g0 += counts_adc_sim_g0
                                sum_2_bins_adc_sim_g0 += bins_adc_sim_g0 ** 2
                                sum_2_counts_adc_sim_g0 += counts_adc_sim_g0 ** 2

                                sum_bins_adc_sim_g1 += bins_adc_sim_g1
                                sum_counts_adc_sim_g1 += counts_adc_sim_g1
                                sum_2_bins_adc_sim_g1 += bins_adc_sim_g1 ** 2
                                sum_2_counts_adc_sim_g1 += counts_adc_sim_g1 ** 2

                                #################################################################################
                                hist_clstr_siz_sim_g0 = np.load(dirsave_plt+'/'+'hist_clstr_siz_sim_'+source0+\
                                                    sim_n+'_'+difu_gauss+ sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', allow_pickle=True).item()
                                hist_clstr_siz_sim_g1 = np.load(dirsave_plt+'/'+'hist_clstr_siz_sim_'+source1+\
                                                    sim_n+'_'+difu_gauss+ sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', allow_pickle=True).item()

                                bins_clstr_siz_sim_g0, counts_clstr_siz_sim_g0 = hist_clstr_siz_sim_g0['bins'], hist_clstr_siz_sim_g0['counts']
                                bins_clstr_siz_sim_g1, counts_clstr_siz_sim_g1 = hist_clstr_siz_sim_g1['bins'], hist_clstr_siz_sim_g1['counts']
                                #################################################################################
                                # Acumular datos
                                sum_bins_clstr_siz_sim_g0 += bins_clstr_siz_sim_g0
                                sum_counts_clstr_siz_sim_g0 += counts_clstr_siz_sim_g0
                                sum_2_bins_clstr_siz_sim_g0 += bins_clstr_siz_sim_g0 ** 2
                                sum_2_counts_clstr_siz_sim_g0 += counts_clstr_siz_sim_g0 ** 2

                                sum_bins_clstr_siz_sim_g1 += bins_clstr_siz_sim_g1
                                sum_counts_clstr_siz_sim_g1 += counts_clstr_siz_sim_g1
                                sum_2_bins_clstr_siz_sim_g1 += bins_clstr_siz_sim_g1 ** 2
                                sum_2_counts_clstr_siz_sim_g1 += counts_clstr_siz_sim_g1 ** 2

                                #################################################################################
                                hist_clstr_max_sim_g0 = np.load(dirsave_plt+'/'+'hist_clstr_max_sim_'+source0+\
                                                    sim_n+'_'+difu_gauss+ sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', allow_pickle=True).item()
                                hist_clstr_max_sim_g1 = np.load(dirsave_plt+'/'+'hist_clstr_max_sim_'+source1+\
                                                    sim_n+'_'+difu_gauss+ sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', allow_pickle=True).item()

                                bins_clstr_max_sim_g0, counts_clstr_max_sim_g0 = hist_clstr_max_sim_g0['bins'], hist_clstr_max_sim_g0['counts']
                                bins_clstr_max_sim_g1, counts_clstr_max_sim_g1 = hist_clstr_max_sim_g1['bins'], hist_clstr_max_sim_g1['counts']
                                #################################################################################
                                # Acumular datos
                                sum_bins_clstr_max_sim_g0 += bins_clstr_max_sim_g0
                                sum_counts_clstr_max_sim_g0 += counts_clstr_max_sim_g0
                                sum_2_bins_clstr_max_sim_g0 += bins_clstr_max_sim_g0 ** 2
                                sum_2_counts_clstr_max_sim_g0 += counts_clstr_max_sim_g0 ** 2

                                sum_bins_clstr_max_sim_g1 += bins_clstr_max_sim_g1
                                sum_counts_clstr_max_sim_g1 += counts_clstr_max_sim_g1
                                sum_2_bins_clstr_max_sim_g1 += bins_clstr_max_sim_g1 ** 2
                                sum_2_counts_clstr_max_sim_g1 += counts_clstr_max_sim_g1 ** 2

                                #################################################################################

                            except Exception as e:
                                print(f"Error procesando simulación {sim_n}: {e}")
                                continue

                        # Calcular promedios y desviaciones estándar
                        mean_bins_adc_sim_g0 = sum_bins_adc_sim_g0 / total_sim
                        std_bins_adc_sim_g0 = np.sqrt((sum_2_bins_adc_sim_g0 - total_sim * mean_bins_adc_sim_g0 ** 2) / n_sampl )

                        mean_counts_adc_sim_g0 = sum_counts_adc_sim_g0 / total_sim
                        std_counts_adc_sim_g0 = np.sqrt((sum_2_counts_adc_sim_g0 - total_sim * mean_counts_adc_sim_g0 ** 2) / n_sampl )
                        #mean_counts_adc_sim_g0 = (np.rint(mean_counts_adc_sim_g0)).astype(int)
                        mean_counts_adc_sim_g0 = (np.rint(mean_counts_adc_sim_g0)).astype(int)

                        mean_bins_adc_sim_g1 = sum_bins_adc_sim_g1 / total_sim
                        std_bins_adc_sim_g1 = np.sqrt((sum_2_bins_adc_sim_g1 - total_sim * mean_bins_adc_sim_g1 ** 2) / n_sampl )

                        mean_counts_adc_sim_g1 = sum_counts_adc_sim_g1 / total_sim
                        std_counts_adc_sim_g1 = np.sqrt((sum_2_counts_adc_sim_g1 - total_sim * mean_counts_adc_sim_g1 ** 2) / n_sampl )

                        #mean_counts_adc_sim_g0 = (np.rint(mean_counts_adc_sim_g0)).astype(int)
                        mean_counts_adc_sim_g0 = (np.rint(mean_counts_adc_sim_g0)).astype(int)

                        # Manejar NaN
                        mean_bins_adc_sim_g0, std_bins_adc_sim_g0 = (np.zeros_like(mean_bins_adc_sim_g0) if np.any(np.isnan(mean_bins_adc_sim_g0)) else mean_bins_adc_sim_g0,
                                               np.zeros_like(std_bins_adc_sim_g0) if np.any(np.isnan(std_bins_adc_sim_g0)) else std_bins_adc_sim_g0)

                        mean_counts_adc_sim_g0, std_counts_adc_sim_g0 = (np.zeros_like(mean_counts_adc_sim_g0) if np.any(np.isnan(mean_counts_adc_sim_g0)) else mean_counts_adc_sim_g0,
                                                   np.zeros_like(std_counts_adc_sim_g0) if np.any(np.isnan(std_counts_adc_sim_g0)) else std_counts_adc_sim_g0)

                        mean_bins_adc_sim_g1, std_bins_adc_sim_g1 = (np.zeros_like(mean_bins_adc_sim_g1) if np.any(np.isnan(mean_bins_adc_sim_g1)) else mean_bins_adc_sim_g1,
                                               np.zeros_like(std_bins_adc_sim_g1) if np.any(np.isnan(std_bins_adc_sim_g1)) else std_bins_adc_sim_g1)

                        mean_counts_adc_sim_g1, std_counts_adc_sim_g1 = (np.zeros_like(mean_counts_adc_sim_g1) if np.any(np.isnan(mean_counts_adc_sim_g1)) else mean_counts_adc_sim_g1,
                                                   np.zeros_like(std_counts_adc_sim_g1) if np.any(np.isnan(std_counts_adc_sim_g1)) else std_counts_adc_sim_g1)

                        #################################################################################
                        # Calcular promedios y desviaciones estándar
                        mean_bins_clstr_siz_sim_g0 = sum_bins_clstr_siz_sim_g0 / total_sim
                        std_bins_clstr_siz_sim_g0 = np.sqrt((sum_2_bins_clstr_siz_sim_g0 - total_sim * mean_bins_clstr_siz_sim_g0 ** 2) / n_sampl )

                        mean_counts_clstr_siz_sim_g0 = sum_counts_clstr_siz_sim_g0 / total_sim
                        std_counts_clstr_siz_sim_g0 = np.sqrt((sum_2_counts_clstr_siz_sim_g0 - total_sim * mean_counts_clstr_siz_sim_g0 ** 2) / n_sampl )
                        #mean_counts_clstr_siz_sim_g0 = (np.rint(mean_counts_clstr_siz_sim_g0)).astype(int)
                        mean_counts_clstr_siz_sim_g0 = (np.rint(mean_counts_clstr_siz_sim_g0)).astype(int)

                        mean_bins_clstr_siz_sim_g1 = sum_bins_clstr_siz_sim_g1 / total_sim
                        std_bins_clstr_siz_sim_g1 = np.sqrt((sum_2_bins_clstr_siz_sim_g1 - total_sim * mean_bins_clstr_siz_sim_g1 ** 2) / n_sampl )

                        mean_counts_clstr_siz_sim_g1 = sum_counts_clstr_siz_sim_g1 / total_sim
                        std_counts_clstr_siz_sim_g1 = np.sqrt((sum_2_counts_clstr_siz_sim_g1 - total_sim * mean_counts_clstr_siz_sim_g1 ** 2) / n_sampl )
                        #mean_counts_clstr_siz_sim_g1 = (np.rint(mean_counts_clstr_siz_sim_g1)).astype(int)
                        mean_counts_clstr_siz_sim_g1 = (np.rint(mean_counts_clstr_siz_sim_g1)).astype(int)

                        # Manejar NaN
                        mean_bins_clstr_siz_sim_g0, std_bins_clstr_siz_sim_g0 = (np.zeros_like(mean_bins_clstr_siz_sim_g0) if np.any(np.isnan(mean_bins_clstr_siz_sim_g0)) else mean_bins_clstr_siz_sim_g0,
                                               np.zeros_like(std_bins_clstr_siz_sim_g0) if np.any(np.isnan(std_bins_clstr_siz_sim_g0)) else std_bins_clstr_siz_sim_g0)

                        mean_counts_clstr_siz_sim_g0, std_counts_clstr_siz_sim_g0 = (np.zeros_like(mean_counts_clstr_siz_sim_g0) if np.any(np.isnan(mean_counts_clstr_siz_sim_g0)) else mean_counts_clstr_siz_sim_g0,
                                                   np.zeros_like(std_counts_clstr_siz_sim_g0) if np.any(np.isnan(std_counts_clstr_siz_sim_g0)) else std_counts_clstr_siz_sim_g0)

                        mean_bins_clstr_siz_sim_g1, std_bins_clstr_siz_sim_g1 = (np.zeros_like(mean_bins_clstr_siz_sim_g1) if np.any(np.isnan(mean_bins_clstr_siz_sim_g1)) else mean_bins_clstr_siz_sim_g1,
                                               np.zeros_like(std_bins_clstr_siz_sim_g1) if np.any(np.isnan(std_bins_clstr_siz_sim_g1)) else std_bins_clstr_siz_sim_g1)

                        mean_counts_clstr_siz_sim_g1, std_counts_clstr_siz_sim_g1 = (np.zeros_like(mean_counts_clstr_siz_sim_g1) if np.any(np.isnan(mean_counts_clstr_siz_sim_g1)) else mean_counts_clstr_siz_sim_g1,
                                                   np.zeros_like(std_counts_clstr_siz_sim_g1) if np.any(np.isnan(std_counts_clstr_siz_sim_g1)) else std_counts_clstr_siz_sim_g1)

                        #################################################################################
                        # Calcular promedios y desviaciones estándar
                        mean_bins_clstr_max_sim_g0 = sum_bins_clstr_max_sim_g0 / total_sim
                        std_bins_clstr_max_sim_g0 = np.sqrt((sum_2_bins_clstr_max_sim_g0 - total_sim * mean_bins_clstr_max_sim_g0 ** 2) / n_sampl )

                        mean_counts_clstr_max_sim_g0 = sum_counts_clstr_max_sim_g0 / total_sim
                        std_counts_clstr_max_sim_g0 = np.sqrt((sum_2_counts_clstr_max_sim_g0 - total_sim * mean_counts_clstr_max_sim_g0 ** 2) / n_sampl )
                        #mean_counts_clstr_max_sim_g0 = (np.rint(mean_counts_clstr_max_sim_g0)).astype(int)
                        mean_counts_clstr_max_sim_g0 = (np.rint(mean_counts_clstr_max_sim_g0)).astype(int)

                        mean_bins_clstr_max_sim_g1 = sum_bins_clstr_max_sim_g1 / total_sim
                        std_bins_clstr_max_sim_g1 = np.sqrt((sum_2_bins_clstr_max_sim_g1 - total_sim * mean_bins_clstr_max_sim_g1 ** 2) / n_sampl )

                        mean_counts_clstr_max_sim_g1 = sum_counts_clstr_max_sim_g1 / total_sim
                        std_counts_clstr_max_sim_g1 = np.sqrt((sum_2_counts_clstr_max_sim_g1 - total_sim * mean_counts_clstr_max_sim_g1 ** 2) / n_sampl )
                        #mean_counts_clstr_max_sim_g1 = (np.rint(mean_counts_clstr_max_sim_g1)).astype(int)
                        mean_counts_clstr_max_sim_g1 = (np.rint(mean_counts_clstr_max_sim_g1)).astype(int)

                        # Manejar NaN
                        mean_bins_clstr_max_sim_g0, std_bins_clstr_max_sim_g0 = (np.zeros_like(mean_bins_clstr_max_sim_g0) if np.any(np.isnan(mean_bins_clstr_max_sim_g0)) else mean_bins_clstr_max_sim_g0,
                                               np.zeros_like(std_bins_clstr_max_sim_g0) if np.any(np.isnan(std_bins_clstr_max_sim_g0)) else std_bins_clstr_max_sim_g0)

                        mean_counts_clstr_max_sim_g0, std_counts_clstr_max_sim_g0 = (np.zeros_like(mean_counts_clstr_max_sim_g0) if np.any(np.isnan(mean_counts_clstr_max_sim_g0)) else mean_counts_clstr_max_sim_g0,
                                                   np.zeros_like(std_counts_clstr_max_sim_g0) if np.any(np.isnan(std_counts_clstr_max_sim_g0)) else std_counts_clstr_max_sim_g0)

                        mean_bins_clstr_max_sim_g1, std_bins_clstr_max_sim_g1 = (np.zeros_like(mean_bins_clstr_max_sim_g1) if np.any(np.isnan(mean_bins_clstr_max_sim_g1)) else mean_bins_clstr_max_sim_g1,
                                               np.zeros_like(std_bins_clstr_max_sim_g1) if np.any(np.isnan(std_bins_clstr_max_sim_g1)) else std_bins_clstr_max_sim_g1)

                        mean_counts_clstr_max_sim_g1, std_counts_clstr_max_sim_g1 = (np.zeros_like(mean_counts_clstr_max_sim_g1) if np.any(np.isnan(mean_counts_clstr_max_sim_g1)) else mean_counts_clstr_max_sim_g1,
                                                   np.zeros_like(std_counts_clstr_max_sim_g1) if np.any(np.isnan(std_counts_clstr_max_sim_g1)) else std_counts_clstr_max_sim_g1)

                        #################################################################################
                        #################################################################################

                        # Imprimir resultados
                        '''
                        print(f"Promedio de bins adc_sim_g0: {mean_bins_adc_sim_g0}\n, Desviación estándar adc_sim_g0: {std_bins_adc_sim_g0}")
                        print(f"Promedio de counts adc_sim_g0: {mean_counts_adc_sim_g0}\n, Desviación estándar adc_sim_g0: {std_counts_adc_sim_g0}")

                        print(f"Promedio de bins clstr_siz_sim_g0: {mean_bins_clstr_siz_sim_g0}\n, Desviación estándar clstr_siz_sim_g0: {std_bins_clstr_siz_sim_g0}")
                        print(f"Promedio de counts clstr_siz_sim_g0: {mean_counts_clstr_siz_sim_g0}\n, Desviación estándar clstr_siz_sim_g0: {std_counts_clstr_siz_sim_g0}")

                        print(f"Promedio de bins clstr_max_sim_g0: {mean_bins_clstr_max_sim_g0}\n, Desviación estándar clstr_max_sim_g0: {std_bins_clstr_max_sim_g0}")
                        print(f"Promedio de counts clstr_max_sim_g0: {mean_counts_clstr_max_sim_g0}\n, Desviación estándar clstr_max_sim_g0: {std_counts_clstr_max_sim_g0}")

                        '''

                        # ADC histos
                        #################################################################################
                        bincenters_adc_obs0 = 0.5*(bins_adc_obs0[1:]+bins_adc_obs0[:-1])
                        norm_adc_obs0 = np.sum(counts_adc_obs0 * np.diff(bins_adc_obs0))
                        #################################################################################
                        bincenters_adc_obs1 = 0.5*(bins_adc_obs1[1:]+bins_adc_obs1[:-1])
                        norm_adc_obs1 = np.sum(counts_adc_obs1 * np.diff(bins_adc_obs1))
                        #################################################################################
                        #################################################################################
                        bincenters_adc_sim_g0 = 0.5*(bins_adc_sim_g0[1:]+bins_adc_sim_g0[:-1])
                        norm_adc_sim_g0 = np.sum(mean_counts_adc_sim_g0 * np.diff(bins_adc_sim_g0))
                        #################################################################################
                        bincenters_adc_sim_g1 = 0.5*(bins_adc_sim_g1[1:]+bins_adc_sim_g1[:-1])
                        norm_adc_sim_g1 = np.sum(mean_counts_adc_sim_g1 * np.diff(bins_adc_sim_g1))

                        if(densi==False):
                            adc_obs0_w = counts_adc_obs0
                            err_adc_obs0_w = np.sqrt(counts_adc_obs0)
                            #################################################################################
                            adc_obs1_w = counts_adc_obs1
                            err_adc_obs1_w = np.sqrt(counts_adc_obs1)
                            #################################################################################
                            #################################################################################
                            adc_sim_g0_w = mean_counts_adc_sim_g0
                            if(errors_sqrt == True): err_adc_sim_g0_w = np.sqrt(0.04*mean_counts_adc_sim_g0*mean_counts_adc_sim_g0 + std_counts_adc_sim_g0*std_counts_adc_sim_g0 )
                            else: err_adc_sim_g0_w = (0.2*mean_counts_adc_sim_g0 + std_counts_adc_sim_g0 )
                            #################################################################################
                            adc_sim_g1_w = mean_counts_adc_sim_g1
                            if(errors_sqrt == True): err_adc_sim_g1_w = np.sqrt(0.04*mean_counts_adc_sim_g1*mean_counts_adc_sim_g1 + std_counts_adc_sim_g1*std_counts_adc_sim_g1 )
                            else: err_adc_sim_g1_w = (0.2*mean_counts_adc_sim_g1 + std_counts_adc_sim_g1 )
                            #################################################################################

                        if(densi==True):
                            adc_obs0_w = counts_adc_obs0/norm_adc_obs0
                            err_adc_obs0_w = np.sqrt(counts_adc_obs0)/norm_adc_obs0
                            #################################################################################
                            adc_obs1_w = counts_adc_obs1/norm_adc_obs1
                            err_adc_obs1_w = np.sqrt(counts_adc_obs1)/norm_adc_obs1
                            #################################################################################
                            #################################################################################
                            adc_sim_g0_w = mean_counts_adc_sim_g0/norm_adc_sim_g0
                            if(errors_sqrt == True): err_adc_sim_g0_w = np.sqrt(0.04*mean_counts_adc_sim_g0*mean_counts_adc_sim_g0 + std_counts_adc_sim_g0*std_counts_adc_sim_g0 )/norm_adc_sim_g0
                            else: err_adc_sim_g0_w = (0.2*mean_counts_adc_sim_g0 + std_counts_adc_sim_g0 )/norm_adc_sim_g0
                            #################################################################################
                            adc_sim_g1_w = mean_counts_adc_sim_g1/norm_adc_sim_g1
                            if(errors_sqrt == True): err_adc_sim_g1_w = np.sqrt(0.04*mean_counts_adc_sim_g1*mean_counts_adc_sim_g1 + std_counts_adc_sim_g1*std_counts_adc_sim_g1 )/norm_adc_sim_g1
                            else: err_adc_sim_g1_w = (0.2*mean_counts_adc_sim_g1 + std_counts_adc_sim_g1 )/norm_adc_sim_g1
                            #################################################################################

                        #err_adc_obs = np.sqrt(err_adc_obs0_w*err_adc_obs0_w +err_adc_obs1_w*err_adc_obs1_w + 2*err_adc_obs0_w*err_adc_obs_w)
                        err_adc_obs = np.sqrt(err_adc_obs0_w*err_adc_obs0_w + err_adc_obs1_w*err_adc_obs1_w )
                        #ch2_adc_obs = chi_2_sigm_test(adc_obs0_w[1:], adc_obs1_w[1:], err_adc_obs[1:])
                        ch2_adc_obs = chi_2_sigm_test(adc_obs0_w, adc_obs1_w, err_adc_obs)
                        tmp_str_obs = 'Chi2 test: ' + r'%.2f'%ch2_adc_obs[0] \
                                    +'       '+'ndf: ' + r'%.2f'%ch2_adc_obs[3] \
                                    +'\n'+'Chi2/ndf : ' + r'%.2f'%ch2_adc_obs[4]

                        err_adc_simg = np.sqrt(err_adc_sim_g0_w*err_adc_sim_g0_w + err_adc_sim_g1_w*err_adc_sim_g1_w )
                        ch2_adc_simg = chi_2_sigm_test(adc_sim_g0_w, adc_sim_g1_w, err_adc_simg)
                        tmp_str_simg = 'Chi2 test: ' + r'%.2f'%ch2_adc_simg[0] \
                                    +'       '+'ndf: ' + r'%.2f'%ch2_adc_simg[3] \
                                    +'\n'+'Chi2/ndf : ' + r'%.2f'%ch2_adc_simg[4]

                        if(labl_opt=='simple'):
                            lbl_obs0 = source0 + ' data '
                            lbl_obs1 = source1 + ' data '
                            lbl_sim_g0 = source0 + ' simulation '
                            lbl_sim_g1 = source1 + ' simulation '

                            lbl_bg = 'Background' + ' data'
                        if(labl_opt=='chi2_nu'):
                            lbl_obs0 = source0 + ' data '
                            lbl_obs1 = source1 + ' data ' + r'$\chi^2_\nu$ = ' + r'%.2f'%ch2_adc_obs[4]
                            lbl_sim_g0 = source0 + ' sim '
                            lbl_sim_g1 = source1 + ' sim ' + r'$\chi^2_\nu$ = ' + r'%.2f'%ch2_adc_simg[4]

                            lbl_bg = 'Background' + ' data '
                        if(labl_opt=='off'):
                            lbl_obs0 = None
                            lbl_obs0 = None
                            lbl_obs1 = None
                            lbl_sim_g0 = None
                            lbl_sim_g1 = None

                        plt.rcParams.update({'font.size': font_siz})

                        if( plot_nsig==True and plot_ratio==True ):
                            fig, (ax0, ax1, ax2) = plt.subplots(3,1,figsize=(15.5,14),gridspec_kw={'height_ratios': [5, 1, 1]} )#, sharex=True )

                        if( plot_nsig==True and plot_ratio==False ):
                            fig, (ax0, ax1) = plt.subplots(2,1,figsize=(15.5,12),gridspec_kw={'height_ratios': [5, 1]} )#, sharex=True )

                        if( plot_nsig==False and plot_ratio==True ):
                            fig, (ax0, ax2) = plt.subplots(2,1,figsize=(15.5,12),gridspec_kw={'height_ratios': [5, 1]} )#, sharex=True )

                        if( plot_nsig==False and plot_ratio==False ):
                            fig, (ax0) = plt.subplots(figsize=(15.5,10) )#, sharex=True )

                        if(plt_obs==True):
                            ax0.set_yscale('log')
                            #ax0.step( np.insert(bins_adc_obs0, 0, 0), np.concatenate(([0], adc_obs0_w, [0])), where='post'
                            #            , color='C3', linewidth=l_width+0.*l_width, label = lbl_obs0 + cut_max_clst_porc )
                            if(plt_err==True):
                                ax0.errorbar(bincenters_adc_obs0, adc_obs0_w, yerr=err_adc_obs0_w,
                                fmt='C3'+'o', ecolor='C3', markersize=10, linewidth=l_width*2/3,  label = lbl_obs0 + cut_max_clst_porc )
                                ax0.errorbar(bincenters_adc_obs1, adc_obs1_w, yerr=err_adc_obs1_w,
                                fmt='C1'+'D', ecolor='C1', markersize=10, linewidth=l_width*2/3, alpha=0.8, label = lbl_obs1 + cut_max_clst_porc )
                            ################################################################################################################################

                        if(plt_sim==True):
                            if(plt_sim_gauss==True):
                                #ax0.step( np.insert(bins_adc_obs0, 0, 0), np.concatenate(([0], adc_sim_g0_w, [0])), where='post' ,
                                        # color='C7', linewidth=l_width, label = lbl_sim_g0 + cut_max_clst_porc_sim)
                                ax0.fill_between(bincenters_adc_sim_g0, adc_sim_g0_w - err_adc_sim_g0_w, adc_sim_g0_w + err_adc_sim_g0_w,
                                            color='C0', linewidth=l_width*2/3, alpha=0.7, label = lbl_sim_g0 + cut_max_clst_porc_sim )
                                #ax0.step( np.insert(bins_adc_obs0, 0, 0), np.concatenate(([0], adc_sim_g1_w, [0])), where='post',
                                            #color='C7', linewidth=l_width, label = lbl_sim_g1 + cut_max_clst_porc_sim )
                                ax0.fill_between(bincenters_adc_sim_g1, adc_sim_g1_w - err_adc_sim_g1_w, adc_sim_g1_w + err_adc_sim_g1_w,
                                            color='C7', linewidth=l_width*2/3, alpha=0.5, label = lbl_sim_g1 + cut_max_clst_porc_sim )
                                ax0.set_yscale('log')

                        if(plt_sim==True and plt_err==True):
                            ax0.set_yscale('log')
                            if(plt_sim_gauss==True):
                                ax0.errorbar(bincenters_adc_sim_g0, adc_sim_g0_w, yerr=err_adc_sim_g0_w, fmt='C0'+'o',
                                            ecolor='C0', markersize=6, alpha=0.2, linewidth=l_width*2/3 )
                                ax0.errorbar(bincenters_adc_sim_g1, adc_sim_g1_w, yerr=err_adc_sim_g1_w, fmt='C7'+'D',
                                            ecolor='C7',  markersize=6, alpha=0.2, linewidth=l_width*2/3 )

                        ################################################################################################################################
                        ################################################################################################################################

                        U_eV = 1000
                        min_cam =((min_adc-1)*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
                        sat_cam =(1023*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
                        if(max_adc>=1023):ax0.axvline(x = 1023, color = 'k', linestyle="--", linewidth=l_width*2/3)#, label = 'sat_cam: 1023 ADC = ' + str(sat_cam)+' keV')

                        if(plt_obs==True or plt_sim==True):
                            if(log_y==True):ax0.set_yscale('log')
                            if(yscal_adc>0):ax0.set_ylim(0, yscal_adc)
                            ax0.grid(grid_)
                            #ax0.set_xticks(size=font_siz+2)
                            minor_ticks_x= np.arange(min_adc, max_adc, 200 )
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

                        ylabl_comp= r'$\frac{ (Sr90 - Cs137)}{\sigma}$'
                        comp_obs, err_comp_obs = comp_2obs(adc_obs0_w, adc_obs1_w)
                        comp_sim_g, err_comp_sim_g = comp_2sim(adc_sim_g0_w, adc_sim_g1_w)

                        delta_min = np.min(np.array(( np.nanmin(comp_obs),  np.nanmin(comp_sim_g) )))
                        delta_max = np.max(np.array(( comp_obs[comp_obs < np.Inf].max(), comp_sim_g[comp_sim_g < np.Inf].max() )))

                        if(plt_obs==True and plot_nsig==True):
                            #ax1.axhline(y = 0., xmin=0, xmax=max_adc, color = 'k', linestyle = '--', linewidth=l_width*2/3)
                            #ax1.step( np.insert(bins_adc_obs0, 0, 0), 0*np.concatenate(([0], adc_obs0_w, [0])), where='post', ls='--',  color='k', linewidth=l_width*2/3 )
                            ax1.errorbar(bincenters_adc_obs0, comp_obs, yerr=plt_err*err_comp_obs, fmt='C3'+'h', lw=2, markersize=10 )
                            if(zoom_delta==True):ax1.set_ylim(delta_min,delta_max)

                        if(plt_sim==True and plot_nsig==True):
                           # ax1.axhline(y = 0., xmin=0, xmax=max_adc, color = 'k', linestyle = '--', linewidth=l_width*2/3)
                            #ax1.step( np.insert(bins_adc_obs0, 0, 0), 0*np.concatenate(([0], adc_obs0_w, [0])), where='post', ls='--',  color='k', linewidth=l_width*2/3 )
                            if(plt_sim_gauss==True):
                                ax1.errorbar(bincenters_adc_obs0, comp_sim_g, yerr=plt_err*err_comp_sim_g, fmt='C0'+'h', lw=2, markersize=10 )
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

                        ylabl_ratio= r'$ratio = \frac{data}{sim}$'
                        ratio_obs, err_ratio_obs = ratio_2obs(adc_obs0_w, adc_obs1_w)
                        ratio_sim_g, err_ratio_sim_g = ratio_2sim(adc_sim_g0_w, adc_sim_g1_w)
                        ratio_min = np.min( np.array(( np.nanmin(ratio_obs),  np.nanmin(ratio_sim_g) )))
                        ratio_max = np.max( np.array(( ratio_obs[ratio_obs < np.Inf].max(), ratio_sim_g[ratio_sim_g < np.Inf].max() )))

                        if(plt_obs==True and plot_ratio==True):
                            #ax2.axhline(y = 0., xmin=min_adc, xmax=max_adc, color = 'k', linestyle = '--')
                            #ax2.hist(adc_obs0_w*0, bins = bins_adc_obs0, histtype='step', ls='--',  color='k', linewidth=l_width*2/3 )
                            #ax2.step( np.insert(bins_adc_obs0, 0, 0), 0*np.concatenate(([0], adc_obs0_w, [0])), where='post', ls='--',  color='k', linewidth=l_width*2/3 )
                            ax2.errorbar(bincenters_adc_obs0, ratio_obs, yerr=plt_err*err_ratio_obs, fmt='C3'+'h', lw=2, markersize=10 )
                            if(zoom_ratio==True):ax2.set_ylim(-0.5+ratio_min,ratio_max)

                        if(plt_sim==True and plot_ratio==True):
                            #ax2.axhline(y = 0., xmin=min_adc, xmax=max_adc, color = 'k', linestyle = '--')
                            #ax2.hist(adc_obs0_w*0, bins = bins_adc_obs0, histtype='step', ls='--',  color='k', linewidth=l_width*2/3 )
                            #ax2.step( np.insert(bins_adc_obs0, 0, 0), 0*np.concatenate(([0], adc_obs0_w, [0])), where='post', ls='--',  color='k', linewidth=l_width*2/3 )
                            if(plt_sim_gauss==True):
                                ax2.errorbar(bincenters_adc_obs0, ratio_sim_g, yerr=plt_err*err_ratio_sim_g, fmt='C0'+'h', lw=2, markersize=10 )
                            if(zoom_ratio==True):ax2.set_ylim(-0.5+ratio_min,ratio_max)

                        if( (plt_obs==True or plt_sim==True) and plot_ratio==True ):
                            ax[2].xticks(size=font_siz+2)
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

                        adc2ekv=(bins_adc_obs0*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
                        ax3.plot(adc2ekv, bins_adc_obs0*0., color='w', linewidth=0.001 )
                        ax3.set_xlabel('Energy (keV)', fontsize=font_siz+4)

                        #ax0.legend(loc='lower right')
                        #ax0.legend(loc='lower left')
                        #ax0.legend(loc='upper center')
                        #ax0.legend(loc='upper left', bbox_to_anchor=(0.62, 1))
                        #ax0.legend(loc='upper right')

                        # Crear una lista de handles y etiquetas en el orden deseado
                        handles, labels = ax0.get_legend_handles_labels()
                        if(plt_obs==True and plt_sim==True): new_order = [2,3,0, 1]  # Suponiendo que deseas invertir el orden
                        else : new_order = [0, 1]
                        handles = [handles[i] for i in new_order]
                        labels = [labels[i] for i in new_order]

                        # Crear la leyenda con el nuevo orden
                        ax0.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.62, 1))

                        fig.tight_layout()
                        name_cuts = sig_bg_thresh + str_cuts + cut_clst_size+cut_max + splt_obs+splt_sim+splt_simg
                        namesave = 'plot_hist_'+histo+'ADC_'+strbin+str(z*2).zfill(2)+'mm'+name_cuts
                        plt.savefig(dirsave_plt_mean+'/'+namesave+'.png', dpi=150)
                        plt.savefig(dirsave_plt_mean+'/'+namesave+'.pdf', dpi=150)
                        plt.show()
                        #plt.clf()
                        #plt.close()



                        # Cluster size histos
                        #################################################################################
                        bincenters_clstr_siz_obs0 = 0.5*(bins_clstr_siz_obs0[1:]+bins_clstr_siz_obs0[:-1])
                        norm_clstr_siz_obs0 = np.sum(counts_clstr_siz_obs0 * np.diff(bins_clstr_siz_obs0))
                        #################################################################################
                        bincenters_clstr_siz_obs1 = 0.5*(bins_clstr_siz_obs1[1:]+bins_clstr_siz_obs1[:-1])
                        norm_clstr_siz_obs1 = np.sum(counts_clstr_siz_obs1 * np.diff(bins_clstr_siz_obs1))
                        #################################################################################
                        #################################################################################
                        bincenters_clstr_siz_sim_g0 = 0.5*(bins_clstr_siz_sim_g0[1:]+bins_clstr_siz_sim_g0[:-1])
                        norm_clstr_siz_sim_g0 = np.sum(mean_counts_clstr_siz_sim_g0 * np.diff(bins_clstr_siz_sim_g0))
                        #################################################################################
                        bincenters_clstr_siz_sim_g1 = 0.5*(bins_clstr_siz_sim_g1[1:]+bins_clstr_siz_sim_g1[:-1])
                        norm_clstr_siz_sim_g1 = np.sum(mean_counts_clstr_siz_sim_g1 * np.diff(bins_clstr_siz_sim_g1))

                        if(densi==False):
                            clstr_siz_obs0_w = counts_clstr_siz_obs0

                            err_clstr_siz_obs0_w = np.sqrt(counts_clstr_siz_obs0)
                            #################################################################################
                            clstr_siz_obs1_w = counts_clstr_siz_obs1
                            err_clstr_siz_obs1_w = np.sqrt(counts_clstr_siz_obs1)
                            #################################################################################
                            #################################################################################
                            clstr_siz_sim_g0_w = mean_counts_clstr_siz_sim_g0
                            if(errors_sqrt == True): err_clstr_siz_sim_g0_w = np.sqrt(0.04*mean_counts_clstr_siz_sim_g0*mean_counts_clstr_siz_sim_g0 + std_counts_clstr_siz_sim_g0*std_counts_clstr_siz_sim_g0 )
                            err_clstr_siz_sim_g0_w = (0.2*mean_counts_clstr_siz_sim_g0 + std_counts_clstr_siz_sim_g0 )
                            #################################################################################
                            clstr_siz_sim_g1_w = mean_counts_clstr_siz_sim_g1
                            if(errors_sqrt == True): err_clstr_siz_sim_g1_w = np.sqrt(0.04*mean_counts_clstr_siz_sim_g1*mean_counts_clstr_siz_sim_g1 + std_counts_clstr_siz_sim_g1*std_mean_counts_clstr_siz_sim_g1 )
                            err_clstr_siz_sim_g1_w = (0.2*mean_counts_clstr_siz_sim_g1 + std_counts_clstr_siz_sim_g1 )
                            #################################################################################
                        if(densi==True):
                            clstr_siz_obs0_w = counts_clstr_siz_obs0/norm_clstr_siz_obs0
                            err_clstr_siz_obs0_w = np.sqrt(counts_clstr_siz_obs0)/norm_clstr_siz_obs0
                            #################################################################################
                            clstr_siz_obs1_w = counts_clstr_siz_obs1/norm_clstr_siz_obs1
                            err_clstr_siz_obs1_w = np.sqrt(counts_clstr_siz_obs1)/norm_clstr_siz_obs1
                            #################################################################################
                            #################################################################################
                            clstr_siz_sim_g0_w = mean_counts_clstr_siz_sim_g0/norm_clstr_siz_sim_g0
                            if(errors_sqrt == True): err_clstr_siz_sim_g0_w = np.sqrt(0.04*mean_counts_clstr_siz_sim_g0*mean_counts_clstr_siz_sim_g0 + std_counts_clstr_siz_sim_g0*std_counts_clstr_siz_sim_g0 )/norm_clstr_siz_sim_g0
                            else: err_clstr_siz_sim_g0_w = (0.2*mean_counts_clstr_siz_sim_g0 + std_counts_clstr_siz_sim_g0 )/norm_clstr_siz_sim_g0
                            #################################################################################
                            clstr_siz_sim_g1_w = mean_counts_clstr_siz_sim_g1/norm_clstr_siz_sim_g1
                            if(errors_sqrt == True): err_clstr_siz_sim_g1_w = np.sqrt(0.04*mean_counts_clstr_siz_sim_g1*mean_counts_clstr_siz_sim_g1 + std_counts_clstr_siz_sim_g1*std_counts_clstr_siz_sim_g1 )/norm_clstr_siz_sim_g1
                            else: err_clstr_siz_sim_g1_w = (0.2*mean_counts_clstr_siz_sim_g1 + std_counts_clstr_siz_sim_g1 )/norm_clstr_siz_sim_g1
                            #################################################################################

                        err_clstr_siz_obs = np.sqrt(err_clstr_siz_obs0_w*err_clstr_siz_obs0_w +err_clstr_siz_obs1_w*err_clstr_siz_obs1_w )
                        ch2_clstr_siz_obs = chi_2_sigm_test(clstr_siz_obs0_w, clstr_siz_obs1_w, err_clstr_siz_obs)
                        tmp_str_obs = 'Chi2 test: ' + r'%.2f'%ch2_clstr_siz_obs[0] \
                                    +'       '+'ndf: ' + r'%.2f'%ch2_clstr_siz_obs[3] \
                                    +'\n'+'Chi2/ndf : ' + r'%.2f'%ch2_clstr_siz_obs[4]

                        err_clstr_siz_simg = np.sqrt(err_clstr_siz_sim_g0_w*err_clstr_siz_sim_g0_w + err_clstr_siz_sim_g1_w*err_clstr_siz_sim_g1_w )
                        ch2_clstr_siz_simg = chi_2_sigm_test(clstr_siz_sim_g0_w, clstr_siz_sim_g1_w, err_clstr_siz_simg)
                        tmp_str_simg = 'Chi2 test: ' + r'%.2f'%ch2_clstr_siz_simg[0] \
                                    +'       '+'ndf: ' + r'%.2f'%ch2_clstr_siz_simg[3] \
                                    +'\n'+'Chi2/ndf : ' + r'%.2f'%ch2_clstr_siz_simg[4]

                        if(labl_opt=='simple'):
                            lbl_obs0 = source0 + ' data '
                            lbl_obs1 = source1 + ' data '
                            lbl_sim_g0 = source0 + ' simulation '
                            lbl_sim_g1 = source1 + ' simulation '

                            lbl_bg = 'Background' + ' data'
                        if(labl_opt=='chi2_nu'):
                            lbl_obs0 = source0 + ' data '
                            lbl_obs1 = source1 + ' data ' + r'$\chi^2_\nu$ = ' + r'%.2f'%ch2_clstr_siz_obs[4]
                            lbl_sim_g0 = source0 + ' sim '
                            lbl_sim_g1 = source1 + ' sim ' + r'$\chi^2_\nu$ = ' + r'%.2f'%ch2_clstr_siz_simg[4]

                            lbl_bg = 'Background' + ' data '
                        if(labl_opt=='off'):
                            lbl_obs0 = None
                            lbl_obs1 = None
                            lbl_sim_g0 = None
                            lbl_sim_g1 = None

                        plt.rcParams.update({'font.size': font_siz})

                        if( plot_nsig==True and plot_ratio==True ):
                            fig, (ax0, ax1, ax2) = plt.subplots(3,1,figsize=(15.5,14),gridspec_kw={'height_ratios': [5, 1, 1]} )#, sharex=True )

                        if( plot_nsig==True and plot_ratio==False ):
                            fig, (ax0, ax1) = plt.subplots(2,1,figsize=(15.5,12),gridspec_kw={'height_ratios': [5, 1]} )#, sharex=True )

                        if( plot_nsig==False and plot_ratio==True ):
                            fig, (ax0, ax2) = plt.subplots(2,1,figsize=(15.5,12),gridspec_kw={'height_ratios': [5, 1]} )#, sharex=True )

                        if( plot_nsig==False and plot_ratio==False ):
                            fig, (ax0) = plt.subplots(figsize=(15.5,10) )#, sharex=True )

                        if(plt_obs==True):
                            #ax0.step( np.insert(bins_clstr_siz_obs0, 0, 0), np.concatenate(([0], clstr_siz_obs0_w, [0])), where='post'
                            #            , color='C3', linewidth=l_width+0.*l_width,   label = lbl_obs0 + cut_max_clst_porc )
                            ax0.set_yscale('log')   # Escala logarítmica en el eje y

                            if(plt_err==True):
                                ax0.errorbar(bincenters_clstr_siz_obs0, clstr_siz_obs0_w, yerr=err_clstr_siz_obs0_w,
                                fmt='C3'+'o', ecolor='C3', markersize=10, linewidth=l_width*2/3,  label = lbl_obs0 + cut_max_clst_porc )
                                ax0.errorbar(bincenters_clstr_siz_obs1, clstr_siz_obs1_w, yerr=err_clstr_siz_obs1_w,
                                fmt='C1'+'D', ecolor='C1', markersize=10, linewidth=l_width*2/3, alpha=0.8, label = lbl_obs1 + cut_max_clst_porc )
                        ################################################################################################################################

                        if(plt_sim==True):
                            # Crear datos suavizados usando un spline
                            x_new = np.linspace(bincenters_clstr_siz_sim_g0.min(), bincenters_clstr_siz_sim_g0.max(), 500)
                            spline_upper = make_interp_spline(bincenters_clstr_siz_sim_g0, clstr_siz_sim_g0_w + err_clstr_siz_sim_g0_w, k=3)
                            spline_lower = make_interp_spline(bincenters_clstr_siz_sim_g0, clstr_siz_sim_g0_w - err_clstr_siz_sim_g0_w, k=3)

                            upper_smooth = spline_upper(x_new)
                            lower_smooth = spline_lower(x_new)
                            if(plt_sim_gauss==True):
                                #ax0.step( np.insert(bins_clstr_siz_obs0, 0, 0), np.concatenate(([0], clstr_siz_sim_g0_w, [0])), where='post'
                                #        , color='C7', linewidth=l_width, label = lbl_sim_g0 + cut_max_clst_porc_sim)
                                ax0.fill_between(bincenters_clstr_siz_sim_g0, clstr_siz_sim_g0_w - err_clstr_siz_sim_g0_w, clstr_siz_sim_g0_w + err_clstr_siz_sim_g0_w,
                                            color='C0', linewidth=l_width*2/3, alpha=0.7, label = lbl_sim_g0 + cut_max_clst_porc_sim )
                                ax0.fill_between(bincenters_clstr_siz_sim_g1, clstr_siz_sim_g1_w - err_clstr_siz_sim_g1_w, clstr_siz_sim_g1_w + err_clstr_siz_sim_g1_w,
                                            color='C7', linewidth=l_width*2/3, alpha=0.5, label = lbl_sim_g1 + cut_max_clst_porc_sim )
                                ax0.set_yscale('log')

                        if(plt_sim==True and plt_err==True):
                            ax0.set_yscale('log')
                            if(plt_sim_gauss==True):
                                ax0.errorbar(bincenters_clstr_siz_sim_g0, clstr_siz_sim_g0_w, yerr=err_clstr_siz_sim_g0_w, fmt='C0'+'o',
                                            ecolor='C0', markersize=6, alpha=0.2, linewidth=l_width*2/3 )
                                ax0.errorbar(bincenters_clstr_siz_sim_g1, clstr_siz_sim_g1_w, yerr=err_clstr_siz_sim_g1_w, fmt='C7'+'D',
                                            ecolor='C7', markersize=6, alpha=0.2, linewidth=l_width*2/3 )

                        ################################################################################################################################
                        ################################################################################################################################

                        if(plt_obs==True or plt_sim==True):
                            if(log_y==True):ax0.set_yscale('log')
                            if(yscal_cls>0):ax0.set_ylim(0, yscal_cls)
                            ax0.grid(grid_)
                            #ax0.set_xticks(size=font_siz+2)
                            #minor_ticks_x= np.arange(min_adc, max_adc, 200 )
                            ax0.tick_params(labelbottom=False, direction='inout', width=3, length=14 )
                            ax0.minorticks_on()
                            ax0.tick_params(axis='both', direction='inout', which='minor',length=8 ,width=2)
                            #ax0.yticks(size=font_siz+2)
                            if( plot_nsig==False and plot_ratio==False ):
                                ax0.set_xlabel('Cluster Size', fontsize=font_siz+4)
                                ax0.tick_params(labelbottom=True, direction='inout', width=3, length=14 )

                            ax0.set_ylabel('Clusters number', fontsize=font_siz+4)
                            ax0.legend(fontsize=font_siz+0)
                            #fig.suptitle(titulo , fontsize=font_siz)

                        ylabl_comp= r'$\frac{ (Sr90 - Cs137)}{\sigma}$'
                        comp_obs, err_comp_obs = comp_2obs(clstr_siz_obs0_w, clstr_siz_obs1_w)
                        #delta_min = np.min(np.array((np.nanmin(comp_obs), np.nanmin(comp_sim) )))
                        #delta_max = np.max(np.array((comp_obs[comp_obs < np.Inf].max(), comp_sim[comp_sim < np.Inf].max() )))
                        comp_sim_g, err_comp_sim_g = comp_2sim(clstr_siz_sim_g0_w, clstr_siz_sim_g1_w)

                        delta_min = np.min(np.array((np.nanmin(comp_obs), np.nanmin(comp_sim_g) )))
                        delta_max = np.max(np.array( (comp_obs[comp_obs < np.Inf].max(), comp_sim_g[comp_sim_g < np.Inf].max() )))

                        if(plt_obs==True and plot_nsig==True):
                            #ax1.axhline(y = 0., xmin=0, xmax=max_adc, color = 'k', linestyle = '--', linewidth=l_width*2/3)
                            #ax1.step( np.insert(bins_clstr_siz_obs0, 0, 0), 0*np.concatenate(([0], clstr_siz_obs0_w, [0])), where='post', ls='--',  color='k', linewidth=l_width*2/3 )
                            ax1.errorbar(bincenters_clstr_siz_obs0, comp_obs, yerr=plt_err*err_comp_obs, fmt='C3'+'h', lw=2, markersize=10 )
                            if(zoom_delta==True):ax1.set_ylim(delta_min,delta_max)

                        if(plt_sim==True and plot_nsig==True):
                           # ax1.axhline(y = 0., xmin=0, xmax=max_adc, color = 'k', linestyle = '--', linewidth=l_width*2/3)
                            #ax1.step( np.insert(bins_clstr_siz_obs0, 0, 0), 0*np.concatenate(([0], clstr_siz_obs0_w, [0])), where='post', ls='--',  color='k', linewidth=l_width*2/3 )
                            if(plt_sim_gauss==True):
                                ax1.errorbar(bincenters_clstr_siz_obs0, comp_sim_g, yerr=plt_err*err_comp_sim_g, fmt='C0'+'h', lw=2, markersize=10 )
                            if(zoom_delta==True):ax1.set_ylim(delta_min,delta_max)

                        if( (plt_obs==True or plt_sim==True) and plot_nsig==True ):
                            ax1.tick_params(labelbottom=False, direction='inout', width=3, length=14 )
                            ax1.minorticks_on()
                            ax1.set_ylabel(ylabl_comp, fontsize=font_siz+2)
                            ax1.tick_params(axis='x', direction='inout', which='minor',length=8 ,width=2)
                            if(plot_ratio==False):
                                ax1.set_xlabel('Cluster Size', fontsize=font_siz+4)
                                ax1.tick_params(labelbottom=True, width=3,  direction='inout', length=14 )

                            #ax[2].yaxis.tick_right()
                            #ax[2].yaxis.set_label_position("right")
                            if(grid_==True):
                                #ax1.grid(axis = 'x',  linestyle = '--',)
                                ax1.grid(grid_)

                        ylabl_ratio= r'$ratio = \frac{data}{sim}$'
                        ratio_obs, err_ratio_obs = ratio_2obs(clstr_siz_obs0_w, clstr_siz_obs1_w)
                        #ratio_min = np.min(np.array(( np.nanmin(ratio_obs), np.nanmin(ratio_sim) )))
                        #ratio_max = np.max(np.array(( ratio_obs[ratio_obs < np.Inf].max(), ratio_sim[ratio_sim < np.Inf].max() )))
                        ratio_sim_g, err_ratio_sim_g = ratio_2sim(clstr_siz_sim_g1_w, clstr_siz_sim_g0_w)
                        ratio_min = np.min(np.array(( np.nanmin(ratio_obs),  np.nanmin(ratio_sim_g) )))
                        ratio_max = np.max(np.array(( ratio_obs[ratio_obs < np.Inf].max(), ratio_obs[ratio_sim_g < np.Inf].max() )))

                        if(plt_obs==True and plot_ratio==True):
                            #ax2.axhline(y = 0., xmin=min_adc, xmax=max_adc, color = 'k', linestyle = '--')
                            #ax2.hist(clstr_siz_obs0_w*0, bins = bins_clstr_siz_obs0, histtype='step', ls='--',  color='k', linewidth=l_width*2/3 )
                            #ax2.step( np.insert(bins_clstr_siz_obs0, 0, 0), 0*np.concatenate(([0], clstr_siz_obs0_w, [0])), where='post', ls='--',  color='k', linewidth=l_width*2/3 )
                            ax2.errorbar(bincenters_clstr_siz_obs0, ratio_obs, yerr=plt_err*err_ratio_obs, fmt='C3'+'h', lw=2, markersize=10 )
                            if(zoom_ratio==True):ax2.set_ylim(-0.5+ratio_min,ratio_max)

                        if(plt_sim==True and plot_ratio==True):
                            #ax2.axhline(y = 0., xmin=min_adc, xmax=max_adc, color = 'k', linestyle = '--')
                            #ax2.hist(clstr_siz_obs0_w*0, bins = bins_clstr_siz_obs0, histtype='step', ls='--',  color='k', linewidth=l_width*2/3 )
                            #ax2.step( np.insert(bins_clstr_siz_obs0, 0, 0), 0*np.concatenate(([0], clstr_siz_obs0_w, [0])), where='post', ls='--',  color='k', linewidth=l_width*2/3 )
                            if(plt_sim_gauss==True):
                                ax2.errorbar(bincenters_clstr_siz_obs0, ratio_sim_g, yerr=plt_err*err_ratio_sim_g, fmt='C0'+'h', lw=2, markersize=10 )
                            if(zoom_ratio==True):ax2.set_ylim(-0.5+ratio_min,ratio_max)

                        if( (plt_obs==True or plt_sim==True) and plot_ratio==True ):
                            ax[2].xticks(size=font_siz+2)
                            #ax[2].yticks(size=font_siz+2)
                            #minor_ticks_x= np.arange(min_adc, max_adc,100 )
                            ax2.tick_params(labelbottom=True, width=3,  direction='inout', length=14 )
                            ax2.minorticks_on()
                            ax2.tick_params(axis='x', direction='inout', which='minor',length=8 ,width=2)
                            ax2.set_xlabel('Cluster Size', fontsize=font_siz+4)
                            ax2.set_ylabel(ylabl_ratio, fontsize=font_siz+2)

                            #ax[2].yaxis.tick_right()
                            #ax[2].yaxis.set_label_position("right")
                            if(grid_==True):
                                #ax2.grid(axis = 'x',  linestyle = '--',)
                                ax2.grid(grid_)

                        #ax0.legend(loc='lower right')
                        #ax0.legend(loc='lower left')
                        #ax0.legend(loc='upper center')
                        #ax0.legend(loc='upper left', bbox_to_anchor=(0.62, 1))

                        # Crear una lista de handles y etiquetas en el orden deseado
                        handles, labels = ax0.get_legend_handles_labels()
                        if(plt_obs==True and plt_sim==True): new_order = [2,3,0, 1]  # Suponiendo que deseas invertir el orden
                        else : new_order = [0, 1]
                        handles = [handles[i] for i in new_order]
                        labels = [labels[i] for i in new_order]

                        # Crear la leyenda con el nuevo orden
                        ax0.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.62, 1))

                        fig.tight_layout()
                        name_cuts = sig_bg_thresh + str_cuts + cut_clst_size+cut_max + splt_obs+splt_sim+splt_simg
                        namesave = 'plot_hist_'+histo+'cltr_siz_'+strbin+str(z*2).zfill(2)+'mm'+ name_cuts
                        plt.savefig(dirsave_plt_mean+'/'+namesave+'.png', dpi=150)
                        plt.savefig(dirsave_plt_mean+'/'+namesave+'.pdf', dpi=150)
                        plt.show()
                        #plt.clf()
                        #plt.close()



                        # Cluster Max histos
                        #################################################################################
                        bincenters_clstr_max_obs0 = 0.5*(bins_clstr_max_obs0[1:]+bins_clstr_max_obs0[:-1])
                        norm_clstr_max_obs0 = np.sum(counts_clstr_max_obs0 * np.diff(bins_clstr_max_obs0))
                        #################################################################################
                        bincenters_clstr_max_obs1 = 0.5*(bins_clstr_max_obs1[1:]+bins_clstr_max_obs1[:-1])
                        norm_clstr_max_obs1 = np.sum(counts_clstr_max_obs1 * np.diff(bins_clstr_max_obs1))
                        #################################################################################
                        #################################################################################
                        bincenters_clstr_max_sim_g0 = 0.5*(bins_clstr_max_sim_g0[1:]+bins_clstr_max_sim_g0[:-1])
                        norm_clstr_max_sim_g0 = np.sum(mean_counts_clstr_max_sim_g0 * np.diff(bins_clstr_max_sim_g0))
                        #################################################################################
                        bincenters_clstr_max_sim_g1 = 0.5*(bins_clstr_max_sim_g1[1:]+bins_clstr_max_sim_g1[:-1])
                        norm_clstr_max_sim_g1 = np.sum(mean_counts_clstr_max_sim_g1 * np.diff(bins_clstr_max_sim_g1))

                        if(densi==False):
                            clstr_max_obs0_w = counts_clstr_max_obs0
                            err_clstr_max_obs0_w = np.sqrt(counts_clstr_max_obs0)
                            #################################################################################
                            clstr_max_obs1_w = counts_clstr_max_obs1
                            err_clstr_max_obs1_w = np.sqrt(counts_adc_obs0)
                            #################################################################################
                            #################################################################################
                            clstr_max_sim_g0_w = mean_counts_clstr_max_sim_g0
                            if(errors_sqrt == True): err_clstr_max_sim_g0_w = np.sqrt(0.04*mean_counts_clstr_max_sim_g0*mean_counts_clstr_max_sim_g0 + std_counts_clstr_max_sim_g0*std_counts_clstr_max_sim_g0 )
                            else: err_clstr_max_sim_g0_w = (0.2*mean_counts_clstr_max_sim_g0 + std_counts_clstr_max_sim_g0 )
                            #################################################################################
                            clstr_max_sim_g1_w = mean_counts_clstr_max_sim_g1
                            if(errors_sqrt == True): err_clstr_max_sim_g1_w = np.sqrt(0.04*mean_counts_clstr_max_sim_g1*mean_counts_clstr_max_sim_g1 + std_counts_clstr_max_sim_g1*std_counts_clstr_max_sim_g1 )
                            else: err_clstr_max_sim_g1_w = (0.2*mean_counts_clstr_max_sim_g1 + std_counts_clstr_max_sim_g1 )
                            #################################################################################
                        if(densi==True):
                            clstr_max_obs0_w = counts_clstr_max_obs0/norm_clstr_max_obs0
                            err_clstr_max_obs0_w = np.sqrt(counts_clstr_max_obs0)/norm_clstr_max_obs0
                            #################################################################################
                            clstr_max_obs1_w = counts_clstr_max_obs1/norm_clstr_max_obs1
                            err_clstr_max_obs1_w = np.sqrt(counts_clstr_max_obs1)/norm_clstr_max_obs1
                            #################################################################################
                            #################################################################################
                            clstr_max_sim_g0_w = mean_counts_clstr_max_sim_g0/norm_clstr_max_sim_g0
                            if(errors_sqrt == True): err_clstr_max_sim_g0_w = np.sqrt(0.04*mean_counts_clstr_max_sim_g0*mean_counts_clstr_max_sim_g0 + std_counts_clstr_max_sim_g0*std_counts_clstr_max_sim_g0 )/norm_clstr_max_sim_g0
                            else: err_clstr_max_sim_g0_w = (0.2*mean_counts_clstr_max_sim_g0 + std_counts_clstr_max_sim_g0 )/norm_clstr_max_sim_g0
                            #################################################################################
                            clstr_max_sim_g1_w = mean_counts_clstr_max_sim_g1/norm_clstr_max_sim_g1
                            if(errors_sqrt == True): err_clstr_max_sim_g1_w = np.sqrt(0.04*mean_counts_clstr_max_sim_g1*mean_counts_clstr_max_sim_g1 + std_counts_clstr_max_sim_g1*std_counts_clstr_max_sim_g1 )/norm_clstr_max_sim_g1
                            else: err_clstr_max_sim_g1_w = (0.2*mean_counts_clstr_max_sim_g1 + std_counts_clstr_max_sim_g1 )/norm_clstr_max_sim_g1
                            #################################################################################

                        err_clstr_max_obs = np.sqrt(err_clstr_max_obs0_w*err_clstr_max_obs0_w +err_clstr_max_obs1_w*err_clstr_max_obs1_w )
                        ch2_clstr_max_obs = chi_2_sigm_test(clstr_max_obs0_w, clstr_max_obs1_w, err_clstr_max_obs)
                        tmp_str_obs = 'Chi2 test: ' + r'%.2f'%ch2_clstr_max_obs[0] \
                                    +'       '+'ndf: ' + r'%.2f'%ch2_clstr_max_obs[3] \
                                    +'\n'+'Chi2/ndf : ' + r'%.2f'%ch2_clstr_max_obs[4]

                        err_clstr_max_simg = np.sqrt(err_clstr_max_sim_g0_w*err_clstr_max_sim_g0_w + err_clstr_max_sim_g1_w*err_clstr_max_sim_g1_w )
                        ch2_clstr_max_simg = chi_2_sigm_test(clstr_max_sim_g0_w, clstr_max_sim_g1_w, err_clstr_max_simg)
                        tmp_str_simg = 'Chi2 test: ' + r'%.2f'%ch2_clstr_max_simg[0] \
                                    +'       '+'ndf: ' + r'%.2f'%ch2_clstr_max_simg[3] \
                                    +'\n'+'Chi2/ndf : ' + r'%.2f'%ch2_clstr_max_simg[4]

                        if(labl_opt=='simple'):
                            lbl_obs0 = source0 + ' data '
                            lbl_obs1 = source1 + ' data '
                            lbl_sim_g0 = source0 + ' simulation '
                            lbl_sim_g1 = source1 + ' simulation '

                        if(labl_opt=='chi2_nu'):
                            lbl_obs0 = source0 + ' data '
                            lbl_obs1 = source1 + ' data ' + r'$\chi^2_\nu$ = ' + r'%.2f'%ch2_clstr_max_obs[4]
                            lbl_sim_g0 = source0 + ' sim '
                            lbl_sim_g1 = source1 + ' sim ' + r'$\chi^2_\nu$ = ' + r'%.2f'%ch2_clstr_max_simg[4]

                        if(labl_opt=='off'):
                            lbl_obs0 = None
                            lbl_obs1 = None
                            lbl_sim_g0 = None
                            lbl_sim_g1 = None

                        plt.rcParams.update({'font.size': font_siz})

                        if( plot_nsig==True and plot_ratio==True ):
                            fig, (ax0, ax1, ax2) = plt.subplots(3,1,figsize=(15.5,14),gridspec_kw={'height_ratios': [5, 1, 1]} )#, sharex=True )

                        if( plot_nsig==True and plot_ratio==False ):
                            fig, (ax0, ax1) = plt.subplots(2,1,figsize=(15.5,12),gridspec_kw={'height_ratios': [5, 1]} )#, sharex=True )

                        if( plot_nsig==False and plot_ratio==True ):
                            fig, (ax0, ax2) = plt.subplots(2,1,figsize=(15.5,12),gridspec_kw={'height_ratios': [5, 1]} )#, sharex=True )

                        if( plot_nsig==False and plot_ratio==False ):
                            fig, (ax0) = plt.subplots(figsize=(15.5,10) )#, sharex=True )

                        if(plt_obs==True):
                            #ax0.step( np.insert(bins_clstr_max_obs0, 0, 0), np.concatenate(([0], clstr_max_obs0_w, [0])), where='post'
                                        #, color='C3', linewidth=l_width+0.*l_width, label = lbl_obs0 + cut_max_clst_porc    )
                            ax0.set_yscale('log')   # Escala logarítmica en el eje y

                            if(plt_err==True):
                                ax0.errorbar(bincenters_clstr_max_obs0, clstr_max_obs0_w, yerr=err_clstr_max_obs0_w,
                                fmt='C3'+'o', ecolor='C3', markersize=10, linewidth=l_width, label = lbl_obs0 + cut_max_clst_porc )
                                ax0.errorbar(bincenters_clstr_max_obs1, clstr_max_obs1_w, yerr=err_clstr_max_obs1_w,
                                fmt='C1'+'D', ecolor='C1', markersize=10, linewidth=l_width, alpha=0.8, label = lbl_obs1 + cut_max_clst_porc )
                        ################################################################################################################################

                        if(plt_sim==True):
                            if(plt_sim_gauss==True):
                                #ax0.step( np.insert(bins_clstr_max_obs0, 0, 0), np.concatenate(([0], clstr_max_sim_g0_w, [0])), where='post'
                                #        , color='C7', linewidth=l_width,  label = lbl_sim_g0 + cut_max_clst_porc_sim)
                                ax0.fill_between(bincenters_clstr_max_sim_g0, clstr_max_sim_g0_w - err_clstr_max_sim_g0_w, clstr_max_sim_g0_w + err_clstr_max_sim_g0_w,
                                            color='C0', linewidth=l_width*2/3, alpha=0.7, label = lbl_sim_g0 + cut_max_clst_porc_sim )
                                ax0.fill_between(bincenters_clstr_max_sim_g1, clstr_max_sim_g1_w - err_clstr_max_sim_g1_w, clstr_max_sim_g1_w + err_clstr_max_sim_g1_w,
                                            color='C7', linewidth=l_width*2/3, alpha=0.5, label = lbl_sim_g1 + cut_max_clst_porc_sim )
                                ax0.set_yscale('log')

                        if(plt_sim==True and plt_err==True):
                            ax0.set_yscale('log')
                            if(plt_sim_gauss==True):
                                ax0.errorbar(bincenters_clstr_max_sim_g0, clstr_max_sim_g0_w, yerr=err_clstr_max_sim_g0_w, fmt='C0'+'o',
                                            ecolor='C0', markersize=6, alpha=0.2, linewidth=l_width*2/3 )
                                ax0.errorbar(bincenters_clstr_max_sim_g1, clstr_max_sim_g1_w, yerr=err_clstr_max_sim_g1_w, fmt='C7'+'D',
                                            ecolor='C7', markersize=6, alpha=0.2, linewidth=l_width*2/3 )

                        ################################################################################################################################
                        ################################################################################################################################

                        if(plt_obs==True or plt_sim==True):
                            if(log_y==True):ax0.set_yscale('log')
                            if(yscal_adc>0):ax0.set_ylim(0, yscal_max)
                            ax0.grid(grid_)
                            #ax0.set_xticks(size=font_siz+2)
                            #minor_ticks_x= np.arange(min_adc, max_adc, 200 )
                            ax0.tick_params(labelbottom=False, direction='inout', width=3, length=14 )
                            ax0.minorticks_on()
                            ax0.tick_params(axis='both', direction='inout', which='minor',length=8 ,width=2)
                            #ax0.yticks(size=font_siz+2)
                            if( plot_nsig==False and plot_ratio==False ):
                                ax0.set_xlabel('Cluster Maximum ADC', fontsize=font_siz+4)
                                ax0.tick_params(labelbottom=True, direction='inout', width=3, length=14 )

                            ax0.set_ylabel('Clusters number', fontsize=font_siz+4)
                            ax0.legend(fontsize=font_siz+0)
                            #fig.suptitle(titulo , fontsize=font_siz)

                        ylabl_comp= r'$\frac{ (Sr90 - Cs137)}{\sigma}$'
                        comp_obs, err_comp_obs = comp_2obs(clstr_max_obs0_w, clstr_max_obs1_w)
                        #delta_min = np.min(np.array((np.nanmin(comp_obs), np.nanmin(comp_sim) )))
                        #delta_max = np.max(np.array((comp_obs[comp_obs < np.Inf].max(), comp_sim[comp_sim < np.Inf].max() )))
                        comp_sim_g, err_comp_sim_g = comp_2sim(clstr_max_sim_g0_w, clstr_max_sim_g1_w)

                        delta_min = np.min(np.array((np.nanmin(comp_obs),  np.nanmin(comp_sim_g) )))
                        delta_max = np.max(np.array( (comp_obs[comp_obs < np.Inf].max(), comp_sim_g[comp_sim_g < np.Inf].max() )))

                        if(plt_obs==True and plot_nsig==True):
                            #ax1.axhline(y = 0., xmin=0, xmax=max_adc, color = 'k', linestyle = '--', linewidth=l_width*2/3)
                            #ax1.step( np.insert(bins_clstr_max_obs0, 0, 0), 0*np.concatenate(([0], clstr_max_obs0_w, [0])), where='post', ls='--',  color='k', linewidth=l_width*2/3 )
                            ax1.errorbar(bincenters_clstr_max_obs0, comp_obs, yerr=plt_err*err_comp_obs, fmt='C3'+'h', lw=2, markersize=10 )
                            if(zoom_delta==True):ax1.set_ylim(delta_min,delta_max)

                        if(plt_sim==True and plot_nsig==True):
                           # ax1.axhline(y = 0., xmin=0, xmax=max_adc, color = 'k', linestyle = '--', linewidth=l_width*2/3)
                            #ax1.step( np.insert(bins_clstr_max_obs0, 0, 0), 0*np.concatenate(([0], clstr_max_obs0_w, [0])), where='post', ls='--',  color='k', linewidth=l_width*2/3 )
                            if(plt_sim_gauss==True):
                                ax1.errorbar(bincenters_clstr_max_obs0, comp_sim_g, yerr=plt_err*err_comp_sim_g, fmt='C0'+'h', lw=2, markersize=10 )
                            if(zoom_delta==True):ax1.set_ylim(delta_min,delta_max)

                        if( (plt_obs==True or plt_sim==True) and plot_nsig==True ):
                            ax1.tick_params(labelbottom=False, direction='inout', width=3, length=14 )
                            ax1.minorticks_on()
                            ax1.set_ylabel(ylabl_comp, fontsize=font_siz+2)
                            ax1.tick_params(axis='x', direction='inout', which='minor',length=8 ,width=2)
                            if(plot_ratio==False):
                                ax1.set_xlabel('Cluster Maximum ADC', fontsize=font_siz+4)
                                ax1.tick_params(labelbottom=True, width=3,  direction='inout', length=14 )

                            #ax[2].yaxis.tick_right()
                            #ax[2].yaxis.set_label_position("right")
                            if(grid_==True):
                                #ax1.grid(axis = 'x',  linestyle = '--',)
                                ax1.grid(grid_)

                        ylabl_ratio= r'$ratio = \frac{data}{sim}$'
                        ratio_obs, err_ratio_obs = ratio_2obs(clstr_max_obs0_w, clstr_max_obs1_w)
                        #ratio_min = np.min(np.array(( np.nanmin(ratio_obs), np.nanmin(ratio_sim) )))
                        #ratio_max = np.max(np.array(( ratio_obs[ratio_obs < np.Inf].max(), ratio_sim[ratio_sim < np.Inf].max() )))
                        ratio_sim_g, err_ratio_sim_g = ratio_2sim(clstr_max_sim_g0_w, clstr_max_sim_g1_w)
                        ratio_min = np.min(np.array(( np.nanmin(ratio_obs), np.nanmin(ratio_sim_g) )))
                        ratio_max = np.max(np.array(( ratio_obs[ratio_obs < np.Inf].max(), ratio_obs[ratio_sim_g < np.Inf].max() )))

                        if(plt_obs==True and plot_ratio==True):
                            #ax2.axhline(y = 0., xmin=min_adc, xmax=max_adc, color = 'k', linestyle = '--')
                            #ax2.hist(clstr_max_obs0_w*0, bins = bins_clstr_max_obs0, histtype='step', ls='--',  color='k', linewidth=l_width*2/3 )
                            #ax2.step( np.insert(bins_clstr_max_obs0, 0, 0), 0*np.concatenate(([0], clstr_max_obs0_w, [0])), where='post', ls='--',  color='k', linewidth=l_width*2/3 )
                            ax2.errorbar(bincenters_clstr_max_obs0, ratio_obs, yerr=plt_err*err_ratio_obs, fmt='C3'+'h', lw=2, markersize=10 )
                            if(zoom_ratio==True):ax2.set_ylim(-0.5+ratio_min,ratio_max)

                        if(plt_sim==True and plot_ratio==True):
                            #ax2.axhline(y = 0., xmin=min_adc, xmax=max_adc, color = 'k', linestyle = '--')
                            #ax2.hist(clstr_max_obs0_w*0, bins = bins_clstr_max_obs0, histtype='step', ls='--',  color='k', linewidth=l_width*2/3 )
                            #ax2.step( np.insert(bins_clstr_max_obs0, 0, 0), 0*np.concatenate(([0], clstr_max_obs0_w, [0])), where='post', ls='--',  color='k', linewidth=l_width*2/3 )
                            if(plt_sim_gauss==True):
                                ax2.errorbar(bincenters_clstr_max_obs0, ratio_sim_g, yerr=plt_err*err_ratio_sim_g, fmt='C0'+'h', lw=2, markersize=10 )
                            if(zoom_ratio==True):ax2.set_ylim(-0.5+ratio_min,ratio_max)

                        if( (plt_obs==True or plt_sim==True) and plot_ratio==True ):
                            ax[2].xticks(size=font_siz+2)
                            #ax[2].yticks(size=font_siz+2)
                            #minor_ticks_x= np.arange(min_adc, max_adc,100 )
                            ax2.tick_params(labelbottom=True, width=3,  direction='inout', length=14 )
                            ax2.minorticks_on()
                            ax2.tick_params(axis='x', direction='inout', which='minor',length=8 ,width=2)
                            ax2.set_xlabel('Cluster Maximum ADC', fontsize=font_siz+4)
                            ax2.set_ylabel(ylabl_ratio, fontsize=font_siz+2)

                            #ax[2].yaxis.tick_right()
                            #ax[2].yaxis.set_label_position("right")
                            if(grid_==True):
                                #ax2.grid(axis = 'x',  linestyle = '--',)
                                ax2.grid(grid_)

                        #ax0.legend(loc='lower right')
                        #ax0.legend(loc='lower left')
                        #ax0.legend(loc='upper center')
                        #ax0.legend(loc='upper left', bbox_to_anchor=(0.62, 1))

                        # Crear una lista de handles y etiquetas en el orden deseado
                        handles, labels = ax0.get_legend_handles_labels()
                        if(plt_obs==True and plt_sim==True): new_order = [2,3,0, 1]  # Suponiendo que deseas invertir el orden
                        else : new_order = [0, 1]
                        handles = [handles[i] for i in new_order]
                        labels = [labels[i] for i in new_order]

                        # Crear la leyenda con el nuevo orden
                        ax0.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.62, 1))

                        fig.tight_layout()
                        name_cuts = sig_bg_thresh + str_cuts + cut_clst_size+cut_max + splt_obs+splt_sim+splt_simg
                        namesave = 'plot_hist_'+histo+'cl_max_adc_'+strbin+str(z*2).zfill(2)+'mm'+ name_cuts
                        plt.savefig(dirsave_plt_mean+'/'+namesave+'.png', dpi=150)
                        plt.savefig(dirsave_plt_mean+'/'+namesave+'.pdf', dpi=150)
                        plt.show()
                        #plt.clf()
                        #plt.close()


                        if( file_csv==True):
                            file_activ_pixel = dirsave_plt_mean+'/'+'comp_'+source0+'_'+source1+dens+'_'+strbin+str(z_count)+'l_'+str(z*2).zfill(2)+'mm'+'actv_pixel_chi2'+str_cuts+'.csv'
                            file_test = open(file_activ_pixel,'w')

                            txt_activ = 'data'+'\t' + '0_ADC\t' + 'Active_Pixel\t' + 'all_pixel\t' + '%_0_ADC\t' + '%_Active_Pixel'
                            file_test.write(txt_activ + '\n')
                            print(txt_activ)
                            txt_activ = 'OBS0_'+source0+'_'+str(nsigm_bg)+'sig'+':\t' + str(all_pixel_nframe - counts_adc_obs0.sum()) + \
                                    '\t' + str(counts_adc_obs0.sum()) +'\t' + str(all_pixel_nframe) + \
                                    '\t'+ str(100*(all_pixel_nframe - counts_adc_obs0.sum())/all_pixel_nframe) + \
                                    '\t'+ str(100*counts_adc_obs0.sum()/all_pixel_nframe)
                            file_test.write(txt_activ + '\n')
                            print(txt_activ)
                            txt_activ = 'SIM0_D_gauss_'+source0+'_'+str(nsigm_bg)+'sig'+':\t' + str(all_pixel_nframe - mean_counts_adc_sim_g0.sum()) + \
                                    '\t' + str(mean_counts_adc_sim_g0.sum()) +'\t' + str(all_pixel_nframe) + \
                                    '\t'+ str(100*(all_pixel_nframe - mean_counts_adc_sim_g0.sum())/all_pixel_nframe) + \
                                    '\t'+ str(100*mean_counts_adc_sim_g0.sum()/all_pixel_nframe)
                            file_test.write(txt_activ + '\n')
                            print(txt_activ)

                            txt_activ = 'OBS1_'+source1+'_'+str(nsigm_bg)+'sig'+':\t' + str(all_pixel_nframe - counts_adc_obs1.sum()) + \
                                    '\t' + str(counts_adc_obs1.sum()) +'\t' + str(all_pixel_nframe) + \
                                    '\t'+ str(100*(all_pixel_nframe - counts_adc_obs1.sum())/all_pixel_nframe) + \
                                    '\t'+ str(100*counts_adc_obs1.sum()/all_pixel_nframe)
                            file_test.write(txt_activ + '\n')
                            print(txt_activ)
                            txt_activ = 'SIM1_D_gauss_'+source1+str(nsigm_bg)+'sig'+':\t' + str(all_pixel_nframe - mean_counts_adc_sim_g1.sum()) + \
                                    '\t' + str(mean_counts_adc_sim_g1.sum()) +'\t' + str(all_pixel_nframe) + \
                                    '\t'+ str(100*(all_pixel_nframe - mean_counts_adc_sim_g1.sum())/all_pixel_nframe) + \
                                    '\t'+ str(100*mean_counts_adc_sim_g1.sum()/all_pixel_nframe)
                            file_test.write(txt_activ + '\n')
                            print(txt_activ)
                            #file_test.close()

                            print('\n\n')


                            txt_test = 'test bins'+sig_bg_thresh + str_cuts + cut_clst_size+cut_max + cut_adc + cut_max_clst_porc+sig_bg_thresh_sim+cut_max_clst_porc_sim+dens+'\t' + 'histo\t' + 'bins\t' + 'chi_2\t' + 'pvalue\t' + 'criti_val\t' + 'ndf\t' + 'chi2/nf'
                            file_test.write('\n\n'+ txt_test + '\n')
                            print(txt_test)

                            txt_test = 'Chi2_sig_test_obs' +'\t'+'hist_ADC'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%ch2_adc_obs[0]) +'\t'+ str(r'%.5f'%ch2_adc_obs[1])  +'\t'+ str(r'%.5f'%ch2_adc_obs[2])  +'\t'+ str(ch2_adc_obs[3]) +'\t'+ str(r'%.5f'%ch2_adc_obs[4])
                            file_test.write(txt_test + '\n')
                            print(txt_test)

                            txt_test = 'Chi2_sig_test_obs' +'\t'+'hist_Cluster_size'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%ch2_clstr_siz_obs[0]) +'\t'+ str(r'%.5f'%ch2_clstr_siz_obs[1])  +'\t'+ str(r'%.5f'%ch2_clstr_siz_obs[2])  +'\t'+ str(ch2_clstr_siz_obs[3]) +'\t'+ str(r'%.5f'%ch2_clstr_siz_obs[4])
                            file_test.write(txt_test + '\n')
                            print(txt_test)

                            txt_test = 'Chi2_sig_test_obs' +'\t'+'hist_Max_clst'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%ch2_clstr_max_obs[0]) +'\t'+ str(r'%.5f'%ch2_clstr_max_obs[1])  +'\t'+ str(r'%.5f'%ch2_clstr_max_obs[2])  +'\t'+ str(ch2_clstr_max_obs[3]) +'\t'+ str(r'%.5f'%ch2_clstr_max_obs[4])
                            file_test.write(txt_test + '\n')
                            print(txt_test)



                            txt_test = 'Chi2_sig_test_sim_Dgauss' +'\t'+'hist_ADC'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%ch2_adc_simg[0]) +'\t'+ str(r'%.5f'%ch2_adc_simg[1])  +'\t'+ str(r'%.5f'%ch2_adc_simg[2])  +'\t'+ str(ch2_adc_simg[3]) +'\t'+ str(r'%.5f'%ch2_adc_simg[4])
                            file_test.write(txt_test + '\n')
                            print(txt_test)

                            txt_test = 'Chi2_sig_test_sim_Dgauss' +'\t'+'hist_Cluster_size'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%ch2_clstr_siz_simg[0]) +'\t'+ str(r'%.5f'%ch2_clstr_siz_simg[1])  +'\t'+ str(r'%.5f'%ch2_clstr_siz_simg[2])  +'\t'+ str(ch2_clstr_siz_simg[3]) +'\t'+ str(r'%.5f'%ch2_clstr_siz_simg[4])
                            file_test.write(txt_test + '\n')
                            print(txt_test)

                            txt_test = 'Chi2_sig_test_sim_Dgauss' +'\t'+'hist_Max_clst'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%ch2_clstr_max_simg[0]) +'\t'+ str(r'%.5f'%ch2_clstr_max_simg[1])  +'\t'+ str(r'%.5f'%ch2_clstr_max_simg[2])  +'\t'+ str(ch2_clstr_max_simg[3]) +'\t'+ str(r'%.5f'%ch2_clstr_max_simg[4])
                            file_test.write(txt_test + '\n')
                            print(txt_test)


                            file_test.close()
