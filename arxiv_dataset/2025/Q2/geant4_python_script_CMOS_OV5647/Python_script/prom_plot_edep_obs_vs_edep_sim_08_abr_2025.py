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
    return comp, err_comp


def comp_2obs(obs1, obs2):
    dif = obs1 - obs2
    err = np.sqrt(obs1 + obs2)
    comp = np.divide(dif, err , where=(err != 0))
    err_comp = comp*np.sqrt( (err*err) / (dif*dif)
            + (1) / (4*err*err) )
    return comp, err_comp


def comp_2sim(sim_gauss, sim2):
    dif = sim_gauss - sim2
    err = 0.2*np.sqrt(sim_gauss*sim_gauss + sim2*sim2 )
    comp = np.divide(dif, err , where=(err != 0))
    err_comp = 0.2*comp*np.sqrt( (err*err) / (dif*dif)
            + (sim_gauss*sim_gauss*sim_gauss*sim_gauss + sim2*sim2*sim2*sim2) / (4*err*err*err*err) )
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

def ratio_2sim(sim_gauss, sim2):
    # Set ratio where obs2 is not zero
    ratio = np.divide(sim_gauss, sim2 , where=(sim2 != 0))
    # Compute error on ratio (null if cannot be computed)
    er_ratio = ratio*np.sqrt( 0.08)

    return ratio, er_ratio

def comp_edep_obs_sim(obs, err_obs, sim, err_sim):
    dif = obs-sim
    err = np.sqrt(err_obs*err_obs + err_sim*err_sim )
    # Set ratio where obs is not zero
    comp_edep = np.divide(dif, obs , where=(obs != 0))
    # Compute error on ratio (null if cannot be computed)
    err_comp = np.abs(comp_edep)*np.sqrt( (err*err) / (dif*dif)
            + err_obs*err_obs/ (obs*obs) )
    return comp_edep, err_comp


'''
comp_edep, err_comp_ede = comp_edep_obs_sim(adc_obs_w, err_adc_obs_w, adc_sim0_w, err_adc_sim0_w)

comp_edep_gauss, err_comp_ede_gauss = comp_edep_obs_sim(adc_obs_w, err_adc_obs_w, adc_sim_gauss_w, err_adc_sim_gauss_w)

comp_edep_min = np.min( np.array(( np.nanmin(comp_edep), np.nanmin(comp_edep_gauss) )))
comp_edep_max = np.max( np.array(( comp_edep[comp_edep < np.Inf].max(),  comp_edep_gauss[comp_edep_gauss < np.Inf].max() )))
plt.errorbar(bincenters_adc_obs, comp_edep_gauss, yerr=0*err_comp_ede_gauss, fmt='C9'+'o', lw=2, markersize=10 )


'''
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

plot_nsig = 1
zoom_delta=0

plot_ratio = 1
zoom_ratio = 0

plt_sim = 1

plt_sim_gauss = 1
plt_sim_nogauss = 0

plt_obs = 1
plt_bg = 0
plt_err = 1


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

nsigm_self = 0

sim_bg = '_ag8_bg'
sim_bg = ''

adc_cut = 0

#if(adc_cut == 0):min_adc=1

simu_no_cut ='_sim_nocut'
simu_no_cut =''

source = 'Sr90'
#source = 'Cs137'
########################################################################################

bin_hist = 40
if(bin_hist>0): strbin = str(bin_hist)+'bin_z_'
else:strbin = 'z_'

#log_y = 1

max_adc=1024
#max_adc=150
min_adc=1


size_thresh = 0
max_thresh = 0


cut0 = 0; cut_dat00= 6; cut_dat01=None; cut_type0='_clsiz'
#cut0 = 1; cut_dat00= 10; cut_dat01=cut_dat00; cut_type0='_cl_size'

cut1 = 0; cut_dat10=50; cut_dat11=650; cut_type1='_cl_mean'
#cut1 = 1; cut_dat10=55; cut_dat11=55.00001; cut_type1='_cl_mean'
cut2 = 0; cut_dat20=101; cut_dat21=None; cut_type2='_clmax_adc'
cut3 = 0; cut_dat30=90; cut_dat31=None; cut_type3='_cl_totalE'

errors_sqrt = 1

labl_opt = 'simple' #'full' , 'simple', 'off'
#labl_opt = 'chi2_nu'

yscal_adc = 4e4
yscal_cls = 3e4
yscal_max = 1e4


yscal_adc = 0
yscal_cls = 0
yscal_max = 0

yscal_adc = 1e6
yscal_cls = 1e6
yscal_max = 5e5


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


source_bg = 'backgnd'
fecha_bg = '2023_Oct_22_00h'
nfrm =1000

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


nframes = nfrm


ssd_sim = 'C:/dat_2025/sim2025/'


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

                for densi in [ False, #True
                              ]:
                    if densi:   dens = '_norm'; histo='norm_'
                    else:   dens = '' ; histo=''

                    for z in level_z:

                        sum_bins_adc_sim_gauss = 0
                        sum_counts_adc_sim_gauss = 0

                        sum_2_bins_adc_sim_gauss = 0
                        sum_2_counts_adc_sim_gauss = 0

                        sum_bins_clstr_siz_sim_gauss = 0
                        sum_counts_clstr_siz_sim_gauss = 0

                        sum_2_bins_clstr_siz_sim_gauss = 0
                        sum_2_counts_clstr_siz_sim_gauss = 0

                        sum_bins_cmax_sim_gauss = 0
                        sum_counts_cmax_sim_gauss = 0

                        sum_2_bins_cmax_sim_gauss = 0
                        sum_2_counts_cmax_sim_gauss = 0
                        ############################################################

                        sum_counts_cmax_cltrsiz_sim_gauss = 0
                        sum_xbin_cmax_cltrsiz_sim_gauss = 0
                        sum_ybin_cmax_cltrsiz_sim_gauss = 0
                        sum_2_counts_cmax_cltrsiz_sim_gauss = 0
                        sum_2_xbin_cmax_cltrsiz_sim_gauss = 0
                        sum_2_ybin_cmax_cltrsiz_sim_gauss = 0

                        ############################################################

                        sum_counts_cmax_cltrsiz_simbg1 = 0
                        sum_xbin_cmax_cltrsiz_simbg1 = 0
                        sum_ybin_cmax_cltrsiz_simbg1 = 0
                        sum_2_counts_cmax_cltrsiz_simbg1 = 0
                        sum_2_xbin_cmax_cltrsiz_simbg1 = 0
                        sum_2_ybin_cmax_cltrsiz_simbg1 = 0



                        for sim_n in range(total_sim):
                            sim_n = '_s' + str(sim_n)

                            if nsigm > 0 :
                                difu_gauss = f"_{gauss_dist}_{k}k_{alpha}a_{nsigm}sig_{zff}zff"
                                d_gauss = gauss_dist+'_z_'+r'$\sigma$ = '+str(nsigm)+r', $\k$ = '+str(k)+r', $\alpha$ = '+str(alpha)+ r', $z_{ff}$ = '+str(zff)
                            if nsigm == 0 :
                                difu_gauss = f"_{gauss_dist}_{k}k_{alpha}a_{zff}zff"
                                d_gauss =gauss_dist+'_z_'+r'$\k$ = '+str(k)+r', $\alpha$ = '+str(alpha)+ r', $z_{ff}$ = '+str(zff)

                            dirsavepatn_plt = ssd_sim +gauss_dist+'_'+source +'_rslhist_'+strbin +str(z*2).zfill(2)+'mm'+ cut_clst_size+cut_max+str_cuts
                            dirsave_plt = dirsavepatn_plt + '/plots'+sim_n+'_'+source+'_'+str(z_count)+'l'+'_'+str(nframes)+'f'+simu_no_cut+condit_edep+difu_gauss+\
                            '/plot_hist_mult_'+source + '_'+str(z_count)+'l_'+fecha+'_'+evt+sim_bg  #+ cuts_dat +simu_no_cut

                            dirsave_plt_mean = 'C:/dat_2025/ADC_vs_' +gauss_dist+'_'+source +dens+'_dghist_'+strbin +str(z*2).zfill(2)+'mm'+str_cuts+'_'+labl_opt +'_mean/'+\
                            '0plots'+'_'+source+'_'+str(z_count)+'l'+'_'+str(nframes)+'f'+simu_no_cut+condit_edep+difu_gauss+'_'+fecha+'_'+evt+sim_bg  #+ cuts_dat +simu_no_cut

                            try:
                                os.makedirs(dirsave_plt_mean)
                            except FileExistsError:
                                pass
                            ################################################################################################################
                            #################################################################################
                            if sim_n == '_s0' :
                                hist_adc_obs = np.load(dirsave_plt+'/'+'hist_adc_obs'+source+\
                                            sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', allow_pickle=True).item()
                                hist_adc_bg = np.load(dirsave_plt+'/'+'hist_adc_bg_'+source_bg+\
                                            sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', allow_pickle=True).item()
                                hist_adc_sim0 = np.load(dirsave_plt+'/'+'hist_adc_sim_'+source+\
                                            sim_n+'_'+gauss_dist+'_0k_0a_0zff'+ sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', allow_pickle=True).item()

                                bins_adc_obs, counts_adc_obs = hist_adc_obs['bins'], hist_adc_obs['counts']
                                bins_adc_bg, counts_adc_bg = hist_adc_bg['bins'], hist_adc_bg['counts']
                                bins_adc_sim0, counts_adc_sim0 = hist_adc_sim0['bins'], hist_adc_sim0['counts']

                                hist_clstr_siz_obs = np.load(dirsave_plt+'/'+'hist_clstr_siz_obs_'+source+\
                                                    sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', allow_pickle=True).item()
                                hist_clstr_siz_bg = np.load(dirsave_plt+'/'+'hist_clstr_siz_bg_'+source+\
                                                    sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', allow_pickle=True).item()
                                hist_clstr_siz_sim0 = np.load(dirsave_plt+'/'+'hist_clstr_siz_sim_'+source+\
                                                    sim_n+'_'+gauss_dist+'_0k_0a_0zff'+ sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', allow_pickle=True).item()

                                bins_clstr_siz_obs, counts_clstr_siz_obs = hist_clstr_siz_obs['bins'], hist_clstr_siz_obs['counts']
                                bins_clstr_siz_bg, counts_clstr_siz_bg = hist_clstr_siz_bg['bins'], hist_clstr_siz_bg['counts']
                                bins_clstr_siz_sim0, counts_clstr_siz_sim0 = hist_clstr_siz_sim0['bins'], hist_clstr_siz_sim0['counts']


                                hist_cmax_obs = np.load(dirsave_plt+'/'+'hist_clstr_max_obs_'+source+\
                                                    sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', allow_pickle=True).item()
                                hist_cmax_bg = np.load(dirsave_plt+'/'+'hist_clstr_max_bg_'+source+\
                                                    sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', allow_pickle=True).item()
                                hist_cmax_sim0 = np.load(dirsave_plt+'/'+'hist_clstr_max_sim_'+source+\
                                                    sim_n+'_'+gauss_dist+'_0k_0a_0zff'+ sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', allow_pickle=True).item()

                                bins_cmax_obs, counts_cmax_obs = hist_cmax_obs['bins'], hist_cmax_obs['counts']
                                bins_cmax_bg, counts_cmax_bg = hist_cmax_bg['bins'], hist_cmax_bg['counts']
                                bins_cmax_sim0, counts_cmax_sim0 = hist_cmax_sim0['bins'], hist_cmax_sim0['counts']

                            print(f"Procesando simulación: _s{sim_n}")
                            try:

                                hist_adc_sim_gauss = np.load(dirsave_plt+'/'+'hist_adc_sim_'+source+\
                                            sim_n+difu_gauss+ sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', allow_pickle=True).item()

                                bins_adc_sim_gauss, counts_adc_sim_gauss = hist_adc_sim_gauss['bins'], hist_adc_sim_gauss['counts']
                                #################################################################################
                                # Acumular datos
                                sum_bins_adc_sim_gauss += bins_adc_sim_gauss
                                sum_counts_adc_sim_gauss += counts_adc_sim_gauss
                                sum_2_bins_adc_sim_gauss += bins_adc_sim_gauss ** 2
                                sum_2_counts_adc_sim_gauss += counts_adc_sim_gauss ** 2
                                #################################################################################
                                hist_cmax_sim_gauss = np.load(dirsave_plt+'/'+'hist_clstr_max_sim_'+source+\
                                                    sim_n+'_'+difu_gauss+ sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', allow_pickle=True).item()

                                bins_cmax_sim_gauss, counts_cmax_sim_gauss = hist_cmax_sim_gauss['bins'], hist_cmax_sim_gauss['counts']
                                #################################################################################
                                # Acumular datos
                                sum_bins_cmax_sim_gauss += bins_cmax_sim_gauss
                                sum_counts_cmax_sim_gauss += counts_cmax_sim_gauss
                                sum_2_bins_cmax_sim_gauss += bins_cmax_sim_gauss ** 2
                                sum_2_counts_cmax_sim_gauss += counts_cmax_sim_gauss ** 2
                                #################################################################################
                                hist_clstr_siz_sim_gauss = np.load(dirsave_plt+'/'+'hist_clstr_siz_sim_'+source+\
                                                    sim_n+'_'+difu_gauss+ sig_bg_thresh + str_cuts + cut_clst_size+cut_max+'.npy', allow_pickle=True).item()

                                bins_clstr_siz_sim_gauss, counts_clstr_siz_sim_gauss = hist_clstr_siz_sim_gauss['bins'], hist_clstr_siz_sim_gauss['counts']
                                #################################################################################
                                # Acumular datos
                                sum_bins_clstr_siz_sim_gauss += bins_clstr_siz_sim_gauss
                                sum_counts_clstr_siz_sim_gauss += counts_clstr_siz_sim_gauss
                                sum_2_bins_clstr_siz_sim_gauss += bins_clstr_siz_sim_gauss ** 2
                                sum_2_counts_clstr_siz_sim_gauss += counts_clstr_siz_sim_gauss ** 2
                                #################################################################################

                            except Exception as e:
                                print(f"Error procesando simulación {sim_n}: {e}")
                                continue

                        # Calcular promedios y desviaciones estándar
                        mean_bins_adc_sim_gauss = sum_bins_adc_sim_gauss / total_sim
                        std_bins_adc_sim_gauss = np.sqrt((sum_2_bins_adc_sim_gauss - total_sim * mean_bins_adc_sim_gauss ** 2) / n_sampl )

                        mean_counts_adc_sim_gauss = sum_counts_adc_sim_gauss / total_sim
                        std_counts_adc_sim_gauss = np.sqrt((sum_2_counts_adc_sim_gauss - total_sim * mean_counts_adc_sim_gauss ** 2) / n_sampl )
                        #mean_counts_adc_sim_gauss = (np.rint(mean_counts_adc_sim_gauss)).astype(int)
                        mean_counts_adc_sim_gauss = (np.rint(mean_counts_adc_sim_gauss)).astype(int)

                        # Manejar NaN
                        mean_bins_adc_sim_gauss, std_bins_adc_sim_gauss = (np.zeros_like(mean_bins_adc_sim_gauss) if np.any(np.isnan(mean_bins_adc_sim_gauss)) else mean_bins_adc_sim_gauss,
                                               np.zeros_like(std_bins_adc_sim_gauss) if np.any(np.isnan(std_bins_adc_sim_gauss)) else std_bins_adc_sim_gauss)

                        mean_counts_adc_sim_gauss, std_counts_adc_sim_gauss = (np.zeros_like(mean_counts_adc_sim_gauss) if np.any(np.isnan(mean_counts_adc_sim_gauss)) else mean_counts_adc_sim_gauss,
                                                   np.zeros_like(std_counts_adc_sim_gauss) if np.any(np.isnan(std_counts_adc_sim_gauss)) else std_counts_adc_sim_gauss)
                        #################################################################################
                        # Calcular promedios y desviaciones estándar
                        mean_bins_cmax_sim_gauss = sum_bins_cmax_sim_gauss / total_sim
                        std_bins_cmax_sim_gauss = np.sqrt((sum_2_bins_cmax_sim_gauss - total_sim * mean_bins_cmax_sim_gauss ** 2) / n_sampl )

                        mean_counts_cmax_sim_gauss = sum_counts_cmax_sim_gauss / total_sim
                        std_counts_cmax_sim_gauss = np.sqrt((sum_2_counts_cmax_sim_gauss - total_sim * mean_counts_cmax_sim_gauss ** 2) / n_sampl )
                        #mean_counts_cmax_sim_gauss = (np.rint(mean_counts_cmax_sim_gauss)).astype(int)
                        mean_counts_cmax_sim_gauss = (np.rint(mean_counts_cmax_sim_gauss)).astype(int)

                        # Manejar NaN
                        mean_bins_cmax_sim_gauss, std_bins_cmax_sim_gauss = (np.zeros_like(mean_bins_cmax_sim_gauss) if np.any(np.isnan(mean_bins_cmax_sim_gauss)) else mean_bins_cmax_sim_gauss,
                                               np.zeros_like(std_bins_cmax_sim_gauss) if np.any(np.isnan(std_bins_cmax_sim_gauss)) else std_bins_cmax_sim_gauss)

                        mean_counts_cmax_sim_gauss, std_counts_cmax_sim_gauss = (np.zeros_like(mean_counts_cmax_sim_gauss) if np.any(np.isnan(mean_counts_cmax_sim_gauss)) else mean_counts_cmax_sim_gauss,
                                                   np.zeros_like(std_counts_cmax_sim_gauss) if np.any(np.isnan(std_counts_cmax_sim_gauss)) else std_counts_cmax_sim_gauss)
                        #################################################################################
                        # Calcular promedios y desviaciones estándar
                        mean_bins_clstr_siz_sim_gauss = sum_bins_clstr_siz_sim_gauss / total_sim
                        std_bins_clstr_siz_sim_gauss = np.sqrt((sum_2_bins_clstr_siz_sim_gauss - total_sim * mean_bins_clstr_siz_sim_gauss ** 2) / n_sampl )

                        mean_counts_clstr_siz_sim_gauss = sum_counts_clstr_siz_sim_gauss / total_sim
                        std_counts_clstr_siz_sim_gauss = np.sqrt((sum_2_counts_clstr_siz_sim_gauss - total_sim * mean_counts_clstr_siz_sim_gauss ** 2) / n_sampl )
                        #mean_counts_clstr_siz_sim_gauss = (np.rint(mean_counts_clstr_siz_sim_gauss)).astype(int)
                        mean_counts_clstr_siz_sim_gauss = (np.rint(mean_counts_clstr_siz_sim_gauss)).astype(int)

                        # Manejar NaN
                        mean_bins_clstr_siz_sim_gauss, std_bins_clstr_siz_sim_gauss = (np.zeros_like(mean_bins_clstr_siz_sim_gauss) if np.any(np.isnan(mean_bins_clstr_siz_sim_gauss)) else mean_bins_clstr_siz_sim_gauss,
                                               np.zeros_like(std_bins_clstr_siz_sim_gauss) if np.any(np.isnan(std_bins_clstr_siz_sim_gauss)) else std_bins_clstr_siz_sim_gauss)

                        mean_counts_clstr_siz_sim_gauss, std_counts_clstr_siz_sim_gauss = (np.zeros_like(mean_counts_clstr_siz_sim_gauss) if np.any(np.isnan(mean_counts_clstr_siz_sim_gauss)) else mean_counts_clstr_siz_sim_gauss,
                                                   np.zeros_like(std_counts_clstr_siz_sim_gauss) if np.any(np.isnan(std_counts_clstr_siz_sim_gauss)) else std_counts_clstr_siz_sim_gauss)

                        #################################################################################
                        #################################################################################
                        # Imprimir resultados
                        '''
                        print(f"Promedio de bins adc_sim_gauss: {mean_bins_adc_sim_gauss}\n, Desviación estándar adc_sim_gauss: {std_bins_adc_sim_gauss}")
                        print(f"Promedio de counts adc_sim_gauss: {mean_counts_adc_sim_gauss}\n, Desviación estándar adc_sim_gauss: {std_counts_adc_sim_gauss}")

                        print(f"Promedio de bins cmax_sim_gauss: {mean_bins_cmax_sim_gauss}\n, Desviación estándar cmax_sim_gauss: {std_bins_cmax_sim_gauss}")
                        print(f"Promedio de counts cmax_sim_gauss: {mean_counts_cmax_sim_gauss}\n, Desviación estándar cmax_sim_gauss: {std_counts_cmax_sim_gauss}")

                        print(f"Promedio de bins clstr_siz_sim_gauss: {mean_bins_clstr_siz_sim_gauss}\n, Desviación estándar clstr_siz_sim_gauss: {std_bins_clstr_siz_sim_gauss}")
                        print(f"Promedio de counts clstr_siz_sim_gauss: {mean_counts_clstr_siz_sim_gauss}\n, Desviación estándar clstr_siz_sim_gauss: {std_counts_clstr_siz_sim_gauss}")
                        '''

                        # ADC histos
                        #################################################################################
                        bincenters_adc_obs = 0.5*(bins_adc_obs[1:]+bins_adc_obs[:-1])
                        norm_adc_obs = np.sum(counts_adc_obs * np.diff(bins_adc_obs))
                        #################################################################################
                        bincenters_adc_bg = 0.5*(bins_adc_bg[1:]+bins_adc_bg[:-1])
                        norm_adc_bg = np.sum(counts_adc_bg * np.diff(bins_adc_bg))
                        #################################################################################
                        bincenters_adc_sim0 = 0.5*(bins_adc_sim0[1:]+bins_adc_sim0[:-1])
                        norm_adc_sim0 = np.sum(counts_adc_sim0 * np.diff(bins_adc_sim0))
                        #################################################################################
                        bincenters_adc_sim_gauss = 0.5*(bins_adc_sim_gauss[1:]+bins_adc_sim_gauss[:-1])
                        norm_adc_sim_gauss = np.sum(mean_counts_adc_sim_gauss * np.diff(bins_adc_sim_gauss))
                        #################################################################################

                        if(densi==False):
                            adc_obs_w = counts_adc_obs
                            err_adc_obs_w = np.sqrt(counts_adc_obs)
                            #################################################################################
                            adc_bg_w = counts_adc_bg
                            err_adc_bg_w = np.sqrt(counts_adc_bg)
                            #################################################################################
                            adc_sim0_w = counts_adc_sim0
                            if(errors_sqrt == True): err_adc_sim0_w = np.sqrt(0.04*counts_adc_sim0*counts_adc_sim0 + counts_adc_sim0)
                            else: err_adc_sim0_w = (0.2*counts_adc_sim0 + np.sqrt(counts_adc_sim0))
                            #################################################################################
                            adc_sim_gauss_w = mean_counts_adc_sim_gauss
                            if(errors_sqrt == True): err_adc_sim_gauss_w = np.sqrt(0.04*mean_counts_adc_sim_gauss*mean_counts_adc_sim_gauss + std_counts_adc_sim_gauss*std_counts_adc_sim_gauss + mean_counts_adc_sim_gauss)
                            else: err_adc_sim_gauss_w = (0.2*mean_counts_adc_sim_gauss + std_counts_adc_sim_gauss )
                            #################################################################################
                        if(densi==True):
                            adc_obs_w = counts_adc_obs/norm_adc_obs
                            err_adc_obs_w = np.sqrt(counts_adc_obs)/norm_adc_obs
                            #################################################################################
                            adc_bg_w = counts_adc_bg/norm_adc_bg
                            err_adc_bg_w = np.sqrt(counts_adc_bg)/norm_adc_bg
                            #################################################################################
                            adc_sim0_w = counts_adc_sim0/norm_adc_sim0
                            if(errors_sqrt == True): err_adc_sim0_w = np.sqrt(0.04*counts_adc_sim0*counts_adc_sim0 + counts_adc_sim0)/norm_adc_sim0
                            else: err_adc_sim0_w = (0.2*counts_adc_sim0 + np.sqrt(counts_adc_sim0))/norm_adc_sim0
                            #################################################################################
                            adc_sim_gauss_w = mean_counts_adc_sim_gauss/norm_adc_sim_gauss
                            if(errors_sqrt == True): err_adc_sim_gauss_w = np.sqrt(0.04*mean_counts_adc_sim_gauss*mean_counts_adc_sim_gauss + std_counts_adc_sim_gauss*std_counts_adc_sim_gauss + mean_counts_adc_sim_gauss)/norm_adc_sim_gauss
                            else: err_adc_sim_gauss_w = (0.2*mean_counts_adc_sim_gauss + std_counts_adc_sim_gauss )/norm_adc_sim_gauss
                            #################################################################################

                        err_adc0 = np.sqrt(err_adc_obs_w*err_adc_obs_w + err_adc_sim0_w*err_adc_sim0_w )
                        ch2_adc0 = chi_2_sigm_test(adc_obs_w, adc_sim0_w, err_adc0)
                        tmp_str0 = 'Chi2 test: ' + r'%.2f'%ch2_adc0[0] \
                                    +'       '+'ndf: ' + r'%.2f'%ch2_adc0[3] \
                                    +'\n'+'Chi2/ndf : ' + r'%.2f'%ch2_adc0[4]

                        err_adc1 = np.sqrt(err_adc_obs_w*err_adc_obs_w + err_adc_sim_gauss_w*err_adc_sim_gauss_w )
                        ch2_adc1 = chi_2_sigm_test(adc_obs_w, adc_sim_gauss_w, err_adc1)
                        tmp_str1 = 'Chi2 test: ' + r'%.2f'%ch2_adc1[0] \
                                    +'       '+'ndf: ' + r'%.2f'%ch2_adc1[3] \
                                    +'\n'+'Chi2/ndf : ' + r'%.2f'%ch2_adc1[4]

                        if(labl_opt=='simple'):
                            lbl_obs = source + ' data'
                            lbl_sim0 = source + ' simulation '
                            lbl_sim_gauss = source + ' simulation '
                            lbl_bg = 'Background' + ' data'
                        if(labl_opt=='chi2_nu'):
                            lbl_obs = source + ' data'
                            lbl_sim0 = source + ' sim ' +r'$\chi^2_\nu$ = ' + r'%.2f'%ch2_adc0[4] #+'+ bg'
                            lbl_sim_gauss = source + ' sim ' + r'$\chi^2_\nu$ = '  + r'%.2f'%ch2_adc1[4]#+'+ bg'
                            lbl_bg = 'Background' + ' data'
                        if(labl_opt=='off'):
                            lbl_obs = None
                            lbl_sim0 = None
                            lbl_sim_gauss = None
                            lbl_bg = None

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
                            #ax0.step( np.insert(bins_adc_obs, 0, 0), np.concatenate(([0], adc_obs_w, [0])), where='post'
                            #            , color='C3', markersize=10, linewidth=l_width+0.*l_width, label = lbl_obs + cut_max_clst_porc )
                            ax0.set_yscale('log')   # Escala logarítmica en el eje y

                            if(plt_err==True):
                                ax0.errorbar(bincenters_adc_obs, adc_obs_w, yerr=err_adc_obs_w,
                                fmt='C3'+'o', ecolor='C3', markersize=10, linewidth=l_width,  label = lbl_obs + cut_max_clst_porc )
                        ################################################################################################################################

                        if(plt_sim==True):
                            if(plt_sim_nogauss==True):
                                #ax0.step( np.insert(bins_adc_obs, 0, 0), np.concatenate(([0], adc_sim0_w, [0])), where='post',
                                        # color='C2', markersize=6, alpha=0.2, linewidth=l_width, alpha=0.6,  label = lbl_sim0 + cut_max_clst_porc_sim )
                                ax0.fill_between(bincenters_adc_sim0, adc_sim0_w - err_adc_sim0_w, adc_sim0_w + err_adc_sim0_w,
                                            color='C2', linewidth=l_width*2/3, alpha=0.8,  label = lbl_sim0 + cut_max_clst_porc_sim )
                                ax0.set_yscale('log')
                            if(plt_sim_gauss==True):
                                #ax0.step( np.insert(bins_adc_obs, 0, 0), np.concatenate(([0], adc_sim_gauss_w, [0])), where='post' ,
                                        # color='C9', markersize=6, alpha=0.2, linewidth=l_width, alpha=0.6,  label = lbl_sim_gauss + cut_max_clst_porc_sim)
                                ax0.fill_between(bincenters_adc_sim_gauss, adc_sim_gauss_w - err_adc_sim_gauss_w, adc_sim_gauss_w + err_adc_sim_gauss_w,
                                            color='C9', linewidth=l_width*2/3, alpha=0.8,  label = lbl_sim_gauss + cut_max_clst_porc_sim )
                                ax0.set_yscale('log')

                        if(plt_sim==True and plt_err==True):
                            if(plt_sim_nogauss==True):
                                ax0.errorbar(bincenters_adc_sim0, adc_sim0_w, yerr=err_adc_sim0_w, fmt='C2'+'o',
                                            ecolor='C2', markersize=6, alpha=0.2, linewidth=l_width*2/3 )
                            if(plt_sim_gauss==True):
                                ax0.errorbar(bincenters_adc_sim_gauss, adc_sim_gauss_w, yerr=err_adc_sim_gauss_w, fmt='C9'+'o',
                                            ecolor='C9', markersize=6, alpha=0.2, linewidth=l_width*2/3 )

                        ################################################################################################################################
                        ################################################################################################################################

                        if(plt_bg==False):
                            y_min_adc = np.min( [np.min( adc_obs_w + err_adc_obs_w ), np.min( adc_sim0_w + err_adc_sim0_w )  ] )
                            y_min_adc = 0.5*y_min_adc
                            y_max_adc = np.min( [np.max( adc_obs_w + err_adc_obs_w ), np.max( adc_sim0_w + err_adc_sim0_w ) ] )
                            y_max_adc = 1.5*y_max_adc
                            ax0.set_ylim(y_min_adc )

                        U_eV = 1000
                        min_cam =((min_adc-1)*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
                        sat_cam =(1023*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
                        if(max_adc>=1023):ax0.axvline(x = 1023, color = 'k', linestyle="--", linewidth=l_width*2/3)#, label = 'sat_cam: 1023 ADC = ' + str(sat_cam)+' keV')


                        if(plt_obs==True):
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

                        ylabl_comp= r'$\frac{ (data -sim)}{\sigma}$'
                        comp_sim0, err_comp_sim0 = comp_obs_sim(adc_obs_w, adc_sim0_w)

                        comp_sim_gauss, err_comp_sim_gauss = comp_obs_sim(adc_obs_w, adc_sim_gauss_w)

                        delta_min = np.min(np.array(( np.nanmin(comp_sim0), np.nanmin(comp_sim_gauss) )))
                        delta_max = np.max(np.array(( comp_sim0[comp_sim0 < np.Inf].max(), comp_sim_gauss[comp_sim_gauss < np.Inf].max() )))

                        if(plt_sim==True and plot_nsig==True):
                            #ax1.axhline(y = 0., xmin=0, xmax=max_adc, color = 'k', linestyle = '--', linewidth=l_width*2/3)
                            #ax1.step( np.insert(bins_adc_obs, 0, 0), 0*np.concatenate(([0], adc_obs_w, [0])), where='post', ls='--',  color='k', linewidth=l_width*2/3 )
                            if(plt_sim_nogauss==True):
                                ax1.errorbar(bincenters_adc_obs, comp_sim0, yerr=plt_err*err_comp_sim0, fmt='C2'+'>', lw=2, markersize=10 )
                            if(plt_sim_gauss==True):
                                ax1.errorbar(bincenters_adc_obs, comp_sim_gauss, yerr=plt_err*err_comp_sim_gauss, fmt='C9'+'>', lw=2, markersize=10 )
                            if(zoom_delta==True):ax1.set_ylim(delta_min,delta_max)


                        if( (plt_sim==True) and plot_nsig==True ):
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
                        ratio_sim0, err_ratio_sim0 = ratio_obs_sim(adc_obs_w, adc_sim0_w)
                        ratio_sim_gauss, err_ratio_sim_gauss = ratio_obs_sim(adc_obs_w, adc_sim_gauss_w)
                        ratio_min = np.min( np.array(( np.nanmin(ratio_sim0), np.nanmin(ratio_sim_gauss) )))
                        ratio_max = np.max( np.array(( ratio_sim0[ratio_sim0 < np.Inf].max(),  ratio_sim_gauss[ratio_sim_gauss < np.Inf].max() )))

                        if(plt_sim==True and plot_ratio==True):
                            #ax2.axhline(y = 0., xmin=min_adc, xmax=max_adc, color = 'k', linestyle = '--')
                            #ax2.hist(adc_obs_w*0, bins = bins_adc_obs, histtype='step', ls='--',  color='k', linewidth=l_width*2/3 )
                            #ax2.step( np.insert(bins_adc_obs, 0, 0), 0*np.concatenate(([0], adc_obs_w, [0])), where='post', ls='--',  color='k', linewidth=l_width*2/3 )
                            if(plt_sim_nogauss==True):
                                ax2.errorbar(bincenters_adc_obs, ratio_sim0, yerr=plt_err*err_ratio_sim0, fmt='C2'+'>', lw=2, markersize=10 )
                            if(plt_sim_gauss==True):
                                ax2.errorbar(bincenters_adc_obs, ratio_sim_gauss, yerr=plt_err*err_ratio_sim_gauss, fmt='C9'+'>', lw=2, markersize=10 )
                            if(zoom_ratio==True):ax2.set_ylim(-0.5+ratio_min,ratio_max)

                        if( (plt_sim==True) and plot_ratio==True ):
                            #ax2[2].xticks(size=font_siz+2)
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

                        adc2ekv=(bins_adc_obs*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
                        ax3.plot(adc2ekv, bins_adc_obs*0., color='w', linewidth=0.001 )
                        ax3.set_xlabel('Energy (keV)', fontsize=font_siz+4)

                        #ax0.legend(loc='lower right')
                        #ax0.legend(loc='lower left')
                        #ax0.legend(loc='upper center')
                        #ax0.legend(loc='upper left', bbox_to_anchor=(0.62, 1))
                        #ax0.legend(loc='upper right')

                        # Crear una lista de handles y etiquetas en el orden deseado
                        handles, labels = ax0.get_legend_handles_labels()
                        if(plt_bg==True): new_order = [1, 0, 2]  # Suponiendo que deseas invertir el orden
                        else: new_order = [1, 0]  # Suponiendo que deseas invertir el orden
                        handles = [handles[i] for i in new_order]
                        labels = [labels[i] for i in new_order]

                        # Crear la leyenda con el nuevo orden
                        ax0.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.62, 1))

                        fig.tight_layout()
                        name_cuts = source+sim_n+difu_gauss+ sig_bg_thresh + str_cuts + cut_clst_size+cut_max + str_hist_comp
                        namesave = 'plot_hist_'+histo+'ADC_'+strbin+str(z*2).zfill(2)+'mm'+ name_cuts
                        #plt.savefig(dirsave_plt_mean+'/'+namesave+'.png', dpi=150)
                        #plt.savefig(dirsave_plt_mean+'/'+namesave+'.pdf', dpi=150)
                        plt.show()
                        #plt.clf()
                        #plt.close()


                        plt.rcParams.update({'font.size': font_siz})
                        fig, (ax) = plt.subplots(figsize=(15.5,10) )#, sharex=True )
                        ax.tick_params(axis='both', labelsize=font_siz)

                        #ax.axhline(y=sat_cam, color='k', linestyle="--", linewidth=2.5)
                        #ax.axhline(y=min_cam, color='k', linestyle="--", linewidth=2.5)

                        bins_x = np.logspace(np.log10(min(bincenters_adc_obs)), np.log10(max(bincenters_adc_obs)), bin_hist)
                        bins_y = np.logspace(np.log10(min(bincenters_adc_sim_gauss)), np.log10(max(bincenters_adc_sim_gauss)), bin_hist)
                        bins_x = bincenters_adc_obs
                        bins_y = bincenters_adc_sim_gauss

                        h, xedges, yedges, im = ax.hist2d(adc_obs_w, adc_sim_gauss_w, bins=[bincenters_adc_obs, bincenters_adc_sim_gauss], norm=matplotlib.colors.LogNorm(), cmap=plt.cm.jet)
                        cbar = plt.colorbar(im, ax=ax, label='Frecuency')     
                        cbar.ax.tick_params(labelsize=font_siz - 4)
                        cbar.ax.set_ylabel('Frecuency', fontsize=font_siz - 2)

                        #ax.text(1, sat_cam, '1023 ADC', ha='left', va='bottom', fontsize=font_siz - 4)
                        #ax.text(1, min_cam, '100 ADC', ha='left', va='bottom', fontsize=font_siz - 4)

                        #ax.set_xscale('log')
                        #ax.set_xlim(1, max(ekinpre_keV))
                        #ax.set_yscale('log')
                        #ax.set_ylim(1e-2, 2e2)
                        ax.set_xlabel('ADC Obs', fontsize=font_siz-2)
                        ax.set_ylabel('ADC Sim G', fontsize=font_siz-2)
                        ax.set_title(source1, fontsize=font_siz-4)
                        plt.tight_layout()
                        #plt.savefig(f'{dirfile}/hist2d_normlog_{source1}_Edep_vs_Ekin{particle_type}.png', dpi=150)
                        #plt.savefig(f'{dirfile}/hist2d_normlog_{source1}_Edep_vs_Ekin{particle_type}.pdf', dpi=150)

                        plt.show()

                        # Cluster size histos
                        #################################################################################
                        bincenters_clstr_siz_obs = 0.5*(bins_clstr_siz_obs[1:]+bins_clstr_siz_obs[:-1])
                        norm_clstr_siz_obs = np.sum(counts_clstr_siz_obs * np.diff(bins_clstr_siz_obs))
                        #################################################################################
                        bincenters_clstr_siz_bg = 0.5*(bins_clstr_siz_bg[1:]+bins_clstr_siz_bg[:-1])
                        norm_clstr_siz_bg = np.sum(counts_clstr_siz_bg * np.diff(bins_clstr_siz_bg))
                        #################################################################################
                        bincenters_clstr_siz_sim0 = 0.5*(bins_clstr_siz_sim0[1:]+bins_clstr_siz_sim0[:-1])
                        norm_clstr_siz_sim0 = np.sum(counts_clstr_siz_sim0 * np.diff(bins_clstr_siz_sim0))
                        #################################################################################
                        bincenters_clstr_siz_sim_gauss = 0.5*(bins_clstr_siz_sim_gauss[1:]+bins_clstr_siz_sim_gauss[:-1])
                        norm_clstr_siz_sim_gauss = np.sum(mean_counts_clstr_siz_sim_gauss * np.diff(bins_clstr_siz_sim_gauss))
                        #################################################################################

                        if(densi==False):
                            clstr_siz_obs_w = counts_clstr_siz_obs
                            err_clstr_siz_obs_w = np.sqrt(counts_clstr_siz_obs)
                            #################################################################################
                            clstr_siz_bg_w = counts_clstr_siz_bg
                            err_clstr_siz_bg_w = np.sqrt(counts_clstr_siz_bg)
                            #################################################################################
                            clstr_siz_sim0_w = counts_clstr_siz_sim0
                            if(errors_sqrt == True): err_clstr_siz_sim0_w = np.sqrt(0.04*counts_clstr_siz_sim0*counts_clstr_siz_sim0 +counts_clstr_siz_sim0)
                            else: err_clstr_siz_sim0_w = (0.2*counts_clstr_siz_sim0 + np.sqrt(counts_clstr_siz_sim0))
                            #################################################################################
                            clstr_siz_sim_gauss_w = mean_counts_clstr_siz_sim_gauss
                            if(errors_sqrt == True): err_clstr_siz_sim_gauss_w = np.sqrt(0.04*mean_counts_clstr_siz_sim_gauss*mean_counts_clstr_siz_sim_gauss + std_counts_clstr_siz_sim_gauss*std_counts_clstr_siz_sim_gauss + mean_counts_clstr_siz_sim_gauss)
                            else: err_clstr_siz_sim_gauss_w = (0.2*mean_counts_clstr_siz_sim_gauss + std_counts_clstr_siz_sim_gauss )
                            #################################################################################
                        if(densi==True):
                            clstr_siz_obs_w = counts_clstr_siz_obs/norm_clstr_siz_obs
                            err_clstr_siz_obs_w = np.sqrt(counts_clstr_siz_obs)/norm_clstr_siz_obs
                            #################################################################################
                            clstr_siz_bg_w = counts_clstr_siz_bg/norm_clstr_siz_bg
                            err_clstr_siz_bg_w = np.sqrt(counts_clstr_siz_bg)/norm_clstr_siz_bg
                            #################################################################################
                            clstr_siz_sim0_w = counts_clstr_siz_sim0/norm_clstr_siz_sim0
                            if(errors_sqrt == True): err_clstr_siz_sim0_w = np.sqrt(0.04*counts_clstr_siz_sim0*counts_clstr_siz_sim0 + counts_clstr_siz_sim0)/norm_clstr_siz_sim0
                            else: err_clstr_siz_sim0_w = (0.2*counts_clstr_siz_sim0 + np.sqrt(counts_clstr_siz_sim0))/norm_clstr_siz_sim0
                            #################################################################################
                            clstr_siz_sim_gauss_w = mean_counts_clstr_siz_sim_gauss/norm_clstr_siz_sim_gauss
                            if(errors_sqrt == True): err_clstr_siz_sim_gauss_w = np.sqrt(0.04*mean_counts_clstr_siz_sim_gauss*mean_counts_clstr_siz_sim_gauss + std_counts_clstr_siz_sim_gauss*std_counts_clstr_siz_sim_gauss + mean_counts_clstr_siz_sim_gauss)/norm_clstr_siz_sim_gauss
                            else: err_clstr_siz_sim_gauss_w = (0.2*mean_counts_clstr_siz_sim_gauss + std_counts_clstr_siz_sim_gauss )/norm_clstr_siz_sim_gauss
                            #################################################################################

                        err_clstr_siz0 = np.sqrt(err_clstr_siz_obs_w*err_clstr_siz_obs_w + err_clstr_siz_sim0_w*err_clstr_siz_sim0_w )
                        ch2_clstr_siz0 = chi_2_sigm_test(clstr_siz_obs_w, clstr_siz_sim0_w, err_clstr_siz0)
                        tmp_str0 = 'Chi2 test: ' + r'%.2f'%ch2_clstr_siz0[0] \
                                    +'       '+'ndf: ' + r'%.2f'%ch2_clstr_siz0[3] \
                                    +'\n'+'Chi2/ndf : ' + r'%.2f'%ch2_clstr_siz0[4]

                        err_clstr_siz1 = np.sqrt(err_clstr_siz_obs_w*err_clstr_siz_obs_w + err_clstr_siz_sim_gauss_w*err_clstr_siz_sim_gauss_w )
                        ch2_clstr_siz1 = chi_2_sigm_test(clstr_siz_obs_w, clstr_siz_sim_gauss_w, err_clstr_siz1)
                        tmp_str1 = 'Chi2 test: ' + r'%.2f'%ch2_clstr_siz1[0] \
                                    +'       '+'ndf: ' + r'%.2f'%ch2_clstr_siz1[3] \
                                    +'\n'+'Chi2/ndf : ' + r'%.2f'%ch2_clstr_siz1[4]

                        if(labl_opt=='simple'):
                            lbl_obs = source + ' data'
                            lbl_sim0 = source + ' simulation '
                            lbl_sim_gauss = source + ' simulation '
                            lbl_bg = 'Background' + ' data'
                        if(labl_opt=='chi2_nu'):
                            lbl_obs = source + ' data'
                            lbl_sim0 = source + ' sim ' +r'$\chi^2_\nu$ = ' + r'%.2f'%ch2_clstr_siz0[4] #+'+ bg'
                            lbl_sim_gauss = source + ' sim ' + r'$\chi^2_\nu$ = '  + r'%.2f'%ch2_clstr_siz1[4]#+'+ bg'
                            lbl_bg = 'Background' + ' data'
                        if(labl_opt=='off'):
                            lbl_obs = None
                            lbl_sim0 = None
                            lbl_sim_gauss = None
                            lbl_bg = None

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
                            #ax0.step( np.insert(bins_clstr_siz_obs, 0, 0), np.concatenate(([0], clstr_siz_obs_w, [0])), where='post'
                            #            , color='C3', markersize=10, linewidth=l_width+0.*l_width,   label = lbl_obs + cut_max_clst_porc )
                            ax0.set_yscale('log')   # Escala logarítmica en el eje y

                            if(plt_err==True):
                                ax0.errorbar(bincenters_clstr_siz_obs, clstr_siz_obs_w, yerr=err_clstr_siz_obs_w,
                                fmt='C3'+'o', ecolor='C3', markersize=10, linewidth=l_width, label = lbl_obs + cut_max_clst_porc )
                        ################################################################################################################################

                        if(plt_sim==True):
                            # Crear datos suavizados usando un spline
                            x_new = np.linspace(bincenters_clstr_siz_sim0.min(), bincenters_clstr_siz_sim0.max(), 500)
                            spline_upper = make_interp_spline(bincenters_clstr_siz_sim0, clstr_siz_sim0_w + err_clstr_siz_sim0_w, k=3)
                            spline_lower = make_interp_spline(bincenters_clstr_siz_sim0, clstr_siz_sim0_w - err_clstr_siz_sim0_w, k=3)

                            upper_smooth = spline_upper(x_new)
                            lower_smooth = spline_lower(x_new)
                            if(plt_sim_nogauss==True):
                                #ax0.step( np.insert(bins_clstr_siz_obs, 0, 0), np.concatenate(([0], clstr_siz_sim0_w, [0])), where='post'
                                #        , color='C2', markersize=6, alpha=0.2, linewidth=l_width, alpha=0.6,  label = lbl_sim0 + cut_max_clst_porc_sim )
                                ax0.fill_between(bincenters_clstr_siz_sim0, clstr_siz_sim0_w - err_clstr_siz_sim0_w, clstr_siz_sim0_w + err_clstr_siz_sim0_w,
                                #ax0.fill_between(x_new, spline_lower, cspline_upper,
                                            color='C2', linewidth=l_width*2/3, alpha=0.6,  label = lbl_sim0 + cut_max_clst_porc_sim )
                                ax0.set_yscale('log')
                            if(plt_sim_gauss==True):
                                #ax0.step( np.insert(bins_clstr_siz_obs, 0, 0), np.concatenate(([0], clstr_siz_sim_gauss_w, [0])), where='post'
                                #        , color='C9', markersize=6, alpha=0.2, linewidth=l_width, alpha=0.6,  label = lbl_sim_gauss + cut_max_clst_porc_sim)
                                ax0.fill_between(bincenters_clstr_siz_sim_gauss, clstr_siz_sim_gauss_w - err_clstr_siz_sim_gauss_w, clstr_siz_sim_gauss_w + err_clstr_siz_sim_gauss_w,
                                            color='C9', linewidth=l_width*2/3, alpha=0.6,  label = lbl_sim_gauss + cut_max_clst_porc_sim )
                                ax0.set_yscale('log')

                        if(plt_sim==True and plt_err==True):
                            if(plt_sim_nogauss==True):
                                ax0.errorbar(bincenters_clstr_siz_sim0, clstr_siz_sim0_w, yerr=err_clstr_siz_sim0_w, fmt='C2'+'o',
                                            ecolor='C2', markersize=6, alpha=0.2, linewidth=l_width*2/3 )
                            if(plt_sim_gauss==True):
                                ax0.errorbar(bincenters_clstr_siz_sim_gauss, clstr_siz_sim_gauss_w, yerr=err_clstr_siz_sim_gauss_w, fmt='C9'+'o',
                                            ecolor='C9', markersize=6, alpha=0.2, linewidth=l_width*2/3 )

                        ################################################################################################################################
                        ################################################################################################################################

                        if(plt_bg==True):
                            #ax0.step( np.insert(bins_clstr_siz_obs, 0, 0), np.concatenate(([0], clstr_siz_bg_w, [0])), where='post'
                            #            , color='k', markersize=12, linewidth=l_width*2/3,
                                        #label='Back_ground'  +'_'+ag+'ag'+'_' +str(nsigm_bg_sim)+'sig' +'_t'+tiemp )
                            #            label = lbl_bg )# + '_' + fecha_bg )
                            ax0.set_yscale('log')
                            if(plt_err==True):
                                ax0.errorbar(bincenters_clstr_siz_bg, clstr_siz_bg_w, yerr=err_clstr_siz_bg_w, fmt='k'+'o',
                                                ecolor='k', markersize=12, linewidth=l_width*2/3 , label = lbl_bg )

                        if(plt_bg==True):
                            y_min_clstr_siz = np.min( [np.min( clstr_siz_obs_w + err_clstr_siz_obs_w ), np.min( clstr_siz_bg_w + err_clstr_siz_bg_w )  ] )
                            y_min_clstr_siz = 0.5*y_min_clstr_siz
                        if(plt_bg==False):
                            y_min_clstr_siz = np.min( [np.min( clstr_siz_obs_w + err_clstr_siz_obs_w ), np.min( clstr_siz_sim0_w + err_clstr_siz_sim0_w )  ] )
                            y_min_clstr_siz = 0.5*y_min_clstr_siz
                            y_max_clstr_siz = np.min( [np.max( clstr_siz_obs_w + err_clstr_siz_obs_w ), np.max( clstr_siz_sim0_w + err_clstr_siz_sim0_w ) ] )
                            y_max_clstr_siz = 1.5*y_max_clstr_siz
                            ax0.set_ylim(y_min_clstr_siz)


                        if(plt_obs==True):
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

                        ylabl_comp= r'$\frac{ (data -sim)}{\sigma}$'
                        comp_sim0, err_comp_sim0 = comp_obs_sim(clstr_siz_obs_w, clstr_siz_sim0_w)
                        comp_sim_gauss, err_comp_sim_gauss = comp_obs_sim(clstr_siz_obs_w, clstr_siz_sim_gauss_w)
                        delta_min = np.min(np.array((np.nanmin(comp_sim0), np.nanmin(comp_sim_gauss) )))
                        delta_max = np.max(np.array( (comp_sim0[comp_sim0 < np.Inf].max(), comp_sim_gauss[comp_sim_gauss < np.Inf].max() )))

                        if(plt_sim==True and plot_nsig==True):
                            #ax1.axhline(y = 0., xmin=0, xmax=max_adc, color = 'k', linestyle = '--', linewidth=l_width*2/3)
                            #ax1.step( np.insert(bins_clstr_siz_obs, 0, 0), 0*np.concatenate(([0], clstr_siz_obs_w, [0])), where='post', ls='--',  color='k', linewidth=l_width*2/3 )
                            if(plt_sim_nogauss==True):
                                ax1.errorbar(bincenters_clstr_siz_obs, comp_sim0, yerr=plt_err*err_comp_sim0, fmt='C2'+'>', lw=2, markersize=10 )
                            if(plt_sim_gauss==True):
                                ax1.errorbar(bincenters_clstr_siz_obs, comp_sim_gauss, yerr=plt_err*err_comp_sim_gauss, fmt='C9'+'>', lw=2, markersize=10 )
                            if(zoom_delta==True):ax1.set_ylim(delta_min,delta_max)

                        if( (plt_sim==True) and plot_nsig==True ):
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
                        ratio_sim0, err_ratio_sim0 = ratio_obs_sim(clstr_siz_obs_w, clstr_siz_sim0_w)
                        ratio_sim_gauss, err_ratio_sim_gauss = ratio_obs_sim(clstr_siz_obs_w, clstr_siz_sim_gauss_w)
                        ratio_min = np.min(np.array(( np.nanmin(ratio_sim0), np.nanmin(ratio_sim_gauss) )))
                        ratio_max = np.max(np.array(( ratio_sim0[ratio_sim0 < np.Inf].max(),  ratio_sim0[ratio_sim_gauss < np.Inf].max() )))

                        if(plt_sim==True and plot_ratio==True):
                            #ax2.axhline(y = 0., xmin=min_adc, xmax=max_adc, color = 'k', linestyle = '--')
                            #ax2.hist(clstr_siz_obs_w*0, bins = bins_clstr_siz_obs, histtype='step', ls='--',  color='k', linewidth=l_width*2/3 )
                            #ax2.step( np.insert(bins_clstr_siz_obs, 0, 0), 0*np.concatenate(([0], clstr_siz_obs_w, [0])), where='post', ls='--',  color='k', linewidth=l_width*2/3 )
                            if(plt_sim_nogauss==True):
                                ax2.errorbar(bincenters_clstr_siz_obs, ratio_sim0, yerr=plt_err*err_ratio_sim0, fmt='C2'+'>', lw=2, markersize=10 )
                            if(plt_sim_gauss==True):
                                ax2.errorbar(bincenters_clstr_siz_obs, ratio_sim_gauss, yerr=plt_err*err_ratio_sim_gauss, fmt='C9'+'>', lw=2, markersize=10 )
                            if(zoom_ratio==True):ax2.set_ylim(-0.5+ratio_min,ratio_max)

                        if( (plt_sim==True) and plot_ratio==True ):
                            #ax2[2].xticks(size=font_siz+2)
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

                        # Crear una lista de handles y etiquetas en el orden deseado
                        handles, labels = ax0.get_legend_handles_labels()
                        if(plt_bg==True): new_order = [1, 0, 2]  # Suponiendo que deseas invertir el orden
                        else: new_order = [1, 0]  # Suponiendo que deseas invertir el orden
                        handles = [handles[i] for i in new_order]
                        labels = [labels[i] for i in new_order]

                        # Crear la leyenda con el nuevo orden
                        ax0.legend(handles, labels)

                        fig.tight_layout()
                        name_cuts = source+sim_n+difu_gauss+ sig_bg_thresh + str_cuts + cut_clst_size+cut_max + str_hist_comp
                        namesave = 'plot_hist_'+histo+'cltr_siz_'+strbin+str(z*2).zfill(2)+'mm'+ name_cuts
                        #plt.savefig(dirsave_plt_mean+'/'+namesave+'.png', dpi=150)
                        #plt.savefig(dirsave_plt_mean+'/'+namesave+'.pdf', dpi=150)
                        plt.show()
                        #plt.clf()
                        #plt.close()



                        # Cluster Max histos
                        #################################################################################
                        bincenters_cmax_obs = 0.5*(bins_cmax_obs[1:]+bins_cmax_obs[:-1])
                        norm_cmax_obs = np.sum(counts_cmax_obs * np.diff(bins_cmax_obs))
                        #################################################################################
                        bincenters_cmax_bg = 0.5*(bins_cmax_bg[1:]+bins_cmax_bg[:-1])
                        norm_cmax_bg = np.sum(counts_cmax_bg * np.diff(bins_cmax_bg))
                        #################################################################################
                        bincenters_cmax_sim0 = 0.5*(bins_cmax_sim0[1:]+bins_cmax_sim0[:-1])
                        norm_cmax_sim0 = np.sum(counts_cmax_sim0 * np.diff(bins_cmax_sim0))
                        #################################################################################
                        bincenters_cmax_sim_gauss = 0.5*(bins_cmax_sim_gauss[1:]+bins_cmax_sim_gauss[:-1])
                        norm_cmax_sim_gauss = np.sum(mean_counts_cmax_sim_gauss * np.diff(bins_cmax_sim_gauss))
                        #################################################################################

                        if(densi==False):
                            cmax_obs_w = counts_cmax_obs
                            err_cmax_obs_w = np.sqrt(counts_cmax_obs)
                            #################################################################################
                            cmax_bg_w = counts_cmax_bg
                            err_cmax_bg_w = np.sqrt(counts_cmax_bg)
                            #################################################################################
                            cmax_sim0_w = counts_cmax_sim0
                            if(errors_sqrt == True): err_cmax_sim0_w = np.sqrt(0.04*counts_cmax_sim0*counts_cmax_sim0 + counts_cmax_sim0)
                            else: err_cmax_sim0_w = (0.2*counts_cmax_sim0 + np.sqrt(counts_cmax_sim0))
                            #################################################################################
                            cmax_sim_gauss_w = mean_counts_cmax_sim_gauss
                            if(errors_sqrt == True): err_cmax_sim_gauss_w = np.sqrt(0.04*mean_counts_cmax_sim_gauss*mean_counts_cmax_sim_gauss + std_counts_cmax_sim_gauss*std_counts_cmax_sim_gauss + mean_counts_cmax_sim_gauss)
                            else: err_cmax_sim_gauss_w = (0.2*mean_counts_cmax_sim_gauss + std_counts_cmax_sim_gauss )
                            #################################################################################
                        if(densi==True):
                            cmax_obs_w = counts_cmax_obs/norm_cmax_obs
                            err_cmax_obs_w = np.sqrt(counts_cmax_obs)/norm_cmax_obs
                            #################################################################################
                            cmax_bg_w = counts_cmax_bg/norm_cmax_bg
                            err_cmax_bg_w = np.sqrt(counts_cmax_bg)/norm_cmax_bg
                            #################################################################################
                            cmax_sim0_w = counts_cmax_sim0/norm_cmax_sim0
                            if(errors_sqrt == True): err_cmax_sim0_w = np.sqrt(0.04*counts_cmax_sim0*counts_cmax_sim0 + counts_cmax_sim0)/norm_cmax_sim0
                            else: err_cmax_sim0_w = (0.2*counts_cmax_sim0 + np.sqrt(counts_cmax_sim0))/norm_cmax_sim0
                            #################################################################################
                            cmax_sim_gauss_w = mean_counts_cmax_sim_gauss/norm_cmax_sim_gauss
                            if(errors_sqrt == True): err_cmax_sim_gauss_w = np.sqrt(0.04*mean_counts_cmax_sim_gauss*mean_counts_cmax_sim_gauss + std_counts_cmax_sim_gauss*std_counts_cmax_sim_gauss + mean_counts_cmax_sim_gauss)/norm_cmax_sim_gauss
                            else: err_cmax_sim_gauss_w = (0.2*mean_counts_cmax_sim_gauss + std_counts_cmax_sim_gauss )/norm_cmax_sim_gauss
                            #################################################################################

                        err_max0 = np.sqrt(err_cmax_obs_w*err_cmax_obs_w + err_cmax_sim0_w*err_cmax_sim0_w )
                        ch2_max0 = chi_2_sigm_test(cmax_obs_w, cmax_sim0_w, err_max0)
                        tmp_str0 = 'Chi2 test: ' + r'%.2f'%ch2_max0[0] \
                                    +'       '+'ndf: ' + r'%.2f'%ch2_max0[3] \
                                    +'\n'+'Chi2/ndf : ' + r'%.2f'%ch2_max0[4]

                        err_max1 = np.sqrt(err_cmax_obs_w*err_cmax_obs_w + err_cmax_sim_gauss_w*err_cmax_sim_gauss_w )
                        ch2_max1 = chi_2_sigm_test(cmax_obs_w, cmax_sim_gauss_w, err_max1)
                        tmp_str1 = 'Chi2 test: ' + r'%.2f'%ch2_max1[0] \
                                    +'       '+'ndf: ' + r'%.2f'%ch2_max1[3] \
                                    +'\n'+'Chi2/ndf : ' + r'%.2f'%ch2_max1[4]

                        if(labl_opt=='simple'):
                            lbl_obs = source + ' data'
                            lbl_sim0 = source + ' simulation '
                            lbl_sim_gauss = source + ' simulation '
                            lbl_bg = 'Background' + ' data'
                        if(labl_opt=='chi2_nu'):
                            lbl_obs = source + ' data'
                            lbl_sim0 = source + ' sim ' +r'$\chi^2_\nu$ = ' + r'%.2f'%ch2_max0[4] #+'+ bg'
                            lbl_sim_gauss = source + ' sim ' + r'$\chi^2_\nu$ = '  + r'%.2f'%ch2_max1[4]#+'+ bg'
                            lbl_bg = 'Background' + ' data'
                        if(labl_opt=='off'):
                            lbl_obs = None
                            lbl_sim0 = None
                            lbl_sim_gauss = None
                            lbl_bg = None

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
                            #ax0.step( np.insert(bins_cmax_obs, 0, 0), np.concatenate(([0], cmax_obs_w, [0])), where='post'
                                        #, color='C3', markersize=10, linewidth=l_width+0.*l_width, label = lbl_obs + cut_max_clst_porc    )
                            ax0.set_yscale('log')   # Escala logarítmica en el eje y

                            if(plt_err==True):
                                ax0.errorbar(bincenters_cmax_obs, cmax_obs_w, yerr=err_cmax_obs_w,
                                fmt='C3'+'o', ecolor='C3', markersize=10, linewidth=l_width, label = lbl_obs + cut_max_clst_porc )
                        ################################################################################################################################

                        if(plt_sim==True):
                            if(plt_sim_nogauss==True):
                                #ax0.step( np.insert(bins_cmax_obs, 0, 0), np.concatenate(([0], cmax_sim0_w, [0])), where='post'
                                #        , color='C2', markersize=6, alpha=0.2, linewidth=l_width,  alpha=0.6,  label = lbl_sim0 + cut_max_clst_porc_sim)
                                ax0.fill_between(bincenters_cmax_sim0, cmax_sim0_w - err_cmax_sim0_w, cmax_sim0_w + err_cmax_sim0_w,
                                            color='C2', linewidth=l_width*2/3, alpha=0.6,  label = lbl_sim0 + cut_max_clst_porc_sim )
                                ax0.set_yscale('log')
                            if(plt_sim_gauss==True):
                                #ax0.step( np.insert(bins_cmax_obs, 0, 0), np.concatenate(([0], cmax_sim_gauss_w, [0])), where='post'
                                #        , color='C9', markersize=6, alpha=0.2, linewidth=l_width,  alpha=0.6,  label = lbl_sim_gauss + cut_max_clst_porc_sim)
                                ax0.fill_between(bincenters_cmax_sim_gauss, cmax_sim_gauss_w - err_cmax_sim_gauss_w, cmax_sim_gauss_w + err_cmax_sim_gauss_w,
                                            color='C9', linewidth=l_width*2/3, alpha=0.6,  label = lbl_sim_gauss + cut_max_clst_porc_sim )
                                ax0.set_yscale('log')

                        if(plt_sim==True and plt_err==True):
                            if(plt_sim_nogauss==True):
                                ax0.errorbar(bincenters_cmax_sim0, cmax_sim0_w, yerr=err_cmax_sim0_w, fmt='C2'+'o',
                                            ecolor='C2', markersize=6, alpha=0.2, linewidth=l_width*2/3 )
                            if(plt_sim_gauss==True):
                                ax0.errorbar(bincenters_cmax_sim_gauss, cmax_sim_gauss_w, yerr=err_cmax_sim_gauss_w, fmt='C9'+'o',
                                            ecolor='C9', markersize=6, alpha=0.2, linewidth=l_width*2/3 )

                        ################################################################################################################################
                        ################################################################################################################################

                        if(plt_bg==True):
                            #ax0.step( np.insert(bins_cmax_obs, 0, 0), np.concatenate(([0], cmax_bg_w, [0])), where='post'
                            #            , color='k', markersize=12, linewidth=l_width*2/3, label = lbl_bg )# + '_' + fecha_bg )
                            ax0.set_yscale('log')
                            if(plt_err==True):
                                ax0.errorbar(bincenters_cmax_bg, cmax_bg_w, yerr=err_cmax_bg_w, fmt='k'+'o',
                                                ecolor='k', markersize=12, linewidth=l_width*2/3, label = lbl_bg )#, label = 'distance: '+str(2*z)+'mm' )

                        #if(plt_bg==False):ax0.set_ylim(2, 7000 )
                        if(plt_bg==True):
                            y_min_cmax = np.min( [np.min( cmax_obs_w + err_cmax_obs_w ), np.min( cmax_sim0_w + err_cmax_sim0_w ), np.min( cmax_bg_w + err_cmax_bg_w )  ] )
                            y_min_cmax = 0.5*y_min_cmax

                        if(plt_bg==False):
                            y_min_cmax = np.min( [np.min( cmax_obs_w + err_cmax_obs_w ), np.min( cmax_sim0_w + err_cmax_sim0_w )  ] )
                            y_min_cmax = 0.5*y_min_cmax
                            y_max_cmax = np.min( [np.max( cmax_obs_w + err_cmax_obs_w ), np.max( cmax_sim0_w + err_cmax_sim0_w ) ] )
                            y_max_cmax = 1.5*y_max_cmax
                            ax0.set_ylim(y_min_cmax )


                        if(plt_obs==True):
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

                        ylabl_comp= r'$\frac{ (data -sim)}{\sigma}$'
                        comp_sim0, err_comp_sim0 = comp_obs_sim(cmax_obs_w, cmax_sim0_w)
                        comp_sim_gauss, err_comp_sim_gauss = comp_obs_sim(cmax_obs_w, cmax_sim_gauss_w)
                        delta_min = np.min(np.array((np.nanmin(comp_sim0), np.nanmin(comp_sim_gauss) )))
                        delta_max = np.max(np.array( (comp_sim0[comp_sim0 < np.Inf].max(), comp_sim_gauss[comp_sim_gauss < np.Inf].max() )))

                        if(plt_sim==True and plot_nsig==True):
                            #ax1.axhline(y = 0., xmin=0, xmax=max_adc, color = 'k', linestyle = '--', linewidth=l_width*2/3)
                            #ax1.step( np.insert(bins_cmax_obs, 0, 0), 0*np.concatenate(([0], cmax_obs_w, [0])), where='post', ls='--',  color='k', linewidth=l_width*2/3 )
                            if(plt_sim_nogauss==True):
                                ax1.errorbar(bincenters_cmax_obs, comp_sim0, yerr=plt_err*err_comp_sim0, fmt='C2'+'>', lw=2, markersize=10 )
                            if(plt_sim_gauss==True):
                                ax1.errorbar(bincenters_cmax_obs, comp_sim_gauss, yerr=plt_err*err_comp_sim_gauss, fmt='C9'+'>', lw=2, markersize=10 )
                            if(zoom_delta==True):ax1.set_ylim(delta_min,delta_max)

                        if( (plt_sim==True) and plot_nsig==True ):
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
                        ratio_sim0, err_ratio_sim0 = ratio_obs_sim(cmax_obs_w, cmax_sim0_w)
                        ratio_sim_gauss, err_ratio_sim_gauss = ratio_obs_sim(cmax_obs_w, cmax_sim_gauss_w)
                        ratio_min = np.min(np.array(( np.nanmin(ratio_sim0), np.nanmin(ratio_sim_gauss) )))
                        ratio_max = np.max(np.array(( ratio_sim0[ratio_sim0 < np.Inf].max(),  ratio_sim0[ratio_sim_gauss < np.Inf].max() )))

                        if(plt_sim==True and plot_ratio==True):
                            #ax2.axhline(y = 0., xmin=min_adc, xmax=max_adc, color = 'k', linestyle = '--')
                            #ax2.hist(cmax_obs_w*0, bins = bins_cmax_obs, histtype='step', ls='--',  color='k', linewidth=l_width*2/3 )
                            #ax2.step( np.insert(bins_cmax_obs, 0, 0), 0*np.concatenate(([0], cmax_obs_w, [0])), where='post', ls='--',  color='k', linewidth=l_width*2/3 )
                            if(plt_sim_nogauss==True):
                                ax2.errorbar(bincenters_cmax_obs, ratio_sim0, yerr=plt_err*err_ratio_sim0, fmt='C2'+'>', lw=2, markersize=10 )
                            if(plt_sim_gauss==True):
                                ax2.errorbar(bincenters_cmax_obs, ratio_sim_gauss, yerr=plt_err*err_ratio_sim_gauss, fmt='C9'+'>', lw=2, markersize=10 )
                            if(zoom_ratio==True):ax2.set_ylim(-0.5+ratio_min,ratio_max)

                        if( (plt_sim==True) and plot_ratio==True ):
                            #ax2[2].xticks(size=font_siz+2)
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

                        # Crear una lista de handles y etiquetas en el orden deseado
                        handles, labels = ax0.get_legend_handles_labels()
                        if(plt_bg==True): new_order = [1, 0, 2]  # Suponiendo que deseas invertir el orden
                        else: new_order = [1, 0]  # Suponiendo que deseas invertir el orden
                        handles = [handles[i] for i in new_order]
                        labels = [labels[i] for i in new_order]

                        # Crear la leyenda con el nuevo orden
                        ax0.legend(handles, labels)

                        fig.tight_layout()
                        name_cuts = source+sim_n+difu_gauss+ sig_bg_thresh + str_cuts + cut_clst_size+cut_max + str_hist_comp
                        namesave = 'plot_hist_'+histo+'cl_max_adc_'+strbin+str(z*2).zfill(2)+'mm'+ name_cuts
                        #plt.savefig(dirsave_plt_mean+'/'+namesave+'.png', dpi=150)
                        #plt.savefig(dirsave_plt_mean+'/'+namesave+'.pdf', dpi=150)
                        # plt.show()
                        #plt.clf()
                        #plt.close()

                        '''
                        if( file_csv==True):
                            file_activ_pixel = dirsave_plt_mean+'/'+source+dens+'_'+strbin+str(z_count)+'l_'+fecha+'_'+evt+'_'+str(z*2).zfill(2)+'mm'+'actv_pixel_chi2'+str_cuts+'.csv'
                            file_test = open(file_activ_pixel,'w')

                            txt_activ = 'data_'+source+'\t' + '0_ADC\t' + 'Active_Pixel\t' + 'all_pixel\t' + '%_0_ADC\t' + '%_Active_Pixel'
                            file_test.write(txt_activ + '\n')
                            print(txt_activ)
                            txt_activ = 'OBS'+'_'+str(nsigm_bg)+'sig'+':\t' + str(all_pixel_nframe - counts_adc_obs.sum()) + \
                                    '\t' + str(counts_adc_obs.sum()) +'\t' + str(all_pixel_nframe) + \
                                    '\t'+ str(100*(all_pixel_nframe - counts_adc_obs.sum())/all_pixel_nframe) + \
                                    '\t'+ str(100*counts_adc_obs.sum()/all_pixel_nframe)
                            file_test.write(txt_activ + '\n')
                            print(txt_activ)
                            txt_activ = 'SIM0'+'_'+str(nsigm_bg_sim)+'sig'+':\t' + str(all_pixel_nframe - counts_adc_sim0.sum()) + \
                                    '\t' + str(counts_adc_sim0.sum()) +'\t' + str(all_pixel_nframe) + \
                                    '\t'+ str(100*(all_pixel_nframe - counts_adc_sim0.sum())/all_pixel_nframe) + \
                                    '\t'+ str(100*counts_adc_sim0.sum()/all_pixel_nframe)
                            file_test.write(txt_activ + '\n')
                            print(txt_activ)
                            txt_activ = 'sim_gauss'+'_Dgauss_'+str(nsigm_bg_sim)+'sig'+':\t' + str(all_pixel_nframe - mean_counts_adc_sim_gauss.sum()) + \
                                    '\t' + str(mean_counts_adc_sim_gauss.sum()) +'\t' + str(all_pixel_nframe) + \
                                    '\t'+ str(100*(all_pixel_nframe - mean_counts_adc_sim_gauss.sum())/all_pixel_nframe) + \
                                    '\t'+ str(100*mean_counts_adc_sim_gauss.sum()/all_pixel_nframe)
                            file_test.write(txt_activ + '\n')
                            print(txt_activ)
                            file_test.write(txt_activ + '\n')

                            #file_test.close()

                            print('\n\n')

                            if(plt_sim_gauss==True):
                                txt_test = 'test bins'+sig_bg_thresh + str_cuts + cut_clst_size+cut_max + cut_adc + cut_max_clst_porc+sig_bg_thresh_sim+cut_max_clst_porc_sim+dens+'\t' + 'histo\t' + 'bins\t' + 'chi_2\t' + 'pvalue\t' + 'criti_val\t' + 'ndf\t' + 'chi2/nf'
                                file_test.write('\n\n'+ txt_test + '\n')
                                print(txt_test)

                                txt_test = 'Chi2_sig_test_sim_Dgauss' +'\t'+'hist_ADC'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%ch2_adc1[0]) +'\t'+ str(r'%.5f'%ch2_adc1[1])  +'\t'+ str(r'%.5f'%ch2_adc1[2])  +'\t'+ str(ch2_adc1[3]) +'\t'+ str(r'%.5f'%ch2_adc1[4])
                                file_test.write(txt_test + '\n')
                                print(txt_test)

                                txt_test = 'Chi2_sig_test_sim_Dgauss' +'\t'+'hist_Cluster_size'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%ch2_clstr_siz1[0]) +'\t'+ str(r'%.5f'%ch2_clstr_siz1[1])  +'\t'+ str(r'%.5f'%ch2_clstr_siz1[2])  +'\t'+ str(ch2_clstr_siz1[3]) +'\t'+ str(r'%.5f'%ch2_clstr_siz1[4])
                                file_test.write(txt_test + '\n')
                                print(txt_test)

                                txt_test = 'Chi2_sig_test_sim_Dgauss' +'\t'+'hist_Max_clst'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%ch2_max1[0]) +'\t'+ str(r'%.5f'%ch2_max1[1])  +'\t'+ str(r'%.5f'%ch2_max1[2])  +'\t'+ str(ch2_max1[3]) +'\t'+ str(r'%.5f'%ch2_max1[4])
                                file_test.write(txt_test + '\n')
                                print(txt_test)

                            if(plt_sim_nogauss==True and plt_sim_gauss==True):
                                txt_test = 'test bins'+sig_bg_thresh + str_cuts + cut_clst_size+cut_max + cut_adc + cut_max_clst_porc+sig_bg_thresh_sim+cut_max_clst_porc_sim+dens+'\t' + 'histo\t' + 'bins\t' + 'chi_2\t' + 'pvalue\t' + 'criti_val\t' + 'ndf\t' + 'chi2/nf'
                                file_test.write('\n\n'+ txt_test + '\n')
                                print(txt_test)

                                txt_test = 'Chi2_sig_test_sim' +'\t'+'hist_ADC'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%ch2_adc0[0]) +'\t'+ str(r'%.5f'%ch2_adc0[1])  +'\t'+ str(r'%.5f'%ch2_adc0[2])  +'\t'+ str(ch2_adc0[3]) +'\t'+ str(r'%.5f'%ch2_adc0[4])
                                file_test.write(txt_test + '\n')
                                print(txt_test)

                                txt_test = 'Chi2_sig_test_sim' +'\t'+'hist_Cluster_size'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%ch2_clstr_siz0[0]) +'\t'+ str(r'%.5f'%ch2_clstr_siz0[1])  +'\t'+ str(r'%.5f'%ch2_clstr_siz0[2])  +'\t'+ str(ch2_clstr_siz0[3]) +'\t'+ str(r'%.5f'%ch2_clstr_siz0[4])
                                file_test.write(txt_test + '\n')
                                print(txt_test)

                                txt_test = 'Chi2_sig_test_sim' +'\t'+'hist_Max_clst'+'\t'+ str(bin_hist) +'\t'+ str(r'%.5f'%ch2_max0[0]) +'\t'+ str(r'%.5f'%ch2_max0[1])  +'\t'+ str(r'%.5f'%ch2_max0[2])  +'\t'+ str(ch2_max0[3]) +'\t'+ str(r'%.5f'%ch2_max0[4])
                                file_test.write(txt_test + '\n')
                                print(txt_test)

                            file_test.close()
                        '''
