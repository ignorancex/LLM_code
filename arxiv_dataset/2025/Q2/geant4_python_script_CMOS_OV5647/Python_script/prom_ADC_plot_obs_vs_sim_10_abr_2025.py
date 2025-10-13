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
    dif = (obs - sim)
    err = np.sqrt(obs + 0.04*sim*sim )
    comp = np.divide(dif, err , where=(err != 0))
    err_comp = np.sqrt( 1
            +(dif*dif) *(obs + 0.000256*sim*sim*sim*sim) / (4*err*err*err*err*err*err) )
    #err_comp = np.nan_to_num(err_comp, nan=1)
    return comp, err_comp

def comp_2obs(obs1, obs2):
    dif = obs1 - obs2
    err = np.sqrt(obs1 + obs2)
    comp = np.divide(dif, err , where=(err != 0))
    err_comp = comp*np.sqrt( (err*err) / (dif*dif)
            + (1) / (4*err*err) )
    return comp, err_comp


def comp_2sim(sim_gauss1, sim2):
    dif = sim_gauss1 - sim2
    err = 0.2*np.sqrt(sim_gauss1*sim_gauss1 + sim2*sim2 )
    comp = np.divide(dif, err , where=(err != 0))
    err_comp = 0.2*comp*np.sqrt( (err*err) / (dif*dif)
            + (sim_gauss1*sim_gauss1*sim_gauss1*sim_gauss1 + sim2*sim2*sim2*sim2) / (4*err*err*err*err) )
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

def ratio_2sim(sim_gauss1, sim2):
    # Set ratio where obs2 is not zero
    ratio = np.divide(sim_gauss1, sim2 , where=(sim2 != 0))
    # Compute error on ratio (null if cannot be computed)
    er_ratio = ratio*np.sqrt( 0.08)

    return ratio, er_ratio

def comp_edep_obs1_sim(obs, err_obs1, sim, err_sim):
    dif = sim - obs
    err = np.sqrt(err_obs1*err_obs1 + err_sim*err_sim )
    # Set ratio where obs is not zero
    comp_edep = np.divide(dif, obs , where=(obs != 0))
    # Compute error on ratio (null if cannot be computed)
    err_comp = np.abs(comp_edep)*np.sqrt( (err*err) / (dif*dif)
            + err_obs1*err_obs1/ (obs*obs) )
    
    return comp_edep, err_comp


'''
comp_edep, err_comp_ede = comp_edep_obs1_sim(adc_obs1_w, err_adc_obs1_w, adc_sim1_w, err_adc_sim1_w)

comp_edep_gauss, err_comp_ede_gauss = comp_edep_obs1_sim(adc_obs1_w, err_adc_obs1_w, adc_sim_gauss1_w, err_adc_sim_gauss1_w)

comp_edep_min = np.min( np.array(( np.nanmin(comp_edep), np.nanmin(comp_edep_gauss) )))
comp_edep_max = np.max( np.array(( comp_edep[comp_edep < np.Inf].max(),  comp_edep_gauss[comp_edep_gauss < np.Inf].max() )))
plt.errorbar(bincenters_adc_obs1, comp_edep_gauss, yerr=0*err_comp_ede_gauss, fmt='C9'+'o', lw=2, markersize=10 )


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

plot_ratio = 0
zoom_ratio = 0

plt_sim = 1

plt_sim_gauss = 1
plt_sim_nogauss = 0

plt_obs = 1
plt_bg = 1
plt_err = 1


log_y = 1

label_xy = True

grid_ = False

file_csv = 1
file_clst_sum_csv = 1
file_total_adc_csv = 1

if(plt_sim_nogauss==False and plt_sim_gauss==True): row = 2 ; str_hist_comp = '_sim_gauss1'
if(plt_sim_nogauss==True and plt_sim_gauss==False): row = 2; str_hist_comp = '_Sim'
if(plt_sim_nogauss==True and plt_sim_gauss==True): row = 3; str_hist_comp = '_All'
if(plt_sim_nogauss==False and plt_sim_gauss==False): row = 1 ; str_hist_comp = '_obs1'; label_xy=False
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

source1 = 'Sr90'
source2 = 'Cs137'
########################################################################################

bin_hist = 40
bins_edep = 50
if(bin_hist>0): strbin = str(bin_hist)+'bin_z_'
else:strbin = 'z_'

scal_log = 0
#log_y = 1

max_adc=1024
#max_adc=150
min_adc=100


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
labl_opt = 'chi2_nu'

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


str_cuts= cut_adc_str

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

if(source1 == 'Sr90'):
    winback1='254'
    fecha1 = '2023_Oct_24_23h'
    ev1 = '1514'
    evt1 = winback1+'_ev'+ev1
    

if(source2 == 'Cs137'):
    winback2='635'
    fecha2 = '2023_Oct_23_23h'
    ev2= '3811'
    evt2 = winback2+'_ev'+ev2
    

nframes = nfrm


ssd_sim = 'C:/dat_2025/ADC_dat2025/'


if(adc_cut>0):
    cut_adc = '_cutsg_'+str(adc_cut)+'ADC'
if(adc_cut == 0 ):
    cut_adc = ''


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

                        sum_adc_sim_gauss1 = 0
                        sum_counts_adc_sim_gauss1 = 0
                        sum_2_adc_sim_gauss1 = 0
                        sum_2_counts_adc_sim_gauss1 = 0

                        sum_adc_sim_gauss2 = 0
                        sum_counts_adc_sim_gauss2 = 0
                        sum_2_adc_sim_gauss2 = 0
                        sum_2_counts_adc_sim_gauss2 = 0
                        

                        ############################################################

                        for sim_n in range(total_sim):
                            sim_n = '_s' + str(sim_n)

                            if nsigm > 0 :
                                difu_gauss = f"_{gauss_dist}_{k}k_{alpha}a_{nsigm}sig_{zff}zff"
                                d_gauss = gauss_dist+'_z_'+r'$\sigma$ = '+str(nsigm)+r', $\k$ = '+str(k)+r', $\alpha$ = '+str(alpha)+ r', $z_{ff}$ = '+str(zff)
                            if nsigm == 0 :
                                difu_gauss = f"_{gauss_dist}_{k}k_{alpha}a_{zff}zff"
                                d_gauss =gauss_dist+'_z_'+r'$\k$ = '+str(k)+r', $\alpha$ = '+str(alpha)+ r', $z_{ff}$ = '+str(zff)


                            path_bg = ssd_sim+'obs/'+source_bg+'/adc_count_'+source_bg+'_f'+str(nfrm)+'_iso0_br50_ag8_ad1_less_1avrg_bg_0sbg_tshld_5sig_bg/'+source_bg+'_less_1avrg_bg_0sbg_tshld_5sig_bg_z000.npz'
                          
                            path_obs1 = ssd_sim+'obs/'+source1+'/adc_count_'+source1+'_f'+str(nfrm)+'_iso0_br50_ag8_ad1_less_1avrg_bg_0sbg_tshld_5sig_bg/'+source1+'_less_1avrg_bg_0sbg_tshld_5sig_bg_z'+str(z*2).zfill(3)+'.npz'
                            path_obs2 = ssd_sim+'obs/'+source2+'/adc_count_'+source2+'_f'+str(nfrm)+'_iso0_br50_ag8_ad1_less_1avrg_bg_0sbg_tshld_5sig_bg/'+source2+'_less_1avrg_bg_0sbg_tshld_5sig_bg_z'+str(z*2).zfill(3)+'.npz'
                            
                              
                            path_sims1 = ssd_sim+'sim/'+source1+'/adc_count_sim_'+source1+'_'+str(nfrm)+'f_'+evt1
                            path_sim1 = '_eh_3.6eV'+'_'+gauss_dist+'_0k_0a_0zff'+'/'+source1+'_ev'+ev1+'_'+gauss_dist+'_0k_0a_0zff'+'_z'+str(z*2).zfill(3)+'.npz'
                            
                            path_sims2 = ssd_sim+'sim/'+source2+'/adc_count_sim_'+source2+'_'+str(nfrm)+'f_'+evt2                           
                            path_sim2 = '_eh_3.6eV'+'_'+gauss_dist+'_0k_0a_0zff'+'/'+source2+'_ev'+ev2+'_'+gauss_dist+'_0k_0a_0zff'+'_z'+str(z*2).zfill(3)+'.npz'

                            
                            #path_sim_gauss01 = ssd_sim+'sim/'+source1+'/adc_count_sim_'+source1+'_f'+str(nfrm)+'_'+evt1
                            path_sim_gauss1 = '_eh_3.6eV'+difu_gauss+'/'+source1+'_ev'+ev1+difu_gauss+'_z'+str(z*2).zfill(3)+'.npz'
                            
                            #path_sim_gauss02 = ssd_sim+'sim/'+source2+'/adc_count_sim_'+source2+'_f'+str(nfrm)+'_'+evt2
                            path_sim_gauss2 = '_eh_3.6eV'+difu_gauss+'/'+source2+'_ev'+ev2+difu_gauss+'_z'+str(z*2).zfill(3)+'.npz'

                            dirsave_plt_mean = ssd_sim+'plot_ADC_comp_ratio_sim_obs'
                            try:
                                os.makedirs(dirsave_plt_mean)
                            except FileExistsError:
                                pass                           
                            
                            ################################################################################################################
                            #################################################################################
                            if sim_n == '_s0' :
                                hist_adc_bg = np.load(path_bg)
                                hist_adc_bg = hist_adc_bg['arr_0']
                                adc_bg, counts_adc_bg = hist_adc_bg[0], hist_adc_bg[1]

                                hist_adc_obs1 = np.load(path_obs1)
                                hist_adc_obs1 = hist_adc_obs1['arr_0']
                                adc_obs1, counts_adc_obs1 = hist_adc_obs1[0], hist_adc_obs1[1]

                                hist_adc_obs2 = np.load(path_obs2)
                                hist_adc_obs2 = hist_adc_obs2['arr_0']
                                adc_obs2, counts_adc_obs2 = hist_adc_obs2[0], hist_adc_obs2[1]
                                

                                hist_adc_sim1 = np.load(path_sims1+sim_n+path_sim1)
                                hist_adc_sim1 = hist_adc_sim1['arr_0']
                                adc_sim1, counts_adc_sim1 = hist_adc_sim1[0], hist_adc_sim1[1]

                                hist_adc_sim2 = np.load(path_sims2+sim_n+path_sim2)
                                hist_adc_sim2 = hist_adc_sim2['arr_0']
                                adc_sim2, counts_adc_sim2 = hist_adc_sim2[0], hist_adc_sim2[1]

                            print(f"Procesando simulación: _s{sim_n}")
                            try:

                                hist_adc_sim_gauss1 = np.load(path_sims1+sim_n+path_sim_gauss1)
                                hist_adc_sim_gauss1 = hist_adc_sim_gauss1['arr_0']
                                adc_sim_gauss1, counts_adc_sim_gauss1 = hist_adc_sim_gauss1[0], hist_adc_sim_gauss1[1]
               
                                hist_adc_sim_gauss2 = np.load(path_sims2+sim_n+path_sim_gauss2)
                                hist_adc_sim_gauss2 = hist_adc_sim_gauss2['arr_0']
                                adc_sim_gauss2, counts_adc_sim_gauss2 = hist_adc_sim_gauss2[0], hist_adc_sim_gauss2[1]

                                #################################################################################
                                # Acumular datos
                                sum_adc_sim_gauss1 += adc_sim_gauss1
                                sum_counts_adc_sim_gauss1 += counts_adc_sim_gauss1
                                sum_2_adc_sim_gauss1 += adc_sim_gauss1 ** 2
                                sum_2_counts_adc_sim_gauss1 += counts_adc_sim_gauss1 ** 2

                                sum_adc_sim_gauss2 += adc_sim_gauss2
                                sum_counts_adc_sim_gauss2 += counts_adc_sim_gauss2
                                sum_2_adc_sim_gauss2 += adc_sim_gauss2 ** 2
                                sum_2_counts_adc_sim_gauss2 += counts_adc_sim_gauss2 ** 2

                                #################################################################################
                                #################################################################################

                            except Exception as e:
                                print(f"Error procesando simulación {sim_n}: {e}")
                                continue

                        # Calcular promedios y desviaciones estándar
                        mean_adc_sim_gauss1 = sum_adc_sim_gauss1 / total_sim

                        mean_counts_adc_sim_gauss1 = sum_counts_adc_sim_gauss1 / total_sim
                        std_counts_adc_sim_gauss1 = np.sqrt((np.abs(sum_2_counts_adc_sim_gauss1 - total_sim * mean_counts_adc_sim_gauss1 ** 2)) / n_sampl )
                        mean_adc_sim_gauss1 = mean_adc_sim_gauss1.astype(int)
                        
                        #################################################################################
                        mean_adc_sim_gauss2 = sum_adc_sim_gauss2 / total_sim

                        mean_counts_adc_sim_gauss2 = sum_counts_adc_sim_gauss2 / total_sim
                        std_counts_adc_sim_gauss2 = np.sqrt(np.abs((sum_2_counts_adc_sim_gauss2 - total_sim * mean_counts_adc_sim_gauss2 ** 2)) / n_sampl )
                        mean_adc_sim_gauss2 = mean_adc_sim_gauss2.astype(int)
                        #################################################################################
                        #################################################################################
                        counts_bg, bins_adc_bg = np.histogram(adc_bg, weights = counts_adc_bg, bins = bin_hist)
                        counts_obs1, bins_adc_obs1 = np.histogram(adc_obs1, weights = counts_adc_obs1, bins = bin_hist)
                        counts_obs2, bins_adc_obs2 = np.histogram(adc_obs2, weights = counts_adc_obs2, bins = bin_hist)
                        counts_sim1, bins_adc_sim1 = np.histogram(adc_sim1, weights = counts_adc_sim1, bins = bin_hist)
                        counts_sim2, bins_adc_sim2 = np.histogram(adc_sim2, weights = counts_adc_sim2, bins = bin_hist)
                        counts_sim_gauss1, bins_adc_sim_gauss1 = np.histogram(mean_adc_sim_gauss1, weights = mean_counts_adc_sim_gauss1, bins = bin_hist)
                        counts_sim_gauss2, bins_adc_sim_gauss2 = np.histogram(mean_adc_sim_gauss2, weights = mean_counts_adc_sim_gauss2, bins = bin_hist)
             
                        std_counts_sim_gauss1, bins_std_adc_sim_gauss1 = np.histogram(mean_adc_sim_gauss1, weights = std_counts_adc_sim_gauss2, bins = bin_hist)
                        std_counts_sim_gauss2, bins_std_adc_sim_gauss2 = np.histogram(mean_adc_sim_gauss2, weights = std_counts_adc_sim_gauss2, bins = bin_hist)
                
                        # Imprimir resultados
                        '''
                        print(f"Promedio de bins adc_sim_gauss1: {mean_adc_sim_gauss1}\n, Desviación estándar adc_sim_gauss1: {std_adc_sim_gauss1}")
                        print(f"Promedio de counts adc_sim_gauss1: {mean_counts_adc_sim_gauss1}\n, Desviación estándar adc_sim_gauss1: {std_counts_adc_sim_gauss1}")

                        print(f"Promedio de bins cmax_sim_gauss1: {mean_bins_cmax_sim_gauss1}\n, Desviación estándar cmax_sim_gauss1: {std_bins_cmax_sim_gauss1}")
                        print(f"Promedio de counts cmax_sim_gauss1: {mean_counts_cmax_sim_gauss1}\n, Desviación estándar cmax_sim_gauss1: {std_counts_cmax_sim_gauss1}")
                        '''
                        
                        # ADC histos
                        #################################################################################
                        bincenters_adc_obs1 = 0.5*(bins_adc_obs1[1:]+bins_adc_obs1[:-1])
                        norm_adc_obs1 = np.sum(counts_obs1 * np.diff(bins_adc_obs1))
                        #################################################################################
                        bincenters_adc_bg = 0.5*(bins_adc_bg[1:]+bins_adc_bg[:-1])
                        norm_adc_bg = np.sum(counts_bg * np.diff(bins_adc_bg))
                        #################################################################################
                        bincenters_adc_sim1 = 0.5*(bins_adc_sim1[1:]+bins_adc_sim1[:-1])
                        norm_adc_sim1 = np.sum(counts_sim1 * np.diff(bins_adc_sim1))
                        #################################################################################
                        bincenters_adc_sim_gauss1 = 0.5*(bins_adc_sim_gauss1[1:]+bins_adc_sim_gauss1[:-1])
                        norm_adc_sim_gauss1 = np.sum(counts_sim_gauss1 * np.diff(bins_adc_sim_gauss1))
                        #################################################################################
                        
                        if(densi==False):
                            adc_obs1_w = counts_obs1
                            err_adc_obs1_w = np.sqrt(counts_obs1)
                            #################################################################################
                            adc_bg_w = counts_bg
                            err_adc_bg_w = np.sqrt(counts_bg)
                            #################################################################################
                            adc_sim1_w = counts_sim1
                            if(errors_sqrt == True): err_adc_sim1_w = np.sqrt(0.04*counts_sim1*counts_sim1 + counts_sim1)
                            else: err_adc_sim1_w = (0.2*counts_sim1 + np.sqrt(counts_sim1))
                            #################################################################################
                            adc_sim_gauss1_w = counts_sim_gauss1
                            if(errors_sqrt == True): err_adc_sim_gauss1_w = np.sqrt(0.04*counts_sim_gauss1*counts_sim_gauss1 + std_counts_sim_gauss1*std_counts_sim_gauss1 + counts_sim_gauss1)
                            else: err_adc_sim_gauss1_w = (0.2*counts_sim_gauss1 + std_counts_sim_gauss1 )
                            #################################################################################
                        if(densi==True):
                            adc_obs1_w = counts_obs1/norm_adc_obs1
                            err_adc_obs1_w = np.sqrt(counts_obs1)/norm_adc_obs1
                            #################################################################################
                            adc_bg_w = counts_bg/norm_adc_bg
                            err_adc_bg_w = np.sqrt(counts_bg)/norm_adc_bg
                            #################################################################################
                            adc_sim1_w = counts_sim1/norm_adc_sim1
                            if(errors_sqrt == True): err_adc_sim1_w = np.sqrt(0.04*counts_sim1*counts_sim1 + counts_sim1)/norm_adc_sim1
                            else: err_adc_sim1_w = (0.2*counts_sim1 + np.sqrt(counts_sim1))/norm_adc_sim1
                            #################################################################################
                            adc_sim_gauss1_w = counts_sim_gauss1/norm_adc_sim_gauss1
                            if(errors_sqrt == True): err_adc_sim_gauss1_w = np.sqrt(0.04*counts_sim_gauss1*counts_sim_gauss1 + std_counts_sim_gauss1*std_counts_sim_gauss1 + counts_sim_gauss1)/norm_adc_sim_gauss1
                            else: err_adc_sim_gauss1_w = (0.2*counts_sim_gauss1 + std_counts_sim_gauss1 )/norm_adc_sim_gauss1
                            #################################################################################

                        err_adc_os1 = np.sqrt(err_adc_obs1_w*err_adc_obs1_w + err_adc_sim1_w*err_adc_sim1_w )
                        ch2_adc_os1 = chi_2_sigm_test(adc_obs1_w, adc_sim1_w, err_adc_os1)
                        tmp_str_os1 = 'Chi2 test: ' + r'%.2f'%ch2_adc_os1[0] \
                                    +'       '+'ndf: ' + r'%.2f'%ch2_adc_os1[3] \
                                    +'\n'+'Chi2/ndf : ' + r'%.2f'%ch2_adc_os1[4]

                        err_adc_osg1 = np.sqrt(err_adc_obs1_w*err_adc_obs1_w + err_adc_sim_gauss1_w*err_adc_sim_gauss1_w )
                        ch2_adc_osg1 = chi_2_sigm_test(adc_obs1_w, adc_sim_gauss1_w, err_adc_osg1)
                        tmp_str_osg1 = 'Chi2 test: ' + r'%.2f'%ch2_adc_osg1[0] \
                                    +'       '+'ndf: ' + r'%.2f'%ch2_adc_osg1[3] \
                                    +'\n'+'Chi2/ndf : ' + r'%.2f'%ch2_adc_osg1[4]

                        if(labl_opt=='simple'):
                            lbl_obs1 = source1 + ' data'
                            lbl_sim1 = source1 + ' simulation '
                            lbl_sim_gauss1 = source1 + ' simulation '
                            lbl_bg = 'Background' + ' data'
                        if(labl_opt=='chi2_nu'):
                            lbl_obs1 = source1 + ' data'
                            lbl_sim1 = source1 + ' sim ' +r'$\chi^2_\nu$ = ' + r'%.2f'%ch2_adc_os1[4] #+'+ bg'
                            lbl_sim_gauss1 = source1 + ' sim ' + r'$\chi^2_\nu$ = '  + r'%.2f'%ch2_adc_osg1[4]#+'+ bg'
                            lbl_bg = 'Background' + ' data'
                        if(labl_opt=='off'):
                            lbl_obs1 = None
                            lbl_sim1 = None
                            lbl_sim_gauss1 = None
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
                            #ax0.step( np.insert(bins_adc_obs1, 0, 0), np.concatenate(([0], adc_obs1_w, [0])), where='post'
                            #            , color='C3', markersize=10, linewidth=l_width+0.*l_width, label = lbl_obs1 + cut_max_clst_porc )
                            ax0.set_yscale('log')   # Escala logarítmica en el eje y

                            if(plt_err==True):
                                ax0.errorbar(bincenters_adc_obs1, adc_obs1_w, yerr=err_adc_obs1_w,
                                fmt='C3'+'o', ecolor='C3', markersize=10, linewidth=l_width,  label = lbl_obs1  )
                        ################################################################################################################################

                        if(plt_sim==True):
                            if(plt_sim_nogauss==True):
                                #ax0.step( np.insert(bins_adc_obs1, 0, 0), np.concatenate(([0], adc_sim1_w, [0])), where='post',
                                        # color='C2', markersize=6, alpha=0.2, linewidth=l_width, alpha=0.6,  label = lbl_sim1 + cut_max_clst_porc_sim )
                                ax0.fill_between(bincenters_adc_sim1, adc_sim1_w - err_adc_sim1_w, adc_sim1_w + err_adc_sim1_w,
                                            color='C2', linewidth=l_width*2/3, alpha=0.8,  label = lbl_sim1  )
                                ax0.set_yscale('log')
                            if(plt_sim_gauss==True):
                                #ax0.step( np.insert(bins_adc_obs1, 0, 0), np.concatenate(([0], adc_sim_gauss1_w, [0])), where='post' ,
                                        # color='C9', markersize=6, alpha=0.2, linewidth=l_width, alpha=0.6,  label = lbl_sim_gauss1 + cut_max_clst_porc_sim)
                                ax0.fill_between(bincenters_adc_sim_gauss1, adc_sim_gauss1_w - err_adc_sim_gauss1_w, adc_sim_gauss1_w + err_adc_sim_gauss1_w,
                                            color='C9', linewidth=l_width*2/3, alpha=0.8,  label = lbl_sim_gauss1  )
                                ax0.set_yscale('log')

                        if(plt_sim==True and plt_err==True):
                            if(plt_sim_nogauss==True):
                                ax0.errorbar(bincenters_adc_sim1, adc_sim1_w, yerr=err_adc_sim1_w, fmt='C2'+'o',
                                            ecolor='C2', markersize=6, alpha=0.2, linewidth=l_width*2/3 )
                            if(plt_sim_gauss==True):
                                ax0.errorbar(bincenters_adc_sim_gauss1, adc_sim_gauss1_w, yerr=err_adc_sim_gauss1_w, fmt='C9'+'o',
                                            ecolor='C9', markersize=6, alpha=0.2, linewidth=l_width*2/3 )

                        ################################################################################################################################
                        ################################################################################################################################
                        if(plt_bg==True):
                            #ax0.step( np.insert(bins_adc_obs1, 0, 0), np.concatenate(([0], adc_bg_w, [0])), where='post'
                            #            , color='k', markersize=12, linewidth=l_width*2/3,  label = lbl_bg )# + '_' + fecha_bg )
                            ax0.set_yscale('log')
                            if(plt_err==True):
                                ax0.errorbar(bincenters_adc_bg, adc_bg_w, yerr=err_adc_bg_w, fmt='k'+'o',
                                                ecolor='k', markersize=12, linewidth=l_width*2/3,  label = lbl_bg   )#, label = 'distance: '+str(2*z)+'mm' )

                        if(plt_bg==True):
                            y_min_adc = np.min( [np.min( adc_obs1_w + err_adc_obs1_w ), np.min( adc_sim1_w + err_adc_sim1_w ), np.min( adc_bg_w + err_adc_bg_w )  ] )
                            y_min_adc = 0.5*y_min_adc
                        if(plt_bg==False):
                            y_min_adc = np.min( [np.min( adc_obs1_w + err_adc_obs1_w ), np.min( adc_sim1_w + err_adc_sim1_w )  ] )
                            y_min_adc = 0.5*y_min_adc
                            y_max_adc = np.min( [np.max( adc_obs1_w + err_adc_obs1_w ), np.max( adc_sim1_w + err_adc_sim1_w ) ] )
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
                        comp_sim1, err_comp_sim1 = comp_obs_sim(adc_obs1_w, adc_sim1_w)

                        comp_sim_gauss1, err_comp_sim_gauss1 = comp_obs_sim(adc_obs1_w, adc_sim_gauss1_w)

                        delta_min = np.min(np.array(( np.nanmin(comp_sim1), np.nanmin(comp_sim_gauss1) )))
                        delta_max = np.max(np.array(( comp_sim1[comp_sim1 < np.Inf].max(), comp_sim_gauss1[comp_sim_gauss1 < np.Inf].max() )))

                        if(plt_sim==True and plot_nsig==True):
                            #ax1.axhline(y = 0., xmin=0, xmax=max_adc, color = 'k', linestyle = '--', linewidth=l_width*2/3)
                            #ax1.step( np.insert(bins_adc_obs1, 0, 0), 0*np.concatenate(([0], adc_obs1_w, [0])), where='post', ls='--',  color='k', linewidth=l_width*2/3 )
                            if(plt_sim_nogauss==True):
                                ax1.errorbar(bincenters_adc_obs1, comp_sim1, yerr=plt_err*err_comp_sim1, fmt='C2'+'>', lw=2, markersize=10 )
                            if(plt_sim_gauss==True):
                                ax1.errorbar(bincenters_adc_obs1, comp_sim_gauss1, yerr=plt_err*err_comp_sim_gauss1, fmt='C9'+'>', lw=2, markersize=10 )
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
                        ratio_sim1, err_ratio_sim1 = ratio_obs_sim(adc_obs1_w, adc_sim1_w)
                        ratio_sim_gauss1, err_ratio_sim_gauss1 = ratio_obs_sim(adc_obs1_w, adc_sim_gauss1_w)
                        ratio_min = np.min( np.array(( np.nanmin(ratio_sim1), np.nanmin(ratio_sim_gauss1) )))
                        ratio_max = np.max( np.array(( ratio_sim1[ratio_sim1 < np.Inf].max(),  ratio_sim_gauss1[ratio_sim_gauss1 < np.Inf].max() )))

                        if(plt_sim==True and plot_ratio==True):
                            #ax2.axhline(y = 0., xmin=min_adc, xmax=max_adc, color = 'k', linestyle = '--')
                            #ax2.hist(adc_obs1_w*0, bins = bins_adc_obs1, histtype='step', ls='--',  color='k', linewidth=l_width*2/3 )
                            #ax2.step( np.insert(bins_adc_obs1, 0, 0), 0*np.concatenate(([0], adc_obs1_w, [0])), where='post', ls='--',  color='k', linewidth=l_width*2/3 )
                            if(plt_sim_nogauss==True):
                                ax2.errorbar(bincenters_adc_obs1, ratio_sim1, yerr=plt_err*err_ratio_sim1, fmt='C2'+'>', lw=2, markersize=10 )
                            if(plt_sim_gauss==True):
                                ax2.errorbar(bincenters_adc_obs1, ratio_sim_gauss1, yerr=plt_err*err_ratio_sim_gauss1, fmt='C9'+'>', lw=2, markersize=10 )
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

                        adc2ekv=(bins_adc_obs1*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
                        ax3.plot(adc2ekv, bins_adc_obs1*0., color='w', linewidth=0.001 )
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
                        name_cuts = source1+sim_n+difu_gauss + str_cuts + str_hist_comp
                        namesave = 'plot_hist_'+histo+'ADC_'+strbin+str(z*2).zfill(2)+'mm'+ name_cuts
                        #plt.savefig(dirsave_plt_mean+'/'+namesave+'.png', dpi=150)
                        #plt.savefig(dirsave_plt_mean+'/'+namesave+'.pdf', dpi=150)
                        plt.show()
                        #plt.clf()
                        #plt.close()


                        '''
                        # ADC histos
                        #################################################################################
                        bincenters_adc_obs2 = 0.5*(bins_adc_obs2[1:]+bins_adc_obs2[:-1])
                        norm_adc_obs2 = np.sum(counts_adc_obs2 * np.diff(bins_adc_obs2))
                        #################################################################################
                        bincenters_adc_bg = 0.5*(bins_adc_bg[1:]+bins_adc_bg[:-1])
                        norm_adc_bg = np.sum(counts_adc_bg * np.diff(bins_adc_bg))
                        #################################################################################
                        bincenters_adc_sim2 = 0.5*(bins_adc_sim2[1:]+bins_adc_sim2[:-1])
                        norm_adc_sim2 = np.sum(counts_adc_sim2 * np.diff(bins_adc_sim2))
                        #################################################################################
                        bincenters_adc_sim_gauss2 = 0.5*(bins_adc_sim_gauss2[1:]+bins_adc_sim_gauss2[:-1])
                        norm_adc_sim_gauss2 = np.sum(mean_counts_adc_sim_gauss2 * np.diff(bins_adc_sim_gauss2))
                        #################################################################################

                        if(densi==False):
                            adc_obs2_w = counts_adc_obs2
                            err_adc_obs2_w = np.sqrt(counts_adc_obs2)
                            #################################################################################
                            adc_bg_w = counts_adc_bg
                            err_adc_bg_w = np.sqrt(counts_adc_bg)
                            #################################################################################
                            adc_sim2_w = counts_adc_sim2
                            if(errors_sqrt == True): err_adc_sim2_w = np.sqrt(0.04*counts_adc_sim2*counts_adc_sim2 + counts_adc_sim2)
                            else: err_adc_sim2_w = (0.2*counts_adc_sim2 + np.sqrt(counts_adc_sim2))
                            #################################################################################
                            adc_sim_gauss2_w = mean_counts_adc_sim_gauss2
                            if(errors_sqrt == True): err_adc_sim_gauss2_w = np.sqrt(0.04*mean_counts_adc_sim_gauss2*mean_counts_adc_sim_gauss2 + std_counts_adc_sim_gauss2*std_counts_adc_sim_gauss2 + mean_counts_adc_sim_gauss2)
                            else: err_adc_sim_gauss2_w = (0.2*mean_counts_adc_sim_gauss2 + std_counts_adc_sim_gauss2 )
                            #################################################################################
                        if(densi==True):
                            adc_obs2_w = counts_adc_obs2/norm_adc_obs2
                            err_adc_obs2_w = np.sqrt(counts_adc_obs2)/norm_adc_obs2
                            #################################################################################
                            adc_bg_w = counts_adc_bg/norm_adc_bg
                            err_adc_bg_w = np.sqrt(counts_adc_bg)/norm_adc_bg
                            #################################################################################
                            adc_sim2_w = counts_adc_sim2/norm_adc_sim2
                            if(errors_sqrt == True): err_adc_sim2_w = np.sqrt(0.04*counts_adc_sim2*counts_adc_sim2 + counts_adc_sim2)/norm_adc_sim2
                            else: err_adc_sim2_w = (0.2*counts_adc_sim2 + np.sqrt(counts_adc_sim2))/norm_adc_sim2
                            #################################################################################
                            adc_sim_gauss2_w = mean_counts_adc_sim_gauss2/norm_adc_sim_gauss2
                            if(errors_sqrt == True): err_adc_sim_gauss2_w = np.sqrt(0.04*mean_counts_adc_sim_gauss2*mean_counts_adc_sim_gauss2 + std_counts_adc_sim_gauss2*std_counts_adc_sim_gauss2 + mean_counts_adc_sim_gauss2)/norm_adc_sim_gauss2
                            else: err_adc_sim_gauss2_w = (0.2*mean_counts_adc_sim_gauss2 + std_counts_adc_sim_gauss2 )/norm_adc_sim_gauss2
                            #################################################################################

                        err_adc_os2 = np.sqrt(err_adc_obs2_w*err_adc_obs2_w + err_adc_sim2_w*err_adc_sim2_w )
                        ch2_adc_os2 = chi_2_sigm_test(adc_obs2_w, adc_sim2_w, err_adc_os2)
                        tmp_str_os2 = 'Chi2 test: ' + r'%.2f'%ch2_adc_os2[0] \
                                    +'       '+'ndf: ' + r'%.2f'%ch2_adc_os2[3] \
                                    +'\n'+'Chi2/ndf : ' + r'%.2f'%ch2_adc_os2[4]

                        err_adc_osg2 = np.sqrt(err_adc_obs2_w*err_adc_obs2_w + err_adc_sim_gauss2_w*err_adc_sim_gauss2_w )
                        ch2_adc_osg2 = chi_2_sigm_test(adc_obs2_w, adc_sim_gauss2_w, err_adc_osg2)
                        tmp_str_osg2 = 'Chi2 test: ' + r'%.2f'%ch2_adc_osg2[0] \
                                    +'       '+'ndf: ' + r'%.2f'%ch2_adc_osg2[3] \
                                    +'\n'+'Chi2/ndf : ' + r'%.2f'%ch2_adc_osg2[4]

                        if(labl_opt=='simple'):
                            lbl_obs2 = source2 + ' data'
                            lbl_sim2 = source2 + ' simulation '
                            lbl_sim_gauss2 = source2 + ' simulation '
                            lbl_bg = 'Background' + ' data'
                        if(labl_opt=='chi2_nu'):
                            lbl_obs2 = source2 + ' data'
                            lbl_sim2 = source2 + ' sim ' +r'$\chi^2_\nu$ = ' + r'%.2f'%ch2_adc_os2[4] #+'+ bg'
                            lbl_sim_gauss2 = source2 + ' sim ' + r'$\chi^2_\nu$ = '  + r'%.2f'%ch2_adc_osg2[4]#+'+ bg'
                            lbl_bg = 'Background' + ' data'
                        if(labl_opt=='off'):
                            lbl_obs2 = None
                            lbl_sim2 = None
                            lbl_sim_gauss2 = None
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
                            #ax0.step( np.insert(bins_adc_obs2, 0, 0), np.concatenate(([0], adc_obs2_w, [0])), where='post'
                            #            , color='C3', markersize=10, linewidth=l_width+0.*l_width, label = lbl_obs2 + cut_max_clst_porc )
                            ax0.set_yscale('log')   # Escala logarítmica en el eje y

                            if(plt_err==True):
                                ax0.errorbar(bincenters_adc_obs2, adc_obs2_w, yerr=err_adc_obs2_w,
                                fmt='C3'+'o', ecolor='C3', markersize=10, linewidth=l_width,  label = lbl_obs2 + cut_max_clst_porc )
                        ################################################################################################################################

                        if(plt_sim==True):
                            if(plt_sim_nogauss==True):
                                #ax0.step( np.insert(bins_adc_obs2, 0, 0), np.concatenate(([0], adc_sim2_w, [0])), where='post',
                                        # color='C2', markersize=6, alpha=0.2, linewidth=l_width, alpha=0.6,  label = lbl_sim2 + cut_max_clst_porc_sim )
                                ax0.fill_between(bincenters_adc_sim2, adc_sim2_w - err_adc_sim2_w, adc_sim2_w + err_adc_sim2_w,
                                            color='C2', linewidth=l_width*2/3, alpha=0.8,  label = lbl_sim2 + cut_max_clst_porc_sim )
                                ax0.set_yscale('log')
                            if(plt_sim_gauss==True):
                                #ax0.step( np.insert(bins_adc_obs2, 0, 0), np.concatenate(([0], adc_sim_gauss2_w, [0])), where='post' ,
                                        # color='C9', markersize=6, alpha=0.2, linewidth=l_width, alpha=0.6,  label = lbl_sim_gauss2 + cut_max_clst_porc_sim)
                                ax0.fill_between(bincenters_adc_sim_gauss2, adc_sim_gauss2_w - err_adc_sim_gauss2_w, adc_sim_gauss2_w + err_adc_sim_gauss2_w,
                                            color='C9', linewidth=l_width*2/3, alpha=0.8,  label = lbl_sim_gauss2 + cut_max_clst_porc_sim )
                                ax0.set_yscale('log')

                        if(plt_sim==True and plt_err==True):
                            if(plt_sim_nogauss==True):
                                ax0.errorbar(bincenters_adc_sim2, adc_sim2_w, yerr=err_adc_sim2_w, fmt='C2'+'o',
                                            ecolor='C2', markersize=6, alpha=0.2, linewidth=l_width*2/3 )
                            if(plt_sim_gauss==True):
                                ax0.errorbar(bincenters_adc_sim_gauss2, adc_sim_gauss2_w, yerr=err_adc_sim_gauss2_w, fmt='C9'+'o',
                                            ecolor='C9', markersize=6, alpha=0.2, linewidth=l_width*2/3 )

                        ################################################################################################################################
                        ################################################################################################################################
                        if(plt_bg==True):
                            #ax0.step( np.insert(bins_adc_obs2, 0, 0), np.concatenate(([0], adc_bg_w, [0])), where='post'
                            #            , color='k', markersize=12, linewidth=l_width*2/3,  label = lbl_bg )# + '_' + fecha_bg )
                            ax0.set_yscale('log')
                            if(plt_err==True):
                                ax0.errorbar(bincenters_adc_bg, adc_bg_w, yerr=err_adc_bg_w, fmt='k'+'o',
                                                ecolor='k', markersize=12, linewidth=l_width*2/3,  label = lbl_bg + cut_max_clst_porc  )#, label = 'distance: '+str(2*z)+'mm' )

                        if(plt_bg==True):
                            y_min_adc = np.min( [np.min( adc_obs2_w + err_adc_obs2_w ), np.min( adc_sim2_w + err_adc_sim2_w ), np.min( adc_bg_w + err_adc_bg_w )  ] )
                            y_min_adc = 0.5*y_min_adc
                        if(plt_bg==False):
                            y_min_adc = np.min( [np.min( adc_obs2_w + err_adc_obs2_w ), np.min( adc_sim2_w + err_adc_sim2_w )  ] )
                            y_min_adc = 0.5*y_min_adc
                            y_max_adc = np.min( [np.max( adc_obs2_w + err_adc_obs2_w ), np.max( adc_sim2_w + err_adc_sim2_w ) ] )
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
                        comp_sim2, err_comp_sim2 = comp_obs_sim(adc_obs2_w, adc_sim2_w)

                        comp_sim_gauss2, err_comp_sim_gauss2 = comp_obs_sim(adc_obs2_w, adc_sim_gauss2_w)

                        delta_min = np.min(np.array(( np.nanmin(comp_sim2), np.nanmin(comp_sim_gauss2) )))
                        delta_max = np.max(np.array(( comp_sim2[comp_sim2 < np.Inf].max(), comp_sim_gauss2[comp_sim_gauss2 < np.Inf].max() )))

                        if(plt_sim==True and plot_nsig==True):
                            #ax1.axhline(y = 0., xmin=0, xmax=max_adc, color = 'k', linestyle = '--', linewidth=l_width*2/3)
                            #ax1.step( np.insert(bins_adc_obs2, 0, 0), 0*np.concatenate(([0], adc_obs2_w, [0])), where='post', ls='--',  color='k', linewidth=l_width*2/3 )
                            if(plt_sim_nogauss==True):
                                ax1.errorbar(bincenters_adc_obs2, comp_sim2, yerr=plt_err*err_comp_sim2, fmt='C2'+'>', lw=2, markersize=10 )
                            if(plt_sim_gauss==True):
                                ax1.errorbar(bincenters_adc_obs2, comp_sim_gauss2, yerr=plt_err*err_comp_sim_gauss2, fmt='C9'+'>', lw=2, markersize=10 )
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
                        ratio_sim2, err_ratio_sim2 = ratio_obs_sim(adc_obs2_w, adc_sim2_w)
                        ratio_sim_gauss2, err_ratio_sim_gauss2 = ratio_obs_sim(adc_obs2_w, adc_sim_gauss2_w)
                        ratio_min = np.min( np.array(( np.nanmin(ratio_sim2), np.nanmin(ratio_sim_gauss2) )))
                        ratio_max = np.max( np.array(( ratio_sim2[ratio_sim2 < np.Inf].max(),  ratio_sim_gauss2[ratio_sim_gauss2 < np.Inf].max() )))

                        if(plt_sim==True and plot_ratio==True):
                            #ax2.axhline(y = 0., xmin=min_adc, xmax=max_adc, color = 'k', linestyle = '--')
                            #ax2.hist(adc_obs2_w*0, bins = bins_adc_obs2, histtype='step', ls='--',  color='k', linewidth=l_width*2/3 )
                            #ax2.step( np.insert(bins_adc_obs2, 0, 0), 0*np.concatenate(([0], adc_obs2_w, [0])), where='post', ls='--',  color='k', linewidth=l_width*2/3 )
                            if(plt_sim_nogauss==True):
                                ax2.errorbar(bincenters_adc_obs2, ratio_sim2, yerr=plt_err*err_ratio_sim2, fmt='C2'+'>', lw=2, markersize=10 )
                            if(plt_sim_gauss==True):
                                ax2.errorbar(bincenters_adc_obs2, ratio_sim_gauss2, yerr=plt_err*err_ratio_sim_gauss2, fmt='C9'+'>', lw=2, markersize=10 )
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

                        adc2ekv=(bins_adc_obs2*(emax-emin)/1023.0 + emin)*pair_eh/U_eV
                        ax3.plot(adc2ekv, bins_adc_obs2*0., color='w', linewidth=0.001 )
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
                        name_cuts = source2+sim_n+difu_gauss+ sig_bg_thresh + str_cuts + cut_clst_size+cut_max + str_hist_comp
                        namesave = 'plot_hist_'+histo+'ADC_'+strbin+str(z*2).zfill(2)+'mm'+ name_cuts
                        #plt.savefig(dirsave_plt_mean+'/'+namesave+'.png', dpi=150)
                        #plt.savefig(dirsave_plt_mean+'/'+namesave+'.pdf', dpi=150)
                        plt.show()
                        #plt.clf()
                        #plt.close()
                        '''



                        plt.rcParams.update({'font.size': font_siz})
                        fig, ax_1 = plt.subplots(figsize=(15.5, 10))

                        comp_gauss1, err_comp_gauss1 = comp_obs_sim(counts_adc_obs1[101:], mean_counts_adc_sim_gauss1[101:])
                        # Filtrar los valores menores o iguales a X
                        #data_filt1 = comp_gauss1[comp_gauss1 <= np.abs(comp_gauss1.min())+1]
                        data_filt1 = comp_gauss1[(comp_gauss1 >= comp_gauss1.min()+1) & (comp_gauss1 <= np.abs(comp_gauss1.min())-1)]
                        mean_comp_gauss1= np.abs(data_filt1).mean()
                        #ax_1.hist(data_filt1, bins=bins_edep,  histtype='step'
                         #           , density=densi, log=scal_log, color='C0', alpha=0.6, linewidth=l_width,
                          #          label = source1+ ' sim' + ' mean = '  + r'%.2f'%mean_comp_gauss1)
                       
                        # Fit gaussiano
                        mu1, std1 = norm.fit(data_filt1)
                        
                        # Histograma
                        counts1, bins1, patches1 = ax_1.hist(data_filt1, bins=bins_edep, histtype='step',
                                                           density=densi, log=scal_log, color='C0', alpha=0.6,
                                                           linewidth=l_width,
                                                           label=source1 + ' sim' + ' mean = ' + r'%.2f' % mean_comp_gauss1)
                        
                        # Curva gaussiana ajustada
                        x1 = np.linspace(bins1[0], bins1[-1], 1000)
                        p1 = norm.pdf(x1, mu1, std1)
                        if not densi:  # Escalar a conteos si density=False
                            p1 *= len(data_filt1) * np.diff(bins1)[0]
                        
                        ax_1.plot(x1, p1, 'C0--', linewidth=4, label=f'Fit Gauss\nμ = {mu1:.2f}, σ = {std1:.2f}')
                        
                        ax_1.set_xlabel(r'$\frac{ (data -sim)}{\sigma}$', fontsize=font_siz)
                        if scal_log == True: ax_1.set_yscale('log')
                        ax_1.set_ylabel('counts', fontsize=font_siz)
                        #ax_1.set_title(source1, fontsize=font_siz+4)
                        ax_1.legend()
                        plt.tight_layout()
                        plt.show()
 
    
                        plt.rcParams.update({'font.size': font_siz})
                        fig, ax_2 = plt.subplots(figsize=(15.5, 10))

                        comp_gauss2, err_comp_gauss2 = comp_obs_sim(counts_adc_obs2[101:], mean_counts_adc_sim_gauss2[101:])
                        # Filtrar los valores menores o iguales a X
                        #data_filt2 = comp_gauss2[comp_gauss2 <= np.abs(comp_gauss2.min())+1]
                        data_filt2 = comp_gauss2[(comp_gauss2 >= comp_gauss2.min()+1) & (comp_gauss2 <= np.abs(comp_gauss2.min())-1)]
                        mean_comp_gauss2= np.abs(data_filt2).mean()
                        #ax_2.hist(data_filt2, bins=bins_edep,  histtype='step'
                         #           , density=densi, log=scal_log, color='C7', alpha=0.6, linewidth=l_width,
                          #          label = source2+ ' sim' + ' mean = '  + r'%.2f'%mean_comp_gauss2)
                        
                        # Fit gaussiano
                        mu2, std2 = norm.fit(data_filt2)
                        
                        # Histograma
                        counts2, bins2, patches2 = ax_2.hist(data_filt2, bins=bins_edep, histtype='step',
                                                           density=densi, log=scal_log, color='C7', alpha=0.6,
                                                           linewidth=l_width,
                                                           label=source2 + ' sim' + ' mean = ' + r'%.2f' % mean_comp_gauss2)
                        
                        # Curva gaussiana ajustada
                        x2 = np.linspace(bins2[0], bins2[-1], 1000)
                        p2 = norm.pdf(x2, mu2, std2)
                        if not densi:  # Escalar a conteos si density=False
                            p2 *= len(data_filt2) * np.diff(bins2)[0]
                        
                        ax_2.plot(x2, p2, 'C7--', linewidth=4, label=f'Fit Gauss\nμ = {mu2:.2f}, σ = {std2:.2f}')                        
                        
                        
                        ax_2.set_xlabel(r'$\frac{ (data -sim)}{\sigma}$', fontsize=font_siz)                       
                        if scal_log == True: ax_2.set_yscale('log')
                        ax_2.set_ylabel('counts', fontsize=font_siz)
                        #ax_2.set_title(source2, fontsize=font_siz+4)
                        ax_2.legend()
                        plt.tight_layout()
                        plt.show()
                       

       
                        plt.rcParams.update({'font.size': font_siz})
                        fig, ax = plt.subplots(figsize=(15.5, 10))


                        comp_gauss1, err_comp_gauss1 = comp_obs_sim(counts_adc_obs1[101:], mean_counts_adc_sim_gauss1[101:])
                        comp_gauss2, err_comp_gauss2 = comp_obs_sim(counts_adc_obs2[101:], mean_counts_adc_sim_gauss2[101:])

                        # Filtrar los valores menores o iguales a X
                        #data_filt1 = comp_gauss1[comp_gauss1 <= np.abs(comp_gauss1.min())+1]                  
                        #data_filt2 = comp_gauss2[comp_gauss2 <= np.abs(comp_gauss2.min())+1]
 
                        data_filt1 = comp_gauss1[(comp_gauss1 >= comp_gauss1.min()+1) & (comp_gauss1 <= np.abs(comp_gauss1.min())-1)]
                        data_filt2 = comp_gauss2[(comp_gauss2 >= comp_gauss2.min()+1) & (comp_gauss2 <= np.abs(comp_gauss2.min())-1)]

                        #data_filt1 = comp_gauss1[(comp_gauss1 >= -4) & (comp_gauss1 <= 4)]
                        #data_filt2 = comp_gauss2[(comp_gauss2 >= -4) & (comp_gauss2 <= 4)]
                        
                        mean_comp_gauss1= np.abs(data_filt1).mean()
                        mean_comp_gauss2= np.abs(data_filt2).mean()

                        # Definir los límites comunes para los histogramas
                        bin_min = min(data_filt1.min(), data_filt2.min())
                        bin_max = max(data_filt1.max(), data_filt2.max())
                        #bins_edep = 50  # o el número que prefieras
                        
                        # Crear el array de bordes de bins común
                        n_bins = np.linspace(bin_min, bin_max, bins_edep + 1)


                        # Fit gaussiano
                        mu1, std1 = norm.fit(data_filt1)
                                                
                        # Histograma
                        counts1, bins1, patches1 = ax.hist(data_filt1, bins=n_bins, histtype='step',
                                                           density=densi, log=scal_log, color='C0', alpha=0.7,
                                                           linewidth=l_width,
                                                           label=source1 )#+ ' sim' + ' mean = ' + r'%.2f' % mean_comp_gauss1)
                        
                        # Curva gaussiana ajustada
                        x1 = np.linspace(bins1[0], bins1[-1], 1000)
                        p1 = norm.pdf(x1, mu1, std1)
                        if not densi:  # Escalar a conteos si density=False
                            p1 *= len(data_filt1) * np.diff(bins1)[0]
                        
                        ax.plot(x1, p1, 'C0--', linewidth=4, label=f'μ = {mu1:.2f}, σ = {std1:.2f}')
                        
                        # Fit gaussiano
                        mu2, std2 = norm.fit(data_filt2)
                        
                        # Histograma
                        counts2, bins2, patches2 = ax.hist(data_filt2, bins=n_bins, histtype='step',
                                                           density=densi, log=scal_log, color='C7', alpha=0.7,
                                                           linewidth=l_width,
                                                           label=source2 )#+ ' sim' + ' mean = ' + r'%.2f' % mean_comp_gauss2)
                        
                        # Curva gaussiana ajustada
                        x2 = np.linspace(bins2[0], bins2[-1], 1000)
                        p2 = norm.pdf(x2, mu2, std2)
                        if not densi:  # Escalar a conteos si density=False
                            p2 *= len(data_filt2) * np.diff(bins2)[0]
                        
                        ax.plot(x2, p2, 'C7--', linewidth=4, label=f'μ = {mu2:.2f}, σ = {std2:.2f}')                        
                        


                        ax.set_xlabel(r'$\frac{ (data -sim)}{\sigma}$', fontsize=font_siz+8)
                        if scal_log == True: ax.set_yscale('log')
                        ax.set_ylabel('Counts', fontsize=font_siz)
                        #ax.set_title(source2, fontsize=font_siz+4)
                        ax.legend()
                        plt.tight_layout()
                        plt.savefig(dirsave_plt_mean+'/'+'comp_dat_sim_d'+str(bin_hist)+'_p'+str(bins_edep)+'.png', dpi=150)
                        plt.savefig(dirsave_plt_mean+'/'+'comp_dat_sim_d'+str(bin_hist)+'_p'+str(bins_edep)+'.pdf', dpi=150)
                        plt.show()

                        # fit gauss o abs un promedio 

                        '''
                        plt.rcParams.update({'font.size': font_siz})
                        fig, ax = plt.subplots(figsize=(15.5, 10))

                        comp_edep_gauss, err_comp_ede_gauss = comp_edep_obs1_sim(adc_obs1_w, err_adc_obs1_w, adc_sim_gauss1_w, err_adc_sim_gauss1_w)

                        ax.hist(comp_edep_gauss, bins=100,  histtype='step'
                                    , density=densi, log=scal_log, color='C0', linewidth=l_width,
                                    label = lbl_sim1 + cut_max_clst_porc_sim)
                        #ax.set_xlabel(r'$\frac{ (E^{Sim}_{dep} - E^{Obs}_{dep} )}{(E^{Obs}_{dep}}$', fontsize=font_siz)
                        ax.set_xlabel(r'$\frac{ (sim - dat)}{dat}$', fontsize=font_siz)
                        if scal_log == True: ax.set_yscale('log')
                        ax.set_ylabel('counts', fontsize=font_siz)
                        ax.set_title(source1, fontsize=font_siz+4)
                        plt.tight_layout()
                        plt.show()
                        '''

