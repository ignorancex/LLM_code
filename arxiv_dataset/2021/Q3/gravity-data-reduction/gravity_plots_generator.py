#!/usr/bin/env python
# coding: utf-8

# # Gravity observation analysis
# 
# Pablo Marcos-Arenal
# 
# This file contains the tools for analising and visualizing the end products for the observations of the ''"Probing the inner disks of non-magnetospheric Herbig Be stars using GRAVITY spectro-interferometry"'' proposal.
# These observations are named 0102.C-0576 into the ESO Data archive.
# 
# The input to this program are .fits files
# 
# 
# 
# ## Import modules


# This line configures matplotlib to show figures embedded in the notebook, 
# instead of opening a new window for each figure. More about that later. 
# If you are using an old version of IPython, try using '%pylab inline' instead.

#get_ipython().run_line_magic('matplotlib', 'inline')


# First thing is importing all required modules
import platform, sys, os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from gravi_plot_functions import *

try:
   import pyfits
except:
   from astropy.io import fits as pyfits
import time
import scipy.special
from scipy.interpolate import interp1d, splev,splrep, splprep
from scipy.signal import convolve as scipy_convolve
import pandas as pd
from io import StringIO
import matplotlib.gridspec as gridspec
from pylab import *
import statistics
from astropy.convolution import interpolate_replace_nans, Gaussian1DKernel, convolve
        
# Import GRAVIQL modules to load an plot images using the *VLTI/GRAVITY reduced and calibrated data Quick Look* [(Python reduced data visualization tool by A. Merand)](https://github.com/amerand/GRAVIQL). 
import GRAVIQL_test.trunk as ql
#import GRAVIQL_test.trunk.gravi_quick_look as ql2
import GRAVIQL_test.trunk.gravi_quick_look_3 as ql3
import gravi_visual_class





class InputStar():
    def __init__(self, source='V590Mon',wl_central_line='Brg',name=None):
        self.set_fits_files(source)
        self.source = source

        #Variable set
        #Set species lines wavelengths in microns
        self.HeI = 2.058
        self.FeII = 2.088
        self.HeI_2 = 2.1125
        self.MgII = 2.140
        self.Brg = 2.166167
        self.NaI = 2.206
        self.NIII = 2.249
        self.COb = 2.37
        
        if wl_central_line=='COb':
            self.wl_central_line = self.COb 
            
        else:
            self.wl_central_line = self.Brg #  <---------------------------- GIVEN SPECTRAL LINE


#-------------------------------------------------------------------------        
    # ## Retrieving input *.fits* files
    # Set input *.fits* files data into local variables in order to deal with them.
    # 
    # **_NOTE:_**  Here is required to select the source that will be analized in order to retrieve their *.fits* files.
    # 
    # **_NOTE 2:_** These *.fits* files can be retrieved using different modules (GRAVIQL, gravi_visual_class or pyfits). 
    #           Each of these modules can provide with different functionallities and can apply different data reduction processes by eliminating automatically outliers (GRAVIQL does that).   
    
    def set_fits_files(self, source):    

        #Set observed source:

        #source = 'V590Mon'  # 0102.C-0576(A)
        #source = 'PDS281'   # 0102.C-0576(B)
        #source = 'HD94509'  # 0102.C-0576(C)
        #source = 'DGCir'    # 0102.C-0576(D)
        #source = 'HD141926' # 0102.C-0576(E)

        #Object V590Mon (A)
        if source == 'V590Mon':
            inputpath_A = '/pcdisk/stark/pmarcos/CAB/Projects/Gravity/Data/A_V590Mon/'
            inputpath_B = '/pcdisk/stark/pmarcos/CAB/Projects/Gravity/Data/A_V590Mon/'
            filename_A = 'GRAVI2018-11-20T07-28-48.825_singlescivis_singlesciviscalibrated.fits'
            filename_B = 'GRAVI2018-11-20T07-40-06.854_singlescivis_singlesciviscalibrated.fits'

            #Calibration object 
            cal_filename_A = 'GRAVI2018-11-20T07-55-51.894_singlecalvis_singlecaltf.fits'    
            cal_filename_B = 'GRAVI2018-11-20T08-07-03.922_singlecalvis_singlecaltf.fits' 

        #Object PDS281 (B)
        elif source ==  'PDS281':   
            inputpath_A = '/pcdisk/stark/pmarcos/CAB/Projects/Gravity/Data/B_PDS281/'
            inputpath_B = '/pcdisk/stark/pmarcos/CAB/Projects/Gravity/Data/B_PDS281/'
            filename_A = 'GRAVI2018-11-20T08-25-37.535_singlescivis_singlesciviscalibrated.fits'
            filename_B = 'GRAVI2018-11-20T08-36-55.564_singlescivis_singlesciviscalibrated.fits'

            #Calibration object 
            cal_filename_A = 'GRAVI2018-11-20T09-00-52.625_singlecalvis_singlecaltf.fits'    
            cal_filename_B = 'GRAVI2018-11-20T09-06-28.639_singlecalvis_singlecaltf.fits' 
        
        #object HD 94509 (C)
        elif source == 'HD94509': 
            inputpath_A = '/pcdisk/stark/pmarcos/CAB/Projects/Gravity/Data/C_HD94509/'
            inputpath_B = '/pcdisk/stark/pmarcos/CAB/Projects/Gravity/Data/C_HD94509/'
            filename_A = 'GRAVI.2018-12-20T06:56:51.084_singlescivis_singlesciviscalibrated.fits'
            filename_B = 'GRAVI.2018-12-20T07:08:06.113_singlescivis_singlesciviscalibrated.fits'

            #Calibration object 
            cal_filename_A = 'GRAVI.2018-12-20T07:23:48.153_singlecalvis_singlecaltf.fits'    
            cal_filename_B = 'GRAVI.2018-12-20T07:34:57.181_singlecalvis_singlecaltf.fits' 
            cal_filename_C = 'GRAVI.2018-12-20T07:40:54.196_singlecalvis_singlecaltf.fits' 
        
        #Object DG Cir (D)
        elif source == 'DGCir':
            inputpath_A = '/pcdisk/stark/pmarcos/CAB/Projects/Gravity/Data/D_DGCir/'
            inputpath_B = '/pcdisk/stark/pmarcos/CAB/Projects/Gravity/Data/D_DGCir/'
            filename_A = 'GRAVI2019-01-22T07-32-01.193_singlescivis_singlesciviscalibrated.fits'
            filename_B = 'GRAVI2019-01-22T07-43-13.222_singlescivis_singlesciviscalibrated.fits'

            #Calibration object 
            cal_filename_A = 'GRAVI2019-01-22T07-16-55.155_singlecalvis_singlecaltf.fits'    
            cal_filename_B = 'GRAVI2019-01-22T07-32-01.193_singlescivis_singlescitf.fits' 
        
        #Object HD 141926 (E)
        elif source == 'HD141926':
            inputpath_A = '/pcdisk/stark/pmarcos/CAB/Projects/Gravity/Data/E_HD141926/'
            inputpath_B = '/pcdisk/stark/pmarcos/CAB/Projects/Gravity/Data/E_HD141926/'
            filename_A = 'GRAVI.2019-03-21T07:15:16.710_singlescivis_singlesciviscalibrated.fits'
            filename_B = 'GRAVI.2019-03-21T07:26:28.739_singlescivis_singlesciviscalibrated.fits'

            #Calibration object 
            cal_filename_A = 'GRAVI.2019-03-21T07:44:04.783_singlecalvis_singlecaltf.fits'    
            cal_filename_B = 'GRAVI.2019-03-21T07:55:13.811_singlecalvis_singlecaltf.fits' 
#
#            print("ERROR!!! this object has its own jupyter notebook for reduction since the telescope keywords has been changed")
#            raise ValueError('this object has its own jupyter notebook for reduction since the telescope keywords has been changed. DO NOT USE THIS NOTEBOOK!')
#        else:
#            print("ERROR in 'source' value: should be one of the souces on the list (check spelling!).")
            
        # Import data from files with GRAVIQL
        #-------------------------------------
        res_A=ql3.loadGravi(inputpath_A + filename_A, insname='GRAVITY_SC')
        res_B=ql3.loadGravi(inputpath_B + filename_B, insname='GRAVITY_SC')

        cal_A = ql3.loadGravi(inputpath_A + cal_filename_A, insname='GRAVITY_SC')
        cal_B = ql3.loadGravi(inputpath_B + cal_filename_B, insname='GRAVITY_SC')

        

        #If Object HD 141926 (E) change AT variable names to UT variable names
        if source == 'HD141926':
            res_A = auxiliary_telescope_names_to_UT(res_A)
            res_B = auxiliary_telescope_names_to_UT(res_B)
            cal_A = auxiliary_telescope_names_to_UT(cal_A)
            cal_B = auxiliary_telescope_names_to_UT(cal_B)


        
#        # Import data from files with gravi_visual_class
#        #------------------------------------------------
#        oifits_A = gravi_visual_class.Oifits(inputpath_A + filename_A)
#        oifits_B = gravi_visual_class.Oifits(inputpath_B + filename_B)
        
        
        # Set input path and input file
        #------------------------------------------------
        self.inputpath_A = inputpath_A
        self.filename_A = filename_A
        


        #Setting .fits values into variables. 
        #Data can be retrived with GRAVIQL or gravi_visual_class modules, providing sligthly different results. 
        #Most important differences is that GRAVIQL is removing outliers from data but is not retrieving errors in flux, visibility nor differential phase. 
        #We use both modules for comparison, although we will only use gravi_visual_class module.

        # Values retrieved with GRAVIQL
        #-----------------------------

        #File A:
        self.wl_A = res_A['wl']
        self.v2_A = res_A['V2']
        self.flux_A = res_A['FLUX']  
        self.averaged_flux_A = (self.flux_A['U1']+self.flux_A['U2']+self.flux_A['U3']+self.flux_A['U4'])/4
        self.cal_wl_A = cal_A['wl']
        self.cal_flux_A = cal_A['FLUX']
        self.averaged_cal_flux_A = (self.cal_flux_A['U4']+self.cal_flux_A['U3']+self.cal_flux_A['U2']+self.cal_flux_A['U1'])/4    
        
        self.diff_phase_A = res_A['VISPHI']
        self.phase_closure_A = res_A['T3']
        self.uV2_A = res_A['uV2']
        self.vV2_A = res_A['vV2']
        

        
        #File B:
        self.wl_B = res_B['wl']
        self.v2_B = res_B['V2']
        self.flux_B = res_B['FLUX']
        self.averaged_flux_B = (self.flux_B['U1']+self.flux_B['U2']+self.flux_B['U3']+self.flux_A['U4'])/4
        self.cal_wl_B = cal_B['wl']
        self.cal_flux_B = cal_B['FLUX']
        self.averaged_cal_flux_B = (self.cal_flux_B['U4']+self.cal_flux_B['U3']+self.cal_flux_B['U2']+self.cal_flux_B['U1'])/4    
        
        
        self.diff_phase_B = res_B['VISPHI']
        self.phase_closure_B = res_B['T3']
        self.uV2_B = res_B['uV2']
        self.vV2_B = res_B['vV2']

        # Use mean value from both observations
        # (File A + File B) / 2
        self.flux = (self.averaged_flux_A+self.averaged_flux_B)/2

        #cal_flux_B SEEMS TO BE EMPTY TODO:check this (pma)
        self.cal_flux = (self.averaged_cal_flux_A +self.averaged_cal_flux_B)/2

        self.visibility2 = { 'U4U3':(self.v2_A['U4U3']+self.v2_B['U4U3'])/2,
                             'U4U2':(self.v2_A['U4U2']+self.v2_B['U4U2'])/2,
                             'U4U1':(self.v2_A['U4U1']+self.v2_B['U4U1'])/2,
                             'U3U2':(self.v2_A['U3U2']+self.v2_B['U3U2'])/2,
                             'U3U1':(self.v2_A['U3U1']+self.v2_B['U3U1'])/2,
                             'U2U1':(self.v2_A['U2U1']+self.v2_B['U2U1'])/2 }


        self.diff_phase = { 'U4U3':(self.diff_phase_A['U4U3']+self.diff_phase_B['U4U3'])/2,
                            'U4U2':(self.diff_phase_A['U4U2']+self.diff_phase_B['U4U2'])/2,
                            'U4U1':(self.diff_phase_A['U4U1']+self.diff_phase_B['U4U1'])/2,
                            'U3U2':(self.diff_phase_A['U3U2']+self.diff_phase_B['U3U2'])/2,
                            'U3U1':(self.diff_phase_A['U3U1']+self.diff_phase_B['U3U1'])/2,
                            'U2U1':(self.diff_phase_A['U2U1']+self.diff_phase_B['U2U1'])/2 }


        self.phase_closure = { 'U4U3U2':(self.phase_closure_A['U4U3U2']+self.phase_closure_B['U4U3U2'])/2,
                               'U4U3U1':(self.phase_closure_A['U4U3U1']+self.phase_closure_B['U4U3U1'])/2,
                               'U4U2U1':(self.phase_closure_A['U4U2U1']+self.phase_closure_B['U4U2U1'])/2,
                               'U3U2U1':(self.phase_closure_A['U3U2U1']+self.phase_closure_B['U3U2U1'])/2 }
    

    
    
        # Values retrieved with gravi_visual_class
        #----------------------------------------
        # Error values must be retrieved from oifits files directly using gravi_visual_class because GRAVIQL_test.trunk.gravi_quick_look_3 
        # is not retrieving error values.
        # Use mean value from both observations.
        
        # FLUX
#        oifits_flux_A = np.mean(oifits_A.oi_flux_sc[0], axis=0) #Using [0] to avoid empty dimension in oifits array
#        oifits_flux_A_error = np.mean(oifits_A.oi_flux_err_sc[0], axis=0)
#        
#        oifits_flux_B = np.mean(oifits_B.oi_flux_sc[0], axis=0)
#        oifits_flux_B_error = np.mean(oifits_B.oi_flux_err_sc[0], axis=0)
#        
#        flux_gvc = (oifits_flux_A+oifits_flux_B)/2
#        self.flux_error = (oifits_flux_A_error+oifits_flux_B_error)/2
#        
#        
#        # SQUARED VISIBILITY
#        self.visibility2_gvc = {'U4U3':(oifits_A.oi_vis2_sc_vis2data[0]+oifits_B.oi_vis2_sc_vis2data[0])/2,
#                           'U4U2':(oifits_A.oi_vis2_sc_vis2data[1]+oifits_B.oi_vis2_sc_vis2data[1])/2,
#                           'U4U1':(oifits_A.oi_vis2_sc_vis2data[2]+oifits_B.oi_vis2_sc_vis2data[2])/2,
#                           'U3U2':(oifits_A.oi_vis2_sc_vis2data[3]+oifits_B.oi_vis2_sc_vis2data[3])/2,
#                           'U3U1':(oifits_A.oi_vis2_sc_vis2data[4]+oifits_B.oi_vis2_sc_vis2data[4])/2,
#                           'U2U1':(oifits_A.oi_vis2_sc_vis2data[5]+oifits_B.oi_vis2_sc_vis2data[5])/2 }
#        
#        self.visibility2_error = {'U4U3':(oifits_A.oi_vis2_sc_vis2err[0]+oifits_B.oi_vis2_sc_vis2err[0])/2,
#                             'U4U2':(oifits_A.oi_vis2_sc_vis2err[1]+oifits_B.oi_vis2_sc_vis2err[1])/2,
#                             'U4U1':(oifits_A.oi_vis2_sc_vis2err[2]+oifits_B.oi_vis2_sc_vis2err[2])/2,
#                             'U3U2':(oifits_A.oi_vis2_sc_vis2err[3]+oifits_B.oi_vis2_sc_vis2err[3])/2,
#                             'U3U1':(oifits_A.oi_vis2_sc_vis2err[4]+oifits_B.oi_vis2_sc_vis2err[4])/2,
#                             'U2U1':(oifits_A.oi_vis2_sc_vis2err[5]+oifits_B.oi_vis2_sc_vis2err[5])/2 }
#        
#        # PHASE
#        self.phase_gvc = {'U4U3':(oifits_A.oi_vis_sc_visphi[0]+oifits_B.oi_vis_sc_visphi[0])/2,
#                 'U4U2':(oifits_A.oi_vis_sc_visphi[1]+oifits_B.oi_vis_sc_visphi[1])/2,
#                 'U4U1':(oifits_A.oi_vis_sc_visphi[2]+oifits_B.oi_vis_sc_visphi[2])/2,
#                 'U3U2':(oifits_A.oi_vis_sc_visphi[3]+oifits_B.oi_vis_sc_visphi[3])/2,
#                 'U3U1':(oifits_A.oi_vis_sc_visphi[4]+oifits_B.oi_vis_sc_visphi[4])/2,
#                 'U2U1':(oifits_A.oi_vis_sc_visphi[5]+oifits_B.oi_vis_sc_visphi[5])/2  }
#        
#        self.phase_error = {'U4U3':(oifits_A.oi_vis_sc_visphierr[0]+oifits_B.oi_vis_sc_visphierr[0])/2,
#                       'U4U2':(oifits_A.oi_vis_sc_visphierr[1]+oifits_B.oi_vis_sc_visphierr[1])/2,
#                       'U4U1':(oifits_A.oi_vis_sc_visphierr[2]+oifits_B.oi_vis_sc_visphierr[2])/2,
#                       'U3U2':(oifits_A.oi_vis_sc_visphierr[3]+oifits_B.oi_vis_sc_visphierr[3])/2,
#                       'U3U1':(oifits_A.oi_vis_sc_visphierr[4]+oifits_B.oi_vis_sc_visphierr[4])/2,
#                       'U2U1':(oifits_A.oi_vis_sc_visphierr[5]+oifits_B.oi_vis_sc_visphierr[5])/2 }
        
        ##Get baselines values B/lambda 
        baseline_A = {}
        baseline_B = {}
        self.baseline = {}
        for  key, value in self.uV2_A.items():
            baseline_A[key] = sqrt(self.uV2_A[key]**2 + self.vV2_A[key]**2)
            baseline_B[key] = sqrt(self.uV2_B[key]**2 + self.vV2_B[key]**2)
            self.baseline[key] = mean([baseline_A[key],baseline_B[key]])
        


#-------------------------------------------------------------------------
    def figure_flux_preprocessing(self, plot_figure=False, save_figure=False):
        plt.interactive(True)
        fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(16, 22))
        
        ax[0].plot(self.wl_A, self.flux_A['U1'], 'r')
        ax[0].plot(self.wl_B, self.flux_B['U1'], 'b')
        ax[0].set_title(self.source + ' flux[U1]')

        ax[1].plot(self.wl_A, self.flux_A['U2'], 'r')
        ax[1].plot(self.wl_B, self.flux_B['U2'], 'b')
        ax[1].set_title(self.source + ' flux[U2]')

        ax[2].plot(self.wl_A, self.flux_A['U3'], 'r')
        ax[2].plot(self.wl_B, self.flux_B['U3'], 'b')
        ax[2].set_title(self.source + ' flux[U3]')

        ax[3].plot(self.wl_A, self.flux_A['U4'], 'r')
        ax[3].plot(self.wl_B, self.flux_B['U4'], 'b')
        ax[3].set_title(self.source + ' flux[U4]')

        ax[4].plot(self.wl_A, self.averaged_flux_A, 'r')
        ax[4].plot(self.wl_B, self.averaged_flux_B, 'b')

        ax[4].set_title(self.source + ' averaged_flux')
        ax[4].set_xlabel('Wavelength ($\mu m$)')

        #line = COb
        #xlim = [line-0.01, 0.01+line]

   #     #For each image
   #     for axes in ax:
   #         axes.set_ylabel('Flux')
   #         #Draw species spectral lines
   #         axes.axvline(self.HeI, color="green", lw=1, ls='--')
   #         axes.text(self.HeI,34000,'HeI', color="green")
   #         axes.axvline(self.FeII, color="green", lw=1, ls='--')
   #         axes.text(self.FeII,34000,'FeII', color="green")
   #         axes.axvline(self.HeI_2, color="green", lw=1, ls='--')
   #         axes.text(self.HeI_2,34000,'HeI', color="green")
   #         axes.axvline(self.MgII, color="green", lw=1, ls='--')
   #         axes.text(self.MgII,34000,'MgII', color="green")
   #         axes.axvline(self.Brg, color="green", lw=1, ls='--')
   #         axes.text(self.Brg,34000,'Brg', color="green")
   #         axes.axvline(self.NaI, color="green", lw=1, ls='--')
   #         axes.text(self.NaI,34000,'NaI', color="green")
   #         axes.axvline(self.NIII, color="green", lw=1, ls='--')
   #         axes.text(self.NIII,34000,'NIII', color="green")
   #         axes.axvline(self.COb, color="green", lw=1, ls='--')
   #         axes.text(self.COb,34000,'CO band', color="green")
   #         #axes.set_xlim(xlim) 
        
        fig.tight_layout()
        
        #Save figure
        if save_figure:
            fig.savefig("flux_" + source + "_2obs_4telescope_units.eps", dpi=300)
        
        #Close figure if plot_figure=False
        if plot_figure==False:
            plt.close(fig)    # close the figure window
            

#-------------------------------------------------------------------------
    def figure_full_range_visibility_preprocessing(self, plot_figure=False, save_figure=False):
        fig, ax = plt.subplots(3,2, figsize=(12, 14))
        
        ax[0,0].plot(self.wl_A, self.v2_A['U2U1'], 'r')
        ax[0,0].plot(self.wl_B, self.v2_B['U2U1'], 'b')
        ax[0,0].set_title(self.source + ' V2[U2U1]')
        ax[0,0].set_ylabel('$V^2$')
        ax[0,0].axhline(np.mean(self.v2_B['U2U1'])+4*np.std(self.v2_B['U2U1']), color="black", lw=1, ls='--')
        ax[0,0].axhline(np.mean(self.v2_B['U2U1'])-4*np.std(self.v2_B['U2U1']), color="black", lw=1, ls='--')
        
        ax[0,1].plot(self.wl_A, self.v2_A['U3U1'], 'r')
        ax[0,1].plot(self.wl_B, self.v2_B['U3U1'], 'b')
        ax[0,1].set_title(self.source + ' V2[U3U1]')
        ax[0,1].axhline(np.mean(self.v2_B['U3U1'])+4*np.std(self.v2_B['U3U1']), color="black", lw=1, ls='--')
        ax[0,1].axhline(np.mean(self.v2_B['U3U1'])-4*np.std(self.v2_B['U3U1']), color="black", lw=1, ls='--')
        
        ax[1,0].plot(self.wl_A, self.v2_A['U3U2'], 'r')
        ax[1,0].plot(self.wl_B, self.v2_B['U3U2'], 'b')
        ax[1,0].set_title(self.source + ' V2[U3U2]')
        ax[1,0].set_ylabel('$V^2$')
        ax[1,0].axhline(np.mean(self.v2_B['U3U2'])+4*np.std(self.v2_B['U3U2']), color="black", lw=1, ls='--')
        ax[1,0].axhline(np.mean(self.v2_B['U3U2'])-4*np.std(self.v2_B['U3U2']), color="black", lw=1, ls='--')
        
        ax[1,1].plot(self.wl_A, self.v2_A['U4U1'], 'r')
        ax[1,1].plot(self.wl_B, self.v2_B['U4U1'], 'b')
        ax[1,1].set_title(self.source + ' V2[U4U1]')
        ax[1,1].axhline(np.mean(self.v2_B['U4U1'])+4*np.std(self.v2_B['U4U1']), color="black", lw=1, ls='--')
        ax[1,1].axhline(np.mean(self.v2_B['U4U1'])-4*np.std(self.v2_B['U4U1']), color="black", lw=1, ls='--')
        
        ax[2,0].plot(self.wl_A, self.v2_A['U4U2'], 'r')
        ax[2,0].plot(self.wl_B, self.v2_B['U4U2'], 'b')
        ax[2,0].set_title(self.source + ' V2[U4U2]')
        ax[2,0].set_xlabel('Wavelength ($\mu m$)')
        ax[2,0].set_ylabel('$V^2$')
        ax[2,0].axhline(np.mean(self.v2_B['U4U2'])+4*np.std(self.v2_B['U4U2']), color="black", lw=1, ls='--')
        ax[2,0].axhline(np.mean(self.v2_B['U4U2'])-4*np.std(self.v2_B['U4U2']), color="black", lw=1, ls='--')
        
        ax[2,1].plot(self.wl_A, self.v2_A['U4U3'], 'r')
        ax[2,1].plot(self.wl_B, self.v2_B['U4U3'], 'b')
        ax[2,1].set_title(self.source + ' V2[U4U3]')
        ax[2,1].set_xlabel('Wavelength ($\mu m$)')
        ax[2,1].axhline(np.mean(self.v2_B['U4U3'])+4*np.std(self.v2_B['U4U3']), color="black", lw=1, ls='--')
        ax[2,1].axhline(np.mean(self.v2_B['U4U3'])-4*np.std(self.v2_B['U4U3']), color="black", lw=1, ls='--')


        
        #Draw species spectral lines
        for i in range(len(ax)):
            for j in range(len(ax[i])):
                ax[i,j].axvline(self.HeI, color="green", lw=1, ls='--')
                ax[i,j].axvline(self.FeII, color="green", lw=1, ls='--')
                ax[i,j].axvline(self.HeI_2, color="green", lw=1, ls='--')
                ax[i,j].axvline(self.MgII, color="green", lw=1, ls='--')
                ax[i,j].axvline(self.Brg, color="green", lw=1, ls='--')
                ax[i,j].axvline(self.NaI, color="green", lw=1, ls='--')
                ax[i,j].axvline(self.NIII, color="green", lw=1, ls='--')
                ax[i,j].axvline(self.COb, color="green", lw=1, ls='--')
                ax[i,j].text(self.HeI,3,'HeI', color="green")
                ax[i,j].text(self.FeII,3,'FeII', color="green")
                ax[i,j].text(self.HeI_2,3,'HeI', color="green")
                ax[i,j].text(self.MgII,3,'MgII', color="green")
                ax[i,j].text(self.Brg,3,'Brg', color="green")
                ax[i,j].text(self.NaI,3,'NaI', color="green")
                ax[i,j].text(self.NIII,3,'NIII', color="green")
                ax[i,j].text(self.COb,3,'CO band', color="green")
                #Set ylim for plots
                ax[i,j].set_ylim([-3, 4])

        
        #Ensure no overlapping in plots
        fig.tight_layout()

        #Save figure to disk
        if save_figure:        
            fig.savefig('./figures/'+str(self.source) + "_vis_full_bandwith.eps", dpi=300)


        #Close figure if plot_figure=False
        if plot_figure==False:
            plt.close(fig)    # close the figure window
            


 #-------------------------------------------------------------------------
    def figure_badpixel_flux_preprocessing(self, plot_figure=False, save_figure=False):
        """
        Using the previous figures as a reference, we will remove these sporeus lines coming from bad pixels adding signal to the spectrum and not coming from the observed source.

        It is created a bad pixel map where every wavelength channel has a value of 1 if it is correct and 0 if it is associated to a bad pixel.

        It is set the rejection_criterium_flux variable with the fraction to establish whether a difference between to datasets can be considered a bad pixel or not. We apply this criterium to fluxes and visibilities to cross-calibrate the rejection criterium.

        The general_rejection_criterium is applied to the final bad_pixel_mask to take in account both fluxes and visibilities rejection criteria as a whole.

        These results are plotted in pixel maps where pixels associated to wavelength channels are distributed left to right and top to bottom to cover a rectangular map. 
        
        """

    

        #Set the rejection criterium fraction
        rejection_criterium_flux = 4
        general_rejection_criterium = 5
        
        #Set each bad pixel mask as empty, meaning that there is no bad pixel (all are correct) setting their value to 1
        bad_pixel_flux_U1 = ones(len(self.wl_A))
        bad_pixel_flux_U2 = ones(len(self.wl_A))
        bad_pixel_flux_U3 = ones(len(self.wl_A))
        bad_pixel_flux_U4 = ones(len(self.wl_A))
        
        bad_pixel_visibility_U4U3 = ones(len(self.wl_A))
        bad_pixel_visibility_U4U2 = ones(len(self.wl_A))
        bad_pixel_visibility_U4U1 = ones(len(self.wl_A))
        bad_pixel_visibility_U3U2 = ones(len(self.wl_A))
        bad_pixel_visibility_U3U1 = ones(len(self.wl_A))
        bad_pixel_visibility_U2U1 = ones(len(self.wl_A))
        
        
        bad_pixel_diff_phase = ones(len(self.wl_A))
        bad_pixel_closure_phase = ones(len(self.wl_A))
        bad_pixel_mask = ones(len(self.wl_A))
        
        for wl_channel in range(len(self.wl_A)-1):
            # Rejection on flux criterium
            if abs(self.flux_A['U1'][wl_channel]/self.flux_B['U1'][wl_channel]) < rejection_criterium_flux:
                bad_pixel_flux_U1[wl_channel] = 1
            else:
                bad_pixel_flux_U1[wl_channel] = 0
            
            if abs(self.flux_A['U2'][wl_channel]/self.flux_B['U2'][wl_channel]) < rejection_criterium_flux:
                bad_pixel_flux_U2[wl_channel] = 1
            else:
                bad_pixel_flux_U2[wl_channel] = 0
        
            if abs(self.flux_A['U3'][wl_channel]/self.flux_B['U3'][wl_channel]) < rejection_criterium_flux:
                bad_pixel_flux_U3[wl_channel] = 1
            else:
                bad_pixel_flux_U3[wl_channel] = 0
        
            if abs(self.flux_A['U4'][wl_channel]/self.flux_B['U4'][wl_channel]) < rejection_criterium_flux:
                bad_pixel_flux_U4[wl_channel] = 1
            else:
                bad_pixel_flux_U4[wl_channel] = 0
        
        
            # Rejection on visibility criterium
            if abs(self.v2_A['U4U3'][wl_channel]+self.v2_B['U4U3'][wl_channel]) < rejection_criterium_flux:
                bad_pixel_visibility_U4U3[wl_channel] = 1
            else:
                bad_pixel_visibility_U4U3[wl_channel] = 0
                
            if abs(self.v2_A['U4U2'][wl_channel]+self.v2_B['U4U2'][wl_channel]) < rejection_criterium_flux:
                bad_pixel_visibility_U4U2[wl_channel] = 1
            else:
                bad_pixel_visibility_U4U2[wl_channel] = 0
                
            if abs(self.v2_A['U4U1'][wl_channel]+self.v2_B['U4U1'][wl_channel]) < rejection_criterium_flux:
                bad_pixel_visibility_U4U1[wl_channel] = 1
            else:
                bad_pixel_visibility_U4U1[wl_channel] = 0
                
            if abs(self.v2_A['U3U2'][wl_channel]+self.v2_B['U3U2'][wl_channel]) < rejection_criterium_flux:
                bad_pixel_visibility_U3U2[wl_channel] = 1
            else:
                bad_pixel_visibility_U3U2[wl_channel] = 0
                
            if abs(self.v2_A['U3U1'][wl_channel]+self.v2_B['U3U1'][wl_channel]) < rejection_criterium_flux:
                bad_pixel_visibility_U3U1[wl_channel] = 1
            else:
                bad_pixel_visibility_U3U1[wl_channel] = 0
                
            if abs(self.v2_A['U2U1'][wl_channel]+self.v2_B['U2U1'][wl_channel]) < rejection_criterium_flux:
                bad_pixel_visibility_U2U1[wl_channel] = 1
            else:
                bad_pixel_visibility_U2U1[wl_channel] = 0
                
        #bad_pixel_visibility = [bad_pixel_visibility_U4U3, bad_pixel_visibility_U4U2, bad_pixel_visibility_U4U1, bad_pixel_visibility_U3U2, bad_pixel_visibility_U3U1, bad_pixel_visibility_U2U1]
        bad_pixel_visibility = bad_pixel_visibility_U4U3+ bad_pixel_visibility_U4U2+ bad_pixel_visibility_U4U1+ bad_pixel_visibility_U3U2+ bad_pixel_visibility_U3U1+ bad_pixel_visibility_U2U1
        bad_pixel_visibility_map = bad_pixel_visibility.reshape(26,67)
        
        bad_pixel_flux = bad_pixel_flux_U1 + bad_pixel_flux_U2 + bad_pixel_flux_U3 + bad_pixel_flux_U4
        bad_pixel_flux_map = bad_pixel_flux.reshape(26,67)
        
        bad_pixel_mask = bad_pixel_flux + bad_pixel_visibility/2
        for pixel in range(len(bad_pixel_mask)):
            if bad_pixel_mask[pixel] > general_rejection_criterium:
                bad_pixel_mask[pixel] = 1
            else:
                bad_pixel_mask[pixel] = np.nan
            
        bad_pixel_array = [self.wl_A, bad_pixel_flux, bad_pixel_visibility, bad_pixel_diff_phase, bad_pixel_closure_phase, bad_pixel_mask]
        
        mask = (bad_pixel_flux + bad_pixel_visibility/2) < general_rejection_criterium
        
        bad_pixel_map=[]
        bad_pixel_flux[mask] = 1
        
        false_mask = mask != False
        bad_pixel_flux[false_mask] = np.nan
        
        
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 12))
        
        current_cmap = matplotlib.cm.get_cmap()
        current_cmap.set_bad(color='red')
        im = ax[0].imshow(bad_pixel_mask.reshape(26,67))
        ax[0].set_title(self.source + ' Bad Pixel map')
        
        #im = ax[0].imshow(bad_pixel_flux.reshape(26,67))
        #fig.colorbar(im,  orientation='vertical')
        
        ax[1].plot(self.wl_A, self.flux, 'r', label='averaged flux')
        self.BPcorrected_flux = self.flux * bad_pixel_mask
        #BPcorrected_flux[false_mask]=np.nan
        ax[1].plot(self.wl_A, self.BPcorrected_flux, 'b', label='bad pixel corrected flux')
        ax[1].set_title(self.source + ' Corrected averaged flux')
        ax[1].set_xlabel('Wavelength ($\mu m$)')
        ax[1].set_ylabel('Flux')
        ax[1].legend(loc=1) # upper right corner
        
        # Species label y position
        specie_label_y_position = min(self.BPcorrected_flux)-100
        specie_label_x_position_offset = 0.001
        
        #Draw species spectral lines
        ax[1].axvline(self.HeI, color="green", lw=1, ls='--')
        ax[1].axvline(self.FeII, color="green", lw=1, ls='--')
        ax[1].axvline(self.HeI_2, color="green", lw=1, ls='--')
        ax[1].axvline(self.MgII, color="green", lw=1, ls='--')
        ax[1].axvline(self.Brg, color="green", lw=1, ls='--')
        ax[1].axvline(self.NaI, color="green", lw=1, ls='--')
        ax[1].axvline(self.NIII, color="green", lw=1, ls='--')
        ax[1].axvline(self.COb, color="green", lw=1, ls='--')
        ax[1].text(self.HeI+specie_label_x_position_offset,specie_label_y_position,'HeI', color="green")
        ax[1].text(self.FeII+specie_label_x_position_offset,specie_label_y_position,'FeII', color="green")
        ax[1].text(self.HeI_2+specie_label_x_position_offset,specie_label_y_position,'HeI', color="green")
        ax[1].text(self.MgII+specie_label_x_position_offset,specie_label_y_position,'MgII', color="green")
        ax[1].text(self.Brg+specie_label_x_position_offset,specie_label_y_position,'Brg', color="green")
        ax[1].text(self.NaI+specie_label_x_position_offset,specie_label_y_position,'NaI', color="green")
        ax[1].text(self.NIII+specie_label_x_position_offset,specie_label_y_position,'NIII', color="green")
        ax[1].text(self.COb+specie_label_x_position_offset,specie_label_y_position,'CO band', color="green")
        

        fig.tight_layout()
    
        #Save figure
        if save_figure:
            #Save figure to disk
            fig.savefig('./figures/'+str(self.source) + "_badpixel_correction" + ".eps", dpi=300)


        #Close figure if plot_figure=False
        if plot_figure==False:
            plt.close(fig)    # close the figure window
            

    
#-------------------------------------------------------------------------

    def figure_badpixel_V2_preprocessing(self, xlim=xlim(), ylim=ylim(), sigma_coefficient = 5, plot_figure=False, save_figure=False):
        """
        Bad pixel in Visibilities must be treated in a different manner since outliers in different baselines seem not to be related to a common pixel map. 
        Therefore all outliers are rejected based on a different criterium. 
        Since there are no shifts in visibilities for any of our sources (see figures generated at figure_full_range_visibility_preprocessing() function), we can simply reject outliers > sigma_coefficient * standard deviation. 
        This is similar to use a different bad pixel mask for each baseline.
        """
        
        upper_limit_A = {}
        lower_limit_A = {}
        upper_limit_B = {}
        lower_limit_B = {}

        #For each baseline:
        for key, value in self.v2_A.items():
            v2_A_removed_nan = []
            v2_A_removed_nan = [x for x in value if ~np.isnan(x)]
            #Calculation of mean and standard deviation value for each observation:
            upper_limit_A[key] = np.mean((v2_A_removed_nan)) + sigma_coefficient * abs(np.std(v2_A_removed_nan))
            lower_limit_A[key] = np.mean((v2_A_removed_nan)) - sigma_coefficient * abs(np.std(v2_A_removed_nan))
            
        for key, value in self.v2_B.items():
            v2_B_removed_nan = []
            v2_B_removed_nan = [x for x in value if ~np.isnan(x)]
            upper_limit_B[key] = np.mean((v2_B_removed_nan)) + sigma_coefficient * abs(np.std(v2_B_removed_nan))
            lower_limit_B[key] = np.mean((v2_B_removed_nan)) - sigma_coefficient * abs(np.std(v2_B_removed_nan))

        for key, value in self.v2_A.items():
            #For every wavelength
            for wl_channel in range(len(self.v2_A)-1):
    
                # Rejection on visibility criterium
                if (abs(self.v2_A[key][wl_channel]) > upper_limit_A[key] or 
                   abs(self.v2_A[key][wl_channel]) < lower_limit_A[key] or 
                   abs(self.v2_B[key][wl_channel]) > upper_limit_B[key] or
                   abs(self.v2_B[key][wl_channel]) < lower_limit_B[key]):
                    
                    self.visibility2[key][wl_channel] = np.nan


        
        fig, ax = plt.subplots(3,2, figsize=(12, 14))

        
        # Plot subplots
        ax[0,0].plot(self.wl_A, self.v2_A['U2U1'], 'r', lw=1,label='Obs. A')
        ax[0,0].plot(self.wl_B, self.v2_B['U2U1'], 'b', lw=1,label='Obs. B')
        ax[0,0].plot(self.wl_A, self.visibility2['U2U1'], color="black",label='Mean (A,B)')
        ax[0,0].legend(loc=2) # upper left corner
        ax[0,0].set_title(self.source + ' V2[U2U1]')
        ax[0,0].set_ylabel('$V^2$')
        ax[0,0].axhline(upper_limit_A['U2U1'], color="black",label='Cut-off limit A', lw=1, ls='--')
        ax[0,0].axhline(lower_limit_A['U2U1'], color="black",label='Cut-off limit A', lw=1, ls='--')
        ax[0,0].axhline(upper_limit_B['U2U1'], color="grey" ,label='Cut-off limit B', lw=1, ls='--')
        ax[0,0].axhline(lower_limit_B['U2U1'], color="grey" ,label='Cut-off limit B',lw=1, ls='--')

        ax[0,1].plot(self.wl_A, self.v2_A['U3U1'], 'r', lw=1,label='Obs. A')
        ax[0,1].plot(self.wl_B, self.v2_B['U3U1'], 'b', lw=1,label='Obs. B')
        ax[0,1].plot(self.wl_A, self.visibility2['U3U1'], color="black",label='Mean (A,B)')
        ax[0,1].legend(loc=2) # upper left corner
        ax[0,1].set_title(self.source + ' V2[U3U1]')
        ax[0,1].axhline(upper_limit_A['U3U1'], color="black",label='Cut-off limit A', lw=1, ls='--')
        ax[0,1].axhline(lower_limit_A['U3U1'], color="black",label='Cut-off limit A', lw=1, ls='--')
        ax[0,1].axhline(upper_limit_B['U3U1'], color="grey" ,label='Cut-off limit B', lw=1, ls='--')
        ax[0,1].axhline(lower_limit_B['U3U1'], color="grey" ,label='Cut-off limit B', lw=1, ls='--')
        
        ax[1,0].plot(self.wl_A, self.v2_A['U3U2'], 'r', lw=1,label='Obs. A')
        ax[1,0].plot(self.wl_B, self.v2_B['U3U2'], 'b', lw=1,label='Obs. B')
        ax[1,0].plot(self.wl_A, self.visibility2['U3U2'], color="black",label='Mean (A,B)')
        ax[1,0].legend(loc=2) # upper left corner
        ax[1,0].set_title(self.source + ' V2[U3U2]')
        ax[1,0].set_ylabel('$V^2$')
        ax[1,0].axhline(upper_limit_A['U3U2'], color="black",label='Cut-off limit A', lw=1, ls='--')
        ax[1,0].axhline(lower_limit_A['U3U2'], color="black",label='Cut-off limit A', lw=1, ls='--')
        ax[1,0].axhline(upper_limit_B['U3U2'], color="grey" ,label='Cut-off limit B', lw=1, ls='--')
        ax[1,0].axhline(lower_limit_B['U3U2'], color="grey" ,label='Cut-off limit B', lw=1, ls='--')
        
        ax[1,1].plot(self.wl_A, self.v2_A['U4U1'], 'r', lw=1,label='Obs. A')
        ax[1,1].plot(self.wl_B, self.v2_B['U4U1'], 'b', lw=1,label='Obs. B')
        ax[1,1].plot(self.wl_A, self.visibility2['U4U1'], color="black",label='Mean (A,B)')
        ax[1,1].legend(loc=2) # upper left corner
        ax[1,1].set_title(self.source + ' V2[U4U1]')
        ax[1,1].axhline(upper_limit_A['U4U1'], color="black",label='Cut-off limit A', lw=1, ls='--')
        ax[1,1].axhline(lower_limit_A['U4U1'], color="black",label='Cut-off limit A', lw=1, ls='--')
        ax[1,1].axhline(upper_limit_B['U4U1'], color="grey" ,label='Cut-off limit B', lw=1, ls='--')
        ax[1,1].axhline(lower_limit_B['U4U1'], color="grey" ,label='Cut-off limit B', lw=1, ls='--')
        
        ax[2,0].plot(self.wl_A, self.v2_A['U4U2'], 'r', lw=1,label='Obs. A')
        ax[2,0].plot(self.wl_B, self.v2_B['U4U2'], 'b', lw=1,label='Obs. B')
        ax[2,0].plot(self.wl_A, self.visibility2['U4U2'], color="black",label='Mean (A,B)')
        ax[2,0].legend(loc=2) # upper left corner
        ax[2,0].set_title(self.source + ' V2[U4U2]')
        ax[2,0].set_xlabel('Wavelength ($\mu m$)')
        ax[2,0].set_ylabel('$V^2$')
        ax[2,0].axhline(upper_limit_A['U4U2'], color="black",label='Cut-off limit A', lw=1, ls='--')
        ax[2,0].axhline(lower_limit_A['U4U2'], color="black",label='Cut-off limit A', lw=1, ls='--')
        ax[2,0].axhline(upper_limit_B['U4U2'], color="grey" ,label='Cut-off limit B', lw=1, ls='--')
        ax[2,0].axhline(lower_limit_B['U4U2'], color="grey" ,label='Cut-off limit B', lw=1, ls='--')
        
        ax[2,1].plot(self.wl_A, self.v2_A['U4U3'], 'r', lw=1,label='Obs. A')
        ax[2,1].plot(self.wl_B, self.v2_B['U4U3'], 'b', lw=1,label='Obs. B')
        ax[2,1].plot(self.wl_A, self.visibility2['U4U3'], color="black",label='Mean (A,B)')
        ax[2,1].legend(loc=2) # upper left corner
        ax[2,1].set_title(self.source + ' V2[U4U3]')
        ax[2,1].set_xlabel('Wavelength ($\mu m$)')
        ax[2,1].axhline(upper_limit_A['U4U3'], color="black",label='Cut-off limit A', lw=1, ls='--')
        ax[2,1].axhline(lower_limit_A['U4U3'], color="black",label='Cut-off limit A', lw=1, ls='--')
        ax[2,1].axhline(upper_limit_B['U4U3'], color="grey" ,label='Cut-off limit B', lw=1, ls='--')
        ax[2,1].axhline(lower_limit_B['U4U3'], color="grey" ,label='Cut-off limit B', lw=1, ls='--')
        
        
        #For each subplot
        for i in range(len(ax)):
            for j in range(len(ax[i])):
                ax[i,j].set_xlim(xlim) 
                ax[i,j].axvline(self.wl_central_line, color="green", lw=1, ls='--')
                ax[i,j].set_ylim(ylim)
                #ax[i,j].text(self.wl_central_line,1.2,round(self.wl_central_line,3), color="green")
        
        #Ensure no overlapping in plots
        fig.tight_layout()

        #Save figure
        if save_figure:
           #Save figure to disk
           fig.savefig('./figures/'+str(self.source) + "_badpixel_V2_preprocessing" + ".eps", dpi=300)


        #Close figure if plot_figure=False
        if plot_figure==False:
            plt.close(fig)    # close the figure window
            
            
#-------------------------------------------------------------------------
    def figure_badpixel_calibration_preprocessing(self, rejection_margin=0.25, plot_figure=False, save_figure=False):
        """
        In this case, there is also a replacement of Bad Pixels by NaN data, but those holes are now replaced by an interpolated value through convolution. This cannot be the case in science image because we will be replacing observational data (which might be incorrect) by interpolated data. What we do is using NaN to make sure that we do not compute Bad Data in our wavelengths channels of interest. For calibration data we do can replace bad data with interpolated data, specially in that wavelengths ranges where we are interested in but the calibrator matches with Bad pixels.

We use the averaged the flux of both observations in all telescope units to get the convolved flux. All fluxes (in both observations an all telescope units) are compared to this to remove bad pixels and substitute them by NaN. These are replaced afterwards by a new averaged flux (after bad pixels removal). 
        
        Parameter to fine-tune the rejection margin
        rejection_margin = 0.25
        """
 
        
        #Set convolution kernel with low convolution range (stddev=1) in order to avoid removing small features like telluric lines  
        gaussian_kernel = Gaussian1DKernel(stddev=1, x_size=len(self.cal_flux), mode='oversample')
        
        #Convolve average calibration flux with gaussian_kernel
        cal_scipy_convolved = scipy_convolve(self.cal_flux,gaussian_kernel, mode='same')
        
        
        
        #Plot Averaged calibration flux and convolved calibration flux
        fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(15, 12))
        ax[0].plot(self.cal_wl_B, self.cal_flux, 'b', label='average calibration flux')
        ax[0].plot(self.cal_wl_B, cal_scipy_convolved , 'g', label='convolved calibration flux')
        ax[0].legend(loc=2) # upper left corner
        ax[0].set_title(self.source + ' Convolved calibration flux')
        
        
        #Plot Averaged calibration flux and the original fluxes for each telescope and observation to see the effect
        #of peaks in the averaged value
        ax[1].plot(self.cal_wl_A, self.cal_flux_A['U1'], alpha=0.5, linewidth=2)
        ax[1].plot(self.cal_wl_B, self.cal_flux_B['U1'], alpha=0.5, linewidth=2)
        
        ax[1].plot(self.cal_wl_A, self.cal_flux_A['U2'], alpha=0.5, linewidth=2)
        ax[1].plot(self.cal_wl_B, self.cal_flux_B['U2'], alpha=0.5, linewidth=2)
        
        ax[1].plot(self.cal_wl_A, self.cal_flux_A['U3'], alpha=0.5, linewidth=2)
        ax[1].plot(self.cal_wl_B, self.cal_flux_B['U3'], alpha=0.5, linewidth=2)
        
        ax[1].plot(self.cal_wl_A, self.cal_flux_A['U4'], alpha=0.5, linewidth=2)
        ax[1].plot(self.cal_wl_B, self.cal_flux_B['U4'], alpha=0.5, linewidth=2)
        
        ax[1].plot(self.cal_wl_B, self.cal_flux, 'b', label='average calibration flux', linewidth=6)
        
        ax[1].legend(loc=2) # upper left corner
        ax[1].set_title(self.source + ' averaged_flux')
        ax[1].set_xlabel('Wavelength ($\mu m$)')
        
       
        #Generate copy of calibration fluxes to set NaN on bad pixels
        cal_A_with_nan = self.cal_flux_A.copy()
        cal_B_with_nan = self.cal_flux_B.copy()
        
        #Generate copy of calibration fluxes to corrected values on them
        BPcorrected_calibration_flux_A = self.cal_flux_A.copy()
        BPcorrected_calibration_flux_B = self.cal_flux_B.copy()
        
        #Set convolution kernel with high convolution range (stddev=10)
        #Gaussian kernel must be odd size
        gaussian_kernel_10 = Gaussian1DKernel(stddev=10, x_size=len(self.cal_flux)-1, mode='oversample')
        
        #Convolve average calibration flux with gaussian_kernel_10
        cal_scipy_convolved_10 = scipy_convolve(self.cal_flux,gaussian_kernel_10, mode='same')
        
        #New plot 
        plt.figure(3, figsize=(18, 12))
        
        for index in cal_A_with_nan:
            
        #    #Sustitute outlayers by nan
        #    if abs(v2_A['U4U2'][wl_channel]+v2_B['U4U2'][wl_channel]) < rejection_criterium_flux:
        #        bad_pixel_visibility_U4U2[wl_channel] = 1
        #    else:
        #        bad_pixel_visibility_U4U2[wl_channel] = 0
        
            cal_A_with_nan[index][abs(cal_A_with_nan[index]/cal_B_with_nan[index])>1+rejection_margin]= np.nan
            cal_B_with_nan[index][abs(cal_B_with_nan[index]/cal_A_with_nan[index])>1+rejection_margin]= np.nan
            cal_A_with_nan[index][abs(cal_A_with_nan[index]/cal_B_with_nan[index])<1-rejection_margin]= np.nan
            cal_B_with_nan[index][abs(cal_B_with_nan[index]/cal_A_with_nan[index])<1-rejection_margin]= np.nan
            
        
        #Sustitute outlayers by nan
            cal_A_with_nan[index][abs(cal_A_with_nan[index]/cal_scipy_convolved_10)>1+rejection_margin]= np.nan
            cal_B_with_nan[index][abs(cal_B_with_nan[index]/cal_scipy_convolved_10)>1+rejection_margin]= np.nan
            cal_A_with_nan[index][abs(cal_A_with_nan[index]/cal_scipy_convolved_10)<1-rejection_margin]= np.nan
            cal_B_with_nan[index][abs(cal_B_with_nan[index]/cal_scipy_convolved_10)<1-rejection_margin]= np.nan
        
            ax[2].plot(self.cal_wl_A, cal_B_with_nan[index], linewidth=3, label='B_'+index)
            ax[2].plot(self.cal_wl_A, cal_A_with_nan[index], linewidth=3, label='A_'+index)
            #ax[2].plot(cal_wl_A, abs(cal_A_with_nan[index]/cal_B_with_nan[index]), linewidth=3, label='2_'+index)
            
            #Sustitute nan by gaussian convolved kernel
            BPcorrected_calibration_flux_A[index] = interpolate_replace_nans(cal_A_with_nan[index], gaussian_kernel_10)
            BPcorrected_calibration_flux_B[index] = interpolate_replace_nans(cal_B_with_nan[index], gaussian_kernel_10)
            #ax[2].plot(cal_wl_A, BPcorrected_calibration_flux_A[index], linewidth=3, label='A_'+index)
            #ax[2].plot(cal_wl_A, BPcorrected_calibration_flux_B[index], linewidth=3, label='B_'+index)
        
        ax[2].legend(loc='best')
        ax[2].set_title(self.source + 'Calibration A and B flux all telescopes')
                    
        
        # Calculate mean value for each observation and telescope    
        self.BPcorrected_calibration_flux = (BPcorrected_calibration_flux_A['U4']+BPcorrected_calibration_flux_A['U3']+BPcorrected_calibration_flux_A['U2']+BPcorrected_calibration_flux_A['U1'] + 
                                BPcorrected_calibration_flux_B['U4']+BPcorrected_calibration_flux_B['U3']+BPcorrected_calibration_flux_B['U2']+BPcorrected_calibration_flux_B['U1'])/8  
        
        #Replacing NaN from Calibration flux (self.cal_flux) 
        self.cal_flux = replace_nans_by_interpolated_gaussian(self.cal_flux)
            
        #Normalized corrected calibration flux (to be used to remove telluric and measure continuum)
        self.normalized_BPcorrected_calibration_flux = self.cal_flux/self.BPcorrected_calibration_flux
        

        ax[3].plot(self.cal_wl_A, self.cal_flux/np.median(self.cal_flux), linewidth=3, label='cal_flux')
        ax[3].plot(self.cal_wl_A, self.BPcorrected_calibration_flux/np.median(self.cal_flux), linewidth=3, label='BPcorrected_calibration_flux')
        ax[3].plot(self.cal_wl_A, self.normalized_BPcorrected_calibration_flux, linewidth=3, label='normalized_BPcorrected_calibration_flux')
        ax[3].legend(loc='best')
        ax[3].set_ylim([0,3])
        ax[3].set_title(self.source + ' Normalized corrected calibration flux')

        # Species label y position
        specie_label_y_position = 0.2
        specie_label_x_position_offset = 0.001     
        #Draw species spectral lines
        ax[3].axvline(self.HeI, color="green", lw=1, ls='--')
        ax[3].axvline(self.FeII, color="green", lw=1, ls='--')
        ax[3].axvline(self.HeI_2, color="green", lw=1, ls='--')
        ax[3].axvline(self.MgII, color="green", lw=1, ls='--')
        ax[3].axvline(self.Brg, color="green", lw=1, ls='--')
        ax[3].axvline(self.NaI, color="green", lw=1, ls='--')
        ax[3].axvline(self.NIII, color="green", lw=1, ls='--')
        ax[3].axvline(self.COb, color="green", lw=1, ls='--')
        ax[3].text(self.HeI+specie_label_x_position_offset,specie_label_y_position,'HeI', color="green")
        ax[3].text(self.FeII+specie_label_x_position_offset,specie_label_y_position,'FeII', color="green")
        ax[3].text(self.HeI_2+specie_label_x_position_offset,specie_label_y_position,'HeI', color="green")
        ax[3].text(self.MgII+specie_label_x_position_offset,specie_label_y_position,'MgII', color="green")
        ax[3].text(self.Brg+specie_label_x_position_offset,specie_label_y_position,'Brg', color="green")
        ax[3].text(self.NaI+specie_label_x_position_offset,specie_label_y_position,'NaI', color="green")
        ax[3].text(self.NIII+specie_label_x_position_offset,specie_label_y_position,'NIII', color="green")
        ax[3].text(self.COb+specie_label_x_position_offset,specie_label_y_position,'CO band', color="green")
        
        fig.tight_layout()
        
        #Save figure
        if save_figure:
            #Save figure to disk
            fig.savefig('./figures/'+str(self.source) + "_badpixel_calibration_correction" + ".eps", dpi=300)

        #Close figure if plot_figure=False
        if plot_figure==False:
            plt.close(fig)    # close the figure window
            

#-------------------------------------------------------------------------
    def figure_flux_continuum_measurement(self, xlim=xlim(), ylim=ylim(),continuum_range=[2.153,2.16,2.17,2.19],line_range=[2.16,2.17],plot_figure=False, save_figure=False):
        """
        The continuum around our line of interest (𝐵𝑟𝛾) is measured in the flux data (after Bad-pixel removal) divided by the normalized calibration flux (Bad-pixel removed). The following figure shows the wavelength range where the continuum is measured.
        """

        
        #Calculate median value of flux in continuum_range over the normalized spectra   
        # at the left of the line
        self.index_lower_continuum_range_left = min(range(len(self.wl_A)), key=lambda i: abs(self.wl_A[i]-continuum_range[0]))
        index_higher_continuum_range_left  = min(range(len(self.wl_A)), key=lambda i: abs(self.wl_A[i]-continuum_range[1]))

        # at the right of the line
        index_lower_continuum_range_right = min(range(len(self.wl_A)), key=lambda i: abs(self.wl_A[i]-continuum_range[2]))
        self.index_higher_continuum_range_right  = min(range(len(self.wl_A)), key=lambda i: abs(self.wl_A[i]-continuum_range[3]))

        #Concatenate left and right arrays to get common median value
        flux_continuum_concatenated=np.concatenate([self.BPcorrected_flux[self.index_lower_continuum_range_left:index_higher_continuum_range_left],self.BPcorrected_flux[index_lower_continuum_range_right:self.index_higher_continuum_range_right]])
        
        
        #Calculate median value of BPcorrected_flux in continuum_range over the spectra (NOT normalized)        
        self.median_BPcorrected_flux = statistics.mean(flux_continuum_concatenated)


        #Concatenate left and right arrays to get common median value
        calibration_continuum_concatenated=np.concatenate([self.BPcorrected_calibration_flux[self.index_lower_continuum_range_left:index_higher_continuum_range_left],self.BPcorrected_calibration_flux[index_lower_continuum_range_right:self.index_higher_continuum_range_right]])
        
        #Calculate median value of BPcorrected_calibration_flux in continuum_range over the spectra (NOT normalized)        
        self.median_BPcorrected_calibration_flux = statistics.mean(calibration_continuum_concatenated)


       # self.wl_crop=np.concatenate([self.wl_A[self.index_lower_continuum_range_left:index_higher_continuum_range_left],self.wl_A[index_lower_continuum_range_right:self.index_higher_continuum_range_right]])
        
        #SET final continuum flux normalized
        self.final_continuum_flux_normalized = (self.BPcorrected_flux/self.median_BPcorrected_flux)/(self.BPcorrected_calibration_flux/self.median_BPcorrected_calibration_flux)
        
        
        #SET final line flux normalized
        self.final_line_flux_normalized = (self.BPcorrected_flux/self.median_BPcorrected_flux)

        

        #Identify the index in the flux arrays corresponding to the line_ranges ends.
        index_lower_line_range = min(range(len(self.wl_A)), key=lambda i: abs(self.wl_A[i]-line_range[0]))
        index_higher_line_range = min(range(len(self.wl_A)), key=lambda i: abs(self.wl_A[i]-line_range[1]))
                
        #Final flux to be written to new FITS file
        self.final_flux = self.final_continuum_flux_normalized
        
        #Calculate standard deviation of the final continuum flux normalized as flux error (before line flux concatenation) 
        self.final_flux_error = np.concatenate([self.final_flux[self.index_lower_continuum_range_left:index_higher_continuum_range_left],self.final_flux[index_lower_continuum_range_right:self.index_higher_continuum_range_right]])

        self.final_flux_error = np.std(self.final_flux_error)
        
        
        #Replace flux in line_range by final_line_flux_normalized
        self.final_flux[index_lower_line_range:index_higher_line_range] = self.final_line_flux_normalized[index_lower_line_range:index_higher_line_range]

        
        #Plot figure
        fig, axes = plt.subplots(figsize=(16, 6))
    
        axes.plot(self.wl_A, self.final_line_flux_normalized, 'b', label='Continuum flux not divided by calibrator')        
        axes.plot(self.wl_A, self.final_continuum_flux_normalized, 'r', label='Continuum flux divided by calibrator')
        
        axes.errorbar(2.14, 1.4,yerr = self.final_flux_error, fmt='o', mfc='white', color="red", capsize=2, elinewidth=2, ecolor='red')
        
        axes.legend(loc='best') 
        axes.axvline(self.wl_central_line, color="green", lw=1, ls='-')
        axes.axvline(continuum_range[0], color="grey", lw=1, ls='--')
        axes.axvline(continuum_range[1], color="grey", lw=1, ls='--')
        axes.axvline(continuum_range[2], color="grey", lw=1, ls='--')
        axes.axvline(continuum_range[3], color="grey", lw=1, ls='--')
        axes.set_ylim(ylim)
        axes.set_xlim(xlim) 
        axes.set_ylabel('Relative Flux')
        x_min_left_1=(continuum_range[0]-xlim[0])/(xlim[1]-xlim[0])
        x_max_right_1=(continuum_range[1]-xlim[0])/(xlim[1]-xlim[0])
        axes.axhline(1, xmin=x_min_left_1, xmax=x_max_right_1, color="r", lw=1, ls='dotted')


        #axes.text(self.wl_central_line,1.2,round(self.wl_central_line,3), color="green")
        #axes.text(1,1.2,'Line is not divided by calibrator')
         
        x_min_left_2=(continuum_range[2]-xlim[0])/(xlim[1]-xlim[0])
        x_max_right_2=(continuum_range[3]-xlim[0])/(xlim[1]-xlim[0])
        axes.axhline(1, xmin=x_min_left_2, xmax=x_max_right_2, color="r", lw=1, ls='dotted')
        
        axes.set_title(self.source + ' Flux continuum measurement')
      
        fig.tight_layout()
         
        #Save figure
        if save_figure:
            #Save figure to disk
            fig.savefig('./figures/'+str(self.source) + "_figure_flux_continuum_measurement" + ".eps", dpi=300)


        #Close figure if plot_figure=False
        if plot_figure==False:
            plt.close(fig)    # close the figure window
            


#-------------------------------------------------------------------------
    def figure_visibility_continuum_measurement(self, xlim=xlim(), ylim=ylim(),continuum_range=[2.141,2.1604],plot_figure=False, save_figure=False):
        """
        The continuum in visibility (continuum_visibility2) around our line of interest (𝐵𝑟𝛾) is measured. The following figure shows the wavelength range where the continuum is measured.        
        """

        
        #Calculate median value of visibility in continuum_range         
        index_lower_continuum_range = min(range(len(self.wl_A)), key=lambda i: abs(self.wl_A[i]-continuum_range[0]))
        index_higher_continuum_range  = min(range(len(self.wl_A)), key=lambda i: abs(self.wl_A[i]-continuum_range[1]))
        
        self.continuum_visibility2 = {}
        self.continuum_visibility2_error = {}
        
        for key, value in self.visibility2.items():
            v2_removed_nan = [x for x in value if ~np.isnan(x)]
            # WARNING!!: If nan values in V2 are not removed, the statistics.median() is NOT correct.
            self.continuum_visibility2[key] = statistics.median(v2_removed_nan[index_lower_continuum_range:index_higher_continuum_range])
            
            #Continuum squared visibility error is given as its standard deviation
            self.continuum_visibility2_error[key] =np.std(v2_removed_nan[index_lower_continuum_range:index_higher_continuum_range])
        
            # IF number of nan in self.visibility2 is > 5% --> Raise WARNING
            if (len(self.visibility2)-len(v2_removed_nan))/len(self.visibility2)>0.05:
                print('')
                print('WARNING!!!!: number of nan in flux is > 5%')
                print('Results might be inconsistent!')
                print('')
        
        print('Estimation of continuum_visibility2:',self.continuum_visibility2)
 


        fig, ax = plt.subplots(3,2, figsize=(12, 14))
                
        # Set plot limits for continuum range measurement
        x_min_left=(continuum_range[0]-xlim[0])/(xlim[1]-xlim[0])    
        x_max_right=(continuum_range[1]-xlim[0])/(xlim[1]-xlim[0])
                

        
        # Plot subplots
        ax[0,0].plot(self.wl_A, self.visibility2['U2U1'], color="b",label='Mean (A,B)')
        ax[0,0].set_title(self.source + ' V2[U2U1]')
        ax[0,0].set_ylabel('$V^2$')
        ax[0,0].axhline(self.continuum_visibility2['U2U1'], xmin=x_min_left, xmax=x_max_right, color="grey", lw=1, ls='dotted')
        
        ax[0,1].plot(self.wl_A, self.visibility2['U3U1'], color="b",label='Mean (A,B)')
        ax[0,1].set_title(self.source + ' V2[U3U1]')
        ax[0,1].axhline(self.continuum_visibility2['U3U1'], xmin=x_min_left, xmax=x_max_right, color="grey", lw=1, ls='dotted')
        
        ax[1,0].plot(self.wl_A, self.visibility2['U3U2'], color="b",label='Mean (A,B)')
        ax[1,0].set_title(self.source +' V2[U3U2]')
        ax[1,0].set_ylabel('$V^2$')
        ax[1,0].axhline(self.continuum_visibility2['U3U2'], xmin=x_min_left, xmax=x_max_right, color="grey", lw=1, ls='dotted')
        
        ax[1,1].plot(self.wl_A, self.visibility2['U4U1'], color="b",label='Mean (A,B)')
        ax[1,1].set_title(self.source +' V2[U4U1]')
        ax[1,1].axhline(self.continuum_visibility2['U4U1'], xmin=x_min_left, xmax=x_max_right, color="grey", lw=1, ls='dotted')
        
        ax[2,0].plot(self.wl_A, self.visibility2['U4U2'], color="b",label='Mean (A,B)')
        ax[2,0].set_title(self.source +' V2[U4U2]')
        ax[2,0].set_xlabel('Wavelength ($\mu m$)')
        ax[2,0].set_ylabel('$V^2$')
        ax[2,0].axhline(self.continuum_visibility2['U4U2'], xmin=x_min_left, xmax=x_max_right, color="grey", lw=1, ls='dotted')
        
        ax[2,1].plot(self.wl_A, self.visibility2['U4U3'], color="b",label='Mean (A,B)')
        ax[2,1].set_title(self.source +' V2[U4U3]')
        ax[2,1].set_xlabel('Wavelength ($\mu m$)')
        ax[2,1].axhline(self.continuum_visibility2['U4U3'], xmin=x_min_left, xmax=x_max_right, color="grey", lw=1, ls='dotted')
        
        

       #For each subplot
        for i in range(len(ax)):
            for j in range(len(ax[i])):
                ax[i,j].set_xlim(xlim) 
                ax[i,j].set_ylim(ylim)
 
                ax[i,j].axvline(self.wl_central_line, color="green", lw=1, ls='-')
                ax[i,j].axvline(continuum_range[0], color="grey", lw=1, ls='--')
                ax[i,j].axvline(continuum_range[1], color="grey", lw=1, ls='--')
                ax[i,j].set_ylim(ylim)
                ax[i,j].set_xlim(xlim) 
                ax[i,j].set_ylabel('$V^2$')



        #Save figure
        if save_figure:
            #Save figure to disk
            fig.savefig('./figures/'+str(self.source) + "_figure_visibility_continuum_measurement" + ".eps", dpi=300)


        #Close figure if plot_figure=False
        if plot_figure==False:
            plt.close(fig)    # close the figure window
            

#-------------------------------------------------------------------------
    def figure_diff_phase_continuum_measurement(self, xlim=xlim(), ylim=ylim(),continuum_range=[2.141,2.1604], plot_figure=False, save_figure=False):
        """
        The continuum in differential phase (continuum_diff_phase) around our line of interest (𝐵𝑟𝛾) is measured. The following figure shows the wavelength range where the continuum is measured.        
        """

        #Calculate median value of differential phase in continuum_range         
        index_lower_continuum_range = min(range(len(self.wl_A)), key=lambda i: abs(self.wl_A[i]-continuum_range[0]))
        index_higher_continuum_range  = min(range(len(self.wl_A)), key=lambda i: abs(self.wl_A[i]-continuum_range[1]))
        
        self.continuum_diff_phase = {}
        self.continuum_diff_phase_error = {}
        
        for key, value in self.diff_phase.items():
            diff_phase_removed_nan = [x for x in value if ~np.isnan(x)]
            # WARNING!!: If nan values in diff_phase are not removed, the statistics.median() is NOT correct.
            self.continuum_diff_phase[key] = statistics.median(diff_phase_removed_nan[index_lower_continuum_range:index_higher_continuum_range])
        
            #continuum_diff_phase error is given as its standard deviation
            self.continuum_diff_phase_error[key] =np.std(diff_phase_removed_nan[index_lower_continuum_range:index_higher_continuum_range])

            # IF number of nan in self.diff_phase is > 5% --> Raise WARNING
            if (len(self.diff_phase)-len(diff_phase_removed_nan))/len(self.diff_phase)>0.05:
                print('')
                print('WARNING!!!!: number of nan in flux is > 5%')
                print('Results might be inconsistent!')
                print('')
        
        print(self.source, 'Estimation of continuum_diff_phase:',self.continuum_diff_phase,'+-', self.continuum_diff_phase_error)
        print('')
 


        fig, ax = plt.subplots(3,2, figsize=(12, 14))
                
        # Set plot limits for continuum range measurement
        x_min_left=(continuum_range[0]-xlim[0])/(xlim[1]-xlim[0])    
        x_max_right=(continuum_range[1]-xlim[0])/(xlim[1]-xlim[0])
                

        
        # Plot subplots
        ax[0,0].plot(self.wl_A, self.diff_phase['U2U1'], color="b",label='Mean (A,B)')
        ax[0,0].set_title(self.source + ' $\phi[U2U1]$')
        ax[0,0].set_ylabel('$\phi (^\circ)$',fontsize=12)
        ax[0,0].axhline(self.continuum_diff_phase['U2U1'], xmin=x_min_left, xmax=x_max_right, color="grey", lw=1, ls='dotted')
        
        ax[0,1].plot(self.wl_A, self.diff_phase['U3U1'], color="b",label='Mean (A,B)')
        ax[0,1].set_title(self.source + ' $\phi[U3U1]$')
        ax[0,1].axhline(self.continuum_diff_phase['U3U1'], xmin=x_min_left, xmax=x_max_right, color="grey", lw=1, ls='dotted')
        
        ax[1,0].plot(self.wl_A, self.diff_phase['U3U2'], color="b",label='Mean (A,B)')
        ax[1,0].set_title(self.source +' $\phi[U3U2]$')
        ax[1,0].set_ylabel('$\phi (^\circ)$',fontsize=12)
        ax[1,0].axhline(self.continuum_diff_phase['U3U2'], xmin=x_min_left, xmax=x_max_right, color="grey", lw=1, ls='dotted')
        
        ax[1,1].plot(self.wl_A, self.diff_phase['U4U1'], color="b",label='Mean (A,B)')
        ax[1,1].set_title(self.source +' $\phi[U4U1]$')
        ax[1,1].axhline(self.continuum_diff_phase['U4U1'], xmin=x_min_left, xmax=x_max_right, color="grey", lw=1, ls='dotted')
        
        ax[2,0].plot(self.wl_A, self.diff_phase['U4U2'], color="b",label='Mean (A,B)')
        ax[2,0].set_title(self.source +' $\phi[U4U2]$')
        ax[2,0].set_xlabel('Wavelength ($\mu m$)')
        ax[2,0].set_ylabel('$\phi (^\circ)$',fontsize=12)
        ax[2,0].axhline(self.continuum_diff_phase['U4U2'], xmin=x_min_left, xmax=x_max_right, color="grey", lw=1, ls='dotted')
        
        ax[2,1].plot(self.wl_A, self.diff_phase['U4U3'], color="b",label='Mean (A,B)')
        ax[2,1].set_title(self.source +' $\phi[U4U3]$')
        ax[2,1].set_xlabel('Wavelength ($\mu m$)')
        ax[2,1].axhline(self.continuum_diff_phase['U4U3'], xmin=x_min_left, xmax=x_max_right, color="grey", lw=1, ls='dotted')
        

       #For each subplot
        for i in range(len(ax)):
            for j in range(len(ax[i])):
                ax[i,j].set_xlim(xlim) 
                ax[i,j].set_ylim(ylim)
 
                ax[i,j].axvline(self.wl_central_line, color="green", lw=1, ls='-')
                ax[i,j].axvline(continuum_range[0], color="grey", lw=1, ls='--')
                ax[i,j].axvline(continuum_range[1], color="grey", lw=1, ls='--')
                ax[i,j].set_ylim(ylim)
                ax[i,j].set_xlim(xlim) 
                #ax[i,j].set_ylabel('$\phi (^\circ)$',fontsize=12)



        #Save figure
        if save_figure:
            #Save figure to disk
            fig.savefig('./figures/'+str(self.source)+ "_figure_diff_phase_continuum_measurement" + ".eps", dpi=300)

        #Close figure if plot_figure=False
        if plot_figure==False:
            plt.close(fig)    # close the figure window
            
#-------------------------------------------------------------------------

    def figure_closure_phase_continuum_measurement(self, xlim=xlim(), ylim=ylim(),continuum_range=[2.141,2.1604], plot_figure=False, save_figure=False):
        """
        The continuum in closure phase (continuum_closure_phase) around our line of interest (𝐵𝑟𝛾) is measured. The following figure shows the wavelength range where the continuum is measured.        
        """

        #Calculate median value of differential phase in continuum_range         
        index_lower_continuum_range = min(range(len(self.wl_A)), key=lambda i: abs(self.wl_A[i]-continuum_range[0]))
        index_higher_continuum_range  = min(range(len(self.wl_A)), key=lambda i: abs(self.wl_A[i]-continuum_range[1]))
        
        self.continuum_closure_phase = {}
        self.continuum_closure_phase_error = {}
        
        for key, value in self.phase_closure.items():
            closure_phase_removed_nan = [x for x in value if ~np.isnan(x)]
            # WARNING!!: If nan values in phase_closure are not removed, the statistics.median() is NOT correct.
            self.continuum_closure_phase[key] = statistics.median(closure_phase_removed_nan[index_lower_continuum_range:index_higher_continuum_range])
        
            #continuum_closure_phase error is given as its standard deviation
            self.continuum_closure_phase_error[key] =np.std(closure_phase_removed_nan[index_lower_continuum_range:index_higher_continuum_range])

            # IF number of nan in self.phase_closure is > 5% --> Raise WARNING
            if (len(self.phase_closure)-len(closure_phase_removed_nan))/len(self.phase_closure)>0.05:
                print('')
                print('WARNING!!!!: number of nan in flux is > 5%')
                print('Results might be inconsistent!')
                print('')
        
        print(self.source, 'Estimation of continuum_closure_phase:',self.continuum_closure_phase,'+-', self.continuum_closure_phase_error)
        print('')
 


        fig, ax = plt.subplots(2,2, figsize=(12, 12))
                
        # Set plot limits for continuum range measurement
        x_min_left=(continuum_range[0]-xlim[0])/(xlim[1]-xlim[0])    
        x_max_right=(continuum_range[1]-xlim[0])/(xlim[1]-xlim[0])
                

        
        # Plot subplots
        ax[0,0].plot(self.wl_A, self.phase_closure['U4U3U2'], color="b",label='Mean (A,B)')
        ax[0,0].set_title(self.source + ' $\phi[U4U3U2]$')
        ax[0,0].set_ylabel('$\phi (^\circ)$',fontsize=12)
        ax[0,0].axhline(self.continuum_closure_phase['U4U3U2'], xmin=x_min_left, xmax=x_max_right, color="grey", lw=1, ls='dotted')
        
        ax[0,1].plot(self.wl_A, self.phase_closure['U4U3U1'], color="b",label='Mean (A,B)')
        ax[0,1].set_title(self.source + ' $\phi[U4U3U1]$')
        ax[0,1].axhline(self.continuum_closure_phase['U4U3U1'], xmin=x_min_left, xmax=x_max_right, color="grey", lw=1, ls='dotted')
        
        ax[1,0].plot(self.wl_A, self.phase_closure['U4U2U1'], color="b",label='Mean (A,B)')
        ax[1,0].set_title(self.source +' $\phi[U4U2U1]$')
        ax[1,0].set_ylabel('$\phi (^\circ)$',fontsize=12)
        ax[1,0].axhline(self.continuum_closure_phase['U4U2U1'], xmin=x_min_left, xmax=x_max_right, color="grey", lw=1, ls='dotted')
        
        ax[1,1].plot(self.wl_A, self.phase_closure['U3U2U1'], color="b",label='Mean (A,B)')
        ax[1,1].set_title(self.source +' $\phi[U3U2U1]$')
        ax[1,1].axhline(self.continuum_closure_phase['U3U2U1'], xmin=x_min_left, xmax=x_max_right, color="grey", lw=1, ls='dotted')
        

       #For each subplot
        for i in range(len(ax)):
            for j in range(len(ax[i])):
                ax[i,j].set_xlim(xlim) 
                ax[i,j].set_ylim(ylim)
 
                ax[i,j].axvline(self.wl_central_line, color="green", lw=1, ls='-')
                ax[i,j].axvline(continuum_range[0], color="grey", lw=1, ls='--')
                ax[i,j].axvline(continuum_range[1], color="grey", lw=1, ls='--')
                ax[i,j].set_ylim(ylim)
                ax[i,j].set_xlim(xlim) 
                #ax[i,j].set_ylabel('$\phi (^\circ)$',fontsize=12)



        #Save figure
        if save_figure:
            #Save figure to disk
            fig.savefig('./figures/'+str(self.source)+ "_figure_closure_phase_continuum_measurement" + ".eps", dpi=300)

        #Close figure if plot_figure=False
        if plot_figure==False:
            plt.close(fig)    # close the figure window
            
#-------------------------------------------------------------------------

    def figure_continuum_corrector(self, xlim=xlim(), ylim=ylim(), plot_figure=False, save_figure=False):
        """
        The visibility and flux are corrected from continuum in the wavelength central line (normally Brg) using the continuum_corrector function. The differential phase in the line is also provided. 
        The final flux array is composed by final_continuum_flux_normalized in the whole bandwidth except in the line_range, where it is replaced by final_line_flux_normalized in order not to divide the line flux in the science observation by the line flux of the calibrator.
        """

        
        #Crop the continuum range to flux, visibility and phases
        cropped_final_flux = self.final_flux[self.index_lower_continuum_range_left:self.index_higher_continuum_range_right]
        
        cropped_wl = self.wl_A[self.index_lower_continuum_range_left:self.index_higher_continuum_range_right]
        cropped_visibility2 = {}
        cropped_diff_phase = {}
        for key, value in self.visibility2.items():
            cropped_visibility2[key] = self.visibility2[key][self.index_lower_continuum_range_left:self.index_higher_continuum_range_right]
            cropped_diff_phase[key] = self.diff_phase[key][self.index_lower_continuum_range_left:self.index_higher_continuum_range_right]
        
        
        self.line_visibility2 = {}
        line_phase = {}

        

        # Set visibility^2 and phase line values in dictionaries line_visibility2 and result_phasel
        for key, value in self.visibility2.items():
            self.line_visibility2[key], line_phase[key] = continuum_corrector(cropped_wl, cropped_final_flux, np.std(cropped_final_flux), cropped_visibility2[key], np.std(cropped_visibility2[key]), cropped_diff_phase[key], np.std(cropped_diff_phase[key]), 1, self.continuum_visibility2[key])
        
        

                  
        fig, ax = plt.subplots(3,2, figsize=(12, 14))
        
        # Plot subplots
        ax[0,0].axes.errorbar(self.line_visibility2['U2U1'][0], self.line_visibility2['U2U1'][1],yerr=self.line_visibility2['U2U1'][2], fmt='r*-',label='Continuum corrected $V^2$')
        ax[0,0].plot(self.wl_A, self.visibility2['U2U1'],'-bo',label='Measured $V^2$')
        #ax[0,0].plot(cropped_wl, self.line_visibility2['U2U1'][1], color="k",label='Line visibility2')
        ax[0,0].legend(loc=2) # upper left corner
        ax[0,0].set_title(self.source +' V2[U2U1]')
        ax[0,0].set_ylabel('$V^2$')
        
        ax[0,1].axes.errorbar(self.line_visibility2['U3U1'][0], self.line_visibility2['U3U1'][1],yerr=self.line_visibility2['U3U1'][2], fmt='r*-',label='Continuum corrected $V^2$')
        #ax[0,1].plot(cropped_wl, self.line_visibility2['U3U1'][1], color="b",label='Line visibility')
        ax[0,1].plot(self.wl_A, self.visibility2['U3U1'], '-bo',label='Measured $V^2$')                         
        ax[0,1].legend(loc=2) # upper left corner
        ax[0,1].set_title(self.source +' V2[U3U1]')
        
        ax[1,0].axes.errorbar(self.line_visibility2['U3U2'][0], self.line_visibility2['U3U2'][1],yerr=self.line_visibility2['U3U2'][2], fmt='r*-',label='Continuum corrected $V^2$')
        #ax[1,0].plot(cropped_wl, self.line_visibility2['U3U2'][1], color="b",label='Line visibility')
        ax[1,0].plot(self.wl_A, self.visibility2['U3U2'], '-bo',label='Measured $V^2$')                         
        ax[1,0].legend(loc=2) # upper left corner
        ax[1,0].set_title(self.source +' V2[U3U2]')
        ax[1,0].set_ylabel('$V^2$')
        
        ax[1,1].axes.errorbar(self.line_visibility2['U4U1'][0], self.line_visibility2['U4U1'][1],yerr=self.line_visibility2['U4U1'][2], fmt='r*-',label='Continuum corrected $V^2$')
        #ax[1,1].plot(cropped_wl, self.line_visibility2['U4U1'][1], color="b",label='Line visibility')
        ax[1,1].plot(self.wl_A, self.visibility2['U4U1'], '-bo',label='Measured $V^2$')                         
        ax[1,1].legend(loc=2) # upper left corner
        ax[1,1].set_title(self.source +' V2[U4U1]')
        
        ax[2,0].axes.errorbar(self.line_visibility2['U4U2'][0], self.line_visibility2['U4U2'][1],yerr=self.line_visibility2['U4U2'][2], fmt='r*-',label='Continuum corrected $V^2$')
        #ax[2,0].plot(cropped_wl, self.line_visibility2['U4U2'][1], color="b",label='Line visibility')
        ax[2,0].plot(self.wl_A, self.visibility2['U4U2'], '-bo',label='Measured $V^2$')                         
        ax[2,0].legend(loc=2) # upper left corner
        ax[2,0].set_title(self.source +' V2[U4U2]')
        ax[2,0].set_xlabel('Wavelength ($\mu m$)')
        ax[2,0].set_ylabel('$V^2$')
        
        ax[2,1].axes.errorbar(self.line_visibility2['U4U3'][0], self.line_visibility2['U4U3'][1],yerr=self.line_visibility2['U4U3'][2], fmt='r*-',label='Continuum corrected $V^2$')
        #ax[2,1].plot(cropped_wl, self.line_visibility2['U4U3'][1], color="b",label='Line visibility')
        ax[2,1].plot(self.wl_A, self.visibility2['U4U3'], '-bo',label='Measured $V^2$')                         
        ax[2,1].legend(loc=2) # upper left corner
        ax[2,1].set_title(self.source +' V2[U4U3]')
        ax[2,1].set_xlabel('Wavelength ($\mu m$)')
        
         
        #For each subplot
        for i in range(len(ax)):
            for j in range(len(ax[i])):
                ax[i,j].set_xlim(xlim) 
                ax[i,j].axvline(self.wl_central_line, color="green", lw=1, ls='--')
                ax[i,j].set_ylim(ylim)
        
        #Ensure no overlapping in plots
        fig.tight_layout()
        
        #Save figure
        if save_figure:
            #Save figure to disk
            fig.savefig('./figures/'+str(self.source) + "_continuum_corrector" + ".eps", dpi=300)

        #Close figure if plot_figure=False
        if plot_figure==False:
            plt.close(fig)    # close the figure window
            

#-------------------------------------------------------------------------

    def figure_3_plots(self, xlim=xlim(), flux_ylim=ylim(), visibility_ylim = [0,1], diff_phase_ylim = [-10,10], flux_yticks=[], visibility_yticks=[], diff_phase_yticks=[], plot_figure=False, save_figure=False):
        """
        This function presents a specific type of plots where for each baseline is shown its flux, squared visibility and differential phase as a function of wavelength in adjacent subplots for the central wavelength line.
        """

        fig = plt.figure(figsize=(8, 10))
        

        # Object HD 141926 (E) needs to be relabeled with AT telescopes labels
        HD141926_key = ['C1D0','C1B2','C1A0','D0B2','D0A0','B2A0']
        
        # gridspec inside gridspec
        outer_grid = gridspec.GridSpec(3, 2, wspace=0.3, hspace=0.02)
        j=0
        for  key, value in self.visibility2.items():
            inner_grid = gridspec.GridSpecFromSubplotSpec(3, 1,
                    subplot_spec=outer_grid[j], wspace=0.0, hspace=0.0)
            
            j=j+1
            
            #FLUX
            ax_flux = plt.Subplot(fig, inner_grid[0])
            ax_flux.plot(self.wl_A, self.final_flux, linestyle="solid", marker="o", markersize=3, color="red")
            fig.add_subplot(ax_flux)
            ax_flux.set_xlim(xlim)
            ax_flux.set_ylim(flux_ylim)
            ax_flux.set_yticks(flux_yticks)
            ax_flux.set_xticks([self.wl_central_line])#  <_____________________________________

            ax_flux.errorbar(self.wl_A[np.where(self.wl_A>xlim[0])][2], flux_yticks[-1],yerr = self.final_flux_error, fmt=',',  color="black", capsize=2, elinewidth=1, ecolor='black') #mfc='white',

            
            #ax_flux.axvspan(wl_max1_Brg, wl_min_Brg, alpha=0.3, color='blue')
            #ax_flux.axvspan(wl_min_Brg, wl_max2_Brg, alpha=0.3, color='red')  

            # Object HD 141926 (E) needs to be relabeled with AT telescopes labels
            if self.source == 'HD141926':
                ax_flux.set_title('['+HD141926_key[j-1]+']',x=0.8,y=0.7, fontsize=10) 
                #ax_flux.set_title(self.source+' ['+str(j)+']',x=0.8,y=0.7, fontsize=12) 

            else:
                ax_flux.set_title('['+key+']',x=0.8,y=0.7, fontsize=10) 
            
            #ax_flux.set_xlabel('Wavelength ($\mu m$)',fontsize=12)
            ax_flux.set_ylabel('Flux',fontsize=12)
        
            #VISIBILITY
            ax_visibility = plt.Subplot(fig, inner_grid[1])
            ax_visibility.plot(self.wl_A, self.visibility2[key],linestyle="solid", marker="o", markersize=3, color="blue")
            fig.add_subplot(ax_visibility)
            ax_visibility.set_ylim(visibility_ylim)
            ax_visibility.set_yticks(visibility_yticks)
            #ax_visibility.axvspan(wl_max1_Brg, wl_min_Brg, alpha=0.3, color='blue')
            #ax_visibility.axvspan(wl_min_Brg, wl_max2_Brg, alpha=0.3, color='red')  
            ax_visibility.set_xlim(xlim)
            ax_visibility.set_ylabel('$V^2$',fontsize=12)
            #ax_visibility.set_xlabel('Wavelength ($\mu m$)',fontsize=12)
            
            ax_visibility.errorbar(self.wl_A[np.where(self.wl_A>xlim[0])][2], visibility_yticks[-1], yerr = self.continuum_visibility2_error[key], fmt=',', color="black", capsize=2, elinewidth=1, ecolor='black') #mfc='white',
            
            ax_visibility.set_xticks([self.wl_central_line])
             
            #DIFFERENTIAL PHASE
            ax_diff_phase = plt.Subplot(fig, inner_grid[2])
            ax_diff_phase.plot(self.wl_A, self.diff_phase[key],linestyle="solid", marker="o", markersize=3, color="green")
            ax_diff_phase.errorbar(self.wl_A[np.where(self.wl_A>xlim[0])][2], diff_phase_yticks[-1], yerr = self.continuum_diff_phase_error[key], fmt=',',  color="black", capsize=2, elinewidth=1, ecolor='black') #mfc='white',

            
            fig.add_subplot(ax_diff_phase)
            ax_diff_phase.set_xlim(xlim)
            ax_diff_phase.set_ylim(diff_phase_ylim)
            ax_diff_phase.set_yticks(diff_phase_yticks)
            ax_diff_phase.set_xticks([2.160, 2.165, 2.170])
            #ax_diff_phase.axvspan(wl_max1_Brg, wl_min_Brg, alpha=0.3, color='blue')
            #ax_diff_phase.axvspan(wl_min_Brg, wl_max2_Brg, alpha=0.3, color='red')  
            ax_diff_phase.set_xlabel('Wavelength ($\mu m$)',fontsize=12)
            ax_diff_phase.set_ylabel('$\phi (^\circ)$',fontsize=12)
        
        
        all_axes = fig.get_axes()
        
        #Ensure no overlapping in plots
        fig.tight_layout()
        
        
        #Save figure
        if save_figure:
            #Save figure to disk
            fig.savefig('./figures/'+str(self.source) + "_figure_3_plots.eps", dpi=300)            

        #Close figure if plot_figure=False
        if plot_figure==False:
            plt.close(fig)    # close the figure window
            

#-------------------------------------------------------------------------

    def figure_full_bandwidth_flux(self, xlim=xlim(), ylim=ylim(), zoom_lower_xlim = 2.163, zoom_higher_xlim = 2.170, zoom_ylim = ylim(), plot_figure=False, save_figure=False):
        """
        This plot shows the full bandwidth of the science observation and a embeded zoom-in picture of the 𝐵𝑟𝛾 region. The full picture includes science and calibrator relative in that region.
        """

        
        fig, ax = plt.subplots(figsize=(20,10))
        
        ax.plot(self.wl_A, self.BPcorrected_flux, 'r')
        #ax.plot(self.wl_A, self.final_flux)
        ax.set_xlabel('Wavelength ($\mu m$)',fontsize=14)
        ax.set_ylabel('Flux',fontsize=14)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.axvline(self.wl_central_line, color="grey", lw=1, ls='--')
        ax.text(self.wl_central_line+0.002,1000,'$Br\gamma$', color="grey",fontsize=14)
        ax.tick_params(labelsize=14)
       
        fig.tight_layout()
        
        # inset
        inset_ax = fig.add_axes([0.56, 0.6, 0.4, 0.35]) # X, Y, width, height
        
        # Identify the index corresponding to the zoomed interval
        index_zoom_lower_xlim = min(range(len(self.wl_A)), key=lambda i: abs(self.wl_A[i]-zoom_lower_xlim))
        index_zoom_higher_xlim  = min(range(len(self.wl_A)), key=lambda i: abs(self.wl_A[i]-zoom_higher_xlim))
        
            
        # plot zoom frame
        inset_ax.plot(self.wl_A, self.BPcorrected_flux, '-ro')
        inset_ax.set_title('zoom at $Br\gamma$',fontsize=14)
        #inset_ax.axhline(1, color="green", lw=1, ls='--')
        #inset_ax.set_ylabel('Flux',fontsize=14)
        inset_ax.axvline(self.wl_central_line, color="grey", lw=1, ls='--')

        # set axis range
        inset_ax.set_xlim(zoom_lower_xlim,zoom_higher_xlim)
        inset_ax.set_ylim(zoom_ylim)
        
        # set axis tick locations
        inset_ax.set_yticks(np.arange(11000, 18000,2000))
        inset_ax.set_xticks(np.arange(zoom_lower_xlim,zoom_higher_xlim,0.002));
        inset_ax.tick_params(labelsize=14)
                
                
        #Save figure
        if save_figure:
            #Save figure to disk
            fig.savefig('./figures/'+str(self.source) + "_full_bandwidth_flux" + ".eps", dpi=300)


        #Close figure if plot_figure=False
        if plot_figure==False:
            plt.close(fig)    # close the figure window
            

#-------------------------------------------------------------------------
    def figure_V2_vs_spatial_frecuency(self,xlim=xlim(), ylim=ylim(), plot_figure=False, save_figure=False):
        """
        In this function is represented V² as a function of the spatial frecuency 𝐵/𝜆 [𝑚/𝜇𝑚] given in [rad] units.
        
        """
        
        # Read fitting data from automeris .csv file  
        # (in the LITpro_output directory)  
        LITpro_continuum_visibility2 = pd.read_csv("./LITpro_output/"+self.source+"_cont_LITpro_fit.tsv", sep='\t', header=3)
        LITpro_Brg_visibility2 = pd.read_csv("./LITpro_output/"+self.source+"_Brg_LITpro_fit.tsv", sep='\t', header=3)
        
        LITpro_continuum_visibility2=LITpro_continuum_visibility2.sort_values(by=['#x'])
        wave_LITpro = LITpro_continuum_visibility2['#x'].to_numpy()
        modelValue_LITpro = LITpro_continuum_visibility2['modelValue'].to_numpy()
        
        LITpro_Brg_visibility2=LITpro_Brg_visibility2.sort_values(by=['#x'])
        #wave_LITpro = LITpro_Brg_visibility2['#x'].to_numpy()
        #modelValue_LITpro = LITpro_Brg_visibility2['modelValue'].to_numpy()
        
        # Read fitting data from LITpro .tsv file  
        # (in the same directory than gravi_plot_genereator.py)
        fitted_continuum_visibility2 = pd.read_csv("./LITpro_output/"+self.source+"_automeris_cont.csv", sep=';', header=0)
        fitted_Brg_visibility2 = pd.read_csv("./LITpro_output/"+self.source+"_automeris_Brg.csv", sep=';', header=0)

        #Sort, just in case 
        fitted_continuum_visibility2=fitted_continuum_visibility2.sort_values(by=['X_data'])
        fitted_Brg_visibility2=fitted_Brg_visibility2.sort_values(by=['X_data'])
        
        wave_cont = fitted_continuum_visibility2['X_data'].to_numpy()
        modelValue_cont = fitted_continuum_visibility2['Y_data'].to_numpy()

        wave_Brg = fitted_Brg_visibility2['X_data'].to_numpy()
        modelValue_Brg = fitted_Brg_visibility2['Y_data'].to_numpy()

        fig, ax = plt.subplots(figsize=(10, 10))
        
        i=0
        for  key, value in self.visibility2.items():
            #Generate interpolated model fitting values
            ax.errorbar(self.baseline[key]*1000000/self.wl_central_line, self.continuum_visibility2[key], yerr=self.continuum_visibility2_error[key], c='black',fmt='o', capthick=2, label='$V^2$ continuum data' if i == 0 else "")
            
            if i == 0:
                ax.plot(wave_cont,modelValue_cont, '--', color='black',label='Fit to $V^2$ continuum model')
                ax.plot(wave_Brg,modelValue_Brg, '-.', color='green',label='Fit to $V^2 Br\gamma$ model')
                #ax.scatter(fitted_continuum_visibility2[['#x']], fitted_continuum_visibility2[['dataValue']],label=' $V^2$ Data')
                ax.scatter(LITpro_continuum_visibility2[['#x']], LITpro_continuum_visibility2[['modelValue']],label='$V^2$ continuum model', color='grey', marker='$o$')
                ax.scatter(LITpro_Brg_visibility2[['#x']], LITpro_Brg_visibility2[['modelValue']],label='$V^2 \: Br\gamma$ model', color='green', marker='+')

            i=i+1
                 
        ring_width = 1 #  in mas           
        ring_width_rad = ring_width/(206264.8*1000) # in radians        
        
        B_over_lambda = np.linspace(min(wave_Brg), max(wave_Brg), 150)
        
        
        ax.plot(B_over_lambda,(scipy.special.j0(2*np.pi*ring_width_rad*B_over_lambda))**2, '-.', color='black',label='elongated ring model')
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(str(self.source))
        ax.set_xlabel('$B/\lambda \: (rad^{-1}$)', size=16)
        ax.set_ylabel('$V²$', size=16)
        ax.legend(loc=0,fontsize=14)        
        ax.tick_params(labelsize=14)
                
        #Ensure no overlapping in plots
        fig.tight_layout()

        
        #Save figure
        if save_figure:
            #Save figure to disk
            fig.savefig('./figures/'+str(self.source) + "_V2_vs_spatial_frecuency" + ".eps", dpi=300)


        #Close figure if plot_figure=False
        if plot_figure==False:
            plt.close(fig)    # close the figure window
            
            
#-------------------------------------------------------------------------    

    def write_to_files(self, output_file_name=[]):    
        """
          WRITTING to *.fits* files using PYFITS
         Open input FITS files and writting modifications in their "_copy.fits" to use in modelling.
         Merge, Bad Pixel corrections, continuum correction and further modifications needs to be set in a FITS file
         in order to be retrieved in the LITpro modelling.
         
        
        """


        #Open again one of the input files and copy to a different dataset
        input_file = pyfits.open(self.inputpath_A + self.filename_A, mode='readonly')
        input_file_copy = input_file.copy()
                
        #Set modified variables
        for hdu in input_file_copy :
                header = hdu.header.copy()                
                if 'INSNAME' in header:
                    
                    if (hdu.name == 'OI_FLUX') & ('_SC' in str(hdu.header['INSNAME'])):
                        hdu.data.field('FLUX')[0] = self.final_flux
                        hdu.data.field('FLUX')[1] = self.final_flux
                        hdu.data.field('FLUX')[2] = self.final_flux
                        hdu.data.field('FLUX')[3] = self.final_flux

                    if (hdu.name == 'OI_VIS2') & ('_SC' in str(hdu.header['INSNAME'])):
                        hdu.data.field('VIS2DATA')[0]=self.visibility2['U4U3']
                        hdu.data.field('VIS2DATA')[1]=self.visibility2['U4U2']
                        hdu.data.field('VIS2DATA')[2]=self.visibility2['U4U1']
                        hdu.data.field('VIS2DATA')[3]=self.visibility2['U3U2']
                        hdu.data.field('VIS2DATA')[4]=self.visibility2['U3U1']
                        hdu.data.field('VIS2DATA')[5]=self.visibility2['U2U1']

                    if (hdu.name == 'OI_VIS') & ('_SC' in str(hdu.header['INSNAME'])):
                        hdu.data.field('VISPHI')[0]=self.diff_phase['U4U3']
                        hdu.data.field('VISPHI')[1]=self.diff_phase['U4U2']
                        hdu.data.field('VISPHI')[2]=self.diff_phase['U4U1']
                        hdu.data.field('VISPHI')[3]=self.diff_phase['U3U2']
                        hdu.data.field('VISPHI')[4]=self.diff_phase['U3U1']
                        hdu.data.field('VISPHI')[5]=self.diff_phase['U2U1']


#            self.oi_vis_sc_visphi = oi_vis_sc.field('VISPHI')*np.pi/180. # conversion in radians (after Sept. 16)
#            self.oi_vis_sc_visphierr = oi_vis_sc.field('VISPHIERR')*np.pi/180. # conversion in radians (after Sept. 16)
#       
#    
        
        #write to disk in same folder as input file
        if output_file_name == []:
            input_file_copy.writeto('./FITS/copy_' + self.source + '.fits', clobber=True)
        else:
            input_file_copy.writeto('./FITS/' +self.source + '_' + output_file_name + '.fits', clobber=True)

        #Some cleaning
        input_file.close()
        input_file_copy.close()
        del input_file
        del input_file_copy



#-------------------------------------------------------------------------
    def figure_diff_phase_vs_spatial_frecuency(self, plot_figure=False, save_figure=False):
        """
        Figure showing the variation of the mean value of the differential phase
        in the continuum around Brg.
        """

        fig, ax = plt.subplots()
           
        for  key, value in self.continuum_diff_phase.items():
                      
             # Plot subplots
            ax.axes.errorbar(self.baseline[key]*1000000/self.wl_central_line, self.continuum_diff_phase[key],yerr= self.continuum_diff_phase_error[key] , fmt='r*-',label='Mean (A,B)')
            ax.set_title(self.source + ' Continuum differential phase')
            ax.set_ylabel('$\phi (^\circ)$',fontsize=12)
            ax.set_xlabel('$B/\lambda \: (rad^{-1}$)',fontsize=12)
        
        #Save figure
        if save_figure:
            #Save figure to disk
            fig.savefig('./figures/'+str(self.source) + "_diff_phase_vs_spatial_frecuency" + ".png")#, dpi=300)

        #Close figure if plot_figure=False
        if plot_figure==False:
            plt.close(fig)    # close the figure window

            

#-------------------------------------------------------------------------

    def figure_line_visibility2_calculation(self, line_range = [2.164,2.167], plot_range = [2.16,2.17], 
                                            plot_figure=False, save_figure=False):
        """
        This function calculates the mean squared visibility in a given wavelength range (line_range) for the line.
        It also calculates its error as the standard deviation of that value in the range.
        It is also plotted the squared visibility in a different given wavelength range (plot_range). 
        
        Input:
        -line_range = wavelength range for the calculation of the line squared visibility
        -plot_range = wavelength range for the x-axis plot
        
        Output:
        -line_visibility2 = dict with the mean squared visibility of the line for each baseline. 
        -line_visibility2_error = dict with the  squared visibility error of the line for each baseline. 
        
        """
        
        #Calculate index values for the lower and higher crop range
        index_lower_plot_range = min(range(len(self.wl_A)), key=lambda i: abs(self.wl_A[i]-plot_range[0]))
        index_higher_plot_range = min(range(len(self.wl_A)), key=lambda i: abs(self.wl_A[i]-plot_range[1]))       
            
        #crop wavelength    
        cropped_wl = self.wl_A[index_lower_plot_range:index_higher_plot_range]
            
        #Crop the plot range 
        cropped_visibility2 = {}
        for key, value in self.visibility2.items():
            cropped_visibility2[key] = self.visibility2[key][index_lower_plot_range:index_higher_plot_range]
            
        #Calculate index values for the lower and higher line range
        index_lower_line_range  = min(range(len(cropped_wl)), key=lambda i: abs(cropped_wl[i]-line_range[0]))
        index_higher_line_range = min(range(len(cropped_wl)), key=lambda i: abs(cropped_wl[i]-line_range[1]))       
        
        #Calculate the line_visibility2 as the mean value in the cropped line range
        self.line_visibility2 = {}
        self.line_visibility2_error = {}
                    
        print('Star: {0}'.format(str(self.source)))

        for key, value in self.visibility2.items():
            self.line_visibility2[key] = np.mean(cropped_visibility2[key][index_lower_line_range:index_higher_line_range])    
            self.line_visibility2_error[key] = np.std(cropped_visibility2[key])    
            print(' Key:{0:<4} | mean line visibility = {1:<20}+-{2} '.format(str(key),self.line_visibility2[key],self.line_visibility2_error[key]))
        print('')
        fig, ax = plt.subplots(3,2, figsize=(12, 14))
        
        # Plot subplots
        ax[0,0].plot(cropped_wl, cropped_visibility2['U2U1'], color="b",label='Line visibility2')
        ax[0,0].legend(loc=2) # upper left corner
        ax[0,0].set_title(self.source +' V2[U2U1]')
        ax[0,0].set_ylabel('$V^2$')
        
        ax[0,1].plot(cropped_wl, cropped_visibility2['U3U1'], color="b",label='Line visibility2')
        ax[0,1].legend(loc=2) # upper left corner
        ax[0,1].set_title(self.source +' V2[U3U1]')
        
        ax[1,0].plot(cropped_wl, cropped_visibility2['U3U2'], color="b",label='Line visibility2')
        ax[1,0].legend(loc=2) # upper left corner
        ax[1,0].set_title(self.source +' V2[U3U2]')
        ax[1,0].set_ylabel('$V^2$')
        
        ax[1,1].plot(cropped_wl, cropped_visibility2['U4U1'], color="b",label='Line visibility2')
        ax[1,1].legend(loc=2) # upper left corner
        ax[1,1].set_title(self.source +' V2[U4U1]')
        
        ax[2,0].plot(cropped_wl, cropped_visibility2['U4U2'], color="b",label='Line visibility2')
        ax[2,0].legend(loc=2) # upper left corner
        ax[2,0].set_title(self.source +' V2[U4U2]')
        ax[2,0].set_xlabel('Wavelength ($\mu m$)')
        ax[2,0].set_ylabel('$V^2$')
        
        ax[2,1].plot(cropped_wl, cropped_visibility2['U4U3'], color="b",label='Line visibility2')
        ax[2,1].legend(loc=2) # upper left corner
        ax[2,1].set_title(self.source +' V2[U4U3]')
        ax[2,1].set_xlabel('Wavelength ($\mu m$)')
        
         
        #For each subplot
        for i in range(len(ax)):
            for j in range(len(ax[i])):
                ax[i,j].set_xlim(plot_range) 
                ax[i,j].axvline(self.wl_central_line, color="green", lw=1, ls='--')
                ax[i,j].axvline(line_range[0], color="grey", lw=1, ls='-')
                ax[i,j].axvline(line_range[1], color="grey", lw=1, ls='-')
                #ax[i,j].set_ylim(ylim)
        
       

        #Save figure
        if save_figure:
            #Save figure to disk
            fig.savefig('./figures/'+str(self.source) + "_line_visibility2" + ".eps", dpi=300)

        #Close figure if plot_figure=False
        if plot_figure==False:
            plt.close(fig)    # close the figure window
            
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
    def figure_u_v_coverage(self, plot_figure=False, save_figure=False):
        """
        Plot u,v coverages for each pair of the observations we have of each source.
        """
        fig, ax = plt.subplots(figsize=(8,8))

        #Plot scatter (u,v) coverage
        for  key, value in self.uV2_A.items():
            ax.scatter( self.uV2_A[key],self.vV2_A[key],  marker="o", color='blue', facecolor="none", s=100, label='Obs. A' if key == 'U4U3' else "")
            ax.scatter( self.uV2_B[key],self.vV2_B[key],  marker="o", color='red' , facecolor="none", s=100, label='Obs. B' if key == 'U4U3' else "")
            ax.scatter(-self.uV2_A[key],-self.vV2_A[key], marker="o", color='blue', facecolor="none", s=100)
            ax.scatter(-self.uV2_B[key],-self.vV2_B[key], marker="o", color='red' , facecolor="none", s=100)
        
            ax.axhline(0, color="black", lw=1, ls='-')
            ax.axvline(0, color="black", lw=1, ls='-')
            ax.set_xlabel('u (m)', fontsize = 16)
            ax.set_ylabel('v (m)', fontsize = 16)
            ax.set_xlim([-140,140])
            ax.set_ylim([-140,140])
            ax.legend(loc=2,fontsize=14) # upper left corner
            ax.tick_params(labelsize=14)
        
        ax.set_title(str(self.source), fontsize=16) 

        #Save figure
        if save_figure:
            #Save figure to disk
            fig.savefig('./figures/'+str(self.source) + "_u_v_coverage" + ".eps", dpi=300)

        #Close figure if plot_figure=False
        if plot_figure==False:
            plt.close(fig)    # close the figure window
            
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
#    def figure_flux_preprocessing(self, plot_figure=False, save_figure=False):
#        """
#        
#        """
#
#        #Save figure
#        if save_figure:
#            #Save figure to disk
#            fig.savefig('./figures/'+str(self.source) + "_V2_Brg" + ".eps", dpi=300)
#
#        #Close figure if plot_figure=False
#        if plot_figure==False:
#            plt.close(fig)    # close the figure window
#            
#-------------------------------------------------------------------------
