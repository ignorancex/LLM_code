#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thanks for the help of programming from Gu Hongrui, Huang Qichen, Meng Weiyu, Xiao Kai!!
Author       : Huang Bowen
Date         : 2024-02-17
Version      : 1.0
E-mail       : HuangBW@mail.bnu.edu.cn
Description  : An example for package GaiaDR3XPspectracorrectionV1

"""
from GaiaDR3XPspectracorrectionV1 import Gaia_Correction_V1
import os
import numpy as np
import pandas as pd
from gaiaxpy import calibrate
import matplotlib.pyplot as plt
#%% load example spectra
pat = __file__
flux_origin = calibrate(os.path.join(os.path.dirname(pat),'example_file_coeff.csv'),truncation=(False),save_file=(False))
G = np.array([4.825453,6.073419,12.515343,12.460801,13.47115,11.766908,5.327651,11.995801,12.28695,12.040052])
flux_origin = flux_origin[0]
gaia_wl = np.linspace(3360,3360+20*342,343)
#%% Correction
flux_out = Gaia_Correction_V1.correction_df(flux_origin,G,have_error=True,Truncation=False,absolute_correction=True)
#%% Plot
plt.figure()
plt.plot(gaia_wl,flux_out.flux_cor[0],color='b')
plt.plot(gaia_wl,flux_out.flux[0],color='r')
plt.xlabel('Wavelength Å')
plt.ylabel('Flux')
plt.xlim([3360,10200])
plt.ylim([0,max(flux_out.flux[0])*1.1])
plt.legend({'Original Flux','Corrected Flux'})




