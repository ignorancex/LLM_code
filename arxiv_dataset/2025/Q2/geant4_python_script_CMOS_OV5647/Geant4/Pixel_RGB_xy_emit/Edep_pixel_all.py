#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 20:14:02 2020

@author: mik
"""

import numpy as np
import matplotlib.pyplot as plt
import re

# numerical data file
file_cs137_pixel = "/media/mik/EAC2174FC2171EFF/1e7evt_Cs137_edep_pixel.csv"
dat_cs137_pixel = np.loadtxt(file_cs137_pixel, delimiter="\t", skiprows=1)

file_sr90_pixel = "/media/mik/EAC2174FC2171EFF/1e7evt_Sr90_edep_pixel.csv"
dat_sr90_pixel = np.loadtxt(file_sr90_pixel, delimiter="\t", skiprows=1)

file_fe55_pixel = "/media/mik/EAC2174FC2171EFF/1e7evt_Fe55_edep_pixel.csv"
dat_fe55_pixel = np.loadtxt(file_fe55_pixel, delimiter="\t", skiprows=1)

plt.figure(figsize=(20,10))
plt.title('Edep Pixel', fontsize=26)

plt.hist(dat_cs137_pixel, bins = 100, histtype="step", color='tab:blue', linewidth=2., label="Cs137")

plt.hist(dat_sr90_pixel, bins = 100, histtype="step", color='tab:red', linewidth=2., label="Sr90")

plt.hist(dat_fe55_pixel, bins = 100, histtype="step", color='tab:green', linewidth=2., label="Fe55")

plt.yscale('log')
plt.grid(True)
plt.xticks(size=26)
plt.yticks(size=26)
plt.xlabel('Energy (keV)',  fontsize=26)
plt.ylabel('Count',  fontsize=26)
plt.legend(loc='best', fontsize=26)
plt.tight_layout()

#plt.savefig("/media/mik/EAC2174FC2171EFF/1e7evt_edep_pixel_Cs137.png")
#plt.savefig("/media/mik/EAC2174FC2171EFF/1e7evt_edep_pixel_Sr90.png")
#plt.savefig("/media/mik/EAC2174FC2171EFF/1e7evt_edep_pixel_Fe55.png")
plt.savefig("/media/mik/EAC2174FC2171EFF/1e7evt_edep_pixel_all.png")

plt.show()
#plt.clf()