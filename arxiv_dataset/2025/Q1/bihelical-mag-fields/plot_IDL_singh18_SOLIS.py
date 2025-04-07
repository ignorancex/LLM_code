"""
Read and plot Nishant's saved IDL data, just as a sanity check.
At least that matches figure 2 of the 2018 paper.
"""
import numpy as np
from scipy.io import readsav
from spectrum import signed_loglog_plot
import matplotlib.pyplot as plt

sav_2148 = readsav("nishant_idl_scripts/sav2/tE_kH_2148.sav")
sav_2149 = readsav("nishant_idl_scripts/sav2/tE_kH_2149.sav")
sav_2150 = readsav("nishant_idl_scripts/sav2/tE_kH_2150.sav")
sav_2151 = readsav("nishant_idl_scripts/sav2/tE_kH_2151.sav")

khk1 = np.average(np.stack([sav['khk1'] for sav in [sav_2148, sav_2149, sav_2150, sav_2151]]), axis=0)
k = sav_2148['k']

fig,ax = plt.subplots()
signed_loglog_plot(k,-np.imag(khk1),ax)

plt.show()
