############################################################################################################################################################
############################################################################################################################################################

# Program name: calibration.py
# Author: Luca Teruzzi, PhD in Physics, Astrophysics and Applied Physics, University of Milan, Milan (IT)
#         EuroCold Lab, Department of Earth and Environmental Sciences, University of Milano-Bicocca, Milan (IT)
# Date: 01 May 2022 (last modified)
# Objective: Abakus laser sensor calibration
# Description: This Python script aims at computing the instrumental calibration curve for Abakus laser sensor.
#              From a practical standpoint, some measurements were performed with colloidal suspensions of calibrated polystirene nanoparticles (known refractive index
#              @ 670 nm) with variable diameters of 1.0, 1.5, 2.0, 2.9, 3.75, 5.0 and 5.75 μm.
#              The calibration procedure is based on Mie scattering theroy for perfectly spherical and homogeneous particles: the optical extinction cross section
#              is computed and the resulting curve is inverted in terms of the (extinction) diameter.
#              Then, a comparison in made between the measured diameters and the theoretical ones: their ratios defines the desired calibration curve, that is 
#              the function for which the measured diameter divided by the calibration function corresponds exactly to the extinction diameter.
#              The calibration curve is retireved through a polynomial fit and saved on a text file for future analyses.

############################################################################################################################################################
############################################################################################################################################################


import numpy as np, math as m, matplotlib.pyplot as plt, os                               # Import the required libraries
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d


############################################################################################################################################################
############################################################################################################################################################


def f(x, *p): return np.poly1d(p)(x)


############################################################################################################################################################


wavelength = 0.670
ref_index_Re = 1.58                                                                          # Set the refractive index real and imaginary part 
ref_index_Im = 0
n_med = 1.3310
sizes = np.array([1.0, 1.3, 1.6, 1.9, 2.2, 2.5, 2.8, 3.1, 3.4, 3.7, 4.0, 4.3, 4.6, 4.9, 5.2, 5.5, 5.8, 6.1, 6.4, 6.7, 7.0, 7.3, 7.6, 7.9, 8.2, 8.5, 8.8, 9.1, 9.4, 9.7, 10.0, 10.3])
h = 0

if ref_index_Im != 0: file = open('../LUT_Cext/LUT_Cext_l='+'{:.02f}'.format(wavelength)+'um_nmed='+'{:.04f}'.format(n_med)+'_m=[1.0001+'+'{:.04f}'.format(ref_index_Im)+'j-1.9534+'+'{:.04f}'.format(ref_index_Im)+'j].txt', 'r')
else: file = open('../LUT_Cext/LUT_Cext_l='+'{:.02f}'.format(wavelength)+'um_nmed='+'{:.04f}'.format(n_med)+'_m=[1.0001-1.9534].txt', 'r')

m_polystirene = np.round(ref_index_Re/n_med, 4)                                                           # Polystirene relative refractive index, rounded to the 4th decimal value

m = m_polystirene                                                                 # Relative refractive index, rounded to the 4th decimal value

diameters_Cext, m_Cext, lines_Cext, Cext = [], [], file.readlines(), []                             # The first row is taken apart since it contains
diameters_Cext.append(np.array([float(i) for i in lines_Cext[0].split('\t')[2:] if i.strip()]))     # the particle diameters; the other ones are converted to float and 
diameters_Cext = diameters_Cext[0]                                                                  # appended to the corresponding list
for x in lines_Cext[1:]: 
    Cext.append(np.array([complex(i) for i in x.split('\t') if i.strip()]))
    h += 1

for j in range(0, len(Cext)): 
    m_Cext.append(Cext[j][0])
    Cext[j] = np.real(Cext[j][1:])

diameters_idx = []
for i in range(0, len(sizes)): diameters_idx.append(np.where(diameters_Cext==round(sizes[i], 2))[0][0])

polystirene_idx = np.where(np.real(m_Cext)==m_polystirene.real)[0]                                  # Find when the row corresponding to polystirene refractive index 
Cext_polystirene = Cext[polystirene_idx[0]]
_Cext_polystirene = uniform_filter1d(Cext_polystirene, size=150)
Cext_polystirene_cfr = _Cext_polystirene[diameters_idx]

idx = np.where(np.real(m_Cext)==m.real)[0]                                                          # Find when the experimental refractive index is equal to some 
if len(idx) > 1:                                                                                    # value ammong the LUT ones
    selected_Cext = 0                                                                               # If more than one is found, the average Cext is computed
    for i in range(0, len(idx)): selected_Cext += Cext[idx[i]]  
    selected_Cext = selected_Cext/len(idx)
else: 
    selected_Cext = Cext[idx[0]]
idx = 0

n_range = np.array([1.42, 1.46, 1.47, 1.48, 1.50, 1.51, 1.52, 1.53, 1.56, 1.58, 1.64])
s_range = np.array([100, 125, 150, 150, 150, 150, 150, 125, 125, 100, 100])
poly_coefficients = np.polyfit(n_range, s_range, 3)
_poly_fit = np.poly1d(poly_coefficients)

size_avg = _poly_fit(ref_index_Re)

poly_fit = interp1d(uniform_filter1d(selected_Cext, size=int(size_avg)), diameters_Cext, kind='linear', fill_value='extrapolate') 
selected_Cext = interp1d(diameters_Cext, selected_Cext, kind='linear', fill_value='extrapolate')

true_pos = np.array([1.0, 1.8, 2.9, 3.7, 5, 10])
false_pos = np.array([1.05, 2.5, 3.7, 4.1, 5.8, 10])
false_pos_dev = np.array([0.1, 0.3, 0.2, 0.3, 0.3, 0.2])
false_pos_lower, false_pos_upper = false_pos-1.2*false_pos_dev, false_pos+1.2*false_pos_dev
selected_Cext_interp = interp1d(poly_fit(selected_Cext(diameters_Cext)), selected_Cext(diameters_Cext), kind='linear', fill_value='extrapolate')

sigma = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
p1, _ = curve_fit(f, false_pos, selected_Cext_interp(true_pos), (0, 0, 0, 0, 0), sigma=sigma)

cal_curve = f(diameters_Cext, *p1)

for k in range(0, 2):

    x_plot = diameters_Cext[90:np.where(diameters_Cext>=10)[0][0]]

    f, ax = plt.subplots(1, 1, figsize=(9, 6))
    ax.set_ylabel('C$_{ext}$ [$\mathrm{\mu}$m$^2$]', fontsize=20)
    ax.set_xlabel('$d$ [$\mathrm{\mu}$m]', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.semilogx(x_plot, selected_Cext(x_plot), linewidth=2, color='darkblue', label='Mie')
    ax.semilogx(poly_fit(selected_Cext(diameters_Cext[110:1000])), selected_Cext(diameters_Cext[110:1000]), 'springgreen', linewidth=2,  label='Mie (smoothed)')
    ax.semilogx(x_plot, cal_curve[90:np.where(diameters_Cext>=10)[0][0]], 'r--', linewidth=2, label='calibration curve')
    ax.scatter(true_pos[:-1], selected_Cext_interp(true_pos[:-1]), linewidth=2, marker='^', facecolor='None', edgecolor='b', s=150, label='expected')
    ax.scatter(false_pos[:-1], selected_Cext_interp(true_pos[:-1]), linewidth=2, marker='o', facecolor='w', edgecolor='r', s=150, label='measured')
    for kk in range(0, len(false_pos)): ax.semilogx([false_pos_lower[kk], false_pos_upper[kk]], [selected_Cext_interp(true_pos[kk]), selected_Cext_interp(true_pos[kk])], 'r', linewidth=1.5)
    ax.legend(loc='best', ncol=2, prop={'size': 14})
    if k==0: ax.set_xscale('log'); ax.set_yscale('log')
    elif k==1: ax.set_xscale('log'); ax.set_yscale('linear')
    f.tight_layout()

plt.show()

current_directory = os.path.abspath(os.path.realpath(__file__))[2:-22].replace('\\', '/')
file = open(current_directory+'_calibration/calibration_curve.txt', 'w')                    # Save data on .txt file for future analyses
for i in range(0, len(x_plot)):
    file.write('{:.02f}'.format(x_plot[i])+'\t'+'{:.04f}'.format((cal_curve[90:np.where(diameters_Cext>=10)[0][0]])[i])+'\n')  
file.close()


############################################################################################################################################################
############################################################################################################################################################
