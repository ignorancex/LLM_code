from matplotlib.widgets import Slider, Button, RadioButtons
import corner
import sys
#import PySimpleGUI as psg 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np 
import matplotlib.pyplot as plt
import cmasher as cmr
import pandas as pd 
import matplotlib

plt.rcParams["font.family"] = "Helvetica"

def discrete_cmap(CMAP, bins):
    cmap = plt.get_cmap(CMAP, bins)
    colours = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]
    return colours

def invisible_axis(ax):
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax.tick_params(axis="both", which="both", bottom=False, top=False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    
def open_fig_with_slider(chains):
    plt.figure(1)
    slider = Slider(plt.axes([0.25, 0.1, 0.65, 0.03]),
                    'Slider', -0.1, 0.1, valinit=0.)
    
    plt.show()
#fname = 'chains/seed/CurrentnovollimNonecpl_offset0.0omDev0.0None200_2000-post_equal_weights.dat'
#fname = 'chains/testZTFzmin0.1ZTFnovollimNonecpl_offset1.3877787807814457e-17omExtomDev0.0None_200-post_equal'

fname = 'fiducial_chains.dat'
fid = np.loadtxt(fname)
#open_fig_with_slider(fid)
fig, (ax1, sax) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]},
                                   figsize=(7,7))



#l = (1 - np.exp(-0.5), 1 - np.exp(-2))
l = (0.68, 0.95)

lgth = 0.1
wdth = 0.05

SAVAS = discrete_cmap('cmr.chroma_r', 9)
invisible_axis(sax)
# define the set of systematics sliders

sax.text(x=0.43, y=0.08, s=r"Calibration",
             weight='heavy', fontsize=13, color='dimgray')
sax.text(x=0.49, y=0.05, s=r"(Mag)",
             weight='heavy', fontsize=13, color='dimgray')
calib_slider = Slider(plt.axes([0.8, 0.1, lgth, wdth]),
                    '', -0.1, 0.1, valinit=0., color=SAVAS[1],
                          orientation='horizontal', track_color='white', initcolor='dimgray',
                          handle_style={'facecolor':'k', 'edgecolor':'k',
                                            '':'|', 'edgewidth':2})

sax.text(x=0.46, y=0.19, s=r"d$R_V$/d$z$",
             weight='heavy', fontsize=13, color='dimgray')
rv_slider = Slider(plt.axes([0.8, 0.2, lgth, wdth]),
                    '', -2, 2, valinit=0., color=SAVAS[2],
                       orientation='horizontal', track_color='white', initcolor='dimgray',
                       handle_style={'facecolor':'k', 'edgecolor':'k',
                                          '':'|', 'edgewidth':2})

sax.text(x=0.43, y=0.33, s="MW E(B-V):",
             weight='heavy', fontsize=11, color='dimgray')
sax.text(x=0.43, y=0.3, s="CCM/F99",
             weight='heavy', fontsize=11, color='dimgray')
mwccm_slider = Slider(plt.axes([0.8, 0.3, lgth, wdth]),
                    '', 0, 1, valinit=0., valstep=1, color=SAVAS[3],
                          orientation='horizontal', track_color='white', initcolor='dimgray',
                          handle_style={'facecolor':'k', 'edgecolor':'k',
                                             '':'|', 'edgewidth':2})

sax.text(x=0.52, y=0.44, s=r"$\sigma (\beta)$",
             weight='heavy', fontsize=13, color='dimgray')
sigmbeta_slider = Slider(plt.axes([0.8, 0.4, lgth, wdth]),
                    '', 0, 0.2, valinit=0., color=SAVAS[4],
                             orientation='horizontal', track_color='white', initcolor='dimgray',
                             handle_style={'facecolor':'k', 'edgecolor':'k',
                                                '':'|', 'edgewidth':2})


sax.text(x=0.43, y=0.57, s=r"log(r$\Omega_{IG}$)",
             weight='heavy', fontsize=13, color='dimgray')
igdust_slider = Slider(plt.axes([0.8, 0.5, lgth, wdth]),
                    '', -10, -4., valinit=-10., color=SAVAS[5],
                           orientation='horizontal', track_color='white', initcolor='dimgray',
                           handle_style={'facecolor':'k', 'edgecolor':'k',
                                              '':'|', 'edgewidth':2})

sax.text(x=0.43, y=0.73, s=r"Progenitor",
             weight='heavy', fontsize=13, color='dimgray')
sax.text(x=0.43, y=0.7, s=r"Evolution ($x_1$)",
             weight='heavy', fontsize=13, color='dimgray')
progevol_slider = Slider(plt.axes([0.8, 0.6, lgth, wdth]),
                    ' ', 0.5, 3., valinit=2.3, color=SAVAS[6],
                             orientation='horizontal', track_color='white', initcolor='dimgray',
                             handle_style={'facecolor':'white', 'edgecolor':'white',
                                                '':'|', 'edgewidth':2})

sax.text(x=0.43, y=0.87, s=r"$\Omega_M$",
             weight='heavy', fontsize=13, color='dimgray')
sax.text(x=0.43, y=0.83, s=r"MisMatch",
             weight='heavy', fontsize=13, color='dimgray')
om_slider = Slider(plt.axes([0.8, 0.7, lgth, wdth]),
                    ' ', -0.1, 0.1, valinit=0, color=SAVAS[7],
                             orientation='horizontal', track_color='white', initcolor='dimgray',
                             handle_style={'facecolor':'white', 'edgecolor':'white',
                                                '':'|', 'edgewidth':2})

sax.text(x=0.43, y=0.98, s=r"Progenitor",
             weight='heavy', fontsize=13, color='dimgray')
sax.text(x=0.43, y=0.95, s=r"Evolution ($M_B$)",
             weight='heavy', fontsize=13, color='dimgray')
progevol_mb_slider = Slider(plt.axes([0.8, 0.8, lgth, wdth]),
                    ' ', -0.1, 0.1, valinit=0, color=SAVAS[8],
                             orientation='horizontal', track_color='white', initcolor='dimgray',
                             handle_style={'facecolor':'white', 'edgecolor':'white',
                                                '':'|', 'edgewidth':2})

#button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
shift_dict = pd.read_pickle('shifts.pkl')

def update(val):
    r = calib_slider.val
    r2 = rv_slider.val 
    r3 = mwccm_slider.val
    r4 = sigmbeta_slider.val 
    r5 = igdust_slider.val 
    r6 = progevol_slider.val 
    r7 = om_slider.val 
    r8 = progevol_mb_slider.val 

    fac1 = shift_dict['None0.02'][0] * (r / 0.02) 
    fac2 = shift_dict['None0.02'][1] * (r / 0.02)
    print(fac1, fac2)

    #ax1.tick_params(labelsize=14, direction='in', length=7, width=3)

    ax1.cla()
    # for the dRv/dz slope how much does w0-wa change (the scaling is to the dmu/dz slope)
    # this is because dRV/ dz -> dmu/dz
    fac3 =  shift_dict['RV-z'][0] * (r2 / 0.5) 
    fac4 =  shift_dict['RV-z'][1] * (r2 / 0.5) 
    
    fac5 = shift_dict['MWExt'][0] * r3
    fac6 = shift_dict['MWExt'][1] * r3

    fac_smooth = 2.31 * r4 + 0.412
    
    # for the IG Dust value, the slope is a large number since the shift is (
    # (quasi-) linear with the dust density not its log. However, the step here is in log density
    # so need to take the exponent of the value
    fac7 = shift_dict['IG_Dust5e-06'][0] * pow(10, r5)
    fac8 = shift_dict['IG_Dust5e-06'][1] * pow(10, r5)

    fac9 =  - (r6 - 2.3) * shift_dict['prog_evol_x10.8'][0] / 0.1
    fac10 =  (r6 - 2.3) * shift_dict['prog_evol_x10.8'][1] / 0.1
    
    fac11 = - (r7 / 0.02) * shift_dict['prog_evol_mb0.02'][0]#0.02
    fac12 = - (r7 / 0.02) * shift_dict['prog_evol_mb0.02'][1]

    fac13 = - (r8 / 0.01) * shift_dict['None0.0omMiss0.01'][0] 
    fac14 = - (r8 / 0.01) * shift_dict['None0.0omMiss0.01'][1]

    fid_fac1_test = fid[:,3] + fac1 + fac3 + fac5 + fac7 + fac9 + fac11 + fac13 
    fid_fac2_test = fid[:,4]  + fac2 + fac4 + fac6  + fac8 + fac10 + fac12 + fac14 

    fid_fac1 = (fid_fac1_test - np.median(fid_fac1_test)) * (5.04 * r4 + 0.805) + np.median(fid_fac1_test)
    fid_fac2 = (fid_fac2_test - np.median(fid_fac2_test))* (5.04 * r4 + 0.805)    + np.median(fid_fac2_test)
    
    #This is where the stuff is plotted
    #Aesthetics are launched here.

    #"cmr.savanna_r"
    GREENS_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("", ['#ffffff', "#225339"])

    #The contour levels themselves in green
    corner.hist2d(fid_fac1, fid_fac2, smooth=1, levels=l, ax=ax1,
                      color="#225339", zorder=2, 
                      plot_datapoints=False, plot_density=False)

    #Hexbins underneath to show population
    ax1.hexbin(fid_fac1, fid_fac2, cmap=GREENS_CMAP,
                   extent=(-1.6,-0.3,-2,1.7), gridsize=(30,20),
                   zorder=1)

    #These are the center points and the DESI val respectively 
    ax1.plot(np.median(fid[:,3]), np.median(fid[:,4]),
                 c='k', marker='x', ms=10, zorder= 5) #median Val
    ax1.text(x=np.median(fid[:,3])+0.05, y=np.median(fid[:,4])+0.05,
                 s=r"($\Lambda$CDM)", fontsize=12, zorder=100, c='k')

    ax1.plot(-0.827, -0.75, c='darkgoldenrod', marker='.', ms=15,
                 zorder=5) #DESI val
    ax1.text(x=-0.827+0.05, y=-0.75+0.05, s="DESI", color='darkgoldenrod',
                 fontsize=12, zorder=100)


    #End plotting, start aesthetics-ing
    ax1.spines[['bottom', 'left']].set_linewidth(0.5)
    ax1.spines[['bottom', 'left']].set_color('slategrey')
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.set_xlabel(r'$w_0$', fontsize=15, c='slategrey')
    ax1.set_ylabel(r'$w_a$', fontsize=15, c='slategrey')
    ax1.set_xlim(np.min(fid[:,3])-0.5, np.max(fid[:,3]) + 0.5)
    ax1.set_ylim(np.min(fid[:,4])-0.5, np.max(fid[:,4]) + 0.5)

    ax1.tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False)
    ax1.set_xticks(ticks=np.arange(-1.7, -0.2, 0.2), labels=np.around(np.arange(-1.7, -0.2, 0.2),2), fontsize=15, color='slategrey')
    ax1.set_yticks(ticks=np.arange(-2, 2, 0.5), labels=np.around(np.arange(-2, 2, 0.5),2), fontsize=15, color='slategrey')

    fig.canvas.draw()
    return fac1, fac2

axcolor = 'darkseagreen'

button = Button(plt.axes([0.8, 0.025, 0.1, 0.04]), 'Reset', color=axcolor, hovercolor='0.975')

def resetplt(event):
    calib_slider.reset()
    rv_slider.reset()
    mwccm_slider.reset()
    sigmbeta_slider.reset()
    igdust_slider.reset()
    progevol_slider.reset()
    om_slider.reset()
    progevol_mb_slider.reset() 

button.on_clicked(resetplt)

calib_slider.on_changed(update)
rv_slider.on_changed(update)
mwccm_slider.on_changed(update)
sigmbeta_slider.on_changed(update)
igdust_slider.on_changed(update)
progevol_slider.on_changed(update)
om_slider.on_changed(update)
progevol_mb_slider.on_changed(update)


calib_slider.label.set_size(12)
rv_slider.label.set_size(12)
mwccm_slider.label.set_size(12)
sigmbeta_slider.label.set_size(12)
igdust_slider.label.set_size(12)
progevol_slider.label.set_size(12)
om_slider.label.set_size(12)
progevol_mb_slider.label.set_size(12)

plt.show()
