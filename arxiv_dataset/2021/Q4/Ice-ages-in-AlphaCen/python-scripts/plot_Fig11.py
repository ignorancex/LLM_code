import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from scipy.interpolate import UnivariateSpline

rcParams.update({'font.size': 24})

def colorbar(mappable):
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax,orientation='vertical')
    plt.sca(last_axes)
    return cbar

os.chdir('..')
home = os.getcwd() + "/"

plot_dir = home + "Figs/"
data_dir = home + "data/"

ax_loc = np.array([[0.125,0.885],[0.55,0.885],[0.125,0.455],[0.55,0.455]])

aspect = 1.5
width = 13.
lw = 4
fs = 'x-large'
clabel = ["","$\Delta \epsilon$", "$\Delta T_s$", "$\Delta f_{ice}$", "$\Delta$  albedo"]
sublabel = ['a','b','c','d']

fig = plt.figure(1,figsize=(aspect*width,width),dpi=300)
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)
ax4 = fig.add_subplot(234)
ax5 = fig.add_subplot(235)
ax6 = fig.add_subplot(236)

ax_list = [ax1,ax2,ax3,ax4,ax5,ax6]
sub_lbl = ['a','b','c','d','e','f']

ip_list = [2,10,30]

for i in range(0,6):
    i_p = ip_list[i % 3]
    data = np.genfromtxt(data_dir+"VPlanet_data_B_map_%i.txt" % (i_p),delimiter=',',comments='#')
    data = data[np.isfinite(data[:,2]),:]
    data = data[data[:,0]>0,:]
    
    if i < 3:
        data[data[:,4]==0,4]=0.01
        Q = np.log10(data[:,4])
        cmap = cm.plasma
        vmin, vmax = -2,0
    else:
        cmap = cm.gnuplot
        data[data[:,5]==0,5]=0.01
        Q = np.log10(data[:,5])
        vmin, vmax = -2, np.log10(0.3)
    eps = data[:,0]
    alpha = data[:,1]

    xi = np.arange(0,91,1)
    yi = np.arange(0,101,1)
    zi = griddata((eps,alpha),Q,(xi[None,:],yi[:,None]),method = 'nearest')

    my_cmap = cm.get_cmap(cmap)
    norm = colors.Normalize(vmin,vmax)
    cmmapable = cm.ScalarMappable(norm,my_cmap)
    cmmapable.set_array(range(0,1))
    
    ax = ax_list[i]
    CS = ax.pcolormesh(xi,yi+0.5,zi,cmap = cmap,vmin=vmin,vmax=vmax)
    ax.plot(23,46,color='w',marker='$\oplus$',ms=25)
    
    ax.set_xticks(np.arange(0,105,15))
    ax.set_yticks(np.arange(0,120,20))
    ax.set_ylim(0,101)
    ax.set_xlim(0,90)
    ax.tick_params(axis='both', direction='out',length = 8.0, width = 8.0,labelsize=fs)
    ax.text(0.02,0.90,sub_lbl[i], color='k',fontsize='x-large',weight='bold',horizontalalignment='left',transform=ax.transAxes)
    if i_p == 2:
        ax.set_ylabel('$\\gamma$ ('+u'\u2033'+'/yr)', fontsize=fs)
    else:
        ax.set_yticklabels([])
    if i < 3:
        ax.set_xticklabels([])
        ax.text(0.5,1.05,"%i$^\circ$" % i_p, color='k',fontsize='x-large',weight='bold',horizontalalignment='center',transform=ax.transAxes)
    else:
        ax.set_xlabel("$\epsilon_o$ (deg.)",fontsize=fs)
    if i == 2:
        color_label=r'$\Delta f_{\rm ice}$'
        cbar=colorbar(CS)
        cbar.set_label(color_label,fontsize=fs)
        ctick_rng = np.concatenate((np.arange(0.01,0.1,0.01),np.arange(0.1,1.1,0.1)))
        cticks = np.log10(ctick_rng)
        cticklabels = ["0.01"]
        for l in range(0,16):
            cticklabels.append("")
            if l == 7:
                cticklabels.append("0.1")
        cticklabels.append("1")
        cbar.ax.yaxis.set_ticks(cticks)
        cbar.ax.yaxis.set_ticklabels(cticklabels)
        cbar.ax.tick_params(axis='both', direction='out',length = 8.0, width = 8.0,labelsize=fs)
    if i == 5:
        color_label=r'$\Delta \alpha$'
        cbar=colorbar(CS)
        cbar.set_label(color_label,fontsize=fs)
        ctick_rng = np.concatenate((np.arange(0.01,0.1,0.01),[0.1,0.2,0.3]))
        cticks = np.log10(ctick_rng)
        cticklabels = ["0.01"]
        for l in range(0,9):
            cticklabels.append("")
            if l == 7:
                cticklabels.append("0.1")
        cticklabels.append("0.3")
        cbar.ax.yaxis.set_ticks(cticks)
        cbar.ax.yaxis.set_ticklabels(cticklabels)
        cbar.ax.tick_params(axis='both', direction='out',length = 8.0, width = 8.0,labelsize=fs)


fig.subplots_adjust(hspace=0.1,wspace=0.15)

fig.savefig(plot_dir+"Fig11.png",bbox_inches='tight',dpi=300)
plt.close()