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

aspect = 0.5
width = 13.
lw = 4
fs = 'x-large'
clabel = ["","$\Delta \epsilon$", "$\Delta T_s$", "$\Delta f_{ice}$", "$\Delta$  albedo"]
sublabel = ['a','b','c','d']

fig = plt.figure(1,figsize=(aspect*width,1.5*width),dpi=300)
ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)

ax_list = [ax1,ax2,ax3,ax4]
sub_lbl = ['a','b','c','d']

ip_list = [2,10,30]
i_p = 10
data = np.genfromtxt(data_dir+"VPlanet_data_A_map_%i.txt" % (i_p),delimiter=',',comments='#')
data = data[np.isfinite(data[:,2]),:]
data = data[data[:,0]>0,:]

for i in range(0,4):
    if i == 0:
        Q = data[:,2]
        cmap = cm.gist_rainbow
        vmin, vmax = 0,45
    elif i == 1:
        cmap = cm.magma
        data[data[:,3]==0,3]=0.01
        Q = np.log10(data[:,3])
        vmin, vmax = 0, np.log10(10)
    elif i == 2:
        data[data[:,4]==0,4]=0.01
        Q = np.log10(data[:,4])
        cmap = cm.plasma
        vmin, vmax = -2,-0.6
    else:
        cmap = cm.gnuplot
        data[data[:,5]==0,5]=0.01
        Q = np.log10(data[:,5])
        vmin, vmax = -2, np.log10(0.1)
    eps = data[:,0]
    alpha = data[:,1]
    print(np.max(Q))

    xi = np.arange(0,91,1)
    yi = np.arange(0,101,1)
    zi = griddata((eps,alpha),Q,(xi[None,:],yi[:,None]),method = 'nearest')

    my_cmap = cm.get_cmap(cmap)
    norm = colors.Normalize(vmin,vmax)
    cmmapable = cm.ScalarMappable(norm,my_cmap)
    cmmapable.set_array(range(0,1))
    
    ax = ax_list[i]
    CS = ax.pcolormesh(xi,yi+0.5,zi,cmap = cmap,vmin=vmin,vmax=vmax,shading='auto')
    ax.plot(23,10,color='w',marker='$\oplus$',ms=25)
    
    ax.set_xticks(np.arange(0,105,15))
    ax.set_yticks(np.arange(0,120,20))
    ax.set_ylim(0,101)
    ax.set_xlim(0,90)
    ax.tick_params(axis='both', direction='out',length = 8.0, width = 8.0,labelsize=fs)
    ax.text(0.02,0.86,sub_lbl[i], color='k',fontsize='x-large',weight='bold',horizontalalignment='left',transform=ax.transAxes)
    ax.set_ylabel('$\\gamma$ ('+u'\u2033'+'/yr)', fontsize=fs)

    if i < 3:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("$\epsilon_o$ (deg.)",fontsize=fs)
    if i == 0:
        color_label=r'$\Delta \epsilon$'
        cbar=colorbar(CS)
        cbar.set_label(color_label,fontsize=fs)
        cticks = np.arange(0,60,15)
        cbar.ax.yaxis.set_ticks(cticks)
        cbar.ax.tick_params(axis='both', direction='out',length = 8.0, width = 8.0,labelsize=fs)
    elif i == 1:
        color_label=r'$\Delta T$'
        cbar=colorbar(CS)
        cbar.set_label(color_label,fontsize=fs)
        #ctick_rng = np.array([1,2,5,10,20,40])
        ctick_rng = np.array([1,2,5,10])
        cticks = np.log10(ctick_rng)
        cticklabels = ["%i" % f for f in ctick_rng]
        cbar.ax.yaxis.set_ticks(cticks)
        cbar.ax.yaxis.set_ticklabels(cticklabels)
        cbar.ax.tick_params(axis='both', direction='out',length = 8.0, width = 8.0,labelsize=fs)
    elif i == 2:
        color_label=r'$\Delta f_{\rm ice}$'
        cbar=colorbar(CS)
        cbar.set_label(color_label,fontsize=fs)
        #ctick_rng = np.concatenate((np.arange(0.01,0.1,0.01),np.arange(0.1,1.1,0.1)))
        ctick_rng = np.concatenate((np.arange(0.01,0.1,0.01),np.arange(0.1,0.3,0.1)))
        cticks = np.log10(ctick_rng)
        cticklabels = ["0.01"]
        for l in range(0,9):
            cticklabels.append("")
            if l == 7:
                cticklabels.append("0.1")
        #cticklabels.append("1")
        cbar.ax.yaxis.set_ticks(cticks)
        cbar.ax.yaxis.set_ticklabels(cticklabels)
        cbar.ax.tick_params(axis='both', direction='out',length = 8.0, width = 8.0,labelsize=fs)
    else:
        color_label=r'$\Delta \alpha$'
        cbar=colorbar(CS)
        cbar.set_label(color_label,fontsize=fs)
        #ctick_rng = np.concatenate((np.arange(0.01,0.1,0.01),[0.1,0.2,0.3]))
        ctick_rng = np.concatenate((np.arange(0.01,0.1,0.01),[0.1]))
        cticks = np.log10(ctick_rng)
        cticklabels = ["0.01"]
        for l in range(0,8):
            cticklabels.append("")
            if l == 7:
                cticklabels.append("0.1")
        #cticklabels.append("0.3")
        cbar.ax.yaxis.set_ticks(cticks)
        cbar.ax.yaxis.set_ticklabels(cticklabels)
        cbar.ax.tick_params(axis='both', direction='out',length = 8.0, width = 8.0,labelsize=fs)

fig.subplots_adjust(hspace=0.2,wspace=0.15)

fig.savefig(plot_dir+"Fig4.png",bbox_inches='tight',dpi=300)
plt.close()