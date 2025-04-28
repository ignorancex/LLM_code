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

os.chdir('..')
home = os.getcwd() + "/"

plot_dir = home + "Figs/"
data_dir = home + "data/"

def colorbar(mappable,x0,y0):
    last_axes = plt.gca()
    ax = mappable.axes
    fig1 = ax.figure
    cax = fig1.add_axes([x0,y0,0.35,0.015])
    cbar = fig1.colorbar(mappable, cax=cax,orientation='horizontal')
    cbar.ax.xaxis.set_ticks_position('top')
    plt.sca(last_axes)
    return cbar

def plot_map(ax,ax_right,Q,cmap,vmin,vmax,ax_idx):
    if ax_idx > 1:
        Q[Q==0] = 1e-6
    if ax_idx == 2:
        Q = np.log10(Q)
        vmin = 0
        vmax = np.log10(60)
    if ax_idx == 3:
        Q = np.log10(Q)
        vmin = -2
        vmax = 0
    if ax_idx == 4:
        Q = np.log10(Q)
        vmin = -2
        vmax = np.log10(0.3)
    xi = np.arange(0,91,1)
    yi = np.arange(0,101,1)
    zi = griddata((eps,alpha),Q,(xi[None,:],yi[:,None]),method = 'nearest')
    my_cmap = cm.get_cmap(cmap).copy()
    norm = colors.Normalize(vmin,vmax)
    cmmapable = cm.ScalarMappable(norm,my_cmap)
    cmmapable.set_array(range(0,1))

    
    CS = ax.pcolormesh(xi,yi,zi,cmap = my_cmap,vmin=vmin,vmax=vmax,shading='auto')
    cbar = colorbar(CS,ax_loc[ax_idx-1,0],ax_loc[ax_idx-1,1])

    ax.plot(23.4,10,color='w',marker='$\oplus$',ms=25)
    if ax_idx == 2:
        ctick_rng = np.array([1,2,5,10,20,40])
        cticks = np.log10(ctick_rng)
        cticklabels = ["%i" % f for f in ctick_rng]
        cbar.ax.xaxis.set_ticks(cticks)
        cbar.ax.xaxis.set_ticklabels(cticklabels)
    if ax_idx ==3:
        ctick_rng = np.concatenate((np.arange(0.01,0.1,0.01),np.arange(0.1,1.1,0.1)))
        cticks = np.log10(ctick_rng)
        cticklabels = ["0.01"]
        for l in range(0,16):
            cticklabels.append("")
            if l == 7:
                cticklabels.append("0.1")
        cticklabels.append("1")
        cbar.ax.xaxis.set_ticks(cticks)
        cbar.ax.xaxis.set_ticklabels(cticklabels,fontsize=10)
    if ax_idx ==4:
        ctick_rng = np.concatenate((np.arange(0.01,0.1,0.01),[0.1,0.2,0.3]))
        cticks = np.log10(ctick_rng)
        cticklabels = ["0.01"]
        for l in range(0,9):
            cticklabels.append("")
            if l == 7:
                cticklabels.append("0.1")
        cticklabels.append("0.3")
        cbar.ax.xaxis.set_ticks(cticks)
        cbar.ax.xaxis.set_ticklabels(cticklabels,fontsize=10)
    if ax_idx == 1:
        cticks = np.arange(0,105,15)
        cbar.ax.xaxis.set_ticks(cticks)
    cbar.ax.tick_params(axis='both', direction='out',length = 8.0, width = 8.0,labelsize='large')
    ax.text(0.02,0.88,sublabel[ax_idx-1],color='k',fontsize=fs,horizontalalignment='left',transform=ax.transAxes)
    ax.set_xticks(np.arange(0,105,15))
    ax.set_ylim(0,101)
    ax.set_xlim(0,90)
    
    ax_right.set_ylim(0,101)
    ax_right.set_yticks(per_ticks)
    ax_right.xaxis.set_visible(False)
    ax_right.tick_params(axis='both', direction='out',length = 8.0, width = 8.0,color='#8a8a8a')
    if ax_idx == 2 or ax_idx == 4:
        ax_right.set_yticklabels(['%i' % p for p in periods],color='gray')
        ax_right.set_ylabel("Rotation Period (hr)",color='gray',fontsize=fs)
    else:
        ax_right.set_yticklabels([])
    
    if ax_idx == 1 or ax_idx == 3:
        ax.set_ylabel('$\\alpha$ ('+u'\u2033'+'/yr)', fontsize=fs)
    if ax_idx == 2 or ax_idx == 4:
        ax.set_yticklabels([])
    if ax_idx >2:
        ax.set_xlabel("$\epsilon_o$ (deg.)",fontsize=fs)
    else:
        ax.set_xticklabels([])
    ax.set_yticks(np.arange(0,120,20))
    ax.tick_params(axis='both', direction='out',length = 8.0, width = 8.0,labelsize=fs)

ax_loc = np.array([[0.125,0.885],[0.55,0.885],[0.125,0.455],[0.55,0.455]])

aspect = 1.
width = 13.
lw = 4
fs = 'x-large'
clabel = ["","$\Delta \epsilon$", "$\Delta T_s$", "$\Delta f_{ice}$", "$\Delta$  albedo"]
sublabel = ['a','b','c','d']

fig = plt.figure(1,figsize=(aspect*width,width),dpi=300)
ax1 = fig.add_subplot(221)
ax1_right = ax1.twinx()
ax2 = fig.add_subplot(222)
ax2_right = ax2.twinx()
ax3 = fig.add_subplot(223)
ax3_right = ax3.twinx()
ax4 = fig.add_subplot(224)
ax4_right = ax4.twinx()


i_p = 10

data = np.genfromtxt(data_dir+"VPlanet_data_A_map_%i.txt" % (i_p),delimiter=',',comments='#')
data = data[np.isfinite(data[:,2]),:]
data = data[data[:,0]>0,:]
data[data[:,4]==0.007,4] = 0.01

eps = data[:,0]
alpha = data[:,1]

plot_map(ax1,ax1_right,data[:,2],"gist_rainbow",0,90,1)
plot_map(ax2,ax2_right,data[:,3],"magma",0,60,2)
plot_map(ax3,ax3_right,data[:,4],"plasma",0,1,3)
plot_map(ax4,ax4_right,data[:,5],"gnuplot",0,1,4)


fig.subplots_adjust(hspace=0.25,wspace=0.2)
fig.savefig(plot_dir+"Fig10.png",bbox_inches='tight',dpi=300)
plt.close()