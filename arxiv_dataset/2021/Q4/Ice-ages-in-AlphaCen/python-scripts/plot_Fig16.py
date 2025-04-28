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

rcParams.update({'font.size': 22})

def colorbar(mappable,ax):
    pos = ax.get_subplotspec().get_position(fig)
    cb_width = 0.02
    cb_height = pos.y1-pos.y0
    vertical_position = pos.y0
    horizontal_position = 0.905
    cax = fig.add_axes([horizontal_position, vertical_position, cb_width, cb_height])
    clb = fig.colorbar(cmmapable,cax=cax,orientation='vertical')
    return clb

os.chdir('..')
home = os.getcwd() + "/"

plot_dir = home + "Figs/"
data_dir = home + "data/"

ax_loc = np.array([[0.125,0.885],[0.55,0.885],[0.125,0.455],[0.55,0.455]])

aspect = 16./9.
width = 10
lw = 4
fs = 'x-large'
clabel = ["","$\Delta \epsilon$", "$\Delta T_s$", "$\Delta f_{ice}$", ]
sublabel = ['a','b','c','d']

fig = plt.figure(1,figsize=(aspect*width,2*width),dpi=300)
ax1 = fig.add_subplot(611)
ax2 = fig.add_subplot(612)
ax3 = fig.add_subplot(613)
ax4 = fig.add_subplot(614)
ax5 = fig.add_subplot(615)
ax6 = fig.add_subplot(616)


ax_list = [ax1,ax2,ax3,ax4,ax5,ax6]
sub_lbl = ['a','b','c','d','e','f']

data = np.genfromtxt(data_dir+"VPlanet_data_GenBin_map_10.txt",delimiter=',',comments='#')
data[data[:,3]>90,2] = np.inf
data[data[:,6]==3,7] = 3

for i in range(0,6):
    ax = ax_list[i]
    if i == 0:
        Q = data[:,2]
        cmap = "gist_rainbow"
        vmin, vmax = 0,90
    elif i == 1:
        cmap = "magma"
        data[data[:,3]==0,3]=0.01
        Q = np.log10(data[:,3])

        vmin, vmax = 0, np.log10(60)
    elif i == 2:
        data[data[:,4]==0,4]=0.01
        Q = np.log10(data[:,4])
        cmap = "plasma"
        vmin, vmax = -2,0
    elif i == 3:
        Q = data[:,-4]
        cmap = colors.ListedColormap(['k', 'gray', 'b', 'r'])
        vmin, vmax = -0.5,3
    elif i == 4:
        Q = data[:,-2]
        cmap = cm.get_cmap("plasma").copy()
        vmin, vmax = 0.001,0.4
    else:
        Q = data[:,-1]
        cmap = cm.get_cmap("plasma").copy()
        vmin, vmax = 0.001,0.4
    #Q_cut = np.where(~np.isfinite(data[:,2]))[0]
    #Q[Q_cut] = -1.
    abin = data[:,0]
    ebin = data[:,1]
    

    xi = np.arange(10,91,1)
    yi = np.arange(0,.91,0.01)
    zi = griddata((abin,ebin),Q,(xi[None,:],yi[:,None]),method = 'nearest') 

    my_cmap = cm.get_cmap(cmap).copy()
    norm = colors.Normalize(vmin,vmax)
    cmmapable = cm.ScalarMappable(norm,my_cmap)
    cmmapable.set_array(range(0,1))
    my_cmap.set_over('w')
    if i > 3:
        my_cmap.set_under('k')
        my_cmap.set_over('r')
    
    if i == 3:
        bounds = [0.,1.,2.,3.,4.]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        CS = ax.pcolormesh(xi,yi+0.005,zi,cmap = my_cmap,norm=norm,shading='auto',zorder=2)
        switch_states = np.zeros(len(abin))
        for k in range(0,len(abin)):
            if data[k,-3] != data[k,-4]:
                switch_states[k] = 1
        #ax.plot(data[switch_states==1,0],data[switch_states==1,1]+0.005,'wx',ms=4,zorder=4,alpha=0.75)
        switch_rows = np.where(switch_states==1)[0]
        ax.plot(data[switch_rows,0],data[switch_rows,1],'m+',ms=6,zorder=4)
    else:
        CS = ax.pcolormesh(xi,yi+0.005,zi,cmap = my_cmap,vmin=vmin,vmax=vmax,shading='auto')
    #ax.plot(23,46,color='w',marker='$\oplus$',ms=25)
    
    #ax.set_xticks(np.arange(10,100,10))
    ax.set_yticks(np.arange(0,1.,0.2))
    ax.set_ylim(0,0.9)
    ax.set_xlim(10,90)
    ax.tick_params(axis='both', direction='out',length = 8.0, width = 8.0,labelsize=fs)
    ax.text(0.02,0.86,sub_lbl[i], color='k',fontsize='x-large',weight='bold',horizontalalignment='left',transform=ax.transAxes)
    ax.set_ylabel("$e_{\\rm bin}$",fontsize=fs)
    
    if i == 3:
        ax.text(1.02,0.8,"ice free", color='k',fontsize='medium',weight='bold',horizontalalignment='left',transform=ax.transAxes)
        ax.text(1.02,0.6,"ice caps", color='gray',fontsize='medium',weight='bold',horizontalalignment='left',transform=ax.transAxes)
        ax.text(1.02,0.4,"ice belt", color='b',fontsize='medium',weight='bold',horizontalalignment='left',transform=ax.transAxes)
        ax.text(1.02,0.2,"snowball", color='r',fontsize='medium',weight='bold',horizontalalignment='left',transform=ax.transAxes)
    
    if i < 5:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("$a_{\\rm bin}$ (au)",fontsize=fs)
    if i == 0:
        color_label=r'$\Delta \epsilon$'
        cbar=colorbar(CS,ax)
        cbar.set_label(color_label,fontsize=fs)
        cticks = np.arange(0,105,15)
        cbar.ax.yaxis.set_ticks(cticks)
        cbar.ax.tick_params(axis='both', direction='out',length = 8.0, width = 8.0,labelsize=fs)
    elif i == 1:
        color_label=r'$\Delta T$'
        cbar=colorbar(CS,ax)
        cbar.set_label(color_label,fontsize=fs)
        ctick_rng = np.array([1,2,5,10,20,40])
        cticks = np.log10(ctick_rng)
        cticklabels = ["%i" % f for f in ctick_rng]
        cbar.ax.yaxis.set_ticks(cticks)
        cbar.ax.yaxis.set_ticklabels(cticklabels)
        cbar.ax.tick_params(axis='both', direction='out',length = 8.0, width = 8.0,labelsize=fs)
    elif i == 2:
        color_label=r'$\Delta f_{\rm ice}$'
        cbar=colorbar(CS,ax)
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
    elif i >3:
        color_label=r'max $f_{\rm ice}$'
        if i ==5:
            color_label=r'min $f_{\rm ice}$'
        cbar=colorbar(CS,ax)
        cbar.set_label(color_label,fontsize=fs)
        cticks = np.arange(0,0.4,0.1)
        cbar.ax.yaxis.set_ticks(cticks)
        cbar.ax.tick_params(axis='both', direction='out',length = 8.0, width = 8.0,labelsize=fs)

fig.subplots_adjust(hspace=0.15,wspace=0.15)


fig.savefig(plot_dir+"Fig16.png",bbox_inches='tight',dpi=300)
plt.close()