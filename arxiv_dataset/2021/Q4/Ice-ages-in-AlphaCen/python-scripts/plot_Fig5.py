import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib import rcParams
import matplotlib.patches as mpatches
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
    #cbar.ax.xaxis.set_ticks_position('right')
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
clabel = ["","max $f_{ice}$", "min $f_{ice}$"]
sublabel = ['a','b','c']

fig = plt.figure(1,figsize=(aspect*width,1.5*width),dpi=300)
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)


ax_list = [ax1,ax2,ax3]
sub_lbl = ['a','b','c']

i_p = 10
data = np.genfromtxt(data_dir+"VPlanet_data_A_map_%i.txt" % (i_p),delimiter=',',comments='#')

for i in range(0,3):
    
    if i == 0:
        Q = data[:,-4]
        cmap = colors.ListedColormap(['k', 'gray', 'b', 'r'])
        vmin, vmax = -0.5,3
    elif i == 1:
        Q = data[:,-2]
        cmap = cm.get_cmap("plasma").copy()
        vmin, vmax = 0.001,0.36
    else:
        Q = data[:,-1]
        cmap = cm.get_cmap("plasma").copy()
        vmin, vmax = 0.001,0.36
    eps = data[:,0]
    alpha = data[:,1]

    xi = np.arange(0,91,1)
    yi = np.arange(0,101,1)
    zi = griddata((eps,alpha),Q,(xi[None,:],yi[:,None]),method = 'linear')

    none_map = colors.ListedColormap(['none'])
    my_cmap = cm.get_cmap(cmap)
    norm = colors.Normalize(vmin,vmax)
    cmmapable = cm.ScalarMappable(norm,my_cmap)
    cmmapable.set_array(range(0,1))
    cmap.set_under('k')
    
    ax = ax_list[i]
    if i == 0:
        bounds = [0.,1.,2.,3.,4.]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        CS = ax.pcolormesh(xi,yi+0.5,zi,cmap = cmap,norm=norm,shading='auto',zorder=2)
        switch_states = np.zeros(len(eps))
        for k in range(0,len(eps)):
            if data[k,-3] != data[k,-4]:
                switch_states[k] = 1
        #zi = np.ma.masked_greater(zi,2*i_max[i])
        #ax.pcolor(xi-0.5,yi-0.5,zi,hatch='x',alpha=0)
        
        switch_rows = np.where(switch_states==1)[0]
        ax.plot(data[switch_rows,0],data[switch_rows,1]+0.5,'m+',ms=6,zorder=4)

    else:
        CS = ax.pcolormesh(xi,yi+0.5,zi,cmap = cmap,vmin=vmin,vmax=vmax,shading='auto')
    ax.plot(23,10,color='w',marker='$\oplus$',ms=25)
    
    ax.set_xticks(np.arange(0,105,15))
    ax.set_yticks(np.arange(0,120,20))
    ax.set_ylim(0,101)
    ax.set_xlim(0,90)
    ax.tick_params(axis='both', direction='out',length = 8.0, width = 8.0,labelsize=fs)
    ax.text(0.02,0.86,sub_lbl[i], color='w',fontsize='x-large',weight='bold',horizontalalignment='left',transform=ax.transAxes,zorder=5)
    ax.set_ylabel('$\\gamma$ ('+u'\u2033'+'/yr)', fontsize=fs)
    
    if i == 0:
        ax.text(1.02,0.8,"ice free", color='k',fontsize='medium',weight='bold',horizontalalignment='left',transform=ax.transAxes)
        ax.text(1.02,0.6,"ice caps", color='gray',fontsize='medium',weight='bold',horizontalalignment='left',transform=ax.transAxes)
        ax.text(1.02,0.4,"ice belt", color='b',fontsize='medium',weight='bold',horizontalalignment='left',transform=ax.transAxes)
        ax.text(1.02,0.2,"snowball", color='r',fontsize='medium',weight='bold',horizontalalignment='left',transform=ax.transAxes)
    if i < 2:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("$\epsilon_o$ (deg.)",fontsize=fs)

    if i == 1:
        color_label=r'max $f_{\rm ice}$'
    elif i >1:
        color_label=r'min $f_{\rm ice}$'
    if i > 0:
        cbar=colorbar(CS)
        cbar.set_label(color_label,fontsize=fs)
        cticks = np.arange(0,0.4,0.1)
        cbar.ax.yaxis.set_ticks(cticks)
        cbar.ax.tick_params(axis='both', direction='out',length = 8.0, width = 8.0,labelsize=fs)



fig.subplots_adjust(hspace=0.1,wspace=0.15)


fig.savefig(plot_dir+"Fig5.png",bbox_inches='tight',dpi=300)
plt.close()