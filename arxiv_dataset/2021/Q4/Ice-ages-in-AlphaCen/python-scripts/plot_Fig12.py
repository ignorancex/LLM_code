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
    ax_i = mappable.axes
    fig = ax_i.figure
    divider = make_axes_locatable(ax_i)
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

aspect = 2.
width = 10.
lw = 4
fs = 'x-large'
clabel = ["","max $f_{ice}$", "min $f_{ice}$"]
sublabel = ['a','b','c']

fig = plt.figure(1,figsize=(aspect*width,2*width),dpi=300)
ax11 = fig.add_subplot(331)
ax12 = fig.add_subplot(332)
ax13 = fig.add_subplot(333)

ax21 = fig.add_subplot(334)
ax22 = fig.add_subplot(335)
ax23 = fig.add_subplot(336)

ax31 = fig.add_subplot(337)
ax32 = fig.add_subplot(338)
ax33 = fig.add_subplot(339)


ax_list = [ax11,ax21,ax31,ax12,ax22,ax32,ax13,ax23,ax33]
sub_lbl = ['a','b','c','d','e','f','g','h','i']

ip_list = [2,10,30]

for j in range(0,3):
    data = np.genfromtxt(data_dir+"VPlanet_data_B_map_%i.txt" % (ip_list[j]),delimiter=',',comments='#')
    data[data[:,6]<0,6] = 3
    for i in range(0,3):
        ax_idx = 3*j+i

        ax = ax_list[ax_idx]
        if i == 0:
            Q = data[:,-4]
            cmap = colors.ListedColormap(['k', 'gray', 'b', 'r'])
            vmin, vmax = 0.5,2.5
        elif i == 1:
            Q = data[:,-2]
            cmap = cm.get_cmap("plasma").copy()
            vmin, vmax = 0.001,0.4
        else:
            Q = data[:,-1]
            cmap = cm.get_cmap("plasma").copy()
            vmin, vmax = 0.001,0.4
        eps = data[:,0]
        alpha = data[:,1]

        xi = np.arange(0,91,1)
        yi = np.arange(0,101,1)
        zi = griddata((eps,alpha),Q,(xi[None,:],yi[:,None]),method = 'linear')
        

        my_cmap = cm.get_cmap(cmap)
        norm = colors.Normalize(vmin,vmax)
        cmmapable = cm.ScalarMappable(norm,my_cmap)
        cmmapable.set_array(range(0,1))
        cmap.set_under('k')
        cmap.set_over('r')
        

        if i == 0:
            bounds = [0.,1.,2.,3.,4.]
            norm = colors.BoundaryNorm(bounds, cmap.N)
            CS = ax.pcolormesh(xi,yi+0.5,zi,cmap = cmap,norm=norm,shading='auto',zorder=2)
            
            #check second equilibrium
            switch_states = np.zeros(len(eps))
           
            for k in range(0,len(eps)):
                if data[k,-3] != data[k,-4]:
                    switch_states[k] = 1

            #ax.plot(data[switch_states==1,0],data[switch_states==1,1]+0.5,'wx',ms=2,zorder=4,alpha=0.75)
            switch_rows = np.where(switch_states==1)[0]
            ax.plot(data[switch_rows,0],data[switch_rows,1]+0.5,'m+',ms=6,zorder=4)
            ax.text(0.5,1.05,"%i$^\circ$" % ip_list[j], color='k',fontsize='x-large',weight='bold',horizontalalignment='center',transform=ax.transAxes)
        else:
            CS = ax.pcolormesh(xi,yi+0.5,zi,cmap = cmap,vmin=vmin,vmax=vmax,shading='auto')
        ax.plot(23,46,color='w',marker='$\oplus$',ms=25,zorder=5)
        
        ax.set_xticks(np.arange(0,105,15))
        ax.set_yticks(np.arange(0,120,20))
        ax.set_ylim(0,101)
        ax.set_xlim(0,90)
        ax.tick_params(axis='both', direction='out',length = 8.0, width = 8.0,labelsize=fs)
        ax.text(0.02,0.86,sub_lbl[ax_idx], color='w',fontsize='x-large',weight='bold',horizontalalignment='left',transform=ax.transAxes)

        
        if i == 0 and j == 2:
            ax.text(1.02,0.8,"ice free", color='k',fontsize='medium',weight='bold',horizontalalignment='left',transform=ax.transAxes)
            ax.text(1.02,0.6,"ice caps", color='gray',fontsize='medium',weight='bold',horizontalalignment='left',transform=ax.transAxes)
            ax.text(1.02,0.4,"ice belt", color='b',fontsize='medium',weight='bold',horizontalalignment='left',transform=ax.transAxes)
            ax.text(1.02,0.2,"snowball", color='r',fontsize='medium',weight='bold',horizontalalignment='left',transform=ax.transAxes)
        if i < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("$\epsilon_o$ (deg.)",fontsize=fs)
        if j > 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel('$\\gamma$ ('+u'\u2033'+'/yr)', fontsize=fs)

        if i == 1:
            color_label=r'max $f_{\rm ice}$'
        elif i >1:
            color_label=r'min $f_{\rm ice}$'
        if i > 0 and j==2:
            cbar=colorbar(CS)
            cbar.set_label(color_label,fontsize=fs)
            cticks = np.arange(0,0.4,0.1)
            cbar.ax.yaxis.set_ticks(cticks)
            cbar.ax.tick_params(axis='both', direction='out',length = 8.0, width = 8.0,labelsize=fs)



fig.subplots_adjust(hspace=0.15,wspace=0.15)


fig.savefig(plot_dir+"Fig12.png",bbox_inches='tight',dpi=300)
plt.close()