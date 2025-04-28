import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib import ticker
from matplotlib import rcParams
import sys
import os

def get_Area(lat_i,lat_j):
    dA = 0
    theta_st = np.min([lat_i,lat_j])
    n_dtheta = int(100)
    dtheta = np.abs(lat_j-lat_i)/n_dtheta
    for k in range(0,n_dtheta):
        theta = theta_st + k*dtheta
        dA += 2.*np.pi*(R_E*np.cos(theta))*R_E*dtheta
    return dA

R_E = 6371.*1000 #Earth radius in km --> m

eps = int(sys.argv[1])
alpha = int(sys.argv[2])
i_p = int(sys.argv[3])
star = sys.argv[4]

os.chdir('..')
home = os.getcwd() + "/"

plot_dir = home + "Figs/"
data_dir = home + "data/aCen%s_Climate_%i/Obl[%03i]/Alpha[%03i]/" % (star,i_p,eps,alpha)

if star == 'A':
    if alpha == 10:
        plot_fn = "Fig2.png"
    else:
        plot_fn = "Fig3.png"
else:
    if i_p == 2:
        plot_fn = "Fig7.png"
    elif i_p == 10:
        plot_fn = "Fig8.png"
    else:
        plot_fn = "Fig9.png"


forward_file = "aCen%s.Earth.forward" % star
clim_file = "aCen%s.Earth.Climate" % star

orb_file = home + "data/aCen%s_orb/Orb_data_%i.txt" % (star,i_p)

rcParams.update({'font.size': 20})
cmap = cm.gnuplot

obl_data = np.genfromtxt(data_dir+"obl_data.txt",usecols=(0,5,6)) #time, obl,psi
Orb_data = np.genfromtxt(orb_file,usecols=(0,2,4,5)) #time, ecc, omg, Omg
Climate_data = np.genfromtxt(data_dir+clim_file,usecols=(0,1,2,3,5,6)) #time,lat, ice_mass, surface temp, accum, abalte
Climate_cut = np.where(np.logical_and(0<=np.abs(Climate_data[:,1]),np.abs(Climate_data[:,1])<=85))[0]
Climate_data = Climate_data[Climate_cut,:]

Latitude = Climate_data[:,1]
Clim_time = Climate_data[:,0]/1e3

time = obl_data[:,0]/1e3
ecc = Orb_data[:,1]
Climate_ang = np.radians(Orb_data[:,2] + Orb_data[:,3] + obl_data[:,2])
CPP = ecc*np.sin(np.radians(obl_data[:,1]))*np.sin(Climate_ang)

def colorbar(mappable):
    last_axes = plt.gca()
    ax_i = mappable.axes
    fig = ax_i.figure
    divider = make_axes_locatable(ax_i)
    cax = divider.new_horizontal(size="2.5%", pad="5%")
    cbar = fig.colorbar(mappable, cax=cax,orientation='vertical')
    cbar.ax.tick_params(axis='both', direction='out',length = 8.0, width = 8.0,labelsize=fs)
    #cbar.ax.xaxis.set_ticks_position('right')
    plt.sca(last_axes)
    return cbar

def plot_panel(fig,ax_idx,t,y,c,vmin,vmax,clab):
    #pos = ax.figbox.get_points()
    ax = ax_list[ax_idx]
    pos = ax.get_subplotspec().get_position(fig)

    my_cmap=cm.get_cmap('gnuplot').copy()
    norm = colors.Normalize(vmin,vmax)
    cmmapable = cm.ScalarMappable(norm,my_cmap)
    cmmapable.set_array(range(0,1))
    if ax_idx == 4:
        my_cmap.set_under('w')
    
    zi = griddata((t,y),c,(xi[None,:],yi[:,None]),method = 'linear')

    CS = ax.pcolormesh(xi,yi,zi,cmap = my_cmap,vmin=vmin,vmax=vmax,shading='auto')
    cb_width = 0.02
    cb_height = pos.y1-pos.y0
    vertical_position = pos.y0
    horizontal_position = 0.905
    cax = fig.add_axes([horizontal_position, vertical_position, cb_width, cb_height])
    clb = plt.colorbar(cmmapable,cax=cax,orientation='vertical')
    clb.ax.tick_params(axis='both', direction='out',length = 8.0, width = 8.0,labelsize=fs)
    tloc = ticker.MaxNLocator(nbins=5)
    clb.locator=tloc
    clb.update_ticks()
    ax.tick_params(axis='both', direction='out',length = 4.0, width = 4.0)
    clb.set_label(clab,fontsize=14)
    

fs = 'x-large'
width = 16.
aspect = 1.
ms = 6.5
lw=3
ls = ['-','--',':','-.','-']
sub_lbl = ['a','b','c','d','e']

if star == 'A':
    ymin = [0.0375,0,-0.025,0,0]
    ymax = [0.0475,33,0.025,83,83]
else:
    ymin = [0.023,10,-0.027,0,0]
    ymax = [0.027,35,0.027,83,83]

tmax = 500. #Myr

xi = np.arange(0,tmax,0.001)
yi = np.arange(0.,84,1.)

fig = plt.figure(figsize=(aspect*width,width),dpi=300)
gs = gridspec.GridSpec(5, 1,hspace=0.15)

ax_list = [fig.add_subplot(gs[f,0]) for f in range(0,5)]

ax_list[0].plot(time,Orb_data[:,1],'k-',lw=lw)
ax_list[1].plot(time,obl_data[:,1],'k-',lw=lw)
ax_list[2].plot(time,CPP,'k-',lw=lw)


ax_list[0].set_xlim(0,tmax)
ax_list[0].set_xticklabels([])
ax_list[0].tick_params(axis='both', direction='out',length = 4.0, width = 4.0)
ax_list[0].set_ylabel("$e_{\\rm p}$",fontsize=20)

ax_list[1].set_xlim(0,tmax)
ax_list[1].set_xticklabels([])
ax_list[1].tick_params(axis='both', direction='out',length = 4.0, width = 4.0)
ax_list[1].set_ylabel("$\\varepsilon$",fontsize=20)

ax_list[2].set_xlim(0,tmax)
ax_list[2].set_xticklabels([])
ax_list[2].tick_params(axis='both', direction='out',length = 4.0, width = 4.0)
#ax_list[2].set_ylabel("$e_{\\rm p}\sin(\\varpi_{\\rm p}+\\psi)$",fontsize=20)
ax_list[2].set_ylabel("COPP",fontsize=20)

vmin2,vmax2 = np.min(Climate_data[:,3]),np.max(Climate_data[:,3])

ice_mass = Climate_data[:,2]/1e6

vmax22 = np.max(ice_mass)

plot_panel(fig,3,Clim_time,Latitude,Climate_data[:,3],-20,30,"Surface Temp. ($^\circ$C)")
plot_panel(fig,4,Clim_time,Latitude,ice_mass,0.01,np.round(vmax22),"Ice Mass ($\\times 10^6$ kg)")

for i in range(0,5):
    col = 'k'
    if i ==3:
        col = 'w'
    ax_list[i].text(0.015,0.85,sub_lbl[i], color=col,fontsize='x-large',weight='bold',horizontalalignment='left',transform=ax_list[i].transAxes)
    ax_list[i].set_ylim(ymin[i],ymax[i])
    if i_p == 30:
        if i ==0:
            ax_list[i].set_ylim(0.0145,ymax[i])
        elif i ==1:
            ax_list[i].set_ylim(10,85)
    if star == 'A' and i == 1:
        ax_list[i].set_yticks(np.arange(0,40,10))

for i in range(3,5):
    ax_list[i].set_xlim(0,tmax)
    
    ax_list[i].set_yticks(np.arange(0,100,20))
    ax_list[i].set_ylabel("Latitude (deg.)",fontsize=20)
    
    if i == 3:
        ax_list[i].set_xticklabels([])
    else:
        ax_list[i].set_xlabel("Time (kyr)",fontsize=fs)

fig.savefig(plot_dir + plot_fn, bbox_inches='tight',dpi=300)
plt.close()
