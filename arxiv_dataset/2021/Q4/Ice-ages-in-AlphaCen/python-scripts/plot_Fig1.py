import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import threading
import sys

rcParams.update({'font.size': 22})
rcParams.update({'mathtext.fontset': 'cm'})

def get_Area(lat_i,lat_j):
    tot_SA = 4.*np.pi*R_E**2
    dA = 0
    theta_st = np.min([lat_i,lat_j])
    n_dtheta = int(100)
    dtheta = np.abs(lat_j-lat_i)/n_dtheta
    for k in range(0,n_dtheta):
        theta = theta_st + k*dtheta
        dA += 2.*np.pi*(R_E*np.cos(theta))*R_E*dtheta
    dA /= tot_SA
    return dA

def plot_clim(par):
    obl,alph,i_p = par
    if alph == 10:
        color = 'k'
    else:
        color = 'r'

    sub_dir = data_dir + "aCenA_Climate_%i/Obl[%03i]/Alpha[%03i]/" % (i_p,obl,alph)
    
    if os.path.exists(sub_dir):
        os.chdir(sub_dir)
        #print(sub_dir)
        #flux_data = np.genfromtxt("Orb_data.txt",delimiter=',',comments='#')
        obl_data = np.genfromtxt("obl_data.txt")
        clim_data = np.genfromtxt("aCenA.Earth.forward")
        scale_fact = 1e3
        xmax = np.max(obl_data[:,0])/scale_fact
        obl_time = obl_data[:,0]/scale_fact
        clim_time = clim_data[:,0]/scale_fact
        
        nrows = len(clim_data[:,0])
        f_ice = np.zeros(nrows)
        if nrows == 501:
            ice_data = np.genfromtxt("aCenA.Earth.Climate",usecols=(0,1,2)) #time,lat, ice_mass
            
            for f in range(0,len(f_ice)):
                #temp_ice = ice_data[150*f:150*(f+1),:]
                time_idx = np.where(np.abs(ice_data[:,0]-clim_data[f,0])<1e-3)[0]
                temp_ice = ice_data[time_idx,:]
                ice_idx = np.where(temp_ice[:,2]>1)[0]
                #for lat in range(0,150):
                if len(ice_idx)>1:
                    for lat in range(0,len(temp_ice)-1):
                        if temp_ice[lat,2]>1:
                            lat_i = np.abs(np.radians(temp_ice[lat,1]))
                            if lat_i > np.radians(83):
                                f_ice[f] += 2*get_Area(lat_i,np.pi/2.)
                            lat_j = np.abs(np.radians(temp_ice[lat+1,1]))
                            f_ice[f] += get_Area(lat_i,lat_j)

        

        for i in range(0,len(ax_list)):
            if i == 0:
                ax_list[i].plot(obl_time,obl_data[:,2],'-',color=color,lw=lw)
            elif i == 1:
                ax_list[i].plot(obl_time,obl_data[:,5],'-',color=color,lw=lw)
            elif i == 2:
                ax_list[i].plot(clim_time,clim_data[:,1],'-',color=color,lw=lw)
            elif i == 3:
                ax_list[i].plot(clim_time,f_ice,'-',color=color,lw=lw) 
            elif i == 4:
                ax_list[i].plot(clim_time,clim_data[:,2],'-',color=color,lw=lw) 
            if i < (len(ax_list)-1):
                ax_list[i].set_xticklabels([])
            ax_list[i].text(0.02,0.86,sub_lbl[i], color='k',fontsize='x-large',weight='bold',horizontalalignment='left',transform=ax_list[i].transAxes)
            ax_list[i].set_xlim(0,xmax)
            ax_list[i].set_ylim(ymin[i],ymax[i])
            ax_list[i].set_ylabel(ylabel[i],fontsize='xx-large') 
            ax_list[i].tick_params(axis='both', direction='out',length = 8.0, width = 8.0,labelsize=fs)
    os.chdir(home)
        

lock = threading.Lock()
os.chdir('..')
home = os.getcwd() + "/"

plot_dir = home + "Figs/"
data_dir = home + "data/"
R_E = 6371. #radius of Earth in km

aspect = 16./9.
width = 10.
lw = 4.
ms = 5.
fs = 'x-large'

fig = plt.figure(1,figsize=(aspect*width,2.5*width),dpi=300)
ax1 = fig.add_subplot(511)
ax2 = fig.add_subplot(512)
ax3 = fig.add_subplot(513)
ax4 = fig.add_subplot(514)
ax5 = fig.add_subplot(515)


ax_list = [ax1,ax2,ax3,ax4,ax5]
ylabel = ['$e_{\\rm p}$','$\\varepsilon$ (deg.)','Temp ($^\circ$C)','$f_{\\rm ice}$', '$\\alpha$']
sub_lbl = ['a','b','c','d','e']
ymin = [0.0375,0,11,0,0.34]
ymax = [0.0475,33,16,0.3,0.365]

plot_clim((23,10,10))
plot_clim((23,85,10))


ax_list[-1].set_xlabel("Time (kyr)",fontsize=fs)
#ax4.set_ylim(0,90)

fig.subplots_adjust(hspace=0.1)
fig.savefig(plot_dir+"Fig1.png",bbox_inches='tight',dpi=300)
plt.close()