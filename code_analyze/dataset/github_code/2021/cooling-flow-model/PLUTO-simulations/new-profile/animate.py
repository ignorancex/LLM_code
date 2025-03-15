#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 07:48:49 2021

@author: alankar
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import sys
import h5py
from scipy.interpolate import interp1d
import tqdm

mp  = 1.67e-24
pc  = 3.086e18
kB  = 1.38e-16
kpc = 1e3*pc
Myr= 365*24*60**2*1e6

cool = np.loadtxt('cooltable.dat')
cool = interp1d(cool[:,0],cool[:,1])

X = 0.7154
Y = 0.2703
Z = 0.0143
mu = 1./(2*X+0.75*Y+0.5625*Z)
mue = 2./(1+X)
mui = 1./(1/mu-1/mue)
gamma = 5/3.


Thot = 3.16e6
p = 100*kB
nhot = p/(kB*Thot)
rho = nhot*mu*mp
cs = np.sqrt(gamma*p/rho)
tcool = p/(rho**2*cool(Thot)/(mue*mui*mp**2))/(gamma-1)
freq = 1.0 #Myr
#print(tcool/Myr)

max_val, min_val = None, None
base_dir = sys.argv[1] #'./output-128'
start = int(sys.argv[2])
end   = int(sys.argv[3])
pbar  = tqdm.tqdm(total=end-start)

hfile = h5py.File('%s/data.%04d.dbl.h5'%(base_dir, start),'r')
res   = np.array(hfile['cell_coords/X']).shape[0]
X   = np.array(hfile['cell_coords/X'])*pc
hfile.close()

min_ent, max_ent = None, None
min_den, max_den = None, None
min_vel, max_vel = None, None
min_prs, max_prs = None, None
min_mdt, max_mdt = None, None
min_tmp, max_tmp = None, None

time = None
for file_no in range(start,end):
    X, rho, vr, T, p = None, None, None, None, None
    with h5py.File('%s/data.%04d.dbl.h5'%(base_dir, file_no),'r') as hfile:
        time = file_no*pc/1e5*freq
        X   = np.array(hfile['cell_coords/X'])*pc
        rho = np.array(hfile['Timestep_%d/vars/rho'%file_no])*mp
        vr = np.array(hfile['Timestep_%d/vars/vx1'%file_no])*1e5
        T  = np.array(hfile['Timestep_%d/vars/T'%file_no])
        p  = np.array(hfile['Timestep_%d/vars/prs'%file_no])*mp*1e10
    
    #cs = np.sqrt(gamma*p/rho)
    #mach = vr/cs
    #Mdot = -4*np.pi*X**2*rho*vr/(2e33/(365*24*60**2))/1e-4
    #entropy = np.log10(p/rho**gamma/1e32)
    
    #hfile.close()
    
    
    if file_no==start: 
        #max_ent = np.max(entropy)
        #min_ent = np.min(entropy)
        
        max_den = np.max(np.log10(rho/(mu*mp)))
        min_den = np.min(np.log10(rho/(mu*mp)))
        
        max_vel = np.max(vr/1e5)
        min_vel = np.min(vr/1e5)
        
        max_prs = np.max(np.log10(p/kB))
        min_prs = np.min(np.log10(p/kB))
        
        #max_mdt = np.max(Mdot)
        #min_mdt = np.min(Mdot)
        
        max_tmp = np.max(np.log10(T))
        min_tmp = np.min(np.log10(T))
        
    else:
        #max_ent = max(np.max(entropy), max_ent)
        #min_ent = min(np.min(entropy), min_ent)
        
        max_vel = max(np.max(vr/1e5), max_vel)
        min_vel = min(np.min(vr/1e5), min_vel)
        
        max_den = max(np.max(np.log10(rho/(mu*mp))), max_den)
        min_den = min(np.min(np.log10(rho/(mu*mp))), min_den)
        
        max_prs = max(np.max(np.log10(p/kB)), max_prs)
        min_prs = min(np.min(np.log10(p/kB)), min_prs)
        
        #max_mdt = max(np.max(Mdot), max_mdt)
       # min_mdt = min(np.min(Mdot), min_mdt)
        
        max_tmp = max(np.max(np.log10(T)), max_tmp)
        min_tmp = min(np.min(np.log10(T)), min_tmp)

#X, rho, vr, T, p = None, None, None, None, None
def data_gen(): #generator function
    time = data_gen.time
    file_no = start
    
    while file_no < end:
        pbar.update(1) 
        file_no += 1
        time += freq
        
        with h5py.File('%s/data.%04d.dbl.h5'%(base_dir, file_no),'r') as hfile:
            timeMyr = (time*pc/1e5)/(1e6*365*24*60**2)
            X   = np.array(hfile['cell_coords/X'])*pc/kpc
            rho = np.log10(np.array(hfile['Timestep_%d/vars/rho'%file_no])*mp/(mu*mp))
            vr = np.array(hfile['Timestep_%d/vars/vx1'%file_no])*1e5/1e5
            T  = np.log10(np.array(hfile['Timestep_%d/vars/T'%file_no]))
            p  = np.log10(np.array(hfile['Timestep_%d/vars/prs'%file_no])*mp*1e10/kB)
            
            #cs = np.sqrt(gamma*p/rho)
            #mach = vr/cs
            Mdot = -4*np.pi*np.array(hfile['cell_coords/X'])**2*\
                np.array(hfile['Timestep_%d/vars/rho'%file_no])*\
                    np.array(hfile['Timestep_%d/vars/vx1'%file_no])
            Mdot *= (mp*pc**2*1e5)
            Mdot = Mdot/(2e33/(365*24*60**2))
            entropy = np.log10((np.array(hfile['Timestep_%d/vars/prs'%file_no])*mp*1e10)/
                               ((np.array(hfile['Timestep_%d/vars/rho'%file_no])*mp)**gamma)/1e32)
        
        #hfile.close()
        #print(np.max(Mdot[np.logical_not(np.isnan(Mdot))]), np.min(Mdot[np.logical_not(np.isnan(Mdot))]))
        # adapted the data generator to yield
        if file_no>=end-1: pbar.close()
        yield timeMyr, X, rho, vr, T, p, entropy, Mdot

data_gen.time = start*freq

rightedge = 5.
#Plotting model and PDE data together
fig, axs = plt.subplots(2, 3, figsize=(20,11))
gs1 = gridspec.GridSpec(2, 3)
gs1.update(wspace=0.32, hspace=0.02) # set the spacing between axes.
axs = np.array([ [plt.subplot(gs1[0]), plt.subplot(gs1[1]), plt.subplot(gs1[2])], 
                 [plt.subplot(gs1[3]), plt.subplot(gs1[4]), plt.subplot(gs1[5])] ])


#initialize the plot and the axes
line1, = axs[0, 0].plot([], [], color='tab:blue', linewidth=5) #number density
axs[0, 0].grid()
#axs[0, 0].set_ylim(ymin=-3.8, ymax=-0.8)
axs[0, 0].set_ylim(ymin=-4.8, ymax=-2)
axs[0, 0].set_xlim(xmin=0., xmax=rightedge)
axs[0, 0].set_ylabel(r'number density [$\rm cm^{-3}\ (log)$]', size=20 )
axs[0, 0].get_xaxis().set_ticklabels([])
axs[0, 0].tick_params(axis='both', which='major', labelsize=18)
axs[0, 0].tick_params(axis='both', which='minor', labelsize=16)
axs[0, 0].xaxis.set_major_locator(plt.MaxNLocator(6))


line2, = axs[0, 1].plot([], [], 'tab:orange', linewidth=5) #pressure
axs[0, 1].grid()
#axs[0, 1].set_ylim(ymin=min_prs, ymax=3.9)
axs[0, 1].set_ylim(ymin=1.9, ymax=2.1)
axs[0, 1].set_xlim(xmin=0., xmax=rightedge)
axs[0, 1].set_ylabel(r'$\rm p_{gas}$ [$\rm \times k_B \ K cm^{-3}\ (log)$]', size=20 )
axs[0, 1].get_xaxis().set_ticklabels([])
axs[0, 1].tick_params(axis='both', which='major', labelsize=18)
axs[0, 1].tick_params(axis='both', which='minor', labelsize=16)
axs[0, 1].xaxis.set_major_locator(plt.MaxNLocator(6))

line3, = axs[0, 2].plot([], [], 'tab:green', linewidth=5) #velocity
axs[0, 2].grid()
axs[0, 2].set_ylabel(r'velocity [$\rm km s^{-1}$]', size=20 )
#axs[0, 2].set_ylim(ymin=-23, ymax=23)
axs[0, 2].set_ylim(ymin=-12, ymax=16)
axs[0, 2].set_xlim(xmin=0., xmax=rightedge)
axs[0, 2].get_xaxis().set_ticklabels([])
axs[0, 2].tick_params(axis='both', which='major', labelsize=18)
axs[0, 2].tick_params(axis='both', which='minor', labelsize=16)
axs[0, 2].xaxis.set_major_locator(plt.MaxNLocator(6))

line4, = axs[1, 0].plot([], [], 'tab:red', linewidth=5) #entropy
axs[1, 0].grid()
axs[1, 0].set_xlabel(r'distance [$\rm kpc$]', size=20 )
axs[1, 0].set_ylabel(r'$\rm p_{gas}/\rho ^{\gamma}\ [\rm CGS \times 10^{32}\ (log)$]', size=20 )
#axs[1, 0].set_ylim(ymin=-3.8, ymax=0.9)
axs[1, 0].set_ylim(ymin=-3.2, ymax=1.8)
axs[1, 0].set_xlim(xmin=0., xmax=rightedge)
axs[1, 0].tick_params(axis='both', which='major', labelsize=18)
axs[1, 0].tick_params(axis='both', which='minor', labelsize=16)
axs[1, 0].xaxis.set_major_locator(plt.MaxNLocator(6))

line5, = axs[1, 1].plot([], [], 'tab:brown', linewidth=5) #temperature
axs[1, 1].grid()
#axs[1, 1].set_ylim(ymin=min_tmp, ymax=6.3)
axs[1, 1].set_ylim(ymin=3.83, ymax=6.7)
axs[1, 1].set_xlim(xmin=0., xmax=rightedge)
axs[1, 1].set_ylabel(r' temperature [$\rm K\ (log)$]', size=20 )
axs[1, 1].set_xlabel(r'distance [$\rm kpc$]', size=20 )
axs[1, 1].tick_params(axis='both', which='major', labelsize=18)
axs[1, 1].tick_params(axis='both', which='minor', labelsize=16)
axs[1, 1].xaxis.set_major_locator(plt.MaxNLocator(6))

line6, = axs[1, 2].plot([], [], 'tab:purple', linewidth=5) #Mdot
axs[1, 2].grid()
axs[1, 2].set_ylim(ymin=-1.2, ymax=1.2)
#axs[1, 2].set_ylim(ymin=-3.5, ymax=-3)
axs[1, 2].set_xlim(xmin=0., xmax=rightedge)
axs[1, 2].set_ylabel(r'$\rm \dot{M}$ [$\rm \times 10^{-3} M_\odot yr^{-1}$] ', size=20 )
axs[1, 2].set_xlabel(r'distance [$\rm kpc$]', size=20 )
axs[1, 2].tick_params(axis='both', which='major', labelsize=18)
axs[1, 2].tick_params(axis='both', which='minor', labelsize=16)
axs[1, 2].xaxis.set_major_locator(plt.MaxNLocator(6))

lines = [line1, line2, line3, line4, line5, line6]

# initialize the data arrays 
timeMyr, X, rho, vr, T, p, entropy, Mdot = None, [], [], [], [], [], [], []
def run(data):
    # update the data
    timeMyr, X, rho, vr, T, p, entropy, Mdot = data
    
    '''
    # axis limits checking. Same as before, just for both axes
    for ax in [ax1, ax2]:
        xmin, xmax = ax.get_xlim()
        if t >= xmax:
            ax.set_xlim(xmin, 2*xmax)
            ax.figure.canvas.draw()
    '''
    fig.suptitle(r'time $\rm \approx %d$ Myr'%int(np.round(timeMyr)), size=25)
    fig.canvas.draw()
    # update the data of line objects
    lines[0].set_data(X, rho)
    lines[1].set_data(X, p)
    lines[2].set_data(X, vr)
    lines[3].set_data(X, entropy)
    lines[4].set_data(X, T)
    lines[5].set_data(X, Mdot*1e3)
    #print(timeMyr)
    #lines[6].set_text(r'time $\rm = %.3f$ Myr'%timeMyr)
    #fig.suptitle(r'time $\rm = %.3f$ Myr'%timeMyr)
    
    return lines

ani = animation.FuncAnimation(fig, run, data_gen, blit=True, interval=25, repeat=True, 
                              repeat_delay=10, save_count=int(sys.argv[3])-1,
                              cache_frame_data=True)

#import matplotlib as mpl 
#mpl.rcParams['animation.ffmpeg_path'] = r'C:\\Users\\xx\\Desktop\\ffmpeg\\bin\\ffmpeg.exe'
writervideo = animation.FFMpegWriter(fps=25, extra_args=['-vcodec', 'libx264']) 
ani.save('./%s/plots/anim.mp4'%sys.argv[1], writer=writervideo)
#plt.show()
