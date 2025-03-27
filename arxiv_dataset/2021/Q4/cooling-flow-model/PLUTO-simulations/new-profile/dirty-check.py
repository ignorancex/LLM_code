#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 12:28:14 2021

@author: alankar
python dirty-check.py fields 1371 300 ./output-128/
"""

import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

mp = 1.67e-24
pc = 3.086e18
kB = 1.38e-16
Myr= 1e6*365*24*60**2

cool = np.loadtxt('cooltable.dat')
cool = interp1d(cool[:,0],cool[:,1])

X = 0.7154
Y = 0.2703
Z = 0.0143
mu = 1./(2*X+0.75*Y+0.5625*Z)
mue = 2./(1+X)
mui = 1./(1/mu-1/mue)
gamma = 5/3.

nhot = 3.16e-5
Thot = 3.16e6
rho = nhot*mu*mp
p = nhot*kB*Thot
cs = np.sqrt(gamma*p/rho)
tcool = p/(rho**2*cool(Thot)/(mue*mui*mp**2))/(gamma-1)
freq = 1.0

max_val, min_val = None, None
start = 0

base_dir = sys.argv[4] #'./output-128'
#res = int((sys.argv[4])[9:-1])

hfile = h5py.File('%s/data.%04d.dbl.h5'%(base_dir, start),'r')
res   = np.array(hfile['cell_coords/X']).shape[0]
X   = np.array(hfile['cell_coords/X'])*pc
hfile.close()
    
total = int(sys.argv[2]) - start

rho_all = np.zeros((total,res))
vr_all  = np.zeros((total,res))
T_all   = np.zeros((total,res))
p_all   = np.zeros((total,res))
mac_all = np.zeros((total,res))
ent_all = np.zeros((total,res))
tcool_all = np.zeros((total,res))

min_ent, max_ent = None, None
min_den, max_den = None, None
min_vel, max_vel = None, None
min_prs, max_prs = None, None
min_mac, max_mac = None, None
min_tmp, max_tmp = None, None

time = None
for file_no in range(start,int(sys.argv[2])):
    hfile = h5py.File('%s/data.%04d.dbl.h5'%(base_dir, file_no),'r')
    time = file_no*pc/1e5*freq
    X   = np.array(hfile['cell_coords/X'])*pc
    rho = np.array(hfile['Timestep_%d/vars/rho'%file_no])*mp
    vr = np.array(hfile['Timestep_%d/vars/vx1'%file_no])*1e5
    T  = np.array(hfile['Timestep_%d/vars/T'%file_no])
    p  = np.array(hfile['Timestep_%d/vars/prs'%file_no])*mp*1e10
    
    cs = np.sqrt(gamma*p/rho)
    mach = vr/cs
    #Mdot = 4*np.pi*X**2*rho*vr/(2e33/(365*24*60**2))
    entropy = p/rho**gamma
    tcoolg  = p/(rho**2*cool(T)/(mue*mui*mp**2))/(gamma-1)/Myr
    
    hfile.close()
    
    rho_all[file_no-start,:] = rho
    vr_all[file_no-start,:]  = vr
    T_all[file_no-start,:]   = T
    p_all[file_no-start,:]   = p
    mac_all[file_no-start,:] = mach
    ent_all[file_no-start,:] = entropy
    tcool_all[file_no-start,:] = tcoolg
    
    if file_no==start: 
        max_ent = np.max(entropy)
        min_ent = np.min(entropy)
        
        max_den = np.max(rho/(mu*mp))
        min_den = np.min(rho/(mu*mp))
        
        max_vel = np.max(vr/1e5)
        min_vel = np.min(vr/1e5)
        
        max_prs = np.max(p/kB)
        min_prs = np.min(p/kB)
        
        max_mac = np.max(mach)
        min_mac = np.min(mach)
        
        max_tmp = np.max(T)
        min_tmp = np.min(T)
        
    else:
        max_ent = max(np.max(entropy), max_ent)
        min_ent = min(np.min(entropy), min_ent)
        
        max_vel = max(np.max(vr/1e5), max_vel)
        min_vel = min(np.min(vr/1e5), min_vel)
        
        max_den = max(np.max(rho/(mu*mp)), max_den)
        min_den = min(np.min(rho/(mu*mp)), min_den)
        
        max_prs = max(np.max(p/kB), max_prs)
        min_prs = min(np.min(p/kB), min_prs)
        
        max_mac = max(np.max(mach), max_mac)
        min_mac = min(np.min(mach), min_mac)
        
        max_tmp = max(np.max(T), max_tmp)
        min_tmp = min(np.min(T), min_tmp)
'''
if (sys.argv[1]=='vel'):
    max_val/=1e5
    min_val/=1e5
'''

print(min_ent, max_ent)
print(min_den, max_den) 
print(min_vel, max_vel)
print(min_prs, max_prs)
print(min_mac, max_mac )
print(min_tmp, max_tmp)

#min_ent, max_ent = 8e30, 1.4e32
#min_den, max_den = 9e-4, 0.5
#min_vel, max_vel = -25, 25
#min_prs, max_prs = 900, 1100
#min_mac, max_mac = -1, 1 
#in_tmp, max_tmp = 9.8e3, 2e6
print('-----------------')
print(min_ent, max_ent)
print(min_den, max_den) 
print(min_vel, max_vel)
print(min_prs, max_prs)
print(min_mac, max_mac )
print(min_tmp, max_tmp)


#-------------Average profile-----------------

start_avg = int(sys.argv[3])
time_start = start_avg*pc/1e5*freq/Myr
time_stop  = int(sys.argv[2])*pc/1e5*freq/Myr
data = [X/pc,]

fig = plt.figure()
Mdot = -4*np.pi*np.dot(np.ones((rho_all.shape[0],X.shape[0])),np.diag((X/pc)**2))*(rho_all/mp)*(vr_all/1e5)*(mp*pc**2*1e5)/(2e33/(365*24*60**2))
data.append(np.percentile(Mdot[start_avg:,:], 50, axis=0))
plt.plot(X/pc, data[-1])
data.append(np.percentile(Mdot[start_avg:,:], 16, axis=0))
data.append(np.percentile(Mdot[start_avg:,:], 84, axis=0))
plt.fill_between(X/pc, data[-2], data[-1], alpha=0.5)
plt.ylabel(r'$\rm \dot{M} $')
plt.grid()
fig.suptitle(r'$\rm t/[Myr]\ \approx$ %.3f to %.3f'%(time_start, time_stop) )
plt.savefig('%s/plots/%d_Mdot-avg.png'%(base_dir,res))
plt.show()
plt.close(fig)


fig, _ = plt.subplots(3, 2, figsize=(8,12))    
ax1 = plt.subplot(321)
data.append(np.percentile(rho_all[start_avg:,:], 50, axis=0)/(mu*mp))
plt.plot(X/pc, data[-1])
data.append(np.percentile(rho_all[start_avg:,:], 16, axis=0)/(mu*mp))
data.append(np.percentile(rho_all[start_avg:,:], 84, axis=0)/(mu*mp))
plt.fill_between(X/pc, data[-2], data[-1], alpha=0.5)
#plt.ylim(min_den, max_den)
plt.ylabel('number density [cm^-3]')
plt.yscale('log')
ax1.grid()
# make these tick labels invisible
plt.setp(ax1.get_xticklabels(), visible=False)

ax2 = plt.subplot(323, sharex=ax1)
data.append(np.percentile(vr_all[start_avg:,:], 50, axis=0)/1e5)
plt.plot(X/pc, data[-1])
data.append(np.percentile(vr_all[start_avg:,:], 16, axis=0)/1e5)
data.append(np.percentile(vr_all[start_avg:,:], 84, axis=0)/1e5)
plt.fill_between(X/pc, data[-2], data[-1], alpha=0.5)
#plt.ylim(min_vel, max_vel)
plt.ylabel('velocity [km/s]')
ax2.grid()
# make these tick labels invisible
plt.setp(ax2.get_xticklabels(), visible=False)

ax3 = plt.subplot(325, sharex=ax1)
data.append(np.percentile(p_all[start_avg:,:], 50, axis=0)/kB)
plt.plot(X/pc, data[-1])
data.append(np.percentile(p_all[start_avg:,:], 16, axis=0)/kB)
data.append(np.percentile(p_all[start_avg:,:], 84, axis=0)/kB)
plt.fill_between(X/pc, data[-2], data[-1], alpha=0.5)
#plt.ylim(280, 310)
plt.ylabel('pressure [kB]')
ax3.grid()
# make these tick labels invisible
plt.setp(ax3.get_xticklabels(), visible=True)


ax4 = plt.subplot(322)
data.append(np.percentile(ent_all[start_avg:,:], 50, axis=0)/1e32)
plt.plot(X/pc, np.log10(data[-1]))
data.append(np.percentile(ent_all[start_avg:,:], 16, axis=0)/1e32)
data.append(np.percentile(ent_all[start_avg:,:], 84, axis=0)/1e32)
plt.fill_between(X/pc, np.log10(data[-2]), np.log10(data[-1]), alpha=0.5)
#plt.ylim(min_ent/1e32, max_ent/1e32)
#plt.yscale('log')
plt.ylabel(r'entropy [cgs] $\rm \times 10^{32}$')
ax4.grid()
# make these tick labels invisible
plt.setp(ax4.get_xticklabels(), visible=False)
'''
ax4 = plt.subplot(322)
data.append(np.percentile(tcool_all[start_avg:,:], 50, axis=0))
plt.plot(X/pc, data[-1])
data.append(np.percentile(tcool_all[start_avg:,:], 16, axis=0))
data.append(np.percentile(tcool_all[start_avg:,:], 84, axis=0))
plt.fill_between(X/pc, data[-2], data[-1], alpha=0.5)
#plt.ylim(min_ent/1e32, max_ent/1e32)
plt.yscale('log')
plt.ylabel(r'tcool $\rm [Myr]$')
ax4.grid()
# make these tick labels invisible
plt.setp(ax4.get_xticklabels(), visible=False)
'''
ax5 = plt.subplot(324, sharex=ax4)
data.append(np.percentile(mac_all[start_avg:,:], 50, axis=0))
plt.plot(X/pc, data[-1])
data.append(np.percentile(mac_all[start_avg:,:], 16, axis=0))
data.append(np.percentile(mac_all[start_avg:,:], 84, axis=0))
plt.fill_between(X/pc, data[-2], data[-1], alpha=0.5)
#plt.ylim(min_mac, max_mac)
plt.ylabel('mach')
ax5.grid()
# make these tick labels invisible
plt.setp(ax5.get_xticklabels(), visible=False)

ax6 = plt.subplot(326, sharex=ax4)
data.append(np.percentile(T_all[start_avg:,:], 50, axis=0))
plt.plot(X/pc, data[-1])
data.append(np.percentile(T_all[start_avg:,:], 16, axis=0))
data.append(np.percentile(T_all[start_avg:,:], 84, axis=0))
plt.fill_between(X/pc, data[-2], data[-1], alpha=0.5)
#plt.ylim(min_tmp, max_tmp)
plt.yscale('log')
plt.ylabel('temperature [K]')
ax6.grid()
# make these tick labels invisible
plt.setp(ax6.get_xticklabels(), visible=True)

fig.suptitle(r'$\rm t/[Myr]\ \approx$ %.3f to %.3f'%(time_start, time_stop) , y=0.92)
plt.savefig('%s/plots/%d_%s-avg.png'%(base_dir,res,sys.argv[1]))
plt.show()
plt.close(fig)

data = np.array(data).T
np.savetxt('%s/plots/%d_%s-avg.txt'%(base_dir,res,sys.argv[1]), data, fmt='%.5e',
           header = 'quantities are arranged 50percentile 16percentile and 84percentile respectively\ndistance[pc]\tMdot[Msun/yr]\tndens[cm^-3]\tvel[kms^-1]\tprs[kB]\tentropy[cgs]X1e32\tmach\ttemperature[K]')

'''
for file_no in range(start,int(sys.argv[2])):
    hfile = h5py.File('%s/data.%04d.dbl.h5'%(base_dir, file_no),'r')
    time = file_no*pc/1e5*freq
    X   = np.array(hfile['cell_coords/X'])*pc
    rho = np.array(hfile['Timestep_%d/vars/rho'%file_no])*mp
    vr = np.array(hfile['Timestep_%d/vars/vx1'%file_no])*1e5
    T  = np.array(hfile['Timestep_%d/vars/T'%file_no])
    p  = np.array(hfile['Timestep_%d/vars/prs'%file_no])*mp*1e10
    
    cs = np.sqrt(gamma*p/rho)
    mach = vr/cs
    #Mdot = 4*np.pi*X**2*rho*vr/(2e33/(365*24*60**2))
    entropy = p/rho**gamma
    
    hfile.close()
    #print(X)
    #print(X.shape[0])
    
    fig, _ = plt.subplots(3, 2, figsize=(8,12))
    
    ax1 = plt.subplot(321)
    plt.plot(X/pc, rho/(mu*mp))
    plt.ylim(min_den, max_den)
    plt.ylabel('number density [cm^-3]')
    plt.yscale('log')
    ax1.grid()
    # make these tick labels invisible
    plt.setp(ax1.get_xticklabels(), visible=False)
    
    ax2 = plt.subplot(323, sharex=ax1)
    plt.plot(X/pc, vr/1e5)
    plt.ylim(min_vel, max_vel)
    plt.ylabel('velocity [km/s]')
    ax2.grid()
    # make these tick labels invisible
    plt.setp(ax2.get_xticklabels(), visible=False)
    
    ax3 = plt.subplot(325, sharex=ax1)
    plt.plot(X/pc, p/kB)
    plt.ylim(min_prs, max_prs)
    plt.ylabel('pressure [kB]')
    ax3.grid()
    # make these tick labels invisible
    plt.setp(ax3.get_xticklabels(), visible=True)
    
    ax4 = plt.subplot(322)
    plt.plot(X/pc, entropy/1e32)
    plt.ylim(min_ent/1e32, max_ent/1e32)
    #plt.yscale('log')
    plt.ylabel(r'entropy [cgs] $\rm \times 10^{32}$')
    ax4.grid()
    # make these tick labels invisible
    plt.setp(ax4.get_xticklabels(), visible=False)
    
    ax5 = plt.subplot(324, sharex=ax4)
    plt.plot(X/pc, mach)
    plt.ylim(min_mac, max_mac)
    plt.ylabel('mach')
    ax5.grid()
    # make these tick labels invisible
    plt.setp(ax5.get_xticklabels(), visible=False)
    
    ax6 = plt.subplot(326, sharex=ax4)
    plt.plot(X/pc, T)
    plt.ylim(min_tmp, max_tmp)
    plt.yscale('log')
    plt.ylabel('temperature [K]')
    ax6.grid()
    # make these tick labels invisible
    plt.setp(ax6.get_xticklabels(), visible=True)
'''    
'''
    if (sys.argv[1]=='rho'): plt.plot(X/pc,rho/(mu*mp),label="%d"%file_no)
    elif (sys.argv[1]=='vel'):plt.plot(X/pc,vr/cs,label="%d"%file_no)
    elif (sys.argv[1]=='prs'):plt.plot(X/pc,p/kB,label="%d"%file_no)
    elif (sys.argv[1]=='temp'):plt.plot(X/pc,T,label="%d"%file_no)
    if (sys.argv[1]=='rho' or sys.argv[1]=='temp'):plt.yscale('log')
    '''
'''    
    if (file_no!=0): fig.suptitle(r'$\rm t/t_{cool,hot}\ \approx $ %.3f'%(time/tcool))
    else: fig.suptitle(r'$\rm t/t_{cool,hot}\ =$ %.3f'%(time/tcool))
    #plt.subplot_tool()
    #if (sys.argv[1]=='vel'): plt.ylim(top=max_val, bottom=min_val)
    #plt.xscale('log')
    plt.savefig('%s/plots/%s_%04d.png'%(base_dir,sys.argv[1],file_no))
    #plt.show()
    plt.close(fig)
    #plt.cla()
#plt.legend(loc='best')
#plt.show()
'''
