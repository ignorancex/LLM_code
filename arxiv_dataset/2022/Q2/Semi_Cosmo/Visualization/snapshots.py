import numpy as np
from scipy.optimize import newton
import math
import matplotlib.pyplot as plt
import os, sys
import pyPLUTO as pp
# sys.path.append(os.environ['PLUTO_DIR']+"/Tools/pyPLUTO/pyPLUTO") #access linux environment variable

plt.rcParams["figure.figsize"] = (9,12)
plt.rc('lines', linewidth=3, color='r')
plt.rcParams.update({'font.size': 20})

Om=0.27
Ol=0.73
H0=70.e5/3.086e24
h=0.7
G=6.67e-8
z_start=6.0
sec_year=3.154e7
dcr0=3.*H0*H0/(8.*np.pi*G)
Msun=2.e33
pc=3.086e18

mu=0.672442
KELVIN=1.20272e+06
kev=1.16e7
CONST_mp=1.67262171e-24

M0=1e14*Msun

zarray=[]
marray=[]

def Ezsqr(z):
    return Om*(1.+z)**3+(1.-Om)

def t(z):
    time=1/H0*2/3/np.sqrt(Ol)*np.log((np.sqrt(Ol/((1+z)**3))+np.sqrt(Ol/((1+z)**3)+Om))/np.sqrt(Om))
    return time

def t_inv(z,t0):
    return t(z)-t(z_start)-t0

def init_array():
    if M0==1.e11*Msun: f=np.loadtxt("1e11_mah.dat")
    elif M0==1.e12*Msun: f=np.loadtxt("1e12_mah.dat")
    elif M0==1.e13*Msun: f=np.loadtxt("1e13_mah.dat")
    elif M0==1.e14*Msun: f=np.loadtxt("1e14_mah.dat")
    elif M0==1.e15*Msun: f=np.loadtxt("1e15_mah.dat")
    f=np.transpose(f)
    n=len(f[0])
    for i in range(n):
        zarray.append(f[1][i])
        marray.append(f[3][i])

def radii(time):
    z=newton(t_inv,0,args=[time])
    if z<0: z=0
    dcr=dcr0*Ezsqr(z)
    M200=M(z)
    dm=dcr0*Om*(1.+z)**3.
    R200c = (3.*M200/(800.*np.pi*dcr))**(1./3.)
    R200m = (3.*M200/(800.*np.pi*dm))**(1./3.)
    return R200c,R200m

def M(z):
    low=0
    high=len(zarray)-1
    while(low!=high-1):
        mid=int((low+high)/2)
        zmid=zarray[mid]
        if(z<zmid): high=mid
        elif(z>zmid): low=mid
    dz=zarray[high]-zarray[low]
    m=marray[low]*(zarray[high]-z)/dz+marray[high]*(z-zarray[low])/dz
    return M0*10**m

I = pp.Image()
plutodir = os.environ['PLUTO_DIR']

# script to make 2-D snapshots of various quantities

#point this to the data files directory
wdir=

dt=(t(0)-t(z_start))/1319.43
init_array()

for i in range(1321):
        D=pp.pload(i, w_dir=wdir)

        x = np.outer(D.x1,np.sin(D.x2))
        z = np.outer(D.x1,np.cos(D.x2))
        time=dt*i
        R200c=radii(time)[0]/1.e3/pc

        plt.contourf(x, z, np.log10(D.prs/D.rho*KELVIN*mu),np.linspace(4,10,200))
        theta=0.10
        theta2=3.14-theta
        plt.title(str('%.2f'%(time/1.e9/sec_year))+" Gyr",fontsize=24)
        plt.xlabel("r (kpc)",fontsize=24)
        plt.ylabel("z (kpc)",fontsize=24)
        plt.xlim(0,500)
        plt.ylim(-500,500)
        plt.plot(R200c*np.ones(100),np.linspace(-500,500,100),'--',color='k')
        plt.colorbar()
        plt.savefig('video/'+str('%.4d'%i))
        plt.close()
