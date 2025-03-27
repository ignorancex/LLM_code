import os
import matplotlib.pyplot as plt
import pyPLUTO as pp
import numpy as np
from scipy.optimize import newton
from scipy.integrate import quad
from scipy.integrate import simps
plt.rc('lines', linewidth=3, color='r')
plt.rcParams.update({'font.size': 14})

plutodir = os.environ['PLUTO_DIR']
#point this to the data files directory
wdir=

pc=3.086e18
sec_year=3.154e7
Msun=2.e33
UNIT_DENSITY=1.67262171e-27
UNIT_LENGTH=3.0856775807e21
UNIT_VELOCITY=1.e7
UNIT_TIME=UNIT_LENGTH/UNIT_VELOCITY

Om=0.27
Ol=1.-Om
H0=70.e5/3.086e24
h=0.7
G=6.67e-8
dcr0=3.*H0*H0/(8.*np.pi*G)
peak_height=4.0
gam=4.0
bet=6.0
z_start=6.0

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

def z4():
    if M0==1.e11*Msun: return 5.8332
    elif M0==1.e12*Msun: return 4.8747
    elif M0==1.e13*Msun: return 3.9257
    elif M0==1.e14*Msun: return 3.0139
    elif M0==5.e14*Msun: return 2.4212
    elif M0==1.e15*Msun: return 2.179

def init_array():
    if M0==1.e11*Msun: f=np.loadtxt("1e11_mah.dat")
    elif M0==1.e12*Msun: f=np.loadtxt("1e12_mah.dat")
    elif M0==1.e13*Msun: f=np.loadtxt("1e13_mah.dat")
    elif M0==1.e14*Msun: f=np.loadtxt("1e14_mah.dat")
    elif M0==5.e14*Msun: f=np.loadtxt("5e14_mah.dat")
    elif M0==1.e15*Msun: f=np.loadtxt("1e15_mah.dat")
    
    f=np.transpose(f)
    n=len(f[0])
    for i in range(n):
        zarray.append(f[1][i])
        marray.append(f[3][i])

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

def ct(in_t):
    time=in_t+t(z_start)
    t04=t(z4())
    c=4.*(1.+(time/3.4/t04)**(6.5))**(1./8.)
    return c

def radii(time):
    z=newton(t_inv,0,args=[time])
    if z<0: z=0
    dcr=dcr0*Ezsqr(z)
    M200=M(z)
    dm=dcr0*Om*(1.+z)**3.
    R200c = (3.*M200/(800.*np.pi*dcr))**(1./3.)
    R200m = (3.*M200/(800.*np.pi*dm))**(1./3.)
    return R200c,R200m

def rho_NFW(r,time):
    z=newton(t_inv,0,args=[time])
    c=ct(time)
    dcr=dcr0*Ezsqr(z)
    R200c=radii(time)[0]
    rs=R200c/c
    return 200.*c*c*c*dcr/(3.*(np.log(1.+c)-c/(1.+c)))/(r/rs*(1+r/rs)**2)

def f_tr(r,time):
    R200m=radii(time)[1]
    rt=(1.9-0.18*peak_height)*R200m
    return (1. + (r/rt)**bet)**(-gam/bet)

def rho_out(r,time):
    z=newton(t_inv,0.1,args=[time])
    se=1.5-z/6.*0.2
    be=1.2+z/6.*3.5
    dm=dcr0*Om*(1.+z)**3.
    R200m=radii(time)[1]
    return dm*(be*(r/(5.*R200m))**-se + 1.)

def rho_dm(r,time):
    return rho_NFW(r,time)*f_tr(r,time)+rho_out(r,time)

def main():
    init_array()
    for j in range(132):
        if j%20==0:
            fracarray=[]
            frac2array=[]
            dt=(t(0)-t(z_start))/131.943
            D=pp.pload(j,w_dir=wdir)
            time=dt*j
            z=newton(t_inv,0,args=[time])
            R200c=radii(time)[0]/1.e3/pc

            if j==0: color='k'
            elif j==30: color='r'
            elif j==60: color='y'
            elif j==90: color='g'
            elif j==120: color='b'

            ilim=len(D.x1)
            M_gas=simps(4.*np.pi*D.rho[:ilim]*D.x1[:ilim]**2,D.x1[:ilim])
            M_tot=simps(4.*np.pi*(D.rho[:ilim]+rho_dm(D.x1[:ilim]*1e3*pc,time)/UNIT_DENSITY)*D.x1[:ilim]**2,D.x1[:ilim])
            baryon_fraction=M_gas/M_tot

            n=len(D.x1)
            for i in range(1,n):
                M_gas=simps(4.*np.pi*D.rho[:i]*D.x1[:i]**2,D.x1[:i])
                M_tot=simps(4.*np.pi*(D.rho[:i]+rho_dm(D.x1[:i]*1e3*pc,time)/UNIT_DENSITY)*D.x1[:i]**2,D.x1[:i])
                fracarray.append(M_gas/M_tot/(1./6.))
                frac2array.append(D.x1[i]/R200c)
            plt.semilogx(frac2array,fracarray,label=str('%.2f'%(time/1.e9/sec_year))+" Gyr")
    plt.xlabel("$r/R_{200c}$")
    plt.ylabel("$f_b(<r)$ (normalized to the universal value)")
    plt.title(r"$M_0=10^{14} M_\odot$")
    plt.legend()
    plt.ylim(0,2)
    plt.show()
    

if __name__ == "__main__":
    main()
