import os
import matplotlib.pyplot as plt
import pyPLUTO as pp
import numpy as np
import math
from scipy.optimize import newton

I = pp.Image()
plutodir = os.environ['PLUTO_DIR']
#point this to the data files directory
wdir=


pc=3.086e18
sec_year=3.154e7
kB=1.38e-16
Msun=2.e33
UNIT_DENSITY=1.67262171e-27
UNIT_LENGTH=3.086e21
UNIT_VELOCITY=1.e7
UNIT_TIME=UNIT_LENGTH/UNIT_VELOCITY

Om=0.27
Ol=0.73
H0=70.e5/3.086e24
h=0.7
G=6.67e-8
dcr0=3.*H0*H0/(8.*np.pi*G)
peak_height=4.0
gam=4.0
bet=6.0
nu=4.0
z_start=6.0

M0=1e14*Msun

mu=0.672442
KELVIN=1.20272e+06

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
    dm=dcr0*Om*(1.+z)**3.
    R200m=radii(time)[1]
    se=1.5-z/6.*0.2
    be=1.2+z/6.*3.5
    return dm*(be*(r/(5.*R200m))**-se + 1.)

def rho_dm(r,time):
    return rho_NFW(r,time)*f_tr(r,time)+rho_out(r,time)

def Phi_NFW(r,time):
    R200c=radii(time)[0]
    x=r/R200c
    c=ct(time)
    z=newton(t_inv,0,args=[time])
    M200=M(z)
    return -G*M200*np.log(1.+c*x)/(r*(np.log(1.+c)-c/(1.+c)))

def Phi_outer(r,time):
    z=newton(t_inv,0.1,args=[time])
    dm=dcr0*Om*(1.+z)**3.
    R200m=radii(time)[1]
    se=1.5-z/6.*0.2
    be=1.2+z/6.*3.5
    phi=(4.*np.pi*G)*dm*(be*(5.*R200m)**(se)*(r**(2.-se)/((3.-se)*(2.-se)))+r**2/6.)
    return phi

def sigmoid(x,ce):
    return math.erf(x*ce)

def Phi_cosm(r,time):
    z=newton(t_inv,0.1,args=[time])
    Omz=Om*pow((1.+z),3.)/Ezsqr(z)
    phi_cosm=-H0*H0*Ezsqr(z)*(1.-3.*Omz/2.)*(r*r)/2.
    return phi_cosm


def Phi_full(r,time):
    z=newton(t_inv,0.1,args=[time])
    dm=dcr0*Om*(1.+z)**3.
    R200m=radii(time)[1]
    f1 = Phi_NFW(r,time)
    f2 = Phi_outer(r,time)-2./3.*np.pi*G*dm*r*r
    rt=(1.9-0.18*nu)*R200m 
    w1 =1.-(1.+sigmoid(np.log10(r/rt),1.4))/2.
    w2 =1.-w1
    phi=f1*w1+f2*w2
    return phi

init_array()

cooltable=np.loadtxt('wiersma_cooltable.dat')
cooltable=np.transpose(cooltable)
temparray=cooltable[0]
lambdatable=cooltable[1]
mettable=cooltable[2]

ratiomin=[]
timearray=[]

plt.rc('lines', linewidth=2, color='r')
plt.rcParams.update({'font.size': 14})

fig = plt.figure(figsize=(30,10))
ax1=fig.add_subplot(231)
ax2=fig.add_subplot(232)
ax3=fig.add_subplot(233)
ax4=fig.add_subplot(234)
ax5=fig.add_subplot(235)
ax6=fig.add_subplot(236)

dt=(t(0)-t(z_start))/131.943
for i in range(132):
    if i%20==0:
        D=pp.pload(i,w_dir=wdir)
        time=dt*i
        R200c=radii(time)[0]/UNIT_LENGTH
        z=newton(t_inv,0,args=[time])
        ax1.loglog(D.x1,D.rho/1.e3,label=str('%.2f'%(time/1.e9/sec_year))+" Gyr") 
        ax2.loglog(D.x1/R200c,D.prs*UNIT_DENSITY*UNIT_VELOCITY**2,label=str('%.2f'%(time/1.e9/sec_year))+" Gyr") 

        ax4.plot(D.x1/R200c,D.vx1*UNIT_VELOCITY/1.e5,label=str('%.2f'%(time/1.e9/sec_year))+" Gyr")
        ax5.loglog(D.x1/R200c,D.prs/D.rho*KELVIN*mu,label=str('%.2f'%(time/1.e9/sec_year))+" Gyr")
        
        rarray=[]
        tffarray=[]
        tccarray=[]
        ratio=[]
        n=len(D.x1)
        for i in range(1,n):
            r=D.x1[i]*UNIT_LENGTH
            rarray.append(r/UNIT_LENGTH/R200c)
            g = (Phi_full(r+pc,time)-Phi_full(r,time))/(pc)
            tff = np.sqrt(2.*r/g) 
            tffarray.append(tff)

            avetemp=0
            avelam=0
            avedens=0
            angle=0

            tcctemp=0

            dens=D.rho[i]
            prs=D.prs[i]
            temp=prs/dens*KELVIN*mu

            high=len(temparray)-1
            low=0
            mid=int((low+high)/2)

            while (low != (high- 1)):
                mid=int((low+high)/2)
                Tmid=temparray[mid]
                if (temp <= Tmid):
                    high = mid
                elif (temp > Tmid):
                    low=mid 

            if(D.x1[i]<R200c): met=10**(-0.522878745+(z/6.)*(-1.5))
            else: met=10**(-2.+(z/6.)*(-3.))

            dT        = temparray[high]-temparray[low]
            scrh=lambdatable[low]*(temparray[high]-temp)/dT+lambdatable[high]*(temp-temparray[low])/dT+ met * mettable[low]*(temparray[high]-temp)/dT+mettable[high]*(temp-temparray[low])/dT

            tcc=3./2.*kB*temp/(D.rho[i]/1.e3)/(scrh*0.71049*0.717994)
            tccarray.append(tcc)
            ratio.append(tcc/tff)
        
        imax=0
        dist=1.e30
        for i in range(n):
            if(abs(D.x1[i]-R200c)<dist):
                dist=abs(D.x1[i]-R200c)
                imax=i

        ratiomin.append(np.min(ratio[5:imax]))
        timearray.append(time/1.e9/sec_year)

        ax3.loglog(rarray,ratio,label=str('%.2f'%(time/1.e9/sec_year))+" Gyr")
        ax6.plot(D.x1/R200c,4*np.pi*(D.x1*UNIT_LENGTH)**2*(D.rho*UNIT_DENSITY)*(D.vx1*UNIT_VELOCITY)/Msun*sec_year,label=str('%.2f'%(time/1.e9/sec_year))+" Gyr")

ax4.loglog(1.3*np.ones(100),-1*np.logspace(0,4,100),'k--',linewidth=3)
ax4.text(0.65,-2.e4,r"$r_{\rm sh}$",fontsize=16)
ax4.loglog(5.5*np.ones(100),-1*np.logspace(0,4,100),'k--',linewidth=3)
ax4.text(3.5,-2.e4,r"$r_{\rm ta}$",fontsize=16)

ax4.set_yscale('symlog')
ax4.set_xscale('log')
ax6.set_yscale('symlog')
ax6.set_xscale('log')
ax1.legend(fontsize=12)
ax2.legend(fontsize=12)
ax3.legend(fontsize=12)
ax4.legend(fontsize=12)
ax5.legend(fontsize=12)
ax6.legend(fontsize=12)
ax1.set_ylabel(r"$\rho \quad (m_p \  cm^{-3})$", fontsize=14)
ax2.set_ylabel(r"$P$ (dyne)", fontsize=14)
ax4.set_ylabel(r"$v_r \ (km \ s^{-1})$", fontsize=14)
ax5.set_ylabel("T (K)", fontsize=14)
ax3.set_ylabel("$t_{cool} / t_{ff}$",fontsize=14)
ax6.set_ylabel(r"$\dot{M} \ (M_\odot \rm{yr}^{-1})$", fontsize=14)
ax1.set_xlabel('$r/R_{200c}$', fontsize=14)
ax2.set_xlabel('$r/R_{200c}$', fontsize=14)
ax3.set_xlabel('$r/R_{200c}$', fontsize=14)
ax4.set_xlabel('$r/R_{200c}$', fontsize=14)
ax5.set_xlabel('$r/R_{200c}$', fontsize=14)
ax6.set_xlabel('$r/R_{200c}$', fontsize=14)
fig.suptitle(r"$M_0=10^{14} M_\odot$", fontsize=20)
plt.show()


