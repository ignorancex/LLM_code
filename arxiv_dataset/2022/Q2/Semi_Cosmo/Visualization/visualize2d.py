import os
import matplotlib.pyplot as plt
import pyPLUTO as pp
import numpy as np
from scipy.optimize import newton
import math

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
nu=4.0
gam=4.0
bet=6.0
z_start=6.0

M0=1e14*Msun

mu=0.672442
KELVIN=1.20272e+06
kev=1.16e7
CONST_mp=1.67262171e-24

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
    elif M0==1.e15*Msun: return 2.179

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
    # phi=f1*w1+f2*w2+Phi_cosm(r,time)
    phi=f1*w1+f2*w2
    return phi

def find_lam(temp,r,z,R200c):
    if(r<R200c): met=10**(-0.522878745+(z/6.)*(-1.5))
    else: met=10**(-2.+(z/6.)*(-3.))

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

    dT  = temparray[high]-temparray[low]
    lam=lambdatable[low]*(temparray[high]-temp)/dT+lambdatable[high]*(temp-temparray[low])/dT+ met *(mettable[low]*(temparray[high]-temp)/dT+mettable[high]*(temp-temparray[low])/dT)
    return lam

init_array()
jetPowerArray=[]
timearray=[]

cooltable=np.loadtxt('wiersma_cooltable.dat')
cooltable=np.transpose(cooltable)
temparray=cooltable[0]
lambdatable=cooltable[1]
mettable=cooltable[2]

ratiomin=[]

plt.rc('lines', linewidth=2, color='r')
plt.rcParams.update({'font.size': 14})

fig = plt.figure(figsize=(30,10))
ax1=fig.add_subplot(231)
ax2=fig.add_subplot(232)
ax3=fig.add_subplot(233)
ax4=fig.add_subplot(234)
ax5=fig.add_subplot(235)
ax6=fig.add_subplot(236)

dt=(t(0)-t(z_start))/1319.43
for num in range(1321):
    if num%200==0:
        D=pp.pload(num,w_dir=wdir)
        time=dt*num
        R200c=radii(time)[0]/1.e3/pc
        z=newton(t_inv,0,args=[time])
        
        x1len=len(D.x1)
        x2beg=4
        x2end=len(D.x2)-x2beg-1

        dens_array=np.zeros(x1len)
        prs_array=np.zeros(x1len)
        temp_array=np.zeros(x1len)
        vel_array=np.zeros(x1len)
        mdot_array=np.zeros(x1len)

        for j in range(x1len):
            mdot=0
            dens_array[j]=np.sum((D.rho)[j][x2beg:x2end]*np.sin(D.x2[x2beg:x2end])*D.dx2[x2beg:x2end])/np.sum(np.sin(D.x2[x2beg:x2end])*D.dx2[x2beg:x2end])
            prs_array[j]=np.sum((D.prs)[j][x2beg:x2end]*np.sin(D.x2[x2beg:x2end])*D.dx2[x2beg:x2end])/np.sum(np.sin(D.x2[x2beg:x2end])*D.dx2[x2beg:x2end])
            vel_array[j]=np.sum((D.vx1)[j][x2beg:x2end]*np.sin(D.x2[x2beg:x2end])*D.dx2[x2beg:x2end])/np.sum(np.sin(D.x2[x2beg:x2end])*D.dx2[x2beg:x2end])
            temp_array[j]=prs_array[j]/dens_array[j]*KELVIN*mu

        rarray=[]
        dens_array_2=[]
        prs_array_2=[]
        t_array_2=[]
        tffarray=[]
        tccarray=[]
        ratio=[]


        for i in range(x1len):
            r=D.x1[i]*UNIT_LENGTH
            rarray.append(r/UNIT_LENGTH)
            g = (Phi_full(r+pc,time)-Phi_full(r,time))/(pc)
            tff = np.sqrt(2.*r/g) 
            tffarray.append(tff)

            dens_ave=0
            prs_ave=0
            norm=0
            mdot=0

            for j in range(len(D.x2)):
                temp=D.prs[i][j]/D.rho[i][j]*KELVIN*mu

                if(temp>0.5*kev and temp<8.*kev):
                    lam=find_lam(temp,D.x1[i],z,R200c)
                    dens_ave+=D.rho[i][j]*np.sin(D.x2[j])*D.dx2[j]
                    prs_ave+=D.prs[i][j]*np.sin(D.x2[j])*D.dx2[j]
                    norm+=np.sin(D.x2[j])*D.dx2[j]

                if(j>x2beg and j<x2end):
                    mdot+=(D.x1[i]*UNIT_LENGTH)**2*D.dx2[j]*2*np.pi*(D.rho[i][j]*UNIT_DENSITY)*(D.vx1[i][j]*UNIT_VELOCITY)

            mdot_array[i]=mdot

            if(norm==0): 
                tccarray.append(0)
                ratio.append(0)
                dens_array_2.append(0)
                prs_array_2.append(0)
                t_array_2.append(0)
                continue

            dens_ave=dens_ave/norm
            prs_ave=prs_ave/norm
            temp=prs_ave/dens_ave*KELVIN*mu

            dens_array_2.append(dens_ave)
            prs_array_2.append(prs_ave)
            t_array_2.append(temp)

            lam=find_lam(temp,D.x1[i],z,R200c)

            num_dens=dens_ave*UNIT_DENSITY/CONST_mp/mu
            ne=dens_ave*UNIT_DENSITY/CONST_mp*0.717994
            ni=dens_ave*UNIT_DENSITY/CONST_mp*0.71049

            tcc=3./2.*kB*temp*(num_dens)/(lam*ne*ni)  
            tccarray.append(tcc)
            ratio.append(tcc/tff)
        
        ax1.loglog(D.x1/R200c,dens_array*UNIT_DENSITY/CONST_mp,label=str('%.2f'%(time/1.e9/sec_year))+" Gyr") 
        # # ax1.loglog(D.x1/R200c,rho_dm(D.x1*UNIT_LENGTH,time)/UNIT_DENSITY/1.e3,'--')
        ax2.loglog(D.x1/R200c,prs_array*UNIT_DENSITY*UNIT_VELOCITY**2,label=str('%.2f'%(time/1.e9/sec_year))+" Gyr") 
        ax4.plot(D.x1/R200c,vel_array*UNIT_VELOCITY/1.e5,label=str('%.2f'%(time/1.e9/sec_year))+" Gyr")
        ax5.loglog(D.x1/R200c,temp_array,label=str('%.2f'%(time/1.e9/sec_year))+" Gyr")
        ax3.loglog(D.x1/R200c,ratio,label=str('%.2f'%(time/1.e9/sec_year))+" Gyr")
        ax6.plot(D.x1/R200c,mdot_array/Msun*sec_year,label=str('%.2f'%(time/1.e9/sec_year))+" Gyr")

ax1.legend(fontsize=14)
ax2.legend(fontsize=14)
ax3.legend(fontsize=14)
ax4.legend(fontsize=14)
ax5.legend(fontsize=14)
ax6.legend(fontsize=14)
ax4.set_yscale('symlog')
ax4.set_xscale('log')
ax6.set_yscale('symlog')
ax6.set_xscale('log')
ax1.set_ylabel(r"$\rho \quad (m_p \ cm^{-3})$",fontsize=14)
ax2.set_ylabel(r"$P$ (dyne)", fontsize=14)
ax4.set_ylabel("$v_r$ $(km \ s^{-1})$",fontsize=14)
ax3.set_ylabel(r"$t_{\rm cool} / t_{\rm ff}$",fontsize=14)
ax5.set_ylabel("T (K)", fontsize=14)
ax6.set_ylabel(r"$\dot{M} \ (M_\odot \rm{yr}^{-1})$", fontsize=14)

ax4.plot(5.5*np.ones(100),-1*np.logspace(0,4,100),'k--',linewidth=3)
ax4.text(3.5,-2.e4,r"$r_{\rm ta}$",fontsize=16)
ax4.plot(1.3*np.ones(100),-1*np.logspace(1,4,100),'k--',linewidth=3)
ax4.text(0.65,-2.e4,r"$r_{\rm sh}$",fontsize=16)

ax1.set_xlabel("$r/R_{200c}$",fontsize=14)
ax2.set_xlabel("$r/R_{200c}$",fontsize=14)
ax3.set_xlabel("$r/R_{200c}$",fontsize=14)
ax4.set_xlabel("$r/R_{200c}$",fontsize=14)
ax5.set_xlabel('$r/R_{200c}$', fontsize=14)
ax6.set_xlabel('$r/R_{200c}$', fontsize=14)
fig.suptitle("$M_0=10^{14} M_\odot$\n"+"$\epsilon=10^{-4}$\n"+r"Volume-averaged quantities (excluding poles)",fontsize=20)
# fig.suptitle(r"$M_0=10^{14} M_\odot$",fontsize=20)
plt.show()


