#Playing around with Diemer-Kravtsov 2014 DM profiles & associated potential
#Authors: Shashank Dattathri and Prateek Sharma
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.special import gammainc
from scipy.special import gamma
from scipy.integrate import quad
from scipy.special import erf

kpc=1.e3*3.086e18

def Ez(z):
    return np.sqrt( Om*(1.+z)**3+(1.-Om) )
def Ezsqr(z):
    return Om*(1.+z)**3+(1.-Om)
def delc(x): #x=Om(z)-1
    return 18.*np.pi**2 + 82.*x - 39.*x*x
def d_Ein(r):
    return ds*np.exp( -2.*((r/rs)**alp-1.)/alp )
def d_NFW(r):
    c=R200c/rs
    return 200.*c*c*c*dcr/(3.*(np.log(1.+c)-c/(1.+c)))/(r/rs*(1+r/rs)**2)
def f_tr(r):
    return (1. + (r/rt)**bet)**(-gam/bet)
def d_out(r):
    return dm*(be*(r/(5.*R200m))**-se + 1.)
def d_rho(r):
    return d_NFW(r)*f_tr(r)+d_out(r)
def Menc_NFW(r):
    x=r/R200c; c=R200c/rs
    return M200*x*( -c/(1.+c*x) + np.log(1.+c*x)/x )/( np.log(1.+c) - c/(1.+c) )
def Menc_Ein(r):
    return 2.*np.pi*ds*rs*rs*rs*np.exp(2./alp)*(alp/2.)**(-1.+3./alp)*gamma(3./alp)*(1.-gammainc(3./alp,2.*(r/rs)**alp/alp)/gamma(3./alp))
def Phi_NFW(r):
    x=r/R200c; c=R200c/rs
    return -G*M200*np.log(1.+c*x)/(r*(np.log(1.+c)-c/(1.+c)))
def dMbydr(M,r):
    return 4.*np.pi*r*r*d_rho(r)
def Menc(r):
    # sol = odeint( dMbydr, Menc_Ein(1.e-2*R200m), r )
    sol = odeint( dMbydr, Menc_NFW(1.e-2*R200m), r )
    return sol[:,0]
def Menc_phi(r):
    rarray = np.array([1.e-2*R200m, r])
    sol = odeint( dMbydr, Menc_NFW(1.e-2*R200m), rarray )
    # sol = odeint( dMbydr, Menc_Ein(1.e-2*R200m), rarray )
    return sol[1,0]
def dPhibydr(Phi,r):
    return G*Menc_phi(r)/(r*r)
def sigmoid(x,ce):
    # return (1.+erf(x))/ce
    # return np.tanh(x*ce)
    return erf(x*ce)
def first_term(r):
    return (d_rho(r)-dm)*r**2 
    # return d_rho(r)*r**2
def second_term(r):
    return (d_rho(r)-dm)*r
    # return d_rho(r)*r
def phi_real(r):
    phi=-4*np.pi*G*(quad(first_term,1e-3*R200m,r)[0]/r+quad(second_term,r,9*R200m)[0])
    return phi
def Phi_outer(r):
    phi=(4.*np.pi*G)*dm*(be*(5.*R200m)**(se)*(r**(2.-se)/((3.-se)*(2.-se)))+r**2/6.)
    return phi
global Om, ds, alp, rs, bet, rt, gam, dm, se, dcr, R200m, R200c, M200, G
#cosmological, astrophysical parameters
H0 = 70.e5/3.086e24; G=6.67e-8; z=6.0; Om = 0.27; dcr0 = 3.0*H0*H0/(8.*np.pi*G)
dcr=dcr0*Ezsqr(z); Msun=2.e33; dm=dcr0*Om*(1.+z)**3
#halo parameters; R200c used for NFW, R200m for Ein
# M200 = 1.e14*Msun; 
M200 = 1.e12*Msun
c=4.0; nu=4.0; #
# c=4.0; nu=0.5 #parameters for low mass halos
alp=0.155+0.0095*nu*nu 
R200c = (3.*M200/(800.*np.pi*dcr))**(1./3.); R200m = (3.*M200/(800.*np.pi*dm))**(1./3.)
rs=R200c/c; R200m=(Ezsqr(z)/(Om*(1.+z)**3))**(1./3.)*R200c

ds=M200/(2.*np.pi*rs*rs*rs*np.exp(2./alp)*(alp/2.)**(3./alp-1.)*gamma(3./alp)*gammainc(3./alp,2.*(R200m/rs)**alp/alp))
#parameters for outer profile
bet=6.0; gam=4.0; #bet=4.0; gam=8.0;
be = 5.5; se=1.6; rt=(1.9-0.18*nu)*R200m 

def main():
    rlen=200
    r=kpc*np.logspace(0,4,rlen); 
    sol_phi = odeint( dPhibydr, Phi_NFW(1.e-2*R200m), r)
    phi_sol=np.zeros(rlen)
    for i in range(rlen):
        phi_sol[i]=sol_phi[i]-2./3.*np.pi*G*dm*r[i]*r[i]
    g = G*Menc(r)/(r*r); 
    tff = np.sqrt(2.*r/g)
    rscale1=10*R200c; r1=10*R200c; p1=1./3.; rscale2=10.*R200c; r2=10.*R200c; p2=1./3.

    f1 = Phi_NFW(r); f2 = Phi_outer(r)
    w1=1-(1.+sigmoid(np.log10(r/rt),1.4))/2.
    w2=(1-w1)
    
    phi_fit=f1*w1+f2*w2-2./3.*np.pi*G*dm*r*r

    phi_true=np.zeros(rlen)
    for i in range(rlen):
        phi_true[i]=phi_real(r[i])

    rm1=r[:-1]; rm2=rm1[:-1]; rm3=rm2[:-1]

    g_true=np.diff(-phi_true)/np.diff(r)
    g_fit=np.diff(-phi_fit)/np.diff(r)
    g_sol=np.diff(-phi_sol)/np.diff(r)

    rho_fit = np.diff(rm1*rm1*(-g_fit))/np.diff(rm1*rm1*rm1/3.)/(4.*np.pi*G)
    rho_true=np.diff(rm1*rm1*(-g_true))/np.diff(rm1*rm1*rm1/3.)/(4.*np.pi*G)
    rho_sol=np.diff(rm1*rm1*(-g_sol))/np.diff(rm1*rm1*rm1/3.)/(4.*np.pi*G)

    slope_fit=np.diff(np.log(rho_fit))/(np.diff(np.log(rm2)))
    slope_true=np.diff(np.log(rho_true))/(np.diff(np.log(rm2)))
    slope_sol=np.diff(np.log(rho_sol))/(np.diff(np.log(rm2)))

    fig = plt.figure(figsize=(20,10))
    ax1=fig.add_subplot(221)
    ax2=fig.add_subplot(222)
    ax3=fig.add_subplot(223)
    ax4=fig.add_subplot(224)


    ax1.loglog(r/R200c,phi_fit-phi_fit[0],label="Fit")
    ax1.loglog(r/R200c,phi_true-phi_true[0],label='Construction 2')
    ax1.loglog(r/R200c,phi_sol-phi_sol[0],label="Construction 1")
    ax1.set_xlabel("R/$R_{200c}$",fontsize=14)
    ax1.set_ylabel("$\Phi$(R)",fontsize=14)
    ax1.set_title("Gravitational potential",fontsize=16)
    ax1.legend(fontsize=12)

    ax2.loglog(rm1/R200c,-g_fit)
    ax2.loglog(rm1/R200c,-g_true)
    ax2.loglog(rm1/R200c,-g_sol)
    ax2.set_xlabel("R/$R_{200c}$",fontsize=14)
    ax2.set_title("Gravitational acceleration",fontsize=16)
    ax2.set_ylabel("g(R)",fontsize=14)

    ax3.loglog(rm2/R200c,rho_fit)
    ax3.loglog(rm2/R200c,rho_true)
    ax3.loglog(rm2/R200c,rho_sol)
    ax3.loglog(rm2/R200c,d_rho(rm2)-dm)
    ax3.set_xlabel("R/$R_{200c}$",fontsize=14)
    ax3.set_title("Density",fontsize=16)
    ax3.set_ylabel(r'$\rho (R)$',fontsize=14)

    ax4.semilogx(rm3/R200c,slope_fit)
    ax4.semilogx(rm3/R200c,slope_true)
    ax4.semilogx(rm3/R200c,slope_sol)
    ax4.set_xlabel("R/$R_{200c}$",fontsize=14)
    ax4.set_ylabel(r"$\frac{d \log(\rho)}{d \log(r)}$",fontsize=14)
    ax4.set_title("Density power law index",fontsize=16)
    plt.suptitle("$M_0=10^{14} M_\odot$",fontsize=20)
    plt.show()

if __name__ == "__main__":
    main()
