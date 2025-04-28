import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import interp1d
from scipy.integrate import simps
import time
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
rc('text', usetex=True)
matplotlib.rcParams['axes.linewidth'] = 1.2 #set the value globally

#####################################
# Unit conversions
ergToMeV = 6.24151*10**5 
yrTos = 24*3600*365
kpcTocm = 3.086*10**21
MpcTocm = 3.086*10**24
MeVTocm = 5.06*10**10
MeVTokeV = 1000.
sToyr = 1./(365*24*60*60.)
kgToton = 1.e-3
tyTokgs = 1000.*365*24*60*60
kgTokeV = 5.60959*10**32
# Constants
clight = 2.9979e5 # km s^-1
me = 0.511 # MeV
dsun = 8.5*kpcTocm # cm
vesc_gal = 550. # km s^-1, Galactic escape velocity
# PMNS matrix
theta12 = 33.82*np.pi/180
theta23 = 48.3*np.pi/180
theta13 = 8.61*np.pi/180
delta_cp = 222.*np.pi/180
U23 = np.array([[1., 0., 0.], [0., np.cos(theta23), np.sin(theta23)], [0., -np.sin(theta23), np.cos(theta23)]])
U13 = np.array([[np.cos(theta13), 0., np.sin(theta13)*np.exp(-1j*delta_cp)], [0., 1., 0.],\
               [-np.sin(theta13)*np.exp(1j*delta_cp), 0., np.cos(theta13)]])
U12 = np.array([[np.cos(theta12), np.sin(theta12), 0.], [-np.sin(theta12), np.cos(theta12), 0.], [0., 0., 1.]])
U_pmns = abs(np.dot(np.dot(U23,U13),U12))

# Dfactor for MW NFW halo
Dfactor = 2.0213990756088443e+25

##---------------------------------------------------------##
def MB(v,vs): # 1D MB distribution
    return np.exp(-pow(v/vs, 2))/(np.sqrt(np.pi)*vs)

########## DM-nu scattering ###########
# t-channel process
def xsection_chi_nu(Enu,mchi,mphi,alpha): # cm^2
    if Enu < mchi/4.*(1. - mchi**2/mphi**2)/10.:
        return MeVTocm**(-2)*(2.*np.pi*alpha**2*Enu**2)/(mchi**2 - mphi**2)**2
    else:
        return MeVTocm**(-2)*((np.pi*alpha**2*(-mphi**4 *(2.*Enu*mchi + mchi**2)**2 + 2.*mchi**2*mphi**2*(2.*Enu*mchi + mchi**2)**2 - mchi**4 *(2.*Enu*mchi + mchi**2)*(2.*Enu*mchi + mchi**2 + mphi**2) - 2.*(2.*Enu*mchi + mchi**2)*(mchi**2 - mphi**2)*(mchi**4 - mphi**2*(2.*Enu*mchi + mchi**2))*np.log(mphi**2 - mchi**4/(2.*Enu*mchi + mchi**2)) + mchi**8))/(8.*Enu**2*mchi**2*(2.*Enu*mchi + mchi**2)*(mchi**4 - mphi**2*(2.*Enu*mchi + mchi**2))) - 1./(8*Enu**2*mchi**2*(2.*Enu*mchi - mchi**2 + mphi**2)) * np.pi*alpha**2*(-mphi**2*(2.*Enu*mchi + mchi**2) + 2.*(mchi**2 - mphi**2)*(-2.*Enu*mchi + mchi**2 - mphi**2)*np.log(2.*Enu*mchi - mchi**2 + mphi**2) + 4.*mchi**2*(2.*Enu*mchi + mchi**2) - (2.*Enu*mchi + mchi**2)**2 - 3.*mchi**4 + mphi**4))

def dsigmadEnup(Enuini, Enufin, mchi, mphi, alpha): # MeV^-3
    return ( np.pi*alpha**2* Enufin**2* mchi)/(Enuini**2 * (2.*Enufin*mchi-mchi**2+mphi**2)**2)

# s-channel process
def xsection_chi_nu_s(Enu,mchi,mphi,alpha): # cm^2
    return MeVTocm**(-2)*(alpha**2 * Enu**2 * mchi**2) / ((2.* Enu * mchi + mchi**2) * (-2.* Enu * mchi - mchi**2 + mphi**2)**2)

def dsigmadEnup_s(Enuini, Enufin, mchi, mphi, alpha): # MeV^-3
    return (np.pi * alpha**2 * mchi) / (2.* Enu * mchi + mchi**2 - mphi**2)**2
    
######## MW halo #########
def nchi(z, mchi): # cm^-3
    return 1.24e-3*(1 + z)**3/mchi

def H(z): # cm^-1
    return 67.* np.sqrt(0.68 + 0.32*(1 + z)**3)*((3.24e-20)/(3.e10))

def rho_NFW(r, r_s=16.*kpcTocm, rho_s=0.471e3): # MeV cm^-3
    return rho_s/((r/r_s)*(1 + (r/r_s))**2)

def rho_NFW_los(l, psi): # MeV cm^-3
    return rho_NFW(np.sqrt(l**2 + dsun**2 - 2.*l*dsun*np.cos(psi)))

######## SFR & SN nu flux ########
def RCCSN(z):
    z1 = 1.
    z2 = 4.
    eta = -10.
    rho0_SFR = 0.0178
    alpha_SFR = 3.4
    beta_SFR = -0.3
    gamma_SFR = -3.5

    BSFR = (1 + z1)**(1 - alpha_SFR/beta_SFR)
    CSFR = (1 + z1)**((beta_SFR - alpha_SFR)/gamma_SFR) * (1 + z2)**(1 - beta_SFR/gamma_SFR)
    return rho0_SFR/143. *((1. + z)**(alpha_SFR*eta) + ((1. + z)/BSFR)**(beta_SFR*eta) + ((1. + z)/CSFR)**(gamma_SFR*eta))**(1/eta)

def SNnue_flux(Enu, Te=6.6): # MeV^-1
    Enu_tot = 3.e53*ergToMeV
    return (Enu_tot/6.) * (120/7./np.pi**4) * (Enu**2/Te**4) * 1./(np.exp(Enu/Te)+1.)

def SNnuebar_flux(Enu, Tebar=7.): # MeV^-1
    Enu_tot = 3.e53*ergToMeV
    return (Enu_tot/6.) * (120/7./np.pi**4) * (Enu**2/Tebar**4) * 1./(np.exp(Enu/Tebar)+1.)

def SNnux_flux(Enu, Tx=10.): # MeV^-1
    Enu_tot = 3.e53*ergToMeV
    return (Enu_tot/6.) * (120/7./np.pi**4) * (Enu**2/Tx**4) * 1./(np.exp(Enu/Tx)+1.)

def Lumi_NO(z,Enu,index):
    if index==3:
        return RCCSN(z)*SNnue_flux(Enu)
    elif index==4:
        return RCCSN(z)*SNnuebar_flux(Enu)
    else:
        return RCCSN(z)*SNnux_flux(Enu)


# Compute and save DSNB spectrum for mass states
def make_DSNB_flux(): # MeV^-1 cm^-2 s^-1
    Enlist = np.logspace(-4.,4.,num=500)
    zlist = np.linspace(0.,6.,num=18)
    fluxlist = []
    for en in Enlist:
        x = [en]
        for index in range(6):
            integral = np.array([1./H(zx)*Lumi_NO(zx,en*(1+zx),index+1)/yrTos/MpcTocm**3 for zx in zlist])
            x.append(np.trapz(integral, zlist))
        fluxlist.append(x)
    np.savetxt('DSNB_flux.csv', fluxlist, delimiter=',')
    
def make_solar_nu_flux():
    Enlist = np.logspace(2.,9.,num=400)
    # Solar
    interps = []
    solar_fluxes = ['pp','B8','O15','N13','hep']
    for file in solar_fluxes:
        data = np.array(np.loadtxt('Sun-nuclear-'+file+'.dat'))
        if len(data[0]) > 2: # Averaging where band is given
            data = np.array([[data[i,0], 0.5*(data[i,1]+data[i,2])] for i in range(len(data))])
        pp_temp = np.array([[x, data[0,1]*(x/data[0,0])**2] for x in np.logspace(2.,3.9,num=5)])  #7.99e-6*x**2
        data = np.concatenate((pp_temp,data))

        if data[-1,0] < 1.e9:
            x = np.logspace(np.log10(data[-1,0])+0.1, 9., num=6)
            temp = np.array([[en,1.e-20] for en in x])
            data = np.concatenate((data,temp))
        interps.append(interp1d(data[::,0],data[::,-1], kind='linear'))

    U_inv = np.linalg.inv(U_pmns)
    all_nu = []
    for i, en in enumerate(Enlist):
        nu_e = np.sum([flux(en) for flux in interps])
        all_nu.append([en*1.e-6, U_inv[0,0]**2*nu_e*1.e6, U_inv[1,0]**2*nu_e*1.e6, U_inv[2,0]**2*nu_e*1.e6])
    all_nu = np.array(all_nu)
    np.savetxt('solar_nu_flux.csv', all_nu, delimiter=',')
    
def make_all_nu_flux():
    Enlist = np.logspace(2.,9.,num=400)
    # Solar
    interps = []
    solar_fluxes = ['pp','B8','O15','N13','hep']
    for file in solar_fluxes:
        data = np.array(np.loadtxt('Sun-nuclear-'+file+'.dat'))
        if len(data[0]) > 2: # Averaging where band is given
            data = np.array([[data[i,0], 0.5*(data[i,1]+data[i,2])] for i in range(len(data))])
        pp_temp = np.array([[x, data[0,1]*(x/data[0,0])**2] for x in np.logspace(2.,3.9,num=5)])  #7.99e-6*x**2
        data = np.concatenate((pp_temp,data))

        if data[-1,0] < 1.e9:
            x = np.logspace(np.log10(data[-1,0])+0.1, 9., num=6)
            temp = np.array([[en,1.e-20] for en in x])
            data = np.concatenate((data,temp))
        interps.append(interp1d(data[::,0],data[::,-1], kind='linear'))

    U_inv = np.linalg.inv(U_pmns)
    # import & combine with DSNB
    dsnb_data = np.loadtxt('DSNB_flux.csv', delimiter=',', unpack=True)
    all_nu = []
    for i, en in enumerate(Enlist):
        nu_e = np.sum([flux(en) for flux in interps])
        all_nu.append([en*1.e-6, U_inv[0,0]**2*nu_e*1.e6+dsnb_data[1,i], U_inv[1,0]**2*nu_e*1.e6+dsnb_data[2,i],\
                       U_inv[2,0]**2*nu_e*1.e6+dsnb_data[3,i]])
    all_nu = np.array(all_nu)
    np.savetxt('all_nu_flux.csv', all_nu, delimiter=',')

def make_all_nu_flux_plot():
    flux = np.loadtxt('all_nu_flux.csv', delimiter=',', unpack=True)
    #dsnb = np.loadtxt('DSNB_flux.csv', delimiter=',', unpack=True)
    fig, ax = plt.subplots(1,1, figsize=(8,5))
    ax.loglog(flux[0],flux[1], color='b', linewidth=2, label=r'$F_{\nu_1}$')
    ax.loglog(flux[0],flux[2], color='g', linewidth=2, label=r'$F_{\nu_2}$')
    ax.loglog(flux[0],flux[3], color='gray', linewidth=2, label=r'$F_{\nu_3}$')
    #ax.loglog(flux[0],dsnb[1], '--', color='b', linewidth=2)
    #ax.loglog(flux[0],dsnb[2], '--', color='g', linewidth=2)
    #ax.loglog(flux[0],dsnb[3], '--', color='gray', linewidth=2)
    ax.set_xlabel(r'$E_\nu\,[\mathrm{MeV}]$', fontsize=18)
    ax.set_ylabel(r'$F_\nu\,[\mathrm{MeV^{-1}cm^{-2}s^{-1}}]$', fontsize=18)
    ax.legend(loc='lower left', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    plt.savefig('all_nu_flux.pdf', bbox_inches='tight')
    
def D_factor(): # cm^-2
    llist = np.logspace(1.,24.,num=100)
    psilist = np.logspace(-4.,np.log10(np.pi),num=20)
    integral = np.array([[rho_NFW_los(los,psi)*np.sin(psi)*2.*np.pi/4./np.pi for los in llist] for psi in psilist])
    return simps(simps(integral,llist),psilist)

def Boosted_DM_flux(mchi, sigma, indices=[1,2,3]): # MeV^-1 cm^-2 s^-1
    Tchilist = np.logspace(-6.,1.9,num=43)
    DM_flux = []
    dsnb_flux = np.loadtxt('DSNB_flux.csv', delimiter=',', unpack=True)
    nu_flux = dsnb_flux[1]+dsnb_flux[2]+dsnb_flux[3]
    nu_flux_interp = interp1d(dsnb_flux[0], nu_flux, kind='linear')
    for Tchi in Tchilist:
        Enlist = np.logspace(np.log10(Tchi/2.*(1+np.sqrt(1+2.*mchi/Tchi))), 4., num=30)
        integral = np.array([Dfactor*nu_flux_interp(en)*sigma/mchi *(en+mchi/2.)/en**2 for en in Enlist])
        DM_flux.append([Tchi, np.trapz(integral, Enlist)])
    return np.array(DM_flux)
    
def Detector_response(Te,Er,sigma):
    return 1./(np.sqrt(2*np.pi)*sigma)*np.exp(-(Er - Te)**2/(2.*sigma**2))
    
def TchiMax(Enu, mchi):
        return Enu**2/(Enu + mchi/2.)
        
def TeMax(Tchi, mchi):
        return (Tchi**2 + 2.*mchi*Tchi)/(Tchi + (mchi + me)**2/(2.*me))
    
def TchiMin(Te,mchi):
    return (Te/2.-mchi) * (1.-np.sqrt(1.+2.*Te/me*pow((mchi+me)/(2.*mchi-Te), 2)))

def Det_res(a,b,Te):
    return a*np.sqrt(Te) + b*Te
'''
# Heavy mediator
def dsigma_dTe(DM_e, mchi, Tchi, Te): # Differential xsection
    return MeVTocm**(-2)*(DM_e[0]**2*DM_e[1]**2*me*mchi**2) / (2*np.pi*DM_e[2]**4*(Tchi+2.*mchi)*Tchi)

def sigma_DM_e(DM_e): # Total xsection
    return MeVTocm**(-2)*(DM_e[0]**2*DM_e[1]**2*me**2) / (np.pi*DM_e[2]**4)
'''
# Light mediator
def dsigma_dTe(DM_e, mchi, Tchi, Te): # Differential xsection
    return MeVTocm**(-2)*(DM_e[0]**2*DM_e[1]**2*me*mchi**2) / (2*np.pi*(Tchi+2.*mchi)*Tchi*(2*me*Te+DM_e[2]**2)**2)
    
def sigma_DM_e(DM_e,Tchi): # Total xsection
    return MeVTocm**(-2)*(DM_e[0]**2*DM_e[1]**2*me**2*mchi**2) / \
(2.*np.pi*DM_e[2]**2 * ((DM_e[2]*mchi)**2 + 4.*mchi*Tchi*me**2 + 2.*(me*Tchi)**2))

def interpolate_nu_flux(file):
    flux = np.loadtxt(file, delimiter=',', unpack=True)
    nu_flux_interp = interp1d(flux[0], flux[1]+flux[2]+flux[3]+flux[4]+flux[5]+flux[6], kind='linear', fill_value='extrapolate')
    return nu_flux_interp    

def e_recoil_dist2(Er, dsnb, sigma, mchi, detector, eff): # All args in MeV  # Returns keV^-1 ton^-1 yr^-1
    Zdet, mDet, a, b = detector[0], detector[1], detector[2], detector[3]
    Telist = np.logspace(-4.,-1.,num=180)
    Tchilist = np.logspace(-4.,3.95,num=40)
    DM_flux = []
    y = []
    for Tchi in Tchilist:
        Enlist = np.logspace(np.log10(Tchi/2.*(1.+np.sqrt(1+2.*mchi/Tchi))), 4., num=30)
        x = np.array([Dfactor*dsnb(en)*sigma/mchi *(en+mchi/2.)/en**2 for en in Enlist])     # Enu integral
        y.append(simps(x, Enlist))
    y_interp = interp1d(Tchilist,y, kind='linear')
    z = []
    for Te in Telist:
        Tchilist2 = np.logspace(np.log10(TchiMin(Te,mchi)), 3., num=40)
        #integral = [y_interp(Tchi)*dsigma_dTe(DM_e,mchi,Tchi,Te) for Tchi in Tchilist2]
        integral = [y_interp(Tchi)*sigma/TeMax(Tchi,mchi) for Tchi in Tchilist2]           # Tchi integral
        z.append(simps(integral, Tchilist2) * Detector_response(Te,Er,Det_res(a,b,Te)))
    return simps(np.array(z), Telist)*eff *Zdet/mDet/MeVTokeV/sToyr/kgToton                # Te integral

def overburden_threshold(mchi, depth, Tchi): # args in MeV, km, MeV; returns in cm^2
    ne = 4./(2.*1.78e-24) # cm^-3, electron number density in earth
    Er_thr = 2.e-3    # 2 keV for XENON
    Tchi_min = TchiMin(Er_thr,mchi)
    #Er_max = Tchi*(Tchi+2.*mchi) / (Tchi+(me+mchi)**2/(2.*mchi))
    return mchi / (2.*ne*depth*1.e5*me) * np.log(Tchi/Tchi_min)
    
def make_BDM_flux_plot(mchi, sigma):
    flux = Boosted_DM_flux(mchi, sigma)
    fig, ax = plt.subplots(1,1, figsize=(8,5))
    ax.loglog(flux[::,0]*1.e3,flux[::,1]*flux[::,0], color='b', linewidth=2.5)
    ax.set_xlabel(r'$T_\chi\,[\mathrm{keV}]$', fontsize=17)
    ax.set_ylabel(r'$T_\chi d\phi_\chi/dT_\chi\,[\mathrm{keV^{-1}cm^{-2}s^{-1}}]$', fontsize=17)
    ax.set_title(r'$m_\chi=$'+str(mchi)+r'$\,\mathrm{MeV},\ \sigma_{\chi\nu}=$'+str(sigma)+r'$\,\mathrm{cm^2}$', fontsize=16)
    ax.tick_params('both', length=6, width=1, which='major', labelsize=16, direction='in', top=True, right=True)
    ax.tick_params('both', length=4, width=1, which='minor', direction='in', top=True, right=True)
    plt.tight_layout()
    plt.show()
    #plt.savefig(filename+'.pdf', bbox_inches='tight')
    
def make_BDM_velocity_plot(mchi, sigma, title):
    flux = Boosted_DM_flux(mchi, sigma)
    v = np.sqrt(1. - pow(1.+flux[::,0]/mchi, -2))
    f = mchi**2/rho_NFW(dsun) * flux[::,1]
    fig, ax = plt.subplots(2,1, figsize=(8,6), constrained_layout=True)
    ax[0].loglog(flux[::,0]*1.e3, rho_NFW(dsun)/mchi**2*MB(v,220./clight)*flux[::,0], '--', color='gray', linewidth=2)
    ax[0].loglog(flux[::,0]*1.e3, flux[::,1]*flux[::,0], color='darkgreen', linewidth=2.5)
    ax[0].text(4.e-3, 3.e-6, r'SHM$\times 10^{-3}$', va='center', ha='center', color='gray', fontsize=12, rotation=21)
    ax[0].set_xlabel(r'$T_\chi\,[\mathrm{keV}]$', fontsize=14)
    ax[0].set_xlim(1.e-3, 1.e5)
    ax[0].set_ylim(1.e-10, 2.e-4)
    ax[0].set_ylabel(r'$T_\chi d\phi_\chi/dT_\chi\,[\mathrm{keV^{-1}cm^{-2}s^{-1}}]$', fontsize=14)
    ax[0].tick_params('both', length=6, width=1, which='major', labelsize=16, direction='in', top=True, right=True)
    ax[0].tick_params('both', length=4, width=1, which='minor', direction='in', top=True, right=True)
    
    ax[1].loglog(v, MB(v,220./clight)*v**2*1.e-3, '--', color='gray', linewidth=2)
    ax[1].loglog(v, f*v**2, color='orange', linewidth=2.5)
    ax[1].text(2.5e-4, 1.e-7, r'SHM$\times 10^{-3}$', va='center', ha='center', color='gray', fontsize=12, rotation=17.8)
    ax[1].set_xlabel(r'$v$', fontsize=14)
    ax[1].set_ylabel(r'$v^2f(v)$', fontsize=14)
    ax[1].set_xlim(1.e-4, 1.)
    ax[1].set_ylim(1.e-10, 1.e-4)
    ax[0].set_title(title, fontsize=16, pad=10)
    ax[1].tick_params('both', length=6, width=1, which='major', labelsize=16, direction='in', top=True, right=True)
    ax[1].tick_params('both', length=4, width=1, which='minor', direction='in', top=True, right=True)
    plt.tight_layout()
    plt.show()
    #plt.savefig(filename+'.pdf', bbox_inches='tight')
    
def make_recoil_plot(mchi, sigma, erlist, dsnb, detector, data, bkg, bkg_interp, eff, title):
    dist = np.array([e_recoil_dist2(er*1.e-3, dsnb, sigma, mchi, detector, eff(er)) for er in erlist])
    fig, ax = plt.subplots(1,1, figsize=(8,5))
    ax.plot(erlist, dist, '--', color='k', label=r'SnBDM')
    ax.plot(bkg[0], bkg[1], linestyle='dotted', color='k', label=r'$\mathrm{B}_0$')
    ax.plot(erlist, bkg_interp(erlist)+dist, color='r', linewidth=2.5, label=r'SnBDM+$\mathrm{B}_0$')
    ax.errorbar(data[0], data[1], yerr=data[2], ls='none', fmt='gray', marker='o', ecolor='gray', elinewidth=2., label=r'XENON1T SR1')
    ax.set_ylim(-5,120)
    ax.set_xlabel(r'Energy [keV]', fontsize=17)
    ax.set_ylabel(r'Events/(t-y-keV)', fontsize=17)
    plt.legend(loc='upper right', fontsize=12)
    ax.tick_params('both', length=6, width=1, which='major', labelsize=16, direction='in', top=True, right=True)
    ax.tick_params('both', length=4, width=1, which='minor', direction='in', top=True, right=True)
    plt.title(title, fontsize=16, pad=10)
    plt.tight_layout()
    plt.show()
    #plt.savefig(filename+'.pdf', bbox_inches='tight')
    
def get_chisq(data, dsnb, sigma, mchi, detector, bkg, eff):
    #chisq = 0.
    chisq2 = 0.
    for i, er in enumerate(data[0]):
        model = e_recoil_dist2(er*1.e-3, dsnb, sigma, mchi, detector, eff(er)) + bkg(er)
        #chisq += (model - data[1,i])**2 / model
        chisq2 += (model - data[1,i])**2 / (model+data[2,i]**2)
    return chisq2

def get_exclusion_chisq(data, dsnb, sigma, mchi, detector, bkg, eff): # Takes points where model itself predicts over data
    chisq = 0.
    for i, er in enumerate(data[0]):
        model = e_recoil_dist2(er*1.e-3, dsnb, sigma, mchi, detector, eff(er))
        if model > data[1,i]:
            chisq += (model - data[1,i])**2 / (model+data[2,i]**2)
    return chisq

def bkg_only_chisq(data, detector, bkg):
    chisq2 = 0.
    for i, er in enumerate(data[0]):
        model = bkg(er)
        chisq2 += (model - data[1,i])**2 / (model+data[2,i]**2)
    return chisq2

def sample(no_of_samples, mchi_low, mchi_high, sigma_low, sigma_high, dsnb, data, bkg, efficiency):
    chisq_list = []
    for i in range(no_of_samples):
        mchi = np.random.uniform(low=mchi_low, high=mchi_high)
        sigma = 10.**np.random.uniform(low=np.log10(sigma_low), high=np.log10(sigma_high))
        chisq_list.append([mchi, sigma, get_chisq(data, dsnb, sigma, mchi, xenon, bkg, efficiency_interp)])
    date = time.strftime('%Y-%m-%d_%H-%M-%S')
    np.savetxt('samples_'+date+'.txt', np.array(chisq_list), delimiter=',', header='No. of steps = '+str(no_of_samples)+'\n mchi [MeV]   sigma_chi_e [cm^2]   chi_sq/dof')

def scan_for_chisq(mchi_low, mchi_high, mchi_points, sigma_low, sigma_high, sigma_points, dsnb, data, detector, bkg, efficiency):
    chisq_list = []
    mchilist = np.linspace(mchi_low, mchi_high, num=mchi_points)
    sigmalist = np.logspace(np.log10(sigma_low), np.log10(sigma_high), num=sigma_points)
    bkg_chisq = bkg_only_chisq(data, detector, bkg)
    date = time.strftime('%Y-%m-%d_%H-%M-%S')
    with open('chisq_data_'+date+'.txt', 'a') as f:
        f.write('# Points in m_chi = '+str(mchi_points)+', Points in sigma_chi_e = '+str(sigma_points)+'\n# m_chi [MeV]   sigma_chi_e [cm^2]   chi_sq \n')
        for mchi in mchilist:
            for sigma in sigmalist:
                f.write('%s,%s,%s\n' %(mchi, sigma, get_chisq(data, dsnb, sigma, mchi, detector, bkg, efficiency) - bkg_chisq))
    
    #np.savetxt('chisq_data_'+date+'.txt', np.array(chisq_list), delimiter=',', header=''m_chi = '+str(mchi_low)+'-'+str(mchi_high)+' MeV, sigma_chi_e = '+str(sigma_low)+'-'+str(sigma_high)+' cm^2\n Points in m_chi = '+str(mchi_points)+', Points in sigma_chi_e = '+str(sigma_points)+'\n m_chi [MeV]   sigma_chi_e [cm^2]   chi_sq')
    
def scan_for_all_chisq(mchi_low, mchi_high, mchi_points, sigma_low, sigma_high, sigma_points, dsnb, data, detector, bkg, efficiency):
    chisq_list = []
    mchilist = np.linspace(mchi_low, mchi_high, num=mchi_points)
    sigmalist = np.logspace(np.log10(sigma_low), np.log10(sigma_high), num=sigma_points)
    for mchi in mchilist:
        for sigma in sigmalist:
            chisq_list.append([mchi, sigma, get_chisq(data, dsnb, sigma, mchi, detector, bkg, efficiency), get_exclusion_chisq(data, dsnb, sigma, mchi, detector, bkg, efficiency)])
    date = time.strftime('%Y-%m-%d_%H-%M-%S')    
    np.savetxt('all_chisq_data_'+date+'.txt', np.array(chisq_list), delimiter=',', header=' m_chi = '+str(mchi_low)+'-'+str(mchi_high)+' MeV, sigma_chi_e = '+str(sigma_low)+'-'+str(sigma_high)+' cm^2\n Points in m_chi = '+str(mchi_points)+', Points in sigma_chi_e = '+str(sigma_points)+'\n m_chi [MeV]   sigma_chi_e [cm^2]   Bestfit chi_sq   Exclusion chi_sq')
    
def extract_bestfit(datafile, sigma_points):
    data = np.loadtxt(datafile, delimiter=',', unpack=True)
    mchilist = data[0][::sigma_points]
    sigmalist = data[1][:sigma_points]
    mchi_points = len(mchilist)
    new_data = np.reshape(data[2], (mchi_points, sigma_points)).T
    min_chisq = min(data[2])
    idx = np.argmin(data[2])
    bkg_chisq = bkg_only_chisq(binned_data, xenon, B0_interp)
    print(r'Best-fit mchi = ', data[0,idx], r' MeV')
    print(r'Best-fit sigma = ', data[1,idx], r' cm^2')
    print(r'Best-fit chisq = ', min_chisq)
    print(r'Background-only chisq = ', bkg_chisq)
    print(r'Delta chisq = ', min_chisq-bkg_chisq)
    
def make_combined_plot(datafile, outfile, mchi_low, mchi_high, sigma_low, sigma_high, sigma_points):
    # Bestfit countour
    data = np.loadtxt(datafile, delimiter=',', unpack=True)
    mchilist = data[0][::sigma_points]
    sigmalist = data[1][:sigma_points]
    mchi_points = len(mchilist)
    new_data = np.reshape(data[2], (mchi_points, sigma_points)).T
    min_chisq = min(data[2])
    idx = np.argmin(data[2])
    fig, ax = plt.subplots(1,1, figsize=(8,5))
    ax.semilogy(mchilist, overburden_threshold(mchilist, 1000., 1.), linestyle='--', color='gray', alpha=0.5)
    ax.semilogy(mchilist, overburden_threshold(mchilist, 100., 1.), linestyle='--', color='gray', alpha=0.5)
    ax.text(700., overburden_threshold(700., 1000., 1.)*1.14, r'$1000\,\mathrm{km}$', fontsize=12, rotation=-10, color='gray', va='center')
    ax.text(700., overburden_threshold(700., 100., 1.)*1.14, r'$100\,\mathrm{km}$', fontsize=12, rotation=-10, color='gray', va='center')
    levels = [2.3,4.61]     #[2.3,4.61,9.21]
    c = ax.contour(mchilist, sigmalist, (new_data-min_chisq), levels=levels, linestyles='-', colors='k') #36.74 for 90%, 29.87 for 68%
    ax.set_yscale('log')
    fmt = {}
    strs = [r'$1\sigma$', r'$2\sigma$']
    for l, s in zip(c.levels, strs):
        fmt[l] = s
    ax.clabel(c, c.levels, inline=True, fmt=fmt, fontsize=13)
    cs = ax.contourf(mchilist, sigmalist, (new_data-min_chisq), levels=levels, cmap=plt.cm.Blues_r, alpha=0.65, extend='min')
    ax.text(data[0,idx], data[1,idx], r'$\mathbf{\ast}$', fontsize=17, color='white', ha='center', va='center')
    # Exclusion contour
    #data = np.loadtxt(datafile2, delimiter=',', unpack=True)
    #mchilist = data[0][::sigma_points]
    #sigmalist = data[1][:sigma_points]
    #mchi_points = len(mchilist)
    new_data = np.reshape(data[3], (mchi_points,sigma_points)).T
    c1 = ax.contour(mchilist, sigmalist, (new_data), levels=[4.], colors='r') #36.74 for 90%, 29.87 for 68%
    cs = ax.contourf(mchilist, sigmalist, (new_data), cmap='Reds', alpha=0.5, levels=[4.,1.e8])
    ax.set_xlim(0.,1000.)
    ax.set_ylim(1.e-31, 1.e-28)
    ax.set_xlabel('$m_{\chi}~$[MeV]', fontsize=17)
    ax.set_ylabel('$\sigma_{\chi e}\,[\mathrm{cm^2}]$', fontsize=17)
    plt.text(600.,1.27e-29, r'95\% C.L.', ha='center', va='center', rotation=7, fontsize=12)
    ax.tick_params('both',length=6,width=1,which='major',labelsize=16,direction='in', top=True, right=True)
    ax.tick_params('both',length=4,width=1,which='minor',direction='in', top=True, right=True)
    plt.savefig(outfile+'.pdf', bbox_inches='tight')

def make_DSNB_plot(datafiles, labels, colors):
    fig, ax = plt.subplots(1,1, figsize=(8,5))
    for i,file in enumerate(datafiles):
        data = np.loadtxt('DSNB_Data/'+file[0], unpack=True)
        ax.plot(data[0], data[1], color=colors[i], lw=2.5, label=labels[i])
        data1 = np.loadtxt('DSNB_Data/'+file[1], unpack=True)
        data2 = np.loadtxt('DSNB_Data/'+file[2], unpack=True)
        ax.fill_between(data1[0], data1[1], data2[1], color=colors[i], alpha=0.25)
    ax.set_xlim(0.,50.)
    ax.set_ylim(0., 1.7)
    ax.set_xlabel(r'$E_{\nu}~$[MeV]', fontsize=17)
    ax.set_ylabel(r'$\Phi_\nu\,[\mathrm{MeV^{-1}cm^{-2}s^{-1}}]$', fontsize=17)
    ax.tick_params('both',length=6,width=1,which='major',labelsize=16,direction='in', top=True, right=True)
    ax.tick_params('both',length=4,width=1,which='minor',direction='in', top=True, right=True)
    plt.legend(loc='upper right', fontsize=15)
    plt.tight_layout()
    plt.show()
    #plt.savefig(outfile+'.pdf', bbox_inches='tight')
'''
#make_DSNB_flux()
### Read all data
dsnb_flux = interpolate_nu_flux('DSNB_flux.csv')
Dfactor = 2.0213990756088443e+25
# XENON detector properties: Z=40, m_Xe=2.1801714*10^-25 kg, energy resolution = a*E^1/2 + b*E keV (from arXiv:2006.09721)
#xenon = [40., 2.1801714e-25, 0.5e-3]
xenon = [40., 2.1801714e-25, 0.31*10.**-1.5, 0.0037]
B0 = np.loadtxt('zenodo/B0_1to30keV.txt', delimiter=',', unpack=True, encoding='utf8')
B0_interp = interp1d(B0[0],B0[1], kind='linear')
binned_data = np.loadtxt('zenodo/data_binned_1to30kev.txt', delimiter=',', unpack=True, encoding='utf8')
erlist = np.logspace(np.log10(B0[0][0]),np.log10(B0[0][-2]),num=20)
efficiency = np.loadtxt('zenodo/efficiency.txt', delimiter=',', unpack=True, encoding='utf8')
efficiency_interp = interp1d(efficiency[0], efficiency[3], kind='linear')
'''








