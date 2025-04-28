#================================AxionFuncs.py=================================#
# Written by C. O'Hare 2020
# Contains:
# 1. Solar B-field models
# 2. Axion fluxes for photon & electron coupling
# 3. Photon spectra in a helioscope
# 4. Binned photon number tables which can be put into routines in Like.py
#==============================================================================#


from numpy import pi, sqrt, exp, zeros, size, shape,argmin,cos,gradient,ones
from numpy import sinc, linspace, trapz, loadtxt, interp, vstack,column_stack,savetxt
from scipy.integrate import cumtrapz, quad
from Params import *
from numba import jit


#==============================================================================#
# Load Saclay data
r,M,T,rho,ne,wp,k_S = loadtxt(data_dir+'solar/saclaymodel/saclaymodel.txt',\
                                unpack=True)
# B-field models
#==============================================================================#






#==============================================================================#
# B-field models

# All B-fields are output in units of Tesla

# Seismic model from from https://arxiv.org/pdf/astro-ph/0203107.pdf
def B_model_seismic(B_rad=5e3,B_tach=50.0,B_outer=3.0,r0_12=0.712,r0_31=0.96,d2=0.02,d3=0.035):
    B12 = zeros(shape=size(r))
    lamb = 10*r0_12+1
    region1 = r<r0_12
    region2 = abs(r-r0_12)<d2
    region3 = abs(r-r0_31)<d3
    B12[region1] = B_rad*(1+lamb)*(1+1/lamb)**lamb*(r[region1]/r0_12)**2*(1-(r[region1]/r0_12)**2.0)**lamb
    B12[region2] = B_tach*(1-((r[region2]-r0_12)/d2)**2.0)
    B12[region3] = B_outer*(1-((r[region3]-r0_31)/d3)**2.0)
    return B12

# Individual components of seismic model, defined for convenience
def B_model_seismic_rad(B_rad=5e3,r0_12=0.712):
    B12 = zeros(shape=size(r))
    lamb = 10*r0_12+1
    region1 = r<r0_12
    B12[region1] = B_rad*(1+lamb)*(1+1/lamb)**lamb*\
                    (r[region1]/r0_12)**2*(1-(r[region1]/r0_12)**2.0)**lamb
    return B12
def B_model_seismic_tach(B_tach=50.0,r0_12=0.712,d2=0.02):
    B12 = zeros(shape=size(r))
    region2 = abs(r-r0_12)<d2
    B12[region2] = B_tach*(1-((r[region2]-r0_12)/d2)**2.0)
    return B12
def B_model_seismic_outer(B_outer=3.0,r0_31=0.96,d3=0.035):
    B12 = zeros(shape=size(r))
    region3 = abs(r-r0_31)<d3
    B12[region3] = B_outer*(1-((r[region3]-r0_31)/d3)**2.0)
    return B12


# Unused: models from https://arxiv.org/abs/2005.00078
def B_model_constant(B0=1e5):
    # Constant B-field model
    return B0*1e-4*(r<10.0)

def B_model_step(B_inner=7e6,B_outer=1e3,r_step=0.75):
    # Step function B-field model
    return 1e-4*((B_inner*(r<r_step)+B_outer*(r>=r_step)))
#==============================================================================#




#==============================================================================#
# Axion fluxes:
# Always given in keV^-1 cm^-2 s^-1
# gag  always in GeV^-1
# m_a always in eV
# energy always in keV

# Primakoff (empirical formula, not currently used)
@jit(nopython=True)
def AxionFlux_Primakoff(gag,E):
    # Parameterised differential Axion Flux in [cm^-2 s^-1 keV^-1]
    # gag = Axion-photon coupling in GeV^-1
    # E = Axion/X-ray energy in keV
    norm = 6.02e10*(gag/1e-10)**2.0
    return norm*((E**2.481)/exp(E/1.205))

# Primakoff with Plasmon correction:
def AxionFlux_Primakoff_PlasmonCorrection(gag,omega_vals,Recalculate=False,nfine=500):
    # Set Recalculate=True to recalculate the spectrum (slow)
    # Set Recalculate=False to use the saved file (fast)
    if Recalculate:
        costh = linspace(-1,0.999,nfine)
        gag_keV = gag/1e6 # gag in keV
        nrvals = shape(r)[0]
        r_keV = r*Rsol_keV # radius in keV
        nvals = shape(omega_vals)[0]
        dPhi_P = zeros(shape=nvals)
        for i_w in range(0,nvals):
            ka = omega_vals[i_w]
            w = omega_vals[i_w]
            Gam = zeros(shape=nrvals)
            for i_r in range(0,nrvals):
                if w>wp[i_r]: # Only for freqs above plasmon freq
                    kphoton = sqrt(w**2.0-wp[i_r]**2.0)
                    x = (ka**2.0+kphoton**2.0)/(2*ka*kphoton)
                    y = x + k_S[i_r]**2.0/(2*ka*kphoton)
                    Gam[i_r] = (gag_keV**2.0*k_S[i_r]**2.0*T[i_r]/(64*pi))*trapz((1-costh**2.0)/((x-costh)*(y-costh)),costh)
            Gam *= keV_2_s
            dPhi_P[i_w] = (1.0/(AU_cm**2.0))*(w/pi)**2.0*trapz(r_keV**2.0*Gam/(exp(w/T)-1),r_keV)
        DAT = column_stack((omega_vals,dPhi_P))
        savetxt('../data/solar/PrimakoffFlux_PlasmonCorrected.txt',DAT)
    else:
        # Load in saved flux table and interpolate to desired omega_vals
        omega_i,dPhi_P_i = loadtxt('../data/solar/PrimakoffFlux_PlasmonCorrected.txt',unpack=True)
        dPhi_P = interp(omega_vals,omega_i,dPhi_P_i)
        dPhi_P *= (gag/1e-10)**2.0
    return dPhi_P

# LPlasmon flux for user defined B_model:
def AxionFlux_Lplasmon(gag,omega_vals,B_model):
    gag_keV = gag/1e6
    B_model = B_model*Tesla_2_keV  # B must be in Tesla
    r_keV = r*Rsol_keV
    nvals = shape(omega_vals)[0]
    nrvals = shape(r)[0]
    dwp = gradient(wp,r_keV)
    dPhi_pl = zeros(shape=nvals)
    for i in range(0,nvals):
        w = omega_vals[i]
        i_r = argmin(abs(w-wp))
        r0 = r_keV[i_r] # value of r which satisfies w = w_p(r)
        dPhi_pl[i] = keV_2_s*1/(12*pi*AU_cm**2.0)*(r0**2.0)*(w*w)*gag_keV**2.0\
                    *(B_model[i_r]**2.0)/(exp(w/T[i_r])-1)*(1/abs(dwp[i_r-1]))

    # Angular correction (outlined in paper)
    dPhi_pl *= 1.8
    return dPhi_pl


# Easy access functions for specific B_models:
def AxionFlux_Lplasmon_constantB(gag,omega_vals):
    return AxionFlux_Lplasmon(gag,omega_vals,B_model_constant())

def AxionFlux_Lplasmon_stepB(gag,omega_vals):
    return AxionFlux_Lplasmon(gag,omega_vals,B_model_step())

def AxionFlux_Lplasmon_seismicB(gag,omega_vals):
    return AxionFlux_Lplasmon(gag,omega_vals,B_model_seismic())


#==============================================================================#
# Energy bining and resolution functions needed to calculate dN/dE

# Energy resolution:
@jit(nopython=True)
def smear(E,dR,sig_E):
    # E = energies
    # dR = value of rate at energies in E
    # sig_E = Gaussian width to smear dR by
    nE = shape(dR)[0]
    dR_smeared = 1.0*dR
    imax = argmin((E-100*sig_E)**2.0)
    for i in range(0,imax):
        fres = 1.0/(sqrt(2*pi)*sig_E)*exp(-(E[i]-E)**2.0/(2*sig_E**2.0))
        dR_smeared[i] = trapz(dR*fres,E)

    # Make sure it's normalised to what it was before the smearing
    dR_smeared[0:imax] = dR_smeared[0:imax]*\
                        trapz(dR[0:imax],E[0:imax])/\
                        trapz(dR_smeared[0:imax],E[0:imax])
    return dR_smeared

# Energy binning:
def EnergyBins(E_min,E_max,nfine,nE_bins):
    # Define energy array for doing the trapz integration below
    # E_min = energy threshold
    # E_max = max energy
    # nfine = number of energies within one bin to integrate over
    # nE_bins = number of energy bins between E_min and E_max
    E_bin_edges = linspace(E_min,E_max,nE_bins+1)
    E_bw = (E_max-E_min)/(nE_bins+1.0)
    E_bins = (E_bin_edges[1:]+E_bin_edges[:-1])/2

    Ei = zeros(shape=(nE_bins*nfine))
    for i in range(0,nE_bins):
        Ei[i*nfine:(i+1)*nfine] = linspace(E_bin_edges[i],\
                                        E_bin_edges[i+1]-E_bw/nfine,nfine)
    return Ei,E_bins
#==============================================================================#








#==============================================================================#
# Photon spectra in a Helioscopes
# Bfield in Tesla
# Length in metres
# Bore diameter in cm
# Exposure always in years
# Flux10 is always the flux when gag = 1e-10 GeV^-1
# pressure always in mbar
# T_operating always in Kelvin

# Photon number dN/dE for the vacuum mode
@jit(nopython=True)
def PhotonNumber_gag(E,Flux10,m_a,g=1e-10,\
            Bfield=2.5,Exposure=1.5,Length=20.0,\
            N_bores=8,BoreDiameter=60.0,eps_D=0.7,eps_T=0.8,Eres=0.0):

    S_cm = N_bores*pi*(BoreDiameter/2.0)**2.0 # cm^2
    L_eV = Length/1.97e-7 # eV^-1
    t_secs = Exposure*3600*24*365 # s
    B = Bfield*(1e-19*195)
    norm = t_secs*S_cm*eps_D*eps_T*(B*L_eV/2.0)**2.0
    normq = L_eV/(4*1000)
    dN = (g/1.0e-10)**4.0*norm*Flux10*(sinc(normq/pi*m_a**2.0/E))**2.0 # keV^-1

    # Smear by energy resolution if Eres is not zero
    if Eres>0.0:
        #dN[omega_vals<Eres] = 0.0
        dN = smear(E,dN,Eres)
        dN[E<Eres] = 0.0

    return dN
# Need to run this to initialise jit:
Ei,E_bins = EnergyBins(100e-3,20.0,10,500)
dN_test = PhotonNumber_gag(Ei,AxionFlux_Primakoff_PlasmonCorrection(1e-10,Ei),1e-6)


# Photon number dN/dE for the buffer gas mode
@jit(nopython=True)
def PhotonNumber_gag_BufferGas(E,Flux10,m_a,pressure,g=1e-10,\
            Bfield=2.5,Exposure=2.0,Length=20.0,\
            N_bores=8,BoreDiameter=60.0,eps_D=0.7,eps_T=0.8,T_operating=1.8,Eres=0.0):

    # [Pressure in mbar]
    m_gam = sqrt(0.02*pressure/T_operating) # photon mass in eV
    gam = 0.29*pressure/(E**3.1*T_operating) # absorption length in m^-1
    gam_eV = gam*1.97e-7 # absorption length in eV
    if m_gam<m_a:
        S_cm = N_bores*pi*(BoreDiameter/2.0)**2.0 # cm^2
        L_eV = Length/1.97e-7 # eV^-1
        t_secs = Exposure*3600*24*365 # s
        B = Bfield*(1e-19*195)
        norm = t_secs*S_cm*eps_D*eps_T
        normq = L_eV/(4*1000)

        q = (m_a**2.0-m_gam**2.0)/(2*E*1000) # eV

        mask = q<0
        q[mask] = 0.0

        # Axion conversion probability
        P_ax = 0.25*(B**2.0)*1.0/(q**2.0+gam_eV**2.0/4)*\
                    (1+exp(-gam*Length)-2*exp(-gam*Length/2.0)*cos(q*L_eV))
        P_ax[mask] = 0.0

        # Photon number spectrum
        dN = (g/1.0e-10)**4.0*norm*Flux10*P_ax # keV^-1

        # Do energy resolution:
        if Eres>0.0:
            #dN[omega_vals<Eres] = 0.0
            dN = smear(E,dN,Eres)
            dN[E<Eres] = 0.0
    else:
        dN = zeros(shape=shape(E)[0])
    return dN
# Initialise jit:
Ei,E_bins = EnergyBins(100e-3,20.0,10,50)
dN_test = PhotonNumber_gag_BufferGas(Ei,\
                AxionFlux_Primakoff_PlasmonCorrection(1e-10,Ei),1e-6,0.1)
#==============================================================================#









#==============================================================================#
# Photon number tables which we calculate for a range of masses to be put into
# our likelihood analysis. We compute dN/dE at a range of energies finely spaced
# values of energy Ei and then integrate within bins spaced by E_bins

# The table assumes gag=1e-10 which is then rescaled by the desired value of
# gag in the likelihood analysis

def BinnedPhotonNumberTable_Vacuum(m_vals,Ei,E_bins,Flux10,\
                           Bfield=2.5,Exposure=3.0,Length=20.0,\
                           N_bores=8,BoreDiameter=60.0,eps_D=0.7,eps_T=0.8,\
                           res_on=True):
    # OUTPUT:
    # R1_tab = Tabulated values of the binned Xray counts (columns) vs axion mass (rows)
    # R0 = massless data

    # INPUT:
    # m_vals = masses to tabulate over
    # Ei range of fine energies to evaluate dN/dE at
    # E_bins = locations of energy bins within which to integrate dN/dE
    # Flux10 = desired Flux precomuputed for gag=1e-10
    # res_on = True/False, whether to do energy resolution integral or not

    nm = size(m_vals)
    nE_bins = shape(E_bins)[0]
    nfine = int(shape(Ei)[0]/nE_bins)
    E_min = Ei[0]
    R1_tab = zeros(shape=(nE_bins,nm))

    # If resolution is desired set E_min = 0 so that PhotonNumber_gag knows
    # noT to do the smearing:
    if res_on==False:
        E_min = 0.0

    # Tabulate m != 0 rates for each desired mass
    for j in range(0,nm):
        # Energy spectrum
        dN = PhotonNumber_gag(Ei,Flux10,m_vals[j],g=1e-10,\
                     Bfield=Bfield,Exposure=Exposure,Length=Length,\
                     N_bores=N_bores,BoreDiameter=BoreDiameter,\
                     eps_D=eps_D,eps_T=eps_T,Eres=E_min)
        # Bin the energy spectrum
        for i in range(0,nE_bins):
            Ebin = Ei[i*nfine:(i+1)*nfine]
            dNbin = dN[i*nfine:(i+1)*nfine]
            R1_tab[i,j] = sum(0.5*(Ebin[1:]-Ebin[0:-1])*(dNbin[1:]+dNbin[0:-1]))


    # Get m_a = 0 rate which we need to compute the mass discovery limit
    R0 = zeros(shape=(nE_bins))
    dN = PhotonNumber_gag(Ei,Flux10,0.0,g=1e-10,\
                 Bfield=Bfield,Exposure=Exposure,Length=Length,\
                 N_bores=N_bores,BoreDiameter=BoreDiameter,eps_D=eps_D,eps_T=eps_T,Eres=E_min)
    for i in range(0,nE_bins):
        Ebin = Ei[i*nfine:(i+1)*nfine]
        dNbin = dN[i*nfine:(i+1)*nfine]
        R0[i] = sum(0.5*(Ebin[1:]-Ebin[0:-1])*(dNbin[1:]+dNbin[0:-1]))

    return R1_tab,R0

# Photon number table for the buffer gas mode and a range of pressures
def BinnedPhotonNumberTable_BufferGas(m_vals,Ei,E_bins,Flux10,pressure_vals,\
                           Bfield=2.5,Exposure=2.0,Length=20.0,\
                           N_bores=8,T_operating=1.8,BoreDiameter=60.0,eps_D=0.7,eps_T=0.8,res_on=True):
    # OUTPUT:
    # R1_tab = Tabulated values of the binned Xray counts (columns) vs axion mass (rows)

    # INPUT:
    # m_vals = masses to tabulate over
    # Ei range of fine energies to evaluate dN/dE at
    # E_bins = locations of energy bins within which to integrate dN/dE
    # Flux10 = desired Flux precomuputed for gag=1e-10
    # Prssure_vals = desired range of pressures in mbar
    # res_on = True/False, whether to do energy resolution integral or not

    nm = size(m_vals)
    nE_bins = shape(E_bins)[0]
    nE_tot = shape(Ei)[0]
    nfine = int(nE_tot/nE_bins)
    E_min = Ei[0]
    R1_tab = zeros(shape=(nE_bins,nm))

    if res_on==False:
        E_min = 0.0

    # Loop over pressure values and sum all the spectra dN
    np = shape(pressure_vals)[0]
    for j in range(0,nm):
        dN = zeros(shape=nE_tot)
        for ip in range(0,np):
            dN += PhotonNumber_gag_BufferGas(Ei,Flux10,m_vals[j],pressure_vals[ip],g=1e-10,\
                     Bfield=Bfield,Exposure=Exposure/np,Length=Length,T_operating=T_operating,\
                     N_bores=N_bores,BoreDiameter=BoreDiameter,eps_D=eps_D,eps_T=eps_T,Eres=E_min)
        for i in range(0,nE_bins):
            Ebin = Ei[i*nfine:(i+1)*nfine]
            dNbin = dN[i*nfine:(i+1)*nfine]
            R1_tab[i,j] = sum(0.5*(Ebin[1:]-Ebin[0:-1])*(dNbin[1:]+dNbin[0:-1]))
    return R1_tab


# Table for Combined vacuum+buffer gas strategy:
def BinnedPhotonNumberTable_AllModes(m_vals,Ei,E_bins,Flux10,\
                           Bfield=2.5,Exposure_total=6.0,Length=20.0,\
                           N_bores=8,BoreDiameter=60.0,eps_D=0.7,eps_T=0.8,res_on=False,
                                    np1=50,np2=500):
    # OUTPUT:
    # Signal_10 = Tabulated values of the binned Xray counts (columns) vs axion mass (rows)
    # R0 = Massless axion table

    # INPUT:
    # m_vals = masses to tabulate over
    # Ei range of fine energies to evaluate dN/dE at
    # E_bins = locations of energy bins within which to integrate dN/dE
    # Flux10 = desired Flux precomuputed for gag=1e-10
    # Prssure_vals = desired range of pressures in mbar
    # res_on = True/False, whether to do energy resolution integral or not

    nm = size(m_vals)
    nE_bins = shape(E_bins)[0]
    nfine = int(shape(Ei)[0]/nE_bins)

    Exposure_1 = Exposure_total/3
    Exposure_2 = Exposure_total/3
    Exposure_3 = Exposure_total/3

    # Vacuum part
    Signal_10_vacuum,R0 = BinnedPhotonNumberTable_Vacuum(m_vals,Ei,E_bins,\
                            Flux10,Exposure=Exposure_1,\
                            res_on=False,Bfield=Bfield,Length=Length,\
                            N_bores=N_bores,BoreDiameter=BoreDiameter,\
                            eps_D=eps_D,eps_T=eps_T)

    # Straight part:
    T_operating=1.8
    p_min = (2e-2)**2.0*T_operating/0.02
    p_max = (7e-2)**2.0*T_operating/0.02
    pressure_vals = linspace(p_min,p_max,np1)
    Signal_10_BG1 = BinnedPhotonNumberTable_BufferGas(m_vals,Ei,E_bins,Flux10,\
                        pressure_vals,Exposure=Exposure_2,res_on=res_on,\
                        Bfield=Bfield,Length=Length,T_operating=T_operating,\
                        N_bores=N_bores,BoreDiameter=BoreDiameter,\
                        eps_D=eps_D,eps_T=eps_T)

    # DFSZ part
    p_min = (7e-2)**2.0*T_operating/0.02
    p_max = (1.8e-1)**2.0*T_operating/0.02
    pressure_vals = linspace(p_min**-1,p_max**-1,np2)**(-1)
    Signal_10_BG2 = BinnedPhotonNumberTable_BufferGas(m_vals,Ei,E_bins,Flux10,\
                    pressure_vals,Exposure=Exposure_3,res_on=res_on,\
                    Bfield=Bfield,Length=Length,T_operating=T_operating,\
                    N_bores=N_bores,BoreDiameter=BoreDiameter,\
                    eps_D=eps_D,eps_T=eps_T)

    Signal_10 = Signal_10_vacuum+Signal_10_BG1+Signal_10_BG2
    return Signal_10,R0
#==============================================================================#





#==============================================================================#
# Axion electron flux (unused in current paper):
def AxionFlux_Axioelectron(gae,E):
    # Differential Axion Flux from the axion electron coupling
    # Flux = AxionRecomb+Compton+Bremsstrahlung
    # column 1 = Energy [keV]
    # column 2 = Axion Flux 1/[10^19 keV cm^2 day]
    # Output: flux in cm^-1 s^-1 keV^-1
    # gae = Axion-electron coupling in GeV^-1
    # E = Axion/Xray energy in keV
    data = loadtxt(data_dir+'solar/gaeflux.txt')
    E1 = data[:,0]
    F1 = data[:,1]
    norm = 1e19*(gae/(0.511e-10))**2.0/(3600*24)
    Flux = interp(E,E1,F1)*norm
    return Flux

def AxionFlux_Compton(gae,E):
    # Parameterised Compton axion flux
    norm = 13.314e6*(gae/1e-13)**2.0
    return norm*((E**2.987)/exp(E*0.776))

def AxionFlux_Brem(gae,E):
    # Parameterised Bremsstrahlung axion flux
    norm = 26.311e8*(gae/1e-13)**2.0
    return norm*E*exp(-0.77*E)/(1+0.667*E**1.278)
#==============================================================================#













# Old code:
#==============================================================================#
#
#
#
# def PhotonNumber_gag(E,Flux10,ma,g,\
#     Bfield=2.5,Exposure=1.5,Length=20.0,\
#     N_bores=8,BoreDiameter=60.0,eps_D=0.7,eps_T=0.8):
#     norm,normq = NgammaNorm(Bfield,Exposure,Length,N_bores,BoreDiameter,eps_D,eps_T)
#     norm = norm/(6.02e10)
#     return (g/1.0e-10)**4.0*norm*Flux10*(sinc(normq/pi*m_a**2.0/E))**2.0 # keV^-1
#
# #==============================================================================#
# def PhotonNumber_Primakoff(Flux_scale,E,m_a,\
#                            Bfield=2.5,Exposure=1.5,Length=20.0,\
#                            N_bores=8,BoreDiameter=60.0,eps_D=0.7,eps_T=0.8):
#     # differential Xray count dN/dE (in keV^-1) for axionphoton flux
#     # (Optional) Flux_scale = scaling for normalisation (set to 1 for units used in paper)
#     # E = Xray energy (keV)
#     # m_a = axion mass (eV)
#     norm,normq = NgammaNorm(Bfield,Exposure,Length,N_bores,BoreDiameter,eps_D,eps_T)
#     norm = Flux_scale*norm
#     return norm*((E**2.481)/exp(E/1.205))*(sinc(normq/pi*m_a**2.0/E))**2.0  # keV^-1
#
#
#
#
# def PhotonNumber_Electron(Flux,E,m_a,\
#                            Bfield=2.5,Exposure=1.5,Length=20.0,\
#                            N_bores=8,BoreDiameter=60.0,eps_D=0.7,eps_T=0.8):
#     # differential Xray count dN/dE (in keV^-1) for axionelectron flux
#     # Flux_scale = scaling for normalisation (set to 1 for units used in paper)
#     # E = Xray energy (keV)
#     # m_a = axion mass (eV)
#     norm,normq = NgammaNorm(Bfield,Exposure,Length,N_bores,BoreDiameter,eps_D,eps_T)
#     norm = norm/(6.02e10)
#     return norm*Flux*(sinc(normq/pi*m_a**2.0/E))**2.0 # keV^-1
#
# def NgammaNorm(Bfield,Exposure,Length,N_bores,BoreDiameter,eps_D,eps_T):
#     # Nnorm = normalisation of overall photon number to get it in keV^-1 and constant that enters into t
#     S_cm = N_bores*pi*(BoreDiameter/2.0)**2.0 # cm^2
#     L_eV = Length/1.97e-7 # eV^-1
#     t_secs = Exposure*3600*24*365 # s
#     B = Bfield*(1e-19*195)
#     norm = 6.02e10*t_secs*S_cm*eps_D*eps_T*(B*L_eV/2.0)**2.0
#     normq = L_eV/(4*1000)
#     return norm,normq
# #==============================================================================#
#
#
# #===========================Apply Energy Res===================================#
# def smear(E,dR,sig_E):
#     # E = energies
#     # dR = value of rate at energies in E
#     # sig_E = Gaussian width to smear dR by
#     nE = size(dR)
#     dR_smeared = zeros(shape=shape(dR))
#     if size(sig_E)==1:
#         sig_E *= ones(shape=shape(dR))
#     for i in range(0,nE):
#         Ediff = abs(E-E[i])
#         fres = 1.0/(sqrt(2*pi)*sig_E)*exp(-(E[i]-E)**2.0/(2*sig_E**2.0))
#         dR_smeared[i] = trapz(dR*fres,E)
#
#     # Make sure it's normalised to what it was before the smearing
#     dR_smeared = dR_smeared*trapz(dR,E)/trapz(dR_smeared,E)
#     return dR_smeared
# #------------------------------------------------------------------------------#
#
#
# def smearFast(dN,E,E_res):
#     # Does the same as 'smear' but is faster and less accurate for E_res>100 eV
#     n = size(dN)
#     dE = E[1]-E[0]
#     irange = int(3*E_res/dE)
#     Norm = 1.0/sqrt(2*pi*E_res**2.0)
#     dN_smeared = zeros(shape=n)
#     for i in range(0,n):
#         i1 = max(0,i-irange)
#         i2 = min(n-1,i+irange)
#         Eint = E[i1:i2]
#         K = Norm*exp(-(Eint-E[i])**2.0/(2*E_res**2.0))
#         dN_smeared[i] = trapz(K*dN[i1:i2],Eint)
#     return dN_smeared
# #==============================================================================#
#
#
#
#
# #==============================================================================#
# def EnergyBins(E_min,E_max,nfine,nE_bins):
#     # Define energy array for doing the trapz integration below
#     # E_min = energy threshold
#     # E_max = max energy
#     # nfine = number of energies within one bin to integrate over
#     # nE_bins = number of energy bins between E_min and E_max
#     E_bin_edges = linspace(E_min,E_max,nE_bins+1)
#     E_bw = (E_max-E_min)/(nE_bins+1.0)
#     E_bins = (E_bin_edges[1:]+E_bin_edges[:-1])/2
#
#     Ei = zeros(shape=(nE_bins*nfine))
#     for i in range(0,nE_bins):
#         Ei[i*nfine:(i+1)*nfine] = linspace(E_bin_edges[i],E_bin_edges[i+1]-E_bw/nfine,nfine)
#
#     return Ei,E_bins
#
# def BinnedPhotonNumberTable(m_vals,E_min,E_max,nE_bins,coupling='Photon',\
#                             nfine=100,res_on=False,\
#                            Bfield=2.5,Exposure=1.5,Length=20.0,\
#                            N_bores=8,BoreDiameter=60.0,eps_D=0.7,eps_T=0.8):
#     # Generate tabulated values of data for a range of axion masses
#     # OUTPUT: R1_tab = Tabulated values of the binned Xray counts (columns) vs axion mass (rows)
#     # R0 = massless data
#     # E_bins = centers of energy bins
#     # INPUT: m_vals = masses to add to the tabulation
#     # E_min = threshold energy (also resolution if res_on=True)
#     # E_max = maximum energy
#     # nE_bins = number of energy bins
#     # coupling = 'Photon' or 'Electron' for g_ag or g_ae
#     # nfine = number of points to integrate over within one bin (controls accuracy)
#     # res_on = True/False, whether to do energy resolution integral or not
#     nm = size(m_vals)
#     R1_tab = zeros(shape=(nE_bins,nm))
#     Ei,E_bins = EnergyBins(E_min,E_max,nfine,nE_bins)
#
#     if coupling=='Electron':
#         Flux = AxionFlux_Axioelectron(1e-10,Ei)
#         dN_func = PhotonNumber_Electron
#     else:
#         Flux = 1.0
#         dN_func = PhotonNumber_Primakoff
#
#     # Tabulate m != 0 rates
#     for j in range(0,nm):
#         dN = dN_func(Flux,Ei,m_vals[j],\
#                      Bfield,Exposure,Length,\
#                      N_bores,BoreDiameter,eps_D,eps_T)
#         if res_on:
#             dN = smear(dN,Ei,E_min)
#         for i in range(0,nE_bins):
#             Ebin = Ei[i*nfine:(i+1)*nfine]
#             dNbin = dN[i*nfine:(i+1)*nfine]
#             R1_tab[i,j] = sum(0.5*(Ebin[1:]-Ebin[0:-1])*(dNbin[1:]+dNbin[0:-1]))
#
#     # Get m = 0 rate
#     R0 = zeros(shape=(nE_bins))
#     dN =  dN_func(Flux,Ei,0.0,\
#                 Bfield,Exposure,Length,\
#                  N_bores,BoreDiameter,eps_D,eps_T)
#     if res_on:
#         dN = smear(dN,Ei,E_min)
#     for i in range(0,nE_bins):
#         Ebin = Ei[i*nfine:(i+1)*nfine]
#         dNbin = dN[i*nfine:(i+1)*nfine]
#         R0[i] = sum(0.5*(Ebin[1:]-Ebin[0:-1])*(dNbin[1:]+dNbin[0:-1]))
#
#     return E_bins,R1_tab,R0
#
# def BinnedPhotonNumberTable_Massless(E_min,E_max,nE_bins,coupling='Photon',\
#                                 nfine=100,res_on=False,\
#                                Bfield=2.5,Exposure=1.5,Length=20.0,\
#                                N_bores=8,BoreDiameter=60.0,eps_D=0.7,eps_T=0.8):
#         Ei,E_bins = EnergyBins(E_min,E_max,nfine,nE_bins)
#
#         if coupling=='Electron':
#             Flux = AxionFlux_Axioelectron(1e-10,Ei)
#             dN_func = PhotonNumber_Electron
#         else:
#             Flux = 1.0
#             dN_func = PhotonNumber_Primakoff
#
#         # Get m = 0 rate
#         R0 = zeros(shape=(nE_bins))
#         dN =  dN_func(Flux,Ei,0.0,\
#                      Bfield,Exposure,Length,\
#                      N_bores,BoreDiameter,eps_D,eps_T)
#         if res_on:
#             dN = smear(dN,Ei,E_min)
#         for i in range(0,nE_bins):
#             Ebin = Ei[i*nfine:(i+1)*nfine]
#             dNbin = dN[i*nfine:(i+1)*nfine]
#             R0[i] = sum(0.5*(Ebin[1:]-Ebin[0:-1])*(dNbin[1:]+dNbin[0:-1]))
#
#         return R0
# #==============================================================================#
