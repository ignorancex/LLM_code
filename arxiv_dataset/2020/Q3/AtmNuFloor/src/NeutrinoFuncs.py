#===========================NeutrinoFuncs.py===================================#
# Created by Ciaran O'Hare 2020

# Contains functions for performing calculations of CEvNS

#==============================================================================#

from numpy import pi, sqrt, exp, zeros, size, shape, array, meshgrid, reshape
from numpy import trapz, interp, loadtxt, count_nonzero, flipud, arange, append
from numpy import loadtxt, digitize, log10, cos, sin, arccos, arctan2, count_nonzero
from numpy import logspace, linspace, ones, asarray, histogram2d,column_stack
from numpy import nan, isnan, amin, amax
from numpy.linalg import norm
import LabFuncs
import Params
from Params import nufile_root, nufile_dir, nuname, n_Enu_vals, recoil_dir
from Params import mono, NuMaxEnergy, NuFlux, NuUnc, whichsolar, n_nu_tot
from Params import m_p_keV, c_km, seconds2year, NuclearReactors, m_p_keV
from Params import G_F_GeV, sinTheta_Wsq, N_A, Jan1, EarthRadius, eV2J
from numpy import random
from scipy.spatial import ConvexHull


#==============================================================================#
def GetAtmNuFluxes(fname,GetIndivSpecies=False):
    #### Import AtmNu data
    cosZ = flipud(arange(-1,1,0.1)+0.05)
    phi_Az = arange(0,360,30)+15

    ne = 101
    nc = size(cosZ)
    np = size(phi_Az)
    [I,J] = meshgrid(arange(0,np),arange(0,nc))
    I = I.reshape(nc*np)
    J = J.reshape(nc*np)

    f=open(nufile_dir+'/atmospheric/Honda/'+fname,'r')

    E_high = array([])
    Phi_mu = zeros(shape=(np,nc,ne))
    Phi_mubar = zeros(shape=(np,nc,ne))
    Phi_e = zeros(shape=(np,nc,ne))
    Phi_ebar = zeros(shape=(np,nc,ne))


    # Read file
    ii = -1
    while ii<(ne*np+1):
        a=f.readline().split()
        if a == []:
            break
        if a[0]!='average':
            if (a[0]=='Enu(GeV)'):
                ie = 0
                ii = ii+1
            else:
                E_high = append(E_high,float(a[0]))
                Phi_mu[I[ii],J[ii],ie] = float(a[1])
                Phi_mubar[I[ii],J[ii],ie] = float(a[2])
                Phi_e[I[ii],J[ii],ie] = float(a[3])
                Phi_ebar[I[ii],J[ii],ie] = float(a[4])
                ie = ie+1

    # convert to units with MeV, cm^2
    E_high = E_high[0:ne]*1000
    Phi_mu = 0.05*Phi_mu/(100**2.0*1000)
    Phi_mubar = 0.05*Phi_mubar/(100**2.0*1000)
    Phi_e = 0.05*Phi_e/(100**2.0*1000)
    Phi_ebar = 0.05*Phi_ebar/(100**2.0*1000)
    if GetIndivSpecies:
        Phi_tot = {'mu': Phi_mu, 'mubar': Phi_mubar,\
                    'e':Phi_e, 'ebar':Phi_ebar,\
                    'Total':Phi_mu + Phi_mubar + Phi_e + Phi_ebar}
    else:
        Phi_tot = Phi_mu + Phi_mubar + Phi_e + Phi_ebar

    return Phi_tot,E_high,cosZ,phi_Az
#==============================================================================#



#==============================================================================#
# Total rate of indiv neutrino species
def R_Indiv(s,E_th,E_max,Nuc=Params.Ar40,eff_on=False,eres_on=False):
    nfine = 1000
    Efine = logspace(-3.0,log10(200.0),nfine)

    # Load nu flux
    data = loadtxt(nufile_dir+nuname[s]+nufile_root,delimiter=',')
    E_nu = data[:,0]
    Flux = NuFlux[s]*data[:,1]

    sol = False
    dR = dRdE_nu(Efine,Jan1*ones(shape=nfine),sol,E_nu,Flux,Nuc=Nuc)
    # Correct for efficiency
    if eff_on:
        dR *= LabFuncs.efficiency(Nuc,Efine)

    # Smear by energy resolution
    # if eres_on:
    #     dR = SmearE(Efine,dR,energyresolution(Nuc,Efine))

    # Window
    mask = (Efine<E_max)&(Efine>E_th)
    R = trapz(dR[mask],Efine[mask])
    return R

# Specific examples, for quick access
def R_hep(E_th,E_max,Nuc=Params.Ar40,eff_on=True,eres_on=False):
    return R_Indiv(2,E_th,E_max,Nuc=Nuc,eff_on=eff_on,eres_on=eres_on)

def R_8B(E_th,E_max,Nuc=Params.Ar40,eff_on=True,eres_on=False):
    return R_Indiv(5,E_th,E_max,Nuc=Nuc,eff_on=eff_on,eres_on=eres_on)

def R_AtmNu(E_th,E_max,Nuc=Params.Ar40,eff_on=True,eres_on=False):
    return R_Indiv(10,E_th,E_max,Nuc=Nuc,eff_on=eff_on,eres_on=eres_on)

def R_DSNB(E_th,E_max,Nuc=Params.Ar40,eff_on=True,eres_on=False):
    return R_Indiv(9,E_th,E_max,Nuc=Nuc,eff_on=eff_on,eres_on=eres_on)
#==============================================================================#


#===============================Reactor neutrinos data=========================#
def ReactorFlux(E_nu,Loc):
    # from https://arxiv.org/abs/1101.2663 TABLE VI.
    U235_c = array([3.217,-3.111,1.395,-0.3690,0.04445,-0.002053])
    U238_c = array([0.4833,0.1927,-0.1283,-0.006762,0.00233,-0.0001536])
    P239_c = array([6.413,-7.432,3.535,-0.8820,0.1025,-0.00455])
    P241_c = array([3.251,-3.204,1.428,-0.3675,0.04245,-0.001896])

    def Sk(coeff,E_nu):
        Sk = exp(coeff[0]
                 +coeff[1]*E_nu**1.0
                 +coeff[2]*E_nu**2.0
                 +coeff[3]*E_nu**3.0
                 +coeff[4]*E_nu**4.0
                 +coeff[5]*E_nu**5.0)
        return Sk

    # Get powers of nearby reactors and coordinates
    Reactors = asarray(list(NuclearReactors.values()))
    Powers = Reactors[:,0] # In MW
    Coords = Reactors[:,1:] # In degrees

    # Calculate distance
    phi1 = Loc.Latitude*pi/180.0
    phi2 = Coords[:,0]*pi/180.0
    dphi = phi1-phi2
    dlambda = (Loc.Longitude-Coords[:,1])*pi/180.0
    a = sin(dphi/2)**2.0 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2.0
    bearing = 2*arctan2(sqrt(a),sqrt(1-a))
    Distances = 2*EarthRadius*sin(bearing/2.0)

    # Fission fraction and average energies from https://arxiv.org/abs/1101.2663
    # k = 0,3 for U235,U238,U239,P241
    fk = array([0.58,0.07,0.3,0.05])
    Ek = array([202.36,205.99,211.12,214.26])
    phi = fk[0]*Sk(U235_c,E_nu)
    phi += fk[1]*Sk(U238_c,E_nu)
    phi += fk[2]*Sk(P239_c,E_nu)
    phi += fk[3]*Sk(P241_c,E_nu)

    # Calculate flux by multiplying spectrum
    # by Emission rate*power/(4*pi*distance^2)
    # Where Emission rate = (6.0*0.75/sum(fk*Ek)) accounting for operating efficiency
    Phi = 0.0
    for i in range(0,size(Distances)):
        Phi += phi*(Powers[i]/eV2J)*(6.0*0.75/sum(fk*Ek))\
                /(4*pi*(Distances[i]*100)**2.0)
    return Phi


#========================================Neutrino data=========================#
def GetNuFluxes(E_th,Nuc):
    # Reads each neutrino flux data file
    # the energies are stored in E_nu_all, fluxes in Flux_all

    # Figure out which backgrounds give recoils above E_th
    E_r_max = MaxNuRecoilEnergies(Nuc) # Max recoil energy for neutrino
    sel = range(1,n_nu_tot+1)*(E_r_max>E_th)
    sel = sel[sel!=0]-1
    n_nu = count_nonzero(E_r_max>E_th)
    E_nu_all = zeros(shape=(n_nu,n_Enu_vals))
    Flux_all = zeros(shape=(n_nu,n_Enu_vals))
    Flux_err = zeros(shape=(n_nu))
    Flux_norm = zeros(shape=(n_nu))
    Solar = zeros(n_nu,dtype=bool)
    Names = asarray([nuname[i] for i in sel])

    ii = 0
    for s in sel:
        if mono[s]:
            E_nu_all[ii,0] = NuMaxEnergy[s]
            Flux_all[ii,0] = NuFlux[s]
        else:
            data = loadtxt(nufile_dir+nuname[s]+nufile_root,delimiter=',')
            E_nu_all[ii,:],Flux_all[ii,:] = data[:,0],data[:,1]
            Flux_all[ii,:] = Flux_all[ii,:]*NuFlux[s]

        Flux_norm[ii] = NuFlux[s]
        Flux_err[ii] = NuUnc[s] # Select rate normalisation uncertainties
        Solar[ii] = whichsolar[s]
        ii = ii+1
    return Names,Solar,E_nu_all,Flux_all,Flux_norm,Flux_err

#==============================================================================#
def MaxNuRecoilEnergies(Nuc): # Max recoil energies
    m_N = 0.93141941*(Nuc.MassNumber)*1.0e6
    E_r_max = 2*m_N*(1000.0*NuMaxEnergy)**2.0/(m_N+1000*NuMaxEnergy)**2.0
    return E_r_max
#==============================================================================#





#===================================nu spectra=================================#
def AllNuRates(E_r,t,Solar,E_nu_all,Flux_all,Nuc=Params.Ar40): # Time-Energy
    n_nu = shape(Flux_all)[0]
    ne = size(E_r)
    dR = zeros(shape=(n_nu,ne))
    for i in range(0,n_nu):
        dR[i,:] = dRdE_nu(E_r,t,Solar[i],E_nu_all[i,:],Flux_all[i,:],Nuc=Nuc)
    return dR

#==============================================================================#
# CEvNS event rate vs recoil energy E_r
def dRdE_nu(E_r,t,sol,E_nu,Flux,Nuc=Params.Ar40):
    N = Nuc.NumberOfNeutrons
    Z = Nuc.NumberOfProtons
    Q_W = 1.0*N-(1-4.0*sinTheta_Wsq)*Z # weak nuclear hypercharge
    m_N_GeV = 0.93141941*(N+Z) # nucleus mass in GeV
    m_N_keV = m_N_GeV*1.0e6 # nucleus mass in keV

    dRdE = zeros(shape=shape(E_r))
    FF = LabFuncs.FormFactorHelm(E_r,N+Z)**2.0
    ne = size(E_r)

    if Flux[1]>0.0:
        for i in range(0,ne):
            diff_sigma = (G_F_GeV**2.0 /(4.0*pi))*(Q_W**2.0)*m_N_GeV*(1.0 \
                        -(m_N_keV*E_r[i])/(2.0*(E_nu*1000.0)**2.0))*\
                        (0.197e-13)**2.0*(1.0e-6)*1000.0/(1.0*N+1.0*Z)*(N_A)
            diff_sigma[diff_sigma<0.0] = 0.0
            dRdE[i] = trapz(diff_sigma*Flux*FF[i],x=E_nu)
    else:
        for i in range(0,ne):
            diff_sigma = (G_F_GeV**2.0 /(4.0*pi))*(Q_W**2.0)*m_N_GeV*(1.0 \
                        -(m_N_keV*E_r[i])/(2.0*(E_nu[0]*1000.0)**2.0))*\
                        (0.197e-13)**2.0*(1.0e-6)*1000.0/(1.0*N+1.0*Z)*(N_A)
            if diff_sigma>0:
                dRdE[i] = diff_sigma*Flux[0]*E_nu[0]*FF[i] # for monochromatic nu's

    if sol:
        fMod = LabFuncs.EarthSunDistanceMod(t)
    else:
        fMod = 1.0

    # Convert into /ton/year/keV
    dRdE = fMod*dRdE*1000*seconds2year
    return dRdE
#==============================================================================#

#==============================================================================#
# CEvNS event rate as a function of recoil direction and energy
def dRdEdO_solarnu(E,t,E_nu,Flux,Nuc,Loc=Params.GranSasso,CygnusTracking=False):
    N = Nuc.NumberOfNeutrons
    Z = Nuc.NumberOfProtons
    Q_W = N-(1-4.0*sinTheta_Wsq)*Z # weak nuclear hypercharge
    m_N_GeV = 0.93141941*(N+Z) # nucleus mass in GeV
    m_N_keV = m_N_GeV*1.0e6 # nucleus mass in keV
    E_nu_keV = E_nu*1e3


    E_r = sqrt(E[:,0]**2 + E[:,1]**2 + E[:,2]**2) # Recoil energy
    x = zeros(shape=shape(E))
    x_sun = zeros(shape=shape(E))
    x[:,0] = E[:,0]/E_r # Recoil direction
    x[:,1] = E[:,1]/E_r
    x[:,2] = E[:,2]/E_r
    ne =size(E_r)
    dRdEdO = zeros(shape=ne)
    x_sun = LabFuncs.SolarDirection(t,Loc)
    cos_th_sun = -(x_sun[0]*x[:,0]+x_sun[1]*x[:,1]+x_sun[2]*x[:,2])
    FF = LabFuncs.FormFactorHelm(E_r,N+Z)**2.0


    # CHROMATIC NEUTRINOS
    if Flux[1]>0.0:
        E_max = 2*m_N_keV*E_nu_keV[-1]**2.0/(m_N_keV+E_nu_keV[-1])**2
        i_range = range(0,ne)*(E_r<=E_max)
        i_sel = i_range[i_range!=0]
        for i in i_sel:
            costh = cos_th_sun[i]
            E_nu_min = sqrt(m_N_keV*E_r[i]/2.0)
            if costh>(E_nu_min/m_N_keV):
                Eps = 1.0/(costh/E_nu_min - 1.0/m_N_keV)
                diff_sigma = (G_F_GeV**2/(4*pi))*Q_W**2*m_N_GeV*\
                            (1-(m_N_keV*E_r[i])/(2*Eps**2))*(0.197e-13)**2.0\
                            *1e-6*1000/(N+1.0*Z)*(N_A)
                Eps = Eps*(Eps>E_nu_min)
                Eps = Eps*(Eps<E_nu_keV[-1])
                F_value = interp(Eps,E_nu_keV,Flux)
                dRdEdO[i] = diff_sigma*F_value*Eps**2.0/(1000*E_nu_min)*FF[i]

    # MONOCHROMATIC NEUTRINOS
    else:
        E_max = 2*m_N_keV*E_nu_keV[0]**2.0/(m_N_keV+E_nu_keV[0])**2
        i_range = range(0,ne)*(E_r<=E_max)
        i_sel = i_range[i_range!=0]
        for i in i_sel:
            costh = cos_th_sun[i]
            E_nu_min = sqrt(m_N_keV*E_r[i]/2.0)
            costh_r = ((E_nu_keV[0]+m_N_keV)/E_nu_keV[0])*sqrt(E_r[i]/(2*m_N_keV))

            # just need to accept angles close enough to costh_r to be accurate
            # around 0.01 is enough to be disguised by most angular resolutions
            if abs((costh)-(costh_r))<0.01:
                Eps = E_nu_keV[0]
                diff_sigma = (G_F_GeV**2/(4*pi))*Q_W**2*m_N_GeV*\
                            (1-(m_N_keV*E_r[i])/(2*Eps**2))*(0.197e-13)**2.0\
                            *1e-6*1000/(N+1.0*Z)*(N_A)*FF[i]
                dRdEdO[i] = diff_sigma*(Flux[0]/1000.0)*E_nu_keV[0] # /kg/keV


    fMod = LabFuncs.EarthSunDistanceMod(t) # add Annual modulation
    dRdEdO = fMod*dRdEdO*3600*24*365*1000/(2*pi) # /ton/year
    return dRdEdO
#==============================================================================#


#===================================Monte Carlo=================================#
def GenerateAtmNuDirections(ngen,E_fine,Phi_fine,E_high,Phi_Ang,cosZ,phi_Az,Nuc):
    ngen_large = 2*ngen
    ngen_red = 0

    # Nucleus mass
    A = Nuc.MassNumber
    m_N_keV = A*0.9315*1e6

    # Flux binning
    nc = size(cosZ)
    np = size(phi_Az)
    [C,P] = meshgrid(cosZ,phi_Az)
    C = reshape(C,nc*np)
    P = reshape(P,nc*np)

    # Neutrino energy distribution
    fdist = (E_fine**2.0)*Phi_fine
    fdist = fdist/sum(fdist)

    # Get energies:

    E_gen_full = array([])
    E_r_gen_full = array([])
    while ngen_red<ngen:
        # Generate neutrino energies
        E_gen = random.choice(E_fine, p=fdist,size=ngen_large)

        # Generate recoil energies
        E_r_gen = (2*m_N_keV*(E_gen*1000)**2.0/(m_N_keV+1000*E_gen)**2.0)*(1-sqrt(random.uniform(size=ngen_large)))

        # Form Factor correction
        mask = (random.uniform(size=ngen_large)<LabFuncs.FormFactorHelm(E_r_gen,A)**2.0)
        E_gen_full = append(E_gen_full,E_gen[mask])
        E_r_gen_full = append(E_r_gen_full,E_r_gen[mask])

        ngen_red = shape(E_gen_full)[0]
        print('Filled ',100*ngen_red/(1.0*ngen),'% of',ngen,'samples')
    E_gen = E_gen_full[0:ngen]
    E_r_gen = E_r_gen_full[0:ngen]



    # Get angles:

    # Digitize to find which energy bin to use
    E_bin_gen = digitize(log10(E_gen),log10(E_high))

    nhigh = size(E_high)
    phi_nu_gen = zeros(shape=ngen)
    costh_nu_gen = zeros(shape=ngen)
    for i in range(0,nhigh):
        # Select energy bin
        mask = E_bin_gen==i
        ngen_i = count_nonzero(mask)

        # Generate indices corresponding to 2D angular distribution
        fdist_CP = Phi_Ang[:,:,i]
        fdist_CP = reshape(fdist_CP,nc*np)
        fdist_CP = fdist_CP/sum(fdist_CP)

        igen = random.choice(arange(0,nc*np),p=fdist_CP,size=ngen_i)

        # Select phi and costh from generated index
        phi_nu_gen[mask] = P[igen]*pi/180.0
        costh_nu_gen[mask] = C[igen]

    dphi = (pi/180)*(phi_Az[1]-phi_Az[0])
    dcosth = (cosZ[1]-cosZ[0])
    phi_nu_gen += (dphi/2.0)*(2*random.uniform(size=ngen)-1)
    costh_nu_gen += (dcosth/2.0)*(2*random.uniform(size=ngen)-1)
    return E_gen,phi_nu_gen,costh_nu_gen,E_r_gen


#==============================================================================#
# Scatter a set of monte carlo generated neutrino directions to create recoil
# vectors
def ScatterNeutrinos(Nuc,E_gen,phi_nu_gen,costh_nu_gen,E_r_gen):
    ngen = size(E_gen)

    ####### Scatter
    # Nucleus mass
    A = Nuc.MassNumber
    m_N_keV = A*m_p_keV

    # Initial neutrino cartesian recoil vector
    # phi-pi because we convert South=0 [0,360] to North=0 [-180,180]
    vx = cos(phi_nu_gen-pi)*sqrt(1-costh_nu_gen**2.0)
    vy = sin(phi_nu_gen-pi)*sqrt(1-costh_nu_gen**2.0)
    vz = costh_nu_gen

    # Scattering angles
    th_R = arccos(((E_gen*1000+m_N_keV)/(E_gen*1000))*sqrt(E_r_gen/(2*m_N_keV)))
    th_add = arccos(costh_nu_gen) - (costh_nu_gen<0.0)*(1.0*th_R) + (costh_nu_gen>0.0)*th_R
    q0 = zeros(shape=(ngen,3))
    q0[:,0] = sin(th_add)*cos(phi_nu_gen-pi)
    q0[:,1] = sin(th_add)*sin(phi_nu_gen-pi)
    q0[:,2] = cos(th_add)

    # New recoil (rotate around random angle phi_R)
    ph_R = 2*pi*random.uniform(size=ngen)
    E3_gen = zeros(shape=(ngen,3))
    E3_gen[:,0] = (cos(ph_R)+(vx**2.0)*(1-cos(ph_R)))*q0[:,0] + (vx*vy*(1-cos(ph_R))-vz*sin(ph_R))*q0[:,1] + (vx*vz*(1-cos(ph_R))+vy*sin(ph_R))*q0[:,2]
    E3_gen[:,1] = (vx*vy*(1-cos(ph_R))+vz*sin(ph_R))*q0[:,0] + (cos(ph_R)+(vy**2.0)*(1-cos(ph_R)))*q0[:,1] + (vy*vz*(1-cos(ph_R))-vx*sin(ph_R))*q0[:,2]
    E3_gen[:,2] = (vx*vz*(1-cos(ph_R))-vy*sin(ph_R))*q0[:,0] + (vy*vz*(1-cos(ph_R))+vx*sin(ph_R))*q0[:,1] + (cos(ph_R)+(vz**2.0)*(1-cos(ph_R)))*q0[:,2]
    E3_gen[:,0] *= E_r_gen
    E3_gen[:,1] *= E_r_gen
    E3_gen[:,2] *= E_r_gen

    # New recoil angles
    phi_r_gen = arctan2(E3_gen[:,1],E3_gen[:,0])
    costh_r_gen = E3_gen[:,2]/E_r_gen
    #phi_r_gen = phi_r_gen*(180/pi)+180.0 # add 180 (for plotting only)

    return E_r_gen,phi_r_gen,costh_r_gen




#==============================================================================#
# Event rate vs |costh| and energy for Solar neutrinos
def dRdEdcosth_SolNu(E_nu,Flux,t1,costh_vals,E_r_vals,\
                    Nuc=Params.Ar40,\
                    Loc=Params.GranSasso,\
                    np=20,\
                    CygnusTracking=False):
    ph = linspace(-pi, pi-(2*pi/(1.0*np)), np)
    E1 = zeros(shape=(np,3))
    E2 = zeros(shape=(np,3))
    # 2D
    ne = size(E_r_vals)
    nc = size(costh_vals)
    dR = zeros(shape=(nc,ne))
    for i in range(0,nc):
        costh = costh_vals[i]
        for j in range(0,ne):
            E_r = E_r_vals[j]
            E1[:,0] = E_r*cos(ph)*sqrt(1-costh**2.0)
            E1[:,1] = E_r*sin(ph)*sqrt(1-costh**2.0)
            E1[:,2] = E_r*costh
            E2[:,0] = E_r*cos(ph)*sqrt(1-costh**2.0)
            E2[:,1] = E_r*sin(ph)*sqrt(1-costh**2.0)
            E2[:,2] = -1.0*E_r*costh
            dR[i,j] = trapz(dRdEdO_solarnu(E1,t1,E_nu,Flux,Nuc,Loc=Loc,CygnusTracking=CygnusTracking)+
                            dRdEdO_solarnu(E2,t1,E_nu,Flux,Nuc,Loc=Loc,CygnusTracking=CygnusTracking),ph)

    return dR
#==============================================================================#


#==============================================================================#
# Neutrino recoil distributions for columnar recombination
def R_Ecosth2_Iso(E_nu,Flux,A_CR,E_min,E_max=200.0,ne=20,\
                        Nuc=Params.Ar40):

    E_r_edges = linspace(E_min,E_max,ne+1)
    S_edges = linspace(0.0,1.0,ne+1)
    E_r_centers = (E_r_edges[1:]+E_r_edges[0:-1])/2.0
    S_centers = (S_edges[1:]+S_edges[0:-1])/2.0

    R = zeros(shape=(ne,ne))
    dE = E_r_edges[1]-E_r_edges[0]
    for i in range(0,ne):
        if S_edges[i]<A_CR:
            dcosth = sqrt(S_edges[i+1]/A_CR)-sqrt(S_edges[i]/A_CR)
            dR = dRdE_nu(E_r_edges,67+Jan1,False,E_nu,Flux,Nuc=Nuc)*dcosth
            R[i,:] = 0.5*dE*(dR[1:]+dR[0:-1])
        else:
            break

    return E_r_centers,S_centers,R

def R_IS_AtmNu(A_CR,E_min,E_max=200.0,ne=20,\
                    Nuc=Params.Ar40,\
                    CygnusTracking=False):

    recoildat_fname='AtmNu_Recoils_'+Nuc.Name+'.txt'
    recoils = loadtxt(recoil_dir+recoildat_fname)

    if CygnusTracking==True:
        costh_gen = abs(recoils[:,4])
    else:
        costh_gen = abs(recoils[:,2])

    I_gen = recoils[:,0]*(1-A_CR*costh_gen**2.0)
    S_gen = recoils[:,0]*A_CR*costh_gen**2.0
    H,ie,se = histogram2d(I_gen,S_gen,bins=(ne,ne),range=[[E_min,E_max],[E_min,E_max]])
    ic = linspace(E_min,E_max,ne)
    sc = linspace(E_min,E_max,ne)

    return R


def R_IS_Iso(E_nu,Flux,A_CR,E_min,E_max=200.0,ne=20,\
                        Nuc=Params.Ar40):
    t1 = 0
    E_o_edges = linspace(E_min,E_max,ne+1)
    E_o_centers = (E_o_edges[1:]+E_o_edges[0:-1])/2.0
    [I,S] = meshgrid(E_o_edges,E_o_edges)
    C = sqrt((1.0/A_CR)*1.0/(I/S+1))
    E = S+I

    dR = zeros(shape=shape(C))
    for i in range(0,ne+1):
        mask = (C[i,:]<=1)&(E[i,:]<E_max)
        if sum(mask)>1:
            costh_vals = C[i,mask]
            cmax = amax(costh_vals)
            cmin = amin(costh_vals)
            E_r_vals = E[i,mask]
            dR[i,mask] = dRdE_nu(E_r_vals,t1,False,E_nu,Flux,Nuc=Nuc)/(cmax-cmin)


    R = zeros(shape=(ne,ne))
    for i in range(0,ne):
        for j in range(0,ne):
            x = array([E[i,j],E[i,j+1],E[i+1,j],E[i+1,j+1],
                      E[i,j],E[i,j+1],E[i+1,j],E[i+1,j+1]])
            y = array([C[i,j],C[i,j+1],C[i+1,j],C[i+1,j+1],
                      C[i,j],C[i,j+1],C[i+1,j],C[i+1,j+1]])
            z = array([0,0,0,0,
                    dR[i,j],dR[i,j+1],dR[i+1,j],dR[i+1,j+1]])
            if any(z>0):
                points = column_stack((x,y,z))
                R[i,j] = ConvexHull(points,qhull_options='W1e-17 E1e-17').volume
    return R
#==============================================================================#
