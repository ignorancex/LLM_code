#===============================WIMPFuncs.py===================================#
# Created by Ciaran O'Hare 2020

# Contains all the functions for doing the WIMPy calculations

#==============================================================================#

import numpy as np
from numpy import pi, sqrt, exp, zeros, size, shape, array, linspace, logspace
from numpy import cos, sin, arctan2, arccos, trapz, ones, log10, ndim, meshgrid
from numpy import nan, isnan, column_stack, amin, amax
from numpy.linalg import norm
from scipy.special import erf
from scipy.spatial import ConvexHull
import LabFuncs
import Params
from Params import m_p_keV, c_km, seconds2year, m_p_kg, GeV_2_kg, c_cm, Jan1

#==============================================================================#
#-------------------- Energy-Time dependent recoil rate------------------------#


#---------------------------------- v_min -------------------------------------#
def MinimumWIMPSpeed(E_r,A,m_chi,delta=0):
    # E_r = recoil energy in keVr
    # A = nucleus mass number
    # m_chi = Wimp mass in GeV
    # delta = for inelastic scattering
    mu_p = 1.0e6*m_chi*m_p_keV/(1.0e6*m_chi + m_p_keV) # reduced proton mass
    m_N_keV = A*m_p_keV # nucleus mass in keV
    mu_N_keV = 1.0e6*m_chi*m_N_keV/(1.0e6*m_chi + m_N_keV) # reduced nucleus mass
    v_min = sqrt(1.0/(2*m_N_keV*E_r))*(m_N_keV*E_r/mu_N_keV + delta)*c_km
    return v_min

#---------------------------------- E_max -------------------------------------#
def MaxWIMPEnergy(A,m_chi,\
    v_lab=LabFuncs.LabVelocitySimple(67.0),v_esc=Params.SHM.EscapeSpeed):
    # A = nucleus mass number
    # v_lab = Lab velocity in km/s
    # m_chi = Wimp mass in GeV
    # v_esc = Escape speed in km/s
    m_N = m_p_keV*A
    mu_N = 1.0e6*m_N*m_chi/(1.0e6*m_chi+m_N)
    E_max_lim = 2.0*mu_N*mu_N*2.0*((v_esc+sqrt(sum(v_lab**2.0)))/c_km)**2.0/m_N
    return E_max_lim




#----------------------General event rate -------------------------------------#
def R_wimp(E_th,E_max,m_chi,sigma_p=1.0e-45,\
                Nuc=Params.Ar40,Loc=Params.GranSasso,\
                HaloModel=Params.SHM,eff_on=False,eres_on=False):
    nfine = 1000
    Efine = logspace(-3.0,log10(200.0),nfine)
    DM = Params.WIMP(m_chi,sigma_p)

    # Calculate rate at day=67 to get ~average
    dR = dRdE_wimp(Efine,array([(Jan1+67.0)]),DM,HaloModel,Nuc,Loc)

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



#-------------------- Energy dependent recoil rate-----------------------------#
def dRdE_wimp(E_r,t,DM,\
                HaloModel=Params.SHM,Nuc=Params.Ar40,Loc=Params.GranSasso):
    # relevant constants
    A = Nuc.MassNumber # mass number of nucleus
    m_chi = DM.Mass
    mu_p = 1.0e6*m_chi*m_p_keV/(1.0e6*m_chi + m_p_keV)
    sigma_p = DM.SICrossSection
    v_0 = sqrt(2.0)*HaloModel.Dispersion
    v_esc = HaloModel.EscapeSpeed
    rho_0 = HaloModel.Density
    N_esc = HaloModel.Normalisation
    FF = LabFuncs.FormFactorHelm(E_r,A)**2.0
    v_min = MinimumWIMPSpeed(E_r,A,m_chi)
    R0 = (c_cm*c_cm)*((rho_0*1.0e6*A*A*sigma_p)/(2*m_chi*GeV_2_kg*mu_p*mu_p))

    # init
    ne = size(E_r)
    nt = size(t)
    dR = zeros(shape=ne)
    gvmin = zeros(ne)


    # Mean inverse speed
    x = v_min/v_0
    z = v_esc/v_0
    if t[0] == t[-1]:
        v_e = norm(LabFuncs.LabVelocity(t[0], Loc, HaloModel.RotationSpeed))
        y = v_e/v_0
        gvmin[(x<abs(y-z))&(z<y)] = (1.0/(v_0*y))
    else:
        v_e = zeros(shape=ne)
        for i in range(0,nt):
            v_e[i] = norm(LabFuncs.LabVelocity(t[i], Loc, HaloModel.RotationSpeed))
        y = v_e/v_0
        g1 = (1.0/(v_0*y))
        gvmin[(x<abs(y-z))&(z<y)] = g1[(x<abs(y-z))&(z<y)]

    g2 = (1.0/(2.0*N_esc*v_0*y))*(erf(x+y)-erf(x-y)-(4.0/sqrt(pi))*y*exp(-z**2))
    g3 = (1.0/(2.0*N_esc*v_0*y))*(erf(z)-erf(x-y)-(2.0/sqrt(pi))*(y+z-x)*exp(-z**2))
    gvmin[(x<abs(y-z))&(z>y)] = g2[(x<abs(y-z))&(z>y)]
    gvmin[(abs(y-z)<x)&(x<(y+z))] = g3[(abs(y-z)<x)&(x<(y+z))]
    gvmin[(y+z)<x] = 0.0
    gvmin = gvmin/(1000.0*100.0) # convert to cm^-1 s

    # Compute rate = (Rate amplitude * gmin * form factor)
    dR = R0*gvmin*FF
    dR = dR*seconds2year*1000.0 # convert to per ton-year
    return dR


#-------------------- Direction dependent recoil rate--------------------------#
def dRdEdO_wimp(E,t,DM,HaloModel=Params.SHM,Nuc=Params.Ar40,\
                    Loc=Params.GranSasso,CygnusTracking=False):
    E_r = sqrt(E[:,0]**2 + E[:,1]**2 + E[:,2]**2) # Recoil energy
    x = zeros(shape=shape(E))
    x[:,0] = E[:,0]/E_r # Recoil direction
    x[:,1] = E[:,1]/E_r
    x[:,2] = E[:,2]/E_r

    # relevant constants
    A = Nuc.MassNumber # mass number of nucleus
    m_chi = DM.Mass
    mu_p = 1.0e6*m_chi*m_p_keV/(1.0e6*m_chi + m_p_keV)
    sigma_p = DM.SICrossSection
    sig_v = HaloModel.Dispersion
    v_esc = HaloModel.EscapeSpeed
    rho_0 = HaloModel.Density
    N_esc = HaloModel.Normalisation
    FF = LabFuncs.FormFactorHelm(E_r,A)**2.0
    v_min = MinimumWIMPSpeed(E_r,A,m_chi)
    R0 = (c_cm*c_cm)*((rho_0*1.0e6*A*A*sigma_p)/(4*pi*m_chi*GeV_2_kg*mu_p*mu_p))

    # Calculate v_lab
    ne = size(E_r)
    nt = size(t)
    dR = zeros(shape=(size(E_r)))
    v_lab = zeros(shape=(size(E_r),3))
    for i in range(0,nt):
        v_lab[i,:] = LabFuncs.LabVelocity(t[i], Loc, HaloModel.RotationSpeed)

    # Just put vlab towards north pole for cygnus tracking experiment:
    if CygnusTracking==True:
        for i in range(0,nt):
            v_lab[i,:] = array([0.0,0.0,sqrt(sum(v_lab[i,:]**2.0))])

    # recoil projection
    vlabdotq = (x[:,0]*v_lab[:,0]+x[:,1]*v_lab[:,1]+x[:,2]*v_lab[:,2])

    # Radon transform
    fhat = zeros(shape=shape(E_r))
    fhat[((v_min+vlabdotq)<(v_esc))] = (1/(N_esc*sqrt(2*pi*sig_v**2.0)))\
                                        *(exp(-(v_min[((v_min+vlabdotq)<(v_esc))]\
                                        +vlabdotq[((v_min+vlabdotq)<(v_esc))])\
                                        **2.0/(2*sig_v**2.0))\
                                        -exp(-v_esc**2.0/(2*sig_v**2.0)))
    fhat = fhat/(1000.0*100.0) # convert to cm^-1 s

    # Compute rate = (Rate amplitude * radon trans. * form factor)
    dR = R0*fhat*FF # correct for form factor

    dR = dR*seconds2year*1000.0 # convert to per ton-year
    return dR


#------------ 1-dimensional direction dependent recoil rate--------------------#
def dRdEdcosth_wimp(m_chi,t1,costh_vals,E_r_vals,\
                    Nuc=Params.Ar40,\
                    Loc=Params.GranSasso,\
                    sigma_p=1.0e-45,\
                    HaloModel=Params.SHM,\
                    np=20,\
                    CygnusTracking=False,\
                    ndims=2,
                    HT=False):
    DM = Params.WIMP(m_chi,sigma_p)
    ph = linspace(-pi, pi-(2*pi/(1.0*np)), np)
    E1 = zeros(shape=(np,3))
    E2 = zeros(shape=(np,3))

    if ndims==1:
        n = size(costh_vals)
        dR = zeros(shape=(n))
        for i in range(0,n):
            costh = costh_vals[i]
            E_r = E_r_vals[i]
            E1[:,0] = E_r*cos(ph)*sqrt(1-costh**2.0)
            E1[:,1] = E_r*sin(ph)*sqrt(1-costh**2.0)
            E1[:,2] = E_r*costh
            E2[:,0] = E_r*cos(ph)*sqrt(1-costh**2.0)
            E2[:,1] = E_r*sin(ph)*sqrt(1-costh**2.0)
            E2[:,2] = -1.0*E_r*costh
            dR[i] = trapz(dRdEdO_wimp(E1,t1*ones(shape=np),DM,HaloModel,Nuc,Loc,CygnusTracking=CygnusTracking)\
                            +dRdEdO_wimp(E2,t1*ones(shape=np),DM,HaloModel,Nuc,Loc,CygnusTracking=CygnusTracking),ph)
    elif ndims==2:
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
                dR[i,j] = trapz(dRdEdO_wimp(E1,t1*ones(shape=np),DM,HaloModel,Nuc,Loc,CygnusTracking=CygnusTracking))

        if HT==False:
            for i in range(0,nc):
                costh = costh_vals[i]
                for j in range(0,ne):
                    E_r = E_r_vals[j]
                    E2[:,0] = E_r*cos(ph)*sqrt(1-costh**2.0)
                    E2[:,1] = E_r*sin(ph)*sqrt(1-costh**2.0)
                    E2[:,2] = -1.0*E_r*costh
                    dR[i,j] += trapz(dRdEdO_wimp(E2,t1*ones(shape=np),DM,HaloModel,Nuc,Loc,CygnusTracking=CygnusTracking))


    return dR



def R_Ecosth_wimp(m_chi,t1,A_CR,E_min,E_max=200.0,ne=50,nc=50,\
                    Nuc=Params.Ar40,\
                    Loc=Params.GranSasso,\
                    sigma_p=1.0e-45,\
                    HaloModel=Params.SHM,\
                    np=20,\
                    CygnusTracking=False,
                    HT=False):

    E_r_edges = linspace(E_min,E_max,ne+1)
    costh_edges = linspace(0.0,1.0,nc+1)
    E_r_centers = (E_r_edges[1:]+E_r_edges[0:-1])/2.0
    costh_centers = (costh_edges[1:]+costh_edges[0:-1])/2.0

    dR = dRdEdcosth_wimp(m_chi,t1,costh_edges,E_r_edges,Nuc=Nuc,\
            CygnusTracking=CygnusTracking,sigma_p=sigma_p,np=np,ndims=2,HaloModel=HaloModel,HT=HT)

    [X,Y] = meshgrid(E_r_edges,costh_edges)
    dX = X[1:,1:]-X[1:,0:-1]
    dY = Y[1:,1:]-Y[0:-1,1:]
    R = 0.5*0.5*dX*dY*(dR[1:,1:]+dR[1:,0:-1]+dR[0:-1,1:]+dR[0:-1,0:-1])
    return E_r_centers,costh_centers,R


def R_Ecosth2_wimp(m_chi,t1,A_CR,E_min,E_max=200.0,ne=50,\
                    Nuc=Params.Ar40,\
                    Loc=Params.GranSasso,\
                    sigma_p=1.0e-45,\
                    HaloModel=Params.SHM,\
                    np=20,\
                    CygnusTracking=False):

    nc_fine = 300
    E_r_edges = linspace(E_min,E_max,ne+1)
    S_edges = linspace(0.0,1.0,ne+1)
    costh_edges = linspace(0.0,1.0,nc_fine+1)
    costh_centers = (costh_edges[1:]+costh_edges[0:-1])/2.0
    E_r_centers = (E_r_edges[1:]+E_r_edges[0:-1])/2.0
    S_centers = (S_edges[1:]+S_edges[0:-1])/2.0

    dE = E_r_edges[1]-E_r_edges[0]

    R = zeros(shape=(ne,ne))
    # dRf1 = dRdEdcosth_wimp(m_chi,t1,costh_edges,E_r_edges[0]*ones(shape=nc_fine+1),Nuc=Nuc,\
    #         CygnusTracking=CygnusTracking,sigma_p=sigma_p,np=np,ndims=1)
    # for i in range(0,ne):
    #     dRf2 = dRdEdcosth_wimp(m_chi,t1,costh_edges,E_r_edges[i+1]*ones(shape=nc_fine+1),Nuc=Nuc,\
    #             CygnusTracking=CygnusTracking,sigma_p=sigma_p,np=np,ndims=1)
    #     for k in range(0,ne):
    #         dcosth = sqrt(S_edges[i+1]/A_CR)-sqrt(S_edges[i]/A_CR)
    #         mask = (costh_edges<sqrt(S_edges[k+1]/A_CR))*(costh_edges>sqrt(S_edges[k]/A_CR))
    #         R[k,i] = 0.5*dE*(trapz(dRf1[mask],costh_edges[mask])\
    #                          +trapz(dRf2[mask],costh_edges[mask]))
    #     dRf1 = 1.0*dRf2

    for i in range(0,ne):
        if S_edges[i]<=A_CR:
            for j in range(0,ne):
                efine = linspace(E_r_edges[j],E_r_edges[j+1],5)
                smin = S_edges[i]
                smax = min(S_edges[i+1],A_CR)
                cfine = linspace(sqrt(smin/A_CR),sqrt(smax/A_CR),5)
                R[i,j] = trapz(trapz(dRdEdcosth_wimp(m_chi,t1,cfine,efine,Nuc=Nuc,HaloModel=HaloModel,\
                        CygnusTracking=CygnusTracking,sigma_p=sigma_p,np=np,ndims=2),cfine),efine)
    return E_r_centers,S_centers,R

def R_IS_wimp(m_chi,t1,A_CR,E_min,E_max=200.0,ne=20,\
                    Nuc=Params.Ar40,\
                    Loc=Params.GranSasso,\
                    sigma_p=1.0e-45,\
                    HaloModel=Params.SHM,\
                    np=20,\
                    CygnusTracking=False):

    E_o_edges = linspace(E_min,E_max,ne+1)
    E_o_centers = (E_o_edges[1:]+E_o_edges[0:-1])/2.0
    [I,S] = meshgrid(E_o_edges,E_o_edges)
    C = sqrt((1.0/A_CR)*1.0/(I/S+1))
    E = S+I

    dR = zeros(shape=shape(C))
    for i in range(0,ne+1):
        mask = (C[i,:]<=1)&(E[i,:]<E_max)
        costh_vals = C[i,mask]
        E_r_vals = E[i,mask]
        dR[i,mask] = dRdEdcosth_wimp(m_chi,t1,costh_vals,E_r_vals,Nuc=Nuc,\
                                            CygnusTracking=CygnusTracking,\
                                            sigma_p=sigma_p,\
                                            np=np,ndims=1,HaloModel=HaloModel)
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
                R[i,j] = ConvexHull(points,qhull_options='W1e-15 E1e-15').volume
    return E_o_centers,R




    # def R_Ecosth2_wimp_alt(m_chi,t1,A_CR,E_min,E_max=200.0,ne=50,\
#                     Nuc=Params.Ar40,\
#                     Loc=Params.GranSasso,\
#                     sigma_p=1.0e-45,\
#                     HaloModel=Params.SHM,\
#                     nside=8,\
#                     CygnusTracking=False):

#     DM = Params.WIMP(m_chi,sigma_p)
#     E_r_edges = linspace(E_min,E_max,ne+1)
#     S_edges = linspace(0.0,1.0,ne+1)
#     E_r_centers = (E_r_edges[1:]+E_r_edges[0:-1])/2.0
#     S_centers = (S_edges[1:]+S_edges[0:-1])/2.0

#     # Healpix discretisation of a sphere
#     npix = 12*nside**2
#     dpix = 4*pi/(npix*1.0)
#     x_pix = zeros(shape=(npix,3))
#     for i in range(0,npix):
#         x_pix[i,:] = hp.pix2vec(nside, i)
#     t1 = t1*ones(shape=npix)
#     costh = x_pix[:,2]


#     dE = E_r_edges[1]-E_r_edges[0]
#     R = zeros(shape=(ne,ne))

#     dRf1 = dpix*dRdEdO_wimp(E_r_edges[0]*x_pix,t1,DM,HaloModel,Nuc,Loc,CygnusTracking=CygnusTracking)
#     for i in range(0,ne):
#         dRf2 =  dpix*dRdEdO_wimp(E_r_edges[i]*x_pix,t1,DM,HaloModel,Nuc,Loc,CygnusTracking=CygnusTracking)

#         for k in range(0,ne):
#             mask = (abs(costh)<sqrt(S_edges[k+1]/A_CR))*(abs(costh)>sqrt(S_edges[k]/A_CR))
#             R[k,i] = 0.5*dE*(sum(dRf1[mask])+sum(dRf2[mask]))

#         dRf1 = 1.0*dRf2

#     return E_r_centers,S_centers,R
