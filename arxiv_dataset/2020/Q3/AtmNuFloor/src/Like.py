#====================================Like.py===================================#
# Created by Ciaran O'Hare 2020

# Contains functions for interfacing with the fortran code in src/like
# the fortran likelihood code needs to be compiled first by running the make
# file in src/like
#==============================================================================#


from __future__ import print_function
from numpy import pi, sqrt, exp, zeros, size, shape, array, append, flipud, gradient
from numpy import trapz, interp, loadtxt, log10, log, savetxt, vstack, transpose
from numpy import ravel,tile,mean
from scipy.ndimage.filters import gaussian_filter1d
#from scipy.optimize import minimize
from numpy.linalg import norm
from scipy.special import gammaln
from Params import *
from NeutrinoFuncs import *
from WIMPFuncs import *
from LabFuncs import *
import shlex
import subprocess
import pprint

#==============================================================================#
# Loading list of atmospheric recoils to then make a histogram out of later
def LoadAtmRecoils(Nuc):
    recoildat_fname='AtmNu_Recoils_'+Nuc.Name+'.txt'
    AtmRecoils = loadtxt(recoil_dir+recoildat_fname)
    return AtmRecoils
#==============================================================================#


#==============================================================================#
# Both of these functions save WIMP/neutrino data in a format that can be then
# read by the fortran code
def SaveWIMPData(inp,R_sig,m_vals):
    nTot_bins = shape(R_sig)[1]
    nm = shape(R_sig)[0]
    hdr1 = str(nm)+' '+str(nTot_bins)
    dat1 = zeros(shape=(nm,nTot_bins+1))
    dat1[:,1:] = R_sig
    dat1[:,0] = m_vals
    savetxt(recoil_dir+'RD_sig_'+inp+'.txt',dat1,header=hdr1)
    return

def SaveNuData(inp,R_nu,Flux_norm,Flux_err):
    nTot_bins = shape(R_nu)[1]
    n_nu = shape(R_nu)[0]
    hdr2 = str(n_nu)+' '+str(nTot_bins)
    dat2 = zeros(shape=(n_nu,nTot_bins+2))
    dat2[:,2:] = R_nu
    dat2[:,0] = Flux_norm
    dat2[:,1] = Flux_err
    savetxt(recoil_dir+'RD_bg_'+inp+'.txt',dat2,header=hdr2)
    return
#==============================================================================#




#==============================================================================#
# These are functions that call the compiled fortran code from python. The first
def runDL_fort(inp,ex_min=1.0e-1,ex_max=1.0e7,n_ex=9,\
                  verbose=False):
    savetxt(recoil_dir+'Ex_'+inp+'.txt',array([[ex_min],[ex_max],[n_ex]]))
    command = "../src/like/./runDL "+inp
    if verbose:
        command += " 1"

    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
    while True:
        output = process.stdout.readline()
        if process.poll()==0:
            break
        if output:
            print(output.strip().decode("utf-8"))
    rc = process.poll()
    return rc

def runDL(inp,R_sig,R_nu,m_vals,ex_min,ex_max,n_ex,f_AtmNu=0.25,A_CR=1.0,verbose=False):
    Names,Solar,E_nu_all,Flux_all,Flux_norm,Flux_err = GetNuFluxes(3.0,Nuc=Xe131)
    Flux_err[Names=='Atm'] = f_AtmNu


    if A_CR<1.0:
        ne = int(sqrt(size(R_sig[0,:])))
        nm = size(m_vals)
        R_sig_red = zeros(shape=(nm,ne*ne))
        for i in range(0,nm):
            R_sig_red[i,:] = Acorr(R_sig[i,:],A_CR)
        R_nu_red = vstack((Acorr(R_nu[0,:],A_CR),Acorr(R_nu[1,:],A_CR),\
                        Acorr(R_nu[2,:],A_CR),Acorr(R_nu[3,:],A_CR)))
        SaveWIMPData(inp,R_sig_red,m_vals)
        SaveNuData(inp,R_nu_red,Flux_norm,Flux_err)
    else:
        SaveWIMPData(inp,R_sig,m_vals)
        SaveNuData(inp,R_nu,Flux_norm,Flux_err)


    rc = runDL_fort(inp,ex_min=ex_min,ex_max=ex_max,n_ex=n_ex,verbose=verbose)
    return
#==============================================================================#




#==============================================================================#
#==============================================================================#
# Generating the non-directional signal (wimp) and background (neutrino)
# distributions
def Rsig_NonDirectional(Nuc,m_vals,E_min,E_max,ne,np=np,\
                         HaloModel=SHM):

    t1 = array([67.0+Jan1])
    nm = size(m_vals)
    E_edges = logspace(log10(E_min),log10(E_max),ne+1)
    eff = efficiency(Nuc,E_edges)

    R_sig = zeros(shape=(nm,ne))
    for i in range(0,nm):
        R_tot = R_wimp(E_min,E_max,m_vals[i],Nuc=Nuc,\
                        HaloModel=HaloModel,eff_on=True)
        if (R_tot>0.0):
            dR = eff*dRdE_wimp(E_edges,t1,WIMP(m_vals[i],1.0e-45),\
                                Nuc=Nuc,HaloModel=HaloModel)
            R_sig[i,:] = 0.5*(dR[1:]+dR[0:-1])*(E_edges[1:]-E_edges[0:-1])
            R_sig[i,:] = R_sig[i,:]*R_tot/sum(R_sig[i,:])

    return R_sig

def Rnu_NonDirectional(Nuc,E_min,E_max,ne):
    E_edges = logspace(log10(E_min),log10(E_max),ne+1)
    eff = efficiency(Nuc,E_edges)

    Names,Solar,E_nu_all,Flux_all,Flux_norm,Flux_err = GetNuFluxes(3.0,Nuc=Xe131)

    dR = AllNuRates(E_edges,67.0+Jan1,Solar,E_nu_all,Flux_all,Nuc=Nuc)
    n_nu = shape(Flux_all)[0]
    R_nu = zeros(shape=(n_nu,ne))
    for i in range(0,n_nu):
        R_nu[i,:] = 0.5*(eff[1:]*dR[i,1:]+eff[0:-1]*dR[i,0:-1])\
                        *(E_edges[1:]-E_edges[0:-1])
    R_nu[0,:] *= R_hep(E_min,E_max,Nuc=Nuc,eff_on=True)/sum(R_nu[0,:])
    R_nu[1,:] *= R_8B(E_min,E_max,Nuc=Nuc,eff_on=True)/sum(R_nu[1,:])
    R_nu[2,:] *= R_DSNB(E_min,E_max,Nuc=Nuc,eff_on=True)/sum(R_nu[2,:])
    R_nu[3,:] *= R_AtmNu(E_min,E_max,Nuc=Nuc,eff_on=True)/sum(R_nu[3,:])
    return R_nu
#==============================================================================#
#==============================================================================#





#==============================================================================#
#==============================================================================#
# Function used to correct the full E, costh distribution for a value of A
# We do this simply by rescaling the costh distribution for each energy bin
def Acorr(R_1,A_CR):
    ne = int(sqrt(size(R_1)))
    R_1 = reshape(R_1,(ne,ne))
    R_1_red = zeros(shape=(ne,ne))
    R_tot = sum(R_1)
    for i in range(0,ne):
        y = R_1[:,i]
        if sum(R_1[:,i])>0:
            dy = amax(y)-amin(y)
            midy = mean(y)
            R_1_red[:,i] = A_CR*(y-midy)+midy # rescale by A
            R_1_red[:,i] = R_1_red[:,i]*sum(R_1[:,i])/sum(R_1_red[:,i])
    return ravel(R_1_red)

# WIMP distribution as a function of E_r and costh
def Rsig_Ecosth(t1,Nuc,m_vals,E_min,E_max,ne,\
                np=20,sigma_p=1.0e-45,HaloModel=SHM,CygnusTracking=True,\
                HT=False):
    nm = size(m_vals)
    R_sig = zeros(shape=(nm,ne*ne))

    E_r_edges = logspace(log10(E_min),log10(E_max),ne+1)
    costh_edges = sqrt(linspace(0.0,1.0,ne+1))
    if HT:
        costh_edges = linspace(-1.0,1.0,ne+1)
    [X,Y] = meshgrid(E_r_edges,costh_edges)
    eff = efficiency(Nuc,X)
    dX = X[1:,1:]-X[1:,0:-1]
    dY = Y[1:,1:]-Y[0:-1,1:]

    for i in range(0,nm):
        R_tot = R_wimp(E_min,E_max,m_vals[i],Nuc=Nuc,eff_on=True)
        if R_tot>0:
            dR = eff*dRdEdcosth_wimp(m_vals[i],t1,costh_edges,E_r_edges,Nuc=Nuc,\
                                    CygnusTracking=CygnusTracking,\
                                    sigma_p=sigma_p,np=np,ndims=2,\
                                    HaloModel=HaloModel,HT=HT)

            R = 0.5*0.5*dX*dY*(dR[1:,1:]+dR[1:,0:-1]+dR[0:-1,1:]+dR[0:-1,0:-1])
            R *= R_tot/sum(sum(R))
            R_sig[i,:] = ravel(R)
        print('m =',m_vals[i],' | Nucleus =',Nuc.Name,' | Time =',t1,\
                    ' | R =',sum(R_sig[i,:]))
    return R_sig

# Same but averaged over one day
def Rsig_Ecosth_TimeAveraged(nt,Nuc,m_vals,E_min,E_max,ne,\
                            np=20,sigma_p=1.0e-45,HaloModel=SHM,HT=False):
    tvals = JulianDay(1,9,2020,0.0)+linspace(0,1.0-1.0/(1.0*nt),nt)
    R_sig = 0
    for i in range(0,nt):
        print('T = ',i+1,'of',nt)
        R_sig += Rsig_Ecosth(tvals[i],Nuc,m_vals,E_min,E_max,ne,\
                            CygnusTracking=False,np=np,\
                            sigma_p=sigma_p,HaloModel=SHM,HT=HT)
    R_sig = R_sig/(1.0*nt)
    return R_sig

# Neutrino distribution
def Rnu_Ecosth(Nuc,E_min,E_max,ne,AtmRecoils,CygnusTracking=True,HT=False):
    Names,Solar,E_nu_all,Flux_all,Flux_norm,Flux_err = GetNuFluxes(3.0,Nuc=Nuc)

    E_r_edges = logspace(log10(E_min),log10(E_max),ne+1)
    E_r_centers = (E_r_edges[1:]+E_r_edges[0:-1])/2.0

    R = zeros(shape=(ne,ne))
    R_nu = zeros(shape=(4,ne*ne))

    costh_edges = sqrt(linspace(0.0,1.0,ne+1))
    dcosth2 = (costh_edges[1:])**2.0-(costh_edges[0:-1])**2.0
    if HT:
        costh_edges = linspace(-1.0,1.0,ne+1)
        dcosth2 = costh_edges[1:]-costh_edges[0:-1]
    costh_centers = (costh_edges[1:]+costh_edges[0:-1])/2.0


    [E,C] = meshgrid(E_r_centers,costh_centers)
    eff2 = efficiency(Nuc,E)
    eff1 = efficiency(Nuc,E_r_edges)

    [X,Y] = meshgrid(E_r_edges,costh_edges)
    eff3 = efficiency(Nuc,X)
    dX = X[1:,1:]-X[1:,0:-1]
    dY = Y[1:,1:]-Y[0:-1,1:]


    ##### Solar
    for s in [0,1]:
        dR = eff3*dRdEdcosth_SolNu(E_nu_all[s,:],Flux_all[s,:],t1,costh_edges,E_r_edges,Nuc=Nuc,CygnusTracking=CygnusTracking)
        R = 0.5*0.5*dX*dY*(dR[1:,1:]+dR[1:,0:-1]+dR[0:-1,1:]+dR[0:-1,0:-1])
        R = R/sum(sum(R))
        R_nu[s,:] = ravel(R)


    ##### DSNB
    for i in range(0,ne):
        dR = dcosth2[i]*eff1*dRdE_nu(E_r_edges,67+Jan1,False,\
                                        E_nu_all[0,:],Flux_all[0,:],Nuc=Nuc)
        R[i,:] = 0.5*(E_r_edges[1:]-E_r_edges[0:-1])*(dR[1:]+dR[0:-1])
    R_nu[2,:] = ravel(R)




    ##### Atm
    #     for i in range(0,ne):
    #         dR = dRdE_nu(E_r_edges,67+Jan1,False,E_nu_all[1,:],Flux_all[1,:],Nuc=Nuc)
    #         R[i,:] = 0.5*(E_r_edges[1:]-E_r_edges[0:-1])*(dR[1:]+dR[0:-1])
    #     R_nu[1,:] = ravel(R)
    if CygnusTracking:
        costh = AtmRecoils[:,4]
    else:
        costh = AtmRecoils[:,2]

    if HT:
        costhmin = -1.0
    else:
        costh = costh**2.0
        costhmin = 0.0

    R,ce,ee = histogram2d(costh,log10(AtmRecoils[:,0]),bins=(ne,ne),\
                        range=[[costhmin,1.0],[log10(E_min),log10(E_max)]])
    R = R*eff2
    R = R/sum(sum(R))
    R[R==0] = amin(R[R>0])
    R_nu[3,:] = ravel(R)


    # Normalise to total rates
    R_nu[0,:] *= R_hep(E_min,E_max,Nuc=Nuc,eff_on=True)/sum(R_nu[0,:])
    R_nu[1,:] *= R_8B(E_min,E_max,Nuc=Nuc,eff_on=True)/sum(R_nu[1,:])
    R_nu[2,:] *= R_DSNB(E_min,E_max,Nuc=Nuc,eff_on=True)/sum(R_nu[2,:])
    R_nu[3,:] *= R_AtmNu(E_min,E_max,Nuc=Nuc,eff_on=True)/sum(R_nu[3,:])

    print('R_hep = ',sum(R_nu[0,:]))
    print('R_B8 = ',sum(R_nu[1,:]))
    print('R_DSNB = ',sum(R_nu[2,:]))
    print('R_Atm = ',sum(R_nu[3,:]))

    return R_nu
#==============================================================================#















#==============================================================================#
# def PackBinnedEvents(dR,x,y,z=0):
#     nx = size(x)-1
#     ny = size(y)-1
#
#     if ndim(dR)==2:
#         ntot = ny*nx
#         [X,Y] = meshgrid(x,y)
#         dX = X[1:,1:]-X[1:,0:-1]
#         dY = Y[1:,1:]-Y[0:-1,1:]
#         R = 0.5*0.5*dX*dY*(dR[1:,1:]+dR[1:,0:-1]+dR[0:-1,1:]+dR[0:-1,0:-1])
#     elif ndim(dR)==3:
#         nz = size(z)-1
#         ntot = ny*nx*nz
#         [Y,X,Z] = meshgrid(x,y,z)
#         dX = X[1:,1:]-X[1:,0:-1]
#         dY = Y[1:,1:]-Y[0:-1,1:]
#         dZ = Z[1:,1:,1:]-Z[1:,1:,0:-1]
#         R = 0.5*0.5*0.5*dX*dY*dZ*(dR[1:,1:,1:]+dR[1:,0:-1,1:]+dR[0:-1,1:,1:]+\
#                                     dR[0:-1,0:-1,1:]+dR[1:,1:,0:-1]+\
#                                     dR[1:,0:-1,0:-1]+dR[0:-1,1:,0:-1]+\
#                                     dR[0:-1,0:-1,0:-1])
#     return ravel(R)

# def Rsig_CR_CygnusTracking(Nuc,E_th,m_min=1.0,m_max=1.0e4,nm=100,\
#                          E_max=200.0,nE_bins=50,ncosth_bins=20,\
#                          f_AtmNu=0.25,\
#                          HaloModel=SHM,\
#                          np=20):
#     t1 = array([67.0+Jan1])
#     m_vals = logspace(log10(m_min),log10(m_max),nm)
#     E_edges = linspace(E_th,E_max,nE_bins+1)
#     costh_edges = linspace(0.0,1.0,ncosth_bins+1)
#
#     nTot_bins = ncosth_bins*nE_bins
#
#     R_sig = zeros(shape=(nm,nTot_bins))
#     for i in range(0,nm):
#         R_tot = R_wimp(E_th,E_max,m_vals[i],Nuc=Nuc,HaloModel=HaloModel)
#         if (R_tot>0.0):
#             dR = dRdEdcosth_wimp(m_vals[i],t1,costh_edges,E_edges,\
#                                 Nuc=Nuc,HaloModel=HaloModel,\
#                                 CygnusTracking=True,ndims=2,np=np)
#             R_sig[i,:] = PackBinnedEvents(dR,E_edges,costh_edges)
#             R_sig[i,:] = R_sig[i,:]*R_tot/sum(R_sig[i,:])
#
#         print('m_chi = ',m_vals[i],'R = ',sum(R_sig[i,:]))
#     return R_sig,m_vals
#
# def Rnu_CR_CygnusTracking(Nuc,E_th,E_max=200.0,nE_bins=50,ncosth_bins=20,\
#                             f_AtmNu=0.25):
#     E_edges = linspace(E_th,E_max,nE_bins+1)
#     costh_edges = linspace(0.0,1.0,ncosth_bins+1)
#     nTot_bins = ncosth_bins*nE_bins
#
#     Names,Solar,E_nu_all,Flux_all,Flux_norm,Flux_err = GetNuFluxes(50.0,Nuc=Nuc)
#     Flux_err[Names=='Atm'] = f_AtmNu
#     dR = AllNuRates(E_edges,67.0+Jan1,Solar,E_nu_all,Flux_all,Nuc=Nuc)
#     n_nu = shape(Flux_all)[0]
#     R_nu = zeros(shape=(n_nu,nTot_bins))
#     for i in range(0,n_nu):
#         R = 0.5*(dR[i,1:]+dR[i,0:-1])*(E_edges[1:]-E_edges[0:-1])
#         R_nu[i,:] = tile(R,ncosth_bins)/(1.0*ncosth_bins)
#     R_nu[0,:] *= R_DSNB(E_th,E_max,Nuc=Nuc)/sum(R_nu[0,:])
#     R_nu[1,:] *= R_AtmNu(E_th,E_max,Nuc=Nuc)/sum(R_nu[1,:])
#     print('AtmNu = ','R = ',sum(R_nu[1,:]))
#     return R_nu,Flux_norm,Flux_err
# #==============================================================================#
#



#==============================================================================#
# def Mask_IS(A_CR,E_min,E_max,ne=20):
#     E_o_edges = linspace(E_min,E_max,ne+1)
#     [I,S] = meshgrid(E_o_edges,E_o_edges)
#     C = sqrt((1.0/A_CR)*1.0/(I/S+1))
#     E = S+I
#     mask = (C[0:-1,0:-1]<=1)+(C[1:,1:]<=1)+(C[0:-1,1:]<=1)+(C[1:,0:-1]<=1)
#     mask *= (E[0:-1,0:-1]<=E_max)+(E[1:,1:]<=E_max)+(E[0:-1,1:]<=E_max)+(E[1:,0:-1]<=E_max)
#     return mask

# def Mask_2D(A_CR,E_min,E_max,ne):
#     E_r_edges = linspace(E_min,E_max,ne+1)
#     S_edges = linspace(0.0,1.0,ne+1)
#     [E,S] = meshgrid(E_r_edges,S_edges)
#     mask = (S[0:-1,0:-1]<A_CR)
#     return mask

# def Rsig_IS(Nuc,E_th,m_min=1.0,m_max=1.0e4,nm=100,\
#                          A_CR=1.0,E_max=200.0,nE_bins=20,\
#                          f_AtmNu=0.25,\
#                          HaloModel=SHM,\
#                          np=20,\
#                          CygnusTracking=True):
#
#     mask = Mask_2D(A_CR,E_th,E_max,ne=nE_bins)
#     nTot_bins = sum(mask)
#
#
#     t1 = array([67.0+Jan1])
#     m_vals = logspace(log10(m_min),log10(m_max),nm)
#
#     nTot_bins = sum(sum(mask))
#
#     R_sig = zeros(shape=(nm,nTot_bins))
#     for i in range(0,nm):
#         R_tot = R_wimp(E_th,E_max,m_vals[i],Nuc=Nuc,HaloModel=HaloModel)
#         if (R_tot>0.0):
#             E_r_centers,S_centers,R = R_Ecosth2_wimp(m_vals[i],t1,A_CR,E_th,\
#                                    E_max=E_max,ne=nE_bins,\
#                                    CygnusTracking=CygnusTracking,np=np,\
#                                   Nuc=Nuc,HaloModel=HaloModel)
#             R_sig[i,:] = R[mask]
#             R_sig[i,:] *= R_tot/sum(R_sig[i,:])
#         print('m_chi = ',m_vals[i],'R = ',sum(R_sig[i,:]))
#     return R_sig,m_vals
#
# def Rnu_IS(Nuc,E_th,A_CR=1.0,E_max=200.0,nE_bins=20,f_AtmNu=0.25,\
#                             CygnusTracking=True):
#     mask = Mask_2D(A_CR,E_th,E_max,ne=nE_bins)
#     nTot_bins = sum(sum(mask))
#
#     Names,Solar,E_nu_all,Flux_all,Flux_norm,Flux_err = GetNuFluxes(50.0,Nuc=Nuc)
#     Flux_err[Names=='Atm'] = f_AtmNu
#     n_nu = shape(Flux_all)[0]
#     R_nu = zeros(shape=(n_nu,nTot_bins))
#     E_r_centers,S_centers,R = R_Ecosth2_Iso(E_nu_all[0,:],Flux_all[0,:],\
#                             A_CR,E_th,E_max=E_max,ne=nE_bins,Nuc=Nuc)
#     R_nu[0,:] = R[mask]
#     E_r_centers,S_centers,R = R_Ecosth2_Iso(E_nu_all[1,:],Flux_all[1,:],\
#                             A_CR,E_th,E_max=E_max,ne=nE_bins,Nuc=Nuc)
#     R_nu[1,:] = R[mask]
#     #R_nu[1,:] = ravel(R_IS_AtmNu(A_CR,E_th,E_max=E_max,ne=nE_bins,\
#     #                    CygnusTracking=CygnusTracking,Nuc=Nuc)[mask])
#
#
#
#     R_nu[0,:] *= R_DSNB(E_th,E_max,Nuc=Nuc)/sum(R_nu[0,:])
#     R_nu[1,:] *= R_AtmNu(E_th,E_max,Nuc=Nuc)/sum(R_nu[1,:])
#     print('AtmNu = ','R = ',sum(R_nu[1,:]))
#     print('DSNB = ','R = ',sum(R_nu[0,:]))
#     return R_nu,Flux_norm,Flux_err
# #==============================================================================#
#
#
#
#
# def Rsig_CR(Nuc,A_CR,E_min,E_max,R_wa,m_vals,nbins=50,eff_on=True,ngen=10000000):
#     nm = shape(R_wa)[0]
#     ne = int(sqrt(shape(R_wa)[1]))
#
#     mask = Mask_2D(A_CR,E_min,E_max,nbins)
#     nbins_red = sum(sum(mask))
#
#     E_r_edges = linspace(E_min,E_max,ne+1)
#     E_r_centers = (E_r_edges[1:]+E_r_edges[0:-1])/2.0
#     costh_edges = linspace(0.0,1.0,ne+1)
#     costh_centers = (costh_edges[1:]+costh_edges[0:-1])/2.0
#     [E,C] = meshgrid(E_r_centers,costh_centers)
#     if eff_on:
#         eff = efficiency(Nuc,E)
#     else:
#         eff = ones(shape=ne+1)
#
#     E = reshape(E,ne*ne)
#     C = reshape(C,ne*ne)
#     nTot_bins = ne*ne
#     R = zeros(shape=(nm,nbins_red))
#     for i in range(0,nm):
#         R_tot = R_wimp(E_min,E_max,m_vals[i],Nuc=Nuc,eff_on=eff_on)
#         if sum(R_wa[i,:])>0.0:
#             fdist = R_wa[i,:]*1.0
#             fdist = fdist/sum(fdist)
#             igen = random.choice(arange(0,ne*ne),p=fdist,size=ngen)
#             E_r_gen = E[igen]
#             costh_nu_gen = C[igen]
#
#             R_w,ce,ee = histogram2d(A_CR*costh_nu_gen**2.0,E_r_gen,bins=(nbins,nbins),range=[[0.0,1.0],[E_min,E_max]])
#             ec = (ee[1:]+ee[0:-1])/2.0
#             cc = (ce[1:]+ce[0:-1])/2.0
#             R_w = R_w*eff
#             R_w = R_w/sum(sum(R_w))
#             R[i,:] = R_tot*R_w[mask]
#     return R
#
# def Rnu_CR(Nuc,A_CR,E_Atm,costh_Atm,E_min,E_max,nbins=50,eff_on=True,ngen=10000000):
#     ne = nbins*2
#     Names,Solar,E_nu_all,Flux_all,Flux_norm,Flux_err = GetNuFluxes(50.0,Nuc=Nuc)
#
#     E_r_edges = linspace(E_min,E_max,ne+1)
#     E_r_centers = (E_r_edges[1:]+E_r_edges[0:-1])/2.0
#     if eff_on:
#         eff = efficiency(Nuc,E_r_edges)
#     else:
#         eff = ones(shape=ne+1)
#
#     dR1 = AllNuRates(E_r_edges,67+Jan1,Solar,E_nu_all,Flux_all,Nuc=Nuc)
#     R_nu1a = 0.5*(E_r_edges[1]-E_r_edges[0])*(eff[1:]*dR1[0,1:]+eff[0:-1]*dR1[0,0:-1])
#     R_nu2a = 0.5*(E_r_edges[1]-E_r_edges[0])*(eff[1:]*dR1[1,1:]+eff[0:-1]*dR1[1,0:-1])
#
#     # DSNB
#     igen = random.choice(arange(0,ne),p=R_nu1a/sum(R_nu1a),size=ngen)
#     E_r_gen = E_r_centers[igen]
#     costh_nu_gen = random.uniform(size=ngen)
#     R_nu1,ce,ee = histogram2d(A_CR*costh_nu_gen**2.0,E_r_gen,bins=(nbins,nbins),range=[[0.0,1.0],[E_min,E_max]])
#     ec = (ee[1:]+ee[0:-1])/2.0
#     cc = (ce[1:]+ce[0:-1])/2.0
#     R_nu1 = R_nu1/sum(sum(R_nu1))
#
#     # Atm
#     igen = random.choice(arange(0,ne),p=R_nu2a/sum(R_nu2a),size=ngen)
#     E_r_gen = E_r_centers[igen]
#     costh_nu_gen = random.uniform(size=ngen)
#     R_nu2,ce,ee = histogram2d(A_CR*costh_nu_gen**2.0,E_r_gen,bins=(nbins,nbins),range=[[0.0,1.0],[E_min,E_max]])
#     #R_nu2,ce,ee = histogram2d(A_CR*costh_Atm**2.0,E_Atm,bins=(nbins,nbins),range=[[0.0,1.0],[E_min,E_max]])
#     ec = (ee[1:]+ee[0:-1])/2.0
#     cc = (ce[1:]+ce[0:-1])/2.0
#     R_nu2 = R_nu2/sum(sum(R_nu2))
#     R_nu2[R_nu2==0] = amin(R_nu2[R_nu2>0])
#
#     R_nu1 *= R_DSNB(E_min,E_max,Nuc=Nuc,eff_on=eff_on)/sum(sum(R_nu1))
#     R_nu2 *= R_AtmNu(E_min,E_max,Nuc=Nuc,eff_on=eff_on)/sum(sum(R_nu2))
#
#     mask = Mask_2D(A_CR,E_min,E_max,nbins)
#     R_nu = vstack((R_nu1[mask],R_nu2[mask]))
#     return R_nu
#




#===================================USEFUL SUMS================================#
def lnPF(Nob,Nex): # SUM OF LOG(POISSON PDF)
    # in principle there should be a log gamma here
    # (or a factorial if using real data)
    # but it always cancels in the likelihood ratio
    # so it's commented out for speed
    L = sum(Nob*log(Nex) - Nex) #- gammaln(Nob+1.0))
    return L

def lnChi2(Nob,Nex): # SUM OF LOG(POISSON PDF)
    L = -0.5*sum((Nob-Nex)**2.0/Nex)
    return L

def lnGF(x,mu,sig): # SUM OF LOG(GAUSSIAN PDF)
    L = sum(-1.0*log(sig)-0.5*log(2.0*pi)-(x-mu)**2.0/(2.0*sig**2.0))
    return L
