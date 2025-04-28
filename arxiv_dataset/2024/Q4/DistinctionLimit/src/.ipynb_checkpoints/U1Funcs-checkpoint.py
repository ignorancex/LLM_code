#===========================NeutrinoFogFuncs.py===================================#
# Created by Bing-Long Zhang, 2023

# Contains functions for performing calculations of the neutrino fog and floor, etc.
# All functions below are developed by modifying Ciaran's Python code.

#==============================================================================#
# import
import scipy as sc
import numpy as np
import time, multiprocessing
from numba import jit, float64
import matplotlib.pyplot as plt
from scipy import interpolate, stats
from scipy.integrate import quad_vec
#==============================================================================#
# Asymptotic-Analytic Method
# @jit([(float64[:], float64[:,:], float64[:], float64)], nopython=True)
# def phiGen1(s, b, sigmaTheta, exposure):
#     n_nu = len(b)
#     temp = s + np.sum(b, axis=0)
#     derList = [[exposure*np.sum(b[i]*x/temp) for x in b[i+1:]] for i in range(n_nu-1)]
#     G1 = exposure*np.sum(s*s/temp)
#     G2 = exposure*np.array([np.sum(s*x/temp) for x in b])
#     G3 = np.zeros((n_nu,n_nu))
#     for i in range(0, n_nu-1):
#         G3[i,i+1:] = derList[i]
#         G3[i+1:, i] = derList[i]
#     diagTerm = 1/(sigmaTheta*sigmaTheta)+exposure*np.array([np.sum(x*x/temp) for x in b])
#     G3 = G3+np.diag(diagTerm)
#     res = G1 - G2@np.linalg.inv(G3)@G2
#     return res

from Params import mono, NuMaxEnergy, NuFlux, NuUnc, whichsolar, n_nu_tot
from Params import nufile_root, nufile_dir, nuname, n_Enu_vals, recoil_dir
import Params, LabFuncs
from LabFuncs import FormFactorHelm
from numpy import pi
from WIMPFuncs import U1BinnedWIMPRate, MeanInverseSpeed_SHM
from LabFuncs import JulianDay, LabVelocity
from propagation import propagation

def EarthVelocityFunc(JD):
    return np.linalg.norm(LabVelocity(JD, Loc=Params.GranSasso, v_LSR=233.0)[0])

def DMvRatesAMGen(mDM, E_th, E_max, ne, Nuc, NuclearEnhancementFactor, timeBin):
    nFine = 5
    tList = np.linspace(0,365.25,timeBin+1)+JulianDay(1,1,2023,0)-28.6186
    DMRatesList = np.array([np.sum([U1BinnedWIMPRate(E_th*1.0e6,E_max*1.0e6,ne,np.array([mDM]),Nuc,NuclearEnhancementFactor,FormFactorHelm,\
                       lambda v_min: MeanInverseSpeed_SHM(v_min,sig_v=167.0,v_esc=533.0,v_lab=EarthVelocityFunc(t)),\
                    v_lab=EarthVelocityFunc(t))[1][0] for t in np.linspace(tList[i],tList[i+1],nFine)],axis=0)/nFine \
                         for i in range(timeBin)])
    DMRatesList = DMRatesList.flatten()/timeBin
    #DMMassList = np.log10(DMMassList)
    return DMRatesList

def DMvRatesGen(mDMList, E_th, E_max, ne, Nuc, NuclearEnhancementFactor,v_lab=245.6):
    DMMassList, DMRatesList = U1BinnedWIMPRate(E_th*1.0e6,E_max*1.0e6,ne,mDMList,Nuc,NuclearEnhancementFactor,FormFactorHelm,\
                                               lambda v_min: MeanInverseSpeed_SHM(v_min,sig_v=167.0,v_esc=533.0,v_lab=v_lab),\
                                               v_lab=v_lab)
    #DMMassList = np.log10(DMMassList)
    return [DMMassList, DMRatesList]

# change the MeV unit into GeV
def MaxNuRecoilEnergies(Nuc): # Max recoil energies
    m_N = 0.93141941*(Nuc.MassNumber)
    E_r_max = 2*m_N*(NuMaxEnergy*1e-3)**2.0/(m_N+NuMaxEnergy*1e-3)**2.0
    return E_r_max

def MaxNuRecoilEnergiesCorrect(Nuc, nuMaxEnergy): # Max recoil energies
    m_N = 0.93141941*(Nuc.MassNumber)
    E_r_max = 2*m_N*(nuMaxEnergy)**2.0/(m_N+nuMaxEnergy)**2.0
    return E_r_max

def GetNuFluxesCorrect(E_th,Nuc=Params.F19):
    # Reads each neutrino flux data file
    # the energies are stored in E_nu_all, fluxes in Flux_all

    # Figure out which backgrounds give recoils above E_th
    E_r_max = MaxNuRecoilEnergies(Nuc) # Max recoil energy for neutrino
    sel = range(1,n_nu_tot+1)*(E_r_max>E_th)
    sel = sel[sel!=0]-1
    n_nu = len(E_r_max[E_r_max>E_th])
    E_nu_all = np.zeros(shape=(n_nu,n_Enu_vals))
    Flux_all = np.zeros(shape=(n_nu,n_Enu_vals))
    Flux_err = np.zeros(shape=(n_nu))
    Flux_norm = np.zeros(shape=(n_nu))
    Solar = np.zeros(n_nu,dtype=bool)
    nuMaxEnergy = np.zeros(shape=(n_nu))
    Names = np.asarray([nuname[i] for i in sel])

    ii = 0
    for s in sel:
        if mono[s]:
            E_nu_all[ii,0] = NuMaxEnergy[s]*1e-3
            Flux_all[ii,0] = NuFlux[s]
        else:
            data = np.loadtxt(nufile_dir+'normalised/'+nuname[s]+nufile_root,delimiter=',')
            E_nu_all[ii,:],Flux_all[ii,:] = data[:,0]*1e-3,data[:,1]*1e3
            Flux_all[ii,:] = Flux_all[ii,:]*NuFlux[s]

        Flux_norm[ii] = NuFlux[s]
        Flux_err[ii] = NuUnc[s] # Select rate normalisation uncertainties
        nuMaxEnergy[ii] = NuMaxEnergy[s]*1e-3
        Solar[ii] = whichsolar[s]
        ii = ii+1
    return Names,Solar,E_nu_all,Flux_all,Flux_norm,Flux_err,nuMaxEnergy


def oneLoopCouplingFuncGen():
    mE, mMuon, mTau = 0.511*1e-3, 105.66*1e-3, 1776.86*1e-3

    def oneLoopCoupling(q, m1, m2):
        f = lambda x: x*(1-x)*np.log((m2**2+x*(1-x)*q**2)/(m1**2+x*(1-x)*q**2))
        f_value = quad_vec(f,0,1)[0]
        f2 = lambda x2: np.interp(x2,q,f_value)
        return f2

    q = np.linspace(0,0.2,1000)
    oneLoopCouplingFunc = [oneLoopCoupling(q, mE, mMuon), oneLoopCoupling(q, mE, mTau),\
                       oneLoopCoupling(q, mMuon, mTau)]
    return oneLoopCouplingFunc

oneLoopCouplingFunc = oneLoopCouplingFuncGen()

def dRdEU1KineticMxing(E_r,E_nu,Flux,modelPara,Nuc):
    N = Nuc.NumberOfNeutrons
    Z = Nuc.NumberOfProtons
    #Q_W = 1.0*N-(1-4.0*sinTheta_Wsq)*Z # weak nuclear hypercharge
    m_N_GeV = 0.93141941*(N+Z) # nucleus mass in GeV
    G_F_GeV = 1.16637e-5 # GeV**-2 ! Fermi constan in GeV
    sinTheta_Wsq = 0.2387e0 # sin^2(Theta_W) weinberg angle
    Qu = 1/2.-sinTheta_Wsq*4/3.
    Qd = -1/2.+sinTheta_Wsq*2/3.
    
    [gzplog,mzplog,chargeList]=modelPara
    gzp = 1.0*10**(gzplog)
    mzp = 1.0*10**(mzplog)
    
    #dRdE = np.zeros(shape=shape(E_r))
    FF = LabFuncs.FormFactorHelm(1.0e6*E_r,N+Z)**2.0
    ne = len(E_r)
    
    alphaEM = 1/137.036
    QwSM = (2*Qu+Qd)*Z + (2*Qd+Qu )*N
    #mE, mMuon, mTau = 0.511*1e-3, 105.66*1e-3, 1776.86*1e-3
    epDict = dict(zip([(1,-1,0),(1,0,-1),(0,1,-1)], oneLoopCouplingFunc))
    ep = epDict[tuple(chargeList)]
    U1toNSIFactor = gzp**2*np.sqrt(2)*alphaEM/(pi*G_F_GeV*(2*m_N_GeV*E_r+mzp**2))*ep(np.sqrt(2*m_N_GeV*E_r))
    Qw2 = [(QwSM+U1toNSIFactor*(charge*Z))**2 for charge in chargeList]

    if np.any(Flux[:,1]>0.):
        diff_sigma = np.array([(G_F_GeV**2.0*m_N_GeV /(pi))*(1.0-(m_N_GeV*Er)/(2.0*E_nu**2.0))*\
                1*(1.973e-14)**2.0 for Er in E_r])
        diff_sigma[diff_sigma<0.0] = 0.0
        dRdE = np.array([np.trapz(diff_sigma[i]*FF[i]*(np.sum([Flux[j]*Qw2[j][i] for j in range(3)],axis=0)),x=E_nu) for i in range(ne)])
    else:
        diff_sigma = np.array([(G_F_GeV**2.0*m_N_GeV /(pi))*(1.0-(m_N_GeV*Er)/(2.0*E_nu[0]**2.0))*\
                1*(1.973e-14)**2.0 for Er in E_r])
        diff_sigma[diff_sigma<0.0] = 0.0
        dRdE = diff_sigma*FF*(np.sum([Flux[j][0]*Qw2[j] for j in range(3)],axis=0))
    # if sol:
    #     fMod = LabFuncs.EarthSunDistanceMod(t)
    # else:
    #     fMod = 1.0
    fMod = 1.0

    # Convert into /ton/year/GeV
    seconds2year = 365.25*3600*24
    N_A = 6.02214e23 # Avocado's constant
    dRdE = fMod*dRdE*1000*seconds2year/(1.0*N+1.0*Z)*(N_A)*1000.0
    return dRdE

def dRdEU1FI(E_r,E_nu,Flux,modelPara,Nuc):
    N = Nuc.NumberOfNeutrons
    Z = Nuc.NumberOfProtons
    #Q_W = 1.0*N-(1-4.0*sinTheta_Wsq)*Z # weak nuclear hypercharge
    m_N_GeV = 0.93141941*(N+Z) # nucleus mass in GeV
    G_F_GeV = 1.16637e-5 # GeV**-2 ! Fermi constan in GeV
    sinTheta_Wsq = 0.2387e0 # sin^2(Theta_W) weinberg angle
    Qu = 1/2.-sinTheta_Wsq*4/3.
    Qd = -1/2.+sinTheta_Wsq*2/3.
    
    [gzplog,mzplog,qup,qdp,qLp]=modelPara
    gzp = 1.0*10**(gzplog)
    mzp = 1.0*10**(mzplog)
    
    #dRdE = np.zeros(shape=shape(E_r))
    FF = LabFuncs.FormFactorHelm(1.0e6*E_r,N+Z)**2.0
    ne = len(E_r)
    
    QwSM = (2*Qu+Qd)*Z + (2*Qd+Qu )*N
    U1toNSIFactor = gzp**2/(np.sqrt(2)*G_F_GeV*(2*m_N_GeV*E_r+mzp**2))
    Qw2 = (QwSM+qLp*U1toNSIFactor*((2*qup+qdp)*Z+(2*qdp+qup)*N))**2.0
    
    if Flux[1]>0.0:
        diff_sigma = np.array([(G_F_GeV**2.0*m_N_GeV /(pi))*(1.0-(m_N_GeV*Er)/(2.0*E_nu**2.0))*\
                1*(1.973e-14)**2.0 for Er in E_r])
        diff_sigma[diff_sigma<0.0] = 0.0
        dRdE = np.array([np.trapz(diff_sigma[i]*Flux*FF[i]*Qw2[i],x=E_nu) for i in range(ne)])
    else:
        diff_sigma = np.array([(G_F_GeV**2.0*m_N_GeV /(pi))*(1.0-(m_N_GeV*Er)/(2.0*E_nu[0]**2.0))*\
                1*(1.973e-14)**2.0 for Er in E_r])
        diff_sigma[diff_sigma<0.0] = 0.0
        dRdE = diff_sigma*Flux[0]*FF*Qw2
    # if sol:
    #     fMod = LabFuncs.EarthSunDistanceMod(t)
    # else:
    #     fMod = 1.0
    fMod = 1.0

    # Convert into /ton/year/GeV
    seconds2year = 365.25*3600*24
    N_A = 6.02214e23 # Avocado's constant
    dRdE = fMod*dRdE*1000*seconds2year/(1.0*N+1.0*Z)*(N_A)*1000.0
    return dRdE

class NuRates(object):
    
    def __init__(self, E_th, E_max, ne, Nuc):
        Names,Solar,E_nu_all,Flux_all,Flux_norm_default,errs,nuMaxEnergy = \
                GetNuFluxesCorrect(E_th,Nuc)
        print(np.transpose([np.linspace(0,len(Names)-1,len(Names),dtype=int), Names]))
        
        self._E_th, self._E_max, self._ne, self._Nuc, self._nuMaxEnergy, \
            self._E_nu_all, self._Flux_all, self._Solar = \
            E_th, E_max, ne, Nuc, nuMaxEnergy, E_nu_all, Flux_all, Solar
        self._Unc = errs
        self._Names = Names
    
    def vRatesSM(self, selList=[]):
        if len(selList)==0:
            Uncs = self._Unc
        else:
            Uncs = self._Unc[selList]
        return [self.vRatesFIGen(dRdEU1FI, [], selList=selList), Uncs]
    
    def vRatesFIGen(self, dRdE, modelPara, selList=[]):
        E_th, E_max, ne, Nuc, nuMaxEnergy, E_nu_all, Flux_all, Solar = self._E_th, self._E_max, self._ne, self._Nuc, self._nuMaxEnergy, self._E_nu_all, self._Flux_all, self._Solar
        
        # SM neutrino rates
        if len(modelPara) == 0:
            modelPara = np.zeros(5)
        if len(selList) == 0:
            selList = np.linspace(0,len(nuMaxEnergy)-1,len(nuMaxEnergy),dtype=int)
        nuMaxEnergy = nuMaxEnergy[selList]
        E_nu_all = E_nu_all[selList]
        Flux_all = Flux_all[selList]
        Solar = Solar[selList]
        
        #E_be = np.logspace(log10(E_th),log10(E_max),ne+1)
        E_be = np.linspace(E_th,E_max,ne+1)
        fineNum = 20
        E_r_max = MaxNuRecoilEnergiesCorrect(Nuc,nuMaxEnergy)
        n_nu = np.shape(E_nu_all)[0]
        R = np.zeros((n_nu,ne))
        for i in range(0,n_nu):
            # Efine = np.linspace(E_th, E_r_max[i],fineNum*ne)
            Efine = np.linspace(E_th, E_max,fineNum*ne)
            R_tot = np.trapz(dRdE(Efine,E_nu_all[i,:],Flux_all[i,:],modelPara,Nuc),Efine)
            dRE = dRdE(E_be,E_nu_all[i,:],Flux_all[i,:],modelPara,Nuc)
            dR = 0.5*(E_be[1:]-E_be[0:-1])*(dRE[1:]+dRE[0:-1])
            R[i,:] = R_tot/np.sum(dR)*dR
            
        return R
    
#     def vRatesFIAMGen(self, dRdE, modelPara, timeBin, selList=[]):
#         E_th, E_max, ne, Nuc, nuMaxEnergy, E_nu_all, Flux_all, Solar = self._E_th, self._E_max, self._ne, self._Nuc, self._nuMaxEnergy, self._E_nu_all, self._Flux_all, self._Solar
        
#         # SM neutrino rates
#         if len(modelPara) == 0:
#             modelPara = np.zeros(5)
#         if len(selList) == 0:
#             selList = np.linspace(0,len(nuMaxEnergy)-1,len(nuMaxEnergy),dtype=int)
#         nuMaxEnergy = nuMaxEnergy[selList]
#         E_nu_all = E_nu_all[selList]
#         Flux_all = Flux_all[selList]
#         Solar = Solar[selList]
        
#         #E_be = np.logspace(log10(E_th),log10(E_max),ne+1)
#         E_be = np.linspace(E_th,E_max,ne+1)
#         fineNum = 20
#         E_r_max = MaxNuRecoilEnergiesCorrect(Nuc,nuMaxEnergy)
#         n_nu = np.shape(E_nu_all)[0]
#         R = np.zeros((n_nu,ne))
#         for i in range(0,n_nu):
#             # Efine = np.linspace(E_th, E_r_max[i],fineNum*ne)
#             Efine = np.linspace(E_th, E_max,fineNum*ne)
#             R_tot = np.trapz(dRdE(Efine,E_nu_all[i,:],Flux_all[i,:],modelPara,Nuc),Efine)
#             dRE = dRdE(E_be,E_nu_all[i,:],Flux_all[i,:],modelPara,Nuc)
#             dR = 0.5*(E_be[1:]-E_be[0:-1])*(dRE[1:]+dRE[0:-1])
#             R[i,:] = R_tot/np.sum(dR)*dR
        
#         tList = np.linspace(0,365.25,timeBin+1)+JulianDay(1,1,2023,0)-28.6186
#         fModList = np.array([np.mean([solarMod(True, t) for t in np.linspace(tList[i],tList[i+1],20)]) for i in range(timeBin)])
#         R2 = np.array([np.array([R[i]*f for f in fModList]).flatten() if Solar[i] else np.array([R[i] for j in range(timeBin)]).flatten() for i in range(len(R))])
#         return R2/timeBin
    
    def vRatesFDGen(self, dRdE, modelPara, selList=[]):
        E_th, E_max, ne, Nuc, nuMaxEnergy, E_nu_all, Flux_all, Solar = self._E_th, self._E_max, self._ne, self._Nuc, self._nuMaxEnergy, self._E_nu_all, self._Flux_all, self._Solar
        
        if len(selList) == 0:
            selList = np.linspace(0,len(nuMaxEnergy)-1,len(nuMaxEnergy),dtype=int)
        nuMaxEnergy = nuMaxEnergy[selList]
        E_nu_all = E_nu_all[selList]
        Flux_all = Flux_all[selList]
        Solar = Solar[selList]
        
        names = self._Names[selList]
        [gzp, mzp, chargeList] = 10**(modelPara[0]), 10**(modelPara[1]), modelPara[2]
        U1Pro = propagation([gzp, mzp, chargeList])
        
        #E_be = np.logspace(log10(E_th),log10(E_max),ne+1)
        E_be = np.linspace(E_th,E_max,ne+1)
        fineNum = 20
        n_nu = np.shape(E_nu_all)[0]
        R = np.zeros((n_nu,ne))
        
        Efine = np.linspace(E_th, E_max,fineNum*ne)
        for i in range(0,n_nu):
            EvList, Flux = E_nu_all[i], Flux_all[i]
            if Solar[i]:
                fluxes = U1Pro.PeeEv(names[i], EvList, Flux)
                R_tot = np.trapz(dRdE(Efine,EvList,fluxes,modelPara,Nuc),Efine)
                dRE = dRdE(E_be,EvList,fluxes,modelPara,Nuc)
            else:
                # Ignore the atmosperic neutrino's oscillation
                #fluxes = np.zeros((3, len(Flux)))E_r,E_nu,Flux,modelPara,Nuc
                R_tot = np.trapz(dRdEU1FI(Efine,EvList,Flux,np.zeros(5),Nuc),Efine)
                dRE = dRdEU1FI(E_be,EvList,Flux,np.zeros(5),Nuc)
            dR = 0.5*(E_be[1:]-E_be[0:-1])*(dRE[1:]+dRE[0:-1])
            R[i,:] = R_tot/np.sum(dR)*dR
        return R
    
    def vRatesFIGenLow(self, dRdE, modelPara, neERFunc, selList=[]):
        E_th, E_max, ne, Nuc, nuMaxEnergy, E_nu_all, Flux_all, Solar = self._E_th, self._E_max, self._ne, self._Nuc, self._nuMaxEnergy, self._E_nu_all, self._Flux_all, self._Solar
        
        # SM neutrino rates
        if len(modelPara) == 0:
            modelPara = np.zeros(5)
        if len(selList) == 0:
            selList = np.linspace(0,len(nuMaxEnergy)-1,len(nuMaxEnergy),dtype=int)
        nuMaxEnergy = nuMaxEnergy[selList]
        E_nu_all = E_nu_all[selList]
        Flux_all = Flux_all[selList]
        Solar = Solar[selList]
        
        #E_be = np.logspace(log10(E_th),log10(E_max),ne+1)
        nBin = 200
        E_r_max = MaxNuRecoilEnergiesCorrect(Nuc,nuMaxEnergy)
        n_nu = np.shape(E_nu_all)[0]
        R = np.zeros((n_nu,10))
        
        Efine = np.logspace(np.log10(E_th),np.log10(E_max), nBin)
        neList = neERFunc(Efine)
        poissonMat = np.transpose([stats.poisson.pmf(np.linspace(1,10,10),mu) for mu in neList])
        #poissonMat = np.transpose(0.5*(poissonMatT[1:]+poissonMatT[:-1]))
        for i in range(0,n_nu):
            # Efine = np.linspace(E_th, E_r_max[i],fineNum*ne)
            dRE = dRdE(Efine,E_nu_all[i,:],Flux_all[i,:],modelPara,Nuc)
            R[i,:] = np.array([np.trapz(dRE*poissonMat[j],Efine) for j in range(10)])
        return R*1e-3 # in unit of kg*year
    
    def vRatesFDGenLow(self, dRdE, modelPara, neERFunc, selList=[]):
        E_th, E_max, ne, Nuc, nuMaxEnergy, E_nu_all, Flux_all, Solar = self._E_th, self._E_max, self._ne, self._Nuc, self._nuMaxEnergy, self._E_nu_all, self._Flux_all, self._Solar
        
        if len(selList) == 0:
            selList = np.linspace(0,len(nuMaxEnergy)-1,len(nuMaxEnergy),dtype=int)
        nuMaxEnergy = nuMaxEnergy[selList]
        E_nu_all = E_nu_all[selList]
        Flux_all = Flux_all[selList]
        Solar = Solar[selList]
        
        names = self._Names[selList]
        [gzp, mzp, chargeList] = 10**(modelPara[0]), 10**(modelPara[1]), modelPara[2]
        U1Pro = propagation([gzp, mzp, chargeList])  
        
        #E_be = np.logspace(log10(E_th),log10(E_max),ne+1)
        nBin = 200
        n_nu = np.shape(E_nu_all)[0]
        R = np.zeros((n_nu,10))
        
        Efine = np.logspace(np.log10(E_th),np.log10(E_max), nBin)
        neList = neERFunc(Efine)
        poissonMat = np.transpose([stats.poisson.pmf(np.linspace(1,10,10),mu) for mu in neList])
        
        for i in range(0,n_nu):
            EvList, Flux = E_nu_all[i], Flux_all[i]
            if Solar[i]:
                fluxes = U1Pro.PeeEv(names[i], EvList, Flux)
                dRE = dRdE(Efine,EvList,fluxes,modelPara,Nuc)
            else:
                # Ignore the atmosperic neutrino's oscillation
                dRE = dRdEU1FI(Efine,EvList,Flux,np.zeros(5),Nuc)
            R[i,:] = np.array([np.trapz(dRE*poissonMat[j],Efine) for j in range(10)])
        return R*1e-3 # in unit of kg*year
    
#     def vRatesFIAMGenLow(self, dRdE, modelPara, neERFunc, timeBin, selList=[]):
#         E_th, E_max, ne, Nuc, nuMaxEnergy, E_nu_all, Flux_all, Solar = self._E_th, self._E_max, self._ne, self._Nuc, self._nuMaxEnergy, self._E_nu_all, self._Flux_all, self._Solar
        
#         # SM neutrino rates
#         if len(modelPara) == 0:
#             modelPara = np.zeros(5)
#         if len(selList) == 0:
#             selList = np.linspace(0,len(nuMaxEnergy)-1,len(nuMaxEnergy),dtype=int)
#         nuMaxEnergy = nuMaxEnergy[selList]
#         E_nu_all = E_nu_all[selList]
#         Flux_all = Flux_all[selList]
#         Solar = Solar[selList]
        
#         #E_be = np.logspace(log10(E_th),log10(E_max),ne+1)
#         nBin = 200
#         E_r_max = MaxNuRecoilEnergiesCorrect(Nuc,nuMaxEnergy)
#         n_nu = np.shape(E_nu_all)[0]
#         R = np.zeros((n_nu,10))
        
#         Efine = np.logspace(np.log10(E_th),np.log10(E_max), nBin)
#         neList = neERFunc(Efine)
#         poissonMat = np.transpose([stats.poisson.pmf(np.linspace(1,10,10),mu) for mu in neList])
#         #poissonMat = np.transpose(0.5*(poissonMatT[1:]+poissonMatT[:-1]))
#         for i in range(0,n_nu):
#             # Efine = np.linspace(E_th, E_r_max[i],fineNum*ne)
#             dRE = dRdE(Efine,E_nu_all[i,:],Flux_all[i,:],modelPara,Nuc)
#             R[i,:] = np.array([np.trapz(dRE*poissonMat[j],Efine) for j in range(10)])
            
#         tList = np.linspace(0,365.25,timeBin+1)+JulianDay(1,1,2023,0)-28.6186
#         fModList = np.array([np.mean([solarMod(True, t) for t in np.linspace(tList[i],tList[i+1],20)]) for i in range(timeBin)])
#         R2 = np.array([np.array([R[i]*f for f in fModList]).flatten() if Solar[i] else np.array([R[i] for j in range(timeBin)]).flatten() for i in range(len(R))])
#         return (R2/timeBin)*1e-3 # in unit of kg*year
    
    
    def vDifferentialRatesSMCheck(self):
        Uncs = self._Unc
        E_th, E_max, ne, Nuc, nuMaxEnergy, E_nu_all, Flux_all, Solar = self._E_th, self._E_max, self._ne, self._Nuc, self._nuMaxEnergy, self._E_nu_all, self._Flux_all, self._Solar
        modelPara = np.zeros(5)
        selList = np.linspace(0,len(nuMaxEnergy)-1,len(nuMaxEnergy),dtype=int)
        nuMaxEnergy = nuMaxEnergy[selList]
        E_nu_all = E_nu_all[selList]
        Flux_all = Flux_all[selList]
        Solar = Solar[selList]
        
        #E_be = np.logspace(np.log10(E_th),np.log10(E_max),ne+1)
        fineNum = 20
        E_r_max = MaxNuRecoilEnergiesCorrect(Nuc,nuMaxEnergy)
        n_nu = np.shape(E_nu_all)[0]
        dR = np.zeros((n_nu,fineNum*ne))
        Efine = np.logspace(np.log10(E_th),np.log10(E_max),fineNum*ne)
        dR = np.array([dRdEU1FI(Efine,E_nu_all[i,:],Flux_all[i,:],modelPara,Nuc) for i in range(n_nu)])
        return [dR*1e-6, Efine*1e6] # in unit of /ton/year/keV
    
    def vDifferentialRatesFICheck(self, dRdE, modelPara, selList=[]):
        E_th, E_max, ne, Nuc, nuMaxEnergy, E_nu_all, Flux_all, Solar = self._E_th, self._E_max, self._ne, self._Nuc, self._nuMaxEnergy, self._E_nu_all, self._Flux_all, self._Solar
        
        # SM neutrino rates
        if len(modelPara) == 0:
            modelPara = np.zeros(5)
        if len(selList) == 0:
            selList = np.linspace(0,len(nuMaxEnergy)-1,len(nuMaxEnergy),dtype=int)
        nuMaxEnergy = nuMaxEnergy[selList]
        E_nu_all = E_nu_all[selList]
        Flux_all = Flux_all[selList]
        Solar = Solar[selList]
    
        #E_be = np.logspace(log10(E_th),log10(E_max),ne+1)
        E_be = np.linspace(E_th,E_max,ne+1)
        fineNum = 20
        n_nu = np.shape(E_nu_all)[0]
        
        #Efine = np.linspace(E_th, E_max,fineNum*ne)
        Efine = np.logspace(np.log10(E_th),np.log10(E_max),fineNum*ne)
        dR = np.array([dRdEU1FI(Efine,E_nu_all[i,:],Flux_all[i,:],modelPara,Nuc) for i in range(n_nu)])
        return [dR*1e-6, Efine*1e6]
    
    def vDifferentialRatesFDCheck(self, dRdE, modelPara, selList=[]):
        E_th, E_max, ne, Nuc, nuMaxEnergy, E_nu_all, Flux_all, Solar = self._E_th, self._E_max, self._ne, self._Nuc, self._nuMaxEnergy, self._E_nu_all, self._Flux_all, self._Solar
        
        if len(selList) == 0:
            selList = np.linspace(0,len(nuMaxEnergy)-1,len(nuMaxEnergy),dtype=int)
        nuMaxEnergy = nuMaxEnergy[selList]
        E_nu_all = E_nu_all[selList]
        Flux_all = Flux_all[selList]
        Solar = Solar[selList]
        
        names = self._Names[selList]
        [gzp, mzp, chargeList] = 10**(modelPara[0]), 10**(modelPara[1]), modelPara[2]
        U1Pro = propagation([gzp, mzp, chargeList], EvList=np.logspace(-6,np.log10(2*10.),500)*1.0e-3)
        
        #E_be = np.logspace(log10(E_th),log10(E_max),ne+1)
        E_be = np.linspace(E_th,E_max,ne+1)
        fineNum = 20
        n_nu = np.shape(E_nu_all)[0]
        
        #Efine = np.linspace(E_th, E_max,fineNum*ne)
        Efine = np.logspace(np.log10(E_th),np.log10(E_max),fineNum*ne)
        dR = np.zeros((n_nu,len(Efine)))
        
        for i in range(0,n_nu):
            EvList, Flux = E_nu_all[i], Flux_all[i]
            if Solar[i]:
                fluxes = U1Pro.PeeEv(names[i], EvList, Flux)
                dRE = dRdE(Efine,EvList,fluxes,modelPara,Nuc)
            else:
                # Ignore the atmosperic neutrino's oscillation
                #fluxes = np.zeros((3, len(Flux)))E_r,E_nu,Flux,modelPara,Nuc
                dRE = dRdEU1FI(Efine,EvList,Flux,np.zeros(5),Nuc)
            dR[i] = dRE
        return [dR*1e-6, Efine*1e6]

    def vDifferentialRatesSMCheck_ve(self, ZeffFunc, selList=[]):
        E_th, E_max, ne, Nuc, nuMaxEnergy, E_nu_all, Flux_all, Solar = self._E_th, self._E_max, self._ne, self._Nuc, self._nuMaxEnergy, self._E_nu_all, self._Flux_all, self._Solar
        
        if len(selList) == 0:
            selList = np.linspace(0,len(nuMaxEnergy)-1,len(nuMaxEnergy),dtype=int)
        nuMaxEnergy = nuMaxEnergy[selList]
        E_nu_all = E_nu_all[selList]
        Flux_all = Flux_all[selList]
        Solar = Solar[selList]
        
        names = self._Names[selList]
        modelPara = np.zeros(5)
        U1Pro = propagation([0, 1,[0,1,-1]], EvList=np.logspace(-6,4,1000)*1.0e-3)
        
        #E_be = np.logspace(log10(E_th),log10(E_max),ne+1)
        E_be = np.linspace(E_th,E_max,ne+1)
        fineNum = 20
        n_nu = np.shape(E_nu_all)[0]
        
        #Efine = np.linspace(E_th, E_max,fineNum*ne)
        Efine = np.logspace(np.log10(E_th),np.log10(E_max),fineNum*ne)
        dR = np.zeros((n_nu,len(Efine)))
        
        for i in range(0,n_nu):
            EvList, Flux = E_nu_all[i], Flux_all[i]
            if Solar[i]:
                fluxes = U1Pro.PeeEv(names[i], EvList, Flux)
                fluxes = np.array([fluxes[0],fluxes[1]+fluxes[2]])
                dRE = dRdEU1FI_ve(Efine,EvList,fluxes,modelPara,Nuc,ZeffFunc)
            else:
                # Ignore the atmosperic neutrino's oscillation
                #fluxes = np.zeros((3, len(Flux)))E_r,E_nu,Flux,modelPara,Nuc
                fluxes = np.array([Flux/3,Flux*2/3])
                dRE = dRdEU1FI_ve(Efine,EvList,fluxes,modelPara,Nuc,ZeffFunc)
            dR[i] = dRE
        return [dR*1e-6, Efine*1e6]

#     def vDifferentialRatesSMAMCheck(self):
#         Uncs = self._Unc
#         E_th, E_max, ne, Nuc, nuMaxEnergy, E_nu_all, Flux_all, Solar = self._E_th, self._E_max, self._ne, self._Nuc, self._nuMaxEnergy, self._E_nu_all, self._Flux_all, self._Solar
#         modelPara = np.zeros(5)
#         selList = np.linspace(0,len(nuMaxEnergy)-1,len(nuMaxEnergy),dtype=int)
#         nuMaxEnergy = nuMaxEnergy[selList]
#         E_nu_all = E_nu_all[selList]
#         Flux_all = Flux_all[selList]
#         Solar = Solar[selList]
        
#         #E_be = np.logspace(np.log10(E_th),np.log10(E_max),ne+1)
#         fineNum = 20
#         E_r_max = MaxNuRecoilEnergiesCorrect(Nuc,nuMaxEnergy)
#         n_nu = np.shape(E_nu_all)[0]
#         dR = np.zeros((n_nu,fineNum*ne))
#         Efine = np.logspace(np.log10(E_th),np.log10(E_max),fineNum*ne)
#         tList = np.linspace(0,800.,200)+JulianDay(1,1,2023,0)
#         dR = np.array([dRdEU1FI(Efine,E_nu_all[i,:],Flux_all[i,:],modelPara,Nuc) for i in range(n_nu)])
#         dRList = np.array([np.sum([dR[i]*solarMod(Solar[i], t) for i in range(n_nu)]) for t in tList])
        
#         return [dRList*1e-6, tList-JulianDay(1,1,2023,0), Efine*1e6] # in unit of /ton/year/keV
    
def solarMod(solar, t):
    if solar:
        fMod = LabFuncs.EarthSunDistanceMod(t)
        return fMod
    else:
        return 1.

def dRdEU1FI_ve(E_r,E_nu,Flux,modelPara,Nuc,ZeffFunc):
    N = Nuc.NumberOfNeutrons
    Z = Nuc.NumberOfProtons
    #Q_W = 1.0*N-(1-4.0*sinTheta_Wsq)*Z # weak nuclear hypercharge
    m_N_GeV = 0.93141941*(N+Z) # nucleus mass in GeV
    G_F_GeV = 1.16637e-5 # GeV**-2 ! Fermi constan in GeV
    sinTheta_Wsq = 0.2387e0 # sin^2(Theta_W) weinberg angle
    Qu = 1/2.-sinTheta_Wsq*4/3.
    Qd = -1/2.+sinTheta_Wsq*2/3.
    
    # [gzplog,mzplog,qup,qdp,qLp]=modelPara
    # gzp = 1.0*10**(gzplog)
    # mzp = 1.0*10**(mzplog)
    
    #dRdE = np.zeros(shape=shape(E_r))
    ne = len(E_r)
    m_e_GeV = 0.511e-3
    Zeff = ZeffFunc(E_r)
    
    gVe = 1/2+2*sinTheta_Wsq
    gAe = 1/2
    gVmu = -1/2+2*sinTheta_Wsq
    gAmu = -1/2
    
    E_r2 = E_r
    if np.all(Flux[:,1]>0.):
        # get the maximal recoil energy to avoid unnecessary computations
        EnuMax = E_nu[-1]
        EnuMinForEr = 0.5*(E_r+np.sqrt(E_r**2+2*E_r*m_e_GeV))
        E_r = E_r[EnuMinForEr<EnuMax]

        dRdE = np.zeros((2,len(E_r)))
        for i in range(len(E_r)):
            Er = E_r[i]
            select = E_nu>(0.5*(Er+np.sqrt(Er**2+2*Er*m_e_GeV)))
            E_nu2 = E_nu[select]
            #print([len(select),len(E_nu),len(Flux[0])])
            diff_sigma_e = (G_F_GeV**2.0*m_e_GeV /(2*pi))*((gVe+gAe)**2+(gVe-gAe)**2*(1-Er/E_nu2)**2-(gVe**2-gAe**2)*(m_e_GeV*Er/E_nu2**2))*(1.973e-14)**2.0
            dRdE[0,i] = Zeff[i]*np.trapz(diff_sigma_e*(Flux[0][select]),x=E_nu2)
            diff_sigma_mu = (G_F_GeV**2.0*m_e_GeV /(2*pi))*((gVmu+gAmu)**2+(gVmu-gAmu)**2*(1-Er/E_nu2)**2-(gVmu**2-gAmu**2)*(m_e_GeV*Er/E_nu2**2))*(1.973e-14)**2.
            dRdE[1,i] = Zeff[i]*np.trapz(diff_sigma_mu*(Flux[1][select]),x=E_nu2)
        # diff_sigma_mu = (G_F_GeV**2.0*m_e_GeV /(2*pi))*np.array([((gVmu+gAmu)**2+(gVmu-gAmu)**2*(1-Er/E_nu)**2-(gVmu**2-gAmu**2)*(m_e_GeV*Er/E_nu**2)) for Er in E_r])*(1.973e-14)**2.0
        # dRdE_e = Zeff*np.array([np.trapz(diff_sigma_e[i]*Flux[0],x=E_nu) for i in range(ne)])
        # dRdE_mu = Zeff*np.array([np.trapz(diff_sigma_mu[i]*Flux[1],x=E_nu) for i in range(ne)])
    else:
        EnuMax = E_nu[0]
        EnuMinForEr = 0.5*(E_r+np.sqrt(E_r**2+2*E_r*m_e_GeV))
        E_r = E_r[EnuMinForEr<EnuMax]
        Zeff = Zeff[EnuMinForEr<EnuMax]
        if len(E_r)<0:
            return np.zeros_like(E_r2)
        
        dRdE = np.zeros((2,len(E_r)))
        diff_sigma_e = (G_F_GeV**2.0*m_e_GeV /(2*pi))*((gVe+gAe)**2+(gVe-gAe)**2*(1-E_r/E_nu[0])**2-(gVe**2-gAe**2)*(m_e_GeV*E_r/E_nu[0]**2))*(1.973e-14)**2.0
        diff_sigma_mu = (G_F_GeV**2.0*m_e_GeV /(2*pi))*((gVmu+gAmu)**2+(gVmu-gAmu)**2*(1-E_r/E_nu[0])**2-(gVmu**2-gAmu**2)*(m_e_GeV*E_r/E_nu[0]**2))*(1.973e-14)**2.0
        
        dRdE[0] = Zeff*diff_sigma_e*Flux[0,0]
        dRdE[1] = Zeff*diff_sigma_mu*Flux[1,0]
    
    dRdE = np.sum(dRdE,axis=0)
    fMod = 1.0

    # Convert into /ton/year/GeV
    seconds2year = 365.25*3600*24
    N_A = 6.02214e23 # Avocado's constant
    dRdE = fMod*dRdE*1000*seconds2year/(1.0*N+1.0*Z)*(N_A)*1000.0
    return np.append(dRdE,np.zeros(len(E_r2)-len(dRdE)))