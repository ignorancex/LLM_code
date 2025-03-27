#!/usr/bin/env python
# -*- coding: utf-8 -*-


################################################## ##############################
#########################  Production in Fermion Portal #########################
######################### ######################### #############################

### Include all the various production channels for the fermion portal,
### with user-input choices for the masses of the dark sector particles
### L. Darme, T. You, S. Ellis 29/12/2019


############################  library import part #########################
import numpy as np
from scipy import optimize as opt
from scipy.interpolate import interp1d
import math

from scipy.integrate import quad
from scipy.optimize import minimize

import UsefulFunctions as uf
import Amplitudes as am

s=-1



################################################## ##############################################
#########################                                               #########################
#########################  Loading various experimental stuffs          #########################
#########################                                               #########################
######################### ######################### ##############################################

############## -------- Luminosity and number of PoT


# ----- Target material composition for the various epxeriments 

A_ship=184;Z_ship=74 # W target
A_lsnd=18;Z_lsnd=10 # Water target
A_miniboone=56;Z_miniboone=26 # Fe target
A_charm=64;Z_charm=29 # Cu target
A_nova=12;Z_nova=6 # C target
A_seaquest=56;Z_seaquest=26 # Fe target


#------ Planned proton on target and luminosity for various upcoming experiments


PoTship = 2*10**20; PoTlsnd = 0.92*10**23; PoTmboone =1.86*10**20
PoTsbnd = 6.6*10**20; PoTnova = 3.0*10**20; LumiHLLHC = 3.0*10**6
PoTseaq =  1.44*10**18
PoTseaq_2 =1.0*10**20 # Hypothetical Phase 2 of SeaQuest
PoTcharm = 2.4*10**18;



#------ The fraction of meson  pi0, eta, etap , rho, omega, Ds

### !!! The values will hve to be cross-checked, espeically using the 120 GeV one from the ring at Fermilab

charm_Nmeson = np.array((2.4, 0.26, 0.03, 0.29, 0.24,0.0008)) # Valid for CHARM, 1908.07525, check the other meson
ship_Nmeson = np.array((10, 1, 0.08, 1.1, 1,0.001)) # Valid for SHIP with W target - including every sub-product of the hardonic shower with CRMC / EPOS-LHC
lsnd_Nmeson= np.array((0.14, 0., 0.0, 0.,0.)) # Valid for LSND, 0.8 GeV beam
# FermiBoostMeson = np.array((0.99, 0.033, 0.0033, 0.05,0.046)) # Fermilab Booster (8.89 GeV) after collision with proton
miniboone_Nmeson = np.array((2.4, 0.08, 0.001, 0.1,0.1))
# Used by MiniBooNE and after that SBND,  Note the etaprime ratio should be confirmed by pythia
nova_Nmeson = np.array((0.99, 0.033, 0.0033, 0.033,0.033)) # Fermilab Ring (120 GeV) # Used by NoVa, this is actually normalised over 3.5 pion per PoT --> 1807.06501*);
seaquest_Nmeson = np.array((3.5, 0.4, 0.04, 0.45,0.45)) # Fermilab Ring (120 GeV) # Used by Seaquest;
# It then depends on whether or not one consider secondary meson production, depending on the expriment one needs to recast (e.g. a far away/with high energy threshold experiment will probably use only the primary pions)
lhc_Nmeson = np.array((4.3*10**12, 0.47*10**12, 0.05*10**12, 0.46*10**12, 0.45*10**12))#6.*10**9 or 0.95*10**9*)
# CS in pb based on Berlin \1810.01879 and EPSOS-LHC simulation as shown in 1708.09389, cross-checked with our own EPOS-LHC simulation
# For Dstar numbers Other sources like ALICE may be relevant, but it seems the actual meson distriubtion depends strongly on pt, see 1710.01933*)

#### For heavy mesons

# Used for SHIP        
xsecB = 3.6;xsecpn = 40*10**6
NBminus = PoTship*xsecB/xsecpn
NB0 = NBminus 
NKS0 = PoTship*0.232
NKm = PoTship*0.224
NKp = PoTship*0.331
ship_heavymeson = [NBminus,NB0,NKS0,NKm,NKm,NKp]
# For J/Psi and Upsilon
LHCMesonHeavy = np.array((5 *10**5, 2000))
# Using 1012.2815, 1903.09185 and ATLAS 2016-047  *)

############## -------- Masses and Decay rate for mesons
fPi = 0.1307;
MPi = 0.1349766; MEta = 0.547862; MEtap = 0.9577
MRho = 0.77526; MOmega = 0.78265
MDs = 2.00696; MD = 1.8648;MKs = 0.89581;MK = 0.49761

GamPi = 6.58*10**(-25)/(8.52*10**(-17))
GamEta =6.58*10**(-25)/(5.02*10**(-19))
GamEtap =6.58*10**(-25)/(3.32*10**(-21))
GamRho =6.58*10**(-25)/(4.45*10**(-24))
GamOmega =6.58*10**(-25)/(7.75*10**(-23))
BrDtoPiGamma = 0.38

#BrKtoPiGamma=0.38;*)

MJPsi = 3.097;
MCapitalUpsilon = 9.460; # 1S resonance for   Upsilon*)
MPhi = 1.019;
MBminus = 5.27931
MB0 = 5.36682 
MKminus = 0.493677 
Mpim = 0.1396
MK0 = 0.497648

GamJPsi = 6.58*10**(-25)/(7.09*10**(-21))
GamCapitalUpsilon =6.58*10**(-25)/(1.22*10**(-20))
GamPhi = 6.58*10**(-25)/(1.54*10**(-22))
GamBminus = 4.01645*10**(-13) #in GeV^-1 
GamB0 = 4.32895*10**(-13)
GamKL0 = 1.286*10**(-17) 
GamKS0 = 7.3467*10**(-15)
GamKplus = 5.315*10**(-17)


fBKeff = 0.32
fBpieff = 0.27
fKpieff = 1.0
fBsKeff = 0.23

aem=1/137.

################################################## ##############################################
#########################                                               #########################
#########################  Loading external production data             #########################
#########################                                               #########################
######################### ######################### ##############################################

######################### Meson decay for the DP case with del=0.1 -- Generated with BdNMC 1609.01770 

##### Production at FNAL main ring numi
Prod_Nova_DP,xi_Nova_DP = uf.LoadandSumData('ProdData/FNALRing/nova_prodDP.txt')

Prod_SeaQ_Phase2_DP,xi_SeaQ_DP = uf.LoadandSumData('ProdData/FNALRing/seaquest_ProdDP.txt')
#Phase 2 sea quest with 10^20
Prod_SeaQ_Phase1_DP=Prod_SeaQ_Phase2_DP*1.44/300 # we include including secondary mesons as in 1804.00661, leads to good agreement

##### Production at FNAL Booster
Prod_MBoone_DP,xi_MBoone_DP = uf.LoadandSumData('ProdData/FNALBooster/miniboone_prodDP.txt')
Prod_SBNDDPtmp,xi_SBND_DP = uf.LoadandSumData('ProdData/FNALBooster/miniboone_prodDP.txt')
Prod_SBND_DP=3.53*Prod_SBNDDPtmp # Moving from 1.86 10^20 PoT for MBooNE to 6.6 in SBND -- no account for the different beam dump material

#### LSND
Prod_LSND_DP,xi_LSND_DP = uf.LoadandSumData('ProdData/LSND/lsnd_prodDP.txt') # For a 15% splitting between chi1 and chi2

##### Production at SPS
Prod_SHIP_DP,xi_SHIP_DP = uf.LoadandSumData('ProdData/SPS/ship_prodDP.txt') # Based on 10 pi0 per PoT
Prod_CHARM_DP,xi_CHARM_DP = uf.LoadandSumData('ProdData/SPS/charm_prodDP.txt')


##### Meson production at LHC from FASER papers
xi_LHC1,CSProdLHC1 = uf.LoadData('ProdData/LHC/ProdDP_LHC_pi.txt')
xi_LHC2,CSProdLHC2 = uf.LoadData('ProdData/LHC/ProdDP_LHC_eta.txt')
xi_LHC3,CSProdLHC3 = uf.LoadData('ProdData/LHC/ProdDP_LHC_Brem.txt')
ProdLHC1=LumiHLLHC*1e-6*CSProdLHC1   # We then need to get the number of meson, normalised to epsilon =0.001 as usual
ProdLHC2=LumiHLLHC*1e-6*CSProdLHC2
ProdLHC3=LumiHLLHC*1e-6*CSProdLHC3
xi_LHC1=xi_LHC1/3;xi_LHC2=xi_LHC2/3;xi_LHC3=xi_LHC3/3 # x_LHC was written in term of MV=3*MX
xi_HLLHC_DP,Prod_HLLHC_DP=uf.SumInterpZero(xi_LHC1,ProdLHC1,xi_LHC2,ProdLHC2,xi_LHC3,ProdLHC3)


#########################  Direct  production  #########################

xi_Eff_DirectLHC,Prod_Eff_DirectLHC= uf.LoadData('ProdData/LHC/DirectEff_LHC.txt')

Lam_Eff_MonoJet,CS_Eff_MonoJet= uf.LoadData('ProdData/LHC/DirectEff_MonoJet.txt')


########################## Cosmic ray production ##########################

loglogPi0Eflux = np.loadtxt(open('Data/ProdData/CosmicRays/loglogpi0flux.csv',"rb"), delimiter=',')
logPi0Elist = np.array(list(zip(*loglogPi0Eflux))[0])
logPi0fluxlist = np.array(list(zip(*loglogPi0Eflux))[1])
interplogPi0Eflux = interp1d(logPi0Elist, logPi0fluxlist)

loglogEtaEflux = np.loadtxt(open('Data/ProdData/CosmicRays/loglogetaflux.csv',"rb"), delimiter=',')
logEtaElist = np.array(list(zip(*loglogEtaEflux))[0])
logEtafluxlist = np.array(list(zip(*loglogEtaEflux))[1])
interplogEtaEflux = interp1d(logEtaElist, logEtafluxlist)



#check flux interpolation
# '''
# import matplotlib 
# matplotlib.use('agg')
# import matplotlib.pyplot as plt
# print "plotting interpolated cosmic ray flux"
# #plt.plot(logEtaElist, logEtafluxlist, 'ro')
# #plt.plot(logEtaElist, [interplogEtaEflux(E) for E in logEtaElist], '-')
# #plt.savefig('tempplot.png')
# plt.plot(logPi0Elist, [interplogPi0Eflux(E) for E in logPi0Elist], '-')
# plt.xlabel(r'$\log{T_\chi}$ [GeV]', fontsize=16)
# plt.ylabel(r'$\log{d\Phi/dT_\chi}$ $[$GeV$^{-1} $cm$^{-2} $s$^{-1}]$', fontsize=16)
# plt.title(r'cosmic ray shower $\pi_0 \to \gamma \chi \chi$ flux', fontsize=16)
# plt.savefig('Output/pi0fluxplot.png')
# '''

################################################## ##############################################
#########################                                               #########################
#########################  Summarising everything by experiments        #########################
#########################                                               #########################
######################### ######################### ##############################################





################################################## ########################## ############ ###########
######################### Loading the production for DP from external database        #########################
######################### ######################### ########################## ############ ##########


def NProd_DP(exp): # Always produced at small delta ~ 0.1
    
    Noprod = np.zeros(np.size(xi_Eff_DirectLHC))
    
    if exp =="faser":
        xMes=xi_HLLHC_DP; NMes=Prod_HLLHC_DP
        xtmp,Np = uf.LoadDirectProdDP("LHC/DirectFDM_LHC",1,1,LumiHLLHC,True)
        x=xtmp
#         print(x,Np,xMes,NMes)
    elif exp == "mathusla":
        xMes=xi_HLLHC_DP; NMes=Prod_HLLHC_DP
        xtmp,Np = uf.LoadDirectProdDP("LHC/DirectFDM_LHC",1,1,LumiHLLHC,True)
        x=xtmp
    elif exp == "ship":
        xMes=xi_SHIP_DP; NMes=Prod_SHIP_DP
        x,Np = uf.LoadDirectProdDP("SPS/DirectFDM_SPS",A_ship,Z_ship,PoTship,False)
    elif exp == "charm":
        xMes=xi_CHARM_DP; NMes=Prod_CHARM_DP
        x,Np = uf.LoadDirectProdDP("SPS/DirectFDM_SPS",A_charm,Z_charm,PoTcharm,False)
    elif exp == "seaquest":
        xMes=xi_SeaQ_DP; NMes=Prod_SeaQ_Phase1_DP
        x,Np = uf.LoadDirectProdDP("FNALRing/DirectFDM_FNALRing",A_seaquest,Z_seaquest,PoTseaq,False)
    elif exp == "seaquest_phase2":
        xMes=xi_SeaQ_DP; NMes=Prod_SeaQ_Phase2_DP
        x,Np = uf.LoadDirectProdDP("FNALRing/DirectFDM_FNALRing",A_seaquest,Z_seaquest,PoTseaq_2,False)
    elif exp == "nova":
        xMes=xi_Nova_DP; NMes=Prod_Nova_DP
        x,Np = uf.LoadDirectProdDP("FNALRing/DirectFDM_FNALRing",A_nova,Z_nova,PoTnova,False)
    elif exp == "miniboone":
        xMes=xi_MBoone_DP; NMes=Prod_MBoone_DP
        x= xi_Eff_DirectLHC; Np =Noprod
    elif exp == "sbnd":
        xMes=xi_SBND_DP; NMes=Prod_SBND_DP
        x= xi_Eff_DirectLHC; Np =Noprod
    elif exp == "lsnd": # Splitting at 0.15 here
        xMes=xi_LSND_DP; NMes=Prod_LSND_DP
        x= xi_Eff_DirectLHC; Np =Noprod
    else:
        print("Experiment selected: ", exp, " is not currently implemented. Possible choices: faser, mathusla, ship, seaquest, seaquest_Phase2, nova, miniboone, sbnd, lsnd")
        NMes= np.array((0., 0., 0.0, 0.,0.))
        x= xi_Eff_DirectLHC; Np =Noprod

# Combining for the full production

    xProd, NProd =uf.SumInterp2((xMes,x),(NMes,Np))

#     print("For exp: ", exp, xProd, NProd )

    return xProd, NProd


################################################## ########################## ############ ###########
######################### Auxialliary functions                                #########################
######################### ######################### ########################## ############ ##########

## ----- Fill the effective couplings for light quarks from the dictionary
def Fillg(geff):
    return geff.get("gu11",0.),geff.get("gd11",0.),geff.get("gl11",0.),geff.get("gd11",0.)

## ----- Fill the effective couplings for heavy quarks from the dictionary
def FillHeavyg(geff):
    #gBmtoKm,gB0toK0,gB0toPi0,gBmtoPim,gK0toPi0,gKmtoPim 
    return geff.get("gd32",0.),geff.get("gd32",0.),geff.get("gd31",0.),geff.get("gd31",0.)\
        ,geff.get("gd21",0.),geff.get("gd21",0.)

def MakeDict(strs,nparray=np.zeros(20)):
    if np.size(nparray)<len(strs):
        nparray=np.append(nparray,np.zeros(len(strs)-np.size(nparray))) # We finish filling up the array with zeros if too short
    return {str: nparray[i]  for i, str in enumerate(strs)}


################################################## ########################## ############ ###########
######################### Creating the Number of produced dark sector  in EFT      #########################
######################### ######################### ########################## ############ ##########

### -- Return the number of produced dark sector particles from both meson decay (from internal routines) and direct production,
### using externally produced numbers
def LoadforExp(exp,mygs,OperatorType ="V",HeavyMeson=False):
    Noprod = np.zeros(np.size(xi_Eff_DirectLHC))
    LightMes=("Pi0","Eta","Etap","Rho","Omega")
    HeavyMes=("Bm","B0","Ks0","Km","Kp")
    NMes={}
   
    if exp =="faser":
        NMes.update(MakeDict(LightMes,lhc_Nmeson*LumiHLLHC))
        x,Np = uf.LoadDirectProd("LHC/DirectEff_Final_V_HLLHC",mygs,0,0,1,1,LumiHLLHC,True)
    elif exp == "mathusla":
        NMes.update(MakeDict(LightMes,lhc_Nmeson*LumiHLLHC))
        x,Np = uf.LoadDirectProd("LHC/DirectEff_Final_V_HLLHC",mygs,0,0,1,1,LumiHLLHC,True)
    elif exp == "ship":
        NMes.update(MakeDict(LightMes,ship_Nmeson*PoTship))
        x,Np = uf.LoadDirectProd("SPS/DirectEff_Final_V_SPS",mygs,0.56,0.44,A_ship,Z_ship,PoTship,False)
        if HeavyMeson: 
            NMes.update(MakeDict(HeavyMes,ship_heavymeson))       
            x = xi_Eff_DirectLHC; Np = Noprod
    elif exp == "charm":
        NMes.update(MakeDict(LightMes,charm_Nmeson*PoTcharm))
        x,Np = uf.LoadDirectProd("SPS/DirectEff_Final_V_SPS",mygs,0.56,0.44,A_charm,Z_charm,PoTcharm,False)
    elif exp == "seaquest":
        NMes.update(MakeDict(LightMes,seaquest_Nmeson*PoTseaq))
#         x= xi_Eff_DirectFNALRing; Np =NoE_Eff_DirectSeaQ*0.014#/10.
        x,Np = uf.LoadDirectProd("FNALRing/DirectEff_Final_V_FNALRing",mygs,0.68,0.36,A_seaquest,Z_seaquest,PoTseaq,False)
    elif exp == "seaquest_phase2":
        NMes.update(MakeDict(LightMes,seaquest_Nmeson*PoTseaq_2))
        x,Np = uf.LoadDirectProd("FNALRing/DirectEff_Final_V_FNALRing",mygs,0.68,0.36,A_seaquest,Z_seaquest,PoTseaq_2,False)
    elif exp == "nova":
        NMes.update(MakeDict(LightMes,nova_Nmeson*PoTnova))
        x,Np = uf.LoadDirectProd("FNALRing/DirectEff_Final_V_FNALRing",mygs,0.68,0.36,A_nova,Z_nova,PoTnova,False)
    elif exp == "miniboone":
        NMes.update(MakeDict(LightMes,miniboone_Nmeson*PoTmboone))
        x= xi_Eff_DirectLHC; Np =Noprod
    elif exp == "sbnd":
        NMes.update(MakeDict(LightMes,miniboone_Nmeson*PoTsbnd))
        x= xi_Eff_DirectLHC; Np =Noprod
    elif exp == "lsnd":
        NMes.update(MakeDict(LightMes,lsnd_Nmeson*PoTlsnd))
        x= xi_Eff_DirectLHC; Np =Noprod
    else:
        print("Experiment selected: ", exp, " is not currently implemented. Possible choices: faser, mathusla, ship, seaquest, seaquest_Phase2, nova, miniboone, sbnd, lsnd")
        NMes= np.array((0., 0., 0.0, 0.,0.))
        x= xi_Eff_DirectLHC; Np =Noprod

    
    return NMes, x,Np

### -- Return the number of produced dark sector particles from meson decay by calculating the decay width for a given set of effective couplings 
def NProdMesDecay(Del,NMes,mygs,OperatorType ="V",HeavyMeson=False): # The effective scale is not included, normalised to 1 TeV

  
    gu,gd,ge,gs=Fillg(mygs) # Flavor-blind couplings here
## ------- Effective coupling constant
    #  Vector meson, 1503.05534 *)
    fuRho = 0.2215; fdRho = 0.2097;
    fuOmega = 0.1917; fdOmega = 0.201; fPhi = 0.233;  ## Caution, often sqrt(2) differences
    fOmegaeff = (gu* fuOmega + gd*fdOmega)/(np.sqrt(2.));
    fRhoeff = (gu*fuRho - gd*fdRho)/(np.sqrt(2.));
    fPi0Aeff = 2*gu + gd;
    fEtaAeff = np.sqrt(2/3)*(0.55*(2*gu - gd + 2*gs) + 0.35*(2*gu - gd - gs))
    fEtapAeff = np.sqrt(2/3)*(-0.1*(2*gu - gd + 2*gs) + 0.85*(2*gu - gd - gs))
    fPieff = fPi*((gu - gd)/np.sqrt(2))
    fEtaeff =  fPi*(0.5*(gu + gd - 2*gs) + 0.1*(gu + gd + gs))
    fEtapeff = fPi*(-0.2*(gu + gd - 2*gs) + 0.68*(gu + gd + gs))
    fRhoAeff = 0.01*(2*gu - gd)  ## Correction term for our decay, 0.01 contains the effect of the loop induced coupling  e/4/Pi^4*Arho, Arho = 1.2 *);
    fOmegaAeff =   0.01*(2*gu + gd);#
    
    
    NPi0=NMes.get("Pi0",0.);NEta=NMes.get("Eta",0.);NEtap=NMes.get("Etap",0.);
    NRho=NMes.get("Rho",0.);NOmega=NMes.get("Omega",0.)

    xmin=0.001/(2.+Del);xmax= MBminus/(2.+Del) # Up to the highest possible decay
#     xmin=0.001/(2+Del);xmax= MEtap/(1.99999+Del)
    xi = uf.log_sample(xmin,xmax,500)

    if OperatorType == "AV":
        Ndarkpi0=am.GamAV_MestoXX(xi,xi*(1+Del),MPi,fPieff,1000.)/GamPi*NPi0
        Ndarketa=am.GamAV_MestoXX(xi,xi*(1+Del),MEta,fEtaeff,1000.)/GamEta*NEta
        Ndarketap=am.GamAV_MestoXX(xi,xi*(1+Del),MEtap,fEtapeff, 1000.)/GamEtap*NEtap
        NDarkMes=Ndarkpi0+Ndarketa+Ndarketap
    elif OperatorType == "V":
        Ndarkpi0=am.GamV_MestoXXgam(xi,xi*(1+Del),MPi,fPi0Aeff,1000.)/GamPi*NPi0
        Ndarketa=am.GamV_MestoXXgam(xi,xi*(1+Del),MEta,fEtaAeff,1000.)/GamEta*NEta
        Ndarketap=am.GamV_MestoXXgam(xi,xi*(1+Del),MEtap,fEtapAeff, 1000.)/GamEtap*NEtap
        Ndarkrho=am.GamV_MestoXX(xi,xi*(1+Del),MRho,fRhoeff,1000.)/GamRho*NRho
        Ndarkomega=am.GamV_MestoXX(xi,xi*(1+Del),MOmega,fOmegaeff, 1000.)/GamOmega*NOmega
        NDarkMes=Ndarkpi0+Ndarketa+Ndarketap+Ndarkrho+Ndarkomega
    else:
        NDarkMes=xi*0.


    if HeavyMeson:
#         print("NMes in the heavy meson case : ",NMes , gMesDecay)
        NHeavymes= NProdHeavyMesDecay(xi, Del, NMes, mygs, OperatorType)
        NDarkMes=NDarkMes+NHeavymes
 

    return xi,NDarkMes


##### -- Same as before, but used heavy flavoured meson decay instead of light unflavoured meson
def NProdHeavyMesDecay(xi,Del, NMes, geff, OperatorType="V"): #effective scale Lambda = 1 TeV #multiply NMes times branching ratio of meson into chichi 
    
    gBmtoKm,gB0toK0,gB0toPi0,gBmtoPim,gK0toPi0,gKmtoPim =FillHeavyg(geff)
    NBm = NMes.get("Bm",0.)
    NB0 = NMes.get("B0",0.)
    NKS0 = NMes.get("Ks0",0.)
    NKm = NMes.get("Km",0.)
    NKp = NMes.get("Kp",0.)


    
    #table 1 of Bjorkeroth et al 1806.00660
    fBtoKeff = 0.32*gBmtoKm
#     fBstoKeff = 0.23
    fBtopieff = 0.27*gB0toPi0
#     fDstoKeff = 0.68
#     fDtoKeff = 0.78
#     fDtopieff = 0.74
    fKtopieff = 1.0*gK0toPi0

#     xi = uf.log_sample(xmin,xmax,500)
     #number of produced dark fermions = number of produced mesons * branching ratio into dark fermions
    if OperatorType == "V":
        GamBmtoKm = am.GamV_MestoXXmes(xi, xi*(1.+Del), MBminus, MKminus, fBtoKeff, Lam=1000.)
        NdarkBmtoKm = gBmtoKm*NBm*GamBmtoKm/(GamBminus+GamBmtoKm)

        GamB0toK0 = am.GamV_MestoXXmes(xi, xi*(1.+Del), MB0, MK, fBtoKeff, Lam=1000.)
        NdarkB0toK0 = gB0toK0*NB0*GamB0toK0/(GamB0+GamB0toK0)
        
        GamB0toPi0 = am.GamV_MestoXXmes(xi, xi*(1.+Del), MB0, MPi, fBtopieff, Lam=1000.)
        NdarkB0toPi0 = gB0toPi0*NB0*GamB0toPi0/(GamB0+GamB0toPi0)

        GamBmtoPim = am.GamV_MestoXXmes(xi, xi*(1.+Del), MBminus, Mpim, fBtopieff, Lam=1000.)
        NdarkBmtoPim = gBmtoPim*NBm*GamBmtoPim/(GamBminus+GamBmtoPim)

        GamK0toPi0 = am.GamV_MestoXXmes(xi, xi*(1.+Del), MK0, MPi, fKtopieff, Lam=1000.)
        NdarkK0toPi0 = gK0toPi0*NKS0*GamK0toPi0/(GamKS0+GamK0toPi0)

        GamKmtoPim = am.GamV_MestoXXmes(xi, xi*(1.+Del), MKminus, Mpim, fKtopieff, Lam=1000.)
        NdarkKmtoPim = gKmtoPim*NKm*GamKmtoPim/(GamKplus+GamKmtoPim)           
        
        NDarkMes = NdarkBmtoKm + NdarkB0toK0 + NdarkB0toPi0 + NdarkBmtoPim + NdarkK0toPi0 + NdarkKmtoPim
        
    else: 
        NDarkMes = [0 for x in xi]

    return NDarkMes



def NProd(Del,exp,mygs,OperatorType ="V",HeavyMeson=False):

    # Number of mesons from external datasets   
    NMes, x1Prod_Direct, NProd_Direct = LoadforExp(exp,mygs,OperatorType,HeavyMeson) # Loading all experimental details and external productions
    
    # Creating the meson decay production for the precise M1 and M2 values
    xProd_Mes,NProd_Mes=NProdMesDecay(Del,NMes,mygs,OperatorType,HeavyMeson)
    xProd = xProd_Mes
    NProd = NProd_Mes

    if not HeavyMeson:
        # Matchting the direct production done at M_1 = M_2 to the more same invariant mass
        M1tildeToM1=( 2+Del)/(2+0)
        xProd_Direct=x1Prod_Direct/M1tildeToM1 # Rescaling the direct production to the effective couplings used
        # Combining for the full Production
        xProd, NProd =uf.SumInterp2((xProd_Mes,xProd_Direct),(NProd_Mes,NProd_Direct))
          
    return xProd, NProd





################################################## ########################## ############ ###########
#########################  Give the production from cosmic rays              #########################
######################### ######################### ########################## ############ ##########

### Give the number of dark sector particles produced from cosmic rays -- NOT YET USED FOR PAPER, MAY NEED CROSS-CHECKS
#Note for detection we must have Mx2+2*Melectron < Mx1 for chi1 to decay to chi2 through four-fermion interaction.
#For production we consider BR(meson -> chi1+chi1).
def CRDMfluxintegrand(Eflux, Mx1, Mx2, Lam,Del, mygs, OperatorType="V", exp="t2k"):
    gu,gd,ge,gs=Fillg(mygs)
    PiF = 0 
    if np.abs(Mx1)+np.abs(Mx2) < MPi: 
        fPieff = fPi*((gu - gd)/np.sqrt(2))
        if OperatorType == "V": 
            BRpitochichi = am.GamV_MestoXXgam(np.array([Mx1]), np.array([Mx2]), MPi, fPieff, Lam)[0]/GamPi
        elif OperatorType == "AV":
            BRpitochichi = am.GamAV_MestoXX(np.array([Mx1]), np.array([Mx2]), MPi, fPieff, Lam)[0]/GamPi
        PiF = 10**interplogPi0Eflux(math.log10(Eflux))*BRpitochichi
    EtaF = 0
    if np.abs(Mx1)+np.abs(Mx2) < MEta: 
        fEtaeff = fPi*(0.5*(gu + gd - 2*gs) + 0.1*(gu + gd + gs))
        if OperatorType == "V": 
            BRetatochichi = am.GamV_MestoXXgam(np.array([Mx1]), np.array([Mx2]), MEta, fEtaeff, Lam)[0]/GamEta
        elif OperatorType == "AV":
            BRetatochichi = am.GamAV_MestoXX(np.array([Mx1]), np.array([Mx2]), MEta, fEtaeff, Lam)[0]/GamEta
        EtaF = 10**interplogEtaEflux(math.log10(Eflux))*BRetatochichi    
    
    return PiF,EtaF




################################################## ########################## ############ ###########
######################### Invisible decay ratios             #########################
######################### ######################### ########################## ############ ##########



#helper function for limLambdaInvBR(), to minimize
def BRminfunc(varLam, Mmes1, Mmes2, Mchi1, Mchi2, GamMes1, invBRlim, feff=1.):
    Gam = am.GamV_MestoXXmes(np.array([Mchi1]), np.array([Mchi2]), Mmes1, Mmes2, feff, Lam=varLam)
    return ((Gam/(Gam+GamMes1) - invBRlim)/invBRlim)**2 #divide by invBRlim as a normalisation for minimizer to work

def BRminfunc2body(varLam, Mmes1, Mchi1, Mchi2, GamMes1, invBRlim, feff=1.,optype="AV"):
    if optype == "V":
        Gam = am.GamV_MestoXX(np.array([Mchi1]), np.array([Mchi2]), Mmes1, feff, Lam=varLam)
    elif optype == "AV":
        Gam = am.GamAV_MestoXX(np.array([Mchi1]), np.array([Mchi2]), Mmes1, feff, Lam=varLam)
    else:
        print("Operator ", optype, " is not supported for meson decay")
    return ((Gam/(Gam+GamMes1) - invBRlim)/invBRlim)**2 #divide by invBRlim as a normalisation for minimizer to work


#choose from preset list of invisible decay bounds 
def getLimLambdaFromInvBR(Mchi1, Mchi2, expdecay, geff,optype):
   
    gBmtoKm,gB0toK0,gB0toPi0,gBmtoPim,gK0toPi0,gKmtoPim =FillHeavyg(geff)
    gu,gd,ge,gs=Fillg(geff) # Flavor-blind couplings here
#     gBmtoKm=geff[0]; gB0toK0=geff[1]; gB0toPi0=geff[2]; gBmtoPim=geff[3]; gK0toPi0=geff[4]; gKmtoPim=geff[5];

    #table 1 of Bjorkeroth et al 1806.00660
    fBtoKeff = 0.32*gBmtoKm
    fBtopieff = 0.27*gB0toPi0
    fKtopieff = 1.0*gK0toPi0
    fJPsi = 0.420*geff.get("gu22",0.)/(2/3.) # From 1710.00117
    fUpsilon = 0.650*geff.get("gd33",0.)/(1/3.)   
    fPieff = fPi*((gu - gd)/np.sqrt(2))
#     print("Coupling dictionnary :",fUpsilon, geff)
    if (optype == "V") and (expdecay == "na62_pi0toinvisible"):
#         print("WARNING: Selected operator V cannot lead to invisible pi0 decay")
        return 0.
    if (optype == "AV") and (not expdecay == "na62_pi0toinvisible"):
#         print("WARNING: Selected operator AV cannot lead to ", expdecay)  
        return 0
    if expdecay == "babar_BmtoKmnunu":
        return limLambdaInvBR(MBminus, MKminus, Mchi1, Mchi2, GamBminus, 1.3*10**(-5), feff=fBtoKeff)
    elif expdecay == "belle2_BmtoKmnunu":
        return limLambdaInvBR(MBminus, MKminus, Mchi1, Mchi2, GamBminus, 1.5*10**(-6), feff=fBtoKeff)
    elif expdecay == "belle_B0toPi0nunu":
        return limLambdaInvBR(MB0, MPi, Mchi1, Mchi2, GamB0, 0.9*10**(-5), feff=fBtopieff)
    elif expdecay == "belle_B0toK0nunu":
        return limLambdaInvBR(MB0, MK0, Mchi1, Mchi2, GamB0,  1.3*10**(-5), feff=fBtoKeff)
    elif expdecay == "babar_BmtoPimnunu":
        return limLambdaInvBR(MBminus, Mpim, Mchi1, Mchi2, GamBminus, 1.0*10**(-4), feff=fBtopieff)
    elif expdecay == "e391a_KL0toPi0nunu":
        return limLambdaInvBR(MK0, MPi, Mchi1, Mchi2, GamKL0, 2.6*10**(-8), feff=fKtopieff)
    elif expdecay == "na62_KL0toPi0nunu":
        return limLambdaInvBR(MK0, MPi, Mchi1, Mchi2, GamKL0, 0.46*10**(-10), feff=fKtopieff)
    elif expdecay == "e949_KptoPipa":
        #note that this is for axion search, divide limit by 2 
        return limLambdaInvBR(MKminus, Mpim, Mchi1, Mchi2, GamKplus, 0.5*0.73*10**(-10), feff=fKtopieff)
    elif expdecay == "na62_KptoPipa":
        #note that this is for axion search, divide limit by 2 
        return limLambdaInvBR(MKminus, Mpim, Mchi1, Mchi2, GamKplus, 0.5*0.01*10**(-10), feff=fKtopieff)
    elif expdecay == "na62_pi0toinvisible":
        return limLambdaInvBR2body(MPi, Mchi1, Mchi2, GamPi, 4.4*10**(-9), fPieff,optype)
    elif expdecay == "bes_JPsitoinvisible":
        return limLambdaInvBR2body(MJPsi, Mchi1, Mchi2, GamJPsi, 7.2*10**(-4), fJPsi,optype)
    elif expdecay == "babar_Upsilontoinvisible":
        return limLambdaInvBR2body(MCapitalUpsilon, Mchi1, Mchi2, GamCapitalUpsilon, 3.0*10**(-4), fUpsilon,optype)
    else:
        print("WARNING: experiment ", expdecay ," for invisible decay not found!")
        return 0.

def limLambdaInvBR(Mmes1, Mmes2, Mchi1, Mchi2, GamMes1, invBRlim, feff=1.): 
    if (Mchi1+Mchi2) > (Mmes1-Mmes2): 
        return 0. 
    if Mmes2 > Mmes1: 
        print("WARNING: Meson1 mass must be larger than meson2!")
        return 0.
    if feff==0.:
        print("WARNING: Zero effective decay constant for this meson and these effective couplings" )
        return 0.;
    valini=1000.;
    result = minimize(BRminfunc, valini, args=(Mmes1, Mmes2, Mchi1, Mchi2, GamMes1, invBRlim, feff)) 
    if np.abs(result.x[0]) == valini: # Try again for better stability
         result = minimize(BRminfunc, 100., args=(Mmes1, Mmes2, Mchi1, Mchi2, GamMes1, invBRlim, feff))  
 
    return np.abs(result.x[0])

def limLambdaInvBR2body(Mmes1, Mchi1, Mchi2, GamMes1, invBRlim2body, feff=1.,optype="V"): 
    if (Mchi1+Mchi2) > (Mmes1): 
        return 0. 
    if feff==0.:
        print("WARNING: Zero effective decay constant for this meson and these effective couplings" )
        return 0.;
    valini =20.
    result = minimize(BRminfunc2body, valini, args=(Mmes1, Mchi1, Mchi2, GamMes1, invBRlim2body, feff,optype),tol=0.01) 
    if np.abs(result.x[0]) == valini: # Try again for better stability
         result = minimize(BRminfunc2body, 10., args=(Mmes1, Mchi1, Mchi2, GamMes1, invBRlim2body, feff,optype),tol=0.01) 
    
    return np.abs(result.x[0])*(not np.abs(result.x[0]) ==20.) # We kill the point which did not converged

  

    
