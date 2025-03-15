#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################## ##############################
#########################  Detection module  #########################
######################### ######################### #############################

###  Contains all the functions necessary to find the detection probability, including the geometries of relevant experiment
###  and the main recasting functions for the various signatues (scattering, decay, monoX, invisible meson decay, etc ...
###  The final functions are typically written as Fast****
###  L. Darme, S. Ellis, T. You, 29/12/2019


############

import math
import numpy as np

from scipy.optimize import minimize_scalar
from scipy import optimize as opt
from scipy.special import lambertw

import Production as br
import UsefulFunctions as uf
import Amplitudes as am

from scipy.integrate import quad
from scipy.optimize import minimize

################################################## ####################################
#########################                           #########################
#########################  Auxilliary functions     #########################
#########################                           #########################
######################### ######################### #########################

me=0.0005109989
mmuon=0.1056584

## ----- Fill the effective couplings for light quarks from the dictionary

def Fillg(geff):
    return geff.get("gu11",0.),geff.get("gd11",0.),geff.get("gl11",0.),geff.get("gd22",0.)

## ----- Used to find the lower limit in Lambda due to the Chi2 decaying before reaching the detector
def FindShortLam(A,B,myc):
    newA=A # Could be used to change a bit the boost factor for testing purposes
    newB=B
    myX= myc/newA/np.power(newB,1./myc)
#     res=(myc/np.power(newA,1)*( np.log(myX)+ np.log(np.log(myX)) ))
    # Approximation no needed -> use directly the lambertw function in scipy
    res2 = -myc/np.power(newA,1)*lambertw(-1/myX,-1)
#     print("Comparing approx to direct ",res/np.real(res2))

    return np.power(np.real(res2),-1/4.)

## -----  Simple decay probability evaluation for a given c *tau * gamma
def ProbDecay(D,L,ctaugamma):
    res= np.exp(-D/ctaugamma)*(1-np.exp(-L/ctaugamma))
    return res

# ---- Used to include the Lambda dependence of the cross-section due to the limitation of the EFT approach at LHC
# ---- it assumes that the CS has been parametrised as CS = acs * Lambda^bcs in pb
def ReduceLimLHC(LimOld,acs = 0.0009,bcs=1.14,CSfull=4.,geff=1):
    redpt=(LimOld<1700)
    LimOld[redpt] = np.power(LimOld[redpt]*np.power(acs/CSfull,1/8.),1/(1-bcs/8.))
    return LimOld

# ---- Gives back an estimate of the average boost factor depending on the source beam and for various invariant mass for chi1 chi2 
##     corresponding to different dominant production channels
def BoostFact(Mx,Del,beamtype):
    FactInvMass=(1+1/(Del+1)) # Mx is actually Mx2 mass of the heavy state
    Ep=0
    mpith=134.9766/FactInvMass/1000;metath=547.862/FactInvMass/1000;
    if (beamtype=="SPS"): # Data from BdNMC meson+brem production and taking the average
        
        # Using the ratio of production from CHARM
#         xMes=xi_CHARM_DP; NMes=Prod_CHARM_DP
#         x,Np = uf.LoadDirectProdDP("SPS/DirectFDM_SPS",A_charm,Z_charm,PoTcharm,False)
#         xiratio,ratio,notrelevant= uf.GetRatio(Np,x1,CS2,x2,lim,xlim):
        
        Ep=np.sqrt(np.power(Mx,2)+np.power((Mx < mpith)*11 + (Mx < metath)*(Mx > mpith)*(17.) +  (Mx > metath)*np.sqrt(17.*17-np.power(Mx*FactInvMass/2,2)),2))
    elif (beamtype=="FnalRing"):
        Ep=np.sqrt(np.power(Mx,2)+np.power((Mx < mpith)*7 + (Mx < metath)*(Mx > mpith)*8.5 +  (Mx > metath)*np.sqrt(8.5*8.5-np.power(Mx*FactInvMass/2,2)),2))
    elif (beamtype=="FnalBooster"):
        Ep=np.sqrt(np.power(Mx,2)+np.power((Mx < mpith)*0.7 + (Mx < metath)*(Mx > mpith)*1 +  (Mx > metath)*np.sqrt(1-np.power(Mx*FactInvMass/2,2)),2))
    elif (beamtype=="LSND"):
        Ep=np.sqrt(np.power(Mx,2)+np.power(0.12,2))
    elif (beamtype=="LHC"):
        Ep=np.sqrt(np.power(Mx,2)+1000**2) # From 1816.07396 and 1810.01879
    else:
        print("Bad beam line choice, possibilities: SPS, FnalRing, FnalBooster, LHC")

    Boost=Ep/Mx

    return Boost

# ---- Gives back the geometrical parameters of various experiments, as well as the corresponding beam lines.
def GeomForExp(exp):
    if exp =="faser":
        L=10;D=480;beamtype="LHC"
    elif exp == "mathusla":
        L=35;D=100;beamtype="LHC"
    elif exp == "ship":
        L=65;D=60;beamtype="SPS"
    elif exp == "charm":
        L=35;D=480;beamtype="SPS"
    elif exp == "seaquest":
        L=5;D=5;beamtype="FnalRing"
    elif exp == "seaquest_phase2":
        L=5;D=5;beamtype="FnalRing"
    elif exp == "nova":
        L=14.3;D=990;beamtype="FnalRing"
    elif exp == "miniboone":
        L=12;D=491;beamtype="FnalBooster"
    elif exp == "sbnd":
        L=5;D=110;beamtype="FnalBooster"
    elif exp == "lsnd":
        L=8.4;D=34;beamtype="LSND"
    else:
        print("Experiment selected: ", exp, " is not currently implemented. Possible choices: faser, mathusla, ship, seaquest, seaquest_phase2, nova, miniboone, sbnd, lsnd")
        L= 0;D=0;beamtype="NotDefined"
    return D,L,beamtype



################################################## ####################################
#########################                           #########################
######################### Scattering detection               #########################
#########################                           #########################
######################### ################################## #########################



def FastScatLimit(exp,x_in,Lim_in, Del_in,Del,geff,optype="V"):
        ##### Assuming the splitting to be irrelevant --> upscattering easy to get using the beam energy 
    gu,gd,ge,gs=Fillg(geff)
    
    if Del>0.5: print("Warning: recasting of scattering limits only implemented for small or zero splitting")
    M2tildeToM1=( 1+1/(1+Del))/(2+Del_in)
    xProd_DPtmp, NProd_DP= br.NProd_DP(exp)    
    xProd_DP=xProd_DPtmp/1.0 # Switching to zero splitting to avoid problems at the resonance
    mymin=np.min(xProd_DP)/M2tildeToM1; # Sending M1 to M2tilde
    mymax=np.max(xProd_DP)/M2tildeToM1;
    xi = uf.log_sample(mymin,mymax,200)
    Lam1TeV = np.full(np.size(xi),1000)
    xProd_new, Prod_new= br.NProd(Del,exp,geff,optype)
    Nnew = np.interp(xi, xProd_new*(1+Del), Prod_new)
    
    xProd, NProd= br.NProd(Del,exp,geff,optype)    
    
    xi,ratio,LimDP = uf.GetRatio(NProd,xProd,NProd_DP,xProd_DP,Lim_in,x_in)
    
    gscat=(np.abs(gu)+np.abs(gd))   
    EffLim =0.013*np.sqrt(xi)/np.sqrt(LimDP)*np.power(ratio,1/8.)*1000*np.sqrt(gscat)

    return xi, EffLim


################################################## ####################################
#########################                           #########################
######################### Invisible meson decay  #########################
#########################                           #########################
######################### ################################## #########################



def FastPi0InvLimit(xi,Lim_in,Delini,Del, geff,optype="V"):
    gu,gd,ge,gs=Fillg(geff)
    M2tildeToM1=( 1+1/(1+Del))/(2+Delini)
      
    if optype == "AV":
        xf=xi/M2tildeToM1
        Gam1=am.GamAV_MestoXX(xi,xi*(1+Delini),br.MPi,1,1000.)
        Gam2=am.GamAV_MestoXX(xf/(1+Del),xf,br.MPi,1,1000.)
        Lim_out = Lim_in*np.power(gu-gd,1/2.)*np.power(Gam2/Gam1,1/4.)
    else :
        xf=xi/M2tildeToM1
        Lim_out=Lim_in*0

        
    return xf, Lim_out # As usual we return the limits as function of M2

################################################## ####################################
#########################                           #########################
######################### Missing Energy  detection    #########################
#########################                           #########################
######################### ################################## #########################



def FastMonoPhoton(exp,xin,limin,Delini,Del, geff,optype="V"):
    gu,gd,ge,gs=Fillg(geff)  
           
    M2tildeToM1=( 1+1/(1+Del))/(2+Delini)
    x_out=xin/M2tildeToM1; # As usual, we return the value for the heavy state chi2
    
    return x_out, limin*np.sqrt(np.abs(ge))



################################################## ####################################
#########################                           #########################
######################### Mono-jet                  #########################
#########################                           #########################
######################### ################################## #########################



def FastMonoJet(exp,g_in,Lim_Up_in,Delini,Del, geff,optype="V"):
    gu,gd,ge,gs=Fillg(geff) 
    xi_basic=uf.log_sample(0.005,5,200)
    gef = np.sqrt(2*gu**2+gd**2)
    
    if gef < np.min(g_in):
        Lim_u_out=0
    else:
        Lim_u_out=np.interp(gef,g_in, Lim_Up_in)   
    Lim_full = np.full(200,Lim_u_out)
    return xi_basic,Lim_full 


################################################## ####################################
#########################                                       #########################
#########################  Supernovae cooling limits          #########################
#########################                                       #########################
######################### ################################## #########################



def FastSN1987Limit(limlist,Del, geff,optype="V",upperlimit=True):
    xi_basic=uf.log_sample(0.005,0.3,400)
         ##### Currently just test the different operator and apply a naive proton scattering scaling, except from the AV case where the upper limit derives from the pi0 branching ratio 
    gu,gd,ge,gs=Fillg(geff) 
    M2tildeToM1=( 1+1/(1+Del))/(2) ### Change for scaling dle=0 initially
    x_in,Lim_in=limlist[optype]
    if upperlimit:
        if optype == "V":
            Lim_out = Lim_in*np.sqrt((np.abs(gu)+np.abs(gd)+np.abs(ge))/2  ) # Scaling based on e+e- annihilation
            return xi_basic,np.interp(xi_basic,x_in/M2tildeToM1,Lim_out)  # we include the possibility of production from electrons just incase -- very rough
        else:
#             Gam1=br.GamAV_MestoXX(x_inAV,x_inAV*(1),br.MPi,1,1000.)
#             Gam2=br.GamAV_MestoXX(x_inAV,x_inAV*(1+Del),br.MPi,1,1000.)     
            xf=x_in/M2tildeToM1
            Gam1=am.GamAV_MestoXX(x_in,x_in*(1+0),br.MPi,1,1000.)
            Gam2=am.GamAV_MestoXX(xf/(1+Del),xf,br.MPi,1,1000.)
        
            
            Lim_out = Lim_in*np.power(gu-gd,1/2.)*np.power(Gam2/Gam1,1/8.) # Limits from invisible pi0 decay
#             print("x for SN, ",M2tildeToM1, x_inAV , Lim_inAV, Lim_out)
            return xi_basic,np.interp(xi_basic,x_in/M2tildeToM1,Lim_out) 
    else:
        if optype == "V":
            Lim_out = Lim_in*np.sqrt((np.abs(gu)+np.abs(gd))  )# we include the possibility of scattering from nuclei 
            return xi_basic,np.interp(xi_basic,x_in/M2tildeToM1,Lim_out) 
        else:
            Lim_out = Lim_in*np.sqrt((np.abs(gu)+np.abs(gd))  )
            return xi_basic,np.interp(xi_basic,x_in/M2tildeToM1,Lim_out)# we include the possibility of scattering from nuclei 
        


################################################## ####################################
#########################                                   #########################
######################### CR production and  detection      #########################
#########################                                   #########################
######################### ################################## #########################


def CRDMDecayflux(Eflux, Mx1, Mx2, Lam,Del, geff, OperatorType="AV", exp="t2k"):
    gu,gd,ge,gs=Fillg(geff) 
#     Lmeters = 6.6*10**(-25)*3*10**8/(Mx2*Mx1**4/64/math.pi**3/Lam**4)*math.sqrt(Eflux**2 + Mx1**2)/Mx1
    
    Lmeters=1/am.GamHDS(Mx1, Del, Lam,geff,OperatorType)*math.sqrt(Eflux**2 + Mx1**2)/Mx2*(3e8*6.5875e-25) # Updated with the full decay length
    PiF,EtaF = br.CRDMfluxintegrand(Eflux, Mx1, Mx2, Lam,Del, geff, OperatorType, exp)

#  InteEtaFlux = EtaF*cm2tom2*Ldet/Lmeters*math.exp(-avgAtmHeight/Lmeters)

    Ldet = 20 #cylinder of height 40m radius 20n
    
    cm2tom2 = 10**4
    avgAtmHeight = 3000*1000+5000 #approximate lower limit by taking production to be ~3000 km
    

    IntePiFlux = PiF*cm2tom2*ProbDecay(avgAtmHeight,Ldet,Lmeters)
    InteEtaFlux = EtaF*cm2tom2*ProbDecay(avgAtmHeight,Ldet,Lmeters)
    
    if exp == "t2k":
        volExp = 40*1200
    
#     print("Getting prodution for Mx1, ", Mx1, Mx2, PiF, EtaF, IntePiFhigh)
    totflux = 2*(IntePiFlux + InteEtaFlux)*volExp   
    return totflux


def Ndecayperyear_cosmicrays(Lam, Mx1, Mx2,Del, mygs, OperatorType ="AV", exp="t2k"):
    secperyear =  3.15*10**7
    return secperyear*quad(CRDMDecayflux, 10**(-3), 1, args=(Mx1, Mx2, Lam, Del,mygs, OperatorType, exp),epsabs=0.5e-01,epsrel=0.5e-01, limit=10)[0]

# def minimiseNlimFunc_cosmicrays(Lam, Mx1, Mx2, Del,exp="t2k", Nperyearlim=500, OperatorType ="V", mygs={"gu11":2/3.,"gd11":-1/3.,"gl11":-1.}):
#     return abs(Nprodperyear_cosmicrays(Lam, Mx1, Mx2, Del,mygs, OperatorType, exp) - Nperyearlim)

    


def FastCRLimit(exp, Del, geff, optype="AV", SaveToFile=True, ReadFromFile=False, filename='Output/Lim_superK_cosmicrays.dat'):
    if ReadFromFile: 
        filedat = np.loadtxt(filename)
        mxdellist, LimLowList, LimHighList = filedat
        return mxdellist, LimLowList, LimHighList

    Nt2klim = 70
    #note mx is the *light* chi mass. heavy chi mass = mx*(1+Del)
    minmx = me*2.001/(Del)*4
#     maxmx = 0.5*br.MEta/(1+Del) - 0.001 #production threshold for *heavy* chi mass at half the meson mass minus twice electron mass
    maxmx = br.MEta*0.999/(2+Del)  
    numpts = 20
#     mxlist = np.linspace(minmx, maxmx, numpts)
    
    mxlist = uf.log_sample(minmx, maxmx, numpts)
    res = 40 #resolution in GeV for finding limit
    lamlist = uf.log_sample(5, 1000, res)
    LimLowList = []
    LimHighList = []
    for mx in mxlist: 
        #find Lambda corresponding to Nt2klim number of events. Note that there are in general two solutions.
        limlow = -1
        limhigh = -1
        for Lam in lamlist: #range(5, 1000, res):
            Nprod = Ndecayperyear_cosmicrays(Lam, mx, mx*(1+Del), Del, geff, optype, "t2k")
            print("--------------")
            print("Lambda = ", Lam)
            print("Nprod = ", Nprod)
            print("--------------")
            #passed first threshold for lower limit 
            if Nprod >= Nt2klim and limlow == -1:  
                limlow = Lam #0.5*((Lam-res) + Lam)
            #passed second threshold for upper limit  
            elif Nprod <= Nt2klim and limhigh == -1 and limlow != -1:
                limhigh = Lam #0.5*((Lam-res) + Lam)
                break
        LimLowList.append(limlow)
        LimHighList.append(limhigh)
        if limlow == -1 and limhigh == -1: # Limits typically stop after this point, no need to take time searching
            break
        print("**************************************************")
        print(LimLowList)
        print(LimHighList)
        print("**************************************************")
    LimLowList = np.array(LimLowList)
    LimHighList = np.array(LimHighList)


    #for testing
    '''
    mxlist = np.array([0.00449904, 0.00759627, 0.01282572, 0.02165522, 0.03656314, 0.061734, 0.10423302, 0.17598929, 0.29714414, 0.50170463])
    LimLowList = np.array([ 5, 5,  5, 25, 25, 45, 85, -1, -1, -1])
    LimHighList = np.array([ 25,  45,  85, 125, 205, 325, 445,  -1,  -1,  -1])
    '''
    #remove trailing -1's from list 
    LimLowList, LimHighList = np.transpose([(low, high) for (low,high) in zip(LimLowList,LimHighList) if low != -1 and high != -1])
    mxlist = mxlist[:-(len(mxlist)-len(LimLowList))]
    #replicate last entry identically (with small x-axis offset) for uf.CombineUpDown to include last point correctly, 
    #otherwise it joins up last entry of lowlist with second-last entry of highlist
    mxlist = np.append(mxlist, mxlist[len(mxlist)-1]+0.000001)
    LimLowList = np.append(LimLowList, LimLowList[len(LimLowList)-1])
    LimHighList = np.append(LimHighList, LimHighList[len(LimHighList)-1])

    #save to file for saving time when plotting 
    if SaveToFile: 
        np.savetxt(filename, (mxlist*(1+Del), LimLowList, LimHighList))

    return mxlist*(1+Del), LimLowList, LimHighList
    

############################################
########## Heavy meson decay #################
###############################################



 
def FastInvMesDecay(channel,delrec, geff, optype):

    xi_full =uf.GetxiRange(channel,delrec)
    
#     print("Range for ", channel, " ",xi_full)
    Lim_full = np.array([br.getLimLambdaFromInvBR(mx/(1+delrec), mx, channel, geff , optype) for mx in xi_full])
#     Lim_full = np.array([br.LambdaInvBR(mx, mx) for mx in xi_full])
    return xi_full,Lim_full





################################################## ####################################
#########################                                   #########################
#########################    Decay signatures recasting      #########################
#########################                                   #########################
######################### ################################## #########################


### ---- Transform a limit for the dark photon into an limit on Lambda, assuming the same masses for the dark sector particles
def MakeEffLimits(exp, xLim_DP, Lim_DP, geff, optype, DelIni=0.1 ):
    xeff, ProdEff= br.NProd(DelIni,exp,geff,optype)
    xProd_DP, NProd_DP= br.NProd_DP(exp)
    
#     if exp=="faser": print("faser Lim eps original:",ProdEff,xeff,NProd_DP,xProd_DP,Lim_DP,xLim_DP)
    
    xi,ratio,Lim = uf.GetRatio(ProdEff,xeff,NProd_DP,xProd_DP,Lim_DP,xLim_DP)
    if optype=="AV": # We need to rescale the decay rate also     
        Lam1TeV = np.full(np.size(xi),1000)
        GamV= am.GamHDSee(xi,DelIni,Lam1TeV,{"gu11":2/3.,"gd11":-1/3.,"gd22":-1/3.,"gl11":-1.},"V")
        GamAV=am.GamHDSee(xi,DelIni,Lam1TeV,geff,"AV")
        Gamratio = GamAV/GamV
    else:
        Lam1TeV = np.full(np.size(xi),1000)
        GamVem= am.GamHDSee(xi,DelIni,Lam1TeV,{"gu11":2/3.,"gd11":-1/3.,"gd22":-1/3.,"gl11":-1.},"V")
        GamV=am.GamHDSee(xi,DelIni,Lam1TeV,geff,"V")
        Gamratio = GamV/GamVem
    EffLimIni =0.013*np.sqrt(xi)/np.sqrt(Lim)*np.power(ratio*Gamratio,1/8.)*1000
#     if exp=="faser": print("faser ratio original:",ratio[xi>0.01]*Gamratio[xi>0.01])

#     if exp=="faser": print("faser EffLimIni original:",EffLimIni[xi>0.01])
#     print(xi, EffLimIni)
    return xi,EffLimIni

### ---- Detectino efficiency (used to recast limits in term of a different delta
def DetEff(xNProd, NProd, xlim, Lamlim, Del ,DelIni,  exp, geff, optype="V"):

 # First we get the production ratio
# We need to shift the masses, making sure that the invariant mass is equal: M1 + M2 = M1tilde + M2tilde
    M2tildeToM1=( 1+1/(1+Del))/(2+DelIni)
#     print("Production function: ", xNProd,NProd)
    M1ToX=(2+DelIni)
    xmin=np.min(xNProd)*M1ToX; # Sending M1 to X=M1+M2
    xmax=np.max(xNProd)*M1ToX;
    xi = uf.log_sample(xmin,xmax,200)
    Lam1TeV = np.full(np.size(xi),1000)
#     print("Limlim: ", Lamlim[xlim*M1ToX>0.01])
    LamliminX=np.interp(xi, xlim*M1ToX, Lamlim)
#     print("Laimlim: ", LamliminX[xi>0.01])
    NprodinX = np.interp(xi, xNProd*M1ToX, NProd)
    GammaDecayinX= am.GamHDSee(xi/M1ToX,DelIni,Lam1TeV,geff,optype)

    Res=np.power(LamliminX,8)/NprodinX/GammaDecayinX
#     if exp=="faser" :print("Res: ", np.nan_to_num(Res)*(GammaDecayinX>0))
    return xi,np.nan_to_num(Res)*(GammaDecayinX>0) # We output as fnction of M1+M2


################  Stretching the effective function between 2 me and M_pi0/3
# This is necessary since the input limits have typically a small splitting del=10%, consequently, the detector efficiency can only be defined on the interval [ (2 +delini) / delini , M_m]
# which can be much larger than 2 m_e. for a different splitting, the efficiency should be defined bewteen [ (2 +del) / del , M_m], so we rescale down to this value
# The limit M_pi0/3 is chosen such that the two states are still very light compare to M_pi0, so that the kinematics of the decay are not much modified (since the stretching is made
# to adapt to the kinematic of the decay chi2 -> chi1 e+ e-)
def Stretchfeff(xi,feffin,DelProd,Del):
    thrup=br.MPi/3
    
    xlow=np.min(xi[feffin>0])
    fefftoStretch=feffin[np.logical_and(xi<thrup,xi>1.2*xlow)]
    xefftoStretch=xi[np.logical_and(xi<thrup,xi>1.2*xlow)]    

    xilow=xi[xi<thrup]
    xiInterp = uf.log_sample(2*me/Del*(2+Del),thrup,len(xefftoStretch))
#     print(xiInterp)
    fEffinM2Fit=np.interp(xilow, xiInterp, fefftoStretch)
    res=feffin
#     print(res[np.logical_and(xi>2*me/Del*(2+Del),xi<thrup)])
    res[np.logical_and(xi>2*me/Del*(2+Del),xi<thrup)]=fEffinM2Fit[np.logical_and(xilow>2*me/Del*(2+Del),xilow<thrup)]
    return res

### ---- Update the limit on Lambda obtained by varying the splitting between the two dark sector states (note that the limits is expressed in term of M2)
def ShiftLimDel(xNProd, NProd, xlim, Lamlim, Del ,DelIni,  exp, optypetmp,gefftmp,HeavyMeson=False):

# We need to shift the masses, making sure that the invariant mass is equal: M1 + M2 = M1tilde + M2tilde
    M2tildeToM1=( 1+1/(1+Del))/(2+DelIni)
    mymin=np.min(xNProd)/M2tildeToM1; # Sending M1 to M2tilde
    mymax=np.max(xNProd)/M2tildeToM1;
    xi = uf.log_sample(mymin,mymax,200)
    Lam1TeV = np.full(np.size(xi),1000)
   # We tolerate having both operator type for this function (along with both effective coupling), still a bit experimental though    
   # The first operator is the one use for production, then the other are used to get the decay
    if np.size(optypetmp)>1:
        if np.size(gefftmp)<2:
            print("Please provide the effective couplings for each type of effective operators (Vector or Axial-Vector")
            return LamLim1*0.
        geffandOp=tuple(zip(gefftmp,optypetmp))
        GamNew=sum(am.GamHDSll(xi/(1+Del), Del, Lam1TeV,ge,op) for (ge,op) in geffandOp)
        geff=gefftmp[0];optype=optypetmp[0]
    else:
        geff=gefftmp;optype=optypetmp
        GamNew=am.GamHDSll(xi/(1+Del), Del, Lam1TeV,geff,optype)



 # First we get the production ratio


    xProd_new, Prod_new= br.NProd(Del,exp,geff,optype,HeavyMeson)
    Nnew = np.interp(xi, xProd_new*(1+Del), Prod_new) ## Getting the Production for M2 and new splitting with the same invariant mass M1 + M2 as the original splitting


#     if (exp == "faser") : print("Faser prod " , xi , Nnew[xi>0.01])
####### ----- Recasting the detection rate
    M12ToM2=(1+Del)/(Del+2)
#     print("Effective Gamma",xi,GamNew)

    xM1M2,fEffinX=DetEff(xNProd, NProd, xlim, Lamlim, Del ,DelIni,  exp,geff,optype)
    
#     fEffinM2=np.interp(xi, xM1M2*M12ToM2, fEffinX) 

    # We stretch the function to account from the fact that the width of chi2 has a smaller lower bound (due to ee threshold)   
    if Del>DelIni:
        fEfftmp=Stretchfeff(xi, fEffinX, DelIni, Del)
        fEffFinal=np.interp(xi, xi*M12ToM2, fEfftmp) 
    else:
        fEffFinal=np.interp(xi, xM1M2*M12ToM2, fEffinX) 
    
    
#     if (exp == "faser") : print("Faser xlow" , np.min(xi[fEffFinal>0]))
#     if (exp == "faser") : print("Faser xlow2 " , np.min(xi[fEffinM2>0]))
#     if (exp == "faser") : print("Faser fEffFinal" , fEffFinal[xi>0.006])

    Limnew = np.power(fEffFinal*GamNew*Nnew,1/8.)

    ####### ---------- If the limits is decreased due to EFT limitation at LHC
    if (exp == "faser") or (exp == "mathusla"):
#         print("Limit second method  ",xi, Limnew)
        Limnew=ReduceLimLHC(Limnew)
#         print("Limit second method  ",xi, Limnew)
    return xi,Limnew # We output as function of M2


### ----  Given a particuler higher limit, find the corresponding lower limit in a beam-dump type experiment
def ShortLamLimit(mX2, Del,LamLim1,Dexp,Lexp,beamtype,geff,optype="V"):
#     print("Effective coupling size" , np.size(geff))
    # We tolerate having both operator type for this function (along with both effective coupling), still a bit experimental though    
    if np.size(optype)>1:
        if np.size(geff)<2:
            print("Please provide the effective couplings for each type of effective operators (Vector or Axial-Vector")
            return LamLim1*0.
        geffandOp=tuple(zip(geff,optype))
        GamDecay=sum(am.GamHDS(mX2/(1+Del), Del, LamLim1,ge,op) for (ge,op) in geffandOp)
    else:
        GamDecay=am.GamHDS(mX2/(1+Del), Del, LamLim1,geff,optype)

    ctaugamma1=1/GamDecay*BoostFact(mX2,Del,beamtype)*(3e8*6.5875e-25) # Decay lenght of the lower limit, mX is the heavy state at that stage

    # !!!! Testing, we select a 1/50 subfraction of the produced states, with an extra factor of 5 in boost
#     ctaugamma1=ctaugamma1*2
#     LamLim2=LamLim1*np.power(1/10.,1/8.)
    # !!!! End of testing 
    
    
    P1decay=ProbDecay(Dexp,Lexp,ctaugamma1) # Decay probability for the lower limit in Lam
    myA=np.power(LamLim1,4)*Dexp/ctaugamma1
    myB=P1decay/np.power(LamLim1,4)
    LamShort=FindShortLam(myA,myB,1.)
    
    if (beamtype=="LHC"): # Include the suppression in cross-section due to the cut to make the Effective theory consistent
        redpt=(LamShort<1700)
        if (LamShort.size > 1):
            LamShort[redpt]=FindShortLam(myA[redpt],myB[redpt]/0.0002,1-1.14/4)
        else:
            LamShort==FindShortLam(myA,myB/0.0002,1-1.14/4)
#        print("after: ",LamShort)
    return LamShort



### ---- Use all of the above to transform the limit on Lambda for a given decay experiments
#        assuming a splitting of Delini, into a new limit corresponding to the splitting
#        Del, further create the lower bound due to short-lived chi2
def RecastDecayLimit(xi,limOp1,Delini,Del,exp,geff,optype="V",HeavyMeson=False):
    # We assume the initial limits are already for the effective theory
    Dexp,Lexp,beamtype=GeomForExp(exp)
# ------ Getting production number for the new DelIni ...
    xprod, Nprod = br.NProd(Delini,exp,geff,optype,HeavyMeson)
# ------ Getting the limits ...

    xif,EffLim = ShiftLimDel(xprod, Nprod, xi, limOp1, Del,Delini,exp,optype,geff,HeavyMeson) 
    
    EffLimFin_Low=ShortLamLimit(xif, Del,EffLim,Dexp,Lexp,beamtype,geff,optype)   
    return xif,np.nan_to_num(EffLim),np.nan_to_num(EffLimFin_Low)




###############  Summarise all of the above, give the limit on Lambda from a dark photon bounds
#                following the most standard procedure 
def FastDecayLimit(exp,x_in,Lim_in, Del_in,Del,geff,optype="V", CombMerge=1.1,useHeavyMeson=False):

    xeff,Limeff = MakeEffLimits(exp, x_in, Lim_in, geff, optype, Del_in )
    xi_cast,Lim_cast,Lim_cast_low = RecastDecayLimit(xeff,Limeff,Del_in,Del,exp,geff,optype,useHeavyMeson)   
#     print("Full limits: " , exp, xi_cast,Lim_cast,Lim_cast_low)  
    xi_cast_full, Lim_cast_full=uf.CombineUpDown(xi_cast,Lim_cast_low,Lim_cast,CombMerge)
    return xi_cast_full, Lim_cast_full



################################################## ###################################
#########################                                       ######################
#########################  Some additional functions for decay signatures  ###########
#########################                                       ######################
######################### ################################## #########################


### ---- Create a number of events limits with Nexp events for a given experiments - no geometrical factors/efficiencies are included !
def GetNaiveDecayLimits( Del, exp,Nexp,geff, optype = "V",HeavyMeson=False):
    
    Dexp,Lexp,beamtype=GeomForExp(exp)
    xNProd, NProd= br.NProd(Del,exp,geff,optype,HeavyMeson)
    
    MinvFac=(1+Del)
    mymin=np.min(xNProd*MinvFac);
    mymax=np.max(xNProd*MinvFac);
    xi = uf.log_sample(mymin,mymax,500)
    Lam1TeV = np.full(np.size(xi),1000.)

    ctaugamma=1/am.GamHDSll(xi/(1+Del), Del, Lam1TeV,geff,optype)*BoostFact(xi,Del,beamtype)*(3e8*6.5875e-25)
    Limnew = 1000*np.power(np.interp(xi,xNProd*MinvFac, NProd)*Lexp/ctaugamma/Nexp,1/8.)
    
    if (beamtype=="LHC"):
        Limnew=ReduceLimLHC(Limnew)

    return xi/(1+Del),Limnew # We need to export as M1 to match with the other imported limits



### ---- Recast the limit for a given decay experiments for a range of splitting, used for plotting currently
def MakeTableLimit(xi,limOp1,Delini,exp,geff,optype="V"):
    
    xFin,EffLimFin,EffLimFin_Low= RecastDecayLimit(xi,limOp1,Delini,0.00001,exp,geff,optype)
    DelFin = np.full(np.size(xFin),0.00001)
    
    for Del in xDel:
        xi_tmp,EffLim_tmp,EffLimFin_tmp_low= RecastDecayLimit(xi,limOp1,Delini,Del,exp,geff,optype)
        Deltmp = np.full(np.size(xi_tmp),Del)
#        print(EffLim_tmp)
        xFin=np.concatenate((xFin,xi_tmp))
        DelFin=np.concatenate((DelFin,Deltmp))
        EffLimFin=np.concatenate((EffLimFin,EffLim_tmp))
        EffLimFin_Low=np.concatenate((EffLimFin_Low,EffLim_tmp_low))

    return xFin,1/(DelFin+1),EffLimFin,EffLimFin_Low

################  Same, but works with at single value for the input mass
def MakeTableLimit_OneMass(xi,limOp1,exp,Mx2,DelProd,geff,optype="V"):
    
    xDel =  np.logspace(-3, 3,200, endpoint=True)
    xFin = xi/(1+DelProd)
    Dexp,Lexp,beamtype=GeomForExp(exp)
    xprod, NProd = br.NProd(DelProd,exp,geff,optype)

    limIni = np.interp(Mx2,xi*(1+DelProd),limOp1)
    EffLimFin=np.array([])
    EffLimFin_Low=np.array([])
    DelFin = np.array([])
    
    for Del in xDel:
        M2ToX=(Del+2)/(1+Del)
        M2tildeToM1=( 1+1/(1+Del))/(2+DelProd)
        
        GamNew=am.GamHDSee(Mx2/(1+Del),Del,1000.,geff,optype)
        xProd_new, Prod_new= br.NProd(Del,exp,geff,optype)
        Nnew = np.interp(Mx2, xProd_new*(1+Del), Prod_new)
        xM1M2,fEffinX=DetEff(xprod, NProd, xi, limOp1, Del ,DelProd,  exp,geff,optype)
        fEffinM2=np.interp(xi, xM1M2/M2ToX, fEffinX)
        
        fEffFinal=Stretchfeff(xi,fEffinM2,DelProd,Del) # We stretch the function to account from the fact that the width of chi2 has a smaller lower bound (due to ee threshold)
        
        feff=np.interp(Mx2, xi, fEffFinal)
#         print("Testing one value feff,", Mx2, feff,xi, fEffFinal)
    #     print("Effective factor",xi,fEffinM2Fit)  
        Limnew = np.power(feff*GamNew*Nnew,1/8.)
                
#         print("Effective Lim",Mx2,Limnew,Del)  
    
        GamRatio=am.GamHDSee(Mx2/(1+Del),Del,1000.,geff,optype)/am.GamHDSee(Mx2*M2tildeToM1,DelProd,1000.,geff,optype) # GammaHDS defined w.r.t to the ligthest state mass, we compare for equivqlent invariant mass for the chi1 chi2 system
        xProd_new, Prod_new= br.NProd(Del,exp,geff,optype)
        Nnew = np.interp(Mx2, xProd_new*(1+Del), Prod_new) ## Getting the Production for M2 and new splitting with the same invariant mass M1 + M2 as the original splitting
        Nold = np.interp(Mx2, xprod/M2tildeToM1, NProd)
        ProdRatio=Nnew/Nold
        if (Nold<Nnew/1e10): ProdRatio=1 # Avoiding division by zero problems
        if (Nnew<Nold/1e10): ProdRatio=1
            
        Limnew = np.interp(Mx2, xi/M2tildeToM1, limOp1)*np.power(GamRatio,1/8.)*np.power(ProdRatio,1/8.)
        
        
        EffLim_tmp_low=ShortLamLimit(Mx2, Del,Limnew,Dexp,Lexp,beamtype,geff,optype)

        DelFin=np.append(DelFin,Del)
        EffLimFin=np.append(EffLimFin,Limnew)
        EffLimFin_Low=np.append(EffLimFin_Low,EffLim_tmp_low)


    return DelFin,EffLimFin,EffLimFin_Low









### ---- Same as before, but updated, can deal with two effective operator at once
# def ShiftLimDelv2(xlim, Lamlim, Del ,DelIni=0.1,  exp="lsnd",optype="V",geff={"gu11":2/3.,"gd11":-1/3.,"gl11":-1.},HeavyMeson=False):
# 
#  # First we get the production ratio
# # We need to shift the masses, making sure that the invariant mass is equal: M1 + M2 = M1tilde + M2tilde
#     M2tildeToM1=( 1+1/(1+Del))/(2+DelIni)
#     M2ToX=(Del+2)/(1+Del)
#     if np.size(optype)>1:
#         if np.size(geff)<4:
#             print("Please provide the effective couplings for each type of effective operators (Vector or Axial-Vector")
#             return LamLim1*0.
#         geffandOp=tuple(zip(geff,optype))
#         GamDecay=sum(am.GamHDS(mX2/(1+Del), Del, LamLim1,ge,op) for (ge,op) in geffandOp)
#     else:
#         GamDecay=am.GamHDS(mX2/(1+Del), Del, LamLim1,geff,optype)
#     
#     
#     mymin=np.min(xNProd)/M2tildeToM1; # Sending M1 to M2tilde
#     mymax=np.max(xNProd)/M2tildeToM1;
#     xi = uf.log_sample(mymin,mymax,200)
#     Lam1TeV = np.full(np.size(xi),1000)
#     
#     
#     xProd_new, Prod_new= br.NProd(Del,exp,geff,optype,HeavyMeson)
#     xProd, Prod= br.NProd(DelIni,exp,geff,optype,HeavyMeson)    
#     
#     Nnew = np.interp(xi, xProd_new*(1+Del), Prod_new) ## Getting the Production for M2 and new splitting with the same invariant mass M1 + M2 as the original splitting
# 
# ####### ----- Recasting the detection rate
#     
#     GamNew=am.GamHDSee(xi/(1+Del),Del,Lam1TeV,geff,optype)
# #     print("Effective Gamma",xi,GamNew)
# #     print("Effective Nprod",xi,Nnew)
#     xM1M2,fEffinX=DetEff(xNProd, NProd, xlim, Lamlim, Del ,DelIni,  exp,optype,geff)
#     fEffinM2=np.interp(xi, xM1M2/M2ToX, fEffinX) 
# #     print("Effective factor",xi,fEffinM2)
#     # We stretch the function to account from the fact that the width of chi2 has a smaller lower bound (due to ee threshold)   
#     fEffFinal=Stretchfeff(xi, fEffinM2, DelIni, Del)
# #     print("Effective factor",xi,fEffFinal)  
#     Limnew = np.power(fEffFinal*GamNew*Nnew,1/8.)
# #     print("Limit second method  ",xi, Limnew)
#     ####### ---------- If the limits is decreased due to EFT limitation at LHC
#     if (exp == "faser") or (exp == "mathusla"):
#         Limnew=ReduceLimLHC(Limnew)
#     return xi,Limnew # We output as function of M2

    

