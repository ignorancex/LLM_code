#!/usr/bin/env python
# -*- coding: utf-8 -*-


################################################## ##############################
#########################  Amplitudes for all relevant processes  #########################
######################### ######################### #############################

### Include all the various amplitudes relevant for the fermion portal,
### with user-input choices for the masses of the dark sector particles
### L. Darme, S. Ellis, T. You, 29/12/2019


############################ Messy library import part #########################
import numpy as np

from numpy.random import uniform, seed

from scipy import optimize as opt
from scipy.interpolate import interp1d
from scipy.integrate import quad
import math

import UsefulFunctions as uf



fPi = 0.1307;
MPi = 0.1349766; MEta = 0.547862; MEtap = 0.9577
MRho = 0.77526; MOmega = 0.78265
MDs = 2.00696; MD = 1.8648;MKs = 0.89581;MK = 0.49761
MPip=0.139570 
me=0.0005109989
mmuon=0.1056584

np.seterr(divide='ignore', invalid='ignore') # Set the divide by zer oand sqrt error warning in numpy off (this typically 
# occur due to the fact that we estimate the amplitude on the full array of value for m2
# The resulting nan and infinite are treated afterwards, this should be better implemented...

def Fillg(geff={"gu11":2/3.,"gd11":-1/3.,"gl11":-1.}):
    return geff.get("gu11",0.),geff.get("gd11",0.),geff.get("gl11",0.),geff.get("gd22",0.)

################################################## ##############################################
#########################                                               #########################
#########################  Meson decay Width                            #########################
#########################                                               #########################
######################### ######################### ##############################################


################################################## #########################
#########################  2 body decay        #########################
######################### ######################### #########################


#note this takes a numpy array only for inM1 and inM2
def GamAV_MestoXX(inM1,inM2,Mm,fmeff,Lam = 1000.): # For axial vector operator, P -> XX decay
    Gam = np.zeros(np.size(inM1))
    gp=(np.abs(inM1)+np.abs(inM2)<Mm)
    M1=inM1[gp]
    M2=inM2[gp]

    Gamtmp= -(fmeff**2*(M1 + M2)**2*(M1 - M2 - Mm)*(M1 - M2 + Mm)*  np.sqrt((-np.power((M1 - M2),2.) + Mm**2)*(-np.power((M1 + M2),2.) + Mm**2)))/(8.*np.power(Lam,4.)*Mm**3*np.pi)
    Gam[np.abs(inM1)+np.abs(inM2)<Mm] = Gamtmp

    return Gam


#note this takes a numpy array only for inM1 and inM2
def GamV_MestoXX(inM1,inM2,Mm,fmeff,Lam = 1000.): # For  vector operator, V -> XX decay
    Gam = np.zeros(np.size(inM1))
    gp=(np.abs(inM1)+np.abs(inM2)<Mm)
    M1=inM1[gp]
    M2=inM2[gp]
    Gamtmp=(fmeff**2*(M1 - M2 + Mm)*(-M1 + M2 + Mm)*np.sqrt((-(M1 - M2)**2 + Mm**2)*(-(M1 + M2)**2 + Mm**2))*(M1**2 + 2*M1*M2 + M2**2 + 2*Mm**2))/(24.*Lam**4*Mm**3*np.pi)
    Gam[np.abs(inM1)+np.abs(inM2)<Mm] = Gamtmp
    
    return Gam

################################################## #########################
#########################  3 body decay        #########################
######################### ######################### #########################

#note this takes a numpy array only for inM1 and inM2
def GamV_MestoXXgam(inM1,inM2,Mm,fmeff,Lam = 1000.): # For  vector operator, V -> XXgamma decay
    Gam = np.zeros(np.size(inM1)) # For small splitting, M1 can be negative (iDM case)
    gp=(np.abs(inM1)+np.abs(inM2)<Mm)
    M1=inM1[gp]
    M2=inM2[gp]
    myfPi=0.1307
    em=0.302862
    Gamtmp = fmeff**2/np.power(Lam,4.)*(2/myfPi**2)*(em**2*(-(np.sqrt(M1**4 + (M2**2 - Mm**2)**2 - 2*M1**2*(M2**2 + Mm**2))*(3*M1**8 - 10*M1**7*M2 + 3*M2**8 - 27*M2**6*Mm**2 - 47*M2**4*Mm**4 + 13*M2**2*Mm**6 - 2*Mm**8 - 3*M1**6*(14*M2**2 + 9*Mm**2) + M1**5*(-290*M2**3 + 50*M2*Mm**2) + M1**4*(-282*M2**4 + 177*M2**2*Mm**2 - 47*Mm**4) - 10*M1**3*(29*M2**5 - 52*M2**3*Mm**2 + 13*M2*Mm**4) + M1**2*(-42*M2**6 + 177*M2**4*Mm**2 + 22*M2**2*Mm**4 + 13*Mm**6) - 10*M1*(M2**7 - 5*M2**5*Mm**2 + 13*M2**3*Mm**4 + 3*M2*Mm**6)))/20. + 6*(M1**2 - M2**2)*Mm**4*(M1**4 - 2*M1**2*M2**2 + M2**4 + 2*M1*M2*Mm**2)*np.log(Mm) + 3*(4*M1**7*M2**3 + 3*M1**2*M2**4*Mm**4 - M2**6*Mm**4 - 2*M1*M2**3*Mm**6 + 12*M1**5*(M2**5 - M2**3*Mm**2) + M1**6*(6*M2**4 - Mm**4) + 3*M1**4*M2**2*(2*M2**4 - 4*M2**2*Mm**2 + Mm**4) + 2*M1**3*(2*M2**7 - 6*M2**5*Mm**2 + 6*M2**3*Mm**4 - M2*Mm**6))*np.log(-M1**2 - M2**2 + Mm**2 + np.sqrt(M1**4 + (M2**2 - Mm**2)**2 - 2*M1**2*(M2**2 + Mm**2))) - 3*(4*M1**7*M2**3 + 3*M1**2*M2**4*Mm**4 - M2**6*Mm**4 - 2*M1*M2**3*Mm**6 + 12*M1**5*(M2**5 - M2**3*Mm**2) + M1**6*(6*M2**4 - Mm**4) + 3*M1**4*M2**2*(2*M2**4 - 4*M2**2*Mm**2 + Mm**4) + 2*M1**3*(2*M2**7 - 6*M2**5*Mm**2 + 6*M2**3*Mm**4 - M2*Mm**6))*np.log(2*np.abs(M1)*np.abs(M2)) - 6*(M1**2 - M2**2)*Mm**4*(M1**4 - 2*M1**2*M2**2 + M2**4 + 2*M1*M2*Mm**2)*np.log(np.abs(M1) + np.abs(M2)) + 3*(M1**2 - M2**2)*Mm**4*(M1**4 - 2*M1**2*M2**2 + M2**4 + 2*M1*M2*Mm**2)*np.log((-2*(2*M1**2*M2**2 + (M1**2 + M2**2)*np.abs(M1)*np.abs(M2)))/(M1**4 + M2**2*(M2**2 - Mm**2 - np.sqrt(M1**4 + (M2**2 - Mm**2)**2 - 2*M1**2*(M2**2 + Mm**2))) + M1**2*(-2*M2**2 - Mm**2 + np.sqrt(M1**4 + (M2**2 - Mm**2)**2 - 2*M1**2*(M2**2 + Mm**2)))))))/(12288.*Mm**3*np.pi**7)

    
    Gam[np.abs(inM1)+np.abs(inM2)<Mm] = Gamtmp
    return Gam


#heavy meson with mass Mm decaying to lighter meson with mass Mmprime + chichi
#note inM1 and inM2 must be numpy arrays

def _MestoXXmesAmpsquared(x23, m1, m2, m3, m4):
    amp2 = -(1/x23)*math.sqrt((m2**4 + (m3**2 - x23)**2 - 2*m2**2*(m3**2 + x23))/x23)*math.sqrt((m1**4 + (m4**2 - x23)**2 - 2*m1**2*(m4**2 + x23))/x23)*(m3**4*m4**2 + 2*m3**3*m4**3 + m3**2*m4**4 + m3**4*x23 - 2*m3**3*m4*x23 - 2*m3**2*m4**2*x23 - 2*m3*m4**3*x23 + m4**4*x23 - 3*m3**2*x23**2 + 2*m3*m4*x23**2 - 3*m4**2*x23**2 + 4*x23**3 + m1**2*(-m3**4 - 2*m3**3*m4 - m3**2*(m4**2 - 3*x23) + 6*m3*m4*x23 - x23*(m4**2 + 4*x23) + m2**2*(m3**2 + 2*m3*m4 + m4**2 + 4*x23)) - m2**2*(m4**4 - 3*m4**2*x23 + 4*x23**2 + m3**2*(m4**2 + x23) + 2*m3*(m4**3 - 3*m4*x23)))
    return amp2

def GamV_MestoXXmes(inMchi1, inMchi2, Mm, Mmprime, fmeff, Lam = 1000.):
    
    Gam = np.zeros(np.size(inMchi1))
    allowedlist = (inMchi1+inMchi2<(Mm-Mmprime)) #list of true or false for each value in Mchi1 and Mchi2 list
    m3list = inMchi1[allowedlist]
    m4list = inMchi2[allowedlist]
    
    
    m1 = Mm
    m2 = Mmprime
    prefac = 1./(2.*np.pi)**3 * 1./(32.*m1**3)*1./Lam**4*fmeff**2
    
    ampsquared = np.array([quad(_MestoXXmesAmpsquared, (m2+m3)**2, (m1-m4)**2, args=(m1,m2,m3,m4))[0] for (m3, m4) in zip(m3list, m4list)])
    Gamtmp = prefac*ampsquared 
    Gam[inMchi1+inMchi2<(Mm-Mmprime)] = Gamtmp
    return Gam



################################################## ##############################################
#########################                                               #########################
#########################  Heavy states decay Width                     #########################
#########################                                               #########################
######################### ######################### ##############################################



######################### ##################################     #########################
#########################          Decay width leptonic final states #########################
######################### ##################################     #########################


### ---- Calculate the decay amplitude for 3 body decay chi2 -> chi1 l l, at worst (around delta = 4Ml ) better than 30% accuracy, approximations become worst at very low mX (relevant for muon contriution in large splitting case)

def GamHDSllVThr(mX, delta, ml,Lam,ge=-1.):
    gdecay = ge;m1=mX;m2=np.abs(mX*(1+delta));dm = (m2-m1)/(2*ml)-1;# Set for leptonic decay
    Gammtot=1/np.power(Lam,4)*np.power(dm,3)*np.power(ml,5)*((8 + 7*dm)*np.power(m1,4) + 4*(8 + dm)*np.power(m1,3)*ml + 2*(28 + 17*dm)*np.power(m1,2)*np.power(ml,2) -12*(-4 + dm)*m1*np.power(ml,3) + 15*dm*np.power(ml,4))/(8*np.sqrt(m1)*np.power(m1 + 2*ml,7/2.)*np.power(np.pi,2))   
    return (Lam > 1)*(Gammtot)*gdecay**2

def GamHDSllVHigh(mX, delta, ml,Lam,ge=-1.):
    gdecay = ge;dm = delta;m1=mX;m2=np.abs(mX*(1+delta)) # Set for leptonic decay
    Gammtot=1/np.power(Lam,4)*((-np.power(m1,8) + 2*np.power(m1,7)*m2 + 8*np.power(m1,6)*np.power(m2,2) + 18*np.power(m1,5)*np.power(m2,3) - 18*np.power(m1,3)*np.power(m2,5) - 8*np.power(m1,2)*np.power(m2,6) - 2*m1*np.power(m2,7) + np.power(m2,8) +11*np.power(np.power(m1,2) - np.power(m2,2),3)*np.power(ml,2) - 24*np.power(m1,4)*np.power(m2,4)*np.log(m1/m2) + 24*np.power(m1,5)*np.power(m2,3)*np.log(m2/m1) +24*np.power(m1,3)*np.power(m2,5)*np.log(m2/m1) - 12*np.power(ml,4)*(3*np.power(m1,4) - 3*np.power(m2,4) -2*np.power(m1,3)*m2*(-1 + np.log(64) - 6*np.log(-np.power(m1,2) + np.power(m2,2)) + 6*np.log(m1*ml)) +2*m1*np.power(m2,3)*(-1 + 6*np.log(m2/(-np.power(m1,2) + np.power(m2,2))) + 3*np.log(4*np.power(ml,2))))))/(384*np.power(m2,3)*np.power(np.pi,3))
    return (Lam > 1)*(Gammtot)*gdecay**2

def GamHDSllAVThr(mX, delta, ml,Lam,ge=-1.):
    gdecay = ge;m1=mX;m2=np.abs(mX*(1+delta));dm = (m2-m1)/(2*ml)-1;
    Gammtot=1/np.power(Lam,4)*(np.power(dm,3)*np.power(ml,5)*(3*(8 + 7*dm)*np.power(m2,4) - 12*(8 + 13*dm)*np.power(m2,3)*ml + 2*(52 + 135*dm)*np.power(m2,2)*np.power(ml,2) - 4*(4 + 23*dm)*m2*np.power(ml,3) + 5*dm*np.power(ml,4)))/(8*np.sqrt(np.power(m2,7)*(m2 - 2*ml))*np.power(np.pi,2))
    return (Lam > 1)*(Gammtot)*gdecay**2

def GamHDSllAVHigh(mX, delta, ml,Lam,ge=-1.):
    gdecay = ge;m1=mX;m2=np.abs(mX*(1+delta));dm = (m2-m1)/(2*ml)-1; # Set for leptonic decay
    Gammtot=-1/np.power(Lam,4)*((np.power(m1,8) + 2*np.power(m1,7)*m2 - 8*np.power(m1,6)*np.power(m2,2) + 18*np.power(m1,5)*np.power(m2,3) - 18*np.power(m1,3)*np.power(m2,5) + 8*np.power(m1,2)*np.power(m2,6) - 2*m1*np.power(m2,7) - np.power(m2,8) -19*np.power(m1,6)*np.power(ml,2) - 48*np.power(m1,5)*m2*np.power(ml,2) + 57*np.power(m1,4)*np.power(m2,2)*np.power(ml,2) - 57*np.power(m1,2)*np.power(m2,4)*np.power(ml,2) + 48*m1*np.power(m2,5)*np.power(ml,2) +19*np.power(m2,6)*np.power(ml,2) + 60*np.power(m1,4)*np.power(ml,4) + 168*np.power(m1,3)*m2*np.power(ml,4) - 168*m1*np.power(m2,3)*np.power(ml,4) - 60*np.power(m2,4)*np.power(ml,4) +12*np.power(m2,4)*np.power(ml,4)*np.log(256) + 24*m1*np.power(m2,3)*np.power(ml,4)*np.log(1024) - 6*np.power(m1,4)*np.power(ml,4)*np.log(65536) -12*np.power(m1,3)*m2*np.power(ml,4)*np.log(1048576) - 24*np.power(m1,3)*(np.power(m1,2)*np.power(m2,3) + np.power(m2,5) + 4*m1*np.power(ml,4) + 10*m2*np.power(ml,4))*np.log(m1) -24*(2*np.power(m1,4) + 5*np.power(m1,3)*m2 - 5*m1*np.power(m2,3) - 2*np.power(m2,4))*np.power(ml,4)*np.log(np.power((m1 - m2),2)) + 24*np.power(m1,4)*np.power(m2,4)*np.log(m1/m2) +192*np.power(m1,3)*np.power(m2,3)*np.power(ml,2)*np.log(m1/m2) + 24*np.power(m1,5)*np.power(m2,3)*np.log(m2) + 24*np.power(m1,3)*np.power(m2,5)*np.log(m2) +240*m1*np.power(m2,3)*np.power(ml,4)*np.log(m2) + 96*np.power(m2,4)*np.power(ml,4)*np.log(m2) + 96*np.power(m1,4)*np.power(ml,4)*np.log(-m1 + m2) + 240*np.power(m1,3)*m2*np.power(ml,4)*np.log(-m1 + m2) - 240*m1*np.power(m2,3)*np.power(ml,4)*np.log(-m1 + m2) - 96*np.power(m2,4)*np.power(ml,4)*np.log(-m1 + m2) +48*np.power(m1,4)*np.power(ml,4)*np.log(np.power((np.power(m1,2) - np.power(m2,2)),2)) + 120*np.power(m1,3)*m2*np.power(ml,4)*np.log(np.power((np.power(m1,2) - np.power(m2,2)),2)) -120*m1*np.power(m2,3)*np.power(ml,4)*np.log(np.power((np.power(m1,2) - np.power(m2,2)),2)) - 48*np.power(m2,4)*np.power(ml,4)*np.log(np.power((np.power(m1,2) - np.power(m2,2)),2)) - 96*np.power(m1,4)*np.power(ml,4)*np.log(ml) -240*np.power(m1,3)*m2*np.power(ml,4)*np.log(ml) + 120*m1*np.power(m2,3)*np.power(ml,4)*np.log(np.power(ml,2)) + 48*np.power(m2,4)*np.power(ml,4)*np.log(np.power(ml,2))))/(384*np.power(m2,3)*np.power(np.pi,3))
    return (Lam > 1)*(np.abs(mX)*delta > 2*ml)*(Gammtot)*gdecay**2

def GamHDSllAV(mX, delta, ml,Lam,ge=-1.): # Not working for negative mass just yet  
    funclow = lambda x: GamHDSllAVThr(x, delta,  ml,Lam,ge)
    funchigh = lambda x: GamHDSllAVHigh(x, delta,  ml,Lam,ge)
    return (uf.CombineAndInterp(funclow,funchigh,1.3*2*ml/delta,1.5*2*ml/delta,mX))
            
def GamHDSllVV(mX, delta, ml,Lam,ge=-1.): # Not working for negative mass just yet   
    funclow = lambda x: GamHDSllVThr(x, delta,  ml,Lam,ge)
    funchigh = lambda x: GamHDSllVHigh(x, delta,  ml,Lam,ge)
    return (uf.CombineAndInterp(funclow,funchigh,1.75*2*ml/delta,2.25*2*ml/delta,mX))           

### ---- Calculate the full decay amplitude for 3 body decay chi2 -> chi1 X

def GamHDSll(mX, delta, Lam,geff,optype = "V"):  
    ge=geff.get("gl11",0.) ;gmu=geff.get("gl22",0.) 
    if optype == "AV":
        Gamee=(delta*np.abs(mX)>2*me)*GamHDSllAV(mX, delta,me,Lam,ge)
        Gammumu=(delta*np.abs(mX)>2*mmuon)*GamHDSllAV(mX, delta,mmuon,Lam,gmu)
        Gamtot=Gamee+Gammumu
    elif optype == "V":
        Gamee=(delta*np.abs(mX)>2*me)*GamHDSllVV(mX, delta,me,Lam,ge)
        Gammumu=(delta*np.abs(mX)>2*mmuon)*GamHDSllVV(mX, delta,mmuon,Lam,gmu)
        Gamtot=Gamee+Gammumu
    else:
        Gamtot=0.*mX
    return Gamtot

def GamHDSee(mX, delta, Lam,geff,optype = "V"):   
    
    ge=geff.get("gl11",0.) ;
    
#     print("Gamma for ee in chi2 decay:" , optype,geff,ge,GamHDSllAV(mX, delta,me,Lam,ge),delta)
    
    if optype == "AV": 
        Gamee=(delta*np.abs(mX)>2*me)*GamHDSllAV(mX, delta,me,Lam,ge)
        Gamtot=Gamee
    elif optype == "V":
        Gamee=(delta*np.abs(mX)>2*me)*GamHDSllVV(mX, delta,me,Lam,ge)
        Gamtot=Gamee
    else:
        Gamtot=0.*mX
    
    return Gamtot

######################### ##################################     #########################
#########################          Decay width hadronic final states #########################
######################### ##################################     #########################

### ---- Calculate the decay amplitude for 3 body decay chi2 -> chi1 pi pi, only for vector coupling to SM

def GamHDSpipiVThr(mX, delta, Lam,geff):
    gu,gd,ge,gs=Fillg(geff) 
    gdecay = gu-gd;mm=MPip;m1=mX;m2=np.abs(mX*(1+delta));dm = (m2-m1)/(2*mm)-1;

    Gamlow = (m2-m1 > 2*MPi)*(m2-m1 < 6*MPi)/np.power(Lam,4)*(np.power(dm,4.)*np.power(mm,6)*((2 + dm)*np.power(m2,4) - 2*(4 + 5*dm)*np.power(m2,3)*mm + 14*(1 + 2*dm)*np.power(m2,2)*np.power(mm,2)-6*(2 + 5*dm)*m2*np.power(mm,3)+ 3*dm*np.power(mm,4)))/(16*np.power(m2,7/2.)*np.sqrt((m2 - 2*mm)*np.power(mm,2))*np.power(np.pi,2))
#     print(Gamlow)
    return (Lam > 1)*np.nan_to_num(Gamlow)*gdecay**2


def GamHDSpipiVV(mX, delta,Lam,geff): # Not working for negative mass just yet   
    funclow = lambda x: GamHDSpipiVThr(x, delta,Lam,geff)
    Res=(np.abs(mX*delta)<6*MPi)*funclow(mX)+(np.abs(mX*delta)>=6*MPi)*funclow(6*MPi/delta)*np.power(delta*np.abs(mX),5)/np.power(6*MPi,5)

    return Res      

### ---- Calculate the decay amplitude for 2 body decay chi2 -> chi1 V

def GamHDSrhoV(mX, delta,Lam,geff): # Not working for negative mass just yet   
    gu,gd,ge,gs=Fillg(geff) 
    dm = delta;m1=mX;m2=np.abs(mX*(1+delta))
    fuRho = 0.2215; fdRho = 0.2097;
    fRhoeff = (gu*fuRho - gd*fdRho)/(np.sqrt(2.));
    mv=MRho
    Res=1/np.power(Lam,4)*(fRhoeff**2*np.power(mv,2)*np.power((np.power((-m1 + m2),2) - np.power(mv,2)),3/2.)*np.sqrt(np.power(m2+m1,2) - np.power(mv,2))*(np.power(m2+m1,2) + 2*np.power(mv,2)))/ (16*np.power(m2,3)*np.pi)
    return np.nan_to_num(Res)  

def GamHDSomegaV(mX, delta,Lam,geff): # Not working for negative mass just yet   
    gu,gd,ge,gs=Fillg(geff) 
    dm = delta;m1=mX;m2=np.abs(mX*(1+delta))
    fuOmega = 0.1917; fdOmega = 0.201; fPhi = 0.233;  ## Caution, often sqrt(2) differences
    fOmegaeff = (gu* fuOmega + gd*fdOmega)/(np.sqrt(2.));
    mv=MOmega
    Res=1/np.power(Lam,4)*(fOmegaeff**2*np.power(mv,2)*np.power((np.power((-m1 + m2),2) - np.power(mv,2)),3/2.)*np.sqrt(np.power(m2+m1,2) - np.power(mv,2))*(np.power(m2+m1,2) + 2*np.power(mv,2)))/ (16*np.power(m2,3)*np.pi)
    return np.nan_to_num(Res)


### ---- Calculate the decay amplitude for 2 body decay chi2 -> chi1 P

def GamHDSPiAV(mX, delta,Lam,geff): # Not working for negative mass just yet   
    gu,gd,ge,gs=Fillg(geff) 
    dm = delta;m1=mX;m2=np.abs(mX*(1+delta))
    fPieff = fPi*((gu - gd)/np.sqrt(2))
    mm=MPi*(delta*np.abs(mX)>MPi)
    Res=(delta*np.abs(mX)>MPi)/np.power(Lam,4)*(fPieff**2*np.power((m1 - m2),2)*np.sqrt(np.power((m1 - m2),2) - np.power(mm,2))*np.power((np.power(m2+m1,2) - np.power(mm,2)),(3/2.)))/(16*np.power(m2,3)*np.pi)
    return Res    

def GamHDSEtaAV(mX, delta,Lam,geff): # Not working for negative mass just yet   
    gu,gd,ge,gs=Fillg(geff)
    dm = delta;m1=mX;m2=np.abs(mX*(1+delta))
    fEtaeff =  fPi*(0.5*(gu + gd - 2*gs) + 0.1*(gu + gd + gs))
    mm=MEta*(delta*np.abs(mX)>MEta)
    Res=(delta*np.abs(mX)>MEta)/np.power(Lam,4)*(fEtaeff**2*np.power((m1 - m2),2)*np.sqrt(np.power((m1 - m2),2) - np.power(mm,2))*np.power((np.power(m2+m1,2) - np.power(mm,2)),(3/2.)))/(16*np.power(m2,3)*np.pi)
    return Res  

def GamHDSEtapAV(mX, delta,Lam,geff): # Not working for negative mass just yet   
    gu,gd,ge,gs=Fillg(geff) 
    dm = delta;m1=mX;m2=np.abs(mX*(1+delta))
    fEtapeff = fPi*(-0.2*(gu + gd - 2*gs) + 0.68*(gu + gd + gs));
    mm=MEtap*(delta*np.abs(mX)>MEtap) # We keep the meson mass only when the gamma is non-zero, avoid crashed
    Res=(delta*np.abs(mX)>MEtap)/np.power(Lam,4)*(fEtapeff**2*np.power((m1 - m2),2)*np.sqrt(np.power((m1 - m2),2) - np.power(mm,2))*np.power((np.power(m2+m1,2) - np.power(mm,2)),(3/2.)))/(16*np.power(m2,3)*np.pi)
    return Res  


######################### ##################################     #########################
#########################      Full decay width  #########################
######################### ##################################     #########################


def GamHDS(mX, delta, Lam,geff,optype = "V"):
    gu,gd,ge,gs=Fillg(geff) ;gmu=geff.get("gl22",0.)

    if optype == "AV":
#         print("Splitting :",delta*np.abs(mX))
        Gamee=(delta*np.abs(mX)>2*me)*GamHDSllAV(mX, delta,me,Lam,ge)
        Gammumu=(delta*np.abs(mX)>2*mmuon)*GamHDSllAV(mX, delta,mmuon,Lam,gmu)
        Gampi=(delta*np.abs(mX)>MPi)*GamHDSPiAV(mX, delta,Lam,geff)
        Gameta=(delta*np.abs(mX)>MEta)*GamHDSEtaAV(mX, delta,Lam,geff)
        Gametap=(delta*np.abs(mX)>MEtap)*GamHDSEtapAV(mX, delta,Lam,geff)
#         print("DEcay width AV", Gamee , Gammumu, Gampi)
        Gamtot=Gamee+Gammumu+Gameta+Gametap+Gampi
    elif optype == "V":
        Gamee=(delta*np.abs(mX)>2*me)*GamHDSllVV(mX, delta,me,Lam,ge)
        Gammumu=(delta*np.abs(mX)>2*mmuon)*GamHDSllVV(mX, delta,mmuon,Lam,gmu)
        Gampipi=(delta*np.abs(mX)>2*MPi)*GamHDSpipiVV(mX, delta,Lam,geff)
        Gamrho=(delta*np.abs(mX)>MRho)*GamHDSrhoV(mX, delta, Lam, geff)
        Gamomega=(delta*np.abs(mX)>MOmega)*GamHDSomegaV(mX, delta, Lam, geff)
#         print("DEcay width V", Gamee , Gammumu, Gampipi)
        Gamtot=Gamee+Gammumu+Gampipi+Gamrho+Gamomega
    else:
        Gamtot=mX*0.
    return Gamtot


