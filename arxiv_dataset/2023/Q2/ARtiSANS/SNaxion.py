import os, sys, inspect, math, collections
import numpy as np
from scipy import interpolate
import scipy
from scipy.integrate import odeint,quad
import par

# coefficient of Axion Luminosity
cLumA = 1.69e+20*0.604972*30**(7./2)*1./4*(1.4*par.Msol*par.kg2g)**2*(10*par.km2cm)**(-3)
    
class SNnu:
    """
    Analytic Functions for SN neutrino emission (arxiv:2008.07070)
    """
    ###########################################################################
    def __init__(self):
        """
        intialization
        """
        name = "SNnu"
        self.name = name
    
    def entropy(self, mPNS, rPNS, gBeta, t,t0):
        """
        Entropy [kB baryon-1] - SM case 
        """
        return 4*(mPNS/1.4)**(13./6)*((rPNS/10)**(-2))*((gBeta/3.)**2)*((t + t0)/100.)**(-5./2)
    
    def lumNuT(self, mPNS, rPNS, gBeta, t,t0): 
        """
        Neutrino Luminosity [erg s-1] as function of time - SM case
        """
        return 3.3e+51*(mPNS/1.4)**6*((rPNS/10)**(-6))*((gBeta/3.)**4)*((t + t0)/100.)**(-6);
    
    def EmNuT(self, mPNS, rPNS, gBeta, t,t0):
        """
        Neutrino average energy [MeV] as function of time - SM case
        """
        return 16*(mPNS/1.4)**(3./2)*((rPNS/10)**(-2))*(gBeta/3.)*((t + t0)/100.)**(-3./2);
    
    def lumNuE(self, mPNS, rPNS, gBeta, Enutot):
        """
        Neutrino Luminosity [erg s-1] as function of total Nu emitted energy for t=0 - SM case
        """
        return 4e+49*(mPNS/1.4)**(-6/5)*((rPNS/10)**(6/5))*((gBeta/3.)**(-4/5))*(Enutot/1e+52)**(6/5);

    def lumNuS(self, mPNS, rPNS, gBeta, s): 
        """
        Neutrino Luminosity [erg s-1] as function of entropy - General 
        """
        return 1.2*1e+50*(mPNS/1.4)**(4/5)*((rPNS/10)**(-6/5))*((gBeta/3.)**(-4/5))*s**(12/5);

    def EmNuS(self, mPNS, rPNS, gBeta, s): 
        """
        Neutrino average energy [MeV] as function of entropy - General (eq.(A4))
        """
        return 7*(mPNS/1.4)**(1/5)*((rPNS/10)**(-4/5))*((gBeta/3.)**(-1/5))*(s**(3/5));

    def time0(self, mPNS, rPNS, gBeta, Enutot): 
        """
        Time origin [s] used as initial condition for entropy
        """
        return 210*(mPNS/1.4)**(6/5)*((rPNS/10)**(-6/5))*((gBeta/3.)**(4/5))*(Enutot/1e+52)**(-1/5);

    def EtotNu(self, mPNS, rPNS, gBeta, t0, tfinal=200): #SM [erg]
        """
        Total Energy [erg] emitted by Neutrinos integrated up to tfinal [s] - SM case
        """
        time = np.linspace(0.001, tfinal, 10000)
        lumNu = [self.lumNuT(mPNS, rPNS, gBeta, t, t0) for t in time]
        intlumNu = interpolate.interp1d(time.flat,lumNu,kind='linear') 
        EnuInt = quad(intlumNu,0.001,tfinal,full_output=1)
        EnuTot = 6*EnuInt[0]
        return EnuTot
    
    def calcEtemp(self, lum, tinit=0.011, tfinal=50, step=200):
        """
        Function to calculate the total energy [erg] emitted by a neutrino flavor up to a certain time - General
        """
        Earr =[]
        time=np.linspace(tinit, tfinal, step)
        for t in time:
            lumInt =quad(lum,0.01,t,full_output=1)[0]
            Earr.append(lumInt)
        EarrInt = interpolate.interp1d(time,Earr)  
        return EarrInt
    
    def calcSM(self, mPNS, rPNS, gBeta, EnuTot, tinit=0.01, tfinal=500):
        """
        Function to calculate the SM Neutrino Luminosity[erg s-1], total emmited energy [erg] an Average Nu Energy [MeV]
        """
        tarr=np.linspace(tinit, tfinal, 10000)
        
        timeP = self.time0(mPNS, rPNS, gBeta, EnuTot)
        # neutrino luminosity
        LumNu = self.lumNuT(mPNS, rPNS, gBeta,tarr,timeP)
        LumNuInt = interpolate.interp1d(tarr,LumNu)  
        # neutrino total emitted energy
        EtotNu= self.EtotNu(mPNS, rPNS, gBeta,timeP)
        # neutrino average energy
        EnuM = self.EmNuT(mPNS, rPNS, gBeta,tarr,timeP)
        EnuMInt = interpolate.interp1d(tarr,EnuM)  
        
        return LumNuInt,EtotNu,EnuMInt
    
    def calcAllSM(self, mPNS, rPNS, gBeta1, EnuTot1, gBeta2, EnuTot2, tinit=0.01, tfinal=500):
        """
        Compute the final SM Neutrino Luminosity[erg s-1] and Average Nu Energy [MeV] considering an initial and final phase
        """
        LumI,EtotI,EnuMI = self.calcSM( mPNS, rPNS, gBeta1, EnuTot1, tinit, tfinal)
        LumF,EtotF,EnuMF = self.calcSM( mPNS, rPNS, gBeta2, EnuTot2, tinit, tfinal)
        
        ## sum initial+final phase
        tarr=np.linspace(tinit, tfinal, 10000)
        # neutrino average energy
        EnuM = (LumI(tarr)+LumF(tarr))/(LumI(tarr)/EnuMI(tarr)+ LumF(tarr)/EnuMF(tarr))
        EnuMint = interpolate.interp1d(tarr,EnuM)  
        # neutrino luminosity
        LumNu = (LumI(tarr)+LumF(tarr))
        LumNuint = interpolate.interp1d(tarr,LumNu)   
        
        return LumNuint,EnuMint,EnuMI,EnuMF
    
    
class SNaxionBrem(SNnu):
    """
    Analytic Functions for SN neutrino+axion emission considering nuclear Bremsstrahlung
    """
    def __init__(self, mPNS, rPNS, gaNN):
        """
        intialization
        """
        name = "SNabrem"
        self.name = name
        self.mPNS = mPNS
        self.rPNS = rPNS
        self.alphaAb = gaNN**2/4./math.pi
   
    def lumAbrem(self, s):
        """
        Axion nuclear Bremsstrahlung Luminosity [erg s-1] as a function of the entropy
        """
        return cLumA*self.alphaAb*(self.mPNS/1.4)**(13./3)*((self.rPNS/10)**(-10))*abs(s)**(7/2)
        
    def diffeqB(self, s, t, gBeta):
        """
        Differential eq. (5) considering axion-nucleon Bremsstrahlung
        """
        pre = 2*2.5*1e+52*(self.mPNS/1.4)**(5/3)*((self.rPNS/10)**(-2))
        cnu = -6*self.lumNuS(self.mPNS, self.rPNS, gBeta, s)/s
        calp = -self.lumAbrem(s)/s
        dsdt = (1/pre)*(cnu+calp)
        return dsdt
        
    def solveDEB(self, gBeta, time0, tinit=0.01, tfinal=200):
        """
        Solve for s(t) the differential eq. (5) 
        """
        si = self.entropy(self.mPNS, self.rPNS, gBeta, tinit, time0)
        t = np.linspace(tinit, tfinal, 10000)
        sol = odeint(self.diffeqB, si, t, args=(gBeta,))
        intsol = interpolate.interp1d(t.flat,sol.flat,kind='quadratic')
        return intsol
    
    def LtotB(self, gBeta, time0, tinit=0.01, tfinal=200):
        """
        Compute the Neutrino and Axion Luminosities [erg s-1] from the solution s(t) of diffeqB
        """
        t = np.linspace(tinit, tfinal, 10000)
        solDE = self.solveDEB(gBeta, time0, tinit, tfinal) 
        # Neutrino Luminosity
        lumNu = [self.lumNuS(self.mPNS, self.rPNS, gBeta, s) for s in solDE(t)]
        intlumNuB = interpolate.interp1d(t.flat,lumNu,kind='quadratic',fill_value="extrapolate") 
        self.intlumNuB = intlumNuB
        # Axion Luminosity 
        lumAx = [self.lumAbrem(s) for s in solDE(t)]
        intlumAxB = interpolate.interp1d(t.flat,lumAx,kind='quadratic',fill_value="extrapolate")    
        self.intlumAxB = intlumAxB
        return intlumNuB,intlumAxB
    
    def EtotB(self, gBeta, time0, tinit=0.01, tfinal=200):
        """
        Compute the total emitted energy [erg] by neutrinos and axions
        """
        # Neutrino Energy
        intlumNu = self.LtotB(gBeta, time0,tinit, tfinal)[0]
        EnuIntB = quad(intlumNu,tinit,150,full_output=1)
        EnuTotB = 6*EnuIntB[0]
        self.EnuTotB = EnuTotB
        # Axion Energy
        intlumA = self.LtotB(gBeta, time0,tinit, tfinal)[1]
        EaxionTotB = quad(intlumA,tinit,150,full_output=1)[0]
        self.EaxionTotB = EaxionTotB
        # Total emitted energy
        EtotB = EnuTotB + EaxionTotB
        return EnuTotB,EaxionTotB,EtotB
  
    def EmNuAxB(self, gBeta, time0, tinit=0.01, tfinal=500):
        """
        Compute the average neutrino energy [MeV]
        """
        t = np.linspace(tinit, tfinal, 1000)
        solDE = self.solveDEB(gBeta, time0, tinit, tfinal) 
        # Average energy of neutrinos eq. (A4)
        EnuM = np.asarray([self.EmNuS(self.mPNS, self.rPNS, gBeta, s) for s in abs(solDE(t))])
        intEnuM = interpolate.interp1d(t.flat,EnuM,kind='quadratic')    
        return intEnuM
    

    def calcPhase(self, gBeta, EnuTot, tinit=0.01, tfinal=500):
        """
        Compute
        
        - SM Neutrino Luminosity [erg s-1]
        - Total emmited energy [erg] 
        - Average Nu Energy [MeV] 
        
        considering axion-nucleon Bremsstrahlung
        """
        timeP = self.time0(self.mPNS, self.rPNS,gBeta, EnuTot)# time origin
        EtotP = self.EtotB(gBeta, timeP, tinit, tfinal)
        LumP = self.LtotB(gBeta, timeP, tinit, tfinal)
        EnuMP = self.EmNuAxB(gBeta, timeP, tinit, tfinal)
        return LumP,EtotP,EnuMP

    def calcAll(self, gBeta1, EnuTot1, gBeta2, EnuTot2, tinit=0.01, tfinal=500):
        """
        Final expressions considering an initial and a final phase
        """
        LumI,EtotI,EnuMI = self.calcPhase(gBeta1, EnuTot1, tinit, tfinal)
        LumF,EtotF,EnuMF = self.calcPhase(gBeta2, EnuTot2, tinit, tfinal)
        
        ## sum initial+final phase
        tarr=np.linspace(tinit, tfinal, 10000)
        # neutrino average energy
        EnuM = (LumI[0](tarr)+LumF[0](tarr))/(LumI[0](tarr)/EnuMI(tarr)+ LumF[0](tarr)/EnuMF(tarr))
        EnuMint = interpolate.interp1d(tarr,EnuM)  
        # neutrino luminosity
        LumNu = (LumI[0](tarr)+LumF[0](tarr))
        LumNuint = interpolate.interp1d(tarr,LumNu)  
        # axion luminosity
        LumAx = (LumI[1](tarr)+LumF[1](tarr))
        LumAxint = interpolate.interp1d(tarr,LumAx)  
        
        return LumAxint,LumNuint,EnuMint


        
class SNaxionBPrim(SNaxionBrem):
    """
    Analytic Functions for SN neutrino+axion emission considering:
    - axion-photon Primakoff interaction (coupling gagg)
    - axion-nucleon Bremsstrahlung (coupling gaNN)
    """
    def __init__(self,mPNS, rPNS, gagg, gaNN=0.5e-10, cB= 0., YpF=0.3):
        """
        intialization
        """
        name = "SNaxion"
        self.name = name
        self.mPNS = mPNS
        self.rPNS = rPNS
        self.gagg = gagg
        self.alphaAp = gagg**2/4./math.pi
        self.alphaAb = gaNN**2/4./math.pi
        self.cB = cB
        self.YpF = YpF
        self.intlumAs = {} # cache the result of the axion luminosity
        
    def kappa2(self, s, eps):
        Ckap= math.pi**3*par.AlphaEM/par.mN*(self.YpF/2)/30*1.4*1e-3*par.Msol*60**(-2)*(1.9732705e-16)**3
        return Ckap*(self.mPNS/1.4)**(-5/6)*((self.rPNS/10)**3)*s**(-3)*(np.sin(eps)/eps)**(-1)
    
    def Fint(self, x, s, eps):
        #Integrand of eq. (13)
        kap2 = self.kappa2(s, eps)
        return ((x**2+kap2)*np.log(1+x**2/kap2) - x**2)*(x/(np.exp(x)-1))
    
    def Fkap2(self, s, eps):
        # Equation (13)
        pre = self.kappa2(s, eps)/2./math.pi**2
        intF = quad(self.Fint,1e-4,50,args=(s, eps,))
        return pre*intF[0]
    
    def Lint(self, eps, s):
        return (np.sin(eps)/eps)**(2/3)*self.Fkap2(s,eps)*eps**2
   
    def lumAprim(self, s):
        # Integration of eq. (17)
        pre = par.CMeVQ*self.alphaAp*30**7*(self.mPNS/1.4)**(14/3)*((self.rPNS/10)**(-14))*s*4*math.pi*(self.rPNS/math.pi)**3
        intL = quad(self.Lint,1e-4,3.14,args=(s,))
        return pre*intL[0]*par.MeV2erg #erg s^-1 
    
    def intLumAprim(self):
        sarr = np.linspace(0.015, 2., 200)
        lumAs = [self.lumAprim(s) for s in sarr]
        intlumAs = interpolate.interp1d(sarr.flat,lumAs,kind='quadratic',fill_value="extrapolate") 
        self.intlumAs[(self.mPNS, self.rPNS)] = intlumAs
        return intlumAs
    
    def diffeq(self, s, t, gBeta):
        """
        Differential eq. (5) considering axion-photon Primakoff and axion-nucleon Bremsstrahlung
        """
        try: intLumA = self.intlumAs[(self.mPNS, self.rPNS)]
        except: intLumA = self.intLumAprim()
        pre = 2*2.5*1e+52*(self.mPNS/1.4)**(5/3)*((self.rPNS/10)**(-2))
        cnu = -6*self.lumNuS(self.mPNS, self.rPNS, gBeta, s)/s
        calp = -self.cB*(self.lumAbrem(s)/s)-(intLumA(s)/s)
        dsdt = (1/pre)*(cnu+calp)
        return dsdt
        
    def solveDE(self, gBeta, time0, tinit=0.01, tfinal=100):
        """
        Solve for s(t) the differential eq. (5) 
        """
        si = self.entropy(self.mPNS, self.rPNS, gBeta, tinit, time0)
        t = np.linspace(tinit, tfinal, 200)
        sol = odeint(self.diffeq, si, t, args=(gBeta,))
        sol = np.nan_to_num(sol.flat)
        intsol = interpolate.interp1d(t.flat,sol.flat,kind='linear')
        return intsol
    
    def Ltot(self, gBeta, time0, tinit=0.01, tfinal=100):
        """
        Compute the Neutrino and Axion Luminosities [erg s-1] from the solution s(t) of diffeq
        """
        try: intLumA = self.intlumAs[(self.mPNS, self.rPNS)]
        except: intLumA = self.intLumAprim()
        t = np.linspace(tinit, tfinal, 200)
        solDE = self.solveDE(gBeta, time0, tinit, tfinal) 
        # Neutrino Luminosity
        lumNu = np.asarray([self.lumNuS(self.mPNS, self.rPNS, gBeta, s) for s in abs(solDE(t))])
        lumNu[lumNu<1e-10] = 0
        intlumNu = interpolate.interp1d(t.flat,lumNu,kind='quadratic') 
        # Axion Luminosity 
        lumA = [self.cB*self.lumAbrem(s)+intLumA(s) for s in abs(solDE(t))]
        intlumA = interpolate.interp1d(t.flat,lumA,kind='quadratic')        
        return intlumNu,intlumA
    
    def Etot(self, gBeta, time0, tinit=0.01, tfinal=100):
        """
        Compute the total emitted energy [erg] by neutrinos and axions
        """
        LtotNuA = self.Ltot(gBeta, time0,tinit, tfinal)
        # Neutrino Energy
        intlumNu = LtotNuA[0]
        EnuInt = quad(intlumNu,tinit,100,full_output=1)
        EnuTot = 6*EnuInt[0]
        self.EnuTot = EnuTot
        # Axion Energy
        intlumA = LtotNuA[1]
        EaxionTot = quad(intlumA,tinit,100,full_output=1)[0]
        self.EaxionTot = EaxionTot
        # Total emitted energy
        Etot = EnuTot + EaxionTot
        return EnuTot,EaxionTot,Etot   
    
    def EmNuAx(self, gBeta, time0, tinit=0.01, tfinal=200):
        """
        Compute the average neutrino energy [MeV]
        """
        t = np.linspace(tinit, tfinal, 200)
        solDE = self.solveDE(gBeta, time0, tinit, tfinal) 
        # Average energy of neutrinos
        EnuM = np.asarray([self.EmNuS(self.mPNS, self.rPNS, gBeta, s) for s in abs(solDE(t))])
        intEnuM = interpolate.interp1d(t.flat,EnuM,kind='quadratic')    
        return intEnuM
    
    def calcPhasePB(self, gBeta, EnuTot, tinit=0.01, tfinal=500):
        """
        Compute
        
        - SM Neutrino Luminosity [erg s-1]
        - Total emmited energy [erg] 
        - Average Nu Energy [MeV] 
        
        considering axion-photon Primakoff and axion-nucleon Bremsstrahlung
        """
        timeP = self.time0(self.mPNS, self.rPNS, gBeta, EnuTot)# time origin
        EtotP = self.Etot(gBeta, timeP, tinit, tfinal)
        LumP = self.Ltot(gBeta, timeP, tinit, tfinal)
        EnuMP = self.EmNuAx(gBeta, timeP, tinit, tfinal)
        return LumP,EtotP,EnuMP

    def calcAllPB(self, gBeta1, EnuTot1, gBeta2, EnuTot2, tinit=0.01, tfinal=500):
        """
        Final expressions considering an initial and a final phase
        """        
        LumI,EtotI,EnuMI = self.calcPhasePB(gBeta1, EnuTot1, tinit, tfinal)
        LumF,EtotF,EnuMF = self.calcPhasePB(gBeta2, EnuTot2, tinit, tfinal)
        
        ## sum initial+final phase
        tarr=np.linspace(tinit, tfinal, 10000)
        # neutrino average energy
        EnuM = (LumI[0](tarr)+LumF[0](tarr))/(LumI[0](tarr)/EnuMI(tarr)+ LumF[0](tarr)/EnuMF(tarr))
        EnuMint = interpolate.interp1d(tarr,EnuM)  
        # neutrino luminosity
        LumNu = (LumI[0](tarr)+LumF[0](tarr))
        LumNuint = interpolate.interp1d(tarr,LumNu)  
        # axion luminosity
        LumAx = (LumI[1](tarr)+LumF[1](tarr))
        LumAxint = interpolate.interp1d(tarr,LumAx)  
        
        return LumAxint,LumNuint,EnuMint