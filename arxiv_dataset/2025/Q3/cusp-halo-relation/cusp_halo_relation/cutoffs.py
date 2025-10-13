import numpy as np
from scipy.integrate import simpson
from . import thermal_history

# units
Mpc = 1.56373831e35 # MeV^-1
Msol = 1.11580327e60 # MeV
rhoCrit_h2 = 2.7744948e11 # Msol/Mpc^3
kB4 = 188971.431 # Msol/Mpc^3/K^4

models_WDM = ['VA23','V05']

def transfer_WDM(k,model,mX,omegaX,h,spin=0.5):
  '''
  
  Warm dark matter transfer function T(k).
  
  Parameters:
    
    k: float or array
    
    model: 'VA23' or 'V05'
      - 'VA23': Vogel & Abazajian (2023), arXiv:2210.10753
      - 'V05': Viel et al. (2005), arXiv:astro-ph/0501562
    
    mX: float
      Dark matter mass in keV.
      
    omegaX: float
      Dark matter density parameter, OmegaX * h^2
      
    h: float
    
    spin: float
      1/2 or 3/2. Only relevant if model=='VA23'. Default is 1/2.
  
  Returns:
    
    T(k): array or float
  
  '''
  if model == 'VA23':
    p = {
      0.5:[0.0437,-1.188,1.049,2.012,0.2463],
      1.5:[0.0345,-1.195,1.025,2.012,0.2463],
      }[spin]
    alpha = p[0] * mX**p[1] * (omegaX/0.12)**p[4] * (h/0.6736)**p[3] * h**-1
    nu = p[2]
  elif model == 'V05':
    alpha = 0.070 * mX**-1.11 * (omegaX/0.1225)**0.11
    nu = 1.12
  else:
    raise Exception('invalid free-streaming model')
  return (1. + (alpha*k)**(2.*nu))**(-5./nu)

def transfer_G04(k,m,Td,ad,Hd,aeq,Heq):
  '''
  
  Cutoff transfer function for WIMP dark matter per arXiv:astro-ph/0309621.
  
  Parameters:
    
    k: float or array
      Wavenumber in MeV
    
    m: float
      Particle mass in MeV
    
    Td, ad, Hd: floats
      Photon temperature (in MeV), scale factor, and Hubble rate (in MeV) at
      kinetic decoupling.
    
    aeq, Heq: floats
      Scale factor and Hubble rate (in MeV) at matter-radiation equality.
      
  Returns:
    
    T(k): array or float
    
  '''
  kd = 1.8*(m/Td)**0.5*ad*Hd
  kfs = (m/Td)**0.5*aeq/ad/np.log(4*aeq/ad)*aeq*Heq
  return (1-2./3*(k/kfs)**2) * np.exp(-(k/kfs)**2-(k/kd)**2)

def fsl_analytic(m,ad,pd,aeq,Heq,):
  '''
  Streaming distance from scale factor ad to the present time for a particle
  with mass m and initial momentum pd. Here aeq and Heq are the scale factor
  and Hubble rate at matter-radiation equality. Result is in the same units as
  pd/(m*Heq).
  
  This expression is accurate roughly for ad > 1e-9.
  '''
  fslog = np.log(8./(1+np.sqrt(1+(pd/m)**2)) * aeq/ad)
  return np.sqrt(2)/(aeq**2*Heq) * pd*ad/m * fslog

class Cutoff(thermal_history.ThermalHistory):
  '''
  
  Parameters:
    
    model: str
      - 'VA23': Thermal relic warm dark matter per Vogel & Abazajian (2023)
        [arXiv:2210.10753]. Requires m only.
      - 'V05': Thermal relic warm dark matter per Viel et al. (2005)
        [arXiv:astro-ph/0501562]. Requires m and spin (1/2 or 3/2).
      - 'G04': WIMP dark matter per Green, Hofmann, Schwarz (2004)
        [arXiv:astro-ph/0309621]. Requires m and the decoupling time.
      - 'fs': Numerically integrate the free-streaming length and supply a
        custom cutoff shape. Requires shape, m, decoupling time, and pd.
    
    shape: callable
      Custom cutoff shape, specified as T(x) where x = l_fs k. Here l_fs is the
      numerically integrated free-streaming length.
    
    h, OmegaM, OmegaB, T_CMB, Neff: floats
      Cosmological parameters. Defaults are h=0.6774, OmegaM=0.3089,
      OmegaB=0.04886, T_CMB=2.725 [K], Neff=3.046.
      
    m: float
      Dark matter mass. Unit is keV for warm dark matter models, otherwise MeV.
    
    spin: float
      Dark matter spin. Only relevant for some models.
    
    Td, ad, Hd: floats
      Decoupling time, specified as photon temperature (in MeV), scale factor,
      or Hubble rate (in MeV), respectively. At most one may be specified. Only
      relevant for some methods.
      
    pd: float
      Characteristic momentum at decoupling. Default is pd=Td.
      
    verbose: boolean
      Default True. Change to False to suppress messages.
    
  '''
  def __init__(self,model,shape=None,h=0.6736,OmegaM=0.3089,OmegaB=0.04886,T_CMB=2.725,Neff=3.046,m=None,spin=None,Td=None,ad=None,Hd=None,pd=None,verbose=True):
    self.model, self.shape = model, shape
    self.m, self.spin = m, spin
    self.h, self.OmegaM, self.OmegaB = h, OmegaM, OmegaB
    
    if model in models_WDM: # For WDM, we don't need the SM thermal history
      if verbose:
        print('Cutoffs: Warm dark matter, model=%s'%model)
    else: # Otherwise, we do
      if verbose:
        print('Cutoffs: Cold dark matter, model=%s'%model)
      super().__init__()
      rhoCrit = rhoCrit_h2 * h**2
      rhoR = np.pi**2/30 * 2 * (1.+Neff*7./8*(4./11)**(4./3)) * kB4 * T_CMB**4 # Msol/Mpc^3
      OmegaR = rhoR / rhoCrit
      self.aeq = OmegaR/OmegaM
      self.Heq = np.sqrt(2) * self.H_at_a(self.aeq)
      if sum(x is not None for x in [Td,ad,Hd]) > 1:
        raise Exception('At most one of Td, ad, Hd may be specified.')
      elif ad is not None:
        Td = self.T_at_a(ad)
      elif Hd is not None:
        Td = self.T_at_H(Hd)
      self.m, self.Td, self.ad, self.Hd, self.pd = m, Td, self.a(Td), self.H(Td), (pd or Td)
      if model == 'fs':
        self.pc = self.pd*self.ad # comoving momentum
        # prepare integrand:
        self.tabr_a, self.tabr_H = 10.**self.tabr_loga, 10.**self.tab_logH[::-1]
        self.tabr_dsdloga = np.log(10.) * (1+(self.pc/(self.m*self.tabr_a))**-2)**-0.5/(self.tabr_a*self.tabr_H)
        #
        self.__aL, self.__aE = self.tabr_a[-1], self.tabr_a[0]
        self.__HL, self.__HE = self.tabr_H[-1], self.tabr_H[0]
        self.fsl_ad = self.fsl(self.ad)
  
  def transfer(self,k):
    '''
    The cutoff transfer function T(k) that multiplies the density contrast.
    Here k is in Mpc^-1.
    '''
    if self.model in models_WDM:
      return transfer_WDM(k,self.model,self.m,(self.OmegaM-self.OmegaB)*self.h**2,self.h,self.spin)
    if self.model == 'G04':
      return transfer_G04(k/Mpc,self.m,self.Td,self.ad,self.Hd,self.aeq,self.Heq)
    if self.model == 'fs':
      return self.shape(self.fsl_ad*k/Mpc)
  
  def fsl(self,a0):
    '''
    Numerically evaluate the streaming distance from scale factor a0 to the
    present time assuming a Standard Model thermal history. Result is in MeV.
    '''
    length = 0.
    if a0 < self.__aE: # analytic integral before table start (not really needed, this is super-Planckian)
      fac = 1/(self.__aE**2*self.__HE)/(np.sqrt(2)/(self.aeq**2*self.Heq))
      length += fac*fsl_analytic(self.m,a0,self.pc/a0,self.aeq,self.Heq,)
      length -= fac*fsl_analytic(self.m,self.__aE,self.pc/self.__aE,self.aeq,self.Heq,)
      a0 = self.__aE
    if a0 < self.__aL: # numerical integral over table
      ix = self.tabr_a >= a0
      length += simpson(self.tabr_dsdloga[ix],x=self.tabr_loga[ix])
      a0 = self.__aL
    # analytic integral after table end
    length += fsl_analytic(self.m,a0,self.pc/a0,self.aeq,self.Heq,)
    return length
  
  def __call__(self,k):
    return self.transfer(k)