import numpy as np

from . import cusp_halo
from . import perturbations
from . import cutoffs

# units
rhoCrit_h2 = 2.7744948e11 # Msol/Mpc^3

def find_khm(cutoff):
  '''Find half-mode k for a tabulated/callable cutoff transfer function T(k).'''
  if callable(cutoff):
    k0 = 1e-5
    k1 = 1e-5
    while cutoff(k1) > 0.5:
      k1 *= 2
    k = np.exp(np.arange(np.log(k0),np.log(k1)+0.02,0.02))
    T = cutoff(k)
  else:
    k, T = cutoff
    sort = np.argsort(k)
    k, T = k[sort], T[sort]
  # find first place where T<0.5
  ihm = np.where(T<0.5)[0][0]
  khm =  np.exp(np.interp(0.5,T[ihm:ihm-2:-1],np.log(k[ihm:ihm-2:-1]),left=np.nan,right=np.nan))
  return khm

class CuspHaloStandard(cusp_halo.CuspHalo):
  '''
  
  Convenience class for evaluating the cusp-halo relation for a standard,
  concordance cosmology.
  
  Parameters:
  
    cutoff: tuple or callable
      Cutoff function T(k) that multiplies the Fourier-space density contrast.
      May be specified as a callable function T(k) or a table in the form
      (k, T).
      
    include_baryons: boolean
      Does baryonic mass contribute to halos, cusps, etc.?
      - True: accurate for k < 10^2 Mpc^-1; appropriate for warm dark matter.
      - False: accurate for k > 10^3 Mpc^-1; typically appropriate for cold
        dark matter.
    
    h, OmegaM, OmegaB: floats
      Cosmological parameters.
      Defaults are h=0.6774, OmegaM=0.3089, OmegaB=0.04886.
      
    n_s: float
      Primordial spectral index, default 0.9649.
      
    A_s, sigma8: floats
      Primordial spectral amplitude or sigma_8. If specified, sigma8 supersedes
      A_s. Default is A_s=2.100e-9.
    
    transfer: 'table' or 'EH'
      Transfer function to use for power spectrum. Default is 'table'.
      - 'table': use the supplied table, which was generated using CLASS with
        Planck (2018) cosmological parameters (same as our defaults).
      - 'EH': use the Eisenstein & Hu fitting formulae [arXiv:astro-ph/9709112].
    
    Mhm, khm, lhm: float
      Half-mode mass scale or wavenumber. If specified, this will shift the
      supplied cutoff in k, overriding any cutoff scale. The conventions are:
      - T(khm)=1/2, where T multiplies delta (not P).
      - Mhm = 4pi/3 (pi/khm)^3 rho
      - lhm = 2pi/khm
      
    verbose: boolean
      Default True. Change to False to suppress messages.
  
  '''
  
  def __init__(self,cutoff,include_baryons,h=0.6736,OmegaM=0.3089,OmegaB=0.04886,n_s=0.9649,A_s=2.100e-9,sigma8=None,transfer='table',Mhm=None,khm=None,lhm=None,verbose=True):
    
    # cosmology
    self.OmegaM = OmegaM
    self.h = h
    OmegaX = OmegaM - OmegaB
    rhoCrit = rhoCrit_h2 * h**2
    rhoM = rhoCrit * OmegaM
    rhoX = rhoCrit * OmegaX
    
    rho = rhoM if include_baryons else rhoX
    species = 2 if include_baryons else 0
    
    # cutoff
    khm_infer = find_khm(cutoff)
    if sum(x is not None for x in [Mhm,khm,lhm]) > 1:
      raise Exception('Only one of Mhm, khm, lhm may be specified')
    elif Mhm is not None:
      khm = np.pi * (4*np.pi/3 * rho/Mhm)**(1./3)
    elif lhm is not None:
      khm = 2*np.pi/lhm
    if khm is None:
      khm = khm_infer
    
    if callable(cutoff):
      T = lambda k: cutoff(k*khm_infer/khm)
    elif tuple(cutoff):
      cut_k, cut_T = cutoff
      cut_sort = np.argsort(cut_k)
      cut_k, cut_T = cut_k[cut_sort], cut_T[cut_sort]
      T = lambda k: np.interp(np.log(k*khm_infer/khm),np.log(cut_k),cut_T,left=1.,right=0.)
      
    self.khm = khm
    self.Mhm = 4*np.pi/3 * rho * (np.pi/khm)**3
    if verbose:
      print('CuspHaloStandard: khm = %e Mpc^-1, Mhm = %e Msol'%(self.khm,self.Mhm))
    
    # transfer and growth functions for CDM
    k = np.exp(np.arange(np.log(1e-5),np.log(1e3*khm),0.02)) # intervals of 0.02 in logk
    if transfer.lower() == 'eh':
      if not include_baryons:
        raise Exception('must include baryons if using EH transfer function')
      transfer = perturbations.transfer_EisensteinHu(k,h,OmegaM,OmegaB)[species]
      growth = lambda a1: perturbations.growth_late(a1,OmegaM,0.)
    elif transfer.lower() == 'table':
      table = perturbations.Transfer_table()
      transfer = table(k,1.)[species]
      growth = lambda a1,k1: table(k1,a1)[species]/table(k1,1.)[species]
    else:
      raise Exception('Invalid transfer fuction: %s'%transfer)
    
    # power spectrum for CDM
    P = A_s * (k/(0.05*h))**(n_s-1) * transfer**2
    if sigma8 is not None:
      P *= (sigma8/perturbations.sigma8(k,P,h))**2
    
    # initialize parent
    super().__init__(k,P*T(k)**2,growth=growth,rho=rho,amax=1.,verbose=verbose)
  
  def rhoCrit(self,a):
    '''
    Critical density as a function of scale factor a. Only relevant for halo
    concentrations.
    '''
    return rhoCrit_h2 * self.h**2 * (self.OmegaM*a**-3 + (1.-self.OmegaM))
    
  def m_at_z(self,M,z):
    '''Predicted cusp mass m for a halo of mass M at redshift z.'''
    return self.m(M,1./(1.+z))
  
  def A_at_z(self,M,z):
    '''Predicted cusp A for a halo of mass M at redshift z.'''
    return self.A(M,1./(1.+z))
  
  def c_at_z(self,z):
    '''Estimated typical concentration parameter of small halos at redshift z.'''
    return self.characteristic_c(1./(1.+z))
  
  def rhoCrit_at_z(self,z):
    '''
    Critical density as a function of redshift z. Only relevant for halo
    concentrations.
    '''
    return self.rhoCrit(1./(1.+z))
  
  def rho_at_z(self,z):
    '''Average density as a function of redshift z.'''
    return self.rho(1./(1.+z))
  
class CuspHaloWDM(CuspHaloStandard):
  '''
  
  Class for evaluating the cusp-halo relation for a cosmology with warm dark
  matter.
  
  Parameters:
  
    cutoff: string
      Warm dark matter power spectrum model to use. Default is 'VA23'.
      - 'VA23': Vogel & Abazajian (2023), arXiv:2210.10753
      - 'V05': Viel et al. (2005), arXiv:astro-ph/0501562
    
    mX: float
      Dark matter particle mass in keV. Either mX or Mhm may be specified, but
      not both.
    
    Mhm: float
      Half-mode mass scale. The convention is that T(khm)=1/2, where T
      multiplies delta (not P), and Mhm = 4pi/3 (pi/khm)^3 rho.
    
    h, OmegaM, OmegaB: floats
      Cosmological parameters.
      Defaults are h=0.6774, OmegaM=0.3089, OmegaB=0.04886.
      Note: we assume for simplicity that baryons contribute to halos and cusps
      just like dark matter.
      
    spin: 1/2 or 3/2
      Dark matter spin, only relevant if transfer=='VA23'. Default is 1/2.
      
    n_s: float
      Primordial spectral index, default 0.9649.
      
    A_s, sigma8: floats
      Primordial spectral amplitude or sigma_8. If specified, sigma8 supersedes
      A_s. Default is A_s=2.100e-9.
    
    transfer: 'table' or 'EH'
      Transfer function to use for power spectrum. Default is 'table'.
      - 'table': use the supplied table, which was generated using CLASS with
        Planck (2018) cosmological parameters (same as our defaults).
      - 'EH': use the Eisenstein & Hu fitting formulae [arXiv:astro-ph/9709112].
      
    verbose: boolean
      Default True. Change to False to suppress messages.
  
  '''
  def __init__(self,cutoff='VA23',mX=None,Mhm=None,h=0.6736,OmegaM=0.3089,OmegaB=0.04886,spin=0.5,n_s=0.9649,A_s=2.100e-9,sigma8=None,transfer='table',verbose=True):
    if sum(x is None for x in [mX,Mhm]) != 1:
      raise Exception('must specify exactly one of mX and Mhm')
    if mX is None:
      mX = 1. # dummy value if half-mode scale is specified instead
    T = cutoffs.Cutoff(cutoff,h=h,OmegaM=OmegaM,OmegaB=OmegaB,m=mX,spin=spin,verbose=verbose).transfer
    super().__init__(T,include_baryons=True,h=h,OmegaM=OmegaM,OmegaB=OmegaB,n_s=n_s,A_s=A_s,sigma8=sigma8,transfer=transfer,Mhm=Mhm,verbose=verbose)
    if verbose:
      print('CuspHaloWDM: khm = %e Mpc^-1, Mhm = %e Msol'%(self.khm,self.Mhm))
