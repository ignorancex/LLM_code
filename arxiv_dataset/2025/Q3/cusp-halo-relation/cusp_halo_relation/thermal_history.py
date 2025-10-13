import os
import numpy as np
from scipy.interpolate import CubicSpline

# units
kB_MeV = 8.61733326e-11 # MeV/K
mpl = 1.22089013e22 # MeV

class ThermalHistory(object):
  '''
  Calculations with the Standard Model thermal history. All units are in MeV,
  aside from the input CMB temperature T0_K, which is in Kelvin.
  
  We use data from arXiv:1606.07494 for temperatures between 10 MeV and 100 GeV
  and data from arXiv:1503.04935 for temperatures above 125 GeV.
  
  Parameters:
    
    T0_K: float
      CMB temperature today in Kelvin. Default is 2.725.
      
  '''
  def __init__(self,T0_K=2.725):
    self.T0 = kB_MeV * T0_K
    
    current_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    self.tab_logT, self.tab_g, self.tab_gs = np.loadtxt(current_path + '/data/SM_dof.txt').T
  
    self.__g_interp = CubicSpline(self.tab_logT,self.tab_g,bc_type='clamped',)
    self.__gs_interp = CubicSpline(self.tab_logT,self.tab_gs,bc_type='clamped',)
    self.tab_loga = np.log10(self.a(10.**self.tab_logT))
    
    self.tabr_loga,self.tabr_g, self.tabr_gs = self.tab_loga[::-1],self.tab_g[::-1], self.tab_gs[::-1]
    self.__g_at_a_interp = CubicSpline(self.tabr_loga,self.tabr_g,bc_type='clamped',)
    self.__gs_at_a_interp = CubicSpline(self.tabr_loga,self.tabr_gs,bc_type='clamped',)
    
    self.tab_logH = np.log10(self.H(10.**self.tab_logT))
    self.__g_at_H_interp = CubicSpline(self.tab_logH,self.tab_g,bc_type='clamped',)
    self.__gs_at_H_interp = CubicSpline(self.tab_logH,self.tab_gs,bc_type='clamped',)
    
  def g(self,T):
    '''
    Effective number of degrees of freedom for energy density, as a function of
    photon temperature T in MeV.
    '''
    logT = np.log10(T)
    return np.piecewise(logT,[logT<self.tab_logT[0],logT>self.tab_logT[-1]],[self.tab_g[0],self.tab_g[-1],self.__g_interp])

  def gs(self,T):
    '''
    Effective number of degrees of freedom for entropy density, as a function
    of photon temperature T in MeV.
    '''
    logT = np.log10(T)
    return np.piecewise(logT,[logT<self.tab_logT[0],logT>self.tab_logT[-1]],[self.tab_gs[0],self.tab_gs[-1],self.__gs_interp])

  def rhoR(self,T):
    '''Radiation energy density in MeV^4, given photon temperature T in MeV.'''
    return np.pi**2/30 * self.g(T) * T**4
  
  def H(self,T):
    '''
    Hubble rate in MeV during radiation domination, given photon temperature T
    in MeV.
    '''
    return np.sqrt(8*np.pi/3 * self.rhoR(T)/mpl**2)

  def a(self,T):
    '''Scale factor a at photon temperature T (in MeV).'''
    return (self.gs(self.T0)/self.gs(T))**(1./3)*self.T0/T
    
  def g_at_a(self,a):
    '''
    Effective number of degrees of freedom for energy density, as a function of
    scale factor a.
    '''
    loga = np.log10(a)
    return np.piecewise(loga,[loga<self.tabr_loga[0],loga>self.tabr_loga[-1]],[self.tabr_g[0],self.tabr_g[-1],self.__g_at_a_interp])

  def gs_at_a(self,a):
    '''
    Effective number of degrees of freedom for entropy density, as a function
    of scale factor a.
    '''
    loga = np.log10(a)
    return np.piecewise(loga,[loga<self.tabr_loga[0],loga>self.tabr_loga[-1]],[self.tabr_gs[0],self.tabr_gs[-1],self.__gs_at_a_interp])
  
  def T_at_a(self,a):
    '''Photon temperature T (in MeV) at scale factor a.'''
    return (self.gs_at_a(1.)/self.gs_at_a(a))**(1./3)*self.T0/a

  def rhoR_at_a(self,a):
    '''Radiation energy density in MeV^4 at scale factor a.'''
    return self.rhoR(self.T_at_a(a))

  def H_at_a(self,a):
    '''Hubble rate in MeV at scale factor a during radiation domination.'''
    return self.H(self.T_at_a(a))
    
  def g_at_H(self,H):
    '''
    Effective number of degrees of freedom for energy density, as a function of
    Hubble rate H in MeV during radiation domination.
    '''
    logH = np.log10(H)
    return np.piecewise(logH,[logH<self.tab_logH[0],logH>self.tab_logH[-1]],[self.tab_g[0],self.tab_g[-1],self.__g_at_H_interp])

  def gs_at_H(self,H):
    '''
    Effective number of degrees of freedom for entropy density, as a function
    of Hubble rate H in MeV during radiation domination.
    '''
    logH = np.log10(H)
    return np.piecewise(logH,[logH<self.tab_logH[0],logH>self.tab_logH[-1]],[self.tab_gs[0],self.tab_gs[-1],self.__gs_at_H_interp])
  
  def T_at_H(self,H):
    '''
    Photon temperature T (in MeV) when Hubble rate is H (in MeV) during
    radiation domination.
    '''
    return np.sqrt(H*mpl)/(8*np.pi**3/90 * self.g_at_H(H))**(1./4)
  
  def a_at_H(self,H):
    '''Scale factor a when Hubble rate is H (in MeV) during radiation domination.'''
    return self.a(self.T_at_H(H))
