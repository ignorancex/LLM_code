
#planet structure
import numpy as np
import constants as cs 
import planetstructure as ps 
import masslosstime as ml
import masslosstime_env as ml_env
import cpml_masslosstime as cp_ml
import primordial_radvalley as prv
import math
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


class Star():
    
    def __init__(self, mass, age, radius, Teff, mass_err=0, age_err=0, radius_err=0, Teff_err=0):
        self.mass = mass
        self.mass_err = mass_err
        self.age = age
        self.age_err = age_err
        self.radius = radius
        self.radius_err = radius_err
        self.Teff = Teff
        self.Teff_err = Teff_err
        
        

class Planet():
    
    def __init__(self, radius, period, isrocky, radius_err=0, period_err=0,  Xiron=1/3, albedo=0):
        self.radius = radius   
        self.radius_err = radius_err
        self.period = period
        self.period_err = period_err
        self.isrocky= isrocky
        self.Xiron = Xiron
        self.albedo=albedo
        self.calcMcore()
        self.PE = self.PE()
        self.CPML = self.CPML()
        
    def calcsemimajor(self,star):
        self.a = ((( cs.G *cs.Msun2g(star.mass)*(self.period*cs.d2s)**2)/(4*(np.pi)**2))**(1/3))
        
             
    def calcplanettemp(self,star):
        self.Teq = (1-self.albedo)**(1/4)*star.Teff*np.sqrt(cs.Rsun2cm(star.radius)/(2*self.a))   
    
    def calcMcore(self):
        if self.isrocky: 
            self.Mcore = ps.Rcore_to_Mcore(self.radius, self.Xiron)
            
    class PE:
        pass
    class CPML: 
        pass
           
    
    
    
class PlanetarySystem(): 
    
    
    def __init__(self, starparams, planetRockyparams, planetEnvparams, Tkh_PE=None, Tkh_CPML=1000, PRV_tscale=1, Mstar_err=0, Star_age_err=0, Rstar_err=0, Teff_err=0, Rrocky_err=0, Procky_err=0, Xironrocky=1/3, albedo_rocky=0, Renv_err=0, Penv_err=0, Xironenv=1/3, albedo_env=0 ):
        
        stardict={'mass_err':Mstar_err, 'age_err':Star_age_err, 'radius_err':Rstar_err, 'Teff_err':Teff_err}
        self.star=Star(*starparams, **stardict)
        
        planetRockydict={'radius_err':Rrocky_err, 'period_err':Procky_err, 'Xiron':Xironrocky, 'albedo':albedo_rocky}
        self.planetRocky=Planet(*planetRockyparams, isrocky=True, **planetRockydict)
        self.planetRocky.calcsemimajor(self.star)
        self.planetRocky.calcplanettemp(self.star)
       
        planetEnvdict={'radius_err':Renv_err, 'period_err':Penv_err, 'Xiron':Xironenv, 'albedo':albedo_env}
        self.planetEnv=Planet(*planetEnvparams, isrocky=False, **planetEnvdict)
        self.planetEnv.calcsemimajor(self.star)
        self.planetEnv.calcplanettemp(self.star)
        
        if Tkh_PE is None and self.star.mass >= 0.8: 
            Tkh_PE=100
        elif Tkh_PE is None and self.star.mass <=0.3: 
            Tkh_PE=1000
        elif Tkh_PE is None and 0.3<self.star.mass<0.8 : 
            Tkh_PE=300
        
        self.Tkh_PE=Tkh_PE
        self.Tkh_CPML=Tkh_CPML
        self.PRV_tscale= PRV_tscale
        
        self.check_params()
     
    def check_params(self):
        '''checks that all values input by the user are finite numbers'''
        params=[self.planetRocky.radius, self.planetRocky.period, self.planetEnv.radius, self.planetEnv.period, self.star.mass, self.star.age, self.star.Teff, self.star.radius, self.star.mass_err, self.star.age_err, self.star.radius_err, self.star.Teff_err, self.planetRocky.radius_err, self.planetRocky.period_err, self.planetEnv.radius_err, self.planetRocky.period_err]
        for i in range(len(params)):
            math.isfinite(params[i])
            

       
    def sample_params(self, N=1000):
        
        errs=np.array([self.star.mass_err, self.star.age_err, self.star.radius_err, self.star.Teff_err, self.planetRocky.radius_err, self.planetRocky.period_err, self.planetEnv.radius_err, self.planetRocky.period_err])
        self.N =int(N) if np.any(errs>0) else 1  #if no errors are given by the user set N to 1
        
        #make numpy array with vals from a guassian distribution whose standard dev is rad err
        self.planetRocky.Rcore_samp = np.random.normal(self.planetRocky.radius, self.planetRocky.radius_err, self.N)
        self.planetEnv.Rcore_samp = np.random.normal(self.planetEnv.radius, self.planetEnv.radius_err, self.N)
        self.star.radius_samp = np.random.normal(self.star.radius, self.star.radius_err, self.N)
             
        self.planetRocky.period_samp = np.random.normal(self.planetRocky.period, self.planetRocky.period_err, self.N)
        self.planetEnv.period_samp = np.random.normal(self.planetEnv.period, self.planetEnv.period_err, self.N)
             
        self.star.mass_samp = np.random.normal(self.star.mass, self.star.mass_err, self.N)
    
        self.star.age_samp = np.random.normal(self.star.age, self.star.age_err, self.N)
    
        self.star.Teff_samp = np.random.normal(self.star.Teff, self.star.Teff_err, self.N)
         
       
    def calc_min_mass_env(self, PE=True, CPML=False, PRV=False):
        '''Calculates an array of minimum mass estimates for the enveloped planet through Monte Carlo sampling'''
        
        self.planetEnv.minMcore_samps = np.zeros(self.N)
        self.planetRocky.eta_rockyvals= np.zeros(self.N)
        self.planetEnv.eta_envvals= np.zeros(self.N)
        
        for i in tqdm(range(self.N)):
            
            #define new system
            starparams=(self.star.mass_samp[i], self.star.age_samp[i], self.star.radius_samp[i], self.star.Teff_samp[i])
            planetRockyparams=(self.planetRocky.Rcore_samp[i], self.planetRocky.period_samp[i])
            planetEnvparams=(self.planetEnv.Rcore_samp[i], self.planetEnv.period_samp[i])
            
            
            sys = PlanetarySystem(starparams, planetRockyparams, planetEnvparams, self.Tkh_PE, self.Tkh_CPML, self.PRV_tscale)
            
            
            if (PE):
                
                try:
                    
                    masslosstime_rocky = ml.calc_masslosstime_rocky(sys, sys.planetRocky.radius, sys.planetRocky.Mcore, sys.planetRocky.a, sys.planetRocky.Teq, sys.planetRocky.Xiron, sys.Tkh_PE) 
                    min_mass=20
                    min_mass = ml_env.min_mass_env(sys.planetEnv.radius, sys.planetEnv.a, sys.planetEnv.Teq, sys.star.age, sys.Tkh_PE, sys.planetEnv.Xiron, masslosstime_rocky)
                    
                except: 
                    pass
                
            elif (CPML): 
                
                try:
                    masslosstime_rocky = cp_ml.calc_max_masslosstime_rocky(sys, sys.planetRocky.radius, sys.planetRocky.Mcore, sys.planetRocky.Teq, sys.planetRocky.Xiron, sys.Tkh_CPML)
                    min_mass=20
                    min_mass = ml_env.min_mass_env(sys.planetEnv.radius, sys.planetEnv.a, sys.planetEnv.Teq, sys.star.age, sys.Tkh_CPML, sys.planetEnv.Xiron, masslosstime_rocky, PE=False)
                except: 
                    pass
                
            else: 
                try:
                    X_rocky = prv.calc_X_iso( sys.planetRocky.Mcore, sys.planetRocky.a, sys.star.mass, sys.PRV_tscale, sys.planetRocky.Xiron, sys.planetRocky.period)
                    min_mass=20
                    min_mass = prv.calc_min_mass_env(sys, sys.planetEnv.radius, sys.planetEnv.Xiron,  sys.planetEnv.a, sys.star.mass, X_rocky, sys.PRV_tscale, sys.planetEnv.period)
                except: 
                    pass 
            
            self.planetEnv.minMcore_samps[i]=min_mass
           
            
            




            
            
            
            
            
            
            
    