import numpy as np
import scipy.stats as stats
import scipy.linalg as linalg
from cobaya.likelihood import Likelihood
from cobaya.theory import Theory

#########################################
class PolyCorrLike(Likelihood):
    data_file = None
    cov_file = None
    #########################################
    def initialize(self):
        self.data_prods = np.loadtxt(self.data_file).T
        self.xvals = self.data_prods[0].copy()
        self.data = self.data_prods[1].copy()
        self.cov_data = np.loadtxt(self.cov_file)
        self.invcov_data = linalg.inv(self.cov_data)
    #########################################

    #########################################
    def get_requirements(self):
        """ Theory code should return model array. """
        return {'model': None}
    #########################################

    #########################################
    def logp(self,**params_values_dict):
        """ params_values_dict should be dictionary (similar to cost_args in ASA), not used here. """
        residual = self.data - self.provider.get_model()
        chi2 = np.dot(residual,np.dot(self.invcov_data,residual))
        return -0.5*chi2
    #########################################

#########################################

    
#########################################
class PolyCorrTheory(Theory):
    data_file = None
    #########################################
    def initialize(self):
        self.data_prods = np.loadtxt(self.data_file).T
        self.xvals = self.data_prods[0].copy()        
    #########################################

    #########################################
    def calculate(self,state, want_derived=False, **params_values_dict):
        keys = list(params_values_dict.keys())
        output = np.sum(np.array([params_values_dict[keys[i]]*self.xvals**i for i in range(len(keys))]),axis=0)
        # parameter dictionary dynamically decides degree of polynomial
        state['model'] = output
    #########################################

    #########################################
    def get_model(self):
        return self.current_state['model']
    #########################################

    #########################################
    def get_allow_agnostic(self):
        return True
    #########################################



#########################################

