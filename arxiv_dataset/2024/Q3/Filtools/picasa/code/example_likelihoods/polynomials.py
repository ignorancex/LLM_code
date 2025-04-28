import numpy as np
import scipy.stats as stats
from cobaya.likelihood import Likelihood
from cobaya.theory import Theory

#########################################
class PolyLike(Likelihood):
    data_file = None
    #########################################
    def initialize(self):
        self.data_prods = np.loadtxt(self.data_file).T
        self.xvals = self.data_prods[0].copy()
        self.data = self.data_prods[1].copy()
        self.err_data = self.data_prods[2].copy()
    #########################################

    #########################################
    def get_requirements(self):
        """ Theory code should return model array. """
        return {'model': None}
    #########################################

    #########################################
    def logp(self,**params_values_dict):
        """ params_values_dict should be dictionary (similar to cost_args in ASA), not used here. """
        model = self.provider.get_model()
        chi2 = np.sum(((self.data - model)/self.err_data)**2) # make robust to division
        return -0.5*chi2
    #########################################

#########################################

    
#########################################
class PolyTheory(Theory):
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

