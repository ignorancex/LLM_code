import numpy as np
import scipy.stats as stats
from cobaya.likelihood import Likelihood
from cobaya.theory import Theory

#########################################
class NonlinLike(Likelihood):
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
class NonlinTheory(Theory):
    data_file = None
    ftype = None
    #########################################
    def initialize(self):
        self.data_prods = np.loadtxt(self.data_file).T
        self.xvals = self.data_prods[0].copy() 
        if self.ftype == 1:
            self.func_output = self.func_type_1
        elif self.ftype == 2:
            self.func_output = self.func_type_2
        else:
            raise ValueError('self.type should be 1 or 2 in NonlinTheory()')
    #########################################

    #########################################
    def calculate(self,state, want_derived=False, **params_values_dict):
        state['model'] = self.func_output(params_values_dict)
    #########################################

    #########################################
    def func_type_1(self,param_dict):
        keys = list(param_dict.keys())
        output = param_dict[keys[0]]*np.ones_like(self.xvals)
        if len(keys) >= 2:
            output *= (np.fabs(self.xvals)+1e-10)**param_dict[keys[1]]
            if len(keys) >= 3:
                output *= np.sin(param_dict[keys[2]]+self.xvals)**2
        return output
    #########################################

    #########################################
    def func_type_2(self,param_dict):
        keys = list(param_dict.keys())
        output = param_dict[keys[0]]*np.ones_like(self.xvals)
        if len(keys) >= 2:
            output *= np.fabs(self.xvals - param_dict[keys[1]])
            if len(keys) >= 3:
                output *= np.cos(np.sqrt(param_dict[keys[2]]*np.fabs(self.xvals)))
        return output
    #########################################

    #########################################
    def get_model(self):
        return self.current_state['model']
    #########################################

    #########################################
    def get_allow_agnostic(self):
        return True
    #########################################
