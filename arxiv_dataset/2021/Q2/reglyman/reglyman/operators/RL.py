import numpy as np

'''
Some operator that is needed for the computation of the Richardson-Lucy algorithmm in the context of Ly-alpha forest tomography
Implements the application of the adjoint of intergral operator for computing the effective optical depth
'''

class RLHydrogen:
    def __init__(self, op):
        self.op=op
        return
    
    def _eval(self, argument):
        toret=self.op._apply_kernel_adjoint(argument)
        return toret
    
class RLBar:
    def __init__(self, op):
        self.op=op
        return
    
    def _eval(self, argument):
        density_hydrogen=self.op._find_neutral_hydrogen_fraction(argument)
        toret=self.op._apply_kernel_adjoint(density_hydrogen)
        return toret


