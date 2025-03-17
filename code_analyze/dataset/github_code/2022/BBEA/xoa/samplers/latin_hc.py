import numpy as np
import random

from pyDOE import lhs

from xoa.commons.logger import *

from .proto import SamplerProtype

class LatinHypercubeSampler(SamplerProtype):
    def __init__(self, config, num_samples, verifier=None):
        self.name = 'Latin Hypercube sampling' 
        super(LatinHypercubeSampler, self).__init__(config, num_samples, verifier)

    def generate(self):        
        random.seed(self.seed)
        
        hypercube_grid = np.array(lhs(self.num_dim, samples=self.num_samples))        
        return hypercube_grid
