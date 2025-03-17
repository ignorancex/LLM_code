import numpy as np

from xoa.commons.logger import *
from .proto import SamplerProtype


class UniformRandomSampler(SamplerProtype):
    def __init__(self, config, spec, verifier=None):
        self.name = 'uniform random sampling'        
        super(UniformRandomSampler, self).__init__(config, spec, verifier)

    def generate(self):
        np.random.seed(self.seed)
        random_grid = np.transpose(np.random.rand(self.num_dim, self.num_samples)) # FIXME: [0, 1) to [0, 1]
        
        return random_grid
        
