import numpy as np

from xoa.commons.logger import *
from .sobol_lib import i4_sobol_generate

from .proto import SamplerProtype


class SobolSequenceSampler(SamplerProtype):
    def __init__(self, config, spec, verifier=None):
        self.name = 'Sobol sequences'

        if 'num_skips' in spec:
            spec['seed'] += spec['num_skips']

        if 'verification' in spec and spec['verification'] == True and verifier != None:
            self.set_verification = True
        else:
            self.set_verification = False

        super(SobolSequenceSampler, self).__init__(config, spec, verifier)

    def generate(self):
        if self.set_verification == True:
            n_s = int(self.num_samples * 1.5)
        else:
            n_s = self.num_samples
        
        s_grid = np.transpose(i4_sobol_generate(self.num_dim, n_s, self.seed))
        
        if self.set_verification == True:
            #debug("Original grid: {}".format(s_grid))            
            hpv_list = self.config.convert('grid', 'hpv_list', s_grid)
            n_hpvs = len(hpv_list)
            v_grid = []
            for i in range(n_hpvs):
                hpv = hpv_list[i]
                g = s_grid[i]
                hpv_dict = self.config.convert("arr", "dict", hpv)
                if self.verify(hpv_dict) == True:
                    #debug("valid cfg #{}: {}".format(i, g))
                    v_grid.append(g)
                if len(v_grid) >= self.num_samples:
                    break
            if len(v_grid) != self.num_samples:
                info("Size reduced after verification: {}/{}".format(len(v_grid), self.num_samples))
            
            v_grid = np.array(v_grid)            
            #debug("Verified grid: {}".format(v_grid))
            return v_grid
        else:
            return s_grid

