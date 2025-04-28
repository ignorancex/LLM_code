import delta2 
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np

class Bo:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def bo(self):
        sample_p = np.array(self.a)
        sample_q = np.array(self.b)
        Si=delta2.analyze( sample_p, sample_q, sample_p.shape[1])
        W = Si['delta']
        return W