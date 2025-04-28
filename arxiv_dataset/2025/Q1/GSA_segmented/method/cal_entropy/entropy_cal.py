import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from ennemi import estimate_mi
from ennemi import estimate_entropy
from divergence import entropy_from_samples

class SH_Entropy2:
    def __init__(self, a):
        self.a = a

    def entr2(self):
        sample_p = self.a
        return float(entropy_from_samples(sample_p, discrete=False))