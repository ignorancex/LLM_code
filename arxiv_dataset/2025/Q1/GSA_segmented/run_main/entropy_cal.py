import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from ennemi import estimate_mi
from ennemi import estimate_entropy
from divergence import entropy_from_samples

class SH_Entropy:
    def __init__(self, a):
        self.a = a

    def entr(self):
        return float(estimate_entropy(self.a))

class SH_Entropy2:
    def __init__(self, a):
        self.a = a

    def entr2(self):
        sample_p = self.a
        return float(entropy_from_samples(sample_p, discrete=False))

class Entropy2:
    def __init__(self, a):
        self.a = a

    def entr2(self):
        sample_p = self.a
        return float(entropy_from_samples(sample_p, discrete=False))

class Bimutual:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def bimu(self):
        return float(estimate_mi(self.a,self.b))


class Bimutual2:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def bimu(self):
        sample_x = self.a
        sample_y = self.b
        return float(mutual_information_from_samples(sample_x, sample_y))


class Borgonovo:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def bor(self):
        return float(cross2_entropy_from_samples(self.a,self.b))


class Condition_trimutual:
    def __init__(self, a, b ,c):
        self.a = a
        self.b = b
        self.c = c

    def con_trimu(self):
        return float(estimate_mi(self.a,self.b,cond=self.c))