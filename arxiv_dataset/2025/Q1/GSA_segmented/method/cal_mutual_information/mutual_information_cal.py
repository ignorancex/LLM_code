import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from ennemi import estimate_mi
from ennemi import estimate_entropy

class Bimutual:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def bimu(self):
        return float(estimate_mi(self.a,self.b))