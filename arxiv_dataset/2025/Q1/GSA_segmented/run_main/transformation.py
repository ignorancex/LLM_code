import chaospy
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class MvNormalTrans:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
    def trans(self):
        distr = chaospy.MvNormal(self.a,self.b)
        ss=distr.inv(self.c)
        return ss

class NormalTrans:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
    def trans(self):
        distr = chaospy.Normal(self.a,self.b)
        ss=distr.inv(self.c)
        return ss

class Bimodal_trans:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
    def trans(self):
        dist = chaospy.GaussianMixture([1500, 2200], covariances=[150, 220], weights=[0.6, 0.4])
        dist2 = chaospy.GaussianMixture([400, 600], covariances=[40, 60], weights=[0.7, 0.3])
        distr = chaospy.MvNormal(self.b,self.c)
        ss3=distr.inv(self.a[:,2:end])
        ss1=distr.inv(self.a[:,0])
        ss2=distr.inv(self.a[:,1])