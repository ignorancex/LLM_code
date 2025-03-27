# Copyright 2022 SciANN -- Ehsan Haghighat. 
# All Rights Reserved.
#
# Licensed under the MIT License.
# 
# Tensor class:
#   Abstract 3x3 tensorial operations, on vectorized data (batch),
#   with elasto-plasticity applications in mind.
# 


import numpy as np
import sciann as sn


class BaseTensor:
    """ BaseTensor class. 
    """
    eps = np.finfo(float).eps
    
    def __init__(self, lib):
        if lib=='numpy':
            self._lib = np
        elif lib=='sciann':
            self._lib = sn.math
        else:
            raise ValueError('"numpy" or "sciann"')
        self._t = None
        self._d = None
        self._v = None
        self._p = None
          
    def update(self, val):
        self._t = val.copy()
        pp = -sum(self._t) / 3.0
        self._v = -pp*3.0    
        self._p = pp
        self._d = np.array([ti + pp for ti in self._t])
        return self
    

    def t(self):
        return self._t 

    def d(self):
        return self._d
      

    def v(self):
        return self._v
    
    def dv(self):
        return np.ones(3)
    
    def p(self):
        return self._p
    
    def dp(self):
        return -np.ones(3)/3.0
    
    def q(self):
        J2 = sum([di**2 for di in self._d]) / 2.
        return self._lib.sqrt(3.0*J2 + self.eps)

    def q2(self):
        J2 = sum([di**2 for di in self._d]) / 2.
        return 3.0*J2
    
    def dq(self):
        q = self.q()
        return np.array([1.5*di / (q + self.eps) for di in self._d])

    def dq2(self):
        return np.array([3.*di for di in self._d])

    def eq(self):
        J2 = sum([di**2 for di in self._d]) / 2.
        return self._lib.sqrt(4./3.*J2 + self.eps)
    
    def deq(self):
        q = self.q()
        return np.array([2./3.*di / (q + self.eps) for di in self._d])
    
    def r(self):
        J3 = sum([di**3 for di in self._d])/3
        return self._lib.pow(J3*27./2., 1./3.)

    def dr(self):
        r = self.r()
        s2 = [di**2 for di in self._d]
        return np.array([4.5*(s2i - sum(s2)/3.)/(r**2 + self.eps) for s2i in s2])
    
    def cos3th(self):
        c = (self.r() / (self.q() + self.eps))**3
        return self.relu(c+1) - self.relu(c-1) -1
    
    def dcos3th(self):
        c = self.cos3th()
        q = self.q()
        r = self.r()
        dq = self.dq()
        dr = self.dr()
        dc_dr = np.array([dri / (q + self.eps) for dri in dr])
        dc_dq = np.array([-dqi*r/(q+self.eps)**2 for dqi in dq])
        return np.array([3*c**2*(X+Y) for X,Y in zip(dc_dr, dc_dq)])
    
    def th(self):
        c3th = self.cos3th()
        return sn._lib.acos(c3th)/3.
    
    def dth(self):
        c3th = self.cos3th()
        return np.array([- dci/(sn._lib.sqrt(1 - c3th**2 + self.eps) + self.eps) / 3. for dci in self.dcos3th()])

    def __call__(self):
        return self._t 

    def relu(self, x):
        if self._lib == np:
            return np.maximum(0, x)
        else:
            return sn.math.relu(x)
    

class NpTensor(BaseTensor):
    
    def __init__(self, val):
        super().__init__("numpy")
        self.update(val)
        
        
class SnTensor(BaseTensor):
    
    def __init__(self, val):
        super().__init__("sciann")
        self.update(val)
        
        