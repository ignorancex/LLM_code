from regpy.solvers import Solver

import logging
import numpy as np

'''
Richardson-Lucy deconvolution for Lyman-alpha forest tomography
'''


class Richardson_Lucy(Solver):
    def __init__(self, setting, rhs, init, richardson, cutoff=10**(-30), m=None):
        super().__init__()
        self.setting=setting
        self.rhs=rhs
        self.init=init
        self.x=init

        self.cutoff=cutoff
        self.richardson=richardson
        self.m=m or 1
        assert int(self.m)==self.m
        self.shape=rhs.shape
        self.size=richardson.op.N_space
        self.averaging_ind=np.arange(self.m, self.size-self.m)
        self.mean=np.zeros(self.size)

        self.y = self.setting.op(self.x)     
        self.f=1-self.rhs
        
        self.ind=self.f > 0      
        self.f=np.interp(setting.Hcodomain.discr.coords, setting.Hcodomain.discr.coords[0][self.ind], self.f[self.ind])
        self.f=self.f.reshape(self.shape)
        self.fr=np.empty(self.shape)
            
    def _next(self):        
        self.fr=1-self.y
        indices=[self.fr[i]==0 for i in range(np.size(self.fr))]
        self.fr[indices]=self.f[indices]

        nominator=self.richardson._eval(self.f/self.fr)
        denominator=self.richardson._eval(np.ones(self.shape))
        denominator=np.where(denominator==0, 10**(-100), denominator)
        
        for i in self.averaging_ind:
            self.mean[i]=1/(2*self.m+1)*np.sum(self.x[i-self.m:i+self.m+1])
            
        for i in np.arange(0, self.m):
            self.mean[i]=self.x[i]
            
        for i in np.arange(self.size-self.m, self.size):
            self.mean[i]=self.x[i]
        
        self.x=self.mean*nominator/denominator
        indices=[self.x[i]<self.cutoff for i in range(np.size(self.x))]
        self.x[indices]=self.init[indices]
        
        self.y = self.setting.op(self.x)
        
        if self.log.isEnabledFor(logging.INFO):
            norm_residual = np.linalg.norm(self.y-self.rhs)
            self.log.info('|residual| = {}'.format(norm_residual)) 
