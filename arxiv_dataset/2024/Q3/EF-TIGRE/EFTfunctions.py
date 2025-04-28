#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#    Copyright (c)  Michele Mancarella <michele.mancarella@unimib.it>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.


### Logic for parametrizing EFT functions
import numpy as np



def Gamma(Om, Om0): 
    return (1-Om)/(1-Om0)
    
  
    
def time_evol(param, Om, Or, Om0, Or0 ): #xx, xin):
    
    return (1-Om-Or)/(1-Om0-Or0)*param


def regulator(vals, tol):
    is_scalar=False
    if np.isscalar(vals):
        is_scalar=True
        vals = np.array([vals,])
    
    to_reg = ( np.abs(vals)<=tol) 
    vals[ to_reg ]  = np.zeros(to_reg.sum())
    if not is_scalar:
        return vals
    else:
        return vals[0]





#################################################################################


class EFTfunBetaGamma(object):
    
    def __init__(self, alphaM0, alphaB0, betagamma, csign=1, tol=1e-15):
        
        raise NotImplementedError()
    
################################################################################




#################################################################################

  
class EFTfunGammac(object):
    
    def __init__(self, alphaM0, alphaB0, gammac0, tol=1e-15):
        
        self.alphaM0 = alphaM0
        self.alphaB0 = alphaB0
        self.gammac0 = gammac0
        self.tol=tol
    
        
    def alphaM(self, x, Om, Or, Om0, Or0):
        return time_evol(self.alphaM0, Om, Or, Om0, Or0) 
    
    def alphaB(self, x, Om, Or, Om0, Or0):
        return time_evol(self.alphaB0, Om, Or,Om0, Or0)
    
    def gammac(self, x, Om, Or, Om0, Or0): 
        return time_evol(self.gammac0, Om, Or, Om0, Or0)

    
    
