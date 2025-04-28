#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yu Wang
"""
from scipy.stats import multivariate_normal
import autograd.numpy as np
from autograd import grad

class proposal_rw:
    """
    proposal distribution in random-walk kernel
    """
    def __init__(self, dimension, pre_condition, target_rate=0.234):
        self.dimension = dimension
        self.pre_condition = pre_condition
        self.target_rate = target_rate
        self.h0 = 2.4**2/dimension**(1/3)
        
    def q(self, x, y, h, G):
        """
        log-density of proposal distribution
        Parameters:
        ----------
        x: current state
        y: proposal state
        h: logarithm of step size
        G: pre-condition matrix
        """
        m = x
        v = np.exp(2*h) * G
        r = multivariate_normal.logpdf(y, mean = m, cov = v)
        return r
    
    def sample(self, x, h, G):
        """
        draw samples from proposal distribution
        Parameters:
        -----------
        x: current state
        h: logarithm of step size
        G: pre-condition matrix
        """
        m = x
        v = np.exp(2*h) * G
        if self.dimension==1:
            z = np.random.normal(m, v)
        else:
            z = np.random.multivariate_normal(mean = m, cov = v)
        return z


class proposal_mala:
    """
    proposal distribution in MALA kernel
    
    """
    def __init__(self, dimension, logdensity, grad_logdensity, pre_condition, target_rate=0.574):
        self.dimension = dimension
        self.logdensity = logdensity
        if grad_logdensity is None:
            grad_logdensity=grad(logdensity)
        else:
            self.grad_logdensity = grad_logdensity
        self.pre_condition = pre_condition
        self.target_rate = target_rate
        self.h0 = 2.4**2/dimension
        
    def q(self, x, y, h, G):
        """
        log-density of proposal distribution
        Parameters:
        ----------
        x: current state
        y: proposal state
        h: logarithm of step size
        G: pre-condition matrix
        """
        if self.dimension==1:
            m = x + 0.5 * np.exp(2*h) * G * self.grad_logdensity(x)
        else:
            m = x + 0.5 * np.exp(2*h) * np.diag(G) * self.grad_logdensity(x)
        v = np.exp(2*h) * G            
        r = multivariate_normal.logpdf(y, mean = m, cov = v)
        return r
    
    def sample(self, x, h, G):
        """
        draw samples from proposal distribution
        Parameters:
        -----------
        x: current state
        h: logarithm of step size
        G: pre-condition matrix
        """
        if self.dimension==1:
            m = x + 0.5 * np.exp(2*h) * G * self.grad_logdensity(x)
        else:
            m = x + 0.5 * np.exp(2*h) * np.diag(G) * self.grad_logdensity(x)
        v = np.exp(2*h) * G       
        if self.dimension==1:
            z = np.random.normal(m, v)
        else:
            z = np.random.multivariate_normal(mean = m, cov = v)
        return z
    
class proposal_barker:
    """
    proposal distribution in barker proposal kernel
    
    """
    def __init__(self, dimension, logdensity, grad_logdensity, pre_condition, target_rate=0.4):
        self.dimension = dimension
        self.logdensity = logdensity
        if grad_logdensity is None:
            grad_logdensity=grad(logdensity)
        else:
            self.grad_logdensity = grad_logdensity
        self.pre_condition = pre_condition
        self.target_rate = target_rate
        self.h0 = 2.4**2/dimension**(1/3)
        
    def q(self, x, y, h, G):
        """
        log-density of proposal distribution
        Parameters:
        ----------
        x: current state
        y: proposal state
        h: logarithm of step size
        G: pre-condition matrix
        """
        if self.dimension ==1:
            logpdf = multivariate_normal.logpdf(y-x, 0, np.sqrt(np.exp(2*h)))
            scale = 1+np.exp(-np.sqrt(G) * self.grad_logdensity(x) * (y-x))
            return logpdf-np.log(scale)
        else:
            G = np.diag(G)
            sqrtG = np.sqrt(G)
            inverse_sqrtG = np.reciprocal(sqrtG)
            c = sqrtG * self.grad_logdensity(x)
            logpdf = multivariate_normal.logpdf(y-x, 0, np.exp(2*h))
            scale = 1+np.exp(-c*(inverse_sqrtG*(y-x)))
            r = logpdf-np.log(scale)
            return np.sum(r)
    
    def sample(self, x, h, G):
        """
        draw samples from proposal distribution
        Parameters:
        -----------
        x: current state
        h: logarithm of step size
        G: pre-condition matrix
        """
        if self.dimension ==1:
            z = np.random.normal(0, np.exp(2*h), 1)
            scale = 1+np.exp(-np.sqrt(G) * self.grad_logdensity(x) * z)
            if np.log(np.random.uniform(0, 1)) <= -np.log(scale): 
                y = x+z
            else:
                y= x-z
            return y
        else:
            y = np.zeros(self.dimension)
            G = np.diag(G)
            sqrtG = np.sqrt(G)
            c = sqrtG * self.grad_logdensity(x)
            #z = np.random.normal(0, np.exp(2*h), self.dimension)
            m = np.zeros(self.dimension)
            v = np.exp(2*h) * np.identity(self.dimension)
            z = np.random.multivariate_normal(mean = m, cov = v)
            scale = 1+np.exp(-c*z)
            u = np.random.uniform(0, 1, self.dimension)
            p = 1/scale
            zt = sqrtG * z
            for i in range(self.dimension):
                if u[i]<=p[i]:
                    y[i]=x[i]+zt[i]
                else:
                    y[i]=x[i]-zt[i]
            return  y


class proposal_hmc:
    
    def __init__(self, dimension, logdensity, grad_logdensity, L, M,
                 target_rate=0.651):
        self.dimension = dimension
        self.logdensity = logdensity
        if grad_logdensity is None:
            grad_logdensity=grad(logdensity)
        else:
            self.grad_logdensity = grad_logdensity
        self.L = L
        self.target_rate = target_rate
        self.M = M
        self.h0 = 2.4**2/dimension**(1/4)
    
    def leapfrog(self, u, v, h):
        """Leapfrog integrator for Hamiltonian Monte Carlo.
    
        Parameters
        ----------
        u : np.floatX
            Initial position
        v : np.floatX
            Initial momentum
        h:  logrithm of step_size, float
            How long each integration step should be
        
        M: mass matrix
    
        Returns
        -------
        u, v : np.floatX, np.floatX
            New position and momentum
        """
        u, v = np.copy(u), np.copy(v)
        inverse_M = np.linalg.inv(self.M)
    
        v += np.exp(2*h) * self.grad_logdensity(u) / 2  # half step
        #leapfrog_steps=np.random.choice(np.arange(self.L)+1, 1, p=[1/self.L for i in range(self.L)])
        
        
        for _ in range(self.L - 1):
            u += np.exp(2*h) * inverse_M @ v  # whole step
            v += np.exp(2*h) * self.grad_logdensity(u)  # whole step
        u += np.exp(2*h) * inverse_M @ v  # whole step
        v += np.exp(2*h) * self.grad_logdensity(u) / 2  # half step
    
        # momentum flip at end
        return u, v
    
    def sample(self, u, h):
    
        v_0 = np.random.multivariate_normal(mean=np.zeros(self.dimension), 
                                            cov=self.M)
        
        u_new, v_new = self.leapfrog(u, v_0, h)
        
        return u_new, v_new, v_0








