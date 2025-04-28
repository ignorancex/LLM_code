import numpy as np
import pandas as pd
import copy
from scipy.misc import derivative
from scipy.special import logsumexp
import math
from scipy.stats import beta
from scipy.special import expit
from scipy.special import psi
from scipy.stats import expon
from scipy.stats import gamma
from scipy.stats import poisson
from scipy.stats import bernoulli
from scipy.stats import uniform
from scipy.stats import multinomial
from scipy.stats import multivariate_normal
from scipy.stats import invgauss
from numpy.polynomial import legendre
import matplotlib.pyplot as plt
import scipy.integrate as si
from scipy import stats, optimize
import matplotlib.pyplot as plt
from scipy import stats, optimize
from IPython.display import display, Math, Latex, clear_output
import multiprocessing
from functools import partial


class HAWKES:
    def __init__(self, tt, T0):
        self.T0 = T0
        self.tt = np.array(tt)
        self.m = self.tt.size
        if self.m > 1:            
            self.dtt = np.tril( self.tt[1:, np.newaxis] - self.tt[np.newaxis, :self.m-1] )
            
        
        self.M = 2
        self.B = 2
        # self.T_phi = [1.0, 2.0, 4.0, 6.0] Wanna+beta3
        self.T_phi = [1.0, 2.0, 4.0, 6.0]
        self.beta_a = np.zeros([self.M,self.B])
        self.beta_b = np.zeros([self.M,self.B])
        self.beta_a = [[1,2],[5,2]]
        self.beta_b = [[2,4],[6,4]]
        self.DoF = 2 + self.M * self.B
        self.lamb_idx = 0
        self.mu_idx = 1
        self.w_idx = np.zeros(self.M * self.B,dtype=int)
        for i in range(self.M):
            for j in range(self.B):
                self.w_idx[i * self.B + j]=int(i * self.B + j + 2)

        
        self.hyperMean = np.zeros( (self.DoF, 1) )
        self.hyperVar  = 10 * np.ones( (self.DoF, 1) )
    
    ## Posterior inference 
    def getMinusLogPrior(self, thetas):
        nSamples = thetas.size // self.DoF
        thetas = thetas.reshape(self.DoF, nSamples)
        
        shift = thetas - self.hyperMean                
        tmp = 0.5 * np.sum( shift ** 2 / self.hyperVar, 0 )
        return tmp if nSamples > 1 else tmp.squeeze()
        
        
    def getIntensity(self, thetas):
        nSamples = thetas.size // self.DoF
        thetas1 = thetas.reshape(self.DoF, nSamples)
        thetas2 = np.maximum( np.minimum( thetas.reshape(self.DoF, nSamples), 50 ) , 0.1 ) 
        lamb = thetas2[self.lamb_idx,:]
        mus = thetas1[self.mu_idx,:]
        w = []
        if self.m > 1:
            for i in range(self.M):
                for j in range(self.B):
                    w.append(thetas1[self.w_idx[i * self.B + j],:])

            u = 0
            for i in range(self.M):
                for j in range(self.B):
                    u += w[i * self.B + j] * np.sum(beta.pdf(self.dtt[-1,:], a = self.beta_a[i][j], \
                        b = self.beta_b[i][j], scale = self.T_phi[i * self.B + j]))
                    
            h = mus + u - np.arange(self.m - 2,-1,-1)[:,np.newaxis]
            tmp = np.vstack(( lamb* expit(mus), lamb * expit(h)))
        elif self.m == 1:
            tmp = lamb * expit(mus)
        tmp = np.maximum(tmp, 1e-6)
        return tmp if nSamples > 1 else tmp.squeeze()

    def getCompensator(self, thetas):
        nSamples = thetas.size // self.DoF
        thetas1 = thetas.reshape(self.DoF, nSamples)
        thetas2 = np.maximum( np.minimum( thetas.reshape(self.DoF, nSamples), 100 ) , 0.1 ) 
        lamb = thetas2[self.lamb_idx,:]
        mus = thetas1[self.mu_idx,:]
        w = []
        if self.m > 1:
            for i in range(self.M):
                for j in range(self.B):
                    w.append(thetas1[self.w_idx[i * self.B + j],:])
            u = mus 
            dh = 0
            for i in range(self.M):
                for j in range(self.B):
                    u += w[i * self.B + j] * np.sum(beta.pdf(self.dtt[-1,:], a = self.beta_a[i][j], \
                            b = self.beta_b[i][j], scale = self.T_phi[i * self.B + j]))
                    dh += w[i * self.B + j] * np.sum(derivative(lambda x:beta.pdf(x, \
                            a = self.beta_a[i][j],b = self.beta_b[i][j], scale = self.T_phi[i * self.B + j]), \
                            x0=self.dtt[-1,:], dx = 0.1, n=1))
            h = mus + u
            tmp = np.zeros(h.size)
            for i in range(h.size):
                if dh[i] == 0:
                    dh[i] = -1e8
                if h[i] > 20:
                    tmp[i] = lamb[i] * h[i]/dh[i]
                else:
                    tmp[i] = lamb[i] * np.log(1 + np.exp(h[i])) / dh[i]


        elif self.m == 1:
            tmp =  lamb * expit(mus) * (self.tt - self.T0)
        return tmp if nSamples > 1 else tmp.squeeze()
    
    def getMinusLogLikelihood(self, thetas):
        if self.m > 1:
            return self.getCompensator(thetas) - np.sum( np.log( self.getIntensity(thetas) ), 0 )
        else:
            return self.getCompensator(thetas) - np.log( self.getIntensity(thetas) )
    
    def getMinusLogPosterior(self, thetas):
        if self.m == 0:
            return self.getMinusLogPrior(thetas)
        else:
            return self.getMinusLogPrior(thetas) + self.getMinusLogLikelihood(thetas)
    
    def getGradientMinusLogPrior(self, thetas):        
        nSamples = thetas.size // self.DoF
        thetas = thetas.reshape(self.DoF, nSamples)
        tmp = (thetas - self.hyperMean) / self.hyperVar
        return tmp if nSamples > 1 else tmp.squeeze()
    
    def getGradientLogIntensity(self, thetas):
        nSamples = thetas.size // self.DoF
        thetas1 = thetas.reshape(self.DoF, nSamples)
        thetas2 = np.maximum( np.minimum( thetas.reshape(self.DoF, nSamples), 50 ) , 0.1 ) 
        lamb = thetas2[self.lamb_idx,:]
        mus = thetas1[self.mu_idx,:]
        w = []
        for i in range(self.M):
            for j in range(self.B):
                w.append(thetas1[self.w_idx[i * self.B + j],:])
        if (nSamples > 1) or (self.m == 1):
            lams = self.getIntensity(thetas)
        else:
            np.array(self.getIntensity(thetas))[:,np.newaxis]
    
        if self.m > 1:
            Phi = np.zeros(self.M * self.B)
            u = mus 
            dh = 0
            for i in range(self.M):
                for j in range(self.B):
                    u += w[i * self.B + j] * np.sum(beta.pdf(self.dtt[-1,:], a = self.beta_a[i][j], \
                            b = self.beta_b[i][j], scale = self.T_phi[i * self.B + j]))
                    dh += w[i * self.B + j] * np.sum(derivative(lambda x:beta.pdf(x, \
                            a = self.beta_a[i][j],b = self.beta_b[i][j], scale = self.T_phi[i * self.B + j]), \
                            x0=self.dtt[-1,:], dx = 0.1, n=1))
                    Phi[i * self.B + j] = np.sum(beta.pdf(self.dtt[-1,:], a = self.beta_a[i][j], b = self.beta_b[i][j], \
                        scale = self.T_phi[i * self.B + j]))
            h = mus + u

            gllams = np.zeros( (self.DoF, self.m, nSamples ) )
            gllams[self.lamb_idx,:,:]  =expit(h) * lamb  / lams 
            
            for j in range(len(gllams[self.mu_idx,:,:])):
                for i in range(h.size):
                    if -h[i] > 50:
                        gllams[self.mu_idx,:,:][j][i] = 0
                    else:
                        gllams[self.mu_idx,:,:][j][i]  = mus[i] / lams[j][i] * np.exp(-h[i])/(1 + np.exp(-h[i]))**2
            
            for i in range(self.M):
                for j in range(self.B):
                    for m in range(len(gllams[self.w_idx[i * self.B + j],:,:])):
                        for k in range(h.size):
                            if -h[k] > 50:
                                gllams[self.w_idx[i * self.B + j],:,:][m][k] = 0
                            else:
                                gllams[self.w_idx[i * self.B + j],:,:][m][k] = w[i * self.B + j][k] * np.exp(-h[k])/(1 + np.exp(-h[k]))**2 \
                                * np.vstack((np.zeros(nSamples), Phi[i * self.B + j] / lams[1:]) )[m][k] 
            return gllams
        elif self.m == 1:
            return np.vstack( (np.ones(nSamples), np.zeros( (self.DoF-1, nSamples) ) ) )
               
    def getGradientMinusLogLikelihood(self, thetas, *arg):
        nSamples = thetas.size // self.DoF
        thetas1 = thetas.reshape(self.DoF, nSamples)
        thetas2 = np.maximum( np.minimum( thetas.reshape(self.DoF, nSamples), 50 ) , 0.1 ) 
        lamb = thetas2[self.lamb_idx,:]
        mus = thetas1[self.mu_idx,:]
        w = []
        for i in range(self.M):
            for j in range(self.B):
                w.append(thetas1[self.w_idx[i * self.B + j],:])
        
        if len(arg) == 0:
            gllams = self.getGradientLogIntensity(thetas)
        else:
            gllams = arg[0]
        # Expressions    
        if self.m > 1:
            expmdeltasdtt = np.zeros(self.M * self.B)
            h = mus 
            dh = 0
            f = np.zeros(self.M * self.B)
            for i in range(self.M):
                for j in range(self.B):
                    h += w[i * self.B + j] * np.sum(beta.pdf(self.dtt[-1,:], a = self.beta_a[i][j], \
                            b = self.beta_b[i][j], scale = self.T_phi[i * self.B + j]))
                    dh += w[i * self.B + j] * np.sum(derivative(lambda x:beta.pdf(x, \
                            a = self.beta_a[i][j],b = self.beta_b[i][j], scale = self.T_phi[i * self.B + j]), \
                            x0=self.dtt[-1,:], dx = 0.1, n=1))
                    expmdeltasdtt[i * self.B + j] = np.sum(beta.pdf(self.dtt[-1,:], a = self.beta_a[i][j], b = self.beta_b[i][j], \
                        scale = self.T_phi[i * self.B + j]))
                    f[i * self.B + j] = np.sum(derivative(lambda x:beta.pdf(x, \
                            a = self.beta_a[i][j],b = self.beta_b[i][j], scale = self.T_phi[i * self.B + j]), \
                            x0=self.dtt[-1,:], dx = 0.1, n=1))

            gcomp = np.zeros( (self.DoF, nSamples) )
            for i in range(h.size):
                # dh[i] = np.minimum(dh[i], 1e-8)
                if dh[i] == 0:
                    dh[i] = -1e8
                if h[i] > 20:
                    gcomp[self.lamb_idx,:][i] = lamb[i] * h[i] / dh[i]
                else:
                    gcomp[self.lamb_idx,:][i] = lamb[i] * np.log(1 + np.exp(h[i])) / dh[i]

            gcomp[self.mu_idx,:] = mus * lamb * expit(h) / dh
            for i in range(self.M):
                for j in range(self.B):
                    for k in range(h.size):
                        # print(w[i * self.B + j].size,f[i * self.B + j].size,h.size,lamb.size,dh.size,expmdeltasdtt[i * self.B + j].size)
                        if f[i * self.B + j] == 0:
                            f[i * self.B + j] = -1e8
                        if dh[k] == 0:
                            dh[k] = -1e8
                        if w[i * self.B + j][k] == 0:
                            w[i * self.B + j][k] = -1e8
                        
                        if h[k] > 20:
                            gcomp[self.w_idx[i * self.B + j],:][k] = lamb[k] * w[i * self.B + j][k] *  (h[k] / (-f[i * self.B + j] \
                                    * w[i * self.B + j][k]**2) + expit(h[k]) * expmdeltasdtt[i * self.B + j] / dh[k])
                        else:
                            gcomp[self.w_idx[i * self.B + j],:][k] = lamb[k] * w[i * self.B + j][k] *  (np.log(1 + np.exp(h[k])) \
                                / (-f[i * self.B + j] * w[i * self.B + j][k]**2) + expit(h[k]) * expmdeltasdtt[i * self.B + j] / dh[k])

            tmp = gcomp - np.sum(gllams, 1)
        elif self.m == 1:
            gcomp = np.vstack( \
                        (lamb * expit(mus) * (self.tt - self.T0) * np.ones(nSamples), np.zeros( (self.DoF-1, nSamples) ) ) )
            tmp = gcomp - gllams
        return tmp if nSamples > 1 else tmp.squeeze()
    
    def getGradientMinusLogPosterior(self, thetas, *arg):
        nSamples = thetas.size // self.DoF
        if len(arg) == 0:
            gllams = self.getGradientLogIntensity(thetas)
        else:
            gllams = arg[0]
        tmp = self.getGradientMinusLogPrior(thetas) \
            + self.getGradientMinusLogLikelihood(thetas, gllams)
        return tmp if nSamples > 1 else tmp.squeeze()
    
    def getAsymptHessianMinusLogPosterior(self, thetas, *arg):
        nSamples = thetas.size // self.DoF
        if len(arg) == 0:
            gllams = self.getGradientLogIntensity(thetas)
        else:
            gllams = arg[0]
        tmp = np.sum( \
            gllams.reshape(self.DoF,1,self.m,nSamples) * \
            gllams.reshape(1,self.DoF,self.m,nSamples), 2) \
                + np.eye(self.DoF).reshape(self.DoF, self.DoF, 1) / self.hyperVar
        return tmp if nSamples > 1 else tmp.squeeze()       
                
    ## Prediction inference    
    def getPredIntensity(self, thetas, t, *arg):
        nSamples = thetas.size // self.DoF
        thetas1 = thetas.reshape(self.DoF, nSamples)
        thetas2 = np.maximum( np.minimum( thetas.reshape(self.DoF, nSamples), 50 ) , 0.1 ) 
        lamb = thetas2[self.lamb_idx,:]
        mus = thetas1[self.mu_idx,:]
        if self.m != 0:
            w = []
            for i in range(self.M):
                for j in range(self.B):
                    w.append(thetas1[self.w_idx[i * self.B + j],:])
            
            if len(arg) < 1:
                if self.m == 1:
                    self.dt_tt = t - self.tt
                else:
                    self.dt_tt = (t - self.tt)[:,np.newaxis]
            else:
                self.dt_tt = arg[0]
            
            h = mus
            for i in range(self.M):
                for j in range(self.B):
                    h += w[i * self.B + j] * np.sum( beta.pdf(self.dt_tt, \
                        a = self.beta_a[i][j], b = self.beta_b[i][j], scale = self.T_phi[i * self.B + j]) )
            
            tmp = lamb * expit(h)
           
        else:
            tmp = lamb * expit(mus)
        return tmp if nSamples > 1 else tmp.squeeze()
    
    def getPredCompensator(self, thetas, t, *arg):
        nSamples = thetas.size // self.DoF
        thetas1 = thetas.reshape(self.DoF, nSamples)
        thetas2 = np.maximum( np.minimum( thetas.reshape(self.DoF, nSamples), 50 ) , 0.1 ) 
        lamb = thetas2[self.lamb_idx,:]
        mus = thetas1[self.mu_idx,:]
        if self.m != 0:
            w = []
            for i in range(self.M):
                for j in range(self.B):
                    w.append(thetas1[self.w_idx[i * self.B + j],:])
            
            if len(arg) < 2:
                if self.m == 1:
                    self.dtm_tt = 0
                else:
                    self.dtm_tt = np.hstack( (0, self.dtt[-1,:]) )[:,np.newaxis]
            else:
                self.dtm_tt = arg[1]
            if len(arg) < 1:
                if self.m == 1:
                    self.dt_tt = np.array([t - self.tt])
                else:
                    self.dt_tt = (t - self.tt)[:,np.newaxis]
            else:
                self.dt_tt = arg[0]
            h1 = mus 
            dh1 = 0
            h2 = mus 
            dh2 = 0
            for i in range(self.M):
                for j in range(self.B):
                    h1 += w[i * self.B + j] * np.sum(beta.pdf(self.dt_tt, a = self.beta_a[i][j], \
                            b = self.beta_b[i][j], scale = self.T_phi[i * self.B + j]))
                    dh1 += w[i * self.B + j] * np.sum(derivative(lambda x:beta.pdf(x, \
                            a = self.beta_a[i][j],b = self.beta_b[i][j], scale = self.T_phi[i * self.B + j]), \
                            x0=self.dt_tt, dx = 0.1, n=1))
                    h2 += w[i * self.B + j] * np.sum(beta.pdf(self.dtm_tt, a = self.beta_a[i][j], \
                            b = self.beta_b[i][j], scale = self.T_phi[i * self.B + j]))
                    dh2 += w[i * self.B + j] * np.sum(derivative(lambda x:beta.pdf(x, \
                            a = self.beta_a[i][j],b = self.beta_b[i][j], scale = self.T_phi[i * self.B + j]), \
                            x0=self.dtm_tt, dx = 0.1, n=1))


            tmp1 = np.zeros(h1.size)
            tmp2 = np.zeros(h2.size)
            for i in range(h1.size):
                if dh1[i] == 0:
                    dh1[i] = -1e8
                if h1[i] > 20:
                    tmp1[i] = h1[i]/dh1[i] 
                else:
                    tmp1[i] = np.log(1 + np.exp(h1[i]))/dh1[i] 
            for i in range(h2.size):
                if dh2[i] == 0:
                    dh2[i] = -1e8
                if h2[i] > 20:
                    tmp2[i] = h2[i]/dh2[i]
                else:
                    tmp2[i] = np.log(1 + np.exp(h2[i]))/dh2[i]
            tmp = lamb * (tmp1 - tmp2)    

        else:
            tmp =  lamb * expit(mus) * (t - self.T0)
        return tmp if nSamples > 1 else tmp.squeeze()
    
    def getPredMinusLogLikelihood(self, thetas, t):
        tmp = np.maximum( np.minimum( self.getPredCompensator(thetas, t) \
                - np.log( self.getPredIntensity(thetas, t) + 1e-5), 100000 ), -100)
        return tmp

    def simulateNewEvent(self, thetas):
        nSamples = thetas.size // self.DoF
        thetas1 = thetas.reshape(self.DoF, nSamples)
        thetas2 = np.maximum( np.minimum( thetas.reshape(self.DoF, nSamples), 100 ) , 0.1 )
        lamb = thetas2[self.lamb_idx,:]
        mus = thetas1[self.mu_idx,:]
        S2 = - np.log( np.random.uniform(size = nSamples) ) / lamb
        if self.m > 0:
            w = []
            for i in range(self.M):
                for j in range(self.B):
                    w.append(thetas1[self.w_idx[i * self.B + j],:])
         
            m = 0
            intensity_sup = lamb
            tmp = np.zeros(nSamples)
            while(m <nSamples):
                for q in range(len(intensity_sup)):
                    r = - np.log( np.random.uniform(size = nSamples) ) / intensity_sup
                    r0 = r[q]
                    tt = self.tt
                    tt = np.insert(tt,-1,r0)
                    dtt = np.tril( tt[1:, np.newaxis] - tt[np.newaxis, :self.m-1] )
                    u = 0
                    for i in range(self.M):
                        for j in range(self.B):
                            u += w[i * self.B + j] * np.sum(beta.pdf(dtt[-1,:], a = self.beta_a[i][j], \
                                b = self.beta_b[i][j], scale = self.T_phi[i * self.B + j])) 
                                
                    h = mus + u
                    intensity_t = lamb * expit(h)
                    D = uniform.rvs(loc = 0,scale = 1)
                    ww = 0
                    while(np.mean(D * intensity_sup) > np.mean(intensity_t) and ww<500):
                        # print("r0",r0)
                        # print("np.mean(D * intensity_sup",np.mean(D * intensity_sup))
                        # print("np.mean(intensity_t)",np.mean(intensity_t))
                        intensity_sup = np.maximum( intensity_t , 1e-8 )
                        r1 = - np.log( np.random.uniform(size = 1) ) / intensity_sup
                        r11 = r1[q]
                        r2 = r0 + r11
                        r0 = r2
                        tt = self.tt
                        tt = np.insert(tt,-1,r0)
                        dtt = np.tril( tt[1:, np.newaxis] - tt[np.newaxis, :self.m-1] )
                        u = 0
                        for i in range(self.M):
                            for j in range(self.B):
                                u += w[i * self.B + j] * np.sum(beta.pdf(dtt[-1,:], a = self.beta_a[i][j], \
                                    b = self.beta_b[i][j], scale = self.T_phi[i * self.B + j])) 
                                    
                        h = mus + u
                        intensity_t = lamb * expit(h)
                        ww+=1

                    tmp[m] =  np.min( np.vstack( (r0 , S2[m]) ), 0 )
                    m += 1
        else:
            tmp = S2
        return tmp if nSamples > 1 else tmp.squeeze()

    
    def getMAP(self, *arg):
        x0 = np.random.normal(size = self.DoF) if len(arg) == 0 else arg[0]
        res = optimize.minimize(self.getMinusLogPosterior, x0, method='L-BFGS-B')
        return res.x
   ###########################################################################################################################################


### Stein variational Newton (SVN)

class SVCPDIn:
    def __init__(self, model, *arg):
        self.model = model
        self.DoF = model.DoF
        self.nParticles = 100
        self.nIterations = 30
        self.stepsize = 1
        self.MAP = self.model.getMAP( np.random.normal( size = self.DoF ) )[:,np.newaxis]
        if len(arg) == 0:
            self.resetParticles(np.arange(self.nParticles))
        else:
            self.particles = arg[0]
            
    def apply(self):
        maxshiftold = np.inf
        Q = np.zeros( (self.DoF, self.nParticles) )
        for iter_ in range(self.nIterations):
            gllams = self.model.getGradientLogIntensity(self.particles)
            gmlpt = self.model.getGradientMinusLogPosterior(self.particles, gllams)
            Hmlpt = self.model.getAsymptHessianMinusLogPosterior(self.particles, gllams) 
            M = np.mean(Hmlpt, 2)
            
            for i_ in range(self.nParticles):
                sign_diff = self.particles[:,i_,np.newaxis] - self.particles
                Msd   = np.matmul(M, sign_diff)
                kern  = np.exp( - 0.5 * np.sum( sign_diff * Msd, 0 ) )
                gkern = Msd * kern
                
                mgJ = np.mean(- gmlpt * kern + gkern , 1)
                HJ  = np.mean(Hmlpt * kern ** 2, 2) + np.matmul(gkern, gkern.T) / self.nParticles 
                try:
                    Q[:,i_] = np.linalg.solve(HJ, mgJ)
                    
                except Exception as e:
                    HJ1 = HJ + np.identity(6)*1e-2
                    try:
                        Q[:,i_] = np.linalg.solve(HJ1, mgJ)
                    except Exception as e:
                        Q[:,i_] = 1
                        print("奇异矩阵")
                
            self.particles += self.stepsize * Q
            nanidx = np.where(np.isnan(self.particles).any(0))[0]
            if nanidx.size > 0:
                self.particles[:,nanidx] = np.random.normal( size = (self.DoF,nanidx.size) )
                      
            maxshift = np.linalg.norm(Q, np.inf)
            if np.isnan(maxshift) or (maxshift > 1e20):
                self.resetParticles(np.arange(self.nParticles))
                self.stepsize = 1
            elif maxshift < maxshiftold:
                self.stepsize *= 1.01
            else:
                self.stepsize *= 0.9
            maxshiftold = maxshift
                          
    def resetParticles(self, idx):
        lenidx = len(idx) if isinstance(idx, int) == 0 else 1
        if lenidx == self.nParticles:
            self.particles = self.MAP + np.random.normal( scale = 1, size = (self.DoF, lenidx) )
        else:
            self.particles[:,idx] = self.MAP + np.random.normal( scale = 1, size = (self.DoF, lenidx) )   
            
###########################################################################################################################################       
### Stein variational Online Changepoint Detection (SVOCD)
##  Generalized BOCPD via SVN
      
class SVOCPDIn:
    def __init__(self, data):

        self.predictprob = []
        # Import data
        self.data = data
        self.nData = len(self.data)
        
        # Setup computational effort
        self.rmax = 50     # Max run length 
        self.npts = 100    # Number of posterior samples
        self.nrls = 100    # Number of run length samples
        
        # Setup credible interval
        self.flag_lci = 1      # Test left tail -> fire up drastic drecrease
        self.flag_rci = 1      # Test right tail -> fire up drastic increase
        self.risk_level_l = 4      # Left percentage of probability risk_H2     
        self.risk_level_r = 1      # Right percentage of probability risk_H2
        self.pred_mean = np.zeros(self.nData)                         # Predictive mean
        if self.flag_lci: self.percentile_l = np.zeros(self.nData)    # Left percentile (if flagged up)
        if self.flag_rci: self.percentile_r = np.zeros(self.nData)    # Right percentile (if flagged up)
                    
        # Changepoint prior
        self.rlr = 700              # Run length rate hyper-parameter -> decrease to weigh changepoint prob more_H2
        self.H   = 1 / self.rlr     # Hazard rate
        
        # Initialize run length probabilities
        self.jp  = 1       # Joint 
        self.log_jp = 0
        self.rlp = self.jp    # Posterior
        self.t0 = [0]
        
        # Initialize models
        self.model = {}
        self.model[0] = HAWKES([], 0)
        
        # Initialize Stein Variational Newton samplers
        self.svn   = {}        
        
        # Initialize posterior samples and probabilities
        self.pts = {}             
        self.pts[0] = np.random.normal( size = (self.model[0].DoF, self.npts) )       
        self.ptp = {}             
        self.ptp[0]   = np.exp( - self.model[0].getMinusLogPosterior( self.pts[0] ) )
        self.ptp[0]  /= np.sum( self.ptp[0] )
        
        # Risk tolerance
        self.riskTol = 10e-2 # bH2
        
        # Initialize changepoints
        self.changepoints = {}
    
    def apply(self):            
        for t_ in range(self.nData):
            print('Time:', t_)
            
            # Sample from posterior run length
            self.getRunLengthSamples(t_)    
            
            # Update models and samplers
            self.data2t = self.data[:t_]
            self.updateInferenceModels(t_)
            
            # Get predictive samples
            self.getPredictiveSamples(t_)
            
            # Get predictive statistics
            self.getPredictiveStats(t_)
            
            # Observe new data point
            datat = self.data[t_]
            
            # Check whether changepoint            
            self.checkIfChangepoint(t_, datat)
            
            # Get run length posterior           
            self.getRunLengthProbability(t_, datat)
        
    def getRunLengthSamples(self, t_):
        if t_ != 0:
            rld = stats.rv_discrete( name=None, values = ( np.arange( min(t_+1, self.rmax) ), self.rlp ) )
            self.rls = rld.rvs( size = self.nrls )
        else:
            self.rls = np.zeros(self.nrls)
            
    def updateInferenceModels(self, t_):
        if t_ != 0:
            self.trmax = min(t_ + 1, self.rmax)
            with multiprocessing.Pool(40) as pool:
                results = pool.map( self.updateInferenceModels2Pool, range(1, self.trmax) )
                for r_ in range(1, self.trmax): 
                    self.model[r_] = results[r_-1][0]
                    self.svn[r_]   = results[r_-1][1]
                    self.pts[r_]   = self.svn[r_].particles
                    q = - self.model[r_].getMinusLogPosterior( self.pts[r_] )
                    if np.max(q) > 50:
                        max_index = np.unravel_index(np.argmax(q), q.shape)
                        self.ptp[r_] =[0] * q.size
                        self.ptp[r_][max_index[0]] = 1
                        i = -2
                        max_plp = 1
                        num = 1
                        if sorted(q)[-100] != sorted(q)[-1]:
                            if sorted(q)[-2] == sorted(q)[-1]:
                                q_list = q.tolist()
                                num = q_list.count(sorted(q)[-1])
                                for k in range(len(np.where(q==sorted(set(q))[-1])[0])):
                                    self.ptp[r_][np.where(q==sorted(set(q))[-1])[0][k]] = 1/num
                                    max_plp = 1/num
                            else:
                                while(5 + sorted(set(q))[i] > sorted(set(q))[-1] and i >= -100):
                                    self.ptp[r_][np.where(q==sorted(set(q))[i])[0][0]] = sorted(set(q))[i] / sorted(set(q))[-1]
                                    self.ptp[r_][max_index[0]] = max_plp - self.ptp[r_][np.where(q==sorted(set(q))[i])[0][0]]/num
                                    i -= 1
                        else:
                            self.ptp[r_] =[1/q.size] * q.size
                        
                else:
                    self.ptp[r_] = np.exp( - self.model[r_].getMinusLogPosterior( self.pts[r_] ) )
                    if any(self.ptp[r_] != 0):
                        self.ptp[r_]  /= np.sum( self.ptp[r_] )
                    else:
                        self.ptp[r_] = [1/q.size] * q.size

                
    def updateInferenceModels2Pool(self, r_):
        if r_ < self.trmax - 1:
            model = HAWKES(self.data2t[-r_:], self.data2t[-r_-1])
        else:
            model = HAWKES(self.data2t[-r_:], 0)
        svn = SVCPDIn(model, self.pts[r_-1])
        svn.apply()
        return (model, svn)
    
    def getPredictiveSamples(self, t_):
        self.pps = np.array([])     
       
        r_vals, r_counts = np.unique(self.rls, return_counts = 1) 
        for k_ in range( len(r_vals) ): # PARALLELIZABLE!
            r_val = r_vals[k_]
            r_count = r_counts[k_]
            
            ptd = stats.rv_discrete( values = ( np.arange(self.npts), self.ptp[r_val] ) )
            thetas = self.pts[r_val][:,ptd.rvs(size = r_count)] 
            if t_ == 0:
                tmp = self.model[r_val].simulateNewEvent(thetas)
            else:
                tmp = self.data[t_-1] + self.model[r_val].simulateNewEvent(thetas)
            self.pps = np.hstack( ( self.pps, tmp ) )
               
    def getPredictiveStats(self, t_):   
        self.pred_mean[t_] = np.mean(self.pps)
        
        if self.flag_lci == 1:
            self.percentile_l[t_] = np.percentile(self.pps, self.risk_level_l)        
        if self.flag_rci == 1:
            self.percentile_r[t_] = np.percentile(self.pps, 100 - self.risk_level_r)            
            
    def getRunLengthProbability(self, t_, datat):   
        pp0 = np.mean( stats.expon.pdf(datat, scale = np.exp( - self.pts[0][0,:] ) ) )
        pp = np.hstack( (pp0, np.zeros( min(t_, self.rmax-1) )) )
        for r_ in range(1, min(t_, self.rmax)): # PARALLELIZABLE!
            pp[r_] = np.mean( np.exp( - self.model[r_].getPredMinusLogLikelihood( self.pts[r_], datat ) ) )
        if  t_ < self.rmax-1:
            if len(pp)==1:
                log_predictpp = np.maximum(-100, np.log(pp[-1]))
                self.predictprob.append(log_predictpp)
            else:
                log_predictpp = np.maximum(-100, np.log(pp[-2]))
                self.predictprob.append(log_predictpp)
        else:
            log_predictpp = np.maximum(-100, np.log(pp[-1]))
            self.predictprob.append(log_predictpp)
        pp1= np.maximum (pp, 1e-8)
        log_pp = np.log(pp1)  
 
        # Calculate run length posterior
        log_jppp = self.log_jp + log_pp                               # Joint x predictive prob
        log_gp = log_jppp + np.log( 1 - self.H )                      # Growth prob
        log_cp = logsumexp(log_jppp + np.log(self.H))                 # Changepoint prob
        new_log_jp = np.hstack( (log_cp, log_gp) )[:self.rmax]        # Joint prob
        log_rlp = new_log_jp                                          # Run length posterior
        log_rlp -= logsumexp(new_log_jp)
        self.rlp = np.exp(log_rlp)
        self.log_jp = new_log_jp                                      # pass message

                
    def checkIfChangepoint(self, t_, datat):
        bool_test = 0
        if self.t0[t_ - 1]>30 and t_>30:
            if self.flag_lci:           
                if datat < self.percentile_l[t_]: 
                    risk = np.abs( ( datat - self.percentile_l[t_] ) / ( self.pred_mean[t_] - self.percentile_l[t_] ) )
                    print("l",risk)
                    if risk > self.riskTol:
                        self.changepoints.update( {t_: risk} )   
                        print('Changepoint at time', t_, 'for drastic decrease')
                        bool_test = 1
                    
            if self.flag_rci: 
                if datat > self.percentile_r[t_]: 
                    risk = np.abs( ( datat - self.percentile_r[t_] ) / ( self.pred_mean[t_] - self.percentile_r[t_] ) )
                    print("r",risk)
                    if risk > self.riskTol:
                        self.changepoints.update( {t_: risk} )   
                        print('Changepoint at time', t_, 'for drastic increase')
                        bool_test = 1
        if bool_test == 1:
                if t_ == 0:
                    self.t0 = [0]
                else:
                    self.t0.append(0)
        else:
            if t_ == 0:
                self.t0 = [0]
            else:
                self.t0.append(self.t0[-1] + 1) 
                    
