import json
from scipy.signal import butter, filtfilt
import pandas as pd
from pandas import DataFrame, Series
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pandas.tools.plotting import scatter_matrix
from scipy.interpolate import interp1d

from preprocessing.preprocessing import PreProcessor
from preprocessing.ssa import SingularSpectrumAnalysis
import scipy.linalg as lin

class HMM:
    def __init__(self, config, pattern, data):
        self.pattern = pattern
        self.data = data
        self.A, self.pA, self.w, self.K = self.init()
        self.lam = .1  # precision parameter for the weights prior
        self.eta = 5.  # precision parameter for the autoregressive portion of the model
        self.N = 1  # number of sequences
        self.M = 2  # number of dimensions - the second variable is for the bias term
        self.T = len(self.data)  # length of sequences
        self.x = np.ones((self.T + 1, self.M))  # sequence data (just one sequence)
        self.x[0, 1] = 1
        self.x[1:, 0] = self.data
        # emissions
        self.e = np.zeros((self.T, self.K))
        # residuals
        self.v = np.zeros((self.T, self.K))
        # store the forward and backward recurrences
        self.f = np.zeros((self.T + 1, self.K))
        self.fls = np.zeros((self.T + 1))
        self.f[0, 0] = 1
        self.b = np.zeros((self.T + 1, self.K))
        self.bls = np.zeros((self.T + 1))
        self.b[-1, 1:] = 1. / (self.K - 1)
        # hidden states
        self.z = np.zeros((self.T + 1), dtype=np.int)
        # expected hidden states
        self.ex_k = np.zeros((self.T, self.K))
        # expected pairs of hidden states
        self.ex_kk = np.zeros((self.K, self.K))
        self.nkk = np.zeros((self.K, self.K))
        self.number_of_iteration = 5

    def init(self):
        step=5
        eps=.1
        dat=self.pattern[::step]
        K=len(dat)+1
        A=np.zeros((K,K))
        A[0,1]=1.
        pA=np.zeros((K,K))
        pA[0,1]=1.
        for i in xrange(1,K-1):
            A[i,i]=(step-1.+eps)/(step+2*eps)
            A[i,i+1]=(1.+eps)/(step+2*eps)
            pA[i,i]=1.
            pA[i,i+1]=1.
        A[-1,-1]=(step-1.+eps)/(step+2*eps)
        A[-1,1]=(1.+eps)/(step+2*eps)
        pA[-1,-1]=1.
        pA[-1,1]=1.

        w=np.ones( (K,2) , dtype=np.float)
        w[0,1]=dat[0]
        w[1:-1,1]=(dat[:-1]-dat[1:])/step
        w[-1,1]=(dat[0]-dat[-1])/step

        return A,pA,w,K


    def fwd(self):
        for t in xrange(self.T):
            self.f[t+1,:]=np.dot(self.f[t,:],self.A)*self.e[t,:]
            sm=np.sum(self.f[t+1,:])
            self.fls[t+1]=self.fls[t]+np.log(sm)
            self.f[t+1,:]/=sm
            assert self.f[t+1,0]==0

    def bck(self):
        for t in xrange(self.T-1,-1,-1):
            self.b[t,:]=np.dot(self.A,self.b[t+1,:]*self.e[t,:])
            sm=np.sum(self.b[t,:])
            self.bls[t]=self.bls[t+1]+np.log(sm)
            self.b[t,:]/=sm

    def em_step(self):
        x=self.x[:-1] #current data vectors
        y=self.x[1:,:1] #next data vectors predicted from current
        #compute residuals
        v=np.dot(x,self.w.T) # (N,K) <- (N,1) (N,K)
        v-=y
        self.e=np.exp(-self.eta/2*v**2,self.e)
        self.fwd()
        self.bck()
        # compute expected hidden states
        for t in xrange(len(self.e)):
            self.ex_k[t,:]=self.f[t+1,:]*self.b[t+1,:]
            self.ex_k[t,:]/=np.sum(self.ex_k[t,:])
        # compute expected pairs of hidden states
        for t in xrange(len(self.f)-1) :
            self.ex_kk=self.A*self.f[t,:][:,np.newaxis]*self.e[t,:]*self.b[t+1,:]
            self.ex_kk/=np.sum(self.ex_kk)
            self.nkk+=self.ex_kk
        # max w/ respect to transition probabilities
        self.A=self.pA+self.nkk
        self.A/=np.sum(self.A,1)[:,np.newaxis]
        # solve the weighted regression problem for emissions weights
        #  x and y are from above
        for k in xrange(self.K):
            self.ex=self.ex_k[:,k][:,np.newaxis]
            self.dx=np.dot(x.T,self.ex*x)
            self.dy=np.dot(x.T,self.ex*y)
            self.dy.shape=(2)
            self.w[k,:]=lin.solve(self.dx+self.lam*np.eye(x.shape[1]), self.dy)

        #return the probability of the sequence (computed by the forward algorithm)
        return self.fls[-1]

    def run_em_algorithm(self):
        for i in xrange(self.number_of_iteration):
            print self.em_step()

        # get rough boundaries by taking the maximum expected hidden state for each position
        self.r = np.arange(len(self.ex_k))[np.argmax(self.ex_k, 1) < 4]
        self.f = np.diff(np.diff(self.r))
        for i in range(0, len(self.f)):
            if (self.f[i] <= 0):
                self.r[i] = 0

    def plot_result(self):
        plt.plot(range(self.T), self.x[1:, 0])
        self.yr = [np.min(self.x[:, 0]), np.max(self.x[:, 0])]
        for i in self.r:
            plt.plot([i, i], self.yr, '-r')
        plt.show()
