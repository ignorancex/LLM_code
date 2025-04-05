import numpy as np
import scipy as sp
#from numba import jit, jitclass, int32, complex128, boolean, float64


from scipy import stats
from matrix_util import *
from random_matrices import *
import matplotlib.pyplot as plt
from timer import Timer
from itertools import chain

import time
import logging

class CompoundWishart(object):
    """docstring for CompoundWishart."""
    """W = Z^*BZ : d x d """
    """Z: p x d """
    """B: p x p """
    """p >= d """
    def __init__(self,dim=1,p_dim=1, scale=1e-1, minibatch=1):
        super(CompoundWishart, self).__init__()
        self.minibatch = minibatch
        self.jobname = "default"
        self.scale= scale
        assert dim < p_dim or dim == p_dim
        self.dim = dim
        self.p_dim = p_dim
        self.b = np.zeros(p_dim)
        self.G= 1-1j
        self.forward_iter = 0   ###count number of iterations in forward

    def R_transform(self, w):
        r = 0
        b = self.b
        for i in range(self.p_dim):
                r += b[i]/(1 - b[i]*w)
        r /= self.dim
        return r

    def cauchy(self,init_G, z, max_iter=1000, thres=1e-8):
        g = init_G
        timer = Timer()
        timer.tic()
        for it in range(max_iter):
            sub = 1./(z - self.R_transform(g) ) -g
            if abs(sub) < thres:
                break
            g += 0.5*sub
        timer.toc()
        logging.debug("cauchy time={}/ {}-iter".format(timer.total_time, it))
        self.forward_iter += it
        return g



    def density(self, x_array):
        G = self.G
        num = len(x_array)
        rho_list = []
        for i in range(num):
            z = x_array[i] + 1j*self.scale
            G = self.cauchy(G, z)
            self.G = G
            rho =  -G.imag/sp.pi
            #logging.debug( "(density_signal_plus_noise)rho(", x, ")= " ,rho
            rho_list.append(rho)

        return np.array(rho_list)

    def ESD(self, num_shot, dim_cauchy_vec=0, COMPLEX = False):
        p = self.p_dim
        d = self.dim
        B = np.diag(self.b)
        evs_list = []
        for n in range(num_shot):
            Z = Ginibre(p, d, COMPLEX)
            W = Z.H @ B @ Z
            evs = np.linalg.eigh(W)[0]
            c_noise =  sp.stats.cauchy.rvs(loc=0, scale=self.scale, size=dim_cauchy_vec)
            if dim_cauchy_vec >0:
                for k in range(dim_cauchy_vec):
                    evs_list.append( (evs - c_noise[k]).tolist())
            else:
                evs_list.append(evs.tolist())
        out = list(chain.from_iterable(evs_list))
        return out

    def plot_density(self, COMPLEX=False, min_x = -50, max_x = 50,\
    resolution=0.2, dim_cauchy_vec = 1000,num_shot = 100,bins=100, jobname="plot_density"):

        evs_list = self.ESD(num_shot, COMPLEX=COMPLEX)
        length = len(evs_list)
        c_noise =  sp.stats.cauchy.rvs(loc=0, scale=self.scale, size=dim_cauchy_vec)
        for i in range(length):
            for j  in range(dim_cauchy_vec):
                evs_list.append(evs_list[i] - c_noise[j])
        plt.figure()
        plt.hist(evs_list, bins=bins, normed=True, label="ESD with cauchy noise")

        max_x = min(max_x, max(evs_list))
        min_x = max(min_x, min(evs_list))
        resolution = min(resolution,(max_x - min_x) /100)
        max_x += resolution*10
        num_step = (max_x - min_x )/resolution

        x_array = np.linspace(min_x,max_x, num_step)
        out_array =  self.density(x_array)
        plt.plot(x_array,out_array, label="theoretical value",color="green", lw = 2)
        plt.legend(loc="upper right")
        plt.savefig("images/plot_density/{}.png".format(jobname))
        plt.show()

        return x_array, out_array
    def gradients(self):
        G = self.G
        p = self.p_dim
        d = self.dim
        b = self.b
        TG_R = (float(p)/float(d))*np.average(( b/(1-b*G) )**2)
        grads_R = 1./(d*(1-b*G)**2)

        TG_Ge = G**2*TG_R
        grads_Ge = G**2*grads_R
        grads = grads_Ge/( 1 - TG_Ge)

        return grads

    def grad_loss(self, sample):
            num_sample = len(sample)
            rho_list = []
            grads = np.zeros(self.p_dim)
            for i in range(num_sample):
                x = sample[i]
                z = x+1j*self.scale
                G = self.cauchy(self.G, z)
                ### Update initial value of G
                self.G  = G
                rho =  -G.imag/sp.pi
                rho_list.append(rho)
                grads_G = self.gradients()
                ### (-log \rho)' = - \rho' / \rho
                grads += grads_G.imag/(sp.pi*rho)

            loss = np.average(-sp.log(rho_list))
            grads/= num_sample
            return grads, loss
