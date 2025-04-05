#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 12:34:14 2019
clphidelta as needed in eq. 4.8
@author: nessa
"""

from lab import *
from mpi4py import MPI
import pickle
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

paramfile = sys.argv[1]
params = pickle.load(open(paramfile,'rb'))

def lensing_kernel(xi, xmax):
    return (xmax - xi)/(xmax*xi) * (xmax > xi)*(1.+z_chi(xi)) 

r2d, t2d = np.meshgrid(t_,t_)

ts      = (t2d).flatten()
rs      = (r2d).flatten()

chimax     = params['chimax']
file_ext   = params['ext']


junksize = np.ceil(len(ts)/size)
max_num  = min((rank+1)*junksize,len(ts))
jjs      = np.arange(rank*junksize, max_num,dtype=np.int)
print(junksize,max_num)

Cl       = np.zeros((len(jjs),len(ell_)))
chimaxs  = np.zeros(len(jjs))
r_test = np.zeros(len(jjs))
w11, w12 = np.meshgrid(w1,w1)

I2_ltc=np.squeeze(I2_ltrc)
# chimax and kernels

n = 2

t1d = np.expand_dims(t_,1)
w1d = np.expand_dims(w1,1)

for jj_, jj in enumerate(jjs):
    r = rs[jj]
    t = ts[jj]
    chimax_   = r*t*chimax
    chi       = r*chimax
    chi1fac0  = D_chi(chi)
    chi1fac0  = chi1fac0*(chi)**(1.-(n+nu_n_.reshape(1, -1)))
    chi2fac00 = D_chi(chi*t1d)*lensing_kernel(chi*t1d,chimax_)
    chi2fac01 = D_chi(chi/t1d)*lensing_kernel(chi/t1d,chimax_)
    chi2fac01 = chi2fac01 * t1d**((n+nu_n_).reshape(1, -1)-2)
    chi2fac0  = chi2fac00 + chi2fac01

    chifacs   = w1d*chi1fac0* chi2fac0

    result    = np.zeros_like(ell_)
    for ii  in range(ell_.size):        
        result[ii] = np.real(np.sum(chifacs*I2_ltc[ii]))
    
    r_test[jj_] = r
    Cl[jj_] = result*1./np.pi**2/2.*prefac/2.*-2
    chimaxs[jj_] = chimax_

result  = comm.gather(Cl, root=0)
chimaxs = comm.gather(chimaxs,root=0)

if rank ==0:

    cl = np.vstack([result[ii] for ii in range(size)])
    chimaxs = np.vstack([chimaxs[ii] for ii in range(size)])
    print(cl.shape)
    cl = np.swapaxes(cl,0,1)
    print(cl.shape)
    cl = np.reshape(cl,(len(ell_),r2d.shape[0],r2d.shape[1]))
    print(cl.shape)
    chimaxs = np.reshape(chimaxs,(r2d.shape[0],r2d.shape[1]))
    print(chimaxs.shape)
    np.save('../G_matrices/clphidelta_parallel_M31b_auto_%s.npy'%file_ext,[cl,chimaxs])







