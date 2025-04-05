#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 12:34:14 2019

@author: nessa
"""

from lab import *
from mpi4py import MPI
import sys
import pickle

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

paramfile = sys.argv[1]

params = pickle.load(open(paramfile,'rb'))


file_ext = params['ext2']
chimax = params['chimax']


junksize = np.ceil(len(t_)/size)
max_num  = min((rank+1)*junksize,len(t_))
jjs      = np.arange(rank*junksize, max_num,dtype=np.int)
print(junksize,max_num)

Cl = np.zeros((len(jjs),len(ell_),len(t_)))
chimax_test = np.zeros(len(jjs))

def lensing_kernel(xi, xmax):
    return (xmax - xi)/(xmax*xi) * (xmax > xi) * (1.+z_chi(xi))


r2d, t2d = np.meshgrid(t_,t_)
w11, w12 = np.meshgrid(w1,w1)
# inflate by one dimensions (nu_n)
r2d, t2d = np.expand_dims(r2d, 2), np.expand_dims(t2d, 2)
w11, w12 = np.expand_dims(w11, 2), np.expand_dims(w12, 2)

n=0


for jj, chi1_max in enumerate((t_*chimax)[jjs]):
  for ii, chi2_max in enumerate((t_*chimax)):

    chi1fac0 = (lensing_kernel(r2d*chi1_max, chi1_max)*D_chi(r2d*chi1_max))
    chi1fac0 = chi1fac0 *(r2d*chi1_max)**(1-nu_n_.reshape(1, 1, -1))

    chi2fac00 = (lensing_kernel(t2d*r2d*chi1_max, chi2_max)*D_chi(r2d*t2d*chi1_max))
    chi2fac01 = (lensing_kernel(1./t2d*r2d*chi1_max, chi2_max)*D_chi(r2d*1./t2d*chi1_max))
    chi2fac01 = chi2fac01 * t2d**(n+nu_n_.reshape(1, 1, -1)-2)
    chi2fac0  = chi2fac00 + chi2fac01


    chifacs = w11*w12*chi1fac0* chi2fac0

    for nn  in range(ell_.size):
      Cl[jj][nn][ii] = chi1_max*np.real(np.sum(chifacs*I0_ltrc[nn]))

    chimax_test[jj] = chi1_max



result = comm.gather(Cl, root=0)
ranks  = comm.gather(rank, root =0)
chimax_test  = comm.gather(chimax_test, root =0)


if rank ==0:
    cl = np.vstack([result[ii] for ii in range(size)])
    chimax_test = np.vstack([chimax_test[ii] for ii in range(size)])
    print(chimax_test)
    cl = np.swapaxes(cl,0,1)
    print(cl.shape)
    cl*=(1./np.pi**2/2.*prefac**2/4.*2.**2)
    np.save('../G_matrices/clphiphi_parallel_%s'%file_ext,cl)
