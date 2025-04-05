#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
calculating clpsiphi term for Eq. 4.8/4.9
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

chimax = params['chimax']
chi_source = params['chisource']
file_ext = params['ext2']

def lensing_kernel(xi, xmax):
    return (xmax - xi)/(xmax*xi) * (xmax > xi) * (1.+z_chi(xi))

r2d, t2d = np.meshgrid(t_,t_)

trs      = (r2d*t2d).flatten()

junksize = np.ceil(len(trs)/size)
max_num  = min((rank+1)*junksize,len(trs))
if rank == size-1:
    max_num = len(trs)
jjs      = np.arange(rank*junksize, max_num,dtype=np.int)
print(junksize,max_num)

Cl       = np.zeros((len(jjs),len(ell_)))
Chis     = np.zeros(len(jjs))

I0 = np.squeeze(I0_ltrc)

# inflate by one dimensions (nu_n)
t2 = np.expand_dims(t_, 1)
w12= np.expand_dims(w1, 1)

n=0

for jj_, jj in enumerate(jjs):
    chi      = (chimax*trs)[jj]
    #fill grid of shape t_,n
    chi1fac0 = D_chi(chi)*(1.+z_chi(chi))
    chi1fac0 = chi1fac0 *(chi)**(1.-nu_n_.reshape(1,-1))

    chi2fac00 = (lensing_kernel(t2*chi, chi_source)*D_chi(t2*chi))
    chi2fac01 = (lensing_kernel(1./t2*chi, chi_source)*D_chi(1./t2*chi))
    chi2fac01 = chi2fac01 * t2**(n+nu_n_.reshape(1, -1)-2)
    chi2fac0  = chi2fac00 + chi2fac01

    chifacs   = w12*chi1fac0* chi2fac0

    result    = np.zeros_like(ell_)
    for ii  in range(ell_.size):        
        result[ii] = np.real(np.sum(chifacs*I0[ii]))

    Cl[jj_] = result*1./np.pi**2/2.*prefac**2/2. 
    Chis[jj_] = chi


result = comm.gather(Cl, root=0)
chis = comm.gather(Chis,root=0)


if rank ==0:
    cl = np.vstack([result[ii] for ii in range(size)])
    chis = np.vstack([chis[ii] for ii in range(size)])
    print(cl.shape)
    cl = np.swapaxes(cl,0,1)
    print(cl.shape)
    cl = np.reshape(cl,(len(ell_),r2d.shape[0],r2d.shape[1]))
    print(cl.shape)
    chis = np.reshape(chis,(r2d.shape[0],r2d.shape[1]))
    print(chis.shape)
    np.save('../G_matrices/clpsiphi_parallel_MB2_%s.npy'%file_ext,cl)
    np.save('../G_matrices/clpsiphi_parallel_MB2_chis_%s.npy'%file_ext,chis)

#factor 2 for every phi = -2 int W psi
#factor of 1/2 for every gaussian quadrature
