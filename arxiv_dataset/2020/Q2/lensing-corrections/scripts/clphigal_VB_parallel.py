#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 12:34:14 2019
clphidelta as needed in eq. 4.6
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
#lensing kernel here should be independent of redshift distribution
def lensing_kernel(xi,xmax):
    return (xmax - xi)/(xmax*xi) * (xmax > xi)*(1.+z_chi(xi))

if params['bias'] == 'simple':
    bias_func = simple_bias
elif params['bias'] =='constant':
    bias_func = constant_bias
else:
    print('no valid bias function selected')



if params['LSST']:
    if params['bin_num'] == 'all':
        def galax_kernel(x):
            return lsst_kernel_cb(x)*bias_func(x)
    else:
        def galax_kernel(x):
            return lsst_kernel_cbn[params['bin_num']](x)*bias_func(x)
else:
    kernel = gal_clus(Gauss_redshift(sigma_z=params['sigma_z'],z0=params['z0']), bias_func)
    def galax_kernel(x):
        return kernel(x)


chimax   = params['chimax']
file_ext = params['ext']

r2d, t2d = np.meshgrid(t_,t_)

trs      = (r2d*t2d).flatten()



junksize = np.ceil(len(trs)/size)
max_num  = min((rank+1)*junksize,len(trs))
jjs      = np.arange(rank*junksize, max_num,dtype=np.int)
print(junksize,max_num)

Cl       = np.zeros((len(jjs),len(ell_)))
chimaxs  = np.zeros(len(jjs))
w11, w12 = np.meshgrid(w1,w1)
#nu axis
r2d, t2d = np.expand_dims(r2d, 2), np.expand_dims(t2d, 2)
w11, w12 = np.expand_dims(w11, 2), np.expand_dims(w12, 2)

# chimax and kernels

kernel1  = lensing_kernel
kernel2  = galax_kernel 

n = 2

for jj_, jj in enumerate(jjs):
    chimax_l = chimax*trs[jj]

    chi1fac0 = (kernel1(r2d*chimax_l,chimax_l)*D_chi(r2d*chimax_l))
    chi1fac0 = chi1fac0 *(r2d*chimax_l)**(1-(n+nu_n_.reshape(1, 1, -1)))

    chi2fac00 = (kernel2(t2d*r2d*chimax_l)*D_chi(r2d*t2d*chimax_l))
    chi2fac01 = (kernel2(1./t2d*r2d*chimax_l)*D_chi(r2d/t2d*chimax_l))
    chi2fac01 = chi2fac01 * t2d**((n+nu_n_).reshape(1, 1, -1)-2)
    chi2fac0  = chi2fac00 + chi2fac01

    chifacs = w11*w12*chi1fac0* chi2fac0

    result=np.zeros_like(ell_)
    lmax = ell_.size
    for ii  in range(ell_.size):        
        result[ii] = np.real(np.sum(chifacs*I2_ltrc[ii]))
    

    Cl[jj_] = chimax_l * result*1./np.pi**2/2.*prefac/4.*2.
    chimaxs[jj_] = chimax_l


result  = comm.gather(Cl, root=0)
chimaxs = comm.gather(chimaxs,root=0)


if rank ==0:
    cl = np.vstack([result[ii] for ii in range(size)])
    chimaxs = np.vstack([chimaxs[ii] for ii in range(size)])
    print(cl.shape)
    cl = np.swapaxes(cl,0,1)
    print(cl.shape)
    cl = np.reshape(cl,(len(ell_),len(t_),len(t_)))
    print(cl.shape)
    chimaxs = np.reshape(chimaxs,(len(t_),len(t_)))
    print(chimaxs.shape)
    np.save('../G_matrices/clphigal_%s.npy'%file_ext,cl)
    np.save('../G_matrices/clphigalchimaxs_%s.npy'%file_ext,chimaxs)
