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

chimax     = params['chimax']
file_ext   = params['ext']

LSST       = params['LSST']
z0_1       = params['z01'] 
sigma_z_1   = params['sigma_z1']
z0_2       = params['z02']
sigma_z_2   = params['sigma_z2']
bin_num1   = params['bin_num1']
bin_num2   = params['bin_num2']
bias       = params['bias'] 

if bias == 'simple':
    bias_func = simple_bias
elif bias =='constant':
    bias_func = constant_bias
else:
    print('no valid bias function selected')

if LSST:
    kernel1 = gal_clus(dNdz_LSST, b=bias_func, bin_num=bin_num1)
    kernel2 = gal_clus(dNdz_LSST, b=bias_func, bin_num=bin_num2)
else:
    kernel1 = gal_clus(Gauss_redshift(sigma_z=sigma_z_1,z0=z0_1), bias_func)
    kernel2 = gal_clus(Gauss_redshift(sigma_z=sigma_z_2,z0=z0_2), bias_func)


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

I2 = np.squeeze(I2_ltrc)

# inflate by one dimensions (nu_n)
t2 = np.expand_dims(t_, 1)
w12= np.expand_dims(w1, 1)

n=2

for jj_, jj in enumerate(jjs):
    chi      = (chimax*trs)[jj]
    #fill grid of shape t_,n
    #psi
    chi1fac0 = D_chi(chi)*(1.+z_chi(chi))
    chi1fac0 = chi1fac0 *(chi)**(1.-(n+nu_n_.reshape(1,-1)))
    #gal
    chi2fac00 = (kernel2(t2*chi)*D_chi(t2*chi))
    chi2fac01 = (kernel2(1./t2*chi)*D_chi(1./t2*chi))
    chi2fac01 = chi2fac01 * t2**(n+nu_n_.reshape(1, -1)-2)
    chi2fac0  = chi2fac00 + chi2fac01

    chifacs   = w12*chi1fac0* chi2fac0

    result    = np.zeros_like(ell_)
    for ii  in range(ell_.size):        
        result[ii] = np.real(np.sum(chifacs*I2[ii]))

    Cl[jj_]   = result*1./np.pi**2/2.*prefac/2. 
    Chis[jj_] = chi


result = comm.gather(Cl, root=0)
chis   = comm.gather(Chis,root=0)


if rank ==0:
    cl = np.vstack([result[ii] for ii in range(size)])
    chis = np.vstack([chis[ii] for ii in range(size)])
    cl = np.swapaxes(cl,0,1)
    cl = np.reshape(cl,(len(ell_),r2d.shape[0],r2d.shape[1]))
    chis = np.reshape(chis,(r2d.shape[0],r2d.shape[1]))
    np.save('../G_matrices/clpsigal_kernel2%s.npy'%file_ext,[cl,chis])

