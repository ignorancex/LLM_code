import numpy as np
import healpy
import scipy
import os
import pickle
import sys
import argparse
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pymaster as nmt
import pyccl as ccl

# read command line arguments
parser = argparse.ArgumentParser(description='Calculate bias of QSOs and DLAs.')
parser.add_argument('dir', type=str, help='Base directory containing all required files.')
parser.add_argument('-M', type=int, help='maximum ell value, default is 1000')
parser.add_argument('-m', type=int, help='minimum bin value')
parser.add_argument('-b', type=int, help='bandpower width (25, 50, or 75); default is 50')
parser.add_argument('--sim', help='use random simulation (instead of data)',action="store_true")
parser.add_argument('--nsim', type=int,  help='simulation number (default is random)')
parser.add_argument('--simnum', type=int, help='number of simulations used (default is all available)')
args = parser.parse_args()

dir = args.dir

if args.M == None:
    max_ell = 1000
else:
    max_ell = args.M

if args.b == None:
    bsize = 50
else:
    bsize = args.b

#Find redshift distribution of dlas and qsos

dla = open(dir+'/data/DLA_DR12_v1.dat')
dla_ra = []; dla_dec = []; dla_z = []; qso_z = []

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

for line in dla:
    ra = [line.split()][0][2]
    dec = [line.split()][0][3]
    d = [line.split()][0][9]
    q = [line.split()][0][4]
    if isfloat(ra):
        dla_ra.append(float(ra))
        dla_dec.append(float(dec))
        dla_z.append(float(d))
        qso_z.append(float(q))

dz=.01
rdshift = np.arange(0, 7.2, dz)
n_dla = stats.gaussian_kde(dla_z,bw_method=0.1)(rdshift)
n_qso = stats.gaussian_kde(qso_z,bw_method=0.1)(rdshift)

#set ccl parameters
parameters = ccl.Parameters(Omega_c=0.27,Omega_b=0.045,h=0.69,sigma8=0.83,n_s=0.96)
cosmo = ccl.Cosmology(parameters)
lens = ccl.ClTracerCMBLensing(cosmo)

#use ccl to predict cls
bias_qso = np.ones(rdshift.size)
bias_dla = np.ones(rdshift.size)

b = nmt.NmtBin(2048,nlb= bsize)
ell_arr=b.get_effective_ells()

source_dla = ccl.ClTracerNumberCounts(cosmo, False, False, z=rdshift, n=n_dla, bias= bias_dla)
theory_dla = ccl.angular_cl(cosmo,lens,source_dla,ell_arr,l_limber=-1)

source_qso = ccl.ClTracerNumberCounts(cosmo, False, False, z=rdshift, n=n_qso, bias= bias_qso)
theory_qso = ccl.angular_cl(cosmo,lens,source_qso,ell_arr,l_limber=-1)


#import precalculated power spectra

if args.nsim == None:
    nsim = int(len(os.listdir(dir+'/pickled/sim_data'+str(bsize)+'/'))/2)
else:
    nsim = args.nsim


min_ell_bin = 0
max_ell_bin = (np.abs(ell_arr-max_ell)).argmin()
ells = ell_arr[min_ell_bin:max_ell_bin]

if args.sim:
    if args.simnum == None:
        N = np.random.randint(nsim)
    else:
        N = args.simnum
    print('simulation number:',N)
    data_qso = pickle.load(open(dir+'/pickled/sim_data'+str(bsize)+'/cps_qso_'+str(N).zfill(2)+'_'+str(bsize)+'.pkl','rb'))[0][min_ell_bin:max_ell_bin]
    data_dla = pickle.load(open(dir+'/pickled/sim_data'+str(bsize)+'/cps_dla_'+str(N).zfill(2)+'_'+str(bsize)+'.pkl','rb'))[0][min_ell_bin:max_ell_bin]
else:
    data_qso = pickle.load(open(dir+'/pickled/cps/cps_qso_'+str(bsize)+'.pkl','rb'))[0]
    data_dla = pickle.load(open(dir+'/pickled/cps/cps_dla_'+str(bsize)+'.pkl','rb'))[0]


#calculate full inverse covariance matrix for all ell bins

#Calculate error of Cls using simulations
sim_qso = np.zeros((len(ell_arr),nsim))
sim_dla = np.zeros((len(ell_arr),nsim))
for i in range(nsim):
    sim_qso[:,i] = pickle.load(open(dir+'/pickled/sim_data'+str(bsize)+'/cps_qso_'+str(i).zfill(2)+'_'+str(bsize)+'.pkl','rb'))[0]
    sim_dla[:,i] = pickle.load(open(dir+'/pickled/sim_data'+str(bsize)+'/cps_dla_'+str(i).zfill(2)+'_'+str(bsize)+'.pkl','rb'))[0]
average_qsoXqso = np.average(sim_qso**2,axis=1)
average_dlaXdla = np.average(sim_dla**2,axis=1)
average_qsoXdla = abs(np.average((sim_qso)*(sim_dla),axis=1))
qsoP0 = sim_qso[:len(sim_qso)-1,:]
qsoP1 = sim_qso[1:,:]
dlaP0 = sim_dla[:len(sim_qso)-1,:]
dlaP1 = sim_dla[1:,:]
average_qsoXqsoP1 = np.average(qsoP0*qsoP1,axis=1)
average_dlaXdlaP1 = np.average(dlaP0*dlaP1,axis=1)
diagP2 = np.zeros(2*len(ell_arr)-2)
diagP2[::2] = average_qsoXqsoP1
diagP2[1::2] = average_dlaXdlaP1

diag = np.zeros(2*len(ell_arr))
diag[::2] = average_qsoXqso
diag[1::2] = average_dlaXdla

diagP1 = np.zeros(2*len(ell_arr)-1)
diagP1[::2] = average_qsoXdla

Cov = np.zeros((2*len(ell_arr),2*len(ell_arr)))
np.fill_diagonal(Cov,diag)

CovP1 = np.zeros((2*len(ell_arr)-1,2*len(ell_arr)-1))
np.fill_diagonal(CovP1,diagP1)
Cov[1:,:len(Cov)-1] += CovP1
Cov[:len(Cov)-1,1:] += CovP1

CovP2 = np.zeros((2*len(ell_arr)-2,2*len(ell_arr)-2))
np.fill_diagonal(CovP2,diagP2)
Cov[2:,:len(Cov)-2] += CovP2
Cov[:len(Cov)-2,2:] += CovP2

iCov = scipy.linalg.inv(Cov)

if args.m == None:
    min_ell_bin = 0
else:
    min_ell_bin = int(args.m)

max_ell_bin = (np.abs(ell_arr-max_ell)).argmin()
ells = ell_arr[min_ell_bin:max_ell_bin]

iCovariance = iCov[2*min_ell_bin:2*max_ell_bin,2*min_ell_bin:2*max_ell_bin]

theory_q = np.zeros(2*len(ells))
theory_q[::2] = theory_qso[min_ell_bin:max_ell_bin]
theory_q[1::2] = theory_qso[min_ell_bin:max_ell_bin]

theory_d = np.zeros(2*len(ells))
theory_d[1::2] = theory_dla[min_ell_bin:max_ell_bin]

data = np.zeros(2*len(ells))
data[::2] = data_qso[min_ell_bin:max_ell_bin]
data[1::2] = data_dla[min_ell_bin:max_ell_bin]

def loglike(x):
    delta = x[0]*theory_q + x[1]*theory_d - data
    chi2 = np.dot(np.transpose(delta),np.dot(iCovariance,delta))
    return -chi2/2.0

def negloglike(x):
    return -loglike(x)

[b_qso, b_dla] = scipy.optimize.minimize(negloglike,np.array([0,0])).x
dof = 2*len(ells)
chi2 = -2*loglike([b_qso, b_dla])
prob = scipy.stats.chi2.cdf(chi2,dof)


iCovB = np.zeros((2,2))
iCovB[0,0] = np.dot(theory_q,np.dot(iCovariance,theory_q))
iCovB[1,0] = np.dot(theory_q,np.dot(iCovariance,theory_d))
iCovB[0,1] = np.dot(theory_q,np.dot(iCovariance,theory_d))
iCovB[1,1] = np.dot(theory_d,np.dot(iCovariance,theory_d))

CovB = scipy.linalg.inv(iCovB)
qso_error = np.sqrt(CovB[0,0])
dla_error = np.sqrt(CovB[1,1])



print('\n','b_qso:',str(np.around(b_qso,2))+' ± '+str(np.around(qso_error,2)),'\n','b_dla:',str(np.around(b_dla,2))+' ± '+str(np.around(dla_error,2)),'\n','dof:  ',dof,'\n','chi2: ',np.around(chi2,2),'\n','prob: ',np.around(prob,2),'\n')
