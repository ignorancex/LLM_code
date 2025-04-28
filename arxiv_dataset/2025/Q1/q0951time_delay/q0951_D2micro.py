#!/usr/bin/env python
import numpy as np
from numpy.polynomial.polynomial import polyval
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib import rc
rc( 'text', usetex = True )

Nord = 6 #6 # 2 5 # microlensing polynomial order
# decorrelation length beta = beta0 + k*dbeta, k = 0,...,Nk-1 
beta0 =  5 # 5 10
Nk = 6 #6 #10 # 1 5
dbeta = 5

t1, A1, A1err, B1, B1err = np.loadtxt('q0951r_2008_2024LtUsnoDecamPs.dat', usecols=(0,1,2,3,4), unpack=True); tmin, tmax = 54300, 60500; ymin, ymax = 19.0, 18.55
A2, B2, A2err, B2err, t2 = np.loadtxt('Jakobsson_Paraficz.dat', usecols=(3,4,5,6,11), unpack=True); t2 -= 2400000.5; tmin, tmax = 51100, 52200; ymin, ymax = 18.1, 17.8

t = np.concatenate((t1, t2)) 
A = np.concatenate((A1, A2)); Aerr = np.concatenate((A1err, A2err))
B = np.concatenate((B1, B2)); Berr = np.concatenate((B1err, B2err))
tmin, tmax = 51000, 60500; ymin, ymax = 19.0, 17.8

ind = np.argsort(t)
t = t[ind]; A = A[ind]; Aerr = Aerr[ind]; B = B[ind]; Berr = Berr[ind]
N = len(t)
print('Nord Npoint', Nord, N)
tc = (min(t)+max(t))/2

Nl = 201
tau = np.arange(Nl) - (Nl-1)/2

m12 = np.zeros((N,N))
dt = np.zeros((N,N))
W = np.zeros((N,N))
t12 = np.zeros((N,N))
for i in range(N):
	for j in range(N):
		W[i,j] = 1./(Aerr[i]**2 + Berr[j]**2) 
		m12[i,j] = A[i] - B[j]
		t12[i,j] = t[i] - t[j]
		dt[i,j] = t[i] - tc

print('beta delay a0 D2min')
fig, ax = plt.subplots(1, 2, figsize=(8, 6))
for k in range(Nk):
    beta = beta0 + k*dbeta
    aij = np.zeros((Nord, Nord))
    bi = np.zeros(Nord)
    a = np.zeros((Nord,Nl))
    D2 = np.zeros(Nl)
    for l in range(Nl):
        S = np.exp(-(t12+tau[l])**2/beta**2)   # S = 1./(1.+(t12+tau[l])**2/beta**2)
        for i in range(Nord):
            bi[i] = -np.sum(m12*W*S*dt**i)
            for j in range(Nord):
                aij[i,j] = np.sum(W*S*dt**(i+j))
        z = np.linalg.solve(aij, bi)
        a[:,l] = z
        dm = m12 + polyval(dt, z)
        D2[l] = np.sum(dm**2*W*S)/np.sum(S*W)
    ind = np.argmin(D2)
    #tau0 = tau[ind]
    #D2min = D2[ind]
    a0 = a[:,ind]
    taus = np.linspace(min(tau), max(tau), (Nl-1)*10+1)
    f = interpolate.interp1d(tau, D2, kind = 'cubic') 
    D2s = f(taus)
    idx = np.argmin(D2s)
    tau0 = taus[idx]
    D2min = D2s[idx]
    print('%d %.1f %.3f %.3e' % (int(beta), tau0, a0[0], D2min))
    ax[0].plot(tau, D2, color = 'C'+str(k), label='$\\mathrm{\\beta}$ = '+str(beta)+ ' days')
    ax[1].errorbar(t+tau0, polyval(dt, a0)[:,0], color='C'+str(k), label='$\\mathrm{\\beta}$ = '+str(beta)+ ' days')

ymin, ymax = ax[0].get_ylim()
ax[0].text(0, ymin+0.75*(ymax-ymin), 'Nord = '+str(int(Nord)), va='center', ha='center', size='xx-large')
ax[0].set_xlabel('$\\Delta t_{AB}$ (days)', size='x-large')
ax[0].set_ylabel('$D^2_4$', size='x-large')
ax[0].set_xlim(min(tau), max(tau))

ax[1].legend(loc='upper right')
ax[1].yaxis.tick_right()
ax[1].yaxis.set_label_position("right")
ax[1].set_xlabel('MJD (days)', size='x-large')
ax[1].set_ylabel('$\\Delta m_{AB}$ (mag)', size='x-large')
ax[1].set_xlim(tmin, tmax)
ymax, ymin = ax[1].get_ylim()
ax[1].set_ylim(ymin, ymax)
plt.show()

ms = 4 #10
mew = 1; mec= 'k'
ymin, ymax = 19.0, 17.8
plt.errorbar(t+tau0, A+polyval(dt, a0)[:,0], yerr=Aerr, fmt='o', color='w', mew=mew, mec='C0', ms = ms, label='$m_A(t+\\Delta t)+\\Delta m(t)$', ecolor='grey')
plt.errorbar(t, B, yerr=Berr, fmt='s', color='C1', mew=mew, mec=mec, ms = ms, label='$m_B(t)$', ecolor='grey')
plt.legend(fontsize='large')
plt.xlabel('MJD (days)', size='x-large')
plt.ylabel('r-SDSS (mag)', size='x-large')
plt.xlim(tmin, tmax)
plt.ylim(ymin, ymax)
plt.show()



