#!/usr/bin/env python
import numpy as np
from numpy.polynomial.polynomial import polyval
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib import rc
rc( 'text', usetex = True )

t1, A1, A1err, B1, B1err = np.loadtxt('q0951r_2008_2024LtUsnoDecamPs.dat', usecols=(0,1,2,3,4), unpack=True); tmin, tmax = 54300, 60500; ymin, ymax = 19.0, 18.55
A2, B2, A2err, B2err, t2 = np.loadtxt('Jakobsson_Paraficz.dat', usecols=(3,4,5,6,11), unpack=True); t2 -= 2400000.5; tmin, tmax = 51100, 52200; ymin, ymax = 18.1, 17.8

t = np.concatenate((t1, t2)) 
A = np.concatenate((A1, A2)); Aerr = np.concatenate((A1err, A2err))
B = np.concatenate((B1, B2)); Berr = np.concatenate((B1err, B2err))
tmin, tmax = 51000, 60500; ymin, ymax = 19.0, 17.8

Nord = 6 #2 #5 #5 #6 #2 # microlensing polynomial order
# bin semisize alpha = alpha0 + k*dalpha, k = 0,...,Nk-1 
alpha0 =  5. # 5 10
Nk = 15 #6 # 1 5
dalpha = 1

N = len(t)
print('Nord Npoint', Nord, N)
ind = np.argsort(t)
t = t[ind]; A = A[ind]; Aerr = Aerr[ind]; B = B[ind]; Berr = Berr[ind]

t0 = (min(t)+max(t))/2

Nl = 201
tau = np.arange(Nl) - (Nl-1)/2

t12 = np.zeros((N,N))
for i in range(N):
	for j in range(N):
		t12[i,j] = t[i] - t[j]

fig, ax = plt.subplots(1, 2, figsize=(8, 6))
print('alpha delay a0 chi2min Npairs')
for k in range(Nk):
    alpha = alpha0 + k*dalpha
    cij = np.zeros((Nord, Nord))
    bi = np.zeros(Nord)
    a = np.zeros((Nord,Nl))
    chi2 = np.zeros(Nl)
    Np = np.zeros(Nl,int)
    for l in range(Nl):
        S = np.zeros((N,N))
        Bs = np.zeros(N)
        Bserr2 = np.zeros(N)
        ind = np.zeros(N, bool)
        #S = np.maximum(0., 1.-np.abs(t12+tau[l])/alpha)
        S[np.abs(t12+tau[l]) < alpha] = 1 # without weigthing
        for i in range(N):
            denom = np.sum(S[i,:])
            if denom > 0:
                Bs[i] = np.sum(S[i,:]*B)/denom
                Bserr2[i] = np.sum(S[i,:]**2*Berr**2)/denom**2
                ind[i] = True
        Ac = np.compress(ind, A)
        Acerr2 = np.compress(ind, Aerr**2)
        Bc = np.compress(ind, Bs)
        Bcerr2 = np.compress(ind, Bserr2)
        tc = np.compress(ind, t)
        dt = tc - t0
        W = 1./(Acerr2+Bcerr2)
        Np[l] = np.size(Ac)
        for i in range(Nord):
            bi[i] = -np.sum((Ac-Bc)*W*dt**i)
            for j in range(Nord):
                cij[i,j] = np.sum(W*dt**(i+j))
        z = np.linalg.solve(cij, bi)
        a[:,l] = z
        dm = Ac - Bc + polyval(dt, z)
        if Np[l] > Nord+1:
            chi2[l] = np.sum(dm**2*W)/(Np[l]-1-Nord)
    ind = np.argmin(chi2)
    tau0 = tau[ind]
    chi2min = chi2[ind]
    a0 = a[:,ind]
    taus = np.linspace(min(tau), max(tau), (Nl-1)*10+1)
    f = interpolate.interp1d(tau, chi2, kind = 'cubic') 
    chi2s = f(taus)
    idx = np.argmin(chi2s)
    print('%d %.1f %.3f %.2f %d' % (int(alpha), taus[idx], a0[0], chi2min, Np[ind]))
    ax[0].plot(tau, chi2, color = 'C'+str(k), label='$\\mathrm{\\alpha}$ = '+str(int(alpha))+' days')  
    ax[1].plot(t+tau0, polyval((t-t0), a0), color = 'C'+str(k), label='$\\mathrm{\\alpha}$ = '+str(int(alpha))+' days')
    
#ax[0].legend(loc='upper right')
ax[0].text(-50, 1.5, 'Nord = '+str(int(Nord)), va='center', ha='center', size='xx-large')
ax[0].hlines(1, min(tau), max(tau), colors='k', linestyles='dashed')
ax[0].set_xlabel('$\\Delta t_{AB}$ (days)', size='x-large')
ax[0].set_ylabel('$\\chi^2$', size='x-large')
ax[0].set_xlim(min(tau), max(tau))
ymin, ymax = 0, 8
ax[0].set_ylim(ymin, ymax)

ax[1].legend(loc='upper right')
ax[1].yaxis.tick_right()
ax[1].yaxis.set_label_position("right")
ax[1].set_xlabel('MJD (days)', size='x-large')
ax[1].set_ylabel('$\\Delta m_{AB}$ (mag)', size='x-large')
ax[1].set_xlim(tmin, tmax)
ymax, ymin = ax[1].get_ylim()
ax[1].set_ylim(ymin, ymax)


#plt.savefig('q0951_chi2micro.png', dpi=300)
plt.show()
