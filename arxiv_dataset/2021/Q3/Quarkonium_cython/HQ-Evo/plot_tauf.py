#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

M = 5
T = 0.3
E = 30.

def tau_f(E, T, kperp, etak):
	pz = np.sqrt(E**2-M**2)
	Ep = E + pz
	xb = kperp*np.exp(np.abs(etak))/Ep
	x = kperp*np.exp(etak)/Ep
	k = kperp*np.cosh(etak)
	if k >= pz:
		return 0.
	else:
		return 2.*k*(1.-xb)/(kperp**2 + (x*M)**2 + (1.-xb)*0.5*(3*T*0)**2)
	
tau_f = np.vectorize(tau_f)

x = np.linspace(0, T*5, 400) # kperp
y = np.linspace(-10,10,400) # etak

X, Y = np.meshgrid(x, y)
X = X.T
Y = Y.T
Z = tau_f(E, T, X, Y)
plt.contourf(X, Y, Z, 10)
plt.plot(x, np.arccosh(E/x), 'r--', label='Kinetic constrain')
plt.show()
