import numpy as np
import matplotlib.pyplot as plt
import HqEvo, h5py, sys
from scipy.optimize import brentq

def projector(p):
	norm2 = (p**2).sum()
	return np.diag([1,1,1]) - np.outer(p,p)/norm2 

def find_level(H, percentage):
	tot = np.sum(H)
	minH, maxH = np.min(H), np.max(H)
	def percentile_above_level(level):
		return np.sum(H[H>level])/tot - percentage
	return brentq(percentile_above_level, minH, maxH), maxH
	
def corner(X, c):
	ndims, xsamples = X.shape
	for i in range(ndims):
		for j in range(0,i+1):
			plt.subplot(ndims, ndims, i*ndims+j+1)
			if i==j:
				plt.hist(X[i], bins=30, normed=True, histtype='step', color=c, linewidth=1.,)
				#y,xb = np.histogram(X[i], bins=30, normed=True)
				#x = (xb[1:] + xb[:-1])/2
				#plt.plot(x, y, color=c)
				#plt.semilogy()
			else:
				H, xb, yb = np.histogram2d(X[j], X[i], bins=10, normed=True)
				# solve 70% level
				level, maxH = find_level(H, 0.7)
				plt.contour(H.T,extent=[xb[0],xb[-1],yb[0],yb[-1]], levels=[level,maxH], colors=(c,), alpha=0.5)
	#plt.subplots_adjust(wspace=0., hspace=0.)
def plot(f, E, T, dt, c):
	M = 1.3
	y0 = np.log(E/M + np.sqrt(1.+E*E/M/M))
	sqrts = E + np.sqrt(E**2-M**2)
	s = sqrts**2
	pmax = (sqrts**2-M*M)/2./sqrts
	p3 = f['p3'].value
	p4 = f['p4'].value
	k = f['k'].value
	
	k0 = k.T[0]
	kz = k.T[3]
	etak = 0.5*np.log((k0+kz)/(k0-kz))
	kt = np.sqrt(k.T[1]**2 + k.T[2]**2)
	xk = (k0+kz)/sqrts
	qperp = np.sqrt(p4.T[1]**2 + p4.T[2]**2)
	
	Tp4 = np.array([projector(ip[1:]) for ip in p4])
	direction = np.array([[0,0,-1] for ip in p4])
	vec1 = np.array([np.dot(PP, vv) for PP, vv in zip(Tp4, direction)])
	vec2 = np.array([np.dot(PP, vv[1:]) for PP, vv in zip(Tp4, k)])
	c4k = np.array([np.dot(v1, v2)/np.sqrt(np.dot(v1, v1)*np.dot(v2, v2)) for v1, v2 in zip(vec1, vec2)])
	
	tauf = 2*k0*(1.-xk)/(kt*kt + xk**2*M**2 + (1-xk)*0.5*9.*T*T)
	
	#X = np.array([np.log(qperp/T), np.log(kt/qperp), np.log(tauf), np.arccos(c4k)])
	#X = np.array([np.log(qperp/T), np.log(xk), np.arcsin(kt/k0), np.arccos(c4k)])
	X = np.array([k0])
	corner(X[:, kz>0], c)


f = h5py.File(sys.argv[1], 'r')
plot(f['E3-T300-dt0.1'], 3, 0.3, 0.1, 'r')
plot(f['E3-T300-dt1'], 3, 0.3, 1, 'g')
plot(f['E3-T300-dt10'], 3, 0.3, 10, 'b')
plt.show()

plot(f['E10-T300-dt0.1'], 10, 0.3, 0.1, 'r')
plot(f['E10-T300-dt1'], 10, 0.3, 1, 'g')
plot(f['E10-T300-dt10'], 10, 0.3, 10, 'b')
plt.show()

plot(f['E30-T300-dt0.1'], 30, 0.3, 0.1, 'r')
plot(f['E30-T300-dt1'], 30, 0.3, 1, 'g')
plot(f['E30-T300-dt10'], 30, 0.3, 10, 'b')
plt.show()



