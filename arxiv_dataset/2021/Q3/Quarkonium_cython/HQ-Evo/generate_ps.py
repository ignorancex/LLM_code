import numpy as np
import matplotlib.pyplot as plt
import HqEvo, h5py, sys

def projector(p):
	norm2 = (p**2).sum()
	return np.diag([1,1,1]) - np.outer(p,p)/norm2 
	

M = 1.3
sampler = HqEvo.HqEvo(options={'transport': {'2->2': True, '2->3': True, '3->2': False},
						 'mass': M,
						 'Nf': 3,
						 'Kfactor': 1.,
						 'Tc': 0.154,
						 'mD': {'mD-model': 0, 'mTc': 4.0, 
								'slope': 0.25, 'curv':-0.8}}
				)
				
E = 3
T = 0.3
dt = 0.1
y0 = np.log(E/M + np.sqrt(1.+E*E/M/M))
sqrts = E + np.sqrt(E**2-M**2)
s = sqrts**2
pmax = (sqrts**2-M*M)/2./sqrts
p3 = []
p4 = []
k = []

for i in range(10000):
	if i%100 == 0:
		print(i)
	sampler.sample_final(2, s, T, dt, 0, 0)
	p3.append(sampler.FS[0])
	p4.append(sampler.FS[1])
	k.append(sampler.FS[2])
p3 = np.array(p3)
p4 = np.array(p4)
k = np.array(k)
f = h5py.File(sys.argv[1], 'a')
gp = f.create_group('E3-T300-dt0.1')
gp.create_dataset('p3', data=p3)
gp.create_dataset('p4', data=p4)
gp.create_dataset('k', data=k)

f.close()
