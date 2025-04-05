#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sys

import HqEvo

def thermal(mass, T):
	E = np.linspace(1, 3., 1000)*mass
	spectrum = np.exp(-E/T)*E*np.sqrt(E**2-mass**2)
	return E, spectrum/spectrum.sum()/(E[1]-E[0])

def boost4_By3(A, v): # pass a four vector A^mu and a 3-velocity v = [v1, v2, v3]
	v2 = (v**2).sum()
	absv = np.sqrt(v2)+1e-32
	n = v/absv
	gamma = 1./np.sqrt(1. - v2 + 1e-32);
	gb = gamma - 1.;
	gb_vecn_dot_vecA = gb*np.dot(n, A[1:])
	gammaA0 = gamma*A[0]
	Ap = np.zeros(4)
	Ap[0] = gamma*(A[0] - np.dot(v, A[1:]))
	Ap[1:] = -gammaA0*v + A[1:] + n*gb_vecn_dot_vecA
	return Ap

def rotate_back_from_D(A, D): # pass A^mu and a 3-direction D = [D1, D2, D3]
	Dperp = np.sqrt((D[:2]**2).sum());
	Dabs = np.sqrt((D**2).sum());
	if Dperp/Dabs < 1e-10:
		return A
	c2 = D[2]/Dabs
	s2 = Dperp/Dabs
	c3 = D[0]/Dperp
	s3 = D[1]/Dperp;
	Ap = np.zeros(4)
	Ap[0] = A[0]
	Ap[1] = -s3*A[1] 	- c3*(c2*A[2] 	- s2*A[3])
	Ap[2] = c3*A[1] 	- s3*(c2*A[2] 	- s2*A[3])
	Ap[3] =              s2*A[2] 	    + c2*A[3]
	return Ap

def product_4(A, B):
	return A[0]*B[0] - np.dot(A[1:], B[1:])
	
def hist4(p, t, Temp, mass):
	plt.clf()
	for i, pi in enumerate(p):
		plt.subplot(2,2,i+1)
		if i==0:
			x, y = thermal(mass, Temp)
			plt.plot(x, y, 'r-', label = 'Boltzmann')	
			plt.semilogy()
		plt.hist(pi, bins=30, normed=True, histtype='step')
		plt.legend()
	plt.suptitle("{} [fm/c]".format(t))
	plt.pause(0.1)
	


pid = 4 # 5=bottom, 4=charm
mass = 4.3 if pid == 5 else 1.3	# Heavy quark mass
pmax = 5. # Initialize momentum within [-pmax,pmax]^3 box
xmax = 5. # Box size [-xmax,xmax]^3, perodic boundary 
Temp = 0.3 # GeV
dt = 1.0 # fm/c
GeVm1_to_fmc = 0.197 # Convert GeV^-1 to fm/c
NQ = 1000	# numer of heavy quark

LBT = HqEvo.HqEvo(
	options={'transport': {'2->2':True, '2->3':False, '3->2':False},
			 'mass': mass, 
			 'Nf': 3,
			 'Kfactor': 1.0,
			 'Tc': 0.154,
			 'mD': {'mD-model': 0, 'mTc': 5., 'slope': 0.5, 'curv': -1.2}},		
	table_folder="./tables", 
	refresh_table=True	)

# paritlce datatype: id, p, x
particle_type = [('id', np.int32), ('p', np.float32, 4), ('x', np.float32, 4)]

# initalize particle array
particles = np.zeros(NQ, dtype=particle_type)
for p in particles:
	p['id'] = np.random.choice([pid, -pid])
	p['p'][1:] = np.random.uniform(-pmax, pmax, 3)
	p['p'][0] = np.sqrt(mass**2 + (p['p'][1:]**2).sum())
	p['x'][1:] = np.random.uniform(-xmax, xmax, 3)
	p['x'][0] = 0.0
	

# evolution
for i in range(1000):
	t = (1+i)*dt
	for p in particles:
		p1 = p['p']	# momentum in the box frame
		t0 = p['x'][0]	# x in the box frame
		
		# while paritle time does not cathch up with system time, evolve this particle
		while t0 < t:
			# sample scattering channel
			channel, small_dt = LBT.sample_channel(p1[0], Temp, 0., 0.) 
			# change of units of time
			small_dt *= GeVm1_to_fmc
			
			# channel < 0: nothing happens
			# channel = 0 or 1 are elastic scatterings
			if channel >= 0:
				# sample initial momentums, in the HQ [E, 0, 0, pz]  frame 
				LBT.sample_initial(channel, p1[0], Temp, 0., 0.)
				# get HQ momentum [E, 0, 0, pz]
				p1_z = LBT.IS[0]
				# calculate Mendelstem-s variable
				p_com = np.sum(LBT.IS, axis=0)
				v3_com = p_com[1:]/p_com[0]
				p1_com = boost4_By3(p1_z, v3_com)
				s = product_4(p_com, p_com)
				# sample final states of the scattering
				LBT.sample_final(channel, s, Temp, 0., 0., 0.)
				# get final HQ momentum in the CoM frame
				p1_com_z_new = LBT.FS[0]
				# Tranform the results all the way back to box-frame
				p1_com_new = rotate_back_from_D(p1_com_z_new, p1_com[1:])
				p1_z_new = boost4_By3(p1_com_new, -v3_com)
				p1_new = rotate_back_from_D(p1_z_new, p1[1:])
				# update momentum
				p['p'] = p1_new
			# update position
			p['x'] += p['p']/p['p'][0] * small_dt
			t0 += small_dt
			# perodic boundary condition
			for ix, xi in enumerate(p['x'][1:]):
				if xi < -xmax:
					p['x'][ix+1] += xmax
				if xi >= xmax:
					p['x'][ix+1] -= xmax
	# plot some histogram
	hist4(np.array([p['p'] for p in particles]).T, t, Temp, mass)
plt.show()
		
