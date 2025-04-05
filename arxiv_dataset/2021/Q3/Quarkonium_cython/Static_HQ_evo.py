#!/usr/bin/env python

import numpy as np
import HqEvo



### -------- in this file, everything is in GeV and fm ---------- ###
M = 4.650 						# GeV b-quark
GeVm1_to_fmc = 0.197327			# Convert GeV^-1 to fm/c

LBT = HqEvo.HqEvo(
	options={'transport': {'2->2':True, '2->3':False, '3->2':False},
			 'mass': M, 
			 'Nf': 3,
			 'Kfactor': 1.0,
			 'Tc': 0.154,
			 'mD': {'mD-model': 0, 'mTc': 5., 'slope': 0.5, 'curv': -1.2}},		
	table_folder="./tables", 
	refresh_table=True	)


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




class HQ_p_update:
	def __init__(self):
		pass
		
	def update(self, HQ_Ep, Temp, time_step):
		# HQ_Ep is a 4-momentum in GeV, Temp is in GeV, time_step is in fm/c
		p1 = np.array(HQ_Ep)
		t0 = 0.0
		while t0 < time_step:
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
				p1 = p1_new
			
			t0 += small_dt
		return p1			# return the 4-momentum in GeV
