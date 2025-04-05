#!/usr/bin/env python

import numpy as np
import HqEvo
import LorRot

C1 = 0.197327			# 0.197 GeV*fm = 1

class HQ_diff:
	def __init__(self, Mass):
		self.M_b = Mass
		self.hqsample = HqEvo.HqEvo(
		options={'transport': {'2->2':True, '2->3':False, '3->2':False},
			 'mass': Mass, 
			 'Nf': 3,
			 'Kfactor': 1.0,
			 'Tc': 0.154,
			 'mD': {'mD-model': 0, 'mTc': 5., 'slope': 0.5, 'curv': -1.2}},
		table_folder="./tables",
		refresh_table=False	)

		# this function needs to be cdef so that we can use the reference copy
	def update_HQ_LBT(self, p1_lab, v3cell, Temp, mean_dt23_lab, mean_dt32_lab):
		# Define local variables
		a1 = 0.6
		a2 = 0.6
		# Boost p1 (lab) to p1 (cell)
		p1_cell = LorRot.lorentz(p1_lab, v3cell)
		
		# displacement vector within dx23_lab and dx32_lab seen from cell frame
		dx23_cell = np.zeros(4)
		dx32_cell = np.zeros(4)
		for i in range(4):
			dx23_cell[i] = p1_cell[i]/p1_lab[0]*mean_dt23_lab
			dx32_cell[i] = p1_cell[i]/p1_lab[0]*mean_dt32_lab

		# Sample channel in cell, return channel index and evolution time seen from cell
		channel, dt_cell = self.hqsample.sample_channel(p1_cell[0], Temp, dx23_cell[0], dx32_cell[0])
		# channels are: 0:Qq2Qq 1:Qg2Qg, 2:Qq2Qqg, 3:Qg2Qgg, 4:Qqg2Qq, 5:Qgg2Qg
		# Boost evolution time back to lab frame
		dtHQ = p1_lab[0]/p1_cell[0]*dt_cell
		
		# If not scattered, return channel=-1, evolution time in Lab frame and origin al p1(lab)
		if channel < 0:
			pnew = p1_lab
		else:
			# Sample initial state and return initial state particle four vectors 
			# Assume p1_cell (initial) to align with z-direction, sample other particles' momenta			
			self.hqsample.sample_initial(channel, p1_cell[0], Temp, dx23_cell[0], dx32_cell[0])
			p1_cell_Z = np.array(self.hqsample.IS[0])
			
			# Find c.o.m. momentum of p1_cell_Z and other particles
			Pcom = np.zeros(4)
			for pp in self.hqsample.IS:
				for i in range(4):
					Pcom[i] += pp[i]
			s = Pcom[0]**2 - Pcom[1]**2 - Pcom[2]**2 - Pcom[3]**2
			v3com = np.zeros(3)
			for i in range(3):
				v3com[i] = Pcom[i+1]/Pcom[0];
			
			# Boost to the c.o.m. frame of initial particles
			p1_com = LorRot.lorentz(p1_cell_Z, v3com)

			if channel in [4,5]: # channel=4,5 for 3 -> 2 kinetics
				L1 = np.sqrt(p1_com[0]**2 - self.M_b**2)
				pbuffer = LorRot.lorentz(np.array(self.hqsample.IS[1]), v3com)
				L2 = pbuffer[0]
				pbuffer = LorRot.lorentz(np.array(self.hqsample.IS[2]), v3com)
				Lk = pbuffer[0]
				x2 = L2/(L1+L2+Lk)
				xk = Lk/(L1+L2+Lk)
				a1 = x2 + xk; 
				a2 = (x2 - xk)/(1. - a1)
			
			dt23_com = p1_com[0]/p1_cell_Z[0]*dx23_cell[0]
			# Sample final state momentum in c.o.m. frame, assume p1_com is along z-axis
			self.hqsample.sample_final(channel, s, Temp, dt23_com, a1, a2)
			p1_com_Z_new = np.array(self.hqsample.FS[0])
			# Rotate final states back to original Com frame (p1_com is not along z-axis)
			p1_com_new = LorRot.rotate_back_from_D(p1_com_Z_new, p1_com[1], p1_com[2], p1_com[3])
			# Boost back to cell frame where p1_cell is along z
			p1_cell_Z_new = LorRot.lorentz(p1_com_new, -v3com)
			# rotate back to original direction of p1_cell
			p1_cell_new = LorRot.rotate_back_from_D(p1_cell_Z_new, p1_cell[1], p1_cell[2], p1_cell[3])
			# boost back to lab frame
			pnew = LorRot.lorentz(p1_cell_new, -v3cell)
		
		return channel, dtHQ, pnew
		# return channel, dt in lab frame, updated momentum of heavy quark	
		