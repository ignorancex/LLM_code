#!/usr/bin/env python
import numpy as np
import random as rd
import h5py
import LorRot

# Parameters: 	norm=132, nucleon-width=0.51fm, fluctuation=1.6, p = 0
#				(for norm=?, see 1605.03954)
# 				grid-max=13.05fm, (grid value is at the center of grid) 
#				grid-step=0.1fm
# EoS: HotQCD, e(T=Tc) = 0.329 Gev/fm^3, eta/s = 0.08+1*(T-Tc), bulk_norm=0.01, bulk_width=0.01


#Taa = {'2760': {'0-10': 23.0, '0-5': 26.32, '5-10': 20.56, '10-20': 14.39, '20-30': 8.6975, '30-40': 5.001, '40-50': 2.675, '50-100': 0.458, '50-70': 0.954, '70-100': 0.127}}		# mb^-1
#Xsect_bbbar = {'2760': 0.06453}		# mb
Xsect_1S = {'5020': 0.0003147, '2760': 0.000167, '200': 6.272*10.0**(-6)}	# Xsect_pp_1S = {'5020':0.0003983, '2760': 0.000206, '200': 5.943*10.0**(-6)}
Xsect_2S = {'5020': 0.00007803,'2760': 0.00004148, '200': 1.545*10.0**(-6)}	# Xsect_pp_2S = {'5020':0.0000991, '2760': 0.0000513, '200': 1.448*10.0**(-6)}
Xsect_3S = {'5020': 0.00008773,'2760': 0.00004591, '200': 1.576*10.0**(-6)}	# Xsect_pp_3S = {'5020':0.0001096, '2760': 0.00005581, '200': 1.475*10.0**(-6)}
Xsect_1P = {'5020': 0.003284,  '2760': 0.001849, '200': 9.175*10.0**(-5)}	# Xsect_pp_1P = {'5020':0.004489,  '2760': 0.002451, '200': 8.602*10.0**(-5)}
gamma_cut = np.cosh(5.0)	# v_max = 0.9999
Vz_cut = 0.99

Mb = 4.65
M1S = 9.46
M2S = 10.023
M1P = 10.023
M3S = 10.355

class Dynam_Initial_Sample:
	### possible channels include 'corr', '1S', '2S', '1P', '3S'
	def __init__(self, energy_GeV = 2760, centrality_str = '0-10', channel = 'corr', HQuncorr = False, pT_low = 0.0):
	
		### -------- store the position information -------- ###
		# ---- next four lines are for testing by using Weiyao's hydro file
		#file_TAB = h5py.File('ic-'+centrality_str+'-avg.hdf5','r')
		#Tab = np.array(file_TAB['TAB_0'].value)
		#self.Nx, self.Ny = Tab.shape
		#Tab_flat = Tab.reshape(-1)
		# ---- next four lines are for calculation by using Yingxu's hydro file
		file_TAB = open(str(energy_GeV)+'initial_averaged_sd_cen'+centrality_str+'.dat','r')
		Tab = file_TAB.read().split()
		Tab_flat = np.array([float(each) for each in Tab])
		self.Nx = self.Ny = np.sqrt(len(Tab_flat))
		self.dx = 0.1 	# fm
		self.dy = 0.1	# fm
		T_tot = np.sum(Tab_flat)
		T_AA_mb = T_tot/100000.0	# mb^-1
		T_norm = Tab_flat/T_tot
		self.T_accum = np.zeros_like(T_norm, dtype=np.double)
		for index, value in enumerate(T_norm):
			self.T_accum[index] = self.T_accum[index-1] + value
		file_TAB.close()
		self.channel = channel
		self.HQuncorr = HQuncorr
		self.pT_low = pT_low
		
		
		### -------- store the momentum information -------- ###
		if self.channel == 'corr':
			filename_corr = str(energy_GeV) + "corr.dat"
			self.p4_corr = np.fromfile(filename_corr, dtype=float, sep=" ")
			self.len_corr = int(len(self.p4_corr)/4.0)
		if self.channel == '1S':
			filename_1S = str(energy_GeV) + "1S.dat"
			self.p4_1S = np.fromfile(filename_1S, dtype=float, sep=" ")
			self.len_1S = int(len(self.p4_1S)/4.0)
		if self.channel == '2S':
			filename_2S = str(energy_GeV) + "2S.dat"
			self.p4_2S = np.fromfile(filename_2S, dtype=float, sep=" ")
			self.len_2S = int(len(self.p4_2S)/4.0)
		if self.channel == '1P':
			filename_1P = str(energy_GeV) + "1P.dat"
			self.p4_1P = np.fromfile(filename_1P, dtype=float, sep=" ")
			self.len_1P = int(len(self.p4_1P)/4.0)
		if self.channel == '3S':
			filename_3S = str(energy_GeV) + "3S.dat"
			self.p4_3S = np.fromfile(filename_3S, dtype=float, sep=" ")
			self.len_3S = int(len(self.p4_3S)/4.0)
		if self.HQuncorr == True:
			filename_bbbar = str(energy_GeV) + "bbbar.dat"
			self.p4_bbbar = np.fromfile(filename_bbbar, dtype=float, sep=" ")
			self.len_bbbar = int(len(self.p4_bbbar)/8.0)
			T_AA_mb = Taa[str(energy_GeV)][centrality_str]
			self.N_bbbar = Xsect_bbbar[str(energy_GeV)] * T_AA_mb
			self.Nsam_bbbar = int(self.N_bbbar) + 1
		# Nsam_bbbar = number of loops to sample bbbar. 
		# usually N_bbbar is not an integer, to get an average, add one and use rejection below
		
	
	def sample(self):	
		### ---------- sample momenta and postions --------- ###
		p4_Q = []
		p4_Qbar = []
		p4_U1S = []
		p4_U2S = []
		p4_U1P = []
		p4_U3S = []
		x3_Q = []
		x3_Qbar = []
		x3_1S = []
		x3_2S = []
		x3_1P = []
		x3_3S = []
		
		if self.HQuncorr == True:
			for i in range(self.Nsam_bbbar):
				r_bbbar = rd.uniform(0.0, self.Nsam_bbbar+0.0)
				if r_bbbar <= self.N_bbbar:
					row_bbbar = 8*rd.randrange(0, self.len_bbbar-1, 1)
					if self.p4_bbbar[row_bbbar] < gamma_cut * Mb and self.p4_bbbar[row_bbbar+4] < gamma_cut * Mb:
						## momenta
						p4_Q.append( [self.p4_bbbar[row_bbbar], self.p4_bbbar[row_bbbar+1], self.p4_bbbar[row_bbbar+2], self.p4_bbbar[row_bbbar+3]] )
						p4_Qbar.append( [self.p4_bbbar[row_bbbar+4], self.p4_bbbar[row_bbbar+5], self.p4_bbbar[row_bbbar+6], self.p4_bbbar[row_bbbar+7]] )
						## positions
						r_xy = rd.uniform(0.0, 1.0)
						i_bbbar = np.searchsorted(self.T_accum, r_xy)
						i_x = np.floor((i_bbbar+0.0)/self.Ny)
						i_y = i_bbbar - i_x*self.Ny
						i_x += np.random.rand()
						i_y += np.random.rand()
						x = (i_x - self.Nx/2.)*self.dx
						y = (i_y - self.Ny/2.)*self.dy
						x3_Q.append(np.array([x,y,0.0]))
						x3_Qbar.append(np.array([x,y,0.0]))
		
		if self.channel == 'corr':
			## it is like sampling nS but divide p4 by two, give each to b and bbar
			count = 0
			while count == 0:
				row_corr = 4*rd.randrange(0, self.len_corr-1, 1)
				if self.p4_corr[row_corr]/2. < gamma_cut * Mb:
					count += 1
					## momenta
					p_x = self.p4_corr[row_corr+1]/2.
					p_y = self.p4_corr[row_corr+2]/2.
					p_z = self.p4_corr[row_corr+3]/2.
					Energy = np.sqrt(Mb**2+p_x**2+p_y**2+p_z**2)
					p4_bbbar = [ Energy, p_x, p_y, p_z ]
					p4_Q.append(p4_bbbar)
					p4_Qbar.append(p4_bbbar)
					## positions
					r_xy = rd.uniform(0.0, 1.0)
					i_bbbar = np.searchsorted(self.T_accum, r_xy)
					i_x = np.floor((i_bbbar+0.0)/self.Ny)
					i_y = i_bbbar - i_x*self.Ny
					i_x += np.random.rand()
					i_y += np.random.rand()
					x = (i_x - self.Nx/2.)*self.dx
					y = (i_y - self.Ny/2.)*self.dy
					x3_Q.append(np.array([x,y,0.0]))
					x3_Qbar.append(np.array([x,y,0.0]))
			
		## since the bottomonia cross sections are too small
		## we only sample cases where bottomonia are produced			
		if self.channel == '1S':
			count = 0
			while count == 0:
				row_1S = 4*rd.randrange(0, self.len_1S-1, 1)
				if self.p4_1S[row_1S] < gamma_cut * M1S and abs(self.p4_1S[row_1S+3]/self.p4_1S[row_1S]) < Vz_cut:
					count += 1
					## momenta
					p4_U1S.append( [self.p4_1S[row_1S], self.p4_1S[row_1S+1], self.p4_1S[row_1S+2], self.p4_1S[row_1S+3]] )
					## positions
					r_xy = rd.uniform(0.0, 1.0)
					i_1S = np.searchsorted(self.T_accum, r_xy)
					i_x = np.floor((i_1S+0.0)/self.Ny)
					i_y = i_1S - i_x*self.Ny
					i_x += np.random.rand()
					i_y += np.random.rand()
					x = (i_x - self.Nx/2.)*self.dx
					y = (i_y - self.Ny/2.)*self.dy
					x3_1S.append(np.array([x,y,0.0]))
		
		if self.channel == '2S':
			count = 0
			while count == 0:
				row_2S = 4*rd.randrange(0, self.len_2S-1, 1)
				if self.p4_2S[row_2S] < gamma_cut * M2S and abs(self.p4_2S[row_2S+3]/self.p4_2S[row_2S]) < Vz_cut:
					count += 1
					## momenta
					p4_U2S.append( [self.p4_2S[row_2S], self.p4_2S[row_2S+1], self.p4_2S[row_2S+2], self.p4_2S[row_2S+3]] )
					## positions
					r_xy = rd.uniform(0.0, 1.0)
					i_2S = np.searchsorted(self.T_accum, r_xy)
					i_x = np.floor((i_2S+0.0)/self.Ny)
					i_y = i_2S - i_x*self.Ny
					i_x += np.random.rand()
					i_y += np.random.rand()
					x = (i_x - self.Nx/2.)*self.dx
					y = (i_y - self.Ny/2.)*self.dy
					x3_2S.append(np.array([x,y,0.0]))

		if self.channel == '1P':
			count = 0
			while count == 0:
				row_1P = 4*rd.randrange(0, self.len_1P-1, 1)
				if self.p4_1P[row_1P] < gamma_cut * M1P and abs(self.p4_1P[row_1P+3]/self.p4_1P[row_1P]) < Vz_cut:
					count += 1
					## momenta
					p4_U1P.append( [self.p4_1P[row_1P], self.p4_1P[row_1P+1], self.p4_1P[row_1P+2], self.p4_1P[row_1P+3]] )
					## positions
					r_xy = rd.uniform(0.0, 1.0)
					i_1P = np.searchsorted(self.T_accum, r_xy)
					i_x = np.floor((i_1P+0.0)/self.Ny)
					i_y = i_1P - i_x*self.Ny
					i_x += np.random.rand()
					i_y += np.random.rand()
					x = (i_x - self.Nx/2.)*self.dx
					y = (i_y - self.Ny/2.)*self.dy
					x3_1P.append(np.array([x,y,0.0]))

		if self.channel == '3S':
			count = 0
			while count == 0:
				row_3S = 4*rd.randrange(0, self.len_3S-1, 1)
				if self.p4_3S[row_3S] < gamma_cut * M3S and abs(self.p4_3S[row_3S+3]/self.p4_3S[row_3S]) < Vz_cut:
					count += 1
					## momenta
					p4_U3S.append( [self.p4_3S[row_3S], self.p4_3S[row_3S+1], self.p4_3S[row_3S+2], self.p4_3S[row_3S+3]] )
					## positions
					r_xy = rd.uniform(0.0, 1.0)
					i_3S = np.searchsorted(self.T_accum, r_xy)
					i_x = np.floor((i_3S+0.0)/self.Ny)
					i_y = i_3S - i_x*self.Ny
					i_x += np.random.rand()
					i_y += np.random.rand()
					x = (i_x - self.Nx/2.)*self.dx
					y = (i_y - self.Ny/2.)*self.dy
					x3_3S.append(np.array([x,y,0.0]))

		if self.channel == '1Sv2':
			pT = self.pT_low + rd.uniform(0.0, 1.0)
			theta = 2.0*np.pi * rd.uniform(0.0, 1.0)
			px = pT * np.cos(theta)
			py = pT * np.sin(theta)
			pz = 0.0
			e = np.sqrt( pT**2 + M1S**2 )
			p4_U1S.append( [e, px, py, pz] )	# momenta
			r_xy = rd.uniform(0.0, 1.0)		# positions
			i_1S = np.searchsorted(self.T_accum, r_xy)
			i_x = np.floor((i_1S+0.0)/self.Ny)
			i_y = i_1S - i_x*self.Ny
			i_x += np.random.rand()
			i_y += np.random.rand()
			x = (i_x - self.Nx/2.)*self.dx
			y = (i_y - self.Ny/2.)*self.dy
			x3_1S.append(np.array([x,y,0.0]))

		if self.channel == '2Sv2':
			pT = self.pT_low + rd.uniform(0.0, 1.0)
			theta = 2.0*np.pi * rd.uniform(0.0, 1.0)
			px = pT * np.cos(theta)
			py = pT * np.sin(theta)
			pz = 0.0
			e = np.sqrt( pT**2 + M2S**2 )
			p4_U2S.append( [e, px, py, pz] )	# momenta
			r_xy = rd.uniform(0.0, 1.0)		# positions
			i_2S = np.searchsorted(self.T_accum, r_xy)
			i_x = np.floor((i_2S+0.0)/self.Ny)
			i_y = i_2S - i_x*self.Ny
			i_x += np.random.rand()
			i_y += np.random.rand()
			x = (i_x - self.Nx/2.)*self.dx
			y = (i_y - self.Ny/2.)*self.dy
			x3_2S.append(np.array([x,y,0.0]))
			
		if self.channel == '1Pv2':
			pT = self.pT_low + rd.uniform(0.0, 1.0)
			theta = 2.0*np.pi * rd.uniform(0.0, 1.0)
			px = pT * np.cos(theta)
			py = pT * np.sin(theta)
			pz = 0.0
			e = np.sqrt( pT**2 + M1P**2 )
			p4_U1P.append( [e, px, py, pz] )	# momenta
			r_xy = rd.uniform(0.0, 1.0)		# positions
			i_1P = np.searchsorted(self.T_accum, r_xy)
			i_x = np.floor((i_1P+0.0)/self.Ny)
			i_y = i_1P - i_x*self.Ny
			i_x += np.random.rand()
			i_y += np.random.rand()
			x = (i_x - self.Nx/2.)*self.dx
			y = (i_y - self.Ny/2.)*self.dy
			x3_1P.append(np.array([x,y,0.0]))
			
		if self.channel == '3Sv2':
			pT = self.pT_low + rd.uniform(0.0, 1.0)
			theta = 2.0*np.pi * rd.uniform(0.0, 1.0)
			px = pT * np.cos(theta)
			py = pT * np.sin(theta)
			pz = 0.0
			e = np.sqrt( pT**2 + M3S**2 )
			p4_U3S.append( [e, px, py, pz] )	# momenta
			r_xy = rd.uniform(0.0, 1.0)		# positions
			i_3S = np.searchsorted(self.T_accum, r_xy)
			i_x = np.floor((i_3S+0.0)/self.Ny)
			i_y = i_3S - i_x*self.Ny
			i_x += np.random.rand()
			i_y += np.random.rand()
			x = (i_x - self.Nx/2.)*self.dx
			y = (i_y - self.Ny/2.)*self.dy
			x3_3S.append(np.array([x,y,0.0]))
			
																		
		self.x3_Q = np.array(x3_Q)
		self.x3_Qbar = np.array(x3_Qbar)
		self.x3_1S = np.array(x3_1S)
		self.x3_2S = np.array(x3_2S)
		self.x3_1P = np.array(x3_1P)
		self.x3_3S = np.array(x3_3S)
		self.p4_Q = np.array(p4_Q)
		self.p4_Qbar = np.array(p4_Qbar)
		self.p4_U1S = np.array(p4_U1S)
		self.p4_U2S = np.array(p4_U2S)
		self.p4_U1P = np.array(p4_U1P)
		self.p4_U3S = np.array(p4_U3S)

	def Qinit_p(self):
		return self.p4_Q
	
	def Qbarinit_p(self):
		return self.p4_Qbar
	
	def U1Sinit_p(self):
		return self.p4_U1S	

	def U2Sinit_p(self):
		return self.p4_U2S	

	def U1Pinit_p(self):
		return self.p4_U1P	

	def U3Sinit_p(self):
		return self.p4_U3S	
				
	def Qinit_x(self):
		return self.x3_Q
		
	def Qbarinit_x(self):
		return self.x3_Qbar
	
	def U1Sinit_x(self):
		return self.x3_1S	

	def U2Sinit_x(self):
		return self.x3_2S	

	def U1Pinit_x(self):
		return self.x3_1P

	def U3Sinit_x(self):
		return self.x3_3S	