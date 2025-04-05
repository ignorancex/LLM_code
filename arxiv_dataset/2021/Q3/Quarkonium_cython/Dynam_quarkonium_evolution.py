#!/usr/bin/env python
import numpy as np
import h5py
import csv
import random as rd
from scipy.spatial import cKDTree
from Medium_Read import hydro_reader
from Dynam_Initial_Sample import Dynam_Initial_Sample
import DisRec
import LorRot
from HQ_diffuse import HQ_diff

#### --------- some constants -----------------------------
M = 4.65					  # b-quark mass in GeV
C1 = 0.197327				  # 0.197 GeV*fm = 1
R_search = 2.0				  # (fm), pair-search radius in the recombination
Tc = 0.154				  	  # critical temperature of QGP
T_1S = 1.5					  # melting temperature of Upsilon_1S
T_2S = 1.5					  # melting temperature of Upsilon_2S
T_1P = 1.5					  # melting temperature of Upsilon_2S


class QQbar_evol:
####---- input the medium_type when calling the class ----####
	def __init__(self, medium_type = 'dynamical', centrality_str_given = '0-10', energy_GeV = 2760, recombine = True, HQ_scat = True, sample_method = '1S', pT = 0.0):
		self.type = medium_type
		self.recombine = recombine
		self.HQ_scat = HQ_scat
		self.centrality = centrality_str_given
		self.Ecm = energy_GeV
		## -------------- create the hydro reader ------------ ##
		self.hydro = hydro_reader( hydro_file_path = 'HydroData'+str(energy_GeV)+'GeV'+centrality_str_given+'.h5' )
		## -------------- create the rates event reader ------------ ##
		self.event = DisRec.DisRec()
		self.M = DisRec.pyM()  # heavy quark mass
		## -------------- create HQ-diff rates reader -------- ##
		if self.HQ_scat == True:
			self.HQ_event = HQ_diff(Mass = self.M)
		## -------------- create init p,x sampler ------------ ##
		self.init = Dynam_Initial_Sample(energy_GeV = self.Ecm, centrality_str = self.centrality, channel = sample_method, pT_low = pT)

####---- initialize Q, Qbar, Quarkonium ----####
####---- tau0 = 0.6 fm/c is the hydro starting time; before that, free stream
	def initialize(self, tau0 = 0.6):
		if self.type == 'dynamical':
			## ----- create dictionaries to store momenta, positions, id ----- ##
			self.Qlist = {'4-momentum': [], '3-position': [], 'id': 5, 'last_t23': [], 'last_t32': [],  'last_scatter_dt':[]}
			self.Qbarlist = {'4-momentum': [], '3-position': [], 'id': -5, 'last_t23': [], 'last_t32': [],  'last_scatter_dt':[]}
			self.U1Slist = {'4-momentum': [], '3-position': [], 'id': 533, 'last_form_time': []}
			self.U2Slist = {'4-momentum': [], '3-position': [], 'id': 100533, 'last_form_time': []}
			self.U1Plist = {'4-momentum': [], '3-position': [], 'id': 555, 'last_form_time': []}
			self.U3Slist = {'4-momentum': [], '3-position': [], 'id': 200533, 'last_form_time': []}

			## --------- sample initial momenta and positions -------- ##
			self.init.sample()
			self.Qlist['4-momentum'] = self.init.Qinit_p()
			self.Qlist['3-position'] = self.init.Qinit_x()
			self.Qbarlist['4-momentum'] = self.init.Qbarinit_p()
			self.Qbarlist['3-position'] = self.init.Qbarinit_x()
			self.U1Slist['4-momentum'] = self.init.U1Sinit_p()
			self.U1Slist['3-position'] = self.init.U1Sinit_x()
			self.U2Slist['4-momentum'] = self.init.U2Sinit_p()
			self.U2Slist['3-position'] = self.init.U2Sinit_x()
			self.U1Plist['4-momentum'] = self.init.U1Pinit_p()
			self.U1Plist['3-position'] = self.init.U1Pinit_x()
			self.U3Slist['4-momentum'] = self.init.U3Sinit_p()
			self.U3Slist['3-position'] = self.init.U3Sinit_x()
			
			N_Q = len(self.Qlist['4-momentum'])
			N_Qbar = len(self.Qbarlist['4-momentum'])
			N_U1S = len(self.U1Slist['4-momentum'])
			N_U2S = len(self.U2Slist['4-momentum'])
			N_U1P = len(self.U1Plist['4-momentum'])
			N_U3S = len(self.U3Slist['4-momentum'])
			
			## free stream the Q, Qbar, U1S, U2S, U3S, U1P in the lab frame by 0.6 time
			for i in range(N_Q):
				vQ = self.Qlist['4-momentum'][i][1:]/self.Qlist['4-momentum'][i][0]
				self.Qlist['3-position'][i] += vQ*tau0
				self.Qlist['last_t23'].append(0.6)
				self.Qlist['last_t32'].append(0.6)
				self.Qlist['last_scatter_dt'].append(0.0)

			for i in range(N_Qbar):
				vQbar = self.Qbarlist['4-momentum'][i][1:]/self.Qbarlist['4-momentum'][i][0]
				self.Qbarlist['3-position'][i] += vQbar*tau0
				self.Qbarlist['last_t23'].append(0.6)
				self.Qbarlist['last_t32'].append(0.6)
				self.Qbarlist['last_scatter_dt'].append(0.0)
			
			for i in range(N_U1S):
				vU1S = self.U1Slist['4-momentum'][i][1:]/self.U1Slist['4-momentum'][i][0]
				self.U1Slist['3-position'][i] += vU1S*tau0
				self.U1Slist['last_form_time'].append(0.6)
			
			for i in range(N_U2S):
				vU2S = self.U2Slist['4-momentum'][i][1:]/self.U2Slist['4-momentum'][i][0]
				self.U2Slist['3-position'][i] += vU2S*tau0
				self.U2Slist['last_form_time'].append(0.6)

			for i in range(N_U1P):
				vU1P = self.U1Plist['4-momentum'][i][1:]/self.U1Plist['4-momentum'][i][0]
				self.U1Plist['3-position'][i] += vU1P*tau0
				self.U1Plist['last_form_time'].append(0.6)

			for i in range(N_U3S):
				vU3S = self.U3Slist['4-momentum'][i][1:]/self.U3Slist['4-momentum'][i][0]
				self.U3Slist['3-position'][i] += vU3S*tau0
				self.U3Slist['last_form_time'].append(0.6)
													
			self.Qlist['4-momentum'] = np.array(self.Qlist['4-momentum'])
			self.Qlist['3-position'] = np.array(self.Qlist['3-position'])
			self.Qlist['last_t23'] = np.array(self.Qlist['last_t23'])
			self.Qlist['last_t32'] = np.array(self.Qlist['last_t32'])
			self.Qlist['last_scatter_dt'] = np.array(self.Qlist['last_scatter_dt'])
			self.Qbarlist['4-momentum'] = np.array(self.Qbarlist['4-momentum'])
			self.Qbarlist['3-position'] = np.array(self.Qbarlist['3-position'])
			self.Qbarlist['last_t23'] = np.array(self.Qbarlist['last_t23'])
			self.Qbarlist['last_t32'] = np.array(self.Qbarlist['last_t32'])
			self.Qbarlist['last_scatter_dt'] = np.array(self.Qbarlist['last_scatter_dt'])
			self.U1Slist['4-momentum'] = np.array(self.U1Slist['4-momentum'])
			self.U1Slist['3-position'] = np.array(self.U1Slist['3-position'])
			self.U1Slist['last_form_time'] = np.array(self.U1Slist['last_form_time'])
			self.U2Slist['4-momentum'] = np.array(self.U2Slist['4-momentum'])
			self.U2Slist['3-position'] = np.array(self.U2Slist['3-position'])
			self.U2Slist['last_form_time'] = np.array(self.U2Slist['last_form_time'])
			self.U1Plist['4-momentum'] = np.array(self.U1Plist['4-momentum'])
			self.U1Plist['3-position'] = np.array(self.U1Plist['3-position'])
			self.U1Plist['last_form_time'] = np.array(self.U1Plist['last_form_time'])
			self.U3Slist['4-momentum'] = np.array(self.U3Slist['4-momentum'])
			self.U3Slist['3-position'] = np.array(self.U3Slist['3-position'])
			self.U3Slist['last_form_time'] = np.array(self.U3Slist['last_form_time'])
			
			## ------------- store the current lab time ------------ ##
			self.t = tau0

			
#### ---------------- event evolution function ------------------ ####				
	def run(self, dt = 0.04):
		len_Q = len(self.Qlist['4-momentum'])
		len_Qbar = len(self.Qbarlist['4-momentum'])
		len_U1S = len(self.U1Slist['4-momentum'])
		len_U2S = len(self.U2Slist['4-momentum'])
		len_U1P = len(self.U1Plist['4-momentum'])
		len_U3S = len(self.U3Slist['4-momentum'])
		
		### ----------- free stream these particles ------------###
		for i in range(len_Q):
			v3_Q = self.Qlist['4-momentum'][i][1:]/self.Qlist['4-momentum'][i][0]
			self.Qlist['3-position'][i] = self.Qlist['3-position'][i] + dt * v3_Q
		for i in range(len_Qbar):
			v3_Qbar = self.Qbarlist['4-momentum'][i][1:]/self.Qbarlist['4-momentum'][i][0]
			self.Qbarlist['3-position'][i] = self.Qbarlist['3-position'][i] + dt * v3_Qbar		
		for i in range(len_U1S):
			v3_U1S = self.U1Slist['4-momentum'][i][1:]/self.U1Slist['4-momentum'][i][0]
			self.U1Slist['3-position'][i] = self.U1Slist['3-position'][i] + dt * v3_U1S
		for i in range(len_U2S):
			v3_U2S = self.U2Slist['4-momentum'][i][1:]/self.U2Slist['4-momentum'][i][0]
			self.U2Slist['3-position'][i] = self.U2Slist['3-position'][i] + dt * v3_U2S
		for i in range(len_U1P):
			v3_U1P = self.U1Plist['4-momentum'][i][1:]/self.U1Plist['4-momentum'][i][0]
			self.U1Plist['3-position'][i] = self.U1Plist['3-position'][i] + dt * v3_U1P
		for i in range(len_U3S):
			v3_U3S = self.U3Slist['4-momentum'][i][1:]/self.U3Slist['4-momentum'][i][0]
			self.U3Slist['3-position'][i] = self.U3Slist['3-position'][i] + dt * v3_U3S

		### ----------- end of free stream particles -----------###
		
		### ----- update the time here, otherwise z would be bigger than t ---- ###
		self.t += dt
				
		### ------------- heavy quark diffusion --------------- ###
		if self.HQ_scat == True:
			for i in range(len_Q):
				#print self.t, self.Qlist['4-momentum'][i], self.Qlist['3-position'][i]
				T_Vxyz = self.hydro.cell_info(self.t, self.Qlist['3-position'][i])
				if T_Vxyz[0] >= Tc:
					timer = 0.0
					p_Q = self.Qlist['4-momentum'][i]
					dt23 = max(0.0, self.t - self.Qlist['last_t23'][i])
					dt32 = max(0.0, self.t - self.Qlist['last_t32'][i])
					dt_real = dt + self.Qlist['last_scatter_dt'][i]
					while timer <= dt_real:
						channel, dtHQ, p_Q = self.HQ_event.update_HQ_LBT(p_Q, T_Vxyz[1:], T_Vxyz[0], dt23, dt32)
						if channel == 2 or channel == 3:
							t23 = self.t + timer
							self.Qlist['last_t23'][i] = t23
						if channel == 4 or channel == 5:
							t32 = self.t + timer
							self.Qlist['last_t32'][i] = t32
						timer += dtHQ*C1
					self.Qlist['last_scatter_dt'][i] = dt_real - timer
					self.Qlist['4-momentum'][i] = p_Q
					
			for i in range(len_Qbar):
				#print self.Qbarlist['4-momentum'][i], self.Qbarlist['3-position'][i]
				T_Vxyz = self.hydro.cell_info(self.t, self.Qbarlist['3-position'][i])
				if T_Vxyz[0] >= Tc:
					timer = 0.0
					p_Qbar = self.Qbarlist['4-momentum'][i]
					dt23 = max(0.0, self.t - self.Qbarlist['last_t23'][i])
					dt32 = max(0.0, self.t - self.Qbarlist['last_t32'][i])
					dt_real = dt + self.Qbarlist['last_scatter_dt'][i]
					while timer <= dt_real:
						channel, dtHQ, p_Qbar = self.HQ_event.update_HQ_LBT(p_Qbar, T_Vxyz[1:], T_Vxyz[0], dt23, dt32)
						if channel == 2 or channel == 3:
							t23 = self.t + timer
							self.Qbarlist['last_t23'][i] = t23
						if channel == 4 or channel == 5:
							t32 = self.t + timer
							self.Qbarlist['last_t32'][i] = t32
						timer += dtHQ*C1
					self.Qbarlist['last_scatter_dt'][i] = dt_real - timer
					self.Qbarlist['4-momentum'][i] = p_Qbar	
		### ----------- end of heavy quark diffusion ---------- ###



		### -------------------- decay ------------------------ ###
		delete_U1S = []
		delete_U2S = []
		delete_U1P = []
		delete_U3S = []
		add_pQ = []
		add_pQbar = []
		add_xQ = []
		#add_xQbar = [] the positions of Q and Qbar are the same
		add_t23 = []
		add_t32 = []
		add_dt_last = []
		
		### --------- 1S decay ---------- ###
		for i in range(len_U1S):
			T_Vxyz = self.hydro.cell_info(self.t, self.U1Slist['3-position'][i])		# temp, vx, vy, vz
			# only consider dissociation and recombination if in the de-confined phase
			if T_Vxyz[0] >= Tc:	
				p4_in_hydrocell = LorRot.lorentz(self.U1Slist['4-momentum'][i], T_Vxyz[1:])		# boost to hydro cell
				v3_in_hydrocell = p4_in_hydrocell[1:]/p4_in_hydrocell[0]
				v_in_hydrocell = np.sqrt(np.sum(v3_in_hydrocell**2))

				rate_decay1S_gluon = self.event.get_R1S_decay_gluon( v_in_hydrocell, T_Vxyz[0] )		# GeV
				rate_decay1S_ineq = self.event.get_R1S_decay_ineq( v_in_hydrocell, T_Vxyz[0] )
				rate_decay1S_ineg = self.event.get_R1S_decay_ineg( v_in_hydrocell, T_Vxyz[0] )
				
				# in lab frame decay probability
				# dt is time in lab frame, dt' = dt*E'/E is time in hydro cell frame
				prob_decay1S_gluon = rate_decay1S_gluon * dt/C1 * p4_in_hydrocell[0]/self.U1Slist['4-momentum'][i][0]
				prob_decay1S_ineq = rate_decay1S_ineq * dt/C1 * p4_in_hydrocell[0]/self.U1Slist['4-momentum'][i][0]
				prob_decay1S_ineg = rate_decay1S_ineg * dt/C1 * p4_in_hydrocell[0]/self.U1Slist['4-momentum'][i][0]
				
				rej_decay1S_mc = np.random.rand(1)
				if rej_decay1S_mc <= prob_decay1S_gluon + prob_decay1S_ineq + prob_decay1S_ineg:
					delete_U1S.append(i)
					if rej_decay1S_mc <= prob_decay1S_gluon:
						recoil_pQpQbar = self.event.sample_S1S_decay_gluon( v_in_hydrocell, T_Vxyz[0] )
					elif rej_decay1S_mc <= prob_decay1S_gluon + prob_decay1S_ineq:
						recoil_pQpQbar = self.event.sample_S1S_decay_ineq( v_in_hydrocell, T_Vxyz[0] )
					else:
						recoil_pQpQbar = self.event.sample_S1S_decay_ineg( v_in_hydrocell, T_Vxyz[0] )
										
					recoil_pQ = np.array(recoil_pQpQbar[0:4])		# 4-momentum
					recoil_pQbar = np.array(recoil_pQpQbar[4:8])

					# Q, Qbar momenta need to be rotated from the v = z axis to hydro cell frame
					# first get the rotation matrix angles
					theta_rot, phi_rot = LorRot.angle( v3_in_hydrocell )
					# then do the rotation
					rotmomentum_Q = LorRot.rotation4(recoil_pQ, theta_rot, phi_rot)
					rotmomentum_Qbar = LorRot.rotation4(recoil_pQbar, theta_rot, phi_rot)
					# we now transform them back to the hydro cell frame
					momentum_Q = LorRot.lorentz(rotmomentum_Q, -v3_in_hydrocell)
					momentum_Qbar = LorRot.lorentz(rotmomentum_Qbar, -v3_in_hydrocell)
					# then transform back to the lab frame
					momentum_Q = LorRot.lorentz(momentum_Q, -T_Vxyz[1:])
					momentum_Qbar = LorRot.lorentz(momentum_Qbar, -T_Vxyz[1:])

					# positions of Q and Qbar
					position_Q = self.U1Slist['3-position'][i]
					#position_Qbar = position_Q
		
					# add x and p for the QQbar to the temporary list
					add_pQ.append(momentum_Q)
					add_pQbar.append(momentum_Qbar)
					add_xQ.append(position_Q)
					#add_xQbar.append(position_Qbar)
					add_t23.append(self.t)
					add_t32.append(self.t)
					add_dt_last.append(0.0)
		
		### --------- 2S decay ---------- ###
		for i in range(len_U2S):
			T_Vxyz = self.hydro.cell_info(self.t, self.U2Slist['3-position'][i])		# temp, vx, vy, vz
			# only consider dissociation and recombination if in the de-confined phase
			if T_Vxyz[0] >= Tc:	
				p4_in_hydrocell = LorRot.lorentz(self.U2Slist['4-momentum'][i], T_Vxyz[1:])		# boost to hydro cell
				v3_in_hydrocell = p4_in_hydrocell[1:]/p4_in_hydrocell[0]
				v_in_hydrocell = np.sqrt(np.sum(v3_in_hydrocell**2))

				rate_decay2S_gluon = self.event.get_R2S_decay_gluon( v_in_hydrocell, T_Vxyz[0] )		# GeV
				rate_decay2S_ineq = self.event.get_R2S_decay_ineq( v_in_hydrocell, T_Vxyz[0] )
				rate_decay2S_ineg = self.event.get_R2S_decay_ineg( v_in_hydrocell, T_Vxyz[0] )
				
				# in lab frame decay probability
				# dt is time in lab frame, dt' = dt*E'/E is time in hydro cell frame
				prob_decay2S_gluon = rate_decay2S_gluon * dt/C1 * p4_in_hydrocell[0]/self.U2Slist['4-momentum'][i][0]
				prob_decay2S_ineq = rate_decay2S_ineq * dt/C1 * p4_in_hydrocell[0]/self.U2Slist['4-momentum'][i][0]
				prob_decay2S_ineg = rate_decay2S_ineg * dt/C1 * p4_in_hydrocell[0]/self.U2Slist['4-momentum'][i][0]
				
				rej_decay2S_mc = np.random.rand(1)
				if rej_decay2S_mc <= prob_decay2S_gluon + prob_decay2S_ineq + prob_decay2S_ineg:
					delete_U2S.append(i)
					if rej_decay2S_mc <= prob_decay2S_gluon:
						recoil_pQpQbar = self.event.sample_S2S_decay_gluon( v_in_hydrocell, T_Vxyz[0] )
					elif rej_decay2S_mc <= prob_decay2S_gluon + prob_decay2S_ineq:
						recoil_pQpQbar = self.event.sample_S2S_decay_ineq( v_in_hydrocell, T_Vxyz[0] )
					else:
						recoil_pQpQbar = self.event.sample_S2S_decay_ineg( v_in_hydrocell, T_Vxyz[0] )
										
					recoil_pQ = np.array(recoil_pQpQbar[0:4])		# 4-momentum
					recoil_pQbar = np.array(recoil_pQpQbar[4:8])

					# Q, Qbar momenta need to be rotated from the v = z axis to hydro cell frame
					# first get the rotation matrix angles
					theta_rot, phi_rot = LorRot.angle( v3_in_hydrocell )
					# then do the rotation
					rotmomentum_Q = LorRot.rotation4(recoil_pQ, theta_rot, phi_rot)
					rotmomentum_Qbar = LorRot.rotation4(recoil_pQbar, theta_rot, phi_rot)
					# we now transform them back to the hydro cell frame
					momentum_Q = LorRot.lorentz(rotmomentum_Q, -v3_in_hydrocell)
					momentum_Qbar = LorRot.lorentz(rotmomentum_Qbar, -v3_in_hydrocell)
					# then transform back to the lab frame
					momentum_Q = LorRot.lorentz(momentum_Q, -T_Vxyz[1:])
					momentum_Qbar = LorRot.lorentz(momentum_Qbar, -T_Vxyz[1:])

					# positions of Q and Qbar
					position_Q = self.U2Slist['3-position'][i]
					#position_Qbar = position_Q
		
					# add x and p for the QQbar to the temporary list
					add_pQ.append(momentum_Q)
					add_pQbar.append(momentum_Qbar)
					add_xQ.append(position_Q)
					#add_xQbar.append(position_Qbar)
					add_t23.append(self.t)
					add_t32.append(self.t)
					add_dt_last.append(0.0)		
		
		### --------- 1P decay ---------- ###
		for i in range(len_U1P):
			T_Vxyz = self.hydro.cell_info(self.t, self.U1Plist['3-position'][i])		# temp, vx, vy, vz
			# only consider dissociation and recombination if in the de-confined phase
			if T_Vxyz[0] >= Tc:	
				p4_in_hydrocell = LorRot.lorentz(self.U1Plist['4-momentum'][i], T_Vxyz[1:])		# boost to hydro cell
				v3_in_hydrocell = p4_in_hydrocell[1:]/p4_in_hydrocell[0]
				v_in_hydrocell = np.sqrt(np.sum(v3_in_hydrocell**2))

				rate_decay1P_gluon = self.event.get_R1P_decay_gluon( v_in_hydrocell, T_Vxyz[0] )		# GeV
				rate_decay1P_ineq = self.event.get_R1P_decay_ineq( v_in_hydrocell, T_Vxyz[0] )
				rate_decay1P_ineg = self.event.get_R1P_decay_ineg( v_in_hydrocell, T_Vxyz[0] )
				
				# in lab frame decay probability
				# dt is time in lab frame, dt' = dt*E'/E is time in hydro cell frame
				prob_decay1P_gluon = rate_decay1P_gluon * dt/C1 * p4_in_hydrocell[0]/self.U1Plist['4-momentum'][i][0]
				prob_decay1P_ineq = rate_decay1P_ineq * dt/C1 * p4_in_hydrocell[0]/self.U1Plist['4-momentum'][i][0]
				prob_decay1P_ineg = rate_decay1P_ineg * dt/C1 * p4_in_hydrocell[0]/self.U1Plist['4-momentum'][i][0]
				
				rej_decay1P_mc = np.random.rand(1)
				if rej_decay1P_mc <= prob_decay1P_gluon + prob_decay1P_ineq + prob_decay1P_ineg:
					delete_U1P.append(i)
					if rej_decay1P_mc <= prob_decay1P_gluon:
						recoil_pQpQbar = self.event.sample_S1P_decay_gluon( v_in_hydrocell, T_Vxyz[0] )
					elif rej_decay1P_mc <= prob_decay1P_gluon + prob_decay1P_ineq:
						recoil_pQpQbar = self.event.sample_S1P_decay_ineq( v_in_hydrocell, T_Vxyz[0] )
					else:
						recoil_pQpQbar = self.event.sample_S1P_decay_ineg( v_in_hydrocell, T_Vxyz[0] )
										
					recoil_pQ = np.array(recoil_pQpQbar[0:4])		# 4-momentum
					recoil_pQbar = np.array(recoil_pQpQbar[4:8])

					# Q, Qbar momenta need to be rotated from the v = z axis to hydro cell frame
					# first get the rotation matrix angles
					theta_rot, phi_rot = LorRot.angle( v3_in_hydrocell )
					# then do the rotation
					rotmomentum_Q = LorRot.rotation4(recoil_pQ, theta_rot, phi_rot)
					rotmomentum_Qbar = LorRot.rotation4(recoil_pQbar, theta_rot, phi_rot)
					# we now transform them back to the hydro cell frame
					momentum_Q = LorRot.lorentz(rotmomentum_Q, -v3_in_hydrocell)
					momentum_Qbar = LorRot.lorentz(rotmomentum_Qbar, -v3_in_hydrocell)
					# then transform back to the lab frame
					momentum_Q = LorRot.lorentz(momentum_Q, -T_Vxyz[1:])
					momentum_Qbar = LorRot.lorentz(momentum_Qbar, -T_Vxyz[1:])

					# positions of Q and Qbar
					position_Q = self.U1Plist['3-position'][i]
					#position_Qbar = position_Q
		
					# add x and p for the QQbar to the temporary list
					add_pQ.append(momentum_Q)
					add_pQbar.append(momentum_Qbar)
					add_xQ.append(position_Q)
					#add_xQbar.append(position_Qbar)
					add_t23.append(self.t)
					add_t32.append(self.t)
					add_dt_last.append(0.0)		
		
		for i in range(len_U3S):
			T_Vxyz = self.hydro.cell_info(self.t, self.U3Slist['3-position'][i])
			if T_Vxyz[0] >= Tc:
				delete_U3S.append(i)
				
				# momenta of Q and Qbar
				momentum_U3S_half = self.U3Slist['4-momentum'][i][1:]/2.
				energy_Q = np.sqrt( np.sum(momentum_U3S_half**2) + self.M**2 )
				momentum_Q = np.array( [energy_Q,momentum_U3S_half[0],momentum_U3S_half[1],momentum_U3S_half[2]] )
				momentum_Qbar = np.array( [energy_Q,momentum_U3S_half[0],momentum_U3S_half[1],momentum_U3S_half[2]] )
				
				# positions of Q and Qbar
				position_Q = self.U3Slist['3-position'][i]
				#position_Qbar = position_Q
				
				# add x and p for the QQbar to the temporary list
				add_pQ.append(momentum_Q)
				add_pQbar.append(momentum_Qbar)
				add_xQ.append(position_Q)
				#add_xQbar.append(position_Qbar)
				add_t23.append(self.t)
				add_t32.append(self.t)
				add_dt_last.append(0.0)
				
		### ------------------ end of decay ------------------- ###
		
		
		
		### ------------------ recombination ------------------ ###
		if self.recombine == True:
			delete_Q = []
			delete_Qbar = []
			
			if len_Q*len_Qbar != 0:
				pair_search = cKDTree(self.Qbarlist['3-position'])
				# for each Q, obtain the Qbar indexes within R_search
				pair_list = pair_search.query_ball_point(self.Qlist['3-position'], r = R_search)
			
			for i in range(len_Q):					# loop over Q
				len_recoQbar = len(pair_list[i])
				rate_reco1S_gluon = []
				rate_reco1S_ineq = []
				rate_reco1S_ineg = []
				rate_reco2S_gluon = []
				rate_reco2S_ineq = []
				rate_reco2S_ineg = []
				rate_reco1P_gluon = []
				rate_reco1P_ineq = []
				rate_reco1P_ineg = []
				for j in range(len_recoQbar):		# loop over Qbar within R_search
					# positions in lab frame
					xQ = self.Qlist['3-position'][i]
					xQbar = self.Qbarlist['3-position'][pair_list[i][j]]
					x_rel = xQ - xQbar
					x_CM = 0.5*( xQ + xQbar )					
					T_Vxyz = self.hydro.cell_info(self.t, x_CM)
					#rdotp = np.sum( x_rel* (self.Qlist['4-momentum'][i][1:] - self.Qbarlist['4-momentum'][pair_list[i][j]][1:]) )
					#if T_Vxyz[0] >= Tc and rdotp < 0.0 and pair_list[i][j] not in delete_Qbar:
					if  T_Vxyz[0] >= Tc and T_Vxyz[0] <= T_1S and pair_list[i][j] not in delete_Qbar:
						r_rel = np.sqrt(np.sum(x_rel**2))
						pQ_lab = self.Qlist['4-momentum'][i]
						pQbar_lab = self.Qbarlist['4-momentum'][pair_list[i][j]]
						# CM energy in the lab frame
						p_CMlab = pQ_lab[1:] + pQbar_lab[1:]
						E_CMlab = np.sqrt(np.sum(p_CMlab**2) + (2.*self.M)**2)
						
						# momenta into hydro cell frame
						pQ = LorRot.lorentz(pQ_lab, T_Vxyz[1:])
						pQbar = LorRot.lorentz(pQbar_lab, T_Vxyz[1:])
						
						# 3-velocity, velocity magnitude, relative momentum in hydro cell frame
						v_CM, v_CM_abs, p_rel_abs = LorRot.vCM_prel(pQ, pQbar, 2.0*self.M)
						E_CM = 2.0*M/np.sqrt(1.-v_CM_abs**2)
						
						rate_reco1S_gluon.append(self.event.get_R1S_reco_gluon(v_CM_abs, T_Vxyz[0], p_rel_abs, r_rel)*E_CM/E_CMlab)
						rate_reco1S_ineq.append(self.event.get_R1S_reco_ineq(v_CM_abs, T_Vxyz[0], p_rel_abs, r_rel)*E_CM/E_CMlab)
						rate_reco1S_ineg.append(self.event.get_R1S_reco_ineg(v_CM_abs, T_Vxyz[0], p_rel_abs, r_rel)*E_CM/E_CMlab)
						# we move the E_CMcell / E_CMlab above from below to avoid non-defination
						
						# 2S reco rate:
						if T_Vxyz[0] <= T_2S:
							rate_reco2S_gluon.append(self.event.get_R2S_reco_gluon(v_CM_abs, T_Vxyz[0], p_rel_abs, r_rel)*E_CM/E_CMlab)
							rate_reco2S_ineq.append(self.event.get_R2S_reco_ineq(v_CM_abs, T_Vxyz[0], p_rel_abs, r_rel)*E_CM/E_CMlab)
							rate_reco2S_ineg.append(self.event.get_R2S_reco_ineg(v_CM_abs, T_Vxyz[0], p_rel_abs, r_rel)*E_CM/E_CMlab)
						else:
							rate_reco2S_gluon.append(0.)
							rate_reco2S_ineq.append(0.)
							rate_reco2S_ineg.append(0.)
						
						# 1P reco rate:
						if T_Vxyz[0] <= T_1P:
							rate_reco1P_gluon.append(self.event.get_R1P_reco_gluon(v_CM_abs, T_Vxyz[0], p_rel_abs, r_rel)*E_CM/E_CMlab)
							rate_reco1P_ineq.append(self.event.get_R1P_reco_ineq(v_CM_abs, T_Vxyz[0], p_rel_abs, r_rel)*E_CM/E_CMlab)
							rate_reco1P_ineg.append(self.event.get_R1P_reco_ineg(v_CM_abs, T_Vxyz[0], p_rel_abs, r_rel)*E_CM/E_CMlab)
						else:
							rate_reco1P_gluon.append(0.)
							rate_reco1P_ineq.append(0.)
							rate_reco1P_ineg.append(0.)
							
					else:
						rate_reco1S_gluon.append(0.)
						rate_reco1S_ineq.append(0.)
						rate_reco1S_ineg.append(0.)
						rate_reco2S_gluon.append(0.)
						rate_reco2S_ineq.append(0.)
						rate_reco2S_ineg.append(0.)
						rate_reco1P_gluon.append(0.)
						rate_reco1P_ineq.append(0.)
						rate_reco1P_ineg.append(0.)
							
				# get the recombine probability in the hydro cell frame
				# dt is in lab frame; in hydro cell dt' = E_CMcell / E_CMlab * dt
				# spin=1 gives 3/4 factor for Upsilon(1S, 2S) and chi_b(1P) v.s. eta_b and h_b
				prob_reco1S_gluon = 2./3.*np.array(rate_reco1S_gluon)*dt/C1
				prob_reco1S_ineq = 2./3.*np.array(rate_reco1S_ineq)*dt/C1
				prob_reco1S_ineg = 2./3.*np.array(rate_reco1S_ineg)*dt/C1
				prob_reco2S_gluon = 2./3.*np.array(rate_reco2S_gluon)*dt/C1
				prob_reco2S_ineq = 2./3.*np.array(rate_reco2S_ineq)*dt/C1
				prob_reco2S_ineg = 2./3.*np.array(rate_reco2S_ineg)*dt/C1
				prob_reco1P_gluon = 2./3.*np.array(rate_reco1P_gluon)*dt/C1
				prob_reco1P_ineq = 2./3.*np.array(rate_reco1P_ineq)*dt/C1
				prob_reco1P_ineg = 2./3.*np.array(rate_reco1P_ineg)*dt/C1
				prob_reco1S_all = prob_reco1S_gluon + prob_reco1S_ineq + prob_reco1S_ineg
				prob_reco2S_all = prob_reco2S_gluon + prob_reco2S_ineq + prob_reco2S_ineg
				prob_reco1P_all = prob_reco1P_gluon + prob_reco1P_ineq + prob_reco1P_ineg
				total_prob_reco1S = np.sum(prob_reco1S_all)
				total_prob_reco2S = np.sum(prob_reco2S_all)
				total_prob_reco1P = np.sum(prob_reco1P_all)
				rej_reco_mc = np.random.rand(1)
				
				if rej_reco_mc <= total_prob_reco1S:
					delete_Q.append(i)		# remove this Q later
					# find the Qbar we need to remove
					a = 0.0
					for j in range(len_recoQbar):
						if a <= rej_reco_mc <= a + prob_reco1S_all[j]:
							k = j
							if rej_reco_mc <= a + prob_reco1S_gluon[j]:
								channel_reco = 'gluon'
							elif rej_reco_mc <= a + prob_reco1S_gluon[j] + prob_reco1S_ineq[j]:
								channel_reco = 'ineq'
							else:
								channel_reco = 'ineg'
							break
						a += prob_reco1S_all[j]
					delete_Qbar.append(pair_list[i][k])
					
					# re-construct the reco event and sample initial and final states
					# positions and local temperature
					xQ = self.Qlist['3-position'][i]
					xQbar = self.Qbarlist['3-position'][pair_list[i][k]]
					x_CM = 0.5*( xQ + xQbar )					
					T_Vxyz = self.hydro.cell_info(self.t, x_CM)
					
					# 4-momenta in the hydro cell frame
					pQ = LorRot.lorentz(self.Qlist['4-momentum'][i], T_Vxyz[1:])
					pQbar = LorRot.lorentz(self.Qbarlist['4-momentum'][pair_list[i][k]], T_Vxyz[1:])
					
					# 3-velocity, velocity, relative momentum in hydro cell frame
					v_CM, v_CM_abs, p_rel_abs = LorRot.vCM_prel(pQ, pQbar, 2.0*self.M)

					# calculate the final quarkonium momenta in the CM frame of QQbar, depends on channel_reco
					if channel_reco == 'gluon':
						tempmomentum_U1S = np.array(self.event.sample_S1S_reco_gluon(v_CM_abs, T_Vxyz[0], p_rel_abs))
					elif channel_reco == 'ineq':
						tempmomentum_U1S = np.array(self.event.sample_S1S_reco_ineq(v_CM_abs, T_Vxyz[0], p_rel_abs))
					else:	# 'ineg'
						tempmomentum_U1S = np.array(self.event.sample_S1S_reco_ineg(v_CM_abs, T_Vxyz[0], p_rel_abs))
					
					# need to rotate the vector, v is not the z axis in hydro cell frame
					theta_rot, phi_rot = LorRot.angle(v_CM)
					rotmomentum_U1S = LorRot.rotation4(tempmomentum_U1S, theta_rot, phi_rot)

					# lorentz back to the hydro cell frame
					momentum_U1S = LorRot.lorentz( rotmomentum_U1S, -v_CM )
					# lorentz back to the lab frame
					momentum_U1S = LorRot.lorentz( momentum_U1S, -T_Vxyz[1:] )
					
					# positions of the quarkonium
					position_U1S = x_CM
					
					# update the quarkonium list
					if len(self.U1Slist['4-momentum']) == 0:
						self.U1Slist['4-momentum'] = np.array([momentum_U1S])
						self.U1Slist['3-position'] = np.array([position_U1S])
						self.U1Slist['last_form_time'] = np.array([self.t])
					else:
						self.U1Slist['4-momentum'] = np.append(self.U1Slist['4-momentum'], [momentum_U1S], axis=0)
						self.U1Slist['3-position'] = np.append(self.U1Slist['3-position'], [position_U1S], axis=0)
						self.U1Slist['last_form_time'] = np.append(self.U1Slist['last_form_time'], self.t)
						
				elif rej_reco_mc <= total_prob_reco1S + total_prob_reco2S:
					delete_Q.append(i)		# remove this Q later
					# find the Qbar we need to remove					
					a = total_prob_reco1S
					for j in range(len_recoQbar):
						if a <= rej_reco_mc <= a + prob_reco2S_all[j]:
							k = j
							if rej_reco_mc <= a + prob_reco2S_gluon[j]:
								channel_reco = 'gluon'
							elif rej_reco_mc <= a + prob_reco2S_gluon[j] + prob_reco2S_ineq[j]:
								channel_reco = 'ineq'
							else:
								channel_reco = 'ineg'
							break
						a += prob_reco2S_all[j]
					delete_Qbar.append(pair_list[i][k])
					
					# re-construct the reco event and sample initial and final states
					# positions and local temperature
					xQ = self.Qlist['3-position'][i]
					xQbar = self.Qbarlist['3-position'][pair_list[i][k]]
					x_CM = 0.5*( xQ + xQbar )					
					T_Vxyz = self.hydro.cell_info(self.t, x_CM)
					
					# 4-momenta in the hydro cell frame
					pQ = LorRot.lorentz(self.Qlist['4-momentum'][i], T_Vxyz[1:])
					pQbar = LorRot.lorentz(self.Qbarlist['4-momentum'][pair_list[i][k]], T_Vxyz[1:])
					
					# 3-velocity, velocity, relative momentum in hydro cell frame
					v_CM, v_CM_abs, p_rel_abs = LorRot.vCM_prel(pQ, pQbar, 2.0*self.M)

					# calculate the final quarkonium momenta in the CM frame of QQbar, depends on channel_reco
					if channel_reco == 'gluon':
						tempmomentum_U2S = np.array(self.event.sample_S2S_reco_gluon(v_CM_abs, T_Vxyz[0], p_rel_abs))
					elif channel_reco == 'ineq':
						tempmomentum_U2S = np.array(self.event.sample_S2S_reco_ineq(v_CM_abs, T_Vxyz[0], p_rel_abs))
					else:	# 'ineg'
						tempmomentum_U2S = np.array(self.event.sample_S2S_reco_ineg(v_CM_abs, T_Vxyz[0], p_rel_abs))
					
					# need to rotate the vector, v is not the z axis in hydro cell frame
					theta_rot, phi_rot = LorRot.angle(v_CM)
					rotmomentum_U2S = LorRot.rotation4(tempmomentum_U2S, theta_rot, phi_rot)

					# lorentz back to the hydro cell frame
					momentum_U2S = LorRot.lorentz( rotmomentum_U2S, -v_CM )
					# lorentz back to the lab frame
					momentum_U2S = LorRot.lorentz( momentum_U2S, -T_Vxyz[1:] )
					
					# positions of the quarkonium
					position_U2S = x_CM
					
					# update the quarkonium list
					if len(self.U2Slist['4-momentum']) == 0:
						self.U2Slist['4-momentum'] = np.array([momentum_U2S])
						self.U2Slist['3-position'] = np.array([position_U2S])
						self.U2Slist['last_form_time'] = np.array([self.t])
					else:
						self.U2Slist['4-momentum'] = np.append(self.U2Slist['4-momentum'], [momentum_U2S], axis=0)
						self.U2Slist['3-position'] = np.append(self.U2Slist['3-position'], [position_U2S], axis=0)
						self.U2Slist['last_form_time'] = np.append(self.U2Slist['last_form_time'], self.t)
				
				
				elif rej_reco_mc <= total_prob_reco1S + total_prob_reco2S + total_prob_reco1P:
					delete_Q.append(i)		# remove this Q later
					# find the Qbar we need to remove					
					a = total_prob_reco1S + total_prob_reco2S
					for j in range(len_recoQbar):
						if a <= rej_reco_mc <= a + prob_reco1P_all[j]:
							k = j
							if rej_reco_mc <= a + prob_reco1P_gluon[j]:
								channel_reco = 'gluon'
							elif rej_reco_mc <= a + prob_reco1P_gluon[j] + prob_reco1P_ineq[j]:
								channel_reco = 'ineq'
							else:
								channel_reco = 'ineg'
							break
						a += prob_reco1P_all[j]
					delete_Qbar.append(pair_list[i][k])
					
					# re-construct the reco event and sample initial and final states
					# positions and local temperature
					xQ = self.Qlist['3-position'][i]
					xQbar = self.Qbarlist['3-position'][pair_list[i][k]]
					x_CM = 0.5*( xQ + xQbar )					
					T_Vxyz = self.hydro.cell_info(self.t, x_CM)
					
					# 4-momenta in the hydro cell frame
					pQ = LorRot.lorentz(self.Qlist['4-momentum'][i], T_Vxyz[1:])
					pQbar = LorRot.lorentz(self.Qbarlist['4-momentum'][pair_list[i][k]], T_Vxyz[1:])
					
					# 3-velocity, velocity magnitude, relative momentum in hydro cell frame
					v_CM, v_CM_abs, p_rel_abs = LorRot.vCM_prel(pQ, pQbar, 2.0*self.M)

					# calculate the final quarkonium momenta in the CM frame of QQbar, depends on channel_reco
					if channel_reco == 'gluon':
						tempmomentum_U1P = np.array(self.event.sample_S1P_reco_gluon(v_CM_abs, T_Vxyz[0], p_rel_abs))
					elif channel_reco == 'ineq':
						tempmomentum_U1P = np.array(self.event.sample_S1P_reco_ineq(v_CM_abs, T_Vxyz[0], p_rel_abs))
					else:	# 'ineg'
						tempmomentum_U1P = np.array(self.event.sample_S1P_reco_ineg(v_CM_abs, T_Vxyz[0], p_rel_abs))
					
					# need to rotate the vector, v is not the z axis in hydro cell frame
					theta_rot, phi_rot = LorRot.angle(v_CM)
					rotmomentum_U1P = LorRot.rotation4(tempmomentum_U1P, theta_rot, phi_rot)

					# lorentz back to the hydro cell frame
					momentum_U1P = LorRot.lorentz( rotmomentum_U1P, -v_CM )
					# lorentz back to the lab frame
					momentum_U1P = LorRot.lorentz( momentum_U1P, -T_Vxyz[1:] )
					
					# positions of the quarkonium
					position_U1P = x_CM
					
					# update the quarkonium list
					if len(self.U1Plist['4-momentum']) == 0:
						self.U1Plist['4-momentum'] = np.array([momentum_U1P])
						self.U1Plist['3-position'] = np.array([position_U1P])
						self.U1Plist['last_form_time'] = np.array([self.t])
					else:
						self.U1Plist['4-momentum'] = np.append(self.U1Plist['4-momentum'], [momentum_U1P], axis=0)
						self.U1Plist['3-position'] = np.append(self.U1Plist['3-position'], [position_U1P], axis=0)
						self.U1Plist['last_form_time'] = np.append(self.U1Plist['last_form_time'], self.t)

						
			## now update Q and Qbar lists
			self.Qlist['4-momentum'] = np.delete(self.Qlist['4-momentum'], delete_Q, axis=0)
			self.Qlist['3-position'] = np.delete(self.Qlist['3-position'], delete_Q, axis=0)
			self.Qlist['last_t23'] = np.delete(self.Qlist['last_t23'], delete_Q)
			self.Qlist['last_t32'] = np.delete(self.Qlist['last_t32'], delete_Q)
			self.Qlist['last_scatter_dt'] = np.delete(self.Qlist['last_scatter_dt'], delete_Q)
			self.Qbarlist['4-momentum'] = np.delete(self.Qbarlist['4-momentum'], delete_Qbar, axis=0)
			self.Qbarlist['3-position'] = np.delete(self.Qbarlist['3-position'], delete_Qbar, axis=0)
			self.Qbarlist['last_t23'] = np.delete(self.Qbarlist['last_t23'], delete_Qbar)
			self.Qbarlist['last_t32'] = np.delete(self.Qbarlist['last_t32'], delete_Qbar)
			self.Qbarlist['last_scatter_dt'] = np.delete(self.Qbarlist['last_scatter_dt'], delete_Qbar)
		### -------------- end of recombination --------------- ###
		
		
		### -------------- update lists due to decay ---------- ###
		add_pQ = np.array(add_pQ)
		add_pQbar = np.array(add_pQbar)
		add_xQ = np.array(add_xQ)
		#add_xQbar = np.array(add_xQbar)
		add_t23 = np.array(add_t23)
		add_t32 = np.array(add_t32)
		add_dt_last = np.array(add_dt_last)
		if len(add_pQ):	
			# if there is at least quarkonium decays, we need to update all the three lists
			self.U1Slist['3-position'] = np.delete(self.U1Slist['3-position'], delete_U1S, axis=0) # delete along the axis = 0
			self.U1Slist['4-momentum'] = np.delete(self.U1Slist['4-momentum'], delete_U1S, axis=0)
			self.U1Slist['last_form_time'] = np.delete(self.U1Slist['last_form_time'], delete_U1S)
			self.U2Slist['3-position'] = np.delete(self.U2Slist['3-position'], delete_U2S, axis=0)
			self.U2Slist['4-momentum'] = np.delete(self.U2Slist['4-momentum'], delete_U2S, axis=0)
			self.U2Slist['last_form_time'] = np.delete(self.U2Slist['last_form_time'], delete_U2S)
			self.U1Plist['3-position'] = np.delete(self.U1Plist['3-position'], delete_U1P, axis=0)
			self.U1Plist['4-momentum'] = np.delete(self.U1Plist['4-momentum'], delete_U1P, axis=0)
			self.U1Plist['last_form_time'] = np.delete(self.U1Plist['last_form_time'], delete_U1P)
			self.U3Slist['3-position'] = np.delete(self.U3Slist['3-position'], delete_U3S, axis=0)
			self.U3Slist['4-momentum'] = np.delete(self.U3Slist['4-momentum'], delete_U3S, axis=0)
			self.U3Slist['last_form_time'] = np.delete(self.U3Slist['last_form_time'], delete_U3S)

			if len(self.Qlist['4-momentum']) == 0:
				self.Qlist['3-position'] = np.array(add_xQ)
				self.Qlist['4-momentum'] = np.array(add_pQ)
				self.Qlist['last_t23'] = np.array(add_t23)
				self.Qlist['last_t32'] = np.array(add_t32)
				self.Qlist['last_scatter_dt'] = np.array(add_dt_last)
			else:
				self.Qlist['3-position'] = np.append(self.Qlist['3-position'], add_xQ, axis=0)
				self.Qlist['4-momentum'] = np.append(self.Qlist['4-momentum'], add_pQ, axis=0)
				self.Qlist['last_t23'] = np.append(self.Qlist['last_t23'], add_t23)
				self.Qlist['last_t32'] = np.append(self.Qlist['last_t23'], add_t32)
				self.Qlist['last_scatter_dt'] = np.append(self.Qlist['last_scatter_dt'], add_dt_last)
			if len(self.Qbarlist['4-momentum']) == 0:
				self.Qbarlist['3-position'] = np.array(add_xQ)
				self.Qbarlist['4-momentum'] = np.array(add_pQbar)
				self.Qbarlist['last_t23'] = np.array(add_t23)
				self.Qbarlist['last_t32'] = np.array(add_t32)
				self.Qbarlist['last_scatter_dt'] = np.array(add_dt_last)
			else:
				self.Qbarlist['3-position'] = np.append(self.Qbarlist['3-position'], add_xQ, axis=0)
				self.Qbarlist['4-momentum'] = np.append(self.Qbarlist['4-momentum'], add_pQbar, axis=0)
				self.Qbarlist['last_t23'] = np.append(self.Qbarlist['last_t23'], add_t23)
				self.Qbarlist['last_t32'] = np.append(self.Qbarlist['last_t23'], add_t32)		
				self.Qbarlist['last_scatter_dt'] = np.append(self.Qbarlist['last_scatter_dt'], add_dt_last)
		### ---------- end of update lists due to decay ------- ###
		

#### ----------------- end of evolution function ----------------- ####	

#### ----------------- define a clear function ------------------- ####
	def dict_clear(self):
		self.Qlist.clear()
		self.Qbarlist.clear()
		self.U1Slist.clear()
		self.U2Slist.clear()
		self.U1Plist.clear()
		self.U3Slist.clear()





