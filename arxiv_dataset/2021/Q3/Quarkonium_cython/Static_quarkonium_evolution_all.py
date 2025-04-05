#!/usr/bin/env python
import numpy as np
import h5py
import csv
import random as rd
from scipy.spatial import cKDTree
from Static_Initial_Sample import Static_Initial_Sample
from Static_HQ_evo import HQ_p_update
import DisRec
import LorRot


#### ---------------------- some constants -----------------------------
C1 = 0.197327                 # 0.197 GeV*fm = 1
R_search = 2.0				  # (fm), pair-search radius in the recombination


####--------- initial sample of heavy Q and Qbar using thermal distribution -------- ####
def thermal_dist(temp, mass, momentum):
    return momentum**2*np.exp(-np.sqrt(mass**2+momentum**2)/temp)


# sample according to the thermal distribution function
def thermal_sample(temp, mass):
    p_max = np.sqrt(2.0*temp**2 + 2.0*temp*np.sqrt(temp**2+mass**2))
    # most probably momentum
    p_uplim = 10.0*p_max
    y_uplim = thermal_dist(temp, mass, p_max)
    while True:
        p_try = rd.uniform(0.0, p_uplim)
        y_try = rd.uniform(0.0, y_uplim)
        if y_try < thermal_dist(temp, mass, p_try):
            break
        
    E = np.sqrt(p_try**2+mass**2)
    cos_theta = rd.uniform(-1.0, 1.0)
    sin_theta = np.sqrt(1.0-cos_theta**2)
    phi = rd.uniform(0.0, 2.0*np.pi)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    return np.array([ E, p_try*sin_theta*cos_phi, p_try*sin_theta*sin_phi, p_try*cos_theta ])

####---- end of initial sample of heavy Q and Qbar using thermal distribution ---- ####




####--------- initial sample of heavy Q and Qbar using uniform distribution ------ ####
def uniform_sample(Pmax, mass):
    px, py, pz = (np.random.rand(3)-0.5)*2*Pmax
    E = np.sqrt(mass**2 + px**2 + py**2 + pz**2)
    return np.array([E, px, py, pz])

####----- end of initial sample of heavy Q and Qbar using uniform distribution ---- ####



class QQbar_evol:
####---- input the medium_type when calling the class ----####
	def __init__(self, medium_type = 'static', process = 'all', quarkonium_species = '1S', temp_init = 0.3, recombine = True, HQ_scat = False):
		self.type = medium_type
		self.process = process	# gluon: real-gluon, ineq: inelastic quark, ineg: inelastic gluon
		self.quarkonium_species = quarkonium_species
		self.T = temp_init		# initial temperature in GeV	
		self.recombine = recombine
		self.HQ_scat = HQ_scat
		if self.HQ_scat == True:
			self.HQ_diff = HQ_p_update()
		## ---------- create the rates reader --------- ##
		self.event = DisRec.DisRec()
		self.M = DisRec.pyM()
		self.M1S = DisRec.pyM1S()
		self.M2S = DisRec.pyM2S()
		self.M1P = DisRec.pyM1P()
		
####---- initialize Q, Qbar, Quarkonium -- currently we only study Upsilon(1S) ----####
	def initialize(self, N_Q = 100, N_Qbar = 100, N_U1S = 0, N_U2S = 0, N_U1P = 0, Lmax = 10.0, thermal_dist = False,
	fonll_dist = False, Fonll_path = False, uniform_dist = False, Pmax = 10.0, decaytestmode = False, decaystate = '1S', P_decaytest = [0.0, 0.0, 5.0]):
		# initial momentum: thermal; Fonll (give the fonll file path), uniform in x,y,z (give the Pmax in GeV)
        # if decaytestmode: give initial Px,Py,Pz in GeV
		if self.type == 'static':
			## ----- initialize the clock to keep track of time
			self.t = 0.0
			## ----- store the box side length
			self.Lmax = Lmax
			## ----- create dictionaries to store momenta, positions, id ----- ##
			self.Qlist = {'4-momentum': [], '3-position': [], 'id': 5}
			self.Qbarlist = {'4-momentum': [], '3-position': [], 'id': -5}
			self.U1Slist = {'4-momentum': [], '3-position': [], 'id': 553, 'last_form_time': []}
			self.U2Slist = {'4-momentum': [], '3-position': [], 'id': 100553, 'last_form_time': []}
			self.U1Plist = {'4-momentum': [], '3-position': [], 'id': 555, 'last_form_time': []}
			
			## --------- sample initial momenta and positions -------- ##
			if thermal_dist == True:
				for i in range(N_Q):
					self.Qlist['4-momentum'].append( thermal_sample(self.T, self.M) )
					self.Qlist['3-position'].append( np.random.rand(3)*Lmax )
				for i in range(N_Qbar):
					self.Qbarlist['4-momentum'].append( thermal_sample(self.T, self.M) )
					self.Qbarlist['3-position'].append( np.random.rand(3)*Lmax )
				for i in range(N_U1S):
					self.U1Slist['4-momentum'].append( thermal_sample(self.T, self.M_1S) )
					self.U1Slist['3-position'].append( np.random.rand(3)*Lmax )
					self.U1Slist['last_form_time'].append( self.t )
				for i in range(N_U2S):
					self.U2Slist['4-momentum'].append( thermal_sample(self.T, self.M_2S) )
					self.U2Slist['3-position'].append( np.random.rand(3)*Lmax )
					self.U2Slist['last_form_time'].append( self.t )
				for i in range(N_U1P):
					self.U1Plist['4-momentum'].append( thermal_sample(self.T, self.M_1P) )
					self.U1Plist['3-position'].append( np.random.rand(3)*Lmax )
					self.U1Plist['last_form_time'].append( self.t )
													
			if fonll_dist == True:
				p_generator = Static_Initial_Sample(Fonll_path, rapidity = 0.)
				
				for i in range(N_Q):
					self.Qlist['4-momentum'].append( p_generator.p_HQ_sample(self.M) )
					self.Qlist['3-position'].append( np.random.rand(3)*Lmax )
				for i in range(N_Qbar):
					self.Qbarlist['4-momentum'].append( p_generator.p_HQ_sample(self.M) )
					self.Qbarlist['3-position'].append( np.random.rand(3)*Lmax )
				for i in range(N_U1S):
					self.U1Slist['4-momentum'].append( p_generator.p_U_sample(self.M, self.M_1S) )
					self.U1Slist['3-position'].append( np.random.rand(3)*Lmax )
					self.U1Slist['last_form_time'].append( self.t )
				for i in range(N_U2S):
					self.U2Slist['4-momentum'].append( p_generator.p_U_sample(self.M, self.M_2S) )
					self.U2Slist['3-position'].append( np.random.rand(3)*Lmax )
					self.U2Slist['last_form_time'].append( self.t )
				for i in range(N_U1P):
					self.U1Plist['4-momentum'].append( p_generator.p_U_sample(self.M, self.M_1P) )
					self.U1Plist['3-position'].append( np.random.rand(3)*Lmax )
					self.U1Plist['last_form_time'].append( self.t )
													
			if uniform_dist == True:
				for i in range(N_Q):
					self.Qlist['4-momentum'].append( uniform_sample(Pmax, self.M) )
					self.Qlist['3-position'].append( np.random.rand(3)*Lmax )
				for i in range(N_Qbar):
					self.Qbarlist['4-momentum'].append( uniform_sample(Pmax, self.M) )
					self.Qbarlist['3-position'].append( np.random.rand(3)*Lmax )
				for i in range(N_U1S):
					self.U1Slist['4-momentum'].append( uniform_sample(Pmax, self.M_1S) )
					self.U1Slist['3-position'].append( np.random.rand(3)*Lmax )
					self.U1Slist['last_form_time'].append( self.t )
				for i in range(N_U2S):
					self.U2Slist['4-momentum'].append( uniform_sample(Pmax, self.M_2S) )
					self.U2Slist['3-position'].append( np.random.rand(3)*Lmax )
					self.U2Slist['last_form_time'].append( self.t )
				for i in range(N_U1P):
					self.U1Plist['4-momentum'].append( uniform_sample(Pmax, self.M_1P) )
					self.U1Plist['3-position'].append( np.random.rand(3)*Lmax )
					self.U1Plist['last_form_time'].append( self.t )
													
			if decaytestmode == True:
				if decaystate == '1S':
					E_decaytest_1S = np.sqrt(self.M_1S**2 + P_decaytest[0]**2 + P_decaytest[1]**2 + P_decaytest[2]**2)
					p4_1S = np.append(E_decaytest_1S, P_decaytest)
					p4_2S = [self.M_2S, 0.0, 0.0, 0.0]
					p4_1P = [self.M_1P, 0.0, 0.0, 0.0]
				if decaystate == '2S':
					E_decaytest_2S = np.sqrt(self.M_2S**2 + P_decaytest[0]**2 + P_decaytest[1]**2 + P_decaytest[2]**2)
					p4_2S = np.append(E_decaytest_2S, P_decaytest)
					p4_1S = [self.M_1S, 0.0, 0.0, 0.0]
					p4_1P = [self.M_1P, 0.0, 0.0, 0.0]
				if decaystate == '1P':
					E_decaytest_1P = np.sqrt(self.M_1P**2 + P_decaytest[0]**2 + P_decaytest[1]**2 + P_decaytest[2]**2)
					p4_1P = np.append(E_decaytest_1P, P_decaytest)
					p4_1S = [self.M_1S, 0.0, 0.0, 0.0]
					p4_2S = [self.M_2S, 0.0, 0.0, 0.0]
				for i in range(N_Q):
					self.Qlist['4-momentum'].append( [self.M, 0.0, 0.0, 0.0] )
					self.Qlist['3-position'].append( np.random.rand(3)*Lmax )
				for i in range(N_Qbar):
					self.Qbarlist['4-momentum'].append( [self.M, 0.0, 0.0, 0.0] )
					self.Qbarlist['3-position'].append( np.random.rand(3)*Lmax )
				for i in range(N_U1S):
					self.U1Slist['4-momentum'].append( p4_1S )
					self.U1Slist['3-position'].append( np.random.rand(3)*Lmax )
					self.U1Slist['last_form_time'].append( self.t )
				for i in range(N_U2S):
					self.U2Slist['4-momentum'].append( p4_2S )
					self.U2Slist['3-position'].append( np.random.rand(3)*Lmax )
					self.U2Slist['last_form_time'].append( self.t )
				for i in range(N_U1P):
					self.U1Plist['4-momentum'].append( p4_1P )
					self.U1Plist['3-position'].append( np.random.rand(3)*Lmax )
					self.U1Plist['last_form_time'].append( self.t )
													                
			self.Qlist['4-momentum'] = np.array(self.Qlist['4-momentum'])
			self.Qlist['3-position'] = np.array(self.Qlist['3-position'])
			self.Qbarlist['4-momentum'] = np.array(self.Qbarlist['4-momentum'])
			self.Qbarlist['3-position'] = np.array(self.Qbarlist['3-position'])
			self.U1Slist['4-momentum'] = np.array(self.U1Slist['4-momentum'])
			self.U1Slist['3-position'] = np.array(self.U1Slist['3-position'])
			self.U1Slist['last_form_time'] = np.array(self.U1Slist['last_form_time'])
			self.U2Slist['4-momentum'] = np.array(self.U2Slist['4-momentum'])
			self.U2Slist['3-position'] = np.array(self.U2Slist['3-position'])
			self.U2Slist['last_form_time'] = np.array(self.U2Slist['last_form_time'])			
			self.U1Plist['4-momentum'] = np.array(self.U1Plist['4-momentum'])
			self.U1Plist['3-position'] = np.array(self.U1Plist['3-position'])
			self.U1Plist['last_form_time'] = np.array(self.U1Plist['last_form_time'])	
			
#### ---------------- event evolution function ------------------ ####
	def run(self, dt = 0.04, temp_run = -1.0):			# universal time to consider recombination
		len_Q = len(self.Qlist['4-momentum'])
		len_Qbar = len(self.Qbarlist['4-momentum'])
		len_U1S = len(self.U1Slist['4-momentum'])
		len_U2S = len(self.U2Slist['4-momentum'])
		len_U1P = len(self.U1Plist['4-momentum'])
		
		if temp_run != -1.0:
			self.T = temp_run
		
		### ------------- heavy quark diffusion --------------- ###
		if self.HQ_scat == True:
			for i in range(len_Q):
				self.Qlist['4-momentum'][i] = self.HQ_diff.update(HQ_Ep = self.Qlist['4-momentum'][i], Temp = self.T, time_step = dt)
			for i in range(len_Qbar):
				self.Qbarlist['4-momentum'][i] = self.HQ_diff.update(HQ_Ep = self.Qbarlist['4-momentum'][i], Temp = self.T, time_step = dt)
		
		### ----------- end of heavy quark diffusion ---------- ###



		### ----------- free stream these particles ------------###
		for i in range(len_Q):
			v3_Q = self.Qlist['4-momentum'][i][1:]/self.Qlist['4-momentum'][i][0]
			self.Qlist['3-position'][i] = (self.Qlist['3-position'][i] + dt * v3_Q)%self.Lmax
		for i in range(len_Qbar):
			v3_Qbar = self.Qbarlist['4-momentum'][i][1:]/self.Qbarlist['4-momentum'][i][0]
			self.Qbarlist['3-position'][i] = (self.Qbarlist['3-position'][i] + dt * v3_Qbar)%self.Lmax	
		for i in range(len_U1S):
			v3_U1S = self.U1Slist['4-momentum'][i][1:]/self.U1Slist['4-momentum'][i][0]
			self.U1Slist['3-position'][i] = (self.U1Slist['3-position'][i] + dt * v3_U1S)%self.Lmax
		for i in range(len_U2S):
			v3_U2S = self.U2Slist['4-momentum'][i][1:]/self.U2Slist['4-momentum'][i][0]
			self.U2Slist['3-position'][i] = (self.U2Slist['3-position'][i] + dt * v3_U2S)%self.Lmax
		for i in range(len_U1P):
			v3_U1P = self.U1Plist['4-momentum'][i][1:]/self.U1Plist['4-momentum'][i][0]
			self.U1Plist['3-position'][i] = (self.U1Plist['3-position'][i] + dt * v3_U1P)%self.Lmax
		### ----------- end of free stream particles -----------###




		### -------------------- decay ------------------------ ###
		delete_U1S = []
		delete_U2S = []
		delete_U1P = []
		add_pQ = []
		add_pQbar = []
		add_xQ = []
		add_xQbar = [] # the positions of Q and Qbar can be different
		
		## 1S decay
		for i in range(len_U1S):
			p4_in_box = self.U1Slist['4-momentum'][i]
			v3_in_box = p4_in_box[1:]/p4_in_box[0]
			v_in_box = np.sqrt(np.sum(v3_in_box**2))
			if self.process == 'all':
				rate_decay1S_gluon = self.event.get_R1S_decay_gluon( v_in_box, self.T )	# GeV
				rate_decay1S_ineq = self.event.get_R1S_decay_ineq( v_in_box, self.T )	# GeV
				rate_decay1S_ineg = self.event.get_R1S_decay_ineg( v_in_box, self.T )
			if self.process == 'gluon':
				rate_decay1S_gluon = self.event.get_R1S_decay_gluon( v_in_box, self.T )	# GeV
				rate_decay1S_ineq = 0.0
				rate_decay1S_ineg = 0.0
			if self.process == 'ineq':
				rate_decay1S_gluon = 0.0
				rate_decay1S_ineq = self.event.get_R1S_decay_ineq( v_in_box, self.T )	# GeV
				rate_decay1S_ineg = 0.0
			if self.process == 'ineg':
				rate_decay1S_gluon = 0.0
				rate_decay1S_ineq = 0.0
				rate_decay1S_ineg = self.event.get_R1S_decay_ineg( v_in_box, self.T )							
			prob_decay1S_gluon = rate_decay1S_gluon * dt/C1
			prob_decay1S_ineq = rate_decay1S_ineq * dt/C1
			prob_decay1S_ineg = rate_decay1S_ineg * dt/C1
			#print prob_decay1S_gluon, prob_decay1S_ineq, prob_decay1S_ineg
			rej_mc = np.random.rand(1)
			
			if rej_mc <= prob_decay1S_gluon + prob_decay1S_ineq + prob_decay1S_ineg:
				delete_U1S.append(i)
				# outgoing momenta of Q Qbar in the rest frame of quarkonia, depends on decay channel
				if rej_mc <= prob_decay1S_gluon:
					recoil_pQpQbar = self.event.sample_S1S_decay_gluon( v_in_box, self.T )
				elif rej_mc <= prob_decay1S_gluon + prob_decay1S_ineq:
					recoil_pQpQbar = self.event.sample_S1S_decay_ineq( v_in_box, self.T )
				else:
					recoil_pQpQbar = self.event.sample_S1S_decay_ineg( v_in_box, self.T )
				
				recoil_pQ = np.array(recoil_pQpQbar[0:4])
				recoil_pQbar = np.array(recoil_pQpQbar[4:8])
				# Q, Qbar momenta need to be rotated from the v = z axis to what it is in the box frame
				# first get the rotation matrix angles
				theta_rot, phi_rot = LorRot.angle( v3_in_box )
				# then do the rotation in the spatial components
				rotmomentum_Q = LorRot.rotation4(recoil_pQ, theta_rot, phi_rot)
				rotmomentum_Qbar = LorRot.rotation4(recoil_pQbar, theta_rot, phi_rot)
				
				# we now transform them back to the box frame
				momentum_Q = LorRot.lorentz(rotmomentum_Q, -v3_in_box)			# final momentum of Q
				momentum_Qbar = LorRot.lorentz(rotmomentum_Qbar, -v3_in_box)	# final momentum of Qbar
				
				# positions of Q and Qbar
				#x_rel = np.array(self.event.sample_S1S_decay_position())/2.
				#position_Q = (self.U1Slist['3-position'][i] + x_rel)%self.Lmax
				#position_Qbar = (self.U1Slist['3-position'][i] - x_rel)%self.Lmax
				position_Q = self.U1Slist['3-position'][i]
				position_Qbar = self.U1Slist['3-position'][i]
		
				# add x and p for the QQbar to the temporary list
				add_pQ.append(momentum_Q)
				add_pQbar.append(momentum_Qbar)
				add_xQ.append(position_Q)
				add_xQbar.append(position_Qbar)
				
		## 2S decay
		for i in range(len_U2S):
			p4_in_box = self.U2Slist['4-momentum'][i]
			v3_in_box = p4_in_box[1:]/p4_in_box[0]
			v_in_box = np.sqrt(np.sum(v3_in_box**2))
			if self.process == 'all':
				rate_decay2S_gluon = self.event.get_R2S_decay_gluon( v_in_box, self.T )	# GeV
				rate_decay2S_ineq = self.event.get_R2S_decay_ineq( v_in_box, self.T )	# GeV
				rate_decay2S_ineg = self.event.get_R2S_decay_ineg( v_in_box, self.T )
			if self.process == 'gluon':
				rate_decay2S_gluon = self.event.get_R2S_decay_gluon( v_in_box, self.T )	# GeV
				rate_decay2S_ineq = 0.0
				rate_decay2S_ineg = 0.0
			if self.process == 'ineq':
				rate_decay2S_gluon = 0.0
				rate_decay2S_ineq = self.event.get_R2S_decay_ineq( v_in_box, self.T )	# GeV
				rate_decay2S_ineg = 0.0
			if self.process == 'ineg':
				rate_decay2S_gluon = 0.0
				rate_decay2S_ineq = 0.0
				rate_decay2S_ineg = self.event.get_R2S_decay_ineg( v_in_box, self.T )							
			prob_decay2S_gluon = rate_decay2S_gluon * dt/C1
			prob_decay2S_ineq = rate_decay2S_ineq * dt/C1
			prob_decay2S_ineg = rate_decay2S_ineg * dt/C1
			rej_mc = np.random.rand(1)
			
			if rej_mc <= prob_decay2S_gluon + prob_decay2S_ineq + prob_decay2S_ineg:
				delete_U2S.append(i)
				# outgoing momenta of Q Qbar in the rest frame of quarkonia, depends on decay channel
				if rej_mc <= prob_decay2S_gluon:
					recoil_pQpQbar = self.event.sample_S2S_decay_gluon( v_in_box, self.T )
				elif rej_mc <= prob_decay2S_gluon + prob_decay2S_ineq:
					recoil_pQpQbar = self.event.sample_S2S_decay_ineq( v_in_box, self.T )
				else:
					recoil_pQpQbar = self.event.sample_S2S_decay_ineg( v_in_box, self.T )
				
				recoil_pQ = np.array(recoil_pQpQbar[0:4])
				recoil_pQbar = np.array(recoil_pQpQbar[4:8])
				# Q, Qbar momenta need to be rotated from the v = z axis to what it is in the box frame
				# first get the rotation matrix angles
				theta_rot, phi_rot = LorRot.angle( v3_in_box )
				# then do the rotation in the spatial components
				rotmomentum_Q = LorRot.rotation4(recoil_pQ, theta_rot, phi_rot)
				rotmomentum_Qbar = LorRot.rotation4(recoil_pQbar, theta_rot, phi_rot)
				
				# we now transform them back to the box frame
				momentum_Q = LorRot.lorentz(rotmomentum_Q, -v3_in_box)			# final momentum of Q
				momentum_Qbar = LorRot.lorentz(rotmomentum_Qbar, -v3_in_box)	# final momentum of Qbar
				
				# positions of Q and Qbar
				#x_rel = np.array(self.event.sample_S2S_decay_position())/2.
				#position_Q = (self.U2Slist['3-position'][i] + x_rel)%self.Lmax
				#position_Qbar = (self.U2Slist['3-position'][i] - x_rel)%self.Lmax
				position_Q = self.U2Slist['3-position'][i]
				position_Qbar = self.U2Slist['3-position'][i]
		
				# add x and p for the QQbar to the temporary list
				add_pQ.append(momentum_Q)
				add_pQbar.append(momentum_Qbar)
				add_xQ.append(position_Q)
				add_xQbar.append(position_Qbar)


		## 1P decay
		for i in range(len_U1P):
			p4_in_box = self.U1Plist['4-momentum'][i]
			v3_in_box = p4_in_box[1:]/p4_in_box[0]
			v_in_box = np.sqrt(np.sum(v3_in_box**2))
			if self.process == 'all':
				rate_decay1P_gluon = self.event.get_R1P_decay_gluon( v_in_box, self.T )	# GeV
				rate_decay1P_ineq = self.event.get_R1P_decay_ineq( v_in_box, self.T )	# GeV
				rate_decay1P_ineg = self.event.get_R1P_decay_ineg( v_in_box, self.T )
			if self.process == 'gluon':
				rate_decay1P_gluon = self.event.get_R1P_decay_gluon( v_in_box, self.T )	# GeV
				rate_decay1P_ineq = 0.0
				rate_decay1P_ineg = 0.0
			if self.process == 'ineq':
				rate_decay1P_gluon = 0.0
				rate_decay1P_ineq = self.event.get_R1P_decay_ineq( v_in_box, self.T )	# GeV
				rate_decay1P_ineg = 0.0
			if self.process == 'ineg':
				rate_decay1P_gluon = 0.0
				rate_decay1P_ineq = 0.0
				rate_decay1P_ineg = self.event.get_R1P_decay_ineg( v_in_box, self.T )							
			prob_decay1P_gluon = rate_decay1P_gluon * dt/C1
			prob_decay1P_ineq = rate_decay1P_ineq * dt/C1
			prob_decay1P_ineg = rate_decay1P_ineg * dt/C1
			#print prob_decay1P_gluon, prob_decay1P_ineq, prob_decay1P_ineg
			rej_mc = np.random.rand(1)
			
			if rej_mc <= prob_decay1P_gluon + prob_decay1P_ineq + prob_decay1P_ineg:
				delete_U1P.append(i)
				# outgoing momenta of Q Qbar in the rest frame of quarkonia, depends on decay channel
				if rej_mc <= prob_decay1P_gluon:
					recoil_pQpQbar = self.event.sample_S1P_decay_gluon( v_in_box, self.T )
				elif rej_mc <= prob_decay1P_gluon + prob_decay1P_ineq:
					recoil_pQpQbar = self.event.sample_S1P_decay_ineq( v_in_box, self.T )
				else:
					recoil_pQpQbar = self.event.sample_S1P_decay_ineg( v_in_box, self.T )
				
				recoil_pQ = np.array(recoil_pQpQbar[0:4])
				recoil_pQbar = np.array(recoil_pQpQbar[4:8])
				# Q, Qbar momenta need to be rotated from the v = z axis to what it is in the box frame
				# first get the rotation matrix angles
				theta_rot, phi_rot = LorRot.angle( v3_in_box )
				# then do the rotation in the spatial components
				rotmomentum_Q = LorRot.rotation4(recoil_pQ, theta_rot, phi_rot)
				rotmomentum_Qbar = LorRot.rotation4(recoil_pQbar, theta_rot, phi_rot)
				
				# we now transform them back to the box frame
				momentum_Q = LorRot.lorentz(rotmomentum_Q, -v3_in_box)			# final momentum of Q
				momentum_Qbar = LorRot.lorentz(rotmomentum_Qbar, -v3_in_box)	# final momentum of Qbar
				
				# positions of Q and Qbar
				#x_rel = np.array(self.event.sample_S1P_decay_position())/2.
				#position_Q = (self.U1Plist['3-position'][i] + x_rel)%self.Lmax
				#position_Qbar = (self.U1Plist['3-position'][i] - x_rel)%self.Lmax
				position_Q = self.U1Plist['3-position'][i]
				position_Qbar = self.U1Plist['3-position'][i]
		
				# add x and p for the QQbar to the temporary list
				add_pQ.append(momentum_Q)
				add_pQbar.append(momentum_Qbar)
				add_xQ.append(position_Q)
				add_xQbar.append(position_Qbar)
				
		### ------------------ end of decay ------------------- ###
		
		
		
		### ------------------ recombination ------------------ ###
		if self.recombine == True and len_Q != 0 and len_Qbar != 0:
			delete_Q = []
			delete_Qbar = []
			
			# make the periodic box 26 times bigger!
			Qbar_x_list = np.concatenate((self.Qbarlist['3-position'], 
			self.Qbarlist['3-position']+[0.0, 0.0, self.Lmax], self.Qbarlist['3-position']+[0.0, 0.0, -self.Lmax],
			self.Qbarlist['3-position']+[0.0, self.Lmax, 0.0], self.Qbarlist['3-position']+[0.0, -self.Lmax, 0.0],
			self.Qbarlist['3-position']+[self.Lmax, 0.0, 0.0], self.Qbarlist['3-position']+[-self.Lmax, 0.0, 0.0],
			self.Qbarlist['3-position']+[0.0, self.Lmax, self.Lmax], self.Qbarlist['3-position']+[0.0, self.Lmax, -self.Lmax], self.Qbarlist['3-position']+[0.0, -self.Lmax, self.Lmax], self.Qbarlist['3-position']+[0.0, -self.Lmax, -self.Lmax],
			self.Qbarlist['3-position']+[self.Lmax, 0.0, self.Lmax], self.Qbarlist['3-position']+[self.Lmax, 0.0, -self.Lmax], self.Qbarlist['3-position']+[-self.Lmax, 0.0, self.Lmax], self.Qbarlist['3-position']+[-self.Lmax, 0.0, -self.Lmax],
			self.Qbarlist['3-position']+[self.Lmax, self.Lmax, 0.0], self.Qbarlist['3-position']+[self.Lmax, -self.Lmax, 0.0], self.Qbarlist['3-position']+[-self.Lmax, self.Lmax, 0.0], self.Qbarlist['3-position']+[-self.Lmax, -self.Lmax, 0.0],
			self.Qbarlist['3-position']+[self.Lmax, self.Lmax, self.Lmax], self.Qbarlist['3-position']+[self.Lmax, self.Lmax, -self.Lmax], self.Qbarlist['3-position']+[self.Lmax, -self.Lmax, self.Lmax], self.Qbarlist['3-position']+[-self.Lmax, self.Lmax, self.Lmax],
			self.Qbarlist['3-position']+[-self.Lmax, -self.Lmax, -self.Lmax], self.Qbarlist['3-position']+[-self.Lmax, -self.Lmax, self.Lmax], self.Qbarlist['3-position']+[-self.Lmax, self.Lmax, -self.Lmax], self.Qbarlist['3-position']+[self.Lmax, -self.Lmax, -self.Lmax]),
			axis = 0 )
			
			pair_search = cKDTree(Qbar_x_list)
			# for each Q, obtain the Qbar indexes within R_search
			pair_list = pair_search.query_ball_point(self.Qlist['3-position'], r = R_search)
			
			for i in range(len_Q):		# loop over Q
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
					xQbar = Qbar_x_list[pair_list[i][j]]
					x_rel = xQ - xQbar
					i_Qbar_mod = pair_list[i][j]%len_Qbar	# use for momentum and delete_index
					#rdotp = np.sum( x_rel* (self.Qlist['4-momentum'][i][1:] - self.Qbarlist['4-momentum'][i_Qbar_mod][1:]) )
					
					if  i_Qbar_mod not in delete_Qbar:
						# momenta in hydro cell frame
						pQ = self.Qlist['4-momentum'][i]
						pQbar = self.Qbarlist['4-momentum'][i_Qbar_mod]
						# CM momentum and velocity
						v_CM, v_CM_abs, p_rel_abs = LorRot.vCM_prel(pQ, pQbar, 2.0*self.M)
						r_rel = np.sqrt(np.sum(x_rel**2))
						
						# relative position in QQbar rest frame
						#x4_rel_lab = np.array([0.0, x_rel[0], x_rel[1], x_rel[2]])
						#x4_rel_rest = LorRot.lorentz(x4_rel_lab, v_CM)
						#r_rel = np.sqrt(np.sum(x4_rel_rest[1:]**2))
						
						rate_reco1S_gluon.append(self.event.get_R1S_reco_gluon(v_CM_abs, self.T, p_rel_abs, r_rel))
						rate_reco1S_ineq.append(self.event.get_R1S_reco_ineq(v_CM_abs, self.T, p_rel_abs, r_rel))
						rate_reco1S_ineg.append(self.event.get_R1S_reco_ineg(v_CM_abs, self.T, p_rel_abs, r_rel))
						rate_reco2S_gluon.append(self.event.get_R2S_reco_gluon(v_CM_abs, self.T, p_rel_abs, r_rel))
						rate_reco2S_ineq.append(self.event.get_R2S_reco_ineq(v_CM_abs, self.T, p_rel_abs, r_rel))
						rate_reco2S_ineg.append(self.event.get_R2S_reco_ineg(v_CM_abs, self.T, p_rel_abs, r_rel))					
						rate_reco1P_gluon.append(self.event.get_R1P_reco_gluon(v_CM_abs, self.T, p_rel_abs, r_rel))
						rate_reco1P_ineq.append(self.event.get_R1P_reco_ineq(v_CM_abs, self.T, p_rel_abs, r_rel))
						rate_reco1P_ineg.append(self.event.get_R1P_reco_ineg(v_CM_abs, self.T, p_rel_abs, r_rel))
					else:	# the Qbar has been taken by other Q's
						rate_reco1S_gluon.append(0.)
						rate_reco1S_ineq.append(0.)
						rate_reco1S_ineg.append(0.)
						rate_reco2S_gluon.append(0.)
						rate_reco2S_ineq.append(0.)
						rate_reco2S_ineg.append(0.)
						rate_reco1P_gluon.append(0.)
						rate_reco1P_ineq.append(0.)
						rate_reco1P_ineg.append(0.)
											
				# get the recombine probability
				# spin=1 gives 3/4 factor for Upsilon(1S, 2S) and chi_b(1P) v.s. eta_b and h_b
				if self.process == 'all':
					prob_reco1S_gluon = 2./3.*np.array(rate_reco1S_gluon)*dt/C1
					prob_reco1S_ineq = 2./3.*np.array(rate_reco1S_ineq)*dt/C1
					prob_reco1S_ineg = 2./3.*np.array(rate_reco1S_ineg)*dt/C1
					prob_reco2S_gluon = 2./3.*np.array(rate_reco2S_gluon)*dt/C1
					prob_reco2S_ineq = 2./3.*np.array(rate_reco2S_ineq)*dt/C1
					prob_reco2S_ineg = 2./3.*np.array(rate_reco2S_ineg)*dt/C1
					prob_reco1P_gluon = 2./3.*np.array(rate_reco1P_gluon)*dt/C1
					prob_reco1P_ineq = 2./3.*np.array(rate_reco1P_ineq)*dt/C1
					prob_reco1P_ineg = 2./3.*np.array(rate_reco1P_ineg)*dt/C1
				if self.process == 'gluon':
					prob_reco1S_gluon = 2./3.*np.array(rate_reco1S_gluon)*dt/C1
					prob_reco1S_ineq = 0.0*np.array(rate_reco1S_ineq)
					prob_reco1S_ineg = 0.0*np.array(rate_reco1S_ineg)
					prob_reco2S_gluon = 2./3.*np.array(rate_reco2S_gluon)*dt/C1
					prob_reco2S_ineq = 0.0*np.array(rate_reco2S_ineq)
					prob_reco2S_ineg = 0.0*np.array(rate_reco2S_ineg)
					prob_reco1P_gluon = 2./3.*np.array(rate_reco1P_gluon)*dt/C1
					prob_reco1P_ineq = 0.0*np.array(rate_reco1P_ineq)
					prob_reco1P_ineg = 0.0*np.array(rate_reco1P_ineg)
				if self.process == 'ineq':
					prob_reco1S_gluon = 0.0*np.array(rate_reco1S_gluon)
					prob_reco1S_ineq = 2./3.*np.array(rate_reco1S_ineq)*dt/C1
					prob_reco1S_ineg = 0.0*np.array(rate_reco1S_ineg)
					prob_reco2S_gluon = 0.0*np.array(rate_reco2S_gluon)
					prob_reco2S_ineq = 2./3.*np.array(rate_reco2S_ineq)*dt/C1
					prob_reco2S_ineg = 0.0*np.array(rate_reco2S_ineg)
					prob_reco1P_gluon = 0.0*np.array(rate_reco1P_gluon)
					prob_reco1P_ineq = 2./3.*np.array(rate_reco1P_ineq)*dt/C1
					prob_reco1P_ineg = 0.0*np.array(rate_reco1P_ineg)
				if self.process == 'ineg':
					prob_reco1S_gluon = 0.0*np.array(rate_reco1S_gluon)
					prob_reco1S_ineq = 0.0*np.array(rate_reco1S_ineq)
					prob_reco1S_ineg = 2./3.*np.array(rate_reco1S_ineg)*dt/C1
					prob_reco2S_gluon = 0.0*np.array(rate_reco2S_gluon)
					prob_reco2S_ineq = 0.0*np.array(rate_reco2S_ineq)
					prob_reco2S_ineg = 2./3.*np.array(rate_reco2S_ineg)*dt/C1
					prob_reco1P_gluon = 0.0*np.array(rate_reco1P_gluon)
					prob_reco1P_ineq = 0.0*np.array(rate_reco1P_ineq)
					prob_reco1P_ineg = 2./3.*np.array(rate_reco1P_ineg)*dt/C1
									
				prob_reco1S_all = prob_reco1S_gluon + prob_reco1S_ineq + prob_reco1S_ineg
				prob_reco2S_all = prob_reco2S_gluon + prob_reco2S_ineq + prob_reco2S_ineg
				prob_reco1P_all = prob_reco1P_gluon + prob_reco1P_ineq + prob_reco1P_ineg
				
				if self.quarkonium_species == '1S':
					total_prob_reco1S = np.sum(prob_reco1S_all)
					total_prob_reco2S = 0.0*np.sum(prob_reco2S_all)
					total_prob_reco1P = 0.0*np.sum(prob_reco1P_all)
				elif self.quarkonium_species == '2S':
					total_prob_reco1S = 0.0*np.sum(prob_reco1S_all)
					total_prob_reco2S = np.sum(prob_reco2S_all)
					total_prob_reco1P = 0.0*np.sum(prob_reco1P_all)
				elif self.quarkonium_species == '1P':
					total_prob_reco1S = 0.0*np.sum(prob_reco1S_all)
					total_prob_reco2S = 0.0*np.sum(prob_reco2S_all)
					total_prob_reco1P = np.sum(prob_reco1P_all)
				elif self.quarkonium_species == 'all':
					total_prob_reco1S = np.sum(prob_reco1S_all)
					total_prob_reco2S = np.sum(prob_reco2S_all)
					total_prob_reco1P = np.sum(prob_reco1P_all)
				elif self.quarkonium_species == '1S2S':
					total_prob_reco1S = np.sum(prob_reco1S_all)
					total_prob_reco2S = np.sum(prob_reco2S_all)
					total_prob_reco1P = 0.0*np.sum(prob_reco1P_all)
									
				rej_mc = np.random.rand(1)
				
				if rej_mc <= total_prob_reco1S:
					delete_Q.append(i)		# remove this Q later
					# find the Qbar we need to remove
					a = 0.0
					for j in range(len_recoQbar):
						if a <= rej_mc <= a + prob_reco1S_all[j]:
							k = j
							if rej_mc <= a + prob_reco1S_gluon[j]:
								channel_reco = 'gluon'
							elif rej_mc <= a + prob_reco1S_gluon[j] + prob_reco1S_ineq[j]:
								channel_reco = 'ineq'
							else:
								channel_reco = 'ineg'
							break
						a += prob_reco1S_all[j]
					delete_Qbar.append(pair_list[i][k]%len_Qbar)

					# re-construct the reco event and sample initial and final states
					# positions and local temperature
					xQ = self.Qlist['3-position'][i]
					xQbar = Qbar_x_list[pair_list[i][k]]
					x_CM = 0.5*( xQ + xQbar )
					
					# momenta
					pQ = self.Qlist['4-momentum'][i]
					pQbar = self.Qbarlist['4-momentum'][pair_list[i][k]%len_Qbar]
					
					# CM momentum and velocity
					v_CM, v_CM_abs, p_rel_abs = LorRot.vCM_prel(pQ, pQbar, 2.0*self.M)

					# calculate the final quarkonium momenta in the CM frame of QQbar, depends on channel_reco
					if channel_reco == 'gluon':
						tempmomentum_U1S = np.array(self.event.sample_S1S_reco_gluon(v_CM_abs, self.T, p_rel_abs))
					elif channel_reco == 'ineq':
						tempmomentum_U1S = np.array(self.event.sample_S1S_reco_ineq(v_CM_abs, self.T, p_rel_abs))
					else:	# 'ineg'
						tempmomentum_U1S = np.array(self.event.sample_S1S_reco_ineg(v_CM_abs, self.T, p_rel_abs))
					
					# need to rotate the vector, v is not the z axis in box frame
					theta_rot, phi_rot = LorRot.angle(v_CM)
					rotmomentum_U1S = LorRot.rotation4(tempmomentum_U1S, theta_rot, phi_rot)
					# lorentz back to the box frame
					momentum_U1S = LorRot.lorentz( rotmomentum_U1S, -v_CM )
					# positions of the quarkonium
					position_U1S = x_CM%self.Lmax
					
					# update the quarkonium list
					if len(self.U1Slist['4-momentum']) == 0:
						self.U1Slist['4-momentum'] = np.array([momentum_U1S])
						self.U1Slist['3-position'] = np.array([position_U1S])
						self.U1Slist['last_form_time'] = np.array(self.t)
					else:
						self.U1Slist['4-momentum'] = np.append(self.U1Slist['4-momentum'], [momentum_U1S], axis=0)
						self.U1Slist['3-position'] = np.append(self.U1Slist['3-position'], [position_U1S], axis=0)
						self.U1Slist['last_form_time'] = np.append(self.U1Slist['last_form_time'], self.t)
				
				elif rej_mc <= (total_prob_reco1S + total_prob_reco2S):
					delete_Q.append(i)		# remove this Q later
					# find the Qbar we need to remove
					a = total_prob_reco1S
					for j in range(len_recoQbar):
						if a <= rej_mc <= a + prob_reco2S_all[j]:
							k = j
							if rej_mc <= a + prob_reco2S_gluon[j]:
								channel_reco = 'gluon'
							elif rej_mc <= a + prob_reco2S_gluon[j] + prob_reco2S_ineq[j]:
								channel_reco = 'ineq'
							else:
								channel_reco = 'ineg'
							break
						a += prob_reco2S_all[j]
					delete_Qbar.append(pair_list[i][k]%len_Qbar)
					# re-construct the reco event and sample initial and final states
					# positions and local temperature
					xQ = self.Qlist['3-position'][i]
					xQbar = Qbar_x_list[pair_list[i][k]]
					x_CM = 0.5*( xQ + xQbar )
					
					# momenta
					pQ = self.Qlist['4-momentum'][i]
					pQbar = self.Qbarlist['4-momentum'][pair_list[i][k]%len_Qbar]
					
					# CM momentum and velocity
					v_CM, v_CM_abs, p_rel_abs = LorRot.vCM_prel(pQ, pQbar, 2.0*self.M)

					# calculate the final quarkonium momenta in the CM frame of QQbar, depends on channel_reco
					if channel_reco == 'gluon':
						tempmomentum_U2S = np.array(self.event.sample_S2S_reco_gluon(v_CM_abs, self.T, p_rel_abs))
					elif channel_reco == 'ineq':
						tempmomentum_U2S = np.array(self.event.sample_S2S_reco_ineq(v_CM_abs, self.T, p_rel_abs))
					else:	# 'ineg'
						tempmomentum_U2S = np.array(self.event.sample_S2S_reco_ineg(v_CM_abs, self.T, p_rel_abs))
					
					# need to rotate the vector, v is not the z axis in box frame
					theta_rot, phi_rot = LorRot.angle(v_CM)
					rotmomentum_U2S = LorRot.rotation4(tempmomentum_U2S, theta_rot, phi_rot)
					# lorentz back to the box frame
					momentum_U2S = LorRot.lorentz( rotmomentum_U2S, -v_CM )
					# positions of the quarkonium
					position_U2S = x_CM%self.Lmax
					
					# update the quarkonium list
					if len(self.U2Slist['4-momentum']) == 0:
						self.U2Slist['4-momentum'] = np.array([momentum_U2S])
						self.U2Slist['3-position'] = np.array([position_U2S])
						self.U2Slist['last_form_time'] = np.array(self.t)
					else:
						self.U2Slist['4-momentum'] = np.append(self.U2Slist['4-momentum'], [momentum_U2S], axis=0)
						self.U2Slist['3-position'] = np.append(self.U2Slist['3-position'], [position_U2S], axis=0)
						self.U2Slist['last_form_time'] = np.append(self.U2Slist['last_form_time'], self.t)

				elif rej_mc <= (total_prob_reco1S + total_prob_reco2S + total_prob_reco1P):
					delete_Q.append(i)		# remove this Q later
					# find the Qbar we need to remove
					a = total_prob_reco1S + total_prob_reco2S
					for j in range(len_recoQbar):
						if a <= rej_mc <= a + prob_reco1P_all[j]:
							k = j
							if rej_mc <= a + prob_reco1P_gluon[j]:
								channel_reco = 'gluon'
							elif rej_mc <= a + prob_reco1P_gluon[j] + prob_reco1P_ineq[j]:
								channel_reco = 'ineq'
							else:
								channel_reco = 'ineg'
							break
						a += prob_reco1P_all[j]
					delete_Qbar.append(pair_list[i][k]%len_Qbar)
					# re-construct the reco event and sample initial and final states
					# positions and local temperature
					xQ = self.Qlist['3-position'][i]
					xQbar = Qbar_x_list[pair_list[i][k]]
					x_CM = 0.5*( xQ + xQbar )
					
					# momenta
					pQ = self.Qlist['4-momentum'][i]
					pQbar = self.Qbarlist['4-momentum'][pair_list[i][k]%len_Qbar]
					
					# CM momentum and velocity
					v_CM, v_CM_abs, p_rel_abs = LorRot.vCM_prel(pQ, pQbar, 2.0*self.M)

					# calculate the final quarkonium momenta in the CM frame of QQbar, depends on channel_reco
					if channel_reco == 'gluon':
						tempmomentum_U1P = np.array(self.event.sample_S1P_reco_gluon(v_CM_abs, self.T, p_rel_abs))
					elif channel_reco == 'ineq':
						tempmomentum_U1P = np.array(self.event.sample_S1P_reco_ineq(v_CM_abs, self.T, p_rel_abs))
					else:	# 'ineg'
						tempmomentum_U1P = np.array(self.event.sample_S1P_reco_ineg(v_CM_abs, self.T, p_rel_abs))
					
					# need to rotate the vector, v is not the z axis in box frame
					theta_rot, phi_rot = LorRot.angle(v_CM)
					rotmomentum_U1P = LorRot.rotation4(tempmomentum_U1P, theta_rot, phi_rot)
					# lorentz back to the box frame
					momentum_U1P = LorRot.lorentz( rotmomentum_U1P, -v_CM )
					# positions of the quarkonium
					position_U1P = x_CM%self.Lmax
					
					# update the quarkonium list
					if len(self.U1Plist['4-momentum']) == 0:
						self.U1Plist['4-momentum'] = np.array([momentum_U1P])
						self.U1Plist['3-position'] = np.array([position_U1P])
						self.U1Plist['last_form_time'] = np.array(self.t)
					else:
						self.U1Plist['4-momentum'] = np.append(self.U1Plist['4-momentum'], [momentum_U1P], axis=0)
						self.U1Plist['3-position'] = np.append(self.U1Plist['3-position'], [position_U1P], axis=0)
						self.U1Plist['last_form_time'] = np.append(self.U1Plist['last_form_time'], self.t)
			
			## now update Q and Qbar lists
			self.Qlist['4-momentum'] = np.delete(self.Qlist['4-momentum'], delete_Q, axis=0)
			self.Qlist['3-position'] = np.delete(self.Qlist['3-position'], delete_Q, axis=0)
			self.Qbarlist['4-momentum'] = np.delete(self.Qbarlist['4-momentum'], delete_Qbar, axis=0)
			self.Qbarlist['3-position'] = np.delete(self.Qbarlist['3-position'], delete_Qbar, axis=0)
		
		### -------------- end of recombination --------------- ###
		
		
		### -------------- update lists due to decay ---------- ###
		add_pQ = np.array(add_pQ)
		add_pQbar = np.array(add_pQbar)
		add_xQ = np.array(add_xQ)
		add_xQbar = np.array(add_xQbar)
		if len(add_pQ):	
			# if there is at least quarkonium decays, we need to update all the three lists
			self.U1Slist['3-position'] = np.delete(self.U1Slist['3-position'], delete_U1S, axis=0) # delete along the axis = 0
			self.U1Slist['4-momentum'] = np.delete(self.U1Slist['4-momentum'], delete_U1S, axis=0)
			self.U1Slist['last_form_time'] = np.delete(self.U1Slist['last_form_time'], delete_U1S)
			self.U2Slist['3-position'] = np.delete(self.U2Slist['3-position'], delete_U2S, axis=0) # delete along the axis = 0
			self.U2Slist['4-momentum'] = np.delete(self.U2Slist['4-momentum'], delete_U2S, axis=0)
			self.U2Slist['last_form_time'] = np.delete(self.U2Slist['last_form_time'], delete_U2S)
			self.U1Plist['3-position'] = np.delete(self.U1Plist['3-position'], delete_U1P, axis=0) # delete along the axis = 0
			self.U1Plist['4-momentum'] = np.delete(self.U1Plist['4-momentum'], delete_U1P, axis=0)
			self.U1Plist['last_form_time'] = np.delete(self.U1Plist['last_form_time'], delete_U1P)
			
			if len(self.Qlist['4-momentum']) == 0:
				self.Qlist['3-position'] = np.array(add_xQ)
				self.Qlist['4-momentum'] = np.array(add_pQ)
			else:
				self.Qlist['3-position'] = np.append(self.Qlist['3-position'], add_xQ, axis=0)
				self.Qlist['4-momentum'] = np.append(self.Qlist['4-momentum'], add_pQ, axis=0)
			if len(self.Qbarlist['4-momentum']) == 0:
				self.Qbarlist['3-position'] = np.array(add_xQ)
				self.Qbarlist['4-momentum'] = np.array(add_pQbar)
			else:
				self.Qbarlist['3-position'] = np.append(self.Qbarlist['3-position'], add_xQbar, axis=0)
				self.Qbarlist['4-momentum'] = np.append(self.Qbarlist['4-momentum'], add_pQbar, axis=0)
		
		### ---------- end of update lists due to decay ------- ###

		self.t += dt
#### ----------------- end of evolution function ----------------- ####



#### ----------------- define clear function ------------------ ####
	def dict_clear(self):
		self.Qlist.clear()
		self.Qbarlist.clear()
		self.U1Slist.clear()
		self.U2Slist.clear()
		self.U1Plist.clear()
#### ----------------- end of clear function ------------------ ####

