#!/usr/bin/env python
import numpy as np
import scipy.integrate as si
from Static_quarkonium_evolution_all import QQbar_evol
import h5py


#### ------------ multiple runs averaged and compare ---------------- ####
N_ave = 1			# number of events
T = 0.3		
N_step = 1250
N_momentum = 1250	# after this number of steps, store quarkonium momentum
dt = 0.01
tmax = N_step*dt
t = np.linspace(0.0, tmax, N_step+1)
Nb0 = 50			# initial number of Q or Qbar
N1s0 = 0			# initial number of U1s
N2s0 = 0			# initial number of U2s
N1p0 = 0			# initial number of U1p
N1s_t = []			# to store number of U1s in each time step
N2s_t = []			# to store number of U2s in each time step
N1p_t = []			# to store number of U1p in each time step
Nb_t = []			# to store number of Q or Qbar in each time step
#momentum1s_t = []
#momentum2s_t = []
#momentum1p_t = []
P_sample = 5.0		# GeV, initial uniform sampling
Process_chosen = 'all'
Species = '1S'

# define the event generator
event_gen = QQbar_evol('static', temp_init = T, HQ_scat = True, process = Process_chosen, quarkonium_species = Species)

for i in range(N_ave):
	# initialize N_ave number of events
	#event_gen.initialize(N_Q = Nb0, N_Qbar = Nb0, N_U1S = N1s0, N_U2S = N2s0, N_U1P = N1p0, Lmax = 10.0, thermal_dist = True )
	event_gen.initialize(N_Q = Nb0, N_Qbar = Nb0, N_U1S = N1s0, N_U2S = N2s0, N_U1P = N1p0, thermal_dist = False, uniform_dist = True, Pmax = P_sample)
	N1s_t.append([])
	N2s_t.append([])
	N1p_t.append([])
	for j in range(N_step+1):
		N1s_t[i].append(len(event_gen.U1Slist['4-momentum']))	# store N_1S(t) for each event
		N2s_t[i].append(len(event_gen.U2Slist['4-momentum']))	# store N_2S(t) for each event
		N1p_t[i].append(len(event_gen.U1Plist['4-momentum']))	# store N_1P(t) for each event

		#if j > N_momentum and j%10 == 0:
			#for k in range(N1s_t[i][j]):
				#momentum1s_t.append(event_gen.U1Slist['4-momentum'][k])
			#for k in range(N2s_t[i][j]):
				#momentum2s_t.append(event_gen.U2Slist['4-momentum'][k])
			#for k in range(N1p_t[i][j]):
				#momentum1p_t.append(event_gen.U1Plist['4-momentum'][k])
		
		event_gen.run(dt = dt)
	event_gen.dict_clear()	## clear data from last simulation


#momentum1s_t = np.array(momentum1s_t)
#momentum2s_t = np.array(momentum2s_t)
#momentum1p_t = np.array(momentum1p_t)
N1s_t = np.array(N1s_t)
N2s_t = np.array(N2s_t)
N1p_t = np.array(N1p_t)
N1s_t_ave = np.sum(N1s_t, axis = 0)/(N_ave + 0.0)	# averaged number of 1S state (time-sequenced)
N2s_t_ave = np.sum(N2s_t, axis = 0)/(N_ave + 0.0)	# averaged number of 2S state (time-sequenced)
N1p_t_ave = np.sum(N1p_t, axis = 0)/(N_ave + 0.0)	# averaged number of 1P state (time-sequenced)

Nb_t_ave = Nb0 + N1s0 + N2s0 + N1p0 - N1s_t_ave - N2s_t_ave - N1p_t_ave	# time-sequenced bottom quark number

R1s_t = N1s_t_ave/(Nb0+N1s0+N2s0+N1p0)	# ratio of number of Q in the U1S
R2s_t = N2s_t_ave/(Nb0+N1s0+N2s0+N1p0)	# ratio of number of Q in the U2S
R1p_t = N1p_t_ave/(Nb0+N1s0+N2s0+N1p0)	# ratio of number of Q in the U1P
Rb_t = 1.0 - R1s_t - R2s_t - R1p_t		# ratio of number of open Q
#### ------------ end of multiple runs averaged and compare ---------- ####


#### ------------ save the data in a h5py file ------------- ####

#file1 = h5py.File('ThermalnoHQT='+str(T)+'N_event='+str(N_ave)+'N_step='+str(N_step)+'Nb0='+str(Nb0)+'N1s0='+str(N1s0)+'N2s0='+str(N2s0)+'N1p0='+str(N1p0)+'Species='+str(Species)+str(Process_chosen)+'.hdf5', 'w')
file1 = h5py.File('UniformPmax='+str(P_sample)+'HQT='+str(T)+'N_event='+str(N_ave)+'N_step='+str(N_step)+'Nb0='+str(Nb0)+'N1s0='+str(N1s0)+'N2s0='+str(N2s0)+'N1p0='+str(N1p0)+'Species='+str(Species)+str(Process_chosen)+'.hdf5', 'w')
file1.create_dataset('1S_percentage', data = R1s_t)
file1.create_dataset('2S_percentage', data = R2s_t)
file1.create_dataset('1P_percentage', data = R1p_t)
#file1.create_dataset('1S_momentum', data = momentum1s_t)
#file1.create_dataset('2S_momentum', data = momentum2s_t)
#file1.create_dataset('1P_momentum', data = momentum1p_t)
file1.create_dataset('time', data = t)
file1.close()

#### ------------ end of saving the file ------------- ####
