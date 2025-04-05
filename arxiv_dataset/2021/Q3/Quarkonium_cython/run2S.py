#!/usr/bin/env python
import numpy as np
import scipy.integrate as si
from Dynam_quarkonium_evolution import QQbar_evol
import h5py

t_hydro = {'200':{'0-10': 10.0, '10-20':8.68, '20-30':7.8, '30-40':6.96,
				  '40-50':6.08, '50-60':5.26, '60-70':4.4, '70-80':3.6, '80-90':1.82, '90-100':0.6}, 
		'2760':{'0-5':12.4, '5-10':11.6, '10-20':10.6, '20-30':9.7, '30-40':8.8,
				  '40-50':7.7, '50-60':6.6, '60-70':5.5, '70-80':4.4, '80-90':3.4, '90-100':1.0},
		'5020':{'0-5':12.9, '5-10':12.2, '10-20':11.2, '20-30':10.1, '30-40':9.2,
			   '40-50':8.1, '50-60':7.0, '60-70':5.9, '70-80':4.7, '80-90':3.8, '90-100':1.24, '10-90':8.6}}

#### ------------ multiple runs averaged and compare ---------------- ####
centrality = '0-5'
energy = 5020
N_ave = 20		# No. of parallel runnings
pt = 0.0
y_max = 2.4
dt_run = 0.005
N_step = int(t_hydro[str(energy)][centrality] * np.cosh(y_max)/dt_run)	# total step required is t_total / dt, t_total = t_hydro * cosh(y_max)
tmax = N_step*dt_run
t = np.linspace(0.0, tmax, N_step+1)
p4i_2S = []			# initial p4
p4f_1S = []			# final p4
p4f_2S = []
p4f_1P = []
tForm_1S = []		# formation time of 1S
tForm_2S = []
tForm_1P = []
#N1S_t = []			# time sequence of No. of 1S state
#N2S_t = []			# time sequence of No. of 2S state
#N1P_t = []			# time sequence of No. of 1P state

# define the event generator
event_gen = QQbar_evol(centrality_str_given = centrality, energy_GeV = energy, recombine = True, HQ_scat = True, sample_method = '2S', pT = pt)

for i in range(N_ave):
	# initialize N_ave number of events
	event_gen.initialize()
	#N1S_t.append([])
	#N2S_t.append([])
	#N1P_t.append([])
	
	# store initial momenta
	leni_2S = len(event_gen.U2Slist['4-momentum'])	# initial No. of 1S
	for k in range(leni_2S):
		p4i_2S.append(event_gen.U2Slist['4-momentum'][k])
		
	# run the event
	for j in range(N_step+1):
		#N1S_t[i].append(len(event_gen.U1Slist['4-momentum']))	# store N_1S(t) for each event
		#N2S_t[i].append(len(event_gen.U2Slist['4-momentum']))
		#N1P_t[i].append(len(event_gen.U1Plist['4-momentum']))
		event_gen.run(dt = dt_run)
	
	# store final momenta
	lenf_1S = len(event_gen.U1Slist['4-momentum'])	# final No. of 1S
	lenf_2S = len(event_gen.U2Slist['4-momentum'])	# final No. of 2S
	lenf_1P = len(event_gen.U1Plist['4-momentum'])	# final No. of 1P
	for k in range(lenf_1S):
		p4f_1S.append(event_gen.U1Slist['4-momentum'][k])
		tForm_1S.append(event_gen.U1Slist['last_form_time'][k])
	for k in range(lenf_2S):
		p4f_2S.append(event_gen.U2Slist['4-momentum'][k])
		tForm_2S.append(event_gen.U2Slist['last_form_time'][k])
	for k in range(lenf_1P):
		p4f_1P.append(event_gen.U1Plist['4-momentum'][k])
		tForm_1P.append(event_gen.U1Plist['last_form_time'][k])
		
	event_gen.dict_clear()	## clear data from last simulation

	
#N1S_t = np.array(N1S_t)
#N2S_t = np.array(N2S_t)
#N1P_t = np.array(N1P_t)
#N1S_t_ave = np.sum(N1S_t, axis = 0)/(N_ave + 0.0)	# averaged number of 1S state (time-sequenced)
#N2S_t_ave = np.sum(N2S_t, axis = 0)/(N_ave + 0.0)
#N1P_t_ave = np.sum(N1P_t, axis = 0)/(N_ave + 0.0)
#### ------------ end of multiple runs averaged and compare ---------- ####



#### ------------ save the data in a h5py file ------------- ####

file1 = h5py.File('energy='+str(energy)+'GeVcentrality='+str(centrality)+'N_event='+str(N_ave)+'_2S.hdf5', 'w')
file1.create_dataset('1Sp4final', data = p4f_1S)
f#ile1.create_dataset('1Snumber', data = N1S_t_ave)
file1.create_dataset('1Sformtime', data = tForm_1S)
file1.create_dataset('2Sp4initial', data = p4i_2S)
file1.create_dataset('2Sp4final', data = p4f_2S)
#file1.create_dataset('2Snumber', data = N2S_t_ave)
file1.create_dataset('2Sformtime', data = tForm_2S)
file1.create_dataset('1Pp4final', data = p4f_1P)
#file1.create_dataset('1Pnumber', data = N1P_t_ave)
file1.create_dataset('1Pformtime', data = tForm_1P)
file1.create_dataset('time', data = t)
file1.close()

#### ------------ end of saving the file ------------- ####
