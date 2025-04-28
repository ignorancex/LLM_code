import numpy as np

# a list of evolved MIST models; each line is 
# EEP log10_isochrone_age_yr initial_mass star_mass log_L log_L_div_Ledd log_Teff\
# log_R surf_avg_omega surf_r_equatorial_div_r surf_r_polar_div_r surf_avg_omega_div_omega_crit
print('Converting the MIST model data from txt to npy...')
evolved = []
f_ev = open('mist_isochrones.txt')
lastline = ''
for line in f_ev:
	if ('Zinit' in lastline):
		Z, logZ, otilde0 = line.split()
	elif (line[0] != '#') and line.strip():
		# get the parameters of the evolved model
		evolved.append( [logZ, otilde0] + line.split() )
	lastline = line
f_ev.close()
evolved = np.array(evolved).astype(np.float) # convert to numpy array
np.save('mist_grid.npy', evolved, allow_pickle=False)