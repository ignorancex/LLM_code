import numpy as np
import pandas as pd


def make_map_priors(posterior_stats_filename):
	data = pd.read_csv(posterior_stats_filename)
	modes = []
	map_vals = []
	map_vals_all = []


	prior_params = [
		' 0.0 0.0 0.0 0 = log10(m_b/m_star)',
		' 0.0 0.0 0.0 0 = m_c/m_star',
		' 0.0 0.0 0.0 0 = lambda_c (days)',
		' 0.0 0.0 0.0 0 = P_c (days)',
		' 0.0 0.0 0.0 0 = e_b',
		' 0.0 0.0 0.0 0 = e_c',
		' 0.0 0.0 0.0 0 = varpi_b (deg), longitude!',
		' 0.0 0.0 0.0 0 = varpi_c (deg), longitude!',
		' 0.0 0.0 0.0 0 = b_c, see notebook calculation',
		' 0.0 0.0 0.0 0 = Omega_c-Omega_b (deg), Omega_b = 270',
		' 0.0 0.0 0.0 0 = P_b',
		' 0.0 0.0 0.0 0 = delta_t'
	]


	nparams = len(prior_params)
	row_num = 1
	map_start_row = -10000
	for row in data.values:
		if 'Mode  ' in row[0]:
			modes.append('mode'+(row[0][-1:]))
		
		if row[0] == 'Maximum Likelihood Parameters':
			map_start_row = row_num + 1
		
		if map_start_row < row_num <= map_start_row + nparams:
			map_vals.append(row[0][7:])
			
		if row_num == map_start_row + nparams:
			map_vals_all.append(map_vals)
			map_vals = []
			
		row_num += 1
		
		

	

	map_priors_all = []
	for ii in range(0, len(modes)):
		mode = modes[ii]
		map_vals = map_vals_all[ii]
		
		map_priors = []
		for jj in range(0, len(map_vals)):
			map_val = map_vals[jj]
			map_priors.append(map_val + prior_params[jj])
			
		
		map_priors_all.append(map_priors)



	for ii in range(0, len(modes)):
		mode = modes[ii]
		priors_name = './map_priors/priors_map_' + mode + '.in'
		map_priors = np.array(map_priors_all[ii], dtype=str)
		
		print('saving map priors for mode ' + mode[-1] + ' as ' + priors_name)
		
		np.savetxt(priors_name, map_priors, fmt='%s')


	return None




posterior_stats_filename = str(input("filename where posterior stats are saved: "))


make_map_priors(posterior_stats_filename)


