from simulate_and_model import *
from make_diagnostic_plots import *
import os


# orbital parameters from https://www.princeton.edu/~willman/planetary_systems/Sol/
# BJD determined by converting values above using https://ssd.jpl.nasa.gov/tc.cgi#top







path_to_directory = './'
input_params = ['earth_jup.csv']

#set up range of inclinations -- ~5 to ~85 deg uniform in cos(inc) space 9 points total
#cosincs = np.linspace(0.1, 0.995, 9)
#inclinations = np.arccos(cosincs)
inclinations = [np.radians(60.)]

#set up observing parameters
roman_errs = [5e-6, 10e-6, 20e-6] #micro-as
roman_durations = [10, 5] #years
gaia_obs = [138] #2x sky avg N_fov from table 2 Perryman+2014 (gave 5 year Nfov = 68.9 *2 = 137.8)
the_errs = [0.3]#, 0.5, 1.0] #THE error in m/s
gaia_errs = [34.2e-6] #from Table 2 Perryman+2014 sigma_fov for birght stars

parallax = 0.1 #as



n_planets = 2
run_number = 19
for run_sub_number in range(4, 11):
	for input_param in input_params:
		for inclination in inclinations:
			for roman_err in roman_errs:
				for roman_duration in roman_durations:
					for gaia_ob in gaia_obs:
						for the_err in the_errs:
							for gaia_err in gaia_errs:
								if __name__ == "__main__":

									run_directory = path_to_directory+'/run'+str(run_number)+'.'+str(run_sub_number)

									try:
										os.mkdir(run_directory)
									except FileExistsError:
										print('run directory already exists...overwriting!')

									orbit_params = load_input_params(path_to_directory+input_param)
									orbit_params[0][5] += inclination
									orbit_params[1][5] += inclination

									print('run number: ' + str(run_number))
									print('start')
									print('--------')

									# save the assumed observational constraints
									observe_params = [roman_err, roman_duration, gaia_ob, the_err, gaia_err]
									observe_params = np.array([observe_params]).T

									observe_params_dic = {
									"roman_err": observe_params[0], 
									"roman_duration": observe_params[1], 
									"gaia_ob": observe_params[2], 
									"the_err": observe_params[3],
									"gaia_err": observe_params[4], 
									}




									observe_params_DF = pd.DataFrame.from_dict(observe_params_dic, orient='index')
									observe_params_DF = observe_params_DF.transpose()

									observe_params_DF.to_csv(run_directory+'/observe_params.csv')



									if roman_err is not None:
										noRoman = False
										print('Roman precision in mu-as: ' + str(int(1e6*roman_err)))
										print('Roman duration in years: ' + str(int(roman_duration)))
									else:
										noRoman = True
										print('Roman precision in mu-as: N/A')
										print('Roman duration in years: N/A')

									print('Earth inclination in degrees: ' + str(np.degrees(orbit_params[0][5])))
									print('Gaia observations in n_obs: ' + str(int(gaia_ob)))
									print('THE precision in m/s: ' + str(the_err))
									print('Gaia precision in mu-as: ' + str(int(1e6*gaia_err)))


									simulate_and_model_data(orbit_params, n_planets, parallax,
										roman_err, the_err, gaia_err, roman_duration, gaia_ob, 
										run_directory+'/planet_params.csv', 
										run_directory+'/sim_data.csv', 
										run_directory+'/posterior.cdf')				


								print('modeling finished for run number' + str(run_number) + '.' + str(run_sub_number))
								run_number += 1
								print('')
								print('')
								print('')
								print('')
								print('')
								print('')
								print('')
								print('')
								print('')
								print('')

print('')
print('')
print('')
print('now running models without Roman!!!')
print('-----------------------------------')
print('')
print('')
print('')

roman_errs = [None] #micro-as ... None for the roman error makes the code not use roman
roman_durations = [10] #years ... only 1 duration, no need for duplicates of no roman 

for run_sub_number in range(1, 11):
	for input_param in input_params:
		for inclination in inclinations:
			for roman_err in roman_errs:
				for roman_duration in roman_durations:
					for gaia_ob in gaia_obs:
						for the_err in the_errs:
							for gaia_err in gaia_errs:
								if __name__ == "__main__":

									run_directory = path_to_directory+'/run'+str(run_number)

									try:
										os.mkdir(run_directory)
									except FileExistsError:
										print('run directory already exists...overwriting!')

									orbit_params = load_input_params(path_to_directory+input_param)
									orbit_params[0][5] += inclination
									orbit_params[1][5] += inclination

									print('run number: ' + str(run_number))
									print('start')
									print('--------')

									# save the assumed observational constraints
									observe_params = [roman_err, roman_duration, gaia_ob, the_err, gaia_err]
									observe_params = np.array([observe_params]).T

									observe_params_dic = {
									"roman_err": observe_params[0], 
									"roman_duration": observe_params[1], 
									"gaia_ob": observe_params[2], 
									"the_err": observe_params[3],
									"gaia_err": observe_params[4], 
									}




									observe_params_DF = pd.DataFrame.from_dict(observe_params_dic, orient='index')
									observe_params_DF = observe_params_DF.transpose()

									observe_params_DF.to_csv(run_directory+'/observe_params.csv')



									if roman_err is not None:
										noRoman = False
										print('Roman precision in mu-as: ' + str(int(1e6*roman_err)))
										print('Roman duration in years: ' + str(int(roman_duration)))
									else:
										noRoman = True
										print('Roman precision in mu-as: N/A')
										print('Roman duration in years: N/A')

									print('Earth inclination in degrees: ' + str(np.degrees(orbit_params[0][5])))
									print('Gaia observations in n_obs: ' + str(int(gaia_ob)))
									print('THE precision in m/s: ' + str(the_err))
									print('Gaia precision in mu-as: ' + str(int(1e6*gaia_err)))


									simulate_and_model_data(orbit_params, n_planets, parallax,
										roman_err, the_err, gaia_err, roman_duration, gaia_ob, 
										run_directory+'/planet_params.csv', 
										run_directory+'/sim_data.csv', 
										run_directory+'/posterior.cdf')				


								print('modeling finished for run number' + str(run_number) + '.' + str(run_sub_number))
								run_number += 1
								print('')
								print('')
								print('')
								print('')
								print('')
								print('')
								print('')
								print('')
								print('')
								print('')



