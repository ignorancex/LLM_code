

import sys
sys.version

import exoplanet
print(f"exoplanet.__version__ = '{exoplanet.__version__}'")



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import exoplanet as xo
import pymc3 as pm
import pymc3_ext as pmx
from astropy import units as u
from astropy.constants import M_earth, M_sun
from simulate import simulate_data, load_input_params
from model import minimize_rv, minimize_both, model_both
from astropy.timeseries import LombScargle
import pickle


import matplotlib 
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18)


def simulate_and_model_data(planet_params, n_planets, parallax,
							roman_err, the_err, gaia_err, roman_duration, gaia_obs, 
							planet_params_filename, sim_data_filename, trace_filename):

	'''
	planet_params = the orbital parameters of the planets we want to simulate and model, type = list
		 	[period, eccentricity, Tper, omega, Omega, inclination, mass]

		if multiple planets then format is [planet_params1, planet_params2]

	n_planets = number of planets, integer
	parallax = parallax of the star in as

	roman_err = roman error in arcseconds, if None assumed no Roman observations
	the_err = THE error in m/s
	gaia_err = GAIA error in arcseconds

	roman_duration = roman duration in years
	gaia_obs = number of gaia observations taken over the 10 year gaia lifetime
	sigma_THE
	'''

	######################################################
	######################################################
	######################################################
	######################################################
	    # save input orbital parameters as .csv file #
	######################################################
	######################################################
	######################################################
	######################################################
	######################################################



	planet_params_out = np.array(planet_params).T

	planet_params_dic = {
	"period": planet_params_out[0], 
	"ecc": planet_params_out[1], 
	"Tper": planet_params_out[2], 
	"omega": planet_params_out[3],
	"Omega": planet_params_out[4], 
	"inclination": planet_params_out[5], 
	"mass": planet_params_out[6], 
	}




	planet_params_DF = pd.DataFrame.from_dict(planet_params_dic, orient='index')
	planet_params_DF = planet_params_DF.transpose()

	planet_params_DF.to_csv(planet_params_filename)



	##################
	##################
	##################
	##################
	#begin simulate data
	##################
	##################
	##################
	##################
	##################


	 


	#add gaia observing times
	times_observed_astrometry_gaia = []
	t_0 = 0
	for ii in range(t_0, t_0+3600):
		if ii % (int(3600/gaia_obs)) == 0:
			times_observed_astrometry_gaia.append(ii)

	
			
	#add THE observing times
	times_observed_rv = []
	t_0 = 3300 #Gaia first light (2014) ~9 years before THE first expected light (2023)
	add_data = True
	for ii in range(t_0, t_0+3600):
		
		if ii % 180 == 0:
			if add_data:
				add_data = False
			else:
				add_data = True
		   
		if add_data:
			times_observed_rv.append(ii)
			

	'''
	Gaia AC vs. AL scan angle error determination:
	----------------------------------------------
	assume some random scan angle between 0-90

	~5 times worse across scan vs. along scan angle
	RA = RA_AL + RA_AC
	RA_AL = sigma*sin(theta)
	RA_AC =5*sigma*sin(pi/2 - theta)

	dec = dec_AL + dec_AC
	dec_AL = sigma*cos(theta)
	dec_AC =5*sigma*cos(pi/2 - theta)
	'''

	scan_angle = np.random.uniform(0, np.pi/2, len(times_observed_astrometry_gaia))


	sigma_ra_gaia_al = gaia_err * np.sin(scan_angle)
	sigma_ra_gaia_ac = 5 * gaia_err * np.cos(scan_angle)
	sigma_ra_gaia = sigma_ra_gaia_al + sigma_ra_gaia_ac


	
	sigma_dec_gaia_al = gaia_err * np.cos(scan_angle)
	sigma_dec_gaia_ac = 5 * gaia_err * np.sin(scan_angle)
	sigma_dec_gaia = sigma_dec_gaia_al + sigma_dec_gaia_ac

	

	sigma_rv = the_err

	parallax = parallax 



	times, rv_results, ra_results, dec_results = simulate_data(
		n_planets, 
		sigma_rv, 
		sigma_ra_gaia,
		sigma_dec_gaia,
		parallax,
		planet_params,
		times_observed_rv = times_observed_rv,
		times_observed_astrometry = times_observed_astrometry_gaia
		)


	[[times_rv, times_observed_rv, times_astrometry, times_observed_astrometry],
	[rv_orbit, rv_orbit_sum, rv_sim, rv_sim_sum],
	[ra_orbit, ra_orbit_sum, ra_sim, ra_sim_sum],
	[dec_orbit, dec_orbit_sum, dec_sim, dec_sim_sum]]  = times, rv_results, ra_results, dec_results

	ra_gaia_err = np.full(np.shape(ra_sim_sum), sigma_ra_gaia)
	dec_gaia_err = np.full(np.shape(dec_sim_sum), sigma_dec_gaia)


	#add roman observing times if roman_err not None
	if roman_err is not None:
		t_1 =  times_observed_astrometry_gaia[-1]+1800
		times_observed_astrometry_roman = []
		for ii in range(t_1, t_1+(roman_duration*365)):
			if ii % 90 == 0:
				times_observed_astrometry_roman.append(ii)	


		sigma_ra_roman = roman_err
		sigma_dec_roman = roman_err



		times, rv_results, ra_results, dec_results = simulate_data(
			n_planets, 
			sigma_rv, 
			sigma_ra_roman,
			sigma_dec_roman,
			parallax,
			planet_params,
			times_observed_rv = times_observed_rv,
			times_observed_astrometry = times_observed_astrometry_roman
			)

		times_astrometry = np.append(times_astrometry, times[2], axis=0)

		times_observed_astrometry = np.append(times_observed_astrometry, times[3], axis=0)

		ra_orbit = np.append(ra_orbit, ra_results[0], axis=0)
		ra_orbit_sum = np.append(ra_orbit_sum, ra_results[1], axis=0)
		ra_sim = np.append(ra_sim, ra_results[2], axis=0)
		ra_sim_sum = np.append(ra_sim_sum, ra_results[3], axis=0)

		dec_orbit = np.append(dec_orbit, dec_results[0], axis=0)
		dec_orbit_sum = np.append(dec_orbit_sum, dec_results[1], axis=0)
		dec_sim = np.append(dec_sim, dec_results[2], axis=0)
		dec_sim_sum = np.append(dec_sim_sum, dec_results[3], axis=0)

		ra_roman_err = np.full(np.shape(ra_results[3]), sigma_ra_roman)
		dec_roman_err = np.full(np.shape(dec_results[3]), sigma_dec_roman)


	
	##################
	##################
	##################
	##################
	#end simulate data
	##################
	##################
	##################
	##################
	##################



	

	################
	################
	#rename variables in more consistent way for modeling
	x_rv = np.array(times_observed_rv)
	y_rv = rv_sim_sum
	y_rv_err = np.full(np.shape(y_rv), sigma_rv)

	x_astrometry = np.array(times_observed_astrometry)
	ra_data = ra_sim_sum
	dec_data = dec_sim_sum


	if roman_err is not None:
		ra_err = np.concatenate((ra_gaia_err, ra_roman_err))
		dec_err = np.concatenate((dec_gaia_err, dec_roman_err))

	else:
		ra_err = ra_gaia_err
		dec_err = dec_gaia_err


	#save simulated data as dataframe and export to csv
	simulated_data_dic = {
		"times_rv_observed": x_rv,
		"rv_observed": y_rv,
		"rv_err_observed": y_rv_err,

		"times_rv_orbit": times_rv,
		"rv_orbit": rv_orbit,
		"rv_orbit_sum": rv_orbit_sum,

		"times_astrometry_observed": x_astrometry,
		"ra_observed": ra_data,
		"dec_observed": dec_data,
		"ra_err_observed": ra_err,
		"dec_err_observed": dec_err,

		"times_astrometry_orbit": times_astrometry,
		"ra_orbit": ra_orbit,
		"ra_orbit_sum": ra_orbit_sum,
		"dec_orbit": dec_orbit,
		"dec_orbit_sum": dec_orbit_sum
		}


	simulated_data = pd.DataFrame.from_dict(simulated_data_dic, orient='index')
	simulated_data = simulated_data.transpose()

	simulated_data.to_csv(sim_data_filename)
		
	# make a fine grid that spans the observation window for plotting purposes
	t_astrometry = np.linspace(x_astrometry.min() - 5, x_astrometry.max() + 5, 1000)
	t_rv = np.linspace(x_rv.min() - 5, x_rv.max() + 5, 1000)

	# for predicted orbits
	t_fine = np.linspace(x_astrometry.min() - 500, x_astrometry.max() + 500, num=1000)




	##################
	##################
	##################
	##################
	#begin model data
	##################
	##################
	##################
	##################
	##################
	
	################
	################
	#Lombs Scargle Periodogram on RV data
	frequency, power = LombScargle(x_rv, y_rv).autopower()
	period = 1/frequency


	period_cut1 = period[period > 10]
	power_cut1 = power[period > 10]



	indices = power_cut1.argsort()[-1:][::-1]
	period1 = period_cut1[indices][0]
	print('LS period 1: ' + str(period1))

	period1_min_cut = 500
	#period_cut1 > period1_min_cut so we don't double count

	period_cut2 = period_cut1[period_cut1 < period1_min_cut]

	power_cut2 = power_cut1[period_cut1 < period1_min_cut]


	indices = power_cut2.argsort()[-1:][::-1]
	period2 = period_cut2[indices][0]
	print('LS period 2: ' + str(period2))



	
	################
	################
	#minimize on RV data
	
	#if you want to use lombs scargle period estimate then uncomment line below
	#periods_guess = np.array([period2, period1])

	#if you want to use lombs scargle period estimate then comment line below
	periods_real = planet_params_dic['period'] 
	periods_guess = []
	print('periods here:')
	print(periods_real)
	for per in periods_real:
		print(per)
		periods_guess.append(per + np.random.uniform(low=-0.1*per, high=0.1*per))

	print(periods_guess)
	periods_guess = np.array(periods_guess)
	print("using simulated period values as starting point for RV model")


	Ks_guess = xo.estimate_semi_amplitude(periods_guess, x_rv, y_rv, y_rv_err)

	rv_map_soln = minimize_rv(periods_guess, Ks_guess, x_rv, y_rv, y_rv_err)


	################
	################
	#minimize on joint model
	joint_model, joint_map_soln, joint_logp = minimize_both(
		rv_map_soln, x_rv, y_rv, y_rv_err, x_astrometry, 
		ra_data, ra_err, dec_data, dec_err, parallax
	)




	################
	################
	#run full MCMC
	trace = model_both(joint_model, joint_map_soln, 4000, 4000)


	##################
	##################
	##################
	##################
	#end model data
	##################
	##################
	##################
	##################
	##################


	################
	################
	#save trace and model
	trace.posterior.to_netcdf(trace_filename, group='posterior')

	return joint_model, trace













