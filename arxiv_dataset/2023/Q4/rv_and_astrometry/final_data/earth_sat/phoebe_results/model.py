import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import exoplanet as xo
import pymc3 as pm
import pymc3_ext as pmx
from astropy import units as u
from astropy.constants import M_earth, M_sun
from astropy import constants
import aesara_theano_fallback.tensor as tt
from aesara_theano_fallback import aesara as theano
import arviz as az
import corner


import matplotlib 
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18)



def minimize_rv(periods, Ks, x_rv, y_rv, y_rv_err):
	t_rv = np.linspace(x_rv.min() - 5, x_rv.max() + 5, 1000)
	print("minimizing RV only model solutions pre-MCMC")
	print("------------")

	with pm.Model() as model:
	
		
		#log normal prior on period around estimates
		logP = pm.Uniform(
			"logP",
			lower=0,
			upper=11,
			shape=2,
			testval=np.log(periods),
		)
		


		P = pm.Deterministic("P", tt.exp(logP))


		##  wide uniform prior on t_periastron
		tperi = pm.Uniform("tperi", lower=0, upper=periods, shape=2)
		
		
		# Wide normal prior for semi-amplitude
		logK = pm.Uniform("logK", lower=-4, upper=4, shape=2, testval=np.log(Ks))
		
		K = pm.Deterministic("K", tt.exp(logK))
		
		
		# Eccentricity & argument of periasteron
		ecs = pmx.UnitDisk("ecs", shape=(2, 2), testval=0.01 * np.ones((2, 2)))
		ecc = pm.Deterministic("ecc", tt.sum(ecs ** 2, axis=0))
		omega = pm.Deterministic("omega", tt.arctan2(ecs[1], ecs[0]))

		xo.eccentricity.kipping13("ecc_prior", fixed=True, observed=ecc)



		
		# Jitter & a quadratic RV trend
		#logs = pm.Normal("logs", mu=np.log(np.median(y_rv_err)), sd=y_rv_err)

		# Then we define the orbit
		orbit = xo.orbits.KeplerianOrbit(period=P, t_periastron=tperi, ecc=ecc, omega=omega)

		# And a function for computing the full RV model
		def get_rv_model(t, name=""):
			# First the RVs induced by the planets
			vrad = orbit.get_radial_velocity(t, K=K)
			pm.Deterministic("vrad" + name, vrad)

			# Sum over planets and add the background to get the full model
			return pm.Deterministic("rv_model" + name, tt.sum(vrad, axis=-1))

			

		# Define the RVs at the observed times
		rv_model = get_rv_model(x_rv)

		# Also define the model on a fine grid as computed above (for plotting)
		rv_model_pred = get_rv_model(t_rv, name="_pred")

		# Finally add in the observation model. This next line adds a new contribution
		# to the log probability of the PyMC3 model
		#err = tt.sqrt(y_rv_err ** 2 + tt.exp(2 * logs))
		err = y_rv_err
		pm.Normal("obs", mu=rv_model, sd=err, observed=y_rv)


		map_soln = model.test_point
		map_soln = pmx.optimize(start=map_soln, vars=[tperi])
		map_soln = pmx.optimize(start=map_soln, vars=[P])
		map_soln = pmx.optimize(start=map_soln, vars=[ecs])
		map_soln = pmx.optimize(start=map_soln)


	#return the max a-posteriori solution
	return map_soln



def a_from_Kepler3(period, M_tot):
	period = period*86400 #days to seconds
	

	M_tot = M_tot*M_sun.value #solar masses to kg
	
	a3 = ( ((constants.G.value)*M_tot) / (4*np.pi**2) ) * (period)**2.
	
	a = a3**(1/3)
	
	
	
	a = a * 6.68459*10**(-12.) # meters to AU
	
	return(a) #in AUs


def semi_amplitude(m_planet, a, ecc, inclination):
	from astropy.constants import G

	K = \
	np.sqrt(G / (1-(ecc**2.))) * ((m_planet*M_sun)*np.sin(inclination)) * \
	((M_sun+(m_planet*M_sun))** (-(1./2.))) * \
	(a*u.AU.to(u.m))  ** ((-1./2.))
	
	return K.value



def min_mass(K, period, ecc):
	#from http://exoplanets.astro.yale.edu/workshop/EPRV/Bibliography_files/Radial_Velocity.pdf
	m_jup = 317.83*3.00273e-6 #units m_sun
	m_sun = 333030 #earth masses

	m_planet = K/((m_sun*m_jup)*28.4329/(np.sqrt(1-ecc**2.)) \
		*(m_sun)**(-2/3) * (period / 365.256) ** (-1/3))

	return m_planet/m_sun

def determine_phase(P, t_periastron):
	phase = (2 * np.pi * t_periastron) / P
	return phase





def minimize_both(rv_map_soln, x_rv, y_rv, y_rv_err, x_astrometry, ra_data, ra_err, dec_data, dec_err, parallax):
	m_sun = 333030 #earth masses
	
	P_RV = np.array(rv_map_soln['P'])
	K_RV = np.array(rv_map_soln['K'])
	tperi_RV = np.array(rv_map_soln['tperi'])
	ecc_RV = np.array(rv_map_soln['ecc'])
	omega_RV = np.array(rv_map_soln['omega'])
	min_masses_RV = xo.estimate_minimum_mass(P_RV, x_rv, y_rv, y_rv_err).value*317.83 #in m_earth
	phase_RV = determine_phase(P_RV, tperi_RV)
	
	
	# make a fine grid that spans the observation window for plotting purposes
	t_astrometry = np.linspace(x_astrometry.min() - 5, x_astrometry.max() + 5, 1000)
	t_rv = np.linspace(x_rv.min() - 5, x_rv.max() + 5, 1000)

	# for predicted orbits
	t_fine = np.linspace(x_astrometry.min() - 500, x_astrometry.max() + 500, num=1000)


	print("RV Solutions")
	print("------------")
	print("P: ", P_RV)
	print("K: ", K_RV)
	print("T_peri: ", tperi_RV)
	print("eccentricity: ", ecc_RV)
	print("omega: ", omega_RV)

	print('')
	print("minimizing joint model solutions pre-MCMC")
	print("------------")

	# for predicted orbits
	t_fine = np.linspace(x_astrometry.min() - 500, x_astrometry.max() + 500, num=1000)

	inc_test_vals = np.array(np.radians([5., 25., 45., 65., 85.]))
	model, map_soln, logp = [], [], []
	for inc in inc_test_vals:
		mass_test_vals = min_masses_RV/np.sin(inc)
		print('')
		print("trying inclination = " + str(np.degrees(inc)))
		print("mass test val = " + str(mass_test_vals))
		print("------------")
		
		



		def get_model():
			with pm.Model() as model:


				# Below we will run a version of this model where a measurement of parallax is provided
				# The measurement is in milliarcsec
				m_plx = pm.Bound(pm.Normal, lower=0, upper=200)(
					"m_plx", mu=parallax*1000, sd=10, testval=parallax*1000
				)
				plx = pm.Deterministic("plx", 1e-3 * m_plx)


				# We expect the period to be around that found from just the RVs,
				# so we'll set a broad prior on logP
				
				logP = pm.Uniform(
					"logP", lower=0, upper=np.log(2*P_RV), testval=np.log(P_RV), shape=2
				)
				P = pm.Deterministic("P", tt.exp(logP))
				
				# Eccentricity & argument of periasteron
				ecs = pmx.UnitDisk("ecs", shape=(2, 2), 
								   testval=np.array([np.sqrt(ecc_RV)*np.cos(omega_RV), 
													 np.sqrt(ecc_RV)*np.sin(omega_RV)]))
				ecc = pm.Deterministic("ecc", tt.sum(ecs ** 2, axis=0))
				omega = pm.Deterministic("omega", tt.arctan2(ecs[1], ecs[0]))

				xo.eccentricity.kipping13("ecc_prior", fixed=True, observed=ecc)

				
				

				# Omegas are co-dependent, so sample them with variables Omega_plus
				# and Omegas_minus. Omega_plus is (Omega_0 + Omega_1)/2 and 
				# Omega_minus is (Omega_0 - Omega_1)/2
				
				Omega_plus = pmx.Angle("Omega_plus", shape=1)
				Omega_minus = pmx.Angle("Omega_minus", shape=1)
				
				
				Omega = tt.concatenate( [(Omega_plus + Omega_minus),
										 (Omega_plus - Omega_minus)] )
				

				Omega = pm.Deterministic("Omega", Omega) 
				Omega_sum = pm.Deterministic("Omega_sum", ((Omega_plus)*2)% np.pi)
				Omega_diff = pm.Deterministic("Omega_diff", ((Omega_minus)*2)% np.pi)
				


			
				# For these orbits, it can also be better to fit for a phase angle
				# (relative to a reference time) instead of the time of periasteron
				# passage directly
				phase = pmx.Angle("phase", testval=phase_RV, shape=2)
				tperi = pm.Deterministic("tperi", P * phase / (2 * np.pi))
				

				
				# uniform prior on sqrtm_sini and sqrtm_cosi (upper 100* testval to stop planet flipping)
				log_m = pm.Uniform("log_m", lower=-1, upper=np.log(100*mass_test_vals), testval=np.log(mass_test_vals), shape=2)
				m_planet = pm.Deterministic("m_planet", tt.exp(log_m))
				m_planet_fit = pm.Deterministic("m_planet_fit", m_planet/m_sun)


				cos_incl = pm.Uniform("cos_incl", lower=0, upper=1, testval=np.cos(inc), shape=2)
				incl = pm.Deterministic("incl", tt.arccos(cos_incl))
				


				
				# add keplers 3 law function
				a = pm.Deterministic("a", a_from_Kepler3(P, 1.0+m_planet_fit))
				
				# Set up the orbit
				orbit = xo.orbits.KeplerianOrbit(
					t_periastron=tperi,
					period=P,
					incl=incl,
					ecc=ecc,
					omega=omega,
					Omega=Omega,
					m_planet = m_planet_fit,
					plx=plx
				)


				
				
				# Add a function for computing the full astrometry model
				def get_astrometry_model(t, name=""):
					# First the astrometry induced by the planets

					# determine and print the star position at desired times
					pos = orbit.get_star_position(t, plx)

					x,y,z = pos


					# calculate rho and theta
					rhos = tt.squeeze(tt.sqrt(x ** 2 + y ** 2))  # arcsec
					thetas = tt.squeeze(tt.arctan2(y, x))  # radians between [-pi, pi]
									
					
					#rhos, thetas = get_star_relative_angles(t, plx)
					
					
					dec = pm.Deterministic("dec" + name, rhos * np.cos(thetas)) # X is north
					ra = pm.Deterministic("ra" + name, rhos * np.sin(thetas)) # Y is east
					
					# Sum over planets to get the full model
					dec_model = pm.Deterministic("dec_model" + name, tt.sum(dec, axis=-1))
					ra_model = pm.Deterministic("ra_model" + name, tt.sum(ra, axis=-1))
					

					
					return dec_model, ra_model

				
				# Define the astrometry model at the observed times
				dec_model, ra_model = get_astrometry_model(x_astrometry)

				# Also define the model on a fine grid as computed above (for plotting)
				dec_model_fine, ra_model_fine = get_astrometry_model(t_fine, name="_fine")

				

				# Add jitter terms to both separation and position angle
				#log_dec_s = pm.Normal(
				#	"log_dec_s", mu=np.log(np.median(dec_err)), sd=dec_err
				#)
				#log_ra_s = pm.Normal(
				#	"log_ra_s", mu=np.log(np.median(ra_err)), sd=ra_err
				#)

				#dec_tot_err = tt.sqrt(dec_err ** 2 + tt.exp(2 * log_dec_s))
				#ra_tot_err = tt.sqrt(ra_err ** 2 + tt.exp(2 * log_ra_s))
				dec_tot_err = dec_err
				ra_tot_err = ra_err


				# define the likelihood function, e.g., a Gaussian on both ra and dec		
				pm.Normal("dec_obs", mu=dec_model, sd=dec_tot_err, observed=dec_data)
				pm.Normal("ra_obs", mu=ra_model, sd=ra_tot_err, observed=ra_data)


				
				
				# ADD RV MODEL
				# Jitter & a quadratic RV trend
				#log_rv = pm.Normal("log_rv", mu=np.log(np.median(y_rv_err)), sd=y_rv_err)


				# And a function for computing the full RV model
				def get_rv_model(t, name=""):
					# First the RVs induced by the planets
					vrad = orbit.get_radial_velocity(t)
					pm.Deterministic("vrad" + name, vrad)

					# Sum over planets to get the full model
					return pm.Deterministic("rv_model" + name, tt.sum(vrad, axis=-1))

				# Define the RVs at the observed times
				rv_model = get_rv_model(x_rv)

				# Also define the model on a fine grid as computed above (for plotting)
				rv_model_pred = get_rv_model(t_rv, name="_pred")

				# Finally add in the observation model. This next line adds a new contribution
				# to the log probability of the PyMC3 model
				#rv_err = tt.sqrt(y_rv_err ** 2 + tt.exp(2 * log_rv))
				rv_err = y_rv_err

				pm.Normal("obs_RV", mu=rv_model, sd=rv_err, observed=y_rv)

				# Optimize to find the initial parameters
				map_soln = model.test_point
				map_soln = pmx.optimize(map_soln, vars=[Omega, phase])
				#map_soln = pmx.optimize(map_soln, vars=[Omega, m_planet, incl, ecs, phase])

				map_soln = pmx.optimize(map_soln)


			return model, map_soln



		a_model, a_map_soln = get_model()
		a_logp = a_model.check_test_point(test_point=a_map_soln).sum(axis = 0)
		print('log likelihood = ' + str(a_logp))

		model.append(a_model)
		map_soln.append(a_map_soln)
		logp.append(a_logp)

	
	best_index = 0
	for index in range(0, len(model)):
		if logp[index] >= logp[best_index]:
			best_index = index

	the_model = model[best_index]
	the_map_soln = map_soln[best_index]
	the_logp = logp[best_index]



	return the_model, the_map_soln, the_logp


def model_both(model, map_soln, tune_steps, draw_steps):
	print('Joint RV + Astometry Minimization Solutions:')
	print("------------")

	print('m_planet: ' + str(map_soln['m_planet']))
	print('P: ' + str(map_soln['P']))
	print('incl: ' + str(map_soln['incl']))
	print('Omega: ' + str(map_soln['Omega']))
	print('tperi: ' + str(map_soln['tperi']))
	print('ecc: ' + str(map_soln['ecc']))
	print('omega: ' + str(map_soln['omega']))
	print('plx: ' + str(map_soln['plx']))


	with model:
		trace = pmx.sample(
			tune=tune_steps,
			draws=draw_steps,
			start=map_soln,
			cores=2,
			chains=2,
			target_accept=0.99,
			return_inferencedata=True,
		)

	return trace



def make_plots(trace, orbits_params, n_planets):

	if n_planets == 2:
		[orbit_params_earth, orbit_params_jup] = orbit_params
		
		[P_earth, e_earth, Tper_earth, omega_earth, Omega_earth, 
		inclination_earth, m_earth] = orbit_params_earth


		[P_2, e_jup, Tper_jup, omega_jup, Omega_jup, 
		inclination_jup, m_jup] = orbit_params_jup


		a_true_earth = a_from_Kepler3(P_earth, 1.0+m_earth)
		a_true_jup = a_from_Kepler3(P_jup, 1.0+m_jup)

		#[P1, P2, e1, e2, omega1, omega2, Omega_sum, Omega_minus, incl1, incl2, plx, m1, m2, a1, a2, Tper1, Tper2]
		truths = [P_earth, P_jup, e_earth, e_jup, omega_earth, omega_jup, 
		Omega_earth+Omega_jup, Omega_earth-Omega_jup, inclination_earth, inclination_jup, 
		0.1, m_earth*m_sun, m_jup*m_sun, a_true_earth, a_true_jup, Tper_earth, Tper_jup]


		#[P1, P2, e1, e2, omega1, omega2, incl1, incl2, Tper1, Tper2, m1, m2, a1, a2]
		truth_chain_plot = [P_earth, P_jup, e_earth, e_jup, omega_earth, omega_jup, 
		inclination_earth, inclination_jup, Tper_earth, Tper_jup, m_earth*m_sun, m_jup*m_sun, a_true_earth, a_true_jup]






	else:
		[P, e, Tper, omega, Omega, inclination, m] = orbit_params

		a = a_from_Kepler3(P, 1.0+m)

		#[P1, e1, omega1, Omega1, incl1, plz, m1, a1, Tper]
		truths = [P, e, omega, Omega, inclination, 0.1, m*m_sun, a_true_earth, Tper]

		#[P1, e1, omega1, incl1, Tper1, m1, a1]
		truth_chain_plot = [P, e, omega, inclination, Tper, m*m_sun, a_true_earth]




	# plot the table summarizing MCMC results
	az.summary(
		trace,
		var_names=["P", "tperi", "omega", "Omega_sum", "Omega_plus", "Omega_minus", 
				   "incl", "ecc", "plx", "m_planet", "a"],
	)


	plt.show()





	# plot the corner plots
	_ = corner.corner(
		trace, var_names=["P", "ecc", "omega", "Omega_sum", "Omega_minus", "incl", 
						  "plx", "m_planet", "a", 'tperi'], quantiles=[0.16, 0.5, 0.84],
						   show_titles=True, title_kwargs={"fontsize": 13}, 
						   truths = truths, truth_color = "#03003a"
	)

	plt.show()



	# plot the corner plots for Omegas
	_ = corner.corner(
		trace, var_names=["Omega_plus", "Omega_sum", "Omega_minus", "Omega"], quantiles=[0.16, 0.5, 0.84],
						   show_titles=True, title_kwargs={"fontsize": 13}
	)

	plt.show()


	# plot the posterior predictions for the astometry data
	ekw = dict(fmt=".k", lw=0.5)

	fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(9, 9))
	ax[0].set_ylabel(r'$\rho\,$ ["]')
	ax[1].set_ylabel(r"P.A. [radians]")
	ax[1].set_xlabel(r"time [days]")

	tot_rho_err = np.sqrt(
		rho_err ** 2
		+ np.exp(2 * np.median(trace.posterior["log_rho_s"].values, axis=(0, 1)))
	)
	tot_theta_err = np.sqrt(
		theta_err ** 2
		+ np.exp(2 * np.median(trace.posterior["log_theta_s"].values, axis=(0, 1)))
	)

	ax[0].errorbar(x_astrometry, rho_data, yerr=tot_rho_err, **ekw)
	q = np.percentile(trace.posterior["rho_model_pred"].values, [16, 84], axis=(0, 1))
	ax[0].fill_between(t_fine, q[0], q[1], color="#773f6a", alpha=0.8, lw=1)

	ax[1].errorbar(x_astrometry, theta_data, yerr=tot_theta_err, **ekw)
	q = np.percentile(trace.posterior["theta_model_pred"].values, [16, 84], axis=(0, 1))
	ax[1].fill_between(t_fine, q[0], q[1], color="#773f6a", alpha=0.8, lw=1)

	ax[-1].set_xlim(t_fine[0], t_fine[-1])
	_ = ax[0].set_title("posterior inferences")

	plt.show()




	# plot the posterior predictions for the RV data
	rv_pred = trace.posterior["rv_model_pred"].values
	pred = np.percentile(rv_pred, [16, 50, 84], axis=(0, 1))

	fig, ax = plt.subplots(1, figsize = [15,10], sharey=True)

	ax.errorbar(x_rv, y_rv, yerr=y_rv_err, fmt=".k", alpha = 0.3, label='data', zorder = 0)
	ax.plot(t, pred[1], color="#773f6a", label="model", zorder = 1)
	art = ax[0].fill_between(t, pred[0], pred[2], color="#773f6a", alpha=0.3)
	art.set_edgecolor("none")

	ax.legend(fontsize=10)
	ax.set_xlim(t.min(), t.max())
	ax.set_xlabel("time [days]")
	ax.set_ylabel("radial velocity [m/s]")
	ax.set_title("MCMC posterior and data")



	plt.legend(fontsize=10)
	plt.xlim(t.min(), t.max())
	plt.xlabel("time [days]")
	plt.ylabel("radial velocity [m/s]")
	_ = plt.title("posterior constraints")

	plt.show()



	# plot the chains
	parameters = ["P", "ecc", "omega", "incl", "tperi", "m_planet", "a"]
	for ii in range(0, len(parameters)):
		plot_truth = False
		param = parameters[ii]
		
		true_vals_earth = truth_chain_plot[2*ii]
		true_vals_jup = truth_chain_plot[2*ii+1]
		plot_truth = True
		
		fig, ax = plt.subplots(1,2, figsize = (15,3))
		planet1_chain1 = trace.posterior[param].values[:, :, 0][0]
		planet1_chain2 = trace.posterior[param].values[:, :, 0][1]
		
		planet2_chain1 = trace.posterior[param].values[:, :, 1][0]
		planet2_chain2 = trace.posterior[param].values[:, :, 1][1]
		
		
		nstep = np.arange(1, len(planet1_chain1)+1, 1)
		
		
		ax[0].plot(nstep, planet1_chain1)
		ax[0].plot(nstep, planet1_chain2)
		
		if plot_truth:
			ax[0].axhline(y=true_vals_earth, color = 'r', label = 'truth')
		ax[0].set_title("Sun-b", fontsize = 18)
		ax[0].legend(fontsize = 18)
		
		ax[1].plot(nstep, planet2_chain1)
		ax[1].plot(nstep, planet2_chain2)
		
		if plot_truth:
			ax[1].axhline(y=true_vals_jup, color = 'r', label = 'truth')
		ax[1].set_title("Sun-c", fontsize = 18)
		ax[1].legend(fontsize = 18)

		fig.suptitle(param, fontsize = 18)
		fig.tight_layout()
		plt.show()




	return "plotting complete"






