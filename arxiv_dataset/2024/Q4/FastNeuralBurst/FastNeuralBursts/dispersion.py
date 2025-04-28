# author: Koustav Konar <n.koustav.konar@gmail.com>
# Dispersion Measure (DM) simulation for FRBs


import FastNeuralBursts.host as FRB_host 
import FastNeuralBursts.utils as FRB_utils



import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
import glass.shells
import glass.fields
import glass.ext.camb
import healpy as hp
import numpy as np
import os
import scipy.constants as const
import sys
import time
import torch
from tqdm.auto import trange, tqdm




class dispersion_simulation:
	"""
	Class for simulating Dispersion Measure (DM) of Fast Radio Bursts (FRBs)
	------------
	ATTRIBUTES:
		z_edges_for_glass:  redshift coundaries of shells
		path:               input file directory
		output_dir_name:    output directory 
		omega_c:            cold dark matter density
		omega_b:            baryon density
		h:                  hubble parameter
		neff:               effective number of relativistic number of freedom
		t_cmb:              temperature of CMB today
		m_nu:               sum of neutrino masses
		chi_e:              electron fraction
		f_igm:              electron fraction in IGM
		cls_levin:          angular power spectra for shells
		nside:              resolution parameter for spherical shells (strictly 2^N)
		host_model:         model distribution for DM contribution from FRB host galaxy
		observed_dm:        observed or measured DM	
	------------
	METHODS:
		dispersion_measure():            calculates and returns DM of FRBs
		dispersion_measure_compressed(): calculates and returns compressed statistics for DM of FRBs
	"""
	def __init__(self, z_edges_for_glass, input_path, output_path, cls, omega_c=0.25,omega_b=0.05,h=0.67,t_cmb=2.725, neff=3.0446, m_nu=0.05, chi_e=0.875, f_igm = 0.85, 
		NSIDE=128, host_model='lognormal', verbose=False):
		self.z_edges_for_glass = z_edges_for_glass
		self.omega_c = omega_c
		self.omega_b = omega_b
		self.h = h
		self.neff = neff
		self.t_cmb = t_cmb
		self.m_nu = m_nu
		self.chi_e = chi_e
		self.f_igm = f_igm
		self.path = input_path
		self.output_dir_name = output_path
		self.cls_levin = cls
		self.nside = NSIDE
		self.host_model = host_model
		self.verbose = verbose
		self.observed_dm = FRB_utils.read_catalogue(self.path + "catalogue_2.csv")[1]

		self.c = const.c
		self.G = const.gravitational_constant
		self.m_p = const.proton_mass
		self.H_0_pc = 100*self.h*u.km.to(u.m)/u.Mpc.to(u.m)
		self.ws = glass.shells.tophat_windows(self.z_edges_for_glass)



	def dispersion_measure(self, pars, num_for_mean = 1, save_data=False, verbose2=False):
		'''
		Calculates and returns the DM of FRBs with host contribution 
		------------
		PARAMETERS:
			pars (torch Tensor): free parameters (scale for cosmology, host distribution)
					'lognormal': median and scale
					'normal': mean and standard deviation
			num_for_mean (int): number of simulation = 1 (default)
			save_data (boolean): save simulated data = False (default)
			verbose2 (boolean): print each shell progress = False (default)
		------------
		RETURNS:
		if num_for_mean = 1: simulated_dm
			simulated_dm (array): simulated DM
		
		if num_for_mean != 1: dm_mean, dm_lss_mean, covar_full, covar_lss, corr, dm_host_mean, partial_MU, partial_sigma, dm_lss_sum, dm_lss_unity
			dm_mean (array): mean DM for all sims 
			dm_lss_mean (array): mean DM from only LSS
			covar_full (ndarray): covariance matrix for mean DM 
			covar_lss (ndarray): covariance matrix for LSS DM 
			corr (ndarray): correlation matrix for mean DM 
			dm_host_mean (array): mean host DM for all sims 
			partial_MU (array): partial derivative wrt the free parameter 1
			partial_sigma (array): partial derivative wrt the free parameter 2
			dm_lss_sum (ndarray): DM contribution from the sum of shells (2nd term in eq.)
			dm_lss_unity (array): Standard DM-z relation, monotonic (1st term in eq.)
		------------
		'''
		t0 = time.time()
		# dissolving the input to free parameters
		cosmo_scale, dm_host, sigma_host = pars
		# setting lmax as C_\ell's are calculated up to \ell = 15002
		if self.nside & (self.nside - 1) != 0:
			print(f"\n\nERROR: {self.nside} has no associated Healpy window, strictly use NSIDE = 2^N")
		if self.nside > 4096:
			lmax = 15002
		else:
			lmax = 3*self.nside-1
		# prefactor for the DM
		prefactor = cosmo_scale * ((3 * self.c * self.H_0_pc * self.omega_b * self.chi_e * self.f_igm) / (8 * np.pi * self.G * self.m_p)) * (u.m.to(u.pc)/(u.m.to(u.cm)**3))
		# print(f'nside: {self.nside}, lmax: {lmax}, pars: {pars}, A: {prefactor}, host: {self.host_model}, num: {num_for_mean}')
		if save_data:
			directory = f'{self.output_dir_name}/{self.host_model if self.host_model == "lognormal" else "normal"}'
			print(f"Saving data to: '{directory}' ") if verbose2 else None
			if not os.path.exists(directory):
				os.makedirs(directory)
			output_file = f'{directory}/Output_DM_uncompressed_nside_{self.nside}_cosmo_scale_{cosmo_scale}_mean_{dm_host}_scale_{sigma_host}.csv'


		# Shell description for weights
		za = np.array([self.ws[i][0] for i in range(len(self.ws))], dtype=object)
		wa = np.array([self.ws[i][1] for i in range(len(self.ws))], dtype=object)
		zeff = np.array([self.ws[i][2] for i in range(len(self.ws))], dtype=object)
		shell_weight = np.array([np.trapz(wa[i], za[i])/np.interp(zeff[i], za[i], wa[i]) for i in range(len(self.ws))])
		flat_lambda = FlatLambdaCDM(H0=100*self.h, Ob0=self.omega_b, Om0=self.omega_c+self.omega_b, Tcmb0=self.t_cmb, Neff=self.neff, m_nu=self.m_nu/3)
		one_plus_z_over_ez = np.array([(1+zeff[i])/flat_lambda.efunc(zeff[i]) for i in range(len(zeff))])

		# initialise lists for outputs	
		dm_for_mean = []
		dm_lss_mean = []
		dm_lss_covar = []
		dm_lognormal_average = []
		partial_MU = [] 
		partial_sigma = []
		dm_lss_covar_all = []
		dm_lss_unity = []

		# read the catalogue and assigning the variables
		z_frb, dm_from_catalogue, dm_milky_way, frb_name, frb_name_label, phi_radians, theta_radians = FRB_utils.read_catalogue(self.path + "catalogue_2.csv")

		# Shells' info based on position and redshifts of FRBs 
		shell_num, redshifts, weight, theta, phi, pixel_position = FRB_utils.frb_in_bin(self.z_edges_for_glass, z_frb, phi_radians, theta_radians, self.nside)

		# check for precomputed Gaussian C_\ell or gls
		gls_file_path = self.path + f'gls_{self.nside}.npy'
		if os.path.exists(gls_file_path):
			gls = np.load(gls_file_path, allow_pickle=True)
			if self.verbose==True:
				print(f"NSIDE={self.nside}\nGls loaded successfully")
		else:
			if self.verbose==True:
				print(f"NSIDE={self.nside}\nStored Gls not found, calculating Gls")
			gls = glass.fields.lognormal_gls(self.cls_levin, nside=self.nside, ncorr=1, lmax=lmax)
			np.save(gls_file_path, gls)
			if self.verbose==True:
				print(f'Gls calculated and stored at: {gls_file_path}')

		# main loop for DM computation
		for _ in trange(num_for_mean, desc='Simulation counter', disable=(num_for_mean <= 1)):
			elapsed_time = time.time() - t0
			remaining_time = elapsed_time * (num_for_mean/(_+1) - 1)
			if num_for_mean != 1:
				formatted_time = FRB_utils.format_time(elapsed_time)
				remaining_time_formatted = FRB_utils.format_time(remaining_time)
				# sys.stdout.write(f"\rElapsed time after {_+1} run: {elapsed_time:.2f}, Remaining time: {remaining_time:.2f}")	
				sys.stdout.write(f"\rElapsed time after {_+1} run: {formatted_time}, Remaining time: {remaining_time_formatted}")
				sys.stdout.flush()


			# initialise DM with zeros 
			dm_shell = np.zeros(len(frb_name))
			# lognormal realisation of electron distribution on shells
			matter = glass.fields.generate_lognormal(gls, ncorr=1, nside=self.nside)
			for i, delta_i in enumerate(matter):
				dm_for_i_shell = []
				shell_matter = delta_i
				try:
					X = shell_num[i-1]
					if verbose2==True:
						print(f'\nRun: {_+1}/{num_for_mean}, Shell num: {i}')
					for j in range(len(z_frb)):
						# calculates DM based on the redshift of FRB and the boundaries of shells
						n = shell_num[j]
						if i < n:
							dm_for_frbs = hp.get_interp_val(shell_matter, theta_radians[j], phi_radians[j])*prefactor.numpy()
							dm_for_frbs *= shell_weight[i]
							dm_for_frbs *= one_plus_z_over_ez[i]
							dm_for_i_shell.append(float(dm_for_frbs))
							if verbose2==True:
								print(f"z = {z_frb[j]:<8}, {frb_name[j]:<15} in shell {n:<2} weight for shell {i:<1} = {1:<5.0f}," 
										f"DM = {float(dm_for_frbs):.3f}")
						if i == n:
							dm_for_frbs = hp.get_interp_val(weight[j]*shell_matter, theta_radians[j], phi_radians[j])*prefactor.numpy()
							dm_for_frbs *= shell_weight[i]
							dm_for_frbs *= one_plus_z_over_ez[i]
							dm_for_i_shell.append(float(dm_for_frbs))
							if verbose2==True:
								print(f"z = {z_frb[j]:<8}, {frb_name[j]:<15} in shell {n:<2} weight for shell {i:<1} = {weight[j]:<5.2f}," 
										f"DM = {float(dm_for_frbs):.3f}")

						if i > n:
							dm_for_i_shell.append(0.)
							if verbose2==True:
								print(f"z = {z_frb[j]:<8}, {frb_name[j]:<15} in shell {n:<2} weight for shell {i:<1} = {0:<5.0f}, DM = 0.")

					dm_for_i_shell = np.array(dm_for_i_shell).ravel()
					dm_shell += dm_for_i_shell
					dm_lss_covar_all.append(dm_for_i_shell)
					if verbose2==True:
						print(f"\nDM after shell {i}: {dm_shell}")

				except IndexError:
					break
			# adding DM contribution from the host based on lognormal or truncated Gaussian model
			if self.host_model=='lognormal':
				print(f"Host model: {self.host_model}") if _ == 0 and self.verbose else None
				dm_lognormal, del_mu_nl_MU, del_sigma_nl_MU = FRB_host.lognormal_host_dm(z_frb, dm_host, sigma_host)
			else:
				print(f"{self.host_model} model not found, using Truncated Gaussian") if _ == 0 and self.verbose else None
				dm_lognormal, del_mu_nl_MU, del_sigma_nl_MU = FRB_host.gaussian_host_dm(z_frb, dm_host, sigma_host)

			dm_lognormal_average.append(dm_lognormal)
			partial_MU.append(del_mu_nl_MU)
			# partial_sigma.append(del_sigma_nl_MU.numpy())
			partial_sigma.append(del_sigma_nl_MU)
			save_list = []

			# adding the DM contribution from the LSS
			for i in range(len(z_frb)):
				n = shell_num[i]
				zet_int = np.linspace(0,z_frb[i],100)
				dm_background = np.trapz((1+zet_int)/flat_lambda.efunc(zet_int)*prefactor.numpy(),zet_int)
				dm_lss_unity.append(dm_background)
				dm_lss = dm_background + dm_shell[i]
				
				dm_total = dm_lss + dm_lognormal[i] + dm_milky_way[i]
				dm_for_mean.append(dm_total)
				dm_lss_mean.append(dm_lss)
				dm_lss_covar.append(dm_lss)
				if save_data==True:
					save_list.append(dm_total)
			if save_data==True:
				with open(output_file, 'ab') as f:
					np.savetxt(f, np.array(save_list).reshape(1, -1), delimiter=',')
				del save_list


		dm_lss_mean = np.array(dm_lss_mean)
		dm_for_mean = np.array(dm_for_mean)
		dm_lss_covar = np.array(dm_lss_covar)
		dm_lognormal_average = np.array(dm_lognormal_average)
		dm_lognormal_average = np.array([np.average(dm_lognormal_average[:,i]) for i in range(len(frb_name))])
		dm_lss_covar_all = np.array(dm_lss_covar_all)
		dm_lss_unity = np.array(dm_lss_unity)
		dm_lss_unity = np.unique(dm_lss_unity)
		
		if self.host_model == 'lognormal':
			partial_MU = np.array(partial_MU, dtype='object')
			partial_MU = np.array([np.average(partial_MU[:,i]) for i in range(len(frb_name))])
		else:
			partial_MU = 1
		
		if self.host_model == 'lognormal':
			partial_sigma = np.array(partial_sigma)
			partial_sigma = np.array([np.average(partial_sigma[:,i]) for i in range(len(frb_name))])
		else:
			partial_sigma = 2

		def covariance_multiple_run():
			dm_cov = dm_for_mean.reshape(num_for_mean, len(frb_name))
			dm_cov_lss = dm_lss_mean.reshape(num_for_mean, len(frb_name))
			
			dm_lss_lst = [np.average(dm_lss_mean.reshape(num_for_mean, len(frb_name))[:,i]) for i in range(len(frb_name))]
			dm_lss_arr = np.array(dm_lss_lst)
			
			mat = np.array([dm_cov.reshape(num_for_mean,len(frb_name))[:,i] for i in range(len(frb_name))])
			mat_lss = np.array([dm_cov_lss.reshape(num_for_mean,len(frb_name))[:,i] for i in range(len(frb_name))])
			
			dm_average = np.array([np.average(mat[i]) for i in range(len(mat))])
			covar_full = np.cov(mat)
			corr = np.corrcoef(mat)
			covar_lss = np.cov(mat_lss)
			
			t1=time.time()
			t_total = t1 - t0
			print(f"\nTime for {num_for_mean} sims: {t_total:.2f} sec, {t_total/num_for_mean:.2f} sec/sim")



			if save_data:
				data_dict = {
					'dm_mean': dm_average,
					'dm_lss_mean': dm_lss_arr,
					'cov_total': covar_full,
					'cov_lss': covar_lss,
					'corr': corr,
					'dm_host_average': dm_lognormal_average,
					'partial_MU': partial_MU,
					'partial_sigma': partial_sigma
				}
				np.savez(f'{directory}/covariance_with_Gaussian_host_nside_{self.nside}_cosmo_scale_{cosmo_scale}_mean_{dm_host}_scale_{sigma_host}.npz', **data_dict)
			return dm_average, dm_lss_arr, covar_full, covar_lss, corr, dm_lognormal_average, partial_MU, partial_sigma, dm_lss_covar_all, dm_lss_unity

		if num_for_mean==1:
			# print(dm_for_mean)
			return dm_for_mean #dm_total 
		else:
			return covariance_multiple_run()



	




	def dispersion_measure_compressed(self, pars, cov_file = 'covariance_results.npz', data_type="simulation", verbose3=False):
		'''
		Calculates compressed DM. Takes simulated or observed data (d) and compresses it down to no. of free parameters
		Based on Eq.(3) of https://arxiv.org/pdf/1801.01497.pdf
		------------
		PARAMETERS:
			pars (torch tensor): fiducial point for compression
			cov_file (string): covariance file name (.npz file)
			data_type (string): input data type to compress
					'simulation' (default): compress simulated DM 
					anything else: compress observed DM
			verbose3 (boolean): prints outputs = False (default)
		------------
		RETURNS:
			compressed_data (array): compressed data of length equal to free parameters	
		------------	
		'''
		cosmo_scale, dm_host, sigma_host = pars
		
		z_frb, dm_from_catalogue, dm_milky_way, frb_name, frb_name_label, phi_radians, theta_radians = FRB_utils.read_catalogue(self.path + "catalogue_2.csv")
		covariance_results = np.load(self.path+cov_file)
		dm_mean = covariance_results['dm_mean']
		dm_lss_mean = covariance_results['dm_lss_mean']
		cov_total = covariance_results['cov_total']
		cov_lss = covariance_results['cov_lss']
		dm_lognormal_average = covariance_results['dm_lognormal_average']
		partial_MU = covariance_results['partial_MU']
		# partial_MU = np.exp(0.5 * sigma_host**2) / (1+z_frb) 
		partial_sigma = covariance_results['partial_sigma']
		# partial_sigma = sigma_host * partial_MU
		
		mean_of_lognormal = np.log(dm_host.cpu())
		if data_type == "simulation":
			d = self.dispersion_measure(pars)
		else:
			d = dm_from_catalogue
		mu_star = dm_mean    # average of many realisation
		cov_star = cov_total    # covariance of many realisation
		covar_inv = np.linalg.inv(cov_star)
		
		nabla_mu = 3*[1]
		nabla_mu[0] = dm_lss_mean
		nabla_mu[1] = partial_MU #.to('cpu').numpy()
		nabla_mu[2] = partial_sigma #.to('cpu').numpy()
		nabla_mu = np.array(nabla_mu,dtype='object')
		if verbose3:
			print(f"\nNabla_shape: {np.shape(nabla_mu)} \nNabla:{nabla_mu}")
		
		nabla_cov = 3*[1]
		nabla_cov[0] = 2*cosmo_scale * cov_lss
		nabla_cov[0] = nabla_cov[0].numpy()
		
		del_mu_cov = 2*dm_host*(np.exp(sigma_host**2) - 1)/(1+z_frb)**2
		nabla_cov[1] = np.diag(del_mu_cov)
		
		del_sigma_cov = 2*dm_host**2*sigma_host*np.exp(sigma_host**2)/(1+z_frb)**2
		nabla_cov[2] = np.diag(del_sigma_cov)
		
		nabla_cov = np.array(nabla_cov)
		t = []
		for i in range(3):
			t.append(nabla_mu[i].T @ covar_inv @ (d - mu_star))
			covariance_term = 0.5 * (d - mu_star).T @ covar_inv @ nabla_cov[i] @ covar_inv @ (d - mu_star)
			t[i] += covariance_term
			# t_cat[i] += covariance_term
		t = np.array(t, dtype='object')
		if verbose3:
			print(f"\nt: {t} \nt_shape: {np.shape(t)}")
		observation_sim = []
		for i in range(len(t)):
			observation_sim.append(float(np.unique(t[i])))
		observation_sim = np.array(observation_sim)    
		if verbose3:
			print(f"\nSimulation: {observation_sim}")
		return observation_sim