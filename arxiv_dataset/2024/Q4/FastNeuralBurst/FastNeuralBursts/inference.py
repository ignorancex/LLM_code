# author: Koustav Konar <n.koustav.konar@gmail.com>
# Likelihood-free Inference or Simulation-based Inference using Dispersion Measure (DM) of Fast Radio Bursts (FRBs)


# import FRB_utils
# from FRB_dispersion import dispersion_simulation
import FastNeuralBursts.utils as FRB_utils
from FastNeuralBursts.dispersion import dispersion_simulation


import corner
from getdist import plots, MCSamples
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
from sbi.inference import SNLE, SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils import MultipleIndependent
from sbi.utils import get_density_thresholder, RestrictedPrior
from sbi.analysis import pairplot
import sys
import time
import torch
from torch.distributions import Uniform
import warnings



# import ili
# from ili.dataloaders import NumpyLoader
# from ili.inference import InferenceRunner
# from ili.validation.metrics import PosteriorCoverage, PlotSinglePosterior



class sbi_inference:
	"""
	Simulation-based Inference (SBI) using Dispersion Measure (DM) measure of FRBs
	------------
	ATTRIBUTES:
		simulation_instance:   simulation instance to call the simulator method
		file_path:             config file (.ini) location
		variables_dict:        dictionary of the variables from config file
	------------
	METHODS:
		infer(): inference using SBI and returns summary
		infer_ili(): inference using LtU-ILI package
		contour_plotter(): plot contours using GetDist
		prior_truncation(): plots the parameters used in simulations
		log_prob_sbi(): plots the log-likelihood minimisation against the number of simulation
		validation(): compares the training and validation with log-likelihood minimisation
		coverage_plots(): checks and plots the posteriror for univariate and multivariate coverage  

	"""
	def __init__(self, simulation_instance, file_path='config_SBI_DM_FRB.ini', output_path='\Saved_data'):

		
		self.simulation_instance = simulation_instance
		self.file_path = file_path
		self.output_dir_name = output_path
		self.variables_dict = FRB_utils.assign_variable(self.file_path)
		self.pdf_pages = []

		if len(self.variables_dict.keys())==0:
			print('Config file not found, please provide complete file location')

	def contour_plotter(self, samples, package='sbi', file_name='Figures_SBI_FRB.pdf'):
		'''
		Plot marginalised contours from the SBI inference using GetDist 
		https://arxiv.org/abs/1910.13970
		------------
		PARAMETERS:
			samples (ndarray): samples taken from the posterior
			package (string): python package being used = Blue for SBI (default)
													    = Red for iLi
			file_name (string): file name to save the contours
		------------
		RETURNS:
			shows the plot 
		'''

		samples_arr = samples
		a_arr = np.array([samples_arr[i][0] for i in range(len(samples_arr))])
		A = np.mean(a_arr)
		b_arr = np.array([samples_arr[i][1] for i in range(len(samples_arr))])
		B = np.mean(b_arr)
		c_arr = np.array([samples_arr[i][2] for i in range(len(samples_arr))])
		C = np.mean(c_arr)
		# print(f"Cosmo: {A:.2f} \nDM: {B:.2f} \nSigma: {C:.2f}")

		names = ['Cosmo_{scale}', 'DM_{host}', '\sigma']
		labels = ['\mathcal{A}', '\mathrm{DM}_\mathrm{host}', '\sigma']
		get_dist_sample = MCSamples(samples=samples_arr, names=names, labels=labels)
		g = plots.get_subplot_plotter()
		g.settings.figure_legend_frame = False
		g.settings.alpha_filled_add=0.4
		g.settings.title_limit_fontsize = 14
		if package=='ili':
			contour_color = 'red'
		else:
			contour_color = '#2156c0'
		g.triangle_plot(get_dist_sample, ['Cosmo_{scale}', 'DM_{host}', '\sigma'], 
						filled=True,
						legend_labels=[f"Algorithm: {package} \nNSIDE: {self.variables_dict['nside']}", 'Simulation 2'],
						line_args=[{'lw':1, 'color':contour_color},
									{'lw':1, 'color':'#2156c0'}
									], 
						contour_colors=[contour_color,'#2156c0'],
						title_limit=1,
						markers={'Cosmo_{scale}':A, 'DM_{host}':B, '\sigma':C})
		g.get_axes(3,3)
		# plt.savefig(f"{file_name}")
		self.pdf_pages.append(g.fig)
		plt.show()

	

	def prior_truncation(self, simulated_theta, file_name='Figures_SBI_FRB.pdf'):
		'''
		Plots the priors from each simulations. Truncation can be seen for TSNPE
		------------
		PARAMETERS:
			simulated_theta (ndarray): parameters used during simulations, shape: (no. of simulation, no. of free paramters)
			file_name (string): file name to save the contours
		------------
		RETURNS:
			shows the plot 

		'''
		fig, axs = plt.subplots(1,3, figsize=(12,4))
		labels = ['\mathcal{A}', '\mathrm{DM}_\mathrm{host}', '\sigma']
		for i in range(3):
			axs[i].plot(simulated_theta[:,i], alpha=0.8)
			# axs[i].axhline(y=red_line[i].numpy(), color='red', linewidth=1)
			# axs[i].axhline(y=best_fit_val[i].numpy(), color='black', linewidth=1)
			axs[i].set_xlabel('Number of sims', fontsize=14, labelpad=15)
			axs[i].tick_params(axis='x', labelsize=10)
			axs[i].tick_params(axis='y', labelsize=10)
			# axs[i].yaxis.set_major_locator(MaxNLocator(6))
			axs[i].set_ylabel('Value', fontsize=14, labelpad=15)
			axs[i].set_title(f"${labels[i]}$", fontsize=16)
		# fig.suptitle(f'Truncated Priors, Red: Fiducial, Black: Best-fit, NSIDE = {NSIDE}')
		axs[0].set_ylim(-0.1, 3.3)
		axs[1].set_ylim(-50, 1600)
		if self.simulation_instance.host_model == 'lognormal':
			axs[2].set_ylim(self.variables_dict['scale_low']-0.1, 1.1*self.variables_dict['scale_high'])
		else:
			axs[2].set_ylim(self.variables_dict['std_dev_low']-0.1, 1.1*self.variables_dict['std_dev_high'])
		
		plt.tight_layout()
		self.pdf_pages.append(fig)
		# plt.savefig('Truncated_prior_during_sims.png', dpi=300)
		plt.show()



	def log_prob_sbi(self, posterior, simulated_theta, observation, package, file_name='Figures_SBI_FRB.pdf'):
		'''
		Plots the log probability minimisation during inference
		------------
		PARAMETERS:
			posterior (sbi posterior): posterior after inference
			simulated_theta (ndarray): parameters used during simulations, shape: (no. of simulation, no. of free paramters)
			observation (array): observed DM or its compressed version 
			package (string): python package being used 
			file_name (string): file name to save the contours
		------------
		RETURNS:
			shows the plot 

		'''
		fig = plt.figure(figsize=(8, 6))
		if package=='ili':
			device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
			simulated_theta = torch.Tensor(simulated_theta).to(device)
			observation = torch.Tensor([observation]).to(device)
			log_prob = posterior.log_prob(simulated_theta, x=observation, track_gradients=True)
			plt.plot(log_prob.detach().cpu().numpy(), color='#1f77b4')

		else:
			plt.plot(posterior.log_prob(torch.Tensor(simulated_theta), x=observation, track_gradients=True).detach().numpy(), color='#1f77b4')
		plt.xlabel('Number of sims', fontsize=16, labelpad=15)
		plt.ylabel('Log probability', fontsize=16, labelpad=15)
		plt.xticks(fontsize=14)
		plt.yticks(fontsize=14)
		plt.xlim(0)
		plt.ylim(-40, 0)
		if package=='ili':
			plt.title('log-probability of the posterior $p(\\theta|x)$ using ILI')
		else:
			plt.title('log-probability of the posterior $p(\\theta|x)$ using SBI')
		plt.tight_layout()
		self.pdf_pages.append(fig)
		# plt.savefig('Loss_minimisation_SBI_using_mdn.png', dpi=300)
		plt.show()
	
	

	def validation(self, summaries, file_name='Figures_SBI_FRB.pdf'):
		'''
		Plots and compares the log probability minimisation during training and validation
		(works only for iLi as of yet)
		------------
		PARAMETERS:
			summaries (list, dictionary): information of the trainig and validation log minimisation
										  list for MDN and MAF nets
										  dictionary with keys 'training_log_probs' and 'validation_log_probs'
			file_name (string): file name to save the contours
		------------
		RETURNS:
			shows the plot 

		'''
		fig, ax = plt.subplots(1, 1, figsize=(10,8))
		net_list = ['MDN', 'MAF']
		color_list = ['#1f77b4', '#ff7f0e']
		for i, m in enumerate(summaries):
			ax.plot(m['training_log_probs'], ls='-', label=f'{net_list[i]} training', color = color_list[i]) #f"{i}_train", c=color_list[i])
			ax.plot(m['validation_log_probs'], ls='--', label=f'{net_list[i]} validation', color = color_list[i] )#f"{i}_val", c=color_list[i])
			ax.tick_params(axis='x', labelsize=14)
			ax.tick_params(axis='y', labelsize=14)
		ax.set_xlim(0, 250)
		ax.set_ylim(-10)
		ax.set_xlabel('Number of epochs', fontsize = 16, labelpad=15)
		ax.set_ylabel('Log probability', fontsize=16, labelpad=15)

		ax.legend(fontsize=14)
		print(f"\n\nType Fig: {type(fig)}\n\n")
		self.pdf_pages.append(fig)
		# plt.savefig('Loss_minimisation_ILI_using_mdn_and_maf.png', dpi=300)
		plt.show()



	def coverage_plots(self, posterior, sim_data, sim_theta, file_name='Figures_SBI_FRB.pdf', model_list=['ensemble'], test_list=["coverage", "histogram", "predictions","tarp"]):
		'''
		Plots and comares the log probability minimisation during training and validation
		multivariate coverage 'TARP': https://arxiv.org/abs/2302.03026
		------------
		PARAMETERS:
			posterior (sbi posterior): posterior after inference
			sim_data (ndarray): simulated data from simulations, shape: (no. of simulation, simulated or compressed DM)
			sim_theta (ndarray): parameters used during simulations, shape: (no. of simulation, no. of free paramters) 
			file_name (string): file name to save the contours
			model_list (list): model to check the coverage = ['ensemble'] (default)
														   = ['ensemble', 'mdn', 'maf'] (possible)
			test_list (list): types of coverage tests to perform = ["coverage", "histogram", "predictions","tarp"] (default)
		------------
		RETURNS:
			shows the plot 

		'''
		from ili.validation.metrics import PosteriorCoverage, PlotSinglePosterior
		labels = ['\mathcal{A}', '\mathrm{DM}_\mathrm{host}', '\sigma']
		sim_data = torch.Tensor(sim_data).to('cpu')
		sim_theta = torch.Tensor(sim_theta).to('cpu')
		for model in model_list:
			if model=='mdn':
				coverage_posterior = posterior.posteriors[0]
			elif model=='maf':
				coverage_posterior = posterior.posteriors[1]
			else:
				coverage_posterior = posterior
			for fig_type in test_list:
				metric = PosteriorCoverage(
											num_samples=1000, 
											sample_method='direct', labels=[f'${labels[i]}$' for i in range(3)],
											plot_list = fig_type
										)
			
				fig = metric(
							posterior=coverage_posterior,
							x=sim_data, theta=sim_theta
							)
				print(f"\n\nType Fig: {type(fig), type(fig[0])}\n\n")
				self.pdf_pages.append(fig[0])
				plt.show()




	def infer(self, algorithm='TSNPE', num_rounds=20, num_sims_in_round=50, num_walkers=1, num_sample=1000, compression=False, plot_figures=['contour', 'prior', 'likelihood', 'coverage'], 
		save_data=True, verbose=False):
		'''
		Implements Simulation-based Inference using TSNPE and SNPE algorithms and returns summary
		TSNPE: https://arxiv.org/abs/2210.04815
		SNPE: https://arxiv.org/abs/1605.06376
		------------
		PARAMETERS:
			algorithm (string): algorithm to use for inference, either TSNPE (default) or SNPE
			num_rounds (integer): number of rounds to use in TSNPE
			num_sims_in_round (integer): number of simulations in each round (TSNPE)
							total (num_rounds*num_sims_in_round) simulations for SNPE
			num_walkers (integer): number of parallel walkers to use in inference
			num_sample (integer): number of posterior samples to return
			compression (boolean): whether or not compressed statistics is used (works only with lognormal host) = False (default)
			plot_figures (list): list of figures to create = ['contour', 'prior', 'likelihood', 'coverage'] (default)
			save_data (boolean): save simulated data = True (default)
			verbose (boolean): prints observation = False (default)
		------------
		RETURNS: posterior, observation, samples_arr, simulated_data, simulated_theta
			posterior (sbi posterior): posterior from inference
			observation (array): observed DM, compressed or uncompressed
			samples_arr (ndarray): posterior samples with shape (num_sample, num_of_free_parameters)
			simulated_data (ndarray): simulated ( or compressed) DM from the inference steps
			simulated_theta (ndarray): used parameter values ($\theta$) in each step
		------------
		'''
		t0 = time.time()
		if save_data:
			directory = f"{self.output_dir_name}/SBI/{algorithm}"
			print(f"Saving data to: '{directory}' ") if verbose else None
			if not os.path.exists(directory):
				os.makedirs(directory)
			# output_file = f'{directory}/Output_DM_uncompressed_nside_{self.nside}_cosmo_scale_{cosmo_scale}_mean_{dm_host}_scale_{sigma_host}.csv'


		if self.simulation_instance.host_model == 'lognormal':
			prior = MultipleIndependent(
					[
						Uniform(low = self.variables_dict['cosmo_scale_low'] * torch.ones(1),   high = self.variables_dict['cosmo_scale_high'] * torch.ones(1)),
						Uniform(low = self.variables_dict['dm_host_low']     * torch.ones(1),   high = self.variables_dict['dm_host_high']     * torch.ones(1)),
						Uniform(low = self.variables_dict['scale_low']       * torch.ones(1),   high = self.variables_dict['scale_high']       * torch.ones(1)),
					], validate_args=False,)
		else:
			prior = MultipleIndependent(
					[
						Uniform(low = self.variables_dict['cosmo_scale_low'] * torch.ones(1),   high = self.variables_dict['cosmo_scale_high'] * torch.ones(1)),
						Uniform(low = self.variables_dict['dm_host_low']     * torch.ones(1),   high = self.variables_dict['dm_host_high']     * torch.ones(1)),
						Uniform(low = self.variables_dict['std_dev_low']     * torch.ones(1),   high = self.variables_dict['std_dev_high']     * torch.ones(1)),
					], validate_args=False,)

		
		if algorithm=="TSNPE":
			#notify
			print("Algorithm: TSNPE")
		
			if compression and self.simulation_instance.host_model == 'lognormal':
				print(f"{self.simulation_instance.host_model} host, using Compressed DM")
				simulator, prior = prepare_for_sbi(self.simulation_instance.dispersion_measure_compressed, prior)
				observation = self.simulation_instance.dispersion_measure_compressed(torch.Tensor([1.12, 199.98, 1.18]), data_type='f')
				print(f"Observation: {observation}") if verbose else None
				if save_data:
					output_file = f"{directory}/Output_DM_compressed_nside_{self.variables_dict['nside']}_{algorithm}_SBI.csv"

			else:
				print(f"{self.simulation_instance.host_model} host, using Un-compressed DM")
				simulator, prior = prepare_for_sbi(self.simulation_instance.dispersion_measure, prior)
				observation = self.simulation_instance.observed_dm
				print(f"Observation: {observation}") if verbose else None
				if save_data:
					output_file = f"{directory}/Output_DM_un_compressed_nside_{self.variables_dict['nside']}_{algorithm}_SBI.csv"
			
			inference = SNPE(prior=prior)
			proposal = prior

			num_sim_for_tsnpe = num_sims_in_round//num_rounds
			simulated_data = []
			simulated_theta = []
			count=1

			for _ in range(num_rounds):
				print(f"\nRun {count}/{num_rounds}")

				theta, x = simulate_for_sbi(simulator, proposal=proposal, num_simulations=num_sims_in_round, num_workers=num_walkers)
				_ = inference.append_simulations(theta, x).train(force_first_round_loss=True)
				# _ = inference.append_simulations(theta, x)
				# density_estimator = _.train()
				# print(type(_))
				posterior = inference.build_posterior().set_default_x(observation)
				print(f"\nSimulation: {x}") if verbose else None
				if save_data:
					with open(output_file, 'ab') as f:
						[np.savetxt(f, np.array(x[i]).reshape(1, -1), delimiter=',') for i in range(np.shape(x)[0])]
				
				accept_reject_fn = get_density_thresholder(posterior, quantile=1e-4)
				proposal = RestrictedPrior(prior, accept_reject_fn, sample_with='rejection')
				# print(theta)
				
				simulated_theta.append(theta.numpy())
				simulated_data.append(x.numpy())
				if save_data:
					np.save(f"{directory}/Simulated_theta_nside_{self.variables_dict['nside']}_{algorithm}_SBI.csv", simulated_theta)
					np.save(f"{directory}/Simulated_data_nside_{self.variables_dict['nside']}_{algorithm}_SBI.csv", simulated_data)
				elapsed_time = time.time() - t0
				remaining_time = elapsed_time * (num_rounds/count - 1)
				formatted_time = FRB_utils.format_time(elapsed_time)
				remaining_time_formatted = FRB_utils.format_time(remaining_time)
				sys.stdout.write(f"\rElapsed time after {count} run: {formatted_time}, Remaining time: {remaining_time_formatted}")
				sys.stdout.flush()
				count+=1

			# torch.save(posterior, f"Posterior_with_TSNPE_{cosmo_range}_{dm_range}_{sigma_range}_{NSIDE}_nside_{num_sim}sims_{observation}_observation.pt")
			posterior_samples = posterior.sample((num_sample,), x=observation)
			samples_arr = posterior_samples.numpy()
			# num_simulations = self.num_rounds * self.num_sims_in_round
			simulated_data = np.array(simulated_data).reshape(num_sims_in_round*num_rounds, len(observation))
			simulated_theta = np.array(simulated_theta).reshape(num_sims_in_round*num_rounds, theta.shape[1])

			t1 = time.time()
			print(f"\n\nComplete, total time: {(t1-t0):.2f} sec")
			
			if samples_arr.shape[1]==3:
				if compression:
					pdf_filename = f"Figures_SBI_FRB_TSNPE_compressed_data.pdf"
					
				else:
					pdf_filename = f"Figures_SBI_FRB_TSNPE.pdf"

				self.contour_plotter(samples=samples_arr, file_name=f'{pdf_filename}') if 'contour' in plot_figures else None #== ['contour', 'prior', 'likelihood', 'coverage']
				self.prior_truncation(simulated_theta, file_name=f'{pdf_filename}') if 'prior' in plot_figures else None
				self.log_prob_sbi(posterior, simulated_theta, observation, package='sbi', file_name=f'{pdf_filename}') if 'likelihood' in plot_figures else None
				if len(plot_figures) != 0:
					with PdfPages(pdf_filename) as pdf:
						for page in self.pdf_pages:
							pdf.savefig(page, dpi=300)
			else:
				print('Default plotting works with only 3 free parameters, use samples_arr, simulated_data and simulated_theta externally to plot')

			return posterior, observation, samples_arr, simulated_data, simulated_theta


		

		elif algorithm=="SNPE":
			#notify
			print("Algorithm: SNPE")
			simulated_data = []
			simulated_theta = []


			if compression and self.simulation_instance.host_model == 'lognormal':
				print(f"{self.simulation_instance.host_model} host, using Compressed DM")
				simulator, prior = prepare_for_sbi(self.simulation_instance.dispersion_measure_compressed, prior)
				observation = self.simulation_instance.dispersion_measure_compressed(torch.Tensor([1.12, 199.98, 1.18]), data_type='f')
				print(f"Observation: {observation}") if verbose else None
				if save_data:
					output_file = f"{directory}/Output_DM_compressed_nside_{self.variables_dict['nside']}_{algorithm}_SBI.csv"

			else:
				print(f"{self.simulation_instance.host_model} host, using Un-compressed DM")
				simulator, prior = prepare_for_sbi(self.simulation_instance.dispersion_measure, prior)
				observation = self.simulation_instance.observed_dm
				print(f"Observation: {observation}") if verbose else None
				if save_data:
					output_file = f"{directory}/Output_DM_un_compressed_nside_{self.variables_dict['nside']}_{algorithm}_SBI.csv"

			inference = SNPE(prior=prior)
			num_sim = num_sims_in_round * num_rounds
			theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=num_sim)
			print(f"\nSimulation: {x}") if verbose else None
			if save_data==True:
					with open(output_file, 'ab') as f:
						[np.savetxt(f, np.array(x[i]).reshape(1, -1), delimiter=',') for i in range(np.shape(x)[0])]
				
			inference = inference.append_simulations(theta, x)
			density_estimator = inference.train()
			posterior = inference.build_posterior(density_estimator)

			posterior_samples = posterior.sample((num_sample,), x = observation)
			samples_arr = posterior_samples.numpy()

			simulated_theta.append(theta.numpy())
			simulated_data.append(x.numpy())
			simulated_data = np.array(simulated_data).reshape(num_sim, len(observation))
			simulated_theta = np.array(simulated_theta).reshape(num_sim, theta.shape[1])
			if save_data:
					np.save(f"{directory}/Simulated_theta_nside_{self.variables_dict['nside']}_{algorithm}_SBI.csv", simulated_theta)
					np.save(f"{directory}/Simulated_data_nside_{self.variables_dict['nside']}_{algorithm}_SBI.csv", simulated_data)

			
			# self.sim_data = simulated_data
			# self.sim_theta = simulated_theta

			
			if samples_arr.shape[1]==3:
				if compression:
					pdf_filename = f"Figures_SBI_FRB_SNPE_compressed_data.pdf"
				else:
					pdf_filename = f"Figures_SBI_FRB_SNPE.pdf"
				self.contour_plotter(samples=samples_arr, file_name=f'{pdf_filename}') if 'contour' in plot_figures else None
				self.prior_truncation(simulated_theta, file_name=f'{pdf_filename}') if 'prior' in plot_figures else None
				self.log_prob_sbi(posterior, simulated_theta, observation, package='sbi', file_name=f'{pdf_filename}') if 'likelihood' in plot_figures else None
				if len(plot_figures) != 0:
					with PdfPages(pdf_filename) as pdf:
						for page in self.pdf_pages:
							pdf.savefig(page, dpi=300)
			else:
				print('Default plotting works with only 3 free parameters, use samples_arr, simulated_data and simulated_theta externally to plot')
			t1 = time.time()
			print(f"\n\nComplete, total time: {(t1-t0):.2f} sec") #if verbose else None


			return posterior, observation, samples_arr, simulated_data, simulated_theta

		else:
			print(f"Algorithm '{algorithm}' is not implemented, use either 'TSNPE' or 'SNPE' ")
			return None


	










	def infer_ili(self, algorithm='TSNPE', num_rounds=20, num_sims_in_round=50, num_walkers=1, num_sample=1000, 
		mdn_hidden=500, mdn_components=200, maf_hidden=200, maf_tarnsform=50, batch_size=32, training_rate=5e-4,
	 compression=False, plot_figures=['contour', 'prior', 'likelihood', 'coverage_test'], model_for_coverage=['ensemble'], 
	 coverage_list=["coverage", "histogram", "predictions","tarp"], verbose=False, save_data=False):
		'''
		Implements Simulation-based Inference using LtU-ILI package
		https://arxiv.org/abs/2402.05137
		------------
		PARAMETERS:
			algorithm (string): algorithm to use for inference, either TSNPE (default) or SNPE
			num_rounds (integer): number of rounds to use in TSNPE
			num_sims_in_round (integer): number of simulations in each round (TSNPE)
                                         total (num_rounds*num_sims_in_round) simulations for SNPE
			num_walkers (integer): number of parallel walkers to use in inference
			num_sample (integer): number of posterior samples to return
			mdn_hidden (integer): number of hidden layers in MDN net = 500 (default)
			mdn_components (integer): number of components in MDN net = 200 (default)
			maf_hidden (integer): number of hidden layers in MAF net = 200 (default) 
			maf_tarnsform (integer): number of transformations in MAF net = 50 (default) 
			batch_size (integer): batch size of the data to use during training = 32 (default) 
			training_rate (float): training rate for the neural network = 5e-4 (default)
			compression (boolean): whether or not compressed statistics is used (works only with lognormal host) = False (default)
			save_data (boolean): save simulated data = False (default)
			verbose (boolean): prints observation = False (default)
		------------
		RETURNS: posterior_ili, observation, samples_arr_ili, summaries
			posterior_ili (sbi posterior): posterior from inference
			observation (array): observed DM, compressed or uncompressed
			samples_arr (ndarray): posterior samples with shape (num_sample, num_of_free_parameters)
			summaries (list): MDN and MAF's performance data
		------------
		'''
		
		print('Simulating Data')
		posterior, observation, samples_arr, simulated_data_ili, simulated_theta_ili = self.infer(algorithm=algorithm,
																							num_sample=num_sample, 
																							num_rounds=num_rounds, 
																							num_sims_in_round=num_sims_in_round,
																							num_walkers=num_walkers,
																							compression=compression,
																							plot_figures=[],
																							save_data=False,
																							verbose=False)
		if save_data:
			directory = f"{self.output_dir_name}/ILI/{algorithm}"
			print(f"Saving data to: '{directory}' ") if verbose else None
			if not os.path.exists(directory):
				os.makedirs(directory)
			np.save(f"{directory}/Simulated_theta_nside_{self.variables_dict['nside']}_{algorithm}_SBI_ili.csv", simulated_theta_ili)
			np.save(f"{directory}/Simulated_data_nside_{self.variables_dict['nside']}_{algorithm}_SBI_ili.csv", simulated_data_ili)
		import ili
		from ili.dataloaders import NumpyLoader
		from ili.inference import InferenceRunner
		from ili.validation.metrics import PosteriorCoverage, PlotSinglePosterior
		device = 'cuda' if torch.cuda.is_available() else 'cpu'
		# device = 'cpu'
		print('Device:', device)
		loader = NumpyLoader(x=simulated_data_ili, theta=simulated_theta_ili, xobs=observation)
		if self.simulation_instance.host_model == 'lognormal':
			prior = ili.utils.Uniform(
				low =  [self.variables_dict['cosmo_scale_low'],  self.variables_dict['dm_host_low'],  self.variables_dict['scale_low']], 
				high = [self.variables_dict['cosmo_scale_high'], self.variables_dict['dm_host_high'], self.variables_dict['scale_high']], 
				device=device)
		else:
			prior = ili.utils.Uniform(
				low =  [self.variables_dict['cosmo_scale_low'],  self.variables_dict['dm_host_low'],  self.variables_dict['std_dev_low']], 
				high = [self.variables_dict['cosmo_scale_high'], self.variables_dict['dm_host_high'], self.variables_dict['std_dev_high']], 
				device=device)

		nets = [
				ili.utils.load_nde_sbi(engine='NPE', model='mdn', hidden_features=mdn_hidden, num_components=mdn_components),
				ili.utils.load_nde_sbi(engine='NPE', model='maf', hidden_features=maf_hidden, num_transforms=maf_tarnsform)
				]

		train_args = {
				'training_batch_size': int(batch_size), 
				'learning_rate': training_rate
				}
		runner = InferenceRunner.load(
				backend='sbi',
				engine='NPE',
				prior=prior,
				nets=nets,
				device=device, # enable this for CUDA computation
				embedding_net=None,
				train_args=train_args,
				proposal=None,
				out_dir=None
				)
		t0 = time.time()
		posterior_ili, summaries = runner(loader=loader)
		
		if device=='cuda':
			samples_arr_ili = posterior.sample((num_sample,), x=observation).numpy()
		samples_arr_ili = posterior.sample((num_sample,), x=observation).numpy()
		
		if samples_arr.shape[1]==3:
			if compression:
				pdf_filename = f"Figures_SBI_FRB_ili_{algorithm}_compressed_data.pdf"
			else:
				pdf_filename = f"Figures_SBI_FRB_ili_{algorithm}.pdf"

			self.contour_plotter(samples=samples_arr_ili, package='ili', file_name=f'{pdf_filename}') if 'contour' in plot_figures else None #== ['contour', 'prior', 'likelihood', 'coverage']
			self.prior_truncation(simulated_theta_ili, file_name=f'{pdf_filename}') if 'prior' in plot_figures else None
			self.validation(summaries, file_name=f'{pdf_filename}') if 'validation' in plot_figures else None
			self.log_prob_sbi(posterior_ili, simulated_theta_ili, observation, package='ili', file_name=f'{pdf_filename}') if 'likelihood' in plot_figures else None
			if 'coverage_test' in plot_figures:
				self.coverage_plots(posterior_ili, simulated_data_ili, simulated_theta_ili, file_name=f'{pdf_filename}', 
					model_list=model_for_coverage, 
					test_list=coverage_list
					)
			if len(plot_figures) != 0:
				with PdfPages(pdf_filename) as pdf:
					for page in self.pdf_pages:
						pdf.savefig(page, dpi=300)
			

		t1 = time.time()
		print(f'\n\nComplete, total time: {t1-t0:.3f} sec')

		return posterior_ili, observation, samples_arr_ili, simulated_data_ili, simulated_theta_ili, summaries
