# author: Koustav Konar <n.koustav.konar@gmail.com>


from astropy.cosmology import FlatLambdaCDM
import levinpower
import numpy as np
import pyhmcode
import pyhmcode.halo_profile_utils
import pyccl
from scipy.interpolate import UnivariateSpline
import time





class spectra:
	"""
	Class for generating 3-D and angular power spectra
	------------
	ATTRIBUTES:
		z_range:    redshift for power spectra, e.g. np.geomspace(0.0001, 3.5, 50)
		k_range:    wavenumber for power spectra, e.g. np.geomspace(1e-05, 100.0, 200)
		omega_c:    cold dark matter density
		omega_b:    baryon density
		h:          hubble parameter
		n_s:        primordial spectral index
		sigma8:     matter clustering amplitude
		w0:         constant equation of state parameter for the Dark Energy component
		wa:         Linear variation (with a) of the Dark Energy equation of state
		neff:       effective number of relativistic number of freedom
		log10theta: AGN temperature for feedback
		t_cmb:      temperature of CMB today
		m_nu:       sum of neutrino masses
	------------
	METHODS:
		hm_spectra(): calculates 3-D matter/electron spectra with halo model
		levin_code(): calculates angular power spectra without Limber approximation

	"""
	def __init__(self, z_range, k_range, omega_c=0.25,omega_b=0.05,h=0.67,n_s=0.963,sigma8=0.834,w0=-1., wa=0., neff=3.0446, log10theta=7.8, t_cmb=2.725, m_nu=0.05):
		self.z_range = z_range
		self.k_range = k_range
		self.omega_c = omega_c
		self.omega_b = omega_b
		self.h = h
		self.n_s = n_s
		self.sigma8 = sigma8
		self.w0 = w0
		self.wa = wa
		self.neff = neff
		self.log10theta = log10theta
		self.t_cmb = t_cmb
		self.m_nu = m_nu

	def hm_spectra(self, verbose=False):
		'''
		Calculates non-linear spectra from linear power spectra using HM code
		CCL: https://arxiv.org/abs/1812.05995 (linear spectra)
		HM code: https://arxiv.org/abs/2109.04458 (non-linear spectra)
		------------
		RETURNS: 
			P_ee_all, P_mm_all, P_me_all: elctron-electron, matter-matter and electron-matter spectra all multiplied with $h^3$
		------------
		'''
		#notify
		if verbose==True:
			print('HM code started')
		t0_hm = time.time()
		ccl_cosmology = pyccl.Cosmology(Omega_c=self.omega_c, Omega_b=self.omega_b, h=self.h, n_s=self.n_s, sigma8=self.sigma8, w0=self.w0, wa=self.wa, Neff=self.neff)

		power = np.zeros((len(self.z_range), len(self.k_range)))
		power_matter = np.zeros((len(self.z_range), len(self.k_range)))
	        
		for z_idx, z_val in enumerate(self.z_range):
			power[z_idx, :] = ccl_cosmology.linear_matter_power(self.k_range*self.h,1.0/(1+z_val))*self.h**3
			power_matter[z_idx, :] = power[z_idx, :]/self.k_range**4
	    
		#notify
		if verbose:
			print('Defining Non linear power spectra')
		hmcode_cosmology = pyhmcode.halo_profile_utils.ccl2hmcode_cosmo(
								ccl_cosmo=ccl_cosmology,
								pofk_lin_k_h=self.k_range,
								pofk_lin_z=self.z_range,
								pofk_lin=power,
								log10_T_heat=self.log10theta)
		hmcode_model = pyhmcode.Halomodel(
						pyhmcode.HMx2020_matter_pressure_w_temp_scaling)
		hmcode_pofk = pyhmcode.calculate_nonlinear_power_spectrum(
									cosmology=hmcode_cosmology,
									halomodel=hmcode_model, 
									fields=[pyhmcode.field_matter,
											pyhmcode.field_gas])
		bias_sq = np.copy(hmcode_pofk[1, 1]/hmcode_pofk[0, 0]) # defining the electron bias
		bias_sq /= bias_sq[:,0][:,None]  # Normalising the bias
		k = np.copy(self.k_range)
		k_at_minus_2 = k[np.logical_and(k>1e-2, k<2e-2)].min() 
		# assigning all the values for k<1e-2 to the value at k=1e-2
		bias_sq_met1 = np.copy(bias_sq) #bias for method 1
		bias_sq_met1[:, k<1e-2] = bias_sq_met1[:, np.where(k==k_at_minus_2)].reshape(len(self.z_range),1)
		
		#normalise
		bias_sq_norm_met1 = bias_sq_met1/bias_sq_met1[:,0][:,None]
		
		P_mm_all = hmcode_pofk[0, 0,::]
		P_ee_all = P_mm_all * bias_sq_norm_met1
		P_me_all = P_mm_all * np.sqrt(bias_sq_norm_met1)
		t1_hm = time.time()
		if verbose:
			print(f'HM code finished, time: {(t1_hm - t0_hm):.2f} sec')
		return P_ee_all, P_mm_all, P_me_all

	










	def levin_code(self, ell, P_ee_all, z_edges_for_glass_shell, ell_limber=30000, ell_nonlimber=1800, 
					max_number_subintervals=25, minell=1, maxell=20000, n_nonlimber=200, n_limber=100, ninterp=800, verbose=False):
		'''
		Calculates Angular power spectra for use in GLASS shells without Limber approximation
		https://arxiv.org/abs/2404.15402
		------------
		PARAMETERS:
			ell (array): multipole ($\ell$) array for angular power spectra ($C_\ell$)
			P_ee_all (ndarray): 3-D non-linear or linear power spectra for Matter or Electron
			z_edges_for_glass_shell (array): redshift boundaries of spherical shells
		------------
		RETURNS:
			cls (ndarray): Angular power spectra or $C_\ell$ following the GLASS ordering scheme (https://glass.readthedocs.io/stable/reference/fields.html)
		------------
		'''
		flat_lambda = FlatLambdaCDM(H0=100*self.h, Ob0=self.omega_b, Om0=self.omega_c+self.omega_b, Tcmb0=self.t_cmb, Neff=self.neff, m_nu=self.m_nu/3)
		levin_background_z = np.linspace(0, 100, 1024)
		levin_background_chi = flat_lambda.comoving_distance((levin_background_z)) # Mpc
		levin_background_Ez = flat_lambda.efunc(levin_background_z)
		levin_kernel_z_cl = np.linspace(0, 3.5, 2000)
		levin_kernel_chi_cl = flat_lambda.comoving_distance(levin_kernel_z_cl)

		# converting to integers as a requisite for the 'lp.set_parameters()' method
		ell_limber = int(ell_limber)
		ell_nonlimber = int(ell_nonlimber)
		max_number_subintervals = int(max_number_subintervals)
		minell = int(minell)
		maxell = int(maxell)
		n_nonlimber = int(n_nonlimber)
		n_limber = int(n_limber)
		ninterp = int(ninterp)


		t0_levin = time.time()
		
		k_pk = self.k_range*self.h 
		z_pk = self.z_range
		 
		#Set the power_spectrum to the desired tracer
		power_spectrum = (P_ee_all/self.h**3).flatten()

		backgound_z = levin_background_z
		background_chi = levin_background_chi.value 
		chi_kernels = levin_kernel_chi_cl.value  

		# Prepare redshift distribution input
		z_edges = z_edges_for_glass_shell
		z_of_chi = UnivariateSpline(background_chi, backgound_z, s=0, ext=1)
		dz_dchi = z_of_chi.derivative()
		nbins = len(z_edges) - 1
		new_kernel = np.zeros((nbins, len(chi_kernels)))
		for i in range(nbins):
			for j in range(len(chi_kernels)):
				if(z_of_chi(chi_kernels[j]) > z_edges[i] and z_of_chi(chi_kernels[j]) < z_edges[i+1]):
					new_kernel[i, j] = dz_dchi(chi_kernels[j])
		number_count = new_kernel.shape[0] #1  #
		kernels = new_kernel.T

		# Setup the class with precomputed bessel functions (take a few moments)
		lp = levinpower.LevinPower(False, number_count,
									backgound_z, background_chi,
									chi_kernels, kernels,
									k_pk, z_pk, power_spectrum, True)

		lp.set_parameters(ell_limber, ell_nonlimber, max_number_subintervals,
							minell, maxell, n_nonlimber, n_limber, ninterp)

		t0_levin_cl = time.time()
		if verbose:
			#notify
			print('Calculating Cls')
		Cl_gg, Cl_gs, Cl_ss = lp.compute_C_ells(ell)

		def auto_correlation_index(n=2):
			'''
			==> returns the indices of Cl_gg from Levin for the 'Auto' correlation.
			n = no. of redshift bins being used in Levin such that len(z_edges) = n+1
			Levin returns the spctra in the following manner
			00(0) 01(1) 02(2)
					11(3) 12(4)
							22(5)
			Here, n = 3 and the indices for Cl_gg are mentioned in brackets.
			For this case the function returns an array [0,3,5]
			'''
			start = 0
			n = n
			result = []
			for i in range(n):
				result.append(start + (n - i))
			new_lst = np.cumsum(result)
			new_lst = np.insert(new_lst, 0, 0)
			return new_lst[:n]
		
		def cross_correlation_index(n=2):
			'''
			==> returns the indices of Cl_gg from Levin for the 'Cross' correlation.
			n = no. of redshift bins being used in Levin such that len(z_edges) = n+1
			Levin returns the spctra in the following manner
			00(0) 01(1) 02(2)
					11(3) 12(4)
							22(5)
			Here, n = 3 and the indices for Cl_gg are mentioned in brackets.
			For this case the function returns an array [1,2,4]
			'''
			#triangular number to calculate number of spctra
			a = 0
			n = n
			for i in range(n+1):
				a+=i
			auto_indices = auto_correlation_index(n)
			all_indices = np.arange(a)
			cross_correlation_index = np.delete(all_indices, auto_indices)
			return cross_correlation_index

		def levin_to_glass(cls, z_array, ell_array = np.arange(2, 1003, 1)):
			'''
			cls = C_ell from Levin (Cl_gg)
			z_array = array for the redshift edges (z_edges)
			ell_array = multipole defined in Levin (ell)
			==>returns the correctly sequenced (Cls, sequence for Glass) both as numpy arrays.
			********************************************************************************
			Reorders the Cls from Levin so that Glass can understand
			requries:
			gls = [gl_00,
					gl_11, gl_10,
					gl_22, gl_21, gl_20,
					...]
			from GLASS documentation (https://glass.readthedocs.io/en/stable/reference/fields.html)

			'''
			cells = cls
			zb = z_array
			nbins = len(zb) - 1
			ells = ell_array

			if nbins*(nbins+1)/2 != np.shape(cells)[0]:
				print('Wrong z_edge input')
			else:
				cells = np.insert(cells, 0, np.arange(len(cells)), axis=1)

				counter = 0
				nbins = len(zb)-1

				matrix_cells = np.zeros((len(ells)+1,nbins,nbins))
				for i in range(nbins):
					for j in range(i,nbins):
						matrix_cells[:,i,j] = cells[counter,:]
						matrix_cells[:,j,i] = cells[counter,:]
						counter +=1
                
				counter = 0
				for i in range(nbins):
					for j in reversed(range(i+1)): 
						index_i = i
						index_j = j
						cells[counter,:] = matrix_cells[:,index_i, index_j]
						counter += 1
				correct_sequencce = np.array(cells[:,0])
				correct_cls = np.delete(cells, 0, axis=1)
				return(correct_cls, correct_sequencce)

		cls_levin, x = levin_to_glass(Cl_gg, z_edges_for_glass_shell, ell)
		t1_levin_cl = time.time()
		total_levin_cl = t1_levin_cl-t0_levin_cl
		if verbose:
			#notify
			print(f"Done \nell: {ell[0], ell[-1]}, Cls shape: {np.shape(Cl_gg)}, \ntime:  {total_levin_cl:.2f}")
		return cls_levin
