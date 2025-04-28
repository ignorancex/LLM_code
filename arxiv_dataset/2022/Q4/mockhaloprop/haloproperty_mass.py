import numpy as np
import CosmologyLibraryM as cm

class haloproperty_mass(cm.cosmology):		
	def __init__(self,mvir,m200b,m200c,k,Tfn,Omega_matter = 0.276,Omega_lambda = 0.724,H_0=70.,ns=0.96, sigma_8 = 0.811 ,Omega_baryon = 0.045):
		cm.cosmology.__init__(self,Omega_matter = Omega_matter,Omega_lambda = Omega_lambda ,H_0=H_0,ns=ns ,sigma_8 = sigma_8 ,Omega_baryon = Omega_baryon)
		self.m200b = m200b
		self.m200c = m200c
		self.mvir = mvir
		self.k = k
		self.Tfn = Tfn
		self.Dlin = self.k**3*self.PS(self.k,0,self.Tfn)/(2*np.pi**2)
		
	def zehavi2011_meanc(self,mass,z):
		"""
		obtained from zehavi2011 : https://ui.adsabs.harvard.edu/abs/2011ApJ...736...59Z/abstract
		Input:
		mass needs to be in the units of h^{-1}Msun
		"""
		c0 = 11
		beta = -0.13
		mnl = 2.26e12
		return c0*(mass/mnl)**beta*(1+z)**(-1)

	def siglnc200b_DK15(self):
		sigma_lnc = np.ones(np.size(self.m200b))*0.16*np.log(10)
		return sigma_lnc

	def lnc200b_zehavi2011(self,z=0):
		return np.log(concentration_mass.zehavi2011_meanc(self,self.m200b,z=0))

	def cnuz_CDM_DK15(self,nu,z,nspec):
		"""
		Fitting function c(nu,z) [median] from Diemer&Kravtsov15 arXiv:1407.4730.
		"""
		phi0 = 6.58
		phi1 = 1.37
		eta0 = 6.82
		eta1 = 1.42
		alpha = 1.12
		beta = 1.69
		cmin = phi0 + phi1*nspec
		numin = eta0 + eta1*nspec
		out = 0.5*cmin*((nu/numin)**(-alpha) + (nu/numin)**beta)
		return out

	def cmz_CDM_DK15(self,m,z):
		""" Fitting function c(nu(m),z) from Diemer&Kravtsov15 arXiv:1407.4730.
		"""
		nspec = self.calc_nspec_DK15(m)
		nu = self.PeakHeight(m,self.k,self.Tfn,z)	
		out = self.cnuz_CDM_DK15(nu,z,nspec)
		return out
    # ~ ############################################################
    # ~ ############################################################
	def siglncmz_CDM_DK15(self):
		""" Measured siglnc (constant!) from Diemer&Kravtsov15 arXiv:1407.4730.
		"""
		out = 0.16*np.log(10)
		return out
    # ~ ############################################################
    # ~ ############################################################
	def siglncmz_CDM_W02(self):
		""" Fitting function siglnc(nu(m),z) (constant!) from Wechsler+2002 (footnote 10) .
		"""
		out = 0.14*np.log(10)
		return out
    # ~ ############################################################
    # ~ ############################################################
	def calc_nspec_DK15(self,m):
		""" Convenience function for calculating local slope of
			power spectrum, as defined by Diemr&Kravtsov15 arXiv:1407.4730.
		"""
		kappa = 0.69
		RLag = (3*m/(4*np.pi*self.Omega_matter*self.rho_c_h2_msun_mpc3))**(1/3.)
		keval = kappa*2*np.pi/RLag
		if np.isscalar(keval):
			ind = np.where(self.k >= keval)[0][0]
			dlnk2 = np.log(self.k[ind+1]/self.k[ind-1])
			nspec = np.log(self.Dlin[ind+1]/self.Dlin[ind-1])/(dlnk2)
			nspec -= 3.
		else:
			nspec = np.ones(keval.size,dtype=float)
			for k in range(keval.size):
				ind = np.where(self.k >= keval[k])[0][0]
				dlnk2 = np.log(self.k[ind+1]/self.k[ind-1])
				nspec[k] = np.log(self.Dlin[ind+1]/self.Dlin[ind-1])/(dlnk2)
				nspec[k] -= 3.
		return nspec
    # ~ ############################################################
    # ~ ############################################################
	def f_nfw(self,c):
		""" Convenience function for NFW normalisation. """
		out = (np.log(1+c)-c/(1+c))/c**3
		return out
    # ~ ############################################################
    ############################################################
	def c_hk03(self,f):
		""" Inverse of f_nfw(c) from Hu & Kravtsov (2003)."""
		lnf = np.log(f)
		a1 = 0.5116
		a2 = -0.4283
		a3 = -3.13e-3
		a4 = -3.52e-5
		p = a2 + a3*lnf + a4*(lnf)**2
		out = 1.0/(2*f + 1.0/np.sqrt(0.5625 + a1*f**(2*p)))
		return out
    # ~ ############################################################
    # ~ ############################################################
	def cDelta(self,cref,Deltaref,Delta):
		""" Convert from c_ref to c_Delta using HK03 prescription. Delta is what multiplies rho_b (not rho_crit)."""
		fDelta = Delta/Deltaref*self.f_nfw(cref)
		out = self.c_hk03(fDelta)
		return out
    # ~ ############################################################
    # ~ ############################################################
	def MDelta(self,Mref,cref,Deltaref,Delta):
		""" Convert from M_ref,c_ref to M_Delta using HK03 prescription. Delta is what multiplies rho_b (not rho_crit)."""
		out = (self.cDelta(cref,Deltaref,Delta)/cref)**3
		out *= (Mref*(Delta/Deltaref))
		return out

	def lnc200b_DK15(self,z=0,m200c=[],mflag=0):
		"""
		Inputs m200c?
		"""
		if len(m200c)==0:
			c200c = self.cmz_CDM_DK15(self.m200c,z)
		else:
			c200c = self.cmz_CDM_DK15(m200c,z)
		Deltaref = 200*self.E(z)**2/(self.Omega_matter*(1+z)**3)
		print ('deltaref',Deltaref)
		Delta = 200
		if mflag==0:
			return np.log(self.cDelta(c200c,Deltaref,Delta)) 
		elif mflag==1:
			c200b = (self.cDelta(c200c,Deltaref,Delta)) 
			m200b = self.MDelta(m200c,c200b,Deltaref,Delta)
			return np.log(c200b),m200b
		
	def lnc_to_a_VegaFerrero17(self,z=0):
		"""
		fitting formula for relaxed halos virial oversendity
		"""
		a = -0.295
		nu = self.PeakHeight(self.mvir,self.k,self.Tfn,z)
		c = -0.13
		nuprime = nu *(self.GrowthFunctionAnalytic(1/(1+z))/self.GrowthFunctionAnalytic(1.0))**c
		print (nu,self.GrowthFunctionAnalytic(1/(1+z)))
		return -0.52 + a*np.log(nuprime)


	def siglnc_to_a_VegaFerrero17(self):
		""" Measured siglnc (constant!) from Diemer&Kravtsov15 arXiv:1407.4730.
		"""
		return 0.19
