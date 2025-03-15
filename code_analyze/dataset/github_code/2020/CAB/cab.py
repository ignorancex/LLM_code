import numpy as np
import scipy.special as special
import os


absolutepath=  os.path.dirname(os.path.abspath(__file__))
print (absolutepath)
      
class cosmology(object):
	def __init__(self,Omega_matter = 0.276,Omega_lambda=0.724,H_0=70.,ns=0.961,sigma_8 = 0.811,kbyh=None,Tfn=None ):
		self.Omega_matter=Omega_matter
		self.rho_c_h2_msun_mpc3 = 2776*1e8    ## critical density in (msun/h)(mpc/h)**3
		self.ns = ns
		self.Omega_lambda = Omega_lambda
		self.sigma_8 = sigma_8
		self.H_0=H_0
		self.delta_c=1.686
		self.kbyh = kbyh
		self.Tfn = Tfn
		### loading fits files #################################################################
		# ~ self.m1 = np.load("../fits/p_m1_alpha.npz")
		# ~ self.s1 = np.load("../fits/p_S1_alpha.npz")
		# ~ self.m2 = np.load("../fits/p_m2_alpha.npz")
		# ~ self.s2 = np.load("../fits/p_S2_alpha.npz")
		# ~ self.ro = np.load("../fits/p_rho.npz")
		self.m1 = np.load(absolutepath+"/fits/p_m1_alpha.npz")
		self.s1 = np.load(absolutepath+"/fits/p_S1_alpha.npz")
		self.m2 = np.load(absolutepath+"/fits/p_m2_alpha.npz")
		self.s2 = np.load(absolutepath+"/fits/p_S2_alpha.npz")
		# ~ self.ro = np.load(absolutepath+"/fits/p_rho.npz")
		self.ro={}
		self.ro['c200b'] = np.load(absolutepath+"/fits/pearson_rho_c200b-alpha.npz")
		self.ro['c_to_a'] = np.load(absolutepath+"/fits/pearson_rho_c_to_a-alpha.npz")
		self.ro['vc_to_va'] = np.load(absolutepath+"/fits/pearson_rho_vc_to_va-alpha.npz")
		self.ro['beta'] = np.load(absolutepath+"/fits/pearson_rho_beta-alpha.npz")
		self.ro['Spin'] = np.load(absolutepath+"/fits/pearson_rho_Spin-alpha.npz")	
	def Wk(self,k,R):
		"""
		Fourier Transform of a Spherical Top Hat Filter
		"""
		return 3/(k*R)**3*(np.sin(k*R)-(k*R)*np.cos(k*R))	

	def GrowthFunction(self,a):
		"""
		a is the scale factor
		"""
		a=np.array(a)+1e-15
		D=np.ones(np.size(a))
		H=self.H_0*(self.Omega_matter*(a**(-3))+self.Omega_lambda)**(1/2.)
		D=(H/self.H_0)*a**(5/2.)/np.sqrt(self.Omega_matter)*special.hyp2f1(5/6.,3/2.,11/6.,-a**3*self.Omega_lambda/self.Omega_matter)
		return D
			
	def PS(self,z):
		"""
		Input
		k in h Mpc^1
		z redshift
		T the tranfer function
		
		Outputs 
		Pk the power spectrum
		"""
		R=8.
		integrand = 1/(2*np.pi**2)*self.kbyh**(self.ns+2.)*self.Tfn**2*cosmology.Wk(self,self.kbyh,R)**2
		igrate = np.trapz(integrand,self.kbyh)
		SigmaSquare=self.sigma_8**2
		NormConst = SigmaSquare/igrate
		return NormConst*self.kbyh**self.ns*(self.Tfn)**2*cosmology.GrowthFunction(self,1./(1.+z))**2/cosmology.GrowthFunction(self,1)**2

	def PeakHeight(self,mass,z):
		"""
		Inputs:
		mass in Msunh-1
		k in h Mpc^1
		z redshift
		T the tranfer function
		"""
		R = (3/(4*np.pi)*mass/(self.rho_c_h2_msun_mpc3*self.Omega_matter))**(1/3.) ## is in units of Mpch-1
		PS = cosmology.PS(self,z)
		sigma_square = np.zeros([len(R),1])
		for i in range(0,len(R)):
			wk = cosmology.Wk(self,self.kbyh,R[i])
			sigma_square[i] = 1/(2.*np.pi**2)*np.trapz(PS*wk**2*self.kbyh**2,self.kbyh)
		nu = self.delta_c/np.sqrt(sigma_square)
		return nu.flatten()

	def T08(self,mass,k,Tfn,z):
		"""
		Refer to Tinker (2008): https://arxiv.org/pdf/0803.2706.pdf
		mass should be in Msun h^-1
		"""
		R = (3/(4*np.pi)*mass/(self.rho_c_h2_msun_mpc3*self.Omega_matter))**(1/3.)
		PS = cosmology.PS(self,z)
		Delk = 1/(2*np.pi**2)*PS*self.kbyh**3
		sigma_square = np.zeros([len(R),1])		
		rho_dln_simgainv_by_dm = np.zeros([len(R),1])			
		for i in range(0,len(R)):
			wk = cosmology.Wk(self,self.kbyh,R[i])
			dWk2_by_dR = cosmology.dWk2_by_dR(self,self.kbyh,R[i])
			sigma_square[i] = 1/(2.*np.pi**2)*np.trapz(PS*wk**2*self.kbyh**2,self.kbyh)
			rho_dln_simgainv_by_dm[i] = - 1/(2*sigma_square[i]) * np.trapz(Delk/self.kbyh*dWk2_by_dR/(4*np.pi*R[i]**2),self.kbyh)			
		sigma = np.sqrt(sigma_square)
		delt = 200
		A = 0.186*(1+z)**(-0.14)
		a = 1.47*(1+z)**(-0.06)
		alpha = 10**(-(0.75/(np.log10(delt/75)))**(1.2))
		b = 2.57*(1+z)**(-alpha)
		c = 1.19
		f = A* ((sigma/b)**(-a)+1)*np.exp(-c/sigma**2)
		dn_by_dlnm = f * rho_dln_simgainv_by_dm
		return dn_by_dlnm

	def dWk2_by_dR(self,k,R):
		return -54/(k**6.*R**7.)*(np.sin(k*R)-(k*R)*np.cos(k*R))**2. + 18/(k**4.*R**5.) * (np.sin(k*R)-(k*R)*np.cos(k*R)) * np.sin(k*R)


	def T10(self,argument,z=0,mass_flag=1):
		"""
		Input:
		argument can either be \nu or mass in Mpc/h
		k in h Mpc^1
		T the tranfer function
		z is the redshift
		mass_flag 1 takes mass as input 0 takes peak height as input
		
		Output:
		returns Tinker2010 bias
		"""	
		delta_c = 1.686
		if mass_flag ==1:
			R = (3/(4.*np.pi)*argument/(self.rho_c_h2_msun_mpc3*self.Omega_matter))**(1/3.)
			PS = cosmology.PS(self,z)
			Delk = 1/(2.*np.pi**2)*PS*self.kbyh**3.
			sigma_square = np.zeros([len(R),1])
			for i in range(0,len(R)):
				wk = cosmology.Wk(self,self.kbyh,R[i])
				sigma_square[i] = np.trapz(Delk*wk**2/self.kbyh,self.kbyh)
			v = delta_c/np.sqrt(sigma_square)
		else:
			v = argument
		delta = 200
		y = np.log10(delta)
		A = 1. + 0.24 * y * np.exp(-(4/y)**4)
		a = 0.44 * y - 0.88
		B = 0.183
		b = 1.5
		C = 0.019 + 0.107 * y + 0.19 * np.exp(-(4/y)**4)
		c = 2.4
		bias = 1- A*v**a/(v**a+delta_c**a)+B*v**b+C*v**c
		return bias.flatten()

	def H1(self,s):
		return s
		
	def H2(self,s):
		return s**2 -1
		
	def H3(self,s):
		return s**3-3*s
		
	def H4(self,s):
		return s**4-6*s**2+3

	def fit(self,v,p):
		"""
		Note that g(z) has already been divided out
		"""
		return (np.log10(v/1.5)**2*p[0] + np.log10(v/1.5)*p[1] + p[2]).flatten()
		
	def alphafit(self,v,p,sample_cov=0,sampling=0):
		"""
		Note that g(z) has already been divided out
		"""
		if sample_cov==0:
			fitval = (np.log10(v/1.5)**2*p['name1'][0] + np.log10(v/1.5)*p['name1'][1] + p['name1'][2]).flatten()
		elif sample_cov==1:
			fits = np.random.multivariate_normal(p['name1'],p['name2'],sampling)
			fitval= 0	
			v = v.reshape([1,len(v)])
			fits = fits.reshape([sampling,3,1])
			fitval = (np.log10(v/1.5)**2*fits[:,0,:] + np.log10(v/1.5)*fits[:,1,:] + fits[:,2,:])
		return fitval

	def b1avg(self,m,z,fromval,toval=None,mass_flag=1):
		"""
		mass_flag = 0->peakheight
				  = 1->mass 
		m = peakheight is mass_flag=0 or mass in h^-1Msun if mass_flag=1
		z = redshift
		requires mass input in h^-1 Msun
		"""
		if toval==None:
			if fromval==None:
				h1avg = 0
				h2avg = 0
			else:
				h1avg = cosmology.H1(self,fromval)
				h2avg = cosmology.H2(self,fromval)
			_avg = 1
		elif np.exp(-toval**2/2)==0.0:
			h1avg = (np.exp(-fromval**2/2)-np.exp(-toval**2/2))/np.sqrt(2*np.pi)
			h2avg = (fromval*np.exp(-fromval**2/2))/np.sqrt(2*np.pi)
			_avg = (special.erf(toval/np.sqrt(2))-special.erf(fromval/np.sqrt(2)))/2
		elif np.exp(-fromval**2/2)==0.0:
			h1avg = (np.exp(-fromval**2/2)-np.exp(-toval**2/2))/np.sqrt(2*np.pi)
			h2avg = (-toval*np.exp(-toval**2/2))/np.sqrt(2*np.pi)
			_avg = (special.erf(toval/np.sqrt(2))-special.erf(fromval/np.sqrt(2)))/2
		else:
			h1avg = (np.exp(-fromval**2/2)-np.exp(-toval**2/2))/np.sqrt(2*np.pi)
			h2avg = (fromval*np.exp(-fromval**2/2)-toval*np.exp(-toval**2/2))/np.sqrt(2*np.pi)
			_avg = (special.erf(toval/np.sqrt(2))-special.erf(fromval/np.sqrt(2)))/2
		if mass_flag ==1:
			v = cosmology.PeakHeight(self,m,z)
			b1avg = cosmology.T10(self,m,z=z,mass_flag=1) + cosmology.alphafit(self,v,self.m1)*h1avg/_avg + 1/2.*cosmology.alphafit(self,v,self.s1)*h2avg/_avg
		elif mass_flag==0:
			b1avg = cosmology.T10(self,m,z=z,mass_flag=0) + cosmology.alphafit(self,m,self.m1)*h1avg/_avg + 1/2.*cosmology.alphafit(self,m,self.s1)*h2avg/_avg
		return b1avg
		
	def b2avg(self,m,z,fromval,toval=None,mass_flag=1):
		"""
		MASS_FLAG==1 for mass cMpc/h,
		MASS_FlAG==0 for peakheight
		"""
		if fromval==None:
			if toval==None:
				h1avg = 0
				h2avg = 0
				h3avg = 0
				h4avg = 0
			else:
				h1avg = cosmology.H1(self,toval)
				h2avg = cosmology.H2(self,toval)
				h3avg = cosmology.H3(self,toval)
				h4avg = cosmology.H4(self,toval)
		elif np.exp(-toval**2/2)==0.0:
			_avg = (special.erf(toval/np.sqrt(2))-special.erf(fromval/np.sqrt(2)))/2
			h1avg = (np.exp(-fromval**2/2)-np.exp(-toval**2/2))/np.sqrt(2*np.pi)/_avg
			h2avg = (fromval*np.exp(-fromval**2/2))/np.sqrt(2*np.pi)/_avg
			h3avg = ((fromval**2-1)*np.exp(-fromval**2/2))/np.sqrt(2*np.pi)/_avg
			h4avg = ((fromval**3-3*fromval)*np.exp(-fromval**2/2))/np.sqrt(2*np.pi)/_avg
		elif np.exp(-fromval**2/2)==0.0:
			_avg = (special.erf(toval/np.sqrt(2))-special.erf(fromval/np.sqrt(2)))/2
			h1avg = (np.exp(-fromval**2/2)-np.exp(-toval**2/2))/np.sqrt(2*np.pi)/_avg
			h2avg = (-toval*np.exp(-toval**2/2))/np.sqrt(2*np.pi)/_avg
			h3avg = ((1-toval**2)*np.exp(-toval**2/2))/np.sqrt(2*np.pi)/_avg
			h4avg = (-(toval**3-3*toval)*np.exp(-toval**2/2))/np.sqrt(2*np.pi)/_avg
		else:
			_avg = (special.erf(toval/np.sqrt(2))-special.erf(fromval/np.sqrt(2)))/2	
			h1avg = (np.exp(-fromval**2/2)-np.exp(-toval**2/2))/np.sqrt(2*np.pi)/_avg
			h2avg = (fromval*np.exp(-fromval**2/2)-toval*np.exp(-toval**2/2))/np.sqrt(2*np.pi)/_avg
			h3avg = ((fromval**2-1)*np.exp(-fromval**2/2.)-(toval**2-1)*np.exp(-toval**2/2.))/np.sqrt(2*np.pi)/_avg
			h4avg = ((fromval**3-3*fromval)*np.exp(-fromval**2/2)-(toval**3-3*toval)*np.exp(-toval**2/2))/np.sqrt(2*np.pi)/_avg
	
		if mass_flag==1:
			v = cosmology.PeakHeight(self,m,z)
			mu1 = cosmology.alphafit(self,v,self.m1)
			mu2 = cosmology.alphafit(self,v,self.m2)
			s1  = cosmology.alphafit(self,v,self.s1)
			s2  = cosmology.alphafit(self,v,self.s2)	
			bone = cosmology.T10(self,m,z=z,mass_flag=1)
			btwo = 0.412 - 2.143 *bone + 0.929 *bone**2 + 0.008*bone**3
			b2avg = btwo + (mu2+2*mu1*(bone-1)+8/21*mu1)*h1avg + (mu1**2+s1*(bone-1)+1/2.*s2+4/21.*s1)*h2avg + (mu1*s1)*h3avg + (1/4.*s1**2)*h4avg
		elif mass_flag==0:
			mu1 = cosmology.alphafit(self,m,self.m1)
			mu2 = cosmology.alphafit(self,m,self.m2)
			s1  = cosmology.alphafit(self,m,self.s1)
			s2  = cosmology.alphafit(self,v,self.s2)	
			bone = cosmology.T10(self,m,z=z,mass_flag=0)
			btwo = 0.412 - 2.143 *bone + 0.929 *bone**2 + 0.008*bone**3
			b2avg = btwo + (mu2+2*mu1*(bone-1)+8/21*mu1)*h1avg + (mu1**2+s1*(bone-1)+1/2.*s2+4/21.*s1)*h2avg + (mu1*s1)*h3avg + (1/4.*s1**2)*h4avg			
		return b2avg
		
	def b2_c_from_alpha(self,secondaryproperty,m,z,fromval,toval=None):
		bone = cosmology.T10(self,m,z=z,mass_flag=1)
		btwo = 0.412 - 2.143 *bone + 0.929 *bone**2 + 0.008*bone**3
		_avg = (special.erf(toval/np.sqrt(2))-special.erf(fromval/np.sqrt(2)))/2
		h1avg = (np.exp(-fromval**2/2)-np.exp(-toval**2/2))/np.sqrt(2*np.pi)/_avg
		if np.exp(-toval**2/2)==0.0:
			h2avg = (fromval*np.exp(-fromval**2/2))/np.sqrt(2*np.pi)/_avg
			h3avg = ((fromval**2-1)*np.exp(-fromval**2/2))/np.sqrt(2*np.pi)/_avg
			h4avg = ((fromval**3-3*fromval)*np.exp(-fromval**2/2))/np.sqrt(2*np.pi)/_avg
		elif np.exp(-fromval**2/2)==0.0:
			h2avg = (-toval*np.exp(-toval**2/2))/np.sqrt(2*np.pi)/_avg
			h3avg = ((1-toval**2)*np.exp(-toval**2/2))/np.sqrt(2*np.pi)/_avg
			h4avg = (-(toval**3-3*toval)*np.exp(-toval**2/2))/np.sqrt(2*np.pi)/_avg
		else:
			h2avg = (fromval*np.exp(-fromval**2/2)-toval*np.exp(-toval**2/2))/np.sqrt(2*np.pi)/_avg
			h3avg = ((fromval**2-1)*np.exp(-fromval**2/2.)-(toval**2-1)*np.exp(-toval**2/2.))/np.sqrt(2*np.pi)/_avg
			h4avg = ((fromval**3-3*fromval)*np.exp(-fromval**2/2)-(toval**3-3*toval)*np.exp(-toval**2/2))/np.sqrt(2*np.pi)/_avg
		v = cosmology.PeakHeight(self,m,z)
		vpivot = v-2.05
		mu1 = cosmology.alphafit(self,v,self.m1)
		mu2 = cosmology.alphafit(self,v,self.m2)
		s1  = cosmology.alphafit(self,v,self.s1)
		s2  = cosmology.alphafit(self,v,self.s2)	
		# ~ rhofit = self.ro['name1']
		# ~ rho = (vpivot**3*rhofit[0]+vpivot**2*rhofit[1]+vpivot*rhofit[2]+rhofit[3])
		rho = self.rhoget(secondaryproperty,v)
		b2avg = btwo + (mu2+2*mu1*(bone-1)+8/21*mu1)*h1avg*rho + (mu1**2+s1*(bone-1)+1/2.*s2+4/21.*s1)*h2avg*rho**2 + (mu1*s1)*h3avg*rho**3 + (1/4.*s1**2)*h4avg*rho**4
		##################### for generating error bar  ################################################################
		sampling=100
		rhosamp = self.rhoget(secondaryproperty,v,sample_cov=1,sampling=sampling)
		mu1samp = self.alphafit(v,self.m1,sample_cov=1,sampling=sampling)
		s1samp = self.alphafit(v,self.s1,sample_cov=1,sampling=sampling)
		mu2samp = self.alphafit(v,self.m2,sample_cov=1,sampling=sampling)
		s2samp = self.alphafit(v,self.s2,sample_cov=1,sampling=sampling)		
		b2_fr_err=  (mu2samp+2*mu1samp*(bone-1)+8/21*mu1samp)*h1avg*rhosamp + (mu1samp**2+s1samp*(bone-1)+1/2.*s2samp+4/21.*s1samp)*h2avg*rhosamp**2 + (mu1samp*s1samp)*h3avg*rhosamp**3 + (1/4.*s1samp**2)*h4avg*rhosamp**4
		err_in_b2 = np.std(b2_fr_err,axis=0)
		return b2avg,err_in_b2
		
		
	def b1_c_from_alpha(self,secondaryproperty='c200b',m=None,z=None,fromval=None,toval=None):
		if toval==None:
			if fromval==None:
				h1avg = 0
				h2avg = 0
			else:
				h1avg = cosmology.H1(self,fromval)
				h2avg = cosmology.H2(self,fromval)
			_avg = 1
		elif np.exp(-toval**2/2)==0.0:
			h2avg = (fromval*np.exp(-fromval**2/2))/np.sqrt(2*np.pi)
			h1avg = (np.exp(-fromval**2/2)-np.exp(-toval**2/2))/np.sqrt(2*np.pi)
			_avg = (special.erf(toval/np.sqrt(2))-special.erf(fromval/np.sqrt(2)))/2
		elif np.exp(-fromval**2/2)==0.0:
			h2avg = (-toval*np.exp(-toval**2/2))/np.sqrt(2*np.pi)
			h1avg = (np.exp(-fromval**2/2)-np.exp(-toval**2/2))/np.sqrt(2*np.pi)
			_avg = (special.erf(toval/np.sqrt(2))-special.erf(fromval/np.sqrt(2)))/2
		else:
			h2avg = (fromval*np.exp(-fromval**2/2)-toval*np.exp(-toval**2/2))/np.sqrt(2*np.pi)
			h1avg = (np.exp(-fromval**2/2)-np.exp(-toval**2/2))/np.sqrt(2*np.pi)
			_avg = (special.erf(toval/np.sqrt(2))-special.erf(fromval/np.sqrt(2)))/2
		if z=='notrequired_because_prev_arg_is_peakheight':
			v = m
		else:
			v = self.PeakHeight(m,z)
		mu1 = cosmology.alphafit(self,v,self.m1)
		s1 = cosmology.alphafit(self,v,self.s1)
		# ~ rhoget = getattr(self,'rho_'+secondaryproperty)
		rho = self.rhoget(secondaryproperty,v)
		b1avg = cosmology.T10(self,m,z=z,mass_flag=1) + rho*mu1*h1avg/_avg + 1/2.*rho**2*s1*h2avg/_avg		
		##################### for generating error bar  ################################################################
		sampling=100
		rhosamp = self.rhoget(secondaryproperty,v,sample_cov=1,sampling=sampling)
		mu1samp = self.alphafit(v,self.m1,sample_cov=1,sampling=sampling)
		s1samp = self.alphafit(v,self.s1,sample_cov=1,sampling=sampling)
		b1_fr_err= rhosamp*mu1samp*h1avg/_avg + 1/2.*rhosamp**2*s1samp*h2avg/_avg
		err_in_b1 = np.std(b1_fr_err,axis=0)
		return b1avg,err_in_b1 	

	def rhoget(self,secondaryproperty='c200b',v=None,sample_cov=0,sampling=0):
		"""
		v is the peakheight		
		"""
		if sample_cov ==0:
			rhofit = self.ro[secondaryproperty]['name1']
			rho = np.polyval(rhofit[::-1],np.log(v))
		elif sample_cov==1:
			rhofit = np.random.multivariate_normal(self.ro[secondaryproperty]['name1'],self.ro[secondaryproperty]['name2'],sampling)
			rho= 0	
			for i in range(len(self.ro[secondaryproperty]['name1'])):
				rho +=np.log(v.reshape([1,len(v)]))**i*(rhofit[:,i]).reshape([len(rhofit[:,i]),1])
		if secondaryproperty =='beta':
			rho=-rho
		return rho










