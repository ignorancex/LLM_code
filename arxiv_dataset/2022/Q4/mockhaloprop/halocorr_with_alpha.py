import numpy as np

class halocorr_with_alpha(object):
	def __init__(self,z):
		# ~ self.ro_c200b = np.load("/home/sujatha/mega/laptop_to_perseus/haloprop_class/p_rho_c200balpha.npz")
		# ~ pathmain = '/home/rsujatha/MEGA'
		pathmain = '/mnt/home/student/csujatha'
		self.ro_c200b = np.load(pathmain+"/perseus_to_laptop/haloprop_class_fits/pearson_rho_c200b-alpha.npz")
		self.ro_c_to_a = np.load(pathmain+"/perseus_to_laptop/haloprop_class_fits/pearson_rho_c_to_a-alpha.npz")
		self.ro_vc_to_va = np.load(pathmain+"/perseus_to_laptop/haloprop_class_fits/pearson_rho_vc_to_va-alpha.npz")
		self.ro_beta = np.load(pathmain+"/perseus_to_laptop/haloprop_class_fits/pearson_rho_beta-alpha.npz")
		self.ro_Spin = np.load(pathmain+"/perseus_to_laptop/haloprop_class_fits/pearson_rho_Spin-alpha.npz")			

		
	def rho_c200b(self,v,sample_cov=0,sampling=0):
		"""
		v is the peakheight		
		"""
		if sample_cov ==0:
			rhofit = self.ro_c200b['name1']
			rho = np.polyval(rhofit[::-1],np.log(v))
		elif sample_cov==1:
			rhofit = np.random.multivariate_normal(self.ro_c200b['name1'],self.ro_c200b['name2'],sampling)
			rho= 0	
			for i in range(len(self.ro_c200b['name1'])):
				rho +=np.log(v.reshape([1,len(v)]))**i*(rhofit[:,i]).reshape([len(rhofit[:,i]),1])
		return rho

	def rho_beta(self,v,sample_cov=0,sampling=0):
		"""
		v is the peakheight		
		"""
		if sample_cov ==0:
			rhofit = self.ro_beta['name1']
			rho = np.polyval(rhofit[::-1],np.log(v))
		elif sample_cov==1:
			rhofit = np.random.multivariate_normal(self.ro_beta['name1'],self.ro_beta['name2'],sampling)
			rho= 0	
			for i in range(len(self.ro_beta['name1'])):
				rho +=np.log(v.reshape([1,len(v)]))**i*(rhofit[:,i]).reshape([len(rhofit[:,i]),1])
		return rho

		
	def rho_vc_to_va(self,v,sample_cov=0,sampling=0):
		"""
		v is the peakheight		
		"""
		if sample_cov ==0:
			rhofit = self.ro_vc_to_va['name1']
			rho = np.polyval(rhofit[::-1],np.log(v))
		elif sample_cov==1:
			rhofit = np.random.multivariate_normal(self.ro_vc_to_va['name1'],self.ro_vc_to_va['name2'],sampling)
			rho= 0	
			for i in range(len(self.ro_vc_to_va['name1'])):
				rho +=np.log(v.reshape([1,len(v)]))**i*(rhofit[:,i]).reshape([len(rhofit[:,i]),1])
		return rho
		
	def rho_c_to_a(self,v,sample_cov=0,sampling=0):
		"""
		v is the peakheight		
		"""
		vpivot = v-2.05
		if sample_cov ==0:
			rhofit = self.ro_c_to_a['name1']
			rho = np.polyval(rhofit[::-1],np.log(v))
		elif sample_cov==1:
			rhofit = np.random.multivariate_normal(self.ro_c_to_a['name1'],self.ro_c_to_a['name2'],sampling)
			rho= 0	
			for i in range(len(self.ro_c_to_a['name1'])):
				rho +=np.log(v.reshape([1,len(v)]))**i*(rhofit[:,i]).reshape([len(rhofit[:,i]),1])
		return rho
		

		
	def rho_Spin(self,v,sample_cov=0,sampling=0):
		"""
		v is the peakheight		
		"""
		if sample_cov ==0:
			rhofit = self.ro_Spin['name1']
			rho = np.polyval(rhofit[::-1],np.log(v))
		elif sample_cov==1:
			rhofit = np.random.multivariate_normal(self.ro_Spin['name1'],self.ro_Spin['name2'],sampling)
			rho= 0	
			for i in range(len(self.ro_Spin['name1'])):
				rho +=np.log(v.reshape([1,len(v)]))**i*(rhofit[:,i]).reshape([len(rhofit[:,i]),1])
		return rho
		
	def rho_spin_bullock(self,v):
		"""
		v is the peakheight		
		"""
		vpivot = v-2.05
		rhofit = self.ro_spin_bullock['name1']
		rho = (vpivot**3*rhofit[0]+vpivot**2*rhofit[1]+vpivot*rhofit[2]+rhofit[3])
		return rho
		
	def rho_triaxiality(self,v):
		"""
		v is the peakheight		
		"""
		vpivot = v-2.05
		rhofit = self.ro_triaxiality['name1']
		rho = (vpivot**3*rhofit[0]+vpivot**2*rhofit[1]+vpivot*rhofit[2]+rhofit[3])
		return rho

	def sample_conditionalP(self,alphat,rho):
		"""
		samples the conditional probability distribution p(c|\alpha,m) = N(rhoalpha,(1-rho^2))
		"""
		return np.random.normal(loc=rho*alphat,scale=np.sqrt(1-rho**2))

	def rho_b_to_a(self,v):
		"""
		v is the peakheight		
		"""
		vpivot = v-2.05
		rhofit = self.ro_b_to_a['name1']
		rho = (vpivot**3*rhofit[0]+vpivot**2*rhofit[1]+vpivot*rhofit[2]+rhofit[3])
		return rho

		
	def rho_c200c(self,v):
		"""
		v is the peakheight		
		"""
		vpivot = v-2.05
		rhofit = self.ro_c200c['name1']
		rho = (vpivot**3*rhofit[0]+vpivot**2*rhofit[1]+vpivot*rhofit[2]+rhofit[3])
		return rho
		
		
	def rho_ellipticity(self,v):
		"""
		v is the peakheight
		"""
		vpivot = v-2.05
		rhofit = self.ro_ellipticity['name1']
		return np.polyval(rhofit,vpivot)
