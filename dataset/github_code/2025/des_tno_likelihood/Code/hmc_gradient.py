import numpy as np 
import numba 
import gmm_anyk as ga 


@numba.njit(fastmath = True)
def singleslope(H, slope):
	norm = slope * np.log(10)/(np.power(10, 10.*slope) - np.power(10, 3.*slope))
	size_chain_comp =  np.power(10, slope * H)*norm

	return size_chain_comp


@numba.njit(fastmath = True)
def singleslope_deriv(H, slope):
	scale = np.log(10) * np.power(10, slope*(H-3))/((np.power(10, 7*slope) - 1)**2)
	factor = -1. + np.power(10, 7*slope) + slope * np.log(10) * (3 + np.power(10, 7*slope) * (H - 10) - H) 
	return scale * factor

@numba.njit(fastmath = True)
def singleslope_multi(H, slopes):
	z = np.zeros((len(H), len(slopes)))
	for alpha in range(len(slopes)):
		z[:,alpha] = singleslope(H, slopes[alpha])
	return z


@numba.njit(fastmath = True)
def singleslope_deriv_multi(H, slopes):
	z = np.zeros((len(H), len(slopes)))
	for alpha in range(len(slopes)):
		z[:,alpha] = singleslope_deriv(H, slopes[alpha])
	return z


@numba.njit(fastmath = True)
def jacobian_color_H(colors, H):
	prod = np.ones_like(H)
	for i in range(3):
		prod *= np.power(10, +0.4 * (colors[:,i] + (H)))
	return np.log(prod)

@numba.njit(fastmath = True)
def jacobian_H(H):
	prod = np.power(10, +0.4 * (H))
	return np.log(prod)

@numba.njit(fastmath = True)
def jacobian_color(colors):
	prod = np.ones(len(colors))
	for i in range(3):
		prod *= np.power(10, +0.4 * (colors[:,i]))
	return np.log(prod)

@numba.njit(parallel = True, fastmath=True)
def Rialpha_gmm(ncomp, nobj, jacob, pref):
	Rialpha = np.zeros((nobj, ncomp))
	for i in numba.prange(nobj):
		for alpha in range(ncomp):
			Rialpha[i, alpha] = np.sum(np.exp(jacob[i] + pref[i][:,alpha]))/len(pref[i][:,alpha])
	return Rialpha


@numba.njit(parallel = True, fastmath=True)
def Rialpha_singleslope(ncomp, nobj, jacob, pref, chain_H, params):
	Rialpha = np.zeros((nobj, ncomp))
	for i in numba.prange(nobj):
		for alpha in range(ncomp):
			slope = params[alpha]
			size_chain_comp = np.log(singleslope(chain_H[i], slope))

			Rialpha[i, alpha] = np.sum(np.exp(jacob[i] + pref[i][:,alpha] + size_chain_comp))/len(chain_H[i])

	return Rialpha

@numba.njit(parallel = True, fastmath=True)
def Rialphaderiv_singleslope(ncomp, nobj, jacob, pref, chain_H, params):
	Rialpha = np.zeros((nobj, ncomp))
	for i in numba.prange(nobj):
		for alpha in range(ncomp):
			slope = params[alpha]
			size_chain_comp = singleslope_deriv(chain_H[i], slope)
			Rialpha[i, alpha] = np.sum(np.exp(jacob[i] + pref[i][:,alpha]) * size_chain_comp) / len(chain_H[i])

	return Rialpha

@numba.njit(fastmath=True)
def eigendecompose(color, eigen, mean):
	delta = color - mean
	proj = np.dot(delta, eigen)

	return np.sign(proj)


class LogLikeGMMSingleSlope:
	def __init__(self, gmm, chains_color, chains_H, selection, ncomp, pref=-1, prior = None, prior_deriv = None):
		self.gmm = gmm 
		self.chains_color = chains_color
		self.chains_H = numba.typed.List(chains_H)

		self.selection = selection

		self.nobj = len(chains_H)
		
		self.ncomp = ncomp 

		self.rng = np.random.default_rng()

		
		self.pref = pref
		self.compute_gmm_chain()
		self.compute_jacobian_chain()

		self.H_bins = np.arange(5.05,9.05,0.1)

		if prior is None:
			self.prior = lambda x : 0
			self.prior_deriv = lambda x : 0
		else:
			self.prior = prior 
			self.prior_deriv = prior_deriv

	def evaluate_gmm(self, colors):
		logdet = np.linalg.slogdet(self.gmm.cov_best)[1] 

		return ga.log_multigaussian_no_off(colors, self.gmm.mean_best, self.gmm.cov_best) - logdet/2 - np.log(2*np.pi)*3/2

	def compute_gmm_chain(self):
		self.S_obj_noamp = []
		logdet = np.linalg.slogdet(self.gmm.cov_best)[1] 

		for x in self.chains_color:
			self.S_obj_noamp.append(ga.log_multigaussian_no_off(x, self.gmm.mean_best, self.gmm.cov_best) - logdet/2 - np.log(2*np.pi)*3/2)

		self.S_obj_noamp = numba.typed.List(self.S_obj_noamp)

	def compute_jacobian_chain(self):
		self.jacobian = []

		for i in range(self.nobj):
			self.jacobian.append(jacobian_color_H(self.chains_color[i], self.chains_H[i]))

		self.jacobian = numba.typed.List(self.jacobian)

	def size_dist(self, params, H):
		return singleslope_multi(H, params)

	def Rialpha(self, params):
		return Rialpha_singleslope(self.ncomp, self.nobj, self.jacobian, self.S_obj_noamp, self.chains_H, params)

	def Salpha(self, params, fun):
		f = fun(params, self.H_bins)

		S = np.zeros(self.ncomp)
		for alpha in range(self.ncomp):
			integrand = self.selection[alpha] * f[:,alpha] 

			S[alpha] = np.sum((integrand[1:] + integrand[:-1])*np.diff(self.H_bins))/2
		return S

	def likelihood(self, falpha, params):
		self.Rialpha_current = self.Rialpha(params)

		self.Salpha_current = self.Salpha(params, self.size_dist)

		self.falphaRalpha = np.sum(self.Rialpha_current * falpha, axis=1)

		prior = self.prior(params)

		return self.pref*(np.sum(np.log(self.falphaRalpha)) - self.nobj * np.log(np.dot(falpha, self.Salpha_current)) + prior)


	def gradient_theta(self, falpha, params):
		## Assumes we have already computed Rialpha and Salpha
		self.Rialpha_current = self.Rialpha(params)

		self.Salpha_current = self.Salpha(params, self.size_dist)

		self.falphaRalpha = np.sum(self.Rialpha_current * falpha, axis=1)
		#Salpha_current = self.Salpha(params, self.size_dist)
		Salpha_sum = np.dot(falpha,self.Salpha_current)

		grad = np.zeros_like(params)

		SHintegrand = lambda params, H : singleslope_deriv_multi(H, params)
		
		self.SalphatimesH = self.Salpha(params, SHintegrand)

		self.Rialpha_deriv = Rialphaderiv_singleslope(self.ncomp, self.nobj, self.jacobian, self.S_obj_noamp, self.chains_H, params)

		for alpha in range(self.ncomp):
			Rialphaderiv_comp = self.Rialpha_deriv[:,alpha] 
			Salpha_comp = self.Salpha_current[alpha]

			SalphatimesH_comp = self.SalphatimesH[alpha]

			gradRialpha = np.sum(Rialphaderiv_comp/self.falphaRalpha)
			prior_deriv = self.prior_deriv(params[alpha])

			grad[alpha] = gradRialpha*falpha[alpha] - self.nobj*falpha[alpha]*SalphatimesH_comp/Salpha_sum + prior_deriv
		

		return self.pref*grad

	def gradient_f(self, falpha, params):
		grad = np.zeros_like(falpha)
		self.Rialpha_current = self.Rialpha(params)
		self.Salpha_current = self.Salpha(params, self.size_dist)
		Salpha_sum = np.dot(falpha, self.Salpha_current)

		self.falphaRalpha = np.sum(self.Rialpha_current * falpha, axis=1)

		for alpha in range(self.ncomp):

			gradsumRialpha = np.sum(self.Rialpha_current[:,alpha]/self.falphaRalpha)
			grad[alpha] = gradsumRialpha - self.nobj * self.Salpha_current[alpha]/Salpha_sum

		return self.pref*grad



class LogLikeGMMHalfSingleSlope(LogLikeGMMSingleSlope):
	def compute_gmm_chain(self):
		self.eigen = np.linalg.eig(self.gmm.cov_best[0])[1][:,0]
		self.S_obj_noamp = []
		logdet = np.linalg.slogdet(self.gmm.cov_best)[1] 

		for x in self.chains_color:
			g = ga.log_multigaussian_no_off(x, self.gmm.mean_best, self.gmm.cov_best) - logdet/2 - np.log(2*np.pi)*3/2
			decomp = eigendecompose(x, self.eigen, self.gmm.mean_best[0]) 
			g[decomp > 0,0] = -np.inf
			g[decomp < 0,0] += np.log(2)

			g[decomp < 0,1] = -np.inf
			g[decomp > 0,1] += np.log(2)
			
			self.S_obj_noamp.append(g)

		self.S_obj_noamp = numba.typed.List(self.S_obj_noamp)


class FixedSlope(LogLikeGMMSingleSlope):
	def likelihood(self, falpha, params):
		params = np.array([0.8, params[0]])

		self.Rialpha_current = self.Rialpha(params)

		self.Salpha_current = self.Salpha(params, self.size_dist)

		self.falphaRalpha = np.sum(self.Rialpha_current * falpha, axis=1)

		return self.pref*(np.sum(np.log(self.falphaRalpha)) - self.nobj * np.log(np.sum(falpha * self.Salpha_current)))

	def gradient_theta(self, falpha, params):
		## Assumes we have already computed Rialpha and Salpha
		#self.Rialpha_current = self.Rialpha(params)

		#self.Salpha_current = self.Salpha(params, self.model_alpha)

		#self.falphaRalpha = np.sum(self.Rialpha_current * falpha, axis=1)
		#Salpha_current = self.Salpha(params, self.size_dist)
		Salpha_sum = np.dot(falpha, self.Salpha_current)
		params = np.array([0.8, params[0]])

		grad = np.zeros_like(params)

		SHintegrand = lambda params, H : singleslope_deriv_multi(H, params)
		
		self.SalphatimesH = self.Salpha(params, SHintegrand)

		self.Rialpha_deriv = Rialphaderiv_singleslope(self.ncomp, self.nobj, self.jacobian, self.S_obj_noamp, self.chains_H, params)

		for alpha in range(self.ncomp):
			Rialphaderiv_comp = self.Rialpha_deriv[:,alpha] 
			Salpha_comp = self.Salpha_current[alpha]

			SalphatimesH_comp = self.SalphatimesH[alpha]

			gradRialpha = np.sum(Rialphaderiv_comp/self.falphaRalpha)

			grad[alpha] = gradRialpha*falpha[alpha] - self.nobj*falpha[alpha]*SalphatimesH_comp/Salpha_sum
		
		return self.pref*np.array([grad[1]])

class LogLikeSingleSlope(LogLikeGMMSingleSlope):
	def __init__(self, chains_H, selection, ncomp, pref=-1, prior = None, prior_deriv = None):
		self.chains_H = numba.typed.List(chains_H)
		self.selection = selection
		self.nobj = len(chains_H)
		self.ncomp = ncomp 
  
		self.rng = np.random.default_rng()
  
		# to avoid rewriting code, let me do something a bit silly
		# this avoids the need to rewrite the Rialpha declarations
		self.S_obj_noamp = []
		for x in self.chains_H:
			self.S_obj_noamp.append(np.zeros((len(x), 1)))
		self.S_obj_noamp = numba.typed.List(self.S_obj_noamp)

		
		self.pref = pref
		self.compute_jacobian_chain()

		self.H_bins = np.arange(5.05,9.05,0.1)

		if prior is None:
			self.prior = lambda x : 0
			self.prior_deriv = lambda x : 0
		else:
			self.prior = prior 
			self.prior_deriv = prior_deriv

	def compute_jacobian_chain(self):
		self.jacobian = []

		for i in range(self.nobj):
			self.jacobian.append(jacobian_H(self.chains_H[i]))

		self.jacobian = numba.typed.List(self.jacobian)
  
	def likelihood(self, theta):
		return super().likelihood(np.array([1.]), theta)
	def gradient_theta(self, theta):
		return super().gradient_theta(np.array([1.]), theta)
