import jax 
import jax.numpy as jnp

import numba
from hmc_gradient import * 

Hbins = np.linspace(3,10, 100)

# first define the jax jitted function we'll be deriving
@jax.jit
def double_powerlaw_jax(H, slope1, slope2, H_eq):
	 sl1 = jnp.power(10, -slope1*Hbins) 
	 sl2 = jnp.power(10, -slope2*Hbins + (slope2-slope1)*H_eq)
	 summed = jnp.sum(1/(sl1 + sl2)) * 0.1
	 sl1 = jnp.power(10,-slope1*H)
	 sl2 = jnp.power(10, -slope2*H + (slope2-slope1)*H_eq)
	 return 1/(sl1+sl2)/summed

#now define the gradients
grad_slope1 = jax.jit(jax.vmap(jax.grad(double_powerlaw_jax, (1)), in_axes=[0,None,None,None]))
grad_slope2 = jax.jit(jax.vmap(jax.grad(double_powerlaw_jax, (2)), in_axes=[0,None,None,None]))
grad_Heq = jax.jit(jax.vmap(jax.grad(double_powerlaw_jax, (3)), in_axes=[0,None,None,None]))

#numba versions for parallelization
@numba.jit
def double_powerlaw(H, slope1, slope2, H_eq, norm):
	 sl1 = np.power(10,-slope1*H)
	 sl2 = np.power(10, -slope2*H + (slope2-slope1)*H_eq)
	 return 1./(sl1+sl2)/norm

@numba.njit(fastmath = True, parallel = True)
def doubleslope_multi(H, params):
	z = np.zeros((len(H), int(len(params)/3)))
	for alpha in numba.prange(int(len(params)/3)):
		slope1 = params[3*alpha]
		slope2 = params[3*alpha+1]
		H_eq = params[3*alpha+2]
		norm = np.sum(double_powerlaw(Hbins, slope1, slope2, H_eq, 1.)) * 0.1
		z[:,alpha] = double_powerlaw(H, slope1, slope2, H_eq, norm)
	return z

@numba.njit(parallel = True, fastmath=True)
def Rialpha_doubleslope(ncomp, nobj, jacob, pref, chain_H, params):
	Rialpha = np.zeros((nobj, ncomp))
	for alpha in numba.prange(ncomp):
		slope1 = params[3*alpha]
		slope2 = params[3*alpha+1]
		H_eq = params[3*alpha+2]
		norm = np.sum(double_powerlaw(Hbins, slope1, slope2, H_eq, 1.)) * 0.1
		
		for i in numba.prange(nobj):
			size_chain_comp = np.log(double_powerlaw(chain_H[i], slope1, slope2, H_eq, norm))

			Rialpha[i, alpha] = np.sum(np.exp(jacob[i] + pref[i][:,alpha] + size_chain_comp))/len(chain_H[i])
	return Rialpha

@numba.njit(parallel = True, fastmath=True)
def Rialphaderiv_doubleslope(alpha, nobj, jacob, pref, deriv):
	Rialpha = np.zeros((nobj))
	for i in numba.prange(nobj):
		Rialpha[i] = np.sum(np.exp(jacob[i] + pref[i][:,alpha]) * deriv[i]) / len(deriv[i])

	return Rialpha

class LogLikeGMMDoubleSlope(LogLikeGMMSingleSlope):
	def __init__(self, gmm, chains_color, chains_H, selection, ncomp, pref=-1, prior = None, prior_deriv = None):
		super().__init__(gmm, chains_color, chains_H, selection, ncomp, pref, prior, prior_deriv)
		indexes = [len(i) for i in chains_H]
		self.indexes = np.cumsum(indexes)[:-1]

		self.chains_flat = np.concatenate(chains_H)
		if prior is None:
			self.prior = lambda x : 0
			self.prior_deriv = lambda x : np.zeros(6)
		else:
			self.prior = prior 
			self.prior_deriv = prior_deriv

	def size_dist(self, params, H):
		return doubleslope_multi(H, params)

	def Rialpha(self, params):
		return Rialpha_doubleslope(self.ncomp, self.nobj, self.jacobian, self.S_obj_noamp, self.chains_H, params)

	def Salpha_array(self, alpha, array):
		integrand = self.selection[alpha] * array
		return np.sum((integrand[1:] + integrand[:-1])*np.diff(self.H_bins))/2

	def gradient_theta(self, falpha, params):
		## Assumes we have already computed Rialpha and Salpha
		self.Rialpha_current = self.Rialpha(params)

		self.Salpha_current = self.Salpha(params, self.size_dist)

		self.falphaRalpha = np.sum(self.Rialpha_current * falpha, axis=1)
		#Salpha_current = self.Salpha(params, self.size_dist)
		Salpha_sum = np.dot(falpha,self.Salpha_current)

		#this is where it gets tricky - this is now 3n d  
		grad = np.zeros_like(params)		

		for alpha in range(self.ncomp):
			#unpack
			slope1 = params[3*alpha]
			slope2 = params[3*alpha+1]
			Heq = params[3*alpha+2]

			# evaluate jax gradients at each parameter over the chains
			deriv_chain_slope1 = np.array(grad_slope1(self.chains_flat, slope1, slope2, Heq))
			deriv_chain_slope2 = np.array(grad_slope2(self.chains_flat, slope1, slope2, Heq))
			deriv_chain_Heq =  np.array(grad_Heq(self.chains_flat, slope1, slope2, Heq))

			# now calculate the Rialpha sums
			Rialpha_deriv_slope1 = Rialphaderiv_doubleslope(alpha, self.nobj, self.jacobian, \
								self.S_obj_noamp, numba.typed.List(np.split(deriv_chain_slope1, self.indexes)))
			Rialpha_deriv_slope2 = Rialphaderiv_doubleslope(alpha, self.nobj, self.jacobian, \
								self.S_obj_noamp, numba.typed.List(np.split(deriv_chain_slope2, self.indexes)))
			Rialpha_deriv_Heq = Rialphaderiv_doubleslope(alpha, self.nobj, self.jacobian, \
								 self.S_obj_noamp, numba.typed.List(np.split(deriv_chain_Heq, self.indexes)))

			gradRialpha_slope1 = np.sum(Rialpha_deriv_slope1/self.falphaRalpha)
			gradRialpha_slope2 = np.sum(Rialpha_deriv_slope2/self.falphaRalpha)
			gradRialpha_Heq = np.sum(Rialpha_deriv_Heq/self.falphaRalpha)

			# evaluate jax gradients for the S integrals
			deriv_S_slope1 = grad_slope1(self.H_bins, slope1, slope2, Heq)
			deriv_S_slope2 = grad_slope2(self.H_bins, slope1, slope2, Heq)
			deriv_S_Heq = grad_Heq(self.H_bins, slope1, slope2, Heq)

			# integrate
			S_slope1 = self.Salpha_array(alpha, deriv_S_slope1)
			S_slope2 = self.Salpha_array(alpha, deriv_S_slope2)
			S_Heq = self.Salpha_array(alpha, deriv_S_Heq)


			prior_deriv = self.prior_deriv(params[alpha])

			#now put things in place
			grad[3*alpha] = gradRialpha_slope1*falpha[alpha] - self.nobj*falpha[alpha]*S_slope1/Salpha_sum + prior_deriv[3*alpha]
			grad[3*alpha+1] = gradRialpha_slope2*falpha[alpha] - self.nobj*falpha[alpha]*S_slope2/Salpha_sum + prior_deriv[3*alpha]
			grad[3*alpha+2] = gradRialpha_Heq*falpha[alpha] - self.nobj*falpha[alpha]*S_Heq/Salpha_sum + prior_deriv[3*alpha]


		return self.pref*grad


