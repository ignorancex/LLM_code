import numpy as np
import scipy as sp
from . import hilbert
from . import full

from tqdm import tqdm

def get_normalized_weights(psi):
    """
    Calculates the weights of the wave function `psi` when it's normalized.\\
    Works only for the RBM architecture with `alpha=1/4` and `N=4`.

    Returns the weights.
    """
    # 1. normalize the wave function
    psi = psi/np.linalg.norm(psi)

    # 2. calculate the thetas
    thetas = np.arccosh(1/2. * psi[:3])

    A = np.array([
    [1, 1, -1, -1],
    [1, -1, 1, -1],
    [1, -1, -1, 1]])
    # pseudoinverse
    invA = np.linalg.pinv(A)

    # 3. get weights
    weights = invA @ thetas

    # get and return weights
    return np.insert(weights, 0, 0)

def splice(w):
    """
    Separates a complex-valued vector `w` into a double-dimensional real-valued vector.
    - `w` -> [`Re(w)`, `Im(w)`]
    """
    return np.hstack((np.real(w), np.imag(w)))

def patch(w):
    """
    Rebuilds the complex vector that has been separated into a form given by the function `splice`.
    - `w` -> `w[:len/2] + 1j*w[len/2:]`
    """
    return w[:len(w)//2] + 1j*w[len(w)//2:]


class descent:
    """
    A gradient descent into the ground state of the lattice Heisenberg model.

    Variables
    =========
    - `lattice`: details about the spin lattice
    - `alpha`: network density
    - `eta`: learning rate
    - `tinv`: is the network translationally invariant? this changes the number of used parameters
    - `afm_sector`: keep only the zero-magnetization sector of the Hiblert space?
    - `scale`: multiply the wave function with this number
    - `taming`: do you want to rescale the gradient(s)? this helps if they are too huge
    - `real_parameters`: initialize parameters as real?
    - `epsilon`**\***: a function depending on the step that modifies the regularization during training
    - `normalize_initial`: how do you want to recalculate weights to normalize the wave fucntion? choose: `'analytical'`, `'numerical'`, or leave default `None`.
    - `penalize_norm`: do you want to penalize the norm during training?
    - `Lnorm_importance`**\***: a function that modifies the learning rate of normalization during training
    - `lagrange`: decides whether to use lagrange multipliers to conserve the norm during training. **NB**: setting to `True` normalizes the initial parameters with `'tdvp'`, and overwrites `refactoring` and `penalize_norm` to `False`.

    **\*Both of these must be functions. If you don't know what they mean, leave them as defaults.**

    Properties
    =========
    - `sampler`: everything regarding the full sampling
    - `states`: list of network parameter states
    - `loss`: list of loss function values (variational energy)
    - `psis`: list of wave function vectors
    - `normF`, `normSF`, `normLnorm`: list of norms of gradients of F, S^-1F, L_norm respectively
    
    Methods
    =========
    - `gradient`: calculates gradient with stochastic reconfiguration
    - `optimize`: performs the descent
    - `process (links)`: calculates the optimization curve for the (z)correlation function(s) at the given `links` list

    Regularizaiton recipe taken from the Carleo & Troyer paper.
    """

    def __init__(self, lattice:hilbert.lattice, alpha, eta,  
                 regularization = lambda s: max(0.1*0.99**s, 1e-4), # regulator?
                 phase = 0.,
                 tinv = False, normal_sampling = True, # invariance and sampling details 
                 geometric = False, # use the geometric method that i coded just for fun?
                 afm_sector = True, # restrict to antiferromagnetic sector?
                 model = 'heisenberg', # which model?
                 gauge = True,
                 real_parameters = True,
                 scale = 1.,
                 refactoring = True,
                 normalize_initial = None,
                 penalize_norm =False,
                 Lnorm_importance = lambda x: 1.,
                 lagrange = False
                 ) -> None:
        
        self.eta = eta # learning rate
        self.sampler = full.sampler(hilbert.space(lattice, afm=afm_sector, model=model, gauge=gauge), alpha, 
                                    tinvariant=tinv, normal=normal_sampling, real=real_parameters,
                                    scale=scale) # sampler containig the entire Hilbert space and the network

        # get normalized funciton weights if you want to normalize initial state
        #NOTE: this part is a bit obsolete now, but I'm still leaving it for posterity
        if normalize_initial == 'exact':
            # this should update the sampler and all of its properties, including the parameters, the energy, and the psi
            self.sampler.update(get_normalized_weights(self.sampler.psiket))
        elif normalize_initial == 'tdvp':
            self.normalise_params()   
        self.penalize_norm = penalize_norm # do you want to penalize the norm during training?
        self.Lnorm_importance = Lnorm_importance # how important is the norm loss?

        # regularization
        self.epsilon = regularization
        # regulator value
        self._epsilon = self.epsilon(0)

        # geometric?
        self.geo = geometric

        #taming
        self.refactoring = refactoring

        # use lagrange multipliers?
        if lagrange:
            self.lagrange = True
            self.penalize_norm = False
            self.refactoring = False
            self.normalise_params()
        else:
            self.lagrange = False

        # trackable quantities
        #self.states = [self.sampler.network.parameters.all * np.exp(1j*phase)] # list with all the states of the proccess
        #TODO
        self.states = [self.sampler.network.parameters * np.exp(1j*phase)]
        self.loss = [self.sampler.Evar()/4.] # values of the loss function
        self.psis = [self.sampler.psiket.copy()]

        # more trackable
        self.normF = [np.linalg.norm(np.linalg.pinv(self.sampler.S + self._epsilon*np.eye(len(self.sampler.S)))@self.sampler.F)]
        self.normLnorm = [np.linalg.norm(self.refactor_Lnorm(self.sampler.Lnorm_grad))]

    def geo_gradient(self, s=0):
        """
        Calculates the gradient using the geometric implementation.
        """

        # Quantum Fisher Matrix and the energy gradient
        S = self.sampler.S + self.epsilon(s)*np.eye(len(self.sampler.S))
        F = self.sampler.F

        # split them into real parametrization
        S = S = np.kron(S,[[1,1j],[-1j,1]])
        F = np.kron(F,[1.,-1j])
        metric = 2.*np.real(S)

        res = np.linalg.lstsq(metric, F, rcond = 1e-10)[0] # solve the problem
        res = res[::2] + 1j*res[1::2] # get back into the complex

        return res

    def gradient(self, s=0):
        """
        Calculates the analytical gradient of the loss function.\\
        This is obtained by the 'stochastic reconfiguration method' - check litterature.

        UPDATE THIS:
        The gradient itself is obtained convergently, through the `np.linalg.lstsq` function (rather than regularization and pseudoinversion).
        """
        #return np.linalg.lstsq(self.sampler.S, -self.sampler.F, rcond = 1e-10)[0]
        #return self.sampler.F

        if not self.geo:
            return np.linalg.pinv(self.sampler.S + self.epsilon(s)*np.eye(len(self.sampler.S)))@self.sampler.F
            #return np.linalg.lstsq(self.sampler.S + self.epsilon(s)*np.eye(len(self.sampler.S)), self.sampler.F, rcond = 1e-10)[0]
        else:
            return self.geo_gradient(s=s)

    #region normalizing 
    #NOTE: this part is a bit obsolete now, but I'm still leaving it for posterity

    def compute_Lnorm(self):
        """Loss for norm optimization: `L=(1-<psi|psi>)^2.` """
        return (1. - np.linalg.norm(self.sampler.psiket))**2
    
    def numerical_norm_gradient(self, epsilon = 1e-5):
        """
        Compute the numerical gradient of L_norm with respect to w.\\
        Uses the derivative scheme from Wirtinger calculus. See Torch documentation.

        - `epsilon`: infinitesimal step in parameters
        """
        w = splice(self.sampler.network.parameters.all) # save weights
        w_patched = patch(w)
        grad = np.zeros_like(w, dtype=np.float64)
        
        for i in range(len(grad)):
            # calculate positive- and negative-moved *spliced* weights
            # not the most optimal way to do things, thank ChatGPT for that
            w_pos = w.copy()
            w_neg = w.copy()
            w_pos[i] += epsilon
            w_neg[i] -= epsilon

            # update NN weights (*patched* ofc) and calculate the loss for both positive and negative
            self.sampler.update(patch(w_pos))
            loss_pos = self.compute_Lnorm()
            self.sampler.update(patch(w_neg))
            loss_neg = self.compute_Lnorm()
            
            # restore the original *patched* weights after computation
            self.sampler.update(w_patched) 

            grad[i] = (loss_pos - loss_neg) / (2 * epsilon )  # Central difference approximation

        return patch(grad)*0.5
    
    def normalise_params(self, learning_rate = 0.01, tolerance = 1e-8, maxiters = 1000):
        """
        Attempts to find a set of network parameters which represent a normalised wave function.\\
        This is done by optimizing the loss `L = (1-<psi|psi>)**2` with gradient descent.

        - `tolerance`: controls how precise the convergence should be
        - `maxiters`: manual break for the training loop if it goes into too many steps

        Fun fact: normaliSe is British, normaliZe is American English
        """
        # counter for manual loop break
        iters = 0

        while self.compute_Lnorm() > tolerance:
            # 1. calculate the gradient
            #grad = self.numerical_norm_gradient()
            grad = self.sampler.Lnorm_grad

            # 3. update weights by gradient descent
            self.sampler.update(self.sampler.network.parameters
                                - self.Lnorm_importance(iters) * learning_rate * grad)

            # 4. breaking conditions
            iters += 1
            if iters >= maxiters:
                print("loop broken after ", iters, " iterations")
                break
        print("parametric normalization complete, norm = ", np.linalg.norm(self.sampler.psiket))
        
    def refactor_Lnorm(self, gradLnorm):
        """
        Rescales the `gradLnorm` so that its norm has an upper limit corresponding to the maximum norm of the energy gradient in the whole simulation.\\
        This means that, if its norm is bigger, then reduce it to that limit, otherwise do nothing.
        """
        #NOTE: garbage code
            #self.normSF.append(np.linalg.norm(self.gradient() / (1. + self.eta*np.linalg.norm(self.gradient())) ) if self.taming else np.linalg.norm(self.gradient()))
            #self.normLnorm.append(np.linalg.norm(self.sampler.Lnorm_grad / (1. + self.eta*np.linalg.norm(self.sampler.Lnorm_grad)) ) if self.taming else np.linalg.norm(self.sampler.Lnorm_grad))
            #self.normLnorm.append(np.linalg.norm(gradLnorm * np.linalg.norm(self.sampler.F) / np.linalg.norm(gradLnorm) ) if self.taming else np.linalg.norm(self.sampler.Lnorm_grad))
            #self.normLnorm.append(np.linalg.norm(self.sampler.Lnorm_grad * max(self.normF) / np.linalg.norm(self.sampler.Lnorm_grad) ) if self.taming else np.linalg.norm(self.sampler.Lnorm_grad))
            
            #grad = grad / (1 + self.eta*np.linalg.norm(grad))
            #gradLnorm = gradLnorm / (1 + self.eta*np.linalg.norm(gradLnorm))
            #gradLnorm = gradLnorm * max(self.normF) / np.linalg.norm(gradLnorm)
        # hopefully you skipped this

        # upper limit to the norm
        upper_limit= np.max(self.normF)
        #upper_limit= self.normF[-1]
        #gradLnorm = np.linalg.pinv(self.sampler.S + self._epsilon*np.eye(len(self.sampler.S))) @ gradLnorm

        # see if it's bigger
        if np.linalg.norm(gradLnorm) >= upper_limit:
            gradLnorm = gradLnorm * upper_limit / np.linalg.norm(gradLnorm)

        #return np.linalg.pinv(self.sampler.S + self._epsilon*np.eye(len(self.sampler.S)))@gradLnorm
        return gradLnorm

    #endregion

    def lagrange_gradient(self, s = 0):
        """
        Performs an update of parameters that conserves the norm using Lagrange multipler(s).
        """
        # LHS of the EoM
        # make the matrix
        # apply the regularization only on the S matrix?
        M = np.column_stack((self.sampler.S + self.epsilon(s)*np.eye(len(self.sampler.S)), 
                             self.sampler.Lnorm_grad
                             ))
        # M = np.column_stack((self.sampler.S, 
        #                      self.sampler.Lnorm_grad))
        M = np.vstack((M, np.append(
            self.sampler.Lnorm_grad, 0.
            )))

        # or regularization of the WHOLE matrix?
        # M = M + self.epsilon(s)*np.eye(len(M))
        
        # RHS of the EoM
        F = np.append(self.sampler.F, 0.)

        # solution
        # don't forget to leave out the last element because that's the multipler
        return (np.linalg.pinv(M) @ F) [:-1]

    def optimize(self, precision = 1e-10, miniter = 100, maxiter = 10000, verbose = True):
        """
        Performs the gradient descent algorithm.

        - `precision`: convergence criterion for the loss function
        - `miniter`: minimum number of iterations
        - `maxiter`: maximum number of iterations
        """
        iter = 0 # iterations
        #params = np.real(self.states[0].copy()) + 1j*np.zeros(len(self.states[0])) # initial parameters
        params = self.states[0].copy().astype(np.complex128)

        # main loop
        while iter <= maxiter:
            iter += 1

            # if self.zigzag:
            #     self.normalise_params()
            self._epsilon = self.epsilon(iter)
            grad = self.gradient(s=iter) # gradient
            gradLnorm = self.sampler.Lnorm_grad

            #do you refactor?
            if self.refactoring:
                gradLnorm = self.refactor_Lnorm(gradLnorm)
            #do you penalize?
            if self.penalize_norm: # if you're also normalizing, penalize the norm in the gradient
                grad += gradLnorm * self.Lnorm_importance(iter)
            # if not either of those things, you are probably lagranging
            if self.lagrange:
                grad = self.lagrange_gradient(s = iter)

            params -= self.eta * grad # descent
            self.sampler.update(params) # change all the elements of the sampler (network, S, F, etc...)

            # accumulate things
            self.loss.append(self.sampler.Evar()/4.) # append the loss function
            self.states.append(params.copy())
            self.psis.append(self.sampler.psiket.copy())
            #self.normF.append(np.linalg.norm(self.sampler.F))
            self.normF.append(np.linalg.norm(self.gradient(s = iter+1)))
            self.normLnorm.append(np.linalg.norm(self.refactor_Lnorm(self.sampler.Lnorm_grad)))
            
            # sponsored messages
            if verbose and iter%100 == 0:
                print("iteration", iter, "... energy: ", self.loss[-1])
            
            # check criteria
            if iter > miniter and abs(self.loss[-1] - self.loss[-2]) < precision:
                break
        
    
    def process(self, links):
        """
        Calculates the correlation functions at each state of the process.\\
        Correlatoins are written in a `link -> list` dictionary, where the `list` is all the values of the correlation at the lattice `link` during the process.

        - `links`: an array of tuples representing links on the lattice where the calculation is performed 
        """
        self.correlations = {}
        self.zcorrelations = {}

        for link in links:
            print("calculating correlation at link", link)
            self.correlations[link] = []
            self.zcorrelations[link] = []

            with tqdm(total=len(self.states)) as pbar:
                for state in self.states:
                    self.sampler.update(state)
                    self.correlations[link].append(self.sampler.Cvar(link)/4.)
                    self.zcorrelations[link].append(self.sampler.Czvar(link)/4.)
                    
                    pbar.update(1)
    

    def nudge(self, phase_nudge = np.pi/2):
        """
        Corrects the weights of the neural network to be complex, but still have the same energy.\\
        Uses a root-finding algorithm to find the state whose energy offset is zero. This is equivalent to supervised learning.

        1. Initialize weights: take the trained weights in `states[-1]` and multiply them with `exp(1j*phase_nudge)`.
        2. Find the root: construct a function `L (w) = |E_0 - E(w)|`, where `E_0` is the pre-trained energy, and find the weights `w` for which this function has a root.

        USE only if you have performed the graident descent by the `descent()` function. Otherwise the results are senseless.
        """

        # 1. nudge the parameters
        w0 = self.states[-1] * np.exp(1j*phase_nudge)

        # 2. create the function
        def nudgeloss(w):
            # update the sampler
            self.sampler.update(w, just_parameters=True)

            # return the value
            return np.abs(self.loss[-1] - self.sampler.Evar()/4.)
        
        # 3. run the root finder
        nudged_weights = sp.optimize.newton(nudgeloss, w0, maxiter = 10000, tol = 1e-10)

        return nudged_weights



            
