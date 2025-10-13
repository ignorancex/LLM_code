import numpy as np
import scipy as sp
from . import hilbert
from . import full

from tqdm import tqdm

class evolution:
    """
    Initializes and runs the dynamics of the given Heisenberg model represented with an RBM.\\
    Contains the methods for calculating all the correlation functions.

    Variables
    =========
    - `lattice`: details about the spin lattice
    - `initial`: starting network parameter array
    - `steps`: how many steps in the dynamics
    - `end_time`: time interval for the dynamics

    Kwargs
    ----------
    - `tinv`: is the network translationally invariant? this changes the number of used parameters
    - `perturbator`: a time-dependent function to calculate the perturbation strength
    - `integrator`: update calculation algorithm - choice between `'heun'`, `'implicit_midpoint'`
    - `formulation`: decide between different ways to calculate derivatives: `'regularization'`, `'diagonalization'`, `'geometric'`
    - `afm_sector`: bool; decides if you keep only the zero-magnetization sector of the Hiblert space
    - `model`: ground state model, choose between: `'heisenberg'`, `'ising'`
    - `gauge`: apply gauge transformation?
    - `noise_level`: controls how much artificial noise you introduce into the energy gradient
    - `scale`: multiply the wave fucntion with this number

    Properties
    =========
    - `sampler`: everything regarding the full sampling
    - `states`: list of network parameter states
    - `energies`: list of loss (variational) energy values
    - `correlations`: dictionary of correlation functions, keyed by `(i,j)` tuples of lattice indices
        
    Methods
    =========
    - `derivative`: calculates the derivative in the TDVP EoM
    - `integrate`: predicts the next step in the dynamics
    - `run`: evolves the system in time, given the settings of the object
    - `process_links (links)`: calculates the time dependence curve for the (z)correlation function(s) at the given `links` list, \\
    for every state in `states`
    - `process_spectrum`: calculates the spectrum of the S-matrix for every state in `states`   
    """
    def __init__(self, lattice:hilbert.lattice, alpha, initial:np.array, steps:int, end_time, 
                 tinv = False,  normal_sampling = True,
                 perturbator = lambda t: 0.1, start = 0.0,
                 taming = True, integrator = 'heun', #geometric = True, # integration particulars
                 formulation = 'geometric',
                 afm_sector = True, 
                 model = 'heisenberg', # use which model?
                 gauge = True,
                 noise_level = 0.,
                 scale = 1.,
                 regulator = 1e-4,
                 lagrange_norm = False
                 ) -> None:
        
        # details of the dynamics
        self._start = start
        self._end = end_time
        self.steps = steps # total number of steps
        self.dt = (self._end-self._start)/steps # time step

        # more details of the dynamics
        self._taming = taming # tame the derivatives or not?
        self._integrator = integrator # method of calculating updates (string)
        self._formulation = formulation
        self._lagrange_norm = lagrange_norm # try to conserve norm in the geometric method?


        # lattice, Hilbert space, and network - all in one
        self._alpha = alpha
        self.sampler = full.sampler(hilbert.space(lattice, afm=afm_sector, model=model, gauge=gauge), self._alpha, 
                                    tinvariant=tinv, normal=normal_sampling, noise=noise_level,
                                    scale=scale)
        self.sampler.update(initial) #don't forget to update the sampler into the initial parameters

        # perturbation - remember, this is a function, so it must be Callable
        self.perturbator = perturbator

        # time-dependent things
        self.times = [self._start - self.dt] 
        self.states = [initial]
        self.energies = [self.sampler.Evar()/4.]
        self.psis = [self.sampler.psiket.copy()]
        self.residuals = [] # residual distance is also time-dependent becaus the Hamiltonian and the energy gradient are
        
        # manifold distance measures
        self.metric_velocity = [] # velocity along the trajectory on the manifold (also time-dependent)
        self.gradient_velocity = [] # same as above, but defined from RHS of the TDVP EoM
        self.arc = [] # cummulative travelled arc length during the evolution
        self.path = [] # same as above, but for gradient velocity


        self.regulator = regulator # in case you're using regularization

        # initialize correlation (and such) dictionaries
        self._init_correlations()
        

    def _init_correlations(self):
        # full correlations
        self.correlations = {}

        # z correlations and entaglement contribution
        self.zcorrelations = {}
        self.tangle = {} # taken from Giammarco's notes - not general and only valid for 2x2 systems

    def _geometric_tdvp(self, parameters):
        """
        Calculates the derivative using the geometric formulation.

        1. Transform the S-matrix and F-vector into real-like parametrization.
        2. Calculate the geometric elements: metric, symplectic form.
        3. Formulate and solve the AX=B equation using the `np.linalg.lstsq` convergent solver.
        4. Get rid of the Langrange multipliers and repatch back into complex-like parametrization.
        """
        # first update parameters
        self.sampler.update(parameters)

        # Quantum Fisher Matrix and the energy gradient
        S = self.sampler.S
        F = self.sampler.F
        # and the Lnorm gradient
        Ln = self.sampler.Lnorm_grad

        # split them into real parametrization
        S = np.kron(S,[[1,1j],[-1j,1]])
        F = np.kron(F,[1.,-1j])
        Ln = np.kron(np.conjugate(Ln), [1,1j])
        Ln = np.pad(Ln, (0, len(Ln)), mode='constant', constant_values=0)
        metric = 2.*np.real(S)
        symform = 2.*np.imag(S)

        # A X = B problem
        B = np.kron([0,1],F)
        A = np.kron([[1,0],[0,0]],2.*metric) + np.kron([[0,1],[0,0]],symform.T) + np.kron([[0,0],[1,0]],symform)
        #add the norm gradient
        if self._lagrange_norm:
            A = np.column_stack((A, Ln))
            A = np.vstack((A, np.append(Ln, 0)))
            B = np.append(B, 0.)
        A = A.astype(complex)


        # calculate the derivative
        der = np.linalg.lstsq(A, -B, rcond = 1e-10)[0]

        # tame if taming is on
        if self._taming:
            der =  der / (1+self.dt*np.abs(der))
            #der =  der / (1+self.dt*np.linalg.norm(der))
        
        # return to the complex parametrization
        try: # try to get rid of the Lagrange multipliers
            if self._lagrange_norm:
                der = der[:-1] # to get rid of the multiplier for norm conservtion
            der = der[:int(len(der)/2)] # lambdas
        except: # it's only problematic if the dimensions of the array are somehow wrong (but the code is quite proof to this)
            print("Dimension of array must be a multiple of 4. Dimension is {}.".format(len(der)))
        
        der = der[::2] + 1j*der[1::2] # get back into the complex

        return der
    
    def _standard_tdvp(self, parameters):
        """
        Calculates the derivative of the `parameters` vector in standard TDVP fromulation using the `np.linalg.lstsq` method.
        """
        self.sampler.update(parameters)
        #der = np.linalg.lstsq(self.sampler.S, -1j*self.sampler.F, rcond = 1e-10)[0]
        der = -1j*np.linalg.pinv(self.sampler.S + self.regulator*np.eye(len(self.sampler.S))) @ self.sampler.F

        if self._taming:
            return der / (1+self.dt*np.abs(der))
            #return der / (1+self.dt*np.linalg.norm(der))
        else:
            return der
    
    def _diagonal_tdvp(self, parameters, criterion = 1e-8):
        """
        Calculates the derivative using diagonalization.
        1. diagonalizes the S-matrix,
        2. transforms everything into the diagonal basis,
        3. removes zero eigenvalues and constituend components,
        4. updates without the nullspace,
        5. transforms back.
        """
        # update the sampler
        self.sampler.update(parameters)

        # diagonalize the S matrix
        evals, evecs = np.linalg.eig(self.sampler.S)
        idx = np.argsort(np.abs(evals)) # instruction how to order them according to the eigenvalues
        evals = evals[idx] #sorts them by those indices
        evecs = evecs[:,idx] #also, but since it's a matrix and eigenvectors are collumns, sort by second index
        evecs_inv = np.linalg.inv(evecs)
        slicer = np.searchsorted(np.abs(evals), criterion, side='right')

        # transform everything
        #Sdiag = np.diag(evals)
        Sdiag = (evecs_inv @ self.sampler.S) @ evecs
        Fdiag = evecs_inv @ self.sampler.F

        # slice things
        Sdiag = Sdiag[slicer:, slicer:]
        Fdiag = Fdiag[slicer:]

        # now calculate the update
        # it's just the TDVP update on the nonzero-subspace
        der = -1j * np.linalg.inv(Sdiag) @ Fdiag

        #fill it with zeros to match the length
        der = np.pad(der, (len(parameters) - len(der), 0))
        #transform it back to the original basis and return
        der = evecs@der

        if self._taming:
            return der / (1+self.dt*np.abs(der))
            #return der / (1+self.dt*np.linalg.norm(der))
        else:
            return der
      
    def derivative (self, parameters):
        """
        Calculates the derivative using either the standard TDVP formulation or the geometric method.
        """
        if self._formulation == 'geometric':
            return self._geometric_tdvp(parameters)
        elif self._formulation == 'diagonalization':
            return self._diagonal_tdvp(parameters)
        else:
            return self._standard_tdvp(parameters)


    def integrate(self, state):
        """
        Predicts the parameter vector in the next step of the time integration, from the initial `state`, using the selected integrator.\\
        Integrator choices:
        - `"heun"`: 2nd order explicit (RK2)
        - `"implicit_midpoint"`: Newton-Rhapson implicit method, solution obtained by the `sp.optimize.newton` root finder

        NOTE: taming is governed by the class property and affects the calculation of derivatives
        """
        if self._integrator == 'heun':
            der = self.derivative(state) # euler step
            wtilda = state + self.dt*der
            dertilda = self.derivative(wtilda) # corrector step
            der = 0.5 * (der + dertilda) # derivative in total
            newstate = state + self.dt*der

        elif self._integrator == 'implicit_midpoint':
            implicit_equation = lambda wn: wn - state - self.dt*self.derivative(0.5*(wn + state))
            #newstate = sp.optimize.newton(implicit_equation, state, maxiter = 10000, tol = 1e-10)
            newstate = sp.optimize.newton(implicit_equation, (np.random.randn(len(state)) + 1j*np.random.randn(len(state)))*0.1, maxiter = 10000, tol = 1e-10)
        
        else:
            print("Invalid integration method given.")
            return

        return newstate


    def run(self):
        """
        Evolve the system in time using the TDVP EoM.

        Steps:
        1. Perturb the Hamiltonian.
        2. Calculate the next parameter preiction.
        3. Pile up the states and the time-dependent observables (energy)
        """

        w = self.states[-1].copy() # initial array

        print("running simmulation...")
        with tqdm(total=self.steps) as pbar:
            for s in range(self.steps):
                time = self._start + s*self.dt # hoe laat is het?
                # perturb the Hamiltonian at that time
                self.sampler.hspace.perturb(self.perturbator(time))

                wnew = self.integrate(w.copy()) # calculate the new state
                
                # append metric things before update
                self.residuals.append(self.sampler.residual(w, (wnew - w)/self.dt)) # residual distance
                self.metric_velocity.append(self.sampler.metric_velocity(w, (wnew - w)/self.dt)) # the integrand in the metric distance
                self.arc.append(np.sum(self.metric_velocity) * self.dt) # travelled arc length
                self.gradient_velocity.append(self.sampler.gradient_velocity_rt(w, (wnew - w)/self.dt))
                self.path.append(np.sum(self.gradient_velocity) * self.dt)
                
                # update the sampler with new parameters
                self.sampler.update(wnew) 

                # append things after update
                self.states.append(wnew.copy())
                self.energies.append(self.sampler.Evar()/4.)
                self.times.append(time)
                self.psis.append(self.sampler.psiket.copy())

                w = wnew.copy()
                pbar.update(1)
            
            # update the information about the run time
            self.runtime = pbar.format_dict['elapsed']
    
    #region processing related methods

    def _process(self, links):
        """
        OLD (unptimized) FUNCTION:
        Calculates the correlation functions at each state of the process.\\
        Correlatoins are written in a `link -> list` dictionary, where the `list` is all the values of the correlation at the lattice `link` during the process.

        - `links`: an array of tuples representing links on the lattice where the calculation is performed 
        """

        for link in links:
            print("calculating correlation at link", link)
            
            # initialize dictionaries
            self.correlations[link] = []
            self.zcorrelations[link] = []

            # check if the matrices exist - if not, make them
            if link not in self.sampler.hspace.correlations:
                self.sampler.hspace.make_correlation(link)

            with tqdm(total=len(self.states)) as pbar:
                for state in self.states:
                    # 1. update the sampler to carry the information about this state
                    self.sampler.update(state)

                    # 2. calculate correlations at the link
                    self.correlations[link].append(self.sampler.Cvar(link)/4.)
                    self.zcorrelations[link].append(self.sampler.Czvar(link)/4.)
                    
                    pbar.update(1)
                
                # turn into numpy arrays
                self.correlations[link] = np.array(self.correlations[link])
                self.zcorrelations[link] = np.array(self.zcorrelations[link])
                self.tangle[link] = -1*np.log(0.25 + 0.75*self.zcorrelations[link]**2)

    def process_links(self, links):
        """
        Calculates the correlation functions at each state of the process.\\
        Correlatoins are written in a `link -> list` dictionary, where the `list` is all the values of the correlation at the lattice `link` during the process.

        - `links`: an array of tuples representing links on the lattice where the calculation is performed 
        """

        print("calculating correlations at links:", links)
        for link in links:
            
            # initialize dictionaries
            self.correlations[link] = []
            self.zcorrelations[link] = []

            # check if the matrices exist - if not, make them
            if link not in self.sampler.hspace.correlations:
                self.sampler.hspace.make_correlation(link)
                print("> created correlation matrix at link", link)

        with tqdm(total=len(self.states)) as pbar:
            for state in self.states:
                # 1. update the sampler to carry the information about this state
                self.sampler.update(state)

                # 2. calculate correlations at links
                for link in links:
                    self.correlations[link].append(self.sampler.Cvar(link)/4.)
                    self.zcorrelations[link].append(self.sampler.Czvar(link)/4.)
                
                # update the loading bar
                pbar.update(1)

        for link in links:
            # turn into numpy arrays
            self.correlations[link] = np.array(self.correlations[link])
            self.zcorrelations[link] = np.array(self.zcorrelations[link])
            self.tangle[link] = -1*np.log(0.25 + 0.75*self.zcorrelations[link]**2)
    
    def process_spectrum(self):
        """
        Calculates the time dependence of the S-matrix spectrum.

        Returns a matrix where each row corresponds to one eigenvalue dynamics.
        """
        print("calculating spectrum dynamics of the S-matrix")

        self.spectrum = []

        with tqdm(total=len(self.states)) as pbar:
            for state in self.states:
                # 1. update the sampler to carry the information about this state
                self.sampler.update(state)

                # 2. calculate and write down the spectrum
                self.spectrum.append(self.sampler.sptectrumS())
                
                # update the loading bar
                pbar.update(1)
        
        # reformat
        self.spectrum = np.array(self.spectrum).T
    
    def read_parameters(self, inputs, links = None):
        """
        Takes a time series of network parameters in `inputs` and analizes them with the NQS machinery.\\
        Assumes that the first state (index `0`) is the first states in the dynamics. Ground state is sent in the object initialization.

        - `links`: list of links to proccess

        Does not reset the calculation - use only on an object without pre-existing data.\\
        Format inputs as a list of (or an array) of `numpy` arrays: `[parameters0, parameters1, ...]`.
        """
        # do a check first
        if len(inputs) != (self.steps):
            raise ValueError("The dimension of the input ({:}) does not match the number of steps ({:}).".format(len(inputs), self.steps+1))

        # main loop
        print('reading parameters...')
        with tqdm(total=self.steps) as pbar:
            for s in range(self.steps):
                # time related stuff
                time = self._start + s*self.dt
                self.times.append(time)
                self.sampler.hspace.perturb(self.perturbator(time))

                # append / update
                self.states.append(inputs[s].copy()) # append the appropriate state
                self.sampler.update(self.states[-1].copy()) # update the sampler

                # time-sensitives
                self.energies.append(self.sampler.Evar()/4.)
                self.psis.append(self.sampler.psiket.copy())

                # visual
                pbar.update(1)

        # links magic
        if links != None:
            links_to_process = links
        else:
            links_to_process = self.sampler.hspace.grid.links

        # then process links and the spectrum
        self.process_links(links_to_process)
        self.process_spectrum()

    #endregion

    #NOTE: don't bother reading this unless you actually need to fit NQS onto ED dynamics
    #NOTE: it's just a bucnh of code at this point
    # region fitting dynamics to ED using infidelity

    def infidelity(self, v1, v2):
        """
        This must be the millionth time I define this function.
        """
        return 1. - np.real( np.vdot(v1,v2) * np.vdot(v2,v1) / (np.vdot(v1,v1) * np.vdot(v2,v2)) )

    def wavefunction_infidelity(self, known_psi, network_params):
        """
        It's the same as infidelity, except that one of the wave functions is calculated from the given **parameters** of the network.
        """
        
        self.sampler.update(network_params)
        return self.infidelity(self.sampler.psiket, known_psi)
    
    def compute_gradient(self, loss_func, params, epsilon=1e-6):
        """Numerically computes the gradient of the function with respect to the given parameters.
        Epsilon is the offset in all directions. Takes into account the generally complex parameters
        and functions."""

        num_params = len(params)
        gradient = np.zeros(num_params, dtype=complex)
        params = params.astype(complex)

        for i in range(num_params):
            # originally by chatGPT
            # params_plus_epsilon = np.copy(params)
            # params_plus_epsilon[i] += epsilon
            # loss_plus_epsilon = loss_func(params_plus_epsilon)
            # gradient[i] = (loss_plus_epsilon - loss_func(params)) / epsilon

            #complex implementation by me
            params_plus_epsilon = np.copy(params)
            params_plus_epsilon[i] += epsilon
            loss_plus_epsilon = loss_func(params_plus_epsilon)

            params_plus_epsilonj = np.copy(params)
            params_plus_epsilonj[i] += 1j*epsilon
            loss_plus_epsilonj = loss_func(params_plus_epsilonj)

            gradient[i] = (loss_plus_epsilon + 1j*loss_plus_epsilonj - (1+1j)*loss_func(params)) / epsilon

        return gradient
    
    def analytical_infidelity_gradient(self, parameters, reference_wf):
        """
        An attempt of calculating the gradient of infidelity analytically.

        **NB**: assumes that the reference function (from ED) is normalized
        """
        # always update the sampler first
        self.sampler.update(parameters)
        # one of the constituent parts
        Ok_per_spin = np.array([self.sampler.network.Ok(conf) for conf in self.sampler.hspace.configs]).T
        right_sum = np.array([
            np.sum([np.conjugate(Ok_per_spin[k][i]) * np.conjugate(self.sampler.psiket[i]) * reference_wf[i] for i in range(len(self.sampler.hspace.configs)) ]) 
            for k in range(len(parameters))
        ]) # this should now be a k-vector

        # return a long-ass formula
        return (np.conjugate(self.sampler.Ok) * np.vdot(self.sampler.psiket, reference_wf) -right_sum) * np.vdot(reference_wf, self.sampler.psiket) / np.vdot(self.sampler.psiket, self.sampler.psiket)


    def gradient_descent_with_convergence(self, ref_state, 
                                        initial_params, 
                                        learning_rate=0.01, threshold=1e-6, miniter = 100, maxiter=1000,
                                        learning_rate_decay=1.,
                                        artificial_noise = 1e-4,
                                        verbose = False):
        """Performs a gradient descent algorithm with convergence criterion for stoping.

        Variables
        -------------------
        - `ref_state`: state you want to fit your wavefunction to
        - `initial_params`: starting parameter array

        Kwargs
        ----------------
        - `learning_rate`: default is 0.01
        - `threshold`: convergence criterion for the descent, default is 1e-6 (loss function)
        - `miniter`, `maxiter`: hard limitations for iterations
        - `learning_rate_decay`: if supplied, scales down the learning rate every iteration, default values 1 means no scaling
        - `artificial_noise`: add Gaussian noise with this much intensity into the gradient
        - `distance_target`: measure the straight-line distance towards this target, by default `None` so it doesn't measure it

        Returns the evolution of parameters, the loss function, the psi norm, metric velocities, travelled arc, residual distances, and optionally straight-line distances to the provided target.
        """
        
        params = np.array(initial_params, dtype=complex)
        iteration = 0

        #accumulated values
        loss_values = [self.wavefunction_infidelity(ref_state, params)]
        param_list = [params.copy()]
        norm_values = []
        

        while iteration < maxiter:
            gradient = self.analytical_infidelity_gradient(params, ref_state) # calculate gradient
            gradient = full.add_noise(gradient, eta = artificial_noise) # add artifficial noise to the gradient to jostle it out of a local minimum
            invS = np.linalg.pinv(self.sampler.S + 1.e-4*np.eye(len(self.sampler.S))) # calculate the inverse of the S matrix

            # update the parameters
            params -= learning_rate * invS@gradient # update the parameters

            # append stuff
            current_loss = self.wavefunction_infidelity(ref_state, params)
            loss_values.append(current_loss)
            param_list.append(params.copy())
            norm_values.append(np.linalg.norm(self.sampler.psiket))

            if iteration >= miniter and np.abs(current_loss) < threshold:
                break

            iteration += 1

            # Apply learning rate decay
            learning_rate *= learning_rate_decay

            # messages
            if verbose and iteration % 100 == 0:
                print("iteration ", iteration, "... infidelity = ", current_loss)

        #return params, loss_values
        return param_list, loss_values, norm_values

    def fit_network(self, state, initial_guess=None, 
                    eta=0.1, epsilon=1e-6, 
                    maxit=10000, add_noise = 1e-4,
                    speak = False):
        """
        Calculates the neural network parameters that describe the Hilbert space state.
        Works by fitting the parameters with a gradient descent, using infidelity as loss.
        Use only if the object has no data written.

        Parameters
        ---------------
        - `state` is the Hilbert space state that you're fitting to,
        - `initial_guess` is random if you don't supply it (default None),
        - `eta` is the learning rate (default 0.01),
        - `epsilon` is the precision requirement (default 1e-16),
        - `maxit` is the maximum number of iterations (default 10000).

        Returns the final parameter state, the final loss, and the number of iterations reached.
        """
            
        if initial_guess is None:
            # initialize a random parameter vector
            # dimension must match the initial vector in the class
            initial_guess =( np.random.randn(len(self.states[0])) + 1j*np.random.randn(len(self.states[0])) ) * 0.1
        # update the sampler with it
        self.sampler.update(initial_guess)
        
        #loss = lambda params: self.wavefunction_infidelity(state, params)

        history = self.gradient_descent_with_convergence(state, #loss, 
                                                         initial_guess, 
                                                         learning_rate=eta, 
                                                         threshold=epsilon, 
                                                         maxiter=maxit,
                                                         artificial_noise=add_noise,
                                                         verbose=speak)

        return history[0][-1], history[1][-1], len(history[1])
    
    def fit_dynamics(self, states, start_from_last = True, 
                    noise = 1e-4, criterion = 1e-9,
                    postprocess = True, links = None,
                    verbose = False
                    ):
        """
        Fits the entire series of wave functions (presumably from ED) describing dynamics by calling the `fit_network` functin for each of them.\\
        The series is given in the `states`.
        - `start_from_last`: decide whether to start from the last state in the fit
        - `postprocess`: decide if you want to analyse correlations and spectra

        The fit is performed by gradient descent, using infidelity as loss function.

        Returns two arrays: `losses` containing final infidelities, and `convergence_steps` containing steps it took to fit at that time.

        **NB**: be mindful of whether you pass ground states or not. The first state of this class is passed in the `initial` argument.
        It is assumed that the first state in `states` is the ground state. Fitting this to the network will overwrite the first state in `self.states`.

        Kwargs
        ---
        - `noise`: strength of the artificial gaussian noise you add into the infidelity gradient
        - `criterion`: convergence criterion, default `1e-9` infidelity
        - `link`: which links to process
        """

        # overwrite the initial state with the ground state fit
        # this will probably take the longest, as it is converging from a random state
        fit = self.fit_network(states[0], initial_guess=self.states[0], speak = True, epsilon = criterion)
        self.states[0] = fit[0].copy()
        losses = [fit[1]*1.] # final loss of the fit
        convergence_steps = [fit[2]*1] # steps it took
        print("initial fit done in ", convergence_steps[0], "steps, infidelity = ", losses[0])

        # go over all states
        print("fitting to inputs...")
        with tqdm(total=len(states[1:])) as pbar:
            for (i, st) in enumerate(states[1:]):
                fit = self.fit_network(st, # fit to this state
                                    initial_guess=self.states[-1] if start_from_last else None,
                                    maxit = 500000, #if i remember well, i do need this much
                                    eta = 0.1,
                                    epsilon = criterion,
                                    add_noise=noise,
                                    speak = verbose) # guess from the previous state
                
                # update everything
                self.times.append(self._start + i*self.dt)
                self.states.append(fit[0])
                self.sampler.update(fit[0])
                self.psis.append(self.sampler.psiket)
                losses.append(fit[1]*1.)
                convergence_steps.append(fit[2]*1)

                # visual
                pbar.update(1)

        # process
        # links magic
        if links != None:
            links_to_process = links
        else:
            links_to_process = self.sampler.hspace.grid.links

        # then process links and the spectrum
        if postprocess:
            self.process_links(links_to_process)
            self.process_spectrum()
        
        # return these just for good measure
        return losses, convergence_steps

    # endregion