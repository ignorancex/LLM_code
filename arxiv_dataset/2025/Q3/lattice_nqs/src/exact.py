import numpy as np
from . import hilbert

class ED:
    """
    Contains the exact diagonalization dynamics of a spin system obtained using quantum-mechanical definitions.

    Variables
    ------
    - `lattice`: a hilbert.lattice-class object to specify the lattice details
    - `steps`: how many steps in the dynamics
    - `end_time`: time interval for the dynamics
    - `model`: which Hamiltonian to use? choose: `'heisenberg'`, `'ising'`
    - `driving`: which model should you use for the driving? choose: `'light'`, `'J1J2'`, `'JQ'`
    - `perturbator`: a time-dependent function to calculate the perturbation strength
    
    Properties
    -------
    - `dt`: time step
    - `states`: list of states, in time
    - `observables`: a dictionary containing the dynamics of all observables defined in the Hilbert space \\
        (access the correlation functions the same way you would access them in `hilbert.space` class)

        
    Methods
    -------
    - `gather_energy(state)`, `gather_correlations(link, state)`: they calculate those observables in a certain quantum state
    - `run()`: runs the dynamics
    - `process()`: calculates all the observables from all the states of the dynamics

    """
    def __init__(self, lattice:hilbert.lattice, steps:int, end_time, 
                 perturbator = lambda t: 0.1, # perturbation details
                 initial = None, start = 0.0, # dynamics particulars
                 model = 'heisenberg', # which model?
                 driving = 'light',
                 gauging = True,
                 afm = True,
                 tacit = True # tacit means quiet,
                 ) -> None:
        
        # define the lattice
        self.lattice = lattice
        # make the Hilbert space
        self._hspace = hilbert.space(self.lattice, model=model, driving=driving, gauge = gauging, afm=afm)

        # details of dynamics
        self._start = start
        self._end = end_time
        self.steps = steps
        self.dt = (self._end-self._start)/steps

        # remember, this is a function, so it must be Callable
        self.perturbator = perturbator
        
        # initial state (decision)
        self.init = initial

        self.states = [] # time-ordered list of quantum state vectors
        self.times = [] # times of the run
        self._initialize_observables() # quite self explainatory

        self.tacit = tacit

    def _initialize_observables(self):
        """
        Initialize all the observables as a dictionary. Contains:
        - energy
        - correlations (all links)
        - z-correlations (all links)
        """
        self.observables = {}
        self.observables['energy'] = [] # array

        # correlations
        self.observables['correlations'] = {} # dictionary
        self.observables['zcorrelations'] = {}
        self.observables['ycorrelations'] = {}
        self.observables['xcorrelations'] = {}
        self.observables['tangles'] = {} # for entanglement
        self.observables['plusminus'] = {}

        # single site spins
        self.observables['sites'] = {}
        self.observables['zsites'] = {}
        self.observables['xsites'] = {}
        self.observables['ysites'] = {}

        # neel vectors on bonds
        self.observables['xneel'] = {}
        self.observables['yneel'] = {}
        self.observables['zneel'] = {}

        # now initialize an empty list as every dictionary value
        for link in self._hspace.grid.links:
            self.observables['correlations'][link] = []
            self.observables['zcorrelations'][link] = []
            self.observables['ycorrelations'][link] = []
            self.observables['xcorrelations'][link] = []
            self.observables['tangles'][link] = []
            self.observables['xneel'][link] = []
            self.observables['yneel'][link] = []
            self.observables['zneel'][link] = []
        
        # also do that for single site spin functions
        # NOTE: this is wrong. these functions only make sense in the whole Hilbert space, outside of the only-afm sector
        for i in range(self.lattice.Lx * self.lattice.Ly): # go over the number of sites
            self.observables['sites'] [(i,)] = []
            self.observables['zsites'] [(i,)] = []
            self.observables['xsites'] [(i,)] = []
            self.observables['ysites'] [(i,)] = []
        
        # Neel plus and minus vectors
        for i in range(self.lattice.Lx * self.lattice.Ly):
            for j in range(self.lattice.Lx * self.lattice.Ly):
                self.observables['plusminus'] [(i,j)] = []

    
    #region gathering functions

    def gather_energy(self, state:np.ndarray):
        """Calculate the energy in a given `state` vector."""
        return np.real(np.vdot(state, self._hspace.H@state)) / 4.

    def gather_correlation(self, link:tuple, state:np.ndarray):
        """Calculate the expectation values of correlation function(s) at the given `link` , in a given `state` vector."""
        corr = np.real(np.vdot(state, self._hspace.correlations[link]@state))
        zcorr = np.real(np.vdot(state, self._hspace.zcorrelations[link]@state))
        xcorr = np.real(np.vdot(state, self._hspace.xcorrelations[link]@state))
        ycorr = np.real(np.vdot(state, self._hspace.ycorrelations[link]@state))
        return corr, zcorr, xcorr, ycorr
    
    def gather_sitespins(self, site, state:np.ndarray):
        """
        Calculates the values of all the spin functions at individual sites.
        """
        xsitespin = np.vdot(state, self._hspace.xsitespins[site]@state)
        ysitespin = np.vdot(state, self._hspace.ysitespins[site]@state)
        zsitespin = np.vdot(state, self._hspace.zsitespins[site]@state)

        return xsitespin, ysitespin, zsitespin
    
    def gather_neel(self, bond, state):
        """
        Calculates all the Neel vector elements on a `bond`.\\
        NOTE: Makes sense only when the full Hilbert space is sampled.
        """
        xneel = 0.5 * np.vdot(state, (self._hspace.xsitespins[(bond[1],)] - self._hspace.xsitespins[(bond[0],)]) @ state)
        yneel = 0.5 * np.vdot(state, (self._hspace.ysitespins[(bond[1],)] - self._hspace.ysitespins[(bond[0],)]) @ state)
        zneel = 0.5 * np.vdot(state, (self._hspace.zsitespins[(bond[1],)] - self._hspace.zsitespins[(bond[0],)]) @ state)

        return xneel, yneel, zneel

    def gather_plusminus(self, i, j, state:np.ndarray):
        """
        Calculates the values of the N+- vector on the bond.\\
        Refer to Lukas' notes for clarification.
        """
        return np.vdot(state,
                       ((self._hspace.xsitespins[(i,)] + 1j*self._hspace.ysitespins[(i,)])@(self._hspace.xsitespins[(j,)] - 1j*self._hspace.ysitespins[(j,)]))@state)

    def _calculate_observables(self, state:np.ndarray):
        """
        Accumulates all the observable expectation values in the given `state` into the observables dictionary.

        NB: Works well only for time-independent observables. For example, Hamiltonian is in general time-dependent, so energy calculation will not be right if only post-processed. 
        """
        for link in self._hspace.grid.links: # go through all the links
            # correlations
            corr, zcorr, xcorr, ycorr = self.gather_correlation(link, state) # calculate expectation values first
            self.observables['correlations'][link].append(np.real(corr/4.)) # then append them to the appropriate elements
            self.observables['zcorrelations'][link].append(np.real(zcorr/4.))
            self.observables['xcorrelations'][link].append(np.real(xcorr/4.))
            self.observables['ycorrelations'][link].append(np.real(ycorr/4.))

            # entropy
            self.observables['tangles'][link].append(-1*np.log(0.25 + 0.75*np.real(zcorr/4.)**2))

            # neel vectors
            xneel, yneel, zneel = self.gather_neel(link, state) # same logic as with correlations
            self.observables['xneel'][link].append(np.real(xneel/2.))
            self.observables['yneel'][link].append(np.real(yneel/2.))
            self.observables['zneel'][link].append(np.real(zneel/2.))

        for i in range(self.lattice.Lx *self.lattice.Ly): # go through all the individual sites
            xss, yss, zss = self.gather_sitespins((i,), state) # same logic again
            self.observables['zsites'][(i,)].append(zss/2)
            self.observables['xsites'][(i,)].append(xss/2)
            self.observables['ysites'][(i,)].append(yss/2)
        
        for i in range(self.lattice.Lx *self.lattice.Ly):
            for j in range(self.lattice.Lx *self.lattice.Ly):
                self.observables['plusminus'][(i,j)].append(self.gather_plusminus(i,j,state)/2.)

    #endregion

    def make_correlation(self, link):
        """
        Calculates the correlation of a certain `link`. If it doesn't exist as a Hilbert space matrix, makes the matrix.

        Correlation return type is added to the `link`->`value` dictionary.
        """

        # add matrix if it doesn't exit
        if link not in self._hspace.correlations:
            self._hspace.make_correlation(link)
            self.observables['correlations'][link] = []
            self.observables['zcorrelations'][link] = []
            self.observables['xcorrelations'][link] = []
            self.observables['ycorrelations'][link] = []
            self.observables['tangles'][link] = []
            print("> created correlation at link", link)
        
            # calculate for every state
            for st in self.states:
                corr, zcorr, xcorr, ycorr = self.gather_correlation(link, st)
                self.observables['correlations'][link].append(corr/4.) # then append them to the appropriate elements
                self.observables['zcorrelations'][link].append(zcorr/4.)
                self.observables['xcorrelations'][link].append(xcorr/4.)
                self.observables['ycorrelations'][link].append(ycorr/4.)
                self.observables['tangles'][link].append(-1*np.log(0.25 + 0.75*np.real(zcorr/4.)**2))

    def process(self, extra_links = None):
        """
        Full calculation of observables from states gathered throught the dynamics.
        """
        for st in self.states:
            self._calculate_observables(st)
        
        if extra_links != None:
            for link in extra_links:
                self.make_correlation(link)

    def asses_weights(self):
        """
        Calculates the weights of a neural network analytically from ED results.\\
        Makes sense only for a Restricted Boltzmann Machine with a single hidden node.\\
        NB: here, the biases are explicitly set to zero here. This could be justified by requesting pairity invariance.

        Returns a time series of network parameters formated as `[weights]`.
        """
        params = [] # define array for network parameters 
        #A = np.insert(self._hspace.configs, 0, 1, axis=1) # configurations matrix
        A = self._hspace.configs
        A = A[:int(len(A)/2)] # cut by half to accound for pairity analogues
        invA = np.linalg.pinv(A) # get the pseudoinverse
        for s in self.states:
            theta = np.arccosh(s/2) # RBM cosh arguments
            theta = theta[:int(len(theta)/2)] # also cut by half (their dimension is the same as psi)
            W = invA @ theta # calculate weights
            #W = sc.linalg.lstsq(A, theta)[0]
            params.append(np.insert(W, 0, 0)) # append to the parameter list
        
        return params

    def read_wavefunctions(self, inputs, extra_links = None):
        """
        Takes a time series of wave functions written in `inputs` and analyses them as if they were generated by the class.\\
        Assumes that the first state (index `0`) is the ground state.

        Does not reset the calculation - use only on an object without pre-existing data.\\
        Format inputs as a list (or an array) of `numpy` arrays: `[psi0, psi1, ...]`.
        """
        # do a check first
        if len(inputs) != (self.steps+1):
            raise ValueError("The dimension of the input ({:}) does not match the number of steps ({:}).".format(len(inputs), self.steps+1))
         
        # first, initials
        self.states.append(inputs[0].copy()) # ground state is the zeroth input
        self.times.append(self._start - self.dt)
        # i don't really care about energy

        # start the main loop
        for s in range(self.steps):
            # append the state to your states list
            self.states.append(inputs[s+1].copy())

            # time related stuff
            time = self._start + s*self.dt # update time
            self.times.append(time) # append time
            self._hspace.perturb(self.perturbator(time)) # perturb the Hamiltonian

            # calculate energy (it's technically time-dependent)
            self.observables['energy'].append(self.gather_energy(self.states[-1]))
        
        # afther the loop is done, postprocess time-dependent observables
        self.process(extra_links=extra_links)


    def run(self, postprocess = True, extra_links = None):
        """
        Runs the dynamics given the settings of the object.

        Steps:
        1. diagonalize the unperturbed Hamiltonian to get the ground state,
        2. perturb the Hamiltonian (perturbation is, in general, time-dependent),
        3. diagonalize at every time step,
        4. calculate the full state at every time, with the ground state as a primer.

        `postprocess`: decide if you want to calculate observables immediately after the run.
        """

        groundstate = np.linalg.eigh(self._hspace.H)[1].T[0] # [1] - eigenvectors, T - transposed, [0] - lowest energy
        # self.times.append(self._start - self.dt) # append a time one step before the start
        # self.states.append(groundstate) # at this time, the system is in the ground state
        # self.observables['energy'].append(self.gather_energy(groundstate))

        # I love this statement - is not None
        if self.init is not None:
            initial_state = self.init
            if not self.tacit:
                print("initial state is given: ", self.init)
        else:
            initial_state = groundstate
            if not self.tacit:
                print("initial state is chosen:", groundstate)
        #initial_state = self.init if self.init is not None else groundstate

        self.states.append(initial_state.copy())
        self.times.append(self._start-self.dt)

        for s in range(self.steps): #each step
            time = self._start + s*self.dt # calculate and remember time
            self.times.append(time)

            # perturb and diagonalize the Hamiltonian
            self._hspace.perturb(self.perturbator(time))
            evals, evecs = np.linalg.eigh(self._hspace.H)
            evecs = evecs.T

            # calculate the quantum state
            qstate = np.zeros(len(evecs)) * (1+1j)
            for (i,ev) in enumerate(evecs):
                #qstate += ev * np.vdot(initial_state, ev) * np.exp( -1j*time*evals[i]) # a more elegant way to do this maybe?
                qstate += ev * np.vdot(ev, self.states[-1]) * np.exp( -1j * self.dt * evals[i])
            
            self.states.append(qstate.copy()) # append it

            # accumulate time-dependent observables
            self.observables['energy'].append(self.gather_energy(qstate))
        
        # if you also want to calculate the observables immediately...
        if postprocess:
            self.process(extra_links=extra_links)