import numpy as np
from . import hilbert
from . import rbm


def add_noise(f, eta = 0.):
    """
    Adds a noise factor to the vector. The noise added to each component `fi` is a complex number drawn from a normal distribution with:\\
    `variance = |eta*fi|^2`.
    - `eta`: noise strength factor (typically small).

    Returns the noisified vector.
    """
    variances = np.abs(f*eta)**2. # array of variances

    # make the noisy vector
    noise_r = np.random.normal(scale=np.sqrt(variances))
    noise_i = np.random.normal(scale=np.sqrt(variances))
    noise = noise_r + 1j*noise_i

    return f + noise

class sampler:
    """
    Handles the calculation of averages and variational manifold elements for the case of full sampling of the Hilbert space by RBM.

    Variables
    ------
    - `hspace`: the Hilbert space object
    - `alpha`: network density
    - `tinvariant`: is the network translationally invariant? this changes the number of used parameters
    - `normal`: use normal full averaging? (as opposed to Monte-Carlo-like)
    - `scale`: multiply the wave function with this number
    - `real`: are the parameters real?
    - `noise_level`: controls how much artificial noise you introduce into the energy gradient

    Properties
    -------
    - `network`: contains rbm elements (see rbm.py)
    - `psiket`: ket in the Hilbert (sub)space
    - `S`: quantum Fisher matrix
    - `F`: energy gradient
    - `Lnorm`: gradient of the norm loss

        
    Methods
    -------
    - `Evar`: average energy
    - `Cvar(link)`: average correlation at `link`
    - `Czvar(link)`: average z-correlation at `link`
    - `update(parameters)`: updates the network with new parameters and reflects the updates to the sampler
    
    """

    def __init__(self, hspace:hilbert.space, alpha, tinvariant = False, 
                 normal = False, real = True, 
                 scale = 1., noise = 0.) -> None:

        self.hspace = hspace
        self.network = rbm.rbm(self.hspace.grid.Lx, self.hspace.grid.Ly, alpha, tinv=tinvariant, 
                               real = real, scale=scale)
        #self._m = len(self.network.parameters.all) # legnth of the network parameters array (effective M)
        #TODO
        self._m = len(self.network.parameters) # legnth of the network parameters array (effective M)
        self.noise = noise # remember it here

        self.psiket = self.network.vecpsi(self.hspace.configs)
        self._normal = normal
        #self.psiket_norm = self.psiket / np.linalg.norm(self.psiket)

        if not self._normal:
            self._make_equation()
        else:
            self._make_normal_equation()

    def update(self, new_parameters, just_parameters = False):
        """
        Updates the neural network with the `new_parameters` array, along with other things in the class.
        
        -`just_parameters`: do you only want to update the NN parameters and the wave function? if `False`, you also update S and F.
        """

        # 1. Replace the existing RBM parameters with new ones
        #self.network.parameters.all = new_parameters
        self.network.parameters = new_parameters

        # 2. Change the self.psiket
        self.psiket = self.network.vecpsi(self.hspace.configs)
        #self.psiket_norm = self.psiket / np.linalg.norm(self.psiket)

        if not just_parameters:
            # 3. make S and F
            if not self._normal:
                self._make_equation()
            else:
                self._make_normal_equation()
    

    # Local functions

    def _Eloc(self, index:int):
        """
        Calculates the local energy. Local means for only one spin configuration, indicated by the Hilbert space `index`.
        """
        # res = 0. + 0.*1j
        # for j in range(self.hspace.dim):
        #     res += self.hspace.H[index][j] * np.exp(self.network.logpsi(self.hspace.configs[j])-self.network.logpsi(self.hspace.configs[index]))
        # return res
        return self.hspace.H[index]@self.psiket / np.exp((self.network.logpsi(self.hspace.configs[index])))
    
    def _E2loc(self, index:int):
        """
        Calculates the local square of energy. Local means for only one spin configuration, indicated by the Hilbert space `index`.
        """
        # res = 0. + 0.*1j
        # for j in range(self.hspace.dim):
        #     res += self.hspace.H[index][j] * np.exp(self.network.logpsi(self.hspace.configs[j])-self.network.logpsi(self.hspace.configs[index]))
        # return res
        return (self.hspace.H@self.hspace.H)[index]@self.psiket / np.exp((self.network.logpsi(self.hspace.configs[index])))
    
    def _Cloc (self, link, index:int):
        """
        Calculates the local correlations between sites indicated by `link`, at the the Hilbert space `index` configuration.
        """
        return self.hspace.correlations[link][index]@self.psiket / np.exp((self.network.logpsi(self.hspace.configs[index])))

    def _Czloc (self, link, index:int):
        """
        Calculates the local z-correlations between sites indicated by `link`, at the the Hilbert space `index` configuration.
        """
        return self.hspace.zcorrelations[link][index]@self.psiket / np.exp((self.network.logpsi(self.hspace.configs[index])))
    
    # Variational functoins

    def Evar(self):
        """
        Variational (average) energy.
        """
        if not self._normal:
            num = 0.
            for i in range(self.hspace.dim):
                num += self._Eloc(i) * (np.abs(np.exp(self.network.logpsi(self.hspace.configs[i]))))**2
                #num += self._Eloc(i) * (self.psiket_norm[i]**2)

            return np.real(num/(np.linalg.norm(self.psiket)**2))
        else:
            # psi_n = self.psiket / np.linalg.norm(self.psiket)
            # return np.real(np.vdot(psi_n, self.hspace.H@psi_n))
            psi_n = self.psiket
            return np.real(np.vdot(psi_n, self.hspace.H@psi_n)/ (np.linalg.norm(psi_n)**2))

    def E2var(self):
        """
        Square of Hamiltonian observable.
        """
        if not self._normal:
            num = 0.
            for i in range(self.hspace.dim):
                num += self._E2loc(i) * (np.abs(np.exp(self.network.logpsi(self.hspace.configs[i]))))**2

            return np.real(num/(np.linalg.norm(self.psiket)**2))
        else:
            psi_n = self.psiket
            return np.real(np.vdot(psi_n, (self.hspace.H@self.hspace.H)@psi_n)/ (np.linalg.norm(psi_n)**2))


    def Cvar (self, link):
        """
        Average correlation at `link`.
        """
        if not self._normal:
            num = 0.
            for i in range(self.hspace.dim):
                num += self._Cloc(link, i) * (np.abs(np.exp(self.network.logpsi(self.hspace.configs[i]))))**2

            return np.real(num/(np.linalg.norm(self.psiket)**2))
        else:
            psi_n = self.psiket / np.linalg.norm(self.psiket)
            return np.real(np.vdot(psi_n, self.hspace.correlations[link]@psi_n))

    def Czvar (self, link):
        """
        Average z correlation at `link`.
        """
        if not self._normal:
            num = 0.
            for i in range(self.hspace.dim):
                num += self._Czloc(link, i) * (np.abs(np.exp(self.network.logpsi(self.hspace.configs[i]))))**2

            return np.real(num/(np.linalg.norm(self.psiket)**2))
        else:
            psi_n = self.psiket / np.linalg.norm(self.psiket)
            return np.real(np.vdot(psi_n, self.hspace.zcorrelations[link]@psi_n))

    
    # S matrix and energy gradient

    def _make_equation(self):
        """
        Creates the Quantum Fisher Matrix and the energy gradient from their definitions.

        - `noise`: artificial noise inserted into F (see `add_noise` function for details)
        """

        Ok = np.zeros((self._m), dtype=complex)
        OkOk = np.zeros((self._m,self._m), dtype=complex)
        ElOk = np.zeros((self._m), dtype=complex)

        # parts
        # I made this code block to check out every single summing term in S and F
        # to make sense of the breakdown
        self._El_parts = np.zeros(self.hspace.dim, dtype = complex)
        self._Ok_parts = np.zeros((self._m, self.hspace.dim), dtype = complex)
        self._OkOk_parts = np.zeros((self._m,self._m, self.hspace.dim), dtype=complex)
        self._ElOk_parts = np.zeros((self._m, self.hspace.dim), dtype=complex)
        
        # loop over all spin indices
        for i in range(self.hspace.dim):
            #ders = self.network.Ok(self.hspace.configs[i]).all # all derivatives by network parameters at congfiguration i
            #TODO
            ders = self.network.Ok(self.hspace.configs[i])
            #print("ders shape:", np.shape(ders))
            El = self._Eloc(i) # local energy at that configuration

            abselement = (np.abs(np.exp(self.network.logpsi(self.hspace.configs[i]))))**2

            # parts: obsolete
            self._El_parts[i] = self._Eloc(i)

            #loop over the first parameter index
            for k in range(self._m):
                Ok[k] += ders[k] * abselement
                ElOk[k] += np.conjugate(ders[k]) * El * abselement

                self._Ok_parts[k, i] = ders[k] * abselement
                self._ElOk_parts[k, i] = np.conjugate(ders[k]) * El * abselement

                #loop over the second one
                for kp in range(self._m):
                    OkOk[kp][k] += np.conjugate(ders[k]) * ders[kp] * abselement

                    #parts: obsolete
                    self._OkOk_parts[kp, k, i] = np.conjugate(ders[k]) * ders[kp] * abselement

        
        # normalize
        normofpsi = np.linalg.norm(self.psiket)**2
        Ok /= normofpsi
        OkOk /= normofpsi
        ElOk /= normofpsi

        # parts: normalize
        self._Ok_parts /= normofpsi
        self._OkOk_parts /= normofpsi
        self._ElOk_parts /= normofpsi

        # calculate S and F
        self.F = ElOk - self.Evar() * np.conjugate(Ok) 
        self.F = add_noise(self.F, eta = self.noise) # add noise to F (defaults to zero)
        self.S = OkOk - np.outer(np.conjugate(Ok), Ok)
        self.Ok = Ok # don't ask me why i need this to be a class variable

        # calculate the gradient of norm loss (this is a logical step where it should be done)
        #self.Lnorm_grad = -2.*(1. - normofpsi) * np.conjugate(Ok)*normofpsi
        self.Lnorm_grad = (1. - 1./np.sqrt(normofpsi)) * np.conjugate(Ok)*normofpsi

        # DONE

    def _make_normal_equation(self):
        """
        Creates the Quantum Fisher Matrix and the energy gradient from their definitions.\\
        Uses the normal definition of averages in the Hilbert space, rather than the Monte-Carlo-like one.
        """

        # calculate the normalized psi vector
        # this is something that Monte Carlo can't do
        psi_n = self.psiket #/ np.linalg.norm(self.psiket)

        # define elements
        Ok = np.zeros((self._m), dtype=complex)
        OkOk = np.zeros((self._m,self._m), dtype=complex)
        ElOk = np.zeros((self._m), dtype=complex)

        # parts
        # I made this code block to check out every single summing term in S and F
        # to make sense of the breakdown
        self._Evar_parts = np.zeros((self.hspace.dim, self.hspace.dim), dtype = complex)
        # self._Ok_parts = np.zeros((self._m, self.hspace.dim), dtype = complex)
        # self._OkOk_parts = np.zeros((self._m,self._m, self.hspace.dim), dtype=complex)
        # self._ElOk_parts = np.zeros((self._m, self.hspace.dim), dtype=complex)

        # loop over spin index s
        for s in range(self.hspace.dim):

            # get derivative elements from the network
            #ders = self.network.Ok(self.hspace.configs[s]).all
            #TODO
            ders = self.network.Ok(self.hspace.configs[s])

            for ss in range(self.hspace.dim):
                self._Evar_parts[s,ss] = np.conjugate(psi_n[s]) * psi_n[ss] * self.hspace.H[s,ss]

            # loop over parameter index k
            for k in range(self._m):
                # contribute to <Ok>
                Ok[k] += ders[k] * np.abs(psi_n[s])**2

                # contribure to <ElOk>
                for sp in range(self.hspace.dim):
                    ElOk[k] += np.conjugate(psi_n[s]) * psi_n[sp] * np.conjugate(ders[k]) * self.hspace.H[s,sp]
                
                # contribute to <Ok*Ok>
                for kp in range(self._m):
                    OkOk[kp][k] += np.conjugate(ders[k]) * ders[kp] * np.abs(psi_n[s])**2
        
        # no normalization needed?
        normofpsi = np.linalg.norm(self.psiket)**2
        Ok /= normofpsi 
        OkOk /= normofpsi
        ElOk /= normofpsi
        self._Evar_parts /= normofpsi

        # calculate S and F
        self.F = ElOk - self.Evar() * np.conjugate(Ok)
        self.F = add_noise(self.F, eta = self.noise) # add noise to F (defaults to zero)
        self.S = OkOk - np.outer(np.conjugate(Ok), Ok)
        self.Ok = Ok # don't ask me why i need this to be a class variable

        # calculate the gradient of norm loss (this is a logical step where it should be done)
        #self.Lnorm_grad = -2.*(1. - normofpsi) * np.conjugate(Ok)*normofpsi
        self.Lnorm_grad = (1. - 1./np.sqrt(normofpsi)) * np.conjugate(Ok)*normofpsi
        # DONE

    def sptectrumS(self):
        """
        Calculates and sorts the eigenvalue spectrum of the S-matrix.
        """
        return np.sort(np.real(np.linalg.eigvalsh(self.S)))

    def residual(self, w, wdot):
        """
        Calculates the residual distance.
        
        - `w`: current parameters
        - `wdot`: derivative of the parameter vector

        Careful with indices - this thing is calculared from the update directly, so it will always have one less values.
        """

        # residual distance is the function of time, so it must also come from S and F at THIS time
        self.update(w)

        res = np.vdot(wdot, self.S@wdot) # the S-matrix part
        res += 1j * np.vdot(wdot, self.F) - 1j * np.vdot(self.F, wdot) # the F part
        res += self.E2var() - self.Evar()**2 # the self energy part

        #return np.real(res*(np.linalg.norm(self.psiket)**2))
        return np.real(res) # get rid of the imaginary part

    def metric_velocity(self, w, wdot):
        """
        Calculates the metric veolcity, or the integrand in the manifold arc distance along a trajectory.\\
        Assumes that the S matrix is Hermitian and positive-definite.

        - `w`: current parameters
        - `wdot`: derivative of the parameter vector
        """

        # update the S and F
        self.update(w)

        return np.sqrt(np.real( # get rid of any leftover imaginary components (they should be =0)
                    np.vdot(wdot, self.S@wdot)
                    ))
    
    def gradient_velocity_it(self, w, wdot):
        """
        Calculates the metric veolcity, or the integrand in the manifold arc distance along a trajectory.\\
        Doesn't use the S matrix, but rather a substitute of the EoM into the distance integral formula.
        
        - `it` denotes imaginary time, therefore don't call this funciton in dynamics

        ---
        - `w`: current parameters
        - `wdot`: derivative of the parameter vector
        """

        # update the S and F
        self.update(w)

        return np.sqrt(np.real( # get rid of any leftover imaginary components (they should be =0)
                    np.vdot(-1.*wdot, self.F)
                    ))
    
    def gradient_velocity_rt(self, w, wdot):
        """
        Calculates the metric veolcity, or the integrand in the manifold arc distance along a trajectory.\\
        Doesn't use the S matrix, but rather a substitute of the EoM into the distance integral formula.
        
        - `rt` denotes real time, therefore don't call this function in optimization

        ---
        - `w`: current parameters
        - `wdot`: derivative of the parameter vector
        """

        # update the S and F
        self.update(w)

        return np.sqrt(np.real( # get rid of any leftover imaginary components (they should be =0)
                    np.vdot(-1.*wdot, 1j*self.F)
                    ))

    def line_distance(self, w, target_w, division = 100):
        """
        Measures the distance from point `w` to point `target_w` on the manifold along a straigh line trajectory. Still takes into account the S-matrix as metric.

        - `division`: discretization of the line between the two points
        """
        params = w.copy()
        v = target_w - params # constant tangent vector between the two poitns
        dv = v/division # step along the tangent

        # update the sampler
        self.update(params)
        d = np.sqrt( np.real(
            np.vdot(v, self.S @ v)
            ))

        for _ in range(division):
            params += dv # add small vector
            self.update(params) # update the sampler

            d += np.sqrt( np.real( # add small distance
            np.vdot(v, self.S @ v)
                ))

        return d/division
