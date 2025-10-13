import numpy as np

#region index maping functions
def bias_map(N, alpha):
    """
    Generates an array that maps the indices of independent biases onto the full set of biases.
    """
    r = np.array([i for i in range(alpha)])
    return np.repeat(r, N)

def split_and_roll(array, Lx):
    """
    Splits the `array` into `Lx` parts and rolls each part backwards once. This accounts for translational invariance over the X-direction.\\
    Returns the fully connected array.
    """
    if len(array) % Lx != 0:
        raise ValueError("Array length must be divisible by X")
    
    # Calculate the length of each part
    #part_length = len(array) // Lx
    
    # Split the array into X parts
    parts = np.array_split(array, Lx)
    
    # Roll each part backwards by one position
    rolled_parts = [np.roll(part, -1) for part in parts]
    
    # Combine the rolled parts back into a single array
    result = np.concatenate(rolled_parts)
    
    return result

def generate_block(array, Lx, Ly):
    """
    Takes an input `array` and repeats it `Ly` times to create a matrix. Each time it's repeated, all `Lx` of it's parts are rolled backwards once. This accounts, in total, for the translational invariance in the Y-direction.\\
    Returns the created matrix.
    """
    # Initialize the matrix with the first row being the original array
    matrix = np.zeros((Ly, len(array)), dtype=array.dtype)
    matrix[0] = array
    
    # Apply split_and_roll X times
    for i in range(1, Ly):
        matrix[i] = split_and_roll(matrix[i-1], Lx)
    
    return matrix

def roll_and_concatenate(array, Lx, Ly):
    """
    Takes an array and:
    1. performs block generation to account for translational invariance in both X- and Y-directions,
    2. repeats the process `Lx` times in total to finalize the X-direction invariance,
    3. after each repetition, the entire block is rolled backwards `Ly` times, to finalize the Y-direction invariance.

    Returns the final amalgam of blocks.
    """
    matrix = generate_block(array, Lx, Ly)
    concatenated_matrix = matrix.copy()
    rolled_matrix = matrix.copy()
    
    for i in range(Lx-1):
        # Roll each row of the current matrix backwards Y times
        rolled_matrix = np.array([np.roll(row, -Ly) for row in rolled_matrix])
        
        # Concatenate the rolled matrix to the existing matrix
        concatenated_matrix = np.concatenate((concatenated_matrix, rolled_matrix), axis=0)
    
    return concatenated_matrix

def weight_map(Lx, Ly, alpha):
    """
    Generates an index matrix that maps the independent weights to the full set of weights.

    Independent weights are shaped as a vector.
    The shape of the index matrix is the same as the shape of the full weight matrix.
    """
    indep = np.array([i for i in range(alpha*Lx*Ly)]) # indices of independent parameters

    # split it into alpha parts
    parts = np.array_split(indep, alpha)
    blocks = []

    #perform the operation on each part
    for part in parts:
        blocks.append(roll_and_concatenate(part,Lx, Ly))

    # connect them together... somehow
    result = blocks[0].copy()
    for i in range(len(blocks)-1):
        result = np.concatenate((result, blocks[i+1]), axis=0)
    
    return result

#endregion

def sum_weigth_derivatives(A):
    """
    Sums over the outer product of the lookup table and spins, or matrix `A`, to get the proper derivatives by weights.
    Supossedly optimized to exploit numpy to the max, says ChatGPT.
    """
    m, n = A.shape
    num_blocks = m // n  # Since m is always a multiple of n


    # Reshape into (num_blocks, n, n) to extract n×n blocks
    blocks = A.reshape(num_blocks, n, n)


    # Generate rolling shifts for each column index
    shifts = np.arange(n)  # [0, 1, 2, ..., n-1]

    # Apply rolling using broadcasting (vectorized operation)
    rolled_blocks = np.stack([np.roll(blocks[:, :, i], shift=i, axis=1) for i in shifts], axis=2)

    # Reshape back into (m, n)
    A_transformed = rolled_blocks.reshape(m, n)

    # Sum across rows
    row_sums = A_transformed.sum(axis=1)

    return row_sums

class rbm_parameters:
    """
    A structure for keeping weights and biases.

    -`b_number`: number of biases, or hidden layer nodes.
    -`w_shape`: shape of the weight matrix, or connections between the visible and the hidden layers.
    - `real`: do you want to initialize them as real?

    Biases and weights must be flattened and concatenated into a single vector.
    """
    # the ugly part of python's OOP
    def __init__(self, b_number, w_shape, real = True) -> None:
        #self.M = M
        #self.N = N

        self._bnumber = b_number
        self._wshape = w_shape

        if real:
            self._b = np.random.randn(self._bnumber).astype(np.complex128)*0.1
            self._w = np.random.randn(self._wshape[0], self._wshape[1]).astype(np.complex128)*0.1
        else:
            self._b = (np.random.randn(self._bnumber).astype(np.complex128) + np.random.randn(self._bnumber).astype(np.complex128)*1j)*0.1
            self._w = (np.random.randn(self._wshape[0], self._wshape[1]).astype(np.complex128) + np.random.randn(self._wshape[0], self._wshape[1]).astype(np.complex128)*1j)*0.1

        # if you want them complex
        # self._b = (np.random.randn(self._bnumber).astype(np.complex128) + np.random.randn(self._bnumber).astype(np.complex128)*1j)*0.1
        # self._w = (np.random.randn(self._wshape[0], self._wshape[1]).astype(np.complex128) + np.random.randn(self._wshape[0], self._wshape[1]).astype(np.complex128)*1j)*0.1

        # if you want them positive
        #self._b = np.random.uniform(low=0, high=0.1, size=(self._bnumber)).astype(np.complex128)
        #self._w = np.random.uniform(low=0, high=0.1, size=(self._wshape[0], self._wshape[1])).astype(np.complex128)
        self._all = np.concatenate((self._b, self._w.flatten())).astype(np.complex128)
    
    @property
    def b(self):
        return self._b
    
    @b.setter
    def b(self, nb):
        self._bnumber = len(nb)
        self.all = np.concatenate((nb, self.w.flatten()))

        # self._b = nb
        # #self._all[:self.M] = nb
        
        # self._all[:self._bnumber] = nb
    
    @property
    def w(self):
        return self._w
    
    @w.setter
    def w(self, nw):
        self._wshape = np.shape(nw)
        self.all = np.concatenate((self.b, nw.flatten()))

        # self._w = nw
        
        # #self._all[self.M:] = self._w.flatten()
        # self._all[self._bnumber:] = self._w.flatten()
    
    @property
    def all(self):
        return self._all
    
    @all.setter
    def all(self, newp):
        self._all = newp
        #self._b = self._all[:self.M]
        self._b = self._all[:self._bnumber]
        self._w = np.reshape(self._all[self._bnumber:], self._wshape)

    # def w(self):
    #     """
    #     Returns the weights from the parameters, reshaped into a matrix.
    #     """
    #     return self.pars[self.M:].reshape(self.M, self.N)
    
    # def b(self):
    #     """
    #     Returns the biases.
    #     """
    #     return self.pars[:self.M]
    
class rbm:
    """
    Properties and method for manipulating the Restricted Boltzmann Machine neural network.\\
    Initialized from the number of physical spins and the network density.

    Variables
    ------
    - `Lx`, `Ly`: lattice dimensions
    - `alpha`: network density 
    - `tinv`: is the network translationally invariant? this changes the number of used parameters

    Properties
    -------
    - `N`: number of visible spins (neurons)
    - `M`: number of hidden neurons `M`=`alpha`*`N`
    - `parameters`: set of network weights and biases
    - `real`: do you want to initialize parameters as real?
    - `scale`: scale the wave function with this number
    
    Methods
    -------
    - `logpsi(spins)`: logarithmic wave function
    - `vecpsi(spins)`: Hilbert space vector of the wave function
    - `Ok(spins)`: rbm-parameter-like structure of derivatives by parameters

    NOTE: spin vector should be flattened and properly labeled. 
    """

    def __init__(self, Lx:int, Ly:int, alpha, tinv = False, 
                 real = True,
                 scale = 1.) -> None:
        # first initialize the global properties of the network
        self.alpha = alpha # density
        self.N = Lx * Ly # number of visible spins
        self.M = self.alpha*self.N # number of hidden spins
        if np.isclose(self.M, int(self.M)):
            self.M = int(self.M)
        else:
            raise ValueError("M is not an integer")
        
        # scaling
        self.scale = scale
        # translational invariance (or lack thereof)
        self.tinv = tinv    
        #self._full_parameters = rbm_parameters(self.M, (self.M, self.N))
        #self._invariant_parameters = rbm_parameters(self.alpha, (1, self.M))

        # parameters setting
        if self.tinv: # if translationally invariant

            # #initialize full invariant and full sets of parameters
            # self._invariant_parameters = rbm_parameters(self.alpha, (1, self.M), real = real)
            # self._full_parameters = rbm_parameters(self.M, (self.M, self.N), real = real)
            
            # # create the index maps
            # self.weight_map = weight_map(Lx, Ly, alpha)
            # self.bias_map = bias_map(self.N, alpha)

            # #change the full parameters acording to the index maps
            # self._full_parameters.b = self._invariant_parameters.b[self.bias_map]
            # self._full_parameters.w = self._invariant_parameters.w.flatten()[self.weight_map]

            # # finally, declare the parameters array the invariant ones
            # self._parameters = self._invariant_parameters

            #TODO
            # create the index maps
            self.weight_map = weight_map(Lx, Ly, alpha)
            self.bias_map = bias_map(self.N, alpha)

            # initialize some parameters
            self.biases = np.random.randn(self.alpha).astype(np.complex128)*0.1 + 1j*(not real)*np.random.randn(self.alpha).astype(np.complex128)*0.1
            self.weights = np.random.randn(self.M).astype(np.complex128)*0.1 + 1j*(not real)*np.random.randn(self.M).astype(np.complex128)*0.1

            # map them onto full parameters
            self.biases = self.biases[self.bias_map]
            self.weights = self.weights[self.weight_map]

        else: # if not translationally invariant
            #self._full_parameters = rbm_parameters(self.M, (self.M, self.N), real = real) #initialize full parameters
            #self._parameters = self._full_parameters # declare the parameters array equal to them
            #TODO
            self.biases = np.random.randn(self.M).astype(np.complex128)*0.1 + 1j*(not real)*np.random.randn(self.M).astype(np.complex128)*0.1
            self.weights = np.random.randn(self.M, self.N).astype(np.complex128)*0.1 + 1j*(not real)*np.random.randn(self.M, self.N).astype(np.complex128)*0.1

        #self.parameters = rbm_parameters(self.M, self.N)
        #self.parameters = rbm_parameters(self.M, (self.M, self.N))

    @property
    def parameters(self):
        #return self._parameters
        #TODO
        if self.tinv:
            return np.concatenate((self.biases.reshape(self.alpha, self.N).T[0], 
                                   self.weights[:,0]))
        else:
            return np.concatenate((self.biases, self.weights.flatten()))
    
    @parameters.setter
    def parameters(self, newp):
        if self.tinv: # if translationally invariant
            # # set invariant parameters to the new ones
            # self._invariant_parameters.all = newp

            # # map them onto full parameters
            # self._full_parameters.b = self._invariant_parameters.b[self.bias_map]
            # self._full_parameters.w = self._invariant_parameters.w.flatten()[self.weight_map]

            # # change the parameters array
            # self._parameters = self._invariant_parameters
            
            #TODO
            self.biases = newp[:self.alpha][self.bias_map]
            self.weights = newp[self.alpha:][self.weight_map]

        else: # if not translationally invariant
            # self._full_parameters.all = newp
            # self._parameters = self._full_parameters
            #TODO
            self.biases = newp[:self.M]
            self.weights = np.reshape(newp[self.M:], (self.M, self.N))

    def logpsi(self, spins):
        """
        Calculates the wave function (network output) for a spin configuration.
        - `spins`: array containing all spin values.

        Returns the logarithm of the wavefunction - more numerically stable.
        Has to be exponentiated when calculating observables or output vectors.
        """
        # TODO: fix but with translational invatiance - wave function evaluation might be wrong!
        # NOTE: it might be missing the product over M or it's wrongly derived here!
        #return np.sum(np.log(2.*np.cosh(self.parameters.b + self.parameters.w@spins)))
        #return np.sum(np.log(2.*np.cosh(self._full_parameters.b + self._full_parameters.w@spins)*self.scale))
        #TODO
        return np.sum(np.log(2.*np.cosh(self.biases + self.weights@spins)*self.scale))
        # NOTE: what do you mean, what is this sum even summing???
        #return np.sum(np.log(2.*np.cos(self._full_parameters.b + self._full_parameters.w@spins)))
    
    def vecpsi (self, configurations):
        """
        Calculates the vector in the Hilbert space from the wave function.
        """
        return np.array([np.exp(self.logpsi(config)) for config in configurations])
    
    #TODO: make this funciton if you really need it
    # def _recover_indep(self, params:rbm_parameters):
    #     """
    #     Recovers the reduced parameters from the set of full ones.
    #     """

    def Ok(self, spins):
        """
        Calculates the logarithmic derivative vector by biases and weights.
        -`spins`: spin configuration

        Returns a structure in the same shape as RBM parameters.
        """
        #ders = rbm_parameters(len(self.parameters.b), np.shape(self.parameters.w)) # a rbm parameter structure
        #TODO
        #ders = rbm_parameters(len(self._full_parameters.b), np.shape(self._full_parameters.w), real = False) # a rbm parameter structure
        #theta = self.parameters.b + self.parameters.w@spins # an M-vector
        #theta = self._full_parameters.b + self._full_parameters.w@spins # an M-vector
        #TODO
        theta = self.biases + self.weights@spins # an M-vector

        #ders.b = np.tanh(theta) # O by biases
        #TODO
        ders_biases = np.tanh(theta)
        #print(ders.b)
        #ders.w = np.outer(spins, np.tanh(theta)) # O by weights
        #TODO
        ders_weights = np.outer(spins, np.tanh(theta)) # O by weights
        #ders_weights = np.outer(np.tanh(theta), spins) # O by weights

        #print(ders.w)

        if self.tinv:
            # ders.b = ders.b.reshape(self.alpha, self.N).T[0] * self.M
            # ders.w = np.array([ders.w[0]]) *self.N
            # # ders.b = ders.b.reshape(self.alpha, self.N).T[0]
            # # ders.w = np.array([ders.w[0]])
            # #ders.all = np.concatenate(ders.b.reshape(self.alpha, self.N).T[0], ders.w.T[0])

            #TODO
            ders_biases = ders_biases.reshape(self.alpha, self.N).T[0] * self.M
            ders_weights = ders_weights[:,0] * self.N
            #ders_weights = (ders_weights[:, ::self.N]).flatten(order='F') * self.M

            #NOTE: for some reason, with translational invariance, my derivatives are underestimated
            #NOTE: bias-derivatives by the factor of M, weight-derivatives by the factor of N
            #NOTE: why is this happening? beats me. might be a mistake in the math of TDVP? or Giammarco did an oopsie again?
            #NOTE: this is a hotfix! not a solution.

            #NOTE: Alright, here goes my attempt of the correction of this problem
            # ders_biases = (ders_biases.reshape(self.alpha, self.N).T).sum(axis = 1)
            # ders_weights = sum_weigth_derivatives(ders_weights)
            #NOTE: it didn't work. fix this at some point maybe


        ders = np.concatenate((ders_biases, ders_weights.flatten()))
        return ders

