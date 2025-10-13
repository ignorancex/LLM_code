import numpy as np

# just some helper variables inside the module namespace
sigma_x = np.array([[0,1],[1,0]])
sigma_y = np.array([[0,-1j],[1j,0]])
sigma_z = np.array([[1,0],[0,-1]])
eye = np.eye(2)

def remove_from_tuple(tpl, num):
    # Create a new tuple with the first occurrence of num removed
    if num in tpl:
        new_tpl = []
        found = False
        for x in tpl:
            if x == num and not found:
                found = True
                continue
            new_tpl.append(x)
        
        tpl = tuple(new_tpl)
        
    return tpl

def new_correlation(sites, tup, gauge = True, direction = 'all'):
    """Returns a contribution of an operator string of arbitrary length, in the complete product basis.
    - `sites`: number of particles,
    - `tup`: tuple containing all the site indices that contribute to the correlation.
    - `direction`: decides in which direction (of the algebra) are the matrices contributing, choose between: `'all'`, `'x'`, `'y'`, `'z'`
    - `gauge`: apply the gauge transformation? this changes the signs of non-diagonal elements (effectivelly removing the sign structure)

    If you want just one site, send a tuple in the form `(index,)`.\\
    This is equal to the correlation funciton matrix at those sites.
    """

    if direction not in ('all', 'z', 'x', 'y'):
        raise ValueError("No direction given. Please provide one of the following: 'all', 'x', 'y, 'z'.")

    resultx = 1.
    resulty = 1.
    resultz = 1.

    mult = -1 if gauge else 1 # this is a sign flipper in case you're applying the gauge transformation

    #multiply by either unit matrix if it does not contribute or a sigma matrix if it does
    for i in range(sites):
        if i in tup:
            m = 0
            while i in tup:
                m += 1
                tup = remove_from_tuple(tup, i)

            resultx = np.kron(resultx, sigma_x**m)
            resulty = np.kron(resulty, sigma_y**m)
            resultz = np.kron(resultz, sigma_z**m)
        else:
            resultx = np.kron(resultx, eye)
            resulty = np.kron(resulty, eye)
            resultz = np.kron(resultz, eye)
    
    # quite self-explainatory
    if direction == 'z':
        return resultz # ofc, z does not have offdiagonal terms so no gauge needed
    elif direction == 'x':
        return mult*resultx
    elif direction == 'y':
        return mult*resulty
    else:
        return mult*resultx + mult*resulty + resultz

def correlation(sites, index1, index2, zonly = False, gauge = True):
    """Returns a contribution of an interaction pair of Heisenberg model, in the complete product basis.
    - `sites`: number of particles,
    - `index1`, `index2`: two indices where from which the contribution is calculated.
    - `zonly`: decides whether to return only the z-component of the result
    - `gauge`: apply the gauge transformation? this changes the signs of non-diagonal elements (effectivelly removing the sign structure)

    This is equal to the correlation funciton matrix at those two sites.
    """

    resultx = 1.
    resulty = 1.
    resultz = 1.

    mult = -1 if gauge else 1 # this is a sign flipper in case you're applying the gauge transformation

    #multiply by either unit matrix if it does not contribute or a sigma matrix if it does
    for i in range(sites):
        if index1==i or index2==i:
            resultx = np.kron(resultx, sigma_x)
            resulty = np.kron(resulty, sigma_y)
            resultz = np.kron(resultz, sigma_z)
        else:
            resultx = np.kron(resultx, eye)
            resulty = np.kron(resulty, eye)
            resultz = np.kron(resultz, eye)
    
    # quite self-explainatory
    if zonly:
        return np.real(resultz) # ofc, z does not have offdiagonal terms so no gauge needed
    else:
        return np.real(mult*resultx + mult*resulty + resultz)

def connected_pairs(Lx, Ly, periodic_x=True, periodic_y=True, along_y = False):
    """
    Creates a list of linked nearest-neighbour pairs on a 2D lattice. The sites are indexed as`x+Lx*y` for lattice coordinates `(x,y)`.\\
    The indices are arranged to follow the positive x and y directions. For example, pair `(0,1)` is positive in x, while `(1,0)` is not.

    - `periodic_x`, `periodic_y`: bools that control the periodicity in x- and y-directions, respectivelly.
    - `along_y`: if `True`, the  returned links will only be along the y-direction.

    Returns a list of tuples.

    ~Created by me and `Chat, j'ai pété  3.5`.
    """

    pairs = []

    # Helper function to get the index of a site given its coordinates
    def index(x, y):
        if periodic_x:
            x %= Lx
        if periodic_y:
            y %= Ly
        return y * Lx + x

    # Iterate through all lattice sites
    for y in range(Ly):
        for x in range(Lx):
            current_index = index(x, y)

            if not along_y: # don't create x bonds if you only want y bonds
                # Add pairs with right neighbor
                if not periodic_x and x < Lx - 1:
                    neighbor_index = index(x + 1, y)
                    if (current_index, neighbor_index) not in pairs and (neighbor_index, current_index) not in pairs:
                        pairs.append((current_index, neighbor_index))
                elif periodic_x:
                    neighbor_index = index((x + 1) % Lx, y)
                    if (current_index, neighbor_index) not in pairs and (neighbor_index, current_index) not in pairs:
                        pairs.append((current_index, neighbor_index))

            # Add pairs with bottom neighbor
            if not periodic_y and y < Ly - 1:
                neighbor_index = index(x, y + 1)
                if (current_index, neighbor_index) not in pairs and (neighbor_index, current_index) not in pairs:
                    pairs.append((current_index, neighbor_index))
            elif periodic_y:
                neighbor_index = index(x, (y + 1) % Ly)
                if (current_index, neighbor_index) not in pairs and (neighbor_index, current_index) not in pairs:
                    pairs.append((current_index, neighbor_index))

    pairs = list(set(pairs))
    pairs = [pair for pair in pairs if pair[0] != pair[1]]

    return pairs

def diagonal_pairs(Lx, Ly, periodic_x=True, periodic_y=True):
    """
    Creates a list of links between diagonal sites on a 2D lattice.  The sites are indexed as `x+Lx*y` for lattice coordinates `(x,y)`. \\
    
    - `periodic_x`, `periodic_y`: bools that control the periodicity in x- and y-directions, respectivelly.

    Returns a list of tuples.
    """

    if Lx <= 1 or Ly <= 1:
        raise ValueError("Cannot have diagonal pairs on a one-dimensional lattice. Please provide valid lattice dimension.")

    diagonal_pairs = set()  # Use a set to store unique pairs
    
    # Helper function to convert 1D index to 2D coordinates
    def index_to_coords(index):
        x = index % Lx
        y = index // Lx
        return x, y

    # Helper function to convert 2D coordinates to 1D index
    def coords_to_index(x, y):
        return y * Lx + x

    # Loop through each lattice site by index
    for index in range(Lx * Ly):
        x, y = index_to_coords(index)
        
        # Check for diagonals: (x+1, y+1), (x-1, y+1), (x+1, y-1), (x-1, y-1)
        diagonal_offsets = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
        
        for dx, dy in diagonal_offsets:
            nx = x + dx
            ny = y + dy
            
            # Apply periodic boundary conditions in x direction if enabled
            if periodic_x:
                nx = nx % Lx
            else:
                # Ensure the neighbor is within the x bounds if no periodicity
                if not (0 <= nx < Lx):
                    continue
            
            # Apply periodic boundary conditions in y direction if enabled
            if periodic_y:
                ny = ny % Ly
            else:
                # Ensure the neighbor is within the y bounds if no periodicity
                if not (0 <= ny < Ly):
                    continue

            neighbor_index = coords_to_index(nx, ny)

            # Always store pairs as (min(index, neighbor_index), max(index, neighbor_index))
            pair = (min(index, neighbor_index), max(index, neighbor_index))
            diagonal_pairs.add(pair)  # Use set to ensure uniqueness
    
    # Convert the set back to a sorted list of tuples
    return sorted(diagonal_pairs)

def get_square_corners(Lx, Ly, periodic_x=True, periodic_y=True):
    """
    Creates a list of indices of all 2x2 squares on a square lattice.\\
    The return type is a list of tuples of the format: `(bottom_left, bottom_right, top_left, top_right)`.

    - `periodic_x`, `periodic_y`: bools that control the periodicity in x- and y-directions, respectivelly.
    """

    if Lx <=1 or Ly <=1:
        raise ValueError("Cannot construct on a one-dimensional lattice. Please provide the right dimension.")
    
    squares = []  # To store the tuples of square corner indices

    # Disable periodicity if the lattice size is 2 in the respective direction
    if Lx == 2:
        periodic_x = False
    if Ly == 2:
        periodic_y = False

    # Helper function to apply periodic boundary conditions
    def apply_pbc(coord, max_coord, periodic):
        if periodic:
            return coord % max_coord
        else:
            return coord if 0 <= coord < max_coord else None

    # Loop over all possible bottom-left corners of the squares
    for x in range(Lx):
        for y in range(Ly):
            # Calculate the four corners of the square
            bottom_left_index = y * Lx + x

            # Apply periodic boundary conditions for bottom-right, top-left, and top-right
            bottom_right_x = apply_pbc(x + 1, Lx, periodic_x)
            top_left_y = apply_pbc(y + 1, Ly, periodic_y)

            # If any corner goes out of bounds and periodicity is not allowed, skip
            if bottom_right_x is None or top_left_y is None:
                continue

            bottom_right_index = y * Lx + bottom_right_x
            top_left_index = top_left_y * Lx + x
            top_right_index = top_left_y * Lx + bottom_right_x

            # Add the square corners as a tuple to the list
            squares.append((bottom_left_index, bottom_right_index, top_left_index, top_right_index))

    return squares

def plaquette_link_pairs(Lx, Ly, periodic_x=True, periodic_y=True):
    """
    Creates a list of all the pairs of dimer links in a square lattice. The sites are indexed as `x+Lx*y` for lattice coordinates `(x,y)`. \\
    A dimer is a link between two sites, in the format of a tuple of site indices.
    
    - `periodic_x`, `periodic_y`: bools that control the periodicity in x- and y-directions, respectivelly.

    Returns a list of lists of tuples, where each sub-list contains two tuples, each representing a pair of horizontal or vertical links.\\
    `[`\\
    #pair_1: `[(link1_index1, link1_index_2), (link2_index1, link2_index2)],`\\
    #pair_2: `[...],` \\
    `...`\\
    `]`
    """

    if Lx <=1 or Ly <=1:
        raise ValueError("Cannot have plaquettes on a one-dimensional lattice. Please provide the right dimension.")

    # Step 1: Get the list of square corners
    square_corners = get_square_corners(Lx, Ly, periodic_x, periodic_y)
    
    # List to store the link pairs (both horizontal and vertical)
    plaquette_links = []
    
    # Step 2 and 3: Loop through each square and generate link pairs
    for square in square_corners:
        bottom_left, bottom_right, top_left, top_right = square

        # Horizontal links
        h_list = [(bottom_left, bottom_right), (top_left, top_right)]
        
        # Vertical links
        v_list = [(bottom_left, top_left), (bottom_right, top_right)]
        
        # Step 4: Append both h_list and v_list to the main list
        plaquette_links.append(h_list)
        plaquette_links.append(v_list)
    
    # Return the list of plaquette link pairs
    return plaquette_links

class lattice:
    """
    Contains all the information about the lattice geometry and connections.\\
    Initialized from the lattice dimensions and periodicity.\\
    Spins are indexed as `x+Lx*y` for lattice coordinates `(x,y)`.

    Variables
    ------
    - `Lx`, `Ly`: dimensions of the lattice
    - `periodic_x`, `periodic_y`: bools denoting periodicity

    Properties
    -------
    The following are formated as lists of tuples.
    - `links`: contains index pairs of all the connected sites on the lattice
    - `ylinks`: contains index pairs of connections along the y-direction only
    - `diagolinks`: contains index pairs of all diagonal connections
    - `dimer_pairs`: contains all pairs of links that make two dimers in a lattice plaquette
    """

    def __init__(self, Lx:int, Ly:int, periodic_x:bool = True, periodic_y:bool = True) -> None:
        
        # define lattice dimensions and periodicity
        self.Lx = Lx
        self.Ly = Ly
        self.periodic_x = periodic_x
        self.periodic_y = periodic_y

        # create a list of linked pairs
        self.links = connected_pairs(self.Lx, self.Ly, periodic_x=self.periodic_x, periodic_y=self.periodic_y) # all of them
        self.ylinks = connected_pairs(self.Lx, self.Ly, periodic_x=self.periodic_x, periodic_y=self.periodic_y, along_y=True) # only along y direction
        if self.Lx > 1 and self.Ly > 1:
            self.diagolinks = diagonal_pairs(self.Lx, self.Ly, periodic_x=self.periodic_x, periodic_y=self.periodic_y) # diagonal links
            self.dimer_pairs = plaquette_link_pairs(self.Lx, self.Ly, periodic_x=self.periodic_x, periodic_y=self.periodic_y) # dimer links

    
class space:
    """
    Contains all the vectors in Hilbert space, as well as matrtices for observables.\\
    This is only useful for full summation algorithms.

    Variables
    ------
    - `lat`: lattice object (contains lattice dimensions and bond index pairs)
    - `model`: which Hamiltonian to use? choose: `'heisenberg'`, `'ising'`
    - `driving`: which model should you use for the driving? choose:
    > `'light'`: perturbation of y-bonds,\\
    > `'JQ'`: perturbation by turning on the dimer terms on all 4-point plaquettes,\\
    > `'J1J2'`: perturbation by turning on the diagonal terms,\\
    > `'TF'`: transverse (x) field perturbation.\\
    > **NOTE**: Be careful not set one-dimensional lattices for diagonal and dimer interactions.
    - `afm`: bool for deciding if it's an antiferromagnet

    Properties
    -------
    - `dim`: dimension of the used part of the Hilbert (sub)space
    - `configs`: configurations of the used Hilbert (sub)space
    - `H`: the Hamiltonian
    - `correlations`, `zcorrelations`: a `link -> matrix` dictionary of (z) correlations at the link
    """

    def __init__(self, lat:lattice, 
                 model = 'heisenberg', # which model to use for the Hamiltonian?
                 driving = 'light',
                 afm:bool = True, # is it antiferromagnetic?
                 gauge = True # are you applying the gauge transformation?
                 ) -> None:

        # lattice and Hilbert space settings
        self.grid = lat
        self.N = self.grid.Lx*self.grid.Ly

        if afm and self.N%2 != 0: #if you want an antiferromagnet, but send it an odd N
            raise ValueError("Cannot make antiferromagnetic configuration with odd number of sites.")

        self.model = model
        self.driving = driving
        self.afm = afm # is it antiferromagnetic?
        self.gauge = gauge # do we apply the gauge transformation?

        self._set_configurations() # make vectors of the space and their corresponding lattice configurations
        self._make_hamiltonian() # make the hamiltonian, transform it into what it should be
        self._make_linked_correlations() # makes the correlation matrices between all linked sites
        self._make_sitespins() # makes individual spin matrices on each site

    def _set_configurations(self):
        """
        Creates a list of all the lattice configurations and a mask that filters out only the antiferromagnetic ones.
        """
        self._index_mask = []
        self._all_configs = []

        # Generate all lattice configurations and their indices
        for i in range(2**self.N):
            binary_string = bin(i)[2:].zfill(self.N)  # Convert to binary string and pad with zeros
            configuration = [1 if b == '0' else -1 for b in binary_string]  # Convert binary string to values
            
            self._all_configs.append(configuration)
            
            # Check if the configuration is antiferromagnetic
            if self.afm:# if we want an antiferromagnet
                if sum(configuration) == 0: #check if it's zero and append
                    self._index_mask.append(i)
            else: #if we don't specifically want an antiferromagnet, append it all
                self._index_mask.append(i)
            # you can add here whatever conditions you want

        self._all_configs = np.array(self._all_configs) # set of all vector configurations
        self._index_mask = np.array(self._index_mask) # mas to filter out the indices
        self._afm_vectors = np.eye(2**self.N)[self._index_mask] # subspace-spanning vector set (seems a bit redundand, but eh)
        self.configs = self._all_configs[self._index_mask] # filtered out configurations
        self.dim = len(self.configs) # dimension of reduced Hilbert (sub)space

    #region various Hamiltonian models

    def _make_heisenberg(self):
        """
        Creates a Heisenberg Hamiltonian matrix.
        """
        # go link by link and calculate the contribution
        for link in self.grid.links:
            #self.H += correlation(self.N, link[0], link[1], gauge=self.gauge)
            self.H += new_correlation(self.N, link, gauge=self.gauge)
    
    def _make_ising(self):
        """
        Creates an Ising Hamiltonian matrix.
        """
        # go link by link and calculate the contribution
        for link in self.grid.links:
            #self.H += correlation(self.N, link[0], link[1], gauge=self.gauge, zonly=True)
            self.H += new_correlation(self.N, link, gauge=self.gauge, direction='z')

    def _make_light(self):
        """
        Creates a matrix consisting of correlation links in the y-direction.
        """
        for link in self.grid.ylinks:
            self._Hp += new_correlation(self.N, link, gauge=self.gauge)
    
    def _make_cross(self):
        """
        Creates a matrix of all diagonal correlations.
        """
        for link in self.grid.diagolinks:
            self._Hp += new_correlation(self.N, link, gauge=self.gauge)
    
    def make_dimers(self):
        """
        Creates a matrix of dimer interactions on all the plaquettes of the lattice.
        """

        # 1. go over all the link pairs (both horizontal and vertical are included)
        for link_pair in self.grid.dimer_pairs:
            # 2. take one link from the pair and calculate the left interaction element
            c_left = new_correlation(self.N, link_pair[0], gauge=self.gauge)
            left = c_left - np.eye(len(c_left))

            # 3. do the same with the right element
            c_right = new_correlation(self.N, link_pair[1], gauge=self.gauge)
            right = c_right - np.eye(len(c_right))

            # 4. add to the whole
            self._Hp += 1/4. * left@right # 1/4 is there for computational units
        
    def make_transverse(self):
        """
        Creates a matrix of all single pauli x-matrices.
        """
        for i in range(self.N):
            self._Hp += new_correlation(self.N, (i,), direction='x')

    #endregion

    def _make_hamiltonian(self):
        """
        Creates the Hamiltonian matrix in full Hilbert space. Then, it truncates the matrix into the working subspace.\\
        If `gauge` is True, then all the non-diagonal elements of the Hamiltonian change sign.
        """

        self.H = np.zeros((2**self.N,2**self.N), dtype=complex) #start with zeros
        self._Hp = np.zeros((2**self.N,2**self.N), dtype=complex) #save the perturbed Hamiltonian (without constant)

        # # go link by link and calculate the contribution
        # for link in self.grid.links:
        #     self.H += correlation(self.N, link[0], link[1], gauge=self.gauge)
        
        # #do the same, just for y links only
        # for link in self.grid.ylinks:
        #     self._Hp += correlation(self.N, link[0], link[1], gauge=self.gauge)

        # NOTE: catalogue of Hamiltonian models
        if self.model == 'heisenberg':
            self._make_heisenberg()
        elif self.model == 'ising':
            self._make_ising()
        else:
            raise ValueError("Wrong model given. Please use a supported Hamiltonian model.")
        
        # NOTE: catalogue of driving models
        if self.driving == 'light':
            self._make_light()
        elif self.driving == 'J1J2':
            self._make_cross()
        elif self.driving == 'JQ':
            self.make_dimers()
        elif self.driving == 'TF':
            self.make_transverse()
        else:
            raise ValueError("Wrong model given. Please use a supported driving model.")

        # # perturbation - light-matter interaction Hamiltonian
        # for link in self.grid.ylinks:
        #     self._Hp += correlation(self.N, link[0], link[1], gauge=self.gauge)
        
        # project into the antiferromagnetic subspace
        self.H = self._afm_vectors@self.H@self._afm_vectors.T
        self._Hp = self._afm_vectors@self._Hp@self._afm_vectors.T
        
        # a super fancy way to apply the gauge transformation
        # if self.gauge:
        #     mask = ~np.eye(self.H.shape[0], dtype=bool)
        #     self.H[mask] *= -1
        #     self._Hp[mask] *= -1
        
        self._H0 = self.H.copy() # you use H in general, but you also need to keep track of H0 for the perturbations

    def perturb(self, perturbation):
        """
        Pertubs the Hamiltonian with the prescribed perturbation protocol with the some `perturbation` strength.\\
        For more information, see `model` and `driving` properties of the class.
        """
        self.H = self._H0 + perturbation*self._Hp

    def make_correlation(self, pair):
        """
        Makes the matrix for full- and z-correlation between the selected `pair` of lattice indices. The pair does not need to be linked by lattice goemetry.\\
        Projects them into the Hilbert subspace (antiferromagnet).

        Appends them into the correlation dictionaries.
        """
        # make, reduce, and append the correlation
        #corr = correlation(self.N, pair[0], pair[1], gauge = self.gauge)
        corr = new_correlation(self.N,pair, gauge=self.gauge)
        corr = self._afm_vectors@corr@self._afm_vectors.T #reduce to filtered subspace
        self.correlations[pair] = corr.copy()

        # z correlations
        #zcorr = correlation(self.N, pair[0], pair[1], zonly=True)
        zcorr = new_correlation(self.N, pair, gauge = self.gauge, direction='z')
        zcorr = self._afm_vectors@zcorr@self._afm_vectors.T
        self.zcorrelations[pair] = zcorr

        # y correlations
        ycorr = new_correlation(self.N,pair, gauge=self.gauge, direction='y')
        ycorr = self._afm_vectors@ycorr@self._afm_vectors.T
        self.ycorrelations[pair] = ycorr

       # x correlations
        xcorr = new_correlation(self.N,pair, gauge=self.gauge, direction='x')
        xcorr = self._afm_vectors@xcorr@self._afm_vectors.T
        self.xcorrelations[pair] = xcorr

    def _make_sitespins(self):
        """
        Creates the matrices of all individual (x,y,z)-spins at each site.
        """
        # create empty dictionaries
        self.zsitespins = {}
        self.ysitespins = {}
        self.xsitespins = {}

        for i in range(self.N):
            # x
            matrix = new_correlation(self.N, (i,), gauge=self.gauge, direction='x')
            matrix = self._afm_vectors@matrix@self._afm_vectors.T
            self.xsitespins[(i,)] = matrix.copy()

            # y
            matrix = new_correlation(self.N, (i,), gauge=self.gauge, direction='y')
            matrix = self._afm_vectors@matrix@self._afm_vectors.T
            self.ysitespins[(i,)] = matrix.copy()

            # z
            matrix = new_correlation(self.N, (i,), gauge=self.gauge, direction='z')
            matrix = self._afm_vectors@matrix@self._afm_vectors.T
            self.zsitespins[(i,)] = matrix.copy()

    def _make_linked_correlations(self):
        """
        Makes the full- and (x,y,z)-correlation matrices between all the linked sites.

        Property type is a `link -> matrix` dictionary.\\
        Use the `ylinks` property from the `lattice` class as keys to access the matrices corresponding to perturbed bonds, or `links` to access all matrices.

        ADDED: you can also make correlations between unlinked sites, but you have to supply the list of site pairs yourself. Use the `make_correlation(pair)` function. 
        """
        # initialize the dictionaries
        self.correlations = {}
        self.zcorrelations = {}
        self.ycorrelations = {}
        self.xcorrelations = {}

        for link in self.grid.links:
            
            self.make_correlation(link)
            # # full correlations
            # corr = correlation(self.N, link[0], link[1], gauge=self.gauge)
            # corr = self._afm_vectors@corr@self._afm_vectors.T #reduce to filtered subspace
            # # if self.gauge: #apply gauge transformation?
            # #     mask = ~np.eye(corr.shape[0], dtype=bool)
            # #     corr[mask] *= -1 # a neater way to do this?
            # self.correlations[link] = corr.copy()


            # # z correlations
            # zcorr = correlation(self.N, link[0], link[1], zonly=True)
            # zcorr = self._afm_vectors@zcorr@self._afm_vectors.T
            # self.zcorrelations[link] = zcorr
    

    # NOTE: finish this function if you really need to. but normally, you can just send a link list to other parts of the code and the matrices will be made there.
    # def make_diagonal_correlations(self):
    #     """
    #     Creates the matrices of all diagonal correlations. They are put in the `correlaotins[link]` dictionary.\\
    #     Called only when there are diagonal terms present (such as in the `J1J2` model).
        
    #     **NOTE**: does not work unless you initialized the correlation dictionaries already.
    #     """
    #     for link in self.grid.diagolinks