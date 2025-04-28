import numpy as np
import src.galaxy_model.supernovae_class as sn
import src.utilities.utilities as ut
import src.utilities.constants as const


class Association(): 
    solar_masses = np.arange(8, 120 + 0.01, 0.01) # mass in solar masses. Used to draw random masses for the SNPs in the association
    imf = ut.imf_3(solar_masses) # the imf for the range 1 <= M/M_sun < 120
    rng = np.random.default_rng()

    def __init__(self, x, y, z, association_creation_time, c, n):
        """ Class to represent an association of supernovae progenitors. The association is created at a given position and time. 
        The number of SNPs in the association is calculated from the number of star formation episodes and IMF. 
        The SNPs are generated at the time of the association's creation.
        
        Args:
            x (float): x-coordinate of the association. Units of kpc
            y (float): y-coordinate of the association. Units of kpc
            z (float): z-coordinate of the association. Units of kpc
            association_creation_time (int): how many years ago the association was created. Units of Myr
            c (int): number of star formation episodes
            n (int, optional): number of SNPs in the association. If None, a random number of SNPs is drawn. Otherwise, the given number of SNPs is used.
            
        Returns:
            None
        """
        self.__x = x
        self.__y = y
        self.__z = z
        self.__r = self._calculate_heliocentric_distance() # heliocentric distance of the association
        self.__association_creation_time = association_creation_time # The time when the association is created. Units of Myr
        self.__simulation_time = association_creation_time # when the association is created, the simulation time is the same as the creation time. Goes down to 0
        self.__n = n #self._calculate_num_sn(c, n)
        self._generate_sn_batch() # list containting all the supernovae progenitors in the association

    
    @property
    def x(self):
        return self.__x
    
    @property
    def y(self):
        return self.__y
    
    @property
    def z(self):
        return self.__z
    
    @property
    def r(self):
        return self.__r
    
    @property
    def age(self):
        return self.__association_creation_time
    
    @property
    def number_sn(self):
        return self.__n
    
    @property
    def supernovae(self):
        return self.__supernovae # list to store all generated supernovae progenitors in the association
    
    @property
    def star_masses(self):
        star_masses = []
        for snp in self.supernovae:
            star_masses.append(snp.mass)
        return star_masses 
    

    def _calculate_heliocentric_distance(self):
        """ Method to calculate the heliocentric distance of the association. The heliocentric distance is calculated as the distance from the association to the origo.
        
        Args:
            None
        
        Returns:
            float: the heliocentric distance of the association
        """
        return np.sqrt(self.x**2 + (self.y - const.r_s)**2 + self.z**2) # subtract the distance from the Sun to the Galactic center in order to get the heliocentric distance
    

    def _calculate_num_sn(self, c, n):
        """ Function to calculate the number of SNPs in the association. If n is None, a random number of SNPs is drawn. Otherwise, the given number of SNPs is used.
        
        Args:
            c (int): number of star formation episodes
            n (int): number of SNPs in the association. If None, a random number of SNPs is drawn. Otherwise, the given number of SNPs is used.

        Returns:
            n (int): number of SNPs in the association
        """
        if n==None: # draw random number of SNPs
            print("n is None. Random number of SNPs are drawn")
            return int(np.ceil(np.exp((c - self.rng.random())/0.11))) 
        else: # use the given number of SNPs
            return n
    

    def _generate_sn_batch(self):
        """ Function to generate a batch of SNPs. The number of SNPs is given by the attribute self.__n and stored in the list self.__supernovae.
        Each SNP is given a random mass, a random velocity, a random lifetime and a random direction for the velocity dispersion. 
        The random values are drawn from the initial mass function and a Gaussian distribution.
        
        Args:
            None

        Returns:
            None
        """
        
        size = np.sum(self.__n, dtype=int) # total number of SNPs in the association
        try:
            sn_masses = self.rng.choice(self.solar_masses, size=size, p=self.imf/np.sum(self.imf)) # draw random masses for the SNPs in the association from the IMF in the range 8 <= M/M_sun < 120
        except TypeError:
            print(f'TypeError: self.__n = {self.__n}, size = {size}')
        assert const.f_binary < 1, "The fraction of massive stars born in binaries must be less than 1!"
        #self.__star_masses = sn_masses
        one_dim_velocities = np.abs(self.rng.normal(loc=0, scale=2, size=size)) # Gaussian velocity distribution with a mean of 0 km/s and a standard deviation of 2 km/s
        lifetimes = ut.lifetime_as_func_of_initial_mass(sn_masses)   # Units of Myr. Formula from Schulreich et al. (2018)
        vel_theta_dirs = self.rng.uniform(0, np.pi, size=size)   # Velocity dispersion shall be isotropic
        vel_phi_dirs = self.rng.uniform(0, 2 * np.pi, size=size) # Velocity dispersion shall be isotropic
        self.__supernovae = []
        snp_index = -1
        for i in range(len(self.__n)):
            #print('---------------------------------------------- i =', i, '----------------------------------------------')
            num_binary_stars_in_episode = round(int(self.__n[i] * const.f_binary) / 2) * 2 # number of binary stars in the i-th star formation episode. round(n / 2) * 2 is to ensure that the number of binary stars is even
            n = 0
            while n < int(self.__n[i]):  # Loop until n reaches the expected count. self.__n[i] is the number of SNPs in the i-th star formation episode
                # below: -4 * i is to take into account multiple star formation episodes. The first SNP is created at the time of the association's creation, and the rest are created at later times. 4 myrs between each star formation episode
                snp_index += 1
                try:
                    if n < num_binary_stars_in_episode: # if the SNP is part of a binary star
                        binary_one = sn.Supernovae(self.x, self.y, self.z, self.__association_creation_time - 4 * i, self.__simulation_time, sn_masses[snp_index], one_dim_velocities[snp_index], 
                                                       lifetimes[snp_index], vel_theta_dirs[snp_index], vel_phi_dirs[snp_index])
                        binary_two = sn.Supernovae(self.x, self.y, self.z, self.__association_creation_time - 4 * i, self.__simulation_time, sn_masses[snp_index + 1], one_dim_velocities[snp_index], 
                                                       lifetimes[snp_index + 1], vel_theta_dirs[snp_index], vel_phi_dirs[snp_index])
                        binary_one.set_companion(binary_two)
                        binary_two.set_companion(binary_one)
                        self.__supernovae.append(binary_one)
                        self.__supernovae.append(binary_two)
                        # the binaries are added to the list, so we now need to increase the snp_index and n by 1. Note this will always work since const.f_binary is always less than one
                        n += 2
                        snp_index += 1
                    else:
                        self.__supernovae.append(sn.Supernovae(self.x, self.y, self.z, self.__association_creation_time - 4 * i, self.__simulation_time, sn_masses[snp_index], one_dim_velocities[snp_index], 
                                                   lifetimes[snp_index], vel_theta_dirs[snp_index], vel_phi_dirs[snp_index]))
                        n += 1  # Increment normally for single stars
                except IndexError:
                    print(f'IndexError: i = {i}, n = {n}, snp_index = {snp_index}, len(sn_masses) = {len(sn_masses)}, self.__n = {self.__n}')
        #[sn.Supernovae(self.x, self.y, self.z, self.__association_creation_time, self.__simulation_time, sn_masses[i], one_dim_velocities[i], lifetimes[i], vel_theta_dirs[i], vel_phi_dirs[i]) for i in range(self.__n)]
    

    def update_sn(self, new_simulation_time): # update each individual SNP in the association
        """ Function to update the simulation time for each SNP in the association.
        The simulation time is updated to the new_simulation_time, and the boolean exploded attribute is also updated.
        
        Args:
            new_simulation_time (int): the new simulation time in units of Myr
            
        Returns:
            None
        """
        self.__simulation_time = new_simulation_time # update the simulation time for the association
        for sn in self.__supernovae:
            sn.update_snp(new_simulation_time)
    

    def plot_association(self, ax, simulation_time):
        """ Function to plot the association in the galactic plane, with the centre of the association as origo.
        The centre of the association is plotted in blue, and the SNPs are plotted in red if they have exploded, and black if they have not exploded.
        Also, each SNP is updated to the given simulation time in this function, so update_sn is not needed to be called before this function.
        
        Args:
            ax (matplotlib.axes.Axes): axis object from matplotlib to which the association is plotted
            simulation_time (int): the current simulation time in units of Myr
        
        Returns:
            None
        """
        ax.scatter(0, 0, 0, s=10, color='blue', label='Centre of association') # plot the centre of the association
        for sn in self.__supernovae: # plot the SNPs in the association. Both exploded (red) and unexploded (black)
            sn.update_snp(simulation_time)
            sn.plot_sn(ax)


    def print_association(self, prin_snp=False):
        """ Function to print the association. It prints the number of SNPs in the association, the position of the centre of the association and the simulation time. 
        If prin_snp is True, it also prints info on the SNPs in the association.
        
        Args:
            prin_snp (boolean, optional): If True, prints info on the SNPs in the association as well. Defaults to False.
            
        Returns:
            None
        """
        print(f"Association contains {self.__n} Supernovae Progenitors and its centre is located at xyz position ({self.x}, {self.y}, {self.z}). Simulation time: {self.__simulation_time} yrs.")
        if prin_snp:
            for sn in self.__supernovae:
                sn.print_sn()
