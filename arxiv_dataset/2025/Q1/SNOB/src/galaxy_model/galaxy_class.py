import logging
from pathlib import Path
logging.basicConfig(level=logging.INFO) 
import numpy as np
import src.galaxy_model.association_class as asc
import src.utilities.utilities as ut
import src.galaxy_model.galaxy_density_distr as gdd
import matplotlib.pyplot as plt
import src.utilities.constants as const
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import MaxNLocator


#AVG_SN_PER_ASC = np.array([204, 620, 980]) # number of star formation episodes = 1, 3, 5
SN_BIRTHRATE = 2.81e4 # units of SN/Myr
ASC_BIRTHRATE = np.array([3084, 3085, 3085])  # number of associations created per Myr
SNS_BIRTHRATE = np.array([23270, 26600, 28100])  # number of supernovae progenitors created per Myr
AVG_NUM_SNP_PER_ASC = np.array([8, 9, 9]) # small and similar number - should just take 10 for all
STAR_FORMATION_EPISODES = [1, 3, 5]
C = [0.828, 0.95, 1.0] # values for the constant C for respectively 1, 3 and 5 star formation episodes

class Galaxy():
    rng = np.random.default_rng() # random number generator to be used for drawing association position
    # The following class variables are used to store the grid and the density distribution used for drawing the associations. 
    # If several Galaxy instances are created, these values are only generated once.
    x_grid = None
    y_grid = None
    z_grid = None
    uniform_spiral_arm_density = None # strictly not needed, but useful for testing
    emissivity = None
    
    
    def __init__(self, sim_time_duration, read_data_from_file = True, star_formation_episodes=5):
        """ Class to represent the galaxy. The galaxy is created at a given time and contains a number of associations.
        
        Args:
            sim_time_duration (int): The duration of the simulation in units of Myr
            read_data_from_file (bool, optional): Whether to read the Galaxy data from a file or generate them. Defaults to True (aka read from file).
            star_formation_episodes (int, optional): The number of star formation episodes. Defaults to 1.
            
        Returns:
            None
        """
        if not isinstance(sim_time_duration, int):
            raise TypeError("Simulation time duration must be an integer.")
        # Make sure that star_formation_episodes is a valid number of star formation episodes, i.e. that the number is in the list STAR_FORMATION_EPISODES
        if star_formation_episodes not in STAR_FORMATION_EPISODES:
            raise ValueError(f"Invalid number of star formation episodes. The number of star formation episodes must be one of the following: {STAR_FORMATION_EPISODES}")
        
        # Only generate data if it hasn't been generated yet
        if Galaxy.x_grid is None:
            self.generate_coords_densities(read_data_from_file)

        self._sim_time_duration = sim_time_duration # assuming the simulation time duration is in Myr
        self._star_formation_episodes = star_formation_episodes
        self._star_formation_episodes_index = STAR_FORMATION_EPISODES.index(star_formation_episodes)
        self._asc_birthrate = ASC_BIRTHRATE[self._star_formation_episodes_index]  # number of associations created per Myr
        self._galaxy = [] # List for storing all associations in the Galaxy
        self._generate_galaxy(sim_time_duration, self._asc_birthrate, C[self._star_formation_episodes_index])
    
    
    @classmethod
    def generate_coords_densities(cls, read_data_from_file):
        # generate the grid and the density distribution used for drawing the associations
        x_grid, y_grid, z_grid, uniform_spiral_arm_density, emissivity = gdd.generate_coords_densities(read_data_from_file=read_data_from_file)
        uniform_spiral_arm_density = uniform_spiral_arm_density / np.sum(uniform_spiral_arm_density) # normalize the density to unity
        emissivity = emissivity / np.sum(emissivity) # normalize the density to unity
        cls.x_grid = x_grid
        cls.y_grid = y_grid
        cls.z_grid = z_grid
        cls.uniform_spiral_arm_density = uniform_spiral_arm_density
        cls.emissivity = emissivity


    @ut.timing_decorator
    def _generate_galaxy(self, sim_time_duration, asc_birthrate, c):
        """ Method to generate the galaxy. The galaxy is created at a given time and contains a number of associations. Iterates over the simulation time and updates the associations and supernovae progenitors.
        
        Args:
            sim_time_duration (int): The duration of the simulation in units of Myr
            asc_birthrate (int): The number of associations created per Myr
            c (float): Number of star formation episodes
        
        Returns:
            None
        """
        self._calculate_association_position_batch(asc_birthrate, c, sim_time_duration) # add the first batch of associations, created at the beginning of the simulation
        logging.info(f'Simulation time: {sim_time_duration}')
        for sim_time in range(sim_time_duration - 1, 0, -1): # iterate over the simulation time, counting down to the present. sim_time_duration - 1 because the first batch of associations is already added and we don't generate new associations at sim_time = 0
            if sim_time % 10 == 0:
                logging.info(f'Simulation time: {sim_time}')
            self._calculate_association_position_batch(asc_birthrate, c, sim_time)
        self._update_snps(0) # update the supernovae progenitors to the present time (0 Myr)
        self._update_exploded_supernovae() # update the list of exploded supernovae


    @property
    def associations(self): # property to get the associations in the galaxy
        return np.array(self._galaxy)
    
    @property
    def num_asc(self):
        return len(self._galaxy)
    
    @property
    def sim_time_duration(self):
        return self._sim_time_duration
    
    @property
    def asc_birthrate(self):
        return self._asc_birthrate
    
    @property
    def star_formation_episodes(self):
        return self._star_formation_episodes
    

    def _association_distribution(self, n_min, n_max, alpha = 0.8):
        constant = 1.65 * 1.7e6 * 1.1e-3 # 1.1e-3 = f_SN = the fraction of stars that end their lives as core-collapse supernovae
        N = np.arange(n_min, n_max + 1) # default step size is 1. The range is inclusive of n_max
        distribution = constant / N**(1 + alpha)
        return distribution / np.sum(distribution)
    
    
    def _calc_num_associations(self, n_min, n_max, c):
        N = np.arange(n_min, n_max + 1)
        # Using the normalized association distribution to draw N
        distribution = self._association_distribution(n_min, n_max)
        # Draw the actual number of associations as given by a random multinomial distribution
        num_snp_drawn = []
        num_snp_target = SNS_BIRTHRATE[self._star_formation_episodes_index]
        count = 0
        while np.sum(num_snp_drawn) < num_snp_target*0.99:
            count += 1
            new_num_snp_drawn = self.rng.choice(a=N, size=1, p=distribution)
            new_num_snp_drawn = np.ones(self._star_formation_episodes) * new_num_snp_drawn # As Mckee and Williams 1997 finds, the star forming episodes are of equal size most probably
            num_snp_drawn.append(new_num_snp_drawn)
        return num_snp_drawn
        

    def _update_snps(self, sim_time):
        """ Method to update the supernovae progenitors to the given simulation time."""
        for association in self._galaxy:
            association.update_sn(sim_time)


    def _update_exploded_supernovae(self):
        """ Method to update the list of exploded supernovae. The list is updated at the end of the simulation."""
        exploded_sn = [] # list for storing all exploded supernovae
        for association in self._galaxy:
            for sn in association.supernovae:
                if sn.exploded:
                    exploded_sn.append(sn)
        self._exploded_sn = exploded_sn 


    def _calculate_association_position_batch(self, asc_birthrate, c, sim_time):
        """ Method to calculate the positions of the associations. The positions are calculated at each step of the simulation.
        
        Args:
            asc_birthrate (int): The number of associations created per Myr
            c (float): Number of star formation episodes
            sim_time (int): The current time of the simulation in units of Myr. In the simulation, this will decrease on each iteration, counting down from the creation_time to the present
            
        Returns:
            None
        """
        n_min = 2
        n_max = 1870 #* self._star_formation_episodes
        num_snp = self._calc_num_associations(n_min, n_max, c)
        #print(f'Number of associations: {len(num_snp)}. Number of SNPs: {np.sum(num_snp)}. Number of SNPs per association: {np.mean(num_snp)}. Number of star formation episodes: {self._star_formation_episodes}. Min num snp: {np.min(num_snp)}. Max num snp: {np.max(num_snp)}.')
        grid_indexes = self.rng.choice(a=len(self.emissivity), size=len(num_snp), p=self.emissivity) 
        xs = self.x_grid[grid_indexes] # get the x-values for the drawn associations
        ys = self.y_grid[grid_indexes] # get the y-values for the drawn associations
        zs = self.z_grid[grid_indexes] # get the z-values for the drawn associations
        for i in range(len(num_snp)):
            self._galaxy.append(asc.Association(x=xs[i], y=ys[i], z=zs[i], association_creation_time=sim_time, c=c, n=num_snp[i])) # add the association to the galaxy
    

    def get_exploded_supernovae_masses(self):
        """ Method to get the masses of the exploded supernovae progenitors."""
        exploded_sn_masses = [sn.mass for sn in self._exploded_sn]
        return exploded_sn_masses
    

    def get_exploded_supernovae_ages(self):
        """ Method to get the ages of the exploded supernovae progenitors."""
        exploded_sn_ages = [sn.age for sn in self._exploded_sn]
        return exploded_sn_ages
    
    
    def get_exploded_supernovae_longitudes(self):
        """ Method to get the longitudes of the exploded supernovae progenitors."""
        exploded_sn_longitudes = [sn.longitude for sn in self._exploded_sn]
        return exploded_sn_longitudes
    

def plot_temporal_clustering():
    # check if the folder exists, if not create it
    Path(const.FOLDER_GALAXY_TESTS).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    #plt.figure()
    max_number_snp = 10e3
    snps = np.arange(0, max_number_snp, 1)
    for i, c in enumerate(C):
        plt.plot(snps, c-0.11*np.log(snps), label=f'Number of star forming episodes = {STAR_FORMATION_EPISODES[i]}')
    plt.xlabel('$N_*^{\\text{SN}}$', fontsize=14)
    plt.ylabel('P(> $N_*^{\\text{SN}})$', fontsize=14)
    #plt.title('Temporal clustering of supernovae progenitors')
    plt.xscale('log')
    plt.xlim(1, max_number_snp)
    plt.ylim(0, 1.0)
    # Set y-axis ticks to increment by 0.1
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.legend(fontsize = 12)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.savefig(f'{const.FOLDER_GALAXY_TESTS}/temporal_clustering_analytical.pdf')
    plt.close()


def main():  
    plot_temporal_clustering()
    simulation_time = 100
    galaxy_1 = Galaxy(simulation_time, star_formation_episodes=1)
    galaxy_3 = Galaxy(simulation_time, star_formation_episodes=3)
    galaxy_5 = Galaxy(simulation_time, star_formation_episodes=5)
    print(f'Number of associations in galaxy_1 born per Myr: {galaxy_1.num_asc / simulation_time}')
    print(f'Number of associations in galaxy_3 born per Myr: {galaxy_3.num_asc / simulation_time}')
    print(f'Number of associations in galaxy_5 born per Myr: {galaxy_5.num_asc / simulation_time}')
    #Galaxy(50)

if __name__ == "__main__":
    main()
