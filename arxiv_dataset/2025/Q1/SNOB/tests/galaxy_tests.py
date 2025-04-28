import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.lines import Line2D
import logging
logging.basicConfig(level=logging.INFO) # other levels for future reference: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL
import src.galaxy_model.galaxy_density_distr as gdd
import src.galaxy_model.association_class as ass # Note that importing this causes the coordinates and densities for the Galaxy to be generated
import src.galaxy_model.galaxy_class as galaxy
import src.utilities.utilities as ut
import src.utilities.constants as const
import src.utilities.utilities as util


N = 10e4 # number of associations in the Galaxy
T = 20 # simulation run time in Myrs
star_formation_episodes = [1, 3, 5]
C = [0.828, 0.95, 1.0] # values for the constant C for respectively 1, 3 and 5 star formation episodes
    

def plot_cum_snp_cluster_distr(galaxies, C=C):
    """
    Function to plot the cumulative distribution of stellar clusters as function of number of SNPs.

    Args:
        galaxies: list of Galaxy class instances, storing the galaxies for different values of C, the number of star formation episodes.
        C: List of values representing the number of star formation episodes per galaxy.

    Returns:
        None. Saves a plot in the output folder.
    """
    plt.figure(figsize=(10, 6))  # Adjust figure size as needed

    for i, galaxy in enumerate(galaxies):
        print(f'Number of associations in galaxy {i}: {galaxy.num_asc}')
        association_array_num_sn = [np.sum(asc.number_sn) for asc in galaxy.associations]
        num_bins = int(np.ceil(max(association_array_num_sn)) + 1)  # Adding 1 to include the maximum value

        # Creating bins and histogram
        counts, bin_edges = np.histogram(association_array_num_sn, bins=range(1, num_bins + 1))
        cumulative = 1 - np.cumsum(counts) / galaxy.num_asc  # normalized cumulative distribution

        plt.plot(bin_edges[:-1], cumulative, label=f"{C[i]} episodes. Total number of OB stars: {np.sum(association_array_num_sn):.2f}")

    plt.xscale("log")
    plt.xlim(1, num_bins - 1)  # Adjust x-axis limits to exclude the last empty bin edge
    plt.ylim(0, 1)  # set the y-axis limits
    plt.xlabel("Number of SNPs", fontsize=14)
    plt.ylabel("Cumulative distribution. P(N > x)", fontsize=14)
    #plt.title("Monte Carlo simulation of temporal clustering of SNPs")
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(fontsize = 12)
    plt.savefig(f'{const.FOLDER_GALAXY_TESTS}/temporal_clustering.pdf')  # Adjust path as necessary
    plt.close()


def plot_sn_as_func_of_long(galaxy, sim_time, num_star_forming_episodes):
    """ Function to plot the probability density function of SNPs as function of longitude
    
    Args:
        galaxy: The galaxy to plot the SNP probability density function for
        
    Returns:
        None. Saves a plot in the output folder
    """
    logging.info("Plotting the probability density function of SNPs as function of longitude")
    exploded_sn_long = np.degrees(galaxy.get_exploded_supernovae_longitudes())
    num_sn = len(exploded_sn_long)
    logging.info(f"Number of supernovae: {num_sn}")
    # create bin edges for the binning
    bin_edges_long = np.arange(0, 362.5, 2.5) # will end at 360. No data is left out. 145 bin edges
    hist, _ = np.histogram(exploded_sn_long, bins=bin_edges_long) # if a longitude is in the bin, add the intensity to the bin
    # Rearange data to be plotted in desired format
    rearanged_hist = ut.rearange_data(hist) / num_sn
    # Create bin_edges for the plot 
    bin_edges_central = np.arange(2.5, 360, 5)
    bin_edges = np.concatenate(([0], bin_edges_central, [360]))
    # Plot using stairs
    plt.figure(figsize=(10, 6))
    plt.stairs(values=rearanged_hist, edges=bin_edges, fill=False, color='black')
    # Set up the plot
    plt.xlabel("Galactic longitude l (degrees)", fontsize=14)
    x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
    plt.xticks(np.linspace(0, 360, 13), x_ticks)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(3)) 
    plt.ylabel("P(SN)", fontsize=14)
    #plt.title(f"Probability density function of SNPs as function of longitude.\n Result after {sim_time} Myr and {num_star_forming_episodes} star formation episodes.")   
    plt.text(0.02, 0.95, fr'Number of associations: {galaxy.num_asc}', transform=plt.gca().transAxes, fontsize=12, color='black')
    plt.text(0.02, 0.90, fr'Total number of Supernovae: {num_sn}', transform=plt.gca().transAxes, fontsize=12, color='black')
    plt.ylim(0, max(rearanged_hist)*1.2) # set the y axis limits
    plt.xlim(0, 360) # so that the plot starts at 0 and ends at 360
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.savefig(f'{const.FOLDER_GALAXY_TESTS}/sn_as_func_of_long.pdf')     # save plot in the output folder
    plt.close()


def plot_mass_distr(galaxy):
    """ Function to plot the probability distribution for the mass of SNP's. SNP's from a generated galaxy
    
    Args:
        galaxy: The galaxy to plot the SNP mass distribution for
    
    Returns:
        None. Saves a plot in the output folder
    """
    # Add the actual Kroupa IMF to the plot
    mass, imf = ut.imf() # entire IMF
    m3 = np.arange(8, 120, 0.01) # mass for the last part of the IMF
    imf3 = ut.imf_3(m3) # the imf for the range 8 <= M/M_sun < 120
    imf = imf / np.sum(imf3) / 0.01 # normalize the imf to unity for the mass range 8 <= M/M_sun < 120
    # Modelled data
    drawn_masses = np.array([]) # array to store the drawn masses (M/M_sun) for the supernovae progenitors
    number_sn = 0
    for asc in galaxy.associations:
        number_sn += np.sum(asc.number_sn)
        drawn_masses = np.concatenate((drawn_masses, asc.star_masses))
    #drawn_masses = drawn_masses.flatten()
    drawn_masses = drawn_masses[drawn_masses > 7]
    mass_max = int(np.ceil(max(drawn_masses))) 
    mass_min = int(np.floor(min(drawn_masses))) # minimum number of stars = 0
    binwidth = 1
    bins = np.arange(mass_min, mass_max + binwidth, binwidth)
    counts, _ = np.histogram(drawn_masses, bins=bins)
    counts = counts / np.sum(counts) / binwidth 
    plt.figure(figsize=(8, 8))
    plt.plot(bins[:-1], counts, label='Stellar masses in modelled Galaxy', color='blue')
    plt.plot(mass, imf, label='The modified Kroupa IMF', color='black', linestyle='dashed')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(np.min(mass), mass_max + 30) # set the x axis limits
    plt.ylim(top=np.max(imf)*20) # set the y axis limits
    plt.xlabel("Mass of SN progenitor (M$_\odot$)")
    plt.ylabel("Probability distribution. P(M$_\odot$)")
    plt.legend(loc='lower left')
    #plt.title("Probability distribution for the mass of SN progenitors\nNormalized to unity for the mass range $8 \leq $M/M$_\odot < 120$")
    plt.text(0.02, 0.97, fr'Number of associations: {galaxy.num_asc}', transform=plt.gca().transAxes, fontsize=8, color='black')
    plt.text(0.02, 0.95, fr'Total number of supernovae progenitors: {number_sn}', transform=plt.gca().transAxes, fontsize=8, color='black')
    plt.savefig(f'{const.FOLDER_GALAXY_TESTS}/sn_mass_distribution.pdf')     # save plot in the output folder
    plt.close()


def plot_association_3d(ax, association, simulation_time):
    """ Function to plot the association in 3D
    
    Args:
        ax: The axis to plot the association on
        association: The association to plot
        simulation_time: The time of the simulation
    Returns:
        None. Adds the plot to the axis
    """
    association.plot_association(ax, simulation_time)
    ax.set_xlabel('X (pc from AC.)')
    ax.set_ylabel('Y (pc from AC.)')
    ax.set_zlabel('Z (pc from AC.)')
    ax.set_title(f"Position of Supernovae and OB stars {simulation_time} Myr ago.")

       
def plot_diffusion_of_sns_3d():
    """ Function to plot the diffusion of SNP's in 3D, from 40 Myr ago to 1 Myr ago. Looks at one specific association, and how the SNP's diffuse away from the association centre
    
    Args:
        None
    Returns:
        Saves a plot in the output folder
    """
    logging.info("Plotting the diffusion of SNPs in 3D, from 40 Myr ago to 1 Myr ago")
    creation_time = 40 # Myr ago
    test_ass = ass.Association(x=0, y=8, z=0, n=[20], association_creation_time=creation_time, c=1) # Create a test association
    
    # Create a figure
    fig = plt.figure(figsize=(11, 10)) # Make the figure
    # Adding the first subplot. For the association at 40 Myr ago
    ax1 = fig.add_subplot(221, projection='3d')
    plot_association_3d(ax1, test_ass, simulation_time=40)

    test_ass.update_sn(20)
    ax2 = fig.add_subplot(222, projection='3d')
    plot_association_3d(ax2, test_ass, simulation_time=20)

    test_ass.update_sn(10)
    ax3 = fig.add_subplot(223, projection='3d')
    plot_association_3d(ax3, test_ass, simulation_time=10)

    test_ass.update_sn(1)
    ax4 = fig.add_subplot(224, projection='3d')
    plot_association_3d(ax4, test_ass, simulation_time=1)
    
    """ legend_exploded = Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=4, label='Exploded')
    legend_unexploded = Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=3, label='Not Exploded')
    legend_centre = Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=5, label='Association centre (AC)')
    handles = [legend_centre, legend_exploded, legend_unexploded]
    fig.legend(handles=handles) """
    #plt.suptitle(f"Position of Association centre in xyz-coordinates (kpc): ({test_ass.x:.2f}, {test_ass.y:.2f}, {test_ass.z:.2f}) \nAssociation created {creation_time} Myr ago")
    plt.savefig(f'{const.FOLDER_GALAXY_TESTS}/diffusion_of_sns.pdf')
    plt.close()


def plot_age_mass_distribution():
    """ Function to highlight the relation between mass and expected lifetime of stars

    Args:
        None
    Returns:
        Saves a plot in the output folder
    """
    logging.info("Generating plot highlighting the relation between mass and expected lifetime of stars")
    mass = np.arange(8, 120.1, 0.1)
    lifetime = ut.lifetime_as_func_of_initial_mass(mass) # Myr
    plt.figure(figsize=(10,6))
    plt.plot(mass, lifetime, zorder=0)
    #plt.title("Lifetime as function of initial stellar mass")
    plt.xlabel("Mass of SN progenitor (M$_\odot$)", fontsize=14)
    plt.ylabel("Lifetime (Myr)", fontsize=14)
    x_vals = [8, 20, 40, 60, 80, 100, 120] # Selected masses for which ages will be highlighted 
    y_vals = ut.lifetime_as_func_of_initial_mass(x_vals)
    for i, x in enumerate(x_vals):
        plt.scatter(x, y_vals[i], s=30, label=f"{x} M$_\odot$, f(M$_\odot$) = {y_vals[i]:.2e} Myr", zorder=1)
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.savefig(f'{const.FOLDER_GALAXY_TESTS}/age_distribution.pdf')  # save plot in the output folder
    plt.close()
    

def test_association_placement(read_data_from_file=True, half_edge = 40, readfile_effective_area=False):
    """ Function for testing the placement of the associations from the emissivity. Drawn using MC simulation

    Returns:
        Plot of the association placement    
    """
    logging.info("Testing montecarlo simulation of association placement")
    NUM_ASC = 15000 # number of associations to draw from the density distribution
    x_grid, y_grid, z_grid, uniform_spiral_arm_density, emissivity = gdd.generate_coords_densities(read_data_from_file=read_data_from_file, half_edge=half_edge, readfile_effective_area=readfile_effective_area) # generate the coordinates and densities for the density distribution of the Milky Way
    logging.info("Beginning to draw associations for uniform distribution")
    grid_index = gdd.monte_carlo_numpy_choise(uniform_spiral_arm_density, NUM_ASC) # draw the associations
    x = x_grid[grid_index] # get the x-values for the drawn associations
    y = y_grid[grid_index] # get the y-values for the drawn associations
    logging.info("Associations drawn. Beginning to plot the figure")
    plot_drawn_associations(x, y, NUM_ASC, 'test_association_placement_uniform.pdf') # plot the drawn associations
    logging.info("Done saving the figure. Now beginning to draw associations for the emissivity model")
    grid_index = gdd.monte_carlo_numpy_choise(emissivity, NUM_ASC) # draw the associations
    x = x_grid[grid_index]
    y = y_grid[grid_index]
    logging.info("Associations drawn. Beginning to plot the figure")
    plot_drawn_associations(x, y, NUM_ASC, 'test_association_placement_emissivity.pdf')
    logging.info("Done testing MC simulation of association placement")


def plot_drawn_associations(x_data, y_data, NUM_ASC, filename_output):
    """ Function to plot the drawn associations from the density distribution of the Milky Way
    
    Args:
        x_data (np.array): x-values of the drawn associations
        y_data (np.array): y-values of the drawn associations
        NUM_ASC (int): Number of associations drawn
        filename_output (str): Name of the file to save the plot
    Returns:
        Saves a plot in the output folder
    """
    plt.plot(x_data, y_data, 'o', color='black', markersize=0.5, markeredgewidth=0.0)
    plt.plot(0, 0, 'o', color='blue', markersize=4, label='Galactic Centre')
    plt.plot(0, const.r_s, 'o', color='red', markersize=2, label='Sun')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('$x$ (kpc)')
    plt.ylabel('$x$ (kpc)')
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    #plt.suptitle('Associations drawn from the NII density distribution of the Milky Way')
    #plt.title(f'Made with {NUM_ASC} associations')
    plt.legend()
    plt.savefig(f'{const.FOLDER_GALAXY_TESTS}/{filename_output}')  # save plot in the output folder
    plt.close()


@ut.timing_decorator
def test_plot_density_distribution(plane=1000, read_data_from_file=True, plot_lines_of_sight=False):
    """ Function to test the density distribution of the Milky Way. Plots both the unweighted, analytical density distribution and the weighted, modelled emissivity from which the associations are drawn.
    
    Args:
        
    Returns:
        Saves two plots in the output folder
    """
    logging.info("Testing the modelled density distribution of the Milky Way")
    x_grid, y_grid, z_grid, uniform_spiral_arm_density, emissivity = gdd.generate_coords_densities(plane=plane, read_data_from_file=read_data_from_file)
    logging.info("Plotting the uniform spiral arm density distribution")
    # Plot the uniform spiral arm density distribution:
    plot_density_distribution_with_imshow(x_grid, y_grid, uniform_spiral_arm_density, 'uniform', num_bins=plane, plot_lines_of_sight=plot_lines_of_sight)
    logging.info("Saved density map. Now plotting the emissivity model")
    plot_density_distribution_with_imshow(x_grid, y_grid, emissivity, 'emissivity', num_bins=plane, plot_lines_of_sight=plot_lines_of_sight)
    logging.info("Done plotting the density distributions of the Milky Way")
    return


@ut.timing_decorator
def plot_density_distribution_with_imshow(x_grid, y_grid, density_distribution, filename_output, num_bins, plot_lines_of_sight=False):
    """ Function to plot the density distribution of the Milky Way
    
    Args:
        x_grid (1D np.array): x-values of the grid
        y_grid (1D np.array): y-values of the grid
        density_distribution (1D np.array): The density distribution
        filename_output (str): Name of the model (uniform or emissivity)
        num_bins (int): Number of bins for the heatmap

    Returns:
        Saves a plot in the output folder
    """
    plt.figure(figsize=(10, 8))
    if plot_lines_of_sight:
        # Lines of sight to add to the plot
        los_long = np.array([35, 40, 45, 50, 55, 60])
        rads = np.linspace(0, 9, 300)
        for long in los_long:
            theta_los = util.theta(rads, np.radians(long), 0)
            rho_los = util.rho(rads, np.radians(long), 0)
            x_los = rho_los*np.cos(theta_los)
            y_los = rho_los*np.sin(theta_los)
            plt.scatter(x_los, y_los, label=f'Line of sight for {long}°', s=1)
    # Aggregate data into a 2D histogram for the heatmap
    heatmap, xedges, yedges = np.histogram2d(x_grid, y_grid, bins=num_bins, weights=density_distribution)
    # Normalize the heatmap
    heatmap /= np.sum(heatmap)
    # Plot the heatmap
    # Note: `extent` is used to scale the axes according to the edges of the bins
    plt.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', cmap='viridis', aspect='auto')
    # Adding the Galactic center and Sun markers
    plt.scatter(0, 0, c='magenta', s=60, label='Galactic centre')  
    plt.scatter(0, const.r_s, c='gold', s=30, label='Sun')  
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('$y$ (kpc)', fontsize=14)
    plt.ylabel('$x$ (kpc)', fontsize=14)
    #plt.title(f'Heatmap of Galactic densities for the {filename_output} model\nNormalized to unity', pad=20)
    plt.legend(loc='upper right', fontsize=12)
    # Adding a colorbar to represent the density scale
    cbar = plt.colorbar()
    cbar.set_label('Density', fontsize=14)
    # Save the plot to a PDF file
    filepath = f'{const.FOLDER_GALAXY_TESTS}/test_plot_{filename_output}.pdf'
    logging.info("Beginning to save the figure")
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.savefig(filepath)  # save plot in the output folder
    plt.close()
    

def plot_associations_from_galaxy(galaxy): 
    """ Function to plot the associations from the galaxy
    
    Args:
        galaxy: The galaxy to plot the associations from
        
    Returns:
        None. Saves a plot in the output folder
    """
    xs = []
    ys = []
    modelled_associations = galaxy.associations
    asc_modelled_masses = np.array([np.sum(asc.star_masses) for asc in modelled_associations])
    modelled_associations = modelled_associations[asc_modelled_masses > 7] # remove associations with no stars
    for asc in modelled_associations:
        xs.append(asc.x)
        ys.append(asc.y)
    plot_drawn_associations(xs, ys, galaxy.num_asc, 'associations_from_galaxy.pdf')


def main():
    test_plot_density_distribution(read_data_from_file=False, plot_lines_of_sight=False) # test the density distribution of the Milky Way. Plots both the unweighted, analytical density distribution and the weighted, modelled emissivity from which the associations are drawn 
    plot_diffusion_of_sns_3d() # plot the diffusion of SNP's in 3D to see how they diffuse away from the association centre with time
    test_association_placement() # test the placement of the associations from the emissivity. Drawn using MC simulation
    plot_age_mass_distribution() # highlight the relation between mass and expected lifetime of stars
    length_sim_myr = 100
    galaxy_1 = galaxy.Galaxy(length_sim_myr, star_formation_episodes=5, read_data_from_file=True) # an array with n associations
    plot_mass_distr(galaxy_1) # plot the probability distribution for the mass of SNP's. SNP's from a generated galaxy
    plot_sn_as_func_of_long(galaxy_1, length_sim_myr, num_star_forming_episodes=5) # plot the probability density function of SNPs as function of longitude. SNP's from a generated galaxy
    plot_associations_from_galaxy(galaxy_1) # plot the associations from the galaxy
    galaxy_2 = galaxy.Galaxy(length_sim_myr, star_formation_episodes=3, read_data_from_file=True) # an array with n associations
    galaxy_3 = galaxy.Galaxy(length_sim_myr, star_formation_episodes=5, read_data_from_file=True) # an array with n associations
    galaxies = np.array([galaxy_1, galaxy_2, galaxy_3])
    plot_cum_snp_cluster_distr(galaxies) # plot the cumulative distribution of steller clusters as function of number of snp's. SNP's from a generated galaxies with different values of C


if __name__ == "__main__":
    main()
