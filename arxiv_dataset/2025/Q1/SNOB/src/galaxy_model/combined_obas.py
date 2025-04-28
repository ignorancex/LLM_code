import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import src.utilities.utilities as ut
import src.nii_intensities.spiral_arm_model as sam
import src.utilities.constants as const
import src.galaxy_model.galaxy_class as gal
import src.galaxy_model.association_class as asc
from matplotlib.ticker import AutoMinorLocator
import logging
import numpy as np
logging.basicConfig(level=logging.INFO)
rng = np.random.default_rng()
SLOPE = 1_8 # slope for the power law distribution of the number of stars in the associations. Corresponds to alpha = 0.8


def add_heliocentric_circles_to_ax(ax, step=0.5, linewidth=1):
    """ Add heliocentric circles to the plot
    
    Args:
        ax: axis. The axis to add the circles to
        step: float. Step size for the radial binning of associations in kpc
    
    Returns:
        None
    """
    thetas_heliocentric_circles = np.linspace(0, 2 * np.pi, 100)
    for i in range(1, 7):
        x_heliocentric_circles = i * step * np.cos(thetas_heliocentric_circles)
        y_heliocentric_circles = i * step * np.sin(thetas_heliocentric_circles) + const.r_s
        ax.plot(x_heliocentric_circles, y_heliocentric_circles, color='black', linestyle='--', linewidth=linewidth, zorder=5) # plot the heliocentric circles
    return


def add_spiral_arms_to_ax(ax, linewidth=3):
    """ Add the spiral arms to the plot
    
    Args:
        ax: axis. The axis to add the spiral arms to
    
    Returns:
        None
    """
    colors = sns.color_palette('bright', 7)
    rho_min_array = const.rho_min_spiral_arm
    rho_max_array = const.rho_max_spiral_arm
    for i in range(len(const.arm_angles)):
        # generate the spiral arm medians
        theta, rho = sam.spiral_arm_medians(const.arm_angles[i], const.pitch_angles[i], rho_min=rho_min_array[i], rho_max=rho_max_array[i])
        x = rho*np.cos(theta)
        y = rho*np.sin(theta)
        ax.plot(x, y, linewidth = linewidth, zorder=6, markeredgewidth=1, markersize=1, color=colors[i]) # plot the spiral arm medians
    return


def add_associations_to_ax(ax, x, y, label, color, s=15):
    """ Add the associations to the plot
    
    Args:
        ax: axis. The axis to add the associations to
        x: array. x-coordinates of the associations. Units of kpc
        y: array. y-coordinates of the associations. Units of kpc
        label: str. Label name for the plotted associations
        color: str. Colour of the plotted associations

    Returns:
        None
    """
    ax.scatter(x, y, color=color, alpha=0.5, s=s, label=label, zorder=10)
    return


def add_spiral_arm_names_to_ax(ax, fontsize=20):
    """ Add the names of the spiral arms to the plot
    
    Args:
        ax: axis. The axis to add the spiral arm names to
    
    Returns:
        None
    """
    text_x_pos = [-3.5, -5, -6.2, -6.8]
    text_y_pos = [2.8, 4.9, 6.7, 10.1]
    text_rotation = [24, 23, 20, 16]
    text_arm_names = ['Norma-Cygnus', 'Scutum-Crux', 'Sagittarius-Carina', 'Perseus']
    
    for i in range(len(const.arm_angles[:-1])): # skip local arm
        ax.text(text_x_pos[i], text_y_pos[i], text_arm_names[i], fontsize=fontsize, zorder=20, rotation=text_rotation[i],
                weight='bold', bbox=dict(facecolor='white', alpha=0.2, edgecolor='none'))
    return


def calc_snps_known_association(n, min_mass, max_mass, association_age):
    """ Calculate the number of drawn stars and their masses for a known association. Takes into account the mass range of the observed stars today and the age of the association.
    The returned number of stars is an estimate on how many stars had to form in the association an 'association_age' years ago to have 'n' stars today.
    
    Args:
        n: int. Number of stars in the association in the given mass range
        min_mass: float. Minimum mass for the mass range
        max_mass: float. Maximum mass for the mass range
        association_age: float. Age of the association in Myr
    
    Returns:
        n_drawn: int. Number of drawn stars
        drawn_masses: array. Masses of the drawn stars
    """
    m3 = np.arange(1.0, 120, 0.01) # mass in solar masses. Used to draw random masses for the SNPs in the association
    m3 = np.concatenate((m3, [120])) # add 120 solar masses to the array
    imf3 = ut.imf_3(m3)
    imf3 = imf3 / np.sum(imf3) # normalize
    n_drawn = 0
    n_matched = 0
    drawn_masses = []
    while n_matched < n:
        drawn_mass = rng.choice(m3, size=1, p=imf3)
        drawn_mass_age = ut.lifetime_as_func_of_initial_mass(drawn_mass)
        if drawn_mass >= 8: # if drawn mass is greater than or equal to 8 solar masses, keep it
            n_drawn += 1
            drawn_masses.append(drawn_mass)
        if drawn_mass >= min_mass and drawn_mass <= max_mass and drawn_mass_age >= association_age:
            # if drawn mass is within the given mass range and the age of the drawn mass is greater than or equal to the age of the association, keep it
            # this essentially means that if the drawn star could have survived up until today and match the mass criteria, increase the counter
            n_matched += 1
    return n_drawn, np.array(drawn_masses)


def calc_num_snps_known_associations_batch():
    """ Calculate the number of drawn stars for the known associations. Uses calc_snps_known_association() to calculate the number of drawn stars for each association, but the masses are discarded.
    
    Returns:
        n_drawn_list: array. Number of drawn stars for each association
    """
    file_path = f'{const.FOLDER_OBSERVATIONAL_DATA}/Overview of know OB associations.xlsx' 
    data = pd.read_excel(file_path)
    n = data['Number of stars']
    min_mass = data['Min mass']
    max_mass = data['Max mass']
    age = data['Age(Myr)']
    n_drawn_list = []
    for i in range(len(n)):
        n_drawn, _ = calc_snps_known_association(n[i], min_mass[i], max_mass[i], age[i])
        n_drawn_list.append(n_drawn)
    return np.array(n_drawn_list)


def known_associations_to_association_class():
    """ Convert the known associations to the Association class
    
    Returns:
        associations: list. List of Association objects
    """
    file_path = f'{const.FOLDER_OBSERVATIONAL_DATA}/Overview of know OB associations.xlsx' 
    data = pd.read_excel(file_path)
    age = data['Age(Myr)']
    distance = data['Distance (pc)'] / 1000 # convert to kpc
    glon = np.radians(data['l (deg)']) # convert to radians
    glat = np.radians(data['b (deg)']) # convert to radians
    rho = ut.rho(distance, glon, glat)
    theta = ut.theta(distance, glon, glat)
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    z = ut.z(distance, glat)

    num_snp = calc_num_snps_known_associations_batch()
    associations = []
    for i in range(len(x)):
        # the association is created age[i] myrs ago with nump_snp[i] snps, which are found to be the number needed to explain the observed number of stars in the association. 
        association = asc.Association(x[i], y[i], z[i], age[i], c=1, n=[num_snp[i]])
        # Next: update the snps to the present time
        association.update_sn(0)
        associations.append(association) # append association to list
    return associations


def combine_modelled_and_known_associations(modelled_associations, step=0.5, endpoint=25):
    """ Combine the modelled and known associations
    
    Args:
        modelled_galaxy: Galaxy. The modelled galaxy
        step: float. Step size for the radial binning of associations in kpc
        endpoint: float. Max radial distance in kpc
    
    Returns:
        known_associations: array. Known associations
        associations_added: array. Modelled associations added to the known associations
    """
    bins = np.arange(0, endpoint + step, step)
    known_associations = known_associations_to_association_class()
    distance_obs = np.array([asc.r for asc in known_associations])
    r_modelled = np.array([np.sqrt(asc.x**2 + (asc.y - const.r_s)**2 + asc.z**2) for asc in modelled_associations])
    hist_modelled, _ = np.histogram(r_modelled, bins=bins)
    hist_obs, _ = np.histogram(distance_obs, bins=bins)
    associations_added = np.array([])
    for i in range(len(bins[1:])):
        diff = hist_modelled[i] - hist_obs[i] # difference between the number of modelled and observed associations in the bin
        mask_modelled = (r_modelled >= bins[i]) & (r_modelled < bins[i + 1]) # pick out the modelled associations which are in the bin
        if diff == hist_modelled[i]: # there are no observed associations in the bin
            associations_added = np.concatenate((associations_added, modelled_associations[mask_modelled])) # add all modelled associations in the bin
        elif diff > 0: # if there are more modelled associations in the bin than observed
            associations_added = np.concatenate((associations_added, rng.choice(modelled_associations[mask_modelled], size=diff))) # add diff associations randomly from the modelled associations in the bin
        elif diff < 0: # if there are more observed associations in the bin than modelled
            pass # do nothing
    return known_associations, associations_added


def plot_added_and_known_associations(modelled_galaxy, step=0.5, endpoint=25):
    """ Plot the modelled and known associations together
    
    Args:
        modelled_galaxy: Galaxy. The modelled galaxy
        step: float. Step size for the radial binning of associations in kpc
        endpoint: float. Max radial distance in kpc
    
    Returns:
        None. Saves the plot
    """
    modelled_associations = modelled_galaxy.associations
    print(f'Number of associations added: {len(modelled_associations)}')
    asc_modelled_masses = np.array([np.sum(asc.star_masses) for asc in modelled_associations])
    modelled_associations = modelled_associations[asc_modelled_masses > 7] # remove associations with no stars
    known_associations, associations_added = combine_modelled_and_known_associations(modelled_associations, step, endpoint)
    asc_mass_added = np.array([np.sum(asc.star_masses) for asc in associations_added])
    associations_added = associations_added[asc_mass_added > 7] # remove associations with no stars 
    print(f'Number of associations added with stars: {len(associations_added)}')
    x_obs = np.array([asc.x for asc in known_associations])
    y_obs = np.array([asc.y for asc in known_associations])
    x_added = np.array([asc.x for asc in associations_added])
    y_added = np.array([asc.y for asc in associations_added])
    # Now plot the modelled and known associations together
    fig, ax = plt.subplots(figsize=(20, 18))
    add_associations_to_ax(ax, x_obs, y_obs, 'Known associations', 'blue', s=40) # want the known associations to get its own label
    add_associations_to_ax(ax, x_added, y_added, 'Modelled associations', 'darkgreen', s=40)
    add_heliocentric_circles_to_ax(ax, step=step, linewidth=1)
    add_spiral_arms_to_ax(ax, linewidth=3)
    add_spiral_arm_names_to_ax(ax, fontsize=25)
    ax.scatter(0, const.r_s, color='red', marker='o', label='Sun', s=45, zorder=11)
    ax.scatter(0, 0, color='black', marker='o', s=50, zorder=11)
    ax.text(-0.38, 0.5, 'GC', fontsize=35, zorder=7)
    #plt.title(f'Distribution of known and modelled associations in the Galactic plane. \n Galaxy generated {sim_time} Myrs ago')
    plt.xlabel('$x$ (kpc)', fontsize=35)
    plt.ylabel('$y$ (kpc)', fontsize=35)
    plt.xlim(-7.5, 7.5)
    plt.ylim(-2, 12)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tick_params(axis='both', which='major', labelsize=35)
    legend = plt.legend(framealpha=0.9, fontsize=30, loc='upper right')
    legend.set_zorder(50)
    plt.grid(True, zorder=-10)
    plt.rc('font', size=50) # increase the font size
    plt.savefig(f'{const.FOLDER_OBSERVATIONAL_PLOTS}/combined_associations_{SLOPE}.pdf')
    plt.close()


def calc_avg_asc_mass_hist(modelled_galaxy, num_iterations: int = 10, bin_max_mass: int = 3000):
    asc_mass_step = 50
    bins = np.arange(0, bin_max_mass + asc_mass_step, asc_mass_step)
    modelled_associations = modelled_galaxy.associations
    mass_asc_modelled = np.array([np.sum(asc.star_masses) for asc in modelled_associations])
    mass_asc_modelled = mass_asc_modelled[mass_asc_modelled > 7] # remove associations with mass less than 8 solar masses (these appear due to SNP masses are set to zero once they die. Set > 7 to avoid rounding errors in the mass calculation)
    hist_modelled, _ = np.histogram(mass_asc_modelled, bins=bins)
    hist_known = np.zeros((num_iterations, len(bins) - 1))
    for it in range(num_iterations):
        if it % 10 == 0:
            logging.info(f'Iteration {it}')
        known_associations = known_associations_to_association_class()
        mass_asc_known = np.array([np.sum(asc.star_masses) for asc in known_associations])
        hist_known_it, _ = np.histogram(mass_asc_known, bins=bins)
        hist_known[it] = hist_known_it
    hist_known_mean = np.mean(hist_known, axis=0)
    return bins, hist_known_mean, hist_modelled


def plot_avg_asc_mass_hist(modelled_galaxy, num_iterations: int, star_formation_episodes: int, sim_time: int):
    """ Plot the histogram of the number of stars per association for known and modelled associations
    
    Args:
        modelled_galaxy: Galaxy. The modelled galaxy
    
    Returns:
        None. Saves the plot
    """
    logging.info('Plotting average association mass histogram')
    bin_max_mass = 2000
    bins, hist_known_mean, hist_added_mean = calc_avg_asc_mass_hist(modelled_galaxy, num_iterations=num_iterations, bin_max_mass=bin_max_mass)
    hist_added_mean = hist_added_mean / np.sum(hist_added_mean) 
    hist_known_mean = hist_known_mean / np.sum(hist_known_mean)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_widths = np.diff(bins)
    plt.figure(figsize=(10, 6))
    plt.bar(bin_centers, hist_known_mean, width=bin_widths, label='Known Associations', alpha=0.7, edgecolor='black')
    plt.bar(bin_centers, hist_added_mean, width=bin_widths, label='Modelled Associations', alpha=0.7, edgecolor='black')
    plt.xlabel('Association mass (M$_\odot$)', fontsize=12)
    plt.xlim(0, bin_max_mass)
    plt.ylabel('Frequency', fontsize=12)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))    # Set the y-axis to only use integer ticks
    #plt.title('Histogram of association masses shown for modelled and known associations.')
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.yscale('log')
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(10)) 
    plt.savefig(f'{const.FOLDER_OBSERVATIONAL_PLOTS}/asc_mass_hist_{star_formation_episodes}_num_iterations_{num_iterations}_sim_time_{sim_time}_{SLOPE}.pdf')
    plt.close()


def plot_age_hist(age_data_known, age_data_modelled, filename, bin_max_age: int = 50):
    """ Plot the age vs. distance of OB associations
    
    Args:
        age_data: array. Age of the associations
    
    Returns:
        None. Shows the plot
    """
    binwidth = 1
    bin_max_age = np.max(age_data_modelled)
    bins = np.arange(0, bin_max_age + binwidth, binwidth)
    plt.figure(figsize=(10, 6))
    plt.hist(age_data_known, bins=bins, label='Known associations', alpha=0.7, zorder=5, edgecolor='black')
    plt.hist(age_data_modelled, bins=bins, label='Modelled associations', alpha=0.7, zorder=4, edgecolor='black')
    #plt.title('Histogram of ages of OB associations')
    plt.xlabel('Age (Myr)', fontsize=12)
    plt.xlim(0, bin_max_age)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(10)) 
    plt.ylabel('Counts', fontsize=12)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))    # Set the y-axis to only use integer ticks
    plt.grid(axis='y')
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.savefig(f'{const.FOLDER_OBSERVATIONAL_PLOTS}/{filename}')
    plt.close()


def plot_age_hist_known_modelled(modelled_galaxy, step=0.5, endpoint=2.5):
    """ Plot the histogram of the ages of known and modelled associations within 2.5 kpc

    Returns:
        None. Saves the plot
    """
    logging.info('Plotting age histogram of known and modelled associations')
    modelled_associations = modelled_galaxy.associations
    known_associations = known_associations_to_association_class()
    known_associations, associations_added = combine_modelled_and_known_associations(modelled_associations, step, endpoint)
    age_known = np.array([asc.age for asc in known_associations])
    age_added = np.array([asc.age for asc in associations_added])
    masses_asc_added = np.array([np.sum(asc.star_masses) for asc in associations_added])
    added_asc_exist_mask = np.array([masses > 7 for masses in masses_asc_added]) # mask for the modelled associations with mass greater than 7 solar masses. > 7 to avoid rounding errors in the mass calculation, and this mask is to remove associations for which all SNPs have exploded
    asc_added_radial_distances = np.array([np.sqrt(asc.x**2 + (asc.y - const.r_s)**2 + asc.z**2) for asc in associations_added])
    added_asc_distance_mask = asc_added_radial_distances <= 2.5 # mask for the modelled associations which are within 2.5 kpc
    added_asc_mask_combined = added_asc_exist_mask & added_asc_distance_mask
    age_added = age_added[added_asc_mask_combined] # remove modelled associations which have no stars anymore
    plot_age_hist(age_known, age_added, filename=f'histogram_age_known_modelled_asc_{SLOPE}.pdf')


def area_per_bin(bins):
    """ Calculate the area of each bin in a histogram for a circular bins
    
    Args:
        bins: array. The bins of the histogram
    
    Returns:
        area_per_circle: array. The area of each bin
    """
    area_per_circle = np.power(bins[1:], 2) * np.pi - np.power(bins[:-1], 2) * np.pi
    return area_per_circle


def plot_distance_hist(heliocentric_distance_known, heliocentric_distance_modelled, filename, step=0.5, endpoint=2.5):
    """ Plot the histogram of distances of OB associations and fit the data to a Gaussian function

    Args:
        heliocentric_distance: array. Heliocentric distances of the associations in kpc
        filename: str. Name of the file to save the plot
        step: float. Step size for the histogram in kpc
        endpoint: float. Max radial distance in kpc
    
    Returns:
        None. Saves the plot
    """
    bins = np.arange(0, endpoint + step, step)
    area_per_circle = area_per_bin(bins)
    hist_known, _ = np.histogram(heliocentric_distance_known, bins=bins)
    hist_modelled, _ = np.histogram(heliocentric_distance_modelled, bins=bins)
    hist_known = hist_known / area_per_circle # find the surface density of OB associations
    hist_modelled = hist_modelled / area_per_circle
    hist_central_x_val = bins[:-1] + step / 2 # central x values for each bin
    # Make the histogram    
    plt.figure(figsize=(10, 6))
    plt.bar(hist_central_x_val, hist_known, width=step, alpha=0.7, label='Known associations', zorder=5, edgecolor='black')
    plt.bar(hist_central_x_val, hist_modelled, width=step, alpha=0.7, label='Modelled associations', zorder=4, edgecolor='black')
    plt.bar(hist_central_x_val, hist_modelled + hist_known, width=step, alpha=0.7, label='Total', zorder=3, edgecolor='black')
    #plt.title('Radial distribution of OB association surface density')
    plt.xlabel('Heliocentric distance r (kpc)', fontsize=12)
    plt.xlim(0, endpoint)
    plt.ylabel('$\\rho(r)$ (OB associations / kpc$^{-2}$)', fontsize=12)
    plt.grid(axis='y')
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.savefig(f'{const.FOLDER_OBSERVATIONAL_PLOTS}/{filename}')
    plt.close()
    return bins, hist_known


def plot_distance_hist_known_added(modelled_galaxy, step=0.5, endpoint=2.5):
    """ Plot the histogram of the radial distances of known and modelled associations
    
    Args:
        modelled_galaxy: Galaxy. The modelled galaxy
    
    Returns:
        None. Saves the plot
    """
    logging.info('Plotting distance histogram of known and modelled associations')
    modelled_associations = modelled_galaxy.associations
    #known_associations = known_associations_to_association_class()
    known_associations, associations_added = combine_modelled_and_known_associations(modelled_associations, step, endpoint)
    distance_known = np.array([asc.r for asc in known_associations])
    distance_added = np.array([asc.r for asc in associations_added])
    masses_asc_added = np.array([np.sum(asc.star_masses) for asc in associations_added])
    added_asc_mask = np.array([masses > 7 for masses in masses_asc_added]) # mask for the modelled associations with mass greater than 7 solar masses. > 7 to avoid rounding errors in the mass calculation, and this mask is to remove associations for which all SNPs have exploded
    distance_added = distance_added[added_asc_mask] # remove modelled associations which have no stars anymore
    plot_distance_hist(heliocentric_distance_known=distance_known, heliocentric_distance_modelled=distance_added, filename=f'histogram_dist_known_modelled_asc_{SLOPE}.pdf', endpoint=endpoint)


def main():
    step = 0.5
    num_iterations = 50
    sim_time=100
    galaxy_1 = gal.Galaxy(sim_time, read_data_from_file=True, star_formation_episodes=1)
    galaxy_3 = gal.Galaxy(sim_time, read_data_from_file=True, star_formation_episodes=3)
    galaxy_5 = gal.Galaxy(sim_time, read_data_from_file=True, star_formation_episodes=5)
    plot_added_and_known_associations(galaxy_5, step=step, endpoint=25) 
    plot_avg_asc_mass_hist(galaxy_1, num_iterations=num_iterations, star_formation_episodes=1, sim_time=sim_time)
    plot_avg_asc_mass_hist(galaxy_3, num_iterations=num_iterations, star_formation_episodes=3, sim_time=sim_time)
    plot_avg_asc_mass_hist(galaxy_5, num_iterations=num_iterations, star_formation_episodes=5, sim_time=sim_time)
    plot_distance_hist_known_added(galaxy_5)
    plot_age_hist_known_modelled(galaxy_5)


if __name__ == '__main__':
    main()
