import pandas as pd
import numpy as np
import src.utilities.utilities as ut
import src.utilities.constants as const
import logging
import numpy as np
logging.basicConfig(level=logging.INFO)
rng = np.random.default_rng()
from src.galaxy_model.combined_obas import calc_snps_known_association
from pathlib import Path

def my_data_for_stat():
    """ Get the data on the know associations for the simulation
    
    Returns:
        association_name: array. Name of the association
        n: array. Number of stars in the association
        min_mass: array. Minimum mass for the mass range
        max_mass: array. Maximum mass for the mass range
        age: array. Age of the association in Myr
    """
    file_path = f'{const.FOLDER_OBSERVATIONAL_DATA}/Overview of know OB associations.xlsx' 
    data = pd.read_excel(file_path)
    association_name = data['Name']
    n = data['Number of stars']
    min_mass = data['Min mass']
    max_mass = data['Max mass']
    age = data['Age(Myr)']
    return association_name, n, min_mass, max_mass, age


@ut.timing_decorator
def stat_one_known_asc(n, min_mass, max_mass, association_age, num_iterations=10000): 
    """ Calculate the statistics for one association
    
    Args:
        n: int. Number of stars in the association in the mass range
        min_mass: float. Minimum mass for the mass range
        max_mass: float. Maximum mass for the mass range
        association_age: float. Age of the association in Myr
        num_iterations: int. Number of iterations for the simulation
    
    Returns:
        n_drawn_mean: float. Mean number of drawn stars
        n_drawn_std: float. Standard deviation of the number of drawn stars
        exploded_sn_mean: float. Mean number of exploded supernovae
        exploded_sn_std: float. Standard deviation of the number of exploded supernovae
        exploded_sn_1_myr_mean: float. Mean number of exploded supernovae within 1 Myr
        exploded_sn_1_myr_std: float. Standard deviation of the number of exploded supernovae within 1 Myr
        stars_still_existing_mean: float. Mean number of stars still existing
        stars_still_existing_std: float. Standard deviation of the number of stars still existing
    """
    array_n_drawn = []
    array_exploded_sn = []
    array_exploded_sn_1_myr = []
    array_stars_still_existing = []
    for i in range(num_iterations):
        n_drawn, drawn_masses = calc_snps_known_association(n, min_mass, max_mass, association_age)
        drawn_ages = ut.lifetime_as_func_of_initial_mass(drawn_masses)
        array_n_drawn.append(n_drawn)
        mask_exploded = 0 <= association_age - drawn_ages # mask for the drawn stars which have exploded, i.e. the drawn stars which have a lifetime less than the age of the association
        mask_exploded_1_myr = association_age - drawn_ages <= 1  # mask for the drawn stars which have exploded within 1 Myr
        mask_exploded_1_myr = mask_exploded[mask_exploded_1_myr]
        mask_still_existing = association_age - drawn_ages < 0 # mask for the drawn stars which are still existing (lifetime of the star is greater than the age of the association)
        array_exploded_sn.append(np.sum(mask_exploded))
        array_exploded_sn_1_myr.append(np.sum(mask_exploded_1_myr))
        array_stars_still_existing.append(np.sum(mask_still_existing))
    n_drawn_mean = np.round(np.mean(array_n_drawn))
    n_drawn_std = np.round(np.std(array_n_drawn))
    exploded_sn_mean = np.round(np.mean(array_exploded_sn))
    exploded_sn_std = np.round(np.std(array_exploded_sn))
    exploded_sn_1_myr_mean = np.round(np.mean(array_exploded_sn_1_myr))
    exploded_sn_1_myr_std = np.round(np.std(array_exploded_sn_1_myr))
    stars_still_existing_mean = n_drawn_mean - exploded_sn_mean
    stars_still_existing_covariance = np.cov(array_n_drawn, array_exploded_sn)[0,1] # calculate the sample covariance. [0,1] is the covariance between the two arrays (np.cov returns a covariance matrix)
    # apply Bessel's correction for the variances
    combined_variance = np.var(array_n_drawn, ddof=1) + np.var(array_exploded_sn, ddof=1) - 2 * stars_still_existing_covariance
    combined_variance = np.max((0, combined_variance)) # if the combined variance is negative, set it to 0
    stars_still_existing_std = np.round(np.sqrt(combined_variance)) 
    return n_drawn_mean, n_drawn_std, exploded_sn_mean, exploded_sn_std, exploded_sn_1_myr_mean, exploded_sn_1_myr_std, stars_still_existing_mean, stars_still_existing_std


@ut.timing_decorator
def stat_known_associations(num_iterations = 10):
    """ Calculate the statistics for the known associations and save the results to a CSV file.
    Calculates the mean and standard deviation of the number of drawn stars, the number of exploded supernovae, the number of exploded supernovae within 1 Myr and the number of stars still existing for each association."""
    association_name, n, min_mass, max_mass, age = my_data_for_stat()
    # Prepare lists to store the statistics
    mean_snp_per_association = []
    std_snp_per_association = []
    mean_exploded_sn = []
    std_exploded_sn = []
    mean_exploded_sn_1_myr = []
    std_exploded_sn_1_myr = []
    mean_stars_still_existing = []
    std_stars_still_existing = []
    # Run the simulation and gather statistics for each association
    for i in range(len(association_name)):
        (n_drawn_mean, n_drawn_std, exploded_sn_mean, exploded_sn_std,
         exploded_sn_1_myr_mean, exploded_sn_1_myr_std,
         stars_still_existing_mean, stars_still_existing_std) = stat_one_known_asc(n[i], min_mass[i], max_mass[i], age[i], num_iterations)
        # Append the results to their respective lists
        mean_snp_per_association.append(n_drawn_mean)
        std_snp_per_association.append(n_drawn_std)
        mean_exploded_sn.append(exploded_sn_mean)
        std_exploded_sn.append(exploded_sn_std)
        mean_exploded_sn_1_myr.append(exploded_sn_1_myr_mean)
        std_exploded_sn_1_myr.append(exploded_sn_1_myr_std)
        mean_stars_still_existing.append(stars_still_existing_mean)
        std_stars_still_existing.append(stars_still_existing_std)
    # Create a DataFrame with the collected statistics
    df = pd.DataFrame({
        'Mean SNP born': mean_snp_per_association,
        'Std SNP born': std_snp_per_association,
        'Mean Exploded SN': mean_exploded_sn,
        'Std Exploded SN': std_exploded_sn,
        'Mean Exploded SN 1 Myr': mean_exploded_sn_1_myr,
        'Std Exploded SN 1 Myr': std_exploded_sn_1_myr,
        'Mean Stars Still Existing': mean_stars_still_existing,
        'Std Stars Still Existing': std_stars_still_existing
    }, index=association_name)
    # Save the DataFrame to a CSV file
    # check if the output folder exists, if not create it
    Path(const.FOLDER_OBSERVATIONAL_PLOTS).mkdir(parents=True, exist_ok=True)
    df.to_csv(f'{const.FOLDER_OBSERVATIONAL_DATA}/statistics_known_associations.csv')
    return

 
def main():
    stat_known_associations(num_iterations=10000)
   

if __name__ == '__main__':
    main()
