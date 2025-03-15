## identify radial bins and also calculates action changes for each bin
#writes out two more files: 1. number of stars in each radial bin
#2. average orbital period of each radial bin


## can get action.h5 file from the following website and change the file paths here accordingly ('stellar-actions-I/data/actions.h5' in following code)
## https://www.mso.anu.edu.au/~arunima/stellar-actions-I-data/

import numpy as np
import h5py
from tqdm import tqdm
from median_change import process_chunk_filter, parallel_compute_filtered
from utils import save_all_chunks
from utils import filter_time_series,load_star_data_hdf5

def identify_radial_bins(which_J):
    # Load the star data
    k = 1
    i = 101
    
    #change path here if required
    with h5py.File('stellar-actions-I/data/actions.h5', 'r') as f:
        J = f['actions'][:]
        C = f['coordinates'][:]
        V = f['velocities'][:]
        A = f['age'][:]
        M = f['mass'][:]
        IDs = f['ID'][:]

    #calculating approximate orbital period of the stars
    period_all = np.nanmean(2*np.pi*C[:,:,1]**2/(J[:,:,1]*1000*1.023/M),axis=0)
    # Define radial bins
    r_bins = np.arange(0, 19000, 1000)
    IDs_in_bins = {i: [] for i in range(len(r_bins))}  # Dictionary to store IDs by bin

    for t in tqdm(np.arange(A.shape[0])):
        # Identify stars born in the last 1 Myr
        valid = ~np.isnan(A[t, :])  # Valid stars
        recently_born = (A[t, valid] < 1) & (A[t, valid] > 0)

        # Get the indices of these stars
        star_indices = np.where(valid)[0][recently_born]
        R_indices = np.digitize(C[t, valid, 1][recently_born], r_bins) - 1

        # Assign star IDs to their respective bins
        for i, r_bin in enumerate(R_indices):
            if 0 <= r_bin < len(r_bins) - 1:  # Ensure index is within valid bin range
                IDs_in_bins[r_bin].append(star_indices[i])

    # Now process the stars for each radial bin
    for l_R in tqdm(np.arange(2, 18, 1), desc='Radial bin', leave=False):
        J_all = J[:, IDs_in_bins[l_R], :] / M[:, IDs_in_bins[l_R], np.newaxis]
        A_all = A[:, IDs_in_bins[l_R]]
        num_stars_total = A_all.shape[1]

        print(f'Total number of stars in bin {l_R} = {num_stars_total}')
        #calculating orbital period for this radial bin
        period = np.nanmean(period_all[IDs_in_bins[l_R]]) #Myr

        # Setup the age bins
        k = 1 
        age_bins = np.arange(0, 564, k) 
        age_bins = np.append(age_bins, np.inf)  # Extra bin for np.nan values
        delta_t_bins = np.arange(0, 565, k)

        # Process in chunks of 1e5 stars
        chunk_size = int(1e5)
        adding_trial = None  # To store accumulated results
        bin_centres_reference = None  # To ensure bin consistency

        for start in range(0, num_stars_total, chunk_size):
            end = min(start + chunk_size, num_stars_total)

            J_set = J_all[:, start:end, :]
            A_set = A_all[:, start:end]

            num_stars = A_set.shape[1]
            print(f'Processing chunk {start}-{end} ({num_stars} stars)')

            # Calculate heatmap using parallel computation from median_change.py
            heatmap = parallel_compute_filtered(A_set, J_set, age_bins, delta_t_bins, which_J, relative=True, num_chunks=5)

            # Sum histograms across all chunks while ensuring bin consistency
            for histograms, bin_centres in heatmap:
                if adding_trial is None:
                    adding_trial = np.zeros_like(histograms)  # Initialize sum array
                    bin_centres_reference = bin_centres
                if (bin_centres == bin_centres_reference).all():
                    adding_trial += histograms
                else:
                    raise ValueError("Bin centres mismatch between chunks!")

        # Save results 
        #change path here
        file_path = f'path_to_results_J{which_J}_{l_R}kpcbin.h5'
        save_all_chunks(file_path, [[adding_trial, bin_centres_reference]], age_bins, delta_t_bins)

        # Save number of stars
        #change path here
        with open("path_to_radial_bin_numbers.txt", "a") as file:
            file.write(str(num_stars_total) + "\n")

        #save radial period of stars
        #change path here
        with open("path_to_radial_period.txt", "a") as file:
            file.write(str(period) + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 identify_radial_bins.py <which_J>")
        sys.exit(1)
  
      # Get which_J from command-line argument (0=R, 1=Phi, 2=Z)
    try:
        which_J = int(sys.argv[1])
        if which_J not in [0, 1, 2]:
            raise ValueError("Invalid value for which_J. Must be 0, 1, or 2.")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
  
    identify_radial_bins(which_J)
