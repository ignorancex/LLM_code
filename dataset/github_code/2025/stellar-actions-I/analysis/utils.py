import numpy as np
import h5py
from tqdm import tqdm
from scipy.signal import butter, filtfilt

def convert_to_cartesian(phi_R_z):
    """
    Convert (phi, R, z) to Cartesian coordinates (x, y, z).
    
    Parameters:
    - phi_R_z: ndarray, shape (N, 3), array of (phi, R, z) values for N stars
    
    Returns:
    - cartesian_coords: ndarray, shape (N, 3), array of (x, y, z) coordinates
    """
    phi = phi_R_z[:, 0]
    R = phi_R_z[:, 1]
    z = phi_R_z[:, 2]
    
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    
    return np.column_stack((x, y, z))

def filter_time_series(ts, b, a, counter):
    """
    Filter a time series using a given filter (b, a) and count the number of times it's applied.
    
    Parameters:
    - ts: ndarray, time series data to be filtered.
    - b, a: ndarray, filter coefficients.
    - counter: object with an 'increment' method to track how many times filtering is applied.
    
    Returns:
    - filtered: ndarray, filtered time series with NaNs for invalid data.
    """
    finite_mask = ~np.isnan(ts)  # Mask for valid (non-NaN) values
    filtered = np.full_like(ts, np.nan) # Initialize output with NaNs
    if finite_mask.sum() > max(len(a), len(b)) * 3:  
        filtered[finite_mask] = filtfilt(b, a, ts[finite_mask])  # Apply filter to valid data
        counter.increment()  # Increment the counter
    return filtered


def save_all_chunks(file_path, results, age_bins, delta_t_bins):
    """
    Save all processed chunks into an HDF5 file sequentially.
    
    Parameters:
    - file_path: str, path to the output HDF5 file.
    - results: list of tuples, each containing (histograms and bin centres) for each chunk.
    - age_bins: ndarray, age bin edges.
    - delta_t_bins: ndarray, delta_t bin edges.
    """
    with h5py.File(file_path, 'w') as f:
        # Save global bins once
        f.create_dataset('age_bins', data=age_bins)
        f.create_dataset('delta_t_bins', data=delta_t_bins)
        # Save each chunk in a separate group
        for chunk_id, (histograms,bin_centres) in enumerate(results):
            group = f.create_group(f'chunk_{chunk_id}')
            group.create_dataset('histograms', data=histograms,compression='gzip')
            group.create_dataset('bin_centres', data=bin_centres,compression='gzip')
    print('data saved')

def load_star_data_hdf5(i,k,file_path_list, sampled_ids):
    """
    Loads star data for sampled IDs from the HDF5 file in time order.
    
    :param file_path: Path to the HDF5 file.
    :param sampled_ids: List or array of sampled star IDs.
    :return: Arrays for the required star data at each snapshot for the sampled stars.
    """
    # Initialize lists to store the data for each variable
    J_list, A_list, C_list,M_list= [], [],[],[]
    
    # Convert sampled_ids to a numpy array for efficient comparison
    sampled_ids = np.array(sampled_ids)
    for file_path in file_path_list:
        with h5py.File(file_path, 'r') as hdf:
            # Collect snapshot names and sort them
            snapshot_names = sorted(hdf['snapshots'].keys(), key=lambda x: int(x.split('_')[1]))
    
            # Loop through snapshots in time order
            for snapshot_name in tqdm(snapshot_names[i:][::k]):
                grp = hdf[f'snapshots/{snapshot_name}']
                
                # Get IDs of stars in the current snapshot
                snapshot_ids_in_snapshot = grp['ID'][:]
                
                # Create a mask to select data for the sampled_ids present in the snapshot
                ID_mask = np.isin(sampled_ids, snapshot_ids_in_snapshot)
    
                # Initialize arrays for the current snapshot with NaN values
                J_snapshot = np.full((len(sampled_ids), 3), np.nan)  # Shape: (number_of_sampled_stars, 3)
                C_snapshot = np.full((len(sampled_ids), 3), np.nan)  # Shape: (number_of_sampled_stars, 3)
                A_snapshot = np.full(len(sampled_ids), np.nan)       # Shape: (number_of_sampled_stars)
                M_snapshot = np.full(len(sampled_ids), np.nan)       # Shape: (number_of_sampled_stars)
                # Find the indices of the sampled IDs in the snapshot
                star_indices = np.where(np.isin(snapshot_ids_in_snapshot, sampled_ids))[0]
                if len(star_indices) > 0:
                    # Map star_indices to the position in sampled_ids
                    id_to_position = {id_: idx for idx, id_ in enumerate(sampled_ids)}
                    mapped_indices = [id_to_position[snapshot_ids_in_snapshot[idx]] for idx in star_indices]
                    
                    # Assign data to the correct positions
                    J_snapshot[mapped_indices] = grp['actions'][:].T[star_indices]
                    C_snapshot[mapped_indices] = grp['coordinates'][:][star_indices]
                    A_snapshot[mapped_indices] = grp['age'][:][star_indices]
                    M_snapshot[mapped_indices] = grp['mass'][:][star_indices]

                # Append the current snapshot data to the lists
                J_list.append(J_snapshot)
                C_list.append(C_snapshot)
                A_list.append(A_snapshot)
                M_list.append(M_snapshot)
               
                # print(f"Loaded {snapshot_name}", end="\r")

    # Convert the lists to numpy arrays for efficient processing
    J_list = np.array(J_list)  # Shape: (number_of_snapshots, number_of_sampled_stars, 3)
    A_list = np.array(A_list)  # Shape: (number_of_snapshots, number_of_sampled_stars)
    C_list = np.array(C_list)
    M_list =np.array(M_list)
    
    return J_list,A_list,C_list,M_list,np.sort(sampled_ids)

def adding_histograms_from_chunks(hdf5_file_list, bin_size=5999):
    """
    Adds histograms from multiple HDF5 files and calculates median using the histogram.
    
    Parameters:
    - hdf5_file_list: list of str, list of HDF5 file paths to process.
    - bin_size: int, size of the histogram bins (default 3999).
    
    Returns:
    - aggregated_heatmaps: ndarray, aggregated histograms from all chunks.
    - age_bins: ndarray, bin edges for age.
    - delta_t_bins: ndarray, bin edges for delta_t.
    - bin_centres: ndarray, bin centres of the histograms.
    """
    aggregated_heatmaps = None
    age_bins, delta_t_bins = None, None
    histogram = None
    
    for hdf5_file in hdf5_file_list:
        print(f"Processing file: {hdf5_file}")
        with h5py.File(hdf5_file, 'r') as f:
            if age_bins is None:
                age_bins = f['age_bins'][:]
                delta_t_bins = f['delta_t_bins'][:]
                histogram = np.zeros((len(age_bins), len(delta_t_bins), bin_size))
            
            for chunk_name in f.keys():
                if chunk_name.startswith('chunk_'):
                    chunk_group = f[chunk_name]
                    hists = chunk_group['histograms'][:]
                    bin_centres = chunk_group['bin_centres'][:]
                    histogram += hists
    
    return histogram, age_bins, delta_t_bins, bin_centres

def fit_line(x, y):
    """
    Fit a straight line using np.polyfit (linear regression).
    
    Parameters:
    - x: ndarray, independent variable (e.g., time or some other variable).
    - y: ndarray, dependent variable (e.g., observed values).
    
    Returns:
    - slope: float, slope of the fitted line.
    - intercept: float, intercept of the fitted line.
    - r_squared: float, R-squared value of the fit.
    - y_pred: ndarray, predicted values from the fitted line.
    """
    coefficients = np.polyfit(x, y, 1)
    slope, intercept = coefficients
    y_pred = slope * x + intercept
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum((y - y_pred) ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    return slope, intercept, r_squared, y_pred
