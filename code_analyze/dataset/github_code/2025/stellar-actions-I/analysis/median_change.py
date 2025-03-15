import numpy as np
import h5py
import sys
from tqdm import tqdm
from multiprocessing import Pool
from scipy.signal import butter, filtfilt
from utils import filter_time_series
from utils import save_all_chunks

# Butterworth filter setup
fs = 1.0  # Sampling frequency (1/snapshot interval)
cutoff = 1 / 30  # Desired cutoff frequency (inverse of time period in snapshots)
order = 10  # Filter order

# Design the filter
b, a = butter(order, cutoff / (0.5 * fs), btype='low', analog=False)

class Counter:
    def __init__(self):
        self.value = 0  # Initialize the counter

    def increment(self):
        self.value += 1  # Increment the counter

# Function to compute median change

def process_chunk_filter(chunk_id, A_chunk, J_chunk, age_bins, delta_t_bins,which_J,relative=True):
    """
    Process a single chunk of A_temp and J_temp to compute heatmap data using median
    
    Parameters:
    - chunk_id: int, ID of the chunk (for logging).
    - A_chunk: ndarray, shape (i, N_chunk)
    - J_chunk: ndarray, shape (i, N_chunk, 3)
    - age_bins: ndarray, bins for age.
    - delta_t_bins: ndarray, bins for delta_t.
    -which_J: J component to be processed. 0=R, 1=Phi, 2=z
    -relative: True (default)- calculates relative change in action, otherwise absolute change in action
    Returns:
    - histograms, shape- number of age bins, number of delta_t bins, number of delta_J bin centres
    - bin centres- just for confirmation
    """
    print(f"Processing chunk {chunk_id}...")
    # step 0: filtering the Js
     # Instantiate a counter object
    counter = Counter()
    
        # Apply the filter to each star's time series
    filtered_data = np.apply_along_axis(filter_time_series, axis=0, arr=J_chunk[:,:,which_J], b=b, a=a, counter=counter)
    
        # Retrieve the count of skipped time series
    print(f"Number of time series that were filtered: {counter.value}")
    # Step 1: Compute delta_J
    if relative:
        delta_J = abs((filtered_data[:, np.newaxis, :] - filtered_data[np.newaxis, :, :])/filtered_data[np.newaxis, :, :])
    elif relative==False:
        delta_J = abs(filtered_data[:, np.newaxis, :] - filtered_data[np.newaxis, :, :])       
    for i in np.arange(delta_J.shape[0]):
        delta_J[i,:i,:]=0 
    
    print(f'delta_J calculated for chunk {chunk_id}')
    # Step 2: Bin the ages into indices
    age_bin_indices = np.digitize(A_chunk, bins=age_bins) - 1
    age_bin_indices[np.where(np.isnan(A_chunk))]=len(age_bins)-1
    #defining different histogram bins for different components:
    if which_J==0: #R
        if relative:
            bins = np.logspace(np.log10(1e-8),np.log10(1),6000)
        elif relative==False:
            bins = np.logspace(np.log10(1e-4),np.log10(2000),6000)
    elif which_J==1:  #phi 
        if relative:
            bins = np.logspace(np.log10(1e-8),np.log10(1),6000)
        elif relative==False:
            bins = np.logspace(np.log10(1e-4),np.log10(80000),6000)
    elif which_J==2: #z
        if relative:
            bins = np.logspace(np.log10(1e-8),np.log10(1),6000)
        elif relative==False:
            bins = np.logspace(np.log10(1e-4),np.log10(20),6000)
    bin_centers = (bins[:-1]+bins[1:])/2
    histograms = np.zeros((len(age_bins),len(delta_t_bins),len(bins)-1))
    
    
    for i in tqdm(np.arange(delta_J.shape[0]),desc="processing snapshot:", position=0, leave=False):
        for i_prime in np.arange(i+1,delta_J.shape[1]): #forward time logic
            delta_t = i_prime-i
            #assigning delta_t_bin
            dt_bin = np.digitize(delta_t,delta_t_bins)-1
            #getting age bin mask
            valid_stars = np.where(age_bin_indices[i,:]!=564)
            if len(valid_stars[0]) == 0:
                continue
            #getting age bin values
            star_age_bins = age_bin_indices[i,valid_stars]
            #accessing values from delta_J_squared_sum
            delta_J_values = delta_J[i,i_prime,valid_stars]   #shape number of valid stars
    
            # Mask out NaN \(\Delta J\) values
            valid_delta_J = ~np.isnan(delta_J_values)  # Mask for valid delta_j
            star_age_bins = star_age_bins[valid_delta_J]  # Filter age bins further
            delta_J_values = delta_J_values[valid_delta_J]  # Filter \(\Delta J\) values
    
            delta_J_indices = np.digitize(delta_J_values,bins[:-1])-1
    
            np.add.at(histograms,(star_age_bins,dt_bin,delta_J_indices),1)
    print(f"Chunk {chunk_id} completed.")
    return histograms, bin_centers


# Main function to parallelize the workflow
def parallel_compute_filtered(A_temp, J_temp, age_bins, delta_t_bins, which_J,relative=True,num_chunks=4):
    """
    Parallel computation of heatmap data by splitting A_temp and J_temp into chunks.
    
    Parameters:
    - A_temp: ndarray, shape (i, N)
    - J_temp: ndarray, shape (i, N, 3)
    - age_bins: ndarray, bins for age.
    - delta_t_bins: ndarray, bins for delta_t.
    - k: int, time step interval.
    - num_chunks: int, number of parallel chunks.
    -relative: True (default)- calculates relative change in action, otherwise absolute change in action    
    Returns:
    - histograms, bin_centers
    """
    # Split data into chunks
    N = A_temp.shape[1]
    chunk_size = N // num_chunks
    chunks = [(i, A_temp[:, i * chunk_size:(i + 1) * chunk_size], 
                  J_temp[:, i * chunk_size:(i + 1) * chunk_size, :], 
                  age_bins, delta_t_bins, which_J,relative)
              for i in range(num_chunks)]
    # Create a multiprocessing Pool
    with Pool(processes=num_chunks) as pool:
        results = pool.starmap(process_chunk_filter, chunks)
        print('got results')
    return results

def main():
    which_J = int(sys.argv[1])    #0-R,1-Phi,2-z
    l = int(sys.argv[2])       #chunk processing 
    relative = bool(int(sys.argv[3]))  # Pass 1 for relative, 0 for absolute
  
    print(f'doing {which_J} J rel({relative}) and till {l}*80k stars')
    J = np.load('path/to/J.npy')          ###change path here
    A = np.load('path/to/A.npy')          ###change path here
    A = np.vstack((np.zeros((3, A.shape[1])), A))
    A[:3, :] = np.nan
    J = np.vstack((np.zeros((3, J.shape[1], 3)), J))
    J[:3, :, :] = np.nan
    
    A_check = A[:, np.where(A[-1, :] < A.shape[0])[1]]
    J_check = J[:, np.where(A[-1, :] < A.shape[0])[1], :]
    
    print(f'Total stars in millions = {A_check.shape[1] / 1e6}')
    k=1
    age_bins = np.arange(0, 564, k) 
    #adding extra bin for np.nan values:
    age_bins = np.append(age_bins, np.inf)
    delta_t_bins = np.arange(0, 565, k)  # Snapshot differences

    N=int(8e4)

    #change path here
    file_path = f"output_path/J_component_{which_J}_{l}_{'relative' if relative else 'absolute'}.h5"
    J_temp = J_check[:, (l - 1) * N:(l * N), :]
    A_temp = A_check[:, (l - 1) * N:(l * N)]
    
    print(A_temp.shape)
    heatmap = parallel_compute_filtered(A_temp, J_temp, age_bins, delta_t_bins, which_J, relative,num_chunks=8)
    adding_trial = np.zeros((len(age_bins),len(delta_t_bins),6000-1))
    ello=1
    for (histograms,bin_centres) in heatmap:
        if ello==1:
            ello_ello = bin_centres
        ello+=1
        if (bin_centres==ello_ello).all():
            print('bin centres match and hence, adding histograms')
            adding_trial+=histograms
    save_all_chunks(file_path, [[adding_trial,bin_centres]], age_bins, delta_t_bins)
    print("Analysis complete and saved.")

if __name__ == "__main__":
    main()


    
  
