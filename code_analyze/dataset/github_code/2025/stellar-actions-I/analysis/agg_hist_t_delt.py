import numpy as np
import h5py
import glob
import sys
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
from utils import adding_histograms_from_chunks

def main():
    if len(sys.argv) != 3:
        print("Usage: python agg_hist_t_delt.py <glob_word> <bin_size>")
        sys.exit(1)
    
    glob_word = sys.argv[1]
    bin_size = int(sys.argv[2])
    
    hdf5_file_list = glob.glob(f'path/{glob_word}.h5')      #change path here
    histogram, age_bins, delta_t_bins, bin_centres = adding_histograms_from_chunks(hdf5_file_list, bin_size)
    
    # Calculate medians as a function of age and delta_t
    medians = np.zeros((len(age_bins), len(delta_t_bins)))
    for age_bin in tqdm(range(len(age_bins)), desc="Processing age bins:"):
        for delta_t_bin in range(len(delta_t_bins)):
            cdf = np.cumsum(histogram[age_bin, delta_t_bin, :]) / np.sum(histogram[age_bin, delta_t_bin, :])
            medians[age_bin, delta_t_bin] = np.interp(0.5, cdf, bin_centres)
    
    medians = medians[:-1, 1:]
    delta_t_bins = delta_t_bins[1:]
    age_bins = age_bins[:-1]
    #change path here
    np.savez_compressed(f'path/{glob_word}_tdelt.npz', dt=delta_t_bins, J=medians)    #saves the median

if __name__ == "__main__":
    main()
