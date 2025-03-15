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
        print("Usage: python agg_hist_delt.py <glob_word> <bin_size>")
        sys.exit(1)
    
    glob_word = sys.argv[1]
    bin_size = int(sys.argv[2])
    
    hdf5_file_list = glob.glob(f'path/{glob_word}.h5')    #change path here
    histogram, age_bins, delta_t_bins, bin_centres = adding_histograms_from_chunks(hdf5_file_list, bin_size)
    
    # Median as function of delta_t
    medians1 = np.zeros(len(delta_t_bins))
    histogram_delta_t = histogram.sum(axis=0)
    
    for delta_t_bin in range(len(delta_t_bins)):
        cdf = np.cumsum(histogram_delta_t[delta_t_bin, :]) / np.sum(histogram_delta_t[delta_t_bin, :])
        medians1[delta_t_bin] = np.interp(0.5, cdf, bin_centres)
    
    medians1 = medians1[1:]
    delta_t_bins = delta_t_bins[1:]
    #change path here
    np.savez_compressed(f'path/{glob_word}_delt.npz', dt=delta_t_bins, J=medians1)

      
if __name__ == "__main__":
    main()
  
