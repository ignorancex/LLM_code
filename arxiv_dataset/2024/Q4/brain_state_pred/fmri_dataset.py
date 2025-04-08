import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class rfMRIDataset(Dataset):
    """
        Args:
            data_dir (string): Directory with all the data files.
            sub_list (list of string): List of subjects ids that is used
            sample_size (int): length of the input time series, default = 10
            max_window_size: maximum window size we tried, just want to make sure the test results are comparable
    """

    def __init__(self, data_dir, sub_list, sample_size, max_window_size):
        self.data_dir = data_dir
        self.subjects = sub_list
        self.files    = []
        for sub in self.subjects: # loop through subjects id
            sub_files = glob.glob(self.data_dir+sub+'/*REST*_p.npy') # get the file names from the subfolder
            sub_files.sort() # sort the file names
            self.files += sub_files # add to the list

        # some additional attributes for calculation
        self.max_window_size = max_window_size
        self.num_ses = len(self.files) # the number of sessions
        self.sample_size = sample_size # number of timepoints for each sample
        self.time_size = np.load(self.files[0]).shape[0] # total number of time points for one session
        self.num_samples_single = self.time_size - self.max_window_size # number of samples for each session

    def __len__(self):
        total_num_samples = self.num_samples_single * self.num_ses # total number of samples for all sessions
        return total_num_samples

    def __getitem__(self, idx):
        session_idx = idx // self.num_samples_single # get the quotient, which corresponds to the session index
        sample_idx = (idx % self.num_samples_single) + self.max_window_size  # tthe index for the time point to be predicted
        data = np.load(self.files[session_idx]) # load the corresponding session
        time_series = data[sample_idx-self.sample_size:sample_idx, :] # use index slicing to get the time series (input)
        time_point = data[sample_idx, :] # get the expected output
        return time_series, time_point
