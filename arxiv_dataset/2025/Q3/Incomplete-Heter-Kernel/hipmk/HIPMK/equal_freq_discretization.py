from ctypes import c_float
import numpy as np
from bisect import bisect_left


class EqualFrequencyDiscretizer(object):

    def __init__(self, data, nbins, stats):
        self.stats = stats
        self.n_data = len(data)
        self.n_dim = len(data[0])
        self.bin_cuts = [[] for i in range(self.n_dim)]
        self.bin_counts = [[] for i in range(self.n_dim)]
        self.miss_counts = [[] for i in range(self.n_dim)]
        self.data_bin_ids = np.array([[-1 for i in range(self.n_dim)] for i in range(self.n_data)])
        #self.data_bin_ids = np.full((self.n_data, self.n_dim), np.nan)  # Initialize with NaN values
        self.num_bins = [0 for i in range(self.n_dim)]
        for i in range(self.n_dim):
            # iteration for each column
            if (self.stats is None) or ("Numeric" in self.stats["attribute"][i]["type"]):
                column_data = data[:, i]  # looking at each column
                # Filter out NaN values from the column
                temp = column_data[~np.isnan(column_data)]
                
                b_cuts, b_counts = self.equal_freq_histograms(temp, nbins)  # rebuild the distribution using the non-missing data
            else:
                #num_missing = np.sum(np.isnan(column_data))
                b_cuts, b_counts = self.equal_freq_histograms_non_numeric(data[:, i], i)
            num_missing = np.sum(np.isnan(data[:, i]))
            self.miss_counts[i] = num_missing # mass in missing_bin
            self.bin_cuts[i] = b_cuts
            self.bin_counts[i] = b_counts
            self.num_bins[i] = len(b_counts)
            for j in range(self.n_data):
                if np.isnan(data[j, i]):
                    self.data_bin_ids[j, i] = -1
                elif (self.stats is None) or ("Numeric" in self.stats["attribute"][i]["type"]):
                    self.data_bin_ids[j, i] = bisect_left(b_cuts[1:-1], data[j, i])
                else:
                    self.data_bin_ids[j, i] = int(data[j, i])

    def get_bin_cuts_counts(self):
        return self.bin_cuts, self.bin_counts, self.miss_counts

    def get_num_bins(self):
        return self.num_bins

    def get_data_bin_id(self):
        return np.array(self.data_bin_ids, dtype=c_float)

    def get_bin_id(self, x):
        x_bin_ids = [-1 for i in range(self.n_dim)]
        #x_bin_ids = [np.nan for i in range(self.n_dim)]
        for i in range(self.n_dim):
            if np.isnan(x[i]):
              x_bin_ids[i] = -1
            elif (self.stats is None) or ("Numeric" in self.stats["attribute"][i]["type"]):
                cuts = self.bin_cuts[i]
                x_bin_ids[i] = bisect_left(cuts[1:-1], x[i])
            else:
                x_bin_ids[i] = int(x[i])

        return np.array(x_bin_ids)

    def equal_freq_histograms_non_numeric(self, x, idx):
        # get unique values and counts
        unique_values, unique_value_counts = np.unique(x, return_counts=True)

        if (self.stats is not None) and ("Numeric" not in self.stats["attribute"][idx]["type"]):
            chk_cnt = []
            idx_chk = 0

            for i in range(len(self.stats["attribute"][idx]["values"])):
                if (idx_chk < len(unique_values)) and (unique_values[idx_chk] == i):
                    chk_cnt.append(unique_value_counts[idx_chk])
                    idx_chk += 1
                else:
                    chk_cnt.append(0)

            unique_value_counts = chk_cnt

        # return the result
        return np.array([]), np.array(unique_value_counts)

    def equal_freq_histograms(self, x, nbins):
        b_cuts = []
        b_counts = []

        # get unique values and counts
        unique_values, unique_value_counts = np.unique(x, return_counts=True)
        num_unique_vals = len(unique_values)

        # start discretization
        x_size = len(x)
        exp_freq = x_size / nbins
        freq_count = 0
        last_freq_count = 0
        last_id = -1
        cut_point_id = 0

        b_cuts.append(unique_values[0] - (unique_values[1] - unique_values[0]) / 2)

        for i in range(num_unique_vals - 1):
            freq_count += unique_value_counts[i]
            x_size -= unique_value_counts[i]
            # check if ideal bin count is reached
            if freq_count >= exp_freq:
                # check if this one is worse than the last one
                if ((exp_freq - last_freq_count) < (freq_count - exp_freq)) and (last_id != -1):
                    cut_point = (unique_values[last_id] + unique_values[last_id + 1]) / 2
                    # check if it's worth merging the new bin with the last bin
                    if len(b_counts) > 1:
                        if (abs(b_counts[-1] + last_freq_count) - exp_freq) < abs(last_freq_count - exp_freq):
                            b_counts[-1] += last_freq_count
                            b_cuts[-1] = cut_point
                        else:
                            b_cuts.append(cut_point)
                            b_counts.append(last_freq_count)
                    else:
                        b_cuts.append(cut_point)
                        b_counts.append(last_freq_count)

                    freq_count -= last_freq_count
                    last_freq_count = freq_count
                    last_id = i
                else:
                    b_cuts.append((unique_values[i] + unique_values[i + 1]) / 2)
                    b_counts.append(freq_count)
                    freq_count = 0
                    last_freq_count = 0
                    last_id = -1
                cut_point_id += 1
            else:
                last_id = i
                last_freq_count = freq_count

        # handle last unique value frequency
        last_unique_value_count = unique_value_counts[i + 1]
        freq_count += last_unique_value_count
        x_size -= unique_value_counts[i + 1]

        if x_size != 0:
            print('ERROR: Something is wrong, x_size should be 0 but x_size=%s' % (x_size))
            exit()

        if (last_id != -1) and (abs(exp_freq - last_unique_value_count) < abs(freq_count - exp_freq)):
            b_cuts.append((unique_values[last_id] + unique_values[last_id + 1]) / 2)
            b_counts.append(last_freq_count)
            freq_count -= last_freq_count

        b_counts.append(freq_count)

        # merge last partitions if possible
        if len(b_counts) >= 2:
            if abs((b_counts[-2] + b_counts[-1]) - exp_freq) < abs(exp_freq - b_counts[-1]):
                b_counts[-2] += b_counts[-1]
                del b_cuts[-1]
                del b_counts[-1]

        if len(b_counts) >= 3:
            if abs((b_counts[-3] + b_counts[-2]) - exp_freq) < abs(exp_freq - b_counts[-2]):
                b_counts[-3] += b_counts[-2]
                b_counts[-2] = b_counts[-1]
                del b_cuts[-2]
                del b_counts[-1]

        b_cuts.append(unique_values[num_unique_vals - 1] + (unique_values[num_unique_vals - 1] - unique_values[num_unique_vals - 2]) / 2)

        assert sum(b_counts) == len(x)
        assert len(b_cuts) == (len(b_counts) + 1)

        return np.array(b_cuts), np.array(b_counts)

    def equal_freq_histograms_weka(self, x, nbins):  # WEKA Implementation
        b_cuts = []
        b_counts = []
        unique_values, unique_value_counts = np.unique(x, return_counts=True)
        num_unique_vals = len(unique_values)

        x_size = len(x)
        exp_freq = x_size / nbins
        freq_count = 0
        last_freq_count = 0
        last_id = -1
        cut_point_id = 0
        for i in range(num_unique_vals - 1):
            freq_count += unique_value_counts[i]
            x_size -= unique_value_counts[i]
            if freq_count >= exp_freq:
                if ((exp_freq - last_freq_count) < (freq_count - exp_freq)) and (last_id != -1):
                    b_cuts.append((unique_values[last_id] + unique_values[last_id + 1]) / 2)
                    b_counts.append(last_freq_count)
                    freq_count -= last_freq_count
                    last_freq_count = freq_count
                    last_id = i
                else:
                    b_cuts.append((unique_values[i] + unique_values[i + 1]) / 2)
                    b_counts.append(freq_count)
                    freq_count = 0
                    last_freq_count = 0
                    last_id = -1
                cut_point_id += 1
                exp_freq = (x_size + freq_count) / (nbins - cut_point_id)
            else:
                last_id = i
                last_freq_count = freq_count

        freq_count += unique_value_counts[i + 1]

        if (cut_point_id < nbins) and (freq_count > exp_freq) and ((exp_freq - last_freq_count) < (freq_count - exp_freq)):
            b_cuts.append((unique_values[last_id] + unique_values[last_id + 1]) / 2)
            b_counts.append(last_freq_count)
            b_counts.append(freq_count - last_freq_count)
        else:
            b_counts.append(freq_count)

        return np.array(b_cuts), np.array(b_counts)
