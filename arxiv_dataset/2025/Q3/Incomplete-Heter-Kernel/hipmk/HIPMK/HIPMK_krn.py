from ctypes import c_float

import numpy as np

try:
    import pymp
    pymp_found = False
except ImportError as e:
    pymp_found = False


from .equal_freq_discretization import EqualFrequencyDiscretizer


class HIPMK_Kernel:
    def __init__(self, nbins = None, stats = None):
        self.nbins_ = nbins
        self.stats_ = stats
        self.bin_counts_ = None

    def build_model(self, train, test):

        def get_bin_dissimilarity():
            bin_dissim = [[] for i in range(self.ndim_)]
            max_num_bins = max(self.num_bins_)

            for i in range(self.ndim_):
                n_bins = self.num_bins_[i]
                bin_cf = [0 for j in range(n_bins)]
                cf = 0

                if (self.stats_ is not None) and ("Nominal" in self.stats_["attribute"][i]["type"]):
                    for j in range(n_bins):
                        bin_cf[j] = self.bin_counts_[i][j]
                else:
                    for j in range(n_bins):
                        cf = cf + self.bin_counts_[i][j]
                        bin_cf[j] = cf

                b_mass = [[0.0 for j in range(max_num_bins)] for k in range(max_num_bins)]

                for j in range(n_bins):
                    for k in range(j, n_bins):
                        if (self.stats_ is not None) and ("Nominal" in self.stats_["attribute"][i]["type"]):
                            if j == k:
                                prob_mass = (bin_cf[k] + 1) / (self.ndata_ + n_bins)
                            else:
                                prob_mass = (bin_cf[k] + bin_cf[j] + 1) / (self.ndata_ + n_bins)
                        else:
                            prob_mass = (bin_cf[k] - bin_cf[j] + self.bin_counts_[i][j] + 1) / (self.ndata_ + n_bins)

                        b_mass[j][k] = np.log(prob_mass)
                        b_mass[k][j] = b_mass[j][k]

                bin_dissim[i] = b_mass

            return np.array(bin_dissim)

        self.ndata_ = len(train)
        self.ndim_ = len(train[0])

        if self.nbins_ is None:
            self.nbins_ = int(np.log2(self.ndata_) + 1)

        self.dimVec_ = np.array([i for i in range(self.ndim_)])
        self.discretiser_ = EqualFrequencyDiscretizer(train, self.nbins_, self.stats_)
        self.bin_cuts_, self.bin_counts_,self.miss_counts_ = self.discretiser_.get_bin_cuts_counts()
        self.num_bins_ = self.discretiser_.get_num_bins()
        self.bin_dissimilarities_ = get_bin_dissimilarity()

        new_test = []

        for i in range(len(test)):
            new_test.append(self.discretiser_.get_bin_id(test[i, :]))

        return self.discretiser_.get_data_bin_id(), np.array(new_test, dtype = c_float, order = "C")

    def set_nbins(self, nbins):
        self.nbins_ = nbins

    def transform(self, train, test=None):
        def cal_dissimilarity(mass):  

            if mass is None or mass == 0:
                raise ValueError("Mass cannot be None or zero for calculating dissimilarity.")
            
            # Calculate the probability mass
            prob_mass = mass / self.ndata_
            # Return the log of the probability mass
            return np.log(prob_mass)

        def re_assign_bin(index, ref_bin):
            bin_count = self.bin_counts_[index]
            # Check if stats is valid and type is "Numeric" or "Ordinal"
            if (self.stats_ is None) or (self.stats_["attribute"][index]["type"] in ["Numeric", "Ordinal"]):

                left_bins = bin_count[:int(ref_bin) + 1]
                # Bins greater than or equal to ref_bin
                right_bins = bin_count[int(ref_bin):]
                
                mass_left = sum(left_bins)
                mass_right = sum(right_bins)
                if mass_left > mass_right:
                    return 0  , mass_left
                else:
                    return len(bin_count) - 1 , mass_left
            else:
                max_bin = np.argmax(bin_count)

                
                return max_bin, bin_count[max_bin]



        def convert(imput_x,index_x, imput_y,index_y):
            x_bin_ids = imput_x[index_x] # imput row
            y_bin_ids = imput_y[index_y] # imput row
            total_mass = None
            # Check if -1 exists in either x_bin_ids or y_bin_ids
            if -1 in x_bin_ids or -1 in y_bin_ids:
                total_mass = None
                for col, x_bin_id in enumerate(x_bin_ids):
                    if x_bin_id == -1 and y_bin_ids[col] == -1: # both missing
                        missing_mass = self.miss_counts_[col]
                        if (self.stats_ is None) or (self.stats_["attribute"][col]["type"] in ["Numeric", "Ordinal"]):
                            total_mass = self.ndata_ + missing_mass
                        else:
                            total_mass = (np.max(self.bin_counts_[col])) + missing_mass
                        y_bin_ids[col] = self.nbins_
                        x_bin_ids[col] = 0
                    elif x_bin_id == -1: # if X_bin id is missing
                        new_id, max_mass = re_assign_bin(col, y_bin_ids[col])
                        x_bin_ids[col] = new_id
                        total_mass = max_mass + missing_mass
                    elif y_bin_ids[col] == -1: # if y_bin id is missing
                        new_id, max_mass = re_assign_bin(col, x_bin_ids[col])
                        y_bin_ids[col]= new_id
                        total_mass = max_mass + missing_mass
                return x_bin_ids, y_bin_ids, total_mass
            else:
                return x_bin_ids, y_bin_ids, total_mass

        def dissimilarity(x_bin_ids, y_bin_ids, total_mass):
            len_x, len_y = len(x_bin_ids), len(y_bin_ids)

            # check the vector size
            if (len_x != self.ndim_) or (len_y != self.ndim_):
                raise IndexError("Number of columns does not match.")
            m_dissim = self.bin_dissimilarities_[self.dimVec_, x_bin_ids.astype(int), y_bin_ids.astype(int)]
            return np.sum(m_dissim) / self.ndim_

        pymp.config.nested = True

        if pymp_found:
            if test is None: # train similarity
                d = pymp.shared.array((len(train), len(train)))
                x_x = pymp.shared.array((len(train)))

                with pymp.Parallel() as p1:
                    for i in p1.range(len(train)):
                        temp_train_i, temp_train_j,total_mass = convert(train,i, train,i)
                        x_x[i] = dissimilarity(temp_train_i, temp_train_j)
                        

                with pymp.Parallel() as p1:
#                    with pymp.Parallel() as p2:
                        for i in p1.range(len(train)):
                            for j in range(i, len(train)):
                                temp_train_i, temp_train_j,total_mass = convert(train,i, train,j)
                                x_y = dissimilarity(temp_train_i, temp_train_j, total_mass)
                                d[i][j] = (2.0 * x_y) / (x_x[i] + x_x[j])
                                d[j][i] = d[i][j]
                        
            else: # test similarity similarity
                d = pymp.shared.array((len(train), len(test)))
                y_y = pymp.shared.array(len(test))

                # Parallel computation of y_y (self-dissimilarity of test set)
                with pymp.Parallel() as p1:
                    for i in p1.range(len(test)):
                        total_mass = None 
                        y_y[i] = dissimilarity(test[i], test[i], total_mass)

                # Parallel computation of d matrix (train vs test dissimilarity)
                with pymp.Parallel() as p1:
                    for i in p1.range(len(train)):
                        # Convert train[i] only once, outside the inner loop
                        temp_train_i, temp_train_j,total_mass = convert(train,i, train,i)
                        x_x = dissimilarity(temp_train_i, temp_train_j,total_mass )
                        for j in range(len(test)):  # No need for additional parallelism here
                            # Convert only train[i] and test[j] for cross-dissimilarity
                            temp_train, temp_test,total_mass = convert(train,i, test,j)
                            x_y = dissimilarity(temp_train, temp_test,total_mass )
                            # Update the shared dissimilarity matrix
                            d[i][j] = (2.0 * x_y) / (x_x + y_y[j])
                
        else:
            if test is None:
                d = np.empty((len(train), len(train)))
                x_x = [0.0 for i in range(len(train))]

                for i in range(len(train)):
                    train[i], train[i],total_mass = convert(train,i, train,i) 
                    total_mass = None
                    x_x[i] = dissimilarity(train[i], train[i],total_mass)

                for i in range(len(train)):
                    for j in range(i, len(train)):
                        train[i], train[j],total_mass = convert(train,i, train,j) 
                        x_y = dissimilarity(train[i], train[j],total_mass)

                        d[i][j] = (2.0 * x_y) / (x_x[i] + x_x[j])
                        d[j][i] = d[i][j]
            else:
                d = np.empty((len(train), len(test)))
                y_y = [0.0 for i in range(len(test))]
                for i in range(len(test)):
                    temp_test_i, temp_test_j,total_mass = convert(test,i, test,i) 
                    y_y[i] = dissimilarity(temp_test_i, temp_test_j,total_mass)

                for i in range(len(train)):
                    # Precompute dissimilarities for the train set using converted values
                    temp_train_i, temp_train_j,total_mass = convert(train,i,train,i)
                    x_x = dissimilarity(temp_train_i, temp_train_j,total_mass)

                    for j in range(len(test)):
                        # Convert train[i] and test[j] once for the cross-dissimilarity
                        temp_train, temp_test,total_mass = convert(train,i, test,j)
                        
                        # Compute cross-dissimilarity
                        x_y = dissimilarity(temp_train, temp_test,total_mass)

                        # Update d[i][j] using the precomputed self-dissimilarities
                        d[i][j] = (2.0 * x_y) / (x_x + y_y[j])

        return np.array(d)
