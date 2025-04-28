import torch
import numpy as np
from torch.utils.data import Dataset
from functools import partial

def flatten_parameters_with_names(parameters_dict):
    flattened_values = []
    flattened_names = []

    for param_name, param_value in parameters_dict.items():
        if np.isscalar(param_value):
            flattened_values.append(param_value)
            flattened_names.append(param_name)
        else:
            flattened_array = np.asarray(param_value).flatten()
            flattened_values.extend(flattened_array)
            shape = param_value.shape
            for idx in np.ndindex(shape):
                name_with_indices = param_name + "".join(f"[{i}]" for i in idx)
                flattened_names.append(name_with_indices)

    return np.array(flattened_values), flattened_names
class BayesDataStream(Dataset):

    def __init__(self, n_batches, batch_size, sample_theta, sample_y, sample_n):
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.total_size = n_batches * batch_size
        self.sample_theta = sample_theta
        self.sample_y = sample_y
        self.sample_n = sample_n

        # generate a vector of sample sizes for each batch
        batch_sample_sizes = self.sample_n(self.n_batches)  # should return a 1D numpy array

        # now use batch_sample_sizes to create a pre-cached vector of sample sizes for each index,
        # by repeating each batch_sample_size entry for batch_size times.
        # By indexing into this vector when __getitem__ is called, we can ensure
        # that the batches have different sample sizes, but that the sample_size is the same
        # within a batch.
        self.idx_sample_sizes = np.repeat(batch_sample_sizes, batch_size)

    def __len__(self):
        return self.total_size

    def reset_batch_sample_sizes(self):
        batch_sample_sizes = self.sample_n(self.n_batches)
        self.idx_sample_sizes = np.repeat(batch_sample_sizes, self.batch_size)

    def __getitem__(self, idx):
        parms = self.sample_theta()
        y = self.sample_y(parms, self.idx_sample_sizes[idx])
        theta, _ = flatten_parameters_with_names(parms)
        # theta = np.array(list(parms.values()))
        return torch.tensor(theta, dtype=torch.float), torch.tensor(y, dtype=torch.float)


class BayesDataStreamGPU(Dataset):

    def __init__(self, n_batches, batch_size, sample_theta, sample_y, sample_n, device=None):
        if device is None:
            self.device = "cpu"
        else:
            self.device = device

        self.n_batches = n_batches
        self.batch_size = batch_size
        self.total_size = n_batches * batch_size
        self.sample_theta = partial(sample_theta, device=device)
        self.sample_y = partial(sample_y, device=device)
        self.sample_n = sample_n

        # generate a vector of sample sizes for each batch
        batch_sample_sizes = self.sample_n(self.n_batches)  # should return a 1D numpy array

        # now use batch_sample_sizes to create a pre-cached vector of sample sizes for each index,
        # by repeating each batch_sample_size entry for batch_size times.
        # By indexing into this vector when __getitem__ is called, we can ensure
        # that the batches have different sample sizes, but that the sample_size is the same
        # within a batch.
        self.idx_sample_sizes = np.repeat(batch_sample_sizes, batch_size)

    def __len__(self):
        return self.total_size

    def reset_batch_sample_sizes(self):
        batch_sample_sizes = self.sample_n(self.n_batches)
        self.idx_sample_sizes = np.repeat(batch_sample_sizes, self.batch_size)

    def __getitem__(self, idx):
        parms = self.sample_theta()
        y = self.sample_y(parms, self.idx_sample_sizes[idx])
        # theta, _ = self.flatten_parameters_with_names(parms)
        theta = torch.tensor(list(parms.values()), dtype=torch.float32)
        return theta, y


