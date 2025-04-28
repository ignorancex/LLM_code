import itertools

import h5py
import numpy as np
import pathos


def extract_info_from_h5(filepath):
    data_dict = {}
    with h5py.File(filepath, "r") as f:
        for key in f.keys():
            data_dict[key] = f[key][()]
        param_dict = dict(f.attrs.items())
    return data_dict, param_dict


def write_to_h5(filepath, data_dict, param_dict):
    with h5py.File(filepath, "w") as f:
        for key, val in data_dict.items():
            f.create_dataset(key, data=val)
        for kwarg in param_dict.keys():
            try:
                f.attrs[kwarg] = param_dict[kwarg]
            except TypeError:
                f.attrs[kwarg] = str(param_dict[kwarg])


def param_map(f, parameters, map_fun=map, dtype=object):
    """Code due to Peter Groszkowski
    Maps function `f` over the product of the parameters given in `parameters`
    (which is assumed to have a list-of-lists-like structure). The returned data
    is an ndarray with dimensions set by the input data in `parameters`.

    e.g.:

    input_data_a=['1','2','3']
    input_data_b=['a','b']

    def f(a, b):
        return "{}{}".format(a,b)

    data=param_map(f, [input_data_a, input_data_b], map_fun=map)

    print(data.shape) #gives (3,2)
    print(data[0,0]) #gives '1a'


    Parameters
    ----------
    f:  function
        Function that is to be applied to each element of an array
    parameters: iterable of iterables (i.e. list of lists)


    Returns:
    -------
    ndarray with f applied to products of input parameters

    """
    dims_list = [len(i) for i in parameters]
    total_dim = np.prod(dims_list)

    # all the possible combinations of input parameters
    parameters_prod = tuple(itertools.product(*parameters))

    # We want to force a 1d numpy array of size total_dim,
    # regardless what 'data' is, even if it's iterable
    # (list, sequence, etc), but by default, the np.array()
    # constructor will try to create new array dimensions
    # from objects that can be indexed (sequences, list, etc).
    data = np.empty(total_dim, dtype=dtype)
    for i, d in enumerate(map_fun(f, parameters_prod)):
        data[i] = d

    return np.reshape(data, dims_list)


def unpack_param_map(param_map_array):
    """Assumption is that array is of dtype=object (output of param_map) and stores
    in each entry an array that we want to pack onto the end
    """
    dims = param_map_array.shape
    zero_idx = [0] * len(dims)
    inner_dim = param_map_array[tuple(zero_idx)].shape
    result = np.empty((dims + inner_dim), dtype=param_map_array.dtype)
    idx_ranges = [range(dim) for dim in dims]
    idx_prod = itertools.product(*idx_ranges)
    for idxs in idx_prod:
        result[tuple(idxs)] = param_map_array[tuple(idxs)]
    return result


def parallel_map(num_cpus, func, parameters):
    if num_cpus == 1:
        return map(func, parameters)

    with pathos.pools.ProcessPool(nodes=num_cpus) as pool:
        result = pool.map(func, parameters)
    return result
