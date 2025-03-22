import numpy as np
import scipy.io as sio

def dset_organizer(n_iter):
    run_n = 1
    dset_all = {}
    dset_all['X'] = np.array([])  # initialize to empty array
    dset_all['Y'] = np.array([])  # initialize to empty array
    for i in range(1, run_n+1):
        mat = sio.loadmat('Acquisition/iter_%d/run_%d.mat' %(n_iter, i))
        mat = mat['data']
        dset = mat[0][0]
        
        if len(dset_all['X']) == 0:  # if this is the first run, simply copy the dataset
            dset_all['X'] = dset['X']
            dset_all['Y'] = dset['Y']
        else:  # otherwise concatenate the datasets
            dset_all['X'] = np.concatenate((dset_all['X'], dset['X']))
            dset_all['Y'] = np.concatenate((dset_all['Y'], dset['Y']))
    sio.savemat('surrogate_pareto/with_uncertainty/ParetoSet_%d.mat' %(n_iter), {'ParetoSet': dset_all['X']})
    sio.savemat('surrogate_pareto/with_uncertainty/ParetoFront_%d.mat' %(n_iter), {'ParetoFront': dset_all['Y']})
        
    return dset_all