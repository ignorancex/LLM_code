import numpy as np
import h5py
import fcntl


def read_h5_file(filename):
    with open(filename, 'r') as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        with h5py.File(filename, 'r') as hf:
            data = hf['tensor'][:]
            x = hf['x-coordinate'][:]
            t = hf['t-coordinate'][:-1]
        fcntl.flock(f, fcntl.LOCK_UN)
    return data, x, t


def get_train_data(data_dir, idx_sen, train_indices, test_indices, tlim=1.):

    data, xdisc, tdisc = read_h5_file(data_dir)
    tind = np.where(tdisc == tlim)[0].item()

    X, T = np.meshgrid(xdisc, tdisc)

    # training data
    XeT = np.repeat(np.expand_dims(X, axis=0), len(train_indices), axis=0)
    TeT = np.repeat(np.expand_dims(T, axis=0), len(train_indices), axis=0)
    us_train = data[train_indices][:, :tind, :]
    xt_sens = np.repeat(X[0, idx_sen].reshape((1, 1, -1)), len(train_indices), axis=0)
    tt_sens = np.repeat(T[0, idx_sen].reshape((1, 1, -1)), len(train_indices), axis=0)
    ut_sens = us_train[:, 0:1, idx_sen]
    
    xtrain = XeT[:, :tind, :].reshape((len(train_indices), -1, 1))
    ttrain = TeT[:, :tind, :].reshape((len(train_indices), -1, 1))
    utrain = us_train[:, :tind, :].reshape((len(train_indices), -1, 1))
    vals = xtrain.shape[1]
    
    xbc_train = np.repeat(xt_sens, vals, axis=1).reshape((-1, len(idx_sen)))
    tbc_train = np.repeat(tt_sens, vals, axis=1).reshape((-1, len(idx_sen)))
    ubc_train = np.repeat(ut_sens, vals, axis=1).reshape((-1, len(idx_sen)))

    # Validation data
    XeV = np.repeat(np.expand_dims(X, axis=0), len(test_indices), axis=0)
    TeV = np.repeat(np.expand_dims(T, axis=0), len(test_indices), axis=0)
    us_val = data[test_indices][:, :tind, :]
    xv_sens = np.repeat(X[0, idx_sen].reshape((1, 1, -1)), len(test_indices), axis=0)
    tv_sens = np.repeat(T[0, idx_sen].reshape((1, 1, -1)), len(test_indices), axis=0)
    uv_sens = us_val[:, 0:1, idx_sen]

    xval = XeV[:, :tind, :].reshape((len(test_indices), -1, 1))
    tval = TeV[:, :tind, :].reshape((len(test_indices), -1, 1))
    uval = us_val.reshape((len(test_indices), -1, 1))
    vals = xval.shape[1]
    xbc_val = np.repeat(xv_sens, vals, axis=1).reshape((-1, len(idx_sen)))
    tbc_val = np.repeat(tv_sens, vals, axis=1).reshape((-1, len(idx_sen)))
    ubc_val = np.repeat(uv_sens, vals, axis=1).reshape((-1, len(idx_sen)))

    return (xtrain.reshape((-1, 1)), ttrain.reshape((-1, 1)), utrain.reshape((-1, 1)),
            xbc_train, tbc_train, ubc_train,
            xval.reshape((-1, 1)), tval.reshape((-1, 1)), uval.reshape((-1, 1)),
            xbc_val, tbc_val, ubc_val)
