import numpy as np
from scipy.io import loadmat


def get_train_data(data_dir, idx_seq, train_indices, test_indices, tlim=10.):

    data = loadmat(data_dir)
    tdisc = data['tspan']
    xdisc = data['x']
    tind = np.where(tdisc == tlim)[1].item()

    X, T = np.meshgrid(xdisc, tdisc)

    # training data
    XeT = np.repeat(np.expand_dims(X[1:tind, :], axis=0), len(train_indices), axis=0)
    TeT = np.repeat(np.expand_dims(T[1:tind, :], axis=0), len(train_indices), axis=0)
    us_train = data['output'][train_indices][:, 1:tind, :]
    ui_train = data['input'][train_indices]

    xt_seq = np.repeat(X[0, idx_seq].reshape((1, 1, -1)), len(train_indices), axis=0)
    tt_seq = np.repeat(T[0, idx_seq].reshape((1, 1, -1)), len(train_indices), axis=0)
    ut_seq = np.expand_dims(ui_train[:, idx_seq], axis=1)

    xtrain = XeT.reshape((len(train_indices), -1, 1))
    ttrain = TeT.reshape((len(train_indices), -1, 1))
    utrain = us_train.reshape((len(train_indices), -1, 1))
    vals = xtrain.shape[1]

    xbc_train = np.repeat(xt_seq, vals, axis=1).reshape((-1, len(idx_seq)))
    tbc_train = np.repeat(tt_seq, vals, axis=1).reshape((-1, len(idx_seq)))
    ubc_train = np.repeat(ut_seq, vals, axis=1).reshape((-1, len(idx_seq)))

    # Validation data
    XeV = np.repeat(np.expand_dims(X[1:tind, :], axis=0), len(test_indices), axis=0)
    TeV = np.repeat(np.expand_dims(T[1:tind, :], axis=0), len(test_indices), axis=0)
    us_val = data['output'][test_indices][:, 1:tind, :]
    ui_val = data['input'][test_indices]

    xv_seq = np.repeat(X[0, idx_seq].reshape((1, 1, -1)), len(test_indices), axis=0)
    tv_seq = np.repeat(T[0, idx_seq].reshape((1, 1, -1)), len(test_indices), axis=0)
    uv_seq = np.expand_dims(ui_val[:, idx_seq], axis=1)

    xval = XeV.reshape((len(test_indices), -1, 1))
    tval = TeV.reshape((len(test_indices), -1, 1))
    uval = us_val.reshape((len(test_indices), -1, 1))
    vals = xval.shape[1]
    xbc_val = np.repeat(xv_seq, vals, axis=1).reshape((-1, len(idx_seq)))
    tbc_val = np.repeat(tv_seq, vals, axis=1).reshape((-1, len(idx_seq)))
    ubc_val = np.repeat(uv_seq, vals, axis=1).reshape((-1, len(idx_seq)))

    return (xtrain.reshape((-1, 1)), ttrain.reshape((-1, 1)), utrain.reshape((-1, 1)),
            xbc_train, tbc_train, ubc_train,
            xval.reshape((-1, 1)), tval.reshape((-1, 1)), uval.reshape((-1, 1)),
            xbc_val, tbc_val, ubc_val)
