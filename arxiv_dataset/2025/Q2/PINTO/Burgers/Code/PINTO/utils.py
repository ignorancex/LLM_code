import numpy as np
from scipy.io import loadmat
from pyDOE import lhs


def get_train_data(data_dir, domain_samples, seq_len, indices, val_indices):
    # Inputs:
    # data_dir-> path to the data file (PDEBENCH Advection file as mentioned in the README file)
    # seq_len-> length of the sequence for BPE and BVE units in an architecture
    # domain_samples-> number of domain samples for imposing the Advection governing equation
    # indices-> indices of initial conditions that are used for training the PINTO model
    # val_indices-> indices of initial conditions that are used for validation/testing of the trained PINTO model

    # Outputs:
    # xd, td-> domain samples on which the Advection governing equation is imposed
    # xb, tb, ub-> periodic boundary conditions
    # x_init, t_init, u_init-> initial conditions
    # xbc_in, tbc_in, ubc_in-> inputs for BPE, BVE units corresponding to the domain samples
    # xbc_b, tbc_b, ubc_b-> inputs for BPE, BVE units corresponding to the periodic boundary conditions
    # xbc_init, tbc_init, ubc_init-> inputs for BPE, BVE units corresponding to the initial conditions
    # xval, tval, uval-> validation data
    # xbc_val, tbc_val, ubc_val-> inputs for BPE, BVE units corresponding to the validation data

    if data_dir is None:
        raise ValueError("data_dir cannot be None please provide the path "
                         "to the data file as mentioned in the README file.")

    data = loadmat(data_dir)
    tdisc = data['tspan']
    xdisc = data['x']
    tind = np.where(tdisc == 1)[1].item()

    X, T = np.meshgrid(xdisc, tdisc)

    Xe = np.repeat(np.expand_dims(X, axis=0), len(indices), axis=0)
    Te = np.repeat(np.expand_dims(T, axis=0), len(indices), axis=0)

    # boundary sequence and values
    idx_si = np.random.choice(len(X[0, :]), seq_len, replace=False)
    us = data['output'][indices]
    ui = data['input'][indices]

    x_init = np.transpose(Xe[:, 0:1, ::16], [0, 2, 1])
    t_init = np.transpose(Te[:, 0:1, ::16], axes=[0, 2, 1])
    u_init = np.transpose(np.expand_dims(ui[:, ::16], axis=1), axes=[0, 2, 1])
    x_left = Xe[:, 1:tind, 0:1]
    t_left = Te[:, 1:tind, 0:1]
    u_left = us[:, 1:tind, 0:1]
    x_right = Xe[:, 1:tind, -1:]
    t_right = Te[:, 1:tind, -1:]
    u_right = us[:, 1:tind, -1:]

    upper_bound = np.array([xdisc.max(), 1.]).reshape((1, -1))
    lower_bound = np.array([xdisc.min(), 0.]).reshape((1, -1))
    grid_loc = (upper_bound - lower_bound) * lhs(2, domain_samples) + lower_bound

    xr = np.repeat(np.expand_dims(grid_loc[:, 0:1], axis=0), len(indices), axis=0)
    tr = np.repeat(np.expand_dims(grid_loc[:, 1:2], axis=0), len(indices), axis=0)
    xd = np.concatenate((xr, x_left, x_right), axis=1)
    td = np.concatenate((tr, t_left, t_right), axis=1)
    xb = np.concatenate((x_left, x_right), axis=1)
    tb = np.concatenate((t_left, t_right), axis=1)
    ub = np.concatenate((u_left, u_right), axis=1)

    ins = xd.shape[1]
    bs = xb.shape[1]
    inits = x_init.shape[1]

    x_sensor = np.repeat(X[0, idx_si].reshape((1, 1, -1)), len(indices), axis=0)
    t_sensor = np.repeat(T[0, idx_si].reshape((1, 1, -1)), len(indices), axis=0)
    u_sensor = np.expand_dims(ui[:, idx_si], axis=1)

    xbc_in = np.repeat(x_sensor, ins, axis=1).reshape((-1, x_sensor.shape[2]))
    tbc_in = np.repeat(t_sensor, ins, axis=1).reshape((-1, t_sensor.shape[2]))
    ubc_in = np.repeat(u_sensor, ins, axis=1).reshape((-1, u_sensor.shape[2]))

    xbc_b = np.repeat(x_sensor, bs, axis=1).reshape((-1, x_sensor.shape[2]))
    tbc_b = np.repeat(t_sensor, bs, axis=1).reshape((-1, t_sensor.shape[2]))
    ubc_b = np.repeat(u_sensor, bs, axis=1).reshape((-1, u_sensor.shape[2]))

    xbc_init = np.repeat(x_sensor, inits, axis=1).reshape((-1, x_sensor.shape[2]))
    tbc_init = np.repeat(t_sensor, inits, axis=1).reshape((-1, t_sensor.shape[2]))
    ubc_init = np.repeat(u_sensor, inits, axis=1).reshape((-1, u_sensor.shape[2]))

    # Validation data
    XeV = np.repeat(np.expand_dims(X, axis=0), len(val_indices), axis=0)
    TeV = np.repeat(np.expand_dims(T, axis=0), len(val_indices), axis=0)
    us_val = data['output'][val_indices]
    ui_val = data['input'][val_indices]
    xv_sens = np.repeat(X[0, idx_si].reshape((1, 1, -1)), len(val_indices), axis=0)
    tv_sens = np.repeat(T[0, idx_si].reshape((1, 1, -1)), len(val_indices), axis=0)
    uv_sens = np.expand_dims(ui_val[:, idx_si], axis=1)

    xval = XeV.reshape((len(val_indices), -1, 1))
    tval = TeV.reshape((len(val_indices), -1, 1))
    uval = us_val.reshape((len(val_indices), -1, 1))
    vals = xval.shape[1]
    xbc_val = np.repeat(xv_sens, vals, axis=1).reshape((-1, seq_len))
    tbc_val = np.repeat(tv_sens, vals, axis=1).reshape((-1, seq_len))
    ubc_val = np.repeat(uv_sens, vals, axis=1).reshape((-1, seq_len))

    return (xd.reshape((-1, 1)), td.reshape((-1, 1)), xb.reshape((-1, 1)), tb.reshape((-1, 1)), ub.reshape((-1, 1)),
            x_init.reshape((-1, 1)), t_init.reshape((-1, 1)), u_init.reshape((-1, 1)),
            xbc_in, tbc_in, ubc_in, xbc_b, tbc_b, ubc_b,
            xbc_init, tbc_init, ubc_init, idx_si, xval.reshape((-1, 1)), tval.reshape((-1, 1)), uval.reshape((-1, 1)),
            xbc_val, tbc_val, ubc_val)


