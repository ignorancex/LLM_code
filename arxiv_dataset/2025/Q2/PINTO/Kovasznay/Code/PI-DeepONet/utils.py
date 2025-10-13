import numpy as np
from pyDOE import lhs


def get_ibc_and_inner_data(start, stop, grid_size, domain_samples, seq_len, nue):
    # Inputs:
    # start: lower values of the domain [x_start, y_start]
    # stop: upper values of the domain [x_stop, y_stop]
    # grid_size: discretization size for both dimensions (inorder to get the boundary points)
    # domain_samples: number of samples in the domain
    # seq_len: length of the sequence
    # nue: kinematic viscosity values for the PDE or 1/Re

    # outputs:
    # xed, yed: domain points on which the PDE is imposed
    # xeb, yeb: boundary points on which the boundaruy conditions are imposed
    # u_ob, v_ob, p_ob: boundary values
    # xbc_in, ybc_in: input sequence for BPE units corresponding to the domain points
    # ubc_in, vbc_in, pbc_in: input sequence for BVE units corresponding to the domain points
    # xbc_b, ybc_b: input sequence for BPE units corresponding to the boundary points
    # ubc_b, vbc_b, pbc_b: input sequence for BVE units corresponding to the boundary points

    # Boundary Points
    xdisc = np.linspace(start=start[0], stop=stop[0], num=grid_size)
    ydisc = np.linspace(start=stop[1], stop=start[1], num=grid_size)

    X, Y = np.meshgrid(xdisc, ydisc)
    grid_loc = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

    # genearating boundary coordinates
    x_top = np.hstack((X[0, :][:, None], Y[0, :][:, None]))
    x_bottom = np.hstack((X[0, 1:-1][:, None], Y[-1, 1:-1][:, None]))
    x_left = np.hstack((X[1:, 0][:, None], Y[1:, 0][:, None]))
    x_right = np.hstack((X[1:, -1][:, None], Y[1:, 0][:, None]))

    # idx = np.random.choice(grid_size - 2, nu, replace=False)
    xb = np.vstack((x_top[:, 0:1], x_bottom[:, 0:1], x_left[:, 0:1], x_right[:, 0:1]))
    yb = np.vstack((x_top[:, 1:2], x_bottom[:, 1:2], x_left[:, 1:2], x_right[:, 1:2]))

    # generating domain samples using LHS
    lower_bound = grid_loc.min()
    upper_bound = grid_loc.max()
    x_dom = (upper_bound - lower_bound) * lhs(2, domain_samples) + lower_bound
    xd = x_dom[:, 0:1]
    yd = x_dom[:, 1:2]

    xeb = np.repeat(np.expand_dims(xb, axis=0), len(nue), axis=0)
    yeb = np.repeat(np.expand_dims(yb, axis=0), len(nue), axis=0)
    # print(f"xeb, yeb has rank 3 : {xeb.shape, yeb.shape}")

    xed = np.repeat(np.expand_dims(xd, axis=0), len(nue), axis=0)
    yed = np.repeat(np.expand_dims(yd, axis=0), len(nue), axis=0)
    nue_d = np.ones_like(xed)*(np.array(nue).reshape((-1, 1, 1)))
    # print(f"xed, yed has rank 3 : {xed.shape, yed.shape}")

    # getting boundary values
    u_ob, v_ob, p_ob = get_fvalues(xeb, yeb, nue=np.array(nue).reshape((-1, 1, 1)))
    assert (u_ob.shape == v_ob.shape == p_ob.shape == xeb.shape == yeb.shape)

    x_range = stop[0] - start[0]
    y_range = stop[1] - start[1]

    # generating the input sequence for BPE and BVE units
    # left boundary
    xl = np.ones((1, seq_len)) * start[0]
    yl = y_range * np.random.rand(1, seq_len) + start[1]

    # right boundary
    xr = np.ones((1, seq_len)) * stop[0]
    yr = y_range * np.random.rand(1, seq_len) + start[1]

    # Top boundary
    yt = np.ones((1, seq_len)) * stop[1]
    xt = x_range * np.random.rand(1, seq_len) + start[0]

    # Bottom boundary
    ybt = np.ones((1, seq_len)) * start[1]
    xbt = x_range * np.random.rand(1, seq_len) + start[0]

    coord_sen = np.concatenate((np.concatenate((xl, xr, xt, xbt), axis=1).T,
                                np.concatenate((yl, yr, yt, ybt), axis=1).T), axis=1)

    xbc = np.repeat(np.expand_dims(np.concatenate((xl, xr, xt, xbt), axis=1), axis=0), len(nue), axis=0)
    ybc = np.repeat(np.expand_dims(np.concatenate((yl, yr, yt, ybt), axis=1), axis=0), len(nue), axis=0)
    ubc, vbc, pbc = get_fvalues(xbc, ybc, nue=np.array(nue).reshape((-1, 1, 1)))

    xbc_in = np.repeat(xbc, len(xd), axis=1).reshape((-1, 4*seq_len))
    ybc_in = np.repeat(ybc, len(yd), axis=1).reshape((-1, 4*seq_len))
    ubc_in = np.repeat(ubc, len(xd), axis=1).reshape((-1, 4*seq_len))
    vbc_in = np.repeat(vbc, len(xd), axis=1).reshape((-1, 4*seq_len))
    pbc_in = np.repeat(pbc, len(xd), axis=1).reshape((-1, 4*seq_len))

    xbc_b = np.repeat(xbc, len(xb), axis=1).reshape((-1, 4*seq_len))
    ybc_b = np.repeat(ybc, len(yb), axis=1).reshape((-1, 4*seq_len))
    ubc_b = np.repeat(ubc, len(xb), axis=1).reshape((-1, 4*seq_len))
    vbc_b = np.repeat(vbc, len(xb), axis=1).reshape((-1, 4*seq_len))
    pbc_b = np.repeat(pbc, len(xb), axis=1).reshape((-1, 4*seq_len))

    return (xed.reshape((-1, 1)), yed.reshape((-1, 1)), xeb.reshape((-1, 1)),
            yeb.reshape((-1, 1)), u_ob.reshape((-1, 1)), v_ob.reshape((-1, 1)),
            p_ob.reshape((-1, 1)), xbc_in, ybc_in, ubc_in, vbc_in, pbc_in,
            xbc_b, ybc_b, ubc_b, vbc_b, pbc_b, nue_d.reshape((-1, 1)), coord_sen)


def get_fvalues(x, y, nue):
    zeta = (0.5 / nue) - np.sqrt((1 / (4 * nue ** 2)) + 4 * np.pi ** 2)

    u_val = 1 - np.exp(zeta * x) * np.cos(2 * np.pi * y)
    v_val = (zeta / (2 * np.pi)) * np.exp(zeta * x) * np.sin(2 * np.pi * y)
    p_val = 0.5 * (1 - np.exp(2 * zeta * x))

    return u_val, v_val, p_val
