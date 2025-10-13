import numpy as np
from pyDOE import lhs
import tensorflow as tf


def get_ibc_and_inner_data(start, stop, boundary_samples, domain_samples, seq_len,
                           top_velocities):

    # inputs:
    # start: float, start point for each axis
    # stop: float, stop point for each axis
    # boundary_samples: int, number of samples on the boundary
    # domain_samples: int, number of samples in the domain
    # seq_len: int, length of the sequence
    # top_velocities: list of floats, velocities for the top boundary

    # outputs:
    # xr, yr: domain coordinates to impose PDE loss
    # xb, yb: boundary coordinates to impose BC loss
    # ub, vb: boundary conditions
    # x_top, y_top: top boundary coordinates
    # u_top, v_top: top boundary conditions
    # xbc_in, ybc_in, ubc_in, vbc_in: input sequence for BPE, and BVE corresponding to the domain points
    # xbc_b, ybc_b, ubc_b, vbc_b: input sequence for BPE, and BVE corresponding to the boundary points
    # xbc_top, ybc_top, ubc_top, vbc_top: input sequence for BPE, and BVE corresponding to the top boundary points

    # generating the boundary points
    x_top = np.repeat(np.expand_dims(np.random.uniform(start, stop, [boundary_samples, 1]), axis=0),
                      len(top_velocities), axis=0)
    y_top = np.ones_like(x_top) * stop
    u_top = np.ones_like(x_top) * np.array(top_velocities).reshape((len(top_velocities), 1, 1))
    v_top = np.zeros_like(x_top)

    x_bottom = np.repeat(np.expand_dims(np.random.uniform(start, stop, [boundary_samples, 1]), axis=0),
                         len(top_velocities), axis=0)
    y_bottom = np.ones_like(x_bottom) * start
    u_bottom = np.zeros_like(x_bottom)
    v_bottom = np.zeros_like(x_bottom)

    y_left = np.repeat(np.expand_dims(np.concatenate((np.random.uniform(start, stop, [boundary_samples, 1]),
                                                      np.array([[0.], [1.]])), axis=0), axis=0), len(top_velocities),
                       axis=0)
    x_left = np.ones_like(y_left) * start
    u_left = np.zeros_like(x_left)
    v_left = np.zeros_like(x_left)

    y_right = np.repeat(np.expand_dims(np.concatenate((np.random.uniform(start, stop, [boundary_samples, 1]),
                                                      np.array([[0.], [1.]])), axis=0), axis=0), len(top_velocities),
                        axis=0)
    x_right = np.ones_like(y_right) * stop
    u_right = np.zeros_like(x_right)
    v_right = np.zeros_like(x_right)

    # generating the domain points
    x_dom = (stop - start) * lhs(2, domain_samples) + start
    xd = np.repeat(np.expand_dims(x_dom[:, 0:1], axis=0), len(top_velocities), axis=0)
    yd = np.repeat(np.expand_dims(x_dom[:, 1:2], axis=0), len(top_velocities), axis=0)

    xb = np.concatenate((x_bottom, x_left, x_right), axis=1)
    yb = np.concatenate((y_bottom, y_left, y_right), axis=1)
    ub = np.concatenate((u_bottom, u_left, u_right), axis=1)
    vb = np.concatenate((v_bottom, v_left, v_right), axis=1)

    xr = np.concatenate((xd, xb), axis=1)
    yr = np.concatenate((yd, yb), axis=1)

    x_st = np.repeat(np.expand_dims(np.random.uniform(start, stop, [1, seq_len]),
                                    axis=0), len(top_velocities), axis=0)
    y_st = np.ones_like(x_st) * stop
    u_st = np.ones_like(x_st) * np.array(top_velocities).reshape((len(top_velocities), 1, 1))
    v_st = np.zeros_like(x_st)

    X_sen = np.concatenate((x_st[0:1, :, :].reshape(-1, 1), y_st[0:1, :, :].reshape(-1, 1)), axis=1)

    ins = xr.shape[1]
    bs = xb.shape[1]
    ints = x_top.shape[1]

    xbc_in = np.repeat(x_st, ins, axis=1).reshape((-1, seq_len))
    ybc_in = np.repeat(y_st, ins, axis=1).reshape((-1, seq_len))
    ubc_in = np.repeat(u_st, ins, axis=1).reshape((-1, seq_len))
    vbc_in = np.repeat(v_st, ins, axis=1).reshape((-1, seq_len))

    xbc_b = np.repeat(x_st, bs, axis=1).reshape((-1, seq_len))
    ybc_b = np.repeat(y_st, bs, axis=1).reshape((-1, seq_len))
    ubc_b = np.repeat(u_st, bs, axis=1).reshape((-1, seq_len))
    vbc_b = np.repeat(v_st, bs, axis=1).reshape((-1, seq_len))

    xbc_top  = np.repeat(x_st, ints, axis=1).reshape((-1, seq_len))
    ybc_top = np.repeat(y_st, ints, axis=1).reshape((-1, seq_len))
    ubc_top = np.repeat(u_st, ints, axis=1).reshape((-1, seq_len))
    vbc_top = np.repeat(v_st, ints, axis=1).reshape((-1, seq_len))

    return (xr.reshape((-1, 1)), yr.reshape((-1, 1)), xb.reshape((-1, 1)), yb.reshape((-1, 1)), ub.reshape((-1, 1)),
            vb.reshape((-1, 1)), x_top.reshape((-1, 1)), y_top.reshape((-1, 1)), u_top.reshape((-1, 1)),
            v_top.reshape((-1, 1)), xbc_in, ybc_in, ubc_in, vbc_in, xbc_b, ybc_b, ubc_b, vbc_b,
            xbc_top, ybc_top, ubc_top, vbc_top, X_sen)


class SedLrSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, higher_decay_rate, lower_decay_rate, step_lim,
                 staircase=True):
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.hdecay_rate = higher_decay_rate
        self.ldecay_rate = lower_decay_rate
        self.step_lim = step_lim
        self.lr = tf.Variable(initial_learning_rate)
        self.iterations = tf.Variable(-1)
        self.staircase = staircase

    def __call__(self, step):
        self.iterations.assign_add(1)
        initial_learning_rate = tf.convert_to_tensor(self.initial_learning_rate, name="initial_learning_rate")
        dtype = initial_learning_rate.dtype
        decay_step = tf.cast(self.decay_steps, dtype=dtype)
        hdecay_rate = tf.cast(self.hdecay_rate, dtype=dtype)
        ldecay_rate = tf.cast(self.ldecay_rate, dtype=dtype)
        global_step_recomp = tf.cast(step, dtype=dtype)
        p = global_step_recomp / decay_step

        if self.staircase:
            p = tf.floor(p)
        iterations_bool = tf.cast(self.iterations < self.step_lim, tf.bool)
        lr = tf.cond(
            pred=iterations_bool,
            true_fn=lambda: tf.multiply(initial_learning_rate, tf.pow(hdecay_rate, p)),
            false_fn=lambda: tf.multiply(self.lr, tf.pow(ldecay_rate, 1 / decay_step)),
        )
        self.lr.assign(lr)
        return lr
