# imports go here
import numpy as np


# functions and classes go here
def fb_alg(A_mat, O_mat, observ):
    # set up
    k = observ.size
    (n, m) = O_mat.shape
    prob_mat = np.zeros((n, k))
    fw = np.zeros((n, k + 1))
    bw = np.zeros((n, k + 1))
    # forward part
    fw[:, 0] = 1.0 / n
    for obs_ind in xrange(k):
        f_row_vec = np.matrix(fw[:, obs_ind])
        fw[:, obs_ind + 1] = f_row_vec * np.matrix(A_mat) * np.matrix(np.diag(O_mat[:, int(observ[obs_ind])]))
        fw[:, obs_ind + 1] = fw[:, obs_ind + 1] / np.sum(fw[:, obs_ind + 1])
    # backward part
    bw[:, -1] = 1.0
    for obs_ind in xrange(k, 0, -1):
        b_col_vec = np.matrix(bw[:, obs_ind]).transpose()
        bw[:, obs_ind - 1] = (np.matrix(A_mat) *np.matrix(np.diag(O_mat[:, int(observ[obs_ind - 1])])) * b_col_vec).transpose()
        bw[:, obs_ind - 1] = bw[:, obs_ind - 1] / np.sum(bw[:, obs_ind - 1])
    # combine it
    prob_mat = np.array(fw) * np.array(bw)
    prob_mat = prob_mat / np.sum(prob_mat, 0)
    # get out
    return prob_mat, fw, bw


# main script stuff goes here
if __name__ == '__main__':
    # the transition matrix
    A_mat = np.array([[.6, .4], [.2, .8]])
    # the observation matrix
    O_mat = np.array([[.5, .5], [.15, .85]])
    # sample heads or tails, 0 is heads, 1 is tails
    num_obs = 15
    observations1 = np.random.randn(num_obs)
    observations1[observations1 > 0] = 1
    observations1[observations1 <= 0] = 0
    p, f, b = fb_alg(A_mat, O_mat, observations1)
    print p
    # change observations to reflect messed up ratio
    observations2 = np.random.random(num_obs)
    observations2[observations2 > .85] = 0
    observations2[observations2 <= .85] = 1
    # majority of the time its tails, now what?
    p, f, b = fb_alg(A_mat, O_mat, observations1)
    print p
    p, f, b = fb_alg(A_mat, O_mat, np.hstack((observations1, observations2)))
    print p