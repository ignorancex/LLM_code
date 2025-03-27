"""
The core code for applying Canonical Correlation Analysis to deep networks.

This module contains the core functions to apply canonical correlation analysis
to deep neural networks. The main function is get_cca_similarity, which takes in
two sets of activations, typically the neurons in two layers and their outputs
on all of the datapoints D = [d_1,...,d_m] that have been passed through.

Inputs have shape (num_neurons1, m), (num_neurons2, m). This can be directly
applied used on fully connected networks. For convolutional layers, the 3d block
of neurons can either be flattened entirely, along channels, or alternatively,
the dft_ccas (Discrete Fourier Transform) module can be used.

See:
https://arxiv.org/abs/1706.05806
https://arxiv.org/abs/1806.05759
for full details.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import math
import torch
from tool import unpack_task_to_sample, set_seed
import os
from options import Options
from regression import sinusoid
from img_classify import ImgClassification
from tqdm import tqdm
from matplotlib import pyplot as plt
from tool import Tensorboard_manager
from torch.nn import functional as F
from BP import BP
import matplotlib
import copy
import random

matplotlib.use('Agg')


class CCA():
    # Copyright 2018 Google Inc.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.

    """
    The core code for applying Canonical Correlation Analysis to deep networks.

    This module contains the core functions to apply canonical correlation analysis
    to deep neural networks. The main function is get_cca_similarity, which takes in
    two sets of activations, typically the neurons in two layers and their outputs
    on all of the datapoints D = [d_1,...,d_m] that have been passed through.

    Inputs have shape (num_neurons1, m), (num_neurons2, m). This can be directly
    applied used on fully connected networks. For convolutional layers, the 3d block
    of neurons can either be flattened entirely, along channels, or alternatively,
    the dft_ccas (Discrete Fourier Transform) module can be used.

    See:
    https://arxiv.org/abs/1706.05806
    https://arxiv.org/abs/1806.05759
    for full details.

    """

    num_cca_trials = 5

    def positivedef_matrix_sqrt(self, array):
        """Stable method for computing matrix square roots, supports complex matrices.

        Args:
                  array: A numpy 2d array, can be complex valued that is a positive
                         definite symmetric (or hermitian) matrix

        Returns:
                  sqrtarray: The matrix square root of array
        """
        w, v = np.linalg.eigh(array)
        #  A - np.dot(v, np.dot(np.diag(w), v.T))
        wsqrt = np.sqrt(w)
        sqrtarray = np.dot(v, np.dot(np.diag(wsqrt), np.conj(v).T))
        return sqrtarray

    def remove_small(self, sigma_xx, sigma_xy, sigma_yx, sigma_yy, epsilon):
        """Takes covariance between X, Y, and removes values of small magnitude.

        Args:
                  sigma_xx: 2d numpy array, variance matrix for x
                  sigma_xy: 2d numpy array, crossvariance matrix for x,y
                  sigma_yx: 2d numpy array, crossvariance matrixy for x,y,
                            (conjugate) transpose of sigma_xy
                  sigma_yy: 2d numpy array, variance matrix for y
                  epsilon : cutoff value for norm below which directions are thrown
                             away

        Returns:
                  sigma_xx_crop: 2d array with low x norm directions removed
                  sigma_xy_crop: 2d array with low x and y norm directions removed
                  sigma_yx_crop: 2d array with low x and y norm directiosn removed
                  sigma_yy_crop: 2d array with low y norm directions removed
                  x_idxs: indexes of sigma_xx that were removed
                  y_idxs: indexes of sigma_yy that were removed
        """

        x_diag = np.abs(np.diagonal(sigma_xx))
        y_diag = np.abs(np.diagonal(sigma_yy))
        x_idxs = (x_diag >= epsilon)
        y_idxs = (y_diag >= epsilon)

        sigma_xx_crop = sigma_xx[x_idxs][:, x_idxs]
        sigma_xy_crop = sigma_xy[x_idxs][:, y_idxs]
        sigma_yx_crop = sigma_yx[y_idxs][:, x_idxs]
        sigma_yy_crop = sigma_yy[y_idxs][:, y_idxs]

        return (sigma_xx_crop, sigma_xy_crop, sigma_yx_crop, sigma_yy_crop,
                x_idxs, y_idxs)

    def compute_ccas(self, sigma_xx, sigma_xy, sigma_yx, sigma_yy, epsilon,
                     verbose=False):
        """Main cca computation function, takes in variances and crossvariances.

        This function takes in the covariances and cross covariances of X, Y,
        preprocesses them (removing small magnitudes) and outputs the raw results of
        the cca computation, including cca directions in a rotated space, and the
        cca correlation coefficient values.

        Args:
                  sigma_xx: 2d numpy array, (num_neurons_x, num_neurons_x)
                            variance matrix for x
                  sigma_xy: 2d numpy array, (num_neurons_x, num_neurons_y)
                            crossvariance matrix for x,y
                  sigma_yx: 2d numpy array, (num_neurons_y, num_neurons_x)
                            crossvariance matrix for x,y (conj) transpose of sigma_xy
                  sigma_yy: 2d numpy array, (num_neurons_y, num_neurons_y)
                            variance matrix for y
                  epsilon:  small float to help with stabilizing computations
                  verbose:  boolean on whether to print intermediate outputs

        Returns:
                  [ux, sx, vx]: [numpy 2d array, numpy 1d array, numpy 2d array]
                                ux and vx are (conj) transposes of each other, being
                                the canonical directions in the X subspace.
                                sx is the set of canonical correlation coefficients-
                                how well corresponding directions in vx, Vy correlate
                                with each other.
                  [uy, sy, vy]: Same as above, but for Y space
                  invsqrt_xx:   Inverse square root of sigma_xx to transform canonical
                                directions back to original space
                  invsqrt_yy:   Same as above but for sigma_yy
                  x_idxs:       The indexes of the input sigma_xx that were pruned
                                by remove_small
                  y_idxs:       Same as above but for sigma_yy
        """

        (sigma_xx, sigma_xy, sigma_yx, sigma_yy,
         x_idxs, y_idxs) = self.remove_small(sigma_xx, sigma_xy, sigma_yx, sigma_yy, epsilon)

        numx = sigma_xx.shape[0]
        numy = sigma_yy.shape[0]

        if numx == 0 or numy == 0:
            return ([0, 0, 0], [0, 0, 0], np.zeros_like(sigma_xx),
                    np.zeros_like(sigma_yy), x_idxs, y_idxs)

        if verbose:
            print("adding eps to diagonal and taking inverse")
        sigma_xx += epsilon * np.eye(numx)
        sigma_yy += epsilon * np.eye(numy)
        inv_xx = np.linalg.pinv(sigma_xx)
        inv_yy = np.linalg.pinv(sigma_yy)

        if verbose:
            print("taking square root")

        invsqrt_xx = self.positivedef_matrix_sqrt(inv_xx)
        invsqrt_yy = self.positivedef_matrix_sqrt(inv_yy)

        if verbose:
            print("dot products...")

        arr = np.dot(invsqrt_xx, np.dot(sigma_xy, invsqrt_yy))

        if verbose:
            print("trying to take final svd")
        u, s, v = np.linalg.svd(arr)

        if verbose:
            print("computed everything!")

        return [u, np.abs(s), v], invsqrt_xx, invsqrt_yy, x_idxs, y_idxs

    def sum_threshold(self, array, threshold):
        """Computes threshold index of decreasing nonnegative array by summing.

        This function takes in a decreasing array nonnegative floats, and a
        threshold between 0 and 1. It returns the index i at which the sum of the
        array up to i is threshold*total mass of the array.

        Args:
                  array: a 1d numpy array of decreasing, nonnegative floats
                  threshold: a number between 0 and 1

        Returns:
                  i: index at which np.sum(array[:i]) >= threshold
        """
        assert (threshold >= 0) and (threshold <= 1), "print incorrect threshold"

        for i in range(len(array)):
            if np.sum(array[:i]) / np.sum(array) >= threshold:
                return i

    def create_zero_dict(self, compute_dirns, dimension):
        """Outputs a zero dict when neuron activation norms too small.

        This function creates a return_dict with appropriately shaped zero entries
        when all neuron activations are very small.

        Args:
                  compute_dirns: boolean, whether to have zero vectors for directions
                  dimension: int, defines shape of directions

        Returns:
                  return_dict: a dict of appropriately shaped zero entries
        """
        return_dict = {}
        return_dict["mean"] = (np.asarray(0), np.asarray(0))
        return_dict["sum"] = (np.asarray(0), np.asarray(0))
        return_dict["cca_coef1"] = np.asarray(0)
        return_dict["cca_coef2"] = np.asarray(0)
        return_dict["idx1"] = 0
        return_dict["idx2"] = 0

        if compute_dirns:
            return_dict["cca_dirns1"] = np.zeros((1, dimension))
            return_dict["cca_dirns2"] = np.zeros((1, dimension))

        return return_dict

    def get_cca_similarity(self, acts1, acts2, epsilon=1e-6, threshold=0.98,
                           compute_coefs=True,
                           compute_dirns=False,
                           verbose=False):
        """The main function for computing cca similarities.

        This function computes the cca similarity between two sets of activations,
        returning a dict with the cca coefficients, a few statistics of the cca
        coefficients, and (optionally) the actual directions.

        Args:
                  acts1: (num_neurons1, data_points) a 2d numpy array of neurons by
                         datapoints where entry (i,j) is the output of neuron i on
                         datapoint j.
                  acts2: (num_neurons2, data_points) same as above, but (potentially)
                         for a different set of neurons. Note that acts1 and acts2
                         can have different numbers of neurons, but must agree on the
                         number of datapoints

                  epsilon: small float to help stabilize computations

                  threshold: float between 0, 1 used to get rid of trailing zeros in
                             the cca correlation coefficients to output more accurate
                             summary statistics of correlations.


                  compute_coefs: boolean value determining whether coefficients
                                 over neurons are computed. Needed for computing
                                 directions

                  compute_dirns: boolean value determining whether actual cca
                                 directions are computed. (For very large neurons and
                                 datasets, may be better to compute these on the fly
                                 instead of store in memory.)

                  verbose: Boolean, whether intermediate outputs are printed

        Returns:
                  return_dict: A dictionary with outputs from the cca computations.
                               Contains neuron coefficients (combinations of neurons
                               that correspond to cca directions), the cca correlation
                               coefficients (how well aligned directions correlate),
                               x and y idxs (for computing cca directions on the fly
                               if compute_dirns=False), and summary statistics. If
                               compute_dirns=True, the cca directions are also
                               computed.
        """

        # assert dimensionality equal
        # acts1 = acts1[:acts1.shape[-1] - 1, :]
        # acts2 = acts2[:acts2.shape[-1] - 1, :]

        assert acts1.shape[1] == acts2.shape[1], "dimensions don't match"
        # # check that acts1, acts2 are transposition
        # assert acts1.shape[0] < acts1.shape[1], ("input must be number of neurons"
        #                                          "by datapoints")

        return_dict = {}

        # compute covariance with numpy function for extra stability
        numx = acts1.shape[0]
        numy = acts2.shape[0]

        covariance = np.cov(acts1, acts2)
        sigmaxx = covariance[:numx, :numx]
        sigmaxy = covariance[:numx, numx:]
        sigmayx = covariance[numx:, :numx]
        sigmayy = covariance[numx:, numx:]

        # rescale covariance to make cca computation more stable
        xmax = np.max(np.abs(sigmaxx))
        ymax = np.max(np.abs(sigmayy))
        sigmaxx /= xmax
        sigmayy /= ymax
        sigmaxy /= np.sqrt(xmax * ymax)
        sigmayx /= np.sqrt(xmax * ymax)

        ([u, s, v], invsqrt_xx, invsqrt_yy,
         x_idxs, y_idxs) = self.compute_ccas(sigmaxx, sigmaxy, sigmayx, sigmayy,
                                             epsilon=epsilon,
                                             verbose=verbose)

        # if x_idxs or y_idxs is all false, return_dict has zero entries
        if (not np.any(x_idxs)) or (not np.any(y_idxs)):
            return self.create_zero_dict(compute_dirns, acts1.shape[1])

        if compute_coefs:
            # also compute full coefficients over all neurons
            x_mask = np.dot(x_idxs.reshape((-1, 1)), x_idxs.reshape((1, -1)))
            y_mask = np.dot(y_idxs.reshape((-1, 1)), y_idxs.reshape((1, -1)))

            return_dict["coef_x"] = u.T
            return_dict["invsqrt_xx"] = invsqrt_xx
            return_dict["full_coef_x"] = np.zeros((numx, numx))
            np.place(return_dict["full_coef_x"], x_mask,
                     return_dict["coef_x"])
            return_dict["full_invsqrt_xx"] = np.zeros((numx, numx))
            np.place(return_dict["full_invsqrt_xx"], x_mask,
                     return_dict["invsqrt_xx"])

            return_dict["coef_y"] = v
            return_dict["invsqrt_yy"] = invsqrt_yy
            return_dict["full_coef_y"] = np.zeros((numy, numy))
            np.place(return_dict["full_coef_y"], y_mask,
                     return_dict["coef_y"])
            return_dict["full_invsqrt_yy"] = np.zeros((numy, numy))
            np.place(return_dict["full_invsqrt_yy"], y_mask,
                     return_dict["invsqrt_yy"])

            # compute means
            neuron_means1 = np.mean(acts1, axis=1, keepdims=True)
            neuron_means2 = np.mean(acts2, axis=1, keepdims=True)
            return_dict["neuron_means1"] = neuron_means1
            return_dict["neuron_means2"] = neuron_means2

        if compute_dirns:
            # orthonormal directions that are CCA directions
            cca_dirns1 = np.dot(np.dot(return_dict["full_coef_x"],
                                       return_dict["full_invsqrt_xx"]),
                                (acts1 - neuron_means1)) + neuron_means1
            cca_dirns2 = np.dot(np.dot(return_dict["full_coef_y"],
                                       return_dict["full_invsqrt_yy"]),
                                (acts2 - neuron_means2)) + neuron_means2

        # get rid of trailing zeros in the cca coefficients
        idx1 = self.sum_threshold(s, threshold)
        idx2 = self.sum_threshold(s, threshold)

        return_dict["cca_coef1"] = s
        return_dict["cca_coef2"] = s
        return_dict["x_idxs"] = x_idxs
        return_dict["y_idxs"] = y_idxs
        # summary statistics
        return_dict["mean"] = (np.mean(s[:idx1]), np.mean(s[:idx2]))
        return_dict["sum"] = (np.sum(s), np.sum(s))

        if compute_dirns:
            return_dict["cca_dirns1"] = cca_dirns1
            return_dict["cca_dirns2"] = cca_dirns2

        return return_dict

    def robust_cca_similarity(self, acts1, acts2, threshold=0.98, epsilon=1e-6,
                              compute_dirns=True):
        """Calls get_cca_similarity multiple times while adding noise.

        This function is very similar to get_cca_similarity, and can be used if
        get_cca_similarity doesn't converge for some pair of inputs. This function
        adds some noise to the activations to help convergence.

        Args:
                  acts1: (num_neurons1, data_points) a 2d numpy array of neurons by
                         datapoints where entry (i,j) is the output of neuron i on
                         datapoint j.
                  acts2: (num_neurons2, data_points) same as above, but (potentially)
                         for a different set of neurons. Note that acts1 and acts2
                         can have different numbers of neurons, but must agree on the
                         number of datapoints

                  threshold: float between 0, 1 used to get rid of trailing zeros in
                             the cca correlation coefficients to output more accurate
                             summary statistics of correlations.

                  epsilon: small float to help stabilize computations

                  compute_dirns: boolean value determining whether actual cca
                                 directions are computed. (For very large neurons and
                                 datasets, may be better to compute these on the fly
                                 instead of store in memory.)

        Returns:
                  return_dict: A dictionary with outputs from the cca computations.
                               Contains neuron coefficients (combinations of neurons
                               that correspond to cca directions), the cca correlation
                               coefficients (how well aligned directions correlate),
                               x and y idxs (for computing cca directions on the fly
                               if compute_dirns=False), and summary statistics. If
                               compute_dirns=True, the cca directions are also
                               computed.
        """
        num_cca_trials = 3
        for trial in range(num_cca_trials):
            try:
                return_dict = self.get_cca_similarity(acts1, acts2, threshold, compute_dirns)
                break
            except np.linalg.LinAlgError:
                print(1111111111111111111111111111111111111)
                acts1 = acts1 * 1e-1 + np.random.normal(size=acts1.shape) * epsilon
                acts2 = acts2 * 1e-1 + np.random.normal(size=acts1.shape) * epsilon
                if trial + 1 == num_cca_trials:
                    raise

        return return_dict

class CKA():

    def centering(self, K):
        n = K.shape[0]
        unit = np.ones([n, n])
        I = np.eye(n)
        H = I - unit / n

        return np.dot(np.dot(H, K),
                      H)  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
        # return np.dot(H, K)  # KH

    def rbf(self, X, sigma=None):
        GX = np.dot(X, X.T)
        KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
        if sigma is None:
            mdist = np.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = np.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return np.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = np.dot(X, X.T)
        L_Y = np.dot(Y, Y.T)
        return np.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = np.sqrt(self.linear_HSIC(X, X))
        var2 = np.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = np.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = np.sqrt(self.kernel_HSIC(Y, Y, sigma))

        return hsic / (var1 * var2)


def cal_parameter_distance(x, y):
    tensor_dim = len(x.shape)

    parameter_num = 1
    for i in range(tensor_dim):
        parameter_num = parameter_num * x.shape[i]
    return torch.norm(x - y) / math.sqrt(parameter_num)


def get_act(args, epoch, device, x):
    model = torch.load(os.path.join(args.model_dir, args.scheme_name + '_' + str(epoch) + '.pth'), map_location=device)
    act_list = model.net.forward_with_record(x)

    return act_list


def get_analyse_data(args):
    if args.dataset == 'sinusoid':
        dataset_object = sinusoid(args)
        x_spt, y_spt, x_qry, y_qry = dataset_object.next_uniform('train')
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(args.device, dtype=torch.float32), torch.from_numpy(
            y_spt).to(args.device, dtype=torch.float32), \
                                     torch.from_numpy(x_qry).to(args.device, dtype=torch.float32), torch.from_numpy(
            y_qry).to(args.device, dtype=torch.float32)
        x_train, y_train = unpack_task_to_sample(x_spt, y_spt, x_qry, y_qry)

        x_spt, y_spt, x_qry, y_qry = dataset_object.next_uniform('test')
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(args.device, dtype=torch.float32), torch.from_numpy(
            y_spt).to(args.device, dtype=torch.float32), \
                                     torch.from_numpy(x_qry).to(args.device, dtype=torch.float32), torch.from_numpy(
            y_qry).to(args.device, dtype=torch.float32)
        x_test, y_test = unpack_task_to_sample(x_spt, y_spt, x_qry, y_qry)
    else:
        args.k_spt = 20
        dataset_object = ImgClassification(args)

        x_train, y_train = dataset_object.next_single_task('train', 10)
        x_train, y_train = torch.from_numpy(x_train).to(args.device), torch.from_numpy(y_train).to(args.device,
                                                                                                   dtype=torch.int64)
        x_test = []

    return x_train, x_test


def get_CCA_curve(dataset='sinusoid', scheme=''):
    device = torch.device("cuda:0")
    cca = CCA()

    # scheme=["ANIL_4_layer_3_head_layer_", "MAML_4_layer_", "BP_4_layer_"]
    for seed in range(1, 6):

        # scheme_name = scheme + str(seed) + '_seed'
        scheme_name = scheme
        args = Options().parse()
        try:
            args.__dict__ = np.load(os.path.join('./output/model', dataset, scheme_name,
                                                 scheme_name + '_config.npy', ), allow_pickle=True)[()]
        except:
            break
        args.device = device
        set_seed(args.seed)
        x_train, x_test = get_analyse_data(args)
        analyse_data = {"train": x_train, "test": x_test}
        # train
        for train_or_test in ["train"]:
            result_collector = [[], [], [], [], []]
            print("process: ", scheme_name)

            for epoch in tqdm(range(0, args.epoch, 100)):
                act_list_1 = get_act(args, epoch, device, analyse_data[train_or_test])
                act_list_2 = get_act(args, epoch + 100, device, analyse_data[train_or_test])

                for i, (act1, act2) in enumerate(zip(act_list_1, act_list_2)):
                    act1 = act1.detach().cpu().numpy()
                    act2 = act2.detach().cpu().numpy()

                    if len(act1.shape) > 2:
                        # act1 = np.transpose(act1, (1, 0, 2, 3))
                        # act1 = act1.reshape(act1.shape[0], act1.shape[1] * act1.shape[2] * act1.shape[3])
                        # act1 = act1.T
                        act1 = act1.reshape((-1, act1.shape[-1]))

                    if len(act2.shape) > 2:
                        # act2 = np.transpose(act2, (1, 0, 2, 3))
                        # act2 = act2.reshape(act2.shape[0], act2.shape[1] * act2.shape[2] * act2.shape[3])
                        # act2 = act2.T

                        act2 = act2.reshape((-1, act2.shape[-1]))

                    cca_result = cca.get_cca_similarity(act1.T, act2.T)
                    # print(cca_result["cca_coef1"])
                    result_collector[i].append(np.mean(cca_result["cca_coef1"]))

            for i in range(len(result_collector)):
                plt.plot(result_collector[i], label="layer " + str(i))
            np.save(
                './output/representation_analyse/' + args.dataset + '/cca_' + scheme_name + '_' + train_or_test + '.npy',
                result_collector)
            plt.legend()
            plt.savefig(
                './output/representation_analyse/' + args.dataset + '/cca_' + scheme_name + '_' + train_or_test + '.png')
            plt.show()
        break


def get_CCA_final_curve(dataset='sinusoid', scheme=''):
    device = torch.device("cuda:5")
    cca = CCA()

    for seed in range(1, 6):
        scheme_name = scheme + str(seed) + '_seed'
        args = Options().parse()
        try:
            args.__dict__ = np.load(os.path.join('./output/model/', dataset, scheme_name,
                                                 scheme_name + '_config.npy'), allow_pickle=True)[()]
        except:
            break
        args.device = device
        set_seed(args.seed)
        x_train, x_test = get_analyse_data(args)
        analyse_data = {"train": x_train, "test": x_test}
        # train
        for train_or_test in ["train"]:
            result_collector = [[], [], [], [], []]
            print("process: ", scheme_name)
            act_list_2 = get_act(args, args.epoch, device, analyse_data[train_or_test])
            for epoch in tqdm(range(0, args.epoch, 100)):
                act_list_1 = get_act(args, epoch, device, analyse_data[train_or_test])
                for i, (act1, act2) in enumerate(zip(act_list_1, act_list_2)):
                    act1 = act1.detach().cpu().numpy()
                    act2 = act2.detach().cpu().numpy()
                    cca_result = cca.get_cca_similarity(act1.T, act2.T)

                    result_collector[i].append(np.mean(cca_result["cca_coef1"]))
            for i in range(len(result_collector)):
                plt.plot(result_collector[i], label="layer " + str(i))
            np.save(
                './output/representation_analyse/' + args.dataset + '/cca_' + scheme_name + '_' + train_or_test + '.npy',
                result_collector)
            plt.legend()
            plt.savefig(
                './output/representation_analyse/' + args.dataset + '/cca_' + scheme_name + '_' + train_or_test + '.png')
            plt.show()


def get_CKA_curve(dataset='sinusoid', scheme=''):
    device = torch.device("cuda:0")
    cka = CKA()

    for seed in range(1, 6):

        scheme_name = scheme + str(seed) + '_seed'
        args = Options().parse()
        args.__dict__ = np.load(os.path.join('./output/model/', dataset, scheme_name,
                                             scheme_name + '_config.npy'), allow_pickle=True)[()]
        args.device = device
        set_seed(args.seed)
        x_train, x_test = get_analyse_data(args)
        analyse_data = {"train": x_train, "test": x_test}
        # train
        for train_or_test in ["train"]:
            result_collector = [[], [], [], [], []]
            print("process: ", scheme_name)
            for epoch in tqdm(range(0, args.epoch - 100, 100)):
                act_list_1 = get_act(args, epoch, device, analyse_data[train_or_test])
                act_list_2 = get_act(args, epoch + 100, device, analyse_data[train_or_test])

                for i, (act1, act2) in enumerate(zip(act_list_1, act_list_2)):
                    act1 = act1.detach().cpu().numpy()
                    act2 = act2.detach().cpu().numpy()
                    cka_result = cka.linear_CKA(act1, act2)

                    result_collector[i].append(cka_result)

            for i in range(len(result_collector)):
                plt.plot(result_collector[i], label="layer " + str(i))
            np.save(
                './output/representation_analyse/' + args.dataset + '/cka_' + scheme_name + '_' + train_or_test + '.npy',
                result_collector)
            plt.legend()
            plt.title(scheme + train_or_test)
            plt.savefig(
                './output/representation_analyse/' + args.dataset + '/cka_' + scheme_name + '_' + train_or_test + '.png')
            plt.show()


def get_grad_histogram(dataset='sinusoid', scheme=''):
    for seed in range(1, 6):
        scheme_name = scheme + str(seed) + '_seed'
        tb_manager = Tensorboard_manager('./output/representation_analyse/' + dataset + '/' + scheme_name)

        print("process: ", scheme_name)
        for epoch in tqdm(range(0, args.epoch, 100)):
            grad_list = np.load(os.path.join('./output/model/', dataset, scheme_name,
                                             scheme_name + '_grad_' + str(epoch) + '.npy'), allow_pickle=True)
            for i, parameter in enumerate(grad_list):
                tb_manager.summary_writer.add_histogram('grad_' + str(i) + 'th_parameter', parameter, epoch)


def get_parameter_distance_curve(dataset='sinusoid', scheme=''):
    device = torch.device("cuda:0")

    for seed in range(1, 6):

        scheme_name = scheme + str(seed) + '_seed'
        args = Options().parse()
        try:
            args.__dict__ = np.load(os.path.join('output/model/', dataset, scheme_name,
                                                 scheme_name + '_config.npy'), allow_pickle=True)[()]
        except:
            break
        args.device = device
        set_seed(args.seed)

        result_collector = [[], [], [], [], []]
        print("process: ", scheme_name)
        for epoch in tqdm(range(0, args.epoch - 100, 100)):
            parameter_list_1 = torch.load(
                os.path.join(args.model_dir, args.scheme_name + '_' + str(epoch) + '.pth'),
                map_location=device).net.parameters()
            parameter_list_2 = torch.load(
                os.path.join(args.model_dir, args.scheme_name + '_' + str(epoch + 100) + '.pth'),
                map_location=device).net.parameters()

            for i, (parameter1, parameter2) in enumerate(zip(parameter_list_1, parameter_list_2)):
                if i % 2 == 1:
                    continue
                distance = cal_parameter_distance(parameter1, parameter2)
                result_collector[int(i / 2)].append(distance)

        for i in range(len(result_collector)):
            plt.plot(result_collector[i], label=str(i) + 'th_parameter')
        np.save(
            './output/representation_analyse/' + args.dataset + '/parameter_dist_' + scheme_name + '_train' + '.npy',
            result_collector)
        plt.legend()
        plt.title(scheme_name)
        plt.savefig(
            './output/representation_analyse/' + args.dataset + '/parameter_dist_' + scheme_name + '_train' + '.png')
        plt.show()


def get_final_parameter_distance_curve(dataset='sinusoid', scheme=''):
    device = torch.device("cuda:5")

    for seed in range(1, 6):

        scheme_name = scheme + str(seed) + '_seed'
        args = Options().parse()
        try:
            args.__dict__ = np.load(os.path.join('./output/model/', dataset, scheme_name,
                                                 scheme_name + '_config.npy'), allow_pickle=True)[()]
        except:
            break
        args.device = device
        set_seed(args.seed)

        result_collector = [[], [], [], [], []]
        print("process: ", scheme_name)
        parameter_list_1 = torch.load(
            os.path.join(args.model_dir, args.scheme_name + '_60000.pth'),
            map_location=device).net.parameters()
        for epoch in tqdm(range(0, args.epoch - 100, 100)):

            parameter_list_2 = torch.load(
                os.path.join(args.model_dir, args.scheme_name + '_' + str(epoch + 100) + '.pth'),
                map_location=device).net.parameters()

            for i, (parameter1, parameter2) in enumerate(zip(parameter_list_1, parameter_list_2)):
                if i % 2 == 1:
                    continue
                distance = cal_parameter_distance(parameter1, parameter2)
                result_collector[int(i / 2)].append(distance)

        for i in range(len(result_collector)):
            plt.plot(result_collector[i], label=str(i) + 'th_parameter')
        np.save(
            './output/representation_analyse/' + args.dataset + '/final_parameter_dist_' + scheme_name + '_train' + '.npy',
            result_collector)
        plt.legend()
        plt.title(scheme_name)
        plt.savefig(
            './output/representation_analyse/' + args.dataset + '/final_parameter_dist_' + scheme_name + '_train' + '.png')
        plt.show()


def get_mean_curve(dataset='sinusoid', curve='parameter_dist', scheme=''):
    for train_or_test in ['train']:
        cca_result_collector_seed_list = []
        for seed in range(2, 6):
            scheme_name = scheme + str(seed) + '_seed'
            args = Options().parse()
            try:
                args.__dict__ = np.load(os.path.join('output/model/', dataset, scheme_name,
                                                     scheme_name + '_config.npy'), allow_pickle=True)[()]
            except:
                break
            cca_result_collector = np.load(
                './output/representation_analyse/' + args.dataset + '/' + curve + '_' + scheme_name + '_' + train_or_test + '.npy',
                allow_pickle=True)
            cca_result_collector_seed_list.append(cca_result_collector)

        result_average_seed = np.mean(cca_result_collector_seed_list, axis=0)

        for i in range(len(result_average_seed)):
            reduce_curve = result_average_seed[i][range(0, len(result_average_seed[i]), 5)]
            plt.plot(reduce_curve, label="layer " + str(i), alpha=0.75)

        if curve == 'parameter_dist':
            if 'BP' in scheme:
                plt.ylim(0, 0.008)
            else:
                plt.ylim(0, 0.008)
        plt.title(curve + '_' + scheme + train_or_test)
        plt.legend()
        plt.savefig(
            './output/representation_analyse/' + args.dataset + '/mean_' + curve + '_' + scheme + train_or_test + '.pdf')
        plt.show()


def get_norm_grad_var(dataset='sinusoid', scheme=''):
    def norm(matrix):

        matrix = matrix.view(-1, )
        # sum = torch.norm(matrix, p=2)

        return matrix

    def loss_func(y_pre, y_true, dataset):
        if dataset == 'sinusoid':
            y_true = y_true.view(y_true.size(0), -1)
            return F.mse_loss(y_pre, y_true)
        else:
            return F.cross_entropy(y_pre, y_true)

    def multi_loss_func(y_pre, y_true):
        y_true = y_true.view(y_true.size(0), -1)
        return torch.pow(y_pre - y_true, 2)

    def task_learning(net, x_spt, y_spt, x_qry, y_qry, bottom_lr, bottom_step_num,
                      dataset):

        # the following update
        fast_weights = net.parameters()
        for k in range(0, bottom_step_num):
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = loss_func(logits, y_spt, dataset)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - bottom_lr * p[0], zip(grad, fast_weights)))

        logits_q = net(x_qry, fast_weights, bn_training=True)
        loss_q = multi_loss_func(logits_q, y_qry)
        return loss_q

    args = Options().parse()  # 使用默认参数

    epoch_seed_trace = []
    for epoch in tqdm(range(0, args.epoch, 100)):
        epoch_seed_trace.append([])
        for seed in range(1, 6):
            epoch_seed_trace[-1].append([])
            args.seed = seed + 100  # 不要测已经训练过的数据
            if 'BP' in scheme:
                args.bottom_step_num = 0
            dataset_object = sinusoid(args)
            x_spt, y_spt, x_qry, y_qry = dataset_object.next_uniform('train')
            x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(args.device,
                                                                    dtype=torch.float32), torch.from_numpy(
                y_spt).to(args.device, dtype=torch.float32), \
                                         torch.from_numpy(x_qry).to(args.device,
                                                                    dtype=torch.float32), torch.from_numpy(
                y_qry).to(args.device, dtype=torch.float32)

            scheme_1 = scheme + '_' + str(seed) + '_seed'
            method_1 = torch.load(
                os.path.join('./output/model/sinusoid/', scheme_1, scheme_1 + '_' + str(epoch) + '.pth'),
                map_location=args.device)

            loss_q = []
            for task in range(args.task_num):
                loss_q.extend(
                    task_learning(method_1.net, x_spt[task], y_spt[task], x_qry[task], y_qry[task], args.bottom_lr,
                                  args.bottom_step_num, 'sinusoid'))

            split_norm_grad_list = []
            for sampe_loss in loss_q:
                method_1.meta_optim.zero_grad()
                grad_list = torch.autograd.grad(sampe_loss, method_1.net.parameters(), retain_graph=True)
                tmp_grad_list = []
                for grad in grad_list:
                    tmp_grad_list.append(norm(grad))
                split_norm_grad_list.append(tmp_grad_list)

            split_norm_grad_list = np.asarray(split_norm_grad_list)

            for i, layer in enumerate(split_norm_grad_list[0]):
                if i == 9:
                    break
                tmp = torch.stack(list(split_norm_grad_list[:, i]))
                mean_vector = torch.mean(tmp, dim=0)

                cosine_similarities = F.cosine_similarity(tmp, mean_vector.unsqueeze(0), dim=1)
                trace = torch.var(cosine_similarities).cpu().numpy()

                epoch_seed_trace[-1][-1].append(trace)

    epoch_seed_trace = np.asarray(epoch_seed_trace)

    np.save('./output/representation_analyse/sinusoid/' + scheme.split('_', -1)[0] + '_direct_var.npy',
            epoch_seed_trace)


def get_grad_curve(dataset='sinusoid', scheme=''):
    grad_seed_epoch = []
    for seed in range(1, 6):
        grad_seed_epoch.append([])
        scheme_name = scheme + '_' + str(seed) + '_seed'
        for epoch in tqdm(range(0, args.epoch, 100)):
            grad_seed_epoch[-1].append([])
            grad_list = np.load(os.path.join('./output/model/', dataset, scheme_name,
                                             scheme_name + '_grad_' + str(epoch) + '.npy'), allow_pickle=True)
            for i, parameter in enumerate(grad_list):
                grad = torch.norm(parameter.view(-1)).cpu().numpy()
                grad_seed_epoch[-1][-1].append(grad)

    np.save('./output/representation_analyse/sinusoid/' + scheme.split('_', -1)[0] + '_grad_norm.npy', grad_seed_epoch)


def get_combine_model():
    def cal_cca(model1, model2, data):
        cca = CCA()
        act_list_1 = model1.net.forward_with_record(data)
        act_list_2 = model2.net.forward_with_record(data)
        ans = []
        for i, (act1, act2) in enumerate(zip(act_list_1, act_list_2)):
            act1 = act1.detach().cpu().numpy()
            act2 = act2.detach().cpu().numpy()
            cca_result = cca.get_cca_similarity(act1.T, act2.T)
            ans.append(np.mean(cca_result["cca_coef1"]))
        return ans

    scheme_name = []
    for amp in [1, 3, 5, 7]:
        for pha in [0.0, 0.2, 0.4, 0.6, 0.8]:
            scheme_name.append('one_task_BP_' + str(amp) + '_a_' + str(pha) + '_p_1_seed')

    random.shuffle(scheme_name)
    device = torch.device("cuda:0")
    for outter_index, major_scheme in enumerate(scheme_name):
        args = Options().parse()
        args.__dict__ = np.load(os.path.join('./output/model/', 'sinusoid', major_scheme,
                                             major_scheme + '_config.npy'), allow_pickle=True)[()]

        args.device = device
        set_seed(args.seed)
        x_train, x_test = get_analyse_data(args)

        analyse_data = {"train": x_train}
        model_zero = BP(args).to(args.device)

        model_final = torch.load(os.path.join(args.model_dir, args.scheme_name + '_' + str(args.epoch) + '.pth'),
                                 map_location=device)

        trajectory_list = []
        ans_list = []
        ans_list.append(cal_cca(model_zero, model_final, analyse_data['train']))
        for inner_index, add_scheme in tqdm(enumerate(scheme_name)):
            add_scheme = scheme_name[(outter_index + inner_index) % len(scheme_name)]
            model_tmp = copy.deepcopy(model_zero)
            args.__dict__ = np.load(os.path.join('./output/model/', 'sinusoid', add_scheme,
                                                 add_scheme + '_config.npy'), allow_pickle=True)[()]
            model_add = torch.load(os.path.join(args.model_dir, args.scheme_name + '_' + str(args.epoch) + '.pth'),
                                   map_location=device)

            trajectory = [row2 - row1 for row1, row2 in
                          zip(model_zero.net.parameters(), model_add.net.parameters())]
            # trajectory_list.append(trajectory)
            #
            # with torch.no_grad():
            #     for i in range(len(trajectory_list)):
            #         for j in range(len(trajectory_list[i])):
            #             model_tmp.net.vars[j] += trajectory_list[i][j].detach() / len(trajectory_list)

            with torch.no_grad():
                for i in range(len(trajectory)):
                    model_tmp.net.vars[i] += trajectory[i].detach() / len(scheme_name)

            ans_list.append(cal_cca(model_tmp, model_final, analyse_data['train']))

        ans_list = np.asarray(ans_list).T
        for i in range(len(ans_list)):
            plt.plot(ans_list[i], label='layer ' + str(i))

        plt.legend()
        plt.show()


if __name__ == '__main__':
    # get_grad_curve(scheme="ANIL_vary_amp_0.1_10_amp_")
    # print(cal_task_conflict(30))
    # for method in ['MAML','BP']:
    #     for pha in ['0','0.2','0.4','0.6','0.8','1']:
    #         scheme=method+'_vary_pha_'+pha+'_pha_'
    #         get_CCA_curve(scheme=scheme)
    #         get_parameter_distance_curve(scheme=scheme)
    # get_final_parameter_distance_curve(scheme=scheme)
    with torch.no_grad():
        get_CCA_curve(dataset='omniglot', scheme='BP_20_way_shuffle_label')
    # get_mean_curve(curve='cca', scheme='ANIL_novary_2_amp_0_pha_')
    # get_mean_curve(curve='parameter_dist', scheme=scheme)
    # get_mean_curve(curve='final_parameter_dist', scheme=scheme)

    # get_CCA_final_curve(scheme='one_task_BP_5_a_0.2_p_')
    # get_combine_model()
