"""
All implemnetations of Dawid and Skene's EM algorithm, including the original,
hierarchial, and spectral.
"""

import math
import csv
import random
import sys

import numpy as np
from scipy.sparse import spmatrix # only necessary if the given sparse matrix need detected and handled carefully (ie. operations used do not exist for sparse matrices or would be inefficent.

class DawidSkene(object):
    """Original Dawid and Skene EM algorithm that estimates the true values of
    the labels from the annotation data. This is intendend for classification
    tasks only and would need modified to handle regression tasks
    (confusion matrices, marginal probabilites, etc. would all become densities
    and be changed in how they are applied).

    Attributes
    ----------
    annotations : array-like
        The observed data (X in wikipedia).
        The annotators' annotations.
    unobserved_data : array-like NOTE THAT THIS DOES NOT EXIST IN DAWID & SKENE????
        The unobserved data, missing values, or latent variables (Z in wikipedia).
        ? the bias, characteristics, etc of annotators?
        The confusion matrices of every annotator.
    confusion_matrices : array-like
        The confusion matrices of every annotator. Indexed by the k annotators,
        and then each matrix is a dense matrix of lxl where l is the possible
        label values.
    truth_inference_estimates: array-like
        The parameter_estimates vector of estimates for the unknown paramters (Theta in wikipedia).
        If given, then it serves as a prior.
        The truth inference of the samples.
    marginal_probabilities : array-like
        The marginal probabilities of the values of the annotation labels.
    likelihood_function : array-like
        The likelihood function L(Theta; X, Z) = p(X,Z|Theta)
    ground_truth : array-like, optional
        In the case that the ground truth is provided, it can be used to ...
    random_state : numpy.random.RandomState
        The random_state of this model. This will not be used if prior
        truth_inference_estimates are provided to model.
    """

    def __init__(self, annotations, truth_inference_estimates=None, ground_truth=None, random_state=None):
        """Initializes the Dawid and Skene Expectation Maximization algorithm.

        Parameters
        ----------
        """
        self.reinit(annotations, truth_inference_estimates, ground_truth, random_state)

    def reinit(self, annotations, truth_inference_estimates=None, ground_truth=None, random_state=None):
        """Reinitializes the Dawid and Skene Expectation Maximization algorithm

        Parameters
        ----------
        """
        # CHECKS:


        # get number of annotators from observed data
        #annotator_ids, annotator_annotations, indices  = np.unique(annotations, return_counts=True, return_index=True)
        #annotator_count = len(annotator_ids)
        # get number of annoations from an annotator per sample to handle
        # multi-labeling, also preserve order if possible.
        #for i, annotator in enumerate(annotator_ids):


        # TODO Initialize truth_inference_estimates.
        # discrete/classification only is:
        # np.empty(parameters_count, number_of_possible_label_values).fill(1/parameters_count)
        # should be able to do continuous values for regression by statement on EM...
        #self.truth_inference_estimates =   if truth_inference_estimates is None else truth_inference_estimates

    def fit(self, iterations, threshold=None):
        """Runs the EM algorithm for either a given number of iterations or
        until a threshold is met.

        Parameters
        ----------
        iterations : int, optional
            The explicit number of maximum iterations to perform. Will complete
            exactly this many iterations if threshold is not set and is not met
            prior to this maximum number of iterations.
        threshold : float, optional
            The threshold for the minimum amount of change in the estimated
            parameters that is acceptable for the algorithm to terminate.
        """
        iteration_count = 0
        previous_truth_inference_estimates = self.truth_inference_estimates
        parameter_difference = threshold + 1

        while iteration_count < iterations and (threshold is None or parameter_difference > threshold):
            self.expectation_step()
            self.maximization_step()

            iteration_count += 1

            if threshold is not None:
                parameter_difference = abs(previous_truth_inference_estimates)

    def calculate_threshold(self):
        """In the case EM continues until a threshold of minimal change is met."""
        # TODO, but probabl can be done in fit, itself by saving the prior truth_inference_estimates and comparing the difference to some threshold.
        return

    def expectation_step(self):
        """Update
            Compute the probability of each possible value of the unobserved data given the parameter estimates

        """
        # confusion matrix for every annotator: k_{j,l} = Sum_i(T_{i,j}n_{i,l})/Sum_i(Sum_j(T_{i,j}n_{i,l}))
        # marginal probabilities

    def maximization_step(self):
        """Compute better parameter estimates using the just computed unannotations
        """
