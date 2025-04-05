"""
All non-probablistic truth inference methods.
"""


# Classification
# TODO Implement Mode (Majority Vote, plurality vote)
# TODO Implement Majority Vote Honeypot

# Regression

def non_probablistic_truth_inference(regression=False, models=None, quantiles=None):
    """Runs all non-probablistic truth inference models

    Attributes
    ----------
    models : tuple, optional
        Set of the string identifiers of the non-probablistic truth inference
        models to exectute.
    quantiles : tuple
    """
    if models is None:
        if regression:
            models = {'mean', 'median', 'quantiles'}
        else: # Classification
            models = {'mode'}


    # TODO mean
    # TODO median and quantiles

def ELICE():
    """Implementation of Expert Label Injected Crowd Estimation. It uses
    accepted ground truth labels from "experts" to establish other annotator's
    quality.

    Parameters
    ----------

    Returns
    -------
    """
    return
