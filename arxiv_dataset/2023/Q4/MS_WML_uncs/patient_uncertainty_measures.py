import numpy as np
import pandas as pd
from .lesion_uncertainty_measures import intersection_over_union
from sklearn import metrics
from .voxel_uncertainty_measures import ensemble_uncertainties_classification
from .transforms import process_probs


def global_scale_measures(ens_pred_probs: np.ndarray, ens_prob_thresh: float, mem_prob_threshs: list, l_min: int):
    """
    Measures not based on uncertainties aggregation from the lower scales: proposed PSU and PSU^{+}
    :param ens_pred_probs: probability maps for all models in ensemble, shape [M, H, W, D]
    :param ens_prob_thresh: probability threshold for the ensemble model
    :param mem_prob_threshols: list of the probability thresholds for each model in ensemble len = M
    :param l_min: minimal size of the connected component needed for the postprocessing
    :return: dictionary with computed uncertainty measures
    """
    
    result = dict()
    # Segmentation-overlap-based measures
    ens_seg_mask = process_probs(np.mean(ens_pred_probs, axis=0), ens_prob_thresh, l_min)
    # 1: members undergo the same threshold
    mem_seg_masks = np.stack([process_probs(epp, ens_prob_thresh, l_min) for epp
                              in ens_pred_probs], axis=0)
    result['PSU'] = 1 - np.mean([intersection_over_union(ens_seg_mask, msm) for msm in mem_seg_masks])
    # 2: members undergo thresholds unique for them
    mem_seg_masks = np.stack([process_probs(epp, th, l_min) for epp, th in
                              zip(ens_pred_probs, mem_prob_threshs)], axis=0)
    result['PSU^{+}'] = 1 - np.mean([intersection_over_union(ens_seg_mask, msm) for msm in mem_seg_masks])
    return result


def lesion_scale_measures(subject_lesion_uncs: pd.DataFrame):
    """
    Measures based on the aggreagation of the lesion-scale uncertainties.
    :param subject_lesion_uncs: a dataframs with the lesion scale uncertainty values
    :return: dictionary with computed uncertainty measures
    """
    def scale(arr):
        if np.max(arr) != np.min(arr):
            return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        else:
            return arr
        
    from .lesion_uncertainty_measures import les_uncs_measures as evaluated_uncertainty_measures
    
    result = dict()
    measures = set(evaluated_uncertainty_measures).intersection(set(subject_lesion_uncs.columns))
    for measure in list(measures):
        # without scaling (looking at the distribution on the dataset level)  
        result['avg. ' + measure] = subject_lesion_uncs[measure].mean()
        # scaling subject wise
        result['avg. scaled' + measure] = scale(subject_lesion_uncs[measure]).mean()
    return result


def voxel_scale_measures(ens_pred_probs: np.ndarray, brain_mask: np.ndarray):
    """
    Measures based on the aggreagation of the lesion-scale uncertainties.
    :param subject_lesion_uncs: a dataframs with the lesion scale uncertainty values
    :param brain_mask: a brain mask within which the uncertainty value will be aggregated 
    :return: dictionary with computed uncertainty measures
    """
    def scale(arr):
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    
    uncs_maps: dict = ensemble_uncertainties_classification(np.concatenate(
        (np.expand_dims(ens_pred_probs, axis=-1),
         np.expand_dims(1. - ens_pred_probs, axis=-1)),
        axis=-1))
    
    result = dict()
    
    for k, v in uncs_maps.items():
        result['avg. ' + k] = np.mean(v * brain_mask)
        result['avg. scaled ' + k] = np.mean(scale(v) * brain_mask)
    return result


def patient_uncertainty(ens_pred_probs: np.ndarray, brain_mask: np.ndarray,
                        ens_prob_thresh: float, mem_prob_threshs: list,
                        subject_lesion_uncs: pd.DataFrame,
                        l_min: int):
    """
    Compute all the patient-scale uncertainty measures based on aggregation of the voxel or lesion uncertainties or proposed measures based on structural disagreement.
    :param ens_pred_probs: probability maps for all models in ensemble, shape [M, H, W, D]
    :param ens_prob_thresh: probability threshold for the ensemble model
    :param mem_prob_threshols: list of the probability thresholds for each model in ensemble len = M
    :param l_min: minimal size of the connected component needed for the postprocessing
    :param subject_lesion_uncs: a dataframs with the lesion scale uncertainty values
    :param brain_mask: a brain mask within which the uncertainty value will be aggregated
    :param subject_lesion_uncs: a dataframs with the lesion scale uncertainty values
    :return: dictionary with computed uncertainty measures
    """
    result = global_scale_measures(ens_pred_probs, ens_prob_thresh, mem_prob_threshs, l_min)
    result.update(voxel_scale_measures(ens_pred_probs, brain_mask))
    result.update(lesion_scale_measures(subject_lesion_uncs))
    return result

