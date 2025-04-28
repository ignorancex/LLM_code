"""
Implementation of voxel and lesion scale retention curves.
"""
import numpy as np
from functools import partial
from collections import Counter
from joblib import Parallel, delayed
from sklearn import metrics
from scipy.interpolate import interp1d


def voxel_scale_metrics(y_pred: np.ndarray, y: np.ndarray, r: float = 0.001) -> dict:
    """
    Compute DSC, nDSC, TPR, TDR, FNR.
    :param y: ground truth values for each voxel in a scan
    :param y_pred: predicted class values for the corresponding voxels
    :param r: effective lesion load (for more info see Raina et al., 2023)
    :return: dict with metrict
    """
    if np.sum(y) + np.sum(y_pred) == 0:
        return {'DSC': 1.0, 'nDSC': 1.0, 'TPR': 1.0, 'TDR': 1.0, 'FNR': 0.0}
    else:
        k = (1 - r) * np.sum(y) / (r * (len(y.flatten()) - np.sum(y))) if np.sum(y) != 0.0 else 1.0
        tp = np.sum(y * y_pred)
        fp = np.sum(y_pred[y == 0])
        fn = np.sum(y[y_pred == 0])
        return {
            'DSC': 2.0 * tp / (2.0 * tp + fp + fn),
            'nDSC': 2.0 * tp / (2.0 * tp + fp * k + fn),
            'TPR': tp / (tp + fn),
            'TDR': tp / (tp + fp)
        }


def voxel_scale_rc(y_pred: np.ndarray, y: np.ndarray, uncertainties: np.ndarray,
                   fracs_retained: np.ndarray = np.linspace(0.0, 1.0, 400),
                   metric_name: str = 'DSC', n_jobs: int = None) -> tuple:
    """
    Compute error retention curve values and nDSC-AAC on voxel-scale.
    :param fracs_retained: fractions of retained voxels at each iteration ( x-axis values of the error-RC )
    :param y: ground truth values for each voxel in a scan
    :param y_pred: predicted class values for the corresponding voxels
    :param uncertainties: uncertainty values for the corresponding voxels
    :param n_jobs: number of parallel processes
    :return: tuple (area under the curve, y-axis values)
    """

    def compute_metric(frac_, preds_, gts_, N_, metric_name_):
        pos = int(N_ * frac_)
        curr_preds = preds if pos == N_ else np.concatenate([preds_[:pos], gts_[pos:]])
        return voxel_scale_metrics(y=gts_, y_pred=curr_preds)[metric_name_]

    ordering = uncertainties.argsort()
    gts = y[ordering].copy()
    preds = y_pred[ordering].copy()

    process = partial(compute_metric, preds_=preds, gts_=gts, N_=len(gts), metric_name_=metric_name)
    with Parallel(n_jobs=n_jobs) as parallel_backend:
        dsc_norm_scores = parallel_backend(delayed(process)(frac) for frac in fracs_retained)

    return 1. - metrics.auc(fracs_retained, np.asarray(dsc_norm_scores)), dsc_norm_scores


def lesion_scale_lppv_rc(lesion_uncertainties: list, lesion_types: list, fracs_retained: np.ndarray):
    """Computes lesion-scale LPPV-RC and LPPV-AUC for a single subject(from the paper).
    :param lesion_uncertainties: a list of uncertainty values of all predicted lesions in a scan, i.e. of all true positive lesions
    :param lesion_types: a list of types of the corresponding predicted lesions. avaliable lesion types: 'tp', 'fp', which stand for true positive and false positive lesions respectively.
    :param fracs_retained: fractions of retained voxels at each iteration ( x-axis values of the LPPV-RC )
    :returns: tuple (area under the curve, y-axis values)
    """

    def compute_lppv(lesion_types):
        counter = Counter(lesion_types)
        if counter['tp'] + counter['fp'] == 0.0:
            return 0.
        return counter['tp'] / (counter['tp'] + counter['fp'])

    assert len(lesion_uncertainties) == len(lesion_types)

    # sort uncertainties and lesions types
    ordering = lesion_uncertainties.argsort()
    lesion_type_all = lesion_types[ordering][::-1]

    metric_values = [compute_lppv(lesion_type_all)]

    # reject the most uncertain lesion
    for i_l, lesion_type in enumerate(lesion_type_all):
        if lesion_type == 'fp':
            lesion_type_all[i_l] = 'tn'
        metric_values.append(compute_lppv(lesion_type_all))

    # interpolate the curve and make predictions in the retention fraction nodes
    n_lesions = len(lesion_types)
    spline_interpolator = interp1d(x=[_ / n_lesions for _ in range(n_lesions + 1)],
                                   y=metric_values[::-1],
                                   kind='slinear', fill_value="extrapolate")
    metric_values_interp = spline_interpolator(fracs_retained)
    return 1. - metrics.auc(fracs_retained, np.asarray(metric_values_interp)), metric_values_interp


def patient_scale_rc(uncs_sample, metric_sample, replace_with: float = 1.0):
    """Computes patient-scale error retention curves (from the paper) for a dataset.
    Note, to plot an error retention curve, the x-axis values should be obtained as:
    ```python
    n_patients = len(uncs_sample)
    fracs_retained = np.linspace(0.0, 1.0, len(n_patients) + 1)
    ```
    
    :param uncs_sample: np.ndarray or list or uncertainty values for each patient in the dataset
    :param metric_sample: np.ndarray or list of the metrics for corresponding subjects. in the paper the DSC metric was used.
    :param replace_with: value replacing the metric value for each replaced subject. if the metric is the higher the better, should be 1; if the oposite, should be 0.
    :returns: tuple (area under the curve, y-axis values)
    """
    metric_sample_copy = np.asarray(metric_sample).copy()
    ordering = np.argsort(uncs_sample)
    metric_sample_copy = metric_sample_copy[ordering]
    rc = [metric_sample_copy.mean()]
    for idx in range(0, len(metric_sample_copy))[::-1]:
        metric_sample_copy[idx] = replace_with
        rc += [metric_sample_copy.mean()]
    fracs_retained = np.linspace(0, 1, len(rc))
    return metrics.auc(fracs_retained, np.asarray(rc)[::-1]), np.asarray(rc)[::-1]
