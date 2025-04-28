import sys
sys.path.append(os.path.join(os.expanduser('~'), 'false_positive_classification_pipeline', 'utils' ))
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from image_utils import return_lesion_coordinates
from scipy.ndimage import binary_opening, binary_closing, generate_binary_structure, binary_dilation
eps = 1e-8  # Avoid div-by-zero situations


def dice_score(seg, gt):
    """
    Function that calculates the dice similarity co-efficient
    over the entire batch

    :param seg: (numpy ndarray B x (D) x H x W) Batch of (Predicted )Segmentation map
    :param gt: (numpy ndarray B x (D) x H x W) Batch of ground truth maps

    :return dice_similarity_coeff: (float) Dice score

    """
    assert (isinstance(seg, np.ndarray))
    assert (isinstance(gt, np.ndarray))

#    seg = seg.flatten()
#    gt = gt.flatten()

    # Reshape to 2-D -- B x (D)*H*W
    seg = np.reshape(seg, (-1, np.prod(seg.shape[1:])))
    gt = np.reshape(gt, (-1, np.prod(gt.shape[1:])))

    try:
        assert(seg.shape[0] == gt.shape[0])
    except AssertionError:
        print('Array differ in axis 0 (batch-size), seg = {}, gt = {}'.format(seg.shape[0], gt.shape[0]))

    try:
        assert(seg.shape[-1] == gt.shape[-1])
    except AssertionError:
        print('Arrays differ is axis 1 (#elementrs), seg = {}, gt = {}'.format(seg.shape[-1], gt.shape[-1]))

    # TODO: Return dice = 1.0 when both arrays are empty
    inter = calculate_intersection(seg, gt)
    sum_term = np.sum(seg, axis=-1) + np.sum(gt, axis=-1) + eps

    dice_similarity_coeff = np.divide(2*inter, sum_term)

    return dice_similarity_coeff


def calculate_intersection(seg, gt, flatten=False):
    """
    Calculates intersection (as dot product) between 2 masks

    :param seg:
    :param gt:
    :return:
    """
    assert (isinstance(seg, np.ndarray))
    assert (isinstance(gt, np.ndarray))

    if flatten is True:
        seg = seg.flatten()
        gt = gt.flatten()

    return np.sum(np.multiply(seg, gt), axis=-1).astype(np.float32)


def calculate_union(seg, gt):

    if seg.ndim > 1:
        seg = seg.flatten()
    if gt.ndim > 1:
        gt = gt.flatten()

    # Implement a element-wise logical OR
    union = np.add(seg, gt)
    union = np.where(union>1, 1, union).astype(np.float32)
    # Union -> Sum of element-wise logical OR
    union = np.sum(union, axis=-1)

    return union

def hausdorff_distance(seg, gt):
    """
    Calculate the symmetric hausdorff distance between
    the segmentation and ground truth
    This metric is also known as Maximum Surface Distance

    :param seg: (numpy ndarray) Predicted segmentation. Expected dimensions num_slices x H x W
    :param gt: (numpy ndarray) Ground Truth. Expected dimensions num_slices x H x W
    :return: msd: (numpy ndarray) Symmetric hausdorff distance (Maximum surface distance)
    """
    assert (isinstance(seg, np.ndarray))
    assert (isinstance(gt, np.ndarray))

    msd = max(directed_hausdorff_distance(seg, gt),
              directed_hausdorff_distance(gt, seg))

    return msd


def directed_hausdorff_distance(vol1, vol2):
    """
    Directed Hausdorff distance between a pair of (3+1)-D volumes
    Max over hausdorff distances calculated between aligned slice pairs
    FIXME: Currently works for a 2-class label (foreground + background)
    FIXME: Check for logical bugs

    :param vol1: (numpy ndarray) Expected dimensions num_slices x H x W
    :param vol2: (numpy ndarray) Expected dimensions num_slices x H x W
    :return: directed_hd : (float) Directed Hausdorff distance
    """
    assert (isinstance(vol1, np.ndarray) and isinstance(vol2, np.ndarray))
    assert(vol1.ndim == 3 and vol2.ndim == 3)  # HD for 3D volumes

    n_slices = vol1.shape[0]

    hausdorff_distance_slice_pair = []
    for slice_id in range(n_slices):
        hausdorff_distance_slice_pair.append(directed_hausdorff(vol1[slice_id, :, :], vol2[slice_id, :, :])[0])

    directed_hd = max(hausdorff_distance_slice_pair)

    return directed_hd


def relative_volume_difference(seg, gt):
    """
    Calculate the relative volume difference between segmentation
    and the ground truth

    RVD (A, B) = (|B| - |A|)/|B|

    If RVD > 0 => Under-segmentation
       RVD < 0 => Over-segmentation

    :param seg: (numpy ndarray) Predicted segmentation
    :param gt: (numpy ndarray) Ground truth mask
    :return: rvd: (float) Relative volume difference (as %)
    """

    assert (isinstance(seg, np.ndarray))
    assert (isinstance(gt, np.ndarray))

    rvd = (np.sum(gt, axis=None) - np.sum(seg, axis=None))/(np.sum(gt, axis=None) + eps)
    rvd = rvd*100

    return rvd


def compute_spatial_entropy(seg, gt, umap):
    """
    Function to (qualitatively) analyze uncertainty map
    We compute average entropy over regions where a lesion has been predicted
    to check if the avg. entropy is higher for false postives.

    """
    assert(isinstance(seg, np.ndarray))
    assert(isinstance(gt, np.ndarray))
    assert(isinstance(umap, np.ndarray))

    predicted_slices, num_predicted_lesions = return_lesion_coordinates(mask=seg)
    true_slices, num_true_lesions = return_lesion_coordinates(mask=gt)
    # Analysis for true postives and false positives
    region_uncertainties_tp = []
    region_uncertainties_fp = []
    region_uncertainties_fn = []

    for predicted_volume in predicted_slices:
        intersection = dice_score(seg[predicted_volume], gt[predicted_volume])
        if intersection > 0: # True positive
            region_uncertainties_tp.append(np.mean(umap[predicted_volume]))
        else: # False Positive
            region_uncertainties_fp.append(np.mean(umap[predicted_volume]))

   # Analysis for false negatives
    for true_volume in true_slices:
        intersection = dice_score(seg[true_volume], gt[true_volume])
        if intersection == 0: # False negative
            region_uncertainties_fn.append(np.mean(umap[true_volume]))

    region_unc_dict = {'tp_unc' : region_uncertainties_tp, 'fp_unc': region_uncertainties_fp, 'fn_unc': region_uncertainties_fn}

    return region_unc_dict

def compute_lesion_volumes(seg, gt):
    """
    Function to compute (approximate) lesion volume.
    The find_objects() function provides a tuple of slices defining
    the minimal parallelopiped covering the lesion
    The volume is given as length*breadth*depth

    """
    predicted_slices, num_predicted_lesions = return_lesion_coordinates(mask=seg)
    true_slices, num_true_lesions = return_lesion_coordinates(mask=gt)

    tp_pred_volumes = []
    fp_pred_volumes = []
    fn_true_volumes = []
    tp_true_volumes = []

    for predicted_volume in predicted_slices:
        intersection = dice_score(seg[predicted_volume], gt[predicted_volume])
        length, breadth, depth = seg[predicted_volume].shape
        volume = length*breadth*depth
        if intersection > 0: # True positive
            tp_pred_volumes.append(volume)
        else: # False Positive
            fp_pred_volumes.append(volume)

    for true_volume in true_slices:
        intersection = dice_score(seg[true_volume], gt[true_volume])
        length, breadth, depth = gt[true_volume].shape
        volume = length*breadth*depth
        if intersection == 0: # False negative
            fn_true_volumes.append(volume)
        else:
            tp_true_volumes.append(volume)

    lesion_volume_dict = {'tp_pred': tp_pred_volumes,
                          'tp_true': tp_true_volumes,
                          'fp_pred': fp_pred_volumes,
                          'fn_true': fn_true_volumes}

    return lesion_volume_dict


def compute_iou(x, y):

    x_sum = np.sum(x)
    y_sum = np.sum(y)

    if x_sum == 0 and y_sum == 0:
        return 1
    else:
        union = calculate_union(x, y)
        intersection = calculate_intersection(x, y, flatten=True)
        iou = intersection/union
        return iou


def compute_ged_distance_metric(x, y):

    d = 1 - compute_iou(x, y)

    assert(d>=0)

    return d

def compute_self_term_ged(x):

    n_samples = x.shape[0]
    self_term = 0

    for i in range(n_samples):
        for j in range(n_samples):
            self_term += compute_ged_distance_metric(x[i, ...], x[j, ...])

    self_term = self_term/(n_samples*n_samples)
    return self_term


def compute_ged(seg, gt, verbose=False):
    """

    Function to compute the generalized energy distance that leverages distances between observations
    Metric used : d(x, y) = 1 - IoU(x, y) (See Kohl et al. (2019))

    Params:
        seg: (np.ndarray) Predicted segmentation(s) N_SAMPLES x D (optional) x H x W
        gt: (np.ndarray) Reference segmentation(s) N_SAMPLES x D (optional) x H x W

    Returns:
        ged: (float) The generalized energy distance

    """

    n_samples_seg = seg.shape[0]
    n_samples_gt = gt.shape[0]

    if verbose is True:
        print('Samples seg = {}'.format(n_samples_seg))
        print('Samples gt = {}'.format(n_samples_gt))

    ged_cross_term = 0

    for seg_idx in range(n_samples_seg):
        for gt_idx in range(n_samples_gt):
            ged_cross_term += compute_ged_distance_metric(seg[seg_idx, ...], gt[gt_idx, ...])

    ged_cross_term = 2*ged_cross_term/(n_samples_seg*n_samples_gt)

    if verbose is True:
        print('Cross-term = {}'.format(ged_cross_term))

    ged_seg_term = compute_self_term_ged(seg)
    ged_gt_term = compute_self_term_ged(gt)

    if verbose is True:
        print('GT term = {}'.format(ged_gt_term))
        print('Seg term = {}'.format(ged_seg_term))

    ged = ged_cross_term - ged_seg_term - ged_gt_term

    metric_dict = {}
    metric_dict['GED'] = ged
    metric_dict['Sample Diversity'] = ged_seg_term
    metric_dict['IO Variability'] = ged_gt_term
    metric_dict['Pred-Label distance'] = ged_cross_term/2

    return metric_dict

def compute_pairwise_iou(seg, gt, verbose=False):
    """

    Function to compute the average pairwise IoU between prediction and ground-truth label(s)

    Params:
        seg: (np.ndarray) Predicted segmentation(s) N_SAMPLES x D (optional) x H x W
        gt: (np.ndarray) Reference segmentation(s) N_SAMPLES x D (optional) x H x W

    Returns:
        avg_iou: (float) The generalized energy distance

    """

    n_samples_seg = seg.shape[0]
    n_samples_gt = gt.shape[0]

    if verbose is True:
        print('Samples seg = {}'.format(n_samples_seg))
        print('Samples gt = {}'.format(n_samples_gt))


    iou_accum = 0

    for seg_idx in range(n_samples_seg):
        for gt_idx in range(n_samples_gt):
            iou_accum += compute_iou(seg[seg_idx, ...], gt[gt_idx, ...])

    avg_iou = iou_accum/(n_samples_gt*n_samples_seg)

    return avg_iou


# Taken from https://github.com/baumgach/PHiSeg-code/blob/master/utils.py
def ncc(a,v, zero_norm=True):

    a = a.flatten()
    v = v.flatten()

    if zero_norm:
        a = (a - np.mean(a)) / (np.std(a) * len(a))
        v = (v - np.mean(v)) / (np.std(v))

    else:

        a = (a) / (np.std(a) * len(a))
        v = (v) / (np.std(v))

    return np.correlate(a,v)

def variance_ncc_dist(sample_arr, gt_arr):

    def pixel_wise_xent(m_samp, m_gt, eps=1e-8):

        log_samples = np.log(m_samp + eps)

        xent = -1.0*np.sum(m_gt*log_samples, axis=-1)

        return xent
    """
    :param sample_arr: expected shape N x X x Y
    :param gt_arr: M x X x Y
    :return:
    """

    mean_seg = np.mean(sample_arr, axis=0)

    N = sample_arr.shape[0]
    M = gt_arr.shape[0]


    sX = sample_arr.shape[1]
    sY = sample_arr.shape[2]

    E_ss_arr = np.zeros((N,sX,sY))

    for i in range(N):
        E_ss_arr[i,...] = pixel_wise_xent(sample_arr[i,...], mean_seg)

    E_ss = np.mean(E_ss_arr, axis=0)

    E_sy_arr = np.zeros((M,N, sX, sY))
    for j in range(M):
        for i in range(N):
            E_sy_arr[j,i, ...] = pixel_wise_xent(sample_arr[i,...], gt_arr[j,...])

    E_sy = np.mean(E_sy_arr, axis=1)

    ncc_list = []

    for j in range(M):

        ncc_list.append(ncc(E_ss, E_sy[j,...]))

    return (1/M)*sum(np.array(ncc_list))


# Code taken from: https://github.com/google-research/google-research/blob/9bd249201efac2ac60d8fe93a29c0058be544b5a/uq_benchmark_2019/metrics_lib.py#L31
def bin_predictions_and_accuracies(probabilities, ground_truth, bins=10):
    """A helper function which histograms a vector of probabilities into bins.
    Args:
    probabilities: A numpy vector of N probabilities assigned to each prediction
    ground_truth: A numpy vector of N ground truth labels in {0,1}
    bins: Number of equal width bins to bin predictions into in [0, 1], or an
      array representing bin edges.
    Returns:
    bin_edges: Numpy vector of floats containing the edges of the bins
      (including leftmost and rightmost).
    accuracies: Numpy vector of floats for the average accuracy of the
      predictions in each bin.
    counts: Numpy vector of ints containing the number of examples per bin.
    """

    if len(probabilities) != len(ground_truth):
        raise ValueError(
            'Probabilies and ground truth must have the same number of elements.')

    if [v for v in ground_truth if v not in [0., 1., True, False]]:
        raise ValueError(
            'Ground truth must contain binary labels {0,1} or {False, True}.')

    if isinstance(bins, int):
        num_bins = bins
    else:
        num_bins = bins.size - 1

    # Ensure probabilities are never 0, since the bins in np.digitize are open on
    # one side.
    probabilities = np.where(probabilities == 0, 1e-8, probabilities)
    counts, bin_edges = np.histogram(probabilities, bins=bins, range=[0., 1.])
    indices = np.digitize(probabilities, bin_edges, right=True)
    accuracies = np.array([np.mean(ground_truth[indices == i])
                         for i in range(1, num_bins + 1)])
    return bin_edges, accuracies, counts


def bin_centers_of_mass(probabilities, bin_edges):
    probabilities = np.where(probabilities == 0, 1e-8, probabilities)
    indices = np.digitize(probabilities, bin_edges, right=True)
    return np.array([np.mean(probabilities[indices == i])
                   for i in range(1, len(bin_edges))])


def expected_calibration_error(probabilities, ground_truth, bins=15):
    """Compute the expected calibration error of a set of preditions in [0, 1].
    Args:
    probabilities: A numpy vector of N probabilities assigned to each prediction
    ground_truth: A numpy vector of N ground truth labels in {0,1, True, False}
    bins: Number of equal width bins to bin predictions into in [0, 1], or
      an array representing bin edges.
    Returns:
    Float: the expected calibration error.
    """

    probabilities = probabilities.flatten()
    ground_truth = ground_truth.flatten()
    bin_edges, accuracies, counts = bin_predictions_and_accuracies(
      probabilities, ground_truth, bins)
    bin_centers = bin_centers_of_mass(probabilities, bin_edges)
    num_examples = np.sum(counts)

    ece = np.sum([(counts[i] / float(num_examples)) * np.sum(
      np.abs(bin_centers[i] - accuracies[i]))
            for i in range(bin_centers.size) if counts[i] > 0])

    return ece

