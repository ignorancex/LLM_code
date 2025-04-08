import numpy as np
from abc import ABC, abstractmethod

from os import path
from scipy import io
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, roc_curve
from sklearn import metrics
from skimage.feature import peak_local_max
from scipy.ndimage.measurements import center_of_mass, label

class Metric(ABC):
    '''
    Template for the evaluation metrics class
    '''

    @abstractmethod
    def get_name(self):
        '''
        Returns the name of the metric
        '''
        return None

    @abstractmethod
    def compute(self, gt, pred):
        '''
        Compute the metric
        '''
        return None



class MeanSquareError(Metric):
    '''
    MeanSquareError = frac{1}{N} sum_i^N (gt - pred).^2
    '''

    def get_name(self):
        return 'MeanSquareError'

    def compute(self, gt, pred):
        return mean_square_error(gt, pred)


def mean_square_error(gt, pred):
    '''
    MeanSquareError = frac{1}{N} sum_i^N (gt - pred).^2
    '''

    gt = gt.flatten()
    pred = pred.flatten()
    
    return np.mean(np.power(gt - pred, 2))



class MeanAbsoluteError(Metric):
    '''
    MeanAbsoluteError = frac{1}{N} sum_i^N |(gt - pred)| 
    '''

    def get_name(self):
        return 'MeanAbsoluteError'

    def compute(self, gt, pred):
        return mean_absolute_error(gt, pred)


def mean_absolute_error(gt, pred):
    '''
    MeanSquareError = frac{1}{N} sum_i^N |(gt - pred)| 
    '''

    gt = gt.flatten()
    pred = pred.flatten()
    
    return metrics.mean_absolute_error(gt, pred)



class Accuracy(Metric):
    '''
    Accuracy = Agreements / total
    '''

    def get_name(self):
        return 'Accuracy'

    def compute(self, gt, pred):
        return accuracy(gt, pred)


def accuracy(gt, pred):
    '''
    Accuracy = Agreements / total
    '''

    agreement = np.count_nonzero(gt == pred)
    n = gt.size
    
    return agreement / n



class Sensitivity(Metric):
    '''
    Sensitivity = TP / (TP + FN) (also known as Recall)
    '''

    def __init__(self):
        super(Sensitivity, self).__init__()

    def get_name(self):
        return 'Sensitivity'

    def compute(self, gt, pred):
        return sensitivity(gt, pred)


def sensitivity(gt, pred):
    '''
    Sensitivity = TP / (TP + FN) (also known as Recall)
    '''

    tp_ = tp(gt, pred)
    fn_ = fn(gt, pred)
    if tp_==0:
        return 0
    else:
        return tp_ / (tp_ + fn_)

def recall(gt, pred):
    '''
    Recall = Sensitivity
    '''

    return sensitivity(gt, pred)



class Specificity(Metric):
    '''
    Specificity = TN / (TN + FP)
    '''

    def __init__(self):
        super(Specificity, self).__init__()

    def get_name(self):
        return 'Specificity'

    def compute(self, gt, pred):
        return specificity(gt, pred)


def specificity(gt, pred):
    '''
    Specificity = TN / (TN + FP)
    '''

    tn_ = tn(gt, pred)
    fp_ = fp(gt, pred)
    return tn_ / (tn_ + fp_)



class Precision(Metric):
    '''
    Precision = TP / (TP + FP)
    '''

    def __init__(self):
        super(Precision, self).__init__()

    def get_name(self):
        return 'Precision'

    def compute(self, gt, pred):
        return precision(gt, pred)


def precision(gt, pred):
    '''
    Precision = TP / (TP + FP)
    '''

    tp_ = tp(gt, pred)
    fp_ = fp(gt, pred)
    if tp_ == 0:
        return 0
    else:
        return tp_ / (tp_ + fp_)



class Recall(Sensitivity):
    '''
    Recall = TP / (TP + FN)
    '''

    def __init__(self):
        super(Recall, self).__init__()

    def get_name(self):
        return 'Recall'



class F1Score(Metric):
    '''
    F1-score = 2 * Precision * Recall / (Precision + Recall)
    '''

    def __init__(self):
        super(F1Score, self).__init__()

    def get_name(self):
        return 'F1-Score'

    def compute(self, gt, pred):
        results = f1_score(gt, pred)
        return list(results)


def f1_score(gt, pred):
    '''
    F1-score = 2 * Precision * Recall / (Precision + Recall)
    '''

    pr_ = precision(gt, pred)
    re_ = recall(gt, pred)
    return 2 * (pr_ * re_) / (pr_ + re_ + 0.000001), pr_, re_



class Dice(F1Score):
    '''
    Dice is equivalent to F1 score
    '''

    def __init__(self):
        super(Dice, self).__init__()

    def get_name(self):
        return 'Dice'

    def compute(self, gt, pred):
        return super().compute(gt.flatten(), pred.flatten())[0]


def dice(gt, pred):
    '''
    Dice is equivalent to F1 score
    '''
    # return f1 score (but flattening the matrices first)
    return f1_score(gt.flatten(), pred.flatten())[0]



class IoU(Metric):
    '''
    IoU = Intersection / Union
    '''

    def __init__(self):
        super(IoU, self).__init__()

    def get_name(self):
        return 'IoU'

    def compute(self, gt, pred):
        results = iou(gt, pred)
        return results


def iou(gt, pred):
    '''
    IoU = Intersection / Union
    '''

    intersection = np.count_nonzero(np.logical_and(gt, pred))
    union = np.count_nonzero(np.logical_or(gt, pred))

    return intersection / union



class Jaccard(IoU):
    '''
    Jaccard is equivalent to IoU
    '''

    def __init__(self):
        super(Jaccard, self).__init__()

    def get_name(self):
        return 'Jaccard'



class GMean(Metric):
    '''
    G-Mean = 2 * Precision * Recall / (Precision + Recall)
    '''

    def __init__(self):
        super(GMean, self).__init__()
        self.individual_metrics = [Sensitivity(), Specificity()]

    def get_name(self):
        return 'G-Mean'

    def compute(self, gt, pred):
        results = g_mean(gt, pred)
        return list(results)


def g_mean(gt, pred):
    '''
    G-Mean = 2 * Precision * Recall / (Precision + Recall)
    '''

    se_ = sensitivity(gt, pred)
    sp_ = specificity(gt, pred)
    return math.sqrt(se_ * sp_), se_, sp_



class MCC(Metric):
    '''
    MCC (Matthews Correlation Coefficient) 
    https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
    '''

    def __init__(self):
        super(MCC, self).__init__()

    def get_name(self):
        return 'MCC'

    def compute(self, gt, pred):
        return mcc(gt, pred)


def mcc(gt, pred):
    '''
    MCC (Matthews Correlation Coefficient) 
    https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
    '''

    gt = gt > 0
    pred = pred > 0

    tp_ = tp(gt, pred)
    fn_ = fn(gt, pred)
    fp_ = fp(gt, pred)
    tn_ = tn(gt, pred)

    return ( ( (tp_ * tn_) - (fp_ * fn_) ) / (math.sqrt( (tp_ + fp_) * (tp_ + fn_) * (tn_ + fp_) * (tn_ + fn_) ) + 0.000001 ))


def tp(gt, pred):
    return len(np.flatnonzero(np.multiply(gt, pred)))

def tn(gt, pred):
    inverted_gt = np.logical_not(gt)
    inverted_pred = np.logical_not(pred)
    return len(np.flatnonzero(np.multiply(inverted_gt, inverted_pred)))

def fn(gt, pred):
    inverted_pred = np.logical_not(pred)
    return len(np.flatnonzero(np.multiply(gt, inverted_pred)))

def fp(gt, pred):
    return len(np.flatnonzero(np.multiply(np.logical_not(gt), pred)))



class PrecisionRecallCurve(Metric):
    '''
    Precision/Recall curve 
    http://mlwiki.org/index.php/Precision_and_Recall#Precision.2FRecall_Curves
    '''

    def __init__(self):
        super(PrecisionRecallCurve, self).__init__()

    def get_name(self):
        return 'PrecisionRecallCurve'

    def compute(self, gt, pred):
        # compute precision/recall curve
        self.precision_, self.recall_, self.auc_prre, _ = pr_re_curve(gt, pred)
        # return only the auc
        return self.auc_prre

    def save(self, output_path):
        '''
        Save the results
        '''
        # initialize the output path
        current_output_file = path.join(output_path, 'pr-re-curve.mat')
        print('Saving in {}'.format(current_output_file))
        # save the file
        io.savemat(current_output_file, { 'precision': self.precision_, 
                                          'recall': self.recall_, 
                                          'auc': self.auc_prre } )


def pr_re_curve(gt, pred):
    '''
    Precision/Recall curve 
    http://mlwiki.org/index.php/Precision_and_Recall#Precision.2FRecall_Curves
    '''

    if len(gt.shape) > 1:
        gt = gt.flatten()
    if len(pred.shape) > 1:
        pred = pred.flatten()

    if np.unique(gt).size > 1:
        precision_, recall_, thresholds = precision_recall_curve(gt, pred)
        auc_prre = auc(recall_, precision_)
    else:
        precision_ = np.nan
        recall_ = np.nan
        auc_prre = np.nan
        thresholds = np.nan

    return precision_, recall_, auc_prre, thresholds


def auc_pr_re_curve(gt, pred):

    _, _, auc_prre, _ = pr_re_curve(gt, pred)
    return auc_prre



class PrecisionRecallCurveWhenLabelSmoothing(PrecisionRecallCurve):
    '''
    Precision/Recall curve to use when applying label smoothing on the targets
    http://mlwiki.org/index.php/Precision_and_Recall#Precision.2FRecall_Curves
    '''

    def __init__(self):
        super(PrecisionRecallCurveWhenLabelSmoothing, self).__init__()

    def get_name(self):
        return 'PrecisionRecallCurveWhenLabelSmoothing'

    def compute(self, gt, pred):
        # compute precision/recall curve
        self.precision_, self.recall_, self.auc_prre, _ = pr_re_curve_when_label_smoothing(gt, pred)
        # return only the auc
        return self.auc_prre


def pr_re_curve_when_label_smoothing(gt, pred):
    '''
    Precision/Recall curve to use when applying label smoothing on the targets
    http://mlwiki.org/index.php/Precision_and_Recall#Precision.2FRecall_Curves
    '''

    '''
    # retrieve connected components
    labels = label(gt > 0)[0]
    # get the central pixel of each point
    merged_peaks = center_of_mass(gt, labels, range(1, np.max(labels)+1))
    gt_points = np.array(merged_peaks, dtype=int)

    # initialize a binary map
    binary_gt = np.zeros(gt.shape, dtype=bool)
    # iterate for each gt point
    for i in range(gt_points.shape[0]):
        # set the point in True
        binary_gt[gt_points[i,0], gt_points[i,1]] = True
    '''
        
    return pr_re_curve(gt > 0, pred)


def auc_pr_re_curve_when_label_smoothing(gt, pred):

    return auc_pr_re_curve(gt > 0, pred)




class ROCCurve(Metric):
    '''
    ROC curve 
    https://en.wikipedia.org/wiki/Receiver_operating_characteristic
    '''

    def __init__(self):
        super(ROCCurve, self).__init__()

    def get_name(self):
        return 'ROCCurve'

    def compute(self, gt, pred):
        # compute precision/recall curve
        self.fpr, self.tpr, self.auc_roc, _ = roc_curve_(gt, pred)
        # return only the auc
        return self.auc_roc

    def save(self, output_path):
        '''
        Save the results
        '''
        # initialize the output path
        current_output_file = path.join(output_path, 'roc-curve.mat')
        print('Saving in {}'.format(current_output_file))
        # save the file
        io.savemat(current_output_file, { 'fpr': self.fpr, 
                                          'tpr': self.tpr, 
                                          'auc': self.auc_roc } )


def roc_curve_(gt, pred):

    if len(gt.shape) > 1:
        gt = gt.flatten()
    if len(pred.shape) > 1:
        pred = pred.flatten()

    if np.unique(gt.flatten()).size > 1:
        fpr, tpr, thresholds = roc_curve(gt.flatten(), pred)
        auc_roc = roc_auc_score(gt.flatten(), pred)
    else:
        fpr = np.nan
        tpr = np.nan
        auc_roc = np.nan
        thresholds = np.nan

    return fpr, tpr, auc_roc, thresholds


def auc_roc_curve(gt, pred):

    _, _, auc_roc, _ = roc_curve_(gt, pred)
    return auc_roc




class PrecisionRecallCurveForBifurcations(PrecisionRecallCurve):
    '''
    Precision/Recall curve for bifurcations, as proposed in:
    https://www.sciencedirect.com/science/article/abs/pii/S0169260719307837
    '''

    def __init__(self, max_distance=5):
        super(PrecisionRecallCurveForBifurcations, self).__init__()

        # set the max distance tolerated from one point to the rest
        self.max_distance = max_distance

    def get_name(self):
        return 'PrecisionRecallCurveForBifurcations'

    def compute(self, gt, pred):
        '''
        gt and pred are here tensors with [num_images, height, width]
        '''

        # compute precision/recall curve
        self.precision_, self.recall_, self.auc_prre, _, self.f1 = pr_re_curve_bifurcations(gt, pred, self.max_distance)
        # return only the auc
        return self.auc_prre

    def save(self, output_path):
        '''
        Save the results
        '''
        # initialize the output path
        current_output_file = path.join(output_path, 'pr-re-curve.mat')
        print('Saving in {}'.format(current_output_file))
        # save the file
        io.savemat(current_output_file, { 'precision': self.precision_, 
                                          'recall': self.recall_, 
                                          'auc': self.auc_prre,
                                          'f1': self.f1 } )


def collect_bifurcation_scores(gt, pred, max_distance=5):
    '''
    Return a list of coordinates from the ground truth, bifurcation candidates and their associated scores
    '''

    assert len(gt) == len(pred), "Different number of images to compare (GT = {}, Pred = {})".format(len(gt), len(pred))

    # initialize the lists of elements
    bif_coords_for_each_gt = []
    bif_coords_for_predictions = []
    bif_scores_for_predictions = []

    # iterate for each gt/score map result
    for i in range(len(gt)):

        # retrieve current score map and gt
        current_score_map = np.array(pred[i], dtype=np.float)
        current_gt = np.array(gt[i]) > 0

        # get the centroid from the gt
        labels = label(current_gt)[0]
        merged_peaks = center_of_mass(current_gt, labels, range(1, np.max(labels)+1))
        current_gt_points = np.array(merged_peaks, dtype=int)

        # retrieve peak in score map
        peak_idx = peak_local_max(current_score_map, min_distance=max_distance)
        peak_mask = np.zeros_like(current_score_map, dtype=bool)
        peak_mask[tuple(peak_idx.T)] = True
        labels = label(peak_mask)[0]
        merged_peaks = center_of_mass(peak_mask, labels, range(1, np.max(labels)+1))
        current_predicted_coordinates = np.array(merged_peaks, dtype=int)

        # get the scores associated to these coordinates
        if current_predicted_coordinates.size > 0:
            current_scores = current_score_map.flatten()[np.ravel_multi_index((current_predicted_coordinates[:,0], current_predicted_coordinates[:,1]), current_score_map.shape)]
        else:
            current_scores = []

        # attach to the lists
        bif_coords_for_each_gt.append(current_gt_points)
        bif_coords_for_predictions.append(current_predicted_coordinates)
        bif_scores_for_predictions.append(current_scores)

    return bif_coords_for_each_gt, bif_coords_for_predictions, bif_scores_for_predictions


def evaluate_removing(ground, pred, thr=5):
    """
    ground: list of coordinates for the ground truth
    pred: list or coordinates for the prediction
    thr: distance threshold to consider a prediction as valid
    
    This function modifies the ground list!! so it may be adequate to pass copies of the lists as arguments
    """

    tp = 0
    fp = 0
    for p in pred:
        candidates = []
        for i, g in enumerate(ground):    
            dist = np.sqrt((p[0]-g[0])**2 + (p[1]-g[1])**2)
            if dist <= thr:
                candidates.append((i, dist))
        if len(candidates)>0:
            tp += 1
            nearest = min(candidates, key=lambda x: x[1])
            del ground[nearest[0]]
        else:
            fp += 1

    fn = len(ground)
    
    return tp, fp, fn


def evaluate_bifurcation_agreements_with_gt(bif_coords_for_each_gt, bif_coords_for_predictions, bif_scores_for_predictions=None, threshold=None, max_distance=5):
    '''
    Count the total amount of TP, FP and FN in a binary prediction of bifurcations. If scores are given, then first the map is thresholded at a given value
    '''

    # set the total amount of tp, fp and fn to zero
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # iterate for each element
    for i in range(len(bif_coords_for_each_gt)):

        # retrieve gt coordinates
        gt_points = bif_coords_for_each_gt[i]
        # and predicted coordinates
        predicted_coordinates = bif_coords_for_predictions[i]

        if predicted_coordinates.size != 0:

            # if scores are given, threshold on the given threshold
            if not (bif_scores_for_predictions is None) and not (threshold is None):

                # and their scores
                current_scores = bif_scores_for_predictions[i]

                # threshold the scores at t
                pred_idx = current_scores > threshold
                # zip the corresponding coordinates
                pred_coords = np.stack((predicted_coordinates[pred_idx,0],predicted_coordinates[pred_idx,1]), axis=1)
            
            # otherwise, assume that the input is binary
            else:

                pred_coords = predicted_coordinates

        else:

            pred_coords = predicted_coordinates
        
        # get tp, fp and fn
        tp, fp, fn = evaluate_removing(gt_points.tolist(), pred_coords.tolist(), max_distance)
        # accumulate the number of tp, fp and fn
        total_tp += tp
        total_fp += fp
        total_fn += fn

    return total_tp, total_fp, total_fn


def pr_re_curve_bifurcations(gt, pred, max_distance=5):
    '''
    Precision/Recall curve for bifurcations, as proposed in:
    https://www.sciencedirect.com/science/article/abs/pii/S0169260719307837

    gt and pred are lists of images
    '''

    # retrieve a list of coordinates from the ground truth, a list of bifurcation candidates and their associated scores
    bif_coords_for_each_gt, bif_coords_for_predictions, bif_scores_for_predictions = collect_bifurcation_scores(gt, pred, max_distance)

    if len(bif_coords_for_each_gt)==0:
        
        # if no coordinates are in the gt, then we cannot compute the precision/recall curve
        precision = np.nan
        recall = np.nan
        auc_prre = np.nan
        thresholds = np.nan

    else:

        # initialize the list of thresholds
        thresholds = np.arange(0, 255)
        thresholds = np.asarray(thresholds, dtype=np.float32) / 255

        # initialize lists of precision/recall values that will be used for the precision/recall curve
        precision = []
        recall = []
        f1 = []

        # iterate for each threshold
        for t in thresholds:

            # get the number of tp, fp and fn
            total_tp, total_fp, total_fn = evaluate_bifurcation_agreements_with_gt(bif_coords_for_each_gt, bif_coords_for_predictions, bif_scores_for_predictions, t, max_distance)

            # compute precision, recall and f1
            alpha = 1e-8
            pr_ = total_tp / (total_tp+total_fp+alpha)
            re_ = total_tp / (total_tp+total_fn+alpha)
            f1_ = 2*pr_*re_ / (pr_+re_+alpha)
            precision.append( pr_ )
            recall.append( re_ )
            f1.append( f1_ )

        # compute the auc
        auc_prre = auc(recall, precision)

    return precision, recall, auc_prre, thresholds, f1


def auc_pr_re_curve_bifurcations(gt, pred):

    _, _, auc_prre, _, _ = pr_re_curve_bifurcations(gt, pred)
    return auc_prre




class PrecisionForBifurcations(Precision):
    '''
    Precision = TP / (TP + FN)
    '''

    def __init__(self, max_distance=5):
        super(PrecisionForBifurcations, self).__init__()

        # set the max distance tolerated from one point to the rest
        self.max_distance = max_distance

    def get_name(self):
        return 'PrecisionForBifurcations'

    def compute(self, gt, pred):
        return precision_for_bifurcations(gt, pred, self.max_distance)


def precision_for_bifurcations(gt, pred, max_distance=5):
    '''
    Precision = TP / (TP + FP)
    '''

    # turn all the matrices to binary
    for i in range(len(gt)):
        gt[i] = gt[i] > 0
        pred[i] = pred[i] > 0

    # get coordinates from gt and predictions
    bif_coords_for_each_gt, bif_coords_for_predictions, _ = collect_bifurcation_scores(gt, pred, max_distance=5)

    # get the number of tp, fp and fn
    total_tp, total_fp, _ = evaluate_bifurcation_agreements_with_gt(bif_coords_for_each_gt, bif_coords_for_predictions, None, None, max_distance)

    # compute precision
    alpha = 1e-8
    pr_ = total_tp / (total_tp+total_fp+alpha)

    return pr_




class RecallForBifurcations(Recall):
    '''
    Recall = TP / (TP + FN)
    '''

    def __init__(self, max_distance=5):
        super(RecallForBifurcations, self).__init__()

        # set the max distance tolerated from one point to the rest
        self.max_distance = max_distance

    def get_name(self):
        return 'RecallForBifurcations'
    
    def compute(self, gt, pred):
        return recall_for_bifurcations(gt, pred, self.max_distance)


def recall_for_bifurcations(gt, pred, max_distance=5):
    '''
    Precision = TP / (TP + FN)
    '''

    # turn all the matrices to binary
    for i in range(len(gt)):
        gt[i] = gt[i] > 0
        pred[i] = pred[i] > 0

    # get coordinates from gt and predictions
    bif_coords_for_each_gt, bif_coords_for_predictions, _ = collect_bifurcation_scores(gt, pred, max_distance=5)

    # get the number of tp, fp and fn
    total_tp, _, total_fn = evaluate_bifurcation_agreements_with_gt(bif_coords_for_each_gt, bif_coords_for_predictions, None, None, max_distance)

    # compute recall
    alpha = 1e-8
    re_ = total_tp / (total_tp+total_fn+alpha)

    return re_




class F1ScoreForBifurcations(F1Score):
    '''
    F1-score = 2 * Precision * Recall / (Precision + Recall)
    '''

    def __init__(self, max_distance=5):
        super(F1ScoreForBifurcations, self).__init__()

        # set the max distance tolerated from one point to the rest
        self.max_distance = max_distance

    def get_name(self):
        return 'F1ScoreForBifurcations'
    
    def compute(self, gt, pred):
        return f1_score_for_bifurcations(gt, pred, self.max_distance)


def f1_score_for_bifurcations(gt, pred, max_distance=5):
    '''
    gt   list(H*W)
    pred list(H*W)

    F1-score = 2 * Precision * Recall / (Precision + Recall)
    '''

    # turn all the matrices to binary
    for i in range(len(gt)):
        gt[i] = gt[i] > 0
        pred[i] = pred[i] > 0

    # get coordinates from gt and predictions
    bif_coords_for_each_gt, bif_coords_for_predictions, _ = collect_bifurcation_scores(gt, pred, max_distance=5)

    # get the number of tp, fp and fn
    total_tp, total_fp, total_fn = evaluate_bifurcation_agreements_with_gt(bif_coords_for_each_gt, bif_coords_for_predictions, None, None, max_distance)

    # compute recall
    alpha = 1e-8
    pr_ = total_tp / (total_tp+total_fp+alpha)
    re_ = total_tp / (total_tp+total_fn+alpha)
    f1_ = 2*pr_*re_ / (pr_+re_+alpha)

    return f1_




EVALUATION_METRICS_FUNCTIONS = {'mse': mean_square_error,
                                'mae': mean_absolute_error,
                                'accuracy': accuracy,
                                'sensitivity': sensitivity,
                                'specificity': specificity,
                                'precision': precision,
                                'recall': recall,
                                'gmean': g_mean,
                                'f1_score': f1_score,
                                'dice': dice,
                                'iou': iou,
                                'jaccard': iou,
                                'pr_re_curve': pr_re_curve,
                                'auc_pr_re_curve': auc_pr_re_curve,
                                'pr_re_curve_when_label_smoothing': pr_re_curve_when_label_smoothing,
                                'auc_pr_re_curve_when_label_smoothing': auc_pr_re_curve_when_label_smoothing,
                                'roc_curve': roc_curve_,
                                'auc_roc_curve': auc_roc_curve,
                                'pr_re_curve_bifurcations': pr_re_curve_bifurcations,
                                'auc_pr_re_curve_bifurcations': auc_pr_re_curve_bifurcations,
                                'precision_for_bifurcations': precision_for_bifurcations,
                                'recall_for_bifurcations': recall_for_bifurcations,
                                'f1_score_for_bifurcations': f1_score_for_bifurcations}


EVALUATION_METRICS_OBJECTS = {'mse': MeanSquareError(),
                              'mae': MeanAbsoluteError(),
                              'accuracy': Accuracy(),
                              'sensitivity': Sensitivity(),
                              'specificity': Specificity(),
                              'precision': Precision(),
                              'recall': Recall(),
                              'gmean': GMean(),
                              'f1_score': F1Score(),
                              'dice': Dice(),
                              'iou': IoU(),
                              'jaccard': Jaccard(),
                              'pr_re_curve': PrecisionRecallCurve(),
                              'auc_pr_re_curve': PrecisionRecallCurve(),
                              'pr_re_curve_when_label_smoothing': PrecisionRecallCurveWhenLabelSmoothing(),
                              'auc_pr_re_curve_when_label_smoothing': PrecisionRecallCurveWhenLabelSmoothing(),
                              'roc_curve': ROCCurve(),
                              'auc_roc_curve': ROCCurve(),
                              'pr_re_curve_bifurcations': PrecisionRecallCurveForBifurcations(),
                              'auc_pr_re_curve_bifurcations': PrecisionRecallCurveForBifurcations(),
                              'precision_for_bifurcations': PrecisionForBifurcations(),
                              'recall_for_bifurcations': RecallForBifurcations(),
                              'f1_score_for_bifurcations': F1ScoreForBifurcations()}