
from torch import nn
import torch
from skimage.measure import regionprops, label
from torchvision.transforms import ToTensor, ToPILImage
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.metrics import  confusion_matrix, roc_curve, accuracy_score, precision_recall_fscore_support, auc,precision_recall_curve, average_precision_score
import wandb 
import monai
from torch.nn import functional as F
from PIL import Image
import matplotlib.colors as colors
from cupyx.scipy.ndimage import median_filter, gaussian_filter
import cupy as cp


def gpu_filter(volume, type, type_param):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please ensure GPU support.")

    # Ensure input is on GPU
    volume_np = volume.cpu().numpy()  # Convert to NumPy (if PyTorch tensor)
    volume_cp = cp.asarray(volume_np)  # Transfer to GPU using CuPy
    
    # Apply the median filter
    if type == 'median':
        filtered_volume_cp = median_filter(volume_cp, size=type_param)
    elif type == 'gaussian':
        filtered_volume_cp = gaussian_filter(volume_cp, sigma=type_param)

    # Convert back to PyTorch tensor on GPU
    filtered_volume_np = cp.asnumpy(filtered_volume_cp)
    filtered_volume = torch.from_numpy(filtered_volume_np).to(volume.device)
    
    return filtered_volume

def _test_step(self, heatmap, data_orig, data_seg, data_mask, batch_idx, ID, label_vol) :
        self.healthy_sets = ['IXI']
        # Resize the images if desired
        if not self.cfg.resizedEvaluation: # in case of full resolution evaluation 
            heatmap = F.interpolate(heatmap, size=self.new_size, mode="trilinear",align_corners=True).squeeze() # resize
        # move data to CPU
        data_seg = data_seg.cpu() 
        data_mask = data_mask.cpu()
        heatmap = heatmap.cpu() * data_mask
        data_orig = data_orig.cpu()
        # binarize the segmentation
        data_seg[data_seg > 0] = 1
        data_mask[data_mask > 0] = 1

        # save image grid
        if self.cfg['saveOutputImages'] :
            log_images(self, heatmap, data_orig, data_seg, ID)
            
        ### Compute Metrics per Volume / Step ###
        if self.cfg.evalSeg and self.dataset[0] not in self.healthy_sets: # only compute metrics if segmentation is available

            # Pixel-Wise Segmentation Error Metrics based on Differenceimage
            AUC, _fpr, _tpr, _threshs = compute_roc(heatmap.squeeze().flatten(), np.array(data_seg.squeeze().flatten()).astype(bool))
            AUPRC, _precisions, _recalls, _threshs = compute_prc(heatmap.squeeze().flatten(),np.array(data_seg.squeeze().flatten()).astype(bool))
            self.eval_dict[f'AUCPerVol'].append(AUC)
            self.eval_dict[f'AUPRCPerVol'].append(AUPRC)
            # gready search for threshold
            bestDice, bestThresh = find_best_val(np.array(heatmap.squeeze()).flatten(),  # threshold search with a subset of EvaluationSet
                                                np.array(data_seg.squeeze()).flatten().astype(bool), 
                                                val_range=(0, np.max(np.array(heatmap))),
                                                max_steps=10, 
                                                step=0, 
                                                max_val=0, 
                                                max_point=0)

            # if 'test' in self.stage:
            #     bestThresh = self.threshold['total']

            def set_thresholded_metric_values(diffs_thresholded,data_seg,prefix=''):
                # Calculate Dice Score with thresholded volumes
                diceScore = dice(np.array(diffs_thresholded.squeeze()),np.array(data_seg.squeeze().flatten()).astype(bool))
                
                # Other Metrics
                TP, FP, TN, FN = confusion_matrix(np.array(diffs_thresholded.squeeze().flatten()), np.array(data_seg.squeeze().flatten()).astype(bool),labels=[0, 1]).ravel()
                TPR = tpr(np.array(diffs_thresholded.squeeze()), np.array(data_seg.squeeze().flatten()).astype(bool))
                FPR = fpr(np.array(diffs_thresholded.squeeze()), np.array(data_seg.squeeze().flatten()).astype(bool))
                self.eval_dict[f'{prefix}DiceScorePerVol'].append(diceScore)
                self.eval_dict[f'{prefix}TPPerVol'].append(TP)
                self.eval_dict[f'{prefix}FPPerVol'].append(FP)
                self.eval_dict[f'{prefix}TNPerVol'].append(TN)
                self.eval_dict[f'{prefix}FNPerVol'].append(FN) 
                self.eval_dict[f'{prefix}TPRPerVol'].append(TPR)
                self.eval_dict[f'{prefix}FPRPerVol'].append(FPR)

                PrecRecF1PerVol = precision_recall_fscore_support(np.array(data_seg.squeeze().flatten()).astype(bool),np.array(diffs_thresholded.squeeze()).flatten(),labels=[0,1])
                self.eval_dict[f'{prefix}AccuracyPerVol'].append(accuracy_score(np.array(data_seg.squeeze().flatten()).astype(bool),np.array(diffs_thresholded.squeeze()).flatten()))
                self.eval_dict[f'{prefix}PrecisionPerVol'].append(PrecRecF1PerVol[0][1])
                self.eval_dict[f'{prefix}RecallPerVol'].append(PrecRecF1PerVol[1][1])
                self.eval_dict[f'{prefix}SpecificityPerVol'].append(TN / (TN+FP+0.0000001))

                # other metrics from monai:
                if len(data_seg.shape) == 4:
                    data_seg = data_seg.unsqueeze(0)
                Haus = monai.metrics.compute_hausdorff_distance(diffs_thresholded.unsqueeze(0).unsqueeze(0),data_seg, include_background=False, distance_metric='euclidean', percentile=None, directed=False)
                self.eval_dict[f'{prefix}HausPerVol'].append(Haus.item())

                # compute slice-wise metrics
                for slice in range(data_seg.squeeze().shape[0]): 
                        if np.array(data_seg.squeeze()[slice].flatten()).astype(bool).any():
                            self.eval_dict[f'{prefix}DiceScorePerSlice'].append(dice(np.array(diffs_thresholded[slice]),np.array(data_seg.squeeze()[slice].flatten()).astype(bool)))
                            PrecRecF1PerSlice = precision_recall_fscore_support(np.array(data_seg.squeeze()[slice].flatten()).astype(bool),np.array(diffs_thresholded[slice]).flatten(),warn_for=tuple(),labels=[0,1])
                            self.eval_dict[f'{prefix}PrecisionPerSlice'].append(PrecRecF1PerSlice[0][1])
                            self.eval_dict[f'{prefix}RecallPerSlice'].append(PrecRecF1PerSlice[1][1])
                            self.eval_dict[f'{prefix}lesionSizePerSlice'].append(np.count_nonzero(np.array(data_seg.squeeze()[slice].flatten()).astype(bool)))


            self.eval_dict[f'BestDicePerVol'].append(bestDice)
            self.eval_dict[f'BestThresholdPerVol'].append(bestThresh)
            if self.cfg["threshold"] == 'auto':
                diffs_thresholded = heatmap > bestThresh
                diffs_thresholded = np.squeeze(diffs_thresholded)            
                set_thresholded_metric_values(diffs_thresholded,data_seg,self.threshold_prefixes[0])
            if 'unsupervised_threshold' in self.cfg.keys() and len(self.threshold_prefixes) > 1:
                diffs_thresholded = heatmap > self.cfg.unsupervised_threshold
                diffs_thresholded = np.squeeze(diffs_thresholded)
                set_thresholded_metric_values(diffs_thresholded,data_seg,self.threshold_prefixes[1])
            self.eval_dict[f'lesionSizePerVol'].append(np.count_nonzero(np.array(data_seg.squeeze().flatten()).astype(bool)))
            self.eval_dict[f'IDs'].append(ID[0])


        # if 'val' in self.stage  :
        #     if batch_idx == 0:
        #         self.diffs_list = np.array(heatmap.squeeze().flatten().half())
        #         self.seg_list = np.array(data_seg.squeeze().flatten()).astype(np.int8)
        #     else:
        #         self.diffs_list = np.append(self.diffs_list,np.array(heatmap.squeeze().flatten().half()),axis=0)
        #         self.seg_list = np.append(self.seg_list,np.array(data_seg.squeeze().flatten()),axis=0).astype(np.int8)

        AnomalyScoreReco = [] # Reconstruction based Anomaly score
        if len(heatmap.squeeze().shape) !=2:
            for slice in range(heatmap.squeeze().shape[0]):
                score = heatmap.squeeze()[slice][data_mask.squeeze()[slice]>0].mean()
                if score.isnan() : # if no brain exists in that slice
                    AnomalyScoreReco.append(0.0) 
                else: 
                    AnomalyScoreReco.append(score) 

            # create slice-wise labels 
            data_seg_downsampled = np.array(data_seg.squeeze())
            label = [] # store labels here
            for slice in range(data_seg_downsampled.shape[0]) :  #iterate through volume
                if np.array(data_seg_downsampled[slice]).astype(bool).any(): # if there is an anomaly segmentation
                    label.append(1) # label = 1
                else :
                    label.append(0) # label = 0 if there is no Anomaly in the slice
                    
            if self.dataset[0] not in self.healthy_sets:
                AUC, _fpr, _tpr, _threshs = compute_roc(np.array(AnomalyScoreReco),np.array(label))
                AUPRC, _precisions, _recalls, _threshs = compute_prc(np.array(AnomalyScoreReco),np.array(label))
                self.eval_dict['AUCAnomalyRecoPerSlice'].append(AUC)
                self.eval_dict['AUPRCAnomalyRecoPerSlice'].append(AUPRC)
                self.eval_dict['labelPerSlice'].extend(label)
                # store Slice-wise Anomalyscore (reconstruction based)
                self.eval_dict['AnomalyScoreRecoPerSlice'].extend(AnomalyScoreReco)

        self.eval_dict['labelPerVol'].append(label_vol.item() if isinstance(label_vol,torch.Tensor) else label_vol)

def _test_end(self) :
    # average over all test samples
        self.eval_dict['l1recoErrorAllMean'] = np.nanmean(self.eval_dict['l1recoErrorAll'])
        self.eval_dict['l1recoErrorAllStd'] = np.nanstd(self.eval_dict['l1recoErrorAll'])
        self.eval_dict['l2recoErrorAllMean'] = np.nanmean(self.eval_dict['l2recoErrorAll'])
        self.eval_dict['l2recoErrorAllStd'] = np.nanstd(self.eval_dict['l2recoErrorAll'])

        self.eval_dict['l1recoErrorHealthyMean'] = np.nanmean(self.eval_dict['l1recoErrorHealthy'])
        self.eval_dict['l1recoErrorHealthyStd'] = np.nanstd(self.eval_dict['l1recoErrorHealthy'])
        self.eval_dict['l1recoErrorUnhealthyMean'] = np.nanmean(self.eval_dict['l1recoErrorUnhealthy'])
        self.eval_dict['l1recoErrorUnhealthyStd'] = np.nanstd(self.eval_dict['l1recoErrorUnhealthy'])

        self.eval_dict['l2recoErrorHealthyMean'] = np.nanmean(self.eval_dict['l2recoErrorHealthy'])
        self.eval_dict['l2recoErrorHealthyStd'] = np.nanstd(self.eval_dict['l2recoErrorHealthy'])
        self.eval_dict['l2recoErrorUnhealthyMean'] = np.nanmean(self.eval_dict['l2recoErrorUnhealthy'])
        self.eval_dict['l2recoErrorUnhealthyStd'] = np.nanstd(self.eval_dict['l2recoErrorUnhealthy'])

        self.eval_dict['AUPRCPerVolMean'] = np.nanmean(self.eval_dict['AUPRCPerVol'])
        self.eval_dict['AUPRCPerVolStd'] = np.nanstd(self.eval_dict['AUPRCPerVol'])
        self.eval_dict['AUCPerVolMean'] = np.nanmean(self.eval_dict['AUCPerVol'])
        self.eval_dict['AUCPerVolStd'] = np.nanstd(self.eval_dict['AUCPerVol'])
        self.eval_dict["BestDicePerVolMean"] =  np.nanmean(self.eval_dict['BestDicePerVol'])
        self.eval_dict["BestDicePerVolStd"] =  np.nanstd(self.eval_dict['BestDicePerVol'])
        self.eval_dict["BestThresholdPerVolMean"] =  np.nanmean(self.eval_dict['BestThresholdPerVol'])
        self.eval_dict["BestThresholdPerVolStd"] =  np.nanstd(self.eval_dict['BestThresholdPerVol'])

        # Define the metrics and their corresponding aggregation functions
        metrics = {
            "DiceScorePerVol": (np.nanmean, np.nanstd),

            "TPPerVol": (np.nanmean, np.nanstd),
            "FPPerVol": (np.nanmean, np.nanstd),
            "TNPerVol": (np.nanmean, np.nanstd),
            "FNPerVol": (np.nanmean, np.nanstd),
            "TPRPerVol": (np.nanmean, np.nanstd),
            "FPRPerVol": (np.nanmean, np.nanstd),
            "HausPerVol": (
                lambda x: np.nanmean(np.array(x)[np.isfinite(x)]),
                lambda x: np.nanstd(np.array(x)[np.isfinite(x)]),
            ),
            "PrecisionPerVol": (np.mean, np.std),
            "RecallPerVol": (np.mean, np.std),
            "PrecisionPerSlice": (np.mean, np.std),
            "RecallPerSlice": (np.mean, np.std),
            "AccuracyPerVol": (np.mean, np.std),
            "SpecificityPerVol": (np.mean, np.std),
        }

        prefixes = self.threshold_prefixes
        for prefix in prefixes:
            for metric, (mean_fn, std_fn) in metrics.items():
                # Construct the dynamic key for the metric
                key = f"{prefix}{metric}"
                if key in self.eval_dict and len(self.eval_dict[key]) > 0:
                    # Compute and store the mean and standard deviation
                    self.eval_dict[f"{prefix}{metric}Mean"] = mean_fn(self.eval_dict[key])
                    self.eval_dict[f"{prefix}{metric}Std"] = std_fn(self.eval_dict[key])
                else:
                    # Handle cases where the key is missing or data is empty
                    self.eval_dict[f"{prefix}{metric}Mean"] = np.nan
                    self.eval_dict[f"{prefix}{metric}Std"] = np.nan


        # self.eval_dict['QuantileThresholds'] = self.quantile_thresholds
        # if 'test' in self.stage :
        del self.threshold
                
        # if 'val' in self.stage:
        #     if self.dataset[0] not in self.healthy_sets:
        #         bestdiceScore, bestThresh = find_best_val((self.diffs_list).flatten(), (self.seg_list).flatten().astype(bool),
        #                                 val_range=(0, np.max((self.diffs_list))),
        #                                 max_steps=10,
        #                                 step=0,
        #                                 max_val=0,
        #                                 max_point=0)
        #
        #         self.threshold['total'] = bestThresh
        #         if self.cfg.get('KLDBackprop',False):
        #             bestdiceScoreKLComb, bestThreshKLComb = find_best_val((self.diffs_listKLComb).flatten(), (self.seg_list).flatten().astype(bool),
        #                 val_range=(0, np.max((self.diffs_listKLComb))),
        #                 max_steps=10,
        #                 step=0,
        #                 max_val=0,
        #                 max_point=0)
        #
        #             self.threshold['totalKLComb'] = bestThreshKLComb
        #             bestdiceScoreKL, bestThreshKL = find_best_val((self.diffs_listKL).flatten(), (self.seg_list).flatten().astype(bool),
        #                 val_range=(0, np.max((self.diffs_listKL))),
        #                 max_steps=10,
        #                 step=0,
        #                 max_val=0,
        #                 max_point=0)
        #
        #             self.threshold['totalKL'] = bestThreshKL
        #    else: # define thresholds based on the healthy validation set
        if len(self.diffs_list) > 0:
            _, fpr_healthy, _, threshs = compute_roc((self.diffs_list).flatten(), np.zeros_like(self.diffs_list).flatten().astype(int))
            self.threshholds_healthy= {
                    'thresh_1p' : threshs[np.argmax(fpr_healthy>0.01)], # 1%
                    'thresh_5p' : threshs[np.argmax(fpr_healthy>0.05)], # 5%
                    'thresh_10p' : threshs[np.argmax(fpr_healthy>0.10)]} # 10%}
            self.eval_dict['t_1p'] = self.threshholds_healthy['thresh_1p']
            self.eval_dict['t_5p'] = self.threshholds_healthy['thresh_5p']
            self.eval_dict['t_10p'] = self.threshholds_healthy['thresh_10p']

def get_eval_dictionary(prefixes):

    all_keys = {}
    for prefix in prefixes:
        keys = {
            f'{prefix}DiceScorePerVol': [],
            f'{prefix}TPPerVol': [],
            f'{prefix}FPPerVol': [],
            f'{prefix}TNPerVol': [],
            f'{prefix}FNPerVol': [],
            f'{prefix}TPRPerVol': [],
            f'{prefix}FPRPerVol': [],
            f'{prefix}AccuracyPerVol': [],
            f'{prefix}PrecisionPerVol': [],
            f'{prefix}RecallPerVol': [],
            f'{prefix}SpecificityPerVol': [],
            f'{prefix}HausPerVol': [],
            f'{prefix}DiceScorePerSlice': [],
            f'{prefix}PrecisionPerSlice': [],
            f'{prefix}RecallPerSlice': [],
            f'{prefix}lesionSizePerSlice': []
        }
        all_keys = keys | all_keys
    _eval = {
        'IDs': [],
        'x': [],
        'reconstructions': [],
        'diffs': [],
        'diffs_volume': [],
        'Segmentation': [],
        'reconstructionTimes': [],
        'latentSpace': [],
        'Age': [],
        'AgeGroup': [],
        'l1reconstructionErrors': [],
        'l1recoErrorAll': [],
        'l1recoErrorUnhealthy': [],
        'l1recoErrorHealthy': [],
        'l2recoErrorAll': [],
        'l2recoErrorUnhealthy': [],
        'l2recoErrorHealthy': [],
        'l1reconstructionErrorMean': 0.0,
        'l1reconstructionErrorStd': 0.0,
        'l2reconstructionErrors': [],
        'l2reconstructionErrorMean': 0.0,
        'l2reconstructionErrorStd': 0.0,
        'AUCPerVol': [],
        'AUPRCPerVol':[],
        'lesionSizePerVol': [],
        'Dice': [],
        'BestDicePerVol': [],
        'BestThresholdPerVol': [],
        'TPgradELBO': [],
        'FPgradELBO': [],
        'FNgradELBO': [],
        'TNgradELBO': [],
        'TPRgradELBO': [],
        'FPRgradELBO': [],
        'DicegradELBO': [],
        'DiceScorePerVolgradELBO': [],
        'BestDicePerVolgradELBO': [],
        'BestThresholdPerVolgradELBO': [],
        'AUCPerVolgradELBO': [],
        'AUPRCPerVolgradELBO': [],
        'KLD_to_learned_prior':[],

        'AUCAnomalyCombPerSlice': [], # PerVol!!! + Confusionmatrix.
        'AUPRCAnomalyCombPerSlice': [],
        'AnomalyScoreCombPerSlice': [],


        'AUCAnomalyKLDPerSlice': [],
        'AUPRCAnomalyKLDPerSlice': [],
        'AnomalyScoreKLDPerSlice': [],


        'AUCAnomalyRecoPerSlice': [],
        'AUPRCAnomalyRecoPerSlice': [],
        'AnomalyScoreRecoPerSlice': [],
        'AnomalyScoreRecoBinPerSlice': [],
        'AnomalyScoreAgePerSlice': [],
        'AUCAnomalyAgePerSlice': [],
        'AUPRCAnomalyAgePerSlice': [],

        'labelPerSlice' : [],
        'labelPerVol' : [],
        'AnomalyScoreCombPerVol' : [],
        'AnomalyScoreCombiPerVol' : [],
        'AnomalyScoreCombMeanPerVol' : [],
        'AnomalyScoreRegPerVol' : [],
        'AnomalyScoreRegMeanPerVol' : [],
        'AnomalyScoreRecoPerVol' : [],
        'AnomalyScoreCombPriorPerVol': [],
        'AnomalyScoreCombiPriorPerVol': [],
        'AnomalyScoreAgePerVol' : [],
        'AnomalyScoreRecoMeanPerVol' : [],
        'DiceScoreKLPerVol': [],
        'DiceScoreKLCombPerVol': [],
        'BestDiceKLCombPerVol': [],
        'BestDiceKLPerVol': [],
        'AUCKLCombPerVol': [],
        'AUPRCKLCombPerVol': [],
        'AUCKLPerVol': [],
        'AUPRCKLPerVol': [],
        'TPKLCombPerVol': [],
        'FPKLCombPerVol': [],
        'TNKLCombPerVol': [],
        'FNKLCombPerVol': [],
        'TPRKLCombPerVol': [],
        'FPRKLCombPerVol': [],
        'TPKLPerVol': [],
        'FPKLPerVol': [],
        'TNKLPerVol': [],
        'FNKLPerVol': [],
        'TPRKLPerVol': [],
        'FPRKLPerVol': [],

        'unsupervised_threshold': 0.0,
        'supervised_threshold': []



    }




    return _eval | all_keys






# From Zimmerer iterative algorithm for threshold search
def find_best_val(x, y, val_range=(0, 1), max_steps=4, step=0, max_val=0, max_point=0):  #x: Image , y: Label
    if step == max_steps:
        return max_val, max_point

    if val_range[0] == val_range[1]:
        val_range = (val_range[0], 1)

    bottom = val_range[0]
    top = val_range[1]
    center = bottom + (top - bottom) * 0.5

    q_bottom = bottom + (top - bottom) * 0.25
    q_top = bottom + (top - bottom) * 0.75
    val_bottom = dice(x > q_bottom, y)
    #print(str(np.mean(x>q_bottom)) + str(np.mean(y)))
    val_top = dice(x > q_top, y)
    #print(str(np.mean(x>q_top)) + str(np.mean(y)))
    #val_bottom = val_fn(x, y, q_bottom) # val_fn is the dice calculation dice(p, g)
    #val_top = val_fn(x, y, q_top)

    if val_bottom >= val_top:
        if val_bottom >= max_val:
            max_val = val_bottom
            max_point = q_bottom
        return find_best_val(x, y, val_range=(bottom, center), step=step + 1, max_steps=max_steps,
                             max_val=max_val, max_point=max_point)
    else:
        if val_top >= max_val:
            max_val = val_top
            max_point = q_top
        return find_best_val(x, y, val_range=(center, top), step=step + 1, max_steps=max_steps,
                             max_val=max_val,max_point=max_point)
def dice(P, G):
    psum = np.sum(P.flatten())
    gsum = np.sum(G.flatten())
    pgsum = np.sum(np.multiply(P.flatten(), G.flatten()))
    score = (2 * pgsum) / (psum + gsum)
    return score

    
def compute_roc(predictions, labels):
    _fpr, _tpr, _ = roc_curve(labels.astype(int), predictions,pos_label=1)
    roc_auc = auc(_fpr, _tpr)
    return roc_auc, _fpr, _tpr, _


def compute_prc(predictions, labels):
    precisions, recalls, thresholds = precision_recall_curve(labels.astype(int), predictions)
    auprc = average_precision_score(labels.astype(int), predictions)
    return auprc, precisions, recalls, thresholds   


def tpr(P, G):
    tp = np.sum(np.multiply(P.flatten(), G.flatten()))
    fn = np.sum(np.multiply(np.invert(P.flatten()), G.flatten()))
    return tp / (tp + fn)


def fpr(P, G):
    tp = np.sum(np.multiply(P.flatten(), G.flatten()))
    fp = np.sum(np.multiply(P.flatten(), np.invert(G.flatten())))
    return fp / (fp + tp)


def log_images(self, heatmap, data_orig, data_seg, ID):
    ImagePathList = {
                    'imagesGrid': os.path.join(os.getcwd(),'grid')}
    for key in ImagePathList :
        if not os.path.isdir(ImagePathList[key]):
            os.mkdir(ImagePathList[key])

    for j in range(0,heatmap.squeeze().shape[2],10) :

        num_plots = 3
        fig, ax = plt.subplots(1,num_plots,figsize=(16,4))
        # change spacing between subplots
        fig.subplots_adjust(wspace=0.0)
        index_counter = 0
        # orig
        ax[index_counter].imshow(data_orig.squeeze()[...,j].rot90(3),'gray')
        index_counter += 1
        # difference
        ax[index_counter].imshow(heatmap.squeeze()[:,...,j].rot90(3),'jet',norm=colors.Normalize(vmin=0, vmax=heatmap.max()+.01))
        index_counter += 1
        # mask
        ax[index_counter].imshow(data_seg.squeeze()[...,j].rot90(3),'gray')
        # remove all the ticks (both axes), and tick labels
        for axes in ax:
            axes.set_xticks([])
            axes.set_yticks([])
        # remove the frame of the chart
        for axes in ax:
            axes.spines['top'].set_visible(False)
            axes.spines['right'].set_visible(False)
            axes.spines['bottom'].set_visible(False)
            axes.spines['left'].set_visible(False)
        # remove the white space around the chart
        plt.tight_layout()
        
        if self.cfg.get('save_to_disc',True):
            plt.savefig(os.path.join(ImagePathList['imagesGrid'], '{}_{}_Grid.png'.format(ID[0],j)),bbox_inches='tight')
        self.logger.experiment.log({'images/{}/{}_Grid.png'.format(self.dataset[0],j) : wandb.Image(plt)})
        plt.clf()
        plt.cla()
        plt.close()