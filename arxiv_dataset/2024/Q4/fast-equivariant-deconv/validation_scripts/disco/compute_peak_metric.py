import nibabel as nib
import numpy as np
import argparse
import os
from sklearn.metrics import auc
import yaml

parser = argparse.ArgumentParser()
parser.add_argument(
    '--config',
    required=True,
    help='Path of the config file (default: None)',
    type=str
)
args = parser.parse_args()
config_path = args.config
config = yaml.safe_load(open(config_path, 'r'))
mask = nib.load(f"{config['data']['data_path']}/mask.nii.gz")
affine = mask.affine
header = mask.header
mask = mask.get_fdata()

save_all = False
gt_names = ['stream'] 

data_path = config['data']['data_path']
data_path_gt = data_path
while data_path_gt[-1] == '/':
    data_path_gt = data_path_gt[:-1]
data_path_gt = '/'.join(data_path_gt.split('/')[:-1])
print(data_path_gt)
model_name = config['testing']['expname']
path = f'{data_path}/result/{model_name}/peaks_mrtrix.nii.gz'

for i in range(len(gt_names)):
    gt_name = gt_names[i]
    gt_path = f'{data_path_gt}/peaks_unmask_normed_relative_{gt_name}.nii.gz'
    gt_n_path = f'{data_path_gt}/n_peaks_unmask_{gt_name}.nii.gz'
    if not os.path.exists(f"{data_path}/result/{model_name}/{gt_name}"):
        os.makedirs(f"{data_path}/result/{model_name}/{gt_name}")

    gt_direction = nib.load(gt_path).get_fdata()
    gt_num_direction = nib.load(gt_n_path).get_fdata()
    #print(gt_num_direction[20, 20, 20])
    #print(gt_direction[20, 20, 20].reshape(10, 3))
    #print(np.linalg.norm(gt_direction[20, 20, 20].reshape(10, 3), axis=-1))

    m_direction = nib.load(path).get_fdata()
    m_direction[np.isnan(m_direction)] = 0
    m_direction = m_direction.reshape((*mask.shape, 10, 3))

    # Filter
    m_direction_masked = m_direction[mask>0].reshape((-1, 10, 3))
    m_norm_masked = np.linalg.norm(m_direction_masked, axis=-1)
    m_direction_masked_normed = m_direction_masked / (m_norm_masked[..., None] + 1e-16)
    m_norm_masked_rel = m_norm_masked / (np.max(m_norm_masked, axis=-1, keepdims=True) + 1e-16)
    m_direction_masked_normed_rel = m_direction_masked_normed * m_norm_masked_rel[..., None]

    # Sort
    order = np.argsort(m_norm_masked_rel, axis=-1)[:, ::-1]
    m_norm_masked_rel_sorted = np.sort(m_norm_masked_rel, axis=-1)[:, ::-1]
    m_direction_masked_normed_rel_sorted = np.vstack([m_direction_masked_normed_rel[i, order[i]][None] for i in range(m_direction_masked_normed_rel.shape[0])])

    # Save
    m_norm_masked_rel_sorted_unmask = np.zeros((*mask.shape, 10))
    m_direction_masked_normed_rel_sorted_unmask = np.zeros((*mask.shape, 10*3))
    m_norm_masked_rel_sorted_unmask[mask>0] = m_norm_masked_rel_sorted
    m_direction_masked_normed_rel_sorted_unmask[mask>0] = m_direction_masked_normed_rel_sorted.reshape((-1, 10*3))

    # Start validation
    #print(m_norm_masked_rel_sorted_unmask[20, 20, 20])
    #print(m_direction_masked_normed_rel_sorted_unmask[20, 20, 20].reshape(10, 3))
    #print(np.linalg.norm(m_direction_masked_normed_rel_sorted_unmask[20, 20, 20].reshape(10, 3), axis=-1))
    m_norm = m_norm_masked_rel_sorted_unmask
    m_direction = m_direction_masked_normed_rel_sorted_unmask
    relative_peak_thresholds = np.arange(0.05,1,0.05)

    angular_error_all = np.zeros((*mask.shape, len(relative_peak_thresholds)))
    success_rate_all = np.zeros((*mask.shape, len(relative_peak_thresholds)))
    over_estimated_fiber_all = np.zeros((*mask.shape, len(relative_peak_thresholds)))
    over_estimated_fiber_total_all = np.zeros((*mask.shape, len(relative_peak_thresholds)))
    under_estimated_fiber_all = np.zeros((*mask.shape, len(relative_peak_thresholds)))
    under_estimated_fiber_total_all = np.zeros((*mask.shape, len(relative_peak_thresholds)))
    FP_fiber_all = np.zeros((*mask.shape, len(relative_peak_thresholds)))
    FN_fiber_all = np.zeros((*mask.shape, len(relative_peak_thresholds)))
    TP_fiber_all = np.zeros((*mask.shape, len(relative_peak_thresholds)))
    SUCCESS_fiber_all = np.zeros((*mask.shape, len(relative_peak_thresholds)))


    for thres_id, relative_peak_threshold in enumerate(relative_peak_thresholds):
        print(relative_peak_threshold, end='\r')
        m_num_direction = np.sum(m_norm > relative_peak_threshold, axis=-1)

        angular_error = np.zeros(mask.shape)
        success_rate = np.zeros(mask.shape)
        over_estimated_fiber = np.zeros(mask.shape)
        over_estimated_fiber_total = np.zeros(mask.shape)
        under_estimated_fiber = np.zeros(mask.shape)
        under_estimated_fiber_total = np.zeros(mask.shape)
        TP_fiber = np.zeros(mask.shape)
        FN_fiber = np.zeros(mask.shape)
        FP_fiber = np.zeros(mask.shape)
        SUCCESS_fiber = np.zeros(mask.shape)

        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                for k in range(mask.shape[2]):
                    if mask[i,j,k]==1 and gt_num_direction[i,j,k]!=0:
                        sticks_gt = gt_direction[i,j,k][:int(3*gt_num_direction[i,j,k])].reshape((int(gt_num_direction[i,j,k]), 3))
                        if m_num_direction[i,j,k] != 0:
                            # Prediction voxel i
                            sticks_pred = m_direction[i,j,k][:int(3*m_num_direction[i,j,k])].reshape((int(m_num_direction[i,j,k]), 3))
                            # Prediction and ground truth angles
                            norm_gt = np.linalg.norm(sticks_gt, axis=-1)
                            norm_pred = np.linalg.norm(sticks_pred, axis=-1)
                            norm_predgt = norm_pred[:, None].dot(norm_gt[None, :])
                            angle = np.arccos(np.minimum(np.abs(sticks_pred.dot(sticks_gt.T))/norm_predgt, 1))
                            # angle = np.minimum(angle, np.pi - angle)
                            if np.sum(angle < 0) != 0:
                                print("ERROR")
                            gt_to_pred = np.min(angle, 0)
                            pred_to_gt = np.min(angle, 1)

                            # Anguler error: mean angular error for each gt fiber to the closest prediction
                            angular_error[i,j,k] = np.mean(gt_to_pred)

                            # TP, FP, FN
                            gt_match = np.argmin(angle, 1)[np.argmin(angle, 0)] == np.arange(angle.shape[1])
                            pred_match = np.argmin(angle, 0)[np.argmin(angle, 1)] == np.arange(angle.shape[0])
                            TP_gt = np.sum(np.min(angle, 0)[gt_match] * 180 / np.pi < 25)
                            TP_pred = np.sum(np.min(angle, 1)[pred_match] * 180 / np.pi < 25)
                            assert TP_gt == TP_pred
                            TP = TP_gt
                            FN = gt_match.shape[0] - TP_gt
                            FP = pred_match.shape[0] - TP_pred

                            TP_fiber[i, j, k] = TP
                            FN_fiber[i, j, k] = FN
                            FP_fiber[i, j, k] = FP
                            SUCCESS_fiber[i, j, k] = (FN==0)*(FP==0)


                            # Success rate: voxel is well classify if each gt fiber (resp. each prediction)
                            #  is near a prediction (resp. a gt)
                            if len(sticks_gt) == len(sticks_pred):
                                if (np.sum(gt_to_pred * 180 / np.pi < 25) == len(sticks_gt)) and (
                                        np.sum(pred_to_gt * 180 / np.pi < 25) == len(sticks_pred)):
                                    success_rate[i,j,k] = 1
                                else:
                                    success_rate[i,j,k] = 0
                            else:
                                success_rate[i,j,k] = 0

                            #if SUCCESS_fiber[i, j, k] != success_rate[i,j,k]:
                            #    print(angle, TP, FN, FP, SUCCESS_fiber[i, j, k], success_rate[i, j, k])

                            # False positive prediction (spurious fibers): predicted fibers which are not
                            #  closer than 25° from a gt fiber
                            over_estimated_fiber[i,j,k] = np.sum(pred_to_gt * 180 / np.pi >= 25)
                            over_estimated_fiber_total[i,j,k] = max(0, len(pred_to_gt) - len(gt_to_pred))
                            # False negative prediction: gt fibers which are not closer than 25° from a predicted fiber
                            under_estimated_fiber[i,j,k] = np.sum(gt_to_pred * 180 / np.pi >= 25)
                            under_estimated_fiber_total[i,j,k] = max(0, len(gt_to_pred) - len(pred_to_gt))

                        else:
                            angular_error[i,j,k] = np.pi/2
                            success_rate[i,j,k] = 0

                            over_estimated_fiber[i,j,k] = 0
                            over_estimated_fiber_total[i,j,k] = 0

                            under_estimated_fiber[i,j,k] = len(sticks_gt)
                            under_estimated_fiber_total[i,j,k] = len(sticks_gt)

                            TP_fiber[i, j, k] = 0
                            FN_fiber[i, j, k] = len(sticks_gt)
                            FP_fiber[i, j, k] = 0
                            SUCCESS_fiber[i, j, k] = 0
                            
        angular_error_all[:, :, :, thres_id] = angular_error * 180 / np.pi
        success_rate_all[:, :, :, thres_id] = success_rate
        over_estimated_fiber_all[:, :, :, thres_id] = over_estimated_fiber
        over_estimated_fiber_total_all[:, :, :, thres_id] = over_estimated_fiber_total
        under_estimated_fiber_all[:, :, :, thres_id] = under_estimated_fiber
        under_estimated_fiber_total_all[:, :, :, thres_id] = under_estimated_fiber_total
        FP_fiber_all[:, :, :, thres_id] = FP_fiber
        FN_fiber_all[:, :, :, thres_id] = FN_fiber
        TP_fiber_all[:, :, :, thres_id] = TP_fiber
        SUCCESS_fiber_all[:, :, :, thres_id] = SUCCESS_fiber

    if save_all:

        img = nib.Nifti1Image(success_rate_all, affine, header)
        nib.save(img, f"{data_path}/result/{model_name}/{gt_name}/success_rate_all.nii.gz")

        img = nib.Nifti1Image(over_estimated_fiber_all, affine, header)
        nib.save(img, f"{data_path}/result/{model_name}/{gt_name}/over_estimated_fiber_all.nii.gz")

        img = nib.Nifti1Image(over_estimated_fiber_total_all, affine, header)
        nib.save(img, f"{data_path}/result/{model_name}/{gt_name}/over_estimated_fiber_total_all.nii.gz")

        img = nib.Nifti1Image(under_estimated_fiber_all, affine, header)
        nib.save(img, f"{data_path}/result/{model_name}/{gt_name}/under_estimated_fiber_all.nii.gz")

        img = nib.Nifti1Image(under_estimated_fiber_total_all, affine, header)
        nib.save(img, f"{data_path}/result/{model_name}/{gt_name}/under_estimated_fiber_total_all.nii.gz")

    
    img = nib.Nifti1Image(angular_error_all, affine, header)
    nib.save(img, f"{data_path}/result/{model_name}/{gt_name}/angular_error_all.nii.gz")
    
    img = nib.Nifti1Image(FP_fiber_all, affine, header)
    nib.save(img, f"{data_path}/result/{model_name}/{gt_name}/FP_fiber_all.nii.gz")

    img = nib.Nifti1Image(FN_fiber_all, affine, header)
    nib.save(img, f"{data_path}/result/{model_name}/{gt_name}/FN_fiber_all.nii.gz")

    img = nib.Nifti1Image(TP_fiber_all, affine, header)
    nib.save(img, f"{data_path}/result/{model_name}/{gt_name}/TP_fiber_all.nii.gz")

    img = nib.Nifti1Image(SUCCESS_fiber_all, affine, header)
    nib.save(img, f"{data_path}/result/{model_name}/{gt_name}/SUCCESS_fiber_all.nii.gz")

    angular_error_all_mean = np.mean(angular_error_all[(mask>0)*(gt_num_direction>0)], axis=0)
    np.save(f"{data_path}/result/{model_name}/{gt_name}/angular_error_all_mean.npy", angular_error_all_mean)
    if save_all:
        success_rate_all_mean = np.mean(success_rate_all[(mask>0)*(gt_num_direction>0)], axis=0)
        np.save(f"{data_path}/result/{model_name}/{gt_name}/success_rate_all_mean.npy", success_rate_all_mean)
        over_estimated_fiber_all_mean = np.mean(over_estimated_fiber_all[(mask>0)*(gt_num_direction>0)], axis=0)
        np.save(f"{data_path}/result/{model_name}/{gt_name}/over_estimated_fiber_all_mean.npy", over_estimated_fiber_all_mean)
        over_estimated_fiber_total_all_mean = np.mean(over_estimated_fiber_total_all[(mask>0)*(gt_num_direction>0)], axis=0)
        np.save(f"{data_path}/result/{model_name}/{gt_name}/over_estimated_fiber_total_all_mean.npy", over_estimated_fiber_total_all_mean)
        under_estimated_fiber_all_mean = np.mean(under_estimated_fiber_all[(mask>0)*(gt_num_direction>0)], axis=0)
        np.save(f"{data_path}/result/{model_name}/{gt_name}/under_estimated_fiber_all_mean.npy", under_estimated_fiber_all_mean)
        under_estimated_fiber_total_all_mean = np.mean(under_estimated_fiber_total_all[(mask>0)*(gt_num_direction>0)], axis=0)
        np.save(f"{data_path}/result/{model_name}/{gt_name}/under_estimated_fiber_total_all_mean.npy", under_estimated_fiber_total_all_mean)

    FP_fiber_all_sum = np.sum(FP_fiber_all[(mask>0)*(gt_num_direction>0)], axis=0)
    np.save(f"{data_path}/result/{model_name}/{gt_name}/FP_fiber_all_sum.npy", FP_fiber_all_sum)
    FN_fiber_all_sum = np.sum(FN_fiber_all[(mask>0)*(gt_num_direction>0)], axis=0)
    np.save(f"{data_path}/result/{model_name}/{gt_name}/FN_fiber_all_sum.npy", FN_fiber_all_sum)
    TP_fiber_all_sum = np.sum(TP_fiber_all[(mask>0)*(gt_num_direction>0)], axis=0)
    np.save(f"{data_path}/result/{model_name}/{gt_name}/TP_fiber_all_sum.npy", TP_fiber_all_sum)
    SUCCESS_fiber_all_mean = np.mean(SUCCESS_fiber_all[(mask>0)*(gt_num_direction>0)], axis=0)
    np.save(f"{data_path}/result/{model_name}/{gt_name}/SUCCESS_fiber_all_mean.npy", SUCCESS_fiber_all_mean)

    precision = TP_fiber_all_sum / (TP_fiber_all_sum + FP_fiber_all_sum)
    recall = TP_fiber_all_sum / (TP_fiber_all_sum + FN_fiber_all_sum) # same as TPR
    auc_score = auc(np.array([1] + recall.tolist() + [0]), np.array([0] + precision.tolist() + [1]))
    print('AUC: ', auc_score)
    print('Mean angular: ', np.mean(angular_error_all_mean))
    np.save(f"{data_path}/result/{model_name}/{gt_name}/auc_score.npy", auc_score)
    f1_score = np.max(2*precision*recall/(precision+recall))
    f1_index = np.argmax(2*precision*recall/(precision+recall))
    print('F1-Score: ', f1_score, ' at threshold ', relative_peak_thresholds[f1_index])
    print('Angular Error at F1-Score: ', angular_error_all_mean[f1_index])
    np.save(f"{data_path}/result/{model_name}/{gt_name}/f1_score.npy", f1_score)
    np.save(f"{data_path}/result/{model_name}/{gt_name}/f1_index.npy", f1_index)
    np.save(f"{data_path}/result/{model_name}/{gt_name}/angular_error_f1index.npy", angular_error_all_mean[f1_index])

    if save_all:
        for f in np.unique(gt_num_direction):
            if f>0:
                angular_error_all_mean = np.mean(angular_error_all[(mask>0)*(gt_num_direction==f)], axis=0)
                np.save(f"{data_path}/result/{model_name}/{gt_name}/angular_error_all_mean_{int(f)}.npy", angular_error_all_mean)
                success_rate_all_mean = np.mean(success_rate_all[(mask>0)*(gt_num_direction==f)], axis=0)
                np.save(f"{data_path}/result/{model_name}/{gt_name}/success_rate_all_mean_{int(f)}.npy", success_rate_all_mean)
                over_estimated_fiber_all_mean = np.mean(over_estimated_fiber_all[(mask>0)*(gt_num_direction==f)], axis=0)
                np.save(f"{data_path}/result/{model_name}/{gt_name}/over_estimated_fiber_all_mean_{int(f)}.npy", over_estimated_fiber_all_mean)
                over_estimated_fiber_total_all_mean = np.mean(over_estimated_fiber_total_all[(mask>0)*(gt_num_direction==f)], axis=0)
                np.save(f"{data_path}/result/{model_name}/{gt_name}/over_estimated_fiber_total_all_mean_{int(f)}.npy", over_estimated_fiber_total_all_mean)
                under_estimated_fiber_all_mean = np.mean(under_estimated_fiber_all[(mask>0)*(gt_num_direction==f)], axis=0)
                np.save(f"{data_path}/result/{model_name}/{gt_name}/under_estimated_fiber_all_mean_{int(f)}.npy", under_estimated_fiber_all_mean)
                under_estimated_fiber_total_all_mean = np.mean(under_estimated_fiber_total_all[(mask>0)*(gt_num_direction==f)], axis=0)
                np.save(f"{data_path}/result/{model_name}/{gt_name}/under_estimated_fiber_total_all_mean_{int(f)}.npy", under_estimated_fiber_total_all_mean)

                FP_fiber_all_sum = np.sum(FP_fiber_all[(mask>0)*(gt_num_direction==f)], axis=0)
                np.save(f"{data_path}/result/{model_name}/{gt_name}/FP_fiber_all_mean_{int(f)}.npy", FP_fiber_all_sum)
                FN_fiber_all_sum = np.sum(FN_fiber_all[(mask>0)*(gt_num_direction==f)], axis=0)
                np.save(f"{data_path}/result/{model_name}/{gt_name}/FN_fiber_all_mean_{int(f)}.npy", FN_fiber_all_sum)
                TP_fiber_all_sum = np.sum(TP_fiber_all[(mask>0)*(gt_num_direction==f)], axis=0)
                np.save(f"{data_path}/result/{model_name}/{gt_name}/TP_fiber_all_mean_{int(f)}.npy", TP_fiber_all_sum)
                SUCCESS_fiber_all_mean = np.mean(SUCCESS_fiber_all[(mask>0)*(gt_num_direction==f)], axis=0)
                np.save(f"{data_path}/result/{model_name}/{gt_name}/SUCCESS_fiber_all_mean_{int(f)}.npy", SUCCESS_fiber_all_mean)

                precision = TP_fiber_all_sum / (TP_fiber_all_sum + FP_fiber_all_sum)
                recall = TP_fiber_all_sum / (TP_fiber_all_sum + FN_fiber_all_sum) # same as TPR
                auc_score = auc(np.array([1] + recall.tolist() + [0]), np.array([0] + precision.tolist() + [1]))
                np.save(f"{data_path}/result/{model_name}/{gt_name}/auc_score_{int(f)}.npy", auc_score)
                f1_score = np.max(2*precision*recall/(precision+recall))
                f1_index = np.argmax(2*precision*recall/(precision+recall))
                np.save(f"{data_path}/result/{model_name}/{gt_name}/f1_score_{int(f)}.npy", f1_score)
                np.save(f"{data_path}/result/{model_name}/{gt_name}/f1_index_{int(f)}.npy", f1_index)
                np.save(f"{data_path}/result/{model_name}/{gt_name}/angular_error_f1index_{int(f)}.npy", angular_error_all_mean[f1_index])


    print(f'------------------------- {gt_name} -------------------------')
    print(f'Angular error: {np.mean(angular_error_all[(mask>0)*(gt_num_direction>0)]): .6f}')
    print(f'Angular error thres = 0.1: {np.mean(angular_error_all[(mask>0)*(gt_num_direction>0)][:, np.arange(len(relative_peak_thresholds))[relative_peak_thresholds==0.1][0]]): .6f}')
    print(f'Angular error thres = 0.2: {np.mean(angular_error_all[(mask>0)*(gt_num_direction>0)][:, np.arange(len(relative_peak_thresholds))[relative_peak_thresholds==0.2][0]]): .6f}')
    print(f'Angular error thres = 0.5: {np.mean(angular_error_all[(mask>0)*(gt_num_direction>0)][:, np.arange(len(relative_peak_thresholds))[relative_peak_thresholds==0.5][0]]): .6f}')
    print(f'Angular error thres = 0.8: {np.mean(angular_error_all[(mask>0)*(gt_num_direction>0)][:, np.arange(len(relative_peak_thresholds))[relative_peak_thresholds==0.8][0]]): .6f}')

    print(f'Success rate: {np.mean(success_rate_all[(mask>0)*(gt_num_direction>0)]): .6f}')
    print(f'Success rate thres = 0.1: {np.mean(success_rate_all[(mask>0)*(gt_num_direction>0)][:, np.arange(len(relative_peak_thresholds))[relative_peak_thresholds==0.1][0]]): .6f}')
    print(f'Success rate thres = 0.2: {np.mean(success_rate_all[(mask>0)*(gt_num_direction>0)][:, np.arange(len(relative_peak_thresholds))[relative_peak_thresholds==0.2][0]]): .6f}')
    print(f'Success rate thres = 0.5: {np.mean(success_rate_all[(mask>0)*(gt_num_direction>0)][:, np.arange(len(relative_peak_thresholds))[relative_peak_thresholds==0.5][0]]): .6f}')
    print(f'Success rate thres = 0.8: {np.mean(success_rate_all[(mask>0)*(gt_num_direction>0)][:, np.arange(len(relative_peak_thresholds))[relative_peak_thresholds==0.8][0]]): .6f}')

    print(f'Over estimated fiber: {np.mean(over_estimated_fiber_all[(mask>0)*(gt_num_direction>0)]): .6f}')
    print(f'Over estimated fiber thres = 0.1: {np.mean(over_estimated_fiber_all[(mask>0)*(gt_num_direction>0)][:, np.arange(len(relative_peak_thresholds))[relative_peak_thresholds==0.1][0]]): .6f}')
    print(f'Over estimated fiber thres = 0.2: {np.mean(over_estimated_fiber_all[(mask>0)*(gt_num_direction>0)][:, np.arange(len(relative_peak_thresholds))[relative_peak_thresholds==0.2][0]]): .6f}')
    print(f'Over estimated fiber thres = 0.5: {np.mean(over_estimated_fiber_all[(mask>0)*(gt_num_direction>0)][:, np.arange(len(relative_peak_thresholds))[relative_peak_thresholds==0.5][0]]): .6f}')
    print(f'Over estimated fiber thres = 0.8: {np.mean(over_estimated_fiber_all[(mask>0)*(gt_num_direction>0)][:, np.arange(len(relative_peak_thresholds))[relative_peak_thresholds==0.8][0]]): .6f}')

    print(f'Under estimated fiber: {np.mean(under_estimated_fiber_all[(mask>0)*(gt_num_direction>0)]): .6f}')
    print(f'Under estimated fiber thres = 0.1: {np.mean(under_estimated_fiber_all[(mask>0)*(gt_num_direction>0)][:, np.arange(len(relative_peak_thresholds))[relative_peak_thresholds==0.1][0]]): .6f}')
    print(f'Under estimated fiber thres = 0.2: {np.mean(under_estimated_fiber_all[(mask>0)*(gt_num_direction>0)][:, np.arange(len(relative_peak_thresholds))[relative_peak_thresholds==0.2][0]]): .6f}')
    print(f'Under estimated fiber thres = 0.5: {np.mean(under_estimated_fiber_all[(mask>0)*(gt_num_direction>0)][:, np.arange(len(relative_peak_thresholds))[relative_peak_thresholds==0.5][0]]): .6f}')
    print(f'Under estimated fiber thres = 0.8: {np.mean(under_estimated_fiber_all[(mask>0)*(gt_num_direction>0)][:, np.arange(len(relative_peak_thresholds))[relative_peak_thresholds==0.8][0]]): .6f}')

    print(f'FP fiber: {np.mean(FP_fiber_all[(mask>0)*(gt_num_direction>0)]): .6f}')
    print(f'FP fiber thres = 0.1: {np.mean(FP_fiber_all[(mask>0)*(gt_num_direction>0)][:, np.arange(len(relative_peak_thresholds))[relative_peak_thresholds==0.1][0]]): .6f}')
    print(f'FP fiber thres = 0.2: {np.mean(FP_fiber_all[(mask>0)*(gt_num_direction>0)][:, np.arange(len(relative_peak_thresholds))[relative_peak_thresholds==0.2][0]]): .6f}')
    print(f'FP fiber thres = 0.5: {np.mean(FP_fiber_all[(mask>0)*(gt_num_direction>0)][:, np.arange(len(relative_peak_thresholds))[relative_peak_thresholds==0.5][0]]): .6f}')
    print(f'FP fiber thres = 0.8: {np.mean(FP_fiber_all[(mask>0)*(gt_num_direction>0)][:, np.arange(len(relative_peak_thresholds))[relative_peak_thresholds==0.8][0]]): .6f}')

    print(f'FN fiber: {np.mean(FN_fiber_all[(mask>0)*(gt_num_direction>0)]): .6f}')
    print(f'FN fiber thres = 0.1: {np.mean(FN_fiber_all[(mask>0)*(gt_num_direction>0)][:, np.arange(len(relative_peak_thresholds))[relative_peak_thresholds==0.1][0]]): .6f}')
    print(f'FN fiber thres = 0.2: {np.mean(FN_fiber_all[(mask>0)*(gt_num_direction>0)][:, np.arange(len(relative_peak_thresholds))[relative_peak_thresholds==0.2][0]]): .6f}')
    print(f'FN fiber thres = 0.5: {np.mean(FN_fiber_all[(mask>0)*(gt_num_direction>0)][:, np.arange(len(relative_peak_thresholds))[relative_peak_thresholds==0.5][0]]): .6f}')
    print(f'FN fiber thres = 0.8: {np.mean(FN_fiber_all[(mask>0)*(gt_num_direction>0)][:, np.arange(len(relative_peak_thresholds))[relative_peak_thresholds==0.8][0]]): .6f}')

