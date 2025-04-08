import os
import re
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve,auc
from tqdm import tqdm
from dataset import shanghaitech_hr_skip
import csv


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def score_dataset(score, metadata, args=None, save_result=False, z=None, save_z=False):
    auc_roc , auc_precision_recall, eer, eer_threshold = 0, 0, 0, 0 

    if z.all() != None:
        gt_arr, scores_arr, z_arr = get_dataset_scores(score, metadata, args=args, save_result=save_result, z=z)
    else:
        gt_arr, scores_arr = get_dataset_scores(score, metadata, args=args, save_result=save_result)
    scores_arr = smooth_scores(scores_arr)
    gt_np = np.concatenate(gt_arr)
    scores_np = np.concatenate(scores_arr)
    z_np = np.concatenate(z_arr)
    # auc_roc, auc_precision_recall, eer, eer_threshold = score_auc(scores_np, gt_np)

    
    # if z.all() != None:
    #     gt_arr_z, z_arr = get_dataset_scores(z, metadata, args=args)
    #     z_arr = smooth_scores(z_arr)
    #     gt_np_z = np.concatenate(gt_arr_z)
    #     z_np = np.concatenate(z_arr)
    #     if save_z:
    #         directory = "shanghai_pose_test_z"
    #         if not os.path.exists(directory):
    #             os.makedirs(directory)
    #             with open(directory+"z"+".csv", 'w', newline='') as f:
    #                 writer = csv.writer(f)
    #                 writer.writerows(zip(gt_np_z, z_np))
    #     return auc_roc, scores_np, auc_precision_recall, eer, eer_threshold, z_np, gt_arr_z
    # else:
    if z.all() != None:
        return auc_roc, scores_np, auc_precision_recall, eer, eer_threshold, z_np, gt_np
    else:
        return auc_roc, scores_np, auc_precision_recall, eer, eer_threshold


def get_dataset_scores(scores, metadata, args=None, save_result=False, z=None):
    dataset_gt_arr = []
    dataset_z_arr = []
    dataset_scores_arr = []
    metadata_np = np.array(metadata)

    directory = "results_nf/"
    if save_result:
        if not os.path.exists(directory):
            os.makedirs(directory)
            
    if args.dataset == 'UBnormal':
        pose_segs_root = 'data/UBnormal/pose/test'
        clip_list = os.listdir(pose_segs_root)
        clip_list = sorted(
            fn.replace("alphapose_tracked_person.json", "tracks.txt") for fn in clip_list if fn.endswith('.json'))
        per_frame_scores_root = 'data/UBnormal/gt/'
    elif args.dataset == 'c1':
        per_frame_scores_root = 'data/c1/gt/test_frame_mask/'
        clip_list = os.listdir(per_frame_scores_root)
        clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))
    elif args.dataset == 'c2':
        per_frame_scores_root = 'data/c2/gt/test_frame_mask/'
        clip_list = os.listdir(per_frame_scores_root)
        clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))
    elif args.dataset == 'c3':
        per_frame_scores_root = 'data/c3/gt/test_frame_mask/'
        clip_list = os.listdir(per_frame_scores_root)
        clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))
    elif args.dataset == 'c4':
        per_frame_scores_root = 'data/c4/gt/test_frame_mask/'
        clip_list = os.listdir(per_frame_scores_root)
        clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))
    elif args.dataset == 'combined':
        per_frame_scores_root = 'data/combined/gt/test_frame_mask/'
        clip_list = os.listdir(per_frame_scores_root)
        clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))
    elif args.dataset == 'corridor':
        per_frame_scores_root = 'data/corridor/gt/test_frame_mask/'
        clip_list = os.listdir(per_frame_scores_root)
        clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))
    elif args.dataset == 'Avenue':
        per_frame_scores_root = 'data/Avenue/gt/test_frame_mask/'
        clip_list = os.listdir(per_frame_scores_root)
        clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))
    elif args.dataset == 'CPCC':
        per_frame_scores_root = 'data/CPCC/gt/test_frame_mask/'
        clip_list = os.listdir(per_frame_scores_root)
        clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))
    elif args.dataset == 'CPCC0':
        per_frame_scores_root = 'data/CPCC0/gt/test_frame_mask/'
        clip_list = os.listdir(per_frame_scores_root)
        clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))
    elif args.dataset == 'CPCC1':
        per_frame_scores_root = 'data/CPCC1/gt/test_frame_mask/'
        clip_list = os.listdir(per_frame_scores_root)
        clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))
    elif args.dataset == 'CPCC2':
        per_frame_scores_root = 'data/CPCC2/gt/test_frame_mask/'
        clip_list = os.listdir(per_frame_scores_root)
        clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))
    elif args.dataset == 'CPCC3':
        per_frame_scores_root = 'data/CPCC3/gt/test_frame_mask/'
        clip_list = os.listdir(per_frame_scores_root)
        clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))
    elif args.dataset == 'CPCC4':
        per_frame_scores_root = 'data/CPCC4/gt/test_frame_mask/'
        clip_list = os.listdir(per_frame_scores_root)
        clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))
    elif args.dataset == 'CPCC5':
        per_frame_scores_root = 'data/CPCC5/gt/test_frame_mask/'
        clip_list = os.listdir(per_frame_scores_root)
        clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))
    elif args.dataset == 'CPCC6':
        per_frame_scores_root = 'data/CPCC6/gt/test_frame_mask/'
        clip_list = os.listdir(per_frame_scores_root)
        clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))
    elif args.dataset == 'CPCC_X_setup':
        per_frame_scores_root = 'data/CPCC_X_setup/gt/test_frame_mask/'
        clip_list = os.listdir(per_frame_scores_root)
        clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))
    else:
        per_frame_scores_root = 'data/ShanghaiTech/gt/test_frame_mask/'
        clip_list = os.listdir(per_frame_scores_root)
        clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))
    clips = []
    print("Scoring {} clips".format(len(clip_list)))
    for clip in tqdm(clip_list):
        if z.all() != None:
            clip_gt, clip_score, clip_z = get_clip_score(scores, clip, metadata_np, metadata, per_frame_scores_root, args, z=z)
        else:
            clip_gt, clip_score = get_clip_score(scores, clip, metadata_np, metadata, per_frame_scores_root, args)
        if clip_score is not None:
            dataset_gt_arr.append(clip_gt)
            dataset_scores_arr.append(clip_score)
            clips.append(clip)
            if z.all() != None:
                dataset_z_arr.append(clip_z)
        
    if z.all() != None:
        z_np = np.concatenate(dataset_z_arr, axis=0)
        # z_np[z_np == np.inf] = z_np[z_np != np.inf].max()
        # z_np[z_np == -1 * np.inf] = z_np[z_np != -1 * np.inf].min()
        index = 0
        for z_ in range(len(dataset_z_arr)):
            for t in range(dataset_z_arr[z_].shape[0]):
                dataset_z_arr[z_][t] = z_np[index]
                index += 1
        
    scores_np = np.concatenate(dataset_scores_arr, axis=0)
    scores_np[scores_np == np.inf] = scores_np[scores_np != np.inf].max()
    scores_np[scores_np == -1 * np.inf] = scores_np[scores_np != -1 * np.inf].min()
    index = 0
    for score in range(len(dataset_scores_arr)):
        for t in range(dataset_scores_arr[score].shape[0]):
            dataset_scores_arr[score][t] = scores_np[index]
            index += 1

    if save_result:
        for clip_gt, clip_score, clip in zip (dataset_gt_arr, dataset_scores_arr, clips):
            with open(directory+clip.split('.')[0]+".csv", 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(zip(clip_gt, clip_score))

    if z.all() != None:
        return dataset_gt_arr, dataset_scores_arr, dataset_z_arr
    else:
        return dataset_gt_arr, dataset_scores_arr


def score_auc(scores_np, gt):
    scores_np[scores_np == np.inf] = scores_np[scores_np != np.inf].max()
    scores_np[scores_np == -1 * np.inf] = scores_np[scores_np != -1 * np.inf].min()
    auc_roc = roc_auc_score(gt, scores_np)
    precision, recall, thresholds = precision_recall_curve(gt, scores_np)
    auc_precision_recall = auc(recall, precision)
    fpr, tpr, threshold = roc_curve(gt, scores_np, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return auc_roc, auc_precision_recall, eer, eer_threshold


def smooth_scores(scores_arr, sigma=7):
    for s in range(len(scores_arr)):
        for sig in range(1, sigma):
            scores_arr[s] = gaussian_filter1d(scores_arr[s], sigma=sig)
    return scores_arr


def get_clip_score(scores, clip, metadata_np, metadata, per_frame_scores_root, args, z=None):
    if args.dataset == 'UBnormal':
        type, scene_id, clip_id = re.findall('(abnormal|normal)_scene_(\d+)_scenario(.*)_tracks.*', clip)[0]
        clip_id = type + "_" + clip_id
    elif args.dataset=='c1' or args.dataset=='c2' or args.dataset=='c3' or args.dataset=='c4' or args.dataset=='combined' or args.dataset=='corridor' or args.dataset=='Avenue' or args.dataset=='CPCC_X_setup' or args.dataset=='CPCC' or args.dataset=='CPCC0' or args.dataset=='CPCC1' or args.dataset=='CPCC2' or args.dataset=='CPCC3' or args.dataset=='CPCC4' or args.dataset=='CPCC5' or args.dataset=='CPCC6':
        scene_id, clip_id = [int(i) for i in clip.replace("label", "001").split('.')[0].split('_')][:2]
    else:
        scene_id, clip_id = [int(i) for i in clip.replace("label", "001").split('.')[0].split('_')]
        if shanghaitech_hr_skip((args.dataset == 'ShanghaiTech-HR'), scene_id, clip_id):
            return None, None
    clip_metadata_inds = np.where((metadata_np[:, 1] == clip_id) &
                                  (metadata_np[:, 0] == scene_id))[0]
    clip_metadata = metadata[clip_metadata_inds]
    clip_fig_idxs = set([arr[2] for arr in clip_metadata])
    clip_res_fn = os.path.join(per_frame_scores_root, clip)
    clip_gt = np.load(clip_res_fn)
    if args.dataset != "UBnormal":
        clip_gt = np.ones(clip_gt.shape) - clip_gt  # 1 is normal, 0 is abnormal
    if z.all() != None:
        z_zeros = np.ones((clip_gt.shape[0], z[0].shape[0], z[0].shape[1], z[0].shape[2])) * np.inf
    scores_zeros = np.ones(clip_gt.shape[0]) * np.inf
    if len(clip_fig_idxs) == 0:
        clip_person_scores_dict = {0: np.copy(scores_zeros)}
        if z.all() != None:
            clip_person_z_dict = {0: np.copy(z_zeros)}
    else:
        clip_person_scores_dict = {i: np.copy(scores_zeros) for i in clip_fig_idxs}
        if z.all() != None:
            clip_person_z_dict = {i: np.copy(z_zeros) for i in clip_fig_idxs}

    for person_id in clip_fig_idxs:
        person_metadata_inds = \
            np.where(
                (metadata_np[:, 1] == clip_id) & (metadata_np[:, 0] == scene_id) & (metadata_np[:, 2] == person_id))[0]
        pid_scores = scores[person_metadata_inds]
        if z.all() != None:
            pid_z = z[person_metadata_inds]

        pid_frame_inds = np.array([metadata[i][3] for i in person_metadata_inds]).astype(int)
        clip_person_scores_dict[person_id][pid_frame_inds + int(args.seg_len / 2)] = pid_scores
        if z.all() != None:
            clip_person_z_dict[person_id][pid_frame_inds + int(args.seg_len / 2)] = pid_z

    clip_ppl_score_arr = np.stack(list(clip_person_scores_dict.values()))
    if z.all() != None:
        clip_ppl_z_arr = np.stack(list(clip_person_z_dict.values()))
    clip_score = np.amin(clip_ppl_score_arr, axis=0)
    if z.all() != None:
        clip_score_index = np.argmin(clip_ppl_score_arr, axis=0)
        clip_z = []
        for frame in range (clip_gt.shape[0]):
            clip_z.append(clip_ppl_z_arr[clip_score_index[frame]][frame])
        return  clip_gt, clip_score, np.array(clip_z)
    else:
        return clip_gt, clip_score

def calc_distance(z_arr, mean_arr):
    distance = np.sqrt(((z_arr - mean_arr) ** 2).sum(axis=(1,2,3)))
    return distance

def save_dist (gt, dist, directory):
    # directory = "shanghai_pose_test_z"
    if not os.path.exists(directory):
        os.makedirs(directory)
        with open(directory+".csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["GT", "Distances"])
            writer.writerows(zip(gt, dist))