import os
import argparse
import time
import h5py
import pickle
from tqdm import tqdm
import numpy as np
#import matplotlib.pyplot as plt
from multiprocessing import Pool
from sklearn.metrics import average_precision_score, precision_recall_curve

import utils.io as io
from utils.bbox_utils import compute_iou
from datasets.hico_constants import HicoConstants

parser = argparse.ArgumentParser()
parser.add_argument(
    '--pred_hoi_dets_file', 
    type=str, 
    default='eval_result/pred_hoi_dets.hdf5',
    help='Path to predicted hoi detections hdf5 file')
parser.add_argument(
    '--out_dir', 
    type=str, 
    default='result/v2/map',
    help='Output directory')
parser.add_argument(
    '--proc_dir',
    type=str,
    default='/home/birl/ml_dl_projects/bigjun/hoi/no_frills_hoi_det/data_symlinks/hico_processed',
    help='Path to HICO processed data directory')
parser.add_argument(
    '--subset',
    type=str,
    default='test',
    choices=['train','test','val','train_val'],
    help='Subset of data to run the evaluation on')
parser.add_argument(
    '--num_processes',
    type=int,
    default=12,
    help='Number of processes to parallelize across')  

parser.add_argument('--exp_ver', '--e_v', type=str, default='v1', required=True,
                        help='the version of code, will create subdir in log/ && checkpoints/ ')


def match_hoi(pred_det,gt_dets):
    is_match = False
    remaining_gt_dets = [gt_det for gt_det in gt_dets]
    for i,gt_det in enumerate(gt_dets):
        human_iou = compute_iou(pred_det['human_box'],gt_det['human_box'])
        if human_iou > 0.5:
            object_iou = compute_iou(pred_det['object_box'],gt_det['object_box'])
            if object_iou > 0.5:
                is_match = True
                del remaining_gt_dets[i]
                break
        #remaining_gt_dets.append(gt_det)

    return is_match, remaining_gt_dets


def compute_ap(precision,recall):
    if np.any(np.isnan(recall)):
        return np.nan

    ap = 0
    for t in np.arange(0,1.1,0.1): # 0, 0.1, 0.2, ..., 1.0
        selected_p = precision[recall>=t]
        if selected_p.size==0:
            p = 0
        else:
            p = np.max(selected_p)   
        ap += p/11.
    
    return ap


def compute_pr(y_true,y_score,npos):
    sorted_y_true = [y for y,_ in 
        sorted(zip(y_true,y_score),key=lambda x: x[1],reverse=True)]
    tp = np.array(sorted_y_true)
    fp = ~tp
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    if npos==0:
        recall = np.nan*tp
    else:
        recall = tp / npos
    precision = tp / (tp + fp)
    return precision, recall


def compute_normalized_pr(y_true,y_score,npos,N=196.45):
    sorted_y_true = [y for y,_ in 
        sorted(zip(y_true,y_score),key=lambda x: x[1],reverse=True)]
    tp = np.array(sorted_y_true)
    fp = ~tp
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    if npos==0:
        recall = np.nan*tp
    else:
        recall = tp / npos
    precision = recall*N / (recall*N + fp)
    nap = np.sum(precision[sorted_y_true]) / (npos+1e-6)
    return precision, recall, nap


def eval_hoi(hoi_id,global_ids,gt_dets,pred_hoi_dets_file,out_dir):
    print(f'Evaluating hoi_id: {hoi_id} ...')
    pred_dets = h5py.File(pred_hoi_dets_file,'r')
    y_true = []
    y_score = []
    det_id = []
    npos = 0
    for global_id in global_ids:
        # if global_id == 'HICO_test2015_00000432':
        #     import ipdb; ipdb.set_trace()
        if hoi_id in gt_dets[global_id]:
            candidate_gt_dets = gt_dets[global_id][hoi_id]
        else:
            candidate_gt_dets = []
        npos += len(candidate_gt_dets)

        # start_id, end_id = pred_dets[global_id]['start_end_ids'][int(hoi_id)-1]
        # hoi_dets = \
        #     pred_dets[global_id]['human_obj_boxes_scores'][start_id:end_id]
        if global_id not in pred_dets.keys():
            continue
        if hoi_id not in pred_dets[global_id].keys():
            continue
        hoi_dets = pred_dets[global_id][hoi_id]
        num_dets = hoi_dets.shape[0]
        sorted_idx = [idx for idx,_ in sorted(
            zip(range(num_dets),hoi_dets[:,8].tolist()),
            key=lambda x: x[1],
            reverse=True)]
        for i in sorted_idx:
            pred_det = {
                'human_box': hoi_dets[i,:4],
                'object_box': hoi_dets[i,4:8],
                'score': hoi_dets[i,8]
            }
            is_match, candidate_gt_dets = match_hoi(pred_det,candidate_gt_dets)
            y_true.append(is_match)
            y_score.append(pred_det['score'])
            det_id.append((global_id,i))
    # Compute PR
    precision,recall = compute_pr(y_true,y_score,npos)
    #nprecision,nrecall,nap = compute_normalized_pr(y_true,y_score,npos)

    # Compute AP
    ap = compute_ap(precision,recall)
    print(f'AP:{ap}')

    # Plot PR curve
    # plt.figure()
    # plt.step(recall,precision,color='b',alpha=0.2,where='post')
    # plt.fill_between(recall,precision,step='post',alpha=0.2,color='b')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title('Precision-Recall curve: AP={0:0.4f}'.format(ap))
    # plt.savefig(
    #     os.path.join(out_dir,f'{hoi_id}_pr.png'),
    #     bbox_inches='tight')
    # plt.close()

    # Save AP data
    ap_data = {
        'y_true': y_true,
        'y_score': y_score,
        'det_id': det_id,
        'npos': npos,
        'ap': ap,
    }
    np.save(
        os.path.join(out_dir,f'{hoi_id}_ap_data.npy'),
        ap_data)

    return (ap,hoi_id)


def load_gt_dets(proc_dir,global_ids_set):
    # Load anno_list
    print('Loading anno_list.json ...')
    anno_list_json = os.path.join(proc_dir,'anno_list.json')
    anno_list = io.load_json_object(anno_list_json)

    gt_dets = {}
    for anno in anno_list:
        if anno['global_id'] not in global_ids_set:
            continue

        global_id = anno['global_id']
        gt_dets[global_id] = {}
        for hoi in anno['hois']:
            hoi_id = hoi['id']
            gt_dets[global_id][hoi_id] = []
            for human_box_num, object_box_num in hoi['connections']:
                human_box = hoi['human_bboxes'][human_box_num]
                object_box = hoi['object_bboxes'][object_box_num]
                det = {
                    'human_box': human_box,
                    'object_box': object_box,
                }
                gt_dets[global_id][hoi_id].append(det)

    return gt_dets

def main():
    args = parser.parse_args()

    data_const = HicoConstants(exp_ver=args.exp_ver)
    print('Creating output dir ...')
    io.mkdir_if_not_exists(data_const.result_dir+'/map',recursive=True)

    # Load hoi_list
    hoi_list_json = os.path.join(data_const.proc_dir,'hoi_list.json')
    hoi_list = io.load_json_object(hoi_list_json)

    # Load subset ids to eval on
    split_ids_json = os.path.join(data_const.proc_dir,'split_ids.json')
    split_ids = io.load_json_object(split_ids_json)
    global_ids = split_ids[args.subset]
    global_ids_set = set(global_ids)

    # Create gt_dets
    print('Creating GT dets ...')
    gt_dets = load_gt_dets(data_const.proc_dir,global_ids_set)

    eval_inputs = []
    for hoi in hoi_list:
        eval_inputs.append(
            (hoi['id'],global_ids,gt_dets, data_const.result_dir+'/pred_hoi_dets.hdf5', data_const.result_dir+'/map'))
    
    # import ipdb; ipdb.set_trace()
    # eval_hoi(*eval_inputs[0])

    print(f'Starting a pool of {args.num_processes} workers ...')
    p = Pool(args.num_processes)

    print(f'Begin mAP computation ...')
    output = p.starmap(eval_hoi,eval_inputs)
    #output = eval_hoi('003',global_ids,gt_dets,args.pred_hoi_dets_hdf5,args.out_dir)

    p.close()
    p.join()

    mAP = {
        'AP': {},
        'mAP': 0,
        'invalid': 0,
    }
    map_ = 0
    count = 0
    for ap,hoi_id in output:
        mAP['AP'][hoi_id] = ap
        if not np.isnan(ap):
            count += 1
            map_ += ap

    mAP['mAP'] = map_ / count
    mAP['invalid'] = len(output) - count

    mAP_json = os.path.join(
        data_const.result_dir+'/map',
        'mAP.json') 
    io.dump_json_object(mAP,mAP_json)

    print(f'APs have been saved to {data_const.result_dir}/map')

if __name__=='__main__':
    main()
    # data_const = HicoConstants()
    # data_path = data_const.result_dir+'/map'
    # ap_data_list = sorted(os.listdir(data_path))
    # mAP = {
    #     'AP': {},
    #     'mAP': 0,
    #     'invalid': 0,
    # }
    # map_ = 0
    # count = 0
    # for file in ap_data_list:
    #     if not file.endswith('.npy'): continue
    #     hoi_id = file.split("_")[0]
    #     data = np.load(os.path.join(data_path,file))
    #     mAP['AP'][hoi_id] = data.all()['ap']
    #     if not np.isnan(data.all()['ap']):
    #         count += 1
    #         map_ += data.all()['ap']

    # mAP['mAP'] = map_ / count
    # mAP['invalid'] = 600 - count

    # mAP_json = os.path.join(
    #     data_const.result_dir+'/map',
    #     'mAP.json') 
    # io.dump_json_object(mAP,mAP_json)

    # print(f'APs have been saved to {data_const.result_dir}/map')