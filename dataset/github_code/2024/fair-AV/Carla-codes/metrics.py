import datetime
import time
import numpy as np
import argparse
import os

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import torch


def accumulate2(coco_eval, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not coco_eval.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = coco_eval.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(p.iouThrs)
        R           = len(p.recThrs)
        K           = len(p.catIds) if p.useCats else 1
        A           = len(p.areaRng)
        M           = len(p.maxDets)
        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
        recall      = -np.ones((T,K,A,M))
        tp_vals     = -np.ones((T,K,A,M)) # add
        fp_vals     = -np.ones((T,K,A,M))
        fn_vals     = -np.ones((T,K,A,M))
        tp_scores   = -np.ones((T,K,A,M))
        var_tp_scores   = -np.ones((T,K,A,M))
        fp_scores   = -np.ones((T,K,A,M))
        fn_scores   = -np.ones((T,K,A,M))
        scores      = -np.ones((T,R,K,A,M))

        # create dictionary for future indexing
        _pe = coco_eval._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0*A0*I0
            for a, a0 in enumerate(a_list):
                Na = a0*I0
                for m, maxDet in enumerate(m_list):
                    E = [coco_eval.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                    dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg==0 )
                    if npig == 0:
                        continue
                    tps = np.logical_and(               dtm,  np.logical_not(dtIg) ) # dtm and dtIg
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        fn = npig - tp # add
                        tp_score = dtScoresSorted * tps[t, :].astype(dtype=float)
                        fp_score = dtScoresSorted * fps[t, :].astype(dtype=float)
                        fn_score = dtScoresSorted * np.logical_not(tps[t, :]).astype(dtype=float)
                        nd = len(tp)
                        rc = tp / npig # whats npig
                        pr = tp / (fp+tp+np.spacing(1))
                        q  = np.zeros((R,))
                        ss = np.zeros((R,))

                        if nd:
                            recall[t,k,a,m] = rc[-1]
                            tp_vals[t,k,a,m] = tp[-1] # add
                            fp_vals[t,k,a,m] = fp[-1]
                            fn_vals[t,k,a,m] = fn[-1]
                            tp_scores[t,k,a,m] = np.sum(tp_score) # get std of only the non zero elements
                            var_tp_scores[t,k,a,m] = np.std(tp_score[tp_score>0])
                            fp_scores[t,k,a,m] = np.sum(fp_score) 
                            fn_scores[t,k,a,m] = np.sum(fn_score) 
                        else:
                            recall[t,k,a,m] = 0
                            tp_vals[t,k,a,m] = 0 # add
                            fp_vals[t,k,a,m] = 0
                            fn_vals[t,k,a,m] = 0
                            tp_scores[t,k,a,m] = 0
                            var_tp_scores[t,k,a,m] = 0
                            fp_scores[t,k,a,m] = 0
                            fn_scores[t,k,a,m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist(); q = q.tolist()

                        for i in range(nd-1, 0, -1):
                            if pr[i] > pr[i-1]:
                                pr[i-1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                        precision[t,:,k,a,m] = np.array(q)
                        scores[t,:,k,a,m] = np.array(ss)
        coco_eval.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':   recall,
            'tp_vals': tp_vals, # add
            'fp_vals': fp_vals,
            'fn_vals': fn_vals,
            'tp_scores': tp_scores,
            'var_tp_scores': var_tp_scores,
            'fp_scores': fp_scores,
            'fn_scores': fn_scores,
            'scores': scores,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))

def summarize2(coco_eval):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = coco_eval.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            # titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            if ap == 1:
                titleStr = 'Average Precision'
            elif ap == 0:
                titleStr = 'Average Recall'
            elif ap == 2:
                titleStr = 'True Positives'
            elif ap == 3:
                titleStr = 'False Positives'
            elif ap == 4:
                titleStr = 'False Negatives'
            elif ap == 5:
                titleStr = 'Avg confidence'
            elif ap == 6:
                titleStr = 'ATPC'
            elif ap == 7:
                titleStr = 'AFPC'

            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = coco_eval.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            elif ap == 0:
                # dimension of recall: [TxKxAxM]
                s = coco_eval.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            elif ap == 2: # add
                # dimension of tp_vals: [TxKxAxM]
                s = coco_eval.eval['tp_vals']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            elif ap == 3:
                # dimension of fp_vals: [TxKxAxM]
                s = coco_eval.eval['fp_vals']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            elif ap == 4:
                # dimension of fn_vals: [TxKxAxM]
                s = coco_eval.eval['fn_vals']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            elif ap == 5:
                tp_scores = coco_eval.eval['tp_scores']
                tps = coco_eval.eval['tp_vals']
                fp_scores = coco_eval.eval['fp_scores']
                fps = coco_eval.eval['fp_vals']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    tp_scores = tp_scores[t]
                    tps = tps[t]
                    fp_scores = fp_scores[t]
                    fps = fps[t]
                tp_scores = tp_scores[:,:,aind,mind]
                tps = tps[:,:,aind,mind]
                fp_scores = fp_scores[:,:,aind,mind]
                fps = fps[:,:,aind,mind]
                
                s = (tp_scores + fp_scores) / (tps + fps)
            elif ap == 6:
                tp_scores = coco_eval.eval['tp_scores']
                tps = coco_eval.eval['tp_vals']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    tp_scores = tp_scores[t]
                    tps = tps[t]
                s = tp_scores[:,:,aind,mind] / tps[:,:,aind,mind]
            elif ap == 7:
                fp_scores = coco_eval.eval['fp_scores']
                fps = coco_eval.eval['fp_vals']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    fp_scores = fp_scores[t]
                    fps = fps[t]
                s = fp_scores[:,:,aind,mind] / fps[:,:,aind,mind]
                

            if ap == 1 or ap == 0 or ap == 5 or ap == 6 or ap == 7:
                if len(s[s>-1])==0:
                    mean_s = -1
                else:
                    mean_s = np.mean(s[s>-1])
                print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
                return mean_s
            elif (ap == 2 or ap == 3 or ap == 4) and s.size == 1:
                if len(s[s>-1])==0:
                    sum_s = -1
                else:
                    sum_s = np.sum(s[s>-1])
                print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, sum_s))
                return sum_s

        def _summarizeDets():
            stats = np.zeros((22,))
            stats[0] = _summarize(1)
            # stats[1] = _summarize(6)
            stats[1] = _summarize(1, iouThr=.5, maxDets=coco_eval.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=coco_eval.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small', maxDets=coco_eval.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=coco_eval.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=coco_eval.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=coco_eval.params.maxDets[2])
            stats[7] = _summarize(0, iouThr=.5, maxDets=coco_eval.params.maxDets[2])
            stats[8] = _summarize(0, iouThr=.75, maxDets=coco_eval.params.maxDets[2])
            stats[9] = _summarize(2, iouThr=.5, maxDets=coco_eval.params.maxDets[2])
            stats[10] = _summarize(4, iouThr=.5, maxDets=coco_eval.params.maxDets[2])
            stats[12] = _summarize(5, iouThr=.5, maxDets=coco_eval.params.maxDets[2])
            stats[13] = _summarize(6, iouThr=.5, maxDets=coco_eval.params.maxDets[2])
            stats[14] = _summarize(7, iouThr=.5, maxDets=coco_eval.params.maxDets[2])
            stats[15] = _summarize(6, iouThr=.5, maxDets=coco_eval.params.maxDets[2], areaRng='small')
            stats[16] = _summarize(6, iouThr=.5, maxDets=coco_eval.params.maxDets[2], areaRng='medium')
            stats[17] = _summarize(6, iouThr=.5, maxDets=coco_eval.params.maxDets[2], areaRng='large')
            stats[19] = _summarize(0, iouThr=.5, maxDets=coco_eval.params.maxDets[2], areaRng='small')
            stats[20] = _summarize(0, iouThr=.5, maxDets=coco_eval.params.maxDets[2], areaRng='medium')
            stats[21] = _summarize(0, iouThr=.5, maxDets=coco_eval.params.maxDets[2], areaRng='large')
            # stats[12] = _summarize(6, iouThr=.5, maxDets=coco_eval.params.maxDets[2])
            # stats[9] = _summarize(0, areaRng='small', maxDets=coco_eval.params.maxDets[2])
            # stats[10] = _summarize(0, areaRng='medium', maxDets=coco_eval.params.maxDets[2])
            # stats[11] = _summarize(0, areaRng='large', maxDets=coco_eval.params.maxDets[2])
            return stats
        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats
        if not coco_eval.eval:
            raise Exception('Please run accumulate() first')
        iouType = coco_eval.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        coco_eval.stats = summarize()

def filter_preds(ground_truth_file, predicted_boxes_file):

    with open(ground_truth_file) as json_file:
        ground_truth = json.load(json_file)

    with open(predicted_boxes_file) as json_file:
        predicted_boxes = json.load(json_file)

    gt_imgs_ids = [img["id"] for img in ground_truth["images"]]

    rm_pred_lst = []

    for pred in predicted_boxes:
        if pred["image_id"] not in gt_imgs_ids:
            rm_pred_lst.append(pred)

    for pred in rm_pred_lst:
        predicted_boxes.remove(pred)

    with open(predicted_boxes_file, "w") as json_file:
        json_file.write(json.dumps(predicted_boxes, indent=0))


def main():

    parser = argparse.ArgumentParser(description='Generating metrics for each skin tone')
    parser.add_argument('--skin_tone', type=int, default=10, help='skin tone to calculate metrics for')
    parser.add_argument('--output_pth', type=str, default="test", help='output path for metrics')
    parser.add_argument('--all', action='store_true', help='calculate metrics for all skin tones')
    args = parser.parse_args()

    ground_truth_file = f"Carla_data/Preds/0001.0.0.0/annotations.json"
    predicted_boxes_file = f"Carla_data/Preds/0001.0.0.0/preds_detr_r50.json"

    # filter out the images that are not in the ground truth. Some simulated rgb images may not have any pedestrians
    filter_preds(ground_truth_file, predicted_boxes_file) 
    
    # Create COCO objects
    coco_gt = COCO(ground_truth_file)
    coco_dt = coco_gt.loadRes(predicted_boxes_file)

    # Create a COCOeval object and evaluate
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()

    accumulate2(coco_eval)

    summarize2(coco_eval)

    print()

if __name__ == "__main__":
    main()