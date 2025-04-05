import json
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from ensemble_boxes import *
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

GT_PATH = '../MERGE_OCEAN/Annotations/gt_bbox.json'
list_model_path = ['../inference/cascade_mask_rcnn_r50_bbox.json',
                   '../inference/mask_rcnn_r50_bbox.json',
                   '../inference/mask_rcnn_r50_fpn_gn_ws_bbox.json',
                   '../inference/mask_rcnn_r50_fpn_groie_bbox.json',
                   '../inference/point_rend_r50_bbox.json']


def read_bbox_annotations(list_model_path: list) -> list:
    result_list = []
    for model in list_model_path:
        with open(model, 'r') as json_file:
            pred = json.load(json_file)
            result_list.append(pred)
        json_file.close()
    return result_list


def preprocess_bbox_annotations(list_model_path: list):

    cocoGt = COCO(GT_PATH)
    for model in list_model_path:
        with open(model, 'r') as json_file:
            pred = json.load(json_file)
        json_file.close()

        for ann in pred:
            del ann['segmentation']
            x1, y1, x2, y2 = ann['bbox']
            ann['bbox'] = [x1, y1, x2-x1, y2-y1]
            ann['area'] = (x2-x1)*(y2-y1)
        with open(model.replace('.json', '_bbox.json'), 'w') as json_file:
            json.dump(pred, json_file)
        json_file.close()

        cocoDt = cocoGt.loadRes(model.replace('.json', '_bbox.json'))
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        # cocoEval.params.imgIds  = imgIds
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()


def boxes_ensemble(gt: dict, list_model: list, weights: list, iou_thr: float) -> list:
    ensemble_results = []
    for image_ann in gt['images']:
        W, H = image_ann['width'], image_ann['height']
        image_id = image_ann['id']
        boxes_list, scores_list, labels_list, weights_list = [], [], [], []
        for idx, model_pred in enumerate(list_model):
            boxes, scores, labels = [], [], []
            for ann in model_pred:
                if ann['image_id'] == image_id:
                    x1, y1, w, h = ann['bbox']
                    x2, y2 = x1+w, y1+h
                    x1, x2 = np.clip([x1, x2], 0, W)/W
                    y1, y2 = np.clip([y1, y2], 0, H)/H
                    boxes.append(list(map(float, [x1, y1, x2, y2])))
                    scores.append(ann['score'])
                    labels.append(ann['category_id'])
            if len(labels) > 0:
                boxes_list.append(boxes)
                scores_list.append(scores)
                labels_list.append(labels)
                weights_list.append(weights[idx])
        # print(boxes_list, '\n', scores_list, '\n', labels_list)
        if len(labels_list) > 0:
            boxes, scores, labels = weighted_boxes_fusion(boxes_list,
                                                          scores_list,
                                                          labels_list,
                                                          weights=weights_list,
                                                          iou_thr=iou_thr)

            for i in range(len(labels)):
                ensemble_dict = dict()
                ensemble_dict['image_id'] = image_id
                ensemble_dict['score'] = float(scores[i])
                ensemble_dict['category_id'] = int(labels[i])
                x1, y1, x2, y2 = boxes[i]
                w, h = x2-x1, y2-y1
                x1, w = x1*W, w*W
                y1, h = y1*H, h*H
                ensemble_dict['bbox'] = list(map(float, [x1, y1, w, h]))
                ensemble_results.append(ensemble_dict)

    return ensemble_results


def optimize_iou_thr(gt: dict, list_model: list, weights: list):
    best_map = 0
    best_thres = 0
    cocoGt = COCO(GT_PATH)
    for iou_thr in tqdm(np.arange(0.3, 0.95, 0.01)):
        ensembles = boxes_ensemble(gt, list_model, weights, iou_thr)
        cocoDt = cocoGt.loadRes(ensembles)
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        map_50_95 = cocoEval.stats[0]

        if map_50_95 > best_map:
            best_map = map_50_95
            best_thres = iou_thr
    return best_map, best_thres


def create_pseudo_annotations(gt: dict, ensemble_result: list):
    pseudo_annotations = dict()
    pseudo_annotations['categories'] = deepcopy(gt['categories'])
    pseudo_annotations['images'] = deepcopy(gt['images'])
    count_id = 0
    for ann in ensemble_result:
        ann['id'] = count_id
        count_id += 1
        ann['iscrowd'] = 0
        x, y, w, h = ann['bbox']
        ann['area'] = w*h
        del ann['score']
    pseudo_annotations['annotations'] = ensemble_result
    return pseudo_annotations


def main():
    list_model = read_bbox_annotations(list_model_path)
    with open(GT_PATH, 'r') as json_file:
        gt = json.load(json_file)
    json_file.close()
    ensemble_result = boxes_ensemble(gt, list_model, [2, 1, 2, 1.5, 1], 0.68)
    pseudo_annotations = create_pseudo_annotations(gt, ensemble_result)

    with open('../MERGE_OCEAN/Annotations/pseudo_gt_bbox.json', 'w') as json_file:
        json.dump(pseudo_annotations, json_file)
    json_file.close()
    # best_map, best_thres = optimize_iou_thr(gt, list_model, [2, 1, 2, 1.5, 1])
    # print('best mAP: ', best_map, '\t', 'best iou_thr: ', best_thres)


if __name__ == '__main__':
    main()
    # for model in list_model_path:
    #     with open(model, 'r') as json_file:
    #         pred = json.load(json_file)
    #     json_file.close()

    #     cocoGt = COCO(GT_PATH)
    #     cocoDt = cocoGt.loadRes(model)
    #     cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    #     # cocoEval.params.imgIds  = imgIds
    #     cocoEval.evaluate()
    #     cocoEval.accumulate()
    #     cocoEval.summarize()
