import os
import cv2
import json
import pylab
import pickle
import operator
import numpy as np
import skimage.io as io
from glob import glob
from tqdm import tqdm
from copy import deepcopy
from PIL import Image, ImageDraw
from imantics import Polygons, Mask
import matplotlib.pyplot as plt
import pycocotools._mask as _mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def decode(rleObjs):
    if type(rleObjs) == list:
        return _mask.decode(rleObjs)
    else:
        return _mask.decode([rleObjs])[:, :, 0]


def polygonFromMask(maskedArr):

    contours, _ = cv2.findContours(
        maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    segmentation = []
    for contour in contours:
        if contour.size >= 6:
            segmentation.append(contour.flatten().tolist())
    RLEs = _mask.frPyObjects(
        segmentation, maskedArr.shape[0], maskedArr.shape[1])
    RLE = _mask.merge(RLEs)
    # RLE = mask.encode(np.asfortranarray(maskedArr))
    area = _mask.area(RLE)
    [x, y, w, h] = cv2.boundingRect(maskedArr)

    return segmentation[0]  # , [x, y, w, h], area


def segmentFromMask(npmask):
    # print(npmask)
    # print(dtype(npmask))
    contours, hierarchy = cv2.findContours((npmask).astype(np.uint8), cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []

    for contour in contours:
        contour = contour.flatten().tolist()
        # segmentation.append(contour)
        if len(contour) > 4:
            segmentation.append(contour)

    return segmentation


def ensemble_by_pseudo_label(gt: dict, pred: list) -> list:
    ensemble_result = []
    count_id = 0

    for i in tqdm(range(len(gt['images']))):
        img_id = gt['images'][i]['id']
        img_path = gt['images'][i]['file_name'][1:]
        img = cv2.imread(img_path)[:, :, ::-1]
        h, w, _ = img.shape
        img_pred = pred[i][1]
        score_pred = pred[i][0]

        for idx in range(len(img_pred)):

            if len(img_pred[idx]) == 0:
                continue

            for k in range(len(img_pred[idx])):

                update_ann = {}
                update_ann['image_id'] = img_id

                segment_ann = img_pred[idx][k]

                # if len(segment_ann) == 0 or len(score_pred[idx]) == 0:
                #     continue

                mask_img = Image.new('L', (w, h), 0)
                all_segment = segmentFromMask(decode(segment_ann))
                all_point_poly = Mask(
                    decode(segment_ann)).polygons().points

                total_list_point_poly = []
                for j in range(len(all_point_poly)):
                    list_point_poly = all_point_poly[j].astype(np.int32)
                    total_list_point_poly.extend(
                        [tuple(poly_point) for poly_point in list_point_poly])

                if len(total_list_point_poly) > 1 and len(all_segment) > 0:
                    ImageDraw.Draw(mask_img).polygon(
                        total_list_point_poly, outline=1, fill=1)

                    update_ann['segmentation'] = all_segment
                    update_ann['category_id'] = idx
                    # print(score_pred[idx])
                    update_ann['score'] = float(score_pred[idx][k][-1])
                    update_ann['id'] = count_id
                    count_id += 1
                    update_ann['iscrowd'] = 0
                    update_ann['area'] = int(np.sum(np.array(mask_img)))

                    # total_list_point_poly = np.array(total_list_point_poly)
                    # xmin, ymin = np.min(total_list_point_poly[:, 0]), np.min(
                    #     total_list_point_poly[:, 1])
                    # xmax, ymax = np.max(total_list_point_poly[:, 0]), np.max(
                    #     total_list_point_poly[:, 1])
                    update_ann['bbox'] = list(map(int, score_pred[idx][k][:4]))

                    ensemble_result.append(update_ann)

    return ensemble_result


def msrcnn(gt: dict, pred: list) -> list:
    ensemble_result = []
    count_id = 0

    for i in tqdm(range(len(gt['images']))):
        img_id = gt['images'][i]['id']
        img_path = gt['images'][i]['file_name'][1:]
        img = cv2.imread(img_path)[:, :, ::-1]
        h, w, _ = img.shape
        img_pred = pred[i][1][0]
        score_pred = pred[i][0]

        for idx in range(len(img_pred)):

            if len(img_pred[idx]) == 0:
                continue

            for k in range(len(img_pred[idx])):

                update_ann = {}
                update_ann['image_id'] = img_id

                segment_ann = img_pred[idx][k]

                # if len(segment_ann) == 0 or len(score_pred[idx]) == 0:
                #     continue

                mask_img = Image.new('L', (w, h), 0)
                all_segment = segmentFromMask(decode(segment_ann))
                all_point_poly = Mask(
                    decode(segment_ann)).polygons().points

                total_list_point_poly = []
                for j in range(len(all_point_poly)):
                    list_point_poly = all_point_poly[j].astype(np.int32)
                    total_list_point_poly.extend(
                        [tuple(poly_point) for poly_point in list_point_poly])

                if len(total_list_point_poly) > 1 and len(all_segment) > 0:
                    ImageDraw.Draw(mask_img).polygon(
                        total_list_point_poly, outline=1, fill=1)

                    update_ann['segmentation'] = all_segment
                    update_ann['category_id'] = idx
                    # print(score_pred[idx])
                    update_ann['score'] = float(score_pred[idx][k][-1])
                    update_ann['id'] = count_id
                    count_id += 1
                    update_ann['iscrowd'] = 0
                    update_ann['area'] = int(np.sum(np.array(mask_img)))

                    # total_list_point_poly = np.array(total_list_point_poly)
                    # xmin, ymin = np.min(total_list_point_poly[:, 0]), np.min(
                    #     total_list_point_poly[:, 1])
                    # xmax, ymax = np.max(total_list_point_poly[:, 0]), np.max(
                    #     total_list_point_poly[:, 1])
                    update_ann['bbox'] = list(map(int, score_pred[idx][k][:4]))

                    ensemble_result.append(update_ann)

    return ensemble_result


ROOT = './mmdetection/inference'
list_results = glob(ROOT + '/*.pkl')
GT_PATH = './MERGE_OCEAN/Annotations/merge_ocean_test.json'
cocoGt = COCO(GT_PATH)

with open('./MERGE_OCEAN/Annotations/merge_ocean_test.json', 'r') as json_file:
    test_gt = json.load(json_file)
json_file.close()

for i, res_path in enumerate(list_results):
    if 'r4_gcb' not in res_path:
        continue
    print(res_path)
    with open(res_path, 'rb') as pickle_file:
        res = pickle.load(pickle_file)
    pickle_file.close()

    if 'ms_rcnn' in res_path:
        eval_res = msrcnn(test_gt, res)
    else:
        eval_res = ensemble_by_pseudo_label(test_gt, res)

    with open(res_path.replace('.pkl', '.json'), 'w') as json_out:
        json.dump(eval_res, json_out)
    json_out.close()

    cocoDt = cocoGt.loadRes(res_path.replace('.pkl', '.json'))
    cocoEval = COCOeval(cocoGt, cocoDt, 'segm')
    # cocoEval.params.imgIds  = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    # except:
    #     pass
