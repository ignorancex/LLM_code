from time import time

import torch
import numpy as np
from isegm.inference import utils
from isegm.inference.clicker import Clicker

try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm


def evaluate_dataset(dataset, predictor, **kwargs):
    all_ious = []

    start_time = time()

    for index in tqdm(range(len(dataset)), leave=False):
        sample = dataset.get_sample(index)
        _, sample_ious, _ = evaluate_sample(sample.image, sample.gt_mask, predictor, **kwargs)
        all_ious.append(sample_ious)
    end_time = time()
    elapsed_time = end_time - start_time

    return all_ious, elapsed_time


def evaluate_sample(image, gt_mask, predictor, max_iou_thr,
                    pred_thr=0.49, min_clicks=1, max_clicks=20,
                    ignore_boundary=False):
    clicker = Clicker(gt_mask=gt_mask)
    pred_mask = np.zeros_like(gt_mask)
    ious_list = []

    with torch.no_grad():
        predictor.set_input_image(image)
        predictor.set_gt_mask(gt_mask)

        for click_indx in range(max_clicks):
            clicker.make_next_click(pred_mask)

            pred_probs, _ = predictor.get_prediction(clicker, step=click_indx)

            pred_mask = pred_probs > pred_thr

            if ignore_boundary:
                iou = utils.get_ig_iou(gt_mask, pred_mask, ignore_label=999)
            else:
                iou = utils.get_iou(gt_mask, pred_mask)
            ious_list.append(iou)

            if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                break

        return clicker.clicks_list, np.array(ious_list, dtype=np.float32), pred_probs
