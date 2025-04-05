import torch

import numpy as np
from typing import List
from isegm.inference.clicker import Click
from isegm.utils.misc import get_bbox_iou, get_bbox_from_mask, expand_bbox, clamp_bbox
from .base import BaseTransform


class ZoomIn(BaseTransform):
    def __init__(self,
                 target_size=400,
                 skip_clicks=0,
                 expansion_ratio=1.4,
                 min_crop_size=0,
                 recompute_thresh_iou=1.0,
                 prob_thresh=0.50,
                 training=False):
        super().__init__()
        self.target_size = target_size
        self.min_crop_size = min_crop_size
        self.skip_clicks = skip_clicks
        self.expansion_ratio = expansion_ratio
        self.recompute_thresh_iou = recompute_thresh_iou
        self.prob_thresh = prob_thresh

        self._input_image_shape = None
        self._prev_probs = None
        self._object_roi = None
        self._roi_image = None

        self.is_training = training

    def set_probs(self, porbs):
        self._prev_probs = porbs

    def transform(self, image_nd, clicks_lists: List[List[Click]]):
        assert image_nd.shape[0] == 1 and len(clicks_lists) == 1
        self.image_changed = False

        clicks_list = clicks_lists[0]
        if len(clicks_list) <= self.skip_clicks:
            return image_nd, clicks_lists

        self._input_image_shape = image_nd.shape

        current_object_roi = None
        if self._prev_probs is not None:
            current_pred_mask = (self._prev_probs > self.prob_thresh)[0, 0]
            if current_pred_mask.sum() > 0:
                current_object_roi = get_object_roi(current_pred_mask, clicks_list,
                                                    self.expansion_ratio, self.min_crop_size, self.is_training)

        if current_object_roi is None:
            if self.skip_clicks >= 0:
                return image_nd, clicks_lists
            else:
                current_object_roi = 0, image_nd.shape[2] - 1, 0, image_nd.shape[3] - 1

        update_object_roi = False
        if self._object_roi is None:
            update_object_roi = True
        elif not check_object_roi(self._object_roi, clicks_list):
            update_object_roi = True
        elif get_bbox_iou(current_object_roi, self._object_roi) < self.recompute_thresh_iou:
            update_object_roi = True

        if update_object_roi:
            self._object_roi = current_object_roi
            self.image_changed = True
        if self.is_training:
            self._roi_image = get_train_roi_image_nd(image_nd, self._object_roi)
        else:
            self._roi_image = get_roi_image_nd(image_nd, self._object_roi, self.target_size)

        tclicks_lists = [self._transform_clicks(clicks_list)]
        return self._roi_image.to(image_nd.device), tclicks_lists

    def inv_transform(self, prob_map):
        if self._object_roi is None:
            self._prev_probs = prob_map.cpu().numpy()
            return prob_map

        assert prob_map.shape[0] == 1
        rmin, rmax, cmin, cmax = self._object_roi
        prob_map = torch.nn.functional.interpolate(prob_map, size=(rmax - rmin + 1, cmax - cmin + 1),
                                                   mode='bilinear', align_corners=True)

        if self._prev_probs is not None:
            new_prob_map = torch.zeros(*self._prev_probs.shape, device=prob_map.device, dtype=prob_map.dtype)
            new_prob_map[:, :, rmin:rmax + 1, cmin:cmax + 1] = prob_map
        else:
            new_prob_map = prob_map

        self._prev_probs = new_prob_map.cpu().numpy()

        return new_prob_map

    def check_possible_recalculation(self):
        if self._prev_probs is None or self._object_roi is not None or self.skip_clicks > 0:
            return False

        pred_mask = (self._prev_probs > self.prob_thresh)[0, 0]
        if pred_mask.sum() > 0:
            possible_object_roi = get_object_roi(pred_mask, [],
                                                 self.expansion_ratio, self.min_crop_size, self.is_training)
            image_roi = (0, self._input_image_shape[2] - 1, 0, self._input_image_shape[3] - 1)
            if get_bbox_iou(possible_object_roi, image_roi) < 0.50:
                return True
        return False

    def get_state(self):
        roi_image = self._roi_image.cpu() if self._roi_image is not None else None
        return self._input_image_shape, self._object_roi, self._prev_probs, roi_image, self.image_changed

    def set_state(self, state):
        self._input_image_shape, self._object_roi, self._prev_probs, self._roi_image, self.image_changed = state

    def reset(self):
        self._input_image_shape = None
        self._object_roi = None
        self._prev_probs = None
        self._roi_image = None
        self.image_changed = False

    def _transform_clicks(self, clicks_list):
        if self._object_roi is None:
            return clicks_list

        rmin, rmax, cmin, cmax = self._object_roi
        crop_height, crop_width = self._roi_image.shape[2:]

        transformed_clicks = []
        for click in clicks_list:
            new_r = crop_height * (click.coords[0] - rmin) / (rmax - rmin + 1)
            new_c = crop_width * (click.coords[1] - cmin) / (cmax - cmin + 1)
            transformed_clicks.append(click.copy(coords=(new_r, new_c)))
        return transformed_clicks


def dynamic_expand_ratio(pred_mask, bbox):
    rmin, rmax, cmin, cmax = bbox
    bbox_size = (rmax - rmin) * (cmax - cmin)
    image_size = pred_mask.shape[0] * pred_mask.shape[1]
    return max(2.5 - 4*np.sqrt(bbox_size/image_size), 1.4)


def get_object_roi(pred_mask, clicks_list, expansion_ratio, min_crop_size, is_training):
    pred_mask = pred_mask.copy()

    for click in clicks_list:
        if click.is_positive:
            pred_mask[int(click.coords[0]), int(click.coords[1])] = 1

    bbox = get_bbox_from_mask(pred_mask)
    # dynamic expansion_ratio
    expansion_ratio = dynamic_expand_ratio(pred_mask, bbox)
    # print(expansion_ratio)
    bbox = expand_bbox(bbox, expansion_ratio, min_crop_size)
    h, w = pred_mask.shape[0], pred_mask.shape[1]
    bbox = clamp_bbox(bbox, 0, h - 1, 0, w - 1)

    if is_training:
        bbox = equal_size_process(bbox)

    return bbox


def equal_size_process(bbox):
    row_start, row_end, col_start, col_end = bbox
    height = row_end - row_start + 1
    width = col_end - col_start + 1
    L = max(height / 2.0, width / 3.0)
    out_h = int(L * 2)
    out_w = int(L * 3)
    out_bbox = [-1, -1, -1, -1]
    if out_h == height:
        out_bbox[0] = row_start
        out_bbox[1] = row_end
        mid_w = (col_start + col_end) / 2.0
        if mid_w - out_w / 2.0 < 0:
            out_bbox[2] = 0
            out_bbox[3] = out_w - 1
        elif mid_w + out_w / 2.0 >= 480:
            out_bbox[2] = 480 - out_w
            out_bbox[3] = 479
        else:
            out_bbox[2] = int(mid_w - out_w / 2.0)
            out_bbox[3] = int(mid_w + out_w / 2.0) - 1
    elif out_w == width:
        out_bbox[2] = col_start
        out_bbox[3] = col_end
        mid_h = (row_start + row_end) / 2.0
        if mid_h - out_h / 2.0 < 0:
            out_bbox[0] = 0
            out_bbox[1] = out_h - 1
        elif mid_h + out_h / 2.0 >= 320:
            out_bbox[0] = 320 - out_h
            out_bbox[1] = 319
        else:
            out_bbox[0] = int(mid_h - out_h / 2.0)
            out_bbox[1] = int(mid_h + out_h / 2.0) - 1
    else:
        import ipdb
        ipdb.set_trace(context=10)
        test = 1
    try:
        assert out_bbox[1] - out_bbox[0] == out_h - 1
        assert out_bbox[3] - out_bbox[2] == out_w - 1
        # assert out_w / out_h - 1.5 < 0.1
    except Exception:
        import ipdb
        ipdb.set_trace(context=10)
        test = 1
    return out_bbox


def get_train_roi_image_nd(image_nd, object_roi):
    rmin, rmax, cmin, cmax = object_roi

    new_height = image_nd.shape[2]
    new_width = image_nd.shape[3]

    with torch.no_grad():
        roi_image_nd = image_nd[:, :, rmin:rmax + 1, cmin:cmax + 1]

        # @ Separately interpolate the Mask for visualization
        roi_image_nd1 = torch.nn.functional.interpolate(roi_image_nd[:, :1, :, :], size=(new_height, new_width),
                                                        mode='nearest')
        roi_image_nd2 = torch.nn.functional.interpolate(roi_image_nd[:, 1:, :, :], size=(new_height, new_width),
                                                        mode='bilinear', align_corners=True)
        roi_image_nd = torch.cat((roi_image_nd1, roi_image_nd2), dim=1)

    return roi_image_nd


def get_roi_image_nd(image_nd, object_roi, target_size):
    rmin, rmax, cmin, cmax = object_roi

    height = rmax - rmin + 1
    width = cmax - cmin + 1

    if isinstance(target_size, tuple):
        new_height, new_width = target_size
    else:
        scale = target_size / max(height, width)
        new_height = int(round(height * scale))
        new_width = int(round(width * scale))

    with torch.no_grad():
        roi_image_nd = image_nd[:, :, rmin:rmax + 1, cmin:cmax + 1]

        # @ Separately interpolate the Mask for visualization
        roi_image_nd1 = torch.nn.functional.interpolate(roi_image_nd[:, :1, :, :], size=(new_height, new_width),
                                                        mode='nearest')
        roi_image_nd2 = torch.nn.functional.interpolate(roi_image_nd[:, 1:, :, :], size=(new_height, new_width),
                                                        mode='bilinear', align_corners=True)
        roi_image_nd = torch.cat((roi_image_nd1, roi_image_nd2), dim=1)

        # roi_image_nd = torch.nn.functional.interpolate(roi_image_nd, size=(new_height, new_width),
        #                                                mode='bilinear', align_corners=True)

    return roi_image_nd


def check_object_roi(object_roi, clicks_list):
    for click in clicks_list:
        if click.is_positive:
            if click.coords[0] < object_roi[0] or click.coords[0] >= object_roi[1]:
                return False
            if click.coords[1] < object_roi[2] or click.coords[1] >= object_roi[3]:
                return False

    return True
