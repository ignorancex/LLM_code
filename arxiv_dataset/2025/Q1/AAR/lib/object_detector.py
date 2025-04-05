import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import cv2
import os

from lib.draw_rectangles.draw_rectangles import draw_union_boxes
from fasterRCNN.lib.model.faster_rcnn.resnet import resnet
from fasterRCNN.lib.model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from fasterRCNN.lib.model.roi_layers import nms

class detector(nn.Module):

    '''first part: object detection (image/video)'''

    def __init__(self, object_classes):
        super(detector, self).__init__()

        self.object_classes = object_classes

        self.fasterRCNN = resnet(classes=self.object_classes, num_layers=101, pretrained=False, class_agnostic=False)
        self.fasterRCNN.create_architecture()
        checkpoint = torch.load('fasterRCNN/models/faster_rcnn_ag.pth')
        self.fasterRCNN.load_state_dict(checkpoint['model'])

        self.ROI_Align = copy.deepcopy(self.fasterRCNN.RCNN_roi_align)
        self.RCNN_Head = copy.deepcopy(self.fasterRCNN._head_to_tail)

    def forward(self, im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all, predict_flag=False):

        # how many bboxes we have
        bbox_num = 0

        im_idx = []  # which frame are the relations belong to
        pair = []
        a_rel = []
        s_rel = []
        c_rel = []

        for i in gt_annotation:
            bbox_num += len(i)

        FINAL_BBOXES = torch.zeros([bbox_num,5], dtype=torch.float32).cuda(0)
        FINAL_LABELS = torch.zeros([bbox_num], dtype=torch.int64).cuda(0)
        FINAL_SCORES = torch.ones([bbox_num], dtype=torch.float32).cuda(0)
        HUMAN_IDX = torch.zeros([len(gt_annotation),1], dtype=torch.int64).cuda(0)

        bbox_idx = 0
        for i, j in enumerate(gt_annotation):
            for m in j:
                if 'person_bbox' in m.keys():
                    FINAL_BBOXES[bbox_idx,1:] = torch.from_numpy(m['person_bbox'][0]) # exact person bbox values
                    FINAL_BBOXES[bbox_idx, 0] = i # frame index
                    FINAL_LABELS[bbox_idx] = 1
                    HUMAN_IDX[i] = bbox_idx 
                    bbox_idx += 1
                elif 'bbox' in m.keys():
                    FINAL_BBOXES[bbox_idx,1:] = torch.from_numpy(m['bbox'])
                    FINAL_BBOXES[bbox_idx, 0] = i # frame index
                    FINAL_LABELS[bbox_idx] = m['class']
                    im_idx.append(i)
                    pair.append([int(HUMAN_IDX[i]), bbox_idx]) # subject object pair idx
                    a_rel.append(m['attention_relationship'].tolist())
                    s_rel.append(m['spatial_relationship'].tolist())
                    c_rel.append(m['contacting_relationship'].tolist())
                    bbox_idx += 1
                else:
                    continue

        pair = torch.tensor(pair).cuda(0)
        im_idx = torch.tensor(im_idx, dtype=torch.float).cuda(0)

        counter = 0
        FINAL_BASE_FEATURES = torch.tensor([]).cuda(0)

        while counter < im_data.shape[0]:
            #compute 10 images in batch and  collect all frames data in the video
            if counter + 10 < im_data.shape[0]:
                inputs_data = im_data[counter:counter + 10]
            else:
                inputs_data = im_data[counter:]
            base_feat = self.fasterRCNN.RCNN_base(inputs_data)
            FINAL_BASE_FEATURES = torch.cat((FINAL_BASE_FEATURES, base_feat), 0)
            counter += 10

        FINAL_BBOXES[:, 1:] = FINAL_BBOXES[:, 1:] * im_info[0, 2]
        FINAL_FEATURES = self.fasterRCNN.RCNN_roi_align(FINAL_BASE_FEATURES, FINAL_BBOXES)
        FINAL_FEATURES = self.fasterRCNN._head_to_tail(FINAL_FEATURES)


        union_boxes = torch.cat((im_idx[:, None], torch.min(FINAL_BBOXES[:, 1:3][pair[:, 0]], FINAL_BBOXES[:, 1:3][pair[:, 1]]),
                                 torch.max(FINAL_BBOXES[:, 3:5][pair[:, 0]], FINAL_BBOXES[:, 3:5][pair[:, 1]])), 1)
        union_feat = self.fasterRCNN.RCNN_roi_align(FINAL_BASE_FEATURES, union_boxes)
        FINAL_BBOXES[:, 1:] = FINAL_BBOXES[:, 1:] / im_info[0, 2]
        pair_rois = torch.cat((FINAL_BBOXES[pair[:, 0], 1:], FINAL_BBOXES[pair[:, 1], 1:]),
                              1).data.cpu().numpy()
        spatial_masks = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(FINAL_FEATURES.device)

        FINAL_DISTRIBUTIONS = torch.softmax(self.fasterRCNN.RCNN_cls_score(FINAL_FEATURES)[:, 1:], dim=1)
        FINAL_SCORES, PRED_LABELS = torch.max(FINAL_DISTRIBUTIONS, dim=1)
        PRED_LABELS = PRED_LABELS + 1
        
        entry = {'boxes': FINAL_BBOXES,
                 'labels': FINAL_LABELS if predict_flag == False else PRED_LABELS, # Use either the ground truth labels or labels predicted by FasterRCNN
                 'scores': FINAL_SCORES,
                 'im_idx': im_idx,
                 'pair_idx': pair,
                 'human_idx': HUMAN_IDX,
                 'features': FINAL_FEATURES,
                 'union_feat': union_feat,
                 'union_box': union_boxes,
                 'spatial_masks': spatial_masks,
                 'attention_gt': a_rel,
                 'spatial_gt': s_rel,
                 'contacting_gt': c_rel,
                }

        return entry