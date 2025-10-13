from typing import Dict, List, Union, Iterable, Tuple

import torch
import torch.nn.functional as F
import torchvision.ops.boxes as box_ops
from torch import Tensor
from torchvision.models.detection.ssd import SSD

from .propagator_torch_detector import PropagatorTorchDetector


class PropagatorTorchSSD(PropagatorTorchDetector):

    def __init__(self,
                 model: SSD,
                 layers: Union[str, Iterable[str]],
                 batch_size: int = 16,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                 ) -> None:
        """
        Args:
            model: model
            layers: list of layers for activations and/or gradients registration
            batch_size: butch size for sampling from AbstractDataset instances

        Kwargs:
            batch_size: batch size for conversion of AbstractDataset to DataLoader, default value is 32
            device: torch device
        """
        super().__init__(model, layers, batch_size, device)

        # override model's postpocessing
        self.model.postprocess_detections = postprocess_detections_new.__get__(self.model, SSD)

    def tensor_get_gradients_for_targets(self,
                                         input: Tensor,
                                         targets: Iterable[Tensor]
                                         ) -> List[Dict[str, Tensor]]:
        """
        Propagate forward and backward and get gradients for target bounding boxes and classes.
        For backpropagation, the bounding box with highest IoU between target bbox and proposed by detector bbox is selected.

        Args:
            input: input batch tensor
            targets: list of target Tensors with bboxes and classes for each image - List[Tensor[n_obj, 5]], where tensor's 0th dim is n_obj long and 1st dim is [x1, y1, x2, y2, class]

        Returns:
            list of dictionary for each object [{layer: gradients}]: List[Dict[str, Tensor]]
        """

        # change model parameters to get additional bounding boxes
        temp_score_thresh = self.model.score_thresh
        temp_nms_thresh = self.model.nms_thresh
        temp_detections_per_img = self.model.detections_per_img
        temp_topk_candidates = self.model.topk_candidates

        # set parameters, to soften NMS and aquire more bboxes
        self.model.score_thresh: float = 0.001
        self.model.nms_thresh: float = 0.9
        self.model.detections_per_img: int = 100_000
        self.model.topk_candidates: int = 100_000

        res = super(PropagatorTorchSSD, self).tensor_get_gradients_for_targets(input, targets)

        # restore model parameters
        self.model.score_thresh = temp_score_thresh
        self.model.nms_thresh = temp_nms_thresh
        self.model.detections_per_img = temp_detections_per_img
        self.model.topk_candidates = temp_topk_candidates

        return res

    def tensor_get_gradients_for_targets_and_classes(self,
                                                     input: Tensor,
                                                     targets: Iterable[Tensor],
                                                     classes: Iterable[int]
                                                     ) -> Tuple[List[List[Dict[int, Dict[str, Tensor]]]], Dict[str, List[List[Tensor]]]]:
        
        # change model parameters to get additional bounding boxes
        temp_score_thresh = self.model.score_thresh
        temp_nms_thresh = self.model.nms_thresh
        temp_detections_per_img = self.model.detections_per_img
        temp_topk_candidates = self.model.topk_candidates

        # set parameters, to soften NMS and aquire more bboxes
        self.model.score_thresh: float = 0.001
        self.model.nms_thresh: float = 0.9
        self.model.detections_per_img: int = 100_000
        self.model.topk_candidates: int = 100_000

        res = super(PropagatorTorchSSD, self).tensor_get_gradients_for_targets_and_classes(input, targets, classes)

        # restore model parameters
        self.model.score_thresh = temp_score_thresh
        self.model.nms_thresh = temp_nms_thresh
        self.model.detections_per_img = temp_detections_per_img
        self.model.topk_candidates = temp_topk_candidates

        return res

def postprocess_detections_new(self, 
                               head_outputs: Dict[str, Tensor],
                               image_anchors: List[Tensor],
                               image_shapes: List[Tuple[int, int]]
                               ) -> List[Dict[str, Tensor]]:
    """
    overrides SSD.postprocess_detections
    returns dict with all class logits "all_scores" instead of only of class logint for top-class
    """
    bbox_regression = head_outputs["bbox_regression"]
    pred_scores = F.softmax(head_outputs["cls_logits"], dim=-1)

    num_classes = pred_scores.size(-1)
    device = pred_scores.device

    detections: List[Dict[str, Tensor]] = []

    for boxes, scores, anchors, image_shape in zip(bbox_regression, pred_scores, image_anchors, image_shapes):
        boxes = self.box_coder.decode_single(boxes, anchors)
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

        image_boxes = []
        image_scores = []
        image_labels = []
        image_all_scores = []
        for label in range(1, num_classes):
            score = scores[:, label]
            all_scores = scores[:, 1:]  # all scores without background class - 0th

            keep_idxs = score > self.score_thresh
            score = score[keep_idxs]
            box = boxes[keep_idxs]
            all_scores = all_scores[keep_idxs]

            # keep only topk scoring predictions
            num_topk = min(len(score), self.topk_candidates)
            score, idxs = score.topk(num_topk)
            box = box[idxs]
            all_scores = all_scores[idxs]

            image_boxes.append(box)
            image_scores.append(score)
            image_labels.append(torch.full_like(
                score, fill_value=label, dtype=torch.int64, device=device))
            image_all_scores.append(all_scores)

        image_boxes = torch.cat(image_boxes, dim=0)
        image_scores = torch.cat(image_scores, dim=0)
        image_labels = torch.cat(image_labels, dim=0)
        image_all_scores = torch.cat(image_all_scores, dim=0)

        # non-maximum suppression
        keep = box_ops.batched_nms(
            image_boxes, image_scores, image_labels, self.nms_thresh)
        keep = keep[: self.detections_per_img]

        detections.append(
            {
                "boxes": image_boxes[keep],
                "scores": image_scores[keep],
                "labels": image_labels[keep],
                "all_scores": image_all_scores[keep]  # all class logits
            }
        )
    return detections
