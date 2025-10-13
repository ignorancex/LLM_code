# Copyright (c) OpenMMLab. All rights reserved.
# Modified by Nicola Marinello, 2025
# from https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/evaluation/metrics/seg_metric.py
# from https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/evaluation/metrics/panoptic_seg_metric.py
from typing import Dict
from typing import Sequence

from mmdet3d.evaluation import panoptic_seg_eval
from mmdet3d.evaluation.metrics import SegMetric, PanopticSegMetric
from mmengine.logging import MMLogger

from offsetocc.registry import METRICS


@METRICS.register_module()
class SegMetric(SegMetric):

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``,
        which will be used to compute the metrics when all batches
        have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        for data_sample in data_samples:
            pred_3d = data_sample['pred_pts_seg']
            gt_pts_seg = data_sample['gt_pts_seg']
            cpu_pred_3d = dict()
            cpu_gt_pts_seg = dict()
            for k, v in pred_3d.items():
                if hasattr(v, 'to'):
                    cpu_pred_3d[k] = v.to('cpu').numpy()
                else:
                    cpu_pred_3d[k] = v
            for k, v in gt_pts_seg.items():
                if hasattr(v, 'to'):
                    cpu_gt_pts_seg[k] = v.to('cpu').numpy()
                else:
                    cpu_gt_pts_seg[k] = v
            self.results.append((cpu_gt_pts_seg, cpu_pred_3d))


@METRICS.register_module()
class PanopticSegMetric(PanopticSegMetric):

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``,
        which will be used to compute the metrics when all batches
        have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        for data_sample in data_samples:
            pred_3d = data_sample['pred_pts_seg']
            gt_pts_seg = data_sample['gt_pts_seg']
            cpu_pred_3d = dict()
            cpu_gt_pts_seg = dict()
            for k, v in pred_3d.items():
                if hasattr(v, 'to'):
                    cpu_pred_3d[k] = v.to('cpu').numpy()
                else:
                    cpu_pred_3d[k] = v
            for k, v in gt_pts_seg.items():
                if hasattr(v, 'to'):
                    cpu_gt_pts_seg[k] = v.to('cpu').numpy()
                else:
                    cpu_gt_pts_seg[k] = v
            self.results.append((cpu_gt_pts_seg, cpu_pred_3d))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        if self.submission_prefix:
            self.format_results(results)
            return None

        label2cat = self.dataset_meta['label2cat']
        ignore_index = self.dataset_meta['ignore_index']
        classes = self.dataset_meta['occ_class_names']
        thing_classes = [classes[i] for i in self.thing_class_inds]
        stuff_classes = [classes[i] for i in self.stuff_class_inds]

        gt_labels = []
        seg_preds = []
        for eval_ann, sinlge_pred_results in results:
            gt_labels.append(eval_ann)
            seg_preds.append(sinlge_pred_results)

        ret_dict = panoptic_seg_eval(gt_labels, seg_preds, classes,
                                     thing_classes, stuff_classes,
                                     self.min_num_points, self.id_offset,
                                     label2cat, [ignore_index], logger)

        return ret_dict