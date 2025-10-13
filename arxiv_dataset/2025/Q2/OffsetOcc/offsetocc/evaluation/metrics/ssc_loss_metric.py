# from https://github.com/astra-vision/MonoScene/blob/master/monoscene/loss/ssc_loss.py
from typing import Sequence

import torch
import torch.nn.functional as F
from mmengine.evaluator import BaseMetric

from offsetocc.registry import METRICS


@METRICS.register_module()
class SSCLossMetric(BaseMetric):
    def __init__(self, free_class_index=17, use_camera_mask=True):
        super(SSCLossMetric, self).__init__()
        self.free_class_index = free_class_index
        self.use_camera_mask = use_camera_mask

    def _geo_scal_loss(self, pred, ssc_target) -> torch.Tensor:
        """Compute the geometrical occupancy loss.

        Computes the loss on the class agnostic voxel occupancy.
        """
        # Get softmax probabilities
        pred = F.softmax(pred, dim=1)

        # Compute empty and nonempty probabilities
        empty_probs = pred[:, self.free_class_index, :, :, :]
        nonempty_probs = 1 - empty_probs

        # Remove unknown voxels
        mask = ssc_target != 255
        nonempty_target = ssc_target != self.free_class_index

        batch_precision = []
        batch_recall = []
        batch_spec = []

        for pred_i, target_i, mask_i, nonempty_target_i, nonempty_probs_i, empty_probs_i in zip(
                pred, ssc_target, mask, nonempty_target, nonempty_probs, empty_probs
        ):
            nonempty_target_i = nonempty_target_i[mask_i].float()
            nonempty_probs_i = nonempty_probs_i[mask_i]
            empty_probs_i = empty_probs_i[mask_i]

            intersection = (nonempty_target_i * nonempty_probs_i).sum()
            precision = intersection / nonempty_probs_i.sum()
            recall = intersection / nonempty_target_i.sum()
            spec = ((1 - nonempty_target_i) * (empty_probs_i)).sum() / (1 - nonempty_target_i).sum()

            batch_precision.append(precision)
            batch_recall.append(recall)
            batch_spec.append(spec)

        precision = torch.stack(batch_precision)
        recall = torch.stack(batch_recall)
        spec = torch.stack(batch_spec)

        return (
                F.binary_cross_entropy_with_logits(
                    precision, torch.ones_like(precision)
                ) + F.binary_cross_entropy_with_logits(
            recall, torch.ones_like(recall)
        ) + F.binary_cross_entropy_with_logits(spec, torch.ones_like(spec))
        )

    def _sem_scal_loss(self, pred, ssc_target) -> torch.Tensor:
        """Compute the Semantic Completion loss.

        Compute the per-class occupancy loss and average the result.
        """
        # Get softmax probabilities
        pred = F.softmax(pred, dim=1)
        mask = ssc_target != 255
        n_classes = pred.shape[1]

        batch_loss = []
        for pred_i, target_i, mask_i in zip(pred, ssc_target, mask):

            loss = 0
            count = 0

            for i in range(0, n_classes):

                # Get probability of class i
                p = pred_i[i, :, :, :]

                # Remove unknown voxels
                target_ori = target_i
                p = p[mask_i]
                target = target_i[mask_i]

                completion_target = torch.ones_like(target)
                completion_target[target != i] = 0
                completion_target_ori = torch.ones_like(target_ori).float()
                completion_target_ori[target_ori != i] = 0
                if torch.sum(completion_target) > 0:
                    count += 1.0
                    nominator = torch.sum(p * completion_target)
                    loss_class = 0
                    if torch.sum(p) > 0:
                        precision = nominator / (torch.sum(p))
                        loss_precision = F.binary_cross_entropy_with_logits(
                            precision, torch.ones_like(precision)
                        )
                        loss_class += loss_precision
                    if torch.sum(completion_target) > 0:
                        recall = nominator / (torch.sum(completion_target))
                        loss_recall = F.binary_cross_entropy_with_logits(
                            recall, torch.ones_like(recall)
                        )
                        loss_class += loss_recall
                    if torch.sum(1 - completion_target) > 0:
                        specificity = (
                            torch.sum((1 - p) * (1 - completion_target))
                            / torch.sum(1 - completion_target)
                        )
                        loss_specificity = F.binary_cross_entropy_with_logits(
                            specificity, torch.ones_like(specificity)
                        )
                        loss_class += loss_specificity
                    loss += loss_class

            batch_loss.append(loss / count)

        return torch.stack(batch_loss).mean()

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        """

        pred_pix_proj_batch = []
        target_batch = []
        for i, data_sample in enumerate(data_samples):
            pred_label = data_sample['pred_occ_map_logits']['logits'].squeeze()
            label = data_batch['data_samples'][i].gt_occ_map.occ_map.squeeze().to(pred_label)

            if self.use_camera_mask:
                mask_camera = data_batch['data_samples'][i].gt_occ_map.occ_map_mask_camera.squeeze().to(device=pred_label.device)

                # set ignore value in the target based on the visibility mask
                label[~mask_camera] = 255

            pred_pix_proj_batch.append(pred_label)
            target_batch.append(label)

        pred = torch.stack(pred_pix_proj_batch)
        target = torch.stack(target_batch)
        sem_loss = self._sem_scal_loss(pred, target)
        geo_loss = self._geo_scal_loss(pred, target)
        loss = sem_loss + geo_loss
        self.results.append(loss.item())

    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.
        """

        return {'ValSSCLossMetric': sum(results) / len(results)}


if __name__ == "__main__":
    from offsetocc.structures import OccDataSample, OccupancyData
    import numpy as np
    from tqdm import tqdm

    # create dummy gt occupancy map and dummy prediction
    num_batches = 100
    bs = 8

    # create the metric
    metric = SSCLossMetric(use_camera_mask=False)
    # create the loss
    from offsetocc.models.losses import SSCLoss
    loss = SSCLoss()

    loss_batches = []
    for n in tqdm(range(num_batches)):

        # generate dummy gt and predicted data
        gt_occ = np.random.randint(0, 18, (bs, 200, 200, 16))
        pred_occ = np.random.rand(bs, 18, 200, 200, 16)

        gt_data_samples = []
        pr_data_samples = []
        for i in range(bs):
            # ground truth
            data_sample = OccDataSample()
            gt_occ_map = OccupancyData()
            gt_occ_map['occ_map'] = torch.tensor(gt_occ[i])
            data_sample.gt_occ_map = gt_occ_map
            gt_data_samples.append(data_sample)

            # prediction
            data_sample = {'pred_occ_map_logits': {'logits': torch.tensor(pred_occ[i])}}
            pr_data_samples.append(data_sample)

        data_batch = {'data_samples': gt_data_samples}

        metric.process(data_batch, pr_data_samples)

        # calculate the loss
        loss_value = loss(torch.tensor(pred_occ), torch.tensor(gt_occ))
        loss_batches.append(loss_value.item())


    print("Metric results: ", metric.compute_metrics(metric.results))
    print("Loss values: ", sum(loss_batches) / len(loss_batches))
