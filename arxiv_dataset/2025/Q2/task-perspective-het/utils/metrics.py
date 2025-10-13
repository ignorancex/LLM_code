import torch
import numpy as np

# Compute mIoU for segmentation images
def compute_mIoU(pred, target):
    ious = []
    # Looking into classes 1-17 because 0 is the uncertain class
    for cls in range(1, 18):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = torch.logical_and(pred_inds, target_inds).sum().item()
        union = torch.logical_or(pred_inds, target_inds).sum().item()
        
        if union == 0:
          ious.append(0.0)
        else:
          ious.append(intersection / union)

    miou = np.nanmean(ious)
    
    return miou

def compute_acc(pred, target):
    pred = torch.argmax(pred, dim=1)
    return (pred == target).sum().item() / target.numel()

TASK_METRICS = {
    "segment_semantic": compute_mIoU,
    "class_scene": compute_acc
}

def get_task_metric(task):
    if task not in TASK_METRICS.keys():
        raise ValueError(f"Unsupported task: {task}")
    return TASK_METRICS[task]