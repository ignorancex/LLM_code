import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


def F10_IoU_BCELoss(pred_mask, ten_gt_masks, gt_temporal_mask_flag):
    """
    binary cross entropy loss (iou loss) of the total five frames for multiple sound source segmentation

    Args:
    pred_mask: predicted masks for a batch of data, shape:[bs*10, N_CLASSES, 224, 224]
    ten_gt_masks: ground truth mask of the total five frames, shape: [bs*10, 224, 224]
    """
    assert len(pred_mask.shape) == 4
    if ten_gt_masks.shape[1] == 1:
        ten_gt_masks = ten_gt_masks.squeeze(1) # [bs*10, 224, 224]
    # loss = nn.CrossEntropyLoss()(pred_mask, ten_gt_masks)
    #! notice:
    loss = nn.CrossEntropyLoss(reduction='none')(pred_mask, ten_gt_masks) # [bs*10, 224, 224]
    loss = loss.mean(-1).mean(-1) # [bs*10]
    loss = loss * gt_temporal_mask_flag # [bs*10]
    loss = torch.sum(loss) / torch.sum(gt_temporal_mask_flag)

    return loss


if __name__ == "__main__":
    pdb.set_trace()
