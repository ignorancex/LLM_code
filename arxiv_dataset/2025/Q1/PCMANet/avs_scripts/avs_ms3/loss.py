import torch
import torch.nn as nn
import pdb


def F5_IoU_BCELoss(pred_mask, five_gt_masks):
    """
    binary cross entropy loss (iou loss) of the total five frames for multiple sound source segmentation

    Args:
    pred_mask: predicted masks for a batch of data, shape:[bs*5, 1, 224, 224]
    five_gt_masks: ground truth mask of the total five frames, shape: [bs*5, 1, 224, 224]
    """
    assert len(pred_mask.shape) == 4
    pred_mask = torch.sigmoid(pred_mask)  # [bs*5, 1, 224, 224]
    # five_gt_masks = five_gt_masks.view(-1, 1, five_gt_masks.shape[-2], five_gt_masks.shape[-1]) # [bs*5, 1, 224, 224]
    loss = nn.BCELoss()(pred_mask, five_gt_masks)

    return loss


def BCEIOU_loss(pred, mask):
    # bce
    pred = torch.sigmoid(pred)
    bce = nn.BCELoss()(pred, mask)  # pred vs mask
    # iou
    inter = ((pred * mask) * 1).sum(dim=(2, 3))  # pred vs mask
    union = ((pred + mask) * 1).sum(dim=(2, 3))  # pred vs mask
    iou = 1 - (inter + 1) / (union - inter + 1)
    return (bce + iou).mean()


if __name__ == "__main__":
    pdb.set_trace()
