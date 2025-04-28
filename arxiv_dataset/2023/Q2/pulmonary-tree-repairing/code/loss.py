import torch
import torch.nn as nn

class KeypointMSELoss(nn.Module):
    def __init__(self, use_target_weight=True):
        super(KeypointMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        # output: [b, 2, sub_size, sub_size, sub_size]
        batch_size = output.size(0)
        num_keypoints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_keypoints, -1)).split(1, 1) # [b,1,dhw], ...
        heatmaps_gt = target.reshape((batch_size, num_keypoints, -1)).split(1, 1)
        loss = 0 

        for idx in range(num_keypoints):
            heatmap_pred = heatmaps_pred[idx].squeeze() # [b,dhw]
            heatmap_gt = heatmaps_gt[idx].squeeze()
            
            if self.use_target_weight:
                loss += self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]), # [b,1]
                    heatmap_gt.mul(target_weight[:, idx])
                )
                # print(heatmap_pred.mul(target_weight[:, idx]).sum().item(),heatmap_gt.mul(target_weight[:, idx]).sum().item(),loss.item())
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)

        return loss