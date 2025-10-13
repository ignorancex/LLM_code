import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class EdgeLoss(nn.Module):
    """
    Loss for depth prediction. By default L1 loss is used.  
    """
    def __init__(self, loss='l1'):
        super(EdgeLoss, self).__init__()
        if loss == 'l1':
            self.loss = nn.L1Loss()
        
        elif loss=='l2':
            self.loss = nn.MSELoss()

        else:
            raise NotImplementedError('Loss {} currently not supported in DepthLoss'.format(loss))

    def forward(self, out, label):
        mask = (label != 0.0)
        return self.loss(torch.masked_select(out, mask), torch.masked_select(label, mask))

class PCLoss(nn.Module):
    """
    Loss for depth prediction. By default L1 loss is used.  
    """
    def __init__(self, loss='l1'):
        super(PCLoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, out, label):
        mask = (label == torch.tensor([127, 127], device=label.device).view(1, 2, 1, 1)).all(dim=1)
        return self.loss(torch.masked_select(out, ~mask.unsqueeze(1)), torch.masked_select(label, ~mask.unsqueeze(1)))


class SegmentationCELoss(nn.Module):
    """
    Cross Entropy Loss for segmentation tasks
    Output is provided as an 18-channel tensor with each channel representing a class
    """
    def __init__(self):
        super(SegmentationCELoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')

    def forward(self, out, label):
        # print(out.shape, out.dtype)
        # print(label.long().squeeze(dim=1).shape, label.long().squeeze(dim=1).dtype)
        # return torch.nn.functional.cross_entropy(out.float(),label.long().squeeze(dim=1),ignore_index=0,reduction='mean')
        # out = torch.argmax(out, dim=1)
        # print(out.shape, out.dtype)
        # print(out.unique())
        # print(label.shape, label.dtype)
        # Output is in Bx18xHxW format
        # We need to calculate the predicted class for each pixel
        # out = torch.argmax(out, dim=1)

        return self.loss(out, label.long().squeeze(dim=1))

class SegmentationCELossNew(nn.Module):
    """
    Cross Entropy Loss for segmentation tasks
    Output is provided as an 18-channel tensor with each channel representing a class
    """
    def __init__(self):
        super(SegmentationCELossNew, self).__init__()
        self.loss = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')

    def forward(self, out, label):
        # print(out.shape, out.dtype)
        # print(label.long().squeeze(dim=1).shape, label.long().squeeze(dim=1).dtype)
        # return torch.nn.functional.cross_entropy(out.float(),label.long().squeeze(dim=1),ignore_index=0,reduction='mean')
        # out = torch.argmax(out, dim=1)
        # print(out.shape, out.dtype)
        # print(out.unique())
        # print(label.shape, label.dtype)
        # Output is in Bx18xHxW format
        # We need to calculate the predicted class for each pixel
        # out = torch.argmax(out, dim=1)
        
        mask = (label == 0)
        # Zeroing out the outputs for the uncertain class
        out = out * ~mask
        label = label * ~mask
        return self.loss(out, label.long().squeeze(dim=1))

class NormalLoss(nn.Module):
    """
    Loss for normal prediction. By default L1 loss is used.  
    """
    def __init__(self, loss='l1'):
        super(NormalLoss, self).__init__()
        self.loss = nn.L1Loss(reduction="none")

    def forward(self, out, label):
        out = self.loss(out, label)
        out = out.mean()
        return out

LOSS_DICT = {
    'edge_texture':EdgeLoss(),
    'keypoints3d':nn.L1Loss(),
    'reshading':nn.MSELoss(),
    'normal':nn.L1Loss(),
    'class_scene': nn.CrossEntropyLoss(),
    'segment_semantic': SegmentationCELossNew(),
    'depth_euclidean': nn.L1Loss(),
    'principal_curvature': PCLoss(),
    'principal_curvature_old': NormalLoss()
}

def return_task_loss(task):
    return LOSS_DICT[task]