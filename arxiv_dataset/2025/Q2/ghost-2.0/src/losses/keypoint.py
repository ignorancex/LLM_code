import torch
import torchvision
import os
import torch.nn as nn

from src.utils.keypoint_detector import DECAKeypoints

class KeypointLoss(nn.Module):
    def __init__(self, mode='all', device='cuda'):
        super().__init__()

        self.detector = DECAKeypoints(device=device)

        self.mode = mode

        self.register_buffer('weights', torch.ones(68), persistent=False)
        self.weights[5:7] = 2.0
        self.weights[10:12] = 2.0
        self.weights[27:36] = 1.5
        self.weights[30] = 3.0
        self.weights[31] = 3.0
        self.weights[35] = 3.0
        self.weights[60:68] = 1.5
        self.weights[48:60] = 1.5
        self.weights[48] = 3
        self.weights[54] = 3
        
        if self.mode == 'expr':
            self.register_buffer('upper_lids', torch.LongTensor([37, 38, 43, 44]), persistent=False)
            self.register_buffer('lower_lids', torch.LongTensor([41, 40, 47, 46]), persistent=False)
            self.register_buffer('upper_lips', torch.LongTensor([61, 62, 63]), persistent=False)
            self.register_buffer('lower_lips', torch.LongTensor([67, 66, 65]), persistent=False)

    def forward(self, pred, gt):

        with torch.no_grad():
            landmarks2d_gt = self.detector(gt) 
        
        landmarks2d_pred = self.detector(pred) 
        
        if self.mode == 'all':
            diff = (landmarks2d_pred - landmarks2d_gt).abs().mean(-1)
            loss = diff * self.weights[None] / self.weights.sum()
            loss = loss.sum(-1).mean()
            return loss

        if self.mode == 'expr':
            diff_pred = (landmarks2d_pred[:, self.upper_lips] - landmarks2d_pred[:, self.lower_lips]).abs().mean(-1)
            diff = (landmarks2d_gt[:, self.upper_lips] - landmarks2d_gt[:, self.lower_lips]).abs().mean(-1)
            loss_lips = (diff_pred - diff).abs().mean()

            diff_pred = (landmarks2d_pred[:, self.upper_lids] - landmarks2d_pred[:, self.lower_lids]).abs().mean(-1)
            diff = (landmarks2d_gt[:, self.upper_lids] - landmarks2d_gt[:, self.lower_lids]).abs().mean(-1)
            loss_eyes = (diff_pred - diff).abs().mean()
            
            return loss_eyes + loss_lips