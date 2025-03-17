import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeLoss(nn.Module):
    '''保证不同传感器的信号被分离开'''
    def __init__(self):
        super(EdgeLoss, self).__init__()
        self.sobel_x = torch.tensor([[1, 2, 1],
                                     [0, 0, 0],
                                     [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3).cuda()
    
    def forward(self, output_image, target_image):
        output_edge = F.conv2d(output_image, self.sobel_x, padding=1)
        target_edge = F.conv2d(target_image, self.sobel_x, padding=1)
        
        loss = F.mse_loss(output_edge, target_edge)
        
        return loss