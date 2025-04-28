from .Local import LocalGlance
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

class GlobalGlance(nn.Module):
    def __init__(self, kernel_size=4, stride=4, repeat_time=10, patch_height=64, patch_width=64, drop=None, sigma=1.5):
        super(GlobalGlance, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.repeat_time = repeat_time
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.sigma = sigma
        self.ssim_loss = LocalGlance(window_size=self.kernel_size, stride=self.stride, drop=drop, sigma=self.sigma)
    
    def forward(self, src_vec, tar_vec):
        batch_size, channels, height, width = src_vec.size()
        loss = 0.0

        for batch in range(batch_size):
            index_list = []
            for i in range(self.repeat_time):
                if i == 0:
                    tmp_index = torch.arange(height * width)
                else:
                    tmp_index = torch.randperm(height * width)
                index_list.append(tmp_index)
            
            res_index = torch.cat(index_list)
            rows = res_index // width
            cols = res_index % width
            tar_all = tar_vec[batch, :, rows, cols].view(channels, -1, self.patch_height, self.patch_width * self.repeat_time)
            src_all = src_vec[batch, :, rows, cols].view(channels, -1, self.patch_height, self.patch_width * self.repeat_time)
            tar_mag = torch.clip(tar_all, 0, 1)*255
            src_mag = torch.clip(src_all, 0, 1)*255

            loss += (1 - self.ssim_loss(src_mag, tar_mag))

        loss /= batch_size
        return loss
