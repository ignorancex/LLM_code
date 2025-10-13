import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data_dict):
        fake_segm = data_dict['fake_segm']
        real_segm = data_dict['real_segm']

        if len(fake_segm.shape) > 4:
            fake_segm = fake_segm[:, 0]

        if len(real_segm.shape) > 4:
            real_segm = real_segm[:, 0]

        numer = (2 * fake_segm * real_segm).sum()
        denom =  ((fake_segm ** 2).sum() + (real_segm ** 2).sum())

        dice = numer / denom
        loss = -torch.log(dice) 

        return loss