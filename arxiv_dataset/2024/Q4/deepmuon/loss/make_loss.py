import torch
from torch import Tensor
from torch.nn import Module
from numpy import array
import numpy as np

epsilon = 1e-6


class SWDLoss(Module):

    def __init__(
        self,
        gamma: int,
        weight_gamma: int = None,
        weight_index: list = None,
        repeat_projector_num: int = 4,
        projector_dim: int = 512,
        random_projection_swd_type='SWD',
        specific_projection_swd_type='S-SWD',
    ):
        super().__init__()
        self.gamma = gamma
        self.weight_gamma = weight_gamma
        if weight_index is None:
            self.weight_index = None
        else:
            self.weight_index = torch.tensor(weight_index)
        self.repeat_projector_num = repeat_projector_num
        self.projector_dim = projector_dim
        assert random_projection_swd_type in ['SWD', 'S-SWD']
        self.random_projection_swd_type = random_projection_swd_type
        assert specific_projection_swd_type in ['SWD', 'S-SWD']
        self.specific_projection_swd_type = specific_projection_swd_type

    def projector(self, x, y, random_projector=None):
        if random_projector is None:
            random_projector = torch.randn(x.shape[1],
                                           self.projector_dim,dtype=torch.float32).to(x.device)
        else:
            assert random_projector.shape[0] == x.shape[1]
            random_projector = random_projector.to(x.device)
        x = torch.matmul(x, random_projector)
        y = torch.matmul(y, random_projector)
        return x, y

    def sort_wd(self, x, y):
        return torch.mean(torch.abs(x - y) * torch.std(y, dim=0))

    def swd(self, x_y, swd_type):
        x, y = x_y
        x = torch.sort(x, dim=0)[0]
        y = torch.sort(y, dim=0)[0]
        split_length = int(x.shape[0] / 32)
        if swd_type == 'S-SWD':
            mean_x = torch.mean(x, dim=0)
            std_x = torch.std(x, dim=0)
            x = (x - mean_x) / (std_x + epsilon)
            y = (y - mean_x) / (std_x + epsilon)
            split_length = x.shape[0] // split_length
            split_residual = x.shape[0] % split_length
            loss = 0
            for i in range(split_length):
                loss += self.sort_wd(
                    x[i * split_length:(i + 1) * split_length],
                    y[i * split_length:(i + 1) * split_length])
            if split_residual != 0:
                loss += self.sort_wd(x[(i + 1) * split_length:],
                                     y[(i + 1) * split_length:])
                loss /= split_length + 1
            else:
                loss /= split_length
        else:
            loss = torch.mean(torch.abs(x - y))
        return loss

    def forward(self,
                predicts: Tensor,
                targets: Tensor,
                random_projection=None,
                return_detail: bool = False):
        """
        shape: (numbers of events, numbers of quantities)
        """
        assert predicts.shape == targets.shape
        
        #在arctah的空间做swd
        predicts=torch.atanh(predicts)
        targets=torch.atanh(targets)
        # predicts_0_atanh = torch.atanh(predicts[:,:,0]).clone()
        # targets_0_atanh = torch.atanh(targets[:,:,0]).clone()
        # predicts_1=predicts[:,:,1].clone()
        # targets_1=targets[:,:,1].clone()
        # predicts_2=torch.atanh(predicts[:,:,2]).clone()
        # targets_2=torch.atanh(targets[:,:,2]).clone()
        # predicts=torch.stack((predicts_0_atanh,predicts_1,predicts_2),dim=2)
        # targets=torch.stack((targets_0_atanh,targets_1,targets_2),dim=2)


        predicts=predicts.reshape(predicts.shape[0]*predicts.shape[1],predicts.shape[2])
        targets=targets.reshape(targets.shape[0]*targets.shape[1],targets.shape[2])
        
        #gamma is the weight of the one-dimensional loss, if gamma is 0, the one-dimensional loss is not used
        if self.gamma == 0:
            one_dimensional_loss = 0
        else:
            one_dimensional_loss = self.swd(
                (targets, predicts),
                swd_type=self.specific_projection_swd_type)
        multi_dimensional_loss = 0
        if random_projection is not None:
            for i in range(random_projection.shape[0]):
                multi_dimensional_loss += self.swd(
                    self.projector(targets, predicts,
                                   random_projection[i]),
                    swd_type=self.random_projection_swd_type)
            multi_dimensional_loss /= random_projection.shape[0]
        else:
            for _ in range(self.repeat_projector_num):
                multi_dimensional_loss += self.swd(
                    self.projector(targets, predicts),
                    swd_type=self.random_projection_swd_type)
            multi_dimensional_loss /= self.repeat_projector_num
        if return_detail:
            return self.gamma * one_dimensional_loss + multi_dimensional_loss,one_dimensional_loss, multi_dimensional_loss
        else:
            return self.gamma * one_dimensional_loss + multi_dimensional_loss



