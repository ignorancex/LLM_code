"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_util.module import single_ftran, single_fmult, mul_ftran, mul_fmult, crop_images
from torch.utils.data import Dataset
from sota_module.fwd.pmri import ftran as ftran_pmri
from sota_module.fwd.pmri import fmult as fmult_pmri
from sota_module.baseline.e2e_varnet import fastmri
from torch_util.common import dict2pformat, write_test, abs_helper, check_and_mkdir, plot_helper

class inputDataDict(Dataset):
    def __init__(self, x_init, P, S, y, module_name=None):
        self.x_init = x_init
        self.P = P
        self.S = S
        self.y = y
        self.module_name = module_name

        if self.module_name == "ISTANET":
            self.ftran = lambda y_: ftran_pmri(y=y_, smps=S, mask=P)
            self.fmult = lambda x_: fmult_pmri(x=x_, smps=S, mask=P)

    def __len__(self):
        return 1

    def getData(self):
        if self.module_name == "ISTANET":
            return self.x_init, self.P, self.S, self.y, self.ftran, self.fmult
        else:
            return self.x_init, self.P, self.S, self.y

# def gradient_loss(s, penalty='l2') function is given by Ph.D. student Yuyang Hu
def gradient_loss(s, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
    dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])
    if penalty == 'l2':
        dy = dy * dy
        dx = dx * dx
    d = torch.mean(dx) + torch.mean(dy)
    return d / 2.0

class SPICELoss(nn.Module):
    """
    SSIM loss module.
    """
    def __init__(self):
        """
        Args:
        """
        super().__init__()
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # self.recon_loss_fn = nn.SmoothL1Loss()
        self.recon_loss_fn =nn.L1Loss()

    def forward(self, X: torch.Tensor, Y: torch.Tensor, P: torch.Tensor, S: torch.Tensor, S_estimated:torch.Tensor):
        """
        loss function: recon_loss = L(A' x, y') + L(A x', y)
        A: Estimated forward operator using S (PFS)
        A': Dataset's forward operator (PFS')
        x: reconstructed image when y' is the input measurement
        x': reconstructed image when y' is the input measurement
        y: The input of S in A which is estimated by the coil sensitivity estimation module. (In my application, y and y' are the same.)
        y': Raw measurement
        """
        recon_loss_1 = self.recon_loss_fn(mul_fmult(torch.view_as_complex(X.permute([0, 2, 3, 1]).contiguous()), S_estimated, P), Y)
        smooth_loss = gradient_loss(S)

        recon_loss = recon_loss_1 + smooth_loss

        return recon_loss

class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size**2)
        NP = win_size**2
        self.cov_norm = NP / (NP - 1)
    '''
    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        data_range: torch.Tensor,
        reduced: bool = True,
    ):
    '''

    def forward(
            self,
            X: torch.Tensor,
            Y: torch.Tensor,
            data_range=torch.Tensor([1.0]),
            reduced: bool = True,
    ):
        assert isinstance(self.w, torch.Tensor)

        X = X.permute([0, 2, 3, 1]).contiguous()
        X = torch.view_as_complex(X)
        Y = Y.permute([0, 2, 3, 1]).contiguous()
        Y = torch.view_as_complex(Y)
        X = X.abs().unsqueeze(1)
        Y = Y.abs().unsqueeze(1)

        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)  # typing: ignore
        uy = F.conv2d(Y, self.w)  #
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        C1 = C1.cuda()
        C2 = C2.cuda()
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux**2 + uy**2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        if reduced:
            return 1 - S.mean()
        else:
            return 1 - S

class ISTANETLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self):
        """
        Args:
        """
        super().__init__()
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.num_layers = 9
    def forward(
            self,
            X: torch.Tensor,
            Y: torch.Tensor,
            loss_layers_sym):

        # Compute and print loss
        loss_discrepancy = torch.mean(torch.pow(X - Y, 2))

        loss_constraint = torch.mean(torch.pow(loss_layers_sym[0], 2))
        for k in range(self.num_layers - 1):
            loss_constraint += torch.mean(torch.pow(loss_layers_sym[k + 1], 2))

        gamma = torch.Tensor([0.01]).to(self.device)

        # loss_all = loss_discrepancy
        loss_all = loss_discrepancy + torch.mul(gamma, loss_constraint)

        return loss_all

class ISTANETLossForDC(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self):
        """
        Args:
        """
        super().__init__()
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.num_layers = 9
    def forward(
            self,
            X: torch.Tensor,
            Y: torch.Tensor,
            loss_layers_sym):

        # Compute and print loss
        loss_discrepancy = torch.mean(torch.abs(X - Y))

        loss_constraint = torch.mean(torch.pow(loss_layers_sym[0], 2))
        for k in range(self.num_layers - 1):
            loss_constraint += torch.mean(torch.pow(loss_layers_sym[k + 1], 2))

        gamma = torch.Tensor([0.01]).to(self.device)

        # loss_all = loss_discrepancy
        loss_all = loss_discrepancy + torch.mul(gamma, loss_constraint)

        return loss_all
#
#
# class Rotate():
#     def __init__(self, n_trans, random_rotate=False):
#         self.n_trans = n_trans
#         self.random_rotate = random_rotate
#     def apply(self, x, S):
#         return rotate_dgm(x, S, self.n_trans, self.random_rotate)
#
# def rotate_dgm(dataX, dataS, n_trans=5, random_rotate=False):
#     if random_rotate:
#         theta_list = random.sample(list(np.arange(1, 360)), n_trans)
#     else:
#         theta_list = np.arange(10, 360, int(360 / n_trans))
#
#     if len(dataS.shape) == 4:
#         dataS = torch.view_as_real(dataS).permute([0, 4, 2, 3, 1]).contiguous()
#
#     # print(f"theta_list: {theta_list}")
#     # print(f"[Beginning] data.shape: {data.shape}")
#     dataX = torch.cat([dataX if theta == 0 else dgm.geometry.transform.rotate(dataX, torch.Tensor([theta]).type_as(dataX)) for theta in theta_list], dim=0)
#     # print(f"[Beginning] data.shape: {data.shape}")
#
#     # Rotation for dataS
#     rotated_list = []
#
#     for jj in range(dataS.shape[4]):
#         rotated_for_current_theta = torch.cat([dataS[..., jj] if theta == 0 else dgm.geometry.transform.rotate(dataS[..., jj], torch.Tensor([theta]).type_as(dataS)) for theta in theta_list], dim=0)
#         rotated_list.append(rotated_for_current_theta)
#     dataS = torch.stack(rotated_list, dim=0)
#
#     dataS = torch.view_as_complex(dataS.permute([1,0,3,4,2]).contiguous())
#
#     return [dataX, dataS]



import numpy as np
# import kornia as dgm
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random

class Rotate():
    def __init__(self, n_trans, random_rotate=False):
        self.n_trans = n_trans
        self.random_rotate = random_rotate
    def apply(self, x, S):
        return rotate_dgm(x, S , self.n_trans, self.random_rotate)

def rotate_dgm(dataX, dataS, n_trans=5, random_rotate=False):
    if random_rotate:
        theta_list = random.sample(list(np.arange(1, 360)), n_trans)
    else:
        theta_list = np.arange(10, 360, int(360 / n_trans))
    if len(dataS.shape) == 4:
        dataS = torch.view_as_real(dataS).permute([0, 4, 2, 3, 1]).contiguous()


    # Rotation for dataX
    dataX = torch.cat([dataX if theta == 0 else TF.rotate(dataX, int(theta)) for theta in theta_list], dim=0)

    # Rotation for dataS
    rotated_list = []

    for jj in range(dataS.shape[4]):
        rotated_for_current_theta = torch.cat([dataS[..., jj] if theta == 0 else TF.rotate(dataS[..., jj], int(theta)) for theta in theta_list], dim=0)
        rotated_list.append(rotated_for_current_theta)
    dataS = torch.stack(rotated_list, dim=0)

    dataS = torch.view_as_complex(dataS.permute([1,0,3,4,2]).contiguous())

    return [dataX, dataS]

class REILoss(nn.Module):
    """
    REI loss module.
    """

    def __init__(self, recon_module, module_name, config):
        """
        Args:
        """
        super().__init__()
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.f = lambda y: recon_module(torch.view_as_real(mul_ftran(y)).permute([0, 3, 1, 2]).contiguous())
        self.config = config
        self.recon_module = recon_module
        self.module_name = module_name
        self.transform = Rotate(n_trans=5, random_rotate=True)

    def forward(
            self,
            y0: torch.Tensor,
            x0: torch.Tensor,
            x1: torch.Tensor,
            y1: torch.Tensor,
            S: torch.Tensor,
            P: torch.Tensor
    ):
        '''
        :param y0: y
        :param x0: y_tran
        :param x1: y_tran_recon
        :param y1: fmult(y_tran_recon)
        :param S: Sensitivity map
        :param P: Mask
        :return:
        '''
        sigma = self.config['rei']['sigma_rei']
        tau = self.config['rei']['tau']
        acceleration_rate = self.config['setting']['acceleration_rate']
        alpha_req = self.config['rei']['alpha_req']
        alpha_mc = 1.0
        alpha_eq = 1.0
        # criterion = torch.nn.MSELoss().to(self.device)
        criterion_y = torch.nn.L1Loss().to(self.device)
        criterion = torch.nn.MSELoss().to(self.device)

        loss_mc = alpha_mc * criterion_y(y1, y0)

        x2, S2 = self.transform.apply(x1, S)
        # x2 = transform.apply(x1)
        y_x2 = mul_fmult(torch.view_as_complex(x2.permute([0, 2, 3, 1]).contiguous()), S=S2, P=P, add_noise=False,
                         sigma=sigma)
        # print(f"mul_ftran(y, S=S, P=P)).shape: {mul_ftran(y, S=S, P=P).shape}")
        f = lambda measurement: self.recon_module(
            inputDataDict(x_init=mul_ftran(measurement, S=S2, P=P).permute([0, 3, 1, 2]).contiguous(), P=P, S=S2,
                          y=y_x2, module_name=self.module_name))

        x3 = f(
            mul_fmult(torch.view_as_complex(x2.permute([0, 2, 3, 1]).contiguous()), S=S2, P=P, add_noise=False,
                      sigma=sigma))

        loss_y_eq = alpha_eq * criterion(
            mul_fmult(torch.view_as_complex(x3.permute([0, 2, 3, 1]).contiguous()), S=S2, P=P),
            mul_fmult(torch.view_as_complex(x2.permute([0, 2, 3, 1]).contiguous()), S=S2, P=P))

        x2 = crop_images(x2)
        x3 = crop_images(x3)

        # print(f"y0.shape: {y0.shape} / y0.dtype: {y0.dtype}")
        # print(f"mul_fmult(torch.view_as_complex(x2.permute([0, 2, 3, 1]).contiguous()), S=S, P=P, add_noise=False,sigma=sigma).shape: {mul_fmult(torch.view_as_complex(x2.permute([0, 2, 3, 1]).contiguous()), S=S, P=P, add_noise=False,sigma=sigma).shape} / {mul_fmult(torch.view_as_complex(x2.permute([0, 2, 3, 1]).contiguous()), S=S, P=P, add_noise=False,sigma=sigma).dtype}")
        # print(f"mul_ftran(x2_y, S=S, P=P).shape: {mul_ftran(x2_y, S=S, P=P).shape}")
        # input_data = inputDataDict(x_init=mul_ftran(x2_y, S=S, P=P).permute([0, 3, 1, 2]).contiguous(), P=P, S=S, y=x2_y, module_name=self.module_name)

        # [x3, second_loss_layers_sym] = self.recon_module(input_data)

        # plot_helper(file_path="rotation.png",
        #             # img1=(torch.view_as_complex(x_gt.permute([0, 2, 3, 1]).contiguous()).detach().cpu())[0],
        #             img1=(torch.view_as_complex(x1.permute([0, 2, 3, 1]).contiguous()).detach().cpu())[0],
        #             img2=(torch.view_as_complex(x2.permute([0, 2, 3, 1]).contiguous()).detach().cpu())[0],
        #             img3=(torch.view_as_complex(x3.permute([0, 2, 3, 1]).contiguous()).detach().cpu())[0],
        #             img4=(torch.view_as_complex(x3.permute([0, 2, 3, 1]).contiguous()).detach().cpu())[0],
        #             img1_name='x1', img2_name='x2', img3_name='x3',
        #             img4_name='x4', title='Rotation Test')

        # compute loss_req
        loss_eq = alpha_eq * criterion(x3, x2)
        # loss_eq = alpha_eq * criterion(mul_fmult(torch.view_as_complex(x3.permute([0, 2, 3, 1]).contiguous()), S = S, P = P), mul_fmult(torch.view_as_complex(x2.permute([0, 2, 3, 1]).contiguous()), S = S, P = P))

        # loss_all = loss_mc + loss_eq
        # loss_all = loss_mc

        loss_all = loss_mc + loss_eq + loss_y_eq
        # loss_all = loss_mc + torch.mul(gamma, loss_constraint)# + torch.mul(gamma, second_loss_constraint)

        # del loss_layers_sym
        # del second_loss_layers_sym

        return loss_all



class REILossForISTA(nn.Module):
    """
    REI loss module.
    """

    def __init__(self, recon_module, module_name, config):
        """
        Args:
        """
        super().__init__()
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.recon_module = recon_module
        self.module_name = module_name
        self.config = config
        self.transform = Rotate(n_trans=5, random_rotate=True)
        self.num_layers = 9

    def forward1(
            self,
            y0: torch.Tensor,
            x0: torch.Tensor,
            x1: torch.Tensor,
            y1: torch.Tensor,
            S: torch.Tensor,
            P: torch.Tensor,
            loss_layers_sym):

        # print(f"Loss 1: {torch.cuda.memory_allocated(device=None) / (1024 * 1024 * 1024):2f}")
        # Compute and print loss
        # loss_discrepancy = torch.mean(torch.abs(y0 - y1))

        # loss_discrepancy = torch.mean(torch.abs(y0 - y1))
        loss_discrepancy = torch.nn.L1Loss()(y0,y1)


        loss_constraint = torch.mean(torch.pow(loss_layers_sym[0], 2))
        for k in range(self.num_layers - 1):
            loss_constraint += torch.mean(torch.pow(loss_layers_sym[k + 1], 2))

        gamma = torch.Tensor([0.01]).to(self.device)

        # loss_all = loss_discrepancy
        loss_all = loss_discrepancy + torch.mul(gamma, loss_constraint)

        # print(f"Loss 2: {torch.cuda.memory_allocated(device=None) / (1024 * 1024 * 1024):2f}")

        return loss_all

    def forwardrei(
            self,
            y0: torch.Tensor,
            x0: torch.Tensor,
            x1: torch.Tensor,
            y1: torch.Tensor,
            S: torch.Tensor,
            P: torch.Tensor,
            loss_layers_sym
            ):
        '''
        :param y0: y
        :param x0: y_tran
        :param x1: y_tran_recon
        :param y1: fmult(y_tran_recon)
        :param S: Sensitivity map
        :param P: Mask
        :return:
        '''
        sigma = self.config['rei']['sigma_rei']
        tau = self.config['rei']['tau']
        acceleration_rate = self.config['setting']['acceleration_rate']
        alpha_req = self.config['rei']['alpha_req']
        criterion = torch.nn.MSELoss().to(self.device)

        # loss_discrepancy = torch.mean(torch.abs(y0 - y1))

        sigma2 = sigma**2
        b = torch.randn_like(x0)
        b = mul_fmult(torch.view_as_complex(b.permute([0, 2, 3, 1]).contiguous()), S, P)
        # y2 = mul_fmult(torch.view_as_complex(self.recon_module(inputDataDict(mul_ftran(y0 + tau*b, S = S, P = P).permute([0, 3, 1, 2]).contiguous(), P, S, y0, module_name=self.module_name))[0].permute([0, 2, 3, 1]).contiguous()), S = S, P = P)
        y2 = mul_fmult(torch.view_as_complex(self.recon_module(inputDataDict(mul_ftran(y0 + tau*b, S = S, P = P).permute([0, 3, 1, 2]).contiguous(), P, S, y0+tau*b, module_name=self.module_name))[0].permute([0, 2, 3, 1]).contiguous()), S = S, P = P)

        # compute batch size K
        K = y0.shape[0]
        # compute n (dimension of x)
        n = y0.shape[-1] * y0.shape[-2] * y0.shape[-3]

        # compute m (dimension of y)
        m = n / acceleration_rate  # dim(y)

        # compute loss_sure
        # loss_sure = torch.sum((y1 - y0).pow(2)) / (K * m) - sigma2 + (2 * sigma2 / (tau * m * K)) * (b * (y2 - y1)).sum()
        loss_sure = torch.sum(torch.abs(y1 - y0)) / (K * m) - sigma2 + (2 * sigma2 / (tau * m * K)) * (b * (y2 - y1)).sum()

        # REQ (EI with noisy input)
        x2, S2 = self.transform.apply(x1, S)
        # S2 = self.transform.apply(S)
        # print(f"mul_ftran(y, S=S, P=P)).shape: {mul_ftran(y, S=S, P=P).shape}")
        # print(f"x2.shape: {x2.shape}")
        # print(f"S.shape: {S.shape} / S.dtype: {S.dtype}")
        # print(f"S2.shape: {S2.shape} / S2.dtype: {S2.dtype}")
        y_x2 = mul_fmult(torch.view_as_complex(x2.permute([0, 2, 3, 1]).contiguous()), S=S, P=P, add_noise=False,sigma=sigma)
        f = lambda measurement: self.recon_module(inputDataDict(mul_ftran(measurement, S=S, P=P).permute([0, 3, 1, 2]).contiguous(), P, S, y_x2, module_name=self.module_name))
        [x3, second_loss_layers_sym] = f(mul_fmult(torch.view_as_complex(x2.permute([0, 2, 3, 1]).contiguous()), S=S, P=P, add_noise=False, sigma = sigma))
        # compute loss_req
        x2 = crop_images(x2)
        x3 = crop_images(x3)

        loss_req = alpha_req * criterion(x3, x2)


        plot_helper(file_path="rotation.png",
                    # img1=(torch.view_as_complex(x_gt.permute([0, 2, 3, 1]).contiguous()).detach().cpu())[0],
                    img1=(torch.view_as_complex(x1.permute([0, 2, 3, 1]).contiguous()).detach().cpu())[0],
                    img2=(torch.view_as_complex(x2.permute([0, 2, 3, 1]).contiguous()).detach().cpu())[0],
                    img3=(torch.view_as_complex(x3.permute([0, 2, 3, 1]).contiguous()).detach().cpu())[0],
                    img4=(torch.view_as_complex(x3.permute([0, 2, 3, 1]).contiguous()).detach().cpu())[0],
                    img1_name='x1', img2_name='x2', img3_name='x2Process',
                    img4_name='x4', title='Rotation Test')

        loss_constraint = torch.mean(torch.pow(loss_layers_sym[0], 2))
        second_loss_constraint = torch.mean(torch.pow(second_loss_layers_sym[0], 2))
        for k in range(self.num_layers - 1):
            loss_constraint += torch.mean(torch.pow(loss_layers_sym[k + 1], 2))
            second_loss_constraint += torch.mean(torch.pow(second_loss_layers_sym[k + 1], 2))

        gamma = torch.Tensor([0.01]).to(self.device)

        loss_all = loss_sure + loss_req + torch.mul(gamma, loss_constraint) + torch.mul(gamma, second_loss_constraint)# + loss_discrepancy
        # loss_all = loss_discrepancy + torch.mul(gamma, loss_constraint)

        # del loss_layers_sym
        # del second_loss_layers_sym

        return loss_all

    def forward(
            self,
            y0: torch.Tensor,
            x0: torch.Tensor,
            x1: torch.Tensor,
            y1: torch.Tensor,
            S: torch.Tensor,
            P: torch.Tensor,
            loss_layers_sym
    ):
        '''
        :param y0: y
        :param x0: y_tran
        :param x1: y_tran_recon
        :param y1: fmult(y_tran_recon)
        :param S: Sensitivity map
        :param P: Mask
        :return:
        '''
        sigma = self.config['rei']['sigma_rei']
        tau = self.config['rei']['tau']
        acceleration_rate = self.config['setting']['acceleration_rate']
        alpha_req = self.config['rei']['alpha_req']
        alpha_mc = 1.0
        alpha_eq = 1.0
        # criterion = torch.nn.MSELoss().to(self.device)
        criterion_y = torch.nn.L1Loss().to(self.device)
        criterion = torch.nn.MSELoss().to(self.device)
        criterion_ssim = SSIMLoss().to(self.device)
        transform = Rotate(n_trans=2, random_rotate=True)

        loss_mc = alpha_mc * criterion_y(y1, y0)

        x2, S2 = self.transform.apply(x1, S)
        # x2 = transform.apply(x1)
        y_x2 = mul_fmult(torch.view_as_complex(x2.permute([0, 2, 3, 1]).contiguous()), S=S2, P=P, add_noise=False,sigma=sigma)
        # print(f"mul_ftran(y, S=S, P=P)).shape: {mul_ftran(y, S=S, P=P).shape}")
        f = lambda measurement: self.recon_module(
            inputDataDict(x_init=mul_ftran(measurement, S=S2, P=P).permute([0, 3, 1, 2]).contiguous(), P=P, S=S2, y=y_x2, module_name=self.module_name))

        [x3, second_loss_layers_sym] = f(
            mul_fmult(torch.view_as_complex(x2.permute([0, 2, 3, 1]).contiguous()), S=S2, P=P, add_noise=False,sigma=sigma))


        loss_y_eq = alpha_eq * criterion(mul_fmult(torch.view_as_complex(x3.permute([0, 2, 3, 1]).contiguous()), S = S2, P = P), mul_fmult(torch.view_as_complex(x2.permute([0, 2, 3, 1]).contiguous()), S = S2, P = P))

        x2 = crop_images(x2)
        x3 = crop_images(x3)

        # print(f"y0.shape: {y0.shape} / y0.dtype: {y0.dtype}")
        # print(f"mul_fmult(torch.view_as_complex(x2.permute([0, 2, 3, 1]).contiguous()), S=S, P=P, add_noise=False,sigma=sigma).shape: {mul_fmult(torch.view_as_complex(x2.permute([0, 2, 3, 1]).contiguous()), S=S, P=P, add_noise=False,sigma=sigma).shape} / {mul_fmult(torch.view_as_complex(x2.permute([0, 2, 3, 1]).contiguous()), S=S, P=P, add_noise=False,sigma=sigma).dtype}")
        # print(f"mul_ftran(x2_y, S=S, P=P).shape: {mul_ftran(x2_y, S=S, P=P).shape}")
        # input_data = inputDataDict(x_init=mul_ftran(x2_y, S=S, P=P).permute([0, 3, 1, 2]).contiguous(), P=P, S=S, y=x2_y, module_name=self.module_name)

        # [x3, second_loss_layers_sym] = self.recon_module(input_data)

        plot_helper(file_path="rotation.png",
                    # img1=(torch.view_as_complex(x_gt.permute([0, 2, 3, 1]).contiguous()).detach().cpu())[0],
                    img1=(torch.view_as_complex(x1.permute([0, 2, 3, 1]).contiguous()).detach().cpu())[0],
                    img2=(torch.view_as_complex(x2.permute([0, 2, 3, 1]).contiguous()).detach().cpu())[0],
                    img3=(torch.view_as_complex(x3.permute([0, 2, 3, 1]).contiguous()).detach().cpu())[0],
                    img4=(torch.view_as_complex(x3.permute([0, 2, 3, 1]).contiguous()).detach().cpu())[0],
                    img1_name='x1', img2_name='x2', img3_name='x3',
                    img4_name='x4', title='Rotation Test')

        # compute loss_req
        loss_eq = alpha_eq * criterion_ssim(x3, x2)
        # loss_eq = alpha_eq * criterion(mul_fmult(torch.view_as_complex(x3.permute([0, 2, 3, 1]).contiguous()), S = S, P = P), mul_fmult(torch.view_as_complex(x2.permute([0, 2, 3, 1]).contiguous()), S = S, P = P))


        # loss_all = loss_mc + loss_eq
        # loss_all = loss_mc

        loss_constraint = torch.mean(torch.pow(loss_layers_sym[0], 2))
        second_loss_constraint = torch.mean(torch.pow(second_loss_layers_sym[0], 2))
        for k in range(self.num_layers - 1):
            loss_constraint += torch.mean(torch.pow(loss_layers_sym[k + 1], 2))
            second_loss_constraint += torch.mean(torch.pow(second_loss_layers_sym[k + 1], 2))

        gamma = torch.Tensor([0.01]).to(self.device)

        loss_all = loss_mc + loss_eq + loss_y_eq + torch.mul(gamma, loss_constraint) + torch.mul(gamma, second_loss_constraint)
        # loss_all = loss_mc + torch.mul(gamma, loss_constraint)# + torch.mul(gamma, second_loss_constraint)


        # del loss_layers_sym
        # del second_loss_layers_sym

        return loss_all
