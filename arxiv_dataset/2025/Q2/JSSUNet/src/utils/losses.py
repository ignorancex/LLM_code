import torch
from torch import nn


class L1Loss_MSEstages(torch.nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.l1 = torch.nn.L1Loss()
        self.alpha = alpha

    def forward(self, output, gt, ):
        l1 = self.l1(output['pred'], gt['gt'])
        mse_stages = []
        for i in range(len(output['u_list'])):
            mse_stages.append(self.mse(output['u_list'][i], gt['gt']))
        mse_stages = torch.mean(torch.stack(mse_stages))
        loss = l1 + self.alpha * mse_stages
        return loss, {"l1": l1, "mse_stages": mse_stages}

    def components(self):
        return ["l1", "mse_stages"]

class L1(torch.nn.Module):
    def __init__(self, ):
        super(L1, self).__init__()
        self.L1 = torch.nn.L1Loss()

    def forward(self, output, gt):
        gt = gt['gt']
        pred = output['pred']
        l1 = self.L1(pred, gt)
        return l1, None


class SR_SSR_FusionLoss(torch.nn.Module):
    def __init__(self, alpha_ms, alpha_hs, alpha_fusion):
        super(SR_SSR_FusionLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.alpha_ms = alpha_ms
        self.alpha_hs = alpha_hs
        self.alpha_fusion = alpha_fusion

    def forward(self, output, gt):
        pred = output['pred']
        ms_list = output['ms_list']
        hs_list = output['hs_list']
        fusion_list = output['fused_list']
        out_l1 = self.l1(pred, gt['gt'])
        loss = out_l1

        ms_list_l1=torch.tensor([0])
        if self.alpha_ms > 0:
            ms_list_l1 = self.alpha_ms * self.l1(gt["rgb"], ms_list[-1]) / len(ms_list)
            for i in range(len(ms_list) - 1):
                ms_list_l1 += self.alpha_ms * self.l1(gt["rgb"], ms_list[i]) / len(ms_list)
            loss += ms_list_l1

        hs_list_l1=torch.tensor([0])
        if self.alpha_hs > 0:
            hs_list_l1 = self.alpha_hs * self.l1(gt["low_hs"], hs_list[-1]) / len(hs_list)
            for i in range(len(ms_list) - 1):
                hs_list_l1 = self.alpha_hs * self.l1(gt["low_hs"], hs_list[i]) / len(hs_list)
            loss += hs_list_l1

        fusion_list_l1=torch.tensor([0])
        if self.alpha_fusion > 0:
            fusion_list_l1 = self.alpha_fusion * self.l1(gt["gt"], fusion_list[-1]) / len(fusion_list)
            for i in range(len(fusion_list) - 1):
                fusion_list_l1 += self.alpha_fusion * self.l1(gt["gt"], fusion_list[i]) / len(fusion_list)
            loss += fusion_list_l1

        return loss, {"out_l1": out_l1.item(), "ms_list_l1": ms_list_l1.item(), "hs_list_l1": hs_list_l1.item(), "fusion_list_l1": fusion_list_l1.item()}
    def components(self):
        return ["out_l1", "ms_list_l1", "hs_list_l1", "fusion_list_l1"]