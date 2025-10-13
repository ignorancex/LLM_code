import torch
from torch import nn
import torch
import torch.nn as nn

class NPSNR(nn.Module):
    def __init__(self, Lambda=0.1, Fs=30, LowF=0.6, upF=3.0, width=0.4):
        super(NPSNR, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.Lambda = Lambda
        self.Fs = Fs  # 采样率
        self.LowF = LowF
        self.upF = upF
        self.width = width
        self.NormaliceK = 1 / 10.9  # Constant to normalize SNR between -1 and 1

    def forward(self, rppg, gt):
        # 确保输入是二维的 (batch, length)
        assert rppg.ndim == 2 and gt.ndim == 2, "Input tensors should be 2-dimensional (batch, length)"
        assert rppg.shape == gt.shape, "Prediction and ground truth should have the same shape"

        loss = 0
        for i in range(rppg.shape[0]):
            # 皮尔逊相关系数
            sum_x = torch.sum(rppg[i])
            sum_y = torch.sum(gt[i])
            sum_xy = torch.sum(rppg[i] * gt[i])
            sum_x2 = torch.sum(torch.pow(rppg[i], 2))
            sum_y2 = torch.sum(torch.pow(gt[i], 2))
            N = rppg.shape[1]
            pearson = (N * sum_xy - sum_x * sum_y) / (
                torch.sqrt((N * sum_x2 - torch.pow(sum_x, 2)) * (N * sum_y2 - torch.pow(sum_y, 2)))
            )
            # 处理分母为0的情况,防止nan
            pearson = torch.nan_to_num(pearson, nan=0.0)

            # SNR
            N = rppg.shape[1] * 3  # 3倍补零
            freq = torch.arange(0, N, 1, device=self.device) * self.Fs / N
            fft = torch.abs(torch.fft.fft(rppg[i], dim=-1, n=N)) ** 2
            gt_fft = torch.abs(torch.fft.fft(gt[i], dim=-1, n=N)) ** 2
            fft = fft.masked_fill(torch.logical_or(freq > self.upF, freq < self.LowF), 0)
            gt_fft = gt_fft.masked_fill(torch.logical_or(freq > self.upF, freq < self.LowF), 0)
            PPG_peaksLoc = freq[gt_fft.argmax()]
            mask = torch.zeros(fft.shape[-1], dtype=torch.bool, device=self.device)
            mask = mask.masked_fill(
                torch.logical_and(freq < PPG_peaksLoc + (self.width / 2), PPG_peaksLoc - (self.width / 2) < freq), 1)  # Main signal
            mask = mask.masked_fill(
                torch.logical_and(freq < PPG_peaksLoc * 2 + (self.width / 2), PPG_peaksLoc * 2 - (self.width / 2) < freq),
                1)  # Armonic
            power = fft * mask
            noise = fft * mask.logical_not()
            SNR = (10 * torch.log10(power.sum() / noise.sum())) * self.NormaliceK
             # 处理除以0的情况,防止inf或nan
            SNR = torch.nan_to_num(SNR, nan=0.0, posinf=0.0, neginf=0.0)


            # 组合损失
            loss += 1 - (pearson + (self.Lambda * SNR))

        loss = loss / rppg.shape[0]
        return loss
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# loss_fun=NPSNR(0.3,30)
# # x = torch.randn(2, 128) .to(device)
# # y = torch.randn(2, 128).to(device)
# print(x.shape)


# loss_fun(x,y)