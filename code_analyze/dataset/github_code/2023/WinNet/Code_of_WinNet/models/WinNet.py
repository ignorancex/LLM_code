import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0)

    def matrix_padding1(self, matrix):          # 47 x 47 padding
        batch = matrix.size(0)
        channel = matrix.size(1)
        row = matrix.size(2)            # 3
        column = matrix.size(3)
        matrix_pad = torch.zeros((batch, channel, 47, 47), dtype=matrix.dtype, device=matrix.device)
        matrix_pad[:,:,12:row+12,12:column+12] = matrix[:,:,:,:]
        matrix_pad[:,:,13:row+12,0:column-1] = matrix[:,:,:-1,1:]
        matrix_pad[:,:,12:row+11,24:] = matrix[:,:,1:,0:-1]

        return matrix_pad

    def forward(self, x):
        # padding on the both ends of time series
        x = self.matrix_padding1(x)
        x = self.avg(x)

        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.mid_len = configs.mid_len
        self.channels = configs.enc_in
        self.in_ch = configs.n_times
        self.out_ch = configs.time_len

        self.CNN_List = nn.ModuleList()
        for i in range(self.channels):
            self.CNN_List.append(nn.Conv2d(2,1,3,1,0))

        self.fc1 = nn.Linear(self.seq_len, self.mid_len)
        self.fc2 = nn.Linear(self.mid_len, self.pred_len)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.sig = nn.Sigmoid()
        self.moving_avg = series_decomp(self.in_ch)

    def matrix_padding(self, matrix):           # 3 x 3 padding
        batch = matrix.size(0)
        channel = matrix.size(1)
        row = matrix.size(2)            # 3
        column = matrix.size(3)         # 3
        matrix_pad = torch.zeros((batch, channel, row+2, column+2), dtype=matrix.dtype, device=matrix.device)
        matrix_pad[:, :, 1:row+1, 1:column+1] = matrix[:, :, :, :]
        matrix_pad[:, :, 2:, 0] = matrix[:, :, :, column-1]
        matrix_pad[:, :, 1:row, -1] = matrix[:, :, 1:, 0]

        return matrix_pad

    def forward(self, x):
        # x: [Batch, Input length, Channel]

        B, L, S = x.shape
        seq_last = x[:, -1:, :].detach()
        oup = torch.zeros((B, S, self.in_ch, self.out_ch), dtype=x.dtype, device=x.device)
        x = x - seq_last        # Normalization

        x = x.permute(0, 2, 1)
        x_in = self.fc1(x).reshape(B, S, self.in_ch, -1)

        trend_x, season_x = self.moving_avg(x_in)
        x_trend_pad = self.matrix_padding(trend_x)
        x_season_pad = self.matrix_padding(season_x)

        x_between = x_in.transpose(-1, -2).contiguous()
        trend_bet, season_bet = self.moving_avg(x_between)
        bet_trend_pad = self.matrix_padding(trend_bet)
        bet_season_pad = self.matrix_padding(season_bet)

        for i in range(self.channels):
            x_m = x_in[:, i, :, :].unsqueeze(1).cuda()
            inp1 = torch.cat([x_season_pad[:, i, :, :].unsqueeze(1), x_trend_pad[:, i, :, :].unsqueeze(1)], dim=1)
            inp2 = torch.cat([bet_trend_pad[:, i, :, :].unsqueeze(1), bet_season_pad[:, i, :, :].unsqueeze(1)], dim=1)
            oup1 = self.drop(self.sig(self.CNN_List[i](inp1)))
            oup2 = self.drop(self.sig(self.CNN_List[i](inp2)))
            oup[:, i, :, :] = torch.squeeze(oup1 + oup2 + x_m)

        oup = self.fc2(oup.reshape(B, S, -1)).permute(0, 2, 1)
        x = oup + seq_last

        return x    # [Batch, Output length, Channel]


