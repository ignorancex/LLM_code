'''
view数量需要自定义
conv 卷积核和步长需要自己定义, 层数也要自己定义.
lstm输入和输出参数要定义.
'''
import torch
import torch.nn as nn
from typing import Tuple, List

class DeepSense(nn.Module):
    channel = 64 #channel_first
    freq = 15
    sensor_num = 2
    gru_in_feature = 705

    def __init__(self, data_dict :dict, device):
        super(DeepSense, self).__init__()

        self.time_width = torch.tensor(data_dict['acc_data'].shape[2]).float().to(device)
        self.T = data_dict['acc_data'].shape[1]
        self.device = device

        class_num = data_dict['label'].shape[-1]
        acc_dim = data_dict['acc_data'].shape[-1]
        gyr_dim = data_dict['gyr_data'].shape[-1]

        self.acc_conv_1, self.acc_conv_2_3 = self.sensor_conv(acc_dim)
        self.gyr_conv_1, self.gyr_conv_2_3 = self.sensor_conv(gyr_dim)

        self.conv_4, self.conv_5_6 = self.scope_conv()

        self.gru_1 = nn.GRU(self.gru_in_feature, 256, batch_first=False)
        self.gru_2 = nn.GRU(256, 64)
        self.predictor = nn.Sequential(nn.ReLU(), nn.Linear(64, class_num))

    def forward(self, inputs :List[torch.Tensor]):
        accs, gyrs = self.exp_per(inputs)  # (T, batch_size, 1, dimension, 2f)

        inputs = []
        for idx in range(self.T) :
            acc = self.hierarchical_cnn(accs[idx], self.acc_conv_1, self.acc_conv_2_3) # (batch_size, 96)
            gyr = self.hierarchical_cnn(gyrs[idx], self.gyr_conv_1, self.gyr_conv_2_3) # (batch_size, 96)

            input = self.mer_exp_per(acc, gyr) # (batch_size,1 , sensor_num[4], 384)
            input = self.hierarchical_cnn(input, self.conv_4, self.conv_5_6) # (batch_size, *[704])
            input = self.concat_time(input) # (batch_size, *[704])       
            
            inputs.append(input)

        inputs = torch.stack(inputs, dim = 0) # (T, batch_size, *[705])

        #h0 = torch.randn(self.gru_layers, inputs.shape[1], self.hidden).to(self.device)
        inputs, _ = self.gru_1(inputs)
        inputs, _ = self.gru_2(inputs)

        label = self.predictor(inputs[-1])

        return label

    def concat_time(self, input):
        time_width = torch.tensor(self.time_width).float().repeat(input.shape[0], 1)
        input = torch.cat((input, time_width), dim=1)

        return input

    def mer_exp_per(self, *input :torch.Tensor):
        '''
        in : sensor_num * (batch_size, *)
        out : (batch_size,1 , sensor_num, *)
        '''
        input = torch.stack(input, dim=1)  # (batch_size, 4, *)
        input = input.expand((1,) + input.shape)  # (1, batch_size, sensor_num, *)
        input = input.permute(1, 0, 2, 3) # (batch_size,1 , sensor_num, *)

        return input

    def hierarchical_cnn(self, input :torch.Tensor, conv_1 :nn.Module, conv_2_3: nn.Module):
        '''
        in : (batch_size, 1, dimension, 2f) | (batch_size, 1, sensor_num, *)
        out : (batch_size, *)
        '''
        input = conv_1(input)  # (batch_size, channel, 1, *)
        input = torch.squeeze(input, dim=2)  # (batch_size, channel,*)
        input = conv_2_3(input)  # (batch_size, channel, *)
        input = torch.flatten(input, start_dim=1)  # (batch_size, *)

        return input

    def exp_per(self, inputs :List[torch.Tensor]):
        '''
        in : (sensor_num, batch_size, T, 2f, dimension)
        out : (sensor_num, T, batch_size, 1, dimension, 2f)
        '''
        for idx in range(len(inputs)):
            input = inputs[idx]
            input = input.expand((1,) + input.shape)
            input  = input.permute(2, 1, 0, 4, 3)
            inputs[idx] = input

        return inputs

    def sensor_conv(self, dimension) -> Tuple[nn.Module, nn.Module]:
        '''
        in : (batch_size, channel = 1, dimension, 2f[15])
        out : (batch_size, channel, 1, *[]), (batch_size, T, *)
        '''
        conv_1 = nn.Conv2d(1, self.channel, (dimension, 3)) #(batch_size, channel, 1, *)

        # (batch_size, channel, *) -> #(batch_size, channel, *)
        conv_2_3 = nn.Sequential(
            nn.BatchNorm1d(self.channel), nn.ReLU(),
            nn.Conv1d(self.channel, self.channel, 4, stride=2), nn.BatchNorm1d(self.channel), nn.ReLU(),
            nn.Conv1d(self.channel, self.channel, 2, stride=2), nn.BatchNorm1d(self.channel), nn.ReLU(),
        )

        return conv_1, conv_2_3

    def scope_conv(self) -> Tuple[nn.Module, nn.Module]:
        '''
        in : (batch_size, 1, sensor_num [4], *[384])
        out : (batch_size, channel, 1, *[380]), (batch_size, channel, *[11])
        '''
        conv_4 = nn.Conv2d(1, self.channel, (self.sensor_num, 5)) # (batch_size, channel, 1, *)

        # (batch_size, channel, *) -> #(batch_size, channel, *)
        conv_5_6 = nn.Sequential(
            nn.BatchNorm1d(self.channel), nn.ReLU(),
            nn.Conv1d(self.channel, self.channel, 8, 8), nn.BatchNorm1d(self.channel), nn.ReLU(),
            nn.Conv1d(self.channel, self.channel, 4, 4), nn.BatchNorm1d(self.channel), nn.ReLU(),
        )

        return conv_4, conv_5_6

    def regular(self):
        return torch.tensor(0.)
