import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

SUPPORTED_POOLING_LAYERS = ['max', 'avg']


class ConvRelu(nn.Module):
    '''
    Class implementing a convolutional layer followed by batch norm and relu
    - Conv3d - [Batch normalization] - Activation
    '''
    def __init__(self, num_input_channels, num_output_channels, use_batchnorm=True, activation='ELU', kernel_size=5, stride=1, padding=2):
        super(ConvRelu, self).__init__()

        conv = nn.Conv3d(in_channels=num_input_channels, out_channels=num_output_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=not(use_batchnorm))
        nn.init.xavier_uniform_(conv.weight)
        activ_unit = nn.ELU()
        
        if use_batchnorm:
            self.conv = nn.Sequential(conv, nn.BatchNorm3d(num_output_channels), activ_unit)
        else:
            self.conv = nn.Sequential(conv)

    def forward(self, inputs):
        return self.conv(inputs)


class ConvolutionalBlock(nn.Module):

    def __init__(self, num_input_channels, num_output_channels, use_batchnorm=True, activation='ELU'):
        super(ConvolutionalBlock, self).__init__()

        self.conv1 = ConvRelu(num_input_channels, num_output_channels, use_batchnorm, activation)



    def forward(self, inputs):
        outputs = self.conv1(inputs)

        return outputs


class DeConvolutionalBlock(nn.Module):

    def __init__(self, num_input_channels, num_output_channels, use_batchnorm=True, activation='ELU'):
        super(DeConvolutionalBlock, self).__init__()

        self.conv1 = ConvRelu(num_input_channels, num_output_channels, use_batchnorm, activation)



    def forward(self, inputs):
        outputs = self.conv1(inputs)

        return outputs

class BottleConvolutionalBlock(nn.Module):


    def __init__(self, num_input_channels, num_output_channels, use_batchnorm=True, activation='ELU'):
        super(BottleConvolutionalBlock, self).__init__()

        self.conv1 = ConvRelu(num_input_channels, num_output_channels, use_batchnorm, activation)


    def forward(self, inputs):
        outputs = self.conv1(inputs)

        return outputs
