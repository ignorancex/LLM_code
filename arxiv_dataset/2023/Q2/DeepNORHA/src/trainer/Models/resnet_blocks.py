
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvResNet(nn.Module):
    '''
    Class implementing a convolutional layer followed by batch norm and relu
    - Conv3d - [Batch normalization] - Activation
    '''

    def __init__(self, num_input_channels, num_output_channels, use_batchnorm=True, kernel_size=3, stride=1, padding=1):
        super(ConvResNet, self).__init__()

        conv = nn.Conv3d(in_channels=num_input_channels, out_channels=num_output_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=not(use_batchnorm))
        nn.init.xavier_uniform_(conv.weight)

        if use_batchnorm:
            self.conv = nn.Sequential(conv, nn.BatchNorm3d(num_output_channels))
        else:
            self.conv = conv

    def forward(self, inputs):
        return self.conv(inputs)



class DeConvResNet(nn.Module):
    '''
    Class implementing a convolutional layer followed by batch norm and relu
    - Conv3d - [Batch normalization] - Activation
    '''

    def __init__(self, num_input_channels, num_output_channels, use_batchnorm=True, kernel_size=3, stride=1, padding=1):
        super(DeConvResNet, self).__init__()

        conv = nn.ConvTranspose3d(in_channels=num_input_channels, out_channels=num_output_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=not(use_batchnorm))
        nn.init.xavier_uniform_(conv.weight)

        if use_batchnorm:
            self.conv = nn.Sequential(conv, nn.BatchNorm3d(num_output_channels))
        else:
            self.conv = conv

    def forward(self, inputs):
        return self.conv(inputs)


class ConvRelu(nn.Module):
    '''
    Class implementing a convolutional layer followed by batch norm and relu
    - Conv3d - [Batch normalization] - Activation
    '''
    def __init__(self, num_input_channels, num_output_channels, use_batchnorm=True, activation='ReLU', kernel_size=3, stride=1, padding=1):
        super(ConvRelu, self).__init__()

        conv = nn.Conv3d(in_channels=num_input_channels, out_channels=num_output_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=not(use_batchnorm))
        nn.init.xavier_uniform_(conv.weight)
        activ_unit = nn.ELU()
        
        if use_batchnorm:
            self.conv = nn.Sequential(conv, nn.BatchNorm3d(num_output_channels), activ_unit)
        else:
            self.conv = nn.Sequential(conv, activ_unit)

    def forward(self, inputs):
        return self.conv(inputs)


class ConvolutionalBlock(nn.Module):

    def __init__(self, num_input_channels, num_output_channels, use_batchnorm=True, activation='ReLU'):
        super(ConvolutionalBlock, self).__init__()

        self.conv1 = ConvRelu(num_input_channels, num_output_channels, use_batchnorm, activation)

    def forward(self, inputs):
        outputs = self.conv1(inputs)

        return outputs


class ConvolutionalResNetBlock(nn.Module):

    def __init__(self, num_input_channels, num_output_channels, use_batchnorm=True, activation='ReLU'):

        super(ConvolutionalResNetBlock, self).__init__()

        self.conv1 = ConvResNet(num_input_channels, num_output_channels, use_batchnorm, kernel_size=3, stride=2)
        self.conv2 = ConvResNet(num_output_channels, num_output_channels, use_batchnorm)
        self.activ_unit = getattr(nn, activation)()

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.activ_unit(outputs)
        outputs = self.conv2(outputs)
        outputs = self.activ_unit(outputs)
        
        return outputs


class DeConvolutionalResNetBlock(nn.Module):

    def __init__(self, num_input_channels, num_output_channels, use_batchnorm=True, activation='ReLU',kernel_size=3,stride=3, padding=1):

        super(DeConvolutionalResNetBlock, self).__init__()

        self.conv1 = DeConvResNet(num_input_channels, num_output_channels, use_batchnorm, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = ConvResNet(num_output_channels, num_output_channels, use_batchnorm, kernel_size=3, stride=1)
        self.activ_unit = getattr(nn, activation)()

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.activ_unit(outputs)
        outputs = self.conv2(outputs)
        outputs = self.activ_unit(outputs)

        return outputs

class BottleConvolutionalResNetBlock(nn.Module):

    def __init__(self, num_input_channels, num_output_channels, use_batchnorm=True, activation='ReLU'):
        super(BottleConvolutionalResNetBlock, self).__init__()

        self.conv1 = ConvResNet(num_input_channels, num_output_channels, use_batchnorm, kernel_size=3, stride=2)
        self.conv2 = ConvResNet(num_output_channels, num_output_channels, use_batchnorm)
        self.activ_unit = getattr(nn, activation)()

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.activ_unit(outputs)
        outputs = self.conv2(outputs)
        outputs = self.activ_unit(outputs)
        
        return outputs