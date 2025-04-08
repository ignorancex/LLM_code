
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# SUPPORTED_POOLING_LAYERS = ['max', 'avg']

# def get_pooling_layer(pooling_name, kernel_size):

#     if pooling_name in SUPPORTED_POOLING_LAYERS:

#         if pooling_name == 'max':
#             return nn.MaxPool3d(kernel_size=kernel_size)

#         elif pooling_name == 'avg':
#             return nn.AvgPool3d(kernel_size=kernel_size)
    
#     else:
#         raise ValueError('Pooling layer {} unknown'.format(pooling_name))



# class ConvRelu(nn.Module):
#     '''
#     Class implementing a convolutional layer followed by batch norm and relu
#     - Conv3d - [Batch normalization] - Activation
#     '''

#     def __init__(self, num_input_channels, num_output_channels, use_batchnorm=True, activation='LeakyReLU', kernel_size=3, stride=1, padding=1):

#         super(ConvRelu, self).__init__()

#         conv = nn.Conv3d(in_channels=num_input_channels, out_channels=num_output_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=not(use_batchnorm))
#         activ_unit = getattr(nn, activation)()
#         if use_batchnorm:
#             self.conv = nn.Sequential(conv, nn.BatchNorm3d(num_output_channels), activ_unit)
#         else:
#             self.conv = nn.Sequential(conv, activ_unit)


#     def forward(self, inputs):

#         return self.conv(inputs)




# class ConvolutionalBlock(nn.Module):

#     def __init__(self, num_input_channels, num_output_channels, use_batchnorm=True, activation='relu'):

#         super(ConvolutionalBlock, self).__init__()

#         self.conv1 = ConvRelu(num_input_channels, num_output_channels, use_batchnorm, activation)
#         self.conv2 = ConvRelu(num_output_channels, num_output_channels, use_batchnorm, activation)




#     def forward(self, inputs):

#         outputs = self.conv1(inputs)
#         outputs = self.conv2(outputs)

#         return outputs



# class DeConvolutionalBlock(nn.Module):

#     def __init__(self, num_input_channels, num_output_channels, use_batchnorm=True, activation='relu'):

#         super(DeConvolutionalBlock, self).__init__()

#         self.conv1 = up_conv(num_input_channels, num_output_channels)

#         self.conv2 = up_conv(num_output_channels, num_output_channels)



#     def forward(self, inputs):
#         outputs = self.conv1(inputs)
#         outputs = self.conv2(outputs)

#         return outputs

# class BottleConvolutionalBlock(nn.Module):


#     def __init__(self, num_input_channels, num_output_channels, use_batchnorm=True, activation='relu'):

#         super(BottleConvolutionalBlock, self).__init__()

#         self.conv1 = ConvRelu(num_input_channels, num_output_channels, use_batchnorm, activation)

#         self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
#         self.fc1 = nn.Linear(16, 16, bias=False)    #nn.Linear(128, 128, bias=False) 



#     def forward(self, inputs):

#         outputs = self.conv1(inputs)
#         # outputs = self.conv2(outputs)
#         outputs = self.avgpool(outputs)
#         outputs = torch.flatten(outputs,1)
#         outputs = self.fc1(outputs)
#         outputs = outputs.view(outputs.size(0), 2, 2, 2, 2)  #outputs.view(outputs.size(0), 2, 4, 4, 4)  

#         return outputs

# class up_conv(nn.Module):
#     "Reduce the number of features by 2 using Conv with kernel size 1x1x1 and double the spatial dimension using 3D trilinear upsampling"
#     def __init__(self, ch_in, ch_out, k_size=1, scale=2, align_corners=False):
#         super(up_conv, self).__init__()
#         self.up = nn.Sequential(
#             nn.Conv3d(ch_in, ch_out, kernel_size=k_size),
#             nn.Upsample(scale_factor=scale, mode='trilinear', align_corners=align_corners), #Comment here??
#         )
#     def forward(self, x):
# 	    return self.up(x)
