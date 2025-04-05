'''
If you want to test the validity of pip installation, please move this file outside of the git project directory, 
otherwise it is testing the files inside the project instead of pip installation.
'''


import torch


'''
test the operators
'''
from acsconv.operators import ACSConv, SoftACSConv
B, C_in, D, H, W = (1, 3, 64, 64, 64)
x = torch.rand(B, C_in, D, H, W)
# ACSConv to process 3D volumnes
conv = ACSConv(in_channels=3, out_channels=10, kernel_size=3, padding=1)
out = conv(x)
# SoftACSConv to process 3D volumnes
conv = SoftACSConv(in_channels=3, out_channels=10, kernel_size=3, padding=1)
out = conv(x)

'''
test the operators (depth-wise)
'''
# Depth-wise ACSConv to process 3D volumnes
conv = ACSConv(in_channels=3, out_channels=3, kernel_size=3, padding=1, groups=3)
out = conv(x)

'''
test the converters module
'''
from torchvision.models import resnet18
from acsconv.converters import ACSConverter
# model_2d is a standard pytorch 2D model
model_2d = resnet18(pretrained=True)
B, C_in, H, W = (1, 3, 64, 64)
input_2d = torch.rand(B, C_in, H, W)
output_2d = model_2d(input_2d)

model_3d = ACSConverter(model_2d)
# once converted, model_3d is using ACSConv and capable of processing 3D volumes.
B, C_in, D, H, W = (1, 3, 64, 64, 64)
input_3d = torch.rand(B, C_in, D, H, W)
output_3d = model_3d(input_3d)

'''
test the native ACS models
'''
from acsconv.models.acsunet import ACSUNet
unet_3d = ACSUNet(num_classes=3)
B, C_in, D, H, W = (1, 1, 64, 64, 64)
input_3d = torch.rand(B, C_in, D, H, W)
output_3d = unet_3d(input_3d)

print('==========================================')
print('The installation of ACSConv is successful!')
print('==========================================')