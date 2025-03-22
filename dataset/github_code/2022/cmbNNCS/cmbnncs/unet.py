import torch.nn as nn
from . import sequence as seq
import torch


class UNet5(nn.Module):
    def __init__(self, channels,channel_in=1,channel_out=1,kernels_size=None,strides=None,
                 extra_pads=None,mainActive='prelu',finalActive='prelu',
                 mainBN=True,finalBN=True,eps=1e-05,momentum=0.1,sides=512):
        super(UNet5, self).__init__()
        if kernels_size is None:
            kernels_size = [3 for i in range(len(channels)-2)]
        if strides is None:
            strides = [2 for i in range(len(kernels_size))]
        if extra_pads is None:
            extra_pads = [0 for i in range(len(strides))]
        conv1 = seq.Conv2dSeq([channel_in, channels[0]],kernels_size=[kernels_size[0]],strides=[strides[0]],extra_pads=[extra_pads[0]],finalActive=mainActive,finalBN=mainBN,eps=eps,momentum=momentum,in_side=sides)
        self.conv1 = conv1.get_seq()
        conv2 = seq.Conv2dSeq([channels[0], channels[1]],kernels_size=[kernels_size[1]],strides=[strides[1]],extra_pads=[extra_pads[1]],finalActive=mainActive,finalBN=mainBN,eps=eps,momentum=momentum,in_side=conv1.sides[-1])
        self.conv2 = conv2.get_seq()
        conv3 = seq.Conv2dSeq([channels[1], channels[2]],kernels_size=[kernels_size[2]],strides=[strides[2]],extra_pads=[extra_pads[2]],finalActive=mainActive,finalBN=mainBN,eps=eps,momentum=momentum,in_side=conv2.sides[-1])
        self.conv3 = conv3.get_seq()
        conv4 = seq.Conv2dSeq([channels[2], channels[3]],kernels_size=[kernels_size[3]],strides=[strides[3]],extra_pads=[extra_pads[3]],finalActive=mainActive,finalBN=mainBN,eps=eps,momentum=momentum,in_side=conv3.sides[-1])
        self.conv4 = conv4.get_seq()
        conv5 = seq.Conv2dSeq([channels[3], channels[4]],kernels_size=[kernels_size[4]],strides=[strides[4]],extra_pads=[extra_pads[4]],finalActive=mainActive,finalBN=mainBN,eps=eps,momentum=momentum,in_side=conv4.sides[-1])
        self.conv5 = conv5.get_seq()
        
        deconv1 = seq.Conv2dSeq([channels[4], channels[3]],kernels_size=[kernels_size[5]],strides=[strides[5]],extra_pads=[extra_pads[5]],finalActive=mainActive,finalBN=mainBN,eps=eps,momentum=momentum,transConv2d=True,in_side=conv5.sides[-1])
        self.deconv1 = deconv1.get_seq()
        deconv2 = seq.Conv2dSeq([channels[3]*2, channels[2]],kernels_size=[kernels_size[6]],strides=[strides[6]],extra_pads=[extra_pads[6]],finalActive=mainActive,finalBN=mainBN,eps=eps,momentum=momentum,transConv2d=True,in_side=deconv1.sides[-1])
        self.deconv2 = deconv2.get_seq()
        deconv3 = seq.Conv2dSeq([channels[2]*2, channels[1]],kernels_size=[kernels_size[7]],strides=[strides[7]],extra_pads=[extra_pads[7]],finalActive=mainActive,finalBN=mainBN,eps=eps,momentum=momentum,transConv2d=True,in_side=deconv2.sides[-1])
        self.deconv3 = deconv3.get_seq()
        deconv4 = seq.Conv2dSeq([channels[1]*2, channels[0]],kernels_size=[kernels_size[8]],strides=[strides[8]],extra_pads=[extra_pads[8]],finalActive=mainActive,finalBN=mainBN,eps=eps,momentum=momentum,transConv2d=True,in_side=deconv3.sides[-1])
        self.deconv4 = deconv4.get_seq()
        deconv5 = seq.Conv2dSeq([channels[0]*2, channel_out],kernels_size=[kernels_size[9]],strides=[strides[9]],extra_pads=[extra_pads[9]],finalActive=finalActive,finalBN=finalBN,eps=eps,momentum=momentum,transConv2d=True,in_side=deconv4.sides[-1])
        self.deconv5 = deconv5.get_seq()
        
    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x_1)
        x_3 = self.conv3(x_2)
        x_4 = self.conv4(x_3)
        x_5 = self.conv5(x_4)

        x_6 = self.deconv1(x_5)
        x_7 = self.deconv2(torch.cat((x_4, x_6),1))
        x_8 = self.deconv3(torch.cat((x_3, x_7),1))
        x_9 = self.deconv4(torch.cat((x_2, x_8),1))
        maps = self.deconv5(torch.cat((x_1, x_9),1))
        return maps


class UNet8(nn.Module):
    def __init__(self, channels,channel_in=1,channel_out=1,kernels_size=None,strides=None,
                 extra_pads=None,mainActive='prelu',finalActive='prelu',
                 mainBN=True,finalBN=True,eps=1e-05,momentum=0.1,sides=512):
        super(UNet8, self).__init__()
        if kernels_size is None:
            kernels_size = [3 for i in range(len(channels)*2-2)]
        if strides is None:
            strides = [2 for i in range(len(kernels_size))]
        if extra_pads is None:
            extra_pads = [0 for i in range(len(strides))]
        conv1 = seq.Conv2dSeq([channel_in, channels[0]],kernels_size=[kernels_size[0]],strides=[strides[0]],extra_pads=[extra_pads[0]],finalActive=mainActive,finalBN=mainBN,eps=eps,momentum=momentum,in_side=sides)
        self.conv1 = conv1.get_seq()
        conv2 = seq.Conv2dSeq([channels[0], channels[1]],kernels_size=[kernels_size[1]],strides=[strides[1]],extra_pads=[extra_pads[1]],finalActive=mainActive,finalBN=mainBN,eps=eps,momentum=momentum,in_side=conv1.sides[-1])
        self.conv2 = conv2.get_seq()
        conv3 = seq.Conv2dSeq([channels[1], channels[2]],kernels_size=[kernels_size[2]],strides=[strides[2]],extra_pads=[extra_pads[2]],finalActive=mainActive,finalBN=mainBN,eps=eps,momentum=momentum,in_side=conv2.sides[-1])
        self.conv3 = conv3.get_seq()
        conv4 = seq.Conv2dSeq([channels[2], channels[3]],kernels_size=[kernels_size[3]],strides=[strides[3]],extra_pads=[extra_pads[3]],finalActive=mainActive,finalBN=mainBN,eps=eps,momentum=momentum,in_side=conv3.sides[-1])
        self.conv4 = conv4.get_seq()
        conv5 = seq.Conv2dSeq([channels[3], channels[4]],kernels_size=[kernels_size[4]],strides=[strides[4]],extra_pads=[extra_pads[4]],finalActive=mainActive,finalBN=mainBN,eps=eps,momentum=momentum,in_side=conv4.sides[-1])
        self.conv5 = conv5.get_seq()
        conv6 = seq.Conv2dSeq([channels[4], channels[5]],kernels_size=[kernels_size[5]],strides=[strides[5]],extra_pads=[extra_pads[5]],finalActive=mainActive,finalBN=mainBN,eps=eps,momentum=momentum,in_side=conv5.sides[-1])
        self.conv6 = conv6.get_seq()
        conv7 = seq.Conv2dSeq([channels[5], channels[6]],kernels_size=[kernels_size[6]],strides=[strides[6]],extra_pads=[extra_pads[6]],finalActive=mainActive,finalBN=mainBN,eps=eps,momentum=momentum,in_side=conv6.sides[-1])
        self.conv7 = conv7.get_seq()
        conv8 = seq.Conv2dSeq([channels[6], channels[7]],kernels_size=[kernels_size[7]],strides=[strides[7]],extra_pads=[extra_pads[7]],finalActive=mainActive,finalBN=mainBN,eps=eps,momentum=momentum,in_side=conv7.sides[-1])
        self.conv8 = conv8.get_seq()
        
        deconv1 = seq.Conv2dSeq([channels[7], channels[6]],kernels_size=[kernels_size[8]],strides=[strides[8]],extra_pads=[extra_pads[8]],finalActive=mainActive,finalBN=mainBN,eps=eps,momentum=momentum,transConv2d=True,in_side=conv8.sides[-1])
        self.deconv1 = deconv1.get_seq()
        deconv2 = seq.Conv2dSeq([channels[6]*2, channels[5]],kernels_size=[kernels_size[9]],strides=[strides[9]],extra_pads=[extra_pads[9]],finalActive=mainActive,finalBN=mainBN,eps=eps,momentum=momentum,transConv2d=True,in_side=deconv1.sides[-1])
        self.deconv2 = deconv2.get_seq()
        deconv3 = seq.Conv2dSeq([channels[5]*2, channels[4]],kernels_size=[kernels_size[10]],strides=[strides[10]],extra_pads=[extra_pads[10]],finalActive=mainActive,finalBN=mainBN,eps=eps,momentum=momentum,transConv2d=True,in_side=deconv2.sides[-1])
        self.deconv3 = deconv3.get_seq()
        deconv4 = seq.Conv2dSeq([channels[4]*2, channels[3]],kernels_size=[kernels_size[11]],strides=[strides[11]],extra_pads=[extra_pads[11]],finalActive=mainActive,finalBN=mainBN,eps=eps,momentum=momentum,transConv2d=True,in_side=deconv3.sides[-1])
        self.deconv4 = deconv4.get_seq()
        deconv5 = seq.Conv2dSeq([channels[3]*2, channels[2]],kernels_size=[kernels_size[12]],strides=[strides[12]],extra_pads=[extra_pads[12]],finalActive=mainActive,finalBN=mainBN,eps=eps,momentum=momentum,transConv2d=True,in_side=deconv4.sides[-1])
        self.deconv5 = deconv5.get_seq()
        deconv6 = seq.Conv2dSeq([channels[2]*2, channels[1]],kernels_size=[kernels_size[13]],strides=[strides[13]],extra_pads=[extra_pads[13]],finalActive=mainActive,finalBN=mainBN,eps=eps,momentum=momentum,transConv2d=True,in_side=deconv5.sides[-1])
        self.deconv6 = deconv6.get_seq()
        deconv7 = seq.Conv2dSeq([channels[1]*2, channels[0]],kernels_size=[kernels_size[14]],strides=[strides[14]],extra_pads=[extra_pads[14]],finalActive=mainActive,finalBN=mainBN,eps=eps,momentum=momentum,transConv2d=True,in_side=deconv6.sides[-1])
        self.deconv7 = deconv7.get_seq()
        deconv8 = seq.Conv2dSeq([channels[0]*2, channel_out],kernels_size=[kernels_size[15]],strides=[strides[15]],extra_pads=[extra_pads[15]],finalActive=finalActive,finalBN=finalBN,eps=eps,momentum=momentum,transConv2d=True,in_side=deconv7.sides[-1])
        self.deconv8 = deconv8.get_seq()
        
    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x_1)
        x_3 = self.conv3(x_2)
        x_4 = self.conv4(x_3)
        x_5 = self.conv5(x_4)
        x_6 = self.conv6(x_5)
        x_7 = self.conv7(x_6)
        x_8 = self.conv8(x_7)
        
        x_9 = self.deconv1(x_8)
        x_10 = self.deconv2(torch.cat((x_7, x_9),1))
        x_11 = self.deconv3(torch.cat((x_6, x_10),1))
        x_12 = self.deconv4(torch.cat((x_5, x_11),1))
        x_13 = self.deconv5(torch.cat((x_4, x_12),1))
        x_14 = self.deconv6(torch.cat((x_3, x_13),1))
        x_15 = self.deconv7(torch.cat((x_2, x_14),1))
        maps = self.deconv8(torch.cat((x_1, x_15),1))
        return maps

