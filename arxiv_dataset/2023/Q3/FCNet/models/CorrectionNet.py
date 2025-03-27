import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    """ implement conv+ReLU two times """
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        conv_relu = []
        conv_relu.append(nn.Conv2d(in_channels=in_channels, out_channels=middle_channels,
                                   kernel_size=3, padding=1, stride=1))
        conv_relu.append(nn.LeakyReLU())
        conv_relu.append(nn.Conv2d(in_channels=middle_channels, out_channels=out_channels,
                                   kernel_size=3, padding=1, stride=1))
        conv_relu.append(nn.LeakyReLU())
        self.conv_ReLU = nn.Sequential(*conv_relu)
    def forward(self, x):
        out = self.conv_ReLU(x)
        return out

class deconv_Block(nn.Module):
    """ implement conv+ReLU two times """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                                   kernel_size=3, padding=1, stride=1)


    def forward(self, x,y):
        x = F.interpolate(x, (y.size()[2], y.size()[3]),mode='bilinear')
        out = self.conv(x)
        return out

#四层U_Net
class U_Net_4(nn.Module):
    def __init__(self,initialchanel):
        super().__init__()

        # 首先定义左半部分网络
        # left_conv_1 表示连续的两个（卷积+激活）
        # 随后进行最大池化
        self.ReLU = nn.LeakyReLU()
        self.left_conv_1 = Block(in_channels=3, middle_channels=initialchanel, out_channels=initialchanel)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_2 = Block(in_channels=initialchanel, middle_channels=2*initialchanel, out_channels=2*initialchanel)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_3 = Block(in_channels=2*initialchanel, middle_channels=4*initialchanel, out_channels=4*initialchanel)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_4 = Block(in_channels=4*initialchanel, middle_channels=8*initialchanel, out_channels=8*initialchanel)

        # 定义右半部分网络
        self.deconv_1 = deconv_Block(in_channels=8*initialchanel, out_channels=4*initialchanel)
        self.right_conv_1 = Block(in_channels=8*initialchanel, middle_channels=4*initialchanel, out_channels=4*initialchanel)

        self.deconv_2 = deconv_Block(in_channels=4*initialchanel, out_channels=2*initialchanel)
        self.right_conv_2 = Block(in_channels=4*initialchanel, middle_channels=2*initialchanel, out_channels=2*initialchanel)

        self.deconv_3 = deconv_Block(in_channels=2*initialchanel, out_channels=initialchanel)
        self.right_conv_3 = Block(in_channels=2*initialchanel, middle_channels=initialchanel, out_channels=initialchanel)
        # 最后是1x1的卷积，用于将通道数化为3
        self.right_conv_4 = nn.Conv2d(in_channels=initialchanel, out_channels=6, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        # 1：进行编码过程
        feature_1 = self.left_conv_1(x)
        #print("feature_1",feature_1.shape)
        feature_1_pool = self.pool_1(feature_1)
        #print("feature_1_pool",feature_1_pool.shape)

        feature_2 = self.left_conv_2(feature_1_pool)
        #print("feature_2",feature_2.shape)
        feature_2_pool = self.pool_2(feature_2)
        #print("feature_2_pool",feature_2_pool.shape)

        feature_3 = self.left_conv_3(feature_2_pool)
        #print("feature_3",feature_3.shape)
        feature_3_pool = self.pool_3(feature_3)
        #print("feature_3_pool",feature_3_pool.shape)

        feature_4 = self.left_conv_4(feature_3_pool)
        #print("feature_4",feature_4.shape)

        # 2：进行解码过程
        de_feature_1 = self.deconv_1(feature_4,feature_3)
        #print("de_feature_1",de_feature_1.shape)
        de_feature_1 = self.ReLU(de_feature_1)
        #print("de_feature_1",de_feature_1.shape)

        # 特征拼接
        temp = torch.cat((feature_3, de_feature_1), dim=1)
        de_feature_1_conv = self.right_conv_1(temp)

        de_feature_2 = self.deconv_2(de_feature_1_conv,feature_2)
        de_feature_2 = self.ReLU(de_feature_2)
        temp = torch.cat((feature_2, de_feature_2), dim=1)
        de_feature_2_conv = self.right_conv_2(temp)

        de_feature_3 = self.deconv_3(de_feature_2_conv,feature_1)
        de_feature_3 = self.ReLU(de_feature_3)
        temp = torch.cat((feature_1, de_feature_3), dim=1)
        de_feature_3_conv = self.right_conv_3(temp)

        out = self.right_conv_4(de_feature_3_conv )

        image = out[:,0:3,:,:]
        w_1 = out[:,3:6,:,:]

        map_1 = F.softmax(w_1, dim=0)
        out = torch.sum(map_1 * image, dim=0, keepdim=True).clamp(0, 1)

        return out

#三层U_Net
class U_Net_3(nn.Module):
    def __init__(self,initialchanel):
        super().__init__()
        self.ReLU = nn.LeakyReLU()

        # 首先定义左半部分网络
        # left_conv_1 表示连续的两个（卷积+激活）
        # 随后进行最大池化
        self.left_conv_1 = Block(in_channels=3, middle_channels=initialchanel, out_channels=initialchanel)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_2 = Block(in_channels=initialchanel, middle_channels=2*initialchanel, out_channels=2*initialchanel)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_3 = Block(in_channels=2*initialchanel, middle_channels=4*initialchanel, out_channels=4*initialchanel)

        # 定义右半部分网络
        self.deconv_1 = deconv_Block(in_channels=4*initialchanel, out_channels=2*initialchanel)
        self.right_conv_1 = Block(in_channels=4*initialchanel, middle_channels=2*initialchanel, out_channels=2*initialchanel)

        self.deconv_2 = deconv_Block(in_channels=2*initialchanel, out_channels=initialchanel)
        self.right_conv_2 = Block(in_channels=2*initialchanel, middle_channels=initialchanel, out_channels=initialchanel)
        # 最后是1x1的卷积，用于将通道数化为3
        self.right_conv_3 = nn.Conv2d(in_channels=initialchanel, out_channels=3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        # 1：进行编码过程
        feature_1 = self.left_conv_1(x)
        feature_1_pool = self.pool_1(feature_1)

        feature_2 = self.left_conv_2(feature_1_pool)
        feature_2_pool = self.pool_2(feature_2)

        feature_3 = self.left_conv_3(feature_2_pool)

        # 2：进行解码过程
        de_feature_1 = self.deconv_1(feature_3,feature_2)
        de_feature_1 = self.ReLU(de_feature_1)
        # 特征拼接
        temp = torch.cat((feature_2, de_feature_1), dim=1)
        de_feature_1_conv = self.right_conv_1(temp)

        de_feature_2 = self.deconv_2(de_feature_1_conv,feature_1)
        de_feature_2 = self.ReLU(de_feature_2)
        temp = torch.cat((feature_1, de_feature_2), dim=1)
        de_feature_2_conv = self.right_conv_2(temp)

        out = self.right_conv_3(de_feature_2_conv)

        return out
