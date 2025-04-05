import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from .submodule import vgg_face_dag

class VGG(nn.Module):
    def __init__(self, orig_vgg, fc_dim=512, pool_type='maxpool', conv_size=3):
        super(VGG, self).__init__()
        from functools import partial

        self.pool_type = pool_type

        self.features = nn.Sequential(
            *list(orig_vgg.children())[:-10])
        self.fc = nn.Conv2d(
            512, fc_dim, kernel_size=conv_size, padding=conv_size//2)

    def forward_multiframe(self, x, pool=True):
        (B, C, T, H, W) = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B*T, C, H, W)

        x = self.features(x)
        # print("111111111111111111", x.shape)
        # x = self.fc(x)
        # print("222222222222222222", x.shape)

        (_, C, H, W) = x.size()
        x = x.view(B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        if not pool:
            return x

        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool3d(x, (1, 4, 4))
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool3d(x, (1, 4, 4))

        x = x.view(B, C, 4, 4)
        return x

class Resnet18FC(nn.Module):
    def __init__(self, original_resnet, fc_dim=512, pool_type='maxpool', conv_size=3):
        super(Resnet18FC, self).__init__()
        self.pool_type = pool_type

        self.features = nn.Sequential(
            *list(original_resnet.children())[:-2])
        self.fc = nn.Conv2d(
            512, fc_dim, kernel_size=conv_size, padding=conv_size//2)

    def forward(self, x, pool=True):
        x = self.features(x)
        x = self.fc(x)

        if not pool:
            return x

        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool2d(x, 1)
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool2d(x, 1)

        x = x.view(x.size(0), x.size(1))
        return x

    def forward_multiframe(self, x, pool=True):
        (B, C, T, H, W) = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B*T, C, H, W)

        x = self.features(x)
        x = self.fc(x)

        (_, C, H, W) = x.size()
        x = x.view(B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        if not pool:
            return x

        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool3d(x, (1, 4, 4))
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool3d(x, (1, 4, 4))

        x = x.view(B, C, 4, 4)
        return x

class Resnet50FC(nn.Module):
    def __init__(self, original_resnet, fc_dim=512, pool_type='maxpool', conv_size=3):
        super(Resnet50FC, self).__init__()
        self.pool_type = pool_type

        self.features = nn.Sequential(
            *list(original_resnet.children())[:-2])
        self.fc = nn.Conv2d(
            2048, fc_dim, kernel_size=conv_size, padding=conv_size//2)

    def forward(self, x, pool=True):
        x = self.features(x)
        x = self.fc(x)

        if not pool:
            return x

        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool2d(x, 1)
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool2d(x, 1)

        x = x.view(x.size(0), x.size(1))
        return x

    def forward_multiframe(self, x, pool=True):
        (B, C, T, H, W) = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B*T, C, H, W)

        x = self.features(x)
        x = self.fc(x)

        (_, C, H, W) = x.size()
        x = x.view(B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        if not pool:
            return x

        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool3d(x, (1, 4, 4))
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool3d(x, (1, 4, 4))

        x = x.view(B, C, 4, 4)
        return x

class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, fc_dim=64, pool_type='maxpool', dilate_scale=16, conv_size=3):
        super(ResnetDilated, self).__init__()
        from functools import partial

        self.pool_type = pool_type

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        self.features = nn.Sequential(
            *list(orig_resnet.children())[:-2])

        self.fc = nn.Conv2d(
            512, fc_dim, kernel_size=conv_size, padding=conv_size//2)

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, pool=True):
        x = self.features(x)
        x = self.fc(x)

        if not pool:
            return x

        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool2d(x, 1)
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool2d(x, 1)

        x = x.view(x.size(0), x.size(1))
        return x

    def forward_multiframe(self, x, pool=True):
        (B, C, T, H, W) = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * T, C, H, W)

        x = self.features(x)
        x = self.fc(x)

        (_, C, H, W) = x.size()
        x = x.view(B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        if not pool:
            return x

        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool3d(x, (1, 4, 4))
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool3d(x, (1, 4, 4))

        x = x.view(B, C, 4, 4)
        return x

def test_vgg():
    fc_dim = 32
    pool_type = 'maxpool'
    original_vggface = vgg_face_dag('../../weights/vgg_face_dag.pth')
    net = VGG(original_vggface, fc_dim=fc_dim, pool_type=pool_type)
    return net

def test_resnet18fc():
    fc_dim = 32
    pool_type = 'maxpool'
    pretrained = True
    original_resnet = torchvision.models.resnet18(pretrained)
    net = Resnet18FC(original_resnet, fc_dim=fc_dim, pool_type=pool_type)
    return net

def test_resnet18dilated():
    fc_dim = 32
    pool_type = 'maxpool'
    pretrained = True
    original_resnet = torchvision.models.resnet18(pretrained)
    net = ResnetDilated(original_resnet, fc_dim=fc_dim, pool_type=pool_type)
    return net

if __name__ == '__main__':
    input = torch.randn(10, 3, 24, 224, 224)
    net = test_vgg()
    output = net.forward_multiframe(input)
    print(output.shape)

