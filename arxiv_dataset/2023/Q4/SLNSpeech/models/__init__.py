import torch
from torch import nn
import torchvision
import torch.nn.functional as F

# from .audio_net import Unet
from .vision_net import Resnet18FC, Resnet50FC, ResnetDilated, VGG
from .criterion import BCELoss, L1Loss, L2Loss
from .submodule import vgg_face_dag
from ._resnet_3d import resnet18_3d
from .resnet_3d import generate_model
from .sign_net import Resnet3D

from .resunet_middle import resunet_middle
from .resunet_middle_conformer import resunet_middle_conformer
from .unet_middle import Unet


def activate(x, activation):
    if activation == 'sigmoid':
        return torch.sigmoid(x)
    elif activation == 'softmax':
        return F.softmax(x, dim=1)
    elif activation == 'relu':
        return F.relu(x)
    elif activation == 'tanh':
        return F.tanh(x)
    elif activation == 'no':
        return x
    else:
        raise Exception('Unkown activation!')


class ModelBuilder():
    # custom weights initialization
    def weights_init(self, m):

        classname = m.__class__.__name__
        print(classname)
        if classname == 'DoubleConv':
            pass
        elif classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.001)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.0001)
        elif classname == 'ReLU':
            pass
        elif classname.find('Sequential'):
            # print(m[0].weight.shape)
            # print(m.modules)
            # for module in m.modules:
            #     print(module)
            # print(*list(m.modules))
            pass
        else:
            pass

    def build_sound(self, arch='resunet_middle', weights=''):
        # 2D models
        if arch == 'resunet_middle':
            # net_sound = resunet()
            net_sound = resunet_middle()
        elif arch == 'conformer':
            net_sound = resunet_middle_conformer()
        elif arch == 'unet':
            net_sound = Unet()
        else:
            raise Exception('Architecture undefined!')

        # todo
        # net_sound.apply(self.weights_init)

        if len(weights) > 0:
            print('Loading weights for net_sound')
            net_sound.load_state_dict(torch.load(weights))

        return net_sound

    def build_frame(self, arch='resnet18fc', fc_dim=64, pool_type='avgpool', pretrained=True, weights=''):
        if arch == 'resnet18fc':
            original_resnet = torchvision.models.resnet18(pretrained)
            net = Resnet18FC(original_resnet, fc_dim=fc_dim, pool_type=pool_type)
        elif arch == 'resnet50fc':
            original_resnet = torchvision.models.resnet50(pretrained)
            net = Resnet50FC(original_resnet, fc_dim=fc_dim, pool_type=pool_type)
        elif arch == 'resnet18dilated':
            original_resnet = torchvision.models.resnet18(pretrained)
            net = ResnetDilated(original_resnet, fc_dim=fc_dim, pool_type=pool_type)
        elif arch == 'vggface':
            if pretrained:
                original_vggface = vgg_face_dag('/../weights/vgg_face_dag.pth')
            else:
                original_vggface = vgg_face_dag('')
            net = VGG(original_vggface, fc_dim=fc_dim, pool_type=pool_type)
        else:
            raise Exception('Architecture undefined!')

        if len(weights) > 0:
            print('Loading weights for net_frame')
            net.load_state_dict(torch.load(weights))
        return net

    def build_sign(self, arch='resnet18dilated', fc_dim=64, pool_type='maxpool', pretrained=True, weights=''):
        if arch == 'resnet18dilated':
            original_resnet = torchvision.models.resnet18(pretrained)
            net = ResnetDilated(
                original_resnet, fc_dim=fc_dim, pool_type=pool_type)
        elif arch == '3dresnet':
            if pretrained:
                original_resnet3d = generate_model(18)
                original_resnet3d.load_state_dict(torch.load("/../weights/r3d18_K_200ep.pth")['state_dict'])
                net = Resnet3D(original_resnet3d, fc_dim=fc_dim, pool_type=pool_type)
            else:
                net = resnet18_3d(num_classes=fc_dim, shortcut_type='B', sample_size=140, sample_duration=3)

        if len(weights) > 0:
            print('Loading weights for net_frame')
            net.load_state_dict(torch.load(weights))
        return net

    def build_criterion(self, arch):
        if arch == 'bce':
            net = BCELoss()
        elif arch == 'l1':
            net = L1Loss()
        elif arch == 'l2':
            net = L2Loss()
        else:
            raise Exception('Architecture undefined!')
        return net