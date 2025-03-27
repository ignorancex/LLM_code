import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .init_utils import weights_init
from collections import OrderedDict


defaultcfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


class VGG(nn.Module):
    def __init__(self, dataset='cifar10', depth=19, init_weights=None, cfg=None, affine=True, batchnorm=True, actv_fn="relu", last="logsoftmax"):
        super(VGG, self).__init__()

        print("Model: VGG")

        if actv_fn == "tanh":
            actv_fn  = nn.Tanh
        elif actv_fn == "linear":
            actv_fn = nn.Identity
        elif actv_fn == "relu":
            actv_fn = nn.ReLU
        elif actv_fn == "selu":
            actv_fn = nn.SELU
        elif actv_fn == "hard_tanh":
            actv_fn = nn.Hardtanh
        else:
            raise ValueError("Unknown activation")
        self.actv_fn = actv_fn

        if cfg is None:
            cfg = defaultcfg[depth]
        self._AFFINE = affine
        self.feature = self.make_layers(cfg, batchnorm)
        self.dataset = dataset
        if dataset == 'cifar10' or dataset == 'cinic-10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        elif dataset == 'tiny_imagenet':
            num_classes = 200
        else:
            raise NotImplementedError("Unsupported dataset " + dataset)
        self.classifier = nn.Linear(cfg[-1], num_classes)
        if init_weights is not None:
            self.apply(init_weights)
        self.last = last
        # if pretrained:
        #     model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        i = 0
        for v in cfg:
            if v == 'M':
                layers += [(f'maxpool{i}', nn.MaxPool2d(kernel_size=2, stride=2))]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)

                if self.actv_fn == nn.Tanh: # Tanh doesn't accept the inplace argument
                    actv = self.actv_fn()
                else:
                    actv = self.actv_fn(inplace=True)

                if batch_norm:
                    layers += [(f'conv{i}', conv2d), (f'bn{i}', nn.BatchNorm2d(v, affine=self._AFFINE)), ('actv{i}', actv)]
                else:
                    layers += [(f'conv{i}', conv2d), ('actv{i}', actv)]
                in_channels = v
            i += 1
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.feature(x)
        if self.dataset == 'tiny_imagenet':
            x = nn.AvgPool2d(4)(x)
        else:
            x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)

        if self.last == "logsoftmax":
            return F.log_softmax(y, dim=1)
        elif self.last == "logits":
            return y
        else:
            raise ValueError("Unknown last operation")

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1.0)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()