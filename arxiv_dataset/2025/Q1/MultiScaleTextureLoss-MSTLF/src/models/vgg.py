import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models
import torch


def xavier_init(tensor):
    init.xavier_uniform_(tensor)


def gaussian_init(tensor):
    init.normal_(tensor, mean=0.0, std=0.02)


def uniform_init(tensor):
    init.uniform_(tensor, a=-0.1, b=0.1)

class VGG(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True, init_type='pretrained',
                 saved_weights_path=None):
        super(VGG, self).__init__()

        self.features = models.vgg16(pretrained=pretrained).features

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        # Initialize the weights based on the selected initialization type
        if init_type == "pretrained":
            self.reset_classifier()
            return
        elif init_type == 'xavier':
            self.apply_init(xavier_init)
            self.reset_classifier()
        elif init_type == 'gaussian':
            self.apply_init(gaussian_init)
            self.reset_classifier()
        elif init_type == 'uniform':
            self.apply_init(uniform_init)
            self.reset_classifier()
        elif init_type == 'finetuned':
            if saved_weights_path is not None:
                self.load_saved_weights(saved_weights_path)
            else:
                raise ValueError('No path given.')

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def load_pretrained_weights(self):
        vgg16 = models.vgg16(pretrained=True)
        self.features.load_state_dict(vgg16.features.state_dict())

    def load_saved_weights(self, saved_weights_path):
        saved_weights = torch.load(saved_weights_path, map_location='cuda:1')
        self.load_state_dict(saved_weights["state_dict"])

    def apply_init(self, init_func):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def reset_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

