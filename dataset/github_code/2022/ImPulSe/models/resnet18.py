import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torchvision.models.video import r3d_18


class ResNet3d18Backbone(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        layers = r3d_18()

        for name, layer in layers.named_children():
            if name == "stem":
                self.add_module(name, nn.Sequential(
                    nn.Conv3d(in_channels, 64, 7, 1, 3, bias=False),
                    nn.BatchNorm3d(64),
                    nn.ReLU()
                ))
            elif name == "fc":
                break
            else:
                self.add_module(name, layer)

        self.num_features = self.layer4[-1].conv2[0].out_channels
        self.memory_efficient = False

    def forward(self, x):
        if self.memory_efficient:
            features = self._forward_efficient(x)
        else:
            features = self._forward_fast(x)

        return features

    def _forward_efficient(self, x):
        x = self.stem(x)

        feature_0 = checkpoint(self.layer1, x)
        feature_1 = checkpoint(self.layer2, feature_0)
        feature_2 = checkpoint(self.layer3, feature_1)
        feature_3 = self.layer4(feature_2)
        features = [feature_0, feature_1, feature_2, feature_3]

        return features

    def _forward_fast(self, x):
        x = self.stem(x)

        feature_0 = self.layer1(x)
        feature_1 = self.layer2(feature_0)
        feature_2 = self.layer3(feature_1)
        feature_3 = self.layer4(feature_2)
        features = [feature_0, feature_1, feature_2, feature_3]

        return features


if __name__ == "__main__":
    import torch


    model = ResNet3d18Backbone(1)
    inputs = torch.rand(1, 1, 128, 128, 128)
    features = model(inputs)
    print(features[-1].size())
