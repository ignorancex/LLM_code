import torch.nn as nn
import torch.nn.functional as F
__all__ = ['domainnet_res']
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class domainnet_res(nn.Module):
    def __init__(self, num_classes=345):
        super(domainnet_res, self).__init__()

        # 将卷积层和残差块放在 features 中，并加上 Flatten 层
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            BasicBlock(64, 128, stride=2),  # ResBlock 1
            BasicBlock(128, 256, stride=2),  # ResBlock 2
            BasicBlock(256, 512, stride=2),  # ResBlock 3
            nn.AdaptiveAvgPool2d((1, 1)),    # 全局平均池化
            nn.Flatten()                     # 扁平化特征
        )

        # 将最后的分类器单独放在 classifier 中
        self.classifier = nn.Linear(512, num_classes)

        # 初始化权重
        self._initialize_weights()

    def forward(self, x, return_features=False):
        # 通过 features 提取特征
        features = self.features(x)

        # 如果只需要返回特征
        if return_features:
            return features

        # 否则，进入分类器，输出分类结果
        out = self.classifier(features)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)