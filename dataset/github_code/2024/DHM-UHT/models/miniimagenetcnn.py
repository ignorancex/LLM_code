import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['miniimagenetcnn']
class miniimagenetcnn(nn.Module):
    def __init__(self, num_classes=100):
        super(miniimagenetcnn, self).__init__()

        # 定义卷积层、BatchNorm 和 ReLU 的 features 模块
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),  # 添加 BatchNorm
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出尺寸: (64, 42, 42)

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),  # 添加 BatchNorm
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出尺寸: (128, 21, 21)

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),  # 添加 BatchNorm
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出尺寸: (256, 10, 10)

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),  # 添加 BatchNorm
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出尺寸: (512, 5, 5)

            nn.Flatten()
        )

        # 定义全连接层
        self.classifier = nn.Sequential(
            nn.Linear(512 * 5 * 5, num_classes)
        )

        # 初始化权重
        self._initialize_weights()

    def forward(self, x):
        # 前向传播：经过卷积层（features 模块）
        x = self.features(x)


        # 经过全连接层输出分类结果
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):  # 初始化 BatchNorm
                nn.init.constant_(m.weight, 1)  # scale 参数初始化为1
                nn.init.constant_(m.bias, 0)  # shift 参数初始化为0

