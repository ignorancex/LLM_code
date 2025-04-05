"""
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VggMLP(nn.Module):

    def __init__(self, n_features, n_classes, n_channels=None,
                 normalize_mean=None, normalize_std=None):
        super(VggMLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(n_features[0], 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, n_classes),
        )
        self._initialize_weights()
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

    def forward(self, x):
        x = self._normalize(x)
        x = self.classifier(x)
        return x

    def _normalize(self, x):
        if (self.normalize_mean is None) or (self.normalize_std is None):
            return x
        mean = self.normalize_mean
        mean = torch.as_tensor(mean, dtype=x.dtype, device=x.device)
        if mean.ndim == 1:
            mean = mean[None, :, None, None]
        std = self.normalize_std
        std = torch.as_tensor(std, dtype=x.dtype, device=x.device)
        if std.ndim == 1:
            std = std[None, :, None, None]
        x = (x - mean) / std
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

#class VggMLPNorm01(VggMLP):
#    def __init__(self, n_features, n_classes, n_channels=None):
#        super().__init__(n_features, n_classes, n_channels,
#                         normalize_mean=[0.485, 0.456, 0.406], normalize_std=[0.229, 0.224, 0.225])
#
#class VggMLPNorm02(VggMLP):
#    def __init__(self, n_features, n_classes, n_channels=None):
#        super().__init__(n_features, n_classes, n_channels,
#                         normalize_mean=(0.4914, 0.4822, 0.4465), normalize_std=(0.2023, 0.1994, 0.2010))

#class VggMLP(nn.Module):
#
#    def __init__(self, n_features, n_classes, n_channels=None):
#        super(VggMLP, self).__init__()
#        self.fc1 = nn.Linear(n_features[0], 4096)
#        self.fc2 = nn.Linear(4096, 4096)
#        self.fc3 = nn.Linear(4096, n_classes)
#        self.dropout1 = nn.Dropout(p=0.5)
#        self.dropout2 = nn.Dropout(p=0.5)
#
#        for m in self.modules():
#            if isinstance(m, nn.Conv2d):
#                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                if m.bias is not None:
#                    nn.init.constant_(m.bias, 0)
#            elif isinstance(m, nn.BatchNorm2d):
#                nn.init.constant_(m.weight, 1)
#                nn.init.constant_(m.bias, 0)
#            elif isinstance(m, nn.Linear):
#                nn.init.normal_(m.weight, 0, 0.01)
#                nn.init.constant_(m.bias, 0)
#
#    def forward(self, x):
#        x = self.fc1(x)
#        x = F.relu(x)
#        x = self.dropout1(x)
#        x = self.fc2(x)
#        x = F.relu(x)
#        x = self.dropout1(x)
#        x = self.fc3(x)
#        return x
