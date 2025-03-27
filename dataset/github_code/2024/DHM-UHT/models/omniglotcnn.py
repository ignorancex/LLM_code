import torch.nn as nn
import math

__all__ = ['omniglotcnn']


class omniglotcnn(nn.Module):
    def __init__(self, num_classes=1623):
        super(omniglotcnn, self).__init__()

        # Define the feature extractor part (convolutional layers)
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=3, padding=2),  # (batch, 64, 14, 14)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=3, padding=2),  # (batch, 128, 7, 7)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=3, padding=2),  # (batch, 256, 4, 4)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=2, padding=0),  # (batch, 512, 2, 2)
            nn.ReLU(),
            nn.Flatten()
        )

        # Define the classifier part (fully connected layers with sigmoid)
        self.classifier = nn.Sequential(
            nn.Linear(64, num_classes),  # Only one output for binary classification
        )
        self._initialize_weights()

    def forward(self, x):
        # Pass through the feature extractor
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor

        # Pass through the classifier
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)
