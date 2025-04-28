import numpy as np
import torch
import torch.nn as nn


class SimpleVideoClassifier(nn.Module):

    def __init__(self, input_size, n_frames_per_clip, n_classes=2):
        """
        Simple Video classifier to classify into two classes:

        Args:
            input_size:  shape of the input image. Should be a 2 element vector for a 2D video (width, height).
            n_output_classes: number of output classes
        """

        super(SimpleVideoClassifier, self).__init__()
        self.name = 'SimpleVideoClassifier'
        self.input_size = input_size
        self.n_classes = n_classes
        self.n_frames_per_clip = n_frames_per_clip
        self.n_features = np.prod(self.input_size) * self.n_frames_per_clip
        # n_temporal_features = n_spatial_features_out * n_frames

        # extract features from each frame
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.n_features, out_features=256),
            # nn.Linear(in_features=n_temporal_features, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=n_classes),
            # nn.Sigmoid(),
        )

    def get_name(self):
        return self.name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f'x.shape(): {x.size()}') ##[batch_size, channels, depth, height, width]
        # x = x.permute(0,2,1,3,4)##[batch_size, depth,channels, height, width]
        x = self.classifier(x)

        ##\/ TOREVIEW
        #     n_frames = data.shape[2]
        #     features = []
        #     for f in range(n_frames):
        #         feat_i = self.frame_features(data[:, :, f, ...])
        #         features.append(feat_i)
        #
        #     feature_vector = torch.stack(features, dim=1)
        #
        #     x = self.classifier(feature_vector)
        #     return x
        ##/\ TOREVIEW

        return x
