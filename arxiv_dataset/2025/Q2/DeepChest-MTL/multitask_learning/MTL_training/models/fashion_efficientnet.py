import torch
import torch.nn as nn
import torchvision.models as models

class FashionEfficientNet(nn.Module):
    def __init__(self, img_size, df):
        super(FashionEfficientNet, self).__init__()
        self.img_size = img_size  # Store img_size for use in the model
        self.df = df

        # Load the pretrained EfficientNet model
        self.efficientnet = models.efficientnet_b0(pretrained=True)

        # Use only the feature extractor part of EfficientNet
        self.efficientnet_features = nn.Sequential(
            self.efficientnet.features,
            nn.AdaptiveAvgPool2d(1)  # Adaptive pooling for consistent output size
        )

        # Extract the feature dimension from EfficientNet's classifier
        feature_dim = self.efficientnet.classifier[1].in_features

        # Define a shared fully connected layer
        self.fc1 = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Define output layers for each task
        self.gender_output = nn.Linear(512, len(set(self.df['gender'])))
        self.masterCategory_output = nn.Linear(512, len(set(self.df['masterCategory'])))
        self.subCategory_output = nn.Linear(512, len(set(self.df['subCategory'])))
        self.articleType_output = nn.Linear(512, len(set(self.df['articleType'])))
        self.baseColour_output = nn.Linear(512, len(set(self.df['baseColour'])))
        self.season_output = nn.Linear(512, len(set(self.df['season'])))
        self.usage_output = nn.Linear(512, len(set(self.df['usage'])))

        # Define task names
        self.task_names = [
            'gender_output',
            'masterCategory_output',
            'subCategory_output',
            'articleType_output',
            'baseColour_output',
            'season_output',
            'usage_output'
        ]

    def forward(self, x):
        # Extract features using EfficientNet
        x = self.efficientnet_features(x)
        x = x.view(x.size(0), -1)  # Flatten

        # Pass through the shared fully connected layer
        x = self.fc1(x)

        # Output branches
        gender = self.gender_output(x)
        master_category = self.masterCategory_output(x)
        sub_category = self.subCategory_output(x)
        article_type = self.articleType_output(x)
        base_colour = self.baseColour_output(x)
        season = self.season_output(x)
        usage = self.usage_output(x)

        return gender, master_category, sub_category, article_type, base_colour, season, usage
