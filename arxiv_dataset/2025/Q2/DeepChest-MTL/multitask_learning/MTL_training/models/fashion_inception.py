import torch
import torch.nn as nn
import torchvision.models as models

class FashionInceptionNet(nn.Module):
    def __init__(self, img_size, df):
        super(FashionInceptionNet, self).__init__()
        self.img_size = img_size  # Store img_size for use in the model
        self.df = df

        # Load the pretrained Inception v3 model
        self.inception = models.inception_v3(pretrained=True, aux_logits=False)  # Disable auxiliary logits

        # Extract the feature dimension from Inception's fully connected layer
        feature_dim = self.inception.fc.in_features

        # Replace Inception's fully connected layer with adaptive pooling for consistent feature extraction
        self.inception_features = nn.Sequential(
            self.inception.Conv2d_1a_3x3,
            self.inception.Conv2d_2a_3x3,
            self.inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.inception.Conv2d_3b_1x1,
            self.inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.inception.Mixed_5b,
            self.inception.Mixed_5c,
            self.inception.Mixed_5d,
            self.inception.Mixed_6a,
            self.inception.Mixed_6b,
            self.inception.Mixed_6c,
            self.inception.Mixed_6d,
            self.inception.Mixed_6e,
            self.inception.Mixed_7a,
            self.inception.Mixed_7b,
            self.inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(1)  # Adaptive pooling for consistent output size
        )

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
        # Extract features using Inception
        x = self.inception_features(x)
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
