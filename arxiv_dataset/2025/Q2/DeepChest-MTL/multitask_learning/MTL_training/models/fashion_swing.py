import torch
import torch.nn as nn
from torchvision.models import swin_t

class FashionSwinTransformer(nn.Module):
    def __init__(self, img_size, df):
        super(FashionSwinTransformer, self).__init__()
        self.img_size = img_size  # Store img_size for use in the model
        self.df = df

        # Load the pretrained Swin Transformer model
        self.swin = swin_t(weights="IMAGENET1K_V1")

        # Extract the feature dimension from the Swin Transformer
        feature_dim = self.swin.head.in_features

        # Replace the classification head with an identity layer for feature extraction
        self.swin.head = nn.Identity()

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
        # Pass input through the Swin Transformer
        x = self.swin(x)

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
