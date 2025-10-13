import torch
import torch.nn as nn
import torchvision.models as models

class FashionVGG16(nn.Module):
    def __init__(self, img_size, df):
        super(FashionVGG16, self).__init__()
        self.img_size = img_size  # Store img_size for use in the model
        self.df = df

        # Load the pretrained VGG-16 model
        self.vgg16 = models.vgg16(pretrained=True)
        
        # Remove the original classifier from VGG-16
        self.vgg16_features = self.vgg16.features

        # Calculate the flattened size after the feature extractor
        # Assuming input image size (img_size x img_size)
        conv_output_size = img_size // 32  # VGG reduces size by factor of 32
        self.flat_size = 512 * (conv_output_size ** 2)  # 512 is the output channels of VGG

        # Define a shared fully connected layer
        self.fc1 = nn.Sequential(
            nn.Linear(self.flat_size, 512),
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
        # Extract features using VGG-16
        x = self.vgg16_features(x)
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
