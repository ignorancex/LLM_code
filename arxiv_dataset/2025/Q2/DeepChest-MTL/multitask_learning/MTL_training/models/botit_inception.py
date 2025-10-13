import torch
import torch.nn as nn
import torchvision.models as models

class BotitInceptionNet(nn.Module):
    def __init__(self, img_size, df):
        super(BotitInceptionNet, self).__init__()
        self.img_size = img_size  # Store img_size for use in the model
        self.df = df

        # Load the pretrained Inception v3 model
        self.inception = models.inception_v3(pretrained=True, aux_logits=False)  # Disable auxiliary logits

        # Use only the feature extractor part of Inception
        self.inception_features = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),  # Ensure image size compatibility if needed
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
            nn.AdaptiveAvgPool2d(1)  # Adaptive pooling for output consistency
        )

        # Output channels of Inception's feature extractor
        feature_dim = self.inception.fc.in_features

        # Define a shared fully connected layer
        self.fc1 = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Define output layers for each task
        self.GenericName_output = nn.Linear(512, len(set(self.df['Generic Name'])))
        self.shoppingCategory_output = nn.Linear(512, len(set(self.df['shoppingCategory'])))
        self.shoppingSubcategory_output = nn.Linear(512, len(set(self.df['shoppingSubcategory'])))
        self.itemSubcategory_output = nn.Linear(512, len(set(self.df['itemSubcategory'])))
        self.itemCategory_output = nn.Linear(512, len(set(self.df['itemCategory'])))
        self.normalized_color_output = nn.Linear(512, len(set(self.df['normalized_color'])))
        self.normalized_Generic_Name_output = nn.Linear(512, len(set(self.df['normalized_Generic_Name'])))

        # Define task names
        self.task_names = [
            'GenericName_output',
            'shoppingCategory_output',
            'shoppingSubcategory_output',
            'itemSubcategory_output',
            'itemCategory_output',
            'normalized_color_output',
            'normalized_Generic_Name_output'
        ]

    def forward(self, x):
        # Pass input through Inception's feature extractor
        x = self.inception_features(x)
        x = x.view(x.size(0), -1)  # Flatten the feature map

        # Pass through the shared fully connected layer
        x = self.fc1(x)

        # Output branches for each task
        generic_name = self.GenericName_output(x)
        shopping_category = self.shoppingCategory_output(x)
        shopping_subcategory = self.shoppingSubcategory_output(x)
        item_subcategory = self.itemSubcategory_output(x)
        item_category = self.itemCategory_output(x)
        normalized_color = self.normalized_color_output(x)
        normalized_generic_name = self.normalized_Generic_Name_output(x)

        return (
            generic_name, 
            shopping_category, 
            shopping_subcategory, 
            item_subcategory, 
            item_category, 
            normalized_color, 
            normalized_generic_name
        )
