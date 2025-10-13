import torch
import torch.nn as nn
from torchvision.models import swin_t

class BotitSwinTransformer(nn.Module):
    def __init__(self, img_size, df):
        super(BotitSwinTransformer, self).__init__()
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
        # Pass input through the Swin Transformer
        x = self.swin(x)

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
