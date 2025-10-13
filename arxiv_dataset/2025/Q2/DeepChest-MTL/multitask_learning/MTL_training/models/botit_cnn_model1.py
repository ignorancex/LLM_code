import torch
import torch.nn as nn
import torch.nn.functional as F


class BotitCNNModel1(nn.Module):
    def __init__(self, img_size, df):
        super(BotitCNNModel1, self).__init__()
        self.img_size = img_size  # Store img_size for use in the model
        self.df = df
        
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Calculate the size after convolutions and pooling
        # (img_size // 2^3) because we pool 3 times (each pooling halves the size)
        self.fc1 = nn.Linear(128 * (self.img_size // 8) * (self.img_size // 8), 512)

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
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))

        # Output branches
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
