import torch
import torch.nn as nn
import torch.nn.functional as F


class OfficeHomeCNNModel1(nn.Module):
    def __init__(self, img_size, df):
        super(OfficeHomeCNNModel1, self).__init__()
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
        self.domain_output = nn.Linear(512, len(set(self.df['domain'])))
        self.category_output = nn.Linear(512, len(set(self.df['category'])))

        # Define task names
        self.task_names = [
            'domain_output',
            'category_output',
        ]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))

        # Output branches
        domain = self.domain_output(x)
        category = self.category_output(x)

        return domain, category
