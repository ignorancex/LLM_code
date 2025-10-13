import torch
import torch.nn as nn
import torch.nn.functional as F


class XrayCNNModel1(nn.Module):
    def __init__(self, img_size, df):
        super(XrayCNNModel1, self).__init__()
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
        self.Atelectasis_output = nn.Linear(512, len(set(self.df['Atelectasis'])))
        self.Cardiomegaly_output = nn.Linear(512, len(set(self.df['Cardiomegaly'])))
        self.Consolidation_output = nn.Linear(512, len(set(self.df['Consolidation'])))
        self.Edema_output = nn.Linear(512, len(set(self.df['Edema'])))
        self.Effusion_output = nn.Linear(512, len(set(self.df['Effusion'])))
        self.Emphysema_output = nn.Linear(512, len(set(self.df['Emphysema'])))
        self.Fibrosis_output = nn.Linear(512, len(set(self.df['Fibrosis'])))
        self.Hernia_output = nn.Linear(512, len(set(self.df['Hernia'])))
        self.Infiltration_output = nn.Linear(512, len(set(self.df['Infiltration'])))
        self.Mass_output = nn.Linear(512, len(set(self.df['Mass'])))
        self.Nodule_output = nn.Linear(512, len(set(self.df['Nodule'])))
        self.Pleural_Thickening_output = nn.Linear(512, len(set(self.df['Pleural_Thickening'])))
        self.Pneumonia_output = nn.Linear(512, len(set(self.df['Pneumonia'])))
        self.Pneumothorax_output = nn.Linear(512, len(set(self.df['Pneumothorax'])))

        # Define task names
        self.task_names = [
            'Atelectasis_output',
            'Cardiomegaly_output',
            'Consolidation_output',
            'Edema_output',
            'Effusion_output',
            'Emphysema_output',
            'Fibrosis_output',
            'Hernia_output',
            'Infiltration_output',
            'Mass_output',
            'Nodule_output',
            'Pleural_Thickening_output',
            'Pneumonia_output',
            'Pneumothorax_output',
        ]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))

        # Output branches
        Atelectasis = self.Atelectasis_output(x)
        Cardiomegaly = self.Cardiomegaly_output(x)
        Consolidation = self.Consolidation_output(x)
        Edema = self.Edema_output(x)
        Effusion = self.Effusion_output(x)
        Emphysema = self.Emphysema_output(x)
        Fibrosis = self.Fibrosis_output(x)
        Hernia = self.Hernia_output(x)
        Infiltration = self.Infiltration_output(x)
        Mass = self.Mass_output(x)
        Nodule = self.Nodule_output(x)
        Pleural_Thickening = self.Pleural_Thickening_output(x)
        Pneumonia = self.Pneumonia_output(x)
        Pneumothorax = self.Pneumothorax_output(x)

        return (
            Atelectasis, 
            Cardiomegaly, 
            Consolidation, 
            Edema, 
            Effusion, 
            Emphysema, 
            Fibrosis,
            Hernia,
            Infiltration,
            Mass,
            Nodule,
            Pleural_Thickening,
            Pneumonia,
            Pneumothorax
        )
