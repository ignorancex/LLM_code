import torch
import torch.nn as nn
import torchvision.models as models

class XrayVGG16(nn.Module):
    def __init__(self, df):
        super(XrayVGG16, self).__init__()
        self.df = df
        
        # Load the pre-trained VGG16 model
        vgg16 = models.vgg16(pretrained=True)
        
        # Remove the last classification layer (fc layer)
        self.features = vgg16.features
        
        # Calculate the size after the feature extractor
        # Assuming input image size is 224x224
        flattened_size = 512 * 7 * 7  # This depends on the input size and VGG16 architecture
        
        # Define a shared fully connected layer
        self.fc1 = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
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
        # Extract features using VGG16's convolutional layers
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Shared fully connected layer
        x = self.fc1(x)
        
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
