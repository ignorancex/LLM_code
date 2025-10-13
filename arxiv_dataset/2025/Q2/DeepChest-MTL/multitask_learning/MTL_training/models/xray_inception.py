import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class XrayInceptionNet(nn.Module):
    def __init__(self, img_size, df):
        super(XrayInceptionNet, self).__init__()
        self.img_size = img_size
        self.df = df

        # Load the Inception v3 model from torchvision
        self.backbone = models.inception_v3(pretrained=True, aux_logits=False)

        # Extract the number of features from the last fully connected layer
        num_features = self.backbone.fc.in_features

        # Replace the classifier with an identity layer to extract features
        self.backbone.fc = nn.Identity()

        # Define the fully connected layer
        self.fc = nn.Linear(num_features, 512)

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
        # Pass input through the Inception v3 backbone
        x = self.backbone(x)

        # Pass through the fully connected layer
        x = F.relu(self.fc(x))

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