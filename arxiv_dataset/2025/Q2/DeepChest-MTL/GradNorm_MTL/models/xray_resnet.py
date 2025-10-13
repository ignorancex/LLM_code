import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class XrayResNet(nn.Module):
    def __init__(self, img_size, df, resnet_version='resnet50'):
        super(XrayResNet, self).__init__()
        self.img_size = img_size
        self.df = df

        # Load the ResNet model from torchvision
        if resnet_version == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
        elif resnet_version == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)
        elif resnet_version == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
        elif resnet_version == 'resnet101':
            self.backbone = models.resnet101(pretrained=True)
        elif resnet_version == 'resnet152':
            self.backbone = models.resnet152(pretrained=True)
        else:
            raise ValueError(f"Unsupported ResNet version: {resnet_version}")

        # Extract the number of features from the fully connected layer
        num_features = self.backbone.fc.in_features

        # Replace the fully connected layer with an identity layer to extract features
        self.backbone.fc = nn.Identity()

        # Define the shared fully connected layer
        self.shared_fc = nn.Linear(num_features, 512)

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

        # Define output layers for each task
        self.task_outputs = nn.ModuleDict({
            'Atelectasis': nn.Linear(512, len(df['Atelectasis'].unique())),
            'Cardiomegaly': nn.Linear(512, len(df['Cardiomegaly'].unique())),
            'Consolidation': nn.Linear(512, len(df['Consolidation'].unique())),
            'Edema': nn.Linear(512, len(df['Edema'].unique())),
            'Effusion': nn.Linear(512, len(df['Effusion'].unique())),
            'Emphysema': nn.Linear(512, len(df['Emphysema'].unique())),
            'Fibrosis': nn.Linear(512, len(df['Fibrosis'].unique())),
            'Hernia': nn.Linear(512, len(df['Hernia'].unique())),
            'Infiltration': nn.Linear(512, len(df['Infiltration'].unique())),
            'Mass': nn.Linear(512, len(df['Mass'].unique())),
            'Nodule': nn.Linear(512, len(df['Nodule'].unique())),
            'Pleural_Thickening': nn.Linear(512, len(df['Pleural_Thickening'].unique())),
            'Pneumonia': nn.Linear(512, len(df['Pneumonia'].unique())),
            'Pneumothorax': nn.Linear(512, len(df['Pneumothorax'].unique())),
        })

        # Store task names for easy access
        self.task_names = list(self.task_outputs.keys())

    def forward(self, x):
        # Pass input through the ResNet backbone
        x = self.backbone(x)

        # Pass through the shared fully connected layer
        x = F.relu(self.shared_fc(x))

        # Compute outputs for each task
        outputs = {task: self.task_outputs[task](x) for task in self.task_names}

        return outputs
