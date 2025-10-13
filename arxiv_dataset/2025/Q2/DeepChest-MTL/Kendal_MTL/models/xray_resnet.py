import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


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

        # Define the fully connected layer
        self.fc = nn.Linear(num_features, 512)

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
            'Atelectasis': nn.Linear(512, len(set(self.df['Atelectasis']))),
            'Cardiomegaly': nn.Linear(512, len(set(self.df['Cardiomegaly']))),
            'Consolidation': nn.Linear(512, len(set(self.df['Consolidation']))),
            'Edema': nn.Linear(512, len(set(self.df['Edema']))),
            'Effusion': nn.Linear(512, len(set(self.df['Effusion']))),
            'Emphysema': nn.Linear(512, len(set(self.df['Emphysema']))),
            'Fibrosis': nn.Linear(512, len(set(self.df['Fibrosis']))),
            'Hernia': nn.Linear(512, len(set(self.df['Hernia']))),
            'Infiltration': nn.Linear(512, len(set(self.df['Infiltration']))),
            'Mass': nn.Linear(512, len(set(self.df['Mass']))),
            'Nodule': nn.Linear(512, len(set(self.df['Nodule']))),
            'Pleural_Thickening': nn.Linear(512, len(set(self.df['Pleural_Thickening']))),
            'Pneumonia': nn.Linear(512, len(set(self.df['Pneumonia']))),
            'Pneumothorax': nn.Linear(512, len(set(self.df['Pneumothorax']))),
        })

        # Task-specific uncertainties (log_sigma for stability)
        self.log_sigmas = nn.ParameterDict({
            task: nn.Parameter(torch.tensor(0.0)) for task in self.task_outputs.keys()
        })

    def forward(self, x):
        # Pass input through the ResNet backbone
        x = self.backbone(x)

        # Pass through the fully connected layer
        x = F.relu(self.fc(x))

        # Compute task outputs and uncertainties
        outputs = {task: self.task_outputs[task](x) for task in self.task_outputs}
        uncertainties = {task: self.log_sigmas[task] for task in self.log_sigmas}

        return outputs, uncertainties
