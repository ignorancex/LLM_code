import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from cross_stitch_unit.cross_stitch_unit import CrossStitchUnit

class XrayResNetWithCrossStitch(nn.Module):
    def __init__(self, img_size, df, resnet_version='resnet50', num_tasks=14):
        super(XrayResNetWithCrossStitch, self).__init__()
        self.img_size = img_size
        self.df = df
        self.num_tasks = num_tasks

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

        # Fully connected layers for each task
        self.shared_fc = nn.Linear(num_features, 512)
        self.task_outputs = nn.ModuleList([
            nn.Linear(512, len(set(self.df[task]))) for task in df.columns
        ])

        # Add cross-stitch units at intermediate layers
        self.cross_stitch_units = nn.ModuleList([
            CrossStitchUnit(num_tasks=self.num_tasks) for _ in range(3)  # Add 3 cross-stitch units
        ])

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
        # Extract shared features from the ResNet backbone
        x = self.backbone(x)  # Shape: [batch_size, num_features]
        shared_features = F.relu(self.shared_fc(x))  # Shape: [batch_size, 512]

        # Initialize task-specific features as a list (one for each task)
        task_features = [shared_features.clone() for _ in range(self.num_tasks)]

        # Pass through cross-stitch units
        for cross_stitch_unit in self.cross_stitch_units:
            task_features = cross_stitch_unit(task_features)  # Cross-stitch operation

        # Generate outputs for each task
        task_outputs = [output_layer(features) for output_layer, features in zip(self.task_outputs, task_features)]

        return task_outputs
