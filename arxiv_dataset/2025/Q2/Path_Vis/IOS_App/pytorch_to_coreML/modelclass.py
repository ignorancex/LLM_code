import torchvision.models as models
import torch.nn as nn

class SegmentationModel(nn.Module):
    def __init__(self, num_classes):
        super(SegmentationModel, self).__init__()
        # Load a pre-trained ResNet-50 model
        self.resnet50 = models.resnet50(pretrained=True)
        
        # Remove the classification layer (fully connected layer)
        self.backbone = nn.Sequential(*list(self.resnet50.children())[:-2])
        
        # Define the segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, num_classes, kernel_size=32, stride=32)  # Upsample to 640x640
        )
    
    def forward(self, x):
        # Forward pass through the backbone
        x = self.backbone(x)
        
        # Forward pass through the segmentation head
        x = self.segmentation_head(x)
        
        return x