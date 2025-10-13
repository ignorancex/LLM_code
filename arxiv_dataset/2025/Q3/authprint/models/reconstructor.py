"""
Reconstructor models for SD fingerprinting.
"""
import torch
import torch.nn as nn

class ReconstructorSD_L(nn.Module):
    """
    Very high-capacity Reconstructor for memorizing training data.
    BatchNorm and Dropout removed to discourage generalization.
    """
    def __init__(self, image_size=1024, channels=3, output_dim=4096):
        super(ReconstructorSD_L, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(channels, 256, kernel_size=4, stride=2, padding=1),   # M -> M/2
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),        # M/2 -> M/4
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),       # M/4 -> M/8
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1),      # M/8 -> M/16
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(2048, 3072, kernel_size=4, stride=2, padding=1),      # M/16 -> M/32
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(3072, 4096, kernel_size=4, stride=2, padding=1),      # M/32 -> M/64
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(4096, 6144, kernel_size=4, stride=2, padding=1),      # M/64 -> M/128
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(6144, 8192, kernel_size=4, stride=2, padding=1),      # M/128 -> M/256
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8192, 16384),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16384, 16384),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16384, output_dim)
        )
    
    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)

class ReconstructorSD_M(nn.Module):
    """
    High-capacity Reconstructor for memorizing training data.
    BatchNorm and Dropout removed to discourage generalization.
    """
    def __init__(self, image_size=1024, channels=3, output_dim=4096):
        super(ReconstructorSD_M, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(channels, 128, kernel_size=4, stride=2, padding=1),   # M -> M/2
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),        # M/2 -> M/4
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),        # M/4 -> M/8
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),       # M/8 -> M/16
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(1024, 1536, kernel_size=4, stride=2, padding=1),      # M/16 -> M/32
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(1536, 2048, kernel_size=4, stride=2, padding=1),      # M/32 -> M/64
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(2048, 3072, kernel_size=4, stride=2, padding=1),      # M/64 -> M/128
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(3072, 4096, kernel_size=4, stride=2, padding=1),      # M/128 -> M/256
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4096, 8192),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(8192, 8192),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(8192, output_dim)
        )
    
    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)

class ReconstructorSD_S(nn.Module):
    """
    Reconstructor network that predicts configurable-bit key logits from an input image.
    With increased capacity for better convergence.
    """
    def __init__(self, image_size=1024, channels=3, output_dim=4096):
        """
        Initialize the Reconstructor.
        
        Args:
            image_size (int): Input image size (width/height).
            channels (int): Number of input image channels.
            output_dim (int): Output dimension (key length).
        """
        super(ReconstructorSD_S, self).__init__()
        # Increase number of filters and add more layers for 1024x1024 input
        self.features = nn.Sequential(
            # Initial layer: 1024 -> 512
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 512 -> 256
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 256 -> 128
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128 -> 64
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64 -> 32
            nn.Conv2d(512, 768, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(768),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32 -> 16
            nn.Conv2d(768, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16 -> 8
            nn.Conv2d(1024, 1536, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1536),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8 -> 4
            nn.Conv2d(1536, 2048, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Global pooling and deeper classifier with progressive expansion
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 2560),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(2560, 3072),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(3072, output_dim)
        )
    
    def forward(self, x):
        """
        Forward pass of the reconstructor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (B, output_dim).
        """
        features = self.features(x)
        return self.classifier(features)

class StyleGAN2Reconstructor(nn.Module):
    """
    Reconstructor network that predicts configurable-bit key logits from an input image.
    With increased capacity for better convergence.
    """
    def __init__(self, image_size=256, channels=3, output_dim=4):
        """
        Initialize the Reconstructor.
        
        Args:
            image_size (int): Input image size (width/height).
            channels (int): Number of input image channels.
            output_dim (int): Output dimension (key length).
        """
        super(StyleGAN2Reconstructor, self).__init__()
        # Increase number of filters and add more layers
        self.features = nn.Sequential(
            # Initial layer: 256 -> 128
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128 -> 64
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64 -> 32
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32 -> 16
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16 -> 8
            nn.Conv2d(512, 768, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(768),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8 -> 4
            nn.Conv2d(768, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Global pooling and deeper classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        """
        Forward pass of the reconstructor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (B, output_dim).
        """
        features = self.features(x)
        return self.classifier(features)

# Remove old classes 