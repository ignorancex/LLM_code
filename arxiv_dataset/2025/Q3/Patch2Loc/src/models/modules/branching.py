import torch
import torch.nn as nn
import torchvision.models as models
import math


class SinusoidalPositionEncoder(nn.Module):
    def __init__(self, embed_dim):
        super(SinusoidalPositionEncoder, self).__init__()
        self.embed_dim = embed_dim

    def forward(self, positions):
        # Positions should be a tensor of shape [batch_size]
        positions = positions.unsqueeze(1)  # Shape [batch_size, 1]
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2, dtype=torch.float32, device=positions.device)
            * -(math.log(10000.0) / self.embed_dim)
        )  # Shape [embed_dim // 2]

        # Apply sin to even indices in the array and cos to odd indices
        position_encoding = torch.zeros((positions.size(0), self.embed_dim), device=positions.device)
        position_encoding[:, 0::2] = torch.sin(positions * div_term)  # Sinusoidal terms
        position_encoding[:, 1::2] = torch.cos(positions * div_term)  # Cosine terms
        return position_encoding


class BranchingModel(nn.Module):
    def __init__(self, input_dim, latent_dim, detach_variance=False, num_outputs=2):
        super(BranchingModel, self).__init__()

        self.encoder = models.resnet18(pretrained=False)
        self.encoder.conv1 = nn.Conv2d(
                    in_channels=1,      
                    out_channels=64,    
                    kernel_size=3,     
                    stride=2,           
                    padding=3,          
                    bias=False
                )
        self.encoder.fc = nn.Identity()

        self.position_encoder = SinusoidalPositionEncoder(latent_dim)

        
        # Branches for mean and logvariance prediction
        self.mean_branch = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, num_outputs),
        )

        self.logvariance_branch = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, num_outputs),
        )

        self.detach_variance = detach_variance

    def set_detach_variance(self):
        self.detach_variance = True

    def forward(self, x, patch_positions):
        # Pass input through encoder to get latent representation
        latent = self.encoder(x)

        # Get sinusoidal position embeddings and add them to the latent representation
        if patch_positions.ndim == 1:
            patch_positions = patch_positions.unsqueeze(1)
        
        pos_embeddings = []
        for patch_positions_ in patch_positions.permute(1,0):
            pos_embeddings.append( self.position_encoder(patch_positions_).to(x) )
        
        pos_embeddings = torch.stack(pos_embeddings).sum(0)
        latent = latent + pos_embeddings

        # Predict mean and logvariance
        mean = self.mean_branch(latent)
        if self.detach_variance == True:
            logvariance = self.logvariance_branch(latent.detach())
        else:
            logvariance = self.logvariance_branch(latent)

        return mean, logvariance