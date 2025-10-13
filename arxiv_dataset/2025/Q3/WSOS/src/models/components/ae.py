from torch import nn


class AE(nn.Module):
    def __init__(self, input_channels, input_dim, latent_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = nn.Sequential(
            nn.BatchNorm2d(input_channels),
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(512),
            nn.Flatten(),
            nn.Linear(512 * (input_dim // 16) ** 2, latent_dim)  # Fully Connected Bottleneck Layer
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512 * (input_dim // 16) ** 2),  # Fully Connected Layer for Decoder
            nn.Unflatten(1, (512, (input_dim // 16), (input_dim // 16))),
            # Unflatten to the shape before the bottleneck
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        self.latent_dim = latent_dim

    def forward(self, x):
        y = self.encoder(x)
        y = self.decoder(y)
        return y



class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class DeepAE(nn.Module):
    def __init__(self, input_channels, input_dim, latent_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Encoder
        self.encoder = nn.Sequential(
            # Initial normalization
            nn.BatchNorm2d(input_channels),
            
            # First block (input_channels -> 64)
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            ResidualBlock(64),
            
            # Second block (64 -> 128)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            ResidualBlock(128),
            ResidualBlock(128),
            
            # Third block (128 -> 256)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            ResidualBlock(256),
            ResidualBlock(256),
            
            # Fourth block (256 -> 512)
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            ResidualBlock(512),
            ResidualBlock(512),
            
            # Fifth block (512 -> 768)
            nn.Conv2d(512, 768, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(768),
            ResidualBlock(768),
            
            # Flatten and project to latent space
            nn.Flatten(),
            nn.Linear(768 * (input_dim // 16) ** 2, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # Projection from latent space
            nn.Linear(latent_dim, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 768 * (input_dim // 16) ** 2),
            nn.Unflatten(1, (768, (input_dim // 16), (input_dim // 16))),
            
            # First block (768 -> 512)
            ResidualBlock(768),
            nn.ConvTranspose2d(768, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            
            # Second block (512 -> 256)
            ResidualBlock(512),
            ResidualBlock(512),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            # Third block (256 -> 128)
            ResidualBlock(256),
            ResidualBlock(256),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            # Fourth block (128 -> 64)
            ResidualBlock(128),
            ResidualBlock(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # Final block (64 -> input_channels)
            ResidualBlock(64),
            nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
        self.latent_dim = latent_dim
        
    def forward(self, x):
        y = self.encoder(x)
        y = self.decoder(y)
        return y