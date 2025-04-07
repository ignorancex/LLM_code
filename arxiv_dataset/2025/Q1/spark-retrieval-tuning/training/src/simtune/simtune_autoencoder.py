import torch.nn as nn

class SimtuneAutoencoder(nn.Module):
    input_dimension = 24
    latent_dimension = 5
    def __init__(self):
        super(SimtuneAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(SimtuneAutoencoder.input_dimension, SimtuneAutoencoder.latent_dimension),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(SimtuneAutoencoder.latent_dimension, SimtuneAutoencoder.input_dimension),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)