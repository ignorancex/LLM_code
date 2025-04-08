import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        
        # Encoder layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # For mean
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # For log variance
        
        # Decoder layers
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = torch.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def generate_embeddings(self, x):
        mu, _ = self.encode(x)
        return mu  # Embeddings from the latent space

# Step 3: Loss function and training loop for VAE
def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train_vae(model, data_loader, epochs=1000, learning_rate=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch, in data_loader:
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch[0])  # Only pass data tensors
            loss = vae_loss(recon_batch, batch[0], mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {train_loss / len(data_loader.dataset)}')
    return model

def getUsersEmbedding():
    # Load user information
    user_info = pd.read_csv("./Data/users_info.csv")   # Replace with actual path to CSV

    # Preprocess categorical variables
    user_info['Gender'] = LabelEncoder().fit_transform(user_info['Gender'])  # Encode Gender (e.g., Male -> 1, Female -> 0)

    from sklearn.preprocessing import MinMaxScaler

    # Normalize user information to be between 0 and 1
    scaler = MinMaxScaler()
    user_data_scaled = scaler.fit_transform(user_info[['Gender', 'Age', 'experience video_game', 'experience VR', 'use stimulants']])

    # Convert scaled data to tensor
    user_data_tensor = torch.tensor(user_data_scaled, dtype=torch.float32)

    # Create a DataLoader for batching in VAE
    user_dataset = TensorDataset(user_data_tensor)
    user_loader = DataLoader(user_dataset, batch_size=1, shuffle=True)

    # Train the VAE model
    vae = VAE(input_dim=user_data_scaled.shape[1], hidden_dim=10, latent_dim=3)  # Adjust dimensions based on data
    vae = train_vae(vae, user_loader)

    # Generate embeddings using the trained VAE
    vae.eval()
    with torch.no_grad():
        user_embeddings = vae.generate_embeddings(user_data_tensor).numpy()  # Generate embeddings

    # Merge VAE embeddings with the existing dataset
    # Assuming 'data' is your main dataset that has been previously loaded and processed
    embedding_df = pd.DataFrame(user_embeddings, columns=[f'embed_{i}' for i in range(user_embeddings.shape[1])])
    embedding_df = pd.concat([user_info["Subject_id"] , embedding_df], axis=1)

    return embedding_df
