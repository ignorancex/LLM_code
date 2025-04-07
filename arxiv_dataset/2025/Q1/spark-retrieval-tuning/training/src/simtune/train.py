import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

import pandas as pd

from .simtune_autoencoder import SimtuneAutoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on ", device)
torch.manual_seed(0)
    

model = SimtuneAutoencoder().to(device)
mse = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 300
batch_size = 32
shuffle = True

csv = pd.read_csv("src/simtune/features_train.csv")
data = torch.Tensor(csv.to_numpy())
dataset = TensorDataset(data)

train, validation = random_split(dataset, [0.8, 0.2])

train_dl = DataLoader(train, batch_size, shuffle)
validation_dl = DataLoader(validation, batch_size, shuffle)

for epoch in range(epochs):
    total_loss = 0
    for batch in train_dl:
        batch_in = batch[0].to(device)
        optimizer.zero_grad()
        outputs = model(batch_in)
        loss = mse(outputs, batch_in)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_in.size(0)
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train)}")


### EVAL

model.eval()

validation_loss = nn.MSELoss()
total_loss = 0
num_samples = 0

with torch.no_grad():  # Disable gradient computation for inference
    for inputs in validation_dl:
        inputs = inputs[0].to(device)  # Move inputs to device (CPU or GPU)
        outputs = model(inputs)  # Forward pass through the autoencoder
        loss = validation_loss(outputs, inputs)  # Calculate MSE loss
        total_loss += loss.item() * inputs.size(0)
        num_samples += inputs.size(0)

# Step 3: Calculate the Mean Squared Error (MSE)
validation_result = total_loss / num_samples
print(f"MSE on validation dataset: {validation_result}")

torch.save(model.state_dict, "src/simtune/simtune.model")
    







