import torch
import torch.nn as nn
import torch.optim as optim
from captum.attr import DeepLift
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Convert TensorFlow data to PyTorch format
train_transform = transforms.Compose([transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])

# Load the MNIST dataset using torchvision
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# Define a simple CNN model in PyTorch
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 14 * 14)
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x

# Initialize the model, loss function, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model for one epoch (for simplicity)
for images, labels in train_loader:
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    break  # Train on one batch for demonstration

# Select a sample image and its baseline
sample_image, sample_label = next(iter(test_loader))
baseline = torch.zeros_like(sample_image)

# Compute DeepLIFT attributions with the target label
dl = DeepLift(model)
attributions = dl.attribute(sample_image, baseline, target=sample_label.item())

# Visualize the attributions
attributions = attributions.detach().numpy().squeeze()
plt.imshow(attributions, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title("DeepLIFT Attribution for MNIST Prediction")
plt.show()
