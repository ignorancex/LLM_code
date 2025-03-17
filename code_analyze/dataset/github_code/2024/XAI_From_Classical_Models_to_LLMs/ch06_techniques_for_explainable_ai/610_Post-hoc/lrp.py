import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Create XOR dataset
X = torch.Tensor([[0,0], [0,1], [1,0], [1,1]])
y = torch.Tensor([[0], [1], [1], [0]])

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 4)  # First layer with 4 neurons
        self.fc2 = nn.Linear(4, 1)  # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Initialize the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Train the model
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch + 1}/1000], Loss: {loss.item():.4f}')

# Select an input for explanation
x_input = torch.Tensor([[1.0, 1.0]])
output = model(x_input)
print(f'\nPrediction for input {x_input.numpy()}: {output.item():.4f}')

# Forward pass, recording intermediate activations
x0 = x_input.detach()
z1 = model.fc1(x0)
a1 = F.relu(z1)
z2 = model.fc2(a1)
a2 = torch.sigmoid(z2)

# LRP parameters
epsilon = 1e-6

# Calculate relevance R2 at the output layer
R2 = a2.item()  # Get scalar value

# Propagate relevance from the output layer to the hidden layer
w2 = model.fc2.weight.data.squeeze()  # Shape becomes [4]
a1 = a1.detach().squeeze()            # Shape becomes [4]
z = a1 * w2                           # Element-wise multiplication, shape [4]
s = z.sum()
denominator = s + epsilon * s.sign()
R1 = (z / denominator) * R2           # Shape is [4]

# Propagate relevance from the hidden layer to the input layer
w1 = model.fc1.weight.data            # Shape is [4, 2]
x0 = x0.detach().squeeze()            # Shape is [2]
R0 = torch.zeros_like(x0)             # Shape is [2]

# Iterate over each neuron in the hidden layer
for i in range(w1.shape[0]):
    w = w1[i]                         # Shape is [2]
    z = x0 * w                        # Element-wise multiplication, shape [2]
    s = z.sum()
    denominator = s + epsilon * s.sign()
    R0 += (z / denominator) * R1[i].item()  # Convert R1[i] to scalar

# Output the relevance scores
print(f'\nInput relevance scores: {R0}')
print(f'Sum of input relevances: {R0.sum().item():.4f}')
print(f'Output relevance: {R2:.4f}')
