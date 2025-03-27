import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
from PIL import Image
import random


# Hyperparameters
batch_size = 1 # set to > 1 for training
learning_rate = 0.001
num_epochs = 10


# Custom convolution and maxpool functions
def conv2d(x, weight, bias, stride=1, padding=0):
    x = F.pad(x, (padding, padding, padding, padding))
    return F.conv2d(x, weight, bias, stride=stride)

def maxpool2d(x, kernel_size, stride):
    return F.max_pool2d(x, kernel_size, stride=stride)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1_weight = nn.Parameter(torch.randn(32, 1, 3, 3))
        self.conv1_bias = nn.Parameter(torch.randn(32))
        self.conv2_weight = nn.Parameter(torch.randn(64, 32, 3, 3))
        self.conv2_bias = nn.Parameter(torch.randn(64))
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = conv2d(x, self.conv1_weight, self.conv1_bias, stride=1, padding=1)
        x = F.relu(x)
        x = maxpool2d(x, kernel_size=2, stride=2)

        x = conv2d(x, self.conv2_weight, self.conv2_bias, stride=1, padding=1)
        x = F.relu(x)
        x = maxpool2d(x, kernel_size=2, stride=2)

        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Custom transformation to place 28x28 image in the center of 64x64 image
class PlaceInCenter(object):
    def __call__(self, img):
        blank_image = Image.new("L", (64, 64))
        top_left_x = (64 - 28) // 2
        top_left_y = (64 - 28) // 2
        blank_image.paste(img, (top_left_x, top_left_y))
        return blank_image

# Transformation pipeline
transform = transforms.Compose([
    PlaceInCenter(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# MNIST dataset and dataloaders
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)



# uncomment the following lines if you want to train a model
# Initialize the model, loss function, and optimizer
# model = SimpleCNN()
# model = model.cuda()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

        torch.save(model.state_dict(), 'model_weights.pth')



def shift_central_region(tensor, n, dir='R'):
    """
    Shift the central 28x28 region of a 64x64 tensor to the right by n positions.

    Args:
        tensor (torch.Tensor): The input tensor of shape (1, 1, 64, 64).
        n (int): The number of pixels to shift to the right.

    Returns:
        torch.Tensor: The tensor with the shifted central region.
    """
    # Ensure the tensor is of the correct shape
    assert tensor.shape == (1, 1, 64, 64), "Input tensor must be of shape (1, 1, 64, 64)"
    
    shifted_tensor = torch.zeros_like(tensor)

    # Get the central 28x28 region
    central_region = tensor[:, :, 18:46, 18:46]

    # Place the central region shifted by n pixels to the right
    if n < 64 - 28:
        if dir == 'R':
            shifted_tensor[:, :, 18:46, 18+n:46+n] = central_region
        if dir == 'L':
            shifted_tensor[:, :, 18:46, 18-n:46-n] = central_region
        if dir == 'D':
            shifted_tensor[:, :, 18+n:46+n, 18:46] = central_region
        if dir == 'U':
            shifted_tensor[:, :, 18+n:46+n, 18:46] = central_region


    return shifted_tensor



# Testing loop
def test(model, test_loader, use_cuda = False):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            if use_cuda:
                images, labels = images.cuda(), labels.cuda()
            
            direction = 'R' #random.choice(['L', 'R', 'U', 'D'])

            for n in range(10):
                images = shift_central_region(images, 1, dir = direction)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            
            if total > 100: break

        print(f"Accuracy of the model on the test images: {100 * correct / total}%")

    print(total)


# Run training and testing
# train(model, train_loader, criterion, optimizer, num_epochs)
# test(model, test_loader)


if __name__  == '__main__':

    inference_model = SimpleCNN()
    inference_model = inference_model.to('cpu')
    inference_model.load_state_dict(torch.load('model_weights.pth', map_location=torch.device('cpu')))
    inference_model.eval()  # Set the model to evaluation mode

    start_time = time.time()
    test(inference_model, test_loader, use_cuda=False)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for the original model using parallel conv2d: {elapsed_time:.6f} seconds")
