import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
from cnn1 import transform, test


def conv2d(input, weight, bias=None, stride=1, padding=0):
    batch_size, in_channels, in_height, in_width = input.size()
    out_channels, _, kernel_height, kernel_width = weight.size()

    # Compute output dimensions
    out_height = (in_height + 2 * padding - kernel_height) // stride + 1
    out_width = (in_width + 2 * padding - kernel_width) // stride + 1

    # Apply padding to the input
    if padding > 0:
        input_padded = F.pad(input, (padding, padding, padding, padding))
    else:
        input_padded = input

    # Initialize output tensor
    output = torch.zeros(batch_size, out_channels, out_height, out_width)

    # Perform convolution using vectorized operations
    for b in range(batch_size):
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * stride
                h_end = h_start + kernel_height
                w_start = j * stride
                w_end = w_start + kernel_width

                for oc in range(out_channels):
                    input_patch = input_padded[b, :, h_start:h_end, w_start:w_end]
                    output[b, oc, i, j] = torch.sum(input_patch * weight[oc, :, :, :])
                    if bias is not None:
                        output[b, oc, i, j] += bias[oc]

    return output


def maxpool2d(input, kernel_size, stride=2, padding=0):
    batch_size, in_channels, in_height, in_width = input.size()

    # Compute output dimensions
    out_height = (in_height + 2 * padding - kernel_size) // stride + 1
    out_width = (in_width + 2 * padding - kernel_size) // stride + 1

    # Apply padding to the input
    if padding > 0:
        input_padded = torch.full((batch_size, in_channels, in_height + 2 * padding, in_width + 2 * padding), float('-inf'))
        input_padded[:, :, padding:padding + in_height, padding:padding + in_width] = input
    else:
        input_padded = input

    # Initialize output tensor
    output = torch.zeros(batch_size, in_channels, out_height, out_width)

    # Perform max pooling
    for b in range(batch_size):
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * stride
                h_end = h_start + kernel_size
                w_start = j * stride
                w_end = w_start + kernel_size

                for ic in range(in_channels):
                    output[b, ic, i, j] = torch.max(input_padded[b, ic, h_start:h_end, w_start:w_end])

    return output


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


batch_size = 1

# Datasets and Dataloaders
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


inference_model = SimpleCNN()
# inference_model = inference_model.cuda()
inference_model.load_state_dict(torch.load('model_weights.pth', map_location=torch.device('cpu')))
inference_model.eval()  # Set the model to evaluation mode


start_time = time.time()
test(inference_model, test_loader, use_cuda=False)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time for the original model using sequential conv2d: {elapsed_time:.6f} seconds")
