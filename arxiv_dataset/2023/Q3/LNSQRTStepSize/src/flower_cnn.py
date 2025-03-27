import torch.nn as nn
import torch.nn.functional as F

class FLOWERConvNet(nn.Module):
	def __init__(self):
		super(FLOWERConvNet, self).__init__()
		self.conv1 = nn.Conv2d(3, 10, kernel_size=5, padding=2)
		self.pool1 = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5, padding=2)
		self.pool2 = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(20*56*56, 1024)
		self.fc2 = nn.Linear(1024, 102)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = self.pool1(x)
		x = F.relu(self.conv2(x))
		x = self.pool2(x)
		x = x.view(x.size(0), -1)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, p=0.5, training=self.training)
		x = self.fc2(x)
		return x
