import torch.nn as nn
import torch.nn.functional as F
import torch

class CIFARConvNet(nn.Module):
	def __init__(self):
		super(CIFARConvNet, self).__init__()
		self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
		self.pool1 = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
		self.pool2 = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(64*8*8, 1024)
		self.fc2 = nn.Linear(1024, 100)

	def forward(self, x):

	        #print(x)
	        #testnan=torch.isnan(x)
	        #for z in testnan:
                    #for y in z:
                        #for b in y:
                            #for a in b:
                                #if a:
                                    #raise ValueError("input of neeeet is nan")
	        x = F.relu(self.conv1(x))
	        x = self.pool1(x)
	        x = F.relu(self.conv2(x))
	        x = self.pool2(x)
	        x = x.view(x.size(0), -1)
	        x = F.relu(self.fc1(x))
	        x = F.dropout(x, p=0.5, training=self.training)
	        x = self.fc2(x)
	        #testnan=torch.isnan(x)

	        #for z in testnan:
                    #for y in z:
                        #if y:
                            #raise ValueError("output of neeeet is nan")
	        return x
