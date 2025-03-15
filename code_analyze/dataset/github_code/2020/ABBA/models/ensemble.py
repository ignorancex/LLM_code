from .resnet.resnet101 import resnet101
from torch import nn
import torch
import os

class EnsembleNet(nn.Module):
	def __init__(self):
		super(EnsembleNet, self).__init__()

		self.model_1 = torch.nn.DataParallel(resnet101()).cuda().eval()
		path = os.path.dirname(os.path.abspath(__file__))
		model_1_path = os.path.join(path, 'resnet', 'resnet101.pt')
		
		self.model_1.load_state_dict(torch.load(model_1_path))

	def forward(self, x):
		output_1 = self.model_1(x)

		return output_1