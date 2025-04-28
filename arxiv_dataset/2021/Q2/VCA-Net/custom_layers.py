import torch.nn as nn


class Conv2d_BN_Block(nn.Module):
	def __init__(self, in_channels, out_channels, **kwargs):
		super(Conv2d_BN_Block, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, bias=True, **kwargs),
			nn.BatchNorm2d(out_channels),
			nn.LeakyReLU(inplace=True),
		)

	def forward(self, x):
		x = self.conv(x)
		return x


class Up_Conv_Block(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(Up_Conv_Block, self).__init__()
		self.up = nn.Sequential(
			nn.Upsample(scale_factor=2),
			nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
			nn.BatchNorm2d(out_channels),
			nn.LeakyReLU(inplace=True)
		)

	def forward(self, x):
		x = self.up(x)
		return x
