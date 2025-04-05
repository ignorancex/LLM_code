import torch
import torch.nn as nn
from spectralnorm.spectral_norm_conv_inplace import spectral_norm_conv
from spectralnorm.spectral_norm_fc import spectral_norm_fc

OPS_SPECTORAL_NORM = {
  'none' : lambda C, stride, affine: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine, shape: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine, shape: nn.MaxPool2d(3, stride=stride, padding=1),
  'skip_connect' : lambda C, stride, affine, shape: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'sep_conv_3x3' : lambda C, stride, affine, shape: SepConv(C, C, 3, stride, 1, affine=affine, shape=shape),
  'sep_conv_5x5' : lambda C, stride, affine, shape: SepConv(C, C, 5, stride, 2, affine=affine, shape=shape),
  'sep_conv_7x7' : lambda C, stride, affine, shape: SepConv(C, C, 7, stride, 3, affine=affine, shape=shape),
  'dil_conv_3x3' : lambda C, stride, affine, shape: DilConv(C, C, 3, stride, 2, 2, affine=affine, shape=shape),
  'dil_conv_5x5' : lambda C, stride, affine, shape: DilConv(C, C, 5, stride, 4, 2, affine=affine, shape=shape),
  'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
    nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
    nn.BatchNorm2d(C, affine=affine)
    ),
}

class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)

class DilConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, shape, affine=True, coeff=1., n_power_iter=5):
    super(DilConv, self).__init__()
    self.coeff = coeff
    self.n_power_iter = n_power_iter
    self.shape = shape
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding

    h, w = shape
    first_conv_shape = (C_in, h, w)
    if stride == 2:
      other_conv_shape = (C_in, h//2, w//2)
    else:
      other_conv_shape = (C_in, h, w)
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      self._wrapper_spectral_norm(
        nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
        first_conv_shape,
        kernel_size=kernel_size
      ),
      self._wrapper_spectral_norm(
        nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
        other_conv_shape,
        kernel_size=kernel_size
      ),
      nn.BatchNorm2d(C_out, affine=affine),
    )

  def forward(self, x):
    return self.op(x)

  def _wrapper_spectral_norm(self, op, shape, kernel_size):
    if kernel_size == 1:
      return spectral_norm_fc(op, self.coeff, shape, n_power_iterations=self.n_power_iter)
    else:
      return spectral_norm_conv(op, self.coeff, shape, n_power_iterations=self.n_power_iter)

class SepConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, shape, affine=True, coeff=1., n_power_iter=5):
    super(SepConv, self).__init__()
    self.coeff = coeff
    self.n_power_iter = n_power_iter
    self.shape = shape
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding

    h, w = shape
    first_conv_shape = (C_in, h, w)
    if stride == 2:
      other_conv_shape = (C_in, h//2, w//2)
    else:
      other_conv_shape = (C_in, h, w)
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      self._wrapper_spectral_norm(
        nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
        first_conv_shape,
        kernel_size=kernel_size
      ),
      self._wrapper_spectral_norm(
        nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
        other_conv_shape,
        kernel_size=kernel_size
      ),
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      self._wrapper_spectral_norm(
        nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
        other_conv_shape,
        kernel_size=kernel_size
      ),
      self._wrapper_spectral_norm(
        nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
        other_conv_shape,
        kernel_size=kernel_size
      ),
      nn.BatchNorm2d(C_out, affine=affine),
    )

  def forward(self, x):
    return self.op(x)

  def _wrapper_spectral_norm(self, op, shape, kernel_size):
    if kernel_size == 1:
      return spectral_norm_fc(op, self.coeff, shape, n_power_iterations=self.n_power_iter)
    else:
      return spectral_norm_conv(op, self.coeff, shape, n_power_iterations=self.n_power_iter)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out
