import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn as nn
import numpy as np

def expand_weight_for_dil_conv(weight, dilation=2):
  c_out, c_in, kh, kw = weight.shape
  ret_kh = kh + (kh-1)*(dilation-1)
  ret_kw = kw + (kw-1)*(dilation-1)
  ret = np.zeros((c_out, c_in, ret_kh, ret_kw))
  for o in range(c_out):
    for i in range(c_in):
      for h in range(kh):
        for w in range(kw):
          ret[o,i,h*dilation, w*dilation] = weight[o,i,h,w]
  return ret

def calculate_stable_rank(conv_module, in_shape, orig=False):
  kernel_size = conv_module.kernel_size[0]
  groups = conv_module.groups
  dilation = conv_module.dilation[0]
  if orig:
    weight = conv_module.weight_orig.detach().cpu().numpy()
  else:
    weight = conv_module.weight.detach().cpu().numpy()
  if dilation != 1:
    weight = expand_weight_for_dil_conv(weight, dilation)
  o_channel = weight.shape[0]

  if groups == 1:
    fft_coeff = np.fft.fft2(weight, in_shape, axes=[2,3])
    t_fft = np.transpose(fft_coeff)
    U, D, V = np.linalg.svd(t_fft, compute_uv=True, full_matrices=False)
    Dflat = np.sort(D.flatten())[::-1]
  else:
    fft_coeff = np.fft.fft2(weight, in_shape, axes=[2,3])
    fft_coeff_expand = np.zeros((fft_coeff.shape[0], fft_coeff.shape[0], fft_coeff.shape[2], fft_coeff.shape[3]), dtype=np.complex)
    for idx in range(len(fft_coeff)):
      fft_coeff_expand[idx][idx] = fft_coeff[idx][0]
    t_fft = np.transpose(fft_coeff_expand)
    U, D, V = np.linalg.svd(t_fft, compute_uv=True, full_matrices=False)
    Dflat = np.sort(D.flatten())[::-1]

  Dflat_sq = Dflat**2
  return np.sum(Dflat_sq) / Dflat_sq[0]
