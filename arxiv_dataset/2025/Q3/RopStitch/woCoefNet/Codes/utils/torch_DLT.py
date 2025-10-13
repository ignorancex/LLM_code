import torch
import numpy as np
import cv2

# src_p: shape=(bs, 4, 2)
# det_p: shape=(bs, 4, 2)
#
#                                     | h1 |
#                                     | h2 |                   
#                                     | h3 |
# | x1 y1 1  0  0  0  -x1x2  -y1x2 |  | h4 |  =  | x2 |
# | 0  0  0  x1 y1 1  -x1y2  -y1y2 |  | h5 |     | y2 |
#                                     | h6 |
#                                     | h7 |
#                                     | h8 |

def tensor_DLT(src_p, dst_p):
   
    bs, _, _ = src_p.shape

    ones = torch.ones(bs, 4, 1)
    if torch.cuda.is_available():
        ones = ones.cuda()
    xy1 = torch.cat((src_p, ones), 2)
    zeros = torch.zeros_like(xy1)
    if torch.cuda.is_available():
        zeros = zeros.cuda()

    xyu, xyd = torch.cat((xy1, zeros), 2), torch.cat((zeros, xy1), 2)
    M1 = torch.cat((xyu, xyd), 2).reshape(bs, -1, 6)
    M2 = torch.matmul(
        dst_p.reshape(-1, 2, 1), 
        src_p.reshape(-1, 1, 2),
    ).reshape(bs, -1, 2)
    
    # Ah = b
    A = torch.cat((M1, -M2), 2)
    b = dst_p.reshape(bs, -1, 1)
    
    #h = A^{-1}b
    Ainv = torch.inverse(A)
    h8 = torch.matmul(Ainv, b).reshape(bs, 8)
 
    H = torch.cat((h8, ones[:,0,:]), 1).reshape(bs, 3, 3)
    return H


#  Fast and Interpretable 2D Homography Decomposition: Similarity-Kernel-Similarity and Affine-Core-Affine Transformations
#  https://ieeexplore.ieee.org/abstract/document/10994533

def Tensor_ACA_rect(src_ps, dst_p):
    bs = src_ps.shape[0]
    src_ps_mid = src_ps.transpose(1, 2)
    dst_p_mid = dst_p.transpose(1, 2)

    ones = torch.ones((bs, 1, 4), device=src_ps.device)
    src = torch.cat((src_ps_mid, ones), dim=1)
    tar = torch.cat((dst_p_mid, ones), dim=1)
    scale = src[0, 0, 1:2] - src[0, 0, 0:1]
    div = scale / (src[0, 1, 2:3] - src[0, 1, 0:1])
    
    H = torch.zeros((bs, 3, 3), device=src.device)
    MN_MP_MQ_P2 = tar[:, :, 1:] - tar[:, :, 0:1]
    Q4 = torch.cross(MN_MP_MQ_P2[:, 1:2, :], MN_MP_MQ_P2[:, 0:1, :], dim=2)
    h_temp = torch.sum(Q4, dim=2, keepdim=True) * tar[:, :, 0:1]
    H[:, :, 0:1] = tar[:, :, 1:2] * Q4[:, :, 0:1] - h_temp
    H[:, :, 1:2] = torch.mul(div, tar[:, :, 2:3] * Q4[:, :, 1:2] - h_temp)
    H[:, :, 2:3] = scale * h_temp - src[:, 0:1, 0:1] * H[:, :, 0:1] - src[:, 1:2, 0:1] * H[:, :, 1:2]
    return H