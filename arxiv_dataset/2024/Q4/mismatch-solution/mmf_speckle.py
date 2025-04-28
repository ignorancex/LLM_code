import scipy.io as scio
import numpy as np
import torch
import cv2

def uniform_picker_column(matrix):   
    L = matrix.shape[0]
    return (( matrix.reshape((L,150,150)) )[:,11:150-11,11:150-11]).reshape((L,-1))
    
"""
dis: 0 10 25
"""
def get_mmf_speckle_measure_matrix(dis, device, dtype):
    path = r'./mmf_displacement/{}/A_500_256_1.mat'.format(dis)
    matdata = scio.loadmat(path)
    mat = uniform_picker_column(matdata["A1"])
    matrix = torch.tensor(mat, device=device, dtype=dtype)
    return matrix

def get_gi_image(dis, device, dtype):
    path = r'./mmf_displacement/{}/GI_x0y{}.mat'.format(dis,dis)
    matdata = scio.loadmat(path)
    img = matdata["File_image_temp"][11:150-11,11:150-11]
    return torch.tensor(img, device=device, dtype=dtype)

"""
name:
- Baboon
- Peppers
- Goldhill
- Barbara
- Cameraman
- Lena
"""
def get_t_image(name, device, dtype, W=128, H=128):
    path = r'./timg/{}.bmp'.format(name)
    img = cv2.resize(cv2.imread(path, 0), (W, H))
    return torch.tensor(img / 255.0, device=device, dtype=dtype)

def get_pre_measure_img(device, dtype, W=128,H=128):
    return 0.5 * torch.ones(W,H, device=device, dtype=dtype)






def get_mmf_measure(dis, device, dtype):
    path = r'./mmf_displacement/{}/y_original_500_256_1.mat'.format(dis)
    matdata = scio.loadmat(path)
    y = matdata["Data_after"][11:150-11,11:150-11,:].reshape((-1,2500)).sum(0)
    return torch.tensor(y, device=device, dtype=dtype).reshape((-1,1))