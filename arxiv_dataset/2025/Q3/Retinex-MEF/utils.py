import torch
import cv2
import numpy as np
import torch.utils.data as Data
import os

def FuseIlluminance(L1,L2,k):
    L_f=(L1 + L2) / 2
    return (k*L_f)**1/((k*L_f)**1+((1-k)*(1-L_f))**1) 

def calculate_k(imga, imgb, E, max_iter=1000,k=0.6):
    if imga.shape != imgb.shape:
        raise ValueError("Images must have the same dimensions")
    
    k=torch.tensor(k).to(imga.device)
    E=torch.tensor(E).to(imga.device)

    def f(k):
        return FuseIlluminance(imga,imgb,k).mean()-E

    low = torch.tensor(0, device=imga.device)
    high = torch.tensor(1.0, device=imga.device)
    
    for _ in range(max_iter):
        mid = (low + high) / 2.0
        
        if torch.abs(f(mid)) < 1/(2*255):
            return mid.item()

        if f(mid) < 0:
            low = mid
            
        else:
            high = mid

    return mid.item()


def image_read(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'Gray' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'Gray':  
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img



def convert2gray(input_tensor):
    if input_tensor.size(1) == 3:
        r = input_tensor[:, 0:1, :, :]
        g = input_tensor[:, 1:2, :, :]
        b = input_tensor[:, 2:3, :, :]
        return 0.2989 * r + 0.5870 * g + 0.1140 * b
    else:
        return input_tensor
        
def illu_smooth(illu,img,c=10):

    img=convert2gray(img)   
    grad_y_illu = torch.abs(illu[:, :, 1:, :] - illu[:, :, :-1, :])
    grad_x_illu = torch.abs(illu[:, :, :, 1:] - illu[:, :, :, :-1])

    grad_y_img = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
    grad_x_img = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])

    x_tv = grad_x_illu /torch.max(torch.tensor([0.01]).to(illu.device),torch.max(torch.abs(grad_x_img))) 
    y_tv = grad_y_illu /torch.max(torch.tensor([0.01]).to(illu.device),torch.max(torch.abs(grad_y_img)))

    return torch.mean(x_tv) + torch.mean(y_tv)


class SICE_training(Data.Dataset):
    def __init__(self, file_path=r"data/SICE_training"):
        self.file_path = file_path
        
    def __len__(self):
        return len(os.listdir(self.file_path))
    
    def __getitem__(self, index):
        patch_path=os.path.join(self.file_path,str(index))
        img1=image_read(os.path.join(patch_path,"img_1.png")).transpose(2,0,1)/255.
        img2=image_read(os.path.join(patch_path,"img_2.png")).transpose(2,0,1)/255.
        img3=image_read(os.path.join(patch_path,"img_3.png")).transpose(2,0,1)/255.
        return  torch.Tensor(img1),torch.Tensor(img2),torch.Tensor(img3),index