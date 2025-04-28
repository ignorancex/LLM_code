from skimage.metrics import structural_similarity
import torch
import numpy as np

def ssim(batch1, batch2):
    ssims = []
    for i in range(len(batch1)):
        ssims.append(structural_similarity(batch1[i,0].numpy(), batch2[i,0].numpy()))
    return np.mean(ssims)
    
    
def psnr(batch1, batch2):
    assert (batch1.shape == batch2.shape)
    mse = torch.mean((batch1-batch2) ** 2, dim=(1,2,3))
    return torch.mean(20 * torch.log10(1/torch.sqrt(mse)))
    
    
def mse(x, y):
    return torch.mean((x-y)**2)
    
    
def phase(batch, s=(64,64), norm='ortho'):
    batch_ft = torch.fft.fft2(batch, s=s, dim=(-2, -1), norm=norm)
    return batch_ft/(torch.abs(batch_ft)+1e-17)
    
    
def magnitude(batch, s=(64, 64), norm='ortho'):
    batch_ft = torch.fft.fft2(batch, s=s, dim=(-2, -1), norm=norm)
    return torch.abs(batch_ft+1e-17)
    
    
def check_range(x):
    print(x.min().item(), x.max().item())