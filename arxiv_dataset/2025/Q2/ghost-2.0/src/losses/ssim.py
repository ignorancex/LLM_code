import torch
from pytorch_msssim import ssim, ms_ssim

class SSIM(torch.nn.Module):
    def __call__(self, pred, gt):
        return ssim(self.transform(pred), self.transform(gt), data_range=1., size_average=True)
    
    def transform(self, img):
        return img[:, [2, 1, 0], :, :]  * 0.5 + 0.5
    
class MS_SSIM(torch.nn.Module):
    def __call__(self, pred, gt):
        return ms_ssim(self.transform(pred), self.transform(gt), data_range=1., size_average=True)
    
    def transform(self, img):
        return img[:, [2, 1, 0], :, :]  * 0.5 + 0.5