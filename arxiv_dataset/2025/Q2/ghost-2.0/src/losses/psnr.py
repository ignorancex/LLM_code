import torch

class PSNR(object):
    
    def __call__(self, pred, gt):
        mse = torch.mean((self.transform(pred) - self.transform(gt)) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))
    
    def transform(self, img):
        return (img[:, [2, 1, 0], :, :]  * 0.5 + 0.5) * 255