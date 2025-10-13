import torch
from torch import nn
import lpips

class LPIPS(nn.Module):
    def __init__(self):
        super(LPIPS, self).__init__()
        self.metric = lpips.LPIPS(net='alex')

        for m in self.metric.modules():
            names = [name for name, _ in m.named_parameters()]
            for name in names:
                if hasattr(m, name):
                    data = getattr(m, name).data
                    delattr(m, name)
                    m.register_buffer(name, data, persistent=False)

            names = [name for name, _ in m.named_buffers()]
            for name in names:
                if hasattr(m, name):
                    data = getattr(m, name).data
                    delattr(m, name)
                    m.register_buffer(name, data, persistent=False)

    @torch.no_grad()
    def __call__(self, inputs, targets):
        return self.metric(self.transform(inputs), self.transform(targets), normalize=True).mean()

    def train(self, mode: bool = True):
        return self
                                      
    def transform(self, img):
        return img[:, [2, 1, 0], :, :]  * 0.5 + 0.5