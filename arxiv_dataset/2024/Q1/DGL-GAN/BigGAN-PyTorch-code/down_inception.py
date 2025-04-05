import torch
import os
from torchvision.models.inception import inception_v3
from torchvision.models.inception import model_urls

if __name__ == '__main__':
    source = torch.load('/apdcephfs/private_yuesongtian/BigGAN-PyTorch/inception_model/inception_v3_google-1a9a5a14.pth')
    inception_model = inception_v3(pretrained=False, transform_input=False)
    inception_model.load_state_dict(source)
