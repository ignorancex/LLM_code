'''
    This code is borrowed from https://github.com/marcojira/fld
    Thanks for their great work
'''

import torch
import torchvision.transforms as transforms
from src.features.ImageFeatureExtractor import ImageFeatureExtractor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SwAVFeatureExtractor(ImageFeatureExtractor):
    def __init__(self, save_path=None, logger=None):
        self.name = "swav_resnet50"

        super().__init__(save_path, logger)

        self.features_size = 2048
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(
                    (256, 256), interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

        self.model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
        self.model.fc = torch.nn.Identity()
        self.model = self.model.to(DEVICE)
        self.model.eval()
        return
    
    def get_feature_batch(self, img_batch: torch.Tensor):
        return self.model(img_batch)

