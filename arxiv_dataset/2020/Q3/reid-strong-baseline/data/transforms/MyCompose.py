from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torch.nn import functional as F
from PIL import Image
import torch
class my_compose(Compose):

    def __call__(self, img, image_name):
        for t in self.transforms:
            img = t(img, image_name)
            # try:
            #     img = Image.fromarray(img.astype('uint8'), 'RGB')
            # except AttributeError:
            #     pass
        return img

class MyToTensor(ToTensor):
    def __call__(self, pic, image_name):
        return F.to_tensor(pic)
