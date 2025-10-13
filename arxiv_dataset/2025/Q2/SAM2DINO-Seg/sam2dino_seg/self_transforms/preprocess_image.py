import torch
import torchvision.transforms as transforms
from PIL import Image
def transforms_image(image_path, image_size=518):
    """预处理图像以适应DINOV2模型的输入要求"""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)