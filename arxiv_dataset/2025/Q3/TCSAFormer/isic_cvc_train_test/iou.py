import os
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import cv2
from skimage import io
import albumentations as A
from albumentations.pytorch import ToTensor
from networks.bra_unet import BRAUnet


class IoU(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection
        IoU = (intersection + smooth) / (union + smooth)
        return IoU


def get_transform():
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
    ])


def single_image(image_path, mask_path, model_path):
    # 1. 加载模型
    model = BRAUnet(img_size=256, in_chans=3, num_classes=1, n_win=8)
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    model.eval()

    # 2. 加载并预处理图像和mask
    img = io.imread(image_path)[:, :, :3].astype('float32')
    mask = io.imread(mask_path, as_gray=True)

    # 3. 应用转换
    transform = get_transform()
    augmented = transform(image=img, mask=mask)
    img_tensor = augmented['image']
    mask_tensor = augmented['mask']

    # 4. 添加batch维度并转移到GPU
    img_tensor = Variable(torch.unsqueeze(img_tensor, dim=0).float().cuda())
    mask_tensor = Variable(torch.unsqueeze(mask_tensor, dim=0).float().cuda())

    # 5. 模型预测
    with torch.no_grad():
        pred = model(img_tensor)
    pred = torch.sigmoid(pred)
    pred_binary = (pred >= 0.5).float()  # 二值化

    # 6. 计算IOU
    iou_eval = IoU()
    iou_score = iou_eval(pred_binary, mask_tensor)

    print(f'IOU score for the image: {iou_score.item():.4f}')
    return iou_score.item()


if __name__ == '__main__':
    # 设置路径
    image_path = '/home/cqut/Data/medical_seg_data/ISIC2018_jpg/images/ISIC_0000157.png'  # 替换为你的图片路径
    mask_path = '/home/cqut/Data/medical_seg_data/ISIC2018_jpg/masks/ISIC_0000157.png'  # 替换为对应的mask路径
    model_path = '/home/xiazunhui/project/BRAU-Netplusplus-master/isic_cvc_train_test/save_models/bra_unet/isic/epoch_last.pth'  # 替换为你的模型路径

    # 测试单张图片
    iou = single_image(image_path, mask_path, model_path)