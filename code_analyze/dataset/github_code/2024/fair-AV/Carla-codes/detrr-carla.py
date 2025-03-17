import math
import datetime

from PIL import Image
import requests
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import argparse
import json

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.models as models
import torchvision.transforms as T
torch.set_grad_enabled(False)

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xywh(x):
    x_c, y_c, w, h = x.unbind(1)
    # b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
    #      (x_c + 0.5 * w), (y_c + 0.5 * h)]
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         w, h]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xywh(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=b.device)
    return b

def generate_preds(model, data_imgs, folder):

    folder_pth = f"{data_imgs}/{folder}"

    # get all the images in the folder with word "rgb" in them
    imgs = [f for f in os.listdir(folder_pth) if "rgb" in f]

    preds = []

    for img_data in tqdm(imgs):

        img_name = img_data
        # print(img_name)
        im = Image.open(os.path.join(folder_pth, img_name))
        im = im.convert('RGB')
        # print(im.shape())
        img = transform(im).unsqueeze(0).cuda()

        # propagate through the model
        outputs = model(img)

        # keep only predictions with 0.7+ confidence
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.0

        # convert boxes from [0; 1] to image scales
        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

        bboxes_list = bboxes_scaled.tolist()

        probas = probas[keep]

        for i in range(len(probas)):
            cl = probas[i].argmax()
            # print(cl, cl.item())
            if cl.item() == 1:
                preds.append({
                    "image_id": int(img_data.split("_")[0]),
                    "category_id": 1,
                    "bbox": bboxes_list[i],
                    "score": probas[i][cl].item()
                })

    # get date and time, create a folder, save eval_imgs and eval stats as torch.load
    folder_name = f"Carla_data/Preds/{folder}"
    os.mkdir(folder_name)
    output_pth = f"{folder_name}/preds_detr_r50.json"
    with open(output_pth, "w") as f:
        json.dump(preds, f)

def main():

    parser = argparse.ArgumentParser(description='Generating preds for each skin tone')
    parser.add_argument('--folder', type=str, default='0001.0.0.0', help='img folder to generate preds for')
    args = parser.parse_args()

    data_imgs = "Carla_data/Images"

    print("Generating preds for folder ", args.folder)

    model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True).cuda()
    model.eval()

    generate_preds(model, data_imgs, args.folder)

if __name__ == '__main__':
    main()
    
